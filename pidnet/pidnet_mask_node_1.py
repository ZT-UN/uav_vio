#!/usr/bin/env python3
"""
pidnet_mask_node.py — 基于 PIDNet 的实时语义分割节点
====================================================
功能：
  订阅左目图像 → PIDNet 推理 → 发布二值语义 mask

发布话题：
  /semantic_mask  sensor_msgs/Image, mono8, 255=静态可用区域, 0=动态候选区域
  /mask_ratio     std_msgs/Float32, 本帧动态候选像素占比

Cityscapes 19 类 trainId:
  road=0, sidewalk=1, building=2, wall=3, fence=4, pole=5,
  traffic light=6, traffic sign=7, vegetation=8, terrain=9, sky=10,
  person=11, rider=12, car=13, truck=14, bus=15, train=16,
  motorcycle=17, bicycle=18

动态候选类别：
  person=11, rider=12, car=13, truck=14, bus=15, motorcycle=17, bicycle=18
"""

import sys

sys.path.insert(0, "/mnt/data/lxx/pidnet/PIDNet-main")

import rospy
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from models.pidnet import get_pred_model


# Cityscapes 19 类 trainId 中的动态候选类别
# 注意：这里使用的是 trainId，不是 Cityscapes 原始 labelId。
DYNAMIC_IDS = {11, 12, 13, 14, 15, 17, 18}
# person, rider, car, truck, bus, motorcycle, bicycle


CLASS_NAMES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


# Cityscapes / ImageNet 常用归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_pidnet(model_path, device):
    """
    加载 PIDNet-S Cityscapes 19 类模型。

    strict=False 用于兼容部分权重格式，但必须检查
    missing keys 和 unexpected keys。
    """
    model = get_pred_model("pidnet_s", 19)

    rospy.loginfo(f"[PIDNet] loading model from: {model_path}")

    ckpt = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        rospy.loginfo("[PIDNet] checkpoint source: ckpt['state_dict']")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        rospy.loginfo("[PIDNet] checkpoint source: ckpt['model']")
    else:
        state = ckpt
        rospy.loginfo("[PIDNet] checkpoint source: checkpoint itself")

    if not isinstance(state, dict):
        raise TypeError(
            f"[PIDNet] 权重内容不是 state_dict，实际类型为 {type(state)}"
        )



        # 去掉权重文件中可能存在的 module. 和 model. 前缀
    def normalize_state_key(key):
        while key.startswith("module.") or key.startswith("model."):
            if key.startswith("module."):
                key = key[len("module."):]
            elif key.startswith("model."):
                key = key[len("model."):]

        return key


    state = {
        normalize_state_key(key): value
        for key, value in state.items()
    }

    model_keys = list(model.state_dict().keys())
    weight_keys = list(state.keys())

    rospy.logwarn(
        f"[PIDNet] first model keys after normalization: {model_keys[:5]}"
    )
    rospy.logwarn(
        f"[PIDNet] first weight keys after normalization: {weight_keys[:5]}"
    )

    ret = model.load_state_dict(state, strict=False)

    rospy.logwarn(
        f"[PIDNet] missing keys num = {len(ret.missing_keys)}"
    )
    rospy.logwarn(
        f"[PIDNet] unexpected keys num = {len(ret.unexpected_keys)}"
    )

    if len(ret.missing_keys) > 0:
        rospy.logwarn(
            f"[PIDNet] first missing keys: {ret.missing_keys[:10]}"
        )

    if len(ret.unexpected_keys) > 0:
        rospy.logwarn(
            f"[PIDNet] first unexpected keys: {ret.unexpected_keys[:10]}"
        )

    model.eval()
    model.to(device)

    rospy.loginfo(f"[PIDNet] 模型加载完成，设备={device}")
    rospy.loginfo(f"[PIDNet] model class = {model.__class__.__name__}")
    rospy.loginfo(
        f"[PIDNet] model augment attribute = "
        f"{getattr(model, 'augment', 'not found')}"
    )

    return model


def preprocess(img_bgr):
    """
    ROS/OpenCV BGR 图像 → RGB → 归一化 → torch tensor

    输入：
      img_bgr: OpenCV BGR uint8 image, shape [H, W, 3]

    输出：
      tensor: shape [1, 3, H, W]
    """
    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    img_rgb = img_rgb / 255.0
    img_rgb = (img_rgb - MEAN) / STD

    tensor = torch.from_numpy(img_rgb)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.float()

    return tensor


def log_class_hist(seg_map):
    """
    打印当前语义预测图的类别占比。
    """
    unique, counts = np.unique(seg_map, return_counts=True)
    hist = dict(zip(unique.tolist(), counts.tolist()))

    top_items = sorted(
        hist.items(),
        key=lambda item: item[1],
        reverse=True
    )[:8]

    msg = []

    for cls_id, num in top_items:
        ratio = num / float(seg_map.size)
        class_name = CLASS_NAMES.get(cls_id, "unknown")
        msg.append(f"{cls_id}:{class_name}={ratio:.3f}")

    rospy.loginfo(
        "[PIDNet class hist] " + " | ".join(msg)
    )


def build_mask(seg_map, h, w, erosion_px=4):
    """
    根据 PIDNet 输出的 Cityscapes trainId 语义图构建二值 mask。

    输入：
      seg_map: numpy array [H, W]
      h, w: 原始图像高度和宽度
      erosion_px: 对动态区域做腐蚀

    输出：
      mask: numpy uint8 [H, W]
            255 = 静态可用区域
            0   = 动态候选区域
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255

    for cls_id in DYNAMIC_IDS:
        dynamic_region = (
            seg_map == cls_id
        ).astype(np.uint8)

        if dynamic_region.sum() == 0:
            continue

        if erosion_px > 0:
            kernel_size = erosion_px * 2 + 1

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size)
            )

            dynamic_region = cv2.erode(
                dynamic_region,
                kernel,
                iterations=1
            )

        mask[dynamic_region == 1] = 0

    # 暂时保持原逻辑：图像下方 25% 强制视为静态可用区域
    mask[int(h * 0.75):, :] = 255

    return mask


class PIDNetMaskNode:
    def __init__(self):
        rospy.init_node("pidnet_mask_node")

        image_topic = rospy.get_param(
            "~image_topic",
            "/zed2/camera/left/image_raw"
        )

        model_path = rospy.get_param(
            "~model_path",
            "/mnt/data/lxx/pidnet/PIDNet_S_Cityscapes_val.pt"
        )

        device_str = rospy.get_param(
            "~device",
            "cuda:0"
        )

        self.erosion = int(
            rospy.get_param("~erosion", 4)
        )

        # 默认每 10 帧打印一次类别分布
        self.debug_every = int(
            rospy.get_param("~debug_every", 10)
        )

        self.frame_idx = 0

        # 控制模型输出结构只打印一次，避免每帧刷屏
        self.output_structure_logged = False

        # 控制选中的语义输出分支只打印一次
        self.selected_output_logged = False

        if (
            device_str.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            rospy.logwarn(
                "[PIDNet] CUDA 不可用，自动切换到 CPU"
            )
            device_str = "cpu"

        self.device = torch.device(device_str)

        self.model = load_pidnet(
            model_path,
            self.device
        )

        self.bridge = CvBridge()

        self.pub_mask = rospy.Publisher(
            "/semantic_mask",
            Image,
            queue_size=5
        )

        self.pub_ratio = rospy.Publisher(
            "/mask_ratio",
            Float32,
            queue_size=5
        )

        rospy.Subscriber(
            image_topic,
            Image,
            self.img_callback,
            queue_size=2
        )

        rospy.loginfo(
            f"[PIDNet] 订阅图像话题: {image_topic}"
        )
        rospy.loginfo(
            "[PIDNet] 发布 mask: /semantic_mask, "
            "255=静态, 0=动态候选"
        )
        rospy.loginfo(
            "[PIDNet] 动态类别: person, rider, car, truck, "
            "bus, motorcycle, bicycle"
        )
        rospy.loginfo(
            f"[PIDNet] erosion={self.erosion}, "
            f"debug_every={self.debug_every}"
        )

    def collect_output_tensors(self, obj, path="output"):
        """
        递归收集模型输出中的所有 tensor。

        支持：
          Tensor
          list
          tuple
          dict

        返回：
          [(输出路径, tensor), ...]
        """
        tensors = []

        if torch.is_tensor(obj):
            tensors.append((path, obj))
            return tensors

        if isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                child_path = f"{path}[{index}]"
                tensors.extend(
                    self.collect_output_tensors(
                        item,
                        child_path
                    )
                )

            return tensors

        if isinstance(obj, dict):
            for key, item in obj.items():
                child_path = f"{path}['{key}']"
                tensors.extend(
                    self.collect_output_tensors(
                        item,
                        child_path
                    )
                )

            return tensors

        return tensors

    def log_output_structure(self, raw_out):
        """
        第一次推理时打印模型输出类型和每个 tensor 分支的 shape。
        """
        rospy.logwarn(
            f"[PIDNet] model output type = {type(raw_out)}"
        )

        if isinstance(raw_out, (list, tuple)):
            rospy.logwarn(
                f"[PIDNet] output branch num = {len(raw_out)}"
            )

        elif isinstance(raw_out, dict):
            rospy.logwarn(
                f"[PIDNet] output dict keys = {list(raw_out.keys())}"
            )

        tensor_outputs = self.collect_output_tensors(
            raw_out,
            path="output"
        )

        rospy.logwarn(
            f"[PIDNet] tensor output num = {len(tensor_outputs)}"
        )

        if len(tensor_outputs) == 0:
            rospy.logerr(
                "[PIDNet] 模型输出中没有找到任何 tensor"
            )
            return

        for path, tensor in tensor_outputs:
            rospy.logwarn(
                f"[PIDNet] {path} shape = {tuple(tensor.shape)}, "
                f"dtype = {tensor.dtype}, "
                f"device = {tensor.device}"
            )

    def select_semantic_output(self, raw_out):
        """
        从模型输出中选择四维、通道数为 19 的语义分割输出。

        规则：
        1. 收集所有 tensor 输出；
        2. 只保留 shape=[N, 19, H, W] 的候选；
        3. 如果只有一个候选，直接使用；
        4. 如果有多个候选，暂时选择遍历顺序中的最后一个。

        PIDNet 开启 augment 时，通常可能同时输出：
          辅助语义分支
          主语义分支
          边界分支

        因此本步骤还会完整打印所有候选路径和 shape。
        """
        tensor_outputs = self.collect_output_tensors(
            raw_out,
            path="output"
        )

        semantic_candidates = []

        for path, tensor in tensor_outputs:
            if (
                tensor.dim() == 4
                and tensor.shape[1] == 19
            ):
                semantic_candidates.append(
                    (path, tensor)
                )

        if len(semantic_candidates) == 0:
            available_shapes = [
                f"{path}: {tuple(tensor.shape)}"
                for path, tensor in tensor_outputs
            ]

            rospy.logerr_throttle(
                2.0,
                "[PIDNet] cannot find a 4D 19-channel semantic output. "
                f"Available tensor outputs: {available_shapes}"
            )

            return None

        # 如果存在多个 19 通道输出，逐个打印
        if not self.selected_output_logged:
            rospy.logwarn(
                f"[PIDNet] semantic candidate num = "
                f"{len(semantic_candidates)}"
            )

            for candidate_index, (
                path,
                tensor
            ) in enumerate(semantic_candidates):
                rospy.logwarn(
                    f"[PIDNet] semantic candidate[{candidate_index}] "
                    f"path = {path}, "
                    f"shape = {tuple(tensor.shape)}"
                )

        # 暂时选择最后出现的 19 通道输出
        # 这样不会再使用原来的 raw_out[-1]，
        # 因为 raw_out[-1] 可能是边界输出。
        selected_path, selected_tensor = semantic_candidates[-1]

        if not self.selected_output_logged:
            rospy.logwarn(
                f"[PIDNet] selected semantic output: "
                f"path = {selected_path}, "
                f"shape = {tuple(selected_tensor.shape)}"
            )

            if len(semantic_candidates) > 1:
                rospy.logwarn(
                    "[PIDNet] multiple 19-channel semantic outputs found; "
                    "temporarily selecting the last 19-channel candidate"
                )

            self.selected_output_logged = True

        return selected_tensor

    def img_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as exc:
            rospy.logwarn(
                f"[PIDNet] 图像解码失败: {exc}"
            )
            return

        if img_bgr is None or img_bgr.size == 0:
            rospy.logwarn(
                "[PIDNet] 收到空图像"
            )
            return

        h, w = img_bgr.shape[:2]

        # 图像预处理
        inp = preprocess(img_bgr)
        inp = inp.to(
            self.device,
            non_blocking=True
        )

        # PIDNet 推理
        try:
            with torch.no_grad():
                raw_out = self.model(inp)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                f"[PIDNet] 模型推理失败: {exc}"
            )
            return

        # 第一次推理时，完整打印模型输出结构
        if not self.output_structure_logged:
            self.log_output_structure(raw_out)
            self.output_structure_logged = True

        # 从模型输出中筛选四维 19 通道语义输出
        out = self.select_semantic_output(raw_out)

        if out is None:
            return

        # 再次检查选中的输出是否合法
        if out.dim() != 4:
            rospy.logerr_throttle(
                2.0,
                f"[PIDNet] selected output dim != 4: "
                f"shape={tuple(out.shape)}"
            )
            return

        if out.shape[0] != 1:
            rospy.logerr_throttle(
                2.0,
                f"[PIDNet] selected output batch size != 1: "
                f"shape={tuple(out.shape)}"
            )
            return

        if out.shape[1] != 19:
            rospy.logerr_throttle(
                2.0,
                f"[PIDNet] selected output channels != 19: "
                f"shape={tuple(out.shape)}"
            )
            return

        # 上采样到原始图像分辨率
        out = F.interpolate(
            out,
            size=(h, w),
            mode="bilinear",
            align_corners=True
        )

        # argmax 得到 Cityscapes 19 类 trainId 语义图
        seg_map = (
            out.argmax(dim=1)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # 类别分布日志
        if (
            self.debug_every > 0
            and self.frame_idx % self.debug_every == 0
        ):
            log_class_hist(seg_map)

        # 构建二值动态候选 mask
        mask = build_mask(
            seg_map,
            h,
            w,
            self.erosion
        )

        # 发布 mask，时间戳与输入图像保持一致
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(
                mask,
                encoding="mono8"
            )
        except Exception as exc:
            rospy.logwarn(
                f"[PIDNet] mask 消息转换失败: {exc}"
            )
            return

        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        # 发布动态候选像素占比
        ratio = float(
            np.count_nonzero(mask == 0)
        ) / float(h * w)

        self.pub_ratio.publish(
            Float32(data=ratio)
        )

        self.frame_idx += 1

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = PIDNetMaskNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal(
            f"[PIDNet] 节点启动或运行失败: {exc}"
        )
        raise