#!/usr/bin/env python3
"""
pidnet_mask_node.py — 基于PIDNet的实时语义分割节点
=====================================================
订阅图像 → PIDNet推理 → 发布二值mask

发布话题:
  /semantic_mask  (sensor_msgs/Image, mono8, 255=静态, 0=动态)
  /mask_ratio     (std_msgs/Float32, 本帧动态像素占比)

动态类别 (Cityscapes 19类):
  person=11, rider=12, car=13, truck=14,
  bus=15, train=16, motorcycle=17, bicycle=18
"""

import sys
sys.path.insert(0, '/mnt/data/lxx/pidnet/PIDNet-main')

import rospy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from models.pidnet import get_pred_model

# Cityscapes动态类别ID
DYNAMIC_IDS = {11, 13, 14, 15, 17, 18}
# person, car, truck, bus, motorcycle, bicycle

# Cityscapes图像归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_pidnet(model_path, device):
    model = get_pred_model('pidnet_s', 19)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt

    # 去掉 module. 前缀（多GPU训练保存的权重）
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    rospy.loginfo(f"[PIDNet] 模型加载完成，设备={device}")
    return model


def preprocess(img_bgr):
    """BGR图像 → 归一化tensor"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return img


def build_mask(seg_map, h, w, erosion_px=4):
    """
    seg_map: numpy array [H, W]，每个像素为Cityscapes类别ID
    返回: mask [H, W]，255=静态，0=动态
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255

    for cls_id in DYNAMIC_IDS:
        dynamic_region = (seg_map == cls_id).astype(np.uint8)
        if dynamic_region.sum() == 0:
            continue
        # 腐蚀：去掉动态区域边缘的不确定像素
        if erosion_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erosion_px * 2 + 1, erosion_px * 2 + 1)
            )
            dynamic_region = cv2.erode(dynamic_region, kernel, iterations=1)
        mask[dynamic_region == 1] = 0

    # 排除图像下方25%区域（引擎盖固定静态区域）
    mask[int(h*0.75):, :] = 255
    return mask


class PIDNetMaskNode:
    def __init__(self):
        rospy.init_node("pidnet_mask_node")

        image_topic = rospy.get_param("~image_topic", "/zed2/camera/left/image_raw")
        model_path  = rospy.get_param("~model_path",
                                       "/mnt/data/lxx/pidnet/PIDNet_S_Cityscapes_val.pt")
        device_str  = rospy.get_param("~device", "cuda:0")
        self.erosion = rospy.get_param("~erosion", 4)

        self.device = torch.device(device_str)
        self.model  = load_pidnet(model_path, self.device)
        self.bridge = CvBridge()

        self.pub_mask  = rospy.Publisher("/semantic_mask", Image, queue_size=5)
        self.pub_ratio = rospy.Publisher("/mask_ratio",    Float32, queue_size=5)

        rospy.Subscriber(image_topic, Image, self.img_callback, queue_size=2)
        rospy.loginfo(f"[PIDNet] 订阅图像话题: {image_topic}")
        rospy.loginfo(f"[PIDNet] 动态类别: person, car, truck, bus, motorcycle, bicycle")

    def img_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"[PIDNet] 图像解码失败: {e}")
            return

        h, w = img_bgr.shape[:2]

        # 预处理
        inp = preprocess(img_bgr).to(self.device)

        # 推理
        with torch.no_grad():
            out = self.model(inp)

        # 取最后一个输出（PIDNet返回list）
        if isinstance(out, (list, tuple)):
            out = out[-1]

        # 上采样到原始分辨率
        out = F.interpolate(out, size=(h, w),
                            mode='bilinear', align_corners=True)

        # 取argmax得到语义图
        seg_map = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 构建mask
        mask = build_mask(seg_map, h, w, self.erosion)

        # 发布mask
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        # 发布动态像素比例
        ratio = float((mask == 0).sum()) / (h * w)
        self.pub_ratio.publish(Float32(data=ratio))

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = PIDNetMaskNode()
    node.spin()
