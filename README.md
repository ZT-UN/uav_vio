# semantic_vio — 双目 SuperPoint + LightGlue + PIDNet 语义 VIO

基于 SuperVINS 框架，整理的双目语义视觉-惯性里程计完整代码包。

## 目录结构

```
semantic_vio/
├── CMakeLists.txt              ← catkin 构建配置 (项目名: semantic_vio)
├── package.xml                 ← ROS 包描述
├── supervins_main.cpp          ← ROS 节点入口 (mask_callback + sync_process)
│
├── estimator/                  ← 后端估计器
│   ├── estimator.cpp/h         ← 滑动窗口 BA 优化
│   ├── parameters.cpp/h        ← YAML 参数读取
│   └── feature_manager.cpp/h   ← 特征管理 (视差/三角化/异常剔除)
│
├── featureTracker/              ← 视觉前端 ★核心★
│   ├── feature_tracker_dpl.cpp/h   ← 主前端: 提取→匹配→三层语义过滤
│   ├── feature_tracker.cpp/h       ← 原始 VINS 前端 (备用)
│   ├── extractor_matcher_dpl.cpp/h ← SuperPoint/DISK + LightGlue ONNX 封装
│   ├── transform_dpl.cpp/h         ← 图像变换辅助
│   └── ort_include/                ← ONNX Runtime C++ 头文件
│
├── factor/                     ← ceres 优化因子
│   ├── projectionTwoFrameTwoCamFactor.cpp/h  ← 双目投影因子
│   ├── projectionTwoFrameOneCamFactor.cpp/h  ← 单目投影因子
│   ├── projectionOneFrameTwoCamFactor.cpp/h
│   ├── projection_factor.cpp/h
│   ├── imu_factor.h            ← IMU 预积分因子
│   ├── marginalization_factor.cpp/h ← 边缘化因子
│   └── ...
│
├── initial/                    ← 初始化
│   ├── initial_aligment.cpp/h  ← 视觉-惯性对齐
│   ├── initial_ex_rotation.cpp/h ← 外参旋转标定
│   ├── initial_sfm.cpp/h       ← SfM 初始化
│   └── solve_5pts.cpp/h        ← 5点法本质矩阵
│
├── utility/                    ← 工具函数
│   ├── utility.cpp/h           ← 坐标变换/数学工具
│   ├── visualization.cpp/h     ← ROS 可视化
│   ├── CameraPoseVisualization.cpp/h ← 相机位姿可视化
│   └── tic_toc.h               ← 计时器
│
├── factor/                     ← 优化因子 (投影/IMU/边缘化)
├── config/                     ← 配置文件
│   ├── extraction_only.yaml    ← 主配置 (UrbanNav)
│   ├── cam0_urbannav.yaml      ← 左目内参
│   └── cam1_urbannav.yaml      ← 右目内参
│
├── launch/                     ← 启动文件
│   └── match_then_filter.launch
│
├── pidnet/                     ← PIDNet 语义分割
│   └── pidnet_mask_node.py     ← ROS 节点: 订阅左目→发布 mask
│
├── weights_dpl/                ← 权重文件 (需手动放置)
│   ├── superpoint.onnx         ← SuperPoint 特征提取
│   ├── superpoint_lightglue_fused.onnx ← SP+LG 融合 (GPU)
│   └── superpoint_voc.bin      ← 回环检测词汇 (可选)
│
└── camera_models/              ← camodocal 相机模型库 (catkin 依赖)
```

## 核心数据流

```
左目图像 ──→ superpoint.onnx ──→ cur_pts + 描述子
                                       │
                                       ├── 回环检测用: cur_dplpts_descriptors_raw (未过滤)
                                       │
                                       ├── LightGlue 前后帧匹配
                                       │     superpoint_lightglue_fused.onnx
                                       │
                                       ├── Layer 1: PIDNet mask 直接删除语义动态特征
                                       │     mask==0 → 从 temp 向量删除
                                       │
                                       ├── Layer 2: IMU 运动一致性几何验证
                                       │     is_sem_dyn AND Sampson残差>IQR阈值 → 删除
                                       │
                                       ├── Layer 3: 新特征屏蔽 (动态区域不创建track)
                                       │
                                       ├── 左目去畸变 + 速度计算
                                       │
                                       ├── 右目独立提取 → 左右 LightGlue 匹配 → 双目深度
                                       │
                                       └── featureFrame → 后端 BA 优化 → 位姿输出
```

## PIDNet 语义剔除详细流程

### 产生 mask (pidnet_mask_node.py)
```
左目图像 (BGR) → PIDNet-S 推理 → seg_map[19类]
  → 标记动态类: person/car/truck/bus/motorcycle/bicycle
  → 椭圆核腐蚀 4px (去边缘)
  → 底部 25% 屏蔽 (引擎盖)
  → 发布 /semantic_mask (255=静态, 0=动态)
```

### 接收 mask (supervins_main.cpp)
```
mask_callback() → mask_buf 队列
sync_process(): 三队列时间戳同步 (img0 / img1 / mask)
  → estimator.setSemanticMask() → featureTracker.semantic_mask_ = mask
  → estimator.inputImage()      → trackImage_dpl() 使用 semantic_mask_
```

## 编译方法

```bash
# 前置依赖
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport
# Ceres, OpenCV, ONNX Runtime, Eigen3

# 在工作区中编译
cd ~/catkin_ws/src
ln -s /path/to/semantic_vio .
cd ~/catkin_ws
catkin_make -j8
source devel/setup.bash
```

## 运行方法

```bash
# 终端1: roscore
roscore

# 终端2: PIDNet 语义分割
python semantic_vio/pidnet/pidnet_mask_node.py \
    _image_topic:=/zed2/camera/left/image_raw \
    _model_path:=/path/to/PIDNet_S_Cityscapes_val.pt

# 终端3: semantic_vio
roslaunch semantic_vio match_then_filter.launch

# 终端4: 播放 bag
rosbag play UrbanNav-HK_TST-20210517_sensors.bag --clock -r 0.5
```

## 权重文件

| 文件 | 大小 | 用途 |
|------|------|------|
| superpoint.onnx | ~5MB | SuperPoint 特征提取 |
| superpoint_lightglue_fused.onnx | ~50MB | SP+LG 联合匹配 |
| PIDNet_S_Cityscapes_val.pt | ~30MB | Cityscapes 语义分割 |

权重文件路径在 config/extraction_only.yaml 中配置。

## 关键参数 (extraction_only.yaml)

| 参数 | 值 | 说明 |
|------|-----|------|
| estimate_extrinsic | 1 | 在线外参估计 |
| body_T_cam0 tx | -0.3427m | 左目外参平移x |
| max_cnt | 200 | 每帧最多特征数 |
| min_dist | 20 | 特征最小间距 (pixel) |
| image | 672×376 | UrbanNav ZED2 分辨率 |

## 实验历史

| 实验 | ATE (SE3) | 策略 |
|------|-----------|------|
| 基线 (无处理) | 39.56m | 纯几何 |
| PID直接删除 | 32.10m | 匹配后 mask==0 硬删除 |
| 0706 匹配后删 | 50.93m | (外参/配置问题) |
| PIDNet标记+IMU验证 | 待测 | mask标记 → IMU残差判断 |

## 修改记录

- 2026-07-13: 从 `vins_clean_ws` 整理为独立 `semantic_vio` 代码包
  - 去掉 CMakeLists.txt 中 `src/` 前缀
  - 项目名改为 `semantic_vio`
  - 配置文件路径改为相对路径
