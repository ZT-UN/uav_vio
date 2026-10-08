cmake_minimum_required(VERSION 2.8.3)
project(semantic_vio)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# ============================================================
# 依赖说明（需根据实际环境修改路径）
#   Ceres:      默认搜索系统路径，也可通过 CMAKE_PREFIX_PATH 指定
#   ONNX Runtime: 默认 /usr/local/onnxruntime
#   camera_models: 本目录下的 catkin 包，需先编译或指向已有安装
# ============================================================

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
set(ONNXRUNTIME_ROOTDIR "/usr/local/onnxruntime")
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

include_directories(${PROJECT_SOURCE_DIR}/featureTracker/ort_include/)
include_directories(${PROJECT_SOURCE_DIR}/estimator/)
include_directories(${PROJECT_SOURCE_DIR}/featureTracker/)
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

# ── 核心库 ──
add_library(semantic_vio_lib
    estimator/parameters.cpp
    estimator/estimator.cpp
    estimator/feature_manager.cpp
    factor/pose_local_parameterization.cpp
    factor/projectionTwoFrameOneCamFactor.cpp
    factor/projectionTwoFrameTwoCamFactor.cpp
    factor/projectionOneFrameTwoCamFactor.cpp
    factor/marginalization_factor.cpp
    utility/utility.cpp
    utility/visualization.cpp
    utility/CameraPoseVisualization.cpp
    initial/solve_5pts.cpp
    initial/initial_aligment.cpp
    initial/initial_sfm.cpp
    initial/initial_ex_rotation.cpp
    featureTracker/feature_tracker.cpp
    featureTracker/feature_tracker_dpl.cpp
    featureTracker/extractor_matcher_dpl.cpp
    featureTracker/transform_dpl.cpp)
target_link_libraries(semantic_vio_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} /usr/local/onnxruntime/lib/libonnxruntime.so)

# ── 可执行文件 ──
get_filename_component(PROJECT_SOURCE_DIR_ABS "${CMAKE_SOURCE_DIR}" ABSOLUTE)
add_executable(semantic_vio_node supervins_main.cpp)
target_compile_definitions(semantic_vio_node PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR_ABS}")
target_link_libraries(semantic_vio_node semantic_vio_lib) 

# add_executable(kitti_odom_test src/KITTIOdomTest.cpp )
# target_link_libraries(kitti_odom_test vins_lib -pthread) 

# add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
# target_link_libraries(kitti_gps_test vins_lib -pthread) 

