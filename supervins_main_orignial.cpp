/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

Estimator estimator;
//数据缓存队列，先进先出
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
queue<sensor_msgs::ImageConstPtr> mask_buf;
//缓冲锁
std::mutex m_buf;

// ── [Semantic Mask Sync] non-blocking exact-or-bypass ─────────────
// Stereo is always allowed to enter VIO once the left/right timestamps match.
// A semantic mask is used only when an already-arrived mask matches the image
// timestamp within g_mask_sync_tolerance. Otherwise this frame falls back to
// ordinary visual+IMU processing with an explicitly cleared semantic mask.
static double g_mask_sync_tolerance = 0.02;
static size_t g_sync_queue_limit = 500;
static uint64_t g_mask_matched = 0;
static uint64_t g_semantic_fallback = 0;
static uint64_t g_stale_masks = 0;
static uint64_t g_stereo_accepted = 0;
// ──────────────────────────────────────────────────────────────────



/// @brief 左目图像回调函数
/// @param img_msg 
void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    //上锁，防止这时候其他函数读取
    m_buf.lock();
    //将数据存入缓冲区
    img0_buf.push(img_msg);
    while (img0_buf.size() > g_sync_queue_limit)
        img0_buf.pop();
    //解锁，这之后其他函数可以读取
    m_buf.unlock();
}

//右目图像回调函数，同左目类似
void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    while (img1_buf.size() > g_sync_queue_limit)
        img1_buf.pop();
    m_buf.unlock();
}


// ── [Semantic Mask] 语义遮罩回调 ─────────────────────────────────
/// 接收来自 semantic_mask_node.py 的 YOLO 分割遮罩
/// 格式: mono8, 0=动态区域, 255=静态区域
void mask_callback(const sensor_msgs::ImageConstPtr &msg)
{
    std::lock_guard<std::mutex> lk(m_buf);
    mask_buf.push(msg);
    while (mask_buf.size() > g_sync_queue_limit)
    {
        mask_buf.pop();
        ++g_stale_masks;
    }
}
// ──────────────────────────────────────────────────────────────────
//将ROS图像消息格式转化为OpenCV格式 cv::Mat
cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
//Synchronize the left and right eye images and input the synchronized images into the estimator 
void sync_process()
{
    while(1)
    {
        if(STEREO)
        {
            cv::Mat image0, image1, semantic_mask;
            double time = 0;
            double mask_time = -1.0;
            bool has_semantic_mask = false;

            {
                std::lock_guard<std::mutex> lk(m_buf);
                if (!img0_buf.empty() && !img1_buf.empty())
                {
                    const double time0 = img0_buf.front()->header.stamp.toSec();
                    const double time1 = img1_buf.front()->header.stamp.toSec();

                    // Keep the original strict stereo synchronization unchanged.
                    if(time0 < time1 - 0.003)
                    {
                        img0_buf.pop();
                        ROS_WARN_STREAM_THROTTLE(
                            5.0,
                            "[MaskSyncNB] throw img0: stereo timestamp mismatch");
                    }
                    else if(time0 > time1 + 0.003)
                    {
                        img1_buf.pop();
                        ROS_WARN_STREAM_THROTTLE(
                            5.0,
                            "[MaskSyncNB] throw img1: stereo timestamp mismatch");
                    }
                    else
                    {
                        // Once stereo is synchronized, ALWAYS accept this pair.
                        time = time0;
                        image0 = getImageFromMsg(img0_buf.front());
                        image1 = getImageFromMsg(img1_buf.front());

                        if (SEMANTIC_FILTER_MODE > 0)
                        {
                            // Drop masks that are already too old for this image.
                            while (!mask_buf.empty() &&
                                   mask_buf.front()->header.stamp.toSec() <
                                       time0 - g_mask_sync_tolerance)
                            {
                                mask_buf.pop();
                                ++g_stale_masks;
                            }

                            // Use a mask only if it has ALREADY arrived and its
                            // source-image timestamp matches this stereo pair.
                            if (!mask_buf.empty())
                            {
                                const double candidate_mask_time =
                                    mask_buf.front()->header.stamp.toSec();

                                if (std::fabs(candidate_mask_time - time0) <=
                                    g_mask_sync_tolerance)
                                {
                                    mask_time = candidate_mask_time;
                                    semantic_mask =
                                        getImageFromMsg(mask_buf.front());
                                    mask_buf.pop();
                                    has_semantic_mask = true;
                                    ++g_mask_matched;
                                }
                            }
                        }

                        // IMPORTANT: semantic availability never controls whether
                        // the stereo pair enters VIO.
                        img0_buf.pop();
                        img1_buf.pop();
                        ++g_stereo_accepted;

                        if (SEMANTIC_FILTER_MODE > 0 && !has_semantic_mask)
                            ++g_semantic_fallback;
                    }
                }
            }

            if(!image0.empty())
            {
                if (SEMANTIC_FILTER_MODE == 0)
                {
                    // Baseline behavior remains unchanged.
                    estimator.setSemanticMask(cv::Mat(), -1.0);
                    estimator.inputImage(time, image0, image1);

                    ROS_INFO_STREAM_THROTTLE(
                        5.0,
                        "[MaskSyncNB] mode0 accepted=" << g_stereo_accepted
                        << " semantic bypass");
                }
                else
                {
                    if (has_semantic_mask)
                    {
                        estimator.setSemanticMask(semantic_mask, mask_time);
                    }
                    else
                    {
                        // Explicitly clear the previous frame's mask so that no
                        // stale semantic state can leak into this image.
                        estimator.setSemanticMask(cv::Mat(), -1.0);
                    }

                    estimator.inputImage(time, image0, image1);

                    ROS_INFO_STREAM_THROTTLE(
                        5.0,
                        "[MaskSyncNB] accepted=" << g_stereo_accepted
                        << " mask_matched=" << g_mask_matched
                        << " fallback_no_mask=" << g_semantic_fallback
                        << " stale_mask=" << g_stale_masks
                        << " has_mask=" << (has_semantic_mask ? 1 : 0)
                        << " mask_dt="
                        << (has_semantic_mask
                                ? std::fabs(mask_time - time)
                                : -1.0));
                }
            }
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            {
                std::lock_guard<std::mutex> lk(m_buf);
                if(!img0_buf.empty())
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                }
            }
            if(!image.empty())
                estimator.inputImage(time, image);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

//IMU回调函数,imu_buf没用，直接把IMU数据往estimator里输入
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}

//特征回调函数，在接收到新的特征时调用,这个函数在VINS-Mono中使用，在VINS-Fusion中feature_tracker内置在了estimator中
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        //ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    
    ros::init(argc, argv, "supervins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2)
    {
        printf("please intput: rosrun supervins supervins_node [config file] \n"
               "for example: upervins supervins_node"
               "~/catkin_ws/src/SuperVINS/config/euroc/euroc_mono_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    //读取参数，包括相机参数、IMU参数、特征参数等等，读入到parameters.cpp这个文件里的变量里
    // parameters, including camera parameters, imu parameters, feature parameters, etc., into variables in the file parameters.cpp
    readParameters(config_file);

    n.param("mask_sync_tolerance", g_mask_sync_tolerance, 0.02);
    int sync_queue_limit = static_cast<int>(g_sync_queue_limit);
    n.param("mask_sync_queue_limit", sync_queue_limit, 500);
    g_sync_queue_limit = static_cast<size_t>(std::max(10, sync_queue_limit));
    if (SEMANTIC_FILTER_MODE == 0)
    {
        ROS_INFO(
            "[MaskSyncNB] mode0: semantic mask bypass enabled; "
            "stereo timestamp synchronization only");
    }
    else
    {
        ROS_INFO(
            "[MaskSyncNB] non-blocking exact-or-bypass: "
            "tolerance=%.3fs queue_limit=%zu; "
            "stereo frames are never dropped for missing semantic masks",
            g_mask_sync_tolerance,
            g_sync_queue_limit);
    }

    // 路径配置
    // Path configuration
    std::string project_source_dir = PROJECT_SOURCE_DIR;
    extractor_weight_global_path = project_source_dir + "/" + extractor_weight_relative_path;
    matcher_weight_global_path = project_source_dir + "/" + matcher_weight_relative_path;
    // 仅对相对路径添加前缀（绝对路径直接使用）
    if (VINS_RESULT_PATH.empty() || VINS_RESULT_PATH[0] != '/')
        VINS_RESULT_PATH = project_source_dir + "/" + VINS_RESULT_PATH;
    // else: 绝对路径保持不变

    //给estimator设置参数，因为一些参数可能被优化，所以可能会重置参数，注意，如果开启了多线程模式，在setParameter()中就已经将状态估计函数放入一个独立线程运行了
    //Set parameters for the estimator. Because some parameters may be optimized, the parameters may be reset. Note that if the multi-thread mode is turned on, the state estimation function has been put into an independent thread to run in set parameter().
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);
    // 订阅IMU话题的接收器
    // Receiver that subscribes to imu topics
    ros::Subscriber sub_imu;
    if(USE_IMU)
    {
        // 接收来自IMU_TOPIC的IMU数据，每接收一个就送入estimator
        // Receive imu data from imu topic, and send each one to the estimator
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    //接收特征、图像
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 500, img0_callback);
    ros::Subscriber sub_img1;
    if(STEREO)
    {
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 500, img1_callback);
    }
    //接收系统重启、IMU切换、图像切换信号
    //Receive system restart, imu switching, image switching signals
    
    // ── [Semantic Mask] 条件订阅 ───────────────────────────────────
    ros::Subscriber sub_mask;

    if (SEMANTIC_FILTER_MODE > 0)
    {
        sub_mask = n.subscribe(
            "/semantic_mask",
            100,
            mask_callback
        );

        ROS_INFO(
            "[SuperVINS] Semantic mask subscriber: ENABLED"
        );
    }
    else
    {
        ROS_INFO(
            "[SuperVINS] Semantic mask subscriber: DISABLED "
            "(mode0 baseline)"
        );
    }
    // ────────────────────────────────────────────────────────────────
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

    //同步处理
    //Synchronization
    std::thread sync_thread{sync_process};
    ros::spin();

    return 0;

    
    
}
