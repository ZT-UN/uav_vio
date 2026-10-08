/*
 * @Author: Hongkun Luo
 * @Date: 2024-11-05 21:16:57
 * @LastEditors: luohongk luohongkun@whu.edu.cn
 * @Description: 
 * 
 * Hongkun Luo
 */
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

#include "feature_tracker_dpl.h"
void FeatureTrackerDPL::initializeExtractorMatcher(int extractor_type_, string &extractor_weight_path, string &matcher_weight_path, float matcher_threshold = 0.5)
{
    extractor_type = extractor_type_;
    FeatureExtractorDPL = std::make_shared<Extractor_DPL>();
    FeatureExtractorDPL->initialize(extractor_weight_path, extractor_type_);

    FeatureMatcherDPL = std::make_shared<Matcher_DPL>();
    FeatureMatcherDPL->initialize(matcher_weight_path, extractor_type_, matcher_threshold);

    if (extractor_type_ == SUPERPOINT)
    {
        descriptor_size = SUPERPOINT_SIZE;
    }
    else if (extractor_type_ == DISK)
    {
        descriptor_size = DISK_SIZE;
    }
}

cv::Mat FeatureTrackerDPL::Extractor_PreProcess(const cv::Mat &Image, float &scale)
{
    float temp_scale = scale;
    cv::Mat tempImage = Image.clone();
    // std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;

    std::string fn = "max";
    std::string interp = "area";
    cv::Mat resize_img = ResizeImage(tempImage, IMAGE_SIZE_DPL, scale, fn, interp);
    cv::Mat resultImage = NormalizeImage(resize_img);
    // if (cfg.extractorType == "superpoint")
    //{
    // std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale" << std::endl;
    // resultImage = RGB2Grayscale(resultImage);
    //}
    // std::cout << "[INFO] Scale from " << temp_scale << " to " << scale << std::endl;

    return resultImage;
}

void FeatureTrackerDPL::goodFeaturesToTrack_dpl(cv::Mat img, vector<cv::Point2f> &pts, vector<pair<cv::Point2f, vector<float>>> &dplpts_descriptors, int max_num, double extractor_threshold, int radius, cv::Mat &mask)
{
    cv::Mat im = img.clone();
    cv::Mat im_preprocessed = Extractor_PreProcess(im, FeatureExtractorDPL->scale);
    std::pair<std::vector<cv::Point2f>, float *> result_dplpts_descriptors = FeatureExtractorDPL->extract_featurepoints(im_preprocessed);
    int n = result_dplpts_descriptors.first.size();
    for (int i = 0; i < n; i++)
    {
        cv::Point2f dplpt = result_dplpts_descriptors.first[i];
        cv::Point2f pt = cv::Point2f((dplpt.x + 0.5) / FeatureExtractorDPL->scale - 0.5, (dplpt.y + 0.5) / FeatureExtractorDPL->scale - 0.5);
        if (!inBorder(pt))
            continue;
        if (mask.at<uchar>(pt) == 255)
        {
            std::vector<float> descriptor(result_dplpts_descriptors.second + i * descriptor_size, result_dplpts_descriptors.second + (i + 1) * descriptor_size);
            pts.push_back(pt);
            dplpts_descriptors.push_back(make_pair(dplpt, descriptor));
        }
    }
}

void FeatureTrackerDPL::extract_features_dpl(cv::Mat img, vector<cv::Point2f> &pts, vector<pair<cv::Point2f, vector<float>>> &dplpts_descriptors)
{
    cv::Mat im = img.clone();
    cv::Mat im_preprocessed = Extractor_PreProcess(im, FeatureExtractorDPL->scale);
    std::pair<std::vector<cv::Point2f>, float *> result_dplpts_descriptors = FeatureExtractorDPL->extract_featurepoints(im_preprocessed);

    int n = result_dplpts_descriptors.first.size();
    for (int i = 0; i < n; i++)
    {

        cv::Point2f dplpt = result_dplpts_descriptors.first[i];
        cv::Point2f pt = cv::Point2f((dplpt.x + 0.5) / FeatureExtractorDPL->scale - 0.5, (dplpt.y + 0.5) / FeatureExtractorDPL->scale - 0.5);
        if (!inBorder(pt))
            continue;
        std::vector<float> descriptor(result_dplpts_descriptors.second + i * descriptor_size, result_dplpts_descriptors.second + (i + 1) * descriptor_size);
        pts.push_back(pt);
        dplpts_descriptors.push_back(make_pair(dplpt, descriptor));
    }
    // std::cout<<dplpts_descriptors[0].first<<endl;
    // std::cout<<dplpts_descriptors[0].second<<endl;
}

bool FeatureTrackerDPL::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

void FeatureTrackerDPL::reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void FeatureTrackerDPL::reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTrackerDPL::FeatureTrackerDPL()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

void FeatureTrackerDPL::match_features_dpl(cv::Mat prev_img_, cv::Mat cur_img_, vector<pair<cv::Point2f, vector<float>>> &prev_dplpts_descriptors_, vector<pair<cv::Point2f, vector<float>>> &cur_dplpts_descriptors_, vector<pair<int, int>> &result_matches,double &ransacReprojThreshold)
{
    // ============================================================
    // B0 ORIGINAL BASELINE
    //   SuperPoint/DISK descriptors + LightGlue pairwise matching
    //   + fundamental-matrix RANSAC only.
    //
    // Deliberately disabled in B0:
    //   - IQR adaptive secondary filtering
    //   - prediction-guided/local-window matching
    //   - TrackManager / Lost-track recovery
    //
    // Track-ID inheritance remains because it is required by VIO:
    // a successful pairwise match inherits the previous ID; an
    // unmatched newly accepted keypoint receives a new ID.
    // ============================================================
    result_matches.clear();

    const int n_pre = static_cast<int>(prev_dplpts_descriptors_.size());
    const int n_cur = static_cast<int>(cur_dplpts_descriptors_.size());

    if (n_pre == 0 || n_cur == 0)
        return;

    vector<cv::Point2f> prev_dplpts, cur_dplpts;
    prev_dplpts.reserve(n_pre);
    cur_dplpts.reserve(n_cur);

    // Keep the original storage style used by the project.
    std::vector<float> prev_descriptors(n_pre * descriptor_size);
    std::vector<float> cur_descriptors(n_cur * descriptor_size);

    for (int i = 0; i < n_pre; ++i)
    {
        prev_dplpts.push_back(prev_dplpts_descriptors_[i].first);
        const vector<float> &desc = prev_dplpts_descriptors_[i].second;
        const int base = i * descriptor_size;
        for (int j = 0; j < descriptor_size && j < static_cast<int>(desc.size()); ++j)
            prev_descriptors[base + j] = desc[j];
    }

    for (int i = 0; i < n_cur; ++i)
    {
        cur_dplpts.push_back(cur_dplpts_descriptors_[i].first);
        const vector<float> &desc = cur_dplpts_descriptors_[i].second;
        const int base = i * descriptor_size;
        for (int j = 0; j < descriptor_size && j < static_cast<int>(desc.size()); ++j)
            cur_descriptors[base + j] = desc[j];
    }

    vector<cv::Point2f> prev_dplpts_normalized =
        FeatureMatcherDPL->pre_process(prev_dplpts, prev_img_.rows, prev_img_.cols);
    vector<cv::Point2f> cur_dplpts_normalized =
        FeatureMatcherDPL->pre_process(cur_dplpts, cur_img_.rows, cur_img_.cols);

    // Raw pairwise LightGlue correspondences.
    vector<pair<int, int>> raw_matches = FeatureMatcherDPL->match_featurepoints(
        prev_dplpts_normalized,
        cur_dplpts_normalized,
        prev_descriptors.data(),
        cur_descriptors.data());

    if (raw_matches.empty())
        return;

    vector<cv::Point2f> points1, points2;
    points1.reserve(raw_matches.size());
    points2.reserve(raw_matches.size());

    for (const auto &match : raw_matches)
    {
        points1.push_back(prev_dplpts_normalized[match.first]);
        points2.push_back(cur_dplpts_normalized[match.second]);
    }

    // F-matrix estimation needs at least 8 correspondences.  This guard
    // is retained only as a crash-prevention measure; it does not change
    // normal B0 matching behavior when enough correspondences exist.
    if (points1.size() < 8 ||
        points2.size() < 8 ||
        points1.size() != points2.size())
    {
        ROS_WARN(
            "[B0-FGuard] skip FundamentalMat: raw=%zu points1=%zu points2=%zu; "
            "fallback to raw LightGlue matches",
            raw_matches.size(), points1.size(), points2.size());
        result_matches = raw_matches;
        return;
    }

    std::vector<uchar> inliersMask;
    cv::Mat fundamentalMatrix;

    try
    {
        // Keep the project's existing normalized-coordinate RANSAC setting.
        // ransacReprojThreshold is intentionally not used here because the
        // previous DPL implementation used the fixed 0.015 threshold.
        (void)ransacReprojThreshold;
        fundamentalMatrix = cv::findFundamentalMat(
            points1,
            points2,
            cv::FM_RANSAC,
            0.015,
            0.99,
            inliersMask);
    }
    catch (const cv::Exception &e)
    {
        ROS_ERROR(
            "[B0-FGuard] findFundamentalMat exception: raw=%zu error=%s; "
            "fallback to raw LightGlue matches",
            raw_matches.size(), e.what());
        result_matches = raw_matches;
        return;
    }

    if (fundamentalMatrix.empty() || inliersMask.size() != raw_matches.size())
    {
        ROS_WARN(
            "[B0-FGuard] invalid FundamentalMat result: raw=%zu mask=%zu; "
            "fallback to raw LightGlue matches",
            raw_matches.size(), inliersMask.size());
        result_matches = raw_matches;
        return;
    }

    // B0 stops here: keep RANSAC inliers directly.  No IQR/top-15 stage.
    for (size_t i = 0; i < raw_matches.size(); ++i)
    {
        if (inliersMask[i])
            result_matches.push_back(raw_matches[i]);
    }

    ROS_INFO(
        "[B0-MatchDiag] raw_lightglue=%zu ransac_inliers=%zu",
        raw_matches.size(), result_matches.size());
}

void FeatureTrackerDPL::match_with_predictions_dpl(cv::Mat prev_img_, cv::Mat cur_img_, vector<pair<cv::Point2f, vector<float>>> &prev_dplpts_descriptors_, vector<pair<cv::Point2f, vector<float>>> &cur_dplpts_descriptors_, vector<cv::Point2f> &predict_pts_, vector<cv::Point2f> &cur_pts_, vector<pair<int, int>> &result_matches,double &ransacReprojThreshold)
{
    // B0 compatibility wrapper.
    // Prediction data may still be produced elsewhere in VINS, but the B0
    // baseline MUST NOT use it to guide matching.  Therefore this function
    // deliberately delegates to the ordinary pairwise SP+LG matcher.
    (void)predict_pts_;
    (void)cur_pts_;

    match_features_dpl(
        prev_img_,
        cur_img_,
        prev_dplpts_descriptors_,
        cur_dplpts_descriptors_,
        result_matches,
        ransacReprojThreshold);
}

/// @brief 设置特征提取掩码，并根据追踪次数排序
/// @brief sets the feature extraction mask and sorts according to the number of tracking
void FeatureTrackerDPL::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    // 将当前帧特征点按照（追踪次数，（特征点坐标，id））存储
    // Store the feature points of the current frame according to (number of tracking, (feature point
    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    // 按照追踪次数从大到小排序
    // Sort by tracking number from largest to smallest
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         { return a.first > b.first; });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    // 将排序后的特征点存回去
    // Save the sorted feature points back
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);

            // 如果掩码内特征点像素位置是255，则在掩码周围设置一个半径为MIN_DIST的“禁止提取区域”,像素置为0
            // If the pixel position of the feature point in the mask is 255, set a "prohibited extraction area" with a radius of min dist around the mask, and set the pixel to 0
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1); 
        }
    }
}

cv::Mat FeatureTrackerDPL::setMask_dpl(vector<cv::Point2f> &matched_points, int radius = MIN_DIST)
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    for (cv::Point2f &pt : matched_points)
    {
        if (mask.at<uchar>(pt) == 255)
        {
            cv::circle(mask, pt, radius, 0, -1); 
            // 如果掩码内特征点像素位置是255，则在掩码周围设置一个半径为MIN_DIST的“禁止提取区域”,像素置为0
            // If the pixel position of the feature point in the mask is 255, set a "prohibited extraction area" with a radius of min dist around the mask, and set the pixel to 0
        }
    }

    return mask.clone();
}

double FeatureTrackerDPL::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

vector<uchar> FeatureTrackerDPL::imuMotionFilter(
    const vector<cv::Point2f> &prev_px,
    const vector<cv::Point2f> &cur_px,
    const vector<bool> &is_semantic_dynamic,
    double *adaptive_threshold,
    vector<double> *motion_residuals)
{
    const size_t n = prev_px.size();
    vector<uchar> status(n, 1);
    if (adaptive_threshold) *adaptive_threshold = 0.0;
    if (motion_residuals) motion_residuals->assign(n, -1.0);
    if (!has_imu_rotation_ || n == 0 || cur_px.size() != n ||
        is_semantic_dynamic.size() != n || m_camera.empty())
        return status;

    vector<Eigen::Vector3d> x1(n), x2(n);
    for (size_t i = 0; i < n; ++i)
    {
        m_camera[0]->liftProjective(Eigen::Vector2d(prev_px[i].x, prev_px[i].y), x1[i]);
        m_camera[0]->liftProjective(Eigen::Vector2d(cur_px[i].x, cur_px[i].y), x2[i]);
        x1[i].normalize();
        x2[i].normalize();
    }

    vector<Eigen::Vector3d> rows;
    for (size_t i = 0; i < n; ++i)
    {
        if (is_semantic_dynamic[i]) continue;
        Eigen::Vector3d a = (delta_R_cam_ * x1[i]).cross(x2[i]);
        if (a.norm() > 1e-6) rows.push_back(a.normalized());
    }

    if (rows.size() < (size_t)IMU_MOTION_MIN_BG_PTS)
    {
        if (SEMANTIC_IMU_DEBUG)
        {
            ROS_WARN("[Stereo-Semantic-IMU] only %zu static matches, need %d; skip IMU soft filter and keep all candidates",
                     rows.size(), IMU_MOTION_MIN_BG_PTS);
        }
        return status;
    }

    auto solveTranslation = [](const vector<Eigen::Vector3d> &eq) {
        Eigen::MatrixXd A(eq.size(), 3);
        for (size_t i = 0; i < eq.size(); ++i) A.row(i) = eq[i].transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        return svd.matrixV().col(2).normalized();
    };
    auto skew = [](const Eigen::Vector3d &v) {
        Eigen::Matrix3d S;
        S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
        return S;
    };
    auto percentile = [](vector<double> v, double q) {
        if (v.empty()) return 0.0;
        size_t k = std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1)));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    };

    Eigen::Vector3d t = solveTranslation(rows);
    Eigen::Matrix3d E = skew(t) * delta_R_cam_;
    auto residual = [&](size_t i) {
        Eigen::Vector3d l2 = E * x1[i], l1 = E.transpose() * x2[i];
        double den = l1.head<2>().squaredNorm() + l2.head<2>().squaredNorm();
        return den < 1e-12 ? 0.0 : std::abs(x2[i].dot(l2)) / std::sqrt(den);
    };

    vector<double> bg;
    for (size_t i = 0; i < n; ++i) if (!is_semantic_dynamic[i]) bg.push_back(residual(i));
    double q1 = percentile(bg, 0.25), q3 = percentile(bg, 0.75);
    double refine_thr = std::max(IMU_MOTION_THR_MIN,
                                 std::min(IMU_MOTION_THR_MAX, q3 + 1.5 * (q3 - q1)));
    vector<Eigen::Vector3d> refined;
    for (size_t i = 0; i < n; ++i)
    {
        if (is_semantic_dynamic[i] || residual(i) > refine_thr) continue;
        Eigen::Vector3d a = (delta_R_cam_ * x1[i]).cross(x2[i]);
        if (a.norm() > 1e-6) refined.push_back(a.normalized());
    }
    if (refined.size() >= 8)
    {
        t = solveTranslation(refined);
        E = skew(t) * delta_R_cam_;
    }

    bg.clear();
    for (size_t i = 0; i < n; ++i) if (!is_semantic_dynamic[i]) bg.push_back(residual(i));
    q1 = percentile(bg, 0.25); q3 = percentile(bg, 0.75);
    const double threshold = std::max(IMU_MOTION_THR_MIN,
                                      std::min(IMU_MOTION_THR_MAX, q3 + 1.5 * (q3 - q1)));
    if (adaptive_threshold) *adaptive_threshold = threshold;
    for (size_t i = 0; i < n; ++i)
    {
        const double r = residual(i);
        if (motion_residuals) (*motion_residuals)[i] = r;
        if (is_semantic_dynamic[i] && r > threshold) status[i] = 0;
    }
    return status;
}

FeatureTrackerDPL::FeatureState FeatureTrackerDPL::updateFeatureState(
    int id, bool semantic_dynamic, double residual, double threshold)
{
    auto &h = feature_history_[id];
    h.last_seen_frame = semantic_frame_index_;
    h.semantic_dynamic.push_back(semantic_dynamic ? 1 : 0);
    while (h.semantic_dynamic.size() > 3) h.semantic_dynamic.pop_front();

    const bool residual_valid = residual >= 0.0 && threshold > 0.0;
    if (residual_valid)
    {
        h.normalized_motion_residual.push_back(residual / threshold);
        while (h.normalized_motion_residual.size() > 3)
            h.normalized_motion_residual.pop_front();
    }

    int semantic_votes = 0;
    for (auto v : h.semantic_dynamic) semantic_votes += v != 0;
    int motion_high = 0, motion_low_relaxed = 0;
    for (double r : h.normalized_motion_residual)
    {
        motion_high += r > 1.0;
        motion_low_relaxed += r < 0.8;
    }

    if (h.state == FeatureState::DYNAMIC)
        return h.state;

    // V3: require persistent semantic evidence (2 of the last 3 frames) and
    // a meaningful motion residual before entering UNCERTAIN.  A single
    // semantic hit must not penalize a stable feature on a parked vehicle.
    if (h.state == FeatureState::STATIC && semantic_votes >= 2)
    {
        if (residual_valid && residual / threshold > 0.8)
            h.state = FeatureState::UNCERTAIN;
    }

    // Confirm dynamic only with both semantic persistence and motion evidence.
    if (semantic_votes >= 2 && motion_high >= 2)
        h.state = FeatureState::DYNAMIC;
    // V3 rescue: two low residuals in a full three-frame history are enough,
    // or all semantic-dynamic votes have disappeared from that history.
    else if (h.state == FeatureState::UNCERTAIN &&
             ((h.normalized_motion_residual.size() == 3 &&
               motion_low_relaxed >= 2) ||
              (h.semantic_dynamic.size() == 3 && semantic_votes == 0)))
        h.state = FeatureState::STATIC;

    return h.state;
}

bool FeatureTrackerDPL::backendUsable(int id) const
{
    // 提取级语义过滤 + imuMotionFilter 已在前端完成动态特征剔除，
    // 不再依赖三状态裁决，所有到达此处的特征均可入后端。
    return true;
}

void FeatureTrackerDPL::pruneFeatureHistory()
{
    for (auto it = feature_history_.begin(); it != feature_history_.end(); )
    {
        if (semantic_frame_index_ - it->second.last_seen_frame > 10)
            it = feature_history_.erase(it);
        else
            ++it;
    }
}



/// @brief 双目特征帧间追踪
/// @param _cur_time
/// @param _img
/// @param _img1
/// @return 图像特征

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTrackerDPL::trackImage_dpl(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    // 当前时间
    // current time
    cur_time = _cur_time;  
    ++semantic_frame_index_;

    ROS_INFO_ONCE(
        "[B0-Baseline] SuperPoint/LightGlue pairwise matching + FM-RANSAC; "
        "IQR=OFF prediction_guidance=OFF TrackManager=OFF LostRecovery=OFF; "
        "TrackIDDiag=ON");

    // 当前左目图像   
    cur_img = _img;  

    // 图像行数 
    // Number of image lines         
    row = cur_img.rows;

    // 图像列数    
    // Number of image columns   
    col = cur_img.cols;

    // 当前右目图像 
    // Current right eye image 
    cv::Mat rightImg = _img1; 

    // 清空当前帧特征点
    // Clear feature points of current frame
    cur_pts.clear();
    cur_dplpts_descriptors.clear();

    // 开始计时
    cout<<"extract feature time"<<endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 使用深度学习进行特征点和描述子提取
    // use deep-learning extractor to extract feature points and descriptors
    extract_features_dpl(cur_img, cur_pts, cur_dplpts_descriptors);

    // ── 提取级语义过滤：已禁用，改为匹配后删除 ──
    // 实验 exp_0706_match_then_filter: 发现PIDNet在香港场景误分类严重(22%像素)
    // 提取前删除导致LightGlue匹配候选不足，ATE从32m恶化到57m
    // 改为在匹配完成后按mask删除（与PID直接删除策略一致）
    // if (!semantic_mask_.empty())
    // {
    //     size_t before = cur_pts.size();
    //     filterBySemanticMask(cur_pts, cur_dplpts_descriptors, semantic_mask_);
    //     ROS_INFO("[Extraction-Filter] semantic mask filtered %zu -> %zu features",
    //              before, cur_pts.size());
    // }

    // 在掩码过滤前保存原始完整描述子，供回环检测使用
    cur_dplpts_descriptors_raw = cur_dplpts_descriptors;

    // Keep extraction arrays intact until temporal/stereo matching has consumed
    // them. Direct semantic deletion is applied synchronously to matched tracks
    // below; dynamic unmatched candidates are rejected before creating IDs.
    // ─────────────────────────────────────────────────────────────

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start; // 计算持续时间

    // 输出时间到控制台
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

    // 打开文件进行保存（以追加模式打开）
    std::ofstream outFile("time_consumption/feature_extraction_matching.txt", std::ios::app);
    if (outFile.is_open()) {
        outFile << duration.count() << ","; // 写入执行时间
        outFile.close(); // 关闭文件
    } else {
        std::cerr << "Unable to open file" << std::endl; // 错误处理
    }

    cout<<"save success"<<endl;

    // 一些变量容器
    // some variables containers
    set<int> cur_matched_indices;

    // ============================================================
    // [TrackIDDiag] Track-ID continuity diagnosis only.
    // This block does NOT change feature extraction, matching,
    // semantic filtering, IMU filtering, or ID assignment logic.
    // "prev_active" is the number of active tracks entering this frame.
    // "diag_prev_matched_indices" records which previous-frame tracks
    // are successfully associated to the current frame by the temporal
    // matcher before semantic/IMU filtering.
    // ============================================================
    const size_t diag_prev_active = prev_pts.size();
    set<int> diag_prev_matched_indices;

    vector<cv::Point2f> temp_prev_pts;
    vector<pair<cv::Point2f, vector<float>>> temp_prev_dplpts_descriptors;
    vector<cv::Point2f> temp_cur_pts;
    vector<pair<cv::Point2f, vector<float>>> temp_cur_dplpts_descriptors;
    vector<int> temp_ids;
    vector<int> temp_track_cnt;

    cout << cur_pts.size() << " points extracted in current frame originally" << endl;

    // 上一帧特征点不能为空，否则没法追踪
    // The feature points of the previous frame cannot be empty, otherwise they cannot be tracked.\

    ROS_INFO_STREAM("prev_pts.size() = " << prev_pts.size());
    
    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<pair<int, int>> matches;

        // ============================================================
        // B0 baseline: always use ordinary pairwise SuperPoint+LightGlue
        // matching.  Any prediction generated by setPrediction() is ignored
        // in this experiment so that the baseline contains no temporal
        // prediction guidance.
        // ============================================================
        auto start_match = std::chrono::high_resolution_clock::now();

        match_features_dpl(
            prev_img,
            cur_img,
            prev_dplpts_descriptors,
            cur_dplpts_descriptors,
            matches,
            ransacReprojThreshold);

        auto end_match = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> match_duration = end_match - start_match;

        ROS_INFO_STREAM("[B0] matches.size() = " << matches.size());

        std::ofstream matchTimeFile(
            "time_consumption/feature_extraction_matching.txt",
            std::ios::app);
        if (matchTimeFile.is_open())
        {
            matchTimeFile << match_duration.count() << std::endl;
            matchTimeFile.close();
        }

        // the number of matched points
        int n_matches = matches.size();
        // cout << n_matches <<" points matched in current frame" <<endl;

        // first, save the successfully matched points in the temporary vector
        for (int i = 0; i < n_matches; i++)
        {
            pair<int, int> match = matches[i];
            temp_prev_pts.push_back(prev_pts[match.first]);
            temp_prev_dplpts_descriptors.push_back(prev_dplpts_descriptors[match.first]);
            temp_cur_pts.push_back(cur_pts[match.second]);
            temp_cur_dplpts_descriptors.push_back(cur_dplpts_descriptors[match.second]);
            temp_ids.push_back(ids[match.first]);
            temp_track_cnt.push_back(track_cnt[match.first] + 1);
            // record the indices of matched points
            cur_matched_indices.insert(match.second);

            // [TrackIDDiag] Record the previous-frame track that
            // successfully inherited its ID through this frame match.
            diag_prev_matched_indices.insert(match.first);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Semantic-IMU dynamic feature filtering
    // mode 0: no semantic filtering
    // mode 1: PIDNet hard delete
    // mode 2: PIDNet candidate + IMU soft delete
    // ─────────────────────────────────────────────────────────────

    cv::Mat local_semantic_mask = semantic_mask_.empty() ? cv::Mat() : semantic_mask_.clone();

    // 仅用于诊断，不改变任何特征过滤逻辑
    const size_t diag_raw_cur_pts = cur_pts.size();
    const size_t diag_matched_before = temp_cur_pts.size();

    // ============================================================
    // [TrackIDDiag] Diagnose track interruption at the temporal
    // matching stage, BEFORE semantic/IMU filtering.  Therefore,
    // these statistics describe SuperPoint + matcher continuity
    // rather than semantic-filter-induced removals.
    // ============================================================
    int diag_prev_lost = 0;
    int diag_lost_le3 = 0;
    int diag_lost_le5 = 0;
    long long diag_lost_track_sum = 0;
    int diag_lost_track_max = 0;

    // track_cnt and prev_pts refer to the same previous-frame active
    // track set at this point.  Use the smaller size defensively.
    const size_t diag_prev_count =
        std::min(diag_prev_active, track_cnt.size());

    for (size_t i = 0; i < diag_prev_count; ++i)
    {
        if (diag_prev_matched_indices.count(static_cast<int>(i)) == 0)
        {
            ++diag_prev_lost;

            const int len = track_cnt[i];
            diag_lost_track_sum += len;
            diag_lost_track_max = std::max(diag_lost_track_max, len);

            if (len <= 3)
                ++diag_lost_le3;
            if (len <= 5)
                ++diag_lost_le5;
        }
    }

    // If prev_pts and track_cnt ever become inconsistent, count the
    // unmatched tail of prev_pts as lost tracks with unknown length.
    if (diag_prev_active > diag_prev_count)
        diag_prev_lost += static_cast<int>(diag_prev_active - diag_prev_count);

    const double diag_track_survival =
        diag_prev_active > 0
            ? static_cast<double>(diag_prev_matched_indices.size()) /
                  static_cast<double>(diag_prev_active)
            : 0.0;

    const double diag_track_break_rate =
        diag_prev_active > 0
            ? static_cast<double>(diag_prev_lost) /
                  static_cast<double>(diag_prev_active)
            : 0.0;

    const int diag_known_lost_count =
        static_cast<int>(diag_prev_count) -
        static_cast<int>(diag_prev_matched_indices.size() > diag_prev_count
                             ? diag_prev_count
                             : diag_prev_matched_indices.size());

    const double diag_lost_mean_len =
        diag_known_lost_count > 0
            ? static_cast<double>(diag_lost_track_sum) /
                  static_cast<double>(diag_known_lost_count)
            : 0.0;

    int diag_hard_deleted = 0;
    int diag_unmatched_candidates = 0;
    int diag_new_semantic_rejected = 0;
    int diag_new_spacing_rejected = 0;
    int diag_new_added = 0;

    vector<bool> is_sem_dyn(temp_cur_pts.size(), false);
    int semantic_candidate_num = 0;

    // mode 1: PIDNet hard delete，保留为消融实验对比组
    if (SEMANTIC_FILTER_MODE == 1 && !local_semantic_mask.empty() && !temp_cur_pts.empty())
    {
        int deleted = 0;

        for (int i = (int)temp_cur_pts.size() - 1; i >= 0; i--)
        {
            int x = cvRound(temp_cur_pts[i].x);
            int y = cvRound(temp_cur_pts[i].y);

            if (x >= 0 && x < local_semantic_mask.cols &&
                y >= 0 && y < local_semantic_mask.rows &&
                local_semantic_mask.at<uchar>(y, x) == 0)
            {
                temp_cur_pts.erase(temp_cur_pts.begin() + i);
                temp_cur_dplpts_descriptors.erase(temp_cur_dplpts_descriptors.begin() + i);
                temp_prev_pts.erase(temp_prev_pts.begin() + i);
                temp_prev_dplpts_descriptors.erase(temp_prev_dplpts_descriptors.begin() + i);
                temp_ids.erase(temp_ids.begin() + i);
                temp_track_cnt.erase(temp_track_cnt.begin() + i);
                deleted++;
            }
        }

        diag_hard_deleted = deleted;

        if (SEMANTIC_IMU_DEBUG && deleted > 0)
        {
            ROS_INFO("[Semantic-hard] removed %d matched features in dynamic regions, %zu remain",
                     deleted, temp_cur_pts.size());
        }
    }

    // mode 2: PIDNet 只做语义动态候选标记，不直接删除
    if (SEMANTIC_FILTER_MODE == 2 && !local_semantic_mask.empty() && !temp_cur_pts.empty())
    {
        is_sem_dyn.assign(temp_cur_pts.size(), false);
        semantic_candidate_num = 0;

        for (size_t i = 0; i < temp_cur_pts.size(); i++)
        {
            int x = cvRound(temp_cur_pts[i].x);
            int y = cvRound(temp_cur_pts[i].y);

            if (x >= 0 && x < local_semantic_mask.cols &&
                y >= 0 && y < local_semantic_mask.rows &&
                local_semantic_mask.at<uchar>(y, x) == 0)
            {
                is_sem_dyn[i] = true;
                semantic_candidate_num++;
            }
        }

        if (SEMANTIC_IMU_DEBUG)
        {
            ROS_INFO("[Semantic-candidate] candidate=%d / total=%zu",
                     semantic_candidate_num, temp_cur_pts.size());
        }
    }

    // mode 2: IMU 几何一致性确认删除
    // 只有 “语义动态候选 + IMU 残差异常” 的点才删除
    if (SEMANTIC_FILTER_MODE == 2 &&
        !temp_cur_pts.empty() &&
        has_imu_rotation_ &&
        semantic_candidate_num > 0)
    {
        double adaptive_thr = 0.0;
        vector<double> residuals;

        vector<uchar> imu_status = imuMotionFilter(
            temp_prev_pts,
            temp_cur_pts,
            is_sem_dyn,
            &adaptive_thr,
            &residuals
        );

        int removed_num = 0;
        int rescued_num = 0;

        for (size_t i = 0; i < imu_status.size(); i++)
        {
            if (is_sem_dyn[i])
            {
                if (imu_status[i] == 0)
                    removed_num++;
                else
                    rescued_num++;
            }
        }

        for (int i = (int)imu_status.size() - 1; i >= 0; i--)
        {
            if (!imu_status[i])
            {
                temp_cur_pts.erase(temp_cur_pts.begin() + i);
                temp_cur_dplpts_descriptors.erase(temp_cur_dplpts_descriptors.begin() + i);
                temp_prev_pts.erase(temp_prev_pts.begin() + i);
                temp_prev_dplpts_descriptors.erase(temp_prev_dplpts_descriptors.begin() + i);
                temp_ids.erase(temp_ids.begin() + i);
                temp_track_cnt.erase(temp_track_cnt.begin() + i);
            }
        }

        if (SEMANTIC_IMU_DEBUG)
        {
            ROS_INFO("[Semantic-IMU-soft] total=%zu, candidate=%d, removed=%d, rescued=%d, thr=%.6f",
                     temp_cur_pts.size() + removed_num,
                     semantic_candidate_num,
                     removed_num,
                     rescued_num,
                     adaptive_thr);
        }
    }
    else if (SEMANTIC_FILTER_MODE == 2 &&
             semantic_candidate_num > 0 &&
             !has_imu_rotation_ &&
             SEMANTIC_IMU_DEBUG)
    {
        ROS_WARN("[Semantic-IMU-soft] no IMU rotation, keep %d semantic candidates",
                 semantic_candidate_num);
    }

    // to avoid point gathering, create a mask depending on matched points in the current image
    
    // ─────────────────────────────────────────────────────────────


    const size_t diag_matched_after_filter = temp_cur_pts.size();

    cv::Mat mask_for_dpl = setMask_dpl(temp_cur_pts, MIN_DIST);
    
    // add new feature points but filter the ones which are close to matched points via the mask
    for (int i = 0; i < cur_pts.size(); i++)
    {
        if (!cur_matched_indices.count(i))
        {
            diag_unmatched_candidates++;

            // 新特征没有上一帧观测，无法执行语义–IMU一致性判别。
            // 第一版采用保守策略：当 SEMANTIC_MASK_NEW_FEATURES=1 时，动态区域内的新点暂不创建 track。
            if (SEMANTIC_FILTER_MODE > 0 &&
                SEMANTIC_MASK_NEW_FEATURES == 1 &&
                !local_semantic_mask.empty())
            {
                int x = cvRound(cur_pts[i].x);
                int y = cvRound(cur_pts[i].y);

                if (x >= 0 && x < local_semantic_mask.cols &&
                    y >= 0 && y < local_semantic_mask.rows &&
                    local_semantic_mask.at<uchar>(y, x) == 0)
                {
                    diag_new_semantic_rejected++;
                    continue;
                }
            }
            if (mask_for_dpl.at<uchar>(cur_pts[i]) == 255) // TO DO: annote this line can alleviate the problem of system crashing
            {
                temp_cur_pts.push_back(cur_pts[i]);
                temp_cur_dplpts_descriptors.push_back(cur_dplpts_descriptors[i]);
                temp_track_cnt.push_back(1);
                temp_ids.push_back(n_id++);
                diag_new_added++;
            }
            else
            {
                diag_new_spacing_rejected++;
            }
        }
    }
    prev_pts = temp_prev_pts;
    prev_dplpts_descriptors = temp_prev_dplpts_descriptors;
    cur_pts.swap(temp_cur_pts);
    cur_dplpts_descriptors.swap(temp_cur_dplpts_descriptors);
    track_cnt = temp_track_cnt;
    ids = temp_ids;
    pruneFeatureHistory();

    // ============================================================
    // [TrackIDDiag] Active-track length distribution after the
    // current frame has been assembled.
    // ============================================================
    long long diag_active_track_sum = 0;
    int diag_ge5 = 0;
    int diag_ge10 = 0;
    int diag_ge20 = 0;

    for (const int len : track_cnt)
    {
        diag_active_track_sum += len;
        if (len >= 5)
            ++diag_ge5;
        if (len >= 10)
            ++diag_ge10;
        if (len >= 20)
            ++diag_ge20;
    }

    const double diag_active_mean_len =
        !track_cnt.empty()
            ? static_cast<double>(diag_active_track_sum) /
                  static_cast<double>(track_cnt.size())
            : 0.0;

    double diag_active_median_len = 0.0;
    if (!track_cnt.empty())
    {
        std::vector<int> diag_track_cnt_sorted = track_cnt;
        std::sort(diag_track_cnt_sorted.begin(),
                  diag_track_cnt_sorted.end());

        const size_t n = diag_track_cnt_sorted.size();
        if (n % 2 == 1)
        {
            diag_active_median_len =
                static_cast<double>(diag_track_cnt_sorted[n / 2]);
        }
        else
        {
            diag_active_median_len =
                0.5 *
                (static_cast<double>(diag_track_cnt_sorted[n / 2 - 1]) +
                 static_cast<double>(diag_track_cnt_sorted[n / 2]));
        }
    }

    // One compact line per frame.  This log is intentionally independent
    // of SEMANTIC_IMU_DEBUG so mode-0 baseline runs always record Track IDs.
    ROS_INFO(
        "[TrackIDDiag] "
        "t=%.6f prev_active=%zu matched_prev=%zu lost=%d "
        "survival=%.4f break_rate=%.4f "
        "lost_mean_len=%.3f lost_max_len=%d lost_le3=%d lost_le5=%d "
        "new_added=%d final=%zu "
        "mean_len=%.3f median_len=%.3f ge5=%d ge10=%d ge20=%d",
        cur_time,
        diag_prev_active,
        diag_prev_matched_indices.size(),
        diag_prev_lost,
        diag_track_survival,
        diag_track_break_rate,
        diag_lost_mean_len,
        diag_lost_track_max,
        diag_lost_le3,
        diag_lost_le5,
        diag_new_added,
        track_cnt.size(),
        diag_active_mean_len,
        diag_active_median_len,
        diag_ge5,
        diag_ge10,
        diag_ge20);

    if (SEMANTIC_IMU_DEBUG)
    {
        ROS_INFO(
            "[FeatureDiag] t=%.6f mode=%d mask=%d "
            "raw=%zu matched_before=%zu hard_deleted=%d "
            "matched_after=%zu unmatched=%d "
            "new_sem_reject=%d new_spacing_reject=%d "
            "new_added=%d final_left=%zu",
            cur_time,
            SEMANTIC_FILTER_MODE,
            local_semantic_mask.empty() ? 0 : 1,
            diag_raw_cur_pts,
            diag_matched_before,
            diag_hard_deleted,
            diag_matched_after_filter,
            diag_unmatched_candidates,
            diag_new_semantic_rejected,
            diag_new_spacing_rejected,
            diag_new_added,
            cur_pts.size()
        );
    }

    // cout << cur_pts.size()<<" points reserved in current frame finally" <<endl;

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);
    int diag_right_matches = 0;

    if (!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        cur_dplpts_right_descriptors.clear();

        vector<cv::Point2f> temp_cur_right_pts;

        if (!cur_pts.empty())
        {
            extract_features_dpl(rightImg, cur_right_pts, cur_dplpts_right_descriptors);
            vector<pair<int, int>> right_matches;

            // double ransac_thresh = 0.5;
            match_features_dpl(cur_img, rightImg, cur_dplpts_descriptors, cur_dplpts_right_descriptors, right_matches,ransacReprojThreshold);
            int n_matches_right = right_matches.size();
            diag_right_matches = n_matches_right;

            for (int i = 0; i < n_matches_right; i++)
            {
                pair<int, int> right_match = right_matches[i];
                temp_cur_right_pts.push_back(cur_right_pts[right_match.second]);
                ids_right.push_back(ids[right_match.first]);
            }
            cur_right_pts = temp_cur_right_pts;
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);                                              // 去畸变加反投影至归一化平面
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map); // 计算右目特征点在相机归一化平面上的速度
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    if (SEMANTIC_IMU_DEBUG)
    {
        ROS_INFO(
            "[StereoDiag] t=%.6f left=%zu right_matches=%d",
            cur_time,
            cur_pts.size(),
            diag_right_matches
        );
    }

    // 要显示追踪图片的话，需要画图
    if (SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    // 基本上追踪完毕了，更新相关的变量
    prev_img = cur_img;               // 当前帧赋给上一帧
    prev_pts = cur_pts;               // 当前帧特征点赋给上一帧特征点
    prev_un_pts = cur_un_pts;         // 当前帧去畸变反投影特征点赋给上一帧去畸变反投影特征点
    prev_un_pts_map = cur_un_pts_map; // 上一帧去畸变反投影特征点的map=当前帧去畸变反投影特征点map：（特征点id,特征点坐标）
    prev_dplpts_descriptors = cur_dplpts_descriptors;
    prev_time = cur_time;  // 当前帧时间戳赋给上一帧时间戳
    hasPrediction = false; // 将特征点预测信号置0

    prevLeftPtsMap.clear(); // 上一帧左目特征点Map清空，下面再填充进当前帧的信息
    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    ROS_INFO_STREAM("start save ");
    

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame; // 要准备返回特征帧了：key值为featureid，value值为map：key值为相机id，value值为7x1向量，分别存放[归一化平面x,归一化平面y,1,像素平面x,像素平面y,归一化平面速度x,归一化平面速度y]
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        // V3 only withholds confirmed DYNAMIC IDs; UNCERTAIN is deferred.
        if (!backendUsable(feature_id))
            continue;
        double x, y, z;
        x = cur_un_pts[i].x; // 归一化平面x
        y = cur_un_pts[i].y; // 归一化平面y
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x; // 像素平面x
        p_v = cur_pts[i].y; // 像素平面y
        int camera_id = 0;  // 相机id，这里0是左目主相机
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x; // 归一化平面x方向速度
        velocity_y = pts_velocity[i].y; // 归一化平面y方向速度

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;      // 把数据存入向量
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity); // 形成当前帧的特征
    }
    // 右目是一样的流程
    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            if (!backendUsable(feature_id))
                continue;
            double x, y, z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1; // 其它和左目差不多，只有这个相机id是1，
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity); // 把右目数据推入
        }
    }

    // printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

void FeatureTrackerDPL::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTrackerDPL::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTrackerDPL::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

/// @brief 将像素平面特征点去畸变，并反投影至归一化相机系下
/// @param pts
/// @param cam
/// @return
vector<cv::Point2f> FeatureTrackerDPL::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);                                   // 针孔相机模型下，返回的b就是去畸变的归一化相机平面坐标[x,y,1]
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z())); // 针孔相机模型下，b.z()是1.0，所以横纵坐标为归一化平面坐标
    }
    return un_pts;
}

/// @brief 计算特征点的速度，这个速度是指在特征点在相机归一化平面中的位移的速度
/// @param ids
/// @param pts 当前这一帧的特征点
/// @param cur_id_pts 名字叫cur_id_pts，但传进来的时候其实保留的是上一帧的特征点
/// @param prev_id_pts 传进来的时候保留的其实也是上一帧的特征点
/// @return
vector<cv::Point2f> FeatureTrackerDPL::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                                   map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity; // 将要返回的特征点速度
    cur_id_pts.clear();               // 清空，下面放入当前帧的特征点信息
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    // 现在cur_id_pts里存的是当前帧特征点了，和prev_id_pts存的上一帧特征点计算速度
    if (!prev_id_pts.empty()) // 若上一帧特征点非空，则计算速度
    {
        double dt = cur_time - prev_time; // 时间间隔

        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end()) // 能找到和上一帧对应的特征点，则计算速度
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0)); // 不能找到和上一帧对应的特征点，则速度为0
        }
    }
    else // 若上一帧特征点为空，则当前帧所有特征点速度初始化为0
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTrackerDPL::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                                  vector<int> &curLeftIds,
                                  vector<cv::Point2f> &curLeftPts,
                                  vector<cv::Point2f> &curRightPts,
                                  map<int, cv::Point2f> &prevLeftPtsMap)
{
    // int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_BGR2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            // cv::Point2f leftPt = curLeftPtsTrackRight[i];
            // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    // draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    // printf("predict pts size %d \n", (int)predict_pts_debug.size());

    // cv::Mat imCur2Compress;
    // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

/// @brief 给FeatureTracker输入预测的空间点
/// @param predictPts
void FeatureTrackerDPL::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true; // 有预测特征点的信号置1
    predict_pts.clear();  // 把老的预测特征点清空
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict; // 设置一个存放路标点的map的迭代器
    // 遍历待追踪的特征点的id
    for (size_t i = 0; i < ids.size(); i++)
    {
        // printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);   // 找对应id的预测路标点
        if (itPredict != predictPts.end()) // 如果没找到
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);       // 将预测的相机系下的空间点投影至图像像素
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y())); // 存入预测特征点像素
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]); // 若是没有预测特征点，则利用历史特征点的位置（这其实就是不通过初始光流时的结果）
    }
}

void FeatureTrackerDPL::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if (itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

cv::Mat FeatureTrackerDPL::getTrackImage()
{
    return imTrack;
}

// ── [Semantic Mask] filterBySemanticMask ─────────────────────────
void FeatureTrackerDPL::filterBySemanticMask(
    std::vector<cv::Point2f> &pts,
    std::vector<std::pair<cv::Point2f, std::vector<float>>> &descriptors,
    const cv::Mat &mask)
{
    if (mask.empty() || pts.empty()) return;

    std::vector<cv::Point2f> pts_out;
    std::vector<std::pair<cv::Point2f, std::vector<float>>> desc_out;
    pts_out.reserve(pts.size());
    desc_out.reserve(pts.size());

    for (size_t i = 0; i < pts.size(); i++) {
        int x = static_cast<int>(pts[i].x);
        int y = static_cast<int>(pts[i].y);
        // 边界检查
        if (x < 0 || x >= mask.cols || y < 0 || y >= mask.rows) {
            pts_out.push_back(pts[i]);
            desc_out.push_back(descriptors[i]);
            continue;
        }
        // mask==0 为动态区域，剔除；mask==255 为静态，保留
        if (mask.at<uchar>(y, x) != 0) {
            pts_out.push_back(pts[i]);
            desc_out.push_back(descriptors[i]);
        }
    }
    pts         = std::move(pts_out);
    descriptors = std::move(desc_out);
}
// ─────────────────────────────────────────────────────────────────
