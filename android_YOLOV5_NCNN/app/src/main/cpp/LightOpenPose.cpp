#include "LightOpenPose.h"

static const std::pair<int, int> limbIdsHeatmap[] = {
        {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6},
        {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11},
        {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16},
        {0, 15}, {15, 17}, {2, 16}, {5, 17}
};
static const std::pair<int, int> limbIdsPaf[] = {
        {12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23},
        {24, 25}, {0, 1}, {2, 3}, {4, 5}, {6, 7},
        {8, 9}, {10, 11}, {28, 29}, {30, 31}, {34, 35},
        {32, 33}, {36, 37}, {18, 19}, {26, 27}
};

namespace human_pose_estimation {
    HumanPose::HumanPose(const std::vector<cv::Point2f> &keypoints,
                         const float &score)
            : keypoints(keypoints),
              score(score) {
    }
} // namespace human_pose_estimation

namespace human_pose_estimation {
    Peak::Peak(const int id, const cv::Point2f &pos, const float score)
            : id(id),
              pos(pos),
              score(score) {
    }

    HumanPoseByPeaksIndices::HumanPoseByPeaksIndices(const int keypointsNumber)
            : peaksIndices(std::vector<int>(keypointsNumber, -1)),
              nJoints(0),
              score(0.0f) {
    }

    TwoJointsConnection::TwoJointsConnection(const int firstJointIdx,
                                             const int secondJointIdx,
                                             const float score)
            : firstJointIdx(firstJointIdx),
              secondJointIdx(secondJointIdx),
              score(score) {
    }

    void findPeaks(const std::vector<cv::Mat> &heatMaps,
                   const float minPeaksDistance,
                   std::vector<std::vector<Peak> > &allPeaks,
                   int heatMapId) {
        const float threshold = 0.1f;
        std::vector<cv::Point> peaks;
        const cv::Mat &heatMap = heatMaps[heatMapId];
        const float *heatMapData = heatMap.ptr<float>();
        size_t heatMapStep = heatMap.step1();
        for (int y = -1; y < heatMap.rows + 1; y++) {
            for (int x = -1; x < heatMap.cols + 1; x++) {
                float val = 0;
                if (x >= 0
                    && y >= 0
                    && x < heatMap.cols
                    && y < heatMap.rows) {
                    val = heatMapData[y * heatMapStep + x];
                    val = val >= threshold ? val : 0;
                }

                float left_val = 0;
                if (y >= 0
                    && x < (heatMap.cols - 1)
                    && y < heatMap.rows) {
                    left_val = heatMapData[y * heatMapStep + x + 1];
                    left_val = left_val >= threshold ? left_val : 0;
                }

                float right_val = 0;
                if (x > 0
                    && y >= 0
                    && y < heatMap.rows) {
                    right_val = heatMapData[y * heatMapStep + x - 1];
                    right_val = right_val >= threshold ? right_val : 0;
                }

                float top_val = 0;
                if (x >= 0
                    && x < heatMap.cols
                    && y < (heatMap.rows - 1)) {
                    top_val = heatMapData[(y + 1) * heatMapStep + x];
                    top_val = top_val >= threshold ? top_val : 0;
                }

                float bottom_val = 0;
                if (x >= 0
                    && y > 0
                    && x < heatMap.cols) {
                    bottom_val = heatMapData[(y - 1) * heatMapStep + x];
                    bottom_val = bottom_val >= threshold ? bottom_val : 0;
                }

                if ((val > left_val)
                    && (val > right_val)
                    && (val > top_val)
                    && (val > bottom_val)) {
                    peaks.push_back(cv::Point(x, y));
                }
            }
        }
        std::sort(peaks.begin(), peaks.end(), [](const cv::Point &a, const cv::Point &b) {
            return a.x < b.x;
        });
        std::vector<bool> isActualPeak(peaks.size(), true);
        int peakCounter = 0;
        std::vector<Peak> &peaksWithScoreAndID = allPeaks[heatMapId];
        for (size_t i = 0; i < peaks.size(); i++) {
            if (isActualPeak[i]) {
                for (size_t j = i + 1; j < peaks.size(); j++) {
                    if (sqrt((peaks[i].x - peaks[j].x) * (peaks[i].x - peaks[j].x) +
                             (peaks[i].y - peaks[j].y) * (peaks[i].y - peaks[j].y)) < minPeaksDistance) {
                        isActualPeak[j] = false;
                    }
                }
                peaksWithScoreAndID.push_back(Peak(peakCounter++, peaks[i], heatMap.at<float>(peaks[i])));
            }
        }
    }

    std::vector<HumanPose> groupPeaksToPoses(const std::vector<std::vector<Peak> > &allPeaks,
                                             const std::vector<cv::Mat> &pafs,
                                             const size_t keypointsNumber,
                                             const float midPointsScoreThreshold,
                                             const float foundMidPointsRatioThreshold,
                                             const int minJointsNumber,
                                             const float minSubsetScore) {
        std::vector<Peak> candidates;
        for (const auto &peaks : allPeaks) {
            candidates.insert(candidates.end(), peaks.begin(), peaks.end());
        }
        std::vector<HumanPoseByPeaksIndices> subset(0, HumanPoseByPeaksIndices(keypointsNumber));
        for (size_t k = 0; k < arraySize(limbIdsPaf); k++) {
            std::vector<TwoJointsConnection> connections;
            const int mapIdxOffset = 0; // keypointsNumber + 1;
            std::pair<cv::Mat, cv::Mat> scoreMid = {pafs[limbIdsPaf[k].first - mapIdxOffset],
                                                    pafs[limbIdsPaf[k].second - mapIdxOffset]};
            const int idxJointA = limbIdsHeatmap[k].first;  // first - 1;
            const int idxJointB = limbIdsHeatmap[k].second; // second - 1;
            const std::vector<Peak> &candA = allPeaks[idxJointA];
            const std::vector<Peak> &candB = allPeaks[idxJointB];
            const size_t nJointsA = candA.size();
            const size_t nJointsB = candB.size();
            if (nJointsA == 0 && nJointsB == 0) {
                continue;
            } else if (nJointsA == 0) {
                for (size_t i = 0; i < nJointsB; i++) {
                    int num = 0;
                    for (size_t j = 0; j < subset.size(); j++) {
                        if (subset[j].peaksIndices[idxJointB] == candB[i].id) {
                            num++;
                            continue;
                        }
                    }
                    if (num == 0) {
                        HumanPoseByPeaksIndices personKeypoints(keypointsNumber);
                        personKeypoints.peaksIndices[idxJointB] = candB[i].id;
                        personKeypoints.nJoints = 1;
                        personKeypoints.score = candB[i].score;
                        subset.push_back(personKeypoints);
                    }
                }
                continue;
            } else if (nJointsB == 0) {
                for (size_t i = 0; i < nJointsA; i++) {
                    int num = 0;
                    for (size_t j = 0; j < subset.size(); j++) {
                        if (subset[j].peaksIndices[idxJointA] == candA[i].id) {
                            num++;
                            continue;
                        }
                    }
                    if (num == 0) {
                        HumanPoseByPeaksIndices personKeypoints(keypointsNumber);
                        personKeypoints.peaksIndices[idxJointA] = candA[i].id;
                        personKeypoints.nJoints = 1;
                        personKeypoints.score = candA[i].score;
                        subset.push_back(personKeypoints);
                    }
                }
                continue;
            }

            std::vector<TwoJointsConnection> tempJointConnections;
            for (size_t i = 0; i < nJointsA; i++) {
                for (size_t j = 0; j < nJointsB; j++) {
                    cv::Point2f pt = candA[i].pos * 0.5 + candB[j].pos * 0.5;
                    cv::Point mid = cv::Point(cvRound(pt.x), cvRound(pt.y));
                    cv::Point2f vec = candB[j].pos - candA[i].pos;
                    double norm_vec = cv::norm(vec);
                    if (norm_vec == 0) {
                        continue;
                    }
                    vec /= norm_vec;
                    float score = vec.x * scoreMid.first.at<float>(mid) + vec.y * scoreMid.second.at<float>(mid);
                    int height_n = pafs[0].rows / 2;
                    float suc_ratio = 0.0f;
                    float mid_score = 0.0f;
                    const int mid_num = 10;
                    const float scoreThreshold = -100.0f;
                    if (score > scoreThreshold) {
                        float p_sum = 0;
                        int p_count = 0;
                        cv::Size2f step((candB[j].pos.x - candA[i].pos.x) / (mid_num - 1),
                                        (candB[j].pos.y - candA[i].pos.y) / (mid_num - 1));
                        for (int n = 0; n < mid_num; n++) {
                            cv::Point midPoint(cvRound(candA[i].pos.x + n * step.width),
                                               cvRound(candA[i].pos.y + n * step.height));
                            cv::Point2f pred(scoreMid.first.at<float>(midPoint),
                                             scoreMid.second.at<float>(midPoint));
                            score = vec.x * pred.x + vec.y * pred.y;
                            if (score > midPointsScoreThreshold) {
                                p_sum += score;
                                p_count++;
                            }
                        }
                        suc_ratio = static_cast<float>(p_count / mid_num);
                        float ratio = p_count > 0 ? p_sum / p_count : 0.0f;
                        mid_score = ratio + static_cast<float>(std::min(height_n / norm_vec - 1, 0.0));
                    }
                    if (mid_score > 0
                        && suc_ratio > foundMidPointsRatioThreshold) {
                        tempJointConnections.push_back(TwoJointsConnection(i, j, mid_score));
                    }
                }
            }
            if (!tempJointConnections.empty()) {
                std::sort(tempJointConnections.begin(), tempJointConnections.end(),
                          [](const TwoJointsConnection &a,
                             const TwoJointsConnection &b) {
                              return (a.score > b.score);
                          });
            }
            size_t num_limbs = std::min(nJointsA, nJointsB);
            size_t cnt = 0;
            std::vector<int> occurA(nJointsA, 0);
            std::vector<int> occurB(nJointsB, 0);
            for (size_t row = 0; row < tempJointConnections.size(); row++) {
                if (cnt == num_limbs) {
                    break;
                }
                const int &indexA = tempJointConnections[row].firstJointIdx;
                const int &indexB = tempJointConnections[row].secondJointIdx;
                const float &score = tempJointConnections[row].score;
                if (occurA[indexA] == 0
                    && occurB[indexB] == 0) {
                    connections.push_back(TwoJointsConnection(candA[indexA].id, candB[indexB].id, score));
                    cnt++;
                    occurA[indexA] = 1;
                    occurB[indexB] = 1;
                }
            }
            if (connections.empty()) {
                continue;
            }

            bool extraJointConnections = (k == 17 || k == 18);
            if (k == 0) {
                subset = std::vector<HumanPoseByPeaksIndices>(
                        connections.size(), HumanPoseByPeaksIndices(keypointsNumber));
                for (size_t i = 0; i < connections.size(); i++) {
                    const int &indexA = connections[i].firstJointIdx;
                    const int &indexB = connections[i].secondJointIdx;
                    subset[i].peaksIndices[idxJointA] = indexA;
                    subset[i].peaksIndices[idxJointB] = indexB;
                    subset[i].nJoints = 2;
                    subset[i].score = candidates[indexA].score + candidates[indexB].score + connections[i].score;
                }
            } else if (extraJointConnections) {
                for (size_t i = 0; i < connections.size(); i++) {
                    const int &indexA = connections[i].firstJointIdx;
                    const int &indexB = connections[i].secondJointIdx;
                    for (size_t j = 0; j < subset.size(); j++) {
                        if (subset[j].peaksIndices[idxJointA] == indexA
                            && subset[j].peaksIndices[idxJointB] == -1) {
                            subset[j].peaksIndices[idxJointB] = indexB;
                        } else if (subset[j].peaksIndices[idxJointB] == indexB
                                   && subset[j].peaksIndices[idxJointA] == -1) {
                            subset[j].peaksIndices[idxJointA] = indexA;
                        }
                    }
                }
                continue;
            } else {
                for (size_t i = 0; i < connections.size(); i++) {
                    const int &indexA = connections[i].firstJointIdx;
                    const int &indexB = connections[i].secondJointIdx;
                    bool num = false;
                    for (size_t j = 0; j < subset.size(); j++) {
                        if (subset[j].peaksIndices[idxJointA] == indexA) {
                            subset[j].peaksIndices[idxJointB] = indexB;
                            subset[j].nJoints++;
                            subset[j].score += candidates[indexB].score + connections[i].score;
                            num = true;
                        }
                    }
                    if (!num) {
                        HumanPoseByPeaksIndices hpWithScore(keypointsNumber);
                        hpWithScore.peaksIndices[idxJointA] = indexA;
                        hpWithScore.peaksIndices[idxJointB] = indexB;
                        hpWithScore.nJoints = 2;
                        hpWithScore.score = candidates[indexA].score + candidates[indexB].score + connections[i].score;
                        subset.push_back(hpWithScore);
                    }
                }
            }
        }
        std::vector<HumanPose> poses;
        for (const auto &subsetI : subset) {
            if (subsetI.nJoints < minJointsNumber || subsetI.score / subsetI.nJoints < minSubsetScore) {
                continue;
            }
            int position = -1;
            HumanPose pose(std::vector<cv::Point2f>(keypointsNumber, cv::Point2f(-1.0f, -1.0f)),
                           subsetI.score * std::max(0, subsetI.nJoints - 1));
            for (const auto &peakIdx : subsetI.peaksIndices) {
                position++;
                if (peakIdx >= 0) {
                    pose.keypoints[position] = candidates[peakIdx].pos;
                    pose.keypoints[position].x += 0.5;
                    pose.keypoints[position].y += 0.5;
                }
            }
            poses.push_back(pose);
        }
        return poses;
    }
} // namespace human_pose_estimation

void postProcess(const ncnn::Mat &pafs, const ncnn::Mat &heatmaps, std::vector<human_pose_estimation::HumanPose> &poses,
                 int img_h, int img_w, int net_h, int net_w) {
    using namespace human_pose_estimation;

    float upsample_ratio = 4;
    // ncnn::Mat -> cv::Mat
    // heatmaps
    std::vector<cv::Mat> cv_heatmaps(heatmaps.c);
    for (int p = 0; p < heatmaps.c; p++) {
        cv_heatmaps[p] = cv::Mat(heatmaps.h, heatmaps.w, CV_32FC1);
        memcpy((float *) cv_heatmaps[p].data, heatmaps.channel(p), heatmaps.h * heatmaps.w * sizeof(float));
    }
//    LOGD("%-11s C:%llu H:%d W:%d\n", "cv_heatmaps", cv_heatmaps.size(), cv_heatmaps[0].rows, cv_heatmaps[0].cols);
    if (cv_heatmaps.size() != heatmaps.c || cv_heatmaps[0].rows != heatmaps.h || cv_heatmaps[0].cols != heatmaps.w) {
        LOGE("ncnn::Mat -> cv::Mat fail\n");
        return;
    }
    if (heatmaps.channel(0).row(1)[2] != cv_heatmaps[0].ptr<float>(1)[2]) // c:0 h:1 w:2
    {
        LOGE("ncnn::Mat != cv::Mat\n");
        return;
    }
    // upsample
    std::vector<cv::Mat> cv_heatmaps_upsample(heatmaps.c);
    for (int p = 0; p < heatmaps.c; p++) {
        cv::resize(cv_heatmaps[p], cv_heatmaps_upsample[p],
                   cv::Size(), upsample_ratio, upsample_ratio, cv::INTER_CUBIC);
    }

    // ncnn::Mat -> cv::Mat
    // pafs
    std::vector<cv::Mat> cv_pafs(pafs.c);
    for (int p = 0; p < pafs.c; p++) {
        cv_pafs[p] = cv::Mat(pafs.h, pafs.w, CV_32FC1);
        memcpy((float *) cv_pafs[p].data, pafs.channel(p), pafs.h * pafs.w * sizeof(float));
    }
//    LOGD("%-11s C:%llu H:%d W:%d\n", "cv_pafs", cv_pafs.size(), cv_pafs[0].rows, cv_pafs[0].cols);
    if (cv_pafs.size() != pafs.c || cv_pafs[0].rows != pafs.h || cv_pafs[0].cols != pafs.w) {
        LOGE("ncnn::Mat -> cv::Mat fail\n");
        return;
    }
    if (pafs.channel(0).row(1)[2] != cv_pafs[0].ptr<float>(1)[2]) // c:0 h:1 w:2
    {
        LOGE("ncnn::Mat != cv::Mat\n");
        return;
    }
    // upsample
    std::vector<cv::Mat> cv_pafs_upsample(pafs.c);
    for (int p = 0; p < pafs.c; p++) {
        cv::resize(cv_pafs[p], cv_pafs_upsample[p],
                   cv::Size(), upsample_ratio, upsample_ratio, cv::INTER_CUBIC);
    }

    // postprocess
    const float minPeaksDistance = 3.0f;
    const int keypointsNumber = 18;
    const float midPointsScoreThreshold = 0.05f;
    const float foundMidPointsRatioThreshold = 0.8f;
    const int minJointsNumber = 3;
    const float minSubsetScore = 0.2f;

    std::vector<std::vector<Peak> > peaksFromHeatMap(cv_heatmaps_upsample.size());

//#pragma omp parallel for
    for (int i = 0; i < cv_heatmaps_upsample.size(); i++) {
        findPeaks(cv_heatmaps_upsample, minPeaksDistance, peaksFromHeatMap, i);
    }

    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < cv_heatmaps_upsample.size(); heatmapId++) {
        peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
        for (auto &peak : peaksFromHeatMap[heatmapId]) {
            peak.id += peaksBefore;
        }
    }
//    std::vector<HumanPose> poses;
    poses = groupPeaksToPoses(
            peaksFromHeatMap, cv_pafs_upsample, keypointsNumber, midPointsScoreThreshold,
            foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);

//    LOGD("human pose total:%llu\n", poses.size());

    // scale keypoint
    float scale_x = 1.0f * net_w / img_w;
    float scale_y = 1.0f * net_h / img_h;
    float stride = 8.0f;
    float upsample = upsample_ratio;

    for (int i = 0; i < poses.size(); i++) {
        HumanPose pose = poses[i];
        for (int j = 0; j < keypointsNumber; j++) {
            if (pose.keypoints[j].x == -1 || pose.keypoints[j].y == -1) {
                continue;
            }
            poses[i].keypoints[j].x = stride / upsample * pose.keypoints[j].x / scale_x;
            poses[i].keypoints[j].y = stride / upsample * pose.keypoints[j].y / scale_y;
        }
    }
}

bool LightOpenPose::hasGPU = true;
bool LightOpenPose::toUseGPU = true;
LightOpenPose *LightOpenPose::detector = nullptr;

LightOpenPose::LightOpenPose(AAssetManager *mgr, bool useGPU) {
    hasGPU = ncnn::get_gpu_count() > 0;
    toUseGPU = hasGPU && useGPU;

    humanPoseNet = new ncnn::Net();
    // opt 需要在加载前设置
    humanPoseNet->opt.use_vulkan_compute = toUseGPU;  // gpu
    humanPoseNet->opt.use_fp16_arithmetic = true;  // fp16运算加速
    humanPoseNet->opt.use_fp16_packed = true;
    humanPoseNet->opt.use_fp16_storage = true;
    humanPoseNet->load_param(mgr, "human_pose_sim_opt.param");
    humanPoseNet->load_model(mgr, "human_pose_sim_opt.bin");
}

LightOpenPose::~LightOpenPose() {
    humanPoseNet->clear();
    delete humanPoseNet;
}

void LightOpenPose::preprocess(JNIEnv *env, jobject image, ncnn::Mat &in) {
    in = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2BGR, input_size_w, input_size_h);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(mean_vals, norm_vals);
}

std::vector<human_pose_estimation::HumanPose> LightOpenPose::detect(JNIEnv *env, jobject image) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);

    int img_w = img_size.width;
    int img_h = img_size.height;
    int net_w = 456;
    int net_h = 456;

    int w = img_w;
    int h = img_h;
    float scale = 1.0f;
    if (w > h) {
        scale = (float) net_w / w;
        w = net_w;
        h = h * scale;
    } else {
        scale = (float) net_h / h;
        h = net_h;
        w = w * scale;
    }
    net_w = w;
    net_h = h;
    input_size_w = w;
    input_size_h = h;

    ncnn::Mat in;
    preprocess(env, image, in);

    // forward
    ncnn::Mat pafs;
    ncnn::Mat heatmaps;
    ncnn::Extractor ex = humanPoseNet->create_extractor();
    ex.input("data", in);
    ex.extract("stage_1_output_1_heatmaps", heatmaps);  // or stage_0_output_1_heatmaps
    ex.extract("stage_1_output_0_pafs", pafs);          // or stage_0_output_0_pafs

    // postprocess
    std::vector<human_pose_estimation::HumanPose> poses;
    postProcess(pafs, heatmaps, poses, img_h, img_w, net_h, net_w);

    return poses;
}


