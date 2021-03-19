#ifndef LIGHT_OPENPOSE_H
#define LIGHT_OPENPOSE_H

#include "ncnn/ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#import <functional>


template<typename T, std::size_t N>
constexpr std::size_t arraySize(const T (&)[N]) noexcept {
    return N;
}

// .h
namespace human_pose_estimation {
    struct HumanPose {
        HumanPose(const std::vector<cv::Point2f> &keypoints = std::vector<cv::Point2f>(),
                  const float &score = 0);

        std::vector<cv::Point2f> keypoints;
        float score;
    };
} // namespace human_pose_estimation

namespace human_pose_estimation {
    struct Peak {
        Peak(const int id = -1,
             const cv::Point2f &pos = cv::Point2f(),
             const float score = 0.0f);

        int id;
        cv::Point2f pos;
        float score;
    };

    struct HumanPoseByPeaksIndices {
        explicit HumanPoseByPeaksIndices(const int keypointsNumber);

        std::vector<int> peaksIndices;
        int nJoints;
        float score;
    };

    struct TwoJointsConnection {
        TwoJointsConnection(const int firstJointIdx,
                            const int secondJointIdx,
                            const float score);

        int firstJointIdx;
        int secondJointIdx;
        float score;
    };

    void findPeaks(const std::vector<cv::Mat> &heatMaps,
                   const float minPeaksDistance,
                   std::vector<std::vector<Peak> > &allPeaks,
                   int heatMapId);

    std::vector<HumanPose> groupPeaksToPoses(
            const std::vector<std::vector<Peak> > &allPeaks,
            const std::vector<cv::Mat> &pafs,
            const size_t keypointsNumber,
            const float midPointsScoreThreshold,
            const float foundMidPointsRatioThreshold,
            const int minJointsNumber,
            const float minSubsetScore);
} // namespace human_pose_estimation


class LightOpenPose {
public:
    LightOpenPose(bool useGPU);

    ~LightOpenPose();

    std::vector<human_pose_estimation::HumanPose> detect(UIImage *image);

private:
    void preprocess(UIImage *image, ncnn::Mat &in);

    ncnn::Net *humanPoseNet;
    int input_size_w = 456;
    int input_size_h = 256;

public:
    static LightOpenPose *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //LIGHT_OPENPOSE_H
