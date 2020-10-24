//
// Created by WZTENG on 2020/08/29 029.
//

#ifndef YOLOV5_FACELANDMARK_H
#define YOLOV5_FACELANDMARK_H

#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <opencv2/core/types.hpp>

#include <android/log.h>


struct FaceKeyPoint {
    cv::Point2f p;
    float prob;
};


class FaceLandmark {
public:
    FaceLandmark(AAssetManager *mgr, bool useGPU);

    ~FaceLandmark();

    std::vector<FaceKeyPoint> detect(JNIEnv *env, jobject image);

private:
    int runlandmark(cv::Mat &roi, int pose_size_width, int pose_size_height,
                std::vector<FaceKeyPoint> &keypoints,
                float x1, float y1);

//    static std::vector<FaceKeyPoint> decode_infer(ncnn::Mat &data, const cv::Size& frame_size, int net_size);
    ncnn::Net *FaceNet;
    ncnn::Net *LandmarkNet;
    int detector_size_width = 320;
    int detector_size_height = 256;
    int landmark_size_width = 112;
    int landmark_size_height = 112;
public:
    static FaceLandmark *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //YOLOV5_FACELANDMARK_H
