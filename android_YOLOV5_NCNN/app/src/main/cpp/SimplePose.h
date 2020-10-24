//
// Created by WZTENG on 2020/08/17 017.
//

#ifndef YOLOV5_SIMPLEPOSE_H
#define YOLOV5_SIMPLEPOSE_H

#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <opencv2/core/types.hpp>

#include "YoloV5.h"

#include <android/log.h>

#ifndef LOG_TAG
#define LOG_TAG "WZT_NCNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif


struct KeyPoint {
    cv::Point2f p;
    float prob;
};

struct PoseResult {
    std::vector<KeyPoint> keyPoints;
    BoxInfo boxInfos;
};


class SimplePose {
public:
    SimplePose(AAssetManager *mgr, bool useGPU);

    ~SimplePose();

    std::vector<PoseResult> detect(JNIEnv *env, jobject image);

private:
    int runpose(cv::Mat &roi, int pose_size_width, int pose_size_height,
                std::vector<KeyPoint> &keypoints,
                float x1, float y1);

//    static std::vector<KeyPoint> decode_infer(ncnn::Mat &data, const cv::Size& frame_size, int net_size);
    ncnn::Net *PersonNet;
    ncnn::Net *PoseNet;
    int detector_size_width = 320;
    int detector_size_height = 320;
    int pose_size_width = 192;
    int pose_size_height = 256;
public:
    static SimplePose *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //YOLOV5_SIMPLEPOSE_H
