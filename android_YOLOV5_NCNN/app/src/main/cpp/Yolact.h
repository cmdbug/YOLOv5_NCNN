//
// Created by WZTENG on 2020/08/19 019.
//

#ifndef YOLOV5_YOLACT_H
#define YOLOV5_YOLACT_H

#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>


struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> maskdata;
    cv::Mat mask;
};

class Yolact {
public:
    Yolact(AAssetManager *mgr, bool useGPU);
    ~Yolact();

    std::vector<Object> detect_yolact(JNIEnv *env, jobject image);

private:
    ncnn::Net *YolactNet;
    int target_size = 550;

public:
    static Yolact *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //YOLOV5_YOLACT_H
