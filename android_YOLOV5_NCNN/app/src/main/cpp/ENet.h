//
// Created by WZTENG on 2020/08/28 028.
//

#ifndef YOLOV5_ENET_H
#define YOLOV5_ENET_H


#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>


class ENet {
public:
    ENet(AAssetManager *mgr, bool useGPU);
    ~ENet();

    ncnn::Mat detect_enet(JNIEnv *env, jobject image);

private:
    ncnn::Net *ENetsim;
    int target_size_w = 512;
    int target_size_h = 512;

public:
    static ENet *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //YOLOV5_ENET_H
