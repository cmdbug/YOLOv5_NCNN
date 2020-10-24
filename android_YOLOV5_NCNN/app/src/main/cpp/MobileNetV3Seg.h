//
// Created by WZTENG on 2020/09/24 028.
//

#ifndef YOLOV5_MOBILENETV3SEG_H
#define YOLOV5_MOBILENETV3SEG_H


#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>


class MBNV3Seg {
public:
    MBNV3Seg(AAssetManager *mgr, bool useGPU);
    ~MBNV3Seg();

    ncnn::Mat detect_mbnseg(JNIEnv *env, jobject image);

private:
    ncnn::Net *MBNSegsim;
    int target_size_w = 512;
    int target_size_h = 512;

public:
    static MBNV3Seg *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //YOLOV5_MOBILENETV3SEG_H
