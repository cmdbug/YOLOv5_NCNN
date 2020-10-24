//
// Created by WZTENG on 2020/09/21 028.
//

#ifndef YOLOV5_MBNFCN_H
#define YOLOV5_MBNFCN_H


#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>


class MbnFCN {
public:
    MbnFCN(AAssetManager *mgr, bool useGPU);
    ~MbnFCN();

    ncnn::Mat detect_mbnfcn(JNIEnv *env, jobject image);

private:
    ncnn::Net *MBNFCNsim;
    int target_size_w = 512;
    int target_size_h = 512;

public:
    static MbnFCN *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //YOLOV5_MBNFCN_H
