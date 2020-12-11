//
// Created by WZTENG on 2020/09/21 028.
//

#ifndef YOLOV5_MBNFCN_H
#define YOLOV5_MBNFCN_H


#include <stdio.h>
#include "ncnn/ncnn/net.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#import <functional>
#include "YoloV5.h"


class MbnFCN {
public:
    MbnFCN(bool useGPU);
    ~MbnFCN();

    ncnn::Mat detect_mbnfcn(UIImage* image);

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
