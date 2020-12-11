//
// Created by WZTENG on 2020/09/24 028.
//

#ifndef YOLOV5_MOBILENETV3SEG_H
#define YOLOV5_MOBILENETV3SEG_H

#include <stdio.h>
#include "ncnn/ncnn/net.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#import <functional>
#include "YoloV5.h"


class MBNV3Seg {
public:
    MBNV3Seg(bool useGPU);
    ~MBNV3Seg();

    ncnn::Mat detect_mbnseg(UIImage *image);

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
