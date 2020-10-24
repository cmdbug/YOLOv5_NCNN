//
// Created by WZTENG on 2020/09/14 014.
//

#ifndef YOLOV5_DBFACE_H
#define YOLOV5_DBFACE_H

#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <opencv2/core/types.hpp>

#include "YoloV5.h"


struct Box {
    float x, y, r, b;
};

struct Landmark {
    std::vector<float> x;
    std::vector<float> y;
};

struct Obj {
    double score;
    Box box;
    Landmark landmark;
};

struct Id {
    double score;
    int idx;
    int idy;
};


class DBFace {
public:
    DBFace(AAssetManager *mgr, bool useGPU);

    ~DBFace();

    std::vector<Obj> detect(JNIEnv *env, jobject image, double threshold, double nms_threshold);

private:
    std::vector<Obj> nms(std::vector<Obj> objs, float iou = 0.5);

    void genIds(ncnn::Mat hm, ncnn::Mat hmPool, int w, double thresh, std::vector<Id> &ids);

    void decode(int w, std::vector<Id> ids, ncnn::Mat tlrb, ncnn::Mat landmark, std::vector<Obj> &objs);

    cv::Mat pad(cv::Mat img, int stride = 32);

    float getIou(Box a, Box b);

    inline float fast_exp(float x);

    inline float myExp(float v);

    float THRESHOLD;
//    float IOU;
    float STRIDE = 4;

    ncnn::Net *DBFaceNet;

public:
    static DBFace *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //YOLOV5_DBFACE_H
