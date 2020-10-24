//
// Created by WZTENG on 2020/08/22 022.
//

#ifndef YOLOV5_NCNNDBNET_H
#define YOLOV5_NCNNDBNET_H

#include "ZData.h"
#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>


class NCNNDBNet {
public:
    NCNNDBNet();
    ~NCNNDBNet();

    std::vector<TextBox> Forward(cv::Mat &srcmat, int inshortsize);

private:
    ncnn::Net DBNet;

    const float mean_vals_dbnet[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
    const float norm_vals_dbnet[3] = {1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0};
    const float unclip_ratio = 2.0;
    const float box_thresh = 0.5;
    const float thresh = 0.3;
    const int min_size = 3;

    const float mean_vals_crnn_angle[3] = {127.5, 127.5, 127.5};
    const float norm_vals_crnn_angle[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};

public:
    static NCNNDBNet *detector;
    static bool hasGPU;
};


#endif //YOLOV5_NCNNDBNET_H
