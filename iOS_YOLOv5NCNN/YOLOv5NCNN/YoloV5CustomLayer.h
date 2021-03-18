#ifndef YOLOV5_CUSTOMLAYER_H
#define YOLOV5_CUSTOMLAYER_H

#include <stdio.h>
#include "ncnn/ncnn/net.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#import <functional>
#include "YoloV5.h"

/**

如果选择重新编译 ncnn 库，需要开启 rtti 选项，即：-DNCNN_DISABLE_RTTI=OFF -DNCNN_DISABLE_EXCEPTION=OFF
或:
Build Settings -> Apple Clang - Language - C++ -> Enable C++ Runtime Types 设置为 NO。
但是由于OCR部分需要使用 OpenCV 4+ 的版本，导致 RTTI 与 EXCEPTION 冲突，所以需要自行编译 ncnn 或去掉 OpenCV、OCR 或去掉 OCR 使用低
版本的 OpenCV 2.4.13 等。
 
使用自定义层记得在 .h 文件中的：ENABLE_CUSTOM_LAYER 改为 1 开启

**/

#define ENABLE_CUSTOM_LAYER 0  // 0:disable 1:enable

struct YoloObject {
//    cv::Rect_<float> rect;
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};


class YoloV5CustomLayer {
public:
    YoloV5CustomLayer(bool useGPU);

    ~YoloV5CustomLayer();

    std::vector<BoxInfo> detect(UIImage *image, float threshold, float nms_threshold);
    std::vector<std::string> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};
private:

    ncnn::Net *Net;
    int input_size = 640;
    int num_class = 80;

    std::vector<YoloLayerData> layers{
            {"output", 8,  {{10,  13}, {16,  30},  {33,  23}}},
            {"781",    16, {{30,  61}, {62,  45},  {59,  119}}},
            {"801",    32, {{116, 90}, {156, 198}, {373, 326}}},
    };

public:
    static YoloV5CustomLayer *detector;
    static bool hasGPU;
    static bool toUseGPU;

};


#endif //YOLOV5_CUSTOMLAYER_H
