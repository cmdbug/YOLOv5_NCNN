#ifndef YOLOV4_H
#define YOLOV4_H

#include <stdio.h>
#include "ncnn/ncnn/net.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#import <functional>
#include "YoloV5.h"

//namespace yolocv{
//    typedef struct{
//        int width;
//        int height;
//    }YoloSize;
//}

//typedef struct {
//    std::string name;
//    int stride;
//    std::vector<yolocv::YoloSize> anchors;
//}YoloLayerData;

//typedef struct BoxInfo {
//    float x1;
//    float y1;
//    float x2;
//    float y2;
//    float score;
//    int label;
//} BoxInfo;

class YoloV4 {
public:
    YoloV4(bool useGPU, const int yoloType);
    ~YoloV4();
    std::vector<BoxInfo> detectv4(UIImage *image, float threshold, float nms_threshold);
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
    static std::vector<BoxInfo> decode_inferv4(ncnn::Mat &data, const yolocv::YoloSize& frame_size, int net_size,int num_classes,float threshold);
//    static void nms(std::vector<BoxInfo>& result,float nms_threshold);
    ncnn::Net* Net;
    int input_size = 640 / 2; // 416
    int num_class = 80;
public:
    static YoloV4 *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //YOLOV4_H
