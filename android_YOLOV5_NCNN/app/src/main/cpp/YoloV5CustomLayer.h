#ifndef YOLOV5_CUSTOMLAYER_H
#define YOLOV5_CUSTOMLAYER_H

#include "ncnn/net.h"
#include "YoloV5.h"

/**

ncnn 与 opencv 库冲突，需要重新编译 ncnn 或去掉 opencv 。（该项目保留 opencv ，如果需要两者都支持请看下面说明）

由于 ncnn 默认编译配置跟 opencv 官方默认编译配置冲突，先关闭 ncnn 自定义层功能，有需要的话可以自行编译 ncnn 库
编译 ncnn 时需要修改下编译配置，编译完成后在该项目 CMakeLists.txt 中使用以下参数开启关闭配置（也可以不改，直接默认）

opencv 官方默认库是开启的，ncnn 官方编译的库默认是关闭的。所以 CMakelista.txt 加上下面这句时 opencv 会报错。
# disable rtti and exceptions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

opencv 官方默认库是开启的，ncnn 官方编译的库默认是关闭的。所以 CMakelista.txt 加上下面这句时 ncnn 会报错。
# enable rtti
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

即：如果选择重新编译 ncnn 库，需要开启 rtti 选项，即：-DNCNN_DISABLE_RTTI=OFF -DNCNN_DISABLE_EXCEPTION=OFF
(没编译过 opencv，不懂)，自行编译完成后，记得把下面的：ENABLE_CUSTOM_LAYER 改为 1 开启

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
    YoloV5CustomLayer(AAssetManager *mgr, const char *param, const char *bin, bool useGPU);

    ~YoloV5CustomLayer();

    std::vector<BoxInfo> detect(JNIEnv *env, jobject image, float threshold, float nms_threshold);
//    std::vector<std::string> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//                                    "hair drier", "toothbrush"};
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
