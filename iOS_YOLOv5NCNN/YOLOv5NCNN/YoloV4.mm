#include "YoloV4.h"

/*********************************************************************************************
                                        YOLOv4-tiny
yolov4官方ncnn模型下载地址
darknet2ncnn:https://drive.google.com/drive/folders/1YzILvh0SKQPS_lrb33dmGNq7aVTKPWS0
********************************************************************************************/

bool YoloV4::hasGPU = false;
bool YoloV4::toUseGPU = false;
YoloV4 *YoloV4::detector = nullptr;

YoloV4::YoloV4(bool useGPU, const int yoloType) {
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    
    Net = new ncnn::Net();
    Net->opt.use_vulkan_compute = toUseGPU;
    Net->opt.use_fp16_arithmetic = true;
    NSString *parmaPath = nil;
    NSString *binPath = nil;
    if (yoloType == 0) {
        parmaPath = [[NSBundle mainBundle] pathForResource:@"yolov4-tiny-opt" ofType:@"param"];
        binPath = [[NSBundle mainBundle] pathForResource:@"yolov4-tiny-opt" ofType:@"bin"];
    } else if (yoloType == 1) {
        parmaPath = [[NSBundle mainBundle] pathForResource:@"MobileNetV2-YOLOv3-Nano-coco" ofType:@"param"];
        binPath = [[NSBundle mainBundle] pathForResource:@"MobileNetV2-YOLOv3-Nano-coco" ofType:@"bin"];
    } else if (yoloType == 2) {
        parmaPath = [[NSBundle mainBundle] pathForResource:@"yolo-fastest-opt" ofType:@"param"];
        binPath = [[NSBundle mainBundle] pathForResource:@"yolo-fastest-opt" ofType:@"bin"];
    }
    int rp = Net->load_param([parmaPath UTF8String]);
    int rm = Net->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model success!");
    } else {
        fprintf(stderr, "net load fail,param:%d model:%d", rp, rm);
    }
}

YoloV4::~YoloV4() {
    Net->clear();
    delete Net;
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

std::vector<BoxInfo> YoloV4::detectv4(UIImage *image, float threshold, float nms_threshold) {
    int w = image.size.width;
    int h = image.size.height;
    unsigned char* rgba = new unsigned char[w * h * 4];
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGContextRef contextRef = CGBitmapContextCreate(rgba, w, h, 8, w * 4, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), image.CGImage);
    CGContextRelease(contextRef);
    
    ncnn::Mat in_net = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, w, h, input_size, input_size);
    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);
    auto ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    if (toUseGPU) {
        ex.set_vulkan_compute(hasGPU);
    }
#endif
    ex.input(0, in_net);
    std::vector<BoxInfo> result;
    ncnn::Mat blob;
    ex.extract("output", blob);
    auto boxes = decode_inferv4(blob, {w, h}, input_size, num_class, threshold);
    result.insert(result.begin(), boxes.begin(), boxes.end());
//    nms(result,nms_threshold);
    delete[] rgba;
    return result;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<BoxInfo>
YoloV4::decode_inferv4(ncnn::Mat &data, const yolocv::YoloSize &frame_size, int net_size, int num_classes, float threshold) {
    std::vector<BoxInfo> result;
    for (int i = 0; i < data.h; i++) {
        BoxInfo box;
        const float *values = data.row(i);
        box.label = values[0] - 1;
        box.score = values[1];
        box.x1 = values[2] * (float) frame_size.width;
        box.y1 = values[3] * (float) frame_size.height;
        box.x2 = values[4] * (float) frame_size.width;
        box.y2 = values[5] * (float) frame_size.height;
        result.push_back(box);
    }
    return result;
}

