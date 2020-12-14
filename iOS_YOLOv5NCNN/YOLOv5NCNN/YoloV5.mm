//
//  YoloV5.cpp
//  YOLOv5NCNN
//
//  Created by WZTENG on 2020/7/6.
//  Copyright © 2020 TENG. All rights reserved.
//

#include "YoloV5.h"

bool YoloV5::hasGPU = false;
bool YoloV5::toUseGPU = false;
YoloV5* YoloV5::detector = nullptr;

YoloV5::YoloV5(bool useGPU) {
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    
    Net = new ncnn::Net();
    Net->opt.use_vulkan_compute = toUseGPU;
    Net->opt.use_fp16_arithmetic = true;
    NSString *parmaPath = [[NSBundle mainBundle] pathForResource:@"yolov5" ofType:@"param"];
    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"yolov5" ofType:@"bin"];
    int rp = Net->load_param([parmaPath UTF8String]);
    int rm = Net->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model success!");
    } else {
        fprintf(stderr, "net load fail,param:%d model:%d", rp, rm);
    }

}


YoloV5::~YoloV5() {
    Net->clear();
    delete Net;
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
    printf("delete YoloV5");
}

std::vector<BoxInfo> YoloV5::dectect(UIImage *image, float threshold, float nms_threshold) {
    int w = image.size.width;
    int h = image.size.height;
    unsigned char* rgba = new unsigned char[w * h * 4];
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGContextRef contextRef = CGBitmapContextCreate(rgba, w, h, 8, w * 4, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), image.CGImage);
    CGContextRelease(contextRef);
    
    ncnn::Mat in_net = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, w, h, input_size/2, input_size/2);
    float norm[3] = {1/255.0f, 1/255.0f, 1/255.0f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);
    auto ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    if (toUseGPU) {
        ex.set_vulkan_compute(toUseGPU);
    }
#endif
    ex.input(0, in_net);
    std::vector<BoxInfo> result;
    for (const auto& layer: layers) {
        ncnn::Mat blob;
        ex.extract(layer.name.c_str(), blob);
        auto boxes = decode_infer(blob, layer.stride, {w, h}, input_size, num_class, layer.anchors, threshold);
        result.insert(result.begin(), boxes.begin(), boxes.end());
    }
    nms(result, nms_threshold);
    delete[] rgba;
    return result;
}

inline float fast_exp(float x) {
    union {uint32_t i; float f;} v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<BoxInfo> YoloV5::decode_infer(ncnn::Mat &data, int stride, const yolocv::YoloSize &frame_size, int net_size, int num_classes, const std::vector<yolocv::YoloSize>& anchors, float threshold) {
    std::vector<BoxInfo> result;
    int grid_size = int(sqrt(data.h));
    float *mat_data[data.c];
    for (int i=0; i<data.c; i++) {
        mat_data[i] = data.channel(i);
    }
    float cx, cy, w, h;
    for (int shift_y = 0; shift_y < grid_size; shift_y++) {
        for (int shift_x = 0; shift_x < grid_size; shift_x++) {
//            int loc = shift_x + shift_y * grid_size;
            for (int i = 0; i < 3; i++) {
                float *record = mat_data[i];
                float *cls_ptr = record + 5;
                for (int cls = 0; cls < num_classes; cls++) {
                    float score = sigmoid(cls_ptr[cls]) * sigmoid(record[4]);
                    if (score > threshold) {
                        cx = (sigmoid(record[0]) * 2.f - 0.5f + (float)shift_x) * (float)stride;
                        cy = (sigmoid(record[1]) * 2.f - 0.5f + (float)shift_y) * (float)stride;
                        w = pow(sigmoid(record[2]) * 2.f, 2) * anchors[i].width;
                        h = pow(sigmoid(record[3]) * 2.f, 2) * anchors[i].height;
                        //printf("[grid size=%d, stride = %d]x y w h %f %f %f %f\n",grid_size,stride,record[0],record[1],record[2],record[3]);
                        BoxInfo box;
                        box.x1 = std::max(0, std::min(frame_size.width,int((cx - w / 2.f) * (float)frame_size.width / (float)net_size)));
                        box.y1 = std::max(0, std::min(frame_size.height,int((cy - h / 2.f) * (float)frame_size.height / (float)net_size)));
                        box.x2 = std::max(0, std::min(frame_size.width,int((cx + w / 2.f) * (float)frame_size.width / (float)net_size)));
                        box.y2 = std::max(0, std::min(frame_size.height,int((cy + h / 2.f) * (float)frame_size.height / (float)net_size)));
                        box.score = score;
                        box.label = cls;
                        result.push_back(box);
                    }
                }
            }
            for (auto& ptr:mat_data) {
                ptr += (num_classes + 5);
            }
        }
    }
    return result;
}

void YoloV5::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b){return a.score > b.score;});
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::max(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::max(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr > NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

