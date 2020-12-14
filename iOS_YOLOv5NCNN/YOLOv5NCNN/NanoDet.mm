#include "NanoDet.h"

bool NanoDet::hasGPU = false;
bool NanoDet::toUseGPU = false;
NanoDet *NanoDet::detector = nullptr;

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

template<typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

NanoDet::NanoDet(bool useGPU) {
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    
    Net = new ncnn::Net();
    Net->opt.use_vulkan_compute = toUseGPU;
    Net->opt.use_fp16_arithmetic = true;
    NSString *parmaPath = [[NSBundle mainBundle] pathForResource:@"nanodet_m" ofType:@"param"];
    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"nanodet_m" ofType:@"bin"];
    int rp = Net->load_param([parmaPath UTF8String]);
    int rm = Net->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model success!");
    } else {
        fprintf(stderr, "net load fail,param:%d model:%d", rp, rm);
    }

}

NanoDet::~NanoDet() {
    Net->clear();
    delete Net;
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

std::vector<BoxInfo> NanoDet::detect(UIImage *image, float score_threshold, float nms_threshold) {
    int width = image.size.width;
    int height = image.size.height;
    unsigned char* rgba = new unsigned char[width * height * 4];
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGContextRef contextRef = CGBitmapContextCreate(rgba, width, height, 8, width * 4, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), image.CGImage);
    CGContextRelease(contextRef);
    
    float width_ratio = (float) width / (float) input_size;
    float height_ratio = (float) height / (float) input_size;

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, width, height, input_size, input_size);
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {0.017429f, 0.017507f, 0.01712475};
    input.substract_mean_normalize(mean_vals, norm_vals);

    auto ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    if (toUseGPU) {
        ex.set_vulkan_compute(toUseGPU);
    }
#endif
    ex.input("input.1", input);
    std::vector<std::vector<BoxInfo>> results;
    results.resize(this->num_class);

    for (const auto &head_info : this->heads_info) {
        ncnn::Mat dis_pred;
        ncnn::Mat cls_pred;
        ex.extract(head_info.dis_layer.c_str(), dis_pred);
        ex.extract(head_info.cls_layer.c_str(), cls_pred);

        decode_infer(cls_pred, dis_pred, head_info.stride, score_threshold, results, width_ratio, height_ratio);
    }

    std::vector<BoxInfo> dets;
    for (int i = 0; i < (int) results.size(); i++) {
        nms(results[i], nms_threshold);

        for (auto box : results[i]) {
            dets.push_back(box);
        }
    }
    delete[] rgba;
    return dets;
}


void NanoDet::decode_infer(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride, float threshold,
                           std::vector<std::vector<BoxInfo>> &results, float width_ratio, float height_ratio) {
    int feature_h = input_size / stride;
    int feature_w = input_size / stride;

    //cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
        const float *scores = cls_pred.row(idx);
        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; label++) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold) {
            //std::cout << "label:" << cur_label << " score:" << score << std::endl;
            const float *bbox_pred = dis_pred.row(idx);
            results[cur_label].push_back(
                    this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride, width_ratio, height_ratio));
            //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
            //cv::imshow("debug", debug_heatmap);
        }

    }
}

BoxInfo NanoDet::disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride, float width_ratio,
                      float height_ratio) {
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        float *dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f) * width_ratio;
    float ymin = (std::max)(ct_y - dis_pred[1], .0f) * height_ratio;
    float xmax = (std::min)(ct_x + dis_pred[2], (float) input_size) * width_ratio;
    float ymax = (std::min)(ct_y + dis_pred[3], (float) input_size) * height_ratio;

    //std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void NanoDet::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}
