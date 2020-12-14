//
// Created by WZTENG on 2020/09/24 028.
//

#include "MobileNetV3Seg.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

bool MBNV3Seg::hasGPU = true;
bool MBNV3Seg::toUseGPU = true;
MBNV3Seg *MBNV3Seg::detector = nullptr;

MBNV3Seg::MBNV3Seg(bool useGPU) {
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    
    MBNSegsim = new ncnn::Net();
    MBNSegsim->opt.use_vulkan_compute = toUseGPU;
    MBNSegsim->opt.use_fp16_arithmetic = true;
    NSString *parmaPath = [[NSBundle mainBundle] pathForResource:@"mbnv3_small" ofType:@"param"];
    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"mbnv3_small" ofType:@"bin"];
    int rp = MBNSegsim->load_param([parmaPath UTF8String]);
    int rm = MBNSegsim->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model success!");
    } else {
        fprintf(stderr, "net load fail,param:%d model:%d", rp, rm);
    }

}

MBNV3Seg::~MBNV3Seg() {
    MBNSegsim->clear();
    delete MBNSegsim;
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

ncnn::Mat MBNV3Seg::detect_mbnseg(UIImage *image) {
    int w = image.size.width;
    int h = image.size.height;
    unsigned char* rgba = new unsigned char[w * h * 4];
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGContextRef contextRef = CGBitmapContextCreate(rgba, w, h, 8, w * 4, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), image.CGImage);
    CGContextRelease(contextRef);
    
    ncnn::Mat in_net = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, w, h, target_size_w, target_size_h);
    float mean[3] = {123.68f, 116.28f, 103.53f};
    float norm[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    in_net.substract_mean_normalize(mean, norm);
    
    ncnn::Mat maskout;
    
    auto ex = MBNSegsim->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    if (toUseGPU) {
        ex.set_vulkan_compute(toUseGPU);
    }
#endif
    ex.input("input", in_net);
    ex.extract("output", maskout);

    int mask_c = maskout.c;
    int mask_w = maskout.w;
    int mask_h = maskout.h;
    
    cv::Mat prediction = cv::Mat::zeros(mask_h, mask_w, CV_8UC1);
    ncnn::Mat chn[mask_c];
    for (int i = 0; i < mask_c; i++) {
        chn[i] = maskout.channel(i);
    }
    for (int i = 0; i < mask_h; i++) {
        const float *pChn[mask_c];
        for (int c = 0; c < mask_c; c++) {
            pChn[c] = chn[c].row(i);
        }

        auto *pCowMask = prediction.ptr<uchar>(i);

        for (int j = 0; j < mask_w; j++) {
            int maxindex = 0;
            float maxvalue = -1000;
            for (int n = 0; n < mask_c; n++) {
                if (pChn[n][j] > maxvalue) {
                    maxindex = n;
                    maxvalue = pChn[n][j];
                }
            }
            pCowMask[j] = maxindex;
        }

    }

//    ncnn::Mat maskMat;
//    maskMat = ncnn::Mat::from_pixels(prediction.data, ncnn::Mat::PIXEL_GRAY, prediction.cols, prediction.rows);

    cv::Mat pred_resize;
    cv::resize(prediction, pred_resize, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);

    ncnn::Mat maskMat;
    maskMat = ncnn::Mat::from_pixels_resize(pred_resize.data, ncnn::Mat::PIXEL_GRAY,
                                            pred_resize.cols, pred_resize.rows,
                                            w, h);

//    LOGD("jni mbnv3seg maskMat 0:%f", maskMat.channel(0).row(0)[0]);
//    LOGD("jni mbnv3seg maskMat end w:%d h:%d", maskMat.w, maskMat.h);

    delete[] rgba;
    return maskMat;

}

