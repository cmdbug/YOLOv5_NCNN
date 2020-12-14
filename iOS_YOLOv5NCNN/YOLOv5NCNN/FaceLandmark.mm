//
// Created by WZTENG on 2020/08/29 029.
//

#include "FaceLandmark.h"
#include "SimplePose.h"

bool FaceLandmark::hasGPU = true;
bool FaceLandmark::toUseGPU = true;
FaceLandmark *FaceLandmark::detector = nullptr;

FaceLandmark::FaceLandmark(bool useGPU) {
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    
    FaceNet = new ncnn::Net();
    FaceNet->opt.use_vulkan_compute = toUseGPU;
    FaceNet->opt.use_fp16_arithmetic = true;
    NSString *parmaPath = [[NSBundle mainBundle] pathForResource:@"yoloface-500k" ofType:@"param"];
    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"yoloface-500k" ofType:@"bin"];
    int rp = FaceNet->load_param([parmaPath UTF8String]);
    int rm = FaceNet->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model success!");
    } else {
        fprintf(stderr, "net load fail,param:%d model:%d", rp, rm);
    }
    
    LandmarkNet = new ncnn::Net();
    LandmarkNet->opt.use_vulkan_compute = toUseGPU;
    LandmarkNet->opt.use_fp16_arithmetic = true;
    parmaPath = [[NSBundle mainBundle] pathForResource:@"landmark106" ofType:@"param"];
    binPath = [[NSBundle mainBundle] pathForResource:@"landmark106" ofType:@"bin"];
    rp = LandmarkNet->load_param([parmaPath UTF8String]);
    rm = LandmarkNet->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model success!");
    } else {
        fprintf(stderr, "net load fail,param:%d model:%d", rp, rm);
    }

}

FaceLandmark::~FaceLandmark() {
    FaceNet->clear();
    LandmarkNet->clear();
    delete FaceNet;
    delete LandmarkNet;
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

int FaceLandmark::runlandmark(cv::Mat &roi, int face_size_w, int face_size_h, std::vector<FaceKeyPoint> &keypoints,
                              float x1, float y1) {
    int w = roi.cols;
    int h = roi.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_BGR2RGB, \
                                                 roi.cols, roi.rows, face_size_w, face_size_h);
//    LOGD("in w:%d h:%d", roi.cols, roi.rows);
    //数据预处理
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    auto ex = LandmarkNet->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
#endif
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("bn6_3_bn6_3_scale", out);
    keypoints.clear();
//    LOGD("pose out.c:%d", out.c);
    float sw, sh;
    sw = (float) w / (float) landmark_size_width;
    sh = (float) h / (float) landmark_size_height;
    for (int i = 0; i < 106; i++) {
        float px, py;
        px = out[i * 2] * landmark_size_width * sw + x1;
        py = out[i * 2 + 1] * landmark_size_height * sh + y1;

        FaceKeyPoint keypoint;
        keypoint.p = cv::Point2f(px, py);
        keypoints.push_back(keypoint);
    }
    return 0;
}

std::vector<FaceKeyPoint> FaceLandmark::detect(UIImage *image) {
//    https://github.com/Tencent/ncnn/wiki/use-ncnn-with-opencv#ncnn-to-opencv
    int img_w = image.size.width;
    int img_h = image.size.height;
    unsigned char* rgba = new unsigned char[img_w * img_h * 4];
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGContextRef contextRef = CGBitmapContextCreate(rgba, img_w, img_h, 8, img_w * 4, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, img_w, img_h), image.CGImage);
    CGContextRelease(contextRef);
    
    ncnn::Mat src_img = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, img_w, img_h, img_w, img_h);
    
    cv::Mat bgr(src_img.h, src_img.w, CV_8UC3);
    src_img.to_pixels(bgr.data, ncnn::Mat::PIXEL_RGB2BGR);
    
    ncnn::Mat in_net = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, img_w, img_h, detector_size_width, detector_size_height);

    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);

    auto ex = FaceNet->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
#endif
    ex.input("data", in_net);
    ncnn::Mat out;
    ex.extract("output", out);

//    LOGD("person out.h:%d", out.h);
    std::vector<FaceKeyPoint> keyPointList;
//    std::vector<BoxInfo> boxInfoList;
    for (int i = 0; i < out.h; i++) {
        float x1, y1, x2, y2, score, label;
        float pw, ph, cx, cy;
        const float *values = out.row(i);

        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;

        pw = x2 - x1;
        ph = y2 - y1;
        cx = x1 + 0.5 * pw;
        cy = y1 + 0.5 * ph;

        x1 = cx - 0.55 * pw;
        y1 = cy - 0.35 * ph;
        x2 = cx + 0.55 * pw;
        y2 = cy + 0.55 * ph;

        score = values[1];
        label = values[0];

        //处理坐标越界问题
        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 < 0) x2 = 0;
        if (y2 < 0) y2 = 0;

        if (x1 > img_w) x1 = img_w;
        if (y1 > img_h) y1 = img_h;
        if (x2 > img_w) x2 = img_w;
        if (y2 > img_h) y2 = img_h;
        //截取脸ROI
//        LOGD("x1:%f y1:%f x2:%f y2:%f\n", x1, y1, x2, y2);
        if (x2 - x1 > 66 && y2 - y1 > 66) {
            cv::Mat roi = bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
//            LOGD("roi w:%d h:%d", roi.cols, roi.rows);
            std::vector<FaceKeyPoint> keypoints;
            runlandmark(roi, landmark_size_width, landmark_size_height, keypoints, x1, y1);
            keyPointList.insert(keyPointList.begin(), keypoints.begin(), keypoints.end());
        }

//        BoxInfo box;
//        box.x1 = x1;
//        box.x2 = x2;
//        box.y1 = y1;
//        box.y2 = y2;
//        box.label = label;
//        box.score = score;
//        boxInfoList.push_back(box);
    }
//    result.insert(result.begin(), boxes.begin(), boxes.end());
    
    delete[] rgba;
    return keyPointList;
}
