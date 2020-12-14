//
// Created by WZTENG on 2020/08/17 017.
//

#include "SimplePose.h"
#include "YoloV5.h"

bool SimplePose::hasGPU = true;
bool SimplePose::toUseGPU = true;
SimplePose *SimplePose::detector = nullptr;

SimplePose::SimplePose(bool useGPU) {
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    
    PersonNet = new ncnn::Net();
    PersonNet->opt.use_vulkan_compute = toUseGPU;
    PersonNet->opt.use_fp16_arithmetic = true;
    NSString *parmaPath = [[NSBundle mainBundle] pathForResource:@"person_detector" ofType:@"param"];
    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"person_detector" ofType:@"bin"];
    int rp = PersonNet->load_param([parmaPath UTF8String]);
    int rm = PersonNet->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model success!");
    } else {
        fprintf(stderr, "net load fail,param:%d model:%d", rp, rm);
    }
    
    PoseNet = new ncnn::Net();
    PoseNet->opt.use_vulkan_compute = toUseGPU;
    PoseNet->opt.use_fp16_arithmetic = true;
    parmaPath = [[NSBundle mainBundle] pathForResource:@"Ultralight-Nano-SimplePose" ofType:@"param"];
    binPath = [[NSBundle mainBundle] pathForResource:@"Ultralight-Nano-SimplePose" ofType:@"bin"];
    rp = PoseNet->load_param([parmaPath UTF8String]);
    rm = PoseNet->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model success!");
    } else {
        fprintf(stderr, "net load fail,param:%d model:%d", rp, rm);
    }

}

SimplePose::~SimplePose() {
    PersonNet->clear();
    PoseNet->clear();
    delete PersonNet;
    delete PoseNet;
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

int SimplePose::runpose(cv::Mat &roi, int pose_size_w, int pose_size_h, std::vector<KeyPoint> &keypoints,
                        float x1, float y1) {
    int w = roi.cols;
    int h = roi.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_BGR2RGB, \
                                                 roi.cols, roi.rows, pose_size_w, pose_size_h);
//    LOGD("in w:%d h:%d", roi.cols, roi.rows);
    //数据预处理
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    auto ex = PoseNet->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
#endif
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("hybridsequential0_conv7_fwd", out);
    keypoints.clear();
    for (int p = 0; p < out.c; p++) {
        const ncnn::Mat m = out.channel(p);

        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for (int y = 0; y < out.h; y++) {
            const float *ptr = m.row(y);
            for (int x = 0; x < out.w; x++) {
                float prob = ptr[x];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_x = x;
                    max_y = y;
                }
            }
        }

        KeyPoint keypoint;
        keypoint.p = cv::Point2f(max_x * w / (float) out.w + x1, max_y * h / (float) out.h + y1);
        keypoint.prob = max_prob;
        keypoints.push_back(keypoint);
    }
    return 0;
}

std::vector<PoseResult> SimplePose::detect(UIImage *image) {
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

    auto ex = PersonNet->create_extractor();
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

    std::vector<PoseResult> poseResults;

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

        x1 = cx - 0.7 * pw;
        y1 = cy - 0.6 * ph;
        x2 = cx + 0.7 * pw;
        y2 = cy + 0.6 * ph;

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
        //截取人体ROI
        cv::Mat roi = bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();

        std::vector<KeyPoint> keypoints;
        runpose(roi, pose_size_width, pose_size_height, keypoints, x1, y1);

        BoxInfo box;
        box.x1 = x1;
        box.x2 = x2;
        box.y1 = y1;
        box.y2 = y2;
        box.label = label;
        box.score = score;

        PoseResult poseResult;
        poseResult.keyPoints = keypoints;
        poseResult.boxInfos = box;
        poseResults.push_back(poseResult);
    }

    delete[] rgba;
    return poseResults;
}

