//
// Created by WZTENG on 2020/08/17 017.
//

#include "SimplePose.h"
#include "YoloV5.h"

bool SimplePose::hasGPU = true;
bool SimplePose::toUseGPU = true;
SimplePose *SimplePose::detector = nullptr;

SimplePose::SimplePose(AAssetManager *mgr, bool useGPU) {
    hasGPU = ncnn::get_gpu_count() > 0;
    toUseGPU = hasGPU && useGPU;

    PersonNet = new ncnn::Net();
    // opt 需要在加载前设置
    PersonNet->opt.use_vulkan_compute = toUseGPU;  // gpu
    PersonNet->opt.use_fp16_arithmetic = true;  // fp16运算加速
    PersonNet->load_param(mgr, "person_detector.param");
    PersonNet->load_model(mgr, "person_detector.bin");
//    LOGD("person_detector");

    PoseNet = new ncnn::Net();
    PoseNet->opt.use_vulkan_compute = toUseGPU;  // gpu
    PoseNet->opt.use_fp16_arithmetic = true;  // fp16运算加速
    PoseNet->load_param(mgr, "Ultralight-Nano-SimplePose.param");
    PoseNet->load_model(mgr, "Ultralight-Nano-SimplePose.bin");
//    LOGD("ultralight-nano-simplepose");

}

SimplePose::~SimplePose() {
    PersonNet->clear();
    PoseNet->clear();
    delete PersonNet;
    delete PoseNet;
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
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("hybridsequential0_conv7_fwd", out);
    keypoints.clear();
//    LOGD("pose out.c:%d", out.c);
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

std::vector<PoseResult> SimplePose::detect(JNIEnv *env, jobject image) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);

    ncnn::Mat src_img = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB,
                                                              img_size.width, img_size.height);
    int img_w = img_size.width;
    int img_h = img_size.height;
    /** ncnn::Mat -> cv::Mat **/
//    cv::Mat bgr(src_img.h, src_img.w, CV_8UC3);
//    for (int c = 0; c < 3; c++) {
//        for (int i = 0; i < src_img.h; i++) {
//            for (int j = 0; j < src_img.w; j++) {
//                float t = ((float *) src_img.data)[j + i * src_img.w + c * src_img.h * src_img.w];
//                bgr.data[(2 - c) + j * 3 + i * src_img.w * 3] = t;
//            }
//        }
//    }
//    https://github.com/Tencent/ncnn/wiki/use-ncnn-with-opencv#ncnn-to-opencv
//    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
//    float mean[3] = {0, 0, 0};
//    src_img.substract_mean_normalize(mean, norm);
    cv::Mat bgr(src_img.h, src_img.w, CV_8UC3);
    src_img.to_pixels(bgr.data, ncnn::Mat::PIXEL_RGB2BGR);
//    LOGD("bgr w:%d h:%d", bgr.cols, bgr.rows);

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB,
                                                         detector_size_width, detector_size_height);


    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in.substract_mean_normalize(mean, norm);

    auto ex = PersonNet->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);

//    LOGD("person out.h:%d", out.h);
    std::vector<PoseResult> poseResults;
//    std::vector<KeyPoint> keyPointList;
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
//        LOGD("x1:%f y1:%f x2:%f y2:%f\n", x1, y1, x2, y2);
        cv::Mat roi = bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
//        LOGD("roi w:%d h:%d", roi.cols, roi.rows);

        std::vector<KeyPoint> keypoints;
        runpose(roi, pose_size_width, pose_size_height, keypoints, x1, y1);
//        draw_pose(image, keypoints);
//        keyPointList.insert(keyPointList.begin(), keypoints.begin(), keypoints.end());

        BoxInfo box;
        box.x1 = x1;
        box.x2 = x2;
        box.y1 = y1;
        box.y2 = y2;
        box.label = label;
        box.score = score;
//        boxInfoList.push_back(box);

        PoseResult poseResult;
        poseResult.keyPoints = keypoints;
        poseResult.boxInfos = box;
        poseResults.push_back(poseResult);
    }
//    result.insert(result.begin(), boxes.begin(), boxes.end());
//    return keyPointList;
    return poseResults;
}

