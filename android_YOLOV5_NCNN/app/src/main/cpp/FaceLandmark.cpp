//
// Created by WZTENG on 2020/08/29 029.
//

#include "FaceLandmark.h"
#include "SimplePose.h"

bool FaceLandmark::hasGPU = true;
bool FaceLandmark::toUseGPU = true;
FaceLandmark *FaceLandmark::detector = nullptr;

FaceLandmark::FaceLandmark(AAssetManager *mgr, bool useGPU) {
    hasGPU = ncnn::get_gpu_count() > 0;
    toUseGPU = hasGPU && useGPU;

    FaceNet = new ncnn::Net();
    // opt 需要在加载前设置
    FaceNet->opt.use_vulkan_compute = toUseGPU;  // gpu
    FaceNet->opt.use_fp16_arithmetic = true;  // fp16运算加速
    FaceNet->load_param(mgr, "yoloface-500k.param");
    FaceNet->load_model(mgr, "yoloface-500k.bin");
//    LOGD("face_detector");

    LandmarkNet = new ncnn::Net();
    LandmarkNet->opt.use_vulkan_compute = toUseGPU;  // gpu
    LandmarkNet->opt.use_fp16_arithmetic = true;  // fp16运算加速
    LandmarkNet->load_param(mgr, "landmark106.param");
    LandmarkNet->load_model(mgr, "landmark106.bin");
//    LOGD("landmark106");

}

FaceLandmark::~FaceLandmark() {
    FaceNet->clear();
    LandmarkNet->clear();
    delete FaceNet;
    delete LandmarkNet;
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
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
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

std::vector<FaceKeyPoint> FaceLandmark::detect(JNIEnv *env, jobject image) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);

    ncnn::Mat src_img = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB,
                                                              img_size.width, img_size.height);
    int img_w = img_size.width;
    int img_h = img_size.height;
    /** ncnn::Mat -> cv::Mat **/
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

    auto ex = FaceNet->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
    ex.input("data", in);
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
    return keyPointList;
}
