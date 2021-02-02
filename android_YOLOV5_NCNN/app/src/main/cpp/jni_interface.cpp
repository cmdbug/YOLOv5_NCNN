#include <jni.h>
#include <string>
#include <ncnn/gpu.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "YoloV5.h"
#include "YoloV4.h"
#include "SimplePose.h"
#include "Yolact.h"
#include "ocr.h"
#include "ENet.h"
#include "FaceLandmark.h"
#include "DBFace.h"
#include "MbnFCN.h"
#include "MobileNetV3Seg.h"
#include "YoloV5CustomLayer.h"
#include "NanoDet.h"


JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    ncnn::create_gpu_instance();
    if (ncnn::get_gpu_count() > 0) {
        YoloV5::hasGPU = true;
        YoloV4::hasGPU = true;
        SimplePose::hasGPU = true;
        Yolact::hasGPU = true;
        OCR::hasGPU = true;
        ENet::hasGPU = true;
        FaceLandmark::hasGPU = true;
        DBFace::hasGPU = true;
        MbnFCN::hasGPU = true;
        MBNV3Seg::hasGPU = true;
    }
//    LOGD("jni onload");
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    ncnn::destroy_gpu_instance();
    delete YoloV5::detector;
    delete YoloV4::detector;
    delete SimplePose::detector;
    delete Yolact::detector;
    delete OCR::detector;
    delete ENet::detector;
    delete FaceLandmark::detector;
    delete DBFace::detector;
    delete MbnFCN::detector;
    delete MBNV3Seg::detector;
//    LOGD("jni onunload");
}


/*********************************************************************************************
                                         Yolov5
 ********************************************************************************************/
extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_YOLOv5_init(JNIEnv *env, jclass, jobject assetManager, jboolean useGPU) {
    if (YoloV5::detector != nullptr) {
        delete YoloV5::detector;
        YoloV5::detector = nullptr;
    }
    if (YoloV5::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        YoloV5::detector = new YoloV5(mgr, "yolov5.param", "yolov5.bin", useGPU);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_wzt_yolov5_YOLOv5_detect(JNIEnv *env, jclass, jobject image, jdouble threshold, jdouble nms_threshold) {
    auto result = YoloV5::detector->detect(env, image, threshold, nms_threshold);

    auto box_cls = env->FindClass("com/wzt/yolov5/Box");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &box:result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, box.x1, box.y1, box.x2, box.y2, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}

// ***************************************[ Yolov5 Custom Layer ]****************************************
extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_YOLOv5_initCustomLayer(JNIEnv *env, jclass, jobject assetManager, jboolean useGPU) {
    if (YoloV5CustomLayer::detector != nullptr) {
        delete YoloV5CustomLayer::detector;
        YoloV5CustomLayer::detector = nullptr;
    }
    if (YoloV5CustomLayer::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        YoloV5CustomLayer::detector = new YoloV5CustomLayer(mgr, "yolov5s_customlayer.param", "yolov5s_customlayer.bin", useGPU);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_wzt_yolov5_YOLOv5_detectCustomLayer(JNIEnv *env, jclass, jobject image, jdouble threshold, jdouble nms_threshold) {
    auto result = YoloV5CustomLayer::detector->detect(env, image, threshold, nms_threshold);

    auto box_cls = env->FindClass("com/wzt/yolov5/Box");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &box:result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, box.x1, box.y1, box.x2, box.y2, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}

/*********************************************************************************************
                                         YOLOv4-tiny
 yolov4官方ncnn模型下载地址
 darknet2ncnn:https://drive.google.com/drive/folders/1YzILvh0SKQPS_lrb33dmGNq7aVTKPWS0
 ********************************************************************************************/

// 20200813 增加 MobileNetV2-YOLOv3-Nano-coco
// 20201124 增加 yolo-fastest-xl

extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_YOLOv4_init(JNIEnv *env, jclass, jobject assetManager, jint yoloType, jboolean useGPU) {
    if (YoloV4::detector != nullptr) {
        delete YoloV4::detector;
        YoloV4::detector = nullptr;
    }
    if (YoloV4::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        if (yoloType == 0) {
            YoloV4::detector = new YoloV4(mgr, "yolov4-tiny-opt.param", "yolov4-tiny-opt.bin", useGPU);
        } else if (yoloType == 1) {
            YoloV4::detector = new YoloV4(mgr, "MobileNetV2-YOLOv3-Nano-coco.param",
                                          "MobileNetV2-YOLOv3-Nano-coco.bin", useGPU);
        } else if (yoloType == 2) {
            YoloV4::detector = new YoloV4(mgr, "yolo-fastest-opt.param", "yolo-fastest-opt.bin", useGPU);
        }
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_wzt_yolov5_YOLOv4_detect(JNIEnv *env, jclass, jobject image, jdouble threshold, jdouble nms_threshold) {
    auto result = YoloV4::detector->detect(env, image, threshold, nms_threshold);

    auto box_cls = env->FindClass("com/wzt/yolov5/Box");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &box:result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, box.x1, box.y1, box.x2, box.y2, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}

/*********************************************************************************************
                                         NanoDet
 ********************************************************************************************/
extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_NanoDet_init(JNIEnv *env, jclass, jobject assetManager, jboolean useGPU) {
    if (NanoDet::detector != nullptr) {
        delete NanoDet::detector;
        NanoDet::detector = nullptr;
    }
    if (NanoDet::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        NanoDet::detector = new NanoDet(mgr, "nanodet_m.param", "nanodet_m.bin", useGPU);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_wzt_yolov5_NanoDet_detect(JNIEnv *env, jclass, jobject image, jdouble threshold, jdouble nms_threshold) {
    auto result = NanoDet::detector->detect(env, image, threshold, nms_threshold);

    auto box_cls = env->FindClass("com/wzt/yolov5/Box");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &box:result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, box.x1, box.y1, box.x2, box.y2, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}


/*********************************************************************************************
                                         SimplePose
 ********************************************************************************************/

extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_SimplePose_init(JNIEnv *env, jclass clazz, jobject assetManager, jboolean useGPU) {
    if (SimplePose::detector != nullptr) {
        delete SimplePose::detector;
        SimplePose::detector = nullptr;
    }
    if (SimplePose::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        SimplePose::detector = new SimplePose(mgr, useGPU);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_wzt_yolov5_SimplePose_detect(JNIEnv *env, jclass clazz, jobject image) {
    auto result = SimplePose::detector->detect(env, image);

    auto box_cls = env->FindClass("com/wzt/yolov5/KeyPoint");
    auto cid = env->GetMethodID(box_cls, "<init>", "([F[FFFFFF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    int KEY_NUM = 17;
    for (auto &keypoint : result) {
        env->PushLocalFrame(1);
        float x[KEY_NUM];
        float y[KEY_NUM];
        for (int j = 0; j < KEY_NUM; j++) {
            x[j] = keypoint.keyPoints[j].p.x;
            y[j] = keypoint.keyPoints[j].p.y;
        }
        jfloatArray xs = env->NewFloatArray(KEY_NUM);
        env->SetFloatArrayRegion(xs, 0, KEY_NUM, x);
        jfloatArray ys = env->NewFloatArray(KEY_NUM);
        env->SetFloatArrayRegion(ys, 0, KEY_NUM, y);

        jobject obj = env->NewObject(box_cls, cid, xs, ys,
                keypoint.boxInfos.x1, keypoint.boxInfos.y1, keypoint.boxInfos.x2, keypoint.boxInfos.y2,
                keypoint.boxInfos.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;

}

/*********************************************************************************************
                                         Yolact
 ********************************************************************************************/
jintArray matToBitmapIntArray(JNIEnv *env, const cv::Mat &image) {
    jintArray resultImage = env->NewIntArray(image.total());
    auto *_data = new jint[image.total()];
    for (int i = 0; i < image.total(); i++) {  // =========== 注意这里再确认下要不要除3
        char r = image.data[3 * i + 2];
        char g = image.data[3 * i + 1];
        char b = image.data[3 * i + 0];
        char a = (char) 255;
        _data[i] = (((jint) a << 24) & 0xFF000000) + (((jint) r << 16) & 0x00FF0000) +
                   (((jint) g << 8) & 0x0000FF00) + ((jint) b & 0x000000FF);
    }
    env->SetIntArrayRegion(resultImage, 0, image.total(), _data);
    delete[] _data;
    return resultImage;
}

jcharArray matToBitmapCharArray(JNIEnv *env, const cv::Mat &image) {
    jcharArray resultImage = env->NewCharArray(image.total());
    auto *_data = new jchar[image.total()];
    for (int i = 0; i < image.total(); i++) {
        char m = image.data[i];
        _data[i] = (m & 0xFF);
    }
    env->SetCharArrayRegion(resultImage, 0, image.total(), _data);
    delete[] _data;
    return resultImage;
}

extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_Yolact_init(JNIEnv *env, jclass clazz, jobject assetManager, jboolean useGPU) {
    if (Yolact::detector != nullptr) {
        delete Yolact::detector;
        Yolact::detector = nullptr;
    }
    if (Yolact::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        Yolact::detector = new Yolact(mgr, useGPU);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_wzt_yolov5_Yolact_detect(JNIEnv *env, jclass clazz, jobject image) {
    auto result = Yolact::detector->detect_yolact(env, image);

    auto yolact_mask = env->FindClass("com/wzt/yolov5/YolactMask");
//    auto cid = env->GetMethodID(yolact_mask, "<init>", "(FFFFIF[F[I)V");
    auto cid = env->GetMethodID(yolact_mask, "<init>", "(FFFFIF[F[C)V");
    jobjectArray ret = env->NewObjectArray(result.size(), yolact_mask, nullptr);
    int i = 0;
    for (auto &mask : result) {
//        LOGD("jni yolact mask rect x:%f y:%f", mask.rect.x, mask.rect.y);
//        LOGD("jni yolact maskdata size:%d", mask.maskdata.size());
//        LOGD("jni yolact mask size:%d", mask.mask.cols * mask.mask.rows);
//        jintArray jintmask = matToBitmapIntArray(env, mask.mask);
        jcharArray jcharmask = matToBitmapCharArray(env, mask.mask);

        env->PushLocalFrame(1);
        jfloatArray maskdata = env->NewFloatArray(mask.maskdata.size());
        auto *jnum = new jfloat[mask.maskdata.size()];
        for (int j = 0; j < mask.maskdata.size(); ++j) {
            *(jnum + j) = mask.maskdata[j];
        }
        env->SetFloatArrayRegion(maskdata, 0, mask.maskdata.size(), jnum);
        delete[] jnum;

        jobject obj = env->NewObject(yolact_mask, cid,
                                     mask.rect.x, mask.rect.y, mask.rect.x + mask.rect.width,
                                     mask.rect.y + mask.rect.height,
                                     mask.label, mask.prob, maskdata, jcharmask);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}


/*********************************************************************************************
                                         chineseocr-lite
 ********************************************************************************************/
jstring str2jstring(JNIEnv *env, const char *pat) {
    //定义java String类 strClass
    jclass strClass = (env)->FindClass("java/lang/String");
    //获取String(byte[],String)的构造器,用于将本地byte[]数组转换为一个新String
    jmethodID ctorID = (env)->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");
    //建立byte数组
    jbyteArray bytes = (env)->NewByteArray(strlen(pat));
    //将char* 转换为byte数组
    (env)->SetByteArrayRegion(bytes, 0, strlen(pat), (jbyte *) pat);
    // 设置String, 保存语言类型,用于byte数组转换至String时的参数
    jstring encoding = (env)->NewStringUTF("UTF-8");
    //将byte数组转换为java String,并输出
    return (jstring) (env)->NewObject(strClass, ctorID, bytes, encoding);
}

std::string jstring2str(JNIEnv *env, jstring jstr) {
    char *rtn = NULL;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("UTF-8");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    jbyteArray barr = (jbyteArray) env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte *ba = env->GetByteArrayElements(barr, JNI_FALSE);
    if (alen > 0) {
        rtn = (char *) malloc(alen + 1);
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    std::string stemp(rtn);
    free(rtn);
    return stemp;
}

extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_ocr_ChineseOCRLite_init(JNIEnv *env, jclass clazz, jobject assetManager, jboolean useGPU) {
    if (OCR::detector != nullptr) {
        delete OCR::detector;
        OCR::detector = nullptr;
    }
    if (OCR::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        OCR::detector = new OCR(env, clazz, mgr, useGPU);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_wzt_yolov5_ocr_ChineseOCRLite_detect(JNIEnv *env, jclass clazz, jobject bitmap, jint short_size) {
    auto ocrResult = OCR::detector->detect(env, bitmap, short_size);
//    LOGD("jni ocr result size:%ld", ocrResult.size());
//    LOGD("jni ocr ocrresult[0].pre_res size:%ld", ocrResult[0].pre_res.size());
//    LOGD("jni ocr ocrresult[0][0]:%s", ocrResult[0].pre_res[0].c_str());

    auto ocr_result = env->FindClass("com/wzt/yolov5/ocr/OCRResult");
    auto cid = env->GetMethodID(ocr_result, "<init>", "([D[DLjava/lang/String;)V");
    jobjectArray ret = env->NewObjectArray(ocrResult.size(), ocr_result, nullptr);
    int i = 0;
    for (auto &info : ocrResult) {
        // boxScore
        env->PushLocalFrame(1);
        jdoubleArray boxScoreData = env->NewDoubleArray(1);
        auto *bsnum = new jdouble[1];
        for (int j = 0; j < 1; ++j) {
            *(bsnum + j) = info.box_score;
        }
        env->SetDoubleArrayRegion(boxScoreData, 0, 1, bsnum);
        delete[] bsnum;

        // text
        char *cp = new char;
        for (auto &pre_re : info.pre_res) {
            strcat(cp, pre_re.c_str());
        }
        jstring text = str2jstring(env, cp);
        delete cp;

        // boxs
        jdoubleArray boxsData = env->NewDoubleArray(info.boxes.size() * 2);
        auto *bnum = new jdouble[info.boxes.size() * 2];
        for (int j = 0; j < info.boxes.size(); j++) {
            *(bnum + j * 2) = info.boxes[j].x;
            *(bnum + j * 2 + 1) = info.boxes[j].y;
        }
        env->SetDoubleArrayRegion(boxsData, 0, info.boxes.size() * 2, bnum);
        delete[] bnum;

        // 合并一下
        jobject obj = env->NewObject(ocr_result, cid, boxScoreData, boxsData, text);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
//    LOGD("jni ocr return");
    return ret;
}


/*********************************************************************************************
                                            ENet
 ********************************************************************************************/
extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_ENet_init(JNIEnv *env, jclass, jobject assetManager, jboolean useGPU) {
    if (ENet::detector != nullptr) {
        delete ENet::detector;
        ENet::detector = nullptr;
    }
    if (ENet::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        ENet::detector = new ENet(mgr, useGPU);
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_wzt_yolov5_ENet_detect(JNIEnv *env, jclass, jobject image) {
    auto result = ENet::detector->detect_enet(env, image);

    int output_w = result.w;
    int output_h = result.h;
//    LOGD("jni enet output w:%d h:%d", output_w, output_h);
    auto *output = new jfloat[output_w * output_h];
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            output[h * output_w + w] = result.row(h)[w];
        }
    }
    jfloatArray jfloats = env->NewFloatArray(output_w * output_h);
    if (jfloats == nullptr) {
        return nullptr;
    }
    env->SetFloatArrayRegion(jfloats, 0, output_w * output_h, output);
    delete[] output;
    return jfloats;
}

/*********************************************************************************************
                                        MobileNetv3_Seg
 ********************************************************************************************/
extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_MbnSeg_init(JNIEnv *env, jclass, jobject assetManager, jboolean useGPU) {
    if (MBNV3Seg::detector != nullptr) {
        delete MBNV3Seg::detector;
        MBNV3Seg::detector = nullptr;
    }
    if (MBNV3Seg::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        MBNV3Seg::detector = new MBNV3Seg(mgr, useGPU);
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_wzt_yolov5_MbnSeg_detect(JNIEnv *env, jclass, jobject image) {
    auto result = MBNV3Seg::detector->detect_mbnseg(env, image);

    int output_w = result.w;
    int output_h = result.h;
//    LOGD("jni mbnv3seg output w:%d h:%d", output_w, output_h);
    auto *output = new jfloat[output_w * output_h];
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            output[h * output_w + w] = result.row(h)[w];
        }
    }
    jfloatArray jfloats = env->NewFloatArray(output_w * output_h);
    if (jfloats == nullptr) {
        return nullptr;
    }
    env->SetFloatArrayRegion(jfloats, 0, output_w * output_h, output);
    delete[] output;
    return jfloats;
}

/*********************************************************************************************
                                        MobileNetv2_FCN
 ********************************************************************************************/
extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_MbnFCN_init(JNIEnv *env, jclass, jobject assetManager, jboolean useGPU) {
    if (MbnFCN::detector != nullptr) {
        delete MbnFCN::detector;
        MbnFCN::detector = nullptr;
    }
    if (MbnFCN::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        MbnFCN::detector = new MbnFCN(mgr, useGPU);
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_wzt_yolov5_MbnFCN_detect(JNIEnv *env, jclass, jobject image) {
    auto result = MbnFCN::detector->detect_mbnfcn(env, image);

    int output_w = result.w;
    int output_h = result.h;
//    LOGD("jni mbnfcn output w:%d h:%d", output_w, output_h);
    auto *output = new jfloat[output_w * output_h];
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            output[h * output_w + w] = result.row(h)[w];
        }
    }
    jfloatArray jfloats = env->NewFloatArray(output_w * output_h);
    if (jfloats == nullptr) {
        return nullptr;
    }
    env->SetFloatArrayRegion(jfloats, 0, output_w * output_h, output);
    delete[] output;
    return jfloats;
}

/*********************************************************************************************
                                         Face_Landmark
 ********************************************************************************************/

extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_FaceLandmark_init(JNIEnv *env, jclass clazz, jobject assetManager, jboolean useGPU) {
    if (FaceLandmark::detector != nullptr) {
        delete FaceLandmark::detector;
        FaceLandmark::detector = nullptr;
    }
    if (FaceLandmark::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        FaceLandmark::detector = new FaceLandmark(mgr, useGPU);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_wzt_yolov5_FaceLandmark_detect(JNIEnv *env, jclass clazz, jobject image) {
    auto result = FaceLandmark::detector->detect(env, image);

    auto box_cls = env->FindClass("com/wzt/yolov5/FaceKeyPoint");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &keypoint : result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, keypoint.p.x, keypoint.p.y);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;

}

/*********************************************************************************************
                                            DBFace
 ********************************************************************************************/
extern "C" JNIEXPORT void JNICALL
Java_com_wzt_yolov5_DBFace_init(JNIEnv *env, jclass clazz, jobject assetManager, jboolean useGPU) {
    if (DBFace::detector != nullptr) {
        delete DBFace::detector;
        DBFace::detector = nullptr;
    }
    if (DBFace::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        DBFace::detector = new DBFace(mgr, useGPU);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_wzt_yolov5_DBFace_detect(JNIEnv *env, jclass clazz, jobject image, jdouble threshold, jdouble nms_threshold) {
    auto result = DBFace::detector->detect(env, image, threshold, nms_threshold);
//    LOGD("jni dbface size:%d %f %f", result.size(), threshold, nms_threshold);

    auto box_cls = env->FindClass("com/wzt/yolov5/KeyPoint");
    auto cid = env->GetMethodID(box_cls, "<init>", "([F[FFFFFF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    int KEY_NUM = 5;
    for (auto &keypoint : result) {
        env->PushLocalFrame(1);
        float x[KEY_NUM];
        float y[KEY_NUM];
        for (int j = 0; j < KEY_NUM; j++) {
            x[j] = keypoint.landmark.x[j];
            y[j] = keypoint.landmark.y[j];
        }
        jfloatArray xs = env->NewFloatArray(KEY_NUM);
        env->SetFloatArrayRegion(xs, 0, KEY_NUM, x);
        jfloatArray ys = env->NewFloatArray(KEY_NUM);
        env->SetFloatArrayRegion(ys, 0, KEY_NUM, y);

        jobject obj = env->NewObject(box_cls, cid, xs, ys,
                                     keypoint.box.x, keypoint.box.y, keypoint.box.r, keypoint.box.b,
                                     (float) keypoint.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;

}
