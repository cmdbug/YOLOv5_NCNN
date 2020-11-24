package com.wzt.yolov5;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class YOLOv4 {
    static {
        System.loadLibrary("yolov5");  // 存放在yolov5.so中
    }

    public static native void init(AssetManager manager, int yoloType, boolean useGPU);
    public static native Box[] detect(Bitmap bitmap, double threshold, double nms_threshold);
}
