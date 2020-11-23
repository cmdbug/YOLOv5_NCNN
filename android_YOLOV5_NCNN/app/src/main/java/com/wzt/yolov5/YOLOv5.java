package com.wzt.yolov5;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class YOLOv5 {
    static {
        System.loadLibrary("yolov5");
    }

    public static native void init(AssetManager manager, boolean useGPU);
    public static native Box[] detect(Bitmap bitmap, double threshold, double nms_threshold);

    public static native void initCustomLayer(AssetManager manager, boolean useGPU);
    public static native Box[] detectCustomLayer(Bitmap bitmap, double threshold, double nms_threshold);
}
