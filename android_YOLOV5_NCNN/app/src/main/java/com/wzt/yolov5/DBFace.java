package com.wzt.yolov5;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

class DBFace {
    static {
        System.loadLibrary("yolov5");
    }

    public static native void init(AssetManager manager, boolean useGPU);
    public static native KeyPoint[] detect(Bitmap bitmap, double threshold, double nms_threshold);
}
