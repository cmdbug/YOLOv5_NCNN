package com.wzt.yolov5;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

class Yolact {
    static {
        System.loadLibrary("yolov5");
    }

    public static native void init(AssetManager manager, boolean useGPU);
    public static native YolactMask[] detect(Bitmap bitmap);
}
