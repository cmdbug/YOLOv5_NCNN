package com.wzt.yolov5;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

class FaceLandmark {
    static {
        System.loadLibrary("yolov5");
    }

    public static native void init(AssetManager manager, boolean useGPU);
    public static native FaceKeyPoint[] detect(Bitmap bitmap);
}
