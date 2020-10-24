package com.wzt.yolov5.ocr;

import android.content.res.AssetManager;
import android.graphics.Bitmap;


public class ChineseOCRLite {
    static {
        System.loadLibrary("yolov5");
    }

    public static native void init(AssetManager manager, boolean useGPU);
    public static native OCRResult[] detect(Bitmap bitmap, int short_size);
}
