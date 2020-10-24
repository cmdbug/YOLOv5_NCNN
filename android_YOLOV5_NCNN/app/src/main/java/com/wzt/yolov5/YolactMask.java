package com.wzt.yolov5;

import android.graphics.Color;

import java.util.Random;

class YolactMask {
    public float left, top, right, bottom;
    public int label;
    public float prob;
    public float[] maskdata;
    public char[] mask;

    public static String[] labels={
            "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"};

    public int colors[][] = {
        {244, 67,  54},
        {233, 30,  99},
        {156, 39,  176},
        {103, 58,  183},
        {63,  81,  181},
        {33,  150, 243},
        {3,   169, 244},
        {0,   188, 212},
        {0,   150, 136},
        {76,  175, 80},
        {139, 195, 74},
        {205, 220, 57},
        {255, 235, 59},
        {255, 193, 7},
        {255, 152, 0},
        {255, 87,  34},
        {121, 85,  72},
        {158, 158, 158},
        {96,  125, 139}
    };

    public YolactMask(float left, float top, float right, float bottom, int label, float prob, float[] maskdata, char[] mask) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
        this.label = label;
        this.prob = prob;
        this.maskdata = maskdata;
        this.mask = mask;
    }

    public int getColor(){
        Random random = new Random(label);
        return Color.argb(255,random.nextInt(256),random.nextInt(256),random.nextInt(256));
//        return Color.argb(255, colors[label % 19][0], colors[label % 19][1], colors[label % 19][2]);
    }

    public float getLeft() {
        return left;
    }

    public void setLeft(float left) {
        this.left = left;
    }

    public float getTop() {
        return top;
    }

    public void setTop(float top) {
        this.top = top;
    }

    public float getRight() {
        return right;
    }

    public void setRight(float right) {
        this.right = right;
    }

    public float getBottom() {
        return bottom;
    }

    public void setBottom(float bottom) {
        this.bottom = bottom;
    }

    public String getLabel() {
        return labels[label];
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public float getProb() {
        return prob;
    }

    public void setProb(float prob) {
        this.prob = prob;
    }

    public float[] getMaskdata() {
        return maskdata;
    }

    public void setMaskdata(float[] maskdata) {
        this.maskdata = maskdata;
    }

    public char[] getMask() {
        return mask;
    }

    public void setMask(char[] mask) {
        this.mask = mask;
    }
}
