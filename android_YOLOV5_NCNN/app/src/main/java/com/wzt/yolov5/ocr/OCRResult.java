package com.wzt.yolov5.ocr;

public class OCRResult {

    public double[] boxScore;  // 直接 boxScore[0] 就行
    public String text;
    public double[] boxes;  // xy xy xy xy


    public OCRResult (double[] boxScore, double[] boxes, String text) {
        this.boxScore = boxScore;
        this.boxes = boxes;
        this.text = text;
    }

    public double[] getBoxScore() {
        return boxScore;
    }

    public void setBoxScore(double[] boxScore) {
        this.boxScore = boxScore;
    }

    public double[] getBoxes() {
        return boxes;
    }

    public void setBoxes(double[] boxes) {
        this.boxes = boxes;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

}
