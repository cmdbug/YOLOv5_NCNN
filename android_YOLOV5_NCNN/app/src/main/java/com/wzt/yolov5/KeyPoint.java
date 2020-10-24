package com.wzt.yolov5;


public class KeyPoint {
    public float[] x;
    public float[] y;
    public float x0, y0, x1, y1;
    public float score;

    public KeyPoint(float[] x, float[] y, float x0, float y0, float x1, float y1, float score) {
        this.x = x;
        this.y = y;
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
        this.score = score;
    }

    public float[] getX() {
        return x;
    }

    public void setX(float[] x) {
        this.x = x;
    }

    public float[] getY() {
        return y;
    }

    public void setY(float[] y) {
        this.y = y;
    }

    public float getX0() {
        return x0;
    }

    public void setX0(float x0) {
        this.x0 = x0;
    }

    public float getY0() {
        return y0;
    }

    public void setY0(float y0) {
        this.y0 = y0;
    }

    public float getX1() {
        return x1;
    }

    public void setX1(float x1) {
        this.x1 = x1;
    }

    public float getY1() {
        return y1;
    }

    public void setY1(float y1) {
        this.y1 = y1;
    }

    public float getScore() {
        return score;
    }

    public void setScore(float score) {
        this.score = score;
    }
}
