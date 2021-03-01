package com.wzt.yolov5;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.wzt.yolov5.ocr.OcrActivity;

public class WelcomeActivity extends AppCompatActivity {

    private ToggleButton tbUseGpu;
    private Button yolov5s;
    private Button yolov4tiny;
    private Button mobilenetyolov3nano;
    private Button simplepose;
    private Button yolact;
    private Button chineseocrlite;
    private Button enet;
    private Button faceLandmark;
    private Button dbface;
    private Button mobilenetv2Fcn;
    private Button mobilenetv3Seg;
    private Button yolov5sCustomLayer;
    private Button nanoDet;
    private Button yoloFastestXL;
    private Button lightOpenPose;

    private boolean useGPU = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_welcome);

        tbUseGpu = findViewById(R.id.tb_use_gpu);
        tbUseGpu.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                useGPU = isChecked;
                MainActivity.USE_GPU = useGPU;
                OcrActivity.USE_GPU = useGPU;
                if (useGPU) {
                    AlertDialog.Builder builder = new AlertDialog.Builder(WelcomeActivity.this);
                    builder.setTitle("Warning");
                    builder.setMessage("If the GPU is too old, it may not work well in GPU mode.");
                    builder.setCancelable(true);
                    builder.setPositiveButton("OK", null);
                    AlertDialog dialog = builder.create();
                    dialog.show();
                } else {
                    Toast.makeText(WelcomeActivity.this, "CPU mode", Toast.LENGTH_SHORT).show();
                }
            }
        });

        yolov5s = findViewById(R.id.btn_start_detect1);
        yolov5s.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.YOLOV5S;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        yolov4tiny = findViewById(R.id.btn_start_detect2);
        yolov4tiny.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.YOLOV4_TINY;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        mobilenetyolov3nano = findViewById(R.id.btn_start_detect3);
        mobilenetyolov3nano.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.MOBILENETV2_YOLOV3_NANO;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        simplepose = findViewById(R.id.btn_start_detect4);
        simplepose.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.SIMPLE_POSE;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        yolact = findViewById(R.id.btn_start_detect5);
        yolact.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.YOLACT;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        chineseocrlite = findViewById(R.id.btn_start_detect6);
        chineseocrlite.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(WelcomeActivity.this, OcrActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        enet = findViewById(R.id.btn_start_detect7);
        enet.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.ENET;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        faceLandmark = findViewById(R.id.btn_start_detect8);
        faceLandmark.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.FACE_LANDMARK;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        dbface = findViewById(R.id.btn_start_detect9);
        dbface.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.DBFACE;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        mobilenetv2Fcn = findViewById(R.id.btn_start_detect10);
        mobilenetv2Fcn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.MOBILENETV2_FCN;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        mobilenetv3Seg = findViewById(R.id.btn_start_detect11);
        mobilenetv3Seg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.MOBILENETV3_SEG;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        yolov5sCustomLayer = findViewById(R.id.btn_start_detect12);
        yolov5sCustomLayer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.YOLOV5_CUSTOM_LAYER;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        nanoDet = findViewById(R.id.btn_start_detect13);
        nanoDet.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.NANODET;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        yoloFastestXL = findViewById(R.id.btn_start_detect14);
        yoloFastestXL.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.YOLO_FASTEST_XL;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        lightOpenPose = findViewById(R.id.btn_start_detect15);
        lightOpenPose.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MainActivity.USE_MODEL = MainActivity.LIGHT_OPEN_POSE;
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

    }


}
