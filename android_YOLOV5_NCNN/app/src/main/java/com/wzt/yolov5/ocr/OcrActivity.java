package com.wzt.yolov5.ocr;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.Toast;

import com.github.chrisbanes.photoview.PhotoView;
import com.wzt.yolov5.R;

import java.io.IOException;
import java.util.Locale;

public class OcrActivity extends AppCompatActivity {

    private static final int REQUEST_PICK_IMAGE = 2;

    protected Button btnPhoto;
    protected ImageView imageSrc;
    //    protected ImageView imageResult;
    protected PhotoView imageResult;
    protected EditText etResult;

    protected Switch swShowText;

    protected boolean showText;
    protected Bitmap mutableBitmap;

    public static boolean USE_GPU = false;

    long startTime = 0;
    long endTime = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_ocr);

        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    1
            );
            finish();
        }
        initView();
        ChineseOCRLite.init(getAssets(), USE_GPU);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (data == null) {
            return;
        }
        final Bitmap image = getPicture(data.getData());
        if (image == null) {
            Toast.makeText(this, "Photo is null", Toast.LENGTH_SHORT).show();
            return;
        }
        imageResult.setImageResource(R.drawable.ic_launcher_foreground);
        imageSrc.setImageBitmap(image);
        etResult.setText("Please wait...");

        startTime = System.currentTimeMillis();
        Thread ocrThread = new Thread(new Runnable() {
            @Override
            public void run() {
                mutableBitmap = image.copy(Bitmap.Config.ARGB_8888, true);
                final OCRResult[] ocrResult = ChineseOCRLite.detect(image, 1080);
                final StringBuilder allText = new StringBuilder();
                if (ocrResult != null && ocrResult.length > 0) {
                    mutableBitmap = drawResult(mutableBitmap, ocrResult);
                    for (OCRResult result : ocrResult) {
                        allText.append(result.text).append("\r\n");
                    }
                }
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        endTime = System.currentTimeMillis();
                        imageResult.setImageBitmap(mutableBitmap);
                        etResult.setText(String.format(Locale.CHINESE,
                                "[ %s ] Chinese_OCR_lite\nSize: %dx%d\nTime: %.3f s\nNum: %d\nText:\n%s",
                                USE_GPU ? "GPU" : "CPU",
                                mutableBitmap.getWidth(),
                                mutableBitmap.getHeight(),
                                (endTime - startTime) / 1000.0,
                                ocrResult != null ? ocrResult.length : 0, allText));
                    }
                });
            }
        });
        ocrThread.start();
    }

    protected Bitmap drawResult(Bitmap mutableBitmap, OCRResult[] results) {
        if (results == null || results.length <= 0) {
            return mutableBitmap;
        }
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint boxPaint = new Paint();
        boxPaint.setAlpha(200);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(2 * Math.min(mutableBitmap.getWidth(), mutableBitmap.getHeight()) / 800.0f);
        boxPaint.setTextSize(15 * Math.min(mutableBitmap.getWidth(), mutableBitmap.getHeight()) / 800.0f);
        boxPaint.setColor(Color.BLUE);
        boxPaint.setAntiAlias(true);
        for (OCRResult result : results) {
            boxPaint.setColor(Color.RED);
            boxPaint.setStyle(Paint.Style.FILL);
            // 框
            canvas.drawLine((float) result.boxes[0], (float) result.boxes[1], (float) result.boxes[2], (float) result.boxes[3], boxPaint);
            canvas.drawLine((float) result.boxes[2], (float) result.boxes[3], (float) result.boxes[4], (float) result.boxes[5], boxPaint);
            canvas.drawLine((float) result.boxes[4], (float) result.boxes[5], (float) result.boxes[6], (float) result.boxes[7], boxPaint);
            canvas.drawLine((float) result.boxes[6], (float) result.boxes[7], (float) result.boxes[0], (float) result.boxes[1], boxPaint);
            // 文字
            if (showText) {  // 防止太乱
                double angle = getBoxAngle(result, true);
                canvas.save();
                canvas.rotate((float) angle, (float) result.boxes[0], (float) result.boxes[1] - 5);
                boxPaint.setColor(Color.BLUE);  // 防止有角度的框与之重叠
                if (angle > 70) {
                    canvas.drawText(String.format(Locale.CHINESE, "%s  (%.3f)", result.text, result.boxScore[0]),
                            (float) result.boxes[0] + 5, (float) result.boxes[1] + 15, boxPaint);
                } else {
                    canvas.drawText(String.format(Locale.CHINESE, "%s  (%.3f)", result.text, result.boxScore[0]),
                            (float) result.boxes[0], (float) result.boxes[1] - 5, boxPaint);
                }
                canvas.restore();
            }
            // 提示
            boxPaint.setColor(Color.YELLOW);  // 左上角画个红点
            canvas.drawPoint((float) result.boxes[0], (float) result.boxes[1], boxPaint);
            boxPaint.setColor(Color.GREEN);  // 右下角画个绿点
            canvas.drawPoint((float) result.boxes[4], (float) result.boxes[5], boxPaint);
        }
        return mutableBitmap;
    }

    /**
     * 自己根据框简单算下角度
     *
     * @param ocrResult
     * @param toDegrees
     * @return
     */
    protected double getBoxAngle(OCRResult ocrResult, boolean toDegrees) {
        double angle = 0.0f;
        if (ocrResult == null) {
            return angle;
        }
        // 0 1  2 3  4 5  6 7
        // x0y0 x1y1 x2y2 x3y3
        double dx1 = ocrResult.boxes[2] - ocrResult.boxes[0];
        double dy1 = ocrResult.boxes[3] - ocrResult.boxes[1];
        double dis1 = dy1 * dy1 + dx1 * dx1;
        double dx2 = ocrResult.boxes[4] - ocrResult.boxes[2];
        double dy2 = ocrResult.boxes[5] - ocrResult.boxes[3];
        double dis2 = dy2 * dy2 + dx2 * dx2;
        if (dis1 > dis2) {
            if (dx1 != 0) {
                angle = Math.asin(dy1 / dx1);
            }
        } else {
            if (dx2 != 0) {
                angle = Math.asin(dx2 / dy2);
            }
        }
        if (toDegrees) {
            angle = Math.toDegrees(angle);
            if (dis2 > dis1) {
                angle = angle + 90;
            }
//            Log.d("wzt", "degrees:" + angle + " dx:" + dx1 + " dy:" + dy1);
            return angle;
        }
//        Log.d("wzt", "angle:" + angle + " dx:" + dx1 + " dy:" + dy1);
        return angle;
    }

    public Bitmap getPicture(Uri selectedImage) {
        String[] filePathColumn = {MediaStore.Images.Media.DATA};
        Cursor cursor = this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
        if (cursor == null) {
            return null;
        }
        cursor.moveToFirst();
        int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
        String picturePath = cursor.getString(columnIndex);
        cursor.close();
        Bitmap bitmap = BitmapFactory.decodeFile(picturePath);
        if (bitmap == null) {
            return null;
        }
        int rotate = readPictureDegree(picturePath);
        return rotateBitmapByDegree(bitmap, rotate);
    }

    public int readPictureDegree(String path) {
        int degree = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degree = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degree = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degree = 270;
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return degree;
    }

    public Bitmap rotateBitmapByDegree(Bitmap bm, int degree) {
        Bitmap returnBm = null;
        Matrix matrix = new Matrix();
        matrix.postRotate(degree);
        try {
            returnBm = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(),
                    bm.getHeight(), matrix, true);
        } catch (OutOfMemoryError e) {
            e.printStackTrace();
        }
        if (returnBm == null) {
            returnBm = bm;
        }
        if (bm != returnBm) {
            bm.recycle();
        }
        return returnBm;
    }

    protected void initView() {
        btnPhoto = findViewById(R.id.btn_photo);
        btnPhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK);
                intent.setType("image/*");
                startActivityForResult(intent, REQUEST_PICK_IMAGE);
            }
        });
        imageSrc = findViewById(R.id.image_src);
        imageResult = findViewById(R.id.image_result);
//        imageResult.enable();
//        imageResult.setScaleType(ImageView.ScaleType.CENTER_INSIDE);
//        imageResult.setMaxScale(3.0f);
        etResult = findViewById(R.id.et_info);
        swShowText = findViewById(R.id.sw_show_text);
        showText = swShowText.isChecked();
        swShowText.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                showText = isChecked;
                Toast.makeText(OcrActivity.this, showText ? "Show text" : "Hide text", Toast.LENGTH_SHORT).show();
            }
        });
    }
}