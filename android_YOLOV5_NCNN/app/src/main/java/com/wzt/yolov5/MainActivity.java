package com.wzt.yolov5;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.camera.core.UseCase;
import androidx.core.app.ActivityCompat;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PaintFlagsDrawFilter;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import wseemann.media.FFmpegMediaMetadataRetriever;

public class MainActivity extends AppCompatActivity {
    public static int YOLOV5S = 1;
    public static int YOLOV4_TINY = 2;
    public static int MOBILENETV2_YOLOV3_NANO = 3;
    public static int SIMPLE_POSE = 4;
    public static int YOLACT = 5;
    public static int ENET = 6;
    public static int FACE_LANDMARK = 7;
    public static int DBFACE = 8;
    public static int MOBILENETV2_FCN = 9;
    public static int MOBILENETV3_SEG = 10;
    public static int YOLOV5_CUSTOM_LAYER = 11;
    public static int NANODET = 12;
    public static int YOLO_FASTEST_XL = 13;
    public static int LIGHT_OPEN_POSE = 14;

    public static int USE_MODEL = MOBILENETV2_YOLOV3_NANO;
    public static boolean USE_GPU = false;

    public static CameraX.LensFacing CAMERA_ID = CameraX.LensFacing.BACK;

    private static final int REQUEST_CAMERA = 1;
    private static final int REQUEST_PICK_IMAGE = 2;
    private static final int REQUEST_PICK_VIDEO = 3;
    private static String[] PERMISSIONS_CAMERA = {
            Manifest.permission.CAMERA
    };
    private Toolbar toolbar;
    private ImageView resultImageView;
    private TextView tvNMS;
    private TextView tvThreshold;
    private SeekBar nmsSeekBar;
    private SeekBar thresholdSeekBar;
    private TextView tvNMNThreshold;
    private TextView tvInfo;
    private Button btnPhoto;
    private Button btnVideo;
    private double threshold = 0.3, nms_threshold = 0.7;
    private TextureView viewFinder;
    private SeekBar sbVideo;
    private SeekBar sbVideoSpeed;

    protected float videoSpeed = 1.0f;
    protected long videoCurFrameLoc = 0;
    public static int VIDEO_SPEED_MAX = 20 + 1;
    public static int VIDEO_SPEED_MIN = 1;

    private AtomicBoolean detectCamera = new AtomicBoolean(false);
    private AtomicBoolean detectPhoto = new AtomicBoolean(false);
    private AtomicBoolean detectVideo = new AtomicBoolean(false);

    private long startTime = 0;
    private long endTime = 0;
    private int width;
    private int height;

    double total_fps = 0;
    int fps_count = 0;

    protected Bitmap mutableBitmap;

    ExecutorService detectService = Executors.newSingleThreadExecutor();

    FFmpegMediaMetadataRetriever mmr;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    PERMISSIONS_CAMERA,
                    REQUEST_CAMERA
            );
            finish();
        }

        initModel();
        initViewID();
        initViewListener();

    }

    protected void initViewListener() {
        toolbar.setNavigationIcon(R.drawable.actionbar_dark_back_icon);
        toolbar.setNavigationOnClickListener(v -> finish());

        if (USE_MODEL != YOLOV5S && USE_MODEL != DBFACE && USE_MODEL != NANODET && USE_MODEL != YOLOV5_CUSTOM_LAYER) {
            nmsSeekBar.setEnabled(false);
            thresholdSeekBar.setEnabled(false);
            tvNMS.setVisibility(View.GONE);
            tvThreshold.setVisibility(View.GONE);
            nmsSeekBar.setVisibility(View.GONE);
            thresholdSeekBar.setVisibility(View.GONE);
            tvNMNThreshold.setVisibility(View.GONE);
        } else if (USE_MODEL == YOLOV5S) {
            threshold = 0.3f;
            nms_threshold = 0.7f;
        } else if (USE_MODEL == DBFACE || USE_MODEL == NANODET) {
            threshold = 0.4f;
            nms_threshold = 0.6f;
        } else if (USE_MODEL == YOLOV5_CUSTOM_LAYER) {
            threshold = 0.25f;
            nms_threshold = 0.45f;
        }
        nmsSeekBar.setProgress((int) (nms_threshold * 100));
        thresholdSeekBar.setProgress((int) (threshold * 100));
        final String format = "THR: %.2f, NMS: %.2f";
        tvNMNThreshold.setText(String.format(Locale.ENGLISH, format, threshold, nms_threshold));
        nmsSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                nms_threshold = i / 100.f;
                tvNMNThreshold.setText(String.format(Locale.ENGLISH, format, threshold, nms_threshold));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
        thresholdSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                threshold = i / 100.f;
                tvNMNThreshold.setText(String.format(Locale.ENGLISH, format, threshold, nms_threshold));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
        btnPhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int permission = ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE);
                if (permission != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(
                            MainActivity.this,
                            new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                            777
                    );
                } else {
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("image/*");
                    startActivityForResult(intent, REQUEST_PICK_IMAGE);
                }
            }
        });

        btnVideo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int permission = ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE);
                if (permission != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(
                            MainActivity.this,
                            new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                            777
                    );
                } else {
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("video/*");
                    startActivityForResult(intent, REQUEST_PICK_VIDEO);
                }
            }
        });

        resultImageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (detectVideo.get() || detectPhoto.get()) {
                    detectPhoto.set(false);
                    detectVideo.set(false);
                    sbVideo.setVisibility(View.GONE);
                    sbVideoSpeed.setVisibility(View.GONE);
                    startCamera();
                }
            }
        });

        viewFinder.addOnLayoutChangeListener(new View.OnLayoutChangeListener() {
            @Override
            public void onLayoutChange(View view, int i, int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
                updateTransform();
            }
        });

        viewFinder.post(new Runnable() {
            @Override
            public void run() {
                startCamera();
            }
        });

        sbVideoSpeed.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                videoSpeed = i;
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                Toast.makeText(MainActivity.this, "Video Speed:" + seekBar.getProgress(), Toast.LENGTH_SHORT).show();
            }
        });

        sbVideo.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                videoCurFrameLoc = i;
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                videoCurFrameLoc = seekBar.getProgress();
            }
        });
    }

    protected void initViewID() {
        toolbar = findViewById(R.id.tool_bar);
        resultImageView = findViewById(R.id.imageView);
        tvNMNThreshold = findViewById(R.id.valTxtView);
        tvInfo = findViewById(R.id.tv_info);
        tvNMS = findViewById(R.id.txtNMS);
        tvThreshold = findViewById(R.id.txtThresh);
        nmsSeekBar = findViewById(R.id.nms_seek);
        thresholdSeekBar = findViewById(R.id.threshold_seek);
        btnPhoto = findViewById(R.id.button);
        btnVideo = findViewById(R.id.btn_video);
        viewFinder = findViewById(R.id.view_finder);
        sbVideo = findViewById(R.id.sb_video);
        sbVideo.setVisibility(View.GONE);
        sbVideoSpeed = findViewById(R.id.sb_video_speed);
        sbVideoSpeed.setMin(VIDEO_SPEED_MIN);
        sbVideoSpeed.setMax(VIDEO_SPEED_MAX);
        sbVideoSpeed.setVisibility(View.GONE);
    }

    protected void initModel() {
        if (USE_MODEL == YOLOV5S) {
            YOLOv5.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == YOLOV4_TINY) {
            YOLOv4.init(getAssets(), 0, USE_GPU);
        } else if (USE_MODEL == MOBILENETV2_YOLOV3_NANO) {
            YOLOv4.init(getAssets(), 1, USE_GPU);
        } else if (USE_MODEL == YOLO_FASTEST_XL) {
            YOLOv4.init(getAssets(), 2, USE_GPU);
        } else if (USE_MODEL == SIMPLE_POSE) {
            SimplePose.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == YOLACT) {
            Yolact.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == ENET) {
            ENet.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == FACE_LANDMARK) {
            FaceLandmark.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == DBFACE) {
            DBFace.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == MOBILENETV2_FCN) {
            MbnFCN.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == MOBILENETV3_SEG) {
            MbnSeg.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == YOLOV5_CUSTOM_LAYER) {
            YOLOv5.initCustomLayer(getAssets(), USE_GPU);
        } else if (USE_MODEL == NANODET) {
            NanoDet.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == LIGHT_OPEN_POSE) {
            LightOpenPose.init(getAssets(), USE_GPU);
        }
    }

    private void updateTransform() {
        Matrix matrix = new Matrix();
        // Compute the center of the view finder
        float centerX = viewFinder.getWidth() / 2f;
        float centerY = viewFinder.getHeight() / 2f;

        float[] rotations = {0, 90, 180, 270};
        // Correct preview output to account for display rotation
        float rotationDegrees = rotations[viewFinder.getDisplay().getRotation()];

        matrix.postRotate(-rotationDegrees, centerX, centerY);

        // Finally, apply transformations to our TextureView
        viewFinder.setTransform(matrix);
    }

    private void startCamera() {
        CameraX.unbindAll();
        // 1. preview
        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(CAMERA_ID)
//                .setTargetAspectRatio(Rational.NEGATIVE_INFINITY)  // 宽高比
                .setTargetResolution(new Size(480, 640))  // 分辨率
                .build();

        Preview preview = new Preview(previewConfig);
        preview.setOnPreviewOutputUpdateListener(new Preview.OnPreviewOutputUpdateListener() {
            @Override
            public void onUpdated(Preview.PreviewOutput output) {
                ViewGroup parent = (ViewGroup) viewFinder.getParent();
                parent.removeView(viewFinder);
                parent.addView(viewFinder, 0);

                viewFinder.setSurfaceTexture(output.getSurfaceTexture());
                updateTransform();
            }
        });
        DetectAnalyzer detectAnalyzer = new DetectAnalyzer();
        CameraX.bindToLifecycle((LifecycleOwner) this, preview, gainAnalyzer(detectAnalyzer));

    }


    private UseCase gainAnalyzer(DetectAnalyzer detectAnalyzer) {
        ImageAnalysisConfig.Builder analysisConfigBuilder = new ImageAnalysisConfig.Builder();
        analysisConfigBuilder.setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE);
        analysisConfigBuilder.setTargetResolution(new Size(480, 640));  // 输出预览图像尺寸
        ImageAnalysisConfig config = analysisConfigBuilder.build();
        ImageAnalysis analysis = new ImageAnalysis(config);
        analysis.setAnalyzer(detectAnalyzer);
        return analysis;
    }

    private Bitmap imageToBitmap(ImageProxy image) {
        byte[] nv21 = imagetToNV21(image);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();
        try {
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private byte[] imagetToNV21(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ImageProxy.PlaneProxy y = planes[0];
        ImageProxy.PlaneProxy u = planes[1];
        ImageProxy.PlaneProxy v = planes[2];
        ByteBuffer yBuffer = y.getBuffer();
        ByteBuffer uBuffer = u.getBuffer();
        ByteBuffer vBuffer = v.getBuffer();
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        byte[] nv21 = new byte[ySize + uSize + vSize];
        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        return nv21;
    }

    private class DetectAnalyzer implements ImageAnalysis.Analyzer {

        @Override
        public void analyze(ImageProxy image, final int rotationDegrees) {
            detectOnModel(image, rotationDegrees);
        }
    }


    private void detectOnModel(ImageProxy image, final int rotationDegrees) {
        if (detectCamera.get() || detectPhoto.get() || detectVideo.get()) {
            return;
        }
        detectCamera.set(true);
        startTime = System.currentTimeMillis();
        final Bitmap bitmapsrc = imageToBitmap(image);  // 格式转换
        if (detectService == null) {
            detectCamera.set(false);
            return;
        }
        detectService.execute(new Runnable() {
            @Override
            public void run() {
                Matrix matrix = new Matrix();
                matrix.postRotate(rotationDegrees);
                width = bitmapsrc.getWidth();
                height = bitmapsrc.getHeight();
                Bitmap bitmap = Bitmap.createBitmap(bitmapsrc, 0, 0, width, height, matrix, false);

                detectAndDraw(bitmap);
                showResultOnUI();
            }
        });
    }

    protected void showResultOnUI() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                detectCamera.set(false);
                resultImageView.setImageBitmap(mutableBitmap);
                endTime = System.currentTimeMillis();
                long dur = endTime - startTime;
                float fps = (float) (1000.0 / dur);
                total_fps = (total_fps == 0) ? fps : (total_fps + fps);
                fps_count++;
                String modelName = getModelName();

                tvInfo.setText(String.format(Locale.CHINESE,
                        "%s\nSize: %dx%d\nTime: %.3f s\nFPS: %.3f\nAVG_FPS: %.3f",
                        modelName, height, width, dur / 1000.0, fps, (float) total_fps / fps_count));
            }
        });
    }

    protected Bitmap drawDBFaceLandmark(Bitmap mutableBitmap, KeyPoint[] keyPoints) {
        if (keyPoints == null || keyPoints.length <= 0) {
            return mutableBitmap;
        }
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint keyPointPaint = new Paint();
        keyPointPaint.setAlpha(200);
        keyPointPaint.setStyle(Paint.Style.STROKE);
        keyPointPaint.setColor(Color.BLUE);
//        Log.d("wzt", "dbface size:" + keyPoints.length);
        for (int i = 0; i < keyPoints.length; i++) {
            // 其它随机颜色
            Random random = new Random(i + 2020);
            int color = Color.argb(255, random.nextInt(256), 125, random.nextInt(256));
            keyPointPaint.setColor(color);
            keyPointPaint.setStrokeWidth(9 * mutableBitmap.getWidth() / 800.0f);
            for (int j = 0; j < 5; j++) {
                canvas.drawPoint(keyPoints[i].x[j], keyPoints[i].y[j], keyPointPaint);
            }
            keyPointPaint.setStrokeWidth(4 * mutableBitmap.getWidth() / 800.0f);
            canvas.drawRect(keyPoints[i].x0, keyPoints[i].y0, keyPoints[i].x1, keyPoints[i].y1, keyPointPaint);
        }
        return mutableBitmap;
    }

    protected Bitmap drawENetMask(Bitmap mutableBitmap, float[] results) {
        if (results == null || results.length <= 0) {
            return mutableBitmap;
        }
        // 0, "road" 1, "sidewalk" 2, "building" 3, "wall" 4, "fence" 5, "pole" 6, "traffic light" 7, "traffic sign" 8, "vegetation"
        // 9, "terrain" 10, "sky" 11, "person" 12, "rider" 13, "car" 14, "truck" 15, "bus" 16, "train" 17, "motorcycle" 18, "bicycle"
        int[][] cityspace_colormap = {
                {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156}, {190, 153, 153}, {153, 153, 153},
                {250, 170, 30}, {220, 220, 0}, {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
                {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32}
        };
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint maskPaint = new Paint();
        maskPaint.setStyle(Paint.Style.STROKE);
        maskPaint.setStrokeWidth(4 * mutableBitmap.getWidth() / 800.0f);
        maskPaint.setTextSize(30 * mutableBitmap.getWidth() / 800.0f);
        float mask = 0;
        int color = 0;
        float tempC = 0;
        int lengthW = 1;
        for (int y = 0; y < mutableBitmap.getHeight(); y++) {
            for (int x = 0; x < mutableBitmap.getWidth(); x++) {
                mask = results[y * mutableBitmap.getWidth() + x];
                if (mask >= cityspace_colormap.length) {
                    continue;
                }
//                color = Color.argb(255,
//                        cityspace_colormap[(int) mask][0],
//                        cityspace_colormap[(int) mask][1],
//                        cityspace_colormap[(int) mask][2]);
//                maskPaint.setColor(color);
//                maskPaint.setAlpha(100);
//                canvas.drawPoint(x, y, maskPaint);
                // fast
                if (mask != tempC) {
                    color = Color.argb(255,
                            cityspace_colormap[(int) tempC][0],
                            cityspace_colormap[(int) tempC][1],
                            cityspace_colormap[(int) tempC][2]);
                    maskPaint.setColor(color);
                    maskPaint.setAlpha(100);
                    canvas.drawLine(x - lengthW, y, x, y, maskPaint);
                    tempC = mask;
                    lengthW = 1;
                } else {
                    lengthW++;
                }
            }
            // fast
            color = Color.argb(255,
                    cityspace_colormap[(int) tempC][0],
                    cityspace_colormap[(int) tempC][1],
                    cityspace_colormap[(int) tempC][2]);
            maskPaint.setColor(color);
            maskPaint.setAlpha(100);
            canvas.drawLine(mutableBitmap.getWidth() - lengthW, y, mutableBitmap.getWidth(), y, maskPaint);
            tempC = mask;
            lengthW = 1;
        }
        return mutableBitmap;
    }

    protected Bitmap drawYolactMask(Bitmap mutableBitmap, YolactMask[] results) {
        if (results == null || results.length <= 0) {
            return mutableBitmap;
        }
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint maskPaint = new Paint();
        maskPaint.setAlpha(200);
        maskPaint.setStyle(Paint.Style.STROKE);
        maskPaint.setStrokeWidth(4 * mutableBitmap.getWidth() / 800.0f);
        maskPaint.setTextSize(30 * mutableBitmap.getWidth() / 800.0f);
        maskPaint.setColor(Color.BLUE);
        for (YolactMask mask : results) {
            if (mask.prob < 0.4f) {
                continue;
            }
            int index = 0;
            char tempC = 0;
            int lengthW = 1;
            for (int y = 0; y < mutableBitmap.getHeight(); y++) {
                for (int x = 0; x < mutableBitmap.getWidth(); x++) {
//                    if (mask.mask[index] != 0) {
//                        maskPaint.setColor(mask.getColor());
//                        maskPaint.setAlpha(100);
//                        canvas.drawPoint(x, y, maskPaint);
//                    }
//                    index++;
                    // fast
                    if (mask.mask[index] != 0) {
                        if (mask.mask[index] != tempC) {
                            maskPaint.setColor(mask.getColor());
                            maskPaint.setAlpha(100);
                            canvas.drawLine(x - lengthW, y, x, y, maskPaint);
                            tempC = mask.mask[index];
                            lengthW = 1;
                        } else {
                            lengthW++;
                        }
                    } else if (lengthW > 1) {
                        maskPaint.setColor(mask.getColor());
                        maskPaint.setAlpha(100);
                        canvas.drawLine(x - lengthW, y, x, y, maskPaint);
                        tempC = mask.mask[index];
                        lengthW = 1;
                    }
                    index++;
                }
                // fast
                if (lengthW > 1) {
                    maskPaint.setColor(mask.getColor());
                    maskPaint.setAlpha(100);
                    canvas.drawLine(mutableBitmap.getWidth() - lengthW, y, mutableBitmap.getWidth(), y, maskPaint);
                    tempC = mask.mask[index - 1];
                    lengthW = 1;
                }
            }
            // 标签跟框放后面画，防止被 mask 挡住
            maskPaint.setColor(mask.getColor());
            maskPaint.setStyle(Paint.Style.FILL);
            canvas.drawText(mask.getLabel() + String.format(Locale.CHINESE, " %.3f", mask.getProb()), mask.left, mask.top - 15 * mutableBitmap.getWidth() / 1000.0f, maskPaint);
            maskPaint.setStyle(Paint.Style.STROKE);
            canvas.drawRect(new RectF(mask.left, mask.top, mask.right, mask.bottom), maskPaint);
        }
        return mutableBitmap;
    }

    protected Bitmap drawBoxRects(Bitmap mutableBitmap, Box[] results) {
        if (results == null || results.length <= 0) {
            return mutableBitmap;
        }
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint boxPaint = new Paint();
        boxPaint.setAlpha(200);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4 * mutableBitmap.getWidth() / 800.0f);
        boxPaint.setTextSize(30 * mutableBitmap.getWidth() / 800.0f);
        for (Box box : results) {
            if (USE_MODEL == MOBILENETV2_YOLOV3_NANO) {
                if (box.getScore() < 0.15f) {
                    // 模型比较小，置信度太低就不要了
                    continue;
                }
                // 有时候差太多了，手动改一下
                box.x0 = box.x0 < 0 ? box.x0 / 9 : box.x0;
                box.y0 = box.y0 < 0 ? box.y0 / 9 : box.y0;
            }
            boxPaint.setColor(box.getColor());
            boxPaint.setStyle(Paint.Style.FILL);
            canvas.drawText(box.getLabel() + String.format(Locale.CHINESE, " %.3f", box.getScore()), box.x0 + 3, box.y0 + 30 * mutableBitmap.getWidth() / 1000.0f, boxPaint);
            boxPaint.setStyle(Paint.Style.STROKE);
            canvas.drawRect(box.getRect(), boxPaint);
        }
        return mutableBitmap;
    }

    protected Bitmap drawFaceLandmark(Bitmap mutableBitmap, FaceKeyPoint[] keyPoints) {
        if (keyPoints == null || keyPoints.length <= 0) {
            return mutableBitmap;
        }
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint keyPointPaint = new Paint();
        keyPointPaint.setAlpha(200);
        keyPointPaint.setStyle(Paint.Style.STROKE);
        keyPointPaint.setStrokeWidth(8 * mutableBitmap.getWidth() / 800.0f);
        keyPointPaint.setColor(Color.BLUE);
//        Log.d("wzt", "facePoint length:" + keyPoints.length);
        for (int i = 0; i < keyPoints.length; i++) {
            // 其它随机颜色
            Random random = new Random(i / 106 + 2020);
            int color = Color.argb(255, random.nextInt(256), 125, random.nextInt(256));
            keyPointPaint.setColor(color);
            canvas.drawPoint(keyPoints[i].x, keyPoints[i].y, keyPointPaint);
        }
        return mutableBitmap;
    }

    protected Bitmap drawPersonPose(Bitmap mutableBitmap, KeyPoint[] keyPoints) {
        if (keyPoints == null || keyPoints.length <= 0) {
            return mutableBitmap;
        }
        // draw bone
        // 0 nose, 1 left_eye, 2 right_eye, 3 left_Ear, 4 right_Ear, 5 left_Shoulder, 6 rigth_Shoulder, 7 left_Elbow, 8 right_Elbow,
        // 9 left_Wrist, 10 right_Wrist, 11 left_Hip, 12 right_Hip, 13 left_Knee, 14 right_Knee, 15 left_Ankle, 16 right_Ankle
        int[][] joint_pairs = {{0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint keyPointPaint = new Paint();
        keyPointPaint.setAlpha(200);
        keyPointPaint.setStyle(Paint.Style.STROKE);
        keyPointPaint.setColor(Color.BLUE);
        int color = Color.BLUE;
        // 画线、画框、画点
        for (int i = 0; i < keyPoints.length; i++) {
            // 其它随机颜色
            Random random = new Random(i + 2020);
            color = Color.argb(255, random.nextInt(256), random.nextInt(256), random.nextInt(256));
            // 画线
            keyPointPaint.setStrokeWidth(5 * mutableBitmap.getWidth() / 800.0f);
            for (int j = 0; j < 16; j++) {  // 17个点连成16条线
                int pl0 = joint_pairs[j][0];
                int pl1 = joint_pairs[j][1];
                // 人体左侧改为红线
                if ((pl0 % 2 == 1) && (pl1 % 2 == 1) && pl0 >= 5 && pl1 >= 5) {
                    keyPointPaint.setColor(Color.RED);
                } else {
                    keyPointPaint.setColor(color);
                }
                canvas.drawLine(keyPoints[i].x[joint_pairs[j][0]], keyPoints[i].y[joint_pairs[j][0]],
                        keyPoints[i].x[joint_pairs[j][1]], keyPoints[i].y[joint_pairs[j][1]],
                        keyPointPaint);
            }
            // 画点
            keyPointPaint.setColor(Color.GREEN);
            keyPointPaint.setStrokeWidth(8 * mutableBitmap.getWidth() / 800.0f);
            for (int n = 0; n < 17; n++) {
                canvas.drawPoint(keyPoints[i].x[n], keyPoints[i].y[n], keyPointPaint);
            }
            // 画框
            keyPointPaint.setColor(color);
            keyPointPaint.setStrokeWidth(3 * mutableBitmap.getWidth() / 800.0f);
            canvas.drawRect(keyPoints[i].x0, keyPoints[i].y0, keyPoints[i].x1, keyPoints[i].y1, keyPointPaint);
        }
        return mutableBitmap;
    }

    protected Bitmap drawOpenPose(Bitmap mutableBitmap, OpenPoseKeyPoint[] humanPoses) {
        if (humanPoses == null || humanPoses.length <= 0) {
            return mutableBitmap;
        }
        int[][] joint_pairs = {
                {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6},
                {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11},
                {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16},
                {0, 15}, {15, 17}, {2, 16}, {5, 17}
        };
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint keyPointPaint = new Paint();
        keyPointPaint.setAlpha(200);
        keyPointPaint.setStyle(Paint.Style.STROKE);
        keyPointPaint.setColor(Color.BLUE);
        int color = Color.BLUE;
        int keyPointNumber = 18;
        int lineNumber = 17;
        // 画线、画点
        for (int i = 0; i < humanPoses.length; i++) {
            Random random = new Random(i + 2021);
            color = Color.argb(255, random.nextInt(256), random.nextInt(256), random.nextInt(256));
            // line
            keyPointPaint.setColor(color);
            keyPointPaint.setStrokeWidth(5 * mutableBitmap.getWidth() / 800.0f);
            OpenPoseKeyPoint human = humanPoses[i];
            for (int j = 0; j < lineNumber; j++) {
                if (human.x[joint_pairs[j][0]] == -1 ||human.y[joint_pairs[j][1]] == -1) {
                    continue;
                }
                canvas.drawLine(human.x[joint_pairs[j][0]], human.y[joint_pairs[j][0]],
                        human.x[joint_pairs[j][1]], human.y[joint_pairs[j][1]],
                        keyPointPaint);
            }

            // point
            keyPointPaint.setColor(Color.GREEN);
            keyPointPaint.setStrokeWidth(8 * mutableBitmap.getWidth() / 800.0f);
            for (int j = 0; j < keyPointNumber; j++) {
                if (humanPoses[i].x[j] == -1 || humanPoses[i].y[j] == -1) {
                    continue;
                }
                canvas.drawPoint(humanPoses[i].x[j], humanPoses[i].y[j], keyPointPaint);
            }
        }
        return mutableBitmap;
    }

    protected Bitmap detectAndDraw(Bitmap image) {
        Box[] result = null;
        KeyPoint[] keyPoints = null;
        YolactMask[] yolactMasks = null;
        FaceKeyPoint[] faceKeyPoints = null;
        float[] enetMasks = null;
        OpenPoseKeyPoint[] openPoseKeyPoints = null;
        if (USE_MODEL == YOLOV5S) {
            result = YOLOv5.detect(image, threshold, nms_threshold);
        } else if (USE_MODEL == YOLOV4_TINY || USE_MODEL == MOBILENETV2_YOLOV3_NANO || USE_MODEL == YOLO_FASTEST_XL) {
            result = YOLOv4.detect(image, threshold, nms_threshold);
        } else if (USE_MODEL == SIMPLE_POSE) {
            keyPoints = SimplePose.detect(image);
        } else if (USE_MODEL == YOLACT) {
            yolactMasks = Yolact.detect(image);
        } else if (USE_MODEL == ENET) {
            enetMasks = ENet.detect(image);
        } else if (USE_MODEL == FACE_LANDMARK) {
            faceKeyPoints = FaceLandmark.detect(image);
        } else if (USE_MODEL == DBFACE) {
            keyPoints = DBFace.detect(image, threshold, nms_threshold);
        } else if (USE_MODEL == MOBILENETV2_FCN) {
            enetMasks = MbnFCN.detect(image);
        } else if (USE_MODEL == MOBILENETV3_SEG) {
            enetMasks = MbnSeg.detect(image);
        } else if (USE_MODEL == YOLOV5_CUSTOM_LAYER) {
            result = YOLOv5.detectCustomLayer(image, threshold, nms_threshold);
        } else if (USE_MODEL == NANODET) {
            result = NanoDet.detect(image, threshold, nms_threshold);
        } else if (USE_MODEL == LIGHT_OPEN_POSE) {
            openPoseKeyPoints = LightOpenPose.detect(image);
        }
        if (result == null && keyPoints == null && yolactMasks == null && enetMasks == null && faceKeyPoints == null
                && openPoseKeyPoints == null) {
            detectCamera.set(false);
            return image;
        }
        if (USE_MODEL == YOLOV5S || USE_MODEL == YOLOV4_TINY || USE_MODEL == MOBILENETV2_YOLOV3_NANO
                || USE_MODEL == YOLOV5_CUSTOM_LAYER || USE_MODEL == NANODET || USE_MODEL == YOLO_FASTEST_XL) {
            mutableBitmap = drawBoxRects(image, result);
        } else if (USE_MODEL == SIMPLE_POSE) {
            mutableBitmap = drawPersonPose(image, keyPoints);
        } else if (USE_MODEL == YOLACT) {
            mutableBitmap = drawYolactMask(image, yolactMasks);
        } else if (USE_MODEL == ENET) {
            mutableBitmap = drawENetMask(image, enetMasks);
        } else if (USE_MODEL == FACE_LANDMARK) {
            mutableBitmap = drawFaceLandmark(image, faceKeyPoints);
        } else if (USE_MODEL == DBFACE) {
            mutableBitmap = drawDBFaceLandmark(image, keyPoints);
        } else if (USE_MODEL == MOBILENETV2_FCN) {
            mutableBitmap = drawENetMask(image, enetMasks);  // 与 enet 相同
        } else if (USE_MODEL == MOBILENETV3_SEG) {
            mutableBitmap = drawENetMask(image, enetMasks);  // 与 enet 相同
        } else if (USE_MODEL == LIGHT_OPEN_POSE) {
            mutableBitmap = drawOpenPose(image, openPoseKeyPoints);
        }
        return mutableBitmap;
    }

    protected String getModelName() {
        String modelName = "ohhhhh";
        if (USE_MODEL == YOLOV5S) {
            modelName = "YOLOv5s";
        } else if (USE_MODEL == YOLOV4_TINY) {
            modelName = "YOLOv4-tiny";
        } else if (USE_MODEL == MOBILENETV2_YOLOV3_NANO) {
            modelName = "MobileNetV2-YOLOv3-Nano";
        } else if (USE_MODEL == SIMPLE_POSE) {
            modelName = "Simple-Pose";
        } else if (USE_MODEL == YOLACT) {
            modelName = "Yolact";
        } else if (USE_MODEL == ENET) {
            modelName = "ENet";
        } else if (USE_MODEL == FACE_LANDMARK) {
            modelName = "YoloFace500k-landmark106";
        } else if (USE_MODEL == DBFACE) {
            modelName = "DBFace";
        } else if (USE_MODEL == MOBILENETV2_FCN) {
            modelName = "MobileNetV2-FCN";
        } else if (USE_MODEL == MOBILENETV3_SEG) {
            modelName = "MBNV3-Segmentation-small";
        } else if (USE_MODEL == YOLOV5_CUSTOM_LAYER) {
            modelName = "YOLOv5s_Custom_Layer";
        } else if (USE_MODEL == NANODET) {
            modelName = "NanoDet";
        } else if (USE_MODEL == YOLO_FASTEST_XL) {
            modelName = "YOLO-Fastest-xl";
        } else if (USE_MODEL == LIGHT_OPEN_POSE) {
            modelName = "Light OpenPose";
        }
        return USE_GPU ? "[ GPU ] " + modelName : "[ CPU ] " + modelName;
    }

    @Override
    protected void onDestroy() {
        detectCamera.set(false);
        detectVideo.set(false);
        if (detectService != null) {
            detectService.shutdown();
            detectService = null;
        }
        if (mmr != null) {
            mmr.release();
        }
        CameraX.unbindAll();
        super.onDestroy();
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera Permission!", Toast.LENGTH_SHORT).show();
                this.finish();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (data == null) {
            return;
        }
        if (requestCode == REQUEST_PICK_IMAGE) {
            // photo
            runByPhoto(requestCode, resultCode, data);
        } else if (requestCode == REQUEST_PICK_VIDEO) {
            // video
            runByVideo(requestCode, resultCode, data);
        } else {
            Toast.makeText(this, "Error", Toast.LENGTH_SHORT).show();
        }
    }

    public void runByPhoto(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode != RESULT_OK || data == null) {
            Toast.makeText(this, "Photo error", Toast.LENGTH_SHORT).show();
            return;
        }
        if (detectVideo.get()) {
            Toast.makeText(this, "Video is running", Toast.LENGTH_SHORT).show();
            return;
        }
        detectPhoto.set(true);
        final Bitmap image = getPicture(data.getData());
        if (image == null) {
            Toast.makeText(this, "Photo is null", Toast.LENGTH_SHORT).show();
            return;
        }
        CameraX.unbindAll();
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                long start = System.currentTimeMillis();
                mutableBitmap = image.copy(Bitmap.Config.ARGB_8888, true);
                width = image.getWidth();
                height = image.getHeight();

                mutableBitmap = detectAndDraw(mutableBitmap);

                final long dur = System.currentTimeMillis() - start;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        String modelName = getModelName();
                        resultImageView.setImageBitmap(mutableBitmap);
                        tvInfo.setText(String.format(Locale.CHINESE, "%s\nSize: %dx%d\nTime: %.3f s\nFPS: %.3f",
                                modelName, height, width, dur / 1000.0, 1000.0f / dur));
                    }
                });
            }
        }, "photo detect");
        thread.start();
    }

    public void runByVideo(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode != RESULT_OK || data == null) {
            Toast.makeText(this, "Video error", Toast.LENGTH_SHORT).show();
            return;
        }
        try {
            Uri uri = data.getData();
            Cursor cursor = getContentResolver().query(uri, null, null, null, null);
            if (cursor != null) {
                cursor.moveToFirst();
                // String imgNo = cursor.getString(0); // 编号
                String v_path = cursor.getString(1); // 文件路径
                String v_size = cursor.getString(2); // 大小
                String v_name = cursor.getString(3); // 文件名
                detectOnVideo(v_path);
            } else {
                Toast.makeText(this, "Video is null", Toast.LENGTH_SHORT).show();
            }
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Video is null", Toast.LENGTH_SHORT).show();
        }
    }

    public void detectOnVideo(final String path) {
        if (detectVideo.get()) {
            Toast.makeText(this, "Video is running", Toast.LENGTH_SHORT).show();
            return;
        }
        detectVideo.set(true);
        Toast.makeText(MainActivity.this, "FPS is not accurate!", Toast.LENGTH_SHORT).show();
        sbVideo.setVisibility(View.VISIBLE);
        sbVideoSpeed.setVisibility(View.VISIBLE);
        CameraX.unbindAll();
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                mmr = new FFmpegMediaMetadataRetriever();
                mmr.setDataSource(path);
                String dur = mmr.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_DURATION);  // ms
                String sfps = mmr.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_FRAMERATE);  // fps
//                String sWidth = mmr.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH);  // w
//                String sHeight = mmr.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT);  // h
                String rota = mmr.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);  // rotation
                int duration = Integer.parseInt(dur);
                float fps = Float.parseFloat(sfps);
                float rotate = 0;
                if (rota != null) {
                    rotate = Float.parseFloat(rota);
                }
                sbVideo.setMax(duration * 1000);
                float frameDis = 1.0f / fps * 1000 * 1000 * videoSpeed;
                videoCurFrameLoc = 0;
                while (detectVideo.get() && (videoCurFrameLoc) < (duration * 1000)) {
                    videoCurFrameLoc = (long) (videoCurFrameLoc + frameDis);
                    sbVideo.setProgress((int) videoCurFrameLoc);
                    final Bitmap b = mmr.getFrameAtTime(videoCurFrameLoc, FFmpegMediaMetadataRetriever.OPTION_CLOSEST);
                    if (b == null) {
                        continue;
                    }
                    Matrix matrix = new Matrix();
                    matrix.postRotate(rotate);
                    width = b.getWidth();
                    height = b.getHeight();
                    final Bitmap bitmap = Bitmap.createBitmap(b, 0, 0, width, height, matrix, false);
                    startTime = System.currentTimeMillis();
                    detectAndDraw(bitmap.copy(Bitmap.Config.ARGB_8888, true));
                    showResultOnUI();
                    frameDis = 1.0f / fps * 1000 * 1000 * videoSpeed;
                }
                mmr.release();
                if (detectVideo.get()) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            sbVideo.setVisibility(View.GONE);
                            sbVideoSpeed.setVisibility(View.GONE);
                            Toast.makeText(MainActivity.this, "Video end!", Toast.LENGTH_LONG).show();
                        }
                    });
                }
                detectVideo.set(false);
            }
        }, "video detect");
        thread.start();
//        startCamera();
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

    /**
     * letterbox (slow)
     *
     * @param srcBitmap
     * @param srcWidth
     * @param srcHeight
     * @param dstWidth
     * @param dstHeight
     * @param matrix
     * @return
     */
    public static Bitmap letterbox(Bitmap srcBitmap, int srcWidth, int srcHeight, int dstWidth, int dstHeight, Matrix matrix) {
        long timeStart = System.currentTimeMillis();
        float scale = Math.min((float) dstWidth / srcWidth, (float) dstHeight / srcHeight);
        int nw = (int) (srcWidth * scale);
        int nh = (int) (srcHeight * scale);
        matrix.postScale((float) nw / srcWidth, (float) nh / srcHeight);
        Bitmap bitmap = Bitmap.createBitmap(srcBitmap, 0, 0, srcWidth, srcHeight, matrix, false);
        Bitmap newBitmap = Bitmap.createBitmap(dstWidth, dstHeight, Bitmap.Config.ARGB_8888);//创建和目标相同大小的空Bitmap
        Canvas canvas = new Canvas(newBitmap);
        Paint paint = new Paint();
        // 针对绘制bitmap添加抗锯齿
        PaintFlagsDrawFilter pfd = new PaintFlagsDrawFilter(0, Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        paint.setFilterBitmap(false);  // 对Bitmap进行滤波处理
        paint.setAntiAlias(true);  // 设置抗锯齿
        canvas.setDrawFilter(pfd);
        canvas.drawBitmap(bitmap, null,
                new Rect((dstHeight - nh) / 2, (dstWidth - nw) / 2,
                        (dstHeight - nh) / 2 + nh, (dstWidth - nw) / 2 + nw),
                paint);
        long timeDur = System.currentTimeMillis() - timeStart;
//        Log.d(TAG, "letterbox time:" + timeDur);
        return newBitmap;
    }

}
