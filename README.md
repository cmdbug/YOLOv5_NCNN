[中文说明](https://github.com/cmdbug/YOLOv5_NCNN/blob/master/README_CN.md)

## :rocket: If it helps you, click a star! :star: ##

### Ncnn deployment on mobile,support:YOLOv5s,YOLOv4-tiny,MobileNetV2-YOLOv3-nano,Simple-Pose,Yolact,ChineseOCR-lite,ENet,Landmark106,DBFace,MBNv2-FCN and MBNv3-Seg-small on camera.

## iOS:
- Xcode 11.5
- macOS 10.15.4
- iPhone 6sp 13.5.1

## Android:
- Android Studio 4.0
- Win10 1909
- Meizu 16x 8.1.0 (CPU:Qualcomm 710 GPU:Adreno 616)

Android has added permission requests, but if it still crashes, please manually confirm whether the relevant permissions are allowed.

> iOS
```code
YOLOv5s:     Select the model to be tested directly on the interface.
YOLOv4-tiny: Select the model to be tested directly on the interface.
YOLOv3-nano: Select the model to be tested directly on the interface.
```
> Android
```
Select the model to be tested directly on the interface.
```
### Models
* ***YOLOv5s***: The input size is reduced, the decoding process uses a large number of for loops and NMS appears to be slower.
* ***YOLOv4-tiny***: Using the default size, the decoding process does not have a lot of for and NMS, so the speed will be faster.
* ***YOLOv3-nano***: Same as v4-tiny.
* ***Simple-Pose***: Only the Android version is written for the time being, and iOS has not been added yet. The internal principle is to first detect the person and then use the area of the person to perform posture detection again, that is, a 2-step process.
* ***Yolact***: Only the Android version is written for the time being, and iOS has not been added yet.
* ***ChineseOCR_lite***: Only the Android version is written for the time being, and iOS has not been added yet. (It should be noted that crashes occasionally occur, please fix it if you have time)
* ***ENet***: Only the Android version is written for the time being, and iOS has not been added yet. (Because the model is too small, the segmentation effect is relatively poor, you can replace the stronger network by yourself. But the effect is too poor, there may be problems)
* ***Landmark106***: Only the Android version is written for the time being, and iOS has not been added yet. The internal principle is to first detect the face and then use the area of the face to perform key point detection again, that is, a 2-step process.
* ***DBFace***: Only the Android version is written for the time being, and iOS has not been added yet.
* ***MBNv2-FCN***: Only the Android version is written for the time being, and iOS has not been added yet.(Thanks to the nick name Persistence for help)
* ***MBNv3-Seg-small***: Only the Android version is written for the time being, and iOS has not been added yet.
* ***YOLOv5s_custom_op***: Only the Android version is written for the time being, and iOS has not been added yet.
* ***nanodet***: Only the Android version is written for the time being, and iOS has not been added yet.

### Note：<br/>
* Due to factors such as mobile phone performance and image size, FPS varies greatly on different mobile phones. This project mainly tests the use of the NCNN framework. For the conversion of specific models, you can go to the NCNN official to view the conversion tutorial.
* Because the opencv library is too large, only arm64-v8a/armeabi-v7a is reserved. If you need other versions, go to the official download.
* ncnn temporarily uses the vulkan version, and acceleration needs to be turned on before loading, which is not turned on in this project. If you want to use the ncnn version, you need to modify the CMakeLists.txt configuration.
* Different AS versions may have various problems with compilation. If the compilation error cannot be solved, it is recommended to use AS4.0 or higher to try.


:art: Screenshot<br/>

> Android

| mbnv2-yolov3-nano | yolov4-tiny | yolov5s |
|-------------------|-------------|---------|
|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_mobilenetv2_yolov3_nano.jpg"/>|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_yolov4_tiny.jpg"/>|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_yolov5s.jpg"/>|

| simple_pose | yolact | chineseocr_lite_01 |
|-------------------|-------------|---------|
|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_simple_pose.jpg"/>|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_yolact.jpg"/>|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_meizu16x_chineseocr_lite_01.jpg"/>|

| chineseocr_lite_02 | ENet | yoloface500k-landmark106 |
|-------------------|-------------|---------|
|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_meizu16x_chineseocr_lite_02.jpg"/>|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_ENet.jpg"/>|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_yoloface500k-landmark106.jpg"/>|

|  dbface | mbnv2_fcn | mbnv3_seg_small |
|-------------------|-------------|---------|
| <img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_dbface.jpg"/>|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_MBNFCN.jpg"/>|<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/Android_Meizu16x_MBNV3_Seg_small.jpg"/>|

|  yolov5s_custom_op | nanodet | xxx |
|-------------------|-------------|---------|
| none | none | xxx |

> iOS

|  mbnv2-yolov3-nano | yolov4-tiny | yolov5s |
|-------------------|-------------|---------|
| <img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/iOS_iPhone6sp_mobilenetv2_yolov3_nano.jpg"/> |<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/iOS_iPhone6sp_yolov4_tiny.jpg"/>| <img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/iOS_iPhone6sp_yolov5s.jpg"/> |


Thanks:<br/>
- sunnyden, dog-qiuqiu, ..., nihui
- https://github.com/Tencent/ncnn

