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
Select the model to be tested directly on the interface.
```
> Android
```
Select the model to be tested directly on the interface.
```
### Models
| model | android | iOS | from |
|-------------------|:--------:|:--------:|:--------:|
| YOLOv5s           | yes | yes |  [Github](https://github.com/ultralytics/yolov5)   |
| YOLOv4-tiny       | yes | yes |  [Github](https://github.com/ultralytics/yolov3)   |
| YOLOv3-nano       | yes | yes |  [Github](https://github.com/dog-qiuqiu/MobileNet-Yolo)   |
| Simple-Pose       | yes | coming soon |  [Github](https://github.com/dog-qiuqiu/MobileNet-Yolo)   |
| Yolact            | yes | coming soon |  [Github](https://github.com/dbolya/yolact) [zhihu](https://zhuanlan.zhihu.com/p/128974102)  |
| ChineseOCR_lite   | yes | probably |  [Github](https://github.com/ouyanghuiyu/chineseocr_lite) [zhihu](https://zhuanlan.zhihu.com/p/113338890)   |
| ENet              | bug | cancel |  [Github](https://github.com/davidtvs/PyTorch-ENet)   |
| Landmark106       | yes | coming soon |  [Github](https://github.com/dog-qiuqiu/MobileNet-Yolo)   |
| DBFace            | yes | coming soon |  [Github](https://github.com/yuanluw/DBface_ncnn_demo)   |
| MBNv2-FCN         | yes | coming soon |  [Github](https://github.com/open-mmlab/mmsegmentation)   |
| MBNv3-Seg-small   | yes | coming soon |  [Github](https://github.com/Tramac/Lightweight-Segmentation)   |
| YOLOv5s_custom_op | yes | yes |  [zhihu](https://zhuanlan.zhihu.com/p/275989233)   |
| NanoDet           | yes | yes |  [Github](https://github.com/RangiLyu/nanodet)   |
| YOLO-Fastest-xl   | yes | bug |  [Github](https://github.com/dog-qiuqiu/Yolo-Fastest)   |

iOS:
- Copy .param and .bin from "android_YOLOV5_NCNN\app\src\main\assets" to "iOS_YOLOv5NCNN\YOLOv5NCNN\res"
- If it prompts that net.h can't be found, you need to download it from the ncnn official website or compile .framework yourself and replace it in the project. If opencv2.framework is useful, you need to download it again and replace it in the project.
- The default library used by iOS does not include vulkan and bitcode.

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

|  yolov5s_custom_op | nanodet | yolo-fastest-xl |
|-------------------|-------------|---------|
| none | none | none |

> iOS

|  mbnv2-yolov3-nano | yolov4-tiny | yolov5s |
|-------------------|-------------|---------|
| <img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/iOS_iPhone6sp_mobilenetv2_yolov3_nano.jpg"/> |<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/iOS_iPhone6sp_yolov4_tiny.jpg"/>| <img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/iOS_iPhone6sp_yolov5s.jpg"/> |

|  yolov5s_custom_op | nanodet | yolo-fastest-xl |
|-------------------|-------------|---------|
| none | none | none |


Thanks:<br/>
- sunnyden, dog-qiuqiu, ..., nihui
- https://github.com/Tencent/ncnn

