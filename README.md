[中文说明](./README_CN.md)

## :rocket: If it helps you, click a star! :star: ##

### Ncnn deployment on mobile,support:YOLOv5s,YOLOv4-tiny,MobileNetV2-YOLOv3-nano,Simple-Pose,Yolact,ChineseOCR-lite,ENet,Landmark106,DBFace,MBNv2-FCN and MBNv3-Seg-small on camera.

## iOS:
- Xcode 12.4
- macOS 11.2.3
- iPhone 6sp 13.5.1

## Android:
- Android Studio 4.1
- Win10 20H2
- CPU:Qualcomm 710 GPU:Adreno 616

> iOS
```code
Select the model to be tested directly on the interface.
```
> Android
```
Select the model to be tested directly on the interface.
```
### Models
| model | android | iOS | from | other |
|-------------------|:--------:|:--------:|:--------:|:--------:|
| YOLOv5s           | yes | yes |  [Github](https://github.com/ultralytics/yolov5)   | [TNN](https://github.com/cmdbug/TNN_Demo) |
| YOLOv4-tiny       | yes | yes |  [Github](https://github.com/ultralytics/yolov3)   |
| YOLOv3-nano       | yes | yes |  [Github](https://github.com/dog-qiuqiu/MobileNet-Yolo)   |
| YOLOv5s_custom_op | yes | yes |  [zhihu](https://zhuanlan.zhihu.com/p/275989233)   |
| NanoDet           | yes | yes |  [Github](https://github.com/RangiLyu/nanodet)   | [TNN](https://github.com/cmdbug/TNN_Demo) [MNN](https://github.com/cmdbug/MNN_Demo) |
| YOLO-Fastest-xl   | yes | yes |  [Github](https://github.com/dog-qiuqiu/Yolo-Fastest)   |
| Simple-Pose       | yes | yes |  [Github](https://github.com/dog-qiuqiu/MobileNet-Yolo)   |
| Yolact            | yes | yes |  [Github](https://github.com/dbolya/yolact) [zhihu](https://zhuanlan.zhihu.com/p/128974102)  |
| ChineseOCR_lite   | yes | yes |  [Github](https://github.com/ouyanghuiyu/chineseocr_lite) [zhihu](https://zhuanlan.zhihu.com/p/113338890)   |
| ENet              | bug | cancel |  [Github](https://github.com/davidtvs/PyTorch-ENet)   |
| Landmark106       | yes | yes |  [Github](https://github.com/dog-qiuqiu/MobileNet-Yolo)   |
| DBFace            | yes | yes |  [Github](https://github.com/yuanluw/DBface_ncnn_demo)   |
| MBNv2-FCN         | yes | yes |  [Github](https://github.com/open-mmlab/mmsegmentation)   |
| MBNv3-Seg-small   | yes | yes |  [Github](https://github.com/Tramac/Lightweight-Segmentation)   |
| Light_OpenPose    | yes | yes |  [Github](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)   |


### iOS:
- Copy .param and .bin from "android_YOLOV5_NCNN\app\src\main\assets" to "iOS_YOLOv5NCNN\YOLOv5NCNN\res"
- If it prompts that net.h can't be found, you need to download it from the ncnn official website or compile .framework(20201208) yourself and replace it in the project. If opencv2.framework(4.3.0) is useful, you need to download it again and replace it in the project.
- The default library used by iOS does not include vulkan and bitcode.
- Normally, you need to re-download ncnn.framework/glslang.framework/openmp.framework/opencv2.framework and replace it with the project.
- For the configuration of Vulkan, please refer to the general configuration mentioned in Issues.

### Android：
* Due to factors such as mobile phone performance and image size, FPS varies greatly on different mobile phones. This project mainly tests the use of the NCNN framework. For the conversion of specific models, you can go to the NCNN official to view the conversion tutorial.
* Because the opencv library is too large, only arm64-v8a/armeabi-v7a is reserved. If you need other versions, go to the official download.
* ncnn temporarily uses the vulkan version, and acceleration needs to be turned on before loading, which is not turned on in this project. If you want to use the ncnn version, you need to modify the CMakeLists.txt configuration.
* Different AS versions may have various problems with compilation. If the compilation error cannot be solved, it is recommended to use AS4.0 or higher to try.
* ncnn has been updated to a new version, which includes ncnn The official import method of cmake.

This project is more about practicing the use and deployment of various models, without too much processing in terms of speed. If you have requirements for speed, you can directly obtain data such as YUV for direct input or use methods such as texture and opengl to achieve data input, reducing intermediate data transmission and conversion.

Convert locally(Will not upload model): [xxxx -> ncnn](https://convertmodel.com/)

Minimal OpenCV:[opencv-mobile](https://github.com/nihui/opencv-mobile)

:art: Screenshot<br/>

| Android | iOS |
|:-----:|:-----:|
|<img width="324" height="145" src="./Screenshots/Android_CPU_or_GPU.jpg"/>| <img width="320" height="166" src="./Screenshots/iOS_CPU_or_GPU.jpg"/> |

> Android

| mbnv2-yolov3-nano | yolov4-tiny | yolov5s |
|-------------------|-------------|---------|
|<img width="270" height="500" src="./Screenshots/Android_Meizu16x_mobilenetv2_yolov3_nano.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_Meizu16x_yolov4_tiny.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_Meizu16x_yolov5s.jpg"/>|

| simple_pose | yolact | chineseocr_lite_01 |
|-------------------|-------------|---------|
|<img width="270" height="500" src="./Screenshots/Android_Meizu16x_simple_pose.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_Meizu16x_yolact.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_meizu16x_chineseocr_lite_01.jpg"/>|

| chineseocr_lite_02 | ENet | yoloface500k-landmark106 |
|-------------------|-------------|---------|
|<img width="270" height="500" src="./Screenshots/Android_meizu16x_chineseocr_lite_02.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_Meizu16x_ENet.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_Meizu16x_yoloface500k-landmark106.jpg"/>|

|  dbface | mbnv2_fcn | mbnv3_seg_small |
|-------------------|-------------|---------|
| <img width="270" height="500" src="./Screenshots/Android_Meizu16x_dbface.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_Meizu16x_MBNFCN.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_Meizu16x_MBNV3_Seg_small.jpg"/>|

|  yolov5s_custom_op | nanodet | yolo-fastest-xl |
|-------------------|-------------|---------|
| <img width="270" height="500" src="./Screenshots/Android_meizu16x_yolov5s_custom_layer.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_meizu16x_nanodet.jpg"/>|<img width="270" height="500" src="./Screenshots/Android_meizu16x_yolo_fastest_xl.jpg"/>|

|  light_openpose  |
|-------------------|
| <img width="270" height="500" src="./Screenshots/Android_Meizu16x_lightopenpose.jpg"/>|


> iOS

|  mbnv2-yolov3-nano | yolov4-tiny | yolov5s |
|-------------------|-------------|---------|
| <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_mobilenetv2_yolov3_nano.jpg"/> |<img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_yolov4_tiny.jpg"/>| <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_yolov5s.jpg"/> |

|  yolov5s_custom_op | nanodet | yolo-fastest-xl |
|-------------------|-------------|---------|
| <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_yolov5s_custom_op.jpg"/> | <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_nanodet.jpg"/> | <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_yolo_fastest_xl.jpg"/> |

|  mbnv2_fcn | mbnv3_seg_small | simple_pose |
|-------------------|-------------|---------|
| <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_mobilenetv2_fcn.jpg"/> | <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_mbnv3_segmentation_small.jpg"/> | <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_simple_pose.jpg"/> |

| chineseocr_lite_01 | chineseocr_lite_02 | light_openpose |
|-------------------|-------------|---------|
| <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_chineseocr_lite_01.jpg"/> | <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_chineseocr_lite_02.jpg"/> | <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_lightopenpose.jpg"/> |

|  yolact | yoloface500k-landmark106 | dbface |
|-------------------|-------------|---------|
| <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_yolact.jpg"/> | <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_yoloface500k_landmark106.jpg"/> | <img width="270" height="480" src="./Screenshots/iOS_iPhone6sp_dbface.jpg"/> |


Thanks:<br/>
- sunnyden, dog-qiuqiu, ..., nihui
- https://github.com/Tencent/ncnn

