[English](https://github.com/cmdbug/YOLOv5_NCNN/blob/master/README.md)

## :rocket: 如果有帮助，点个star！:star: ##

### 移动端NCNN部署，项目支持YOLOv5s、YOLOv4-tiny、MobileNetV2-YOLOv3-nano、Simple-Pose、Yolact、ChineseOCR-lite、ENet、Landmark106、DBFace、MBNv2-FCN与MBNv3-Seg-small模型，摄像头实时捕获视频流进行检测。

## iOS:
- Xcode 11.5
- macOS 10.15.4
- iPhone 6sp 13.5.1

## Android:
- Android Studio 4.0
- Win10 1909
- Meizu 16x 8.1.0 (CPU:Qualcomm 710 GPU:Adreno 616)

安卓已经增加权限申请，但如果还是闪退请手动确认下相关权限是否允许。

> iOS
```code
YOLOv5s:     从界面中选择需要测试的模型。
YOLOv4-tiny: 从界面中选择需要测试的模型。
YOLOv3-nano: 从界面中选择需要测试的模型。
```
> Android
```
从界面中选择需要测试的模型。
```
### 模型
* ***YOLOv5s*** 输入尺寸减小，解码过程使用了大量的 for 循环与 NMS 表现出来会比较慢。
* ***YOLOv4-tiny*** 使用默认尺寸，解码过程没有大量的 for 与 NMS 所以速度会快些。
* ***YOLOv3-nano*** 与 v4-tiny 一样。
* ***Simple-Pose*** 暂时只写了安卓版本，iOS目前还没有增加。内部原理是先检测人再用人的区域再次进行姿态检测，即2步过程。
* ***Yolact*** 暂时只写了安卓版本，iOS目前还没有增加。
* ***ChineseOCR_lite*** 暂时只写了安卓版本，iOS目前还没有增加。(需要注意的是偶尔会发生崩溃现象，有空修复一下)
* ***ENet*** 暂时只写了安卓版本，iOS目前还没有增加。(由于模型太小所以分割效果比较差，可以自行更换更强的网络。但是效果太差了可能哪里有问题)
* ***Landmark106*** 暂时只写了安卓版本，iOS目前还没有增加。内部原理是先检测脸再用脸的区域再次进行关键点检测，即2步过程。
* ***DBFace*** 暂时只写了安卓版本，iOS目前还没有增加。
* ***MBNv3-FCN*** 暂时只写了安卓版本，iOS目前还没有增加。(感谢网名Persistence提供帮助)
* ***MBNv3-Seg-small*** 暂时只写了安卓版本，iOS目前还没有增加。

### Note：
* 由于手机性能、图像尺寸等因素导致FPS在不同手机上相差比较大。该项目主要测试NCNN框架的使用，具体模型的转换可以去NCNN官方查看转换教程。<br/>
* 由于opencv库太大只保留 arm64-v8a/armeabi-v7a 有需要其它版本的自己去官方下载。
* ncnn暂时使用vulkan版本，在加载前需要打开加速，本项目中没有打开。如果要用ncnn版本需要修改CMakeLists.txt配置。
* AS版本不一样可能编译会有各种问题，如果编译错误无法解决、建议使用AS4.0以上版本尝试一下。

已知部分ncnn大佬网名：.... nihui qiuqiu

:art: 截图<br/>

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

> iOS

|  mbnv2-yolov3-nano | yolov4-tiny | yolov5s |
|-------------------|-------------|---------|
| <img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/iOS_iPhone6sp_mobilenetv2_yolov3_nano.jpg"/> |<img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/iOS_iPhone6sp_yolov4_tiny.jpg"/>| <img width="270" height="500" src="https://github.com/cmdbug/YOLOv5_NCNN/blob/master/Screenshots/iOS_iPhone6sp_yolov5s.jpg"/> |


感谢:<br/>
- sunnyden, dog-qiuqiu, ..., nihui
- https://github.com/Tencent/ncnn

