//
//  ELCameraControlCapture.h
//  AVCaptureDemo
//
//  Created by yin linlin on 2018/5/17.
//  Copyright © 2018年 yin linlin. All rights reserved.
//

#import "ELCameraBaseCapture.h"
@interface ELCameraControlCapture : ELCameraBaseCapture

#pragma mark 切换摄像头
- (BOOL)switchCamera;
- (BOOL)canSwitchCamera;
#pragma mark 设置闪光灯模式
- (BOOL)setFlashMode:(AVCaptureFlashMode)flashMode;

#pragma mark 设置手电筒开关
- (BOOL)setTorchMode:(AVCaptureTorchMode)torchMode;

#pragma mark 焦距调整
- (BOOL)setFocusMode:(AVCaptureFocusMode)focusMode;
- (BOOL)setFocusPoint:(CGPoint)point;

#pragma mark 曝光量调节
- (BOOL)setExposureMode:(AVCaptureExposureMode)exposureMode;

#pragma mark 白平衡
- (BOOL)setWhiteBalanceMode:(AVCaptureWhiteBalanceMode)whiteBalanceMode;


@end
