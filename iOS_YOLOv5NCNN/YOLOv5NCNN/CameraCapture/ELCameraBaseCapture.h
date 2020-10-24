//
//  ELCameraBaseCapture.h
//  AVCaptureDemo
//
//  Created by yin linlin on 2018/5/17.
//  Copyright © 2018年 yin linlin. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

@interface ELCameraBaseCapture : NSObject

@property (nonatomic, strong) AVCaptureSession *captureSession;

//输入设备
@property (nonatomic, strong) AVCaptureDevice *captureDevice;

//输入流
@property (nonatomic, strong) AVCaptureDeviceInput *captureInput;

//照片输出流
@property (nonatomic, strong) AVCaptureStillImageOutput *imageOutput;

//视频帧流
@property (nonatomic, strong) AVCaptureVideoDataOutput *videoOutput;
@property (copy, atomic) void (^imageBlock)(UIImage *image);

//配置session
- (void)sessionConfig;

- (void)startSession;

- (void)stopSession;

- (void)takePhoto:(void(^)(UIImage *image, NSError *error))complete;


@end
