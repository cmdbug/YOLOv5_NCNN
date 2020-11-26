//
//  ViewController.m
//  YOLOv5NCNN
//
//  Created by WZTENG on 2020/7/5.
//  Copyright © 2020 TENG. All rights reserved.
//

#import "ViewController.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <ncnn/ncnn/net.h>  // 新版本(如果报错请尝试从官网下载后重新导入。download ncnn.framework/openmp.framework from ncnn and replace)
//#include <ncnn/net.h>  // 旧版本
#include "YoloV5.h"
#include "YoloV4.h"
#include "NanoDet.h"

#import <AVFoundation/AVFoundation.h>
#import <AVFoundation/AVMediaFormat.h>
#import "ELCameraControlCapture.h"
#import <opencv2/opencv.hpp>  // download opencv2.framework to project


@interface ViewController ()
@property (strong, nonatomic) IBOutlet UILabel *resultLabel;
@property (strong, nonatomic) IBOutlet UIImageView *imageView;
@property (strong, nonatomic) IBOutlet UISlider *nmsSlider;
@property (strong, nonatomic) IBOutlet UISlider *thresholdSlider;
@property (strong, nonatomic) IBOutlet UILabel *valueShowLabel;
@property (strong, nonatomic) IBOutlet UIView *preView;

@property (assign, nonatomic) float threshold;
@property (assign, nonatomic) float nms_threshold;

@property (assign, atomic) double total_fps;
@property (assign, atomic) double fps_count;

// 相机部分
@property (strong, nonatomic) ELCameraControlCapture *cameraCapture;
@property (strong, nonatomic) AVCaptureVideoPreviewLayer *preLayer;

@property YoloV5 *yolo;
@property YoloV4 *yolov4;
@property NanoDet *nanoDet;

@property (assign, atomic) Boolean isDetecting;
@property (nonatomic) dispatch_queue_t queue;

//@property (assign, nonatomic) Boolean USE_YOLOV5;  // YES:YOLOv5  NO:YOLOv4-tiny


@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    AVAuthorizationStatus authStatus = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    if (authStatus == AVAuthorizationStatusRestricted || authStatus == AVAuthorizationStatusDenied) {
        // 没有权限
        NSLog(@"没有权限");
    } else {
        // 有权限
    }
    self.queue = dispatch_queue_create("ncnn", DISPATCH_QUEUE_CONCURRENT);
    [self initTitleName];
    self.isDetecting = NO;
    [self initView];
    [self setCameraUI];
}

#pragma mark 显示标题
- (void)initTitleName {
    self.title = [[[self getModelName] stringByReplacingOccurrencesOfString:@"[CPU] " withString:@""] stringByReplacingOccurrencesOfString:@"[GPU] " withString:@""];
}

- (CGFloat)degreesToRadians:(CGFloat)degrees {
    return M_PI * (degrees / 180.0);
}

- (UIImage *)rotatedByDegrees:(CGFloat)degrees image:(UIImage *)image {
    // calculate the size of the rotated view's containing box for our drawing space
    UIView *rotatedViewBox = [[UIView alloc] initWithFrame:CGRectMake(0, 0, image.size.width, image.size.height)];
    CGAffineTransform t = CGAffineTransformMakeRotation([self degreesToRadians:degrees]);
    rotatedViewBox.transform = t;
    CGSize rotatedSize = rotatedViewBox.frame.size;
    
    // Create the bitmap context
    UIGraphicsBeginImageContext(rotatedSize);
    CGContextRef bitmap = UIGraphicsGetCurrentContext();
    
    // Move the origin to the middle of the image so we will rotate and scale around the center.
    CGContextTranslateCTM(bitmap, rotatedSize.width/2, rotatedSize.height/2);
    
    // Rotate the image context
    CGContextRotateCTM(bitmap, [self degreesToRadians:degrees]);
    
    // Now, draw the rotated/scaled image into the context
    CGContextScaleCTM(bitmap, 1.0, -1.0);
    CGContextDrawImage(bitmap, CGRectMake(-image.size.width / 2, -image.size.height / 2, image.size.width, image.size.height), [image CGImage]);
    
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}

#pragma mark 设置相机并回调
- (void)setCameraUI {
    [self setVideoPreview];
    __weak typeof(self) weakSelf = self;
    self.cameraCapture.imageBlock = ^(UIImage *image) {
//        NSLog(@"%f", image.size.height);
        if (weakSelf.isDetecting) {
            return;
        }
        weakSelf.isDetecting = YES;
        // 根据方向旋转图片
        __block float degrees = 0.0f;
        __block UIImage *temp = nil;
        dispatch_sync(dispatch_get_main_queue(), ^{
            if ([[UIApplication sharedApplication] statusBarOrientation] == UIInterfaceOrientationPortrait) {
                degrees = 90.0f;
            } else {
                degrees = -90.0f;
            }
            temp = [weakSelf rotatedByDegrees:degrees image:image];
        });
        dispatch_sync(weakSelf.queue, ^{
            [weakSelf detectImage:temp];
            weakSelf.isDetecting = NO;
        });
    };
    [self.cameraCapture startSession];
}

- (ELCameraControlCapture *)cameraCapture {
    if (!_cameraCapture) {
        _cameraCapture = [[ELCameraControlCapture alloc] init];
    }
    return _cameraCapture;
}

#pragma mark 相机预览画面位置
- (void)setVideoPreview {
    self.preLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.cameraCapture.captureSession];
    self.preLayer.backgroundColor = [[UIColor redColor] CGColor];
    UIEdgeInsets insets = self.view.safeAreaInsets;
    CGRect screen = [[UIScreen mainScreen] bounds];
    self.preLayer.frame = CGRectMake(5, screen.size.height - 145 - insets.bottom - 5, 110, 145);
    self.preLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
    [self.view.layer addSublayer:self.preLayer];
}

- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    [self.cameraCapture startSession];
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [self.cameraCapture stopSession];
}

- (void)viewDidDisappear:(BOOL)animated {
    [super viewDidDisappear:animated];
    dispatch_barrier_async(self.queue, ^{
        [self releaseModel];
    });
}


- (void)initView {
    self.threshold = 0.3f;
    self.nms_threshold = 0.7f;
    if (self.USE_MODEL == W_YOLOV5S) {
        self.threshold = 0.3f;
        self.nms_threshold = 0.7f;
    } else if (self.USE_MODEL == W_DBFACE || self.USE_MODEL == W_NANODET) {
        self.threshold = 0.3f;
        self.nms_threshold = 0.6f;
    } else if (self.USE_MODEL == W_YOLOV5S_CUSTOM_OP) {
        self.threshold = 0.25f;
        self.nms_threshold = 0.45f;
    }
    
//    self.imageView.image = [UIImage imageNamed:@"000000000650.jpg"];
    [self.nmsSlider addTarget:self action:@selector(nmsChange:forEvent:) forControlEvents:UIControlEventValueChanged];
    [self.thresholdSlider addTarget:self action:@selector(nmsChange:forEvent:) forControlEvents:UIControlEventValueChanged];
    if (self.USE_MODEL != W_YOLOV5S && self.USE_MODEL != W_DBFACE && self.USE_MODEL != W_NANODET && self.USE_MODEL != W_YOLOV5S_CUSTOM_OP) {
        self.nmsSlider.enabled = NO;
        self.thresholdSlider.enabled = NO;
    }
}

#pragma mark 顶部控件
- (void)nmsChange:(UISlider *)slider forEvent:(UIEvent *)event {
    UITouch *torchEvent = [[event allTouches] anyObject];
    switch (torchEvent.phase) {
        case UITouchPhaseBegan: {
            break;
        }
        case UITouchPhaseMoved: {
            NSString *value = [NSString stringWithFormat:@"Threshold:%.2f NMS:%.2f", self.thresholdSlider.value, self.nmsSlider.value];
            self.valueShowLabel.text = value;
            self.threshold = self.thresholdSlider.value;
            self.nms_threshold = self.nmsSlider.value;
            break;
        }
        case UITouchPhaseEnded: {
            NSString *value = [NSString stringWithFormat:@"Threshold:%.2f NMS:%.2f", self.thresholdSlider.value, self.nmsSlider.value];
            self.valueShowLabel.text = value;
            self.threshold = self.thresholdSlider.value;
            self.nms_threshold = self.nmsSlider.value;
            break;
        }
        default: {
            break;
        }
    }
    if (self.threshold <= 0) {
        self.threshold = 0.01;
        NSString *value = [NSString stringWithFormat:@"Threshold:%.2f NMS:%.2f", self.threshold, self.nmsSlider.value];
        self.valueShowLabel.text = value;
    }
}

#pragma mark 照片
- (IBAction)predict:(id)sender {
    // load image
//    UIImage* image = [UIImage imageNamed:@"000000000650.jpg"];
//    self.imageView.image = image;
//    if (self.USE_MODEL == W_YOLOV5S) {
//        YoloV5 *yolo = new YoloV5("", "");
//        std::vector<BoxInfo> result = yolo->dectect(image, self.threshold, self.nms_threshold);
//        printf("result size:%lu", result.size());
//        NSString *detect_result = @"";
//        for (int i = 0; i < result.size(); i++) {
//            BoxInfo boxInfo = result[i];
//            detect_result = [NSString stringWithFormat:@"%@\n%s %.3f", detect_result, yolo->labels[boxInfo.label].c_str(), boxInfo.score];
//        }
//        delete yolo;
//        self.resultLabel.text = detect_result;
//        self.imageView.image = [self drawBox:self.imageView image:image boxs:result];
//    } else if (self.USE_MODEL == W_YOLOV4TINY) {
//        YoloV4 *yolo = new YoloV4("", "", YES);
//        std::vector<BoxInfo> result = yolo->detectv4(image, self.threshold, self.nms_threshold);
//        printf("result size:%lu", result.size());
//        NSString *detect_result = @"";
//        for (int i = 0; i < result.size(); i++) {
//            BoxInfo boxInfo = result[i];
//            detect_result = [NSString stringWithFormat:@"%@\n%s %.3f", detect_result, yolo->labels[boxInfo.label].c_str(), boxInfo.score];
//        }
//        delete yolo;
//        self.resultLabel.text = detect_result;
//        self.imageView.image = [self drawBox:self.imageView image:image boxs:result];
//    }
}

#pragma mark 相机回调图片
- (void)detectImage:(UIImage *)image {
    // create model
    [self createModel];
    
    NSDate *start = [NSDate date];
    std::vector<BoxInfo> result;
    if (self.USE_MODEL == W_YOLOV5S) {
        result = self.yolo->dectect(image, self.threshold, self.nms_threshold);
    } else if (self.USE_MODEL == W_YOLOV4TINY
               || self.USE_MODEL == W_MOBILENETV2_YOLOV3_NANO
               || self.USE_MODEL == W_YOLO_FASTEST_XL) {
        result = self.yolov4->detectv4(image, self.threshold, self.nms_threshold);
    } else if (self.USE_MODEL == W_NANODET) {
        result = self.nanoDet->detect(image, self.threshold, self.nms_threshold);
    }
    __weak typeof(self) weakSelf = self;
    dispatch_sync(dispatch_get_main_queue(), ^{
        long dur = [[NSDate date] timeIntervalSince1970]*1000 - start.timeIntervalSince1970*1000;
        float fps = 1.0 / (dur / 1000.0);
        weakSelf.total_fps = (weakSelf.total_fps == 0) ? fps : (weakSelf.total_fps + fps);
        weakSelf.fps_count++;
        NSString *info = [NSString stringWithFormat:@"%@\nSize:%dx%d\nTime:%.3fs\nFPS:%.2f\nAVG_FPS:%.2f", [self getModelName], int(image.size.width), int(image.size.height), dur / 1000.0, fps, (float)weakSelf.total_fps / weakSelf.fps_count];
        weakSelf.resultLabel.text = info;
        if (weakSelf.USE_MODEL == W_YOLOV5S
            || weakSelf.USE_MODEL == W_YOLOV4TINY
            || weakSelf.USE_MODEL == W_MOBILENETV2_YOLOV3_NANO
            || weakSelf.USE_MODEL == W_YOLOV5S_CUSTOM_OP
            || weakSelf.USE_MODEL == W_NANODET
            || weakSelf.USE_MODEL == W_YOLO_FASTEST_XL) {
            weakSelf.imageView.image = [weakSelf drawBox:weakSelf.imageView image:image boxs:result];
        } else if (weakSelf.USE_MODEL == W_SIMPLE_POSE) {
            
        } else if (weakSelf.USE_MODEL == W_YOLACT) {
                   
        } else if (weakSelf.USE_MODEL == W_FACE_LANDMARK) {
                          
        } else if (weakSelf.USE_MODEL == W_DBFACE) {
                                 
        } else if (weakSelf.USE_MODEL == W_MOBILENETV2_FCN || weakSelf.USE_MODEL == W_MOBILENETV3_SEG) {
                                        
        }
    });
    
}

#pragma mark 创建模型
- (void)createModel {
    if (!self.yolo && self.USE_MODEL == W_YOLOV5S) {
        NSLog(@"new YoloV5");
        self.yolo = new YoloV5(self.USE_GPU);
    } else if (!self.yolov4 && self.USE_MODEL == W_YOLOV4TINY) {
        NSLog(@"new YoloV4");
        self.yolov4 = new YoloV4(self.USE_GPU, 0);
    } else if (!self.yolov4 && self.USE_MODEL == W_MOBILENETV2_YOLOV3_NANO) {
        NSLog(@"new YoloV3-nano");
        self.yolov4 = new YoloV4(self.USE_GPU, 1);
    } else if (!self.yolov4 && self.USE_MODEL == W_SIMPLE_POSE) {
        NSLog(@"new Simple-Pose");
    } else if (!self.yolov4 && self.USE_MODEL == W_YOLACT) {
        NSLog(@"new Yolact");
    } else if (!self.yolov4 && self.USE_MODEL == W_FACE_LANDMARK) {
        NSLog(@"new face-landmark");
    } else if (!self.yolov4 && self.USE_MODEL == W_DBFACE) {
        NSLog(@"new dbface");
    } else if (!self.yolov4 && self.USE_MODEL == W_MOBILENETV2_FCN) {
        NSLog(@"new mbnv2 fcn");
    } else if (!self.yolov4 && self.USE_MODEL == W_MOBILENETV3_SEG) {
        NSLog(@"new mbnv3 seg");
    } else if (!self.yolov4 && self.USE_MODEL == W_YOLOV5S_CUSTOM_OP) {
        NSLog(@"new yolov5s custom op");
    } else if (!self.nanoDet && self.USE_MODEL == W_NANODET) {
        NSLog(@"new nanodet");
        self.nanoDet = new NanoDet(self.USE_GPU);
    } else if (!self.yolov4 && self.USE_MODEL == W_YOLO_FASTEST_XL) {
        NSLog(@"new yolo fastest xl");
        self.yolov4 = new YoloV4(self.USE_GPU, 2);
    }
}

- (void)releaseModel {
    NSLog(@"release model");
    delete self.yolo;
    delete self.yolov4;
    delete self.nanoDet;
}

#pragma mark 获取模型名称
- (NSString *)getModelName {
    NSString *name = @"ohhhhh";
    if (self.USE_MODEL == W_YOLOV5S) {
        name = @"YOLOv5s";
    } else if (self.USE_MODEL == W_YOLOV4TINY) {
        name = @"YOLOv4-tiny";
    } else if (self.USE_MODEL == W_MOBILENETV2_YOLOV3_NANO) {
        name = @"MobileNetV2-YOLOv3_Nano";
    } else if (self.USE_MODEL == W_SIMPLE_POSE) {
        name = @"Simple-Pose";
    } else if (self.USE_MODEL == W_YOLACT) {
        name = @"Yolact";
    } else if (self.USE_MODEL == W_FACE_LANDMARK) {
        name = @"YoloFace500k-landmark106";
    } else if (self.USE_MODEL == W_DBFACE) {
        name = @"DBFace";
    } else if (self.USE_MODEL == W_MOBILENETV2_FCN) {
        name = @"MobileNetV2-FCN";
    } else if (self.USE_MODEL == W_MOBILENETV3_SEG) {
        name = @"MBNv3-Segmentation-small";
    } else if (self.USE_MODEL == W_YOLOV5S_CUSTOM_OP) {
        name = @"YOLOv5s_Custom_Layer";
    } else if (self.USE_MODEL == W_NANODET) {
        name = @"NanoDet";
    } else if (self.USE_MODEL == W_YOLO_FASTEST_XL) {
        name = @"YOLO-Fastest-xl";
    }
    NSString *modelName = self.USE_GPU ? [NSString stringWithFormat:@"[GPU] %@", name] : [NSString stringWithFormat:@"[CPU] %@", name];
    return modelName;
}

#pragma mark 绘制结果
- (UIImage *)drawBox:(UIImageView *)imageView image:(UIImage *)image boxs:(std::vector<BoxInfo>)boxs {
    UIGraphicsBeginImageContext(image.size);

    [image drawAtPoint:CGPointMake(0,0)];

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetLineWidth(context, fmax(image.size.width/200, 1));
    NSMutableParagraphStyle *style = [[NSMutableParagraphStyle alloc] init];
    for (int i = 0; i < boxs.size(); i++) {
        BoxInfo box = boxs[i];
        srand(box.label + 2020);
        UIColor *color = [UIColor colorWithRed:rand()%256/255.0f green:rand()%256/255.0f blue:rand()%255/255.0f alpha:1.0f];
        CGContextAddRect(context, CGRectMake(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1));
        NSString *label = nil;
        if (self.USE_MODEL == W_YOLOV5S) {
            label = [NSString stringWithFormat:@"%s %.3f", self.yolo->labels[box.label].c_str(), box.score];
        } else if (self.USE_MODEL == W_YOLOV4TINY
                   || self.USE_MODEL == W_MOBILENETV2_YOLOV3_NANO
                   || self.USE_MODEL == W_YOLO_FASTEST_XL) {
            label = [NSString stringWithFormat:@"%s %.3f", self.yolov4->labels[box.label].c_str(), box.score];
        } else if (self.USE_MODEL == W_NANODET) {
            label = [NSString stringWithFormat:@"%s %.3f", self.nanoDet->labels[box.label].c_str(), box.score];
        }
        [label drawAtPoint:CGPointMake(box.x1 + 2, box.y1 - 3) withAttributes:@{NSFontAttributeName:[UIFont systemFontOfSize:20], NSParagraphStyleAttributeName:style, NSForegroundColorAttributeName:color}];
        
        CGContextSetStrokeColorWithColor(context, [color CGColor]);
        CGContextStrokePath(context);
    }
//    CGContextSetStrokeColorWithColor(context, [lineColor CGColor]);
//    CGContextStrokePath(context);
    //创建新图像
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();

    UIGraphicsEndImageContext();
    return newImage;
}

@end

