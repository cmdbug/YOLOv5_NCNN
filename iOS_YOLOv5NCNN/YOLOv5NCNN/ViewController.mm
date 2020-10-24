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
#include <ncnn/ncnn/net.h>  // 新版本
//#include <ncnn/net.h>  // 旧版本
#include "YoloV5.h"
#include "YoloV4.h"

#import <AVFoundation/AVFoundation.h>
#import <AVFoundation/AVMediaFormat.h>
#import "ELCameraControlCapture.h"


@interface ViewController ()
@property (strong, nonatomic) IBOutlet UILabel *resultLabel;
@property (strong, nonatomic) IBOutlet UIImageView *imageView;
@property (strong, nonatomic) IBOutlet UISlider *nmsSlider;
@property (strong, nonatomic) IBOutlet UISlider *thresholdSlider;
@property (strong, nonatomic) IBOutlet UILabel *valueShowLabel;
@property (strong, nonatomic) IBOutlet UIView *preView;

@property (assign, nonatomic) float threshold;
@property (assign, nonatomic) float nms_threshold;

// 相机部分
@property (strong, nonatomic) ELCameraControlCapture *cameraCapture;
@property (strong, nonatomic) AVCaptureVideoPreviewLayer *preLayer;

@property YoloV5 *yolo;
@property YoloV4 *yolov4;
@property (assign, atomic) Boolean isDetecting;

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
    [self initTitleName];
    self.isDetecting = NO;
    [self initView];
    [self setCameraUI];
}

- (void)initTitleName {
    if (self.USE_MODEL == W_YOLOV5S) {
        self.title = @"YOLOv5s";
    } else if (self.USE_MODEL == W_YOLOV4TINY) {
        self.title = @"YOLOV4-tiny";
    } else {
        self.title = @"ohhhhhh";
    }
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
    
    //   // Rotate the image context
    CGContextRotateCTM(bitmap, [self degreesToRadians:degrees]);
    
    // Now, draw the rotated/scaled image into the context
    CGContextScaleCTM(bitmap, 1.0, -1.0);
    CGContextDrawImage(bitmap, CGRectMake(-image.size.width / 2, -image.size.height / 2, image.size.width, image.size.height), [image CGImage]);
    
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}

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
        dispatch_sync(dispatch_get_global_queue(0, 0), ^{
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

- (void)setVideoPreview {
    self.preLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.cameraCapture.captureSession];
    self.preLayer.backgroundColor = [[UIColor redColor] CGColor];
    UIEdgeInsets insets = self.view.safeAreaInsets;
    CGRect screen = [[UIScreen mainScreen] bounds];
    self.preLayer.frame = CGRectMake(15, screen.size.height - 145 - insets.bottom - 15, 110, 145);
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


- (void)initView {
    self.threshold = 0.3f;
    self.nms_threshold = 0.7f;
//    self.imageView.image = [UIImage imageNamed:@"000000000650.jpg"];
    [self.nmsSlider addTarget:self action:@selector(nmsChange:forEvent:) forControlEvents:UIControlEventValueChanged];
    [self.thresholdSlider addTarget:self action:@selector(nmsChange:forEvent:) forControlEvents:UIControlEventValueChanged];
    if (!(self.USE_MODEL == W_YOLOV5S)) {
        self.nmsSlider.enabled = NO;
        self.thresholdSlider.enabled = NO;
    }
}

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

// 推理
- (IBAction)predict:(id)sender {
    // load image
    UIImage* image = [UIImage imageNamed:@"000000000650.jpg"];
    self.imageView.image = image;
    if (self.USE_MODEL == W_YOLOV5S) {
        YoloV5 *yolo = new YoloV5("", "");
        std::vector<BoxInfo> result = yolo->dectect(image, self.threshold, self.nms_threshold);
        printf("result size:%lu", result.size());
        NSString *detect_result = @"";
        for (int i = 0; i < result.size(); i++) {
            BoxInfo boxInfo = result[i];
            detect_result = [NSString stringWithFormat:@"%@\n%s %.3f", detect_result, yolo->labels[boxInfo.label].c_str(), boxInfo.score];
        }
        delete yolo;
        self.resultLabel.text = detect_result;
        self.imageView.image = [self drawBox:self.imageView image:image boxs:result];
    } else if (self.USE_MODEL == W_YOLOV4TINY) {
        YoloV4 *yolo = new YoloV4("", "");
        std::vector<BoxInfo> result = yolo->detectv4(image, self.threshold, self.nms_threshold);
        printf("result size:%lu", result.size());
        NSString *detect_result = @"";
        for (int i = 0; i < result.size(); i++) {
            BoxInfo boxInfo = result[i];
            detect_result = [NSString stringWithFormat:@"%@\n%s %.3f", detect_result, yolo->labels[boxInfo.label].c_str(), boxInfo.score];
        }
        delete yolo;
        self.resultLabel.text = detect_result;
        self.imageView.image = [self drawBox:self.imageView image:image boxs:result];
    }
}

- (void)detectImage:(UIImage *)image {
    if (!self.yolo && self.USE_MODEL == 1) {
        NSLog(@"new YoloV5");
        self.yolo = new YoloV5("", "");
    } else if (!self.yolov4 && self.USE_MODEL == 2) {
        NSLog(@"new YoloV4");
        self.yolov4 = new YoloV4("", "");
    }
    if (self.USE_MODEL == W_YOLOV5S) {
        NSDate *start = [NSDate date];
        std::vector<BoxInfo> result = self.yolo->dectect(image, self.threshold, self.nms_threshold);
        NSString *detect_result = @"";
        for (int i = 0; i < result.size(); i++) {
            BoxInfo boxInfo = result[i];
            detect_result = [NSString stringWithFormat:@"%@\n%s %.3f", detect_result, self.yolo->labels[boxInfo.label].c_str(), boxInfo.score];
        }
//        delete self.yolo;
        __weak typeof(self) weakSelf = self;
        dispatch_sync(dispatch_get_main_queue(), ^{
            long dur = [[NSDate date] timeIntervalSince1970]*1000 - start.timeIntervalSince1970*1000;
            NSString *info = [NSString stringWithFormat:@"YOLOv5s\nSize:%dx%d\nTime:%.3fs\nFPS:%.2f", int(image.size.width), int(image.size.height), dur / 1000.0, 1.0 / (dur / 1000.0)];
            weakSelf.resultLabel.text = info;
            weakSelf.imageView.image = [weakSelf drawBox:weakSelf.imageView image:image boxs:result];
        });
    } else if (self.USE_MODEL == W_YOLOV4TINY) {
        NSDate *start = [NSDate date];
        std::vector<BoxInfo> result = self.yolov4->detectv4(image, self.threshold, self.nms_threshold);
        NSString *detect_result = @"";
        for (int i = 0; i < result.size(); i++) {
            BoxInfo boxInfo = result[i];
            detect_result = [NSString stringWithFormat:@"%@\n%s %.3f", detect_result, self.yolov4->labels[boxInfo.label].c_str(), boxInfo.score];
        }
//        delete self.yolov4;
        __weak typeof(self) weakSelf = self;
        dispatch_sync(dispatch_get_main_queue(), ^{
            long dur = [[NSDate date] timeIntervalSince1970]*1000 - start.timeIntervalSince1970*1000;
            NSString *info = [NSString stringWithFormat:@"YOLOv4-tiny\nSize:%dx%d\nTime:%.3fs\nFPS:%.2f", int(image.size.width), int(image.size.height), dur / 1000.0, 1.0 / (dur / 1000.0)];
            weakSelf.resultLabel.text = info;
            weakSelf.imageView.image = [weakSelf drawBox:weakSelf.imageView image:image boxs:result];
        });
    }
    
}

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
        if (self.USE_MODEL == W_YOLOV5S) {
            NSString *name = [NSString stringWithFormat:@"%s %.3f", self.yolo->labels[box.label].c_str(), box.score];
            [name drawAtPoint:CGPointMake(box.x1, box.y1) withAttributes:@{NSFontAttributeName:[UIFont systemFontOfSize:35], NSParagraphStyleAttributeName:style, NSForegroundColorAttributeName:color}];
        } else if (self.USE_MODEL == W_YOLOV4TINY) {
            NSString *name = [NSString stringWithFormat:@"%s %.3f", self.yolov4->labels[box.label].c_str(), box.score];
            [name drawAtPoint:CGPointMake(box.x1, box.y1) withAttributes:@{NSFontAttributeName:[UIFont systemFontOfSize:35], NSParagraphStyleAttributeName:style, NSForegroundColorAttributeName:color}];
        }
        
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

