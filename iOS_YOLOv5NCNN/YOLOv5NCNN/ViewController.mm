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
#include "YoloV5CustomLayer.h"
#include "MobileNetV3Seg.h"
#include "MbnFCN.h"
#include "SimplePose.h"
#include "FaceLandmark.h"
#include "DBFace.h"
#include "Yolact.h"
#include "LightOpenPose.h"

#import <AVFoundation/AVFoundation.h>
#import <AVFoundation/AVMediaFormat.h>
#import "ELCameraControlCapture.h"
#import <opencv2/opencv.hpp>  // download opencv2.framework to project


@interface ViewController () <UINavigationControllerDelegate, UIImagePickerControllerDelegate>
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
@property YoloV5CustomLayer *yolov5CustomOp;
@property MBNV3Seg *mbnv3Seg;
@property MbnFCN *mbnFCN;
@property SimplePose *simplePose;
@property FaceLandmark *faceLandmark;
@property DBFace *dbFace;
@property Yolact *yolact;
@property LightOpenPose *lightOpenpose;

@property (assign, atomic) Boolean isDetecting;
@property (assign, atomic) Boolean isPhoto;

@property (nonatomic) dispatch_queue_t queue;


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
    self.isPhoto = NO;
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
        if (weakSelf.isDetecting || weakSelf.isPhoto) {
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

- (void)tapImageViewRecognizer {
    UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(tapClick)];
    tap.numberOfTapsRequired = 1;
    self.imageView.userInteractionEnabled = YES;
    [self.imageView addGestureRecognizer:tap];
}

- (void)tapClick {
    self.isPhoto = NO;
    self.isDetecting = NO;
    [self.cameraCapture startSession];
}

- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    if (!self.isPhoto) {
        [self.cameraCapture startSession];
    }
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [self.cameraCapture stopSession];
}

- (void)viewDidDisappear:(BOOL)animated {
    [super viewDidDisappear:animated];
    if (!self.isPhoto) {
        dispatch_barrier_async(self.queue, ^{
            [self releaseModel];
        });
    }
}


- (void)initView {
    self.threshold = 0.3f;
    self.nms_threshold = 0.7f;
    if (self.USE_MODEL == W_YOLOV5S) {
        self.threshold = 0.3f;
        self.nms_threshold = 0.7f;
    } else if (self.USE_MODEL == W_DBFACE || self.USE_MODEL == W_NANODET) {
        self.threshold = 0.4f;
        self.nms_threshold = 0.6f;
    } else if (self.USE_MODEL == W_YOLOV5S_CUSTOM_OP) {
        self.threshold = 0.25f;
        self.nms_threshold = 0.45f;
    }
    [self.thresholdSlider setValue:self.threshold];
    [self.nmsSlider setValue:self.nms_threshold];
    NSString *value = [NSString stringWithFormat:@"Threshold:%.2f NMS:%.2f", self.threshold, self.nms_threshold];
    self.valueShowLabel.text = value;
    
    [self.nmsSlider addTarget:self action:@selector(nmsChange:forEvent:) forControlEvents:UIControlEventValueChanged];
    [self.thresholdSlider addTarget:self action:@selector(nmsChange:forEvent:) forControlEvents:UIControlEventValueChanged];
    if (self.USE_MODEL != W_YOLOV5S && self.USE_MODEL != W_DBFACE && self.USE_MODEL != W_NANODET && self.USE_MODEL != W_YOLOV5S_CUSTOM_OP) {
        self.nmsSlider.enabled = NO;
        self.thresholdSlider.enabled = NO;
    }
    
    [self tapImageViewRecognizer];
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
    self.isPhoto = YES;
    UIImagePickerController *picVC = [[UIImagePickerController alloc] init];
    picVC.delegate = self;
    picVC.allowsEditing = NO;
    [self presentViewController:picVC animated:YES completion:nil];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<UIImagePickerControllerInfoKey,id> *)info {
    NSLog(@"did pick image");
    self.isDetecting = YES;
    UIImage *image = info[@"UIImagePickerControllerOriginalImage"];
    self.imageView.image = image;
    __weak typeof(self) weakSelf = self;
    dispatch_async(self.queue, ^{
        [weakSelf detectImage:image];
    });
    
    [self dismissViewControllerAnimated:YES completion:^{
        self.isPhoto = NO;
    }];
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
    NSLog(@"did cancel image");
    self.isPhoto = NO;
    [self dismissViewControllerAnimated:YES completion:^{
        self.isPhoto = NO;
        self.isDetecting = NO;
    }];
}

#pragma mark 相机回调图片
- (void)detectImage:(UIImage *)image {
    // create model
    [self createModel];
    
    NSDate *start = [NSDate date];
    std::vector<BoxInfo> result;
    ncnn::Mat maskMat;
    std::vector<PoseResult> poseResult;
    std::vector<FaceKeyPoint> faceLandmarkResult;
    std::vector<Obj> dbFaceResult;
    std::vector<Object> yolactResult;
    std::vector<human_pose_estimation::HumanPose> humanPose;
    if (self.USE_MODEL == W_YOLOV5S) {
        result = self.yolo->dectect(image, self.threshold, self.nms_threshold);
    } else if (self.USE_MODEL == W_YOLOV4TINY
               || self.USE_MODEL == W_MOBILENETV2_YOLOV3_NANO
               || self.USE_MODEL == W_YOLO_FASTEST_XL) {
        result = self.yolov4->detectv4(image, self.threshold, self.nms_threshold);
    } else if (self.USE_MODEL == W_NANODET) {
        result = self.nanoDet->detect(image, self.threshold, self.nms_threshold);
    } else if (self.USE_MODEL == W_YOLOV5S_CUSTOM_OP) {
        result = self.yolov5CustomOp->detect(image, self.threshold, self.nms_threshold);
    } else if (self.USE_MODEL == W_SIMPLE_POSE) {
        poseResult = self.simplePose->detect(image);
    } else if (self.USE_MODEL == W_DBFACE) {
        dbFaceResult = self.dbFace->detect(image, self.threshold, self.nms_threshold);
    } else if (self.USE_MODEL == W_FACE_LANDMARK) {
        faceLandmarkResult = self.faceLandmark->detect(image);
    } else if (self.USE_MODEL == W_YOLACT) {
        yolactResult = self.yolact->detect_yolact(image);
    } else if (self.USE_MODEL == W_MOBILENETV2_FCN) {
        maskMat = self.mbnFCN->detect_mbnfcn(image);
    } else if (self.USE_MODEL == W_MOBILENETV3_SEG) {
        maskMat = self.mbnv3Seg->detect_mbnseg(image);
    } else if (self.USE_MODEL == W_LIGHT_OPENPOSE) {
        humanPose = self.lightOpenpose->detect(image);
    }
    __weak typeof(self) weakSelf = self;
    dispatch_sync(dispatch_get_main_queue(), ^{
        if (weakSelf.USE_MODEL == W_YOLOV5S
            || weakSelf.USE_MODEL == W_YOLOV4TINY
            || weakSelf.USE_MODEL == W_MOBILENETV2_YOLOV3_NANO
            || weakSelf.USE_MODEL == W_YOLOV5S_CUSTOM_OP
            || weakSelf.USE_MODEL == W_NANODET
            || weakSelf.USE_MODEL == W_YOLO_FASTEST_XL) {
            weakSelf.imageView.image = [weakSelf drawBox:weakSelf.imageView image:image boxs:result];
        } else if (weakSelf.USE_MODEL == W_SIMPLE_POSE) {
            weakSelf.imageView.image = [weakSelf drawPose:weakSelf.imageView image:image pose:poseResult];
        } else if (weakSelf.USE_MODEL == W_YOLACT) {
            weakSelf.imageView.image = [weakSelf drawYolactMask:weakSelf.imageView image:image dbface:yolactResult];
        } else if (weakSelf.USE_MODEL == W_FACE_LANDMARK) {
            weakSelf.imageView.image = [weakSelf drawFaceLandmark:weakSelf.imageView image:image faceLandmark:faceLandmarkResult];
        } else if (weakSelf.USE_MODEL == W_DBFACE) {
            weakSelf.imageView.image = [weakSelf drawDBFace:weakSelf.imageView image:image dbface:dbFaceResult];
        } else if (weakSelf.USE_MODEL == W_MOBILENETV2_FCN || weakSelf.USE_MODEL == W_MOBILENETV3_SEG) {
            // slow !!!
            weakSelf.imageView.image = [weakSelf drawSegMask:weakSelf.imageView image:image mask:maskMat];
        } else if (weakSelf.USE_MODEL == W_LIGHT_OPENPOSE) {
            weakSelf.imageView.image = [weakSelf drawLightOpenpose:weakSelf.imageView image:image humanPose:humanPose];
        }
        
        // include draw
        long dur = [[NSDate date] timeIntervalSince1970]*1000 - start.timeIntervalSince1970*1000;
        float fps = 1.0 / (dur / 1000.0);
        weakSelf.total_fps = (weakSelf.total_fps == 0) ? fps : (weakSelf.total_fps + fps);
        weakSelf.fps_count++;
        NSString *info = [NSString stringWithFormat:@"%@\nSize:%dx%d\nTime:%.3fs\nFPS:%.2f\nAVG_FPS:%.2f", [self getModelName], int(image.size.width), int(image.size.height), dur / 1000.0, fps, (float)weakSelf.total_fps / weakSelf.fps_count];
        weakSelf.resultLabel.text = info;
        
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
    } else if (!self.simplePose && self.USE_MODEL == W_SIMPLE_POSE) {
        NSLog(@"new Simple-Pose");
        self.simplePose = new SimplePose(self.USE_GPU);
    } else if (!self.yolact && self.USE_MODEL == W_YOLACT) {
        NSLog(@"new Yolact");
        self.yolact = new Yolact(self.USE_GPU);
    } else if (!self.faceLandmark && self.USE_MODEL == W_FACE_LANDMARK) {
        NSLog(@"new face-landmark");
        self.faceLandmark = new FaceLandmark(self.USE_GPU);
    } else if (!self.dbFace && self.USE_MODEL == W_DBFACE) {
        NSLog(@"new dbface");
        self.dbFace = new DBFace(self.USE_GPU);
    } else if (!self.mbnFCN && self.USE_MODEL == W_MOBILENETV2_FCN) {
        NSLog(@"new mbnv2 fcn");
        self.mbnFCN = new MbnFCN(self.USE_GPU);
    } else if (!self.mbnv3Seg && self.USE_MODEL == W_MOBILENETV3_SEG) {
        NSLog(@"new mbnv3 seg");
        self.mbnv3Seg = new MBNV3Seg(self.USE_GPU);
    } else if (!self.yolov5CustomOp && self.USE_MODEL == W_YOLOV5S_CUSTOM_OP) {
        NSLog(@"new yolov5s custom op");
        self.yolov5CustomOp = new YoloV5CustomLayer(self.USE_GPU);
    } else if (!self.nanoDet && self.USE_MODEL == W_NANODET) {
        NSLog(@"new nanodet");
        self.nanoDet = new NanoDet(self.USE_GPU);
    } else if (!self.yolov4 && self.USE_MODEL == W_YOLO_FASTEST_XL) {
        NSLog(@"new yolo fastest xl");
        self.yolov4 = new YoloV4(self.USE_GPU, 2);
    } else if (!self.lightOpenpose && self.USE_MODEL == W_LIGHT_OPENPOSE) {
        NSLog(@"new light openpose");
        self.lightOpenpose = new LightOpenPose(self.USE_GPU);
    }
}

- (void)releaseModel {
    NSLog(@"release model");
    delete self.yolo;
    delete self.yolov4;
    delete self.nanoDet;
    delete self.yolov5CustomOp;
    delete self.mbnv3Seg;
    delete self.mbnFCN;
    delete self.simplePose;
    delete self.faceLandmark;
    delete self.dbFace;
    delete self.yolact;
    delete self.lightOpenpose;
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
    } else if (self.USE_MODEL == W_LIGHT_OPENPOSE) {
        name = @"Light Openpose";
    }
    NSString *modelName = self.USE_GPU ? [NSString stringWithFormat:@"[GPU] %@", name] : [NSString stringWithFormat:@"[CPU] %@", name];
    return modelName;
}

#pragma mark 绘制目标检测结果
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
        } else if (self.USE_MODEL == W_YOLOV5S_CUSTOM_OP) {
            label = [NSString stringWithFormat:@"%s %.3f", self.yolov5CustomOp->labels[box.label].c_str(), box.score];
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

#pragma mark 绘制语义分割结果
- (UIImage *)drawSegMask:(UIImageView *)imageView image:(UIImage *)image mask:(ncnn::Mat)mask {
    // 0, "road" 1, "sidewalk" 2, "building" 3, "wall" 4, "fence" 5, "pole" 6, "traffic light" 7, "traffic sign" 8, "vegetation"
    // 9, "terrain" 10, "sky" 11, "person" 12, "rider" 13, "car" 14, "truck" 15, "bus" 16, "train" 17, "motorcycle" 18, "bicycle"
    int cityspace_colormap[19][3] = {
            {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156}, {190, 153, 153}, {153, 153, 153},
            {250, 170, 30}, {220, 220, 0}, {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
            {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32}
    };
    
    UIGraphicsBeginImageContext(image.size);

    [image drawAtPoint:CGPointMake(0,0)];

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetLineWidth(context, 1);
    
    int tempC = -1;
    int lengthW = 1;
    for (int h = 0; h < image.size.height; h++) {
        float *rowx = mask.row(h);
        for (int w = 0; w < image.size.width; w++) {
            int classes = rowx[w];
            if (tempC != classes) {
                float r = cityspace_colormap[tempC][0] / 255.0f;
                float g = cityspace_colormap[tempC][1] / 255.0f;
                float b = cityspace_colormap[tempC][2] / 255.0f;
                
                UIColor *color = [UIColor colorWithRed:r green:g blue:b alpha:0.5f];
                CGContextSetStrokeColorWithColor(context, [color CGColor]);
                // not draw point fun ???
                CGContextMoveToPoint(context, w - lengthW, h);
                CGContextAddLineToPoint(context, w, h);
                CGContextStrokePath(context);
                tempC = classes;
                lengthW = 1;
            } else {
                lengthW++;
            }
        }
        float r = cityspace_colormap[tempC][0] / 255.0f;
        float g = cityspace_colormap[tempC][1] / 255.0f;
        float b = cityspace_colormap[tempC][2] / 255.0f;
        
        UIColor *color = [UIColor colorWithRed:r green:g blue:b alpha:0.5f];
        CGContextSetStrokeColorWithColor(context, [color CGColor]);
        // not draw point fun ???
        CGContextMoveToPoint(context, image.size.width - lengthW, h);
        CGContextAddLineToPoint(context, image.size.width, h);
        CGContextStrokePath(context);
        tempC = -1;
        lengthW = 1;
    }
    
    // slow !!!
//    for (int h = 0; h < image.size.height; h++) {
//        float *rowx = mask.row(h);
//        for (int w = 0; w < image.size.width; w++) {
//            int classes = rowx[w];
//            float r = cityspace_colormap[classes][0] / 255.0f;
//            float g = cityspace_colormap[classes][1] / 255.0f;
//            float b = cityspace_colormap[classes][2] / 255.0f;
//
//            UIColor *color = [UIColor colorWithRed:r green:g blue:b alpha:0.5f];
//            CGContextSetStrokeColorWithColor(context, [color CGColor]);
//            // not draw point fun ???
//            CGContextMoveToPoint(context, w, h);
//            CGContextAddLineToPoint(context, w + 1, h);
//            CGContextStrokePath(context);
//        }
//    }
    
    //创建新图像
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();

    UIGraphicsEndImageContext();
    return newImage;
}

- (UIImage *)drawPose:(UIImageView *)imageView image:(UIImage *)image pose:(std::vector<PoseResult>)pose {
    UIGraphicsBeginImageContext(image.size);

    [image drawAtPoint:CGPointMake(0,0)];

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetLineWidth(context, fmax(image.size.width/200, 1));
        
    // draw bone
    // 0 nose, 1 left_eye, 2 right_eye, 3 left_Ear, 4 right_Ear, 5 left_Shoulder, 6 rigth_Shoulder, 7 left_Elbow, 8 right_Elbow,
    // 9 left_Wrist, 10 right_Wrist, 11 left_Hip, 12 right_Hip, 13 left_Knee, 14 right_Knee, 15 left_Ankle, 16 right_Ankle
    int joint_pairs[17][2] = {{0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};
    
    for (int i = 0; i < pose.size(); i++) {
        PoseResult posex = pose[i];
        srand(i + 2020);
        UIColor *color = [UIColor colorWithRed:rand()%256/255.0f green:rand()%256/255.0f blue:rand()%255/255.0f alpha:1.0f];
        CGContextSetStrokeColorWithColor(context, [color CGColor]);
        CGContextStrokePath(context);
        CGContextAddRect(context, CGRectMake(posex.boxInfos.x1, posex.boxInfos.y1, posex.boxInfos.x2 - posex.boxInfos.x1, posex.boxInfos.y2 - posex.boxInfos.y1));
        
        for (int j = 0; j < 16; j++) {
            int pl0 = joint_pairs[j][0];
            int pl1 = joint_pairs[j][1];
            // 人体左侧改为红线
            if ((pl0 % 2 == 1) && (pl1 % 2 == 1) && pl0 >= 5 && pl1 >= 5) {
                CGContextSetStrokeColorWithColor(context, UIColor.redColor.CGColor);
            } else {
                CGContextSetStrokeColorWithColor(context, [color CGColor]);
            }
            CGContextMoveToPoint(context, posex.keyPoints[joint_pairs[j][0]].p.x, posex.keyPoints[joint_pairs[j][0]].p.y);
            CGContextAddLineToPoint(context, posex.keyPoints[joint_pairs[j][1]].p.x, posex.keyPoints[joint_pairs[j][1]].p.y);
            CGContextStrokePath(context);
        }
        for (int j = 0; j < 17; j++) {
            CGContextFillRect(context, CGRectMake(posex.keyPoints[j].p.x - 2, posex.keyPoints[j].p.y - 2, 5, 5));
        }
    }
    
    //创建新图像
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();

    UIGraphicsEndImageContext();
    return newImage;
}


- (UIImage *)drawFaceLandmark:(UIImageView *)imageView image:(UIImage *)image faceLandmark:(std::vector<FaceKeyPoint>)faceLandmark {
    UIGraphicsBeginImageContext(image.size);

    [image drawAtPoint:CGPointMake(0,0)];

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetLineWidth(context, fmax(image.size.width/200, 1));
    
    for (int i = 0; i < faceLandmark.size(); i++) {
        FaceKeyPoint faceLandmarkx = faceLandmark[i];
        srand(i / 106 + 2020);
        UIColor *color = [UIColor colorWithRed:rand()%256/255.0f green:rand()%256/255.0f blue:rand()%255/255.0f alpha:1.0f];
        CGContextSetFillColorWithColor(context, [color CGColor]);
        CGContextFillRect(context, CGRectMake(faceLandmarkx.p.x - 2, faceLandmarkx.p.y - 2, 5, 5));
    }
    
    //创建新图像
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();

    UIGraphicsEndImageContext();
    return newImage;
}

- (UIImage *)drawDBFace:(UIImageView *)imageView image:(UIImage *)image dbface:(std::vector<Obj>)dbFace {
    UIGraphicsBeginImageContext(image.size);

    [image drawAtPoint:CGPointMake(0,0)];

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetLineWidth(context, fmax(image.size.width / 200, 1));
    
    for (int i = 0; i < dbFace.size(); i++) {
        Obj dbfacex = dbFace[i];
        srand(i + 2020);
        UIColor *color = [UIColor colorWithRed:rand()%256/255.0f green:rand()%256/255.0f blue:rand()%255/255.0f alpha:1.0f];
        CGContextSetStrokeColorWithColor(context, [color CGColor]);
        CGContextStrokePath(context);
        CGContextAddRect(context, CGRectMake(dbfacex.box.x, dbfacex.box.y, dbfacex.box.r - dbfacex.box.x, dbfacex.box.b - dbfacex.box.y));
        CGContextDrawPath(context, kCGPathStroke);
        
        CGContextSetFillColorWithColor(context, [color CGColor]);
        for (int j = 0; j < 5; j++) {
            CGContextFillRect(context, CGRectMake(dbfacex.landmark.x[j] - 2, dbfacex.landmark.y[j] - 2, 5, 5));
        }
    }
    
    //创建新图像
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();

    UIGraphicsEndImageContext();
    return newImage;
}

- (UIImage *)drawYolactMask:(UIImageView *)imageView image:(UIImage *)image dbface:(std::vector<Object>)yolactMask {
    UIGraphicsBeginImageContext(image.size);

    [image drawAtPoint:CGPointMake(0,0)];

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetLineWidth(context, fmax(image.size.width / 200, 1));
    NSMutableParagraphStyle *style = [[NSMutableParagraphStyle alloc] init];
    
    NSString *label = nil;
    for (int i = 0; i < yolactMask.size(); i++) {
        Object yolactx = yolactMask[i];
        if (yolactx.prob < 0.4f) {
            continue;
        }
        
        srand(i + 2020);
        UIColor *color = [UIColor colorWithRed:rand()%256/255.0f green:rand()%256/255.0f blue:rand()%255/255.0f alpha:1.0f];
        UIColor *acolor = [UIColor colorWithRed:rand()%256/255.0f green:rand()%256/255.0f blue:rand()%255/255.0f alpha:0.4f];
        CGContextSetLineWidth(context, fmax(image.size.width / 200, 1));
        CGContextSetStrokeColorWithColor(context, [color CGColor]);
        CGContextAddRect(context, CGRectMake(yolactx.rect.x, yolactx.rect.y, yolactx.rect.width, yolactx.rect.height));
        CGContextStrokePath(context);
        
        label = [NSString stringWithFormat:@"%s %.3f", self.yolact->labels[yolactx.label].c_str(), yolactx.prob];
        [label drawAtPoint:CGPointMake(yolactx.rect.x + 2, yolactx.rect.y - 25) withAttributes:@{NSFontAttributeName:[UIFont systemFontOfSize:20], NSParagraphStyleAttributeName:style, NSForegroundColorAttributeName:color}];
        
        CGContextSetStrokeColorWithColor(context, [acolor CGColor]);
        CGContextSetLineWidth(context, 1);
        int lengthW = 0;
        for (int h = 0; h < yolactx.mask.rows; h++) {
            const auto *pCowMask = yolactx.mask.ptr(h);
            for (int w = 0; w < yolactx.mask.cols; w++) {
                // slow
//                if (pCowMask[w] != 0) {
//                    CGContextMoveToPoint(context, w, h);
//                    CGContextAddLineToPoint(context, w + 1, h);
//                    CGContextStrokePath(context);
//                }
                // fast
                if (pCowMask[w] != 0) {
                    lengthW++;
                } else if (lengthW > 0) {
                    CGContextMoveToPoint(context, w - lengthW, h);
                    CGContextAddLineToPoint(context, w, h);
                    CGContextStrokePath(context);
                    lengthW = 0;
                }
            }
            // fast
            if (lengthW > 0) {
                CGContextMoveToPoint(context, yolactx.mask.cols - lengthW, h);
                CGContextAddLineToPoint(context, yolactx.mask.cols, h);
                CGContextStrokePath(context);
                lengthW = 0;
            }
        }
    }
    
    //创建新图像
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();

    UIGraphicsEndImageContext();
    return newImage;
}

- (UIImage *)drawLightOpenpose:(UIImageView *)imageView image:(UIImage *)image humanPose:(std::vector<human_pose_estimation::HumanPose>)humanPoses {
    int joint_pairs[19][2] = {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 16}, {5, 17}};
    int keyPointNumber = 18;
    int lineNumber = 17;
    
    UIGraphicsBeginImageContext(image.size);
    [image drawAtPoint:CGPointMake(0,0)];

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetLineWidth(context, fmax(image.size.width / 200, 1));
    
    for (int i = 0; i < humanPoses.size(); i++) {
        human_pose_estimation::HumanPose humanPose = humanPoses[i];
        srand(i + 2020);
        UIColor *color = [UIColor colorWithRed:rand()%256/255.0f green:rand()%256/255.0f blue:rand()%255/255.0f alpha:1.0f];
        CGContextSetStrokeColorWithColor(context, [color CGColor]);
        CGContextStrokePath(context);
        
        // draw line
        for (int j = 0; j < lineNumber; j++) {
            if (humanPose.keypoints[joint_pairs[j][0]].x == -1 || humanPose.keypoints[joint_pairs[j][1]].y == -1) {
                continue;
            }
            CGContextMoveToPoint(context, humanPose.keypoints[joint_pairs[j][0]].x, humanPose.keypoints[joint_pairs[j][0]].y);
            CGContextAddLineToPoint(context, humanPose.keypoints[joint_pairs[j][1]].x, humanPose.keypoints[joint_pairs[j][1]].y);
            CGContextStrokePath(context);
        }
        // draw point
        for (int j = 0; j < keyPointNumber; j++) {
            if (humanPose.keypoints[j].x == -1 || humanPose.keypoints[j].y == -1) {
                continue;
            }
            CGContextFillRect(context, CGRectMake(humanPose.keypoints[j].x - 2, humanPose.keypoints[j].y - 2, 5, 5));
        }
    }
    //创建新图像
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}


@end

