//
//  WelcomeVC.m
//  YOLOv5NCNN
//
//  Created by WZTENG on 2020/9/2.
//  Copyright © 2020 TENG. All rights reserved.
//

#include "ncnn/ncnn/platform.h"
#import "WelcomeVC.h"
#import "ViewController.h"
#import "OCRViewController.h"

@interface WelcomeVC ()

@property (strong, nonatomic) IBOutlet UIButton *btnYolov5s;
@property (strong, nonatomic) IBOutlet UIButton *btnYolov4tiny;
@property (strong, nonatomic) IBOutlet UIButton *btnMBV2Yolov3nano;
@property (strong, nonatomic) IBOutlet UIButton *btnSimplePose;
@property (strong, nonatomic) IBOutlet UIButton *btnYolact;
@property (strong, nonatomic) IBOutlet UIButton *btnChineseOCRLite;
@property (strong, nonatomic) IBOutlet UIButton *btnFaceLandmark;
@property (strong, nonatomic) IBOutlet UIButton *btnDBFace;
@property (strong, nonatomic) IBOutlet UIButton *btnMobilenetv2FCN;
@property (strong, nonatomic) IBOutlet UIButton *btnmobilenetv3Seg;
@property (strong, nonatomic) IBOutlet UIButton *btnYOLOv5sCustomLayer;
@property (strong, nonatomic) IBOutlet UIButton *btnNanoDet;
@property (strong, nonatomic) IBOutlet UIButton *btnYOLOFastestXL;
@property (strong, nonatomic) IBOutlet UIButton *btnLightOpenPose;


@property (strong, nonatomic) UIScrollView *scrollView;
@property (strong, nonatomic) UIView *boxView;

@property (strong, nonatomic) UIButton *btnRight;

@property (assign, nonatomic) Boolean useGPU;

@end

@implementation WelcomeVC

- (void)viewDidLoad {
    [super viewDidLoad];
    [self initView];
    
    self.title = @"WZTENG";
}

- (void)changeMode {
    self.useGPU = NO;

    self.btnRight = [UIButton buttonWithType:UIButtonTypeCustom];
    self.btnRight.imageView.contentMode = UIViewContentModeScaleAspectFit;
    [self.btnRight setImage:[UIImage imageNamed:@"mode_cpu"] forState:UIControlStateNormal];
    [self.btnRight addTarget:self action:@selector(changeNcnnMode) forControlEvents:UIControlEventTouchUpInside];
    [self.btnRight.widthAnchor constraintEqualToConstant:45].active = YES;
    [self.btnRight.heightAnchor constraintEqualToConstant:30].active = YES;
    UIBarButtonItem *barRight = [[UIBarButtonItem alloc] initWithCustomView:self.btnRight];
    self.navigationItem.rightBarButtonItem = barRight;
}

- (void)changeNcnnMode {
    self.useGPU = self.useGPU ? NO : YES;
    NSString *title = @"Warning";
    NSString *message = @"ohhhhh";
    if (self.useGPU) {
        [self.btnRight setImage:[UIImage imageNamed:@"mode_gpu"] forState:UIControlStateNormal];
#if NCNN_VULKAN  // #include "ncnn/ncnn/platform.h"
        title = @"Warning";
        message = @"If the GPU is too old, it may not work well in GPU mode.";
#else
        title = @"Warning";
        message = @"You should download ncnn vulkan version.(see github ncnn wiki)";
#endif
    } else {
        [self.btnRight setImage:[UIImage imageNamed:@"mode_cpu"] forState:UIControlStateNormal];
        title = @"Warning";
        message = @"Run on CPU mode.";
    }
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:title message:message preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *sure = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil];
    [alert addAction:sure];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)initView {
    [self changeMode];
    
    int btnWidth = self.view.bounds.size.width;
    int offsetY = self.view.bounds.size.width * 0.6f;
    int offsetBottom = 30;
    int btnHeight = 35;
    int btnY = 35;
    int btnCount = 14;
    int i = 0;
    
    self.boxView = [[UIView alloc] initWithFrame:CGRectMake(self.view.bounds.origin.x, self.view.bounds.origin.y, self.view.bounds.size.width, offsetY + btnHeight * btnCount + offsetBottom)];
//    self.boxView.backgroundColor = [UIColor redColor];

    UIImageView *tipImageView = [[UIImageView alloc] initWithFrame:CGRectMake(0, 0, btnWidth, offsetY)];
    tipImageView.image = [UIImage imageNamed:@"ohhh"];
    tipImageView.contentMode = UIViewContentModeScaleToFill;
    [self.boxView addSubview:tipImageView];
    
    _btnYolov5s = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnYolov5s setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnYolov5s setTitle:@"YOLOv5s" forState:UIControlStateNormal];
    [_btnYolov5s addTarget:self action:@selector(pressYolov5s:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnYolov5s];
    
    _btnYolov4tiny = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnYolov4tiny setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnYolov4tiny setTitle:@"YOLOv4-tiny" forState:UIControlStateNormal];
    [_btnYolov4tiny addTarget:self action:@selector(pressYolov4tiny:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnYolov4tiny];
    
    _btnMBV2Yolov3nano = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnMBV2Yolov3nano setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnMBV2Yolov3nano setTitle:@"MBNv2-YOLOv3-nano" forState:UIControlStateNormal];
    [_btnMBV2Yolov3nano addTarget:self action:@selector(pressMBNv2Yolov3Nano:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnMBV2Yolov3nano];
    
    _btnSimplePose = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnSimplePose setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnSimplePose setTitle:@"Simple-Pose" forState:UIControlStateNormal];
    [_btnSimplePose addTarget:self action:@selector(pressSimplePose:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnSimplePose];
    
    _btnYolact = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnYolact setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnYolact setTitle:@"Yolact" forState:UIControlStateNormal];
    [_btnYolact addTarget:self action:@selector(pressYolact:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnYolact];
    
    _btnChineseOCRLite = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnChineseOCRLite setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnChineseOCRLite setTitle:@"Chinese OCR lite [Beta]" forState:UIControlStateNormal];
    [_btnChineseOCRLite addTarget:self action:@selector(pressChineseOCRLite:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnChineseOCRLite];
    
    _btnFaceLandmark = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnFaceLandmark setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnFaceLandmark setTitle:@"YoloFace500k-landmark106" forState:UIControlStateNormal];
    [_btnFaceLandmark addTarget:self action:@selector(pressFaceLandmark:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnFaceLandmark];
    
    _btnDBFace = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnDBFace setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnDBFace setTitle:@"DBFace" forState:UIControlStateNormal];
    [_btnDBFace addTarget:self action:@selector(pressDBFace:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnDBFace];
    
    _btnMobilenetv2FCN = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnMobilenetv2FCN setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnMobilenetv2FCN setTitle:@"MobileNetV2-FCN" forState:UIControlStateNormal];
    [_btnMobilenetv2FCN addTarget:self action:@selector(pressMBNv2FCN:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnMobilenetv2FCN];
    
    _btnmobilenetv3Seg = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnmobilenetv3Seg setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnmobilenetv3Seg setTitle:@"MBNv3-Segmentation-small" forState:UIControlStateNormal];
    [_btnmobilenetv3Seg addTarget:self action:@selector(pressMBNv3SEG:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnmobilenetv3Seg];
    
    _btnYOLOv5sCustomLayer = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnYOLOv5sCustomLayer setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnYOLOv5sCustomLayer setTitle:@"YOLOv5s-Custom-Layer (请看.h说明)" forState:UIControlStateNormal];
    [_btnYOLOv5sCustomLayer addTarget:self action:@selector(pressYOLOv5CustomOP:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnYOLOv5sCustomLayer];
    
    _btnNanoDet = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnNanoDet setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnNanoDet setTitle:@"NanoDet" forState:UIControlStateNormal];
    [_btnNanoDet addTarget:self action:@selector(pressNanoDet:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnNanoDet];
    
    _btnYOLOFastestXL = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnYOLOFastestXL setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnYOLOFastestXL setTitle:@"YOLO-Fastest-xl" forState:UIControlStateNormal];
    [_btnYOLOFastestXL addTarget:self action:@selector(pressYOLOFastestXL:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnYOLOFastestXL];
    
    _btnLightOpenPose = [[UIButton alloc] initWithFrame:CGRectMake(0, offsetY + btnY * i++, btnWidth, btnHeight)];
    [_btnLightOpenPose setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
    [_btnLightOpenPose setTitle:@"Light Openpose" forState:UIControlStateNormal];
    [_btnLightOpenPose addTarget:self action:@selector(pressLightOpenpose:) forControlEvents:UIControlEventTouchUpInside];
    [self.boxView addSubview:_btnLightOpenPose];
    
    self.scrollView = [[UIScrollView alloc] initWithFrame:self.view.bounds];
    self.scrollView.contentSize = self.boxView.frame.size;
    [self.scrollView addSubview:self.boxView];
    [self.view addSubview:self.scrollView];
    
}

- (void)pressYolov5s:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_YOLOV5S;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressYolov4tiny:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_YOLOV4TINY;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressMBNv2Yolov3Nano:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_MOBILENETV2_YOLOV3_NANO;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressSimplePose:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_SIMPLE_POSE;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressYolact:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_YOLACT;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressChineseOCRLite:(UIButton *)btn {
    OCRViewController *ocr = [[OCRViewController alloc] init];
    ocr.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:ocr animated:NO];
}

- (void)pressFaceLandmark:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_FACE_LANDMARK;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressDBFace:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_DBFACE;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressMBNv2FCN:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_MOBILENETV2_FCN;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressMBNv3SEG:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_MOBILENETV3_SEG;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressYOLOv5CustomOP:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_YOLOV5S_CUSTOM_OP;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressNanoDet:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_NANODET;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressYOLOFastestXL:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_YOLO_FASTEST_XL;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressLightOpenpose:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_LIGHT_OPENPOSE;
    vc.USE_GPU = self.useGPU;
    [self.navigationController pushViewController:vc animated:NO];
}

/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
