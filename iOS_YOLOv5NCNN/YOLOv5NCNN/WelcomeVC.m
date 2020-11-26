//
//  WelcomeVC.m
//  YOLOv5NCNN
//
//  Created by WZTENG on 2020/9/2.
//  Copyright © 2020 TENG. All rights reserved.
//

#import "WelcomeVC.h"
#import "ViewController.h"

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

@property (strong, nonatomic) IBOutlet UIImageView *btnUseGPU;
@property (assign, nonatomic) Boolean useGPU;

@end

@implementation WelcomeVC

- (void)viewDidLoad {
    [super viewDidLoad];
    [self initView];
    
    self.title = @"TENG";
}

- (void)changeMode {
    self.useGPU = NO;
    self.btnUseGPU.userInteractionEnabled = YES;
    UITapGestureRecognizer *modeTap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(changeNcnnMode)];
    [self.btnUseGPU addGestureRecognizer:modeTap];
}

- (void)changeNcnnMode {
    self.useGPU = self.useGPU ? NO : YES;
    NSString *title = @"Warning";
    NSString *message = @"ohhhhh";
    if (self.useGPU) {
        title = @"Warning";
        message = @"If the GPU is too old, it may not work well in GPU mode.";
    } else {
        title = @"Warning";
        message = @"Run on CPU.";
    }
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:title message:message preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *sure = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil];
    [alert addAction:sure];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)initView {
    [self changeMode];
    [_btnYolov5s addTarget:self action:@selector(pressYolov5s:) forControlEvents:UIControlEventTouchUpInside];
    [_btnYolov4tiny addTarget:self action:@selector(pressYolov4tiny:) forControlEvents:UIControlEventTouchUpInside];
    [_btnMBV2Yolov3nano addTarget:self action:@selector(pressMBNv2Yolov3Nano:) forControlEvents:UIControlEventTouchUpInside];
    [_btnSimplePose addTarget:self action:@selector(pressSimplePose:) forControlEvents:UIControlEventTouchUpInside];
    [_btnYolact addTarget:self action:@selector(pressYolact:) forControlEvents:UIControlEventTouchUpInside];
    [_btnFaceLandmark addTarget:self action:@selector(pressFaceLandmark:) forControlEvents:UIControlEventTouchUpInside];
    [_btnDBFace addTarget:self action:@selector(pressDBFace:) forControlEvents:UIControlEventTouchUpInside];
    [_btnMobilenetv2FCN addTarget:self action:@selector(pressMBNv2FCN:) forControlEvents:UIControlEventTouchUpInside];
    [_btnmobilenetv3Seg addTarget:self action:@selector(pressMBNv3SEG:) forControlEvents:UIControlEventTouchUpInside];
    [_btnYOLOv5sCustomLayer addTarget:self action:@selector(pressYOLOv5CustomOP:) forControlEvents:UIControlEventTouchUpInside];
    [_btnNanoDet addTarget:self action:@selector(pressNanoDet:) forControlEvents:UIControlEventTouchUpInside];
    [_btnYOLOFastestXL addTarget:self action:@selector(pressYOLOFastestXL:) forControlEvents:UIControlEventTouchUpInside];
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

/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
