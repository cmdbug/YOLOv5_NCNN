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

@end

@implementation WelcomeVC

- (void)viewDidLoad {
    [super viewDidLoad];
    [self initView];
    
    self.title = @"TENG";
}

- (void)initView {
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
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressYolov4tiny:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_YOLOV4TINY;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressMBNv2Yolov3Nano:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_MOBILENETV2_YOLOV3_NANO;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressSimplePose:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_SIMPLE_POSE;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressYolact:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_YOLACT;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressFaceLandmark:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_FACE_LANDMARK;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressDBFace:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_DBFACE;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressMBNv2FCN:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_MOBILENETV2_FCN;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressMBNv3SEG:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_MOBILENETV3_SEG;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressYOLOv5CustomOP:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_YOLOV5S_CUSTOM_OP;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressNanoDet:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_NANODET;
    [self.navigationController pushViewController:vc animated:NO];
}

- (void)pressYOLOFastestXL:(UIButton *)btn {
    ViewController *vc = [self.storyboard instantiateViewControllerWithIdentifier:@"ViewController"];
    vc.USE_MODEL = W_YOLO_FASTEST_XL;
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
