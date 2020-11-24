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

@end

@implementation WelcomeVC

- (void)viewDidLoad {
    [super viewDidLoad];
    [self initView];
    
    self.title = @"NCNN Demo";
}

- (void)initView {
    [_btnYolov5s addTarget:self action:@selector(pressYolov5s:) forControlEvents:UIControlEventTouchUpInside];
    [_btnYolov4tiny addTarget:self action:@selector(pressYolov4tiny:) forControlEvents:UIControlEventTouchUpInside];
    [_btnMBV2Yolov3nano addTarget:self action:@selector(pressMBNv2Yolov3Nano:) forControlEvents:UIControlEventTouchUpInside];
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

/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
