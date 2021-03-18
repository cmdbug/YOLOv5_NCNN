//
//  OCRViewController.m
//  YOLOv5NCNN
//
//  Created by WZTENG on 2021/3/17.
//  Copyright © 2021 TENG. All rights reserved.
//

#import "OCRViewController.h"
#include "ocr.h"
#include <numeric>

@interface OCRViewController () <UINavigationControllerDelegate, UIImagePickerControllerDelegate, UIGestureRecognizerDelegate>

@property (nonatomic, strong) UIImageView *resultImageView;
@property (nonatomic, strong) UIImageView *srcImageView;
@property (nonatomic, strong) UILabel *textLabel;
@property (nonatomic, strong) UIButton *photoBtn;

@property (nonatomic) dispatch_queue_t queue;
@property (assign, atomic) Boolean isDetecting;
@property (assign, atomic) Boolean isPhoto;

@property OCR *ocr;

@end

@implementation OCRViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.title = @"Chinese OCR Lite";
    
    [self initView];
    [self createModel];
    
    self.queue = dispatch_queue_create("ncnnocr", DISPATCH_QUEUE_CONCURRENT);
}

- (void)initView {
    float statusHeight = [UIApplication sharedApplication].statusBarFrame.size.height;
    float navBarHeight = self.navigationController.navigationBar.frame.size.height;
    
    UIEdgeInsets safeAreaInsets = UIEdgeInsetsZero;
    if (@available(iOS 11.0, *)) {
        safeAreaInsets = [[[[UIApplication sharedApplication] delegate] window] safeAreaInsets];
    } else {
        safeAreaInsets.top = 20;
    }
    
    self.view.backgroundColor = UIColor.lightGrayColor;
    int width = self.view.bounds.size.width;
    int height = self.view.bounds.size.height;
    
    _resultImageView = [[UIImageView alloc] initWithFrame:CGRectMake(safeAreaInsets.left + 3,
                                                                     safeAreaInsets.top + navBarHeight + 3,
                                                                     width - 6,
                                                                     height - 6 - 150 - statusHeight - navBarHeight)];
    [self addGestureRecognizerToView:_resultImageView];
    [_resultImageView setContentMode:UIViewContentModeScaleAspectFit];
    [_resultImageView setUserInteractionEnabled:YES];
    [_resultImageView setMultipleTouchEnabled:YES];
    _resultImageView.backgroundColor = UIColor.lightTextColor;
    [_resultImageView setImage:[UIImage imageNamed:@"ncnn_icon"]];
    _srcImageView = [[UIImageView alloc] initWithFrame:CGRectMake(3, height - 150, 110, 150 - 3)];
    [_srcImageView setContentMode:UIViewContentModeScaleAspectFit];
    _srcImageView.backgroundColor = UIColor.lightTextColor;
    _textLabel =[[UILabel alloc] initWithFrame:CGRectMake(110 + 6, height - 150, width - 110 - 6 - 3, 150 - 6 - 30)];
    _textLabel.backgroundColor = UIColor.lightTextColor;
    [_textLabel setNumberOfLines:0];
    [_textLabel setFont:[UIFont systemFontOfSize:14]];
    _photoBtn = [[UIButton alloc] initWithFrame:CGRectMake(110 + 6, height - 30 - 3, width - 110 - 6 - 3, 30)];
    [_photoBtn setTitle:@"Photo" forState:UIControlStateNormal];
    _photoBtn.backgroundColor = UIColor.blueColor;
    [self.view addSubview:_resultImageView];
    [self.view addSubview:_srcImageView];
    [self.view addSubview:_textLabel];
    [self.view addSubview:_photoBtn];
    [_photoBtn addTarget:self action:@selector(predict:) forControlEvents:UIControlEventTouchUpInside];
    self.textLabel.text = @"Please select a pciture";
}

#pragma mark 创建模型
- (void)createModel {
    if (!self.ocr) {
        NSLog(@"new ocr");
        self.ocr = new OCR(self.USE_GPU);
    }
}

- (void)releaseModel {
    NSLog(@"release model");
    delete self.ocr;
}

#pragma mark 绘制检测结果
- (UIImage *)drawBoxAndText:(UIImageView *)imageView image:(UIImage *)image ocrResults:(std::vector<OCRResult>)ocrs {
    UIGraphicsBeginImageContext(image.size);

    [image drawAtPoint:CGPointMake(0,0)];

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetLineWidth(context, fmax(image.size.width/200, 1));
    NSMutableParagraphStyle *style = [[NSMutableParagraphStyle alloc] init];
    for (int i = 0; i < ocrs.size(); i++) {
        OCRResult ocr = ocrs[i];
        UIColor *color = [UIColor colorWithRed:rand()%256/255.0f green:rand()%256/255.0f blue:rand()%255/255.0f alpha:1.0f];
        int x1 = ocr.boxes[0].x;
        int y1 = ocr.boxes[0].y;
        int x2 = ocr.boxes[1].x;
        int y2 = ocr.boxes[1].y;
        int x3 = ocr.boxes[2].x;
        int y3 = ocr.boxes[2].y;
        int x4 = ocr.boxes[3].x;
        int y4 = ocr.boxes[3].y;
        CGContextMoveToPoint(context, x1, y1);
        CGContextAddLineToPoint(context, x2, y2);
        CGContextAddLineToPoint(context, x3, y3);
        CGContextAddLineToPoint(context, x4, y4);
        CGContextAddLineToPoint(context, x1, y1);
        CGContextStrokePath(context);
        
        std::string txt;
        txt = accumulate(ocr.pre_res.begin(), ocr.pre_res.end(), txt);
        
        NSString *temp = [NSString stringWithCString:txt.c_str() encoding:NSUTF8StringEncoding];
        temp = [temp stringByReplacingOccurrencesOfString:@"\r" withString:@""];
        NSString *text = [NSString stringWithFormat:@"%@ %.3f", temp, ocr.box_score];
        float fontSize = MAX(20, image.size.width / 40.0f);
        [text drawAtPoint:CGPointMake(x1, y1 - fontSize - 5) withAttributes:@{NSFontAttributeName:[UIFont systemFontOfSize:fontSize], NSParagraphStyleAttributeName:style, NSForegroundColorAttributeName:color}];
        
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

- (void)ocrDetectImage:(UIImage *)image {
    [self createModel];
    
    __weak typeof(self) weakSelf = self;
    dispatch_sync(dispatch_get_main_queue(), ^{
        weakSelf.textLabel.text = @"Please wait...";
    });
    
    std::vector<OCRResult> results;
    results = self.ocr->detect(image, 1080);
    NSLog(@"ocr result:%lu", results.size());
    
    dispatch_sync(dispatch_get_main_queue(), ^{
        weakSelf.textLabel.text = [NSString stringWithFormat:@"[%@] %@\nSize:%.0fx%.0f\nResults:%lu",
                                   weakSelf.USE_GPU ? @"GPU" : @"CPU", @"Chinese_OCR_lite",
                                   image.size.width, image.size.height, results.size()];
        weakSelf.resultImageView.image = [weakSelf drawBoxAndText:weakSelf.resultImageView image:image ocrResults:results];
    });
}

#pragma mark 照片
- (void)predict:(id)sender {
    if (self.isDetecting) {
        return;
    }
    self.isPhoto = YES;
    UIImagePickerController *picVC = [[UIImagePickerController alloc] init];
    picVC.delegate = self;
    picVC.allowsEditing = NO;
    [self presentViewController:picVC animated:YES completion:nil];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<UIImagePickerControllerInfoKey,id> *)info {
    NSLog(@"did pick image");
    UIImage *image = info[@"UIImagePickerControllerOriginalImage"];
    self.srcImageView.image = image;
    
    __weak typeof(self) weakSelf = self;
    dispatch_async(self.queue, ^{
        weakSelf.isDetecting = YES;
        [weakSelf ocrDetectImage:image];
        weakSelf.isDetecting = NO;
    });
    [self dismissViewControllerAnimated:YES completion:^{
        weakSelf.isPhoto = NO;
    }];
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
    NSLog(@"did cancel image");
    __weak typeof(self) weakSelf = self;
    [self dismissViewControllerAnimated:YES completion:^{
        weakSelf.isPhoto = NO;
    }];
}

- (void)viewDidDisappear:(BOOL)animated {
    [super viewDidDisappear:animated];
    if (!self.isPhoto) {
        dispatch_barrier_async(self.queue, ^{
            [self releaseModel];
        });
    }
}

// 添加所有的手势
- (void)addGestureRecognizerToView:(UIView *)view {
    // 旋转手势
    UIRotationGestureRecognizer *rotationGestureRecognizer = [[UIRotationGestureRecognizer alloc] initWithTarget:self action:@selector(rotateView:)];
    [view addGestureRecognizer:rotationGestureRecognizer];
    
    // 缩放手势
    UIPinchGestureRecognizer *pinchGestureRecognizer = [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(pinchView:)];
    [view addGestureRecognizer:pinchGestureRecognizer];
    
    // 移动手势
    UIPanGestureRecognizer *panGestureRecognizer = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(panView:)];
    [view addGestureRecognizer:panGestureRecognizer];
}

// 处理旋转手势
- (void)rotateView:(UIRotationGestureRecognizer *)rotationGestureRecognizer {
    UIView *view = rotationGestureRecognizer.view;
    if (rotationGestureRecognizer.state == UIGestureRecognizerStateBegan || rotationGestureRecognizer.state == UIGestureRecognizerStateChanged) {
        view.transform = CGAffineTransformRotate(view.transform, rotationGestureRecognizer.rotation);
        [rotationGestureRecognizer setRotation:0];
    }
}

// 处理缩放手势
- (void)pinchView:(UIPinchGestureRecognizer *)pinchGestureRecognizer {
    UIView *view = pinchGestureRecognizer.view;
    if (pinchGestureRecognizer.state == UIGestureRecognizerStateBegan || pinchGestureRecognizer.state == UIGestureRecognizerStateChanged) {
        view.transform = CGAffineTransformScale(view.transform, pinchGestureRecognizer.scale, pinchGestureRecognizer.scale);
        pinchGestureRecognizer.scale = 1;
    }
}

// 处理拖拉手势
- (void)panView:(UIPanGestureRecognizer *)panGestureRecognizer {
    UIView *view = panGestureRecognizer.view;
    if (panGestureRecognizer.state == UIGestureRecognizerStateBegan || panGestureRecognizer.state == UIGestureRecognizerStateChanged) {
        CGPoint translation = [panGestureRecognizer translationInView:view.superview];
        [view setCenter:(CGPoint){view.center.x + translation.x, view.center.y + translation.y}];
        [panGestureRecognizer setTranslation:CGPointZero inView:view.superview];
    }
}

@end
