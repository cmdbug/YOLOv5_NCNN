//
//  ViewController.h
//  YOLOv5NCNN
//
//  Created by WZTENG on 2020/7/5.
//  Copyright © 2020 TENG. All rights reserved.
//

#import <UIKit/UIKit.h>

#define W_YOLOV5S 1
#define W_YOLOV4TINY 2

@interface ViewController : UIViewController

// 1:yolov5s 2:yolov4_tiny 3:mbnv2_yolov3_nano
// 4:simple-pose 5:yolact 6:enet 7:facelandmark
@property (assign, nonatomic) int USE_MODEL;



@end

