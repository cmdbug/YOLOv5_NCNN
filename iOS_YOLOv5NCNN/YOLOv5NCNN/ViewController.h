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
#define W_MOBILENETV2_YOLOV3_NANO 3
#define W_SIMPLE_POSE 4
#define W_YOLACT 5
#define W_ENET 6  // cancel
#define W_FACE_LANDMARK 7
#define W_DBFACE 8
#define W_MOBILENETV2_FCN 9
#define W_MOBILENETV3_SEG 10
#define W_YOLOV5S_CUSTOM_OP 11
#define W_NANODET 12
#define W_YOLO_FASTEST_XL 13
#define W_LIGHT_OPENPOSE 14

@interface ViewController : UIViewController

// 1:yolov5s 2:yolov4_tiny 3:mbnv2_yolov3_nano
// 4:simple-pose 5:yolact 6:enet 7:facelandmark
// 8:dbface 9:mbnv2-fcn 10:mbnv3-seg
@property (assign, nonatomic) int USE_MODEL;
@property (assign, nonatomic) bool USE_GPU;



@end

