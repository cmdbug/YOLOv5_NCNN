#include "YoloV5CustomLayer.h"
#include "ncnn/layer.h"


/**

ncnn 与 opencv 库冲突，需要重新编译 ncnn 或去掉 opencv 。（该项目保留 opencv ，如果需要两者都支持请看下面说明）

由于 ncnn 默认编译配置跟 opencv 官方默认编译配置冲突，先关闭 ncnn 自定义层功能，有需要的话可以自行编译 ncnn 库
编译 ncnn 时需要修改下编译配置，编译完成后在该项目 CMakeLists.txt 中使用以下参数开启关闭配置（也可以不改，直接默认）

opencv 官方默认库是开启的，ncnn 官方编译的库默认是关闭的。所以 CMakelista.txt 加上下面这句时 opencv 会报错。
# disable rtti and exceptions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

opencv 官方默认库是开启的，ncnn 官方编译的库默认是关闭的。所以 CMakelista.txt 加上下面这句时 ncnn 会报错。
# enable rtti
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

即：如果选择重新编译 ncnn 库，需要开启 rtti 选项，即：-DNCNN_DISABLE_RTTI=OFF -DNCNN_DISABLE_EXCEPTION=OFF
(没编译过 opencv，不懂)，自行编译完成后，记得把下面的：ENABLE_CUSTOM_LAYER 改为 1 开启

**/


#if ENABLE_CUSTOM_LAYER

class YoloV5Focus : public ncnn::Layer {
public:
    YoloV5Focus();

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const;
};

DEFINE_LAYER_CREATOR(YoloV5Focus)

YoloV5Focus::YoloV5Focus() {
    one_blob_only = true;
}

int YoloV5Focus::forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = w / 2;
    int outh = h / 2;
    int outc = channels * 4;

    top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++) {
        const float *ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
        float *outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++) {
            for (int j = 0; j < outw; j++) {
                *outptr = *ptr;

                outptr += 1;
                ptr += 2;
            }

            ptr += w;
        }
    }

    return 0;
}

#endif  // custom layer

// ===================================================================================================

inline float intersection_area(const YoloObject &a, const YoloObject &b) {
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y) {
        // no intersection
        return 0.f;
    }
    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

void qsort_descent_inplace(std::vector<YoloObject> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<YoloObject> &faceobjects) {
    if (faceobjects.empty()) {
        return;
    }
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<YoloObject> &faceobjects, std::vector<int> &picked, float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++) {
        const YoloObject &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const YoloObject &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold) {
                keep = 0;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }
}

//inline float fast_exp(float x) {
//    union {
//        uint32_t i;
//        float f;
//    } v{};
//    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
//    return v.f;
//}
//
//inline float sigmoid(float x) {
//    return 1.0f / (1.0f + fast_exp(-x));
//}

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void generate_proposals(const ncnn::Mat &anchors, int stride,
                        const ncnn::Mat &in_pad, const ncnn::Mat &feat_blob,
                        float prob_threshold, std::vector<YoloObject> &objects) {
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                const float *featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++) {
                    float score = featptr[5 + k];
                    if (score > class_score) {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold) {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    YoloObject obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}


bool YoloV5CustomLayer::hasGPU = true;
bool YoloV5CustomLayer::toUseGPU = true;
YoloV5CustomLayer *YoloV5CustomLayer::detector = nullptr;

YoloV5CustomLayer::YoloV5CustomLayer(AAssetManager *mgr, const char *param, const char *bin, bool useGPU) {
    hasGPU = ncnn::get_gpu_count() > 0;
    toUseGPU = hasGPU && useGPU;

    Net = new ncnn::Net();
    // opt 需要在加载前设置
    Net->opt.use_vulkan_compute = toUseGPU;  // gpu
    Net->opt.use_fp16_arithmetic = true;  // fp16运算加速
#if ENABLE_CUSTOM_LAYER
    // 注册自定义层
    Net->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
#endif
    Net->load_param(mgr, param);
    Net->load_model(mgr, bin);
}

YoloV5CustomLayer::~YoloV5CustomLayer() {
    Net->clear();
    delete Net;
}

std::vector<BoxInfo> YoloV5CustomLayer::detect(JNIEnv *env, jobject image, float threshold, float nms_threshold) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);
//    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env,image,ncnn::Mat::PIXEL_BGR2RGB,input_size/2,input_size/2);

    // letterbox pad to multiple of 32
    int w = img_size.width;
    int h = img_size.height;
    float scale = 1.f;
    if (w > h) {
        scale = (float) input_size / w;
        w = input_size;
        h = h * scale;
    } else {
        scale = (float) input_size / h;
        h = input_size;
        w = w * scale;
    }
    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB, w, h);
    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in_net, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT,
                           114.f);

    float mean[3] = {0, 0, 0};
    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(mean, norm);
    auto ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
    ex.input("images", in_pad);

    std::vector<YoloObject> proposals;
    // anchor setting from yolov5/models/yolov5s.yaml

    for (const auto &layer: layers) {
        ncnn::Mat blob;
        ex.extract(layer.name.c_str(), blob);
        ncnn::Mat anchors(6);
        anchors[0] = layer.anchors[0].width;
        anchors[1] = layer.anchors[0].height;
        anchors[2] = layer.anchors[1].width;
        anchors[3] = layer.anchors[1].height;
        anchors[4] = layer.anchors[2].width;
        anchors[5] = layer.anchors[2].height;
        std::vector<YoloObject> objectsx;
        generate_proposals(anchors, layer.stride, in_pad, blob, threshold, objectsx);

        proposals.insert(proposals.end(), objectsx.begin(), objectsx.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    std::vector<YoloObject> objects;
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].x - (wpad / 2)) / scale;
        float y0 = (objects[i].y - (hpad / 2)) / scale;
        float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
        float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_size.width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_size.height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_size.width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_size.height - 1)), 0.f);

        objects[i].x = x0;
        objects[i].y = y0;
        objects[i].w = x1 - x0;
        objects[i].h = y1 - y0;
    }

    std::vector<BoxInfo> result;
    for (int i = 0; i < count; i++) {
        BoxInfo box;
        box.x1 = objects[i].x;
        box.y1 = objects[i].y;
        box.x2 = objects[i].x + objects[i].w;
        box.y2 = objects[i].y + objects[i].h;
        box.label = objects[i].label;
        box.score = objects[i].prob;
        result.push_back(box);
    }
    return result;

}

