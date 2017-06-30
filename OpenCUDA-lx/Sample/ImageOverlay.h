// ImageOverlay.h
//
// 图像叠加（ImageOverlay）
// 功能说明：将 n 幅输入图像叠加到一起，输出图像的灰度值等于所有输入图像对应点灰
//           度值乘以相应权重值并求和。

#ifndef __IMAGEOVERLAY_H__
#define __IMAGEOVERLAY_H__

#include "Image.h"
#include "RoiCopy.h"
#include "ErrorCode.h"

// 类：ImageOverlay
// 继承自：无
// 将 n 幅输入图像的像素灰度值逐点与相应权重相乘，权重值范围为[0, 1]且总和为 1，
// 然后求和，结果去尾为整数，且范围为[0, 255]，得到图像 outimg。
class ImageOverlay {

protected:

    // 成员变量：weight（权重）
    // 存储输入图像对应权重，有效时指向 device 端。
    float *weight;

    // 成员变量：weightLength（权重数组大小）
    // 存储权重数量。
    int weightLength;

    // 成员变量：roiCopy（子图像拷贝器）
    // 运用其将输入图像的子图像直接拷贝到输出中。
    RoiCopy roiCopy;

public:

    // 构造函数：ImageOverlay
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    ImageOverlay()
    {
        // 将权重数组设为空，长度为 0。
        weight = NULL;
        weightLength = 0;
    }

    // 析构函数：~ImageOverlay
    // 析构函数，释放 weight 申请的内存空间。
    // 内容。
    __host__ __device__
    ~ImageOverlay()
    {
        // 如果 weight 权重数组不为空，则释放其内存。
        if (this->weight != NULL)
            cudaFree(this->weight);
    }
 
    // Host 成员方法：setValue（设置权重值）
    // 将 weight 设置到权重数组 weight 处。
    __host__ int                      // 返回值：函数是否正确执行，若函数正
                                      // 确执行，返回 NO_ERROR。
    setValue(
            float *weight,            // 新的权重数组。
            int weightLength,         // 新的权重数组大小。
            bool onhostweight = true  // 判断 weight 是否是 Host 内存的指针，
                                      // 默认为“true”。
    );

    // Host 成员方法：imageOverlay（图像叠加）
    // 将 n 幅输入图像的像素灰度值逐点与相应权重相乘，权重值范围[0, 1]总和为 1，
    // 然后求和，结果去尾为整数，且范围为[0, 255]，
    // 得到图像 outimg。
    __host__ int             // 返回值：函数是否正确执行，若函数正
                             // 确执行，返回 NO_ERROR。
    imageOverlay(
            Image *inimg[],  // 输入图像集合。
            int n,           // 输入图像个数。
            Image *outimg    // 输出图像。
    );
};

#endif
