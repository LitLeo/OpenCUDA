// ImageDiff.h 
//
// 图像做差（Difference of two Image）
// 功能说明：根据输入的两幅灰度图像，对其相应为位置的像素值做差得到差值图像。


#ifndef __IMAGEDIFF_H__ 
#define __IMAGEDIFF_H__ 

#include "Image.h"
 
// 类:ImageDiff(图像做差)
// 继承自：无
// 根据输入的两幅灰度图像，对其相应为位置的像素值做差得到差值图像 outimg
class ImageDiff {

public:  
    // 构造函数：ImageDiff
    // 无参数版本的构造函数.
    __host__ __device__
    ImageDiff() {
    }

    // 成员方法:imageDiff:
    // 对输入的 inimg1 和 inimg 做差，得到插值图像 outimg
    __host__ int            // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR。
    imageDiff(
            Image *inimg1,  // 相减前的原图像 1
            Image *inimg2,  // 相减前的原图像 2
            Image *outimg   // 相减后生成的差值图像
    );
};

#endif

