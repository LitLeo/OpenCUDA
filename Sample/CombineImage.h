// CombineImage.h 
//
// 融合图像（Combine Image）
// 功能说明：将若干幅图像融合成一幅图像。要求这些图像的 ROI 子区域的尺寸完全相
// 同。

#ifndef __COMBINEIMAGE_H__
#define __COMBINEIMAGE_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：CombineImage
// 继承自：无
// 通过一些归约性质的计算将多幅图片组合成一幅图片。这些计算包括取最大值、取平
// 均值等。
class CombineImage
{
public:

    // Host 成员方法：combineImageMax（以最大值的方式合并图像）
    // 输出图像的对应坐标位置的像素值为各个输入图像对应位置的像素的最大值。
    __host__ int            // 返回值：错误码，如果处理无误则返回 NO_ERROR。
    combineImageMax(
            Image **inimg,  // 输入图像，多幅图像组成的数组 
            int imgcnt,     // 输入图像的数量
            Image *outimg   // 输出图像
    );

};

#endif

