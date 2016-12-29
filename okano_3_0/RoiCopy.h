// RoiCopy.h
// 创建人：罗劼
//
// 拷贝子图（copy roi）
// 功能说明：将图像的 ROI 子图拷贝出来
//
// 修订历史：
// 2012年12月05日（罗劼）
//     初始版本

#ifndef __ROICOPY_H__
#define __ROICOPY_H__

#include "Image.h"

// 类：RoiCopy（拷贝图片的 ROI 子图）
// 继承自：无
// 拷贝图片的 ROI 子图到一张新的图中
class RoiCopy {

public:

    // 构造函数：RoiCopy
    // 无参构造函数，此函数不做任何操作
    __host__ __device__
    RoiCopy() 
    {
        // 该函数无任何操作。
    }

    // 成员方法：roiCopyAtHost（拷贝图片的 ROI 子图 Host 版本）
    // 根据输入图片设置的 ROI 子图，将其子图拷贝到输出图像中，将拷贝得到的图片
    // 放到 Host 端
    __host__ int           // 返回值：函数是否正确执行，若正确执行，返回
                           // NO_ERROR
    roiCopyAtHost(
            Image *inimg,  // 输入图片
            Image *outimg  // 输出图片
    );

    // 成员方法：roiCopyAtDevice（拷贝图片的 ROI 子图 Device 版本）
    // 根据输入图片设置的 ROI 子图，将其子图拷贝到输出图像中，将拷贝得到的图片
    // 放到当前 Device 端
    __host__ int           // 返回值：函数是否正确执行，若正确执行，返回
                           // NO_ERROR
    roiCopyAtDevice(
            Image *inimg,  // 输入图片
            Image *outimg  // 输出图片
    );
};

#endif

