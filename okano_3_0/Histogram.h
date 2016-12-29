// Histogram.h
// 创建人：侯怡婷
//
// 图像直方图（Histogram）
// 功能说明：实现计算图像的直方图。输出的结果保存在int型的数组中，数组的下标代表
//           灰度图像的灰度值，数组中存储的为等于该灰度值的像素点总数。
//
// 修订历史：
// 2012年08月15日（侯怡婷）
//     初始版本。
// 2012年10月25日（侯怡婷）
//     更正了代码的一些注释规范。
// 2012年11月15日（侯怡婷）
//     在核函数执行后添加 cudaGetLastError 判断语句。
// 2012年11月23日（侯怡婷）
//     增加对输入指针 histogram 有效性判断。
// 2013年06月26日（于玉龙）
//     修正了单线程处理多个点的一处 Bug。

#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

#include "Image.h"


// 类：Histogram（图像直方图）
// 继承自：无
// 功能说明：实现计算图像的直方图。输出的结果保存在int型的数组中，数组的下标代表
//           灰度图像的灰度值，数组中存储的为等于该灰度值的像素点总数。
class Histogram {
   
public:

    // 构造函数：Histogram
    // 无参数版本的构造函数，因为该类没有成员变量，所以该构造函数为空。
    // 没有需要设置默认值的成员变量。
    __host__ __device__
    Histogram() {}

    // Host 成员方法：histogram（图像直方图）
    // 实现计算图像的直方图。输出的结果保存在 int 型的数组中，数组的下标代表
    // 灰度图像的灰度值，数组中存储的为等于该灰度值的像素点总数。
    __host__ int                      // 返回值：函数是否正确执行，若函数正确执
                                      // 行，返回
                                      // NO_ERROR。
    histogram(
            Image *inimg,             // 输入图像
            unsigned int *histogram,  // 输出参数，直方图。
            bool onhostarray = true   // 判断 histogram 是否是 Host 内存的指针，
                                      // 默认为“是”。			
    );
};

#endif

