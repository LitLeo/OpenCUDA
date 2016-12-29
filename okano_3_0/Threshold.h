// Threshold.h
// 创建人：邓建平
//
// 阈值分割（Threshold）
// 功能说明：根据图像的某点的像素值是否在阈值内决定该点分割后的像素值，根据项目
//           要求，分割处理分为两种情况：（1）、未指定高低像素值，某点像素值在
//           阈值范围内时，不做改变，否则将改点像素值置 0; （2）、指定高低像素
//           值某点像素值在阈值范围内时，该点像素值置为高像素值，否则置为低像素
//           值
//
// 修订历史：
// 2012年09月01日（邓建平）
//     初始版本
// 2012年09月04日（邓建平）
//     根据需求重构了类，修正了大量的格式错误
// 2102年10月25日（邓建平）
//     按照最新版的编码规范对代码进行了调整，并修正了一些之前未发现的格式错误 

#ifndef __THRESHOLD_H__
#define __THRESHOLD_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：Threshold（阈值分割）
// 继承自：无
// 根据图像的某点的像素值是否在阈值内决定该点分割后的像素值，根据项目 要求，分
// 割处理分为两种情况：（1）、未指定高低像素值，某点像素值在阈值范围内时，不做
// 改变，否则将改点像素值置 0; （2）、指定高低像素值，某点像素值在阈值范围内时，
// 该点像素值置为高像素值，否则置为低像素值
class Threshold {

protected:

    // 成员变量：最小像素值，最大像素值，指定阈值的边界（不包括其本身）
    unsigned char  minPixelVal,maxPixelVal;   

public:

    // 构造函数：Threshold
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Threshold()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->maxPixelVal = 0;  // 最大像素值默认值为 0
        this->minPixelVal = 0;  // 最小像素值默认值为 0
    }

    // 构造函数：Threshold
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__                   
    Threshold(
            unsigned char  minpixelval,  // 最小像素值
            unsigned char  maxpixelval   // 最大像素值
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。 
        this->maxPixelVal = 0;  // 最大像素值默认值为 0
        this->minPixelVal = 0;  // 最小像素值默认值为 0
        
        // 根据参数列表中的值设定成员变量的初值
        this->setMinPixelVal(minPixelVal);
        this->setMaxPixelVal(maxpixelval);  
    }

    // 成员方法：getMinPixelVal（读取最小像素值）
    // 读取 minPixelVal 成员变量的值。
    __host__ __device__ unsigned char    // 返回: 当前  minPixelVal 成员变量的值
    getMinPixelVal() const
    {
        // 返回 minPixelVal 成员变量的值。
        return this->minPixelVal;
    }

    // 成员方法：setMinPixelVal（设置最小像素值）
    // 设置 minPixelVal 成员变量的值。
    __host__ __device__ int            // 返回值：函数是否正确执行，若函数正确
                                       // 执行，返回 NO_ERROR。
    setMinPixelVal(
            unsigned char minpixelval  // 最小像素值
    ) {
        // 将 minPixelVal 成员变量赋成新值
        this->minPixelVal = minpixelval;
        return NO_ERROR;
    }

    // 成员方法：getMaxPixelVal（读取最大像素值）
    // 读取 maxPixelVal 成员变量的值。
    __host__ __device__ unsigned char    // 返回: 当前  maxPixelVal 成员变量的值
    getMaxPixelVal() const
    {
        // 返回 maxPixelVal 成员变量的值。
        return this->maxPixelVal;
    }

    // 成员方法：setMaxPixelVal（设置最大像素值）
    // 设置 maxPixelVal 成员变量的值。
    __host__ __device__ int            // 返回值：函数是否正确执行，若函数正确
                                       // 执行，返回 NO_ERROR。
    setMaxPixelVal(
            unsigned char maxpixelval  // 最大像素值
    ) {
        // 将 maxPixelVal 成员变量赋成新值
        this->maxPixelVal = maxpixelval;
        return NO_ERROR;
    }

    // Host 成员方法：threshold（阈值分割）
    // 未指定高低像素值且输出图像不为 NULL 的阈值分割。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。 
    threshold(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );

    // Host 成员方法：threshold（阈值分割）
    // 未指定高低像素值且输出图像为 NULL 的阈值分割。
    __host__ int              // 返回值：函数是否正确执行，若函数正确执行，返回
                              // NO_ERROR。 
    threshold(
            Image *inoutimg   // 输入输出图像
    );

   // Host 成员方法：threshold（阈值分割）
   // 已指定高低像素值（low, high）且输出图像不为 NULL 的阈值分割。
    __host__ int                // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回 NO_ERROR。 
    threshold(
            Image *inimg,       // 输入图像
            Image *outimg,      // 输出图像
            unsigned char low,  // 低像素值
            unsigned char high  // 高像素值
    );

    // Host 成员方法：threshold（阈值分割）
    // 已指定高低像素值（low, high）且输出图像为 NULL 的阈值分割。
    __host__ int                // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回 NO_ERROR。  
    threshold(
            Image *inoutimg,    // 输入输出图像
            unsigned char low,  // 低像素值
            unsigned char high  // 高像素值
    );
};


#endif
