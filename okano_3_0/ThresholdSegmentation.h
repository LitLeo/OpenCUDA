// ThresholdSegmentation.h 
// 创建人：王媛媛
//
// 阈值化（ThresholdSegmentation）
// 功能说明：根据设定的阈值T, D，如果像素的灰度值与 T 之差小于 D，
//           此像素的灰度值赋值为 255。否则，此像素的灰度值赋值为 0。
//
// 修订历史：
// 2013年09月09日（王媛媛）
//     初始版本
// 2013年09月22日（王媛媛）
//     修订注释

#ifndef __THRESHOLDSEGMENTATION_H__
#define __THRESHOLDSEGMENTATION_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：ThresholdSegmentation
// 继承自：无
// 根据设定的阈值，对灰度图像进行二值化处理，得到二值图像。
class ThresholdSegmentation {

protected:

    // 成员变量：T，D
    // 二值化判断的标准，范围是 [0, 255]。
    unsigned char T;
    unsigned char D;

public:
    // 构造函数：ThresholdSegmentation
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    ThresholdSegmentation()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->T = 200;
        this->D = 50;
    }

    // 构造函数：ThresholdSegmentation
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    ThresholdSegmentation(
            unsigned char T,
            unsigned char D
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->T = 200;
        this->D = 50;

        // 根据参数列表中的值设定成员变量的初值
        setT(T);
        setD(D);
    }
   
    // 成员方法：getT（获取灰度阈值）
    // 获取成员变量 T 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 T 的值
    getT() const
    {
        // 返回 T 成员变量的值。
        return this->T;
    } 

    // 成员方法：setT（设置灰度阈值）
    // 设置成员变量 T 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setT(
        unsigned char T      // 设定新的灰度阈值
    ) {
        // 将 T 成员变量赋成新值
        this->T = T;
        return NO_ERROR;
    }

    // 成员方法：getD（获取灰度阈值）
    // 获取成员变量 D 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 D 的值
    getD() const
    {
        // 返回 D 成员变量的值。
        return this->D;
    } 

    // 成员方法：setD（设置灰度阈值）
    // 设置成员变量 D 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setD(
            unsigned char D  // 设定新的灰度阈值
    ) {
        // 将 T 成员变量赋成新值
        this->D = D;

        return NO_ERROR;
    }

    // Host 成员方法：thresholdSeg_parallel（阈值分割并行处理）
    // 根据阈值对图像进行二值化处理。如果像素的灰度值与 T 之差小于 D，
    // 此像素的灰度值赋值为 255。否则，此像素的灰度值赋值为 0。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    thresholdSeg_parallel(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );

    // Host 成员方法：thresholdSeg_serial（阈值分割串行处理）
    // 根据阈值对图像进行二值化处理。如果像素的灰度值与 T 之差小于 D，
    // 此像素的灰度值赋值为 255。否则，此像素的灰度值赋值为 0。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    thresholdSeg_serial(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
};

#endif

