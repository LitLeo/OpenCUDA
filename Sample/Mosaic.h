// Mosaic.h
//
// 马赛克（Mosaic）
// 功能说明：在给定的图像范围内，将图像编程马赛克的样子（电视上常见的遮挡人脸的
//           效果）。

#ifndef __MOSAIC_H__
#define __MOSAIC_H__

#include <stdio.h>

#include "Image.h" 
#include "ErrorCode.h"

// 类：Mosaic
// 继承自：无
// 将图像划分为 n * n 的小块若干，每个小块内求平均值，并在输出图像对应的小
// 块内各个像素点均赋为该值。
class Mosaic {

protected:

    // 成员变量：mossize（两个方向划分为多少小块）
    // 范围：[0, min(outimg.width, outimg.height)]
    int mossize;

public:
    // 构造函数： Mosaic
    // 无参数版本的构造函数，成员变量初始化为默认值
    __host__ __device__
    Mosaic()
    {
        // 使用默认值为个各类成员变量赋值。
        this->mossize = 32;  // 默认马赛克块大小为 32。
    }

    // 构造函数：Mosaic
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Mosaic(
            int mossize  // 每个马赛克小块的尺寸大小（单位：像素）
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非
        // 法的初始值而使系统进入一个未知的状态。
        this->mossize = 32;  // 默认马赛克块大小为 32

        // 根据参数列表中的值设定成员变量的初值
        setMossize(mossize);
    }

    // 成员方法： getMossize（获取马赛克块的尺寸）
    // 获取成员变量 mossize 的值。
    __host__ __device__ int  // 返回值；成员变量 mossize 的值。
    getMossize() const
    {
        // 返回 mossize 成员变量的值。
        return mossize;
    }

    // 成员方法： setMossize（设置马赛克块的尺寸）
    // 设置成员变量 mossize 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setMossize(
            int mossize      // 每个马赛克小块的尺寸大小（单位：像素）
    ) {
        if (mossize < 1)                     
            return INVALID_DATA;

        // 将 mossize 赋为新值。
        this->mossize = mossize;
        return NO_ERROR;
    }

    // Host 成员方法： mosaic（马赛克处理）
    // 给图像的指定区域打马赛克，这是一个 Out-Place 形式的。将图像指定区域划
    // 分为 n * n 的小块若干，每个小块内求平均值，并在输出图像对应的小块内各
    // 个像素点均赋为该值。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    mosaic(
            Image *inimg,  // 输入图像 
            Image *outimg  // 输出图像
    );

    // Host 成员方法： mosaic（马赛克处理）
    // 给图像的指定区域打马赛克，这是一个 In-Place 形式的。将图像指定区域划
    // 分为 n * n 的小块若干，每个小块内求平均值，并在输出图像对应的小块内各
    // 个像素点均赋为该值。
    __host__ int          // 返回值：函数是否正确执行，若函数正确执行，返回
                          // NO_ERROR。
    mosaic(
            Image *inimg  //输入图像
    );
};

#endif

