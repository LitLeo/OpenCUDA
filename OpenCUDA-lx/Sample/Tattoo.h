// Tattoo.h 
//
// 贴图（Tattoo）
// 功能说明：输入两幅图像，一幅为前景图，一幅为背景图，输出一幅图像：其中输出图
//           像满足当前景图灰度值与指定透明像素相同时，则输出背景图对应的灰度值
//           否则输出前景图灰度值。

#ifndef __TATTOO_H__
#define __TATTOO_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：Tattoo（贴图）
// 继承自：无
// 输入两幅图像，一幅为前景图，一幅为背景图，输出一幅图像，其中输出图像满足当前
// 景图灰度值与指定透明像素相同时，则输出背景图对应的灰度值，否则输出前景图灰度
// 值。
class Tattoo {

protected:

    // 成员变量：dummyPixel（透明像素）
    // 贴图标准，范围为 [0，255]。
    unsigned char dummyPixel;

public:
    // 构造函数：Tattoo
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Tattoo()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->dummyPixel = 128;  // 透明像素默认为128。
    }

    // 构造函数：Tattoo
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    Tattoo(
            unsigned char dummypixel  // 透明像素
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->dummyPixel = 128;  // 透明像素默认为128。

        // 根据参数列表中的值设定成员变量的初值。
        setDummyPixel(dummypixel);
    }

    // 成员方法：getDummyPixel（获取透明像素）
    // 获取成员变量 dummyPixel 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 dummyPixel 的值
    getDummyPixel() const
    {
        // 返回 dummyPixel 成员变量的值
        return this->dummyPixel;
    }

    // 成员方法：setDummyPixel(设置透明像素)
    // 设置成员变量 dummyPixel 的值
    __host__ __device__ int           // 返回值：函数是否正确执行，若函数正确
                                      // 执行，返回 NO_ERROR。
    setDummyPixel(
            unsigned char dummypixel  // 设定新的透明像素
    ) {
        // 将 dummyPixel 成员变量赋值。
        this->dummyPixel = dummypixel;

        return NO_ERROR;
    }

    // Host 成员方法：Tattoo（贴图）
    // 输入两幅图像，一幅为前景图，一幅为背景图，输出一幅图像，其中输出图像满足
    // 当前景图灰度值与指定透明像素相同时，则输出背景图对应的灰度值，否则输出前
    // 景图灰度值。
    __host__ int         // 返回值：函数是否正确执行，若函数正确执行，返回 
                         // NO_ERROR。
    tattoo(
          Image *frimg,  // 前景图
          Image *baimg,  // 背景图
          Image *outimg  // 输出图像
    );
};

#endif

