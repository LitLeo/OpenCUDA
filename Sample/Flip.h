// Flip.h
// 创建人：张丽洁
// 
// 图像翻转（Flip）
// 功能说明：实现图像的水平和竖直翻转。

#ifndef __FLIP_H__
#define __FLIP_H__

#include "Image.h"


// 类：Flip（图像翻转）
// 继承自：无
// 通过对对应像素的像素值的的交换翻转，从而实现图像的水平和竖直的翻转。
class Flip {

public:

    // 构造函数：Flip
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Flip()
	{
        // 无任何操作。
    }

    // Host 成员方法：flipHorizontal（图像水平翻转）
    // 实现图像的水平翻转。
    __host__ int           // 返回值：函数是否正确执行，若正确执行，返回
                           // NO_ERROR。
    flipHorizontal(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );

    // Host 成员方法：flipVertical（图像竖直翻转）
    // 实现图像的竖直翻转。
    __host__ int           // 返回值：函数是否正确执行，若正确执行，返回
                           // NO_ERROR。
    flipVertical(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
};

#endif

