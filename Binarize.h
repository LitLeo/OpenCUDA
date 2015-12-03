// Binarize.h 
//
// 二值化（Binarize）
// 功能说明：根据设定的阈值，对灰度图像进行二值化处理，得到二值图像。
//
// 修订历史：
//     初始版本
//     完善了 Out-Place 版本的图像二值化处理。更正了代码中的一些错误。
//     更正了代码中的一些错误。
//     按照最新版的编码规范对代码进行了调整，并修正了一些之前未发现的格式错误。 

#ifndef __BINARIZE_H__
#define __BINARIZE_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：Binarize
// 继承自：无
// 根据设定的阈值，对灰度图像进行二值化处理，得到二值图像。如果像素的灰度值大
// 于等于阈值，此像素的灰度值赋值为 255。如果像素的灰度值小于阈值，此像素的灰
// 度值赋值为 0。
class Binarize {

protected:

    // 成员变量：threshold(灰度阈值)
    // 二值化判断的标准，范围是 [0, 255]。
    unsigned char threshold;

public:
    // 构造函数：Binarize
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Binarize()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->threshold = 128;  // 灰度阈值默认为 128。
    }

    // 构造函数：Binarize
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Binarize(
            unsigned char threshold  // 灰度阈值
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->threshold = 128;  // 灰度阈值默认为 128。
	
        // 根据参数列表中的值设定成员变量的初值
        setThreshold(threshold);
    }
   
    // 成员方法：getThreshold（获取灰度阈值）
    // 获取成员变量 threshold 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 threshold 的值
    getThreshold() const
    {
        // 返回 threshold 成员变量的值。
        return this->threshold;
    } 

    // 成员方法：setThreshold（设置灰度阈值）
    // 设置成员变量 threshold 的值。
    __host__ __device__ int          // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    setThreshold(
            unsigned char threshold  // 设定新的灰度阈值
    ) {
        // 将 threshold 成员变量赋成新值
        this->threshold = threshold;

        return NO_ERROR;
    }

    // Host 成员方法：binarize（二值化处理）
    // 根据阈值对图像进行二值化处理。这是一个 Out-Place 形式的。如果像素的灰度
    // 值大于等于阈值，此像素的灰度值赋值为 255。如果像素的灰度值小于阈值，此
    // 像素的灰度值赋值为 0。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    binarize(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );

    // Host 成员方法：binarize（二值化处理）
    // 根据阈值对图像进行二值化处理。这是一个 In-Place 形式的。如果像素的灰度
    // 值大于等于阈值，此像素的灰度值赋值为 255。如果像素的灰度值小于阈值，此
    // 像素的灰度值赋值为 0。
    __host__ int        // 返回值：函数是否正确执行，若函数正确执行，返回 
                        // NO_ERROR。
    binarize(
            Image *img  // 输入图像
    );
};

#endif

