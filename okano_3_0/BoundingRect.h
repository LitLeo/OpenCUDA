// BoundingRect.h 
// 创建人：刘瑶
//
// 包围矩形（Bounding Rectangle）
// 功能说明：找出图像中给定点集的包围矩形，根据给定点集的协方差矩阵，找出主方
//           向，求得包围矩形及相应参数。
//
// 修订历史：
// 2012年09月06日（刘瑶）
//     初始版本
// 2012年09月16日（刘瑶）
//     修改了一些错误和注释
// 2012年10月25日（刘瑶）
//     按照最新版的编码规范对代码进行了调整，并修正了一些之前未发现的格式错误
// 2012年11月23日（刘瑶）
//     添加对 *outrect 指针的有效性进行检查
// 2013年03月19日（刘瑶）
//     修改了计算旋转角度的错误，将误加的弧度值改为角度值。
// 2013年06月26日（刘瑶）
//     修正了单线程处理多个点的一处 Bug。
// 2013年09月07日（于玉龙）
//     增加了串行的包围矩形算法实现。
// 2013年09月22日（于玉龙）
//     修正了角度旋转过程中的计算 BUG。

#ifndef __BOUNDINGRECT_H__
#define __BOUNDINGRECT_H__

#include "Image.h"
#include "ErrorCode.h"
#include "Rectangle.h"

// 类：BoundingRect
// 继承自：无
// 根据给定的对象的像素值，找出图像中的所要包围的对象，根据给定对象点集的协方差
// 矩阵，找出主方向，求得包围矩形的相应参数。
class BoundingRect {

protected:

    // 成员变量：value（对象的像素值）
    // 找出图像中符合条件的点集的像素值，范围是 [0, 255]。
    unsigned char value;

    // 成员变量：pixelCount（符合条件的像素点数量）
    unsigned long long int pixelCount; 

public:

    // 构造函数：BoundingRect
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    BoundingRect()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->value = 128;  // 对象的像素值默认为 128。
    }

    // 构造函数：BoundingRect
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    BoundingRect(
            unsigned char value  // 对象的像素值
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->value = 128;  // 对象的像素值默认为 128。

        // 根据参数列表中的值设定成员变量的初值
        setValue(value);
    }
	
    // 成员方法：getValue（获取对象的像素值）
    // 获取成员变量 value 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 value 的值
    getValue() const
    {
        // 返回 value 成员变量的值。
        return this->value;    
    } 

    // 成员方法：setValue（设置对象的像素值）
    // 设置成员变量 value 的值。
    __host__ __device__ int      // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    setValue(
            unsigned char value  // 设定新的对象的像素值
    ) {
        // 将 value 成员变量赋成新值。
        this->value = value;
        
        return NO_ERROR;
    }
	
    // Host 成员方法：boundingRect（求像素值给定的对象的包围矩形）
    // 根据给定的像素值在图像中找出对象，根据协方差的方法，找出对象的包围矩形，
    // 输出为求得的包围矩形。
    __host__ int                 // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    boundingRect(
            Image *inimg,        // 输入图像
            Quadrangle *outrect  // 包围矩形
    );

    // Host 成员方法：boundingRectHost（求像素值给定的对象的包围矩形）
    // 根据给定的像素值在图像中找出对象，根据协方差的方法，找出对象的包围矩形，
    // 输出为求得的包围矩形。该函数在 Host 端由 CPU 串行计算。
    __host__ int                 // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    boundingRectHost(
            Image *inimg,        // 输入图像
            Quadrangle *outrect  // 包围矩形
    );

    // Host 成员方法：boundingRect（求像素值给定的对象的包围矩形）
    // 根据给定的像素值在图像中找出对象，根据协方差的方法，找出对象的包围矩形，
    // 输出为求得的包围矩形。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确执行，
                                   // 返回 NO_ERROR。
    boundingRect(
            Image *inimg,          // 输入图像
            DirectedRect *outrect  // 包围矩形
    );

    // Host 成员方法：boundingRectHost（求像素值给定的对象的包围矩形）
    // 根据给定的像素值在图像中找出对象，根据协方差的方法，找出对象的包围矩形，
    // 输出为求得的包围矩形。该函数在 Host 端由 CPU 串行计算。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确执行，
                                   // 返回 NO_ERROR。
    boundingRectHost(
            Image *inimg,          // 输入图像
            DirectedRect *outrect  // 包围矩形
    );
};

#endif

