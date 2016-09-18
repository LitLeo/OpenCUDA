// ImageScaling.h
//
// 图像扩缩（ImageScaling）
// 功能说明：根据给定扩缩中心和扩缩系数，实现图像的扩大或缩小

#ifndef __IMAGESCALING_H__
#define __IMAGESCALING_H__

#include "ErrorCode.h"
#include "Image.h"

class ImageScaling {

protected:

    // 成员变量：x 和 y（扩缩中心）
    // 用于设定扩缩中心，以输入图像的该点为中心进行图像扩缩
    int x, y;

    // 成员变量：scalCoefficient（扩缩系数）
    // 设定图像扩缩的倍数
    float scalCoefficient;

public:

    // 构造函数：ImageScaling
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值
    __host__ __device__ 
    ImageScaling()
    {
        // 使用默认值为类的各个成员变量赋值
        x = 0;                   // 扩缩中心默认值为 (0, 0)
        y = 0;
        scalCoefficient = 1.0f;  // 扩缩系数默认值为1
    }

    // 构造函数：ImageScaling
    // 有参数版本的构造函数，根据需要给定各个参数
    __host__ __device__
    ImageScaling(
            int x, int y,              // 扩缩中心
            float scalcoefficient      // 扩缩系数
    ) { 
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->x = 0;                   // 扩缩中心默认值为 (0, 0)
        this->y = 0;
        this->scalCoefficient = 1.0f;  // 扩缩系数默认值为1  
        
        // 根据参数列表中的值设定成员变量的初值
        this->setX(x);
        this->setY(y);
        this->setScalCoefficient(scalcoefficient);
    }

    // 成员方法：getX（读取扩缩中心 x 分量）
    // 读取成员变量 x 的值
    __host__ __device__ int  // 返回值：当前成员变量 x 的值
    getX() const
    {
        // 返回成员变量 x 的值
        return this->x;
    }

    // 成员方法：setX（设置扩缩中心 x 分量）
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。 
    setX(
            int x            // 新的扩缩中心 x 分量
    ) {
        // 将成员变量 x 赋成新值
        this->x = x;
        return NO_ERROR;
    }

    // 成员方法：getY（读取扩缩中心 y 分量）
    // 读取成员变量 y 的值
    __host__ __device__ int  // 返回值：当前成员变量 y 的值
    getY() const
    {
        // 返回成员变量 y 的值
        return this->y;
    }

    // 成员方法：setY（设置扩缩中心 y 分量）
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR
    setY(
            int y            // 新的扩缩中心 y 分量
    ) {
        // 将成员变量 y 赋成新值
        this->y = y;
        return NO_ERROR;
    }

    // 成员方法：getScalCoefficient（读取扩缩系数）
    // 读取成员变量 scalCoefficient 的值
    __host__ __device__ int  // 返回值：当前成员变量 scalCoefficient 的值
    getScalCoefficient() const
    {
        // 返回成员变量 scalCoefficient 的值
        return this->scalCoefficient;
    }

    // 成员方法：setScalCoefficient（设置扩缩系数）
    __host__ __device__ int        // 返回值：函数是否正确执行，若函数正确执行，
                                   // 返回NO_ERROR                               
    setScalCoefficient(
            float scalcoefficient  // 新的扩缩系数
    ) {
        // 检查输入参数是否合法
        if (scalcoefficient <= 0)
            return INVALID_DATA;

        // 将成员变量 scalCoefficient 赋成新值
        this->scalCoefficient = scalcoefficient;
        return NO_ERROR;
    }

    // Host 成员方法：scaling（图像扩缩）
    // 以输入图像扩缩中心为中心，进行扩缩系数倍数的扩缩
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    scaling(
            Image *inimg,  // 输入图像 
            Image *outimg  // 输出图像
    );
};

#endif

