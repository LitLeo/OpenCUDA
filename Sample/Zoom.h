// Zoom.h
//
// 放大镜定义（Zoom）
// 功能说明：定义了放大镜类，将图像进行局部放大处理，根据用户要求的对图像放大的
// 中心点、放大尺寸局部放大图片。

#ifndef __ZOOM_H__
#define __ZOOM_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：Zoom
// 继承自：无
// 根据设定的中心点、半径、放大倍数，对灰度图像进行局部放大处理，得到局部放大图
// 像。假如我们定义放大镜的坐标为(O_X, O_Y)，半径为 Radius，而放大倍数为 M，那
// 么其实就是将原图中的坐标为(O_X, O_Y)、半径为 Radius 的区域的图像放大到放大镜
// 覆盖的区域即可，对图片上的每一个点，求其与(CenterX, CenterY)的距离 Distance，
// 并且根据函数变化获得实际放大率M，若 Distance < Radius，则取原图中坐标为
// ((C1_x - O_x) / M + O_x, (C1_y - O_y) / M + O_y)的像素的灰度值作新的颜色值。
class Zoom{

protected:

    // 成员变量：centreX（中心点横坐标）
    // 中心点横坐标，范围是 [0, 图像宽]。
    int centreX;

    // 成员变量：centreY（中心点纵坐标）
    // 中心点纵坐标，范围是 [0, 图片高]。
    int centreY;

    // 成员变量：circleRadius（放大区域半径）
    unsigned int circleRadius;

    // 成员变量：magnifyMul（放大倍数）
    float magnifyMul;

public:

    // 构造函数：Zoom
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Zoom()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->centreX = 0;       // 默认中心点横坐标是 0
        this->centreY = 0;       // 默认中心点纵坐标是 0
        this->circleRadius = 1;  // 默认区域半径为 1
        this->magnifyMul = 2.0;  // 默认放大倍数为 2
    }

    // 构造函数：Zoom
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Zoom(
            int centreX,                // 中心横坐标
            int centreY,                // 中心纵坐标
            unsigned int circleRadius,  // 放大半径
            float magnifyMul            // 放大倍数
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->centreX = 0;       // 中心横坐标默认为 0
        this->centreY = 0;       // 中心纵坐标默认为 0
        this->circleRadius = 1;  // 默认区域半径为 1
        this->magnifyMul = 2;    // 默认放大倍数为 2

        // 根据参数列表中的值设定成员变量的初值
        setCentreX(centreX);
        setCentreY(centreY);
        setCircleRadius(circleRadius);
        setMagnifyMul(magnifyMul);
    }

    // 成员方法：getCentreX（获取中心横坐标值）
    // 获取成员变量 centreX 的值。
    __host__ __device__ int  // 返回值：成员变量 centreX 的值
    getCentreX() const
    {
        // 返回 centreX 成员变量的值。
        return this->centreX;
    } 

    // 成员方法：setCentreX（设置中心横坐标值）
    // 设置成员变量 centreX 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，
                             // 若函数正确执行，返回 NO_ERROR
                             // 否则返回 INVALID_DATA。
    setCentreX( 
            int centreX      // 设置新的中心点横坐标
    ) {
        // 若参数小于0，返回对应的错误码并退出
        if (centreX < 0)
            return INVALID_DATA;

        // 将 centreX 成员变量赋新值
        this->centreX = centreX;

        return NO_ERROR;
    }

    // 成员方法：getCentreY（获取中心纵坐标值）
    // 获取成员变量 centreY 的值。
    __host__ __device__ int  // 返回值：成员变量 centreY 的值
    getCentreY() const
    {
        // 返回 centreY 成员变量的值。
        return this->centreY;
    } 

    // 成员方法：setCentreY（设置中心纵坐标值）
    // 设置成员变量 centreY 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，
                             // 若函数正确执行，返回 NO_ERROR
                             // 否则返回错误码 INVALID_DATA。
    setCentreY( 
            int centreY      // 设置新的中心点纵坐标
    ) {
        // 若参数小于0，返回对应的错误码并退出
        if (centreY < 0)
            return INVALID_DATA;

        // 将 centreY 成员变量赋新值
        this->centreY = centreY;

        return NO_ERROR;
    } 

    // 成员方法：getCircleRadius（获取半径值）
    // 获取成员变量 circleRadius 的值。
    __host__ __device__ unsigned int  // 返回值：成员变量 circleRadius 的值
    getCircleRadius() const
    {
        // 返回 circleRadius 成员变量的值。
        return this->circleRadius;
    } 

    // 成员方法：setCircleRadius（设置半径值）
    // 设置成员变量 circleRadius 的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，
                              // 若函数正确执行，返回 NO_ERROR
                              // 否则返回错误码 INVALID_DATA。
    setCircleRadius( 
            int circleRadius  // 设置新的放大半径
    ) {
        // 若参数小于0，返回对应的错误码并退出
        if (circleRadius < 0)
            return INVALID_DATA;

        // 设置 circleRadius 成员变量的值。
        this->circleRadius = circleRadius;

        return NO_ERROR;
    } 

    // 成员方法：getMagnifyMul（获取放大倍数值）
    // 获取成员变量 magnifyMul 的值。
    __host__ __device__ float  // 返回值：成员变量 magnifyMul 的值
    getMagnifyMul() const
    {
        // 返回 magnifyMul 成员变量的值。
        return this->magnifyMul;
    } 

    // 成员方法：setMagnifyMul（设置放大倍数值）
    // 设置成员变量 magnifyMul 的值。
    __host__ __device__ int    // 返回值：函数是否正确执行，
                               // 若函数正确执行，返回 NO_ERROR
                               // 否则返回 INVALID_DATA。
    setMagnifyMul( 
            float magnifyMul   // 设置新的放大倍数
    ) {
        // 若参数小于0，返回对应的错误码并退出
        if (magnifyMul < 1.0e-8)
            return INVALID_DATA;

        // 设置 magnifyMul 成员变量的值。
        this->magnifyMul = magnifyMul;

        return NO_ERROR;
    } 

    // Host 成员方法：zoom（放大镜处理）
    // 根据中心及半径和放大倍数对图像进行放大镜处理。
    // 以中心点为中心，对半径区域进行放大处理
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    zoom(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
};

#endif
