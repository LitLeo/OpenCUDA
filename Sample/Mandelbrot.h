// Mandelbrot.h
//
// 生成Mandelbrot集
// 功能说明：以显示区域W内所有的点(mc, mr)作为初始迭代值 p = mc + mri 对
//           z = z * z + p 进行 times 次迭代，z 的初始值为 0。迭代完成后比较 z
//           的模和逃逸半径 radius 的大小，根据比较结果进行着色。

#ifndef __MANDELBROT_H__
#define __MANDELBROT_H__

#include "Image.h"
#include "ErrorCode.h"
#include "Complex.h"

// 类：Mandelbrot
// 继承自：无
// 以显示区域W内所有的点(mc, mr)作为初始迭代值 p = mc + mri 对 z = z * z + p
// 进行 times 次迭代，z 的初始值为 0。迭代完成后比较 z 的模和逃逸半径 radius 的
// 大小，根据比较结果进行着色。
class Mandelbrot {

protected:

    // 成员变量：radius（逃逸半径）
    // 设定逃逸半径
    float radius;

    // 成员变量：times（逃逸次数）
    // 通过迭代次数设定逃逸时间限制
    unsigned int times;

    // 成员变量：colorCount（颜色数量）
    // 绘制图像中包含的颜色数量。
    unsigned int colorCount;

    // 成员变量：exponent（计算指数）
    // 绘图使用的迭代方程的阶数。
    unsigned int exponent;

    // 成员变量：scopeFrom, scopeTo（绘图范围）
    // 指定所要绘制的 Mandelbrot 集的范围。
    Complex scopeFrom, scopeTo;

public:
    // 构造函数：Mandelbrot
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Mandelbrot(){
        // 使用默认值为类的各个成员变量赋值。
        this->radius = 500.0f;                  // 逃逸半径为 500
        this->times = 100;                      // 迭代次数为 100
        this->colorCount = 5;                   // 颜色数量为 5
        this->exponent = 2;                     // 迭代方程指数为 2
        this->scopeTo = Complex(-1.5f, -1.5f);  // 绘图起始点为 -1.5 - 1.5i
        this->scopeFrom = Complex(1.5f, 1.5f);  // 绘图终点为 1.5 + 1.5i
    }

    // 构造函数：Mandelbrot
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Mandelbrot(
            float radius,           // 逃逸半径
            unsigned int times,     // 逃逸时间
            unsigned int exp,       // 迭代方程指数
            const Complex &spfrom,  // 绘图起始点
            const Complex &spto     // 绘图终点
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->radius = 500.0f;                  // 逃逸半径为 500
        this->times = 100;                      // 迭代次数为 100
        this->colorCount = 5;                   // 颜色数量为 5
        this->exponent = 2;                     // 迭代方程指数为 2
        this->scopeTo = Complex(-1.5f, -1.5f);  // 绘图起始点为 -1.5 - 1.5i
        this->scopeFrom = Complex(1.5f, 1.5f);  // 绘图终点为 1.5 + 1.5i

        // 根据参数列表中的值设定成员变量的初值
        setRadius(radius);
        setTimes(times);
        //setExponent(exp);
        setScopeFrom(spfrom);
        setScopeTo(spto);
    }
   
    // 成员方法：getRadius（获取逃逸半径）
    // 获取成员变量 radius 的值。
    __host__ __device__ float  // 返回值：返回 radius 成员变量的值
    getRadius() const
    {
        // 返回 radius 成员变量的值
        return this->radius;
    } 

    // 成员方法：setRadius（设置逃逸半径）
    // 设置成员变量 radius 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，
                             // 返回 NO_ERROR。
    setRadius(
            float radius     // 逃逸半径
    ) {
        // 判断 radius 是否为大于 0 的值，如果大于 0，进行赋值
        if (radius > 0.0f) {
            // 将 radius 成员变量赋成新值
            this->radius = radius;

            // 若函数正确执行，返回 NO_ERROR。
            return NO_ERROR;
        }
        // 如果 radius 不大于 0，返回错误码（无效数据）
        else
            return INVALID_DATA;
    }

    // 成员方法：getTimes（获取迭代次数）
    // 获取成员变量 times 的值。
    __host__ __device__ unsigned int  // 返回值：返回 times 成员变量的值
    getTimes() const
    {
        // 返回 times 成员变量的值
        return this->times;
    }

    // 成员方法：setTimes（设置迭代次数）
    // 设置成员变量 times 的值。
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回 NO_ERROR。
    setTimes(
            unsigned int times  // 逃逸时间
    ) {        
        // 将 times 成员变量赋成新值
        this->times = times;

        // 若函数正确执行，返回 NO_ERROR。
        return NO_ERROR;
    }

    // 成员方法：getColorCount（获取颜色数量）
    // 获取成员变量 colorCount 的值。
    __host__ __device__ unsigned int  // 返回值：返回 colorCount 成员变量的值
    getColorCount() const
    {
        // 返回 colorCount 成员变量的值
        return this->colorCount;
    }

    // 成员方法：setColorCount（设置颜色数量）
    // 设置成员变量 colorCount 的值。
    __host__ __device__ int        // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    setColorCount(
            unsigned int colorcnt  // 逃逸时间
    ) {
        // 检查新的颜色数量的合法性。
        if (colorcnt < 1 || colorcnt > times || colorcnt > 256)
            return INVALID_DATA;

        // 将 colorCount 成员变量赋成新值
        this->colorCount = colorcnt;

        // 若函数正确执行，返回 NO_ERROR。
        return NO_ERROR;
    }

    // 成员方法：getExponent（获取迭代方程阶数）
    // 获取成员变量 exponent 的值。
    __host__ __device__ unsigned int  // 返回值：返回 exponent 成员变量的值
    getExponent() const
    {
        // 返回 times 成员变量的值
        return this->exponent;
    }

    // 成员方法：setTimes（设置迭代方程阶数）
    // 设置成员变量 exponent 的值。
    __host__ __device__ int      // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    setExponent(
            unsigned int newexp  // 迭代方程阶数
    ) {
        // 如果迭代方程阶数小于 2 或者大于 10 则报错。从原理上来讲该值可以是大
        // 于等于 2 的任何值，但过大的值可能会导致计算上过于复杂，而且这些情况
        // 也极少用到，因此我们直接屏蔽了这些值。
        if (newexp < 2 || newexp > 10)
            return INVALID_DATA;

        // 将 exponent 成员变量赋成新值
        this->exponent = newexp;

        // 若函数正确执行，返回 NO_ERROR。
        return NO_ERROR;
    }

    // 成员方法：getScopeFrom（获取绘图起始点）
    // 获取成员变量 scopeFrom 的值。
    __host__ __device__ Complex  // 返回值：返回 scopeFrom 成员变量的值
    getScopeFrom() const
    {
        // 返回成员变量 scopeFrom 的值。
        return this->scopeFrom;
    }

    // 成员方法：setScopeFrom（设置绘图起始点）
    // 设置成员变量 scopeFrom 的值，我们这里不判断 scopeFrom 和 scopeTo 之间的
    // 大小关系，如果 scopeFrom 大于 scopeTo，则绘制出来的图像是镜像样式的。
    __host__ __device__ int           // 返回值：函数是否正确执行，若函数正确执
                                      // 行，返回 NO_ERROR。
    setScopeFrom(
            const Complex &newspfrom  // 新的绘图起始点
    ) {
        // 将 scopeFrom 成员变量赋为新值。
        this->scopeFrom = newspfrom;

        // 函数正确执行，返回 NO_ERROR。
        return NO_ERROR;
    }

    // 成员方法：getScopeTo（获取绘图终点）
    // 获取成员变量 scopeTo 的值。
    __host__ __device__ Complex  // 返回值：返回 scopeTo 成员变量的值
    getScopeTo() const
    {
        // 返回成员变量 scopeTo 的值。
        return this->scopeTo;
    }

    // 成员方法：setScopeTo（设置绘图终点）
    // 设置成员变量 scopeTo 的值，我们这里不判断 scopeFrom 和 scopeTo 之间的大
    // 小关系，如果 scopeFrom 大于 scopeTo，则绘制出来的图像是镜像样式的。
    __host__ __device__ int         // 返回值：函数是否正确执行，若函数正确执
                                    // 行，返回 NO_ERROR。
    setScopeTo(
            const Complex &newspto  // 新的绘图起始点
    ) {
        // 将 scopeTo 成员变量赋为新值。
        this->scopeTo = newspto;

        // 函数正确执行，返回 NO_ERROR。
        return NO_ERROR;
    }

    // Host 成员方法：mandelbrot（Mandelbrot 集的生成）
    // 以显示区域 W 内所有的点 (mc, mr) 作为初始迭代值 p = mc + mri 对
    // z = z * z + p 进行 times 次迭代，z 的初始值为 0。迭代完成后比较 z 的模和
    // 逃逸半径 radius 的大小，根据比较结果进行着色。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    mandelbrot(
            Image *outimg  // 输出图像
    ); 

    // Host 成员方法：mandelbrotHost（Mandelbrot 集的生成）
    // 以显示区域 W 内所有的点 (mc, mr) 作为初始迭代值 p = mc + mri 对 
    // z = z * z + p 进行 times 次迭代，z 的初始值为 0。迭代完成后比较 z 的模和
    // 逃逸半径 radius 的大小，根据比较结果进行着色。该接口是利用 CPU 串行实
    // 现。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    mandelbrotHost(
            Image *outimg  // 输出图像
    );
};

#endif

