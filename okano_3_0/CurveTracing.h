// CurveTracing.h
// 创建者：欧阳翔
// 
// 曲线跟踪（CurveTracing）
// 功能说明：对于给定的一副单一像素宽度的二值图像，跟踪图像上的所有曲线，得到
//           闭合曲线和非闭合曲线的有序序列，如果图中有曲线相交情况，就会产生
//           更多的闭合和非闭合曲线，需要得到所有曲线组合的情况。

// 修订历史：
// 2013年9月10日（欧阳翔）
//     初始版本，曲线跟踪初步实现。
// 2013年10月10日（欧阳翔）
//     修改了曲线交点处理的方法。
// 2013年10月15日（欧阳翔）
//     曲线跟踪的串行和并行的完全实现
// 2013年11月10日（欧阳翔）
//     修改了一些格式上的错误，减少了部分代码的冗余情况
// 2013年12月01日（欧阳翔）
//     增加了断点连接的曲线跟踪实现
// 2013年12月10日（欧阳翔）
//     修改了断点连接的方法，使得断点连接的处理结果相对自然

#ifndef __CURVETRACING_H__
#define __CURVETRACING_H__

#include "Curve.h"
#include "Graph.h"
#include "Image.h"
#include "DynamicArrays.h"
#include "ErrorCode.h"


// 类：CurveTracing（曲线跟踪）
// 继承自：无
// 曲线跟踪：对于给定的一副单一像素宽度的二值图像，跟踪图像上的所有曲线，得到
// 闭合曲线和非闭合曲线的有序序列，如果图中有曲线相交情况，就会产生更多的闭合
// 和非闭合曲线，需要得到所有曲线组合的情况。
class CurveTracing {

protected:

    // 成员变量：radius（圆形邻域参数）
    // 外部设定的圆形领域的半径大小。用于判断端点和交点可否连接，如果在某一端点
    // 领域内有其他端点，则这些端点可以认为是连着的
    int radius;

public:

    // 构造函数：CurveTracing
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    CurveTracing() {
        // 使用默认值为类的各个成员变量赋值。
        radius = 1;               // 默认圆的半径大小为 1
    }

    // 构造函数：CurveTracing
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    CurveTracing(
            int radius  // 圆形邻域参数（具体解释见成员变量）
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->radius = 1;               // 默认圆的半径大小为 1
        
        // 根据参数列表中的值设定成员变量的初值
        this->setRadius(radius);
    }
    
    // 成员方法：getRadius（读取圆形邻域参数）
    // 读取 radius 成员变量的值。
    __host__ __device__ int  // 返回值：当前 radius 成员变量的值。
    getRadius() const
    {
        // 返回 radius 成员变量的值。
        return this->radius;
    }

    // 成员方法：setRadius（设置圆形邻域参数）
    // 设置 radius 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setRadius(
            int radius       // 新的圆形邻域参数
    ) {
        // 将 radius 成员变量赋成新值
        this->radius = radius;
        return NO_ERROR;
    }

    // Host 成员方法：curveTracing（曲线跟踪）
    // 对图像进行曲线跟踪，得到非闭合曲线和闭合曲线的有序序列，根据输入参数设置
    // 大小具有一定的断点连接功能。
    __host__ int                 // 返回值：函数是否正确执行，若函数正确执行，返回
                                 // NO_ERROR。
    curveTracing(
            Image *inimg,        // 输入为单一像素宽的二值图像
            Curve ***curveList,  // 输出结果，得到的曲线序列
            int *openNum,        // 得到非闭合曲线数量
            int *closeNum        // 得到闭合曲线数量
    ); 
    
    // Host 成员方法：curveTracingCPU（曲线跟踪串行方法）
    // 对图像进行曲线跟踪，得到非闭合曲线和闭合曲线的有序序列，没有实现断点连接
    __host__ int                 // 返回值：函数是否正确执行，若函数正确执行，返回
                                 // NO_ERROR。
    curveTracingCPU(
            Image *inimg,        // 输入为单一像素宽的二值图像
            Curve ***curveList,  // 输出结果，得到的曲线序列
            int *openNum,        // 得到非闭合曲线数量
            int *closeNum        // 得到闭合曲线数量
    ); 
};

#endif
