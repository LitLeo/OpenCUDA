// SmallestDirRect.h
// 创建者：刘瑶
//
// 计算最小有向外接矩形（SmallestDirRect）
// 功能说明：对于已知点集，计算点集的最小有向外接矩形，并最终输出最小有向外接矩
// 形的信息。
// 核心算法采用 Rotating Clipers 算法。
//
// 修订历史：
// 2013年01月04日（刘瑶）
//     初始版本。搭建框架。
// 2013年01月09日（刘瑶）
//     完善接口设计，完成核函数设计。
// 2013年01月12日（王雪菲，刘瑶）
//     完成计算相邻凸壳点构成的凸壳边的角度信息的函数。
// 2013年03月17日（刘瑶）
//     完成最小有向外接矩形的初步计算。
// 2013年03月18日（刘瑶）
//     修改了部分接口，完成其他接口的最小有向外接矩形的计算。
// 2013年03月19日（刘瑶）
//     修正了旋转角度的计算 bug，旋转角度方向现在统一为跟包围矩形的长边平行。
//     修改了包围矩形的四个顶点的取整舍入问题，分别处理。
// 2013年03月21日（刘瑶）
//     修改了部分代码结构错误。
// 2013年03月21日（刘瑶）
//     将核函数合并，由目前的 3 个核函数合并为 2 个。计算凸壳点集中每相邻两点的
//     旋转矩阵信息的核函数和计算新坐标系下凸壳的有向外接矩形的边界信息的核函数
//     合并，用第二个核函数中每个线程块的第一个线程来计算第一个核函数所需要求的
//     旋转矩阵信息，利用共享内存的优势。
// 2013年03月21日（刘瑶）
//     对 4 种点集输入接口，添加了点集的输入检查，处理输入点集为 1 或者 2 个点
//     的特殊情况。
// 2013年03月22日（刘瑶）
//     根据图像转换点集的函数接口，修改了部分代码，修改了构造函数和 get 与 set
//     函数，删掉了成员变量像素阈值。
// 2013年03月23日（刘瑶）
//     修改了最小有向外接矩形的 6 个接口，添加了判断输出是否是 host 端的指针，
//     对于不同的情况分别计算。
// 2013年03月24日（刘瑶）
//     根据修改的图像转换点集的函数接口，修改对应的调用代码。
// 2013年09月21日（于玉龙）
//     修正了计算 VALUE 值的一处 BUG。
// 2013年12月23日（于玉龙）
//     增加了最小有向外接矩形串行算法接口。

#ifndef __SMALLESTDIRRECT_H__
#define __SMALLESTDIRRECT_H__

#include "Image.h"
#include "ErrorCode.h"
#include "CoordiSet.h"
#include "ConvexHull.h"
#include "Rectangle.h"
#include "ImgConvert.h"

// 类：SmallestDirRect
// 继承自：无
// 根据给定的对象的像素值，找出图像中的所要包围的对象，最终输出最小有向外接矩形
// 的信息。
class SmallestDirRect {

protected:

    // 成员变量：imgCvt（图像与坐标集转换）
    // 根据给定的图像和阈值，转换成坐标集形式。
    ImgConvert imgCvt;

    // 成员变量：cvHull（凸壳）
    // 凸壳，主要计算给定点集的凸壳。
    ConvexHull cvHull;

    // 成员变量：value（图像转换点集的像素阈值）
    // 用于图像转换点集的像素阈值。
    unsigned char value;

    // Host 成员方法：sdrComputeBoundInfo（计算凸壳点集中每相邻两点的旋转矩阵
    // 信息,进而计算新坐标系下凸壳的有向外接矩形的边界信息）
    // 根据输入的凸壳点，计算顺时针相邻两点的构成的直线与 x 轴的角度，同时计算
    // 旋转矩阵信息。在此基础上，计算新坐标系下各点的坐标。从而计算每个有向外接
    // 矩形的边界点的坐标信息。
    __host__ int                        // 返回值：函数是否正确执行，若函数正确
                                        // 执行，返回 NO_ERROR。
    sdrComputeBoundInfo(
            CoordiSet *convexcst,       // 输入凸壳点集。
            RotationInfo rotateinfo[],  // 输出，旋转矩阵信息数组。
            BoundBox bbox[]             // 输出，找出的有向外接矩形的边界坐标
                                        // 信息数组。
    );

    // Host 成员方法：sdrComputeBoundInfoCpu（计算凸壳点集中每相邻两点的旋转矩阵
    // 信息,进而计算新坐标系下凸壳的有向外接矩形的边界信息）
    // 根据输入的凸壳点，计算顺时针相邻两点的构成的直线与 x 轴的角度，同时计算
    // 旋转矩阵信息。在此基础上，计算新坐标系下各点的坐标。从而计算每个有向外接
    // 矩形的边界点的坐标信息。
    __host__ int                        // 返回值：函数是否正确执行，若函数正确
                                        // 执行，返回 NO_ERROR。
    sdrComputeBoundInfoCpu(
            CoordiSet *convexcst,       // 输入凸壳点集。
            RotationInfo rotateinfo[],  // 输出，旋转矩阵信息数组。
            BoundBox bbox[]             // 输出，找出的有向外接矩形的边界坐标
                                        // 信息数组。
    );

    // Host 成员方法：sdrComputeSDR（计算有向外接矩形中面积最小的）
    // 根据输入的目前的每个有向外接矩形的长短边长度，计算最小有向外接矩形的
    // 标号索引。
    __host__ int              // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    sdrComputeSDR(
            int cstcnt,       // 输入，点集中点的数量。
            BoundBox bbox[],  // 输入，找出的有向外接矩形的边界坐标信息。
            int *index        // 输出，计算出的最小有向外接矩形的标号索引。
    );

    // Host 成员方法：sdrComputeSDRCpu（计算有向外接矩形中面积最小的）
    // 根据输入的目前的每个有向外接矩形的长短边长度，计算最小有向外接矩形的
    // 标号索引。
    __host__ int              // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    sdrComputeSDRCpu(
            int cstcnt,       // 输入，点集中点的数量。
            BoundBox bbox[],  // 输入，找出的有向外接矩形的边界坐标信息。
            int *index        // 输出，计算出的最小有向外接矩形的标号索引。
    );

    // Host 成员方法：sdrParamOnConvex（求凸壳点集的最小有向外接矩形的参数）
    // 根据输入的凸壳点集，利用选择卡尺算法，找出点集的最小有向外接矩形的参数。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确
                                   // 执行，返回 NO_ERROR。
    sdrParamOnConvex(
            CoordiSet *convexcst,  // 输入凸壳点集。
            BoundBox *bbox,        // 有向外接矩形的边界点的坐标信息。
            RotationInfo *rotinfo  // 旋转信息。
    );

    // Host 成员方法：sdrParamOnConvexCpu（求凸壳点集的最小有向外接矩形的参数）
    // 根据输入的凸壳点集，利用选择卡尺算法，找出点集的最小有向外接矩形的参数。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确
                                   // 执行，返回 NO_ERROR。
    sdrParamOnConvexCpu(
            CoordiSet *convexcst,  // 输入凸壳点集。
            BoundBox *bbox,        // 有向外接矩形的边界点的坐标信息。
            RotationInfo *rotinfo  // 旋转信息。
    );

public:

    // 构造函数：SmallestDirRect
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    SmallestDirRect()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->imgCvt.clearAllConvertFlags();             // 清除所有的转换标志位
        this->value = 128;                               // 设置转换阈值。
        this->imgCvt.setConvertFlags(this->value, 255);  // 置位某范围
                                                         // 内的转换标志位。
    }

    // 成员方法：getValue（获取图像转换点集的像素阈值）
    // 获取图像转换点集的像素阈值。
    __host__ __device__ unsigned char  // 返回值：图像转换点集的像素阈值。
    getValue() const
    {
        // 返回图像转换点集的像素阈值。
        return this->value;    
    } 

    // 成员方法：setValue（设置图像转换点集的像素阈值）
    // 设置图像转换点集的像素阈值。
    __host__ __device__ int      // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    setValue(
            unsigned char value  // 设定新的图像转换点集的像素阈值。
    ) {
        // 根据阈值设置转换标志位。
        if (value < this->value)
            this->imgCvt.setConvertFlags(value, this->value - 1);
        else if (value > this->value)
            this->imgCvt.clearConvertFlags(this->value, value - 1);

        // 将图像转换点集的像素阈值赋成新值。
        this->value = value;

        return NO_ERROR;
    }

    // Host 成员方法：smallestDirRect（求像素值给定的对象的最小有向外接矩形）
    // 根据给定的阈值在图像中找出对象，求出对象的凸壳，找出对象的最小有向
    // 外接矩形
    __host__ int                  // 返回值：函数是否正确执行，若函数正确
                                  // 执行，返回 NO_ERROR。
    smallestDirRect(
            Image *inimg,         // 输入图像。
            Quadrangle *outrect,  // 有向外接矩形。
            bool hostrect = true  // 判断 outrect 是否是 Host 端的指针，
                                  // 默认为“是”。
    );

    // Host 成员方法：smallestDirRectCpu（求像素值给定的对象的最小有向外接矩形）
    // 根据给定的阈值在图像中找出对象，求出对象的凸壳，找出对象的最小有向
    // 外接矩形
    __host__ int                  // 返回值：函数是否正确执行，若函数正确
                                  // 执行，返回 NO_ERROR。
    smallestDirRectCpu(
            Image *inimg,         // 输入图像。
            Quadrangle *outrect,  // 有向外接矩形。
            bool hostrect = true  // 判断 outrect 是否是 Host 端的指针，
                                  // 默认为“是”。
    );

    // Host 成员方法：smallestDirRect（求像素值给定的对象的最小有向外接矩形）
    // 根据给定的阈值在图像中找出对象，求出对象的凸壳，找出对象的最小有向
    // 外接矩形。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确
                                    // 执行，返回 NO_ERROR。
    smallestDirRect(
            Image *inimg,           // 输入图像。
            DirectedRect *outrect,  // 有向外接矩形。
            bool hostrect = true    // 判断 outrect 是否是 Host 端的指针，
                                    // 默认为“是”。
    );

    // Host 成员方法：smallestDirRectCpu（求像素值给定的对象的最小有向外接矩形）
    // 根据给定的阈值在图像中找出对象，求出对象的凸壳，找出对象的最小有向
    // 外接矩形。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确
                                    // 执行，返回 NO_ERROR。
    smallestDirRectCpu(
            Image *inimg,           // 输入图像。
            DirectedRect *outrect,  // 有向外接矩形。
            bool hostrect = true    // 判断 outrect 是否是 Host 端的指针，
                                    // 默认为“是”。
    );

    // Host 成员方法：smallestDirRect（求给定点集的最小有向外接矩形）
    // 根据输入的点集，求出点集的凸壳，找出点集的最小有向外接矩形。
    __host__ int                  // 返回值：函数是否正确执行，若函数正确执行，
                                  // 返回 NO_ERROR。
    smallestDirRect(
            CoordiSet *cst,       // 输入点集。
            Quadrangle *outrect,  // 有向外接矩形。
            bool hostrect = true  // 判断 outrect 是否是 Host 端的指针，
                                  // 默认为“是”。
    );

    // Host 成员方法：smallestDirRectCpu（求给定点集的最小有向外接矩形）
    // 根据输入的点集，求出点集的凸壳，找出点集的最小有向外接矩形。
    __host__ int                  // 返回值：函数是否正确执行，若函数正确执行，
                                  // 返回 NO_ERROR。
    smallestDirRectCpu(
            CoordiSet *cst,       // 输入点集。
            Quadrangle *outrect,  // 有向外接矩形。
            bool hostrect = true  // 判断 outrect 是否是 Host 端的指针，
                                  // 默认为“是”。
    );

    // Host 成员方法：smallestDirRect（求给定点集的最小有向外接矩形）
    // 根据输入的点集，求出点集的凸壳，找出点集的最小有向外接矩形。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确执行
                                    // 返回 NO_ERROR。
    smallestDirRect(
            CoordiSet *cst,         // 输入点集。
            DirectedRect *outrect,  // 有向外接矩形。
            bool hostrect = true    // 判断 outrect 是否是 Host 端的指针，
                                    // 默认为“是”。
    );

    // Host 成员方法：smallestDirRectCpu（求给定点集的最小有向外接矩形）
    // 根据输入的点集，求出点集的凸壳，找出点集的最小有向外接矩形。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确执行
                                    // 返回 NO_ERROR。
    smallestDirRectCpu(
            CoordiSet *cst,         // 输入点集。
            DirectedRect *outrect,  // 有向外接矩形。
            bool hostrect = true    // 判断 outrect 是否是 Host 端的指针，
                                    // 默认为“是”。
    );

    // Host 成员方法：smallestDirRectOnConvex（求凸壳点集的最小有向外接矩形）
    // 根据输入的凸壳点集，利用选择卡尺算法，找出点集的最小有向外接矩形。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确执行
                                   // 返回 NO_ERROR。
    smallestDirRectOnConvex(
            CoordiSet *convexcst,  // 输入凸壳点集。
            Quadrangle *outrect,   // 有向外接矩形。
            bool hostrect = true   // 判断 outrect 是否是 Host 端的指针，
                                   // 默认为“是”。
    );

    // Host 成员方法：smallestDirRectCpuOnConvex（求凸壳点集的最小有向外接矩形）
    // 根据输入的凸壳点集，利用选择卡尺算法，找出点集的最小有向外接矩形。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确执行
                                   // 返回 NO_ERROR。
    smallestDirRectCpuOnConvex(
            CoordiSet *convexcst,  // 输入凸壳点集。
            Quadrangle *outrect,   // 有向外接矩形。
            bool hostrect = true   // 判断 outrect 是否是 Host 端的指针，
                                   // 默认为“是”。
    );

    // Host 成员方法：smallestDirRectOnConvex（求凸壳点集的最小有向外接矩形）
    // 根据输入的凸壳点集，利用选择卡尺算法，找出点集的最小有向外接矩形。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确执行
                                    // 返回 NO_ERROR。
    smallestDirRectOnConvex(
            CoordiSet *convexcst,   // 输入凸壳点集。
            DirectedRect *outrect,  // 有向外接矩形。
            bool hostrect = true    // 判断 outrect 是否是 Host 端的指针，
                                    // 默认为“是”。
    );

    // Host 成员方法：smallestDirRectCpuOnConvex（求凸壳点集的最小有向外接矩形）
    // 根据输入的凸壳点集，利用选择卡尺算法，找出点集的最小有向外接矩形。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确执行
                                    // 返回 NO_ERROR。
    smallestDirRectCpuOnConvex(
            CoordiSet *convexcst,   // 输入凸壳点集。
            DirectedRect *outrect,  // 有向外接矩形。
            bool hostrect = true    // 判断 outrect 是否是 Host 端的指针，
                                    // 默认为“是”。
    );
};

#endif

