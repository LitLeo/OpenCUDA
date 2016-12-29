// GeometryProperties.h
// 创建人：刘宇
//
// 几何形状方向等的测度（GeometryProperties）
// 功能说明：计算几何形状的一些特征属性。包括 5 类属性，分别是
//          （1）边缘平滑度的计算;（2）直线的线性度计算；（3）圆度的计算；
//          （4）凸面度计算；（5）分布中心和方向的计算。
//
// 修订历史：
// 2012年10月24日（刘宇）
//     初始版本
// 2012年10月25日（于玉龙、刘宇）
//     修改代码规范
// 2012年11月13日（刘宇）
//     在核函数执行后添加 cudaGetLastError 判断语句
// 2012年11月23日（刘宇）
//     添加输入输出参数的空指针判断
// 2012年12月25日（刘宇）
//     完成直线的线性度计算和分布方向中心的计算
// 2012年12月30日（刘宇）
//     完成边缘平滑度计算
// 2013年03月15日（刘宇）
//     完成凸面度计算
// 2013年03月21日（刘宇）
//     完成圆度计算

#ifndef __GEOMETRYPROPERTIES_H__
#define __GEOMETRYPROPERTIES_H__

#include "Image.h"
#include "Template.h"
#include "CoordiSet.h"
#include "ErrorCode.h"
#include "Rectangle.h"
#include "HoughLine.h"
#include "Histogram.h"
#include "RobustEdgeDetection.h"
#include "Thinning.h"
#include "Moments.h"
#include "Morphology.h"
#include "ConvexHull.h"
#include "SmallestDirRect.h"

// 结构体：RegionRound（区域的圆度测度）
// 该结构体定了区域圆度的数据结构，其中包含了 6 种圆度测量，分别从不同侧面反映
// 了区域的圆度属性。
typedef struct RegionRound_st {
    int pointcount;         // 区域或曲线上的点数
    float anisometry;       // 各向异性的
    float bulkiness;        // 蓬松度
    float structurefactor;  // 结构因子
    float circularity;      // 圆环度
    float compactness;      // 紧密度
} RegionRound;

// 类：GeometryProperties
// 继承自：无
// 计算几何形状的一些特征属性。包括 5 类属性，分别是
// （1）边缘平滑度的计算；（2）直线的线性度计算；
// （3）圆度的计算；（4）凸面度计算；（5）分布中心和方向的计算。
class GeometryProperties {

protected:

    // 成员变量：highPixel（高像素）
    // 二值图像中轮廓区域的像素值。
    unsigned char highPixel;
    
    // 成员变量：lowPixel（低像素）
    // 二值图像中背景部分的像素值。
    unsigned char lowPixel;
    
    // 成员变量：smoothWidth（边缘平滑宽度）
    // 计算边缘平滑度时，指定的平滑邻域的宽度，邻域的点的总个数等于 
    // 2 * smooththred + 1。
    int smoothWidth;

    // 成员变量：smooththred（边缘平滑度的阈值大小）
    // 当边缘平滑度大于阈值大小时，设置 errmap 图像。
    float smooththred;

    // 成员变量：linethred（直线线性度的阈值大小）
    // 当直线线性度小于阈值大小时，设置 errmap 图像。
    float linethred;

    // 静态成员成员：tpl（模版）
    // 长度为 12 的模板。因为以每个像素为中心，有 4 个方向的 3 邻域。
    static Template *tpl;

    // Host 静态方法：initTemplate（在算法操作前初始化模版）
    // 对模版中的数据进行赋值，并将模版拷贝到 Device 端。
    static __host__ int  // 返回值：函数是否正确执行，若正确执行，返回
                         // NO_ERROR 
    initTemplate();

    // 成员变量：isconst(乘积项标识)
    // 如果 isconst 等于 true，则在计算几何矩时乘积项恒等于 1；否则等于
    // 正常的灰度值。
    bool isconst;

    // 形态学类声明。
    Morphology mo;
 
    // 边缘检测类声明。
    RobustEdgeDetection edgedete;
    
    // 图像细化类声明。
    Thinning thin;

    // 直线检测类声明。
    HoughLine hline;
    LineParam lineparam;

    // 凸壳类声明。
    ConvexHull cvhull;

    // 最小包围盒类声明。
    SmallestDirRect smallrect;
    DirectedRect outrect;

    // 直方图类声明。
    Histogram hist;

    // 几何矩类声明。
    Moments mom;

public:

    // 构造函数：GeometryProperties
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    GeometryProperties()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->highPixel = 255;     // 高像素值默认为 255。
        this->lowPixel = 0;        // 低像素值默认为 0。
        this->smoothWidth = 5;     // 边缘平滑宽度设为 5。
        this->smooththred = 1.0f;  // 边缘平滑度的阈值设为 1。
        this->linethred = 1.0f;    // 直线线性度的阈值设为 1。
        this->isconst = false;     // 图像的灰度值标识默认为 false。
    }

    // 构造函数：GeometryProperties
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    GeometryProperties(
            unsigned int highpixel,
            unsigned int lowpixel,
            int smoothwidth,
            float smooththred,
            float linethred,
            bool isconst
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->highPixel = 255;     // 高像素值默认为 255。
        this->lowPixel = 0;        // 低像素值默认为 0。
        this->smoothWidth = 5;     // 边缘平滑宽度设为 5。
        this->smooththred = 1.0f;  // 边缘平滑度的阈值设为 1。
        this->linethred = 1.0f;    // 直线线性度的阈值设为 1。
        this->isconst = false;     // 图像的灰度值标识默认为 false。

        setHighLowPixel(highpixel, lowpixel);
        setSmoothWidth(smoothwidth);
        setSmooththred(smooththred);
        setLinethred(linethred);
        setIsconst(isconst);
    }

    // 成员函数：getHighPixel（获取高像素的值）
    // 获取成员变量 highPixel 的值。
    __host__ __device__ unsigned char  // 返回值：返回 hignPixel 的值。
    getHighPixel() const
    { 
        // 返回 highPixel 成员变量的值。
        return highPixel;   
    }

    // 成员函数：setHighPixel（设置高像素）
    // 设置成员变量 highPixel 的值。
    __host__ __device__ int          // 返回值：若函数正确执行，返回 NO_ERROR。
    setHighPixel(
            unsigned char highpixel  // 设定新的高像素的值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (highPixel == lowPixel)
            return INVALID_DATA;

        // 将 highPixel 成员变量赋成新值
        this->highPixel = highpixel;

        return NO_ERROR;
    }

    // 成员函数：getLowPixel（获取低像素的值）
    // 获取成员变量 lowPixel 的值。
    __host__ __device__ unsigned char  // 返回值：返回 lowPixel 的值。
    getLowPixel() const
    { 
        // 返回 lowPixel 成员变量的值。
        return lowPixel;   
    }

    // 成员函数：setLowPixel（设置低像素的值）
    // 设置成员变量 lowPixel 的值。
    __host__ __device__ int         // 返回值：若函数正确执行，返回 NO_ERROR。
    setLowPixel(
            unsigned char lowpixel  // 设定新的低像素的值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (highPixel == lowPixel)
            return INVALID_DATA;

        // 将 lowPixel 成员变量赋成新值
        this->lowPixel = lowpixel;

        return NO_ERROR;
    }

    // 成员函数：setHighLowPixel（设置高低像素）
    // 设置成员变量 highPixel 和 lowPixel 的值。
    __host__ __device__ int           // 返回值：函数正确执行，返回 NO_ERROR。
    setHighLowPixel(
            unsigned char highpixel,  // 设定新的高像素的值
            unsigned char lowpixel    // 设定新的低像素的值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (highPixel == lowPixel)
            return INVALID_DATA;

        // 将 highpixel 成员变量赋成新值
        this->highPixel = highpixel;

        // 将 lowPixel 成员变量赋成新值
        this->lowPixel = lowpixel;

        return NO_ERROR;
    }

    // 成员函数：getSmoothWidth（获取边缘平滑宽度）
    // 获取成员变量 smoothWidth 的值。
    __host__ __device__ int  // 返回值：返回 smoothWidth 的值。
    getSmoothWidth() const
    { 
        // 返回 smoothWidth 成员变量的值。
        return smoothWidth;   
    }

    // 成员函数：setSmoothWidth（设置边缘平滑宽度）
    // 设置成员变量 smoothWidth 的值。
    __host__ __device__ int  // 返回值：若函数正确执行，返回 NO_ERROR。
    setSmoothWidth(
            int smoothwidth  // 设定新的边缘平滑宽度
    ) {
        // 将 smoothWidth 成员变量赋成新值
        this->smoothWidth = smoothwidth;

        return NO_ERROR;
    }

    // 成员方法：getSmooththred（获取边缘平滑度的阈值大小）
    // 获取成员变量 smooththred 的值。
    __host__ __device__ float  // 返回值：成员变量 smooththred 的值
    getSmooththred() const
    {
        // 返回 smooththred 成员变量的值。
        return this->smooththred;
    }

    // 成员方法：setSmooththred（设置边缘平滑度的阈值大小）
    // 设置成员变量 smooththred 的值。
    __host__ __device__ int    // 返回值：函数是否正确执行，若函数正确执
                               // 行，返回 NO_ERROR。
    setSmooththred(
            float smooththred  // 设定新的边缘平滑度的阈值大小。
    ) {
        // 将 smooththred 成员变量赋成新值。
        this->smooththred = smooththred;

        return NO_ERROR;
    }

    // 成员方法：getLinethred（获取直线线性度的阈值大小）
    // 获取成员变量 linethred 的值。
    __host__ __device__ float  // 返回值：成员变量 linethred 的值
    getLinethred() const
    {
        // 返回 linethred 成员变量的值。
        return this->linethred;
    }

    // 成员方法：setLinethred（设置直线线性度的阈值大小）
    // 设置成员变量 linethred 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setLinethred(
            float linethred  // 设定新的直线线性度的阈值大小。
    ) {
        // 将 linethred 成员变量赋成新值。
        this->linethred = linethred;

        return NO_ERROR;
    }

    // 成员方法：getIsconst（获取乘积项标识）
    // 获取成员变量 isconst 的值。
    __host__ __device__ bool  // 返回值：成员变量 isconst 的值
    getIsconst() const;  

    // 成员方法：setIsconst（设置乘积项标识）
    // 设置成员变量 isconst 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setIsconst(
            bool isconst     // 设定新的乘积项标识
    );

    // Host 成员方法：edgeSmoothness（计算边缘的平滑度）
    // 首先对输入坐标集做膨胀和细化处理，去除孤立点。然后根据尺度平滑坐标点，
    // 计算平滑前后的平均误差度，如果大于指定阈值，则设置错误码图像。
    __host__ int                // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回 NO_ERROR。
    edgeSmoothness(
            CoordiSet *cdset,   // 输入坐标集
            float *smoothness,  // 输出形状边缘的平滑度
            Image *errmap       // 错误码图像
    );

    // Host 成员方法：calcLinearity（计算直线的线性度）
    // 对给定的坐标集，利用 Hough 变换计算其直线参数，然后计算各点到拟合直线的
    // 垂直距离，最终得到直线的线性度大小。
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    calcLinearity(
            CoordiSet *cdset,  // 输入坐标集
            float *linearity,  // 输出直线的线性度
            Image *errmap      // 错误码图像
    );

    // Host 成员方法：calcConvexity（计算形状的凸面度）
    // 对给定的坐标集，计算其凸壳包围的面积，然后和形状自身面积进行对比，得到
    // 凸面度大小。
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    calcConvexity(
            CoordiSet *cdset,  // 输入坐标集
            float *convexity   // 输出直线的线性度
    );

    // Host 成员方法：calcAnisometry（计算区域的各向异性）
    // 计算区域的各向异性，等于最小有向外接矩形的短径除以长径。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确
                                   // 执行，返回 NO_ERROR。
    calcAnisometry(
            DirectedRect outrect,  // 最小有向外接矩形
            RegionRound *round     // 圆度测度
    );

    // Host 成员方法：calcBulkiness（计算区域的蓬松度）
    // 计算区域的蓬松度，等于最小有向外接矩形的面积除以区域的实际面积。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确
                                   // 执行，返回 NO_ERROR。
    calcBulkiness(
            DirectedRect outrect,  // 最小有向外接矩形
            int area,              // 区域的实际面积
            RegionRound *round     // 圆度测度
    );

    // Host 成员方法：calcStructurefactor（计算区域的结构因子）
    // 计算区域的结构因子，等于最小有向外接矩形的长径平方除以区域的实际面积。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确
                                   // 执行，返回 NO_ERROR。
    calcStructurefactor(
            DirectedRect outrect,  // 最小有向外接矩形
            int area,              // 区域的实际面积
            RegionRound *round     // 圆度测度
    );

    // Host 成员方法：calcCircularity（计算区域的圆环度）
    // 计算区域的结圆环度。区域的实际面积除以区域边缘到几何中心的平均距离的
    // 平方。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确
                                   // 执行，返回 NO_ERROR。
    calcCircularity(
            Image *img,            // 输入图像
            DirectedRect outrect,  // 最小有向外接矩形
            RegionRound *round     // 圆度测度
    );

    // Host 成员方法：contourLength（计算轮廓的长度）
    // 计算轮廓的长度，此处的轮廓是由边缘细化得到的输出结果。轮廓长度的计算采用
    // 需求文档中的并行策略。对于图像上的每个非轮廓点，统计其 3 邻域模版中含有
    // 1 个， 2 个， 3 个轮廓点的情况。最后根据给出公式计算出轮廓的长度。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，
                           // 返回 NO_ERROR。
    contourLength(
            Image *img,    // 输入图像
            float *length  // 输出形状的紧密度
    );

    // Host 成员方法：calcCompactness（计算形状的紧密度）
    // 首先计算轮廓的长度，然后根据公式除以形状区域的面积，得到形状紧密度大小。
    __host__ int                // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回 NO_ERROR。
    calcCompactness(
            Image *img,         // 输入图像
            int area,           // 区域的实际面积
            RegionRound *round  // 圆度测度
    );

    // Host 成员方法：calcRoundness（计算区域的圆度）
    // 通过不同属性计算形状区域的圆度属性。包括：区域或曲线上的点数，各向异性的
    // ,蓬松度,结构因子,圆环度,紧密度
    __host__ int                // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回 NO_ERROR。
    calcRoundness(
            Image *img,         // 输入图像
            RegionRound *round  // 圆度测度
    );

    // Host 成员方法：calcCentOrient（计算形状的分布重心和方向）
    // 根据中心矩计算形状的分布中心和方向，调用 Moments 中方法。
    __host__ int                // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回 NO_ERROR。
    calcCentOrient(
            Image *img,          // 输入图像
            double centroid[2],  // 形状分布重心
            double *orientation  // 形状分布方向
    );
};

#endif

