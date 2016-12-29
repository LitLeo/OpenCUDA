// HoughCircle.h 
// 创建人：侯怡婷
//
// Hough变换检测圆（HoughCircle）
// 功能说明：实现 Hough 变换检测圆。输入参数为坐标集 guidingset 或者输
//           入图像 inimg，若 guidingset 不为空，则只处理该坐标集;若为
//           空，则对图像 inimg 进行圆检测。
//
// 修订历史：
// 2012年12月12日（侯怡婷）
//     初始版本。
// 2012年12月28日（侯怡婷）
//     基本完成的 HoughCircle 算法，初始版本。
// 2012年12月30日（侯怡婷）
//     改善 HoughCircle 算法。
// 2012年12月31日（侯怡婷）
//     完善注释规范。
// 2013年01月02日（侯怡婷）
//     改善计算得票数时的方法，缩小了每个点的查找范围。
// 2013年01月06日（侯怡婷）
//     加入处理坐标点集的算法。
// 2013年01月14日（侯怡婷）
//     修改确定最终圆的个数的方法。
// 2013年03月20日（侯怡婷）
//     修改代码中一处 bug。
// 2013年04月13日（侯怡婷）
//     将寻找输入坐标集横纵坐标最小、最大值的过程改成并行算法,
//     并尽量合并代码，减少冗余。

#ifndef __HOUGHCIRCLE_H__
#define __HOUGHCIRCLE_H__

#include "Image.h"
#include "CoordiSet.h"
#include "ErrorCode.h"
#include "ImgConvert.h"

// 结构体：CircleParam（圆返回参数）
// 将检测到圆的参数：圆心的坐标，圆的半径等定义为结构体，作为函数最终
// 输出结果。
typedef struct CircleParam_st {
    int a;       // 圆心的横坐标。
    int b;       // 圆心的纵坐标。
    int radius;  // 圆的半径。
    int votes;   // 得票数，即 BufHough 矩阵中的数据。
} CircleParam;

// 类：HoughLCircle
// 继承自：无
// 实现 Hough 变换检测圆。输入参数为坐标集 guidingset 或者输入图像 inimg，
// 若 guidingset 不为空，则只处理该坐标集；若为空，则对图像 inimg 进行圆检测。
class HoughCircle {

protected:
    
    // 成员变量： minradius（最小检测的圆半径）
    // 描述在检测圆时，最小的圆半径。
    int radiusMin;
    
    // 成员变量： maxradius（最小检测的圆半径）
    // 描述在检测圆时，最大的圆半径。
    int radiusMax;

    // 成员变量：distMin（判定相似圆的圆心之间距离阈值）
    // 用于明显区分两个不同圆之间的最小距离。
    float distThres;

    // 成员变量：radius（判定相似圆的半径相差阈值）
    // 用于区分不同圆的最小半径阈值
    float rThres;

    // 成员变量：cirThreshold（圆形阈值）
    // 若累加器中相应的累计值大于该参数，则认为是一个圆。
    int cirThreshold;

public:
    // 构造函数：HoughCircle
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
        HoughCircle()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->radiusMin = 5;       // 最小检测的圆半径为 5。
        this->radiusMax = 50;      // 最大小检测的圆半径为 50。
        this->distThres = 50.0f;     // 两个圆之间的最小距离为 50.0f。
        this->rThres=20;
        this->cirThreshold = 100;  // 圆的阈值为100。
    }

    // 构造函数：HoughCircle
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
        HoughCircle(
        int radiusMin,    // 最小检测的圆半径。
        int radiusMax,    // 最小检测的圆半径。
        float distMin,    // 两个圆之间的最小距离。
        int rThres,
        int cirThreshold  // 圆的阈值
        ) {
            // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中
            // 给了非法的初始值而使系统进入一个未知的状态。
            this->radiusMin = 5;       // 最小检测的圆半径为 5。
            this->radiusMax = 50;      // 最大检测的圆半径为 50。 
            this->distThres = 50.0f;     // 两个圆之间的最小距离为 50.0f。
            this->rThres=20;
            this->cirThreshold = 100;  // 圆的阈值为100。

            // 根据参数列表中的值设定成员变量的初值
            setRadiusMin(radiusMin);
            setRadiusMax(radiusMax);
            setDistThres(distMin);
            setRThres(rThres);
            setCirThreshold(cirThreshold);
    }

    // 成员方法：getRadiusMin（获取最小检测的圆半径）
    // 获取成员变量 radiusMin 的值。
    __host__ __device__ int  // 返回值：成员变量 radiusMin 的值
        getRadiusMin() const
    {
        // 返回 radiusMin 成员变量的值。
        return this->radiusMin;
    } 

    // 成员方法：setRadiusMin（设置最小检测的圆半径）
    // 设置成员变量 radiusMin 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
        // 行，返回 NO_ERROR。
        setRadiusMin(
        int radiusMin    // 设定新的最小检测的圆半径
        ) { 
            // 检查 radiusMin 的合法性。
            if (radiusMin < 0)
                return INVALID_DATA;
            // 将 radiusMin 成员变量赋成新值
            this->radiusMin = radiusMin;

            return NO_ERROR;
    }

    // 成员方法：getRadiusMax（获取最大检测的圆半径）
    // 获取成员变量 radiusMax 的值。
    __host__ __device__ int  // 返回值：成员变量 radiusMax 的值
        getRadiusMax() const
    {
        // 返回 radiusMax 成员变量的值。
        return this->radiusMax;
    } 

    // 成员方法：setRadiusMax（设置最大检测的圆半径）
    // 设置成员变量 radiusMax 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
        // 行，返回 NO_ERROR。
        setRadiusMax(
        int radiusMax    // 设定最大检测的圆半径
        ) { 
            // 检查 radiusMax 的合法性。
            if (radiusMax < 0)
                return INVALID_DATA;
            // 将 radiusMax 成员变量赋成新值
            this->radiusMax = radiusMax;

            return NO_ERROR;
    }

    // 成员方法：getDistMin（获取两个不同圆之间的最小距离）
    // 获取成员变量 distMin 的值。
    __host__ __device__ float  // 返回值：成员变量 distMin 的值
        getDistThres() const
    {
        // 返回 distMin 成员变量的值。
        return this->distThres;
    }  

    // 成员方法：setDistMin（设置两个不同圆之间的最小距离）
    // 设置成员变量 distMin 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
        // 行，返回 NO_ERROR。
        setDistThres(
        float dist       // 设定两个不同圆之间的最小距离
        ) {
            // 将 distMin 成员变量赋成新值
            this->distThres=dist;

            return NO_ERROR;
    }
    // 成员方法：getDistMin（获取两个不同圆之间的最小距离）
    // 获取成员变量 distMin 的值。
    __host__ __device__ int  // 返回值：成员变量 distMin 的值
        getRThres() const
    {
        // 返回 distMin 成员变量的值。
        return this->rThres;
    }  

    // 成员方法：setDistMin（设置评定相似园的半径差别值）
    // 设置成员变量 rThresMin 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
        // 行，返回 NO_ERROR。
        setRThres(
        int  r        // 设定两个不同圆之间的最小距离
        ) {
            this->rThres=r;
            return NO_ERROR;
    }
    // 成员方法：getCirThreshold（获取圆的阈值）
    // 获取成员变量 cirThreshold 的值。
    __host__ __device__ int	// 返回值：成员变量 cirThreshold 的值
        getCirThreshold() const
    {
        // 返回 cirThreshold 成员变量的值。
        return this->cirThreshold;
    }

    // 成员方法：setCirThreshold（设置圆的阈值）
    // 设置成员变量 cirThreshold 的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执
        // 行，返回 NO_ERROR。
        setCirThreshold(
        int cirThreshold  // 设定新的圆的阈值
        ) {
            // 检查 cirThreshold 的合法性。
            if (cirThreshold < 0)
                return INVALID_DATA;
            // 将 cirThreshold 成员变量赋成新值
            this->cirThreshold = cirThreshold;

            return NO_ERROR;
    }

    
    // Host 成员方法：houghline（Hough 变换检测圆）
    // 若输入的坐标集 guidingset 不为空，则只处理该坐标集；若该坐标集为空，则 
    // Hough 变换就处理输入图像的 ROI 区域，并把最终检测圆的结果返回
    // 到定义的参数结构体中，并且返回检测到的圆的数量。输入图像可以为空。
    __host__ int                      // 返回值：函数是否正确执行，若函数
                                      // 正确执行，返回NO_ERROR。
    houghcircle(
            Image *inimg,             // 输入图像
            CoordiSet *guidingset,    // 输入坐标集
            int *circleMax,           // 检测圆的最大数量
            CircleParam *circleparam, // 圆返回参数结构体
            bool writetofile          // 是否需要将检测结果写入图像文件result.bmp中
    );
    // Host 成员方法：pieceCircle(分片检测inimg中的圆形，放入数组返回)
    __host__ int 
    pieceCircle(
        Image *inimg,                   // 输入待检测的图形
        int piecenum,                   // 每个维度上分块数量
        int *circleMax,                 // 返回检测到的圆数量
        CircleParam *circleparam,       // 返回检测到的圆参数
        bool writetofile                // 是否把检测结果写到文件中
        );

};

#endif

