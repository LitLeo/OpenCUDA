// ClusterLocalGray.h
// 创建人：杨伟光
//
// 图像分类降噪平滑算法（ClusterLocalGray）
// 功能说明：对灰度图像进行分类处理，使高像素更亮，低像素更暗，中间值更平庸。
// 有图像平滑和消除毛刺、噪声的效果。

// 修订历史：
// 2014年08月21日（杨伟光）
//     初始版本。
// 2014年08月30日（杨伟光）
//     功能完成后第一次规范代码，初始版本完成。 
// 2014年09月17日（杨伟光）
//     代码注释进行了一些完善。
// 2014年09月21日（杨伟光）
//      对代码逻辑进行了改进，速度提升了一倍多。
// 2014年09月25日（杨伟光）
//      根据河边老师要求替换了一处判断逻辑。
// 2014年09月28日（杨伟光）
//      修改了几处注释。

#ifndef __CLUSTERLOCALGRAY_H__
#define __CLUSTERLOCALGRAY_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：ClusterLocalGray（图像分类降噪平滑算法）
// 继承自：无。
// 实现了图像分类降噪平滑算法，根据当前像素点周围一定领域内像素点
// 的特征，依据不同的输入参数对图像进行不同程度的分类降噪平滑处理。
class ClusterLocalGray {                                                                

protected:

    // 成员变量：neighborsSideSpan（单像素点处理范围的宽度）
    // 以像素点为中心，上下左右 neighborsSideSpan 宽度的范围为处理范围，
    // 值域为 0 - 15。
    unsigned char neighborsSideSpan;

    // 成员变量：hGrayPercentTh（高像素比例）
    // 统计像素点处理范围内点的像素值，使 hGrayPercentTh 比例内的点更亮。
    // 建议值域为 5 - 30。
    unsigned char hGrayPercentTh;

    // 成员变量：lGrayPercentTh（低像素比例）
    // 统计像素点处理范围内点的像素值，使 lGrayPercentTh 比例内的点更暗。
    // 建议值域为 5 - 30。
    unsigned char lGrayPercentTh;

    // 成员变量：grayGapTh（临界值）
    // 根据 hGrayPercentTh 和 hGrayPercentTh 计算出平均高和平均低，然后
    // 判断跟临界值的关系从而进行相应的操作。建议值域为 50 - 150。
    unsigned char grayGapTh;
    
public:

    // 构造函数：ClusterLocalGray
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    ClusterLocalGray()
    {
        this->neighborsSideSpan = 2;  // 单像素点处理范围的宽度 默认为 2。
        this->hGrayPercentTh = 20;    // 高像素比例默认为 20。
        this->lGrayPercentTh = 20;    // 低像素比例默认为 20。
    }
    
    // 构造函数：ClusterLocalGray
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    ClusterLocalGray(
            unsigned char neighborsSideSpan,
            unsigned char hGrayPercentTh,
            unsigned char lGrayPercentTh
    ) {
        this->neighborsSideSpan = 2;  // 单像素点处理范围的宽度 默认为 2。
        this->hGrayPercentTh = 20;    // 高像素比例默认为 20。
        this->lGrayPercentTh = 20;    // 低像素比例默认为 20。

        // 根据参数列表中的值设定成员变量的初值。
        this->setNeighborsSideSpan(neighborsSideSpan);
        this->setHGrayPercentTh(hGrayPercentTh);
        this->setLGrayPercentTh(lGrayPercentTh);
    }

    // 成员函数：getNeighborsSideSpan（获取单像素点处理范围的宽度的值）
    // 获取成员变量 neighborsSideSpan 的值。
    __host__ __device__ unsigned char    // 返回值：返回 neighborsSideSpan 的值。
    getNeighborsSideSpan() const
    {
        // 返回单像素点处理范围的宽度的值。
        return this->neighborsSideSpan;  
    }

    // 成员函数：setNeighborsSideSpan（设定单像素点处理范围的宽度的值）
    // 设定成员变量 neighborsSideSpan 的值。
    __host__ __device__ int                  // 返回值：若函数正确执行，
                                             // 返回 NO_ERROR。
    setNeighborsSideSpan(
            unsigned char neighborsSideSpan  // 单像素点处理范围的宽度。
    ) {
        // 判断设定值是否大于值域范围。
        if (neighborsSideSpan >= 16)
            return INVALID_DATA;

        // 设定 neighborsSideSpan 的值。
        this->neighborsSideSpan = neighborsSideSpan;
        
        return NO_ERROR;
    }

    // 成员函数：getHGrayPercentTh（获取高像素比例的值）
    // 获取成员变量 hGrayPercentTh 的值。
    __host__ __device__ unsigned char  // 返回值：返回 hGrayPercentTh 的值。
    getHGrayPercentTh() const
    { 
        // 返回成员变量 hGrayPercentTh 的值。
        return this->hGrayPercentTh; 
    }

    // 成员函数：setHGrayPercentTh（设置高像素比例的值）
    // 设定成员变量 hGrayPercentTh 的值。
    __host__ __device__ int               // 返回值：若函数正确执行，返回 
                                          // NO_ERROR。
    setHGrayPercentTh(
            unsigned char hGrayPercentTh  // 高像素比例。
    ) {
        // 设定成员变量 hGrayPercentTh 的值。
        this->hGrayPercentTh = hGrayPercentTh;                                          

        return NO_ERROR;
    }

    // 成员函数：getLGrayPercentTh（获取低像素比例的值）
    // 获取成员变量 lGrayPercentTh 的值。
    __host__ __device__ unsigned char  // 返回值：返回 lGrayPercentTh 的值。
    getLGrayPercentTh() const
    { 
        // 返回成员变量 lGrayPercentTh 的值。
        return this->lGrayPercentTh; 
    }

    // 成员函数：setLGrayPercentTh（设置低像素比例的值）
    // 设定成员变量 lGrayPercentTh 的值。
    __host__ __device__ int               // 返回值：若函数正确执行，返回 
                                          // NO_ERROR。
    setLGrayPercentTh(
            unsigned char lGrayPercentTh  // 低像素比例。
    ) {
        // 设定成员变量 lGrayPercentTh 的值。
        this->lGrayPercentTh = lGrayPercentTh;

        return NO_ERROR;
    }

    // 成员函数：getGrayGapTh（获取临界值的值）
    // 获取成员变量 grayGapTh 的值。
    __host__ __device__ unsigned char  // 返回值：返回 grayGapTh 的值。
    getGrayGapTh() const
    { 
        // 返回成员变量 grayGapTh 的值。
        return this->grayGapTh; 
    }

    // 成员函数：setGrayGapTh（设定临界值的值）
    // 设定成员变量 grayGapTh 的值。
    __host__ __device__ int          // 返回值：若函数正确执行，返回 NO_ERROR。
    setGrayGapTh(
            unsigned char grayGapTh  // 临界值
    ) {
        // 设定成员变量 grayGapTh 的值。
        this->grayGapTh = grayGapTh;

        return NO_ERROR;
    }

    // 成员方法：clusterLocalGray（图像分类降噪处理）
    // 实现了图像分类降噪平滑算法。主要思想为根据当前像素点周围一定领域
    // （领域大小由 neighborsSideSpan 决定）内像素点的特征，对像素点进行统计，
    // 再依据不同的输入参数（高像素点比例和低像素点比例）对图像进行不同程度的
    // 分类降噪平滑处理。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    clusterLocalGray(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
};

#endif


