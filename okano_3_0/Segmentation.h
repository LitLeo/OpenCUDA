// Segmentation.h
// 创建人：邱孝兵
//
// 二分类（Segmentation）
// 功能说明：SmoothVector 处理过程完了后，生成了一个结果 Vector 集合(收束点集合)
// 针对该收束点集合，根据给定的范围限定数据结构（包含：坐标范围，range 范围，
// 向量，偏角三个指标），由种子点出发对它们进行分割。
//
// 修订历史：
// 2012年12月14日（邱孝兵）
//     初始版本
// 2012年12月20日（邱孝兵）
//     删除 lvl1 和 lvl2 两个无用的类变量
// 2013年01月03日（邱孝兵）
//     完成 _countW1Ker 和 _labelVectorsKer 两个核函数
// 2013年01月04日（邱孝兵）
//     完成 _countAppointW1Ker 和 _countAppointW1Ker 两个核函数

#include "Image.h"
#include "ErrorCode.h"
#include "FeatureVecArray.h"

#ifndef __SEGMENTATION_H__
#define __SEGMENTATION_H__

// 结构体：BandWidth（分割的时候所依据的范围）
// 该结构体中定义了，在对收束点集合进行二分类的时候所依据的范围，
// 包括坐标范围，特征值范围和向量偏角范围三个指标
typedef struct BandWidth_st {
    float spaceWidth;  // 坐标范围，向量坐标的欧式距离范围
    float rangeWidth;  // 特征值范围，三维特征值向量的欧式距离范围
    float angleWidth;  // 向量偏角范围，以 cos 值度量
} BandWidth;

// 类：Segmentation
// 继承自：无
// SmoothVector 处理过程完了后，生成了一个结果 Vector 集合(收束点集合)
// 针对该收束点集合，根据给定的范围限定数据结构（包含：坐标范围，range 范围，
// 向量，偏角三个指标），由种子点出发对它们进行分割。
class Segmentation {

protected:

    // 成员变量：alpha（外部指定系数）
    // 根据相邻点的 label 判定某个点的 class 时需要用到个一个参数。
    float alpha;        
    
    // 成员变量：beta（外部指定系数）
    // 根据相邻点的 label 判定某个点的 class 时需要用到个一个参数。
    float beta;

    // 成员变量：bw1（寻找种子点的范围限定）
    // 分类时根据该值寻找种子点
    BandWidth bw1;

    // 成员变量：bw2（分类时使用的范围限定）
    // 分类时需要使用该值。
    BandWidth bw2;

public:

    // 构造函数：Segmentation
    // 无参数版本的构造函数，所有成员变量均初始化为默认值
    __host__ __device__
    Segmentation() {
        // 无参数的构造函数，使用默认值初始化各个变量
        this->alpha = 1;
        this->beta = 1;       
        this->bw1.spaceWidth = 100;
        this->bw1.rangeWidth = 0.5;
        this->bw1.angleWidth = 0.5;
        this->bw2.spaceWidth = 100;
        this->bw2.rangeWidth = 0.5;
        this->bw2.angleWidth = 0.5;
    }
    
    // 构造函数：Segmentation
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Segmentation(
            float alpha,        // 外部参数 alpha
            float beta,         // 外部参数 beta
            BandWidth bw1,      // 寻找 class - 1 的种子点的范围
            BandWidth bw2       // 分类时使用的范围
    ) {
        // 使用默认值初始化各个变量
        this->alpha = 1;
        this->beta = 1;       
        this->bw1.spaceWidth = 100;
        this->bw1.rangeWidth = 0.5;
        this->bw1.angleWidth = 0.5;
        this->bw2.spaceWidth = 100;
        this->bw2.rangeWidth = 0.5;
        this->bw2.angleWidth = 0.5;

        // 调用 seters 给各个成员变量赋值
        this->setAlpha(alpha);
        this->setBeta(beta);
        this->setBw1(bw1);
        this->setBw2(bw2);
    }

    // 成员方法：getAlpha （获取 alpha 的值）
    // 获取成员变量  alpha 的值
    __host__ __device__ float  // 返回值：成员变量 alpha 的值
    getAlpha() const
    {
        // 返回成员变量 alpha 的值
        return this->alpha;
    }
    
    // 成员方法：setAlpha（设置 alpha 的值）
    // 设置成员变量 alpha 的值
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setAlpha(
            float alpha      // 外部参数 alpha
    ) {
        // 设置成员变量 beta 的值
        this->alpha = alpha;
        
        return NO_ERROR;
    }
    
    // 成员方法：getBeta （获取 beta 的值）
    // 获取成员变量  beta 的值
    __host__ __device__ float  // 返回值：成员变量 beta 的值
    getBeta() const
    {
        // 返回成员变量 beta 的值
        return this->beta;
    }
    
    // 成员方法：setBeta（设置 beta 的值）
    // 设置成员变量 beta 的值
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setBeta(
            float beta       // 外部参数 beta
    ) {
        // 设置成员变量 beta 的值
        this->beta = beta;
        
        return NO_ERROR;
    }    

    // 成员方法：getBw1（ 获取 bw1 的值）
    // 设置成员变量 bw1 的值
    __host__ __device__ BandWidth  // 返回值，成员变量 bw1 的值
    getBw1() const
    {
        // 返回成员变量 bw1 的值
        return this->bw1;
    }

    // 成员方法：setBw1（设置 bw1 的值）
    // 设置成员变量 bw1 的值
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setBw1(
            BandWidth bw1    // 用于寻找种子点的范围
    ) {
        // 设置成员变量 bw1 的值
        this->bw1.spaceWidth = bw1.spaceWidth;
        this->bw1.rangeWidth = bw1.rangeWidth;
        this->bw1.angleWidth = bw1.angleWidth;

        return NO_ERROR;
    }
    
    // 成员方法：getBw2（ 获取 bw2 的值）
    // 设置成员变量 bw2 的值
    __host__ __device__ BandWidth  // 返回值，成员变量 bw2 的值
    getBw2() const
    {
        // 返回成员变量 bw2 的值
        return this->bw2;
    }

    // 成员方法：setBw2（设置 bw2 的值）
    // 设置成员变量 bw2 的值
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setBw2(
            BandWidth bw2    // 用于分类的范围
    ) {
        // 设置成员变量 bw1 的值
        this->bw2.spaceWidth = bw2.spaceWidth;
        this->bw2.rangeWidth = bw2.rangeWidth;
        this->bw2.angleWidth = bw2.angleWidth;

        return NO_ERROR;
    }

    // 成员函数：segregate（对收束点集进行二分类）
    // SmoothVector 处理过程完了后，生成了一个结果 Vector 集合(收束点集合)
    // 针对该收束点集合，根据给定的范围限定数据结构（包含：坐标范围，
    // range 范围，向量，偏角三个指标），由种子点出发对它们进行二分类。
    __host__ int                           // 返回值：函数是否正确执行，若函数
                                           // 正确执行，返回 NO_ERROR。
    segregate(
        FeatureVecArray *featurevecarray,  // 输入的收束点集
        int *outlabel                      // 输出的标记数组
    );
};

#endif