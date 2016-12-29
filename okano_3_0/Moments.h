// Moments.h 
// 创建人：刘宇
//
// 几何矩（Moments）
// 功能说明：计算空间矩（spatial moments），中心矩（central moments），
//           以及 Hu 矩（Hu moments），affine moment invariants 等。
//
// 修订历史：
// 2012年10月19日（刘宇）
//     初始版本
// 2012年11月13日（刘宇）
//     在核函数执行后添加 cudaGetLastError 判断语句
// 2012年11月23日（刘宇）
//     添加输入输出参数的空指针判断
// 2013年7月14日（刘宇）
//     修正 Hu 矩一处符号错误
// 2013年9月12日（刘宇）
//     完成 affine moment invriants
// 2013年10月6日（刘宇）
//     完成 flusser moments

#ifndef __MOMENTS_H__
#define __MOMENTS_H__

#include "Image.h"


// 结构体：MomentSet（几何矩数据集合）
// 该结构体包含了空间矩，中心矩和 Hu 矩的所有数据。
typedef struct MomentSet_st {
    double m00, m10, m01, m20, m11, m02, m30, m21,          // 空间矩
           m12, m03, m22, m31, m13, m40, m04;   
    double mu00, mu10, mu01, mu20, mu11, mu02, mu30, mu21, 
           mu12, mu03, mu22, mu31, mu13, mu40, mu04;        // 中心矩
    double hu1, hu2, hu3, hu4, hu5, hu6, hu7, hu8;          // Hu 矩
    double ami1, ami2, ami3, ami4, ami6, ami7,              // Affine Moments
           ami8, ami9, ami19; 
    double flu1, flu2, flu3, flu4, flu5, flu6, flu7,        // Fluuser Moments
           flu8, flu9, flu10, flu11;
} MomentSet;


// 类：Moments
// 继承自：无
// 利用差分矩因子算法计算几何矩。差分求和定理：两个离散函数数组的乘积等于
// 将其中一个差分、另一个累进求和后的乘积。基于该定理对图像计算矩，将矩做
// 为一个离散函数数组对其差分，将图像函数作为另一个离散数组对其累进求和。
// 由于矩因子数组差分一次或多次后，除边界附近点外，其余元素值皆为 0，这样
// 本需对所有数组元素作相乘运算，变为只需对数组边界附近不为 0 的数组元素
// 作相乘运算，从而减少了大量的运算。根据算法需求，计算矩的分为两种情况，
// 一种是乘以正常的灰度值，一种是设置乘积项恒等于 1。
// 参考文献《快速不变矩算法基于 CUDA 的并行实现》，计算机应用，2010年07月。
class Moments {

protected:

    // 成员变量：isconst(乘积项标识)
    // 如果 isconst 等于 true，则在计算几何矩时乘积项恒等于 1；否则等于
    // 正常的灰度值。
    bool isconst;

    // 成员变量：adjustcenter(调整中心标识)
    // 如果 adjustcenter 等于 true，则调整中心坐标；否则使用
    // 原始的中心坐标。
    bool adjustcenter;

public:

    // 构造函数：Moments
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Moments()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->isconst = false;       // 图像的灰度值标识默认为 false。
        this->adjustcenter = false;  // 调整中心标识默认为 false。
    }

    // 构造函数：Moments
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Moments(
            bool isconst,            // 乘积项标识
            bool adjustcenter        // 调整中心标识
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->isconst = false;       // 图像的灰度值标识默认为 false。
        this->adjustcenter = false;  // 调整中心标识默认为 false。

        // 根据参数列表中的值设定成员变量的初值
        setIsconst(isconst);
        setAdjustcenter(adjustcenter);
    }

    // 成员方法：getIsconst（获取乘积项标识）
    // 获取成员变量 isconst 的值。
    __host__ __device__ bool  // 返回值：成员变量 isconst 的值
    getIsconst() const
    {
        // 返回 isconst 成员变量的值。
        return this->isconst;
    }

    // 成员方法：setIsconst（设置乘积项标识）
    // 设置成员变量 isconst 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setIsconst(
            bool isconst     // 设定新的乘积项标识
    ) {
        // 将 isconst 成员变量赋成新值
        this->isconst = isconst;

        return NO_ERROR;
    }

    // 成员方法：getAdjustcenter（获取调整中心标识）
    // 获取成员变量 isconst 的值。
    __host__ __device__ bool  // 返回值：成员变量 adjustcenter 的值
    getAdjustcenter() const
    {
        // 返回 adjustcenter 成员变量的值。
        return this->adjustcenter;
    }

    // 成员方法：setAdjustcenter（设置调整中心标识）
    // 设置成员变量 adjustcenter 的值。
    __host__ __device__ int    // 返回值：函数是否正确执行，若函数正确执
                               // 行，返回 NO_ERROR。
    setAdjustcenter(
            bool adjustcenter  // 设定新的调整中心标识
    ) {
        // 将 adjustcenter 成员变量赋成新值
        this->adjustcenter = adjustcenter;

        return NO_ERROR;
    }

    // Host 成员方法：spatialMoments（计算空间矩）
    // 首先基于差分矩因子算法计算累进求和，然后根据求和结果推导出空间矩的
    // 各项值。
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    spatialMoments(
            Image *img,        // 输入图像
            MomentSet *momset  // 输出空间矩数据集合
    );

    // Host 成员方法：centralMoments（计算中心矩）
    // 首先需要计算空间矩，然后根据公式推导出中心距。
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    centralMoments(
            Image *img,        // 输入图像
            MomentSet *momset  // 输出中心矩数据集合
    );

    // Host 成员方法：centralMoments（计算形状的分布重心和方向）
    // 通过中心矩推导出形状的分布中心和方向。
    __host__ int                // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回NO_ERROR。
    centralMoments(
            Image *img,         // 输入图像
            double centers[2],  // 分布重心
            double *angle       // 分布方向
    );

    // Host 成员方法：huMoments（计算 Hu 矩）
    // 首先需要计算空间矩和中心矩，然后根据公式推导出 Hu 距。
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    huMoments(
            Image *img,        // 输入图像
            MomentSet *momset  // 输出 hu 矩数据集合
    );

    // Host 成员方法：affineMoments（计算 affine 矩）
    // 首先需要计算空间矩和中心矩，然后根据公式推导出 affine 距。
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    affineMoments(
            Image *img,        // 输入图像
            MomentSet *momset  // 输出 affine 矩数据集合
    );

    // Host 成员方法：flusserMoments（计算 flusser 矩）
    // 首先需要计算空间矩和中心矩，然后根据公式推导出 flusser 距。
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    flusserMoments(
            Image *img,        // 输入图像
            MomentSet *momset  // 输出 flusser 矩数据集合
    );
};

#endif

