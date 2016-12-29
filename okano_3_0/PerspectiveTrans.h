// PerspectiveTrans.h
// 创建人：于玉龙
//
// 射影透视变换（Perspective Transform）
// 功能说明：实现图像的四点射影透视变换，给定图像的四个点以及这四个点经过变换后
//           的坐标位置，求出整个图像随之变换后的样子。这种变换通常是绘画中使用
//           的透视画法的技术是相一致的。这里，该算法主要被需求方用来完成待匹配
//           图像的形变问题，使匹配算法更加准确。这里我们实现了两种插值算法，第
//           一种使用 CUDA 默认的硬件插值方法，第二种则使用 Fanczos 软件插值。
//
// 修订历史：
// 2012年12月09日（于玉龙）
//     初始版本。
// 2012年12月11日（于玉龙）
//     调整了算法中对于图像 ROI 的处理方式，使其更加符合常理。
// 2012年12月12日（于玉龙）
//     增加了通过单位矩形的形式输入射影变换的方法。
//     增加了基于软件插值的射影透视变换。

#ifndef __PERSPECTIVETRANS_H__
#define __PERSPECTIVETRANS_H__

#include "ErrorCode.h"
#include "Image.h"

// 宏：PERSPECT_SOFT_IPL
// 用于设置 PerspectiveTrans 类中的 impType 成员变量，告知类的实例选用软件实现
// 的 Fanczos 插值实现旋转变换。
#define PERSPECT_SOFT_IPL  1

// 宏：PERSPECT_HARD_IPL
// 用于设置 PerspectiveTrans 类中的 impType 成员变量，告知类的实例选用基于纹理
// 内存的硬件插值功能实现的旋转变换。
#define PERSPECT_HARD_IPL  2

// 结构体：PerspectiveMatrix（射影矩阵）
// 以矩阵的形式表达射影透视变换的参数。该结构体也是算法内部实际使用的变换表达方
// 式，这种表达方式虽然不够直观，但是却能够很好的用于计算。所有通过坐标点形式输
// 入的参数，最终也会转化位这种形式。
typedef struct PerspectiveMatrix_st {
    float elem[3][3];  // 矩阵元素，是一个 3×3 的矩阵
} PerspectiveMatrix;

// 结构体：PerspectiveUnitRect（单位矩形变换参数）
// 射影变换使用的一种参数形式，给定单位矩形在进行影变换后的四个点坐标。其中，
// ptXY 单位矩形上的四个角点 (X, Y) 的新坐标，下标 [0] 表示 x 坐标，下标 [1] 表
// 示 y 坐标。
typedef struct PerspectiveUnitRect_st {
    float pt00[2];  // 点 (0, 0) 对应的新坐标。
    float pt10[2];  // 点 (1, 0) 对应的新坐标。
    float pt01[2];  // 点 (0, 1) 对应的新坐标。
    float pt11[2];  // 点 (1, 1) 对应的新坐标。
} PerspectiveUnitRect;

// 结构体：PerspectivePoints（射影变换坐标点形式参数）
// 射影变换使用的一种最形象化的参数。通过给定四个原始坐标点，和对应的四个新坐
// 标，来刻画射影变换。其中，下标 [0] 表示 x 坐标，下标 [1] 表示 y 坐标。
typedef struct PerspectivePoints_st {
    float srcPts[4][2];  // 四个原始坐标点
    float dstPts[4][2];  // 变换后四个点对应的新坐标
} PerspectivePoints;


// 类：PerspectiveTrans
// 继承自：无
// 实现图像的四点射影透视变换，给定图像的四个点以及这四个点经过变换后的坐标位
// 置，求出整个图像随之变换后的样子。这种变换通常是绘画中使用的透视画法的技术是
// 相一致的。这里，该算法主要被需求方用来完成待匹配图像的形变问题，使匹配算法更
// 加准确。这里我们实现了两种插值算法，第一种使用 CUDA 默认的硬件插值方法，第二
// 种则使用 Fanczos 软件插值。
class PerspectiveTrans {

protected:

    // 成员变量：impType（实现类型）
    // 设定三种实现类型中的一种，在调用仿射变换函数的时候，使用对应的实现方式。
    int impType;

    // 成员变量：psptMatrix（射影透视变换矩阵）
    // 描述射影透视变换的矩阵参数形式。由于这种矩阵可以直接用于计算，因此所有通
    // 过其他形式传递进来的参数，都会实现转换成这种表示形式。由于实际计算过程中
    // 是通过目标点反算回源图像坐标点，因此，该矩阵实际上是逆运算的矩阵。
    PerspectiveMatrix psptMatrix;

public:

    // 构造函数：PerspectiveTrans
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    PerspectiveTrans()
    {
        // 使用默认值为类的各个成员变量赋值。
        impType = PERSPECT_HARD_IPL;   // 实现类型的默认值为“硬件插值”
        psptMatrix.elem[0][0] = 1.0f;  // 变换矩阵初始化为一个单位矩阵。选择单
        psptMatrix.elem[0][1] = 0.0f;  // 位矩阵的原因是由于单位阵不进行任何变
        psptMatrix.elem[0][2] = 0.0f;  // 化，输入图像和输出图像是完全一致的。
        psptMatrix.elem[1][0] = 0.0f;
        psptMatrix.elem[1][1] = 1.0f;
        psptMatrix.elem[1][2] = 0.0f;
        psptMatrix.elem[2][0] = 0.0f;
        psptMatrix.elem[2][1] = 0.0f;
        psptMatrix.elem[2][2] = 1.0f;
    }

    // 构造函数：PerspectiveTrans
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    PerspectiveTrans (
            int imptype  // 实现类型（具体解释见成员变量）
    ) {
        // 使用默认值为类的各个成员变量赋值。
        impType = PERSPECT_HARD_IPL;   // 实现类型的默认值为“硬件插值”
        psptMatrix.elem[0][0] = 1.0f;  // 变换矩阵初始化为一个单位矩阵。选择单
        psptMatrix.elem[0][1] = 0.0f;  // 位矩阵的原因是由于单位阵不进行任何变
        psptMatrix.elem[0][2] = 0.0f;  // 化，输入图像和输出图像是完全一致的。
        psptMatrix.elem[1][0] = 0.0f;
        psptMatrix.elem[1][1] = 1.0f;
        psptMatrix.elem[1][2] = 0.0f;
        psptMatrix.elem[2][0] = 0.0f;
        psptMatrix.elem[2][1] = 0.0f;
        psptMatrix.elem[2][2] = 1.0f;

        // 根据参数列表中的值设定成员变量的初值
        this->setImpType(imptype);
    }

    // 成员方法：getImpType（读取实现类型）
    // 读取 impType 成员变量的值。
    __host__ __device__ int  // 返回值：当前 impType 成员变量的值。
    getImpType() const
    {
        // 返回 impType 成员变量的值。
        return this->impType;
    }
    
    // 成员方法：setImpType（设置实现类型）
    // 设置 impType 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setImpType(
            int imptype      // 新的实现类型
    ) {
        // 检查输入参数是否合法
        if (imptype != PERSPECT_HARD_IPL && imptype != PERSPECT_SOFT_IPL)
            return INVALID_DATA;

        // 将 impType 成员变量赋成新值
        this->impType = imptype;
        return NO_ERROR;
    }

    // 成员方法：getPerspectiveMatrix（读取射影透视变换矩阵）
    // 读取 psptMatrix 成员变量的值。
    __host__ __device__ PerspectiveMatrix  // 返回值：当前 psptMatrix 成员变量
                                           // 的值。
    getPerspectiveMatrix() const
    {
        // 返回 psptMatrix 成员变量的值。
        return this->psptMatrix;
    }

    // Host 成员方法：setPerspectiveMatrix（设置射影透视变换矩阵）
    // 设置 psptMatrix 成员变量的值。
    __host__ int                            // 返回值：函数是否正确执行，若函数
                                            // 正确执行，返回 NO_ERROR。
    setPerspectiveMatrix(
            const PerspectiveMatrix &newpm  // 新的射影变换矩阵
    );

    // Host 成员方法：setPerspectivePoints（设置射影透视变换四点参数）
    // 设置射影变换的参数，虽然给定的形式是四点变换，但是该函数会将这个四点参数
    // 转换为矩阵形式。
    __host__ int                            // 返回值：函数是否正确执行，若函数
                                            // 正确执行，返回 NO_ERROR。
    setPerspectivePoints(
            const PerspectivePoints &newpp  // 新的摄影变换四点参数
    );

    // Host 成员方法：setPerspectiveUnitRect（以单位矩形的形式设定变换参数）
    // 通过使用单位矩形的形式设定射影变换的参数，这个方式给定单位矩形上四个角点
    // 在变换后的坐标来表述射影变换的参数。
    __host__ int                              // 返回值：函数是否正确执行，若函
                                              // 数正确执行，返回 NO_ERROR。
    setPerspectiveUnitRect(
            const PerspectiveUnitRect &newur  // 新的单位矩形形式的参数
    );

    // Host 成员方法：perspectiveTrans（射影透视变换）
    // 射影透视变换的主函数。其中输入图像与输出图像的原点坐标对齐。如果输出图像
    // 为一个空图像，则输出图像的尺寸会和输入图像等大小。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    perspectiveTrans(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
};

#endif

