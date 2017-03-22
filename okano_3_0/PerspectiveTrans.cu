
#include "PerspectiveTrans.h"

#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 全局变量：_hardIplInimgTex（作为输入图像的纹理内存引用）
// 纹理内存只能用于全局变量，因此将硬件插值的旋转变换的 Kernel 函数的输入图像列
// 于此处。
static texture<unsigned char, 2, cudaReadModeElementType> _hardIplInimgTex;

// Kernel 函数：_hardPerspectiveKer（利用硬件插值实现的射影变换）
// 利用纹理内存提供的硬件插值功能，实现的并行射影变换。没有输入图像的参数，是因
// 为输入图像通过纹理内存来读取数据，纹理内存只能声明为全局变量。
static __global__ void        // Kernel 函数无返回值。
_hardPerspectiveKer(
        ImageCuda outimg,     // 输出图像
        PerspectiveMatrix pm  // 旋转变换的参数
);

// Kernel 函数：_softPerspectiveKer（利用软件插值实现的射影变换）
// 利用 Fanczos 软件插值功能，实现的并行射影变换。
static __global__ void        // Kernel 函数无返回值。
_softPerspectiveKer(
        ImageCuda inimg,      // 输入图像
        ImageCuda outimg,     // 输出图像
        PerspectiveMatrix pm  // 旋转变换的参数
);

// Host 函数：_pointsToPsptMatrix（单侧变换标准矩阵）
// 该函数计算从单位矩形到给定四个点的射影变换所对应的矩阵。由于一个由给定四个点
// 到新的四个点的变换所对应的射影变换可以看成是现将给定四个点变换为一个单位矩
// 形，再由单位矩形变换到新的四个点的两个射影变换的组合，因此这个函数称之为单侧
// 变换的标准矩阵。求的一个完整的变换矩阵的过程也是将两个单侧变换的矩阵通过矩阵
// 乘法进行组合的过程（其中的一个矩阵是逆矩阵）。
static __host__ int            // 返回值：函数是否正确执行，若函数正确执行，返
                               // 回 NO_ERROR。
_pointsToPsptMatrix(
        const float pts[4][2],       // 给定的四个坐标点
        PerspectiveMatrix *pm  // 单侧变换的标准矩阵
);

// Host 函数：_detPsptMatrix（射影变换矩阵的行列式）
// 计算射影变换矩阵的行列式。通过这个函数可以判断矩阵是否为满秩的。
static __host__ float                // 返回值：行列式计算结果（如果计算发生错
                                     // 误，则返回 0.0f）。
_detPsptMatrix(
        const PerspectiveMatrix &pm  // 输入矩阵
);

// Host 函数：_invPsptMatrix（射影变换矩阵求逆）
// 为了得到一个变换所对应的逆变换的矩阵，实现了该函数用来求一个矩阵对应的逆矩
// 阵。
static __host__ int                     // 返回值：函数是否正确执行，若函数正确
                                        // 执行，返回 NO_ERROR。
_invPsptMatrix(
        const PerspectiveMatrix &inpm,  // 输入矩阵
        PerspectiveMatrix *outpm        // 输出的逆矩阵
);

// Host 函数：_mulPsptMatrix（射影变换矩阵乘法计算）
// 计算两个矩阵的乘积。这一函数用于将两个连续的变化拼接成一个单次的变化。
static __host__ int                      // 返回值：函数是否正确执行，若函数正
                                         // 确执行，返回 NO_ERROR。
_mulPsptMatrix(
        const PerspectiveMatrix &inpm1,  // 输入矩阵 1
        const PerspectiveMatrix &inpm2,  // 输入矩阵 2
        PerspectiveMatrix *outpm         // 输出矩阵
);

// Host 函数：_pointsToPsptMatrix（单侧变换标准矩阵）
static __host__ int _pointsToPsptMatrix(const float pts[4][2],
                                        PerspectiveMatrix *pm)
{
    // 检查输入参数是否为 NULL
    if (pts == NULL || pm == NULL)
        return NULL_POINTER;

    // 局部变量，比例系数
    float d = (pts[1][0] - pts[3][0]) * (pts[2][1] - pts[3][1]) -
              (pts[2][0] - pts[3][0]) * (pts[1][1] - pts[3][1]);
    if (fabs(d) < 1.0e-8f)
        return INVALID_DATA;

    // 按照射影变换的公式（这些公式在任何一本介绍图像处理或图形学的书中都会有介
    // 绍，这里就不再赘述）求出矩阵的各个元素。
    pm->elem[2][0] = ((pts[0][0] - pts[1][0] + pts[3][0] - pts[2][0]) *
                      (pts[2][1] - pts[3][1]) -
                      (pts[0][1] - pts[1][1] + pts[3][1] - pts[2][1]) *
                      (pts[2][0] - pts[3][0])) / d;
    pm->elem[2][1] = ((pts[0][1] - pts[1][1] + pts[3][1] - pts[2][1]) *
                      (pts[1][0] - pts[3][0]) -
                      (pts[0][0] - pts[1][0] + pts[3][0] - pts[2][0]) *
                      (pts[1][1] - pts[3][1])) / d;
    pm->elem[2][2] = 1.0f;
    pm->elem[0][0] = pts[1][0] - pts[0][0] + pm->elem[2][0] * pts[1][0];
    pm->elem[0][1] = pts[2][0] - pts[0][0] + pm->elem[2][1] * pts[2][0];
    pm->elem[0][2] = pts[0][0];
    pm->elem[1][0] = pts[1][1] - pts[0][1] + pm->elem[2][0] * pts[1][1];
    pm->elem[1][1] = pts[2][1] - pts[0][1] + pm->elem[2][1] * pts[2][1];
    pm->elem[1][2] = pts[0][1];

    // 计算结束，退出
    return NO_ERROR;
}

// Host 函数：_detPsptMatrix（射影变换矩阵的行列式）
static __host__ float _detPsptMatrix(const PerspectiveMatrix &pm)
{
    // 按照行列式的计算公式，返回行列式的值。计算行列式的方法在所有的关于线性代
    // 数的书中都有详细的介绍，这里不在赘述。
    return pm.elem[0][0] * pm.elem[1][1] * pm.elem[2][2] +
           pm.elem[0][1] * pm.elem[1][2] * pm.elem[2][0] +
           pm.elem[0][2] * pm.elem[1][0] * pm.elem[2][1] -
           pm.elem[0][0] * pm.elem[1][2] * pm.elem[2][1] -
           pm.elem[0][1] * pm.elem[1][0] * pm.elem[2][2] -
           pm.elem[0][2] * pm.elem[1][1] * pm.elem[2][0];
}

// Host 函数：_invPsptMatrix（射影变换矩阵求逆）
static __host__ int _invPsptMatrix(
        const PerspectiveMatrix &inpm, PerspectiveMatrix *outpm)
{
    // 检查输入参数是否为 NULL
    if (outpm == NULL)
        return NULL_POINTER;

    // 求出矩阵的行列式，这个行列式可以辅助求逆计算，同时可以检查该矩阵是否为奇
    // 异阵。
    float det = _detPsptMatrix(inpm);
    if (fabs(det) < 1.0e-8f)
        return INVALID_DATA;

    // 根据 3×3 矩阵求逆的公式，计算给定矩阵的逆矩阵，这个公式可以在任何一本介
    // 绍线性代数的书中找到，此处不再赘述。
    outpm->elem[0][0] = (inpm.elem[1][1] * inpm.elem[2][2] -
                         inpm.elem[1][2] * inpm.elem[2][1]) / det;
    outpm->elem[0][1] = (inpm.elem[0][2] * inpm.elem[2][1] -
                         inpm.elem[0][1] * inpm.elem[2][2]) / det;
    outpm->elem[0][2] = (inpm.elem[0][1] * inpm.elem[1][2] -
                         inpm.elem[0][2] * inpm.elem[1][1]) / det;
    outpm->elem[1][0] = (inpm.elem[1][2] * inpm.elem[2][0] -
                         inpm.elem[1][0] * inpm.elem[2][2]) / det;
    outpm->elem[1][1] = (inpm.elem[0][0] * inpm.elem[2][2] -
                         inpm.elem[0][2] * inpm.elem[2][0]) / det;
    outpm->elem[1][2] = (inpm.elem[0][2] * inpm.elem[1][0] -
                         inpm.elem[0][0] * inpm.elem[1][2]) / det;
    outpm->elem[2][0] = (inpm.elem[1][0] * inpm.elem[2][1] -
                         inpm.elem[1][1] * inpm.elem[2][0]) / det;
    outpm->elem[2][1] = (inpm.elem[0][1] * inpm.elem[2][0] -
                         inpm.elem[0][0] * inpm.elem[2][1]) / det;
    outpm->elem[2][2] = (inpm.elem[0][0] * inpm.elem[1][1] -
                         inpm.elem[0][1] * inpm.elem[1][0]) / det;

    // 计算完毕返回
    return NO_ERROR;
}

// Host 函数：_mulPsptMatrix（射影变换矩阵乘法计算）
static __host__ int _mulPsptMatrix(const PerspectiveMatrix &inpm1,
                                   const PerspectiveMatrix &inpm2,
                                   PerspectiveMatrix *outpm)
{
    // 检查输出指针是否为 NULL。
    if (outpm == NULL)
        return NULL_POINTER;

    // 按照矩阵乘法的公式进行计算。由于矩阵乘法的计算公式在任何一本介绍线性代数
    // 的书中均有介绍，此处不再赘述。
    outpm->elem[0][0] = inpm1.elem[0][0] * inpm2.elem[0][0] +
                        inpm1.elem[0][1] * inpm2.elem[1][0] +
                        inpm1.elem[0][2] * inpm2.elem[2][0];
    outpm->elem[0][1] = inpm1.elem[0][0] * inpm2.elem[0][1] +
                        inpm1.elem[0][1] * inpm2.elem[1][1] +
                        inpm1.elem[0][2] * inpm2.elem[2][1];
    outpm->elem[0][2] = inpm1.elem[0][0] * inpm2.elem[0][2] +
                        inpm1.elem[0][1] * inpm2.elem[1][2] +
                        inpm1.elem[0][2] * inpm2.elem[2][2];
    outpm->elem[1][0] = inpm1.elem[1][0] * inpm2.elem[0][0] +
                        inpm1.elem[1][1] * inpm2.elem[1][0] +
                        inpm1.elem[1][2] * inpm2.elem[2][0];
    outpm->elem[1][1] = inpm1.elem[1][0] * inpm2.elem[0][1] +
                        inpm1.elem[1][1] * inpm2.elem[1][1] +
                        inpm1.elem[1][2] * inpm2.elem[2][1];
    outpm->elem[1][2] = inpm1.elem[1][0] * inpm2.elem[0][2] +
                        inpm1.elem[1][1] * inpm2.elem[1][2] +
                        inpm1.elem[1][2] * inpm2.elem[2][2];
    outpm->elem[2][0] = inpm1.elem[2][0] * inpm2.elem[0][0] +
                        inpm1.elem[2][1] * inpm2.elem[1][0] +
                        inpm1.elem[2][2] * inpm2.elem[2][0];
    outpm->elem[2][1] = inpm1.elem[2][0] * inpm2.elem[0][1] +
                        inpm1.elem[2][1] * inpm2.elem[1][1] +
                        inpm1.elem[2][2] * inpm2.elem[2][1];
    outpm->elem[2][2] = inpm1.elem[2][0] * inpm2.elem[0][2] +
                        inpm1.elem[2][1] * inpm2.elem[1][2] +
                        inpm1.elem[2][2] * inpm2.elem[2][2];

    // 运算完毕，返回
    return NO_ERROR;
}

// Host 成员方法：setPerspectiveMatrix（设置放射透视变换矩阵）
__host__ int PerspectiveTrans::setPerspectiveMatrix(
        const PerspectiveMatrix &newpm)
{
    // 如果给定的矩阵是一个奇异阵（通过判断行列式是否为 0，可以得到一个矩阵是否
    // 为奇异阵），则直接报错返回，因为，一个奇异阵无法进行映射变换的计算。
    if (fabs(_detPsptMatrix(newpm)) < 1.0e-8f)
        return INVALID_DATA;

    // 将 impType 成员变量赋成新值
    this->psptMatrix = newpm;
    return NO_ERROR;
}

// Host 成员方法：setPerspectivePoints（设置射影透视变换四点参数）
__host__ int PerspectiveTrans::setPerspectivePoints(
        const PerspectivePoints &newpp)
{
    // 局部变量声明
    int errcode;
    PerspectiveMatrix a1, a2, inva2;  // 由于计算四点参数到矩阵，需要使用单位矩
                                      // 形作为中间过度，因此这里需要计算出两个
                                      // 矩阵，最后将这两个矩阵拼合起来。

    // 首先，计算源坐标点到单位矩形的单侧变换矩阵。
    errcode = _pointsToPsptMatrix(newpp.srcPts, &a1);
    if (errcode != NO_ERROR)
        return errcode;

    // 然后计算目标坐标点到单位矩阵的单侧变换矩阵。
    errcode = _pointsToPsptMatrix(newpp.dstPts, &a2);
    if (errcode != NO_ERROR)
        return errcode;

    // 由于需要拼合的两个变换，是首先从源四点参数变换到单位矩形，然后再由单位矩
    // 形变换到目标四点参数，这样，需要对第二个单侧矩阵求逆，这样后续步骤才有实
    // 际的物理含义。
    errcode = _invPsptMatrix(a2, &inva2);
    if (errcode != NO_ERROR)
        return errcode;

    // 通过矩阵乘法将两两步变换进行整合，形成一个综合的矩阵。
    errcode = _mulPsptMatrix(a1, inva2, &this->psptMatrix);
    if (errcode != NO_ERROR)
        return errcode;

    // 处理完毕，返回退出。
    return NO_ERROR;
}

// Host 成员方法：setPerspectivePoints（设置射影透视变换四点参数）
__host__ int PerspectiveTrans::setPerspectiveUnitRect(
        const PerspectiveUnitRect &newur)
{
    // 将给定的单位矩形类型的数据转化成内部函数可识别的数据类型。这里将
    // PerspectiveUnitRect 转换为 float[4][2]。
    float tmppts[4][2];
    tmppts[0][0] = newur.pt00[0];
    tmppts[0][1] = newur.pt00[1];
    tmppts[1][0] = newur.pt10[0];
    tmppts[1][1] = newur.pt10[1];
    tmppts[2][0] = newur.pt01[0];
    tmppts[2][1] = newur.pt01[1];
    tmppts[3][0] = newur.pt11[0];
    tmppts[3][1] = newur.pt11[1];

    // 调用内部的转换函数完成转换。
    return _pointsToPsptMatrix(tmppts, &this->psptMatrix);
}

// Kernel 函数：_hardPerspectiveKer（利用硬件插值实现的射影变换）
static __global__ void _hardPerspectiveKer(ImageCuda outimg,
                                           PerspectiveMatrix pm)
{
    // 计算想成对应的输出点的位置，其中 dstc 和 dstr 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻
    // 4 行上，因此，对于 dstr 需要进行乘 4 计算。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (dstc >= outimg.imgMeta.width || dstr >= outimg.imgMeta.height)
        return;

    // 计算第一个输出坐标点对应的图像数据数组下标。
    int dstidx = dstr * outimg.pitchBytes + dstc;

    // 声明目标图像输出像素对应的源图像中的坐标点，由于计算会得到小数结果，因此
    // 使用浮点型存储该做标。
    float srcc, srcr;

    // 在计算过程中会使用到一些中间变量。
    float dstc1 = dstc + outimg.imgMeta.roiX1;  // 由于输入 Kernel 的都是 ROI 
    float dstr1 = dstr + outimg.imgMeta.roiY1;  // 子图像，所以，需要校正出 ROI
                                                // 图像中像素在原图像中的像素坐
                                                // 标。
    float tmpc, tmpr;  // 没有除以比例系数的临时坐标。
    float hh;          // 比例系数，这个系数随着坐标位置变化而变化。

    // 计算第一个输出坐标点对应的源图像中的坐标点。计算过程实际上是目标点的坐标
    // 组成的二维向量扩展成三维向量后乘以映射变换矩阵，之后再将得到的向量的 z
    // 分量归一化到 1 后得到新的坐标（具体步骤可参看任何一本图像处理的书籍）。
    // 反映到代码上分为三个步骤，首先计算得到比例系数，然后计算得到没有初期比例
    // 系数的临时坐标，最后在将这个临时坐标除以比例系统，得到最终的源图像坐标。
    hh = pm.elem[2][0] * dstc1 + pm.elem[2][1] * dstr1 + pm.elem[2][2];
    tmpc = pm.elem[0][0] * dstc1 + pm.elem[0][1] * dstr1 + pm.elem[0][2];
    tmpr = pm.elem[1][0] * dstc1 + pm.elem[1][1] * dstr1 + pm.elem[1][2];
    srcc = tmpc / hh;
    srcr = tmpr / hh;

    // 通过上面的步骤，求出了第一个输出坐标对应的源图像坐标。这里利用纹理内存的
    // 硬件插值功能，直接使用浮点型的坐标读取相应的源图像“像素”值，并赋值给目
    // 标图像。这里没有进行对源图像读取的越界检查，这是因为纹理内存硬件插值功能
    // 可以处理越界访问的情况，越界访问会按照事先的设置得到一个相对合理的像素颜
    // 色值，不会引起错误。
    outimg.imgMeta.imgData[dstidx] = tex2D(_hardIplInimgTex, srcc, srcr);

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++dstr >= outimg.imgMeta.height)
            return;
        
        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        dstidx += outimg.pitchBytes;
        
        // 根据上一个源坐标位置计算当前的源坐标位置。由于只有 y 分量增加 1，因
        // 此，对应的源坐标只有在涉及到 dstr 的项上有变化，从而消除了一系列乘法
        // 计算，而通过两个源坐标的差值进行简单的加减法而得。
        hh += pm.elem[2][1];
        tmpc += pm.elem[0][1];
        tmpr += pm.elem[1][1];
        srcc = tmpc / hh;
        srcr = tmpr / hh;
        
        // 将对应的源坐标位置出的插值像素写入到目标图像的当前像素点中。
        outimg.imgMeta.imgData[dstidx] = tex2D(_hardIplInimgTex, srcc, srcr);
    }
}

// Kernel 函数：_softPerspectiveKer（利用软件插值实现的射影变换）
static __global__ void _softPerspectiveKer(
        ImageCuda inimg, ImageCuda outimg, PerspectiveMatrix pm)
{
    // 计算想成对应的输出点的位置，其中 dstc 和 dstr 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻
    // 4 行上，因此，对于 dstr 需要进行乘 4 计算。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (dstc >= outimg.imgMeta.width || dstr >= outimg.imgMeta.height)
        return;

    // 计算第一个输出坐标点对应的图像数据数组下标。
    int dstidx = dstr * outimg.pitchBytes + dstc;

    // 声明目标图像输出像素对应的源图像中的坐标点，由于计算会得到小数结果，因此
    // 使用浮点型存储该做标。
    float srcc, srcr;

    // 在计算过程中会使用到一些中间变量。
    float dstc1 = dstc + outimg.imgMeta.roiX1;  // 由于输入 Kernel 的都是 ROI 
    float dstr1 = dstr + outimg.imgMeta.roiY1;  // 子图像，所以，需要校正出 ROI
                                                // 图像中像素在原图像中的像素坐
                                                // 标。
    float tmpc, tmpr;  // 没有除以比例系数的临时坐标。
    float hh;          // 比例系数，这个系数随着坐标位置变化而变化。

    // 计算第一个输出坐标点对应的源图像中的坐标点。计算过程实际上是目标点的坐标
    // 组成的二维向量扩展成三维向量后乘以映射变换矩阵，之后再将得到的向量的 z
    // 分量归一化到 1 后得到新的坐标（具体步骤可参看任何一本图像处理的书籍）。
    // 反映到代码上分为三个步骤，首先计算得到比例系数，然后计算得到没有初期比例
    // 系数的临时坐标，最后在将这个临时坐标除以比例系统，得到最终的源图像坐标。
    hh = pm.elem[2][0] * dstc1 + pm.elem[2][1] * dstr1 + pm.elem[2][2];
    tmpc = pm.elem[0][0] * dstc1 + pm.elem[0][1] * dstr1 + pm.elem[0][2];
    tmpr = pm.elem[1][0] * dstc1 + pm.elem[1][1] * dstr1 + pm.elem[1][2];
    srcc = tmpc / hh;
    srcr = tmpr / hh;

    // 通过上面的步骤，求出了第一个输出坐标对应的源图像坐标。这里利用纹理内存的
    // 硬件插值功能，直接使用浮点型的坐标读取相应的源图像“像素”值，并赋值给目
    // 标图像。这里没有进行对源图像读取的越界检查，这是因为纹理内存硬件插值功能
    // 可以处理越界访问的情况，越界访问会按照事先的设置得到一个相对合理的像素颜
    // 色值，不会引起错误。
    outimg.imgMeta.imgData[dstidx] = _fanczosInterpoDev(inimg, srcc, srcr);

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++dstr >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        dstidx += outimg.pitchBytes;

        // 根据上一个源坐标位置计算当前的源坐标位置。由于只有 y 分量增加 1，因
        // 此，对应的源坐标只有在涉及到 dstr 的项上有变化，从而消除了一系列乘法
        // 计算，而通过两个源坐标的差值进行简单的加减法而得。
        hh += pm.elem[2][1];
        tmpc += pm.elem[0][1];
        tmpr += pm.elem[1][1];
        srcc = tmpc / hh;
        srcr = tmpr / hh;

        // 将对应的源坐标位置出的插值像素写入到目标图像的当前像素点中。
        outimg.imgMeta.imgData[dstidx] = _fanczosInterpoDev(inimg, 
                                                            srcc, srcr);
    }
}

// Host 成员方法：perspectiveTrans（射影透视变换）
__host__ int PerspectiveTrans::perspectiveTrans(Image *inimg, Image *outimg)
{
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图像
        // 尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->width, inimg->height);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 获得输入图像对应的 ImageCuda 型的结构体。没有取 ROI 子图像，是因为输入图
    // 像的 ROI 区域相对于输出图像来说是没有意义的。与其在输出图像中把 ROI 区域
    // 以外的区域视为越界区域，还不如把它们纳入计算范畴更为方便。处理起来也不容
    // 易收到硬件对齐因素的限制。
    ImageCuda *inimgCud = IMAGE_CUDA(inimg);
    
    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 针对不同的实现类型，选择不同的路径进行处理。
    cudaError_t cuerrcode;
    switch (this->impType) {
    // 使用硬件插值实现射影变换：
    case PERSPECT_HARD_IPL:
        // 设置数据通道描述符，因为只有一个颜色通道（灰度图），因此描述符中只有
        // 第一个分量含有数据。概述据通道描述符用于纹理内存的绑定操作。
        struct cudaChannelFormatDesc chndesc;
        chndesc = cudaCreateChannelDesc(sizeof (unsigned char) * 8, 0, 0, 0,
                                        cudaChannelFormatKindUnsigned);

        // 将输入图像的 ROI 子图像绑定到纹理内存。
        cuerrcode = cudaBindTexture2D(
                NULL, &_hardIplInimgTex, inimg->imgData, &chndesc, 
                inimg->width, inimg->height, inimgCud->pitchBytes);
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 调用 Kernel 函数，完成实际的图像射影变换。
        _hardPerspectiveKer<<<gridsize, blocksize>>>(
                outsubimgCud, this->psptMatrix);
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;
        break;
    
    // 使用软件插值实现射影变换：
    case PERSPECT_SOFT_IPL:
        // 调用 Kernel 函数，完成实际的图像射影变换。
        _softPerspectiveKer<<<gridsize, blocksize>>>(
                *inimgCud, outsubimgCud, this->psptMatrix);
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;
        break;

    // 其他方式情况下，直接返回非法数据错误。由于 NPP 实现已在前面跳转入了相应
    // 的其他函数，该 switch-case 语句中未包含对 NPP 实现的处理。
    default:
        return INVALID_DATA;
    }
    
    // 处理完毕，退出。
    return NO_ERROR;
}

