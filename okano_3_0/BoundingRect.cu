// BoundingRect.cu
// 找出图像中给定点集的包围矩形

#include "BoundingRect.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：DEF_SHARED_LENGTH
// 定义了核函数中共享内存的长度
#define DEF_SHARED_LENGTH(sharedarray) (DEF_BLOCK_X * DEF_BLOCK_Y *  \
                                        sizeof (sharedarray))

// 宏：BR_LARGE_ENOUGH
// 定义了计算共享内存的函数中循环的上界。
#define BR_LARGE_ENOUGH ((1 << 30) - 1)

// 宏：FINDPIXEL_PACK_LEVEL
// 定义了 FINDPIXEL 核函数中一个线程中计算的像素点个数，若该值为 5，则在一个线
// 程中计算 2 ^ 5 = 32 个像素点。根据实验结果，5 是最好的选择。
#define FINDPIXEL_PACK_LEVEL 5

// 宏：FINDPIXEL_PACK_NUM
// 定义了计算每个线程中计算的次数。
#define FINDPIXEL_PACK_NUM (1 << FINDPIXEL_PACK_LEVEL)

// 宏：FINDPIXEL_PACK_MASK
// 定义了计算线程时向上取整的MASK。
#define FINDPIXEL_PACK_MASK (FINDPIXEL_PACK_NUM - 1)

// 列出了计算范围，如果超出范围，返回错误码。
#if (FINDPIXEL_PACK_LEVEL < 1 || FINDPIXEL_PACK_LEVEL > 5)
#  error Unsupport FINDPIXEL_PACK_LEVEL Value!!!
#endif

// 宏：COMPUTECOV_PACK_LEVEL
// 定义了 COMPUTECOV 核函数中一个线程中计算的像素点个数，若该值为 5，则在一个线
// 程中计算 2 ^ 5 = 32个像素点。根据实验结果，5 是最好的选择。
#define COMPUTECOV_PACK_LEVEL 5

// 宏：COMPUTECOV_PACK_NUM
// 定义了计算每个线程中计算的次数。
#define COMPUTECOV_PACK_NUM (1 << COMPUTECOV_PACK_LEVEL)

// 宏：COMPUTECOV_PACK_MASK
// 定义了计算线程时向上取整的MASK
#define COMPUTECOV_PACK_MASK (COMPUTECOV_PACK_NUM - 1)

// 列出了计算范围，如果超出范围，返回错误码。
#if (COMPUTECOV_PACK_LEVEL < 1 || COMPUTECOV_PACK_LEVEL > 5)
#  error Unsupport COMPUTECOV_PACK_LEVEL Value!!!
#endif

// 宏：EXTREAMPOINT_PACK_LEVEL
// 定义了 EXTREAMPOINT 核函数中一个线程中计算的像素点个数，若该值为 5，则在一个
// 线程中计算 2 ^ 5 = 32个像素点。根据实验结果，5 是最好的选择。
#define EXTREAMPOINT_PACK_LEVEL 5

// 宏：EXTREAMPOINT_PACK_NUM
// 定义了计算每个线程中计算的次数。
#define EXTREAMPOINT_PACK_NUM (1 << EXTREAMPOINT_PACK_LEVEL)

// 宏：EXTREAMPOINT_PACK_MASK
// 定义了计算线程时向上取整的MASK
#define EXTREAMPOINT_PACK_MASK (EXTREAMPOINT_PACK_NUM - 1)

// 列出了计算范围，如果超出范围，返回错误码。
#if (EXTREAMPOINT_PACK_LEVEL < 1 || EXTREAMPOINT_PACK_LEVEL > 5)
#  error Unsupport EXTREAMPOINT_PACK_LEVEL Value!!!
#endif

// 结构体：ObjPixelPosSumInfoInner（符合条件的对象的像素点信息）
// 该结构体定义了图像中符合条件的对象的像素点信息，其中包含了像素点数量，x 坐标
// 总和， y 坐标总和。该结构的使用可以减少数据的申请和释放。
typedef struct ObjPixelPosSumInfoInner_st {
    unsigned long long int pixelCount;  // 符合条件的像素点数量
    unsigned long long int posSumX;     // 符合条件的像素点的 x 坐标总和
    unsigned long long int posSumY;     // 符合条件的像素点的 y 坐标总和
} ObjPixelPosSumInfoInner;

// 结构体：CovarianceMatrix（协方差矩阵）
// 该结构体定义了 2 维的协方差矩阵的数据结构。协方差矩阵中第二个和第三个元素相
// 等，所以忽略第三个元素计算。
typedef struct CovarianceMatrix_st{
    float a11;    // 协方差矩阵给的第一个元素 Covariance11 =
                  // E{[X-E(X)][X-E(X)]}。
    float a12;    // 协方差矩阵给的第二个，第三个元素 Covariance1 =
                  // E{[X-E(X)][Y-E(Y)]}。
    //float a21;  // 协方差矩阵给的第三个元素，等于上值，忽略 Covariance21 =
                  // E{[Y-E(Y)][X-E(X)]}。
    float a22;    // 协方差矩阵给的第四个元素 Covariance12 =
                  // E{[Y-E(Y)][Y-E(Y)]}。
} CovarianceMatrix;

// 结构体：Coordinate（点的坐标）
// 该结构体定义了点的坐标。坐标的数据类型为 float。
typedef struct Coordinate_st
{
    float x;  // x 坐标。
    float y;  // y 坐标。
} Coordinate;

// 结构体：CoordinateInt（点的坐标）
// 该结构体定义了点的坐标。坐标的数据类型为 int。
typedef struct CoordinateInt_st
{
    int x;  // x 坐标。
    int y;  // y 坐标。
} CoordinateInt;

// Kernel 函数: _objCalcPixelInfoKer（计算符合条件的对象的像素信息）
// 计算符合条件的对象的像素点的信息，包括像素点个数，横纵坐标总和。
static __global__ void                    // Kernel 函数无返回值
_objCalcPixelInfoKer(
        ImageCuda inimg,                  // 输入图像
        unsigned char value,              // 对象的像素值
        int blksize,                      // 块大小，等于 blocksize.x *
                                          // blocksize.y * blocksize.z。
        int blksize2p,                    // 优化的块大小，方便规约方法。
        ObjPixelPosSumInfoInner *suminfo  // 对象的像素信息。
);

// Host 函数：_objCalcPixelInfo（计算符合条件的对象的像素信息）
// 计算符合条件的对象的像素点的信息，包括像素点个数，横纵坐标总和。该函数在
// Host 端由 CPU 串行实现。
static __host__ void                      // 该函数无返回值
_objCalcPixelInfo(
        ImageCuda *insubimg,              // 输入图像
        unsigned char value,              // 对象的像素值
        ObjPixelPosSumInfoInner *suminfo  // 返回的对象像素信息。
);

// Kernel 函数: _objCalcCovMatrixKer（计算符合条件的对象的协方差矩阵）
// 根据符合条件的像素点的信息和中心值，计算对象的协方差矩阵。
static __global__ void               // Kernel 函数无返回值
_objCalcCovMatrixKer(
        ImageCuda inimg,             // 输入图像
        Coordinate *expcenter,       // 像素坐标的期望
        unsigned char value,         // 对象的像素值
        int blksize,                 // 块大小，等于 blocksize.x *
                                     // blocksize.y * blocksize.z。
        int blksize2p,               // 优化的块大小，方便规约方法。
        CovarianceMatrix *covmatrix  // 协方差矩阵
);

// Kernel 函数: _brCalcExtreamPointKer（计算对象包围矩形的边界点）
// 根据对象的旋转信息和中心点，通过逐次比较，找出对象的包围矩形的边界点。
static __global__ void             // Kernel 函数无返回值
_brCalcExtreamPointKer(
        ImageCuda inimg,           // 输入图像
        unsigned char value,       // 对象的像素值
        int blksize,               // 块大小，等于 blocksize.x *
                                   // blocksize.y * blocksize.z。
        int blksize2p,             // 优化的块大小，方便规约方法。
        CoordinateInt *expcenter,  // 像素坐标的期望, 类型为 int
        RotationInfo *rtinfo,      // 旋转矩阵信息
        BoundBoxInt *boundbox      // 包围矩形信息
);

// 函数：_calcBoundingRectParam（计算包围矩形的参数）
// 计算BoundingRect使用到的参数，计算结果用于随后的成员方法，这样做的目的是简
// 化代码，维护方便。
static __host__ int                // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
_calcBoundingRectParam(
        Image *inimg,              // 输入图像
        unsigned char value,       // 对象的像素值
        RotationInfo *rotateinfo,  // 旋转信息
        BoundBoxInt *boundboxint   // 包围矩形的四个点
);

// Host 函数： brCelling2PInner（计算适合规约方法的共享内存长度）
// 这个函数的目的是通过迭代的方法找出不小于 n 的最大的 2^n。
// 结果是 n2p，用来作为规约方法的共享内存的长度。
static __host__ int  // 返回值：函数是否正确执行，若函数正确执行，
                     // 返回NO_ERROR。
brCelling2PInner(
        int n,       // 块的大小。
        int *n2p     // 计算的适合规约方法的共享内存长度。
);

// Kernel 函数: _objCalcExpectedCenterKer（计算符合条件的对象的中心点坐标）
static __global__ void                          // Kernel 函数无返回值。
_objCalcExpectedCenterKer(
        ObjPixelPosSumInfoInner *pixelsuminfo,  // 对象的像素信息。
        Coordinate *expcenter                   // 对象的中心点坐标。
);

// Kernel 函数：_brCalcParamforExtreamPointKer
//（为核函数 _brCalcExtreamPointKer 计算参数）
static __global__ void                // Kernel 函数无返回值
_brCalcParamforExtreamPointKer(
        CovarianceMatrix *covmatrix,  // 协方差矩阵
        RotationInfo *rtinfo,         // 旋转信息
        BoundBoxInt *bboxint,         // 包围矩形
        Coordinate *expcenter,        // 中心点坐标
        Coordinate *rtexpcenter,      // 旋转后的中心点坐标（float 类型）
        CoordinateInt *expcenterint   // 旋转后中心点坐标（int 类型）
);

// Host 函数： brCelling2PInner（计算适合规约方法的共享内存长度）
// 这个函数的目的是通过迭代的方法找出不小于 n 的最大的 2^n。
// 结果是 n2p，用来作为规约方法的共享内存的长度。
__host__ int brCelling2PInner(int n, int *n2p)
{
    // 局部变量 i。
    int i;

    // 检查输出指针是否为 NULL。
    if (n2p == NULL)
        return NULL_POINTER;

    // 计算找出不小于 n 的最大的 2^n。
    for (i = 1; i < BR_LARGE_ENOUGH; i <<= 1) {
        // 如果找到了，就返回正确。
        if (i >= n) {
            *n2p = i;
            return NO_ERROR;
        }
    }
	
    // 如果找不到，就返回错误。
    return UNKNOW_ERROR;
}

// Kernel 函数: _objCalcPixelInfoKer（计算符合条件的对象的像素信息）
static __global__ void _objCalcPixelInfoKer(
        ImageCuda inimg, unsigned char value,
        int blksize, int blksize2p,
        ObjPixelPosSumInfoInner *suminfo)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理多个输出像素，这多个像素位于统一列的相邻多行
    // 上，因此，对于 r 需要进行乘以 FINDPIXEL_PACK_LEVEL 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) << FINDPIXEL_PACK_LEVEL;

    // 本地变量。inidx 为块内索引。
    int inidx = threadIdx.y * blockDim.x + threadIdx.x;
    int inidx2;
    int currdsize;
    ObjPixelPosSumInfoInner blksuminfo_temp;

    // 声明共享内存。
    extern __shared__ ObjPixelPosSumInfoInner blksuminfo[];

    // 初始化。
    blksuminfo_temp.pixelCount = 0UL;
    blksuminfo_temp.posSumX = 0UL;
    blksuminfo_temp.posSumY = 0UL;

    // 找到图像中符合条件的像素点，计算像素点的数量和坐标总和。
    do {
        // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资
        // 源，一方面防止由于段错误导致的程序崩溃。
        if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
            return;

        // 计算第一个输入坐标点对应的图像数据数组下标。
        int inindex = r * inimg.pitchBytes + c;
        // 读取第一个输入坐标点对应的像素值。
        unsigned char intemp;
        intemp = inimg.imgMeta.imgData[inindex];

        // 如果当前的像素值为 value，那么像素点的计数加 1，x 和 y 的坐标总和累
        // 加。由于 c 和 r 只是 ROI 图像中的坐标索引，所以需加上 ROI 的边界值。
        if (intemp == value) {
            blksuminfo_temp.pixelCount++;
            blksuminfo_temp.posSumX += c;
            blksuminfo_temp.posSumY += r;
        }

        // 宏：FINDPIXEL_KERNEL_MAIN_PHASE
        // 该宏定义了一个线程中进行的计算，计算下一个像素点和对应的操作。
        // 使用宏定义简化代码
#define FINDPIXEL_KERNEL_MAIN_PHASE               \
        if (++r >= inimg.imgMeta.height)          \
            break;                                \
        inindex += inimg.pitchBytes;              \
        intemp = inimg.imgMeta.imgData[inindex];  \
        if (intemp == value) {                    \
            blksuminfo_temp.pixelCount++;         \
            blksuminfo_temp.posSumX += c;         \
            blksuminfo_temp.posSumY += r;         \
        }

#define FINDPIXEL_KERNEL_MAIN_PHASEx2             \
        FINDPIXEL_KERNEL_MAIN_PHASE               \
        FINDPIXEL_KERNEL_MAIN_PHASE

#define FINDPIXEL_KERNEL_MAIN_PHASEx4             \
        FINDPIXEL_KERNEL_MAIN_PHASEx2             \
        FINDPIXEL_KERNEL_MAIN_PHASEx2

#define FINDPIXEL_KERNEL_MAIN_PHASEx8             \
        FINDPIXEL_KERNEL_MAIN_PHASEx4             \
        FINDPIXEL_KERNEL_MAIN_PHASEx4

#define FINDPIXEL_KERNEL_MAIN_PHASEx16            \
        FINDPIXEL_KERNEL_MAIN_PHASEx8             \
        FINDPIXEL_KERNEL_MAIN_PHASEx8

        // 对于线程中的最后一个像素处理操作。
        FINDPIXEL_KERNEL_MAIN_PHASE

        // 根据不同的 FINDPIXEL_PACK_LEVEL 定义，进行不同的线程操作
#if (FINDPIXEL_PACK_LEVEL >= 2)
        FINDPIXEL_KERNEL_MAIN_PHASEx2
#  if (FINDPIXEL_PACK_LEVEL >= 3)
        FINDPIXEL_KERNEL_MAIN_PHASEx4
#    if (FINDPIXEL_PACK_LEVEL >= 4)
        FINDPIXEL_KERNEL_MAIN_PHASEx8
#      if (FINDPIXEL_PACK_LEVEL >= 5)
        FINDPIXEL_KERNEL_MAIN_PHASEx16
#      endif
#    endif
#  endif
#endif

#undef FINDPIXEL_KERNEL_MAIN_PHASEx16
#undef FINDPIXEL_KERNEL_MAIN_PHASEx8
#undef FINDPIXEL_KERNEL_MAIN_PHASEx4
#undef FINDPIXEL_KERNEL_MAIN_PHASEx2
#undef FINDPIXEL_KERNEL_MAIN_PHASE
    } while (0);

    // 将线程中计算得到的临时变量赋给共享内存。
    blksuminfo[inidx].pixelCount = blksuminfo_temp.pixelCount;
    blksuminfo[inidx].posSumX = blksuminfo_temp.posSumX;
    blksuminfo[inidx].posSumY = blksuminfo_temp.posSumY;

    __syncthreads();

    // 对于 blksize2p 长度的值进行折半累加。
    currdsize = (blksize2p >> 1);
    inidx2 = inidx + currdsize;
    if (inidx2 < blksize) {
        blksuminfo[inidx].pixelCount += blksuminfo[inidx2].pixelCount ;
        blksuminfo[inidx].posSumX += blksuminfo[inidx2].posSumX;
        blksuminfo[inidx].posSumY += blksuminfo[inidx2].posSumY;
    }
    __syncthreads();

    // 使用规约的方法，累加像素信息值到共享内存的开头。
    for (currdsize >>= 1; currdsize > 0; currdsize >>= 1) {
        if (inidx < currdsize) {
            inidx2 = inidx + currdsize;
            blksuminfo[inidx].pixelCount += blksuminfo[inidx2].pixelCount;
            blksuminfo[inidx].posSumX += blksuminfo[inidx2].posSumX;
            blksuminfo[inidx].posSumY += blksuminfo[inidx2].posSumY;
        }
        __syncthreads();
    }

    // 把共享内存的像素信息值累加到总和上。每个线程块第一个线程会进行这个操作。
    if (inidx == 0 && blksuminfo[0].pixelCount != 0) {
        atomicAdd(&(suminfo->pixelCount), blksuminfo[0].pixelCount);
        atomicAdd(&(suminfo->posSumX), blksuminfo[0].posSumX);
        atomicAdd(&(suminfo->posSumY), blksuminfo[0].posSumY);
    }
}

// Host 函数：_objCalcPixelInfo（计算符合条件的对象的像素信息）
static __host__ void _objCalcPixelInfo(
        ImageCuda *insubimg, unsigned char value, 
        ObjPixelPosSumInfoInner *suminfo)
{
    // 检查输入指针的合法性。
    if (insubimg == NULL || suminfo == NULL)
        return /*NULL_POINTER*/;

    // 初始化返回值，为下面的累加做准备。
    suminfo->pixelCount = 0UL;
    suminfo->posSumX = 0UL;
    suminfo->posSumY = 0UL;

    // 迭代图像内所有的像素点，判断每个像素点的像素值，如果像素值满足要求，则累
    // 加相应的计数信息。
    for (int r = 0; r < insubimg->imgMeta.height; r++) {
        int inidx = r * insubimg->pitchBytes;
        for (int c = 0; c < insubimg->imgMeta.width; c++) {
            unsigned char inpixel = insubimg->imgMeta.imgData[inidx];
            if (inpixel == value) {
                suminfo->pixelCount += 1;
                suminfo->posSumX += c;
                suminfo->posSumY += r;
            }
            inidx++;
        }
    }
    //return NO_ERROR;
}


// Kernel 函数: _objCalcExpectedCenterKer（计算符合条件的对象的中心点坐标）
static __global__ void _objCalcExpectedCenterKer(
        ObjPixelPosSumInfoInner *pixelsuminfo, Coordinate *expcenter)
{
    // 利用符合条件的像素点的坐标总和与坐标个数，计算对象的中心点坐标。
    expcenter->x = (float)pixelsuminfo->posSumX /
                   (float)pixelsuminfo->pixelCount;
    expcenter->y = (float)pixelsuminfo->posSumY /
                   (float)pixelsuminfo->pixelCount;
}

// Host 函数: _objCalcExpectedCenter（计算符合条件的对象的中心点坐标）
static __host__ void _objCalcExpectedCenter(
        ObjPixelPosSumInfoInner *pixelsuminfo, Coordinate *expcenter)
{
    // 利用符合条件的像素点的坐标总和与坐标个数，计算对象的中心点坐标。
    expcenter->x = (float)pixelsuminfo->posSumX /
                   (float)pixelsuminfo->pixelCount;
    expcenter->y = (float)pixelsuminfo->posSumY /
                   (float)pixelsuminfo->pixelCount;
}

// Kernel 函数: _objCalcCovMatrixKer（计算符合条件的对象的协方差矩阵）
static __global__ void _objCalcCovMatrixKer(
        ImageCuda inimg, Coordinate *expcenter,
        unsigned char value, int blksize, int blksize2p,
        CovarianceMatrix *covmatrix)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理多个输出像素，这多个像素位于统一列的相邻多行
    // 上，因此，对于 r 需要进行乘以 COMPUTECOV_PACK_LEVEL 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) <<
            COMPUTECOV_PACK_LEVEL;

    // 局部变量
    int inidx = threadIdx.y * blockDim.x + threadIdx.x;
    int inidx2, currdsize;
    float dx, dy, dxx, dxy, dyy;
    CovarianceMatrix cov_temp;

    // 声明共享内存
    extern __shared__ CovarianceMatrix shdcov[];

    // 临时变量初始化
    cov_temp.a11 = 0.0f;
    cov_temp.a12 = 0.0f;
    cov_temp.a22 = 0.0f;

    // 找到图像中符合条件的像素点，计算对象的协方差矩阵。
    do {
        // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资
        // 源，一方面防止由于段错误导致的程序崩溃。
        if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
            return;

        // 计算第一个输入坐标点对应的图像数据数组下标。
        int inindex = r * inimg.pitchBytes + c;
        // 读取第一个输入坐标点对应的像素值。
        unsigned char intemp;
        intemp = inimg.imgMeta.imgData[inindex];

        // 计算坐标值减去中心点坐标值。
        dx = c - expcenter->x;
        dy = r - expcenter->y;

        // 计算协方差矩阵的各个值.
        dxx = dx * dx;
        dxy = dx * dy;
        dyy = dy * dy;

        // 如果当前点的像素值符合要求，那么累加到协方差元素中。
        if (intemp == value) {
            cov_temp.a11 += dxx;
            cov_temp.a12 += dxy;
            cov_temp.a22 += dyy;
        }

        // 宏：COMPUTECOV_KERNEL_MAIN_PHASE
        // 该宏定义了一个线程中进行的计算，计算下一个像素点和对应的操作。
        // 使用宏定义简化代码
#define COMPUTECOV_KERNEL_MAIN_PHASE              \
        if (++r >= inimg.imgMeta.height)          \
            break;                                \
        dxy += dx;                                \
        dyy += dy + dy + 1.0f;                    \
        dy += 1.0f;                               \
        inindex += inimg.pitchBytes;              \
        intemp = inimg.imgMeta.imgData[inindex];  \
        if (intemp == value) {                    \
            cov_temp.a11 += dxx;                  \
            cov_temp.a12 += dxy;                  \
            cov_temp.a22 += dyy;                  \
        }

#define COMPUTECOV_KERNEL_MAIN_PHASEx2            \
        COMPUTECOV_KERNEL_MAIN_PHASE              \
        COMPUTECOV_KERNEL_MAIN_PHASE

#define COMPUTECOV_KERNEL_MAIN_PHASEx4            \
        COMPUTECOV_KERNEL_MAIN_PHASEx2            \
        COMPUTECOV_KERNEL_MAIN_PHASEx2

#define COMPUTECOV_KERNEL_MAIN_PHASEx8            \
        COMPUTECOV_KERNEL_MAIN_PHASEx4            \
        COMPUTECOV_KERNEL_MAIN_PHASEx4

#define COMPUTECOV_KERNEL_MAIN_PHASEx16           \
        COMPUTECOV_KERNEL_MAIN_PHASEx8            \
        COMPUTECOV_KERNEL_MAIN_PHASEx8

        // 对于线程中的最后一个像素处理操作。
        COMPUTECOV_KERNEL_MAIN_PHASE

        // 根据不同的 COMPUTECOV_PACK_LEVEL 定义，进行不同的线程操作
#if (COMPUTECOV_PACK_LEVEL >= 2)
        COMPUTECOV_KERNEL_MAIN_PHASEx2
#  if (COMPUTECOV_PACK_LEVEL >= 3)
        COMPUTECOV_KERNEL_MAIN_PHASEx4
#    if (COMPUTECOV_PACK_LEVEL >= 4)
        COMPUTECOV_KERNEL_MAIN_PHASEx8
#      if (COMPUTECOV_PACK_LEVEL >= 5)
        COMPUTECOV_KERNEL_MAIN_PHASEx16
#      endif
#    endif
#  endif
#endif

#undef COMPUTECOV_KERNEL_MAIN_PHASEx16
#undef COMPUTECOV_KERNEL_MAIN_PHASEx8
#undef COMPUTECOV_KERNEL_MAIN_PHASEx4
#undef COMPUTECOV_KERNEL_MAIN_PHASEx2
#undef COMPUTECOV_KERNEL_MAIN_PHASE
    } while (0);

    // 累加线程计算得到的临时变量到共享内存内。
    shdcov[inidx].a11 = cov_temp.a11;
    shdcov[inidx].a12 = cov_temp.a12;
    shdcov[inidx].a22 = cov_temp.a22;

    __syncthreads();

    // 对于 blksize2p 长度的值进行折半累加
    currdsize = (blksize2p >> 1);
    inidx2 = inidx + currdsize;
    if (inidx2 < blksize) {
        shdcov[inidx].a11 += shdcov[inidx2].a11;
        shdcov[inidx].a12 += shdcov[inidx2].a12;
        shdcov[inidx].a22 += shdcov[inidx2].a22;
    }
    __syncthreads();

    // 使用规约的方法，累加像素信息值到共享内存的开头。
    for (currdsize >>= 1; currdsize > 0; currdsize >>= 1) {
        if (inidx < currdsize) {
            inidx2 = inidx + currdsize;
            shdcov[inidx].a11 += shdcov[inidx2].a11;
            shdcov[inidx].a12 += shdcov[inidx2].a12;
            shdcov[inidx].a22 += shdcov[inidx2].a22;
        }
        __syncthreads();
    }

    // 把共享内存的像素信息值累加到总和上。每个线程块的第一个线程会进行这个操
    // 作。
    if (inidx == 0) {
        atomicAdd(&(covmatrix->a11), shdcov[0].a11);
        atomicAdd(&(covmatrix->a12), shdcov[0].a12);
        atomicAdd(&(covmatrix->a22), shdcov[0].a22);
    }
}

// Host 函数: _objCalcCovMatrix（计算符合条件的对象的协方差矩阵）
static __host__ void _objCalcCovMatrix(
        ImageCuda *insubimg, Coordinate *expcenter, unsigned char value,
        CovarianceMatrix *covmatrix)
{
    // 检查输入指针的合法性。
    if (insubimg == NULL || expcenter == NULL || covmatrix == NULL)
        return /*NULL_POINTER*/;

    // 初始化返回值，为下面的累加做准备。
    covmatrix->a11 = 0.0f;
    covmatrix->a12 = 0.0f;
    covmatrix->a22 = 0.0f;

    // 迭代图像内所有的像素点，判断每个像素点的像素值，如果像素值满足要求，则累
    // 加相应的计数信息。
    for (int r = 0; r < insubimg->imgMeta.height; r++) {
        int inidx = r * insubimg->pitchBytes;

        // 计算坐标值减去中心点坐标值。
        float dx = 0.0f - expcenter->x;
        float dy = r - expcenter->y;

        // 计算协方差矩阵的各个值.
        float dxx = dx * dx;
        float dxy = dx * dy;
        float dyy = dy * dy;

        for (int c = 0; c < insubimg->imgMeta.width; c++) {
            unsigned char inpixel = insubimg->imgMeta.imgData[inidx];
            // 如果当前坐标满足要求，则进行偏移量的累加。
            if (inpixel == value) {
                covmatrix->a11 += dxx;
                covmatrix->a12 += dxy;
                covmatrix->a22 += dyy;
            }
            // 利用两个点之间坐标相关性减少一部分计算。
            inidx++;
            dxx += 2 * dx + 1.0f;
            dxy += dy;
            dx += 1.0f;
        }
    }
    //return NO_ERROR;
}

// 函数：_brCalcParamforExtreamPointIn（为核函数 _brCalcExtreamPointKer 
// 计算参数）
static __host__ __device__ void _brCalcParamforExtreamPointIn(
        CovarianceMatrix *covmatrix,
        RotationInfo *rtinfo,
        Coordinate *expcenter,
        Coordinate *rtexpcenter)
{
    // 局部变量。
    float apd, amd, bmc, det;
    float eigen, solx, soly, soldt;

    // 为计算矩阵的特征值做准备计算。
    apd = covmatrix->a11 + covmatrix->a22;
    amd = covmatrix->a11 - covmatrix->a22;
    bmc = covmatrix->a12 * covmatrix->a12;

    // 计算矩阵的特征值。
    det = sqrt(4.0f * bmc + amd * amd);
    eigen = (apd + det) / 2.0f;

    // 计算旋转角度通过 asin()。
    // 求解方程式 (covmatrix - eigen * E) * sol = 0
    solx = covmatrix->a12 + covmatrix->a22 - eigen;
    soly = eigen - covmatrix->a11 - covmatrix->a12;
    soldt = sqrt(solx * solx + soly * soly);

    // 如果解的的 x 在第二或者第三象限，转化坐标到第四或者第一象限。
    if (solx < 0) {
        solx = -solx;
        soly = -soly;
    }

    // 计算旋转角度信息。
    rtinfo->sin = soly / soldt;
    rtinfo->cos = solx / soldt;
    rtinfo->radian = asin(rtinfo->sin);
	
    // 当旋转角度为负时，进行调整操作。
    if (rtinfo->radian < 0) {
        (rtinfo)->radian = -(rtinfo)->radian;
        (rtinfo)->sin = -(rtinfo)->sin;
        (rtinfo)->cos = (rtinfo)->cos;
    }
	
    // 根据旋转信息，计算中心点 expcenter 旋转后的坐标 rtexpcenter。
    rtexpcenter->x = expcenter->x * rtinfo->cos - expcenter->y * rtinfo->sin;
    rtexpcenter->y = expcenter->x * rtinfo->sin + expcenter->y * rtinfo->cos;
}

// Kernel 函数：_brCalcParamforExtreamPointKer（为核函数 _brCalcExtreamPointKer 
// 计算参数）
static __global__ void _brCalcParamforExtreamPointKer(
        CovarianceMatrix *covmatrix,
        RotationInfo *rtinfo,
        BoundBoxInt *bboxint,
        Coordinate *expcenter,
        Coordinate *rtexpcenter,
        CoordinateInt *expcenterint)
{
    _brCalcParamforExtreamPointIn(covmatrix, rtinfo, expcenter, rtexpcenter);

    // 初始化包围矩形的信息，边界的四个值全部用中心点的值来初始化。
    // 初始化中心点坐标。
    expcenterint->x = (int)rtexpcenter->x;
    expcenterint->y = (int)rtexpcenter->y;

    // 初始化包围矩形的边界。
    bboxint->bottom = expcenterint->y;
    bboxint->top = expcenterint->y;
    bboxint->left = expcenterint->x;
    bboxint->right = expcenterint->x;
}

// Host 函数：_brCalcParamforExtreamPoint（计算旋转角度与初始化）
static __host__ void _brCalcParamforExtreamPoint(
        CovarianceMatrix *covmatrix,
        RotationInfo *rtinfo,
        BoundBox *bbox,
        Coordinate *expcenter,
        Coordinate *rtexpcenter)
{
    _brCalcParamforExtreamPointIn(covmatrix, rtinfo, expcenter, rtexpcenter);

    // 初始化包围矩形的边界。
    bbox->bottom = rtexpcenter->y;
    bbox->top = rtexpcenter->y;
    bbox->left = rtexpcenter->x;
    bbox->right = rtexpcenter->x;
}

// Kernel 函数: _brCalcExtreamPointKer（计算对象包围盒的边界点）
static __global__ void _brCalcExtreamPointKer(
        ImageCuda inimg, unsigned char value,
        int blksize, int blksize2p,
        CoordinateInt *expcenter, RotationInfo *rtinfo,
        BoundBoxInt *boundbox)
{
    // 局部变量
    int ptsor[2];
    float pt[2];
    int x, y, x_, y_;

    // 块内索引。
    int inidx = threadIdx.y * blockDim.x + threadIdx.x;
    int inidx2, currdsize;
    BoundBoxInt bbox;

    // 声明共享内存。
    extern __shared__ int shdbbox[];
    int *shdbboxLeft = shdbbox;
    int *shdbboxRight = shdbboxLeft + blksize;
    int *shdbboxTop = shdbboxRight + blksize;
    int *shdbboxBottom = shdbboxTop + blksize;

    // 计算线程对应的输出点的位置，其中 ptsor[0] 和 ptsor[1] 分别表示线程处理的
    // 像素点的坐标的 x 和 y 分量（其中，ptsor[1] 表示 column；ptsor[1] 表示
    // row）。由于我们采用了并行度缩减的策略，令一个线程处理多个输出像素，这多
    // 个像素位于统一列的相邻多行上，因此，对于 r 需要进行乘以
    // EXTREAMPOINT_PACK_LEVEL 计算。
    ptsor[0] = blockIdx.x * blockDim.x + threadIdx.x;
    ptsor[1] = (blockIdx.y * blockDim.y + threadIdx.y) <<
               EXTREAMPOINT_PACK_LEVEL;

    // 初始化包围矩形		   
    bbox.left = expcenter->x;
    bbox.right = expcenter->x;
    bbox.top = expcenter->y;
    bbox.bottom = expcenter->y;

    // 找到图像中符合条件的像素点，计算对象的包围矩形的边界。
    do {
        // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资
        // 源，一方面防止由于段错误导致的程序崩溃。
        if (ptsor[0] >= inimg.imgMeta.width ||
            ptsor[1] >= inimg.imgMeta.height)
            break;

        // 计算第一个输入坐标点对应的图像数据数组下标。
        int inindex = ptsor[1] * inimg.pitchBytes + ptsor[0];
        // 读取第一个输入坐标点对应的像素值。
        unsigned char intemp;
        intemp = inimg.imgMeta.imgData[inindex];

        // 如果像素值符合要求，计算旋转以后的点，比较找出边界值。
        if (intemp == value) {
            RECT_ROTATE_POINT(ptsor, pt, *rtinfo);
            x = (int)(pt[0]);
            x_ = x + 1;
            y = (int)(pt[1]);
            y_ = y + 1;

            // 比较当前像素坐标和初始值。
            bbox.left = min(bbox.left, x);
            bbox.right = max(bbox.right, x_);
            bbox.bottom = min(bbox.bottom, y);
            bbox.top = max(bbox.top, y_);
        }

        // 宏：EXTREAMPOINT_KERNEL_MAIN_PHASE
        // 该宏定义了一个线程中进行的计算，计算下一个像素点和对应的操作。
        // 使用宏定义简化代码
#define EXTREAMPOINT_KERNEL_MAIN_PHASE                     \
        if (++ptsor[1] >= inimg.imgMeta.height)            \
            break;                                         \
        inindex = ptsor[1] * inimg.pitchBytes + ptsor[0];  \
        intemp = inimg.imgMeta.imgData[inindex];           \
        if (intemp == value) {                             \
            RECT_ROTATE_POINT(ptsor, pt, *rtinfo);         \
            x = (int)(pt[0]);                              \
            x_ = x + 1;                                    \
            y = (int)(pt[1]);                              \
            y_ = y + 1;                                    \
            bbox.left = min(bbox.left, x);                 \
            bbox.right = max(bbox.right, x_);              \
            bbox.bottom = min(bbox.bottom, y);             \
            bbox.top = max(bbox.top, y_);                  \
        }

#define EXTREAMPOINT_KERNEL_MAIN_PHASEx2                   \
        EXTREAMPOINT_KERNEL_MAIN_PHASE                     \
        EXTREAMPOINT_KERNEL_MAIN_PHASE

#define EXTREAMPOINT_KERNEL_MAIN_PHASEx4                   \
        EXTREAMPOINT_KERNEL_MAIN_PHASEx2                   \
        EXTREAMPOINT_KERNEL_MAIN_PHASEx2

#define EXTREAMPOINT_KERNEL_MAIN_PHASEx8                   \
        EXTREAMPOINT_KERNEL_MAIN_PHASEx4                   \
        EXTREAMPOINT_KERNEL_MAIN_PHASEx4

#define EXTREAMPOINT_KERNEL_MAIN_PHASEx16                  \
        EXTREAMPOINT_KERNEL_MAIN_PHASEx8                   \
        EXTREAMPOINT_KERNEL_MAIN_PHASEx8

        // 对于线程中的最后一个像素处理操作。
        EXTREAMPOINT_KERNEL_MAIN_PHASE

        // 根据不同的 EXTREAMPOINT_PACK_LEVEL 定义，进行不同的线程操作
#if (EXTREAMPOINT_PACK_LEVEL >= 2)
        EXTREAMPOINT_KERNEL_MAIN_PHASEx2
#  if (EXTREAMPOINT_PACK_LEVEL >= 3)
        EXTREAMPOINT_KERNEL_MAIN_PHASEx4
#    if (EXTREAMPOINT_PACK_LEVEL >= 4)
        EXTREAMPOINT_KERNEL_MAIN_PHASEx8
#      if (EXTREAMPOINT_PACK_LEVEL >= 5)
        EXTREAMPOINT_KERNEL_MAIN_PHASEx16
#      endif
#    endif
#  endif
#endif

#undef EXTREAMPOINT_KERNEL_MAIN_PHASEx16
#undef EXTREAMPOINT_KERNEL_MAIN_PHASEx8
#undef EXTREAMPOINT_KERNEL_MAIN_PHASEx4
#undef EXTREAMPOINT_KERNEL_MAIN_PHASEx2
#undef EXTREAMPOINT_KERNEL_MAIN_PHASE
    } while (0);

    // 比较结果存入共享内存中。
    shdbboxLeft[inidx] = bbox.left;
    shdbboxRight[inidx] = bbox.right + 1;
    shdbboxBottom[inidx] = bbox.bottom;
    shdbboxTop[inidx] = bbox.top;

    __syncthreads();

    // 对于 blksize2p 长度的值进行折半比较包围矩形的边界值。
    currdsize = (blksize2p >> 1);
    inidx2 = inidx + currdsize;
    if (inidx2 < blksize) {
        atomicMin(&(shdbboxLeft[inidx]), shdbboxLeft[inidx2]);
        atomicMax(&(shdbboxRight[inidx]), shdbboxRight[inidx2]);
        atomicMin(&(shdbboxBottom[inidx]), shdbboxBottom[inidx2]);
        atomicMax(&(shdbboxTop[inidx]), shdbboxTop[inidx2]);
    }
    __syncthreads();

    // 使用规约的方法，把比较的结果保存到共享内存的开头。
    for (currdsize >>= 1; currdsize > 0; currdsize >>= 1) {
        if (inidx < currdsize) {
            inidx2 = inidx + currdsize;
            atomicMin(&(shdbboxLeft[inidx]), shdbboxLeft[inidx2]);
            atomicMax(&(shdbboxRight[inidx]), shdbboxRight[inidx2]);
            atomicMin(&(shdbboxBottom[inidx]), shdbboxBottom[inidx2]);
            atomicMax(&(shdbboxTop[inidx]), shdbboxTop[inidx2]);
        }
        __syncthreads();
    }

    // 比较共享内存里的边界值和初始值，更新边界值。每个线程块的第一个线程会进行
    // 这个操作。
    if (inidx == 0) {
        if (shdbboxLeft[0] != expcenter->x)
            atomicMin(&(boundbox->left), shdbboxLeft[0]);
        if (shdbboxRight[0] != expcenter->x)
            atomicMax(&(boundbox->right), shdbboxRight[0]);
        if (shdbboxBottom[0] != expcenter->y)
            atomicMin(&(boundbox->bottom), shdbboxBottom[0]);
        if (shdbboxTop[0] != expcenter->y)
            atomicMax(&(boundbox->top), shdbboxTop[0]);
    }
}

// Host 函数: _brCalcExtreamPoint（计算对象包围盒的边界点）
static __host__ void _brCalcExtreamPoint(
        ImageCuda *insubimg, unsigned char value,
        Coordinate *expcenter, RotationInfo *rtinfo,
        BoundBox *boundbox)
{
    // 检查输入指针的合法性。
    if (insubimg == NULL || expcenter == NULL || 
        rtinfo == NULL || boundbox == NULL)
        return /*NULL_POINTER*/;

    // 迭代图像内所有的像素点，判断每个像素点的像素值，如果像素值满足要求，则累
    // 加相应的计数信息。
    int ptsor[2];
    float pt[2];
    for (ptsor[1] = 0; ptsor[1] < insubimg->imgMeta.height; ptsor[1]++) {
        int inidx = ptsor[1] * insubimg->pitchBytes;
        for (ptsor[0] = 0; ptsor[0] < insubimg->imgMeta.width; ptsor[0]++) {
            unsigned char inpixel = insubimg->imgMeta.imgData[inidx];
            if (inpixel == value) {
                RECT_ROTATE_POINT(ptsor, pt, *rtinfo);
                boundbox->left = min(boundbox->left, pt[0]);
                boundbox->right = max(boundbox->right, pt[0]);
                boundbox->bottom = min(boundbox->bottom, pt[1]);
                boundbox->top = max(boundbox->top, pt[1]);
            }
            inidx++;
        }
    }
    //return NO_ERROR;

}

// 函数：_calcBoundingRectParam（计算包围矩形的参数）
static __host__ int _calcBoundingRectParam(Image *inimg, unsigned char value, 
                                           RotationInfo *rotateinfo,
                                           BoundBoxInt *boundboxint)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 局部变量
    CoordinateInt *rtexpcenterint_dev;
    Coordinate *expcenter_dev, *rtexpcenter_dev;
    CovarianceMatrix *covmatrix_dev;
    RotationInfo *rtinfo_dev, *rtinfon_dev;
    BoundBoxInt *bdboxint_dev;
    ObjPixelPosSumInfoInner *pixelsuminfo_dev;
    float *temp_dev;

    // 在设备端申请内存，然后分配给各个变量。
    cudaError_t cuerrcode;
    cuerrcode = cudaMalloc((void **)&temp_dev,
                           sizeof (ObjPixelPosSumInfoInner) +
                           2 * sizeof (Coordinate) +
                           2 * sizeof (RotationInfo) +
                           sizeof (CovarianceMatrix) +
                           sizeof (CoordinateInt) +
                           sizeof (BoundBoxInt));
    if (cuerrcode != cudaSuccess)
        return cuerrcode;

    // 为变量分配内存。
    pixelsuminfo_dev = (ObjPixelPosSumInfoInner*)(temp_dev);
    expcenter_dev = (Coordinate*)(pixelsuminfo_dev + 1);
    rtexpcenter_dev = expcenter_dev + 1;
    rtinfo_dev = (RotationInfo*)(rtexpcenter_dev + 1);
    rtinfon_dev = rtinfo_dev + 1;
    covmatrix_dev = (CovarianceMatrix*)(rtinfon_dev + 1);
    rtexpcenterint_dev = (CoordinateInt*)(covmatrix_dev + 1);
    bdboxint_dev = (BoundBoxInt*)(rtexpcenterint_dev + 1);

    // 初始化存放像素信息的数组。
    cuerrcode = cudaMemset(pixelsuminfo_dev, 0,
                           sizeof (ObjPixelPosSumInfoInner));
    if (cuerrcode != cudaSuccess)
        return cuerrcode;

    // 设置存放协方差矩阵的数组初始化全为 0。
    cuerrcode = cudaMemset(covmatrix_dev, 0, sizeof (CovarianceMatrix));
    if (cuerrcode != cudaSuccess)
        return cuerrcode;

    // 计算调用 _objCalcPixelInfoKer 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    int height = (insubimgCud.imgMeta.height +
                  FINDPIXEL_PACK_MASK) / FINDPIXEL_PACK_NUM;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y - 1) / blocksize.y;

    // 计算 _objCalcPixelInfoKer 共享内存的长度 blksize2p。
    int blkthdcnt, blksize2p;
    blkthdcnt = blocksize.x * blocksize.y * blocksize.z;
    brCelling2PInner(blkthdcnt, &blksize2p);

    // 计算对象的像素信息。
    _objCalcPixelInfoKer<<<gridsize, blocksize,
                           blkthdcnt * sizeof (ObjPixelPosSumInfoInner)>>>(
            insubimgCud, value,
            blkthdcnt, blksize2p, pixelsuminfo_dev);
            
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 计算中心点。
    _objCalcExpectedCenterKer<<<1, 1>>>(pixelsuminfo_dev, expcenter_dev);
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 计算调用 _objCalcCovMatrixKer 函数的线程块的尺寸和线程块的数量。
    height = (insubimgCud.imgMeta.height +
              COMPUTECOV_PACK_MASK) / COMPUTECOV_PACK_NUM;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y - 1) / blocksize.y;

    // 计算 _objCalcCovMatrixKer 共享内存的长度 blksize2p。
    blkthdcnt = blocksize.x * blocksize.y * blocksize.z;
    brCelling2PInner(blkthdcnt, &blksize2p);

    // 计算协方差矩阵。
    _objCalcCovMatrixKer<<<gridsize, blocksize,
                           blkthdcnt * sizeof (CovarianceMatrix)>>>(
            insubimgCud, expcenter_dev, value, blkthdcnt, blksize2p,
            covmatrix_dev);
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
		
    // 计算 _brCalcExtreamPointKer 需要用到的一些参数	
    _brCalcParamforExtreamPointKer<<<1, 1>>>(covmatrix_dev, rtinfo_dev, 
                                             bdboxint_dev, expcenter_dev,
                                             rtexpcenter_dev, 
                                             rtexpcenterint_dev);
                                      
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;			

    // 计算调用 _brCalcExtreamPointKer 函数的线程块的尺寸和线程块的数量。
    height = (insubimgCud.imgMeta.height + EXTREAMPOINT_PACK_MASK) /
	     EXTREAMPOINT_PACK_NUM;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y - 1) / blocksize.y;

    // 计算 _brCalcExtreamPointKer 共享内存的长度 blksize2p。
    blkthdcnt = blocksize.x * blocksize.y * blocksize.z;
    brCelling2PInner(blkthdcnt, &blksize2p);

    // 计算包围矩形的边界点。
    _brCalcExtreamPointKer<<<gridsize, blocksize,
                             blkthdcnt * sizeof (BoundBoxInt)>>>(
            insubimgCud, value, blkthdcnt, blksize2p,
            rtexpcenterint_dev, rtinfo_dev,
            bdboxint_dev);
            
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 从设备端拷贝回主存。
    cuerrcode = cudaMemcpy(boundboxint, bdboxint_dev,
                           sizeof (BoundBoxInt),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess)
        return cuerrcode;
		
    cuerrcode = cudaMemcpy(rotateinfo, rtinfo_dev,
                           sizeof (RotationInfo),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess)
        return cuerrcode;
		
    cudaFree(temp_dev);
        return NO_ERROR;
}

// 函数：_calcBoundingRectParamHost（计算包围矩形的参数）
static __host__ int _calcBoundingRectParamHost(
        Image *inimg, unsigned char value, 
         RotationInfo *rotateinfo, BoundBox *boundbox)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || rotateinfo == NULL || boundbox == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToHost(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 局部变量
    Coordinate expcenter, rtexpcenter;
    CovarianceMatrix covmatrix;
    ObjPixelPosSumInfoInner pixelsuminfo;

    // 计算对象的像素信息。
    _objCalcPixelInfo(&insubimgCud, value, &pixelsuminfo);

    // 计算中心点。
    _objCalcExpectedCenter(&pixelsuminfo, &expcenter);

    // 计算协方差矩阵。
    _objCalcCovMatrix(&insubimgCud, &expcenter, value, &covmatrix);
		
    // 计算 _brCalcExtreamPointKer 需要用到的一些参数	
    _brCalcParamforExtreamPoint(&covmatrix, rotateinfo, 
                                boundbox, &expcenter, &rtexpcenter);			

    // 计算包围矩形的边界点。
    _brCalcExtreamPoint(&insubimgCud, value, &rtexpcenter, 
                        rotateinfo, boundbox);
 
    return NO_ERROR;
}


// Host 成员方法：boundingRect（求像素值给定的对象的包围矩形）
__host__ int BoundingRect::boundingRect(Image *inimg, Quadrangle *outrect)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outrect == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 局部变量
    RotationInfo rotateinfo;
    BoundBoxInt bdboxint;
	
    // 调用函数_calcBoundingRectParam。
    errcode = _calcBoundingRectParam(inimg, value, &rotateinfo, &bdboxint);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 计算包围矩形的角度。
    outrect->angle = RECT_RAD_TO_DEG(rotateinfo.radian);

    // 计算包围矩形的边界点值。
    Quadrangle temprect;
    temprect.points[0][0] = bdboxint.left;
    temprect.points[0][1] = bdboxint.top;
    temprect.points[1][0] = bdboxint.right;
    temprect.points[1][1] = bdboxint.top;
    temprect.points[2][0] = bdboxint.right;
    temprect.points[2][1] = bdboxint.bottom;
    temprect.points[3][0] = bdboxint.left;
    temprect.points[3][1] = bdboxint.bottom;

    // 计算旋转后的包围矩形的边界点值。即结果的边界点值。
    rotateinfo.sin = -rotateinfo.sin;
    RECT_ROTATE_POINT(temprect.points[0], outrect->points[0], rotateinfo);
    RECT_ROTATE_POINT(temprect.points[1], outrect->points[1], rotateinfo);
    RECT_ROTATE_POINT(temprect.points[2], outrect->points[2], rotateinfo);
    RECT_ROTATE_POINT(temprect.points[3], outrect->points[3], rotateinfo);

    return NO_ERROR;
}

// Host 成员方法：boundingRectHost（求像素值给定的对象的包围矩形）
__host__ int BoundingRect::boundingRectHost(Image *inimg, Quadrangle *outrect)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outrect == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 局部变量
    RotationInfo rotateinfo;
    BoundBox bdbox;
	
    // 调用函数_calcBoundingRectParam。
    errcode = _calcBoundingRectParamHost(inimg, value, &rotateinfo, &bdbox);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 计算包围矩形的角度。
    outrect->angle = RECT_RAD_TO_DEG(rotateinfo.radian);

    // 计算包围矩形的边界点值。
    Quadrangle temprect;
    temprect.points[0][0] = bdbox.left;
    temprect.points[0][1] = bdbox.top;
    temprect.points[1][0] = bdbox.right;
    temprect.points[1][1] = bdbox.top;
    temprect.points[2][0] = bdbox.right;
    temprect.points[2][1] = bdbox.bottom;
    temprect.points[3][0] = bdbox.left;
    temprect.points[3][1] = bdbox.bottom;

    // 计算旋转后的包围矩形的边界点值。即结果的边界点值。
    rotateinfo.sin = -rotateinfo.sin;
    RECT_ROTATE_POINT(temprect.points[0], outrect->points[0], rotateinfo);
    RECT_ROTATE_POINT(temprect.points[1], outrect->points[1], rotateinfo);
    RECT_ROTATE_POINT(temprect.points[2], outrect->points[2], rotateinfo);
    RECT_ROTATE_POINT(temprect.points[3], outrect->points[3], rotateinfo);

    return NO_ERROR;
}

// Host 成员方法：boundingRect（求像素值给定的对象的包围矩形）
__host__ int BoundingRect::boundingRect(Image *inimg, DirectedRect *outrect)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outrect == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 局部变量
    RotationInfo rotateinfo;
    BoundBoxInt bdboxint;
    float boxcenter[2];
	
    // 调用函数_calcBoundingRectParam。
    errcode = _calcBoundingRectParam(inimg, value, &rotateinfo, &bdboxint);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算旋转角。
    outrect->angle = RECT_RAD_TO_DEG(rotateinfo.radian);

    // 计算中心坐标。
    boxcenter[0] = (bdboxint.left + bdboxint.right) / 2.0f;
    boxcenter[1] = (bdboxint.top + bdboxint.bottom) / 2.0f;
    RECT_ROTATE_POINT(boxcenter, outrect->centerPoint, rotateinfo);

    // 计算矩形的长宽。
    outrect->length1 = bdboxint.right - bdboxint.left;
    outrect->length2 = bdboxint.top - bdboxint.bottom;

    // 选择长的作为矩形的长。
    if (outrect->length1 < outrect->length2) {
        int length_temp;
        length_temp = outrect->length1;
        outrect->length1 = outrect->length2;
        outrect->length2 = length_temp;
    } else {
        // 对于旋转角度出现负值的情况，进行处理。
        if (outrect->angle < 0.0f)
            outrect->angle += 90.0f;
        else
            outrect->angle -= 90.0f;
    }

    return NO_ERROR;
}

// Host 成员方法：boundingRectHost（求像素值给定的对象的包围矩形）
__host__ int BoundingRect::boundingRectHost(Image *inimg, DirectedRect *outrect)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outrect == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 局部变量
    RotationInfo rotateinfo;
    BoundBox bdbox;
    float boxcenter[2];
	
    // 调用函数_calcBoundingRectParam。
    errcode = _calcBoundingRectParamHost(inimg, value, &rotateinfo, &bdbox);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算旋转角。
    outrect->angle = RECT_RAD_TO_DEG(rotateinfo.radian);

    // 计算中心坐标。
    boxcenter[0] = (bdbox.left + bdbox.right) / 2.0f;
    boxcenter[1] = (bdbox.top + bdbox.bottom) / 2.0f;
    RECT_ROTATE_POINT(boxcenter, outrect->centerPoint, rotateinfo);

    // 计算矩形的长宽。
    outrect->length1 = bdbox.right - bdbox.left;
    outrect->length2 = bdbox.top - bdbox.bottom;

    // 选择长的作为矩形的长。
    if (outrect->length1 < outrect->length2) {
        int length_temp;
        length_temp = outrect->length1;
        outrect->length1 = outrect->length2;
        outrect->length2 = length_temp;
    } else {
        // 对于旋转角度出现负值的情况，进行处理。
        if (outrect->angle < 0.0f)
            outrect->angle += 90.0f;
        else
            outrect->angle -= 90.0f;
    }

    return NO_ERROR;
}

