// AffineTrans.cu
// 实现图像的旋转变换

#include "AffineTrans.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi.h>

#include "ErrorCode.h"
#include "FanczosIpl.h"

// 宏：M_PI
// π值。对于某些操作系统，M_PI可能没有定义，这里补充定义 M_PI。
#ifndef M_PI
#define M_PI 3.14159265359
#endif

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8


// 结构体：AffineTransParam（旋转仿射变换的内部参数）
// 该结构体定义了旋转仿射变换的内部参数。它的作用在于简化参数传递的形式，在调用
// 算法函数的时候，Host 代码会首先根据类的成员变量和用户的参数计算出这个形似的
// 内部参数，然后再将这个内部参数传递给 Kernel，进而由 Kernel 完成对图像的并行
// 处理。
typedef struct AffineTransParam_st {
    float x0, y0;              // 旋转前平移的向量
    float cosalpha, sinalpha;  // 旋转角度对应的余弦和正弦值
    float x1, y1;              // 旋转后平移的向量
} AffineTransParam;

// 全局变量：_hardIplInimgTex（作为输入图像的纹理内存引用）
// 纹理内存只能用于全局变量，因此将硬件插值的旋转变换的 Kernel 函数的输入图像列
// 于此处。
static texture<unsigned char, 2, cudaReadModeElementType> _hardIplInimgTex;

// Kernel 函数：_hardRotateKer（利用硬件插值实现的旋转变换）
// 利用纹理内存提供的硬件插值功能，实现的并行旋转变换。没有输入图像的参数，是因
// 为输入图像通过纹理内存来读取数据，纹理内存只能声明为全局变量。
static __global__ void          // Kernel 函数无返回值。
_hardRotateKer(
        ImageCuda outimg,       // 输出图像
        AffineTransParam param  // 旋转变换的参数
);

// Kernel 函数：_softRotateKer（利用软件件插值实现的旋转变换）
// 利用 Fanczos 软件硬件插值算法，实现的并行旋转变换。
static __global__ void          // Kernel 函数无返回值。
_softRotateKer(
        ImageCuda inimg,        // 输入图像
        ImageCuda outimg,       // 输出图像
        AffineTransParam param  // 旋转变换的参数
);

// Host 函数：_rotateNpp（基于 NPP 的旋转变换实现）
// 由于调用 NPP 支持库中的函数同 Runtime API 的 CUDA Kernel 调用具有较大的差
// 别，这里我们单独将 NPP 的旋转变换实现单独提出来作为一个函数以方便代码阅读。
// 注意，这个函数没有对输入输出图像进行前后处理工作，因此，必须要求输入输出图像
// 在当前 Device 上合法可用的数据空间，否则会带来不可预知的错误。
static __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                              // NO_ERROR。
_rotateNpp(
        Image *inimg,         // 输入图像，要求该图像必须在当前 Device 上有数
                              // 据。
        Image *outimg,        // 输出图像，要求该图像必须在当前 Device 上有数
                              // 据。
        AffineTransParam atp  // 旋转变换参数
);

// Host 函数：_rotateGeneral（通用旋转变换）
// 作为整个旋转仿射变换的枢纽函数，所有的上层函数调用都会汇聚于此，并游该函数分
// 配调度下一层调度。在这个函数中包含了如下的功能：（1）对输入和输出图像进行数
// 据的准备工作，包括申请当前 Device 存储空间等；（2）针对不同的实现类型，对图
// 像数据进行个别的加工，如对于 NPP 实现调用 _rotateNpp 函数，对于硬插值实现调
// 纹理内存绑定操作等。
static __host__ int            // 返回值：函数是否正确执行，若函数正确执行，返
                               // 回 NO_ERROR。
_rotateGeneral(
        Image *inimg,          // 输入图像
        Image *outimg,         // 输出图像
        AffineTransParam atp,  // 旋转变换参数
        int imptype            // 实现方式
);

// 函数：_calcRotateCenterParam（按照中心点旋转计算内部参数）
// 根据给定的 AffineTrans 类，根据其中的成员变量，计算处内部的参数，内部参数随
// 后用于调用 Kernel 函数。
static __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
_calcRotateCenterParam(
        AffineTrans *at,        // 输入参数，需要计算内部参数的类
        Image *inimg,           // 输入参数，用于考虑 ROI 子图像的问题。
        Image *outimg,          // 输出参数，用于考虑 ROI 子图像的问题。
        AffineTransParam *atp   // 输出参数，转换出来的内部参数，参数中原来的数
                                // 据将会被抹除。
);

// 函数：calcRotateShiftParam（按照平移旋转计算内部参数）
// 根据给定的 AffineTrans 类，根据其中的成员变量，计算处内部的参数，内部参数随
// 后用于调用 Kernel 函数。
static __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
_calcRotateShiftParam(
        AffineTrans *at,        // 输入参数，需要计算内部参数的类
        Image *inimg,           // 输入参数，用于考虑 ROI 子图像的问题。
        Image *outimg,          // 输出参数，用于考虑 ROI 子图像的问题。
        AffineTransParam *atp   // 输出参数，转换出来的内部参数，参数中原来的数
);


// Kernel 函数：_hardRotateKer（利用硬件插值实现的旋转变换）
static __global__ void _hardRotateKer(ImageCuda outimg, AffineTransParam param)
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

    // 由于是通过目标坐标点反推回源图像中的坐标点，因此这里实用的是逆向的旋转变
    // 换。首先进行的是旋转后的平移，由于是逆向操作，这里是减法。
    int tmpc = dstc - param.x1;
    int tmpr = dstr - param.y1;
    // 利用旋转矩阵，进行旋转变换，由于是逆向操作，这里的旋转矩阵也是正向变换的
    // 旋转矩阵的逆矩阵。最后，进行旋转前的平移，同样也是逆向操作，故用减法。
    srcc = tmpc * param.cosalpha - tmpr * param.sinalpha - param.x0;
    srcr = tmpc * param.sinalpha + tmpr * param.cosalpha - param.y0;
    
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
        srcc -= param.sinalpha;
        srcr += param.cosalpha;
        
        // 将对应的源坐标位置出的插值像素写入到目标图像的当前像素点中。
        outimg.imgMeta.imgData[dstidx] = tex2D(_hardIplInimgTex, srcc, srcr);
    }
}

// Kernel 函数：_softRotateKer（利用软件插值实现的旋转变换）
static __global__ void _softRotateKer(ImageCuda inimg, ImageCuda outimg,
                                      AffineTransParam param)
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

    // 由于是通过目标坐标点反推回源图像中的坐标点，因此这里实用的是逆向的旋转变
    // 换。首先进行的是旋转后的平移，由于是逆向操作，这里是减法。
    int tmpc = dstc - param.x1;
    int tmpr = dstr - param.y1;
    // 利用旋转矩阵，进行旋转变换，由于是逆向操作，这里的旋转矩阵也是正向变换的
    // 旋转矩阵的逆矩阵。最后，进行旋转前的平移，同样也是逆向操作，故用减法。
    srcc = tmpc * param.cosalpha - tmpr * param.sinalpha - param.x0;
    srcr = tmpc * param.sinalpha + tmpr * param.cosalpha - param.y0;
   
    // 调用 Fanczos 软件插值算法实现，获得源图像中对应坐标下的插值值。由于插值
    // 算法实现函数处理了越界的情况，因此这里可以安全的把一些问题丢给插值算法实
    // 现函数来处理。
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
        srcc -= param.sinalpha;
        srcr += param.cosalpha;
        
        // 将对应的源坐标位置出的插值像素写入到目标图像的当前像素点中。
        outimg.imgMeta.imgData[dstidx] = _fanczosInterpoDev(inimg, srcc, srcr);
    }
}

// Host 函数：_rotateNpp（基于 NPP 的旋转变换实现）
static __host__ int _rotateNpp(Image *inimg, Image *outimg,
                               AffineTransParam atp)
{
    // 检查输入和输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 获得输入输出图像对应的 ImageCuda 型指针。
    ImageCuda *inimgCud = IMAGE_CUDA(inimg);
    ImageCuda *outimgCud = IMAGE_CUDA(outimg);

    NppiRect srcroi;   // 输入图像的 ROI
    srcroi.x = inimg->roiX1;
    srcroi.y = inimg->roiY1;
    srcroi.width = inimg->roiX2 - inimg->roiX1;
    srcroi.height = inimg->roiY2 - inimg->roiY1;

    // 计算出 4 个基准点的坐标变换，这四个基本坐标选择了源图像 ROI 上的四个角
    // 点。
    double afquad[4][2];  // 存放 4 个基准点在目标图像中的坐标。
    // NPP 函数仅支持旋转后图像平移，因此需要根据内部参数中的旋转前平移和旋转后
    // 平移推算出整体的平移量。
    double xshift = atp.x0 * atp.cosalpha + atp.y0 * atp.sinalpha + atp.x1;
    double yshift = atp.y0 * atp.cosalpha - atp.x0 * atp.sinalpha + atp.y1;
    // 旋转角度，为了软件工程的美观，以及 double 类型的需求，这里是通过计算重新
    // 计算角度，当然，也可以通过改造 AffineTransParam 来减去这一计算。
    double alpha = asin(atp.sinalpha) * 180.0f / M_PI;
    // 调用 NPP 函数获取 ROI 四点对应到目标图像中的坐标。
    NppStatus nppstatus;
    nppstatus = nppiGetRotateQuad(srcroi, afquad, alpha, xshift, yshift);
    if (nppstatus < NPP_SUCCESS) // 这里使用小于号的原因是 NPP 的错误码中，正数
        return CUDA_ERROR;       // 表示无错误的警告，0 表示无错误，负数才表示
                                 // 真正发生了错误。

    // 利用 NPP 函数求出旋转变换对应的仿射矩阵。利用这个矩阵可以将旋转变换的实
    // 现，转化为调用仿射函数实行仿射变换。
    double afcoeff[2][3];
    nppstatus = nppiGetAffineTransform(srcroi, afquad, afcoeff);
    if (nppstatus < NPP_SUCCESS)
        return CUDA_ERROR;

    // 为调用 NPP 仿射函数做一些数据准备工作。由于 NPP 很好的支持了 ROI，所以，
    // 这里没有使用 ROI 子图像，而直接使用了整幅图像和 ROI 信息。
    Npp8u *psrc = (Npp8u *)(inimg->imgData);   // 输入图像的指针
    Npp8u *pdst = (Npp8u *)(outimg->imgData);  // 输出图像的指针

    Npp32s srcstep = inimgCud->pitchBytes;   // 输入图像的 Pitch
    Npp32s dststep = outimgCud->pitchBytes;  // 输出图像的 Pitch
    NppiSize srcsize;                // 输入图像的总尺寸
    srcsize.width = inimg->width;    // 宽
    srcsize.height = inimg->height;  // 高

    NppiRect dstroi;           // 输出图像的 ROI，这里输入图像的 ROI 已在前面完
    dstroi.x = outimg->roiX1;  // 成了赋值，此处无需再赋值。
    dstroi.y = outimg->roiY1;
    dstroi.width = outimg->roiX2 - outimg->roiX1;
    dstroi.height = outimg->roiY2 - outimg->roiY1;

    int iplmode = NPPI_INTER_LINEAR;  // 插值方式（这里我们采用了线性插值）

    // 调用 NPP 的仿射变换函数完成图像的旋转变换。
    nppstatus = nppiWarpAffine_8u_C1R(psrc, srcsize, srcstep, srcroi,
                                      pdst, dststep, dstroi,
                                      afcoeff, iplmode);
    // 现已确定，现行的 NPP 版本并不算稳定，在某些 ROI 的情况下会出现莫名其妙的
    // 无法处理的情况，这时会报告 NPP_ERROR 错误码，但是这个错误码的具体含义
    // NVIDIA 并未给出一个明确的说法。显然这时由于 NPP 内部不稳定造成的。在 NPP
    // 文档的第 644 页关于函数 nppiWarpAffine_8u_C1R 的介绍中并未说明该函数会产
    // 生 NPP_ERROR 的错误。希望未来的 NPP 版本可以解决不稳定的问题。
    if (nppstatus < NPP_SUCCESS)
        return CUDA_ERROR;

    // 处理完毕返回。
    return NO_ERROR;
}

// Host 函数：_rotateGeneral（通用旋转变换）
static __host__ int _rotateGeneral(Image *inimg, Image *outimg,
                                   AffineTransParam atp, int imptype)
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

    // 如果实现方式为调用 NPP 支持库，由于实现方式同其他 CUDA Kernel 的实现法方
    // 法差别较大，则在此直接转入 NPP 处理函数。
    if (imptype == AFFINE_NVIDIA_LIB)
        return _rotateNpp(inimg, outimg, atp);

    // 提取输入图像的 ROI 子图像。
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
    switch (imptype) {
    // 使用硬件插值实现旋转变换：
    case AFFINE_HARD_IPL:
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

        // 调用 Kernel 函数，完成实际的图像旋转变换。
        _hardRotateKer<<<gridsize, blocksize>>>(outsubimgCud, atp);
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;
        break;
    
    // 使用软件插值实现旋转变换：
    case AFFINE_SOFT_IPL:
        // 调用 Kernel 函数，完成实际的图像旋转变换。
        _softRotateKer<<<gridsize, blocksize>>>(*inimgCud, outsubimgCud,
                                                atp);
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

// 函数：calcRotateCenterParam（按照中心点旋转计算内部参数）
static __host__ __device__ int _calcRotateCenterParam(
        AffineTrans *at, Image *inimg, Image *outimg, AffineTransParam *atp)
{
    // 如果两个参数都为 NULL 则报错。如果 at 为 NULL，则无法计算；如果 atp 为
    // NULL，则无法保存计算结果。
    if (at == NULL || atp == NULL || inimg == NULL || outimg == NULL)
        return NULL_POINTER;
    
    // 获取图像旋转的中心点
    int xc = at->getX();
    int yc = at->getY();

    // 如果旋转中心点角度在输入图像之外，则报错退出。
    if (xc < 0 || xc >= inimg->width || yc < 0 || yc >= inimg->height)
        return INVALID_DATA;

    // 设置旋转前平移向量。基于中心的旋转相当于先将旋转中心移动到原点，在旋转后
    // 再将图像移动回去。
    atp->x0 = -xc;
    atp->y0 = -yc;
    
    // 计算旋转角度的余弦和正弦的值。
    float alpharad = at->getAlpha() * M_PI / 180.0f;
    atp->cosalpha = cos(alpharad);
    atp->sinalpha = sin(alpharad);
    
    // 设置旋转后平移向量。
    atp->x1 = xc ;
    atp->y1 = yc;

    // 针对 ROI 信息调整平移， AffineTrans 中的基准坐标是相对于整幅图像而言的，
    // 因此这里需要进行一下差值，从相对于整幅图像的基准坐标计算得到相对于 ROI
    // 子图像的坐标。注意，在 NPP 实现中，由于 NPP 具有对 ROI 的处理能力，因
    // 此，对于 NPP 实现不需要对旋转中心点作出调整。
    if (at->getImpType() != AFFINE_NVIDIA_LIB) {
        // 由于输入图像不需要考虑 ROI 范围，因此，我们在实现过程中，撇开了输入
        // 图像的 ROI 区域，直接使用更加容易计算的整幅图像作为数据来源。
        //atp->x0 += inimg->roiX1;
        //atp->y0 += inimg->roiY1;
        atp->x1 -= outimg->roiX1;
        atp->y1 -= outimg->roiY1;
    }
    
    // 处理完毕，成功返回。
    return NO_ERROR;
}

// Host 成员方法：rotateCenter（基于中心的旋转）
__host__ int AffineTrans::rotateCenter(Image *inimg, Image *outimg)
{
    // 检查输入和输出图像，若有一个为 NULL，则报错。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;
    
    // 转换参数，将类内的参数转换成 Kernel 函数使用的内部参数。
    AffineTransParam atp;
    int errcode;
    errcode = _calcRotateCenterParam(this, inimg, outimg, &atp);
    if (errcode != NO_ERROR) {
        // 如果在参数转换过程中发生错误，则记录 stateFlag 并退出。
        this->stateFlag = errcode;
        return errcode;
    }

    // 交由枢纽函数 _rotateGeneral 进行后续的实际旋转变换。
    errcode = _rotateGeneral(inimg, outimg, atp, this->impType);
    if (errcode != NO_ERROR) {
        // 如果在枢纽函数处理过程中发生错误，则记录 stateFlag 并退出。
        this->stateFlag = errcode;
        return errcode;
    }

    // 处理完毕退出。
    return NO_ERROR;
}

// 函数：calcRotateShiftParam（按照平移旋转计算内部参数）
static __host__ __device__ int _calcRotateShiftParam(
        AffineTrans *at, Image *inimg, Image *outimg, AffineTransParam *atp)
{
    // 如果两个参数都为 NULL 则报错。如果 at 为 NULL，则无法计算；如果 atp 为
    // NULL，则无法保存计算结果。
    if (at == NULL || atp == NULL || inimg == NULL || outimg == NULL)
        return NULL_POINTER;
    
    // 获取图像旋转的中心点
    int xc = inimg->width / 2;
    int yc = inimg->height / 2;
    
    // 设置旋转前平移向量。基于中心的旋转相当于先将旋转中心移动到原点，在旋转后
    // 再将图像移动回去。
    atp->x0 = -xc + at->getX();
    atp->y0 = -yc + at->getY();
    
    // 计算旋转角度的余弦和正弦的值。
    float alpharad = at->getAlpha() * M_PI / 180.0f;
    atp->cosalpha = cos(alpharad);
    atp->sinalpha = sin(alpharad);
    
    // 设置旋转后平移向量。
    atp->x1 = xc ;
    atp->y1 = yc;

    // 针对 ROI 信息调整平移， AffineTrans 中的基准坐标是相对于整幅图像而言的，
    // 因此这里需要进行一下差值，从相对于整幅图像的基准坐标计算得到相对于 ROI
    // 子图像的坐标。注意，在 NPP 实现中，由于 NPP 具有对 ROI 的处理能力，因
    // 此，对于 NPP 实现不需要对旋转中心点作出调整。
    if (at->getImpType() != AFFINE_NVIDIA_LIB) {
        atp->x0 += inimg->roiX1;
        atp->y0 += inimg->roiY1;
        atp->x1 -= outimg->roiX1;
        atp->y1 -= outimg->roiY1;
    }
    
    // 处理完毕，成功返回。
    return NO_ERROR;
}

// Host 成员方法：rotateShift（基于平移的旋转）
__host__ int AffineTrans::rotateShift(Image *inimg, Image *outimg)
{
    // 检查输入和输出图像，若有一个为 NULL，则报错。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;
    
    // 转换参数，将类内的参数转换成 Kernel 函数使用的内部参数。
    AffineTransParam atp;
    int errcode;
    errcode = _calcRotateShiftParam(this, inimg, outimg, &atp);
    if (errcode != NO_ERROR) {
       // 如果在参数转换过程中发生错误，则记录 stateFlag 并退出。
        this->stateFlag = errcode;
        return errcode;
    }

    // 交由枢纽函数 _rotateGeneral 进行后续的实际旋转变换。
    errcode = _rotateGeneral(inimg, outimg, atp, this->impType);
    if (errcode != NO_ERROR) {
        // 如果在枢纽函数处理过程中发生错误，则记录 stateFlag 并退出。
        this->stateFlag = errcode;
        return errcode;
    }

    // 处理完毕退出。
    return NO_ERROR;
}
