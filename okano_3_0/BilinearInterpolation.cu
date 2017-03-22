// BilinearInterpolation.cu
// 实现图像的双线性插值

#include "BilinearInterpolation.h"
#include <iostream>
#include <cmath>
using namespace std;


// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 全局变量：_bilInterInimgTex（作为输入图像的纹理内存引用）
// 纹理内存只能用于全局变量，此处是为了使用纹理内存提供的硬件插值功能
static texture<unsigned char, 2, cudaReadModeNormalizedFloat> _bilInterInimgTex;

// Host 函数：initTexture（初始化纹理内存）
// 将输入图像数据绑定到纹理内存
static __host__ int   // 返回值：若正确执行返回 NO_ERROR
initTexture(
        Image* inimg  // 输入图像
);

// Kernel 函数：_bilInterpolKer（使用 ImageCuda 实现的双边滤波）
// 空域参数只影响高斯表，在调用该方法前初始化高斯表即可
static __global__ void      // kernel 函数无返回值
_bilInterpolKer(
        ImageCuda outimg,   // 输出图像
        float scaleinverse  // 双边滤波半径
);


// Host 函数：initTexture（初始化纹理内存）
static __host__ int initTexture(Image* inimg)
{
    cudaError_t cuerrcode;
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

    // 设置数据通道描述符，因为只有一个颜色通道（灰度图），因此描述符中只有第一
    // 个分量含有数据。概述据通道描述符用于纹理内存的绑定操作。
    struct cudaChannelFormatDesc chndesc;
    chndesc = cudaCreateChannelDesc(sizeof (unsigned char) * 8, 0, 0, 0,
                                    cudaChannelFormatKindUnsigned);
    // 将输入图像数据绑定至纹理内存（texture） 
    cuerrcode = cudaBindTexture2D(
            NULL, &_bilInterInimgTex, insubimgCud.imgMeta.imgData, &chndesc, 
            insubimgCud.imgMeta.width, insubimgCud.imgMeta.height, 
            insubimgCud.pitchBytes);
    
    // 将纹理内存（texture）的过滤模式设置为线性插值模式
    // （cudaFilterModeLinear）
    _bilInterInimgTex.filterMode = cudaFilterModeLinear;
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;
    return NO_ERROR;
}

// Kernel 函数：_bilInterpolKer（使用 ImageCuda 实现的双线性插值）
static __global__ void _bilInterpolKer(ImageCuda outimg, float scaleinverse)
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
    
    // 将放大后图像的坐标点映射到源图像中的坐标点
    float oldx = dstc * scaleinverse, oldy = dstr * scaleinverse;
    
    // 使用 texture 硬件插值得到归一化的 [0, 1] 的 float 归一化数据，再将数据转
    // 换为对应的 8 位 unsigned char 数据
    outimg.imgMeta.imgData[dstidx] = (unsigned char)0xFF * 
                                     tex2D(_bilInterInimgTex, oldx, oldy);

    // 处理剩下的 3 个点
    for (int i = 0; i <= 3; i++)
    {
        if (++dstr >= outimg.imgMeta.height)
            return ;
        // 由于只有纵坐标加 1，故映射左边只需加上 scale 的倒数即可
        oldy += scaleinverse;
        // 将数据指针移至下一行像素点
        dstidx += outimg.pitchBytes;
        // 使用 texture 进行硬件插值
        outimg.imgMeta.imgData[dstidx] = (unsigned char) 0xFF * 
                                         tex2D(_bilInterInimgTex, oldx, oldy);
    }
}

// Host 成员方法：doInterpolation（执行插值）
__host__ int BilinearInterpolation::doInterpolation(Image *inimg, Image *outimg)
{
    // 若图像的放大倍数为 0 ，则不进行插值返回正确执行
    if (scale <= 0)
        return NO_ERROR;

    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;

    int errcode;  // 局部变量，错误码

    // 初始化纹理内存，将输入图像与之绑定
    initTexture(inimg);

    // 将输出图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 拷贝失败时需要在host端创建放大后大小的图像空间
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, scale * (inimg->roiX2 - inimg->roiX1), 
                scale * (inimg->roiY2 - inimg->roiY1));
        if (errcode != NO_ERROR)
            return errcode;
    }
        
    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 由于输出图像是输入图像的 scale 倍，此处不进行 ROI 的大小调整
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    // gridsize 由 blocksize 对应大小决定
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);
    
    // 放大倍数的倒数，将核函数内的除法运算变成乘法运算，同时减少除运算的次数
    float scaleInverse = 1.0f / scale;

    // 调用核函数进行插值
    _bilInterpolKer<<<gridsize, blocksize>>>(outsubimgCud, scaleInverse);
    
    return NO_ERROR;
}

