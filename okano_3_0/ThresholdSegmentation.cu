#include "ThresholdSegmentation.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_thresholdSegmentationKer（二值化）
// 根据给定的阈值对图像进行二值化处理。如果像素的灰度值与 T 之差小于 D，
// 此像素的灰度值赋值为 255。否则，此像素的灰度值赋值为 0。
static __global__ void           // Kernel 函数无返回值
_thresholdSegmentationKer(
        ImageCuda inimg,         // 输入图像
        ImageCuda outimg,        // 输出图像
        unsigned char T,         // 灰度阈值
        unsigned char D
);

// Kernel 函数: _thresholdSegmentationKer（二值化）
static __global__ void _thresholdSegmentationKer(
        ImageCuda inimg, ImageCuda outimg,
        unsigned char T, unsigned char D)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp;
    intemp = inimg.imgMeta.imgData[inidx];

    // 一个线程处理四个像素点.
    // 如果输入图像的该位置的像素值如果像素的灰度值与 T 之差小于 D，
    // 则将输出图像中对应位置的像素值置为 255；
    // 否则将输出图像中对应位置的像素值置为 0。
    // 线程中处理的第一个点。
    outimg.imgMeta.imgData[outidx] = (abs(intemp - T) < D ? 255 : 0);

    // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += outimg.pitchBytes;
        intemp = inimg.imgMeta.imgData[inidx];

        // 如果输入图像的该位置的像素值如果像素的灰度值与 T 之差小于 D，
        // 则将输出图像中对应
        // 位置的像素值置为 255；否则将输出图像中对应位置的像素值置为 0。线程
        // 中处理的第一个点。
        outimg.imgMeta.imgData[outidx] = (abs(intemp - T) < D ? 255 : 0);
    }
}

// Host 成员方法：thresholdSegmentation（阈值分割处理）
__host__ int ThresholdSegmentation::thresholdSeg_parallel(
        Image *inimg, Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->roiX2 - inimg->roiX1, 
                inimg->roiY2 - inimg->roiY1);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一	
    if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width)
        insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
    else
        outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;

    if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height)
        insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
    else
        outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 调用核函数，根据阈值 threshold 进行二值化处理。
    _thresholdSegmentationKer<<<gridsize, blocksize>>>(
            insubimgCud, outsubimgCud, T, D);
            
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

__host__ int ThresholdSegmentation::thresholdSeg_serial(Image *inimg, Image *outimg )
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;
    int i, j, index;
    for(i = 0;i < inimg->width; i++)
       for (j = 0; j < inimg->height;j++) {
           index = j * inimg->width + i;
           if ((inimg->imgData[index] - T > - D) &&
               (inimg->imgData[index] - T < D))
               outimg->imgData[index] = 255;
           else
               outimg->imgData[index] = 0;
       }

    // 处理完毕，退出。
    return NO_ERROR;
}
