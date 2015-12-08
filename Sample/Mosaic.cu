// Mosaic.cu
// 实现给图像打马赛克的操作

#include <iostream>
using namespace std;

#include "ErrorCode.h"
#include "Mosaic.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y  32

// Kernel 函数：_mosaicKer（马赛克）
// 给图像的指定区域打马赛克。将图像指定区域划分为 n * n 的小块若干，每个小块
// 内求平均值，并在输出图像对应的小块内各个像素点均赋为该值。
static __global__ void     // kernel 函数无返回值
_mosaicKer(
        ImageCuda inimg,   // 输入图像 
        ImageCuda outimg,  // 输出图像
        int mossize        // 马赛克块尺寸
);

// Kernel 函数：_mosaicKer（马赛克）
static __global__ void _mosaicKer(ImageCuda inimg, ImageCuda outimg, 
                                  int mossize)
{   
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * mossize + threadIdx.x;
    int r = blockIdx.y * mossize + threadIdx.y;

    // 动态分配共享内存，用于进行像素累加。
    extern __shared__ int temp[];

    // 计算该线程在块内的相对位置。
    int inindex = threadIdx.y * blockDim.x + threadIdx.x;

    // 为共享内存附初值 0。
    temp[inindex] = 0;

    // 分配共享内存，用于存放该线程块要处理的马赛克块在 y 方向上的大小
    __shared__ int blocksizey[1]; 
    // 分配共享内存，用于存放该线程块要处理的马赛克块在 x 方向上的大小
    __shared__ int blocksizex[1];

    // 计算当前需要处理的马赛克块的有效尺寸
    if (inindex == 0) {
        // 为共享内存附初值 0
        blocksizex[0] = 0;
        blocksizey[0] = 0;
    
        // 计算 mosblockx，和 blocksizey 的值，先计算 x 方向。
        if (blockIdx.x == (gridDim.x - 1) &&
            inimg.imgMeta.width % mossize != 0) {
            // 如果该线程块是水平方向最后一块 ，并且图像宽度不能被马赛克块整
            // 除，则 blocksizex 的值等于图像宽度与马赛克块宽度求余得到的值。
            blocksizex[0] = inimg.imgMeta.width % mossize;
        } else {
            // 否则 blocksizex 等于用户规定的马赛克块大小
            blocksizex[0] = mossize;
        }

        // 计算 y 方向。
        if (blockIdx.y == (gridDim.y - 1) &&
            inimg.imgMeta.height % mossize != 0) {
            // 如果该线程块是竖直方向最后一块 ，并且图像高度不能被马赛克块整
            // 除，则 blocksizey 的值等于图像高度与马赛克块高度求余得到的值。
            blocksizey[0] = inimg.imgMeta.height % mossize;
        } else {
            // 否则 blocksizey 等于用户规定的马赛克块大小
            blocksizey[0] = mossize;
        }
    }
    
    // 进行块内同步，保证执行到此处，线程已经计算出该块的有效大小。
    __syncthreads();

    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;

    // 计算出该线程当前处理的点在马赛克块内的坐标，其中 inblockc 和 inblockr
    // 分别表示马赛克块内的 x 和 y 分量。
    int inblockc = threadIdx.x;
    int inblockr = threadIdx.y;

    // 将整个马赛克块内的点的像素值按照一定的方式累加到共享内存中。当马赛克块小
    // 于或等于线程块大小时，直接将线程对应的点的像素累加到 temp 中对应位置上；
    // 当马赛克块大于线程块时，逻辑上将马赛克块分成 n 个于线程块等大的块，每个
    // 线程处理 n 个块内相同位置上的点。
    // 当线程目前处理的点在马赛克块内的纵坐标未超出马赛克块 y 方向的有效大小时
    // 循环。
    while (inblockr < blocksizey[0]) { 
        // 当线程目前处理的点在马赛克块内的横坐标未超出马赛克块 x 方向的有效大
        // 时循环。
        while (inblockc < blocksizex[0]) {
            // 将该坐标点对应的像素值累加到 temp 中对应的点上。
            temp[inindex] += inimg.imgMeta.imgData[inidx];
            // 马赛克块内横坐标定位到水平方向下一个待处理点的位置。
            inblockc += blockDim.x;
            // 计算出该点的索引。
            inidx += blockDim.x;
        }
        // 当水平方向处理完以后，则开始处理下一行。
        // 将马赛克块内横坐标恢复到初始状态。
        inblockc = threadIdx.x;
        // 马赛克块内纵坐标定位到竖直方向第一个待处理点的位置。
        inblockr += blockDim.y;
        // 计算出该点的索引。
        inidx = (blockIdx.y * mossize + inblockr) * inimg.pitchBytes + c;
    }
    // 程序执行至此，待处理区域中所有像素值均累加到 temp 中，等待归约。
    __syncthreads();

    // 接下来利用金字塔操作将一维数组 temp 中的元素归约到 temp[0] 位置处。该归
    // 约操作要求 temp 数组长度必须是 2 的整数次幂。
    // cursize 等于 temp 数组长度的一半。
    int cursize = (blockDim.x * blockDim.y) / 2;

    // 循环进行线程规约。
    for(; cursize > 0; cursize /= 2) {
        // 将 temp 数组的后半部分累加到前半部分上。
        if (inindex < cursize)
            temp[inindex] += temp[inindex + cursize];

        // 程序运行至此，完成一轮累加
        __syncthreads();
    }
    // 代码运行至此，线程块内水平归约全部结束。
    __syncthreads();

    // 用一个线程来计算该线程块最终的像素值。
    if (inindex == 0) {
        temp[0] = temp[0] / (blocksizex[0] * blocksizey[0]);
    }
    // 程序运行至此，该线程块的最终像素值以计算出。
    __syncthreads();  

    // 将计算出的最终的值赋回输出图像。
    // 马赛克块内横纵坐标恢复到初始值。
    inblockc = threadIdx.x;
    inblockr = threadIdx.y;

    // 用同样的方式对应线程与像素点，将计算出的像素值赋给各点。当马赛克块小于或
    // 等于线程块大小时，直接将线程对应的点的像素值赋为 temp[0] 中对应位置上；
    // 当马赛克块大于线程块时，逻辑上将马赛克块分成 n 个与线程块等大的块，每个
    // 线程给 n 个块内相同位置上的点赋值。
    // 当线程目前处理的点在马赛克块内的纵坐标未超出马赛克块 y 方向的有效大小时
    // 循环。
    while (inblockr < blocksizey[0]) {
        // 当线程目前处理的点在马赛克块内的横坐标未超出马赛克块 x 方向的有效大
        // 时循环。
        while (inblockc < blocksizex[0]) {
            // 将该点的坐标值赋为刚刚计算出的平均值。
            outimg.imgMeta.imgData[outidx] = temp[0];
            // 马赛克块内横坐标定位到水平方向下一个待处理点的位置。
            inblockc += blockDim.x;
            // 计算出该点的索引。
            outidx += blockDim.x;
        }

        // 当水平方向处理完以后，则开始处理下一行。
        // 将马赛克块内横坐标恢复到初始状态。
        inblockc = threadIdx.x;
        // 马赛克块内纵坐标定位到竖直方向第一个待处理点的位置。
        inblockr += blockDim.y;
        // 计算出该点的索引。
        outidx = (blockIdx.y * mossize + inblockr) * outimg.pitchBytes + c;
    }
}

// Host 成员方法： mosaic (打马赛克)
__host__ int Mosaic::mosaic(Image *inimg, Image * outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;

    // 如果输出图像为 NULL，直接调用 In—place 版本的成员方法。
    if (outimg == NULL)
        return mosaic(inimg);

    // 如果 mossize 大小超过了roi子区域的大小，则直接报错返回。
    if (mossize > (inimg->roiX2 - inimg->roiX1) || 
        mossize > (inimg->roiY2 - inimg->roiY1))
        return INVALID_DATA;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;   // 局部变量，错误码。

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

    // 将输入图像 inimg 复制到输出图像 outimg 中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入图像的 roi 子图像大小设置为与输入图像的一致
    outimg->roiX1 = inimg->roiX1;
    outimg->roiY1 = inimg->roiY1;
    outimg->roiX2 = inimg->roiX2;
    outimg->roiY2 = inimg->roiY2;

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
    gridsize.x = (outsubimgCud.imgMeta.width + mossize - 1) / mossize;
    gridsize.y = (outsubimgCud.imgMeta.height + mossize - 1) / mossize;

    // 动态申请需要的共享内存大小。
    int memsize = sizeof (int) * (DEF_BLOCK_X * DEF_BLOCK_Y);

    // 调用核函数，对图像 roi 子图指定区域进行马赛克处理。
    _mosaicKer<<<gridsize, blocksize, memsize>>>(
            insubimgCud, outsubimgCud, mossize);           
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法： mosaic (打马赛克)
__host__ int Mosaic::mosaic(Image *inimg)
{
    // 检查图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;

    // 如果 mossize 大小超过了roi子区域的大小，则直接报错返回。
    if (mossize > (inimg->roiX2 - inimg->roiX1) || 
        mossize > (inimg->roiY2 - inimg->roiY1))
        return INVALID_DATA;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 图像准备内存空间，以便盛放数据。   
    int errcode;  // 局部变量，错误码。

    // 将图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
   
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + mossize - 1) / mossize;
    gridsize.y = (insubimgCud.imgMeta.height + mossize - 1) / mossize;

    // 动态申请需要的共享内存大小。
    int memsize = sizeof (int) * (DEF_BLOCK_X * DEF_BLOCK_Y);

    // 调用核函数，对图像 roi 子图指定区域进行马赛克处理。
    _mosaicKer<<<gridsize, blocksize, memsize>>>(
            insubimgCud, insubimgCud, mossize);           
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

