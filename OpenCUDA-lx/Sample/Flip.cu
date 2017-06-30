// Flip.cu
// 实现图像的水平和竖直方向的翻转。

#include"Flip.h"

#include<iostream>
#include<cmath>
using namespace std;

#include"ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认线程快的尺寸。
#define DEF_BLOCK_X 32
#define DEF_BLOCK_Y  8

// Kernel 函数：flipHorizontalKer（图像水平翻转）
// 通过交换水平相对位置像素点的值来实现图像的水平翻转。
static __global__ void    // Kernel 函数无返回值
_flipHorizontalKer(
        ImageCuda inimg,  // 输入图像
        ImageCuda outimg  // 输出图像
);

// Kernel 函数：flipVerticalKer（图像竖直翻转）
// 通过交换竖直相对位置像素点的值来实现图像的竖直翻转。
static __global__ void    // Kernel 函数无返回值
_flipVerticalKer(
        ImageCuda inimg,  // 输入图像
        ImageCuda outimg  // 输出图像
);

// Kernel 函数：flipHorizontalKer（图像水平翻转）
static __global__ void _flipHorizontalKer(ImageCuda inimg,ImageCuda outimg)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    int sorc;  // 设定临时变量，用于存放所要操作图像的对应像素点列坐标。
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= (inimg.imgMeta.width + 1) / 2 || r >= inimg.imgMeta.height)
        return;

    // 因为是水平翻转，所以行坐标相同，只需计算对应图像像素点的横坐标。
    sorc = inimg.imgMeta.width - c - 1;
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + sorc;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp1,intemp2;
    intemp1 = inimg.imgMeta.imgData[inidx];
    intemp2 = inimg.imgMeta.imgData[outidx];

    // 一个线程处理四个像素点。
    // 进行水平方向上相应的像素点的像素值的交换。
    // 线程中处理的第一个像素点。
    outimg.imgMeta.imgData[outidx] = intemp1;
    outimg.imgMeta.imgData[inidx] = intemp2;

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
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

        // 处理此线程中后三个像素点。
        intemp1 = inimg.imgMeta.imgData[inidx];
        intemp2 = inimg.imgMeta.imgData[outidx];
        outimg.imgMeta.imgData[outidx] = intemp1;
        outimg.imgMeta.imgData[inidx] = intemp2;
    }
}

// Host 成员方法：flipHorizontal（图像水平翻转）
__host__ int Flip::flipHorizontal(Image *inimg,Image *outimg)
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
    gridsize.x = ((outsubimgCud.imgMeta.width + 1) / 2 + blocksize.x - 1) /
                 blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 调用核函数
    _flipHorizontalKer<<<gridsize,blocksize>>>(
            insubimgCud,outsubimgCud);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Kernel 函数：_flipVerticalKer（图像竖直翻转）
static __global__ void _flipVerticalKer(ImageCuda inimg,ImageCuda outimg)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    int sorr;  // 设定临时变量，用于存放所要操作图像的对应像素点行坐标。

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= (inimg.imgMeta.height + 1) / 2)
        return;

    // 因为是竖直翻转，所以列坐标相同，只需计算对应图像像素点的行坐标。
    sorr = inimg.imgMeta.height - r - 1;
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = sorr * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp1,intemp2;
    intemp1 = inimg.imgMeta.imgData[inidx];
    intemp2 = inimg.imgMeta.imgData[outidx];

    // 一个线程处理四个像素点。
    // 进行水平方向上相应的像素点的像素值的交换。
    // 线程中处理的第一个像素点。
    outimg.imgMeta.imgData[outidx] = intemp1;
    outimg.imgMeta.imgData[inidx] = intemp2;

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= (outimg.imgMeta.height + 1) / 2)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx -= outimg.pitchBytes;

        // 处理此线程中后三个像素点。
        intemp1 = inimg.imgMeta.imgData[inidx];
        intemp2 = inimg.imgMeta.imgData[outidx];
        outimg.imgMeta.imgData[outidx] = intemp1;
        outimg.imgMeta.imgData[inidx] = intemp2;
    }
}

// Host 成员方法：flipVertical（图像竖直翻转）
__host__ int Flip::flipVertical(Image *inimg,Image *outimg)
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
    gridsize.y = ((outsubimgCud.imgMeta.height + 1) /
                  2 + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 调用核函数
    _flipVerticalKer<<<gridsize,blocksize>>>(
            insubimgCud, outsubimgCud);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

