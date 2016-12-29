// MultiThreshold.cu
// 实现图像的多阈值二值化图像生成操作

#include "MultiThreshold.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏： DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_multithresholdKer（多阈值二值化图像生成）
// 对输入图像进行多阈值二值化处理。以 1 - 254 内的所有的灰度为阈值,同时生成
// 254个2值化结果(0-1)图像。
static __global__ void      // Kernel 函数无返回值
_multithresholdKer(
        ImageCuda inimg,    // 输入图像
        ImageCuda outimg[]  // 输出图像
);

// Kernel 函数: _multithresholdKer（多阈值二值化图像生成）
static __global__ void _multithresholdKer(ImageCuda inimg, 
                                          ImageCuda *outimg)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标的
    // x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻行上，
    // 因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    int z = blockIdx.z;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height || z >= 254)
        return;
        
    // 计算对应阈值，由 1 - 254
    int threshold = z + 1;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg[z].pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp;
    intemp = inimg.imgMeta.imgData[inidx];
    outimg[z].imgMeta.imgData[outidx] = (intemp >= threshold ? 255 : 0);
    
    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= outimg[z].imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += outimg[z].pitchBytes;
        intemp = inimg.imgMeta.imgData[inidx];

        // 如果输入图像的该位置的像素值大于等于 threshold，则将输出图像中对应
        // 位置的像素值置为 255；否则将输出图像中对应位置的像素值置为 0。线程
        // 中处理的第一个点。
        outimg[z].imgMeta.imgData[outidx] = (intemp >= threshold ? 255 : 0);
    }
}

// Host 成员方法：multithreshold（多阈值二值化处理）
__host__ int MultiThreshold::multithreshold(Image *inimg, Image *outimg[254])
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;
    
    // 检查输出图像是否为 NULL，如果为 NULL 直接报错返回。
    for(int i = 0; i < 254; i++) {
        if (outimg[i] == NULL)
            return NULL_POINTER;
    }
    
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
        	
    // 将输出图像拷贝入 Device 内存。
    for (int i = 0; i < 254; i++) {
        errcode = ImageBasicOp::copyToCurrentDevice(outimg[i]);
        if (errcode != NO_ERROR) {
            // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输
            // 入图像的 ROI 子图像尺寸相同的图像。
            errcode = ImageBasicOp::makeAtCurrentDevice(outimg[i], 
            inimg->roiX2 - inimg->roiX1, 
            inimg->roiY2 - inimg->roiY1);

            // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
            if (errcode != NO_ERROR)
                return errcode;
        }
    }
	
    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud[254], *outsubimgCudDev;
    
    // 对 254 幅输出图像分别进行提取。
    for (int i = 0; i < 254; i++) {
    errcode = ImageBasicOp::roiSubImage(outimg[i], &outsubimgCud[i]);
    if (errcode != NO_ERROR)
        return errcode;
    }
    
    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一。
    for (int i = 0; i < 254; i++) {
        if (insubimgCud.imgMeta.width > outsubimgCud[i].imgMeta.width)
            insubimgCud.imgMeta.width = outsubimgCud[i].imgMeta.width;
        else
            outsubimgCud[i].imgMeta.width = insubimgCud.imgMeta.width;
          
        if (insubimgCud.imgMeta.height > outsubimgCud[i].imgMeta.height)
            insubimgCud.imgMeta.height = outsubimgCud[i].imgMeta.height;
        else
            outsubimgCud[i].imgMeta.height = insubimgCud.imgMeta.height;
    }
    
    // 为 outsubimgCudDev 分配内存空间。
    errcode = cudaMalloc((void **)&outsubimgCudDev, 
                         254 * sizeof (ImageCuda));
    if (errcode != NO_ERROR) 
        return errcode;
    
    // 将 Host 上的 outsubimgCudDev 拷贝到 Device 上。
    errcode = cudaMemcpy(outsubimgCudDev, outsubimgCud,
                         254 * sizeof (ImageCuda), cudaMemcpyHostToDevice);
    if (errcode != NO_ERROR) {
        cudaFree(outsubimgCudDev);
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = 1;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 设定 gridsize.z 的大小为 254，存储 1 - 254 的阈值。
    gridsize.z = 254;
    
    // 调用 Kernel 函数，实现 254 幅图像生成
    _multithresholdKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCudDev);
    
    // 释放已分配的数组内存，避免内存泄露
    cudaFree(outsubimgCudDev);
                                            
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
        
    // 处理完毕，退出。	
    return NO_ERROR;
}

