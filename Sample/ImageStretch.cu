// ImageStretch.cu
// 实现对图像的拉伸处理

#include "ImageStretch.h"
#include "ErrorCode.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 全局变量：texRef（作为输入图像的纹理内存引用）
// 纹理内存只能用于全局变量，因此将硬件插值的旋转变换的 Kernel 函数的输入图像列
// 于此处。
static texture<unsigned char, 2, cudaReadModeElementType> texRef;

// Kernel 函数：_performImgStretchKer（拉伸图像）
// 根据给定的拉伸倍数 timesWidth 和 timesHeight，将输入图像拉伸，将其尺寸从
// width * height 变成(width * timesWidth) * (height * timesHeight)。
static __global__ void     // Kernel 函数无返回值
_performImgStretchKer(
        ImageCuda outimg,  // 输出图像
        float timeswidth,  // 宽度拉伸倍数
        float timesheight  // 高度拉伸倍数
);
  
// Kernel 函数：_performImgStretchKer（拉伸图像）
static __global__ void _performImgStretchKer(
        ImageCuda outimg, float timeswidth, float timesheight)
{
    // 计算当前线程的位置。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，另
    // 一方面防止由于段错误导致程序崩溃。
    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;
  
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;

    // 通过目标坐标点反推回源图像中的坐标点。
    float inc = c / timeswidth;
    float inr = r / timesheight;

    // 通过上面的计算，求出了第一个输出坐标对应的源图像坐标。这里利用纹理内存的
    // 硬件插值功能，直接使用浮点型的坐标读取相应的源图像“像素”值，并赋值给目标
    // 图像。这里没有进行对源图像读取的越界检查，这是因为纹理内存硬件插值功能可
    // 以处理越界访问的情况，越界访问会按照事先的设置得到一个相对合理的像素颜色
    // 值，不会引起错误。
    outimg.imgMeta.imgData[outidx] = tex2D(texRef, inc, inr);
}
  
// Host 成员方法：performImgStretch（图像拉伸处理）
__host__ int ImageStretch::performImgStretch(Image *inimg, Image *outimg)
{
    // 检查输入图像，输出图像是否为空
    if (inimg == NULL)
        return NULL_POINTER;
  
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码。
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图像
        // 的 ROI 子图像拉伸后尺寸相同的图像。
        int outwidth = (inimg->roiX2 - inimg->roiX1) * timesWidth;
        int outheight = (inimg->roiY2 - inimg->roiY1) * timesHeight;

        // 判断输出图像尺寸是否为 0，若为 0，报错退出。
        if(outwidth == 0 || outheight == 0)
            return INVALID_DATA;
    
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, outwidth, outheight);
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
  
    // 为CUDA数组分配内存，并将输入图像拷贝到内存
    struct cudaChannelFormatDesc channelDesc;
    channelDesc = cudaCreateChannelDesc(sizeof (unsigned char) * 8, 0, 0, 0, 
                                        cudaChannelFormatKindUnsigned);
    
    // 纹理和数组绑定
    cudaError_t cuerrcode;
    cuerrcode = cudaBindTexture2D(NULL, &texRef, inimg->imgData, &channelDesc, 
                                  inimg->width, inimg->height, 
                                  insubimgCud.pitchBytes);
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;
  
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;
    
    // 调用核函数。
    _performImgStretchKer<<<gridsize, blocksize>>>(outsubimgCud, timesWidth, 
                                                   timesHeight);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
    // 处理完毕，退出。
    return NO_ERROR;
}
  