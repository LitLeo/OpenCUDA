// SimpleBrightnessGradient.cu
// 实现简单亮度渐变算法

#include "SimpleBrightnessGradient.h"                                          
#include <math.h> 

#include <iostream> 
using namespace std;

#include "ErrorCode.h" 

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  16 
#define DEF_BLOCK_Y  16 

// Kernel 函数： SimpleBrightnessGradient （简单亮度渐变） 
static __global__ void    // Kernel 函数无返回值
_simpleBrightnessGradientKer( 
        ImageCuda inimg,  // 输入图像
        ImageCuda outimg  // 输出图像
);

// Kernel 函数：SimpleBrightnessGradient（简单亮度渐变）
static __global__ void _simpleBrightnessGradientKer(ImageCuda inimg, 
                                                    ImageCuda outimg) 
{ 
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 
    //（其中，c 表示 column， r 表示 row）。由于采用并行度缩减策略 ，
    // 令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行上，
    // 因此，对于r 需要进行乘 4 的计算
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，
    //一方面节省计算资源，一方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height) 
        return;

    // 计算输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;

    // 计算输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;

    // 读取输入坐标点对应的像素值
    unsigned char inpixel;
    inpixel = inimg.imgMeta.imgData[inidx];

    unsigned char intemp;
    const unsigned char temp = sqrtf(inimg.imgMeta.width * 
                               inimg.imgMeta.width + 
                               inimg.imgMeta.height * inimg.imgMeta.height);
    intemp = (1 - (1 / temp) * sqrtf(c * c + r * r)) * 255 + 
             (1 / temp) * sqrtf(c * c + r * r) * inpixel;

    //输出图像像素
    outimg.imgMeta.imgData[outidx] = intemp;

    // 处理剩下的 3 个点
    for(int i = 1; i < 4; i++) { 
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查
        if (r + i > inimg.imgMeta.height) 
            return;

        // 获取当前列的下一行的位置指针
        inidx += inimg.pitchBytes;
        outidx += outimg.pitchBytes;

        intemp = (1 - (1 / temp) * sqrtf(c * c + r * r)) * 255 +  
                 (1 / temp) * sqrtf(c * c + r * r) * inpixel;

        //输出图像像素
        outimg.imgMeta.imgData[outidx] = intemp;
    }
}

// Host 成员方法: SimpleBrightnessGradient（简单亮度渐变）
__host__ int SimpleBrightnessGradient::simpleBrightnessGradient(Image *inimg, 
                                                                Image *outimg) 
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
        // 计算 roi 子图的宽和高
        int roiwidth = inimg->roiX2 - inimg->roiX1; 
        int roiheight = inimg->roiY2 - inimg->roiY1;

        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, roiwidth, 
                                                    roiheight); 
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

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / 
                  blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) / 
                 (blocksize.y * 4);

    // 调用核函数
    _simpleBrightnessGradientKer<<<gridsize, blocksize>>>(insubimgCud, 
                                                         outsubimgCud);
    if (cudaGetLastError() != cudaSuccess) 
        return CUDA_ERROR;
    
    // 处理完毕，退出。
    return NO_ERROR;
} 

