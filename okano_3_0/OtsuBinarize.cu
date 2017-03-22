// OtsuBinarize.cu
// 根据两个领域之间的分散程度,自动找到最佳二值化结果

#include "OtsuBinarize.h"


#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏： DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_OtsuBinarizeKer（最佳二值化自动生成）
// 用 Histogram searching 法自动找到最佳阈值(最佳二值化结果)
// 判断的根据:两个领域之间的分散最大(内分散最小)
static __global__ void      // Kernel 函数无返回值
_OtsuBinarizeKer(
        ImageCuda inimg,              // 输入图像
        ImageCuda outimg,             // 输出图像
        unsigned char threshold       // 阈值
);

// Kernel 函数: _OtsuBinarizeKer（最佳二值化自动生成）
static __global__ void _OtsuBinarizeKer(ImageCuda inimg, 
                                        ImageCuda outimg,        
                                        unsigned char threshold)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线 程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像 素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理， 一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >=  inimg.imgMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp;
    intemp = inimg.imgMeta.imgData[inidx];
    
    // 一个线程处理四个像素点.
    // 如果输入图像的该位置的像素值大于等于 threshold，则 将输出图像中对应位
    // 置的像素值置为 255；否则将输出图像中对应位置的像素 值置为 0。
    // 线程中处理的第一个点。
    outimg.imgMeta.imgData[outidx] = (intemp >=  threshold ? 255 : 0);
    
    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而  x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y  分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图 像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可 ，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += outimg.pitchBytes;
        intemp = inimg.imgMeta.imgData[inidx];

        // 如果输入图像的该位置的像素值大于等于 threshold ，则将输出图像中对应
        // 位置的像素值置为 255；否则将输出图像中对应位置 的像素值置为 0。线程
        // 中处理的第一个点。
        outimg.imgMeta.imgData[outidx] = (intemp >= threshold ? 255 : 0);
    }
}

// Host 成员方法：otsuBinarize（最佳二值化自动生成）
__host__ int OtsuBinarize::otsuBinarize(Image *inimg, Image *outimg)
{
    // 检查输入图像和输出图像是否为 NULL，如果为 NULL 直接报错返回 。
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
    errcode = ImageBasicOp::copyToCurrentDevice (outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败）， 则会创建一个和输入图
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
    
    // 根据子图像的大小对长，宽进行调整，选择长度小的长， 宽进行子图像的统一	
    if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width)
        insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
    else
        outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;
		
    if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height)
        insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
    else
        outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;
         
    // 调用直方图，获取图像的像素信息        
    Histogram h;
    
    // 图像的像素信息
    unsigned int his[256];
    h.histogram(inimg, his, true);
    
    // 图像总像素数 
    int sumpixel = 0;                    
    for (int i = 0; i < 256; i++) {
        sumpixel += his[i];
    }
    
    // 获取图像的最小和最大有效像素值
    int imin = 0, imax = 255;       
    for (int i = 0; his[i] == 0 && i <= 255; i++) {
        imin++;
    }
    for (int i = 255; his[i] == 0 && i >= 0; i--) {
        imax--;
    }
    
    // 根据图像信息计算最大类间方差,使用如下公式：
    // max =(μ_T ω_1 (k)-μ_1 (k))^2/(ω_1 (k)(1-ω_1(k)))
    float pix_w[256];    // 数组 pix_w[i] 记录图像中值为i的像素占总像素数的比例
    float weights[256];  // 数组 weights[i] 记录图像以i为阈值时，前景的像素数占
                         // 总像素数的比例
    float means[256];    // 数组 means[i] 记录图像以i为阈值时，前景的平均灰度值
    
    // 计算 pix_w[i] 的值
    for (int i = 0; i < 256; i++) {
        pix_w[i] = ((float)his[i]) / sumpixel;
    }
    
    // 计算 weights[i] 和 means[i] 的值
    weights[0] = pix_w[0];
    means[0] = 0.0;
    for (int i = 1; i < 256; i++) {
        weights[i] = weights[i - 1] + pix_w[i];
        means[i] = means[i - 1] + (i * pix_w[i]);
    }
    
    // 计算类间方差，并找出最大类间方差，同时记录最佳阈值
    float mean = means[255];
    float max = 0.0;
    int threshold = 0;
    for (int i = imin; i <= imax; i++) {
        float bcv = mean * weights[i] - means[i];
        bcv *= bcv / (weights[i] * (1.0 - weights[i]));
        if (max < bcv) {
            max = bcv;
            threshold = i;
        }
    }
    
    // 将阈值进行类型转换。
    unsigned char thresholds = (unsigned char)threshold;
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);
    
    // 调用核函数，使用最佳阈值对图像进行二值化
    _OtsuBinarizeKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud, 
                                              thresholds);
    
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
        
    // 处理完毕，退出。	
    return NO_ERROR;
}

