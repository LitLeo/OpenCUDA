// Threshold.cu
// 实现图像的阈值分割

#include "Threshold.h"
#include <iostream>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_thresholdKer（使用 ImageCuda 实现的阈值分割）
// 有输出图像但无高低像素值（low,high）
static __global__ void           // kernel 函数无返回值
_thresholdKer( 
        ImageCuda in,            // 输入图像
        ImageCuda out,           // 输出图像
        unsigned char minpixel,  // 最小像素值
        unsigned char maxpixel   // 最大像素值
);

// Kernel 函数：_thresholdKer（使用 ImageCuda 实现的阈值分割）
// 无输出图像且无高低像素值（low, high）
static __global__ void           // kernel 函数无返回值
_thresholdKer(
        ImageCuda inout,         // 输入输出图像
        unsigned char minpixel,  // 最小像素值
        unsigned char maxpixel   // 最大像素值
);

// Kernel 函数：_thresholdKer（使用 ImageCuda 实现的阈值分割）
// 有输出图像且有高低像素值（low, high）
static __global__ void           // kernel 函数无返回值
_thresholdKer(
        ImageCuda in,            // 输入图像
        ImageCuda out,           // 输出图像
        unsigned char minpixel,  // 最小像素值
        unsigned char maxpixel,  // 最大像素值
        unsigned char low,       // 低像素值
        unsigned char high       // 高像素值
); 

// Kernel 函数：_thresholdKer（使用 ImageCuda 实现的阈值分割）
// 无输出图像但有高低像素值（low, high）
static __global__ void           // kernel 函数无返回值
_thresholdKer(
        ImageCuda inout,         // 输入输出图像
        unsigned char minpixel,  // 最小像素值
        unsigned char maxpixel,  // 最大像素值
        unsigned char low,       // 低像素值
        unsigned char high       // 高像素值
); 


// Kernel 函数：_thresholdKer（使用ImageCuda实现的阈值分割）
static __global__ void _thresholdKer(
        ImageCuda in, ImageCuda out, unsigned char minpixel, 
        unsigned char maxpixel)
{
    // 计算想成对应的输出点的位置，其中 dstc 和 dstr 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻
    // 4 行上，因此，对于 dstr 需要进行乘 4 计算。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (dstc >= in.imgMeta.width || dstr >= in.imgMeta.height)
        return;
     
    // 计算第一个输入坐标点和输出坐标点对应的图像数据数组下标。
    int dstidx = dstr * in.pitchBytes + dstc;
    int outidx = dstr * out.pitchBytes + dstc; 

    // 根据点的像素值进行阈值分割
    if(in.imgMeta.imgData[dstidx] < minpixel || in.imgMeta.imgData[dstidx] > 
       maxpixel)
        out.imgMeta.imgData[outidx] = 0;
    else
        out.imgMeta.imgData[outidx] = in.imgMeta.imgData[dstidx];  

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++dstr >= out.imgMeta.height)
            return;
        
        // 计算输入坐标点以及输出坐标点，由于只有 y 分量增加 1，所以下标只需要加
        // 上对应的 pitch 即可，不需要在进行乘法计算
        dstidx += in.pitchBytes;
        outidx += out.pitchBytes;
        
        // 若输入点像素在阈值范围内，输出点像素即为对应输入点像素，否则为 0 
        if(in.imgMeta.imgData[dstidx] < minpixel || 
           in.imgMeta.imgData[dstidx] > maxpixel)
            out.imgMeta.imgData[outidx] = 0;
        else
            out.imgMeta.imgData[outidx] = in.imgMeta.imgData[dstidx];
    }
}

// Host 成员方法：threshold（阈值分割）
// 未指定高低像素值且输出图像不为 NULL 的阈值分割。
__host__ int Threshold::threshold(Image *inimg, Image *outimg)
{  
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;
        
    // 如果输出图像为NULL，直接调用 In—Place 版本的成员方法。    
    if (outimg == NULL)
        return threshold(inimg);

    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->roiX2 - inimg->roiX1, 
                inimg->roiY2 - inimg->roiY1);
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

    // 调用对应的 kernel 函数进行计算
    _thresholdKer<<<gridsize, blocksize>>>(
               insubimgCud, outsubimgCud, minPixelVal, maxPixelVal);
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
    return NO_ERROR;
}

// Kernel 函数：_thresholdKer（使用 ImageCuda 实现的阈值分割）
static __global__ void _thresholdKer(
        ImageCuda inout, unsigned char minpixel, unsigned char maxpixel)
{
    // 计算对应输出点的下标
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
        
    // 越界检查，若越界则不作任何处理直接退出
    if (dstc >= inout.imgMeta.width || dstr >= inout.imgMeta.height)
        return;
    
    int dstidx = dstr * inout.pitchBytes + dstc;
    
    // 在图像本身上进行阈值分割，不在阈值内置0否则保持不变
    if(inout.imgMeta.imgData[dstidx] < minpixel || 
       inout.imgMeta.imgData[dstidx] > maxpixel)
        inout.imgMeta.imgData[dstidx] = 0;
    
    // 处理剩下的三个点
    for (int i = 0; i < 3; i++) {
        
        if (++dstr >= inout.imgMeta.height)
            return;
        // 计算输入坐标点，由于只有 y 分量增加 1，所以下标只需要加
        // 上一个 pitch 即可，不需要在进行乘法计算
        dstidx += inout.pitchBytes;
    
        //若输入点像素在阈值范围内，输出点像素保持不变，否则为 0 
        if(inout.imgMeta.imgData[dstidx] < minpixel || 
           inout.imgMeta.imgData[dstidx] > maxpixel)
            inout.imgMeta.imgData[dstidx] = 0;
    }
}

// Host 成员方法：threshold（阈值分割）
__host__ int Threshold::threshold(Image *inoutimg)
{
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inoutimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的子图像
    ImageCuda inoutimgCud;
    errcode = ImageBasicOp::roiSubImage(inoutimg, &inoutimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (inoutimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (inoutimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 调用对应的 kernel 函数进行计算
    _thresholdKer<<<gridsize, blocksize>>>(inoutimgCud, minPixelVal, 
                                              maxPixelVal);
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
    return NO_ERROR;
}

// Kernel 函数：_thresholdKer（使用 ImageCuda 实现的阈值分割）
static __global__ void _thresholdKer(
        ImageCuda in, ImageCuda out, unsigned char minpixel, 
        unsigned char maxpixel, unsigned char low, unsigned char high)
{
    // 计算对应输出点的下标
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    
    // 越界检查，若越界则不作任何处理直接退出
    if (dstc >= in.imgMeta.width || dstr >= in.imgMeta.height)
        return;
    
    // 计算第一个输入坐标点和输出坐标点对应的图像数据数组下标。
    int dstidx = dstr * in.pitchBytes + dstc;
    int outidx = dstr * out.pitchBytes + dstc;
    
    // 根据输入图像进行阈值分割，不在阈值内置为低像素：low否则置为高像素：high
    if(in.imgMeta.imgData[dstidx] < minpixel || in.imgMeta.imgData[dstidx] > 
       maxpixel)
        out.imgMeta.imgData[outidx] = low;
    else
        out.imgMeta.imgData[outidx] = high;
   
    // 处理剩下的三个点
    for (int i = 0; i < 3; i++) {
        
        if (++dstr >= out.imgMeta.height)
            return;
        
        // 计算输入坐标点以及输出坐标点，由于只有 y 分量增加 1，所以下标只需要
        // 加上对应的 pitch 即可，不需要在进行乘法计算
        dstidx += in.pitchBytes;
        outidx += out.pitchBytes;
        
        // 若输入点像素在阈值范围内，输出点像素为高像素：high，否则为低像素：
        // low
        if(in.imgMeta.imgData[dstidx] < minpixel || 
           in.imgMeta.imgData[dstidx] > maxpixel)
            out.imgMeta.imgData[outidx] = low;
        else
            out.imgMeta.imgData[outidx] = high;
    }
}

// Host 成员方法：threshold（阈值分割）
__host__ int Threshold::threshold(
        Image *inimg, Image *outimg, unsigned char low, unsigned char high)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;
        
    // 如果输出图像为NULL，直接调用 In—Place 版本的成员方法。    
    if (outimg == NULL)
        return threshold(inimg, low, high);

    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 将输出图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->roiX2 - inimg->roiX1, 
                inimg->roiY2 - inimg->roiY1);
    if (errcode != NO_ERROR)
        return errcode;
    }

    // 提取输入图像的子图像
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的子图像
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

    // 调用对应的 kernel 函数进行计算
    _thresholdKer<<<gridsize, blocksize>>>(
           insubimgCud, outsubimgCud, minPixelVal, maxPixelVal,low,high);
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
    return NO_ERROR;
}

// Kernel 函数：_thresholdKer（使用 ImageCuda 实现的阈值分割）
static __global__ void _thresholdKer(
        ImageCuda inout, unsigned char minpixel, unsigned char maxpixel,
        unsigned char low, unsigned char high)
{
    // 计算对应输出点的下标
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 越界检查，若越界则不作任何处理直接退出
    if (dstc >= inout.imgMeta.width || dstr >= inout.imgMeta.height)
        return;
    
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int dstidx = dstr * inout.pitchBytes + dstc;

    // 在图像自身上进行阈值分割，不在阈值内置为低像素：low 否则置为高像素：high
    if(inout.imgMeta.imgData[dstidx] < minpixel || 
       inout.imgMeta.imgData[dstidx] > maxpixel)
        inout.imgMeta.imgData[dstidx] = low;
    else
        inout.imgMeta.imgData[dstidx] = high;
    
    // 处理剩下的三个点
    for (int i = 0; i < 3; i++) {
        
        if (++dstr >= inout.imgMeta.height)
            return;
        
        // 计算输入坐标点以及输出坐标点，由于只有 y 分量增加 1，所以下标只需要加
        // 上对应的 pitch 即可，不需要在进行乘法计
        dstidx += inout.pitchBytes;
        
        // 在图像自身上进行阈值分割，不在阈值内置为低像素：low 否则置为高像素：
        // high
        if(inout.imgMeta.imgData[dstidx] < minpixel || 
           inout.imgMeta.imgData[dstidx] > maxpixel)
            inout.imgMeta.imgData[dstidx] = low;
        else
            inout.imgMeta.imgData[dstidx] = high;
    } 
}

// Host 成员方法：threshold（阈值分割）
__host__ int Threshold::threshold(Image *inoutimg, unsigned char low,  
                                  unsigned char high)
{
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inoutimg);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输入图像的ROI子图像
    ImageCuda inoutimgCud;
    errcode = ImageBasicOp::roiSubImage(inoutimg, &inoutimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (inoutimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (inoutimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 调用对应的 kernel 函数进行计算
    _thresholdKer<<<gridsize, blocksize>>>(
        inoutimgCud, minPixelVal, maxPixelVal, low, high);
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
    return NO_ERROR;
}
