// ImageDiff.cu
// 实现两幅图像相减

#include "Image.h"
#include "ImageDiff.h"
#include "ErrorCode.h"

#include <iostream>
#include <stdio.h>
#include <cmath>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_imageDiffKer
// 根据输入的两幅灰度图像，对其相应为位置的像素值做差得到差值图像 outimg
static __global__ void     // 无返回值
_imageDiffKer(
        ImageCuda inimg1,  // 输入图像 1
        ImageCuda inimg2,  // 输入图像 2
        ImageCuda outimg   // 输出图像
); 
   
// Kernel 函数:_imageDiffKer
static __global__ void _imageDiffKer(ImageCuda inimg1,
                                     ImageCuda inimg2, ImageCuda outimg)
{
    // 获取线程索引 
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    // 由于我们采用了并行度缩减的策略，令一个线程处理 4 个输出像素,这四
    // 个像素位于统一列的相邻 4 行上因此，对于 yidx 需要进行乘 4 计算。
    // 定义变量 idx_x 和 idx_y 用来保存 xidx 和 yidx * 4 的 
    // 值，并作为新的索引
    int idx_x = xidx;
    int idx_y = yidx * 4;

    // 判断 idx _y 和 idx_x 是否超过了图像的尺寸
    if (idx_x > inimg1.imgMeta.width ||
        idx_x > inimg2.imgMeta.width ||
        idx_y > inimg1.imgMeta.height ||
        idx_y > inimg2.imgMeta.height)
        return;

    // 转化为图像数组下标
    int idout = idx_y * outimg.pitchBytes + idx_x;
    int idimg1 = idx_y * inimg1.pitchBytes + idx_x;
    int idimg2 = idx_y * inimg2.pitchBytes + idx_x;

    // 处理第一个点  
    // 对应灰度值相减,为了保证差值为正数，对差取绝对值
    outimg.imgMeta.imgData[idout] = abs(inimg1.imgMeta.imgData[idimg1] -
                                        inimg2.imgMeta.imgData[idimg2]);

    // 处理统一列上剩下的三个点   
    for ( int i = 1; i < 4; i ++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查
        if (idx_y + i > inimg1.imgMeta.height ||
            idx_y + i > inimg2.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitchBytes 即可，不需要在进行
        // 乘法计算。
        idout = idout + outimg.pitchBytes;
        idimg1 = idimg1 + inimg1.pitchBytes;
        idimg2 = idimg2 + inimg2.pitchBytes;

        // 对应灰度值相减,为了保证差值为正数，对差取绝对值
        outimg.imgMeta.imgData[idout] = abs(inimg1.imgMeta.imgData[idimg1] -
                                            inimg2.imgMeta.imgData[idimg2]);
    }     
}

// Host成员方法:imageDiff
// 实现两幅图像的相减
__host__ int ImageDiff::imageDiff(Image *inimg1,
                                  Image *inimg2, Image *outimg)
{ 
    // 检查输入图像是否为 NULL
    if (inimg1 == NULL || inimg2 == NULL)
        return NULL_POINTER;
   
    // 检查图像是否为空
    if (inimg1->imgData == NULL || inimg2->imgData == NULL )
        return UNMATCH_IMG;

    // 如何两幅输入图像的 ROI 区域大小不一样，
    // 那么将大图像的 ROI 大小设定为小图像的 ROI 大小
    int img1roilx = inimg1->roiX2 - inimg1->roiX1;
    int img1roily = inimg1->roiY2 - inimg1->roiY1;
    int img2roilx = inimg2->roiX2 - inimg2->roiX1;
    int img2roily = inimg2->roiY2 - inimg2->roiY1;

    if (img1roilx > img2roilx) {
        // 当 img1 的 ROI 比 img2 宽,将 img1 的 ROIX 大小该为与 img2 的一样
        inimg1->roiX2 = inimg2->roiX2;
        inimg1->roiX1 = inimg2->roiX1;
    } else {
        // 当 img2 的 ROI 比img1 宽,将 img2 的 ROIX 大小该为与 img1 的一样
        inimg2->roiX2 = inimg1->roiX2; 
        inimg2->roiX1 = inimg1->roiX1;
    }  
                 
    if (img1roily > img2roily) {
        // 当 img1 的 ROI 比 img2 长,将 img1 的 ROIY 大小该为与 img2 的一样
        inimg1->roiY2 = inimg2->roiY2;
        inimg1->roiY1 = inimg2->roiY1;
    } else {
        // 当 img2 的 ROI 比 img1 长,将 img2 的 ROIY 大小该为与 img2 的一样
        inimg2->roiY2 = inimg1->roiY2; 
        inimg2->roiY1 = inimg1->roiY1;
    }  
        
    // 将输入图像 1 复制到 device
    int errcode;
    errcode = ImageBasicOp::copyToCurrentDevice(inimg1);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入图像 2 复制到 device
    errcode = ImageBasicOp::copyToCurrentDevice(inimg2);
    if (errcode != NO_ERROR)
        return errcode;
        
    // 将 outimg 复制到 device
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    
    // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和
    // 小输入图像尺寸相同的图像。
    int outw = inimg1->width > inimg2->width ? inimg1->width : 
               inimg1->width;
    int outh = inimg1->height > inimg2->height ? inimg1->height : 
               inimg1->height;
                 
    if (errcode != NO_ERROR) {
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, 
                                                    outw, outh);
        // 如果创建图像也操作失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }
    
    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud1,insubimgCud2;
    errcode = ImageBasicOp::roiSubImage(inimg1, &insubimgCud1);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::roiSubImage(inimg2, &insubimgCud2);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud1.imgMeta.width  + blocksize.x - 1) / 
                  blocksize.x;
    gridsize.y = (insubimgCud1.imgMeta.height + blocksize.y * 4 - 1) / 
                  (blocksize.y * 4);
    // 调用 kernel 函数_imageDiffKer
    _imageDiffKer<<<gridsize,blocksize>>>(insubimgCud1,
                                          insubimgCud2, outsubimgCud);

    // 调用 cudaGetLastError 判断程序是否出错
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        return CUDA_ERROR;
        
    // 处理完毕，退出             
    return NO_ERROR;    
}

