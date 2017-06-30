// ImageFilter.cu
// 实现多阈值图像过滤

#include "ImageFilter.h"
#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X 32
#define DEF_BLOCK_Y 8

//static __global__ void _calImageFilter(ImageCuda inimg, ImageCuda outimg, 
//                                       int lowthreshold, int highthreshold);

// kernel 函数：_calImageFilterOpt (多阈值图像过滤)
// 根据输入图像的灰度值的大小，判断其位于阈值数组中的区间，然后根据区间号将对应
// 图像的对应位置设成改灰度值，将其余区间的对应的图像的对应位置的灰度值设为0。
// 该核函数运用了并行度缩减策略，每一个线程处理同一列的四个像素点
static __global__ void      // kernel函数无返回值
_calImageFilterOpt(
        ImageCuda inimg,    // 输入图像  
        ImageCuda *outimg,  // 输出图像数组
        int num,            // 输出图像数组大小和阈值数组大小 
        int *threshold      // 阈值数组
);

/*static __global__ void _calImageFilter(ImageCuda inimg, ImageCuda outimg, 
                                         int lowthreshold, int highthreshold)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if(c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    int index = r * inimg.pitchBytes + c;

    if(inimg.imgMeta.imgData[index] <= highthreshold && inimg.imgMeta.imgData[index] >= lowthreshold) {
        outimg.imgMeta.imgData[index] = inimg.imgMeta.imgData[index];
    }
    else {
        outimg.imgMeta.imgData[index] = 0;
    }
}*/

// kernel 函数：_calImageFilterOpt (多阈值图像过滤)
static __global__ void _calImageFilterOpt(ImageCuda inimg, ImageCuda *outimg, 
                                          int num, int *threshold)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标的
    // x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻行上，
    // 因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 局部变量 i，用于循环
    int i;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if(c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算第一个输入坐标点对应的图像数据数组下标。
    int index = r * inimg.pitchBytes + c;

    // 循环遍历每一个阈值区间，判断输入图像该像素点的区间位置
    for(i = 0; i < num - 1; i++) {
        // 判断像素点的区间位置，如果是该区间，将该区间对应的图像的对应位置的灰度值设为该灰度值，如果不是
        // 则设为0。
        if(inimg.imgMeta.imgData[index] >= threshold[i] && inimg.imgMeta.imgData[index] <= threshold[i + 1])
            ((ImageCuda)outimg[i]).imgMeta.imgData[index] = inimg.imgMeta.imgData[index];
        else
             ((ImageCuda)outimg[i]).imgMeta.imgData[index] = 0;
    }

    // 对最后一个区间处理，之所以放在循环外，因为阈值数组中没有255。
    if(inimg.imgMeta.imgData[index] >= threshold[i])
         ((ImageCuda)outimg[i]).imgMeta.imgData[index] = inimg.imgMeta.imgData[index];
    else
         ((ImageCuda)outimg[i]).imgMeta.imgData[index] = 0;

     // 处理剩下的三个像素点。
    for(int j = 0; j < 3; j++) {

        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if(++r >= inimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        index += inimg.pitchBytes;

        // 循环遍历每一个阈值区间，判断输入图像该像素点的区间位置
        for(i = 0; i < num - 1; i++) {
            // 判断像素点的区间位置，如果是该区间，将该区间对应的图像的对应位置的灰度值设为该灰度值，如果不是
            // 则设为0。
            if(inimg.imgMeta.imgData[index] >= threshold[i] && inimg.imgMeta.imgData[index] <= threshold[i + 1])
                ((ImageCuda)outimg[i]).imgMeta.imgData[index] = inimg.imgMeta.imgData[index];
            else
                ((ImageCuda)outimg[i]).imgMeta.imgData[index] = 0;
        }

        // 对最后一个区间处理，之所以放在循环外，因为阈值数组中没有255。
        if(inimg.imgMeta.imgData[index] >= threshold[i])
             ((ImageCuda)outimg[i]).imgMeta.imgData[index] = inimg.imgMeta.imgData[index];
        else
            ((ImageCuda)outimg[i]).imgMeta.imgData[index] = 0;
    }
}

/*void ImageFilter::calImageFilter(Image *inimg, Image ***outimg)
{
    dim3 gridsize;
    dim3 blocksize;

    if(inimg == NULL)
        return;

    *outimg = new Image *[this->thresholdNum];

    for(int i = 0; i < this->thresholdNum; i++) {
        ImageBasicOp::newImage(&((*outimg)[i]));
        ImageBasicOp::makeAtHost((*outimg)[i], inimg->width, inimg->height);
    }

    ImageBasicOp::copyToCurrentDevice(inimg);


    for(int i = 0; i < this->thresholdNum; i++) {
        
        ImageBasicOp::copyToCurrentDevice((*outimg)[i]);

        ImageCuda insubimgCud;
        ImageCuda outsubimgCud;

        ImageBasicOp::roiSubImage(inimg, &insubimgCud);
        ImageBasicOp::roiSubImage((*outimg)[i], &outsubimgCud);

        if(insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width)
            insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
        else
            outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;
        if(insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height)
            insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
        else
            outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;


        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;

        gridsize.x = (insubimgCud.imgMeta.width + DEF_BLOCK_X - 1) / DEF_BLOCK_X;
        gridsize.y = (insubimgCud.imgMeta.height + DEF_BLOCK_Y - 1) / DEF_BLOCK_Y;

        if(i == thresholdNum - 1){
            _calImageFilter<<<gridsize,blocksize>>>(insubimgCud, outsubimgCud, threshold[i], 255);
        }
        else
            _calImageFilter<<<gridsize,blocksize>>>(insubimgCud, outsubimgCud, threshold[i], threshold[i + 1]);
        
    }
}*/

// 成员方法：calImageFilterOpt (多阈值图像过滤)
int ImageFilter::calImageFilterOpt(Image *inimg, Image ***outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if(inimg == NULL)
        return NULL_POINTER;

    // 局部变量，错误码
    int errcode; 

    // 定义 device 中的 threshold 数组。
    int *d_threshold;

    // 在 device 中为 d_threshold 开辟空间
    errcode = cudaMalloc((void **)&d_threshold, thresholdNum * sizeof(int));
    if (errcode != NO_ERROR)
        return errcode;
    // 将 host 中的 threshold 拷贝到 device 中的 d_threshold
    errcode = cudaMemcpy(d_threshold, threshold, thresholdNum * sizeof(int),  
                         cudaMemcpyHostToDevice);
    if (errcode != NO_ERROR) {
        cudaFree(d_threshold);
        return errcode;
    }

    //  为输出图像开辟空间
    *outimg = new Image *[thresholdNum];
    for(int i = 0; i < thresholdNum; i++) {
        errcode = ImageBasicOp::newImage(&((*outimg)[i]));
        if (errcode != NO_ERROR) {
            cudaFree(d_threshold);
            return errcode;
        }

        errcode = ImageBasicOp::makeAtCurrentDevice((*outimg)[i], inimg->width,
                                                     inimg->height);
        if (errcode != NO_ERROR) {
            cudaFree(d_threshold);
            return errcode;
        }
    }

    // 将输入图像拷贝到目前设备中
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR) {
        cudaFree(d_threshold);
        return errcode;
    }

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR) {
        cudaFree(d_threshold);
        return errcode;
    }

    // 提取输出图像的 ROI 子图像。
    // 在 host 上为 outsubimgCud 开辟空间
    ImageCuda *outsubimgCud = new ImageCuda[thresholdNum];

    // 在 device 上为 d_outsubimgCud 开辟空间
    ImageCuda *d_outsubimgCud;
    errcode = cudaMalloc((void **)&d_outsubimgCud, thresholdNum *              
                         sizeof(ImageCuda));
    if (errcode != NO_ERROR) {
        cudaFree(d_threshold);
        return errcode;
    }
    // 对每一幅输出图像提取 ROI 子图像
    for(int i = 0; i < thresholdNum; i++) {
        errcode = ImageBasicOp::roiSubImage((*outimg)[i], &(outsubimgCud[i])); 
        if (errcode != NO_ERROR) {
            cudaFree(d_threshold);
            cudaFree(d_outsubimgCud);
            return errcode;
        }

        // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行
        // 子图像的统一。    
        if(insubimgCud.imgMeta.width > outsubimgCud[i].imgMeta.width)
            insubimgCud.imgMeta.width = outsubimgCud[i].imgMeta.width;
        else
            outsubimgCud[i].imgMeta.width = insubimgCud.imgMeta.width;
        if(insubimgCud.imgMeta.height > outsubimgCud[i].imgMeta.height)
            insubimgCud.imgMeta.height = outsubimgCud[i].imgMeta.height;
        else
            outsubimgCud[i].imgMeta.height = insubimgCud.imgMeta.height;       
    }

    // 将 Host 上的 outsubimgCud 拷贝到 Device 上的 d_outsubimgCud。
    errcode = cudaMemcpy(d_outsubimgCud, outsubimgCud, thresholdNum * 
                         sizeof(ImageCuda), cudaMemcpyHostToDevice);
    if (errcode != NO_ERROR) {
        cudaFree(d_threshold);
        cudaFree(d_outsubimgCud);
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize;
    dim3 gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;

    gridsize.x = (insubimgCud.imgMeta.width + DEF_BLOCK_X - 1) / DEF_BLOCK_X;  
    gridsize.y = (insubimgCud.imgMeta.height + DEF_BLOCK_Y * 4 - 1) / 
                 (DEF_BLOCK_Y * 4);

    // 调用 Kernel 函数，实现多阈值图像过滤
    _calImageFilterOpt<<<gridsize,blocksize>>>(insubimgCud, d_outsubimgCud, 
                                               thresholdNum, d_threshold);

    // 释放已分配的数组内存，避免内存泄露
    cudaFree(d_threshold);
    cudaFree(d_outsubimgCud);

    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}