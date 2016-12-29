// SalientRegionDetect.cu
// 实现图像显著性区域检测

#include "SalientRegionDetect.h"
#include "Template.h"
#include "ConnectRegion.h"
#include "Histogram.h"

#include <iostream>
#include <stdio.h>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：MAX_TEMPLATE
// 定义领域模板的最大值。
#ifndef MAX_TEMPLATE
#define MAX_TEMPLATE 32
#endif

// 宏：GRAY_LEVEL
// 定义灰度级范围。
#ifndef GRAY_LEVEL
#define GRAY_LEVEL 256
#endif


// Device 全局常量：_gaussCoeffDev（高斯模板权重）
static __device__ float _gaussCoeffDev[6][170] = {
    // 3 * 3 模板
    {  1.0f / 16.0f,  2.0f / 16.0f,  1.0f / 16.0f,  
       2.0f / 16.0f,  4.0f / 16.0f,  2.0f / 16.0f,
       1.0f / 16.0f,  2.0f / 16.0f,  1.0f / 16.0f },
    // 5 * 5 模板
    {   1.0f / 352.0f,   5.0f / 352.0f,   8.0f / 352.0f,   5.0f / 352.0f,
        1.0f / 352.0f,   5.0f / 352.0f,  21.0f / 352.0f,  34.0f / 352.0f, 
       21.0f / 352.0f,   5.0f / 352.0f,   8.0f / 352.0f,  34.0f / 352.0f,  
       56.0f / 352.0f,  34.0f / 352.0f,   8.0f / 352.0f,   5.0f / 352.0f,   
       21.0f / 352.0f,  34.0f / 352.0f,  21.0f / 352.0f,   5.0f / 352.0f,  
        1.0f / 352.0f,   5.0f / 352.0f,   8.0f / 352.0f,   5.0f / 352.0f, 
        1.0f / 352.0f },
    // 7 * 7 模板
    {   1.0f / 50888.0f,  12.0f / 50888.0f,  55.0f / 50888.0f,  
       90.0f / 50888.0f,  55.0f / 50888.0f,  12.0f / 50888.0f,
        1.0f / 50888.0f,
         12.0f / 50888.0f,  148.0f / 50888.0f,  665.0f / 50888.0f,
       1097.0f / 50888.0f,  665.0f / 50888.0f,  148.0f / 50888.0f,
         12.0f / 50888.0f,
         55.0f / 50888.0f,  665.0f / 50888.0f,  2981.0f / 50888.0f,
       4915.0f / 50888.0f,  2981.0f / 50888.0f, 665.0f / 50888.0f,
         55.0f / 50888.0f,
         90.0f / 50888.0f,  1097.0f / 50888.0f,  4915.0f / 50888.0f,
       8104.0f / 50888.0f,  4915.0f / 50888.0f,  1097.0f / 50888.0f,
         90.0f / 50888.0f,
         55.0f / 50888.0f,  665.0f / 50888.0f,  2981.0f / 50888.0f,
       4915.0f / 50888.0f,  2981.0f / 50888.0f, 665.0f / 50888.0f,
         55.0f / 50888.0f,
         12.0f / 50888.0f,  148.0f / 50888.0f,  665.0f / 50888.0f,
       1097.0f / 50888.0f,  665.0f / 50888.0f,  148.0f / 50888.0f,
         12.0f / 50888.0f,
        1.0f / 50888.0f,  12.0f / 50888.0f,  55.0f / 50888.0f,  
       90.0f / 50888.0f,  55.0f / 50888.0f,  12.0f / 50888.0f,
        1.0f / 50888.0f }
};

// Kernel 函数：_saliencyMapByDiffValueKer（差值法计算显著性值）
// 计算图像中每个像素值的显著性值。以每个像素为中心，计算其与邻域 radius 内所有
// 像素的灰度差值；对所有差值按照降序进行排序，去掉排序中先头的若干值和末尾的若
// 干值（通过设置 highPercent 和 lowPercent），只保留中间部分的排序结果，计算平
// 均值作为显著性值，形成一个初期的 saliency map。然后改变 radius 值，重复上述
// 计算，得到若干个初期 saliency map，将所有的 saliency map 进行累加平均，就得
// 到最终的平均 saliency map，输出到 outimg 中。
static __global__ void       // Kernel 函数无返回值。
_saliencyMapByDiffValueKer(
        ImageCuda inimg,     // 输入图像
        ImageCuda outimg,    // 输出图像
        int *radius,         // 模板半径
        int iteration,       // 迭代次数
        float hightpercent,  // 数组的高位段
        float lowpercent     // 数组的低位段
);

// Kernel 函数：_saliencyMapByDiffValueKer（差值法计算显著性值）
// 计算图像中每个像素值的显著性值。以每个像素为中心，计算其与邻域 radius 内所有
// 像素的灰度差值；不进行数组的筛选，计算所有差值的平均值作为显著性值，形成一个
// 初期的 saliency map。然后改变 radius 大小，重复上述计算，就会得到若干个初期 
// saliency map，将所有的 saliency map 进行累加平均，就得到最终的平均显著性图
// saliency map，输出到 outimg 中。
static __global__ void     // Kernel 函数无返回值。
_saliencyMapByDiffValueKer(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 输出图像
        int *radius,       // 模板半径
        int iteration      // 迭代次数
);

// Kernel 函数：_saliencyMapBySmoothKer（高斯平滑法计算显著值）
// 计算图像中每个像素值的显著性值。利用高斯平滑滤波对原始图像进行处理，设置 
// smoothWidth 表示平滑尺度大小，将平滑结果与邻域算数几何平均的图像进行整体差分
// ，就得到一个初期的 saliency map。改变 smoothWidth 的值，重复上述计算，得到若
// 干个初期 saliency map，将所有的 saliency map 进行累加平均，就得到最终的平均 
// saliency map，输出到 outimg 中。
static __global__ void     // Kernel 函数无返回值。
_saliencyMapBySmoothKer(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 输出图像
        int *smoothwidth,  // 平滑模板
        int iteration      // 迭代次数
);

// Kernel 函数：_saliencyMapAverageKer（计算平均显著性值）
// 将差值法计算显著性值和高斯平滑法计算显著值的结果进行加权平均，weightSM1 是
// 差值法计算显著性值的权重，weightSM2 是高斯平滑法计算显著值的权重。加权结果
// 保存到 sm2img 中。
static __global__ void     // Kernel 函数无返回值。
_saliencyMapAverageKer(
        ImageCuda sm1img,  // 输入图像
        ImageCuda sm2img,  // 输出图像
        float weightsm1,   // 差值法计算显著性值的权重
        float weightsm2    // 高斯平滑法计算显著值的权重
);

// Kernel 函数：_regionSaliencyKer（计算区域累计显著性值）
// 计算每个区域所有像素值的累计显著性值。默认区域的最大个数为 256。
static __global__ void             // Kernel 函数无返回值。
_regionSaliencyKer(
        ImageCuda smimg,           // 输入图像
        ImageCuda connimg,         // 输出图像
        unsigned int *regionaverg  // 区域显著性累计数组
);

// Kernel 函数：_regionAverageKer（计算区域的平均显著性值）
// 通过区域累计显著性数组除以区域面积，得到区域平均显著性值。
static __global__ void              // Kernel 函数无返回值。
_regionAverageKer(
        unsigned int *regionaverg,  // 区域显著性累计数组
        unsigned int *regionarea,   // 区域面积数组
        unsigned int *flag,         // 判断是否满足阈值标识位
        int saliencythred           // 区域显著性阈值大小
);

// Kernel 函数: _regionShowKer（显示显著性区域）
// 根据标识数组，将显著性区域的灰度值设为 255，背景设为 0。
static __global__ void 
_regionShowKer(
        ImageCuda inimg,         // 输入图像
        unsigned int *flaglabel  // 标识数组
);

// Kernel 函数：_saliencyMapByDiffValueKer（差值法计算显著性值）
static __global__ void _saliencyMapByDiffValueKer( 
        ImageCuda inimg, ImageCuda outimg, int *radius, int iteration)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，
    // c 表示 column， r 表示 row）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (dstc >= inimg.imgMeta.width || dstr >= inimg.imgMeta.height)
        return;

    // 用来记录当前处理像素的灰度值。
    unsigned char *curinptr;
    curinptr = inimg.imgMeta.imgData + dstc + dstr * inimg.pitchBytes;

    // 存放邻域像素的灰度值。
    unsigned char neighpixel;

    // 邻域内所有灰度值的差值。
    int diffvalue;
    // 差值累积和。
    float tempSum = 0;
    // 邻域像素个数。
    int neighbors;

    for (int num = 0; num < iteration; num++) {
        int r = radius[num];
        // 计算邻域差值之和。
        float diffsum = 0;
        // 计算邻域像素个数。
        neighbors = (2 * r - 1) * (2 * r - 1);

        // 对当前像素的 (2 * r - 1) * (2 * r - 1)领域，计算出
        // 各像素值以及相应的个数。
        for (int j = dstr - (r - 1); j <= dstr + (r - 1); j++) {
            for (int i = dstc - (r - 1); i <= dstc + (r - 1); i++) {
                // 判断当前像素是否越界。
                if (j >= 0 && j < inimg.imgMeta.height && 
                    i >= 0 && i < inimg.imgMeta.width) {
                    // 循环计算每个邻域的灰度值差值。
                    neighpixel = *(inimg.imgMeta.imgData + i + 
                                   j * inimg.pitchBytes);         
                    diffvalue = *curinptr - neighpixel;
                    if (diffvalue < 0)
                        diffvalue = -diffvalue;
                    // 累加邻域内的差值。
                    diffsum += diffvalue;
                }
            }
        }    
        // 计算一次迭代的显著性值。
        diffsum = diffsum / neighbors;

        // 将多次迭代的结果累加。
        tempSum += diffsum;
    }

    // 多次迭代结果计算平均显著性值输出到图像中。
    unsigned char *curoutptr;
    curoutptr = outimg.imgMeta.imgData + dstc + dstr * outimg.pitchBytes;
    *curoutptr = (int)(tempSum + 0.5f) / iteration;
}

// Kernel 函数：_saliencyMapByDiffValueKer（差值法计算显著性值）
static __global__ void _saliencyMapByDiffValueKer(
        ImageCuda inimg, ImageCuda outimg, int *radius, int iteration,
        float hightpercent, float lowpercent)
{    

    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，
    // c 表示 column， r 表示 row）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (dstc >= inimg.imgMeta.width || dstr >= inimg.imgMeta.height)
        return;

    // 用来记录当前处理像素的灰度值。
    unsigned char *curinptr;
    curinptr = inimg.imgMeta.imgData + dstc + dstr * inimg.pitchBytes;
        
    // 存放邻域像素的灰度值。
    unsigned char neighpixel;

    // 邻域内所有灰度值的差值。
    int diffvalue;
    // 差值累积和。
    float tempSum = 0;
    // 邻域像素个数。
    int neighbors;
   
    for (int num = 0; num < iteration; num++) {
        int r = radius[num];
        // 定义数组，下标代表图像灰度值，数组里存相应的个数
        int count[256] = { 0 };

        // 计算邻域像素个数。
        neighbors = (2 * r - 1) * (2 * r - 1);

        // 对当前像素的 (2 * r - 1) * (2 * r - 1)领域，计算出
        // 各像素值以及相应的个数。
        for (int j = dstr - (r - 1); j <= dstr + (r - 1); j++) {
            for (int i = dstc - (r - 1); i <= dstc + (r - 1); i++) {
                // 判断当前像素是否越界。
                if (j >= 0 && j < inimg.imgMeta.height && 
                    i >= 0 && i < inimg.imgMeta.width) {
                    // 循环计算每个邻域的灰度值差值。
                    neighpixel = *(inimg.imgMeta.imgData + i + 
                                   j * inimg.pitchBytes);         
                    diffvalue = *curinptr - neighpixel;
                    if (diffvalue < 0)
                        diffvalue = -diffvalue;
                    // 当前灰度值差值的计数器加 1。
                    count[diffvalue]++;
                }
            }
        }    

        // 去掉排序结果中先头的若干值和末尾的若干值。
        int hp = (int)(hightpercent * neighbors + 0.5f);
        int lp = (int)(lowpercent * neighbors + 0.5f);    
        // 定义一些临时变量。
        int lpcount = 0, hpcount = 0;
        // lp 和 hp 完成标识位。
        bool lpover = false, hpover = false;
        // lp 和 hp 结果索引。
        int lpindex = 0, hpindex = 0;
        int lpresidual = 0, hpresidual = 0;

        // 循环查找数组，找到 lp 和 hp 位置。
        for (int lpi = 0; lpi < 256; lpi++) {
            // 筛选结束。
            if (lpover == true && hpover == true)
                break;

            // 处理低段数据 lp。
            lpcount += count[lpi];
            if (lpover == false && lpcount >= lp) {
                lpindex = lpi + 1;
                lpover = true;
                lpresidual = lpi * (lpcount - lp);
            }

            // 处理高段数据 hp。
            int hpi = 255 - lpi;
            hpcount += count[hpi];
            if (hpover == false && hpcount >= hp) {
                hpindex = hpi - 1;
                hpover = true;
                hpresidual = hpi * (hpcount - hp);
            }
        }   
        // 如果 lp 大于 hp，则错误退出。
        if (lpindex > hpindex)
            return;
 
        // 计算保留部分的均值。
        float sum = lpresidual + hpresidual;
        for (int j = lpindex; j <= hpindex; j++) {
            sum += count[j] * j;
        }

        // 计算一次迭代的显著性值。
        sum = sum / (neighbors - lp - hp);
    
        // 将多次迭代的结果累加。
        tempSum += sum;
    }

    // 多次迭代结果计算平均显著性值输出到图像中。
    unsigned char *curoutptr;
    curoutptr = outimg.imgMeta.imgData + dstc + dstr * outimg.pitchBytes;
    *curoutptr = (int)(tempSum + 0.5f) / iteration;
}

// Host 成员方法：saliencyMapByDiffValue（差值法计算显著值）
__host__ int SalientRegionDetect::saliencyMapByDiffValue(Image *inimg, 
                                                         Image *outimg)
{
    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 检查输入参数是否为空。
    if (this->radius == NULL || iterationSM1 == 0)
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

    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一。
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
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / DEF_BLOCK_X;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / DEF_BLOCK_Y;
  
    // 在 Device 端申请模板半径数组。
    int *devradius;
    cudaError_t cudaerrcode;
    cudaerrcode = cudaMalloc((void** )&devradius, iterationSM1 * sizeof (int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 将 Host 上的 radius 拷贝到 Device 上的 devradius 中。
    cudaerrcode = cudaMemcpy(devradius, radius, iterationSM1 * sizeof (int),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode; 
    
    if (this->isSelect == true) {    
        // 调用核函数，筛选差值数组，计算每个像素的显著性值。
        _saliencyMapByDiffValueKer<<<gridsize, blocksize>>>( 
                insubimgCud, outsubimgCud, devradius, 
                iterationSM1, highPercent, lowPercent);
    } else {
        // 调用核函数，不筛选差值数组，计算每个像素的显著性值。
        _saliencyMapByDiffValueKer<<<gridsize, blocksize>>>(
                insubimgCud, outsubimgCud, devradius, iterationSM1);
    }

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(devradius);
        return CUDA_ERROR;
    }

    // 释放 Device 端半径空间。
    cudaFree(devradius);

    // 处理完毕，退出。
    return NO_ERROR;
}

// Kernel 函数：_saliencyMapBySmoothKer（高斯平滑法计算显著值）
static __global__ void _saliencyMapBySmoothKer(
        ImageCuda inimg, ImageCuda outimg, int *smoothwidth, int iteration)
{
    // 计算想成对应的输出点的位置，其中 dstc 和 dstr 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (dstc >= inimg.imgMeta.width || dstr >= inimg.imgMeta.height)
        return;

    // 用来保存临时像素点的坐标的 x 和 y 分量。
    int dx, dy; 

    // 存放邻域像素的灰度值。
    unsigned char neighpixel;
    
    float gaussitera = 0.0f, meanitera = 0.0f;
    for (int num = 0; num < iteration; num++) {
        int w = smoothwidth[num];
        int neighbors = w * w;
        // 获取高斯模板权重。
        float *gaussweight = _gaussCoeffDev[num];
        // 算术平均权重。
        float meanweight = 1.0f / neighbors;
        // 统计加权和。
        float gausssum = 0.0f, meansum = 0.0f;
        // 循环处理邻域内每个像素点。
        for (int i = 0; i < neighbors; i++) {
            // 分别计算每一个点的横坐标和纵坐标
            dx = dstc + i % w - 1;
            dy = dstr + i / w - 1;

            // 先判断当前像素是否越界，如果越界，则跳过，扫描下一个点。
            if (dx >= 0 && dx < inimg.imgMeta.width &&
                dy >= 0 && dy < inimg.imgMeta.height) {
                // 根据 dx 和 dy 获取邻域像素的灰度值。
                neighpixel = *(inimg.imgMeta.imgData + dx + 
                             dy * inimg.pitchBytes);
                // 累积一次迭代中高斯平滑的加权和。
                gausssum +=  neighpixel * gaussweight[i];
                // 累积一次迭代中算术平均的加权和。
                meansum += neighpixel * meanweight;
            }
        }
        
        // 累积多次迭代的结果。
        gaussitera += gausssum;
        meanitera += meansum;
    }
    
    // 多次迭代结果计算平均显著性值输出到图像中。
    unsigned char *curoutptr;
    curoutptr = outimg.imgMeta.imgData + dstc + dstr * outimg.pitchBytes;
    *curoutptr = (unsigned char)((gaussitera - meanitera) / iteration + 0.5f);
    
}

// Host 成员方法：saliencyMapBySmooth（高斯平滑法计算显著值）
__host__ int SalientRegionDetect::saliencyMapBySmooth(Image *inimg, 
                                                      Image *outimg)
{
    // 检查输入图像，输出图像是否为空
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 检查输入参数 smoothWidth 是否为空。
    if (this->smoothWidth == NULL || iterationSM2 == 0)
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

    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一。
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
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / DEF_BLOCK_X;    
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / DEF_BLOCK_Y; 

    // 在 Device 端分配平滑模板数组。
    int *devwidth;
    cudaError_t cudaerrcode;
    cudaerrcode = cudaMalloc((void** )&devwidth, iterationSM2 * sizeof (int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 将 Host 上的 smoothWidth 拷贝到 Device 上的 devwidth 中。
    cudaerrcode = cudaMemcpy(devwidth, smoothWidth, iterationSM2 * sizeof (int),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;        

    // 调用核函数，计算高斯平滑平均值。
    _saliencyMapBySmoothKer<<<gridsize, blocksize>>>(
            insubimgCud, outsubimgCud, devwidth, iterationSM2);

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(devwidth);
        return CUDA_ERROR;
    }

    // 释放 Device 端空间。
    cudaFree(devwidth);

    // 处理完毕，退出。
    return NO_ERROR;
}
       
// Kernel 函数：_saliencyMapAverageKer（计算平均显著性值）
static __global__ void _saliencyMapAverageKer(
        ImageCuda inimg, ImageCuda outimg, float w1, float w2)
{
    // 计算想成对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻
    // 4 行上，因此，对于 r 需要进行乘 4 计算。
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
    // 读取第一个输出坐标点对应的像素值。
    unsigned char outtemp;
    outtemp = outimg.imgMeta.imgData[outidx];

    // 一个线程处理四个像素点.
    // 将差值法计算显著性值和高斯平滑法计算显著值的结果进行加权平均，weightSM1
    // 是差值法计算显著性值的权重，weightSM2 是高斯平滑法计算显著值的权重。
    // 线程中处理的第一个点。
    outimg.imgMeta.imgData[outidx] = 
            (unsigned char)(intemp * w1 + outtemp * w2 + 0.5f);

    // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += outimg.pitchBytes;
        intemp = inimg.imgMeta.imgData[inidx];
        outtemp = outimg.imgMeta.imgData[outidx];
        
        // 将显著性值输出到输出图像中。
        outimg.imgMeta.imgData[outidx] = 
                (unsigned char)(intemp * w1 + outtemp * w2 + 0.5f);
    }
}

// Kernel 函数：_regionSaliencyKer（计算区域累计显著性值）
static __global__ void _regionSaliencyKer(
        ImageCuda smimg, ImageCuda connimg, unsigned int *regionsacy)
{
    // 计算想成对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻
    // 4 行上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= smimg.imgMeta.width || r >= smimg.imgMeta.height)
        return;

    // 计算第一个输入坐标点对应的图像数据数组下标。
    int smidx = r * smimg.pitchBytes + c;    
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int connidx = r * connimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char smtemp;
    smtemp = smimg.imgMeta.imgData[smidx];
    unsigned char conntemp;
    conntemp = connimg.imgMeta.imgData[connidx];

    // 一个线程处理四个像素点。
    // 计算区域的累计显著性值。
    // 线程中处理的第一个点。
    regionsacy[conntemp] += smtemp ;

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++r >= connimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        smidx += smimg.pitchBytes;
        connidx += connimg.pitchBytes;
        smtemp = smimg.imgMeta.imgData[smidx];
        conntemp = connimg.imgMeta.imgData[connidx];
        
        // 如果输入图像的该像素值等于 label, 则将其拷贝到输出图像中；
        // 否则将输出图像中该位置清 0。
        regionsacy[conntemp] += smtemp ;
    }
}

// Kernel 函数：_regionAverageKer（计算区域的平均显著性值）
static __global__ void _regionAverageKer(
        unsigned int *regionaverg, unsigned int *regionarea, 
        unsigned int *flag, int saliencythred)
{
    // 读取线程号。
    int tid = threadIdx.x;

    // 通过区域累计显著性数组除以区域面积，得到区域平均显著性值。
    if (regionarea[tid] > 0)
        regionaverg[tid] = ((float)regionaverg[tid]  + 0.5f) / regionarea[tid];

    if (regionaverg[tid] > saliencythred)
        flag[tid] = 1;
    else 
        flag[tid] = 0;
}

// Kernel 函数: _regionShowKer（显示显著性区域）
static __global__ void _regionShowKer(ImageCuda inimg, unsigned int *flaglabel)
{
    // 计算想成对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻
    // 4 行上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;    

    // 一个线程处理四个像素点.
    // 如果输入图像的该像素值对应的 flag 等于 0, 则将像素值设为 0；
    // 否则设为 255。
    // 线程中处理的第一个点。
    if (flaglabel[inimg.imgMeta.imgData[inidx]] == 0)
        inimg.imgMeta.imgData[inidx] = 0;
    else
        inimg.imgMeta.imgData[inidx] = 255;

    // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++r >= inimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;

        // 如果输入图像的该像素值对应的 flag 等于 0, 则将像素值设为 0；
        // 否则设为 255。
        if (flaglabel[inimg.imgMeta.imgData[inidx]] == 0)
            inimg.imgMeta.imgData[inidx] = 0;
        else
            inimg.imgMeta.imgData[inidx] = 255;
    }
}

// Host 成员方法：saliencyRegionDetect（显著性区域检测）
__host__ int SalientRegionDetect::saliencyRegionDetect(Image *inimg, 
                                                       Image *outimg)
{
    // 检查输入图像，输出图像是否为空
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 检查输入参数 smoothWidth 是否为空。
    if (this->smoothWidth == NULL || this->radius == NULL)
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

    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一。
    if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width)
        insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
    else
        outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;

    if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height)
        insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
    else
        outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;
    
    // 申请 SM1 和 SM2 中间图像。
    Image *sm1, *sm2;
    ImageBasicOp::newImage(&sm1);
    ImageBasicOp::newImage(&sm2);
    ImageBasicOp::makeAtCurrentDevice(sm1, inimg->width, inimg->height);
    ImageBasicOp::makeAtCurrentDevice(sm2, inimg->width, inimg->height);
                
    // 差值法计算显著值。
    errcode = saliencyMapByDiffValue(inimg, sm1);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 高斯平滑法计算显著值。
    errcode = saliencyMapBySmooth(inimg, sm2);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取 sm1 图像的 ROI 子图像。
    ImageCuda sm1subimgCud;
    errcode = ImageBasicOp::roiSubImage(sm1, &sm1subimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取 sm2 图像的 ROI 子图像。
    ImageCuda sm2subimgCud;
    errcode = ImageBasicOp::roiSubImage(sm2, &sm2subimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一。
    if (sm1subimgCud.imgMeta.width > sm2subimgCud.imgMeta.width)
        sm1subimgCud.imgMeta.width = sm2subimgCud.imgMeta.width;
    else
        sm2subimgCud.imgMeta.width = sm1subimgCud.imgMeta.width;

    if (sm1subimgCud.imgMeta.height > sm2subimgCud.imgMeta.height)
        sm1subimgCud.imgMeta.height = sm2subimgCud.imgMeta.height;
    else
        sm2subimgCud.imgMeta.height = sm1subimgCud.imgMeta.height;   
             
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (sm2subimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (sm2subimgCud.imgMeta.height + blocksize.y * 4 - 1) / 
                 (blocksize.y * 4); 

    // 合并两种计算显著性值的算法，加权求和，结果保存在 sm2 中。
    _saliencyMapAverageKer<<<gridsize, blocksize>>>(sm1subimgCud, sm2subimgCud,
                                                    weightSM1, weightSM2);

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess) {
        ImageBasicOp::deleteImage(sm1);
        ImageBasicOp::deleteImage(sm2);
        return CUDA_ERROR;
    }

    // 利用连通区域分割 saliency map。
    ConnectRegion cr;
    cr.setThreshold(this->threshold);
    cr.setMinArea(this->minRegion);
    cr.setMaxArea(this->maxRegion);
    cr.connectRegion(sm2, outimg);  
              
    // 计算每个区域的平均显著性值。
    // 在 Device 上分配临时空间。一次申请所有空间，然后通过偏移索引各个数组。
    cudaError_t cudaerrcode;
    unsigned int *alldevicedata;
    unsigned int *devhistogram, *devregionAverg, *devflag;
    cudaerrcode = cudaMalloc((void** )&alldevicedata,
                             3 * GRAY_LEVEL * sizeof (unsigned int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 初始化 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldevicedata, 0,
                             3 * GRAY_LEVEL * sizeof (unsigned int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 通过偏移读取 devhistogram 内存空间。
    devhistogram = alldevicedata;

    // 通过直方图计算区域面积，保存到 devhistogram 中。
    Histogram hist;
    errcode = hist.histogram(outimg, devhistogram, 0);
    if (errcode != NO_ERROR)
        return errcode;   
     
    // 计算每个区域的累积显著性值。
    devregionAverg =  alldevicedata + GRAY_LEVEL;
    _regionSaliencyKer<<<gridsize, blocksize>>>(sm2subimgCud, outsubimgCud,
                                                devregionAverg);

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess) {
        ImageBasicOp::deleteImage(sm1);
        ImageBasicOp::deleteImage(sm2);
        cudaFree(alldevicedata);
        return CUDA_ERROR;
    }

    devflag = devregionAverg + GRAY_LEVEL;
    // 计算每个区域的平均显著性值。
    _regionAverageKer<<<1, GRAY_LEVEL>>>(devregionAverg, devhistogram, 
                                         devflag, saliencyThred);

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess) {
        ImageBasicOp::deleteImage(sm1);
        ImageBasicOp::deleteImage(sm2);
        cudaFree(alldevicedata);
        return CUDA_ERROR;
    }

    // 将显著性区域设为白色，背景设为黑色。
    _regionShowKer<<<gridsize, blocksize>>>(outsubimgCud, devflag);

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess) {
        ImageBasicOp::deleteImage(sm1);
        ImageBasicOp::deleteImage(sm2);
        cudaFree(alldevicedata);
        return CUDA_ERROR;
    }
           
    // 释放中间图像。
    ImageBasicOp::deleteImage(sm1);
    ImageBasicOp::deleteImage(sm2);
    
    // 释放显存上的临时空间
    cudaFree(alldevicedata);
    
    return NO_ERROR;
}
