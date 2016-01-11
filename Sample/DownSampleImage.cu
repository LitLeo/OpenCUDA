// DownSampleImage.cu
// 实现对图像的缩小处理

#include <iostream> 
using namespace std;

#include "DownSampleImage.h"
#include "ErrorCode.h"
#include "stdio.h"
#include "time.h"
#include "stdlib.h"
#include "curand.h"
#include "curand_kernel.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：METHODSELECT_THRES
// 定义了图像缩小倍数 3，根据这个临界值采用不同的 Kernel 函数处理方式。
#define METHODSELECT_THRES 3

// Kernel 函数：_downImgbyDomLessKer（优势法缩小图像）
// 根据给定的缩小倍数 N，将输入图像缩小，将其尺寸从 width * height 变成 
// (width / N) * (height / N)
static __global__ void         // Kernel 函数无返回值
_downImgbyDomLessKer(
        ImageCuda inimg,       // 输入图像
        ImageCuda outimg,      // 输出图像
        int times,             // 缩小倍数
        int *alldevicepointer  // 数组指针
);

// Kernel 函数：_downImgbyDomGreaterKer（优势法缩小图像）
// 根据给定的缩小倍数 N，将输入图像缩小，将其尺寸从 width * height 变成 
// (width / N) * (height / N)
static __global__ void     // Kernel 函数无返回值
_downImgbyDomGreaterKer(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 输出图像
        int times          // 缩小倍数
); 

// Kernel 函数：_genRandomKer（生成随机数）
// 在 Device 端生成一个跟输出图片大小一样的随机数矩阵，用于概率法
// 缩小图像。
static __global__ void 
_genRandomKer(
        int *randnumdev,  // 随机数矩阵
        int times,        // 缩小倍数
        int time,         // 时间参数
        int width         // 随机数矩阵的宽度
);

// Kernel 函数：_downImgbyProKer（概率法缩小图像）
// 根据给定的缩小倍数 N，用概率法将输入图像缩小，将其尺寸从 width * height 变成 
// (width / N) * (height / N)。
static __global__ void     // Kernel 函数无返回值
_downImgbyProKer(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 输出图像
        int *randnumdev,   // 随机数矩阵
        int times          // 缩小倍数
);

// Kernel 函数：_downImgbyDomLessKer（用优势法缩小图像）
static __global__ void _downImgbyDomLessKer(
        ImageCuda inimg, ImageCuda outimg, int times, int *alldevicepointer)
{
    // 计算当前线程的位置。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 对应输入图像的位置
    int inc = times * (c + 1) - 1;
    int inr = times * (r + 1) - 1;
    
    // 计算当前像素的领域大小 
    int pixnum = (2 * times - 1) * (2 * times - 1);
    // 获取当前线程中存放像素值的区域指针。
    int *pixel = alldevicepointer + (r * outimg.imgMeta.width + c) * pixnum;
    // 获取当前线程中存放像素值个数的区域指针。
    int *count = alldevicepointer +  
                 pixnum * outimg.imgMeta.width * outimg.imgMeta.height + 
                 (r * outimg.imgMeta.width + c) * pixnum;
                 
    // 定义变量
    int i, j;
    unsigned char *outptr;
    unsigned char curvalue;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (r >= outimg.imgMeta.height || c >= outimg.imgMeta.width)
        return;
       
    // 得到输出图像当前像素的位置。
    outptr = outimg.imgMeta.imgData + c + r * outimg.pitchBytes;
  
    // 中间数组，用来临时记录下标
    int flag[256];
    
    // 为数组 flag 赋初值
    for(int i = 0; i < 256; i++)
        flag[i] = -1;
    
    // pixel 数组的下标
    int m = 0;   
    
    // 对当前像素的 (2 * times - 1) * (2 * times - 1)领域，计算出
    // 各像素值以及相应的个数。
    for (j = inr - (times - 1); j <= inr + (times - 1); j++) {
        for (i = inc - (times - 1); i <= inc + (times - 1); i++) {

            // 判断当前像素是否越界。
            if (j >= 0 && j < inimg.imgMeta.height && 
                i >= 0 && i < inimg.imgMeta.width) {
                // 得到当前位置的像素值。
                curvalue = *(inimg.imgMeta.imgData + i + j * inimg.pitchBytes);
          
                // 如果当前像素值的 flag 为 -1，即说明该像素值在邻域
                // 中没有出现过，则对该像素值进行数量统计，并把标记 flag 
                // 赋值为当前位置，从而建立一个索引。
                if (flag[curvalue] == -1) {
                    pixel[m] = curvalue;
                    flag[curvalue] = m;
                    count[m]++;
                    m++;
                } else {
                    // 如果当年像素值的 flag 不为 -1，则找到当前像素值
                    // 对应的索引值，并把计数器中该位置的数字加 1。
                    count[flag[curvalue]]++; 
                }       
            }
        }
    }

    // 选出领域内像素值个数最多的三个。
    // 声明局部变量。
    int p, q;
    int maxest;
    int maxindex;
    int tempmax[3], tempindex[3];
   
    // 使用选择排序，找到最大的 3 个像素值。
    for (p = 0; p < 3; p++) {
        // 暂存计数器中的第一个数据，以及对应的索引。
        maxest = count[0];
        maxindex = 0;
        // 对于邻域中所有值，与暂存的值进行比较，从而找到最大值。
        for (q = 1; q < pixnum; q++) {
            if (count[q] > maxest) {
                maxest = count[q];
                maxindex = q;
            }
        } 
        // 记录下找到的最大值以及索引，并把计数器中的
        // 最大值位置清 0。
        tempmax[p] = maxest;
        tempindex[p] = maxindex;
        count[maxindex] = 0;
    }
    
    // 求这 3 个像素峰值的加权平均值，并四舍五入取整。
    int v;
    int sum = tempmax[0] + tempmax[1] + tempmax[2];
    v = (pixel[tempindex[0]] * tempmax[0] + 
         pixel[tempindex[1]] * tempmax[1] +
         pixel[tempindex[2]] * tempmax[2] + (sum >> 1)) / sum;

    // 将用优势法计算出的像素值赋给输出图像
    *outptr = v;
}

// Kernel 函数：_downImgbyDomGreaterKer（用优势法缩小图像）
static __global__ void _downImgbyDomGreaterKer(
        ImageCuda inimg, ImageCuda outimg, int times)
{
    // 计算当前线程的位置。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 对应输入图像的位置
    int inc = times * (c + 1) - 1;
    int inr = times * (r + 1) - 1;

    // 定义变量
    int i, j;
    unsigned char *outptr;
    unsigned char curvalue;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，
    // 一方面节省计算资源，另一方面防止由于段错误导致程序崩溃。
    if (r >= outimg.imgMeta.height || c >= outimg.imgMeta.width)
        return;
       
    // 得到输出图像当前像素的位置。  
    outptr = outimg.imgMeta.imgData + c + r * outimg.pitchBytes;
  
    // 定义数组，下标代表图像灰度值，数组里存相应的个数
    int count[256] = { 0 };

    // 对当前像素的 (2 * times - 1) * (2 * times - 1)领域，计算出
    // 各像素值以及相应的个数。
    for (j = inr - (times - 1); j <= inr + (times - 1); j++) {
        for (i = inc - (times - 1); i <= inc + (times - 1); i++) {
            // 判断当前像素是否越界。
            if (j >= 0 && j < inimg.imgMeta.height && 
                i >= 0 && i < inimg.imgMeta.width) {
                // 得到当前位置的像素值。
                curvalue = *(inimg.imgMeta.imgData + i + j * inimg.pitchBytes);
                // 当前像素值的计数器加 1。
                count[curvalue]++;
            }
        }
    }

    // 选出领域内像素值个数最多的三个。
    // 声明局部变量。
    int p, q;
    int maxest;
    int maxindex;
    int tempmax[3], tempindex[3];
    
    // 使用选择排序，找到最大的 3 个像素值。
    for (p = 0; p < 3; p++) {
        // 暂存计数器中的第一个数据，以及对应的索引。
        maxest = count[0];
        maxindex = 0;       
        // 对于邻域中所有值，与暂存的值进行比较，从而找到最大值。
        for (q = 1; q < 256; q++) {      
            if (count[q] > maxest) {
                maxest = count[q];
                maxindex = q;
            }
        }
        // 记录下找到的最大值以及索引，并把计数器中的最大值位置清 0。
        tempmax[p] = maxest;
        tempindex[p] = maxindex;
        count[maxindex] = 0;
    }
    
    // 求这 3 个像素峰值的加权平均值，并四舍五入取整。
    int v;
    int sum = tempmax[0] + tempmax[1] + tempmax[2];
    v = (tempindex[0] * tempmax[0] + 
         tempindex[1] * tempmax[1] +
         tempindex[2] * tempmax[2] + (sum >> 1)) / sum;
        
    // 将用优势法计算出的像素值赋给输出图像
    *outptr = v;
}

// Host 成员方法：dominanceDownSImg（优势法图像缩小处理）
__host__ int DownSampleImage::dominanceDownSImg(Image *inimg, Image *outimg)
{
    // 检查输入图像，输出图像是否为空
    if (inimg == NULL || outimg == NULL )
        return NULL_POINTER;
    
    // 判断缩小倍数是否合理  
    if (times <= 1)
        return INVALID_DATA;
        
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成
    // 在 Device 内存上为输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
        
    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建
        // 一个和输入图像的 ROI 子图像缩小 times 倍后尺寸相同的图像。
        int outwidth = (inimg->roiX2 - inimg->roiX1) / times;
        int outheight = (inimg->roiY2 - inimg->roiY1) / times;
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
        
    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一
    if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width * times)
        insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width * times;
    else
        outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width / times;

    if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height * times)
        insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height * times;
    else
        outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height / times;

    // 计算每个像素点需要处理的邻域的大小。
    int devsize = (2 * times - 1) * (2 * times - 1);
    // 得到输出图像的大小。
    int size = outsubimgCud.imgMeta.width * outsubimgCud.imgMeta.height;
    
    // 声明局部变量。
    int *alldevicepointer;
    cudaError_t cudaerrcode;
    // 一次性申请全部的 device 端的内存空间。
    cudaerrcode = cudaMalloc(
            (void **)&alldevicepointer, 2 * devsize * size * sizeof (int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;
    
    // 初始化所有 Device 上的内存空间。
    cudaerrcode = cudaMemset(
            alldevicepointer, 0, 2 * devsize * size * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;
    
    if (times < METHODSELECT_THRES) {
        // 调用核函数，根据缩小倍数 times 进行图像缩小处理。
        _downImgbyDomLessKer<<<gridsize, blocksize>>>(
                insubimgCud, outsubimgCud, times, alldevicepointer);
        if (cudaerrcode != cudaSuccess) {
            cudaFree(alldevicepointer);
            return cudaerrcode; 
        }

    } else {
        // 调用核函数，根据缩小倍数 times 进行图像缩小处理。
        _downImgbyDomGreaterKer<<<gridsize, blocksize>>>(
                insubimgCud, outsubimgCud, times);
        if (cudaerrcode != cudaSuccess) {
            cudaFree(alldevicepointer);
            return cudaerrcode; 
        }
    }
    
    // 释放内存空间。
    cudaFree(alldevicepointer);
    
    // 处理完毕，退出。
    return NO_ERROR;
}

// Kernel 函数：_genRandomKer（生成随机数）
static __global__ void _genRandomKer(int *randnumdev, int times, 
                                     int time, int width)
{
    // 使用一个线程生成 4 行随机数。
    // 计算当前线程的位置。
    int index = blockIdx.x * 4;
    // 获取当前的时间参数
    int position;
    
    // curand随机函数初始化
    curandState state;
    curand_init(time, index, 0, &state);
    
    // 得到当前行在随机数矩阵中的偏移
    position = index * width;
    // 一次性生成 4 行随机数。
    for (int k = 0; k < 4; k ++) {
        for (int i = 0; i < width; i++) { 
            // 生成一个随机数。
            *(randnumdev + position + i) = curand(&state) % times;
        }
        // 获取下一行的偏移。
        position += width;
    }    
}

// Kernel 函数：_downImgbyProKer（概率法缩小图像）
static __global__ void _downImgbyProKer(
        ImageCuda inimg, ImageCuda outimg, int *randnumdev, int times)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = (blockIdx.x * blockDim.x + threadIdx.x);
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    
    // 声明局部变量。
    unsigned char *inptr, *outptr;
    int randnum, index;
    int rex, rey, x, y;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (r >= outimg.imgMeta.height || c >= outimg.imgMeta.width)
        return;

    // 得到当前线程在随机数矩阵中的位置 index。
    index = r * outimg.imgMeta.width + c;   
    // 得到当前 index 位置上对应的随机数。
    randnum = *(randnumdev + index);
    
    // 得到在输入图像的偏移量。
    rex = randnum % (2 * times - 1);
    rey = randnum / (2 * times - 1); 
    
    // 对应输入图像的位置
    x = times * c + rex;  //n * (c + 1) - 1 + rex - (n - 1);
    y = times * r + rey;  //n * (r + 1) - 1 + rey - (n - 1); 
    
    // 处理边界点的特殊情况
    if (x >= inimg.imgMeta.width)
        x = x - (times - 1);
    if (y >= inimg.imgMeta.height)
        y = y - (times - 1);
    if (x < 0)
        x = x + (times - 1);
    if (y < 0)
        y = y + (times - 1);
   
    // 第一个像素点
    inptr = inimg.imgMeta.imgData + x + y * inimg.pitchBytes;
    outptr = outimg.imgMeta.imgData + c + r * outimg.pitchBytes;
    
    // 为输出图像当前点赋值。
    *outptr = *inptr;

    // 处理剩下的 3 个像素点。    
    for (int k = 1; k < 4; k++) {
        // 判断是否越界。
        if (++r >= outimg.imgMeta.height)
            return;
        // 得到下一个随机数的位置以及获取随机数。
        index += outimg.imgMeta.width;
        randnum = *(randnumdev + index);
        
        // 得到在输入图像的偏移量
        rex = randnum % (2 * times - 1);
        rey = randnum / (2 * times - 1); 
       
        // 对应输入图像的位置
        x = times * c + rex;
        y = times * r + rey;
        
        // 处理边界点
        if (x >= inimg.imgMeta.width)
            x = x - (times - 1);
        if (y >= inimg.imgMeta.height)
            y = y - (times - 1);
        if (x < 0)
            x = x + (times - 1);
        if (y < 0)
            y = y + (times - 1);
            
        // 为后面 3 个像素点赋值。
        inptr = inimg.imgMeta.imgData + x + y * inimg.pitchBytes;
        outptr = outimg.imgMeta.imgData + c + r * outimg.pitchBytes;   
        *outptr = *inptr;
    }
}

// Host 成员方法：probabilityShrink（概率法图像缩小处理）
__host__ int DownSampleImage::probabilityDownSImg(Image *inimg, Image *outimg)
{
    // 定义变量
    int errcode;  
    cudaError_t cudaerrcode;
    
    // 检查输入图像，输出图像是否为空
    if (inimg == NULL || outimg == NULL )
        return NULL_POINTER;
        
    // 判断缩小倍数是否合理    
    if (times <= 1)
        return INVALID_DATA;
        
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
        
    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像的 ROI 子图像缩小 times 倍后尺寸相同的图像。
        int outwidth = (inimg->roiX2 - inimg->roiX1) / times;
        int outheight = (inimg->roiY2 - inimg->roiY1) / times;
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
    
    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一
    if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width * times)
        insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width * times;
    else
        outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width / times;

    if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height * times)
        insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height * times;
    else
        outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height / times;

    // 计算随机数矩阵的大小，以及随机数的范围。  
    int positionsize, randomsize;   
    positionsize = outsubimgCud.imgMeta.width * outsubimgCud.imgMeta.height;
    randomsize = (2 * times - 1) * (2 * times - 1);

    // 在 Device 端一次性申请所需要的空间
    int *randnumdev;
    
    // 为 Device 端分配内存空间。  
    cudaerrcode = cudaMalloc((void**)&randnumdev, positionsize * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        return cudaerrcode;
    }

    // 在 Host 端获取时间。由于使用标准 C++ 库获得的时间是精确到秒的，这个时间
    // 精度是远远大于两次可能的调用间隔，因此，我们只在程序启动时取当前时间，之
    // 之后对程序的时间直接进行自加，以使得每次的时间都是不同的，这样来保证种子
    // 在各次调用之间的不同，从而获得不同的随机数。  
    static int timehost = (int)time(NULL);
    timehost++;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。  
    dim3 gridsize, blocksize;
    gridsize.x = (outsubimgCud.imgMeta.height + 3) / 4;
    blocksize.x = 1;
    // 随机数矩阵的宽度。
    int width = outsubimgCud.imgMeta.width; 

    // 调用生成随机数的 Kernel 函数。
    _genRandomKer<<<gridsize, blocksize>>>(randnumdev, randomsize, 
                                           timehost, width);  
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(randnumdev);
        return CUDA_ERROR;
    }       
        
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) / 
                 (blocksize.y * 4);
    
    // 调用核函数，根据缩小倍数 times 进行图像缩小处理。
    _downImgbyProKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud,
                                             randnumdev, times);  
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(randnumdev);
        return CUDA_ERROR;
    }
    
    // 释放 Device 内存中的数据。
    cudaFree(randnumdev);

    // 处理完毕，退出。
    return NO_ERROR;
}

