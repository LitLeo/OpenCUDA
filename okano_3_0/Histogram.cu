// Hitogram.cu
// 实现计算图像直方图算法

#include "Histogram.h"

#include <iostream>
using namespace std;

#include "ErrorCode.h"

// 宏：HISTOGRAM_PACK_LEVEL
// 定义了一个线程中计算的像素点个数，若该值为4，则在一个线程中计算 2 ^ 4 = 16
// 个像素点。
#define HISTOGRAM_PACK_LEVEL 4

#define HISTOGRAM_PACK_NUM   (1 << HISTOGRAM_PACK_LEVEL)
#define HISTOGRAM_PACK_MASK  (HISTOGRAM_PACK_NUM - 1)

#if (HISTOGRAM_PACK_LEVEL < 1 || HISTOGRAM_PACK_LEVEL > 5)
#  error Unsupport HISTOGRAM_PACK_LEVEL Value!!!
#endif

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32 
#define DEF_BLOCK_Y  8   

// Kernel 函数: _histogramKer（计算图像的直方图）
// 根据输入图像每个像素点的灰度值，累加到直方图数组中的相应的位置，从而得到
// 输入图像的直方图。
static __global__ void         // Kernel 函数无返回值。
_histogramKer(
        ImageCuda inimg,       // 输入图像。
        unsigned int *devhist  //图像直方图。
);

// Kernel 函数: _histogramKer（计算图像的直方图）
static __global__ void _histogramKer(ImageCuda inimg, unsigned int *devhist)
{
    // 申请大小为灰度图像灰度级 256 的共享内存，其中下标代表图像的灰度值，数
    // 组用来累加等于该灰度值的像素点个数。
    __shared__ unsigned int temp[256];

    // 计算想成对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，默认令一个线程处理 16 个输出像素，这四个像素位于统一列
    // 的相邻 16 行上，因此，对于 r 需要进行右移计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) << HISTOGRAM_PACK_LEVEL;
    
    // 计算该线程在块内的相对位置。
    int inindex = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 临时变量，curgray 用于存储当前点的像素值，inptrgray 存储下一个点的像素值。
    // cursum 用于存储局部累加和。
    unsigned int curgray = 0, inptrgray;
    unsigned int curnum = 0;
    
    // 若线程在块内的相对位置小于 256，即灰度级大小，则用来给共享内存赋初值 0。
    if (inindex < 256)
        temp[inindex] = 0;
    // 进行块内同步，保证执行到此处，共享内存的数组中所有元素的值都为 0。
    __syncthreads();
    
    do {
        // 线程中处理第一个点。
        // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资
        // 源，一方面防止由于段错误导致的程序崩溃。
        if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
            break;
			
        // 计算第一个输入坐标点对应的图像数据数组下标。
        int inidx = r * inimg.pitchBytes + c;    

        // 读取第一个输入坐标点对应的像素值。
        curgray = inimg.imgMeta.imgData[inidx];
        curnum = 1;

        // 处理第二个点。
        // 此后的像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++r >= inimg.imgMeta.height)
            break;

        // 得到第二个点的像素值。			
        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计算。
        inidx += inimg.pitchBytes;
        inptrgray = inimg.imgMeta.imgData[inidx];
		
        // 若当前第二个点的像素值不等于前一个，把当前临时变量 cursum 中的统计结
        // 果增加到共享内存中的相应区域；若该值等于前一个点的像素值，则临时变量
        // cursum 加 1，继续检查下一个像素点。
        if (curgray != inptrgray) {
            // 使用原子操作把临时变量 curnum 的结果加到共享内存中，可以防止多个
            // 线程同时更改数据而发生的写错误。
            atomicAdd(&temp[curgray], curnum);
            curgray = inptrgray;
            //curnum = 1;
        } else {
            curnum++;
        }
              
        // 宏：HISTOGRAM_KERNEL_MAIN_PHASE
        // 定义计算下一个像素点的程序片段。使用这个宏可以实现获取下一个点的像素
        // 值，并累加到共享内存，并且简化编码量。
#define HISTOGRAM_KERNEL_MAIN_PHASE                            \
        if (++r >= inimg.imgMeta.height)                       \
            break;                                             \
        inidx += inimg.pitchBytes;                             \
        inptrgray = inimg.imgMeta.imgData[inidx];              \
        if (curgray != inptrgray) {                            \
            atomicAdd(&temp[curgray], curnum);                 \
            curgray = inptrgray;                               \
            curnum = 1;                                        \
        } else {                                               \
            curnum++;                                          \
        }

#define HISTOGRAM_KERNEL_MAIN_PHASEx2                           \
        HISTOGRAM_KERNEL_MAIN_PHASE                             \
        HISTOGRAM_KERNEL_MAIN_PHASE

#define HISTOGRAM_KERNEL_MAIN_PHASEx4                           \
        HISTOGRAM_KERNEL_MAIN_PHASEx2                           \
        HISTOGRAM_KERNEL_MAIN_PHASEx2

#define HISTOGRAM_KERNEL_MAIN_PHASEx8                           \
        HISTOGRAM_KERNEL_MAIN_PHASEx4                           \
        HISTOGRAM_KERNEL_MAIN_PHASEx4

#define HISTOGRAM_KERNEL_MAIN_PHASEx16                          \
        HISTOGRAM_KERNEL_MAIN_PHASEx8                           \
        HISTOGRAM_KERNEL_MAIN_PHASEx8

// 对于不同的 HISTOGRAM_PACK_LEVEL ，定义不同的执行次数，从而使一个线程内部
// 实现对多个点的像素值的统计。
#if (HISTOGRAM_PACK_LEVEL >= 2)
    HISTOGRAM_KERNEL_MAIN_PHASEx2
#  if (HISTOGRAM_PACK_LEVEL >= 3)
      HISTOGRAM_KERNEL_MAIN_PHASEx4
#    if (HISTOGRAM_PACK_LEVEL >= 4)
        HISTOGRAM_KERNEL_MAIN_PHASEx8
#      if (HISTOGRAM_PACK_LEVEL >= 5)
          HISTOGRAM_KERNEL_MAIN_PHASEx16
#      endif
#    endif
#  endif
#endif

// 取消前面的宏定义。
#undef HISTOGRAM_KERNEL_MAIN_PHASEx16
#undef HISTOGRAM_KERNEL_MAIN_PHASEx8
#undef HISTOGRAM_KERNEL_MAIN_PHASEx4
#undef HISTOGRAM_KERNEL_MAIN_PHASEx2
#undef HISTOGRAM_KERNEL_MAIN_PHASE

    } while (0);

    // 使用原子操作把临时变量 curnum 的结果加到共享内存中，可以防止多个
    // 线程同时更改数据而发生的写错误。
    if (curnum != 0)
        atomicAdd(&temp[curgray], curnum);
    
    // 块内同步。此处保证图像中所有点的像素值都被统计过。
    __syncthreads();
    
    // 用每一个块内前 256 个线程，将共享内存 temp 中的结果保存到输出数组中。
    if (inindex < 256)
        atomicAdd(&devhist[inindex], temp[inindex]);
}

// Host 成员方法：histogram（计算图像直方图）
__host__ int Histogram::histogram(Image *inimg, 
                                  unsigned int *histogram, bool onhostarray)
{
    // 检查图像是否为 NULL。
    if (inimg == NULL || histogram == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    int height = (insubimgCud.imgMeta.height + 
                 HISTOGRAM_PACK_MASK) / HISTOGRAM_PACK_NUM;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y - 1) / blocksize.y;

    // 判断当前 histogram 数组是否存储在 Host 端。若是，则需要在 Device 端为直
    // 方图申请一段空间；若该数组是在 Device端，则直接调用核函数。
    if (onhostarray){	
        // 在 Device 上分配存储临时直方图的空间。
        cudaError_t cudaerrcode;
        unsigned int *devhisto;
        cudaerrcode = cudaMalloc((void**)&devhisto,
                                 256 * sizeof (unsigned int));
        if (cudaerrcode != cudaSuccess) {
            return cudaerrcode;
        }

        // 初始化 Device 上的内存空间。
        cudaerrcode = cudaMemset(devhisto, 0, 256 * sizeof (unsigned int));
        if (cudaerrcode != cudaSuccess) {
            cudaFree(devhisto);
            return cudaerrcode;
        }

        // 调用核函数，计算输入图像的直方图。
        _histogramKer<<<gridsize, blocksize>>>(insubimgCud, devhisto);
	if (cudaGetLastError() != cudaSuccess) {
            cudaFree(devhisto);
            return CUDA_ERROR;
        }

        // 将直方图的结果拷回 Host 端内存中。
        cudaerrcode = cudaMemcpy(
                histogram, devhisto, 256 * sizeof (unsigned int), 
                cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            cudaFree(devhisto);
            return cudaerrcode;
        }

        // 释放 Device 端的直方图存储空间。	
        cudaFree(devhisto);

        // 如果 histogram 在 Device 端，直接调用核函数。
        } else {
	    _histogramKer<<<gridsize, blocksize>>>(insubimgCud, histogram);
            if (cudaGetLastError() != cudaSuccess) {
                return CUDA_ERROR;
            }
        }

    return NO_ERROR;
}

