// Binarization.cu
// 实现图像的多阈值二值化图像生成操作

#include "Binarization.h"
#include "Binarize.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：BINARIZE_PACK_LEVEL
// 定义了一个线程中计算的像素点个数，若该值为4，则在一个线程中计算2 ^ 4 = 16
// 个像素点
#define BINARIZE_PACK_LEVEL 7

#define BINARIZE_PACK_NUM   (1 << BINARIZE_PACK_LEVEL)
#define BINARIZE_PACK_MASK  (BINARIZE_PACK_LEVEL - 1)

#if (BINARIZE_PACK_LEVEL < 1 || BINARIZE_PACK_LEVEL > 8)
#  error Unsupport BINARIZE_PACK_LEVEL Value!!!
#endif

// 宏： DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_computeAreaKer
// 利用共享内存和原子操作计算输入图像分别以1-255为阈值进行二值化之后的面积值，// 并将面积值保存在数组 result 中。
static __global__ void
_computeAreaKer(
        ImageCuda inimg,  // 输入图像
        int *result       // 面积值数组
);
// Kernel 函数：_computeAreaKer（计算图像的255种面积值）
static __global__ void _computeAreaKer(
        ImageCuda inimg, int *result)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，默认令一个线程处理 16 个输出像素，这四个像素位于统一列
    // 的相邻 16 行上，因此，对于 r 需要进行右移计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) << BINARIZE_PACK_LEVEL;
    int z = blockIdx.z;
    int inidx = r * inimg.pitchBytes + c;
    int cursum = 0;
    int threshold = z + 1;
    do {
        // 线程中处理第一个点。
        // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资
        // 源，一方面防止由于段错误导致的程序崩溃。
        if (r >= inimg.imgMeta.height || c >= inimg.imgMeta.width)
            break;
        // 得到第一个输入坐标点对应的标记值。
        //curlabel = label[inidx];
        if (inimg.imgMeta.imgData[inidx] >= threshold)
            cursum++;

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
        if (inimg.imgMeta.imgData[inidx] >= threshold)
            cursum++;

        // 宏：BINARIZE_KERNEL_MAIN_PHASE
        // 定义计算下一个像素点的程序片段。使用这个宏可以实现获取下一个点的像素
        // 值，并累加到共享内存，并且简化编码量
#define BINARIZE_KERNEL_MAIN_PHASE                                 \
        if (++r >= inimg.imgMeta.height)                             \
            break;                                                   \
        inidx += inimg.pitchBytes;                                  \
        if (inimg.imgMeta.imgData[inidx] >= threshold)              \
            cursum++;


#define BINARIZE_KERNEL_MAIN_PHASEx2                           \
        BINARIZE_KERNEL_MAIN_PHASE                             \
        BINARIZE_KERNEL_MAIN_PHASE

#define BINARIZE_KERNEL_MAIN_PHASEx4                           \
        BINARIZE_KERNEL_MAIN_PHASEx2                           \
        BINARIZE_KERNEL_MAIN_PHASEx2

#define BINARIZE_KERNEL_MAIN_PHASEx8                           \
        BINARIZE_KERNEL_MAIN_PHASEx4                           \
        BINARIZE_KERNEL_MAIN_PHASEx4

#define BINARIZE_KERNEL_MAIN_PHASEx16                          \
        BINARIZE_KERNEL_MAIN_PHASEx8                           \
        BINARIZE_KERNEL_MAIN_PHASEx8

#define BINARIZE_KERNEL_MAIN_PHASEx32                          \
        BINARIZE_KERNEL_MAIN_PHASEx16                          \
        BINARIZE_KERNEL_MAIN_PHASEx16

#define BINARIZE_KERNEL_MAIN_PHASEx64                          \
        BINARIZE_KERNEL_MAIN_PHASEx32                          \
        BINARIZE_KERNEL_MAIN_PHASEx32

// 对于不同的 BINARIZE_PACK_LEVEL ，定义不同的执行次数，从而使一个线程内部
// 实现对多个点的像素值的统计。
#if (BINARIZE_PACK_LEVEL >= 2)
        BINARIZE_KERNEL_MAIN_PHASEx2
#  if (BINARIZE_PACK_LEVEL >= 3)
        BINARIZE_KERNEL_MAIN_PHASEx4
#    if (BINARIZE_PACK_LEVEL >= 4)
        BINARIZE_KERNEL_MAIN_PHASEx8
#      if (BINARIZE_PACK_LEVEL >= 5)
           BINARIZE_KERNEL_MAIN_PHASEx16
#        if (BINARIZE_PACK_LEVEL >= 6)
              BINARIZE_KERNEL_MAIN_PHASEx32
#          if (BINARIZE_PACK_LEVEL >= 7)
                BINARIZE_KERNEL_MAIN_PHASEx64
#          endif
#        endif
#      endif
#    endif
#  endif
#endif

// 取消前面的宏定义。
#undef BINARIZE_KERNEL_MAIN_PHASEx64
#undef BINARIZE_KERNEL_MAIN_PHASEx32
#undef BINARIZE_KERNEL_MAIN_PHASEx16
#undef BINARIZE_KERNEL_MAIN_PHASEx8
#undef BINARIZE_KERNEL_MAIN_PHASEx4
#undef BINARIZE_KERNEL_MAIN_PHASEx2
#undef BINARIZE_KERNEL_MAIN_PHASE
    } while (0);

    // 使用原子操作来保证操作的正确性
    if (cursum != 0)
        atomicAdd(&result[threshold - 1], cursum);
}


// Host 成员方法：binarization（多阈值二值化处理）
__host__ int Binarization::binarization(Image *inimg,
                                        Image *outimg,
                                        float areaRatio)
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

    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一
    if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width)
        insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
    else
        outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;

    if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height)
        insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
    else
        outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;
    // 创建 host 端和 device 端的存放面积值的数组。
    int *devResult;
    int *hostResult;
    hostResult = new int[254];
    cudaError_t cudaerrcode;

    // 为标记数组分配大小。
    cudaerrcode = cudaMalloc((void **)&devResult, 254 * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devResult);
        return cudaerrcode;
    }
    cudaerrcode = cudaMemset(devResult, 0, 254 * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devResult);
        return cudaerrcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = 1;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y ;
    gridsize.z = 254;

    // 设定 gridsize.z 的大小为 254，存储 1 - 254 的阈值。
    //gridsize.z = 254;

   _computeAreaKer<<<gridsize, blocksize>>>(insubimgCud, devResult);


    cudaerrcode = cudaMemcpy(hostResult, devResult, 254 * sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devResult);
        return cudaerrcode;
    }
    // 通过遍历面积值数组找到与标准面积差别最小的面积值，并获取其对应的最佳阈值。
    // 初始化最佳二值化结果的阈值为0, 图像的面积值与标准面积的最小差值初始化为1000。
    int bestnum = 0;
    float min = 1000;

    // |标准面积-TEST图像上的OBJECT面积| / 标准面积 < areaRatio的二值化结果
    for (int i = 0; i < 254; i++) {
        float tmpnum = areaRatio - (float)(fabs(normalArea - hostResult[i])
                       / normalArea);
        if (tmpnum > 0) {
            if (tmpnum < min) {
                min = tmpnum;
                bestnum = i+1;
            }
        }
    }

    // 选出最佳面积比的二值化结果后，设定成员变量 area 的值。
    this->area = hostResult[bestnum - 1];

    // 调用 Binarize 函数使用最佳阈值对输入图像进行二值化。
    Binarize b;
    b.setThreshold(bestnum);
    b.binarize(inimg,outimg);

    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

