// ClusterLocalGray.cu
// 实现图像的分类降噪操作

#include "ClusterLocalGray.h"
#include <iostream>
#include <stdio.h>

using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  16                                                                 
#define DEF_BLOCK_Y  16

// 宏：MAX_NB_SIDE_SPAN
// 像素点最大处理范围。
#define MAX_NB_SIDE_SPAN  16

// 宏：IMG_SMEM_SPAN
// 一个 block 对应的有效共享内存大小。
#define IMG_SMEM_SPAN  48

// 宏：GRAY_RESULTION
// 每一个 bin 的大小。
#define GRAY_RESULTION  4

// 宏：GRAYBIN_NUM
// 将 256 个灰度值按 4 个为一组进行划分，划分成 64 个 bin。
#define GRAYBIN_NUM 64

// 宏：CHECK_SHARED_MOMORY_DEBUG
// 条件编译开关。
// #define CHECK_SHARED_MOMORY_DEBUG 

// __device__ 函数：checkSharedMemory（判断内存拷贝是否成功）
// 条件编译执行函数。判断 block 内是否正确地将图像相应位置拷贝到共享内存中。
// 成功返回 0，不成功则返回 1。
__device__ int checkSharedMemory(
        unsigned char *imgSharedMem,  // 共享内存数组
        ImageCuda inimg,              // 输入图像
        int imgX,                     // block 对应的图像范围的开始 x 坐标
        int imgY,                     // block 对应的图像范围的开始 x 坐标
        int sharedMenSpan             // 共享内存的大小
);

// __device__ 函数：checkSharedMemory（判断内存拷贝是否成功）
__device__ int checkSharedMemory(unsigned char *imgSharedMem, ImageCuda inimg, 
                                 int imgX, int imgY, int sharedMenSpan)
{
    for (int y = 0; y < sharedMenSpan; y++) {
        for (int x = 0; x < sharedMenSpan; x++) {
            // 如果共享内存与所对应的图像上区域有像素点值不同，则认为拷贝失败。
            if (imgSharedMem[y * sharedMenSpan + x] != 
                inimg.imgMeta.imgData[(imgY + y) * 
                inimg.pitchBytes + imgX + x]) {

                // 打印错误点的坐标。
                printf("%d, %d ", imgSharedMem[y * sharedMenSpan + x], 
                        inimg.imgMeta.imgData[(imgY + y) * inimg.pitchBytes + 
                        imgX + x]);
                printf("%d, %d %d, %d  \n", x, y, 
                        imgX + x, (imgY + y));
                // return 1;
            }
        }
    }

    return 0;
}

// Kernel 函数：_clusterLocalGrayKer（图像的分类降噪）
// 每一个 block 的大小为 32 * 32， 在一个 block 内需要将 64 * 64 大小的图像
// 拷贝到共享内存中，即一个线程拷贝 4 个像素点。再根据输入参数 neighborsSideSpan
// 统计像素点领域内像素点的个数，将其分成 64 个 bin，然后再根据 hGrayPercentTh
// 和 lGrayPercentTh 计算出当前像素点高像素比例和低像素比例之间的差值，与 
// grayGapTh 进行对比从而选择对该点进行增强、降低、中庸操作。
static __global__ void _clusterLocalGrayKer(
        ImageCuda inimg,                  // 输入图像
        ImageCuda outimg,                 // 输出图像
        unsigned char neighborsSideSpan,  // 领域大小
        unsigned char hGrayPercentTh,     // 高像素比例
        unsigned char lGrayPercentTh,     // 低像素比例
        unsigned char grayGapTh           // 外部参数
);

// Kernel 函数：_clusterLocalGrayKer（图像的分类降噪）
static __global__ void _clusterLocalGrayKer(ImageCuda inimg, ImageCuda outimg,
                                            unsigned char neighborsSideSpan,
                                            unsigned char hGrayPercentTh,
                                            unsigned char lGrayPercentTh,
                                            unsigned char grayGapTh)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量（其中，
    // c 表示 column， r 表示 row）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;


    // 获取当前像素点在暂存图像中的相对位置。
    int curpos = (dstr + MAX_NB_SIDE_SPAN) * inimg.pitchBytes + dstc +
                  MAX_NB_SIDE_SPAN;

    // 为 block 内的线程需要处理的图像开辟共享内存，大小为 48 * 48。
    __shared__ unsigned char imgSharedMem[2304];

    // 将本 block 对应在图像上的区域拷贝到共享内存中，考虑合并访问的问题，
    // 不是拷贝一个方块，而是尽量同一行。同时将一些变量先计算出来以便减少
    // 计算量。
    unsigned char *tempImgData = inimg.imgMeta.imgData;

    int temp1 = threadIdx.y * IMG_SMEM_SPAN + threadIdx.x;
    int temp2 = dstr * inimg.pitchBytes + dstc;
    imgSharedMem[temp1] = tempImgData[temp2];
    imgSharedMem[temp1 + MAX_NB_SIDE_SPAN] = 
                tempImgData[temp2 + MAX_NB_SIDE_SPAN];
    imgSharedMem[temp1 + MAX_NB_SIDE_SPAN * 2] =
                tempImgData[temp2 + MAX_NB_SIDE_SPAN * 2];

    temp1 = (threadIdx.y + MAX_NB_SIDE_SPAN) * IMG_SMEM_SPAN;
    temp2 = (dstr + MAX_NB_SIDE_SPAN) * inimg.pitchBytes + dstc;
    imgSharedMem[temp1 + threadIdx.x] = tempImgData[temp2];
    imgSharedMem[temp1 + threadIdx.x + MAX_NB_SIDE_SPAN] = 
                tempImgData[temp2 + MAX_NB_SIDE_SPAN];
    imgSharedMem[temp1 + threadIdx.x + MAX_NB_SIDE_SPAN * 2] = 
                tempImgData[temp2 + MAX_NB_SIDE_SPAN * 2];

    temp1 = (threadIdx.y + MAX_NB_SIDE_SPAN * 2) * IMG_SMEM_SPAN;
    temp2 = (dstr + MAX_NB_SIDE_SPAN * 2) * inimg.pitchBytes + dstc;
    imgSharedMem[temp1 + threadIdx.x] = tempImgData[temp2];
    imgSharedMem[temp1 + threadIdx.x + MAX_NB_SIDE_SPAN] = 
                tempImgData[temp2 + MAX_NB_SIDE_SPAN];
    imgSharedMem[temp1 + threadIdx.x + MAX_NB_SIDE_SPAN * 2] = 
                tempImgData[temp2 + MAX_NB_SIDE_SPAN * 2];

    __syncthreads();

    #ifdef CHECK_SHARED_MOMORY_DEBUG
    // 在每一个块的第一个线程内判断内存拷贝是否成功。
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            if (checkSharedMemory(imgSharedMem, inimg, dstc, dstr, IMG_SMEM_SPAN) == 0) {
                printf("blockIdx.x = %d, blockIdx.y = %d, copy sharedMen successful!\n", 
                    blockIdx.x, blockIdx.y);
            } else {
                printf("blockIdx.x = %d, blockIdx.y = %d, copy sharedMen failed!\n", 
                    blockIdx.x, blockIdx.y);
            }
        }
            
    #endif

    // 存放该点处理区域范围内像素值的分布，根据 bin 进行划分。
    unsigned short cp[GRAYBIN_NUM];

    // 共享内存初始化
    for (int i = 0; i < GRAYBIN_NUM; ++i)
        cp[i] = 0;

    // 获得该像素点处理范围的边长。
    unsigned short neighborsSpan = neighborsSideSpan * 2 + 1;

    // 获得该像素点处理范围的面积。
    unsigned short neighborsArea = neighborsSpan * neighborsSpan;

    // 对该像素点处理范围内的点根据 bin 进行基数排序。
    for (int i = 0; i < neighborsArea; ++i)
    {
        cp[imgSharedMem[threadIdx.x + MAX_NB_SIDE_SPAN - neighborsSideSpan + i % 
                        neighborsSpan + 
                        (threadIdx.y + MAX_NB_SIDE_SPAN - neighborsSideSpan + i / 
                        neighborsSpan)
                        * IMG_SMEM_SPAN] >> 2] += 1;
    }

    // 比例范围内点的总数。
    int m = 0;

    // 比例范围内像素点的灰度累计值。
    int gSum = 0;

    // 根据处理范围和低处理比例获得低处理点的数量。
    int  lNumTh = lGrayPercentTh * neighborsArea / 100;
    for ( int n = 0;  m < lNumTh && n < GRAYBIN_NUM; n++) {
        // 点数累积。
        m += cp[n];

        // 灰度累积。
        gSum += n * cp[n];  
    }

    // 计算低处理比例内的平均灰度值。
    unsigned char aveLg = (unsigned char)(GRAY_RESULTION * gSum / m + 2 + 0.5f);  

    // 重新赋值为初值。
    m = 0;
    gSum = 0;

    // 根据处理范围和高处理比例获得高处理点的数量。
    int  hNumTh = hGrayPercentTh * neighborsArea / 100;
    for (int n= 64 - 1; m < hNumTh && n >= 0; n--) {
        // 点数累积。
        m += cp[n];

        // 灰度累积。
        gSum += n * cp[n]; 
    }

    // 计算高处理比例内的平均灰度值。
    unsigned char aveHg = (unsigned char)(GRAY_RESULTION * gSum / m + 2 + 0.5f);

    // 当前 pixel 的 gray 值。
    unsigned char gc = imgSharedMem[IMG_SMEM_SPAN * (threadIdx.y + MAX_NB_SIDE_SPAN) + threadIdx.x + MAX_NB_SIDE_SPAN];  

    // 计算平庸值。
    unsigned char ag = (aveHg + aveLg) >> 1;

    // 根据 grayGapTh 外部参数决定对当前点进行何种处理。
    if ((aveHg - aveLg) < grayGapTh)  
        outimg.imgMeta.imgData[curpos] = ag;
    else if (gc >= aveHg ) 
        outimg.imgMeta.imgData[curpos] = fmin(aveHg * 1.03f, 255); // Enhancing high gray.
    else if ( gc <= aveLg ) 
        outimg.imgMeta.imgData[curpos] = aveLg * 0.98f;           // Depressing low gray.
    else if ( gc >= ag ) 
        outimg.imgMeta.imgData[curpos] = (aveHg + ag) >> 1;
    else 
        outimg.imgMeta.imgData[curpos] = (aveLg + ag) >> 1; 
}

// 成员方法：clusterLocalGray（图像分类降噪处理）
__host__ int ClusterLocalGray::clusterLocalGray(Image *inimg, Image *outimg)
{
    // 局部变量，错误码。
    int errcode;  

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 将输入图像复制到 device
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入图像 inimg 完全拷贝到输出图像 outimg ，并将 outimg 拷贝到 
    // device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR) 
        return errcode;

    // 提取输出图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) 
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;

    // 这里需要减去边界的宽度来计算图像的有效位置。 
    gridsize.x = (outsubimgCud.imgMeta.width - MAX_NB_SIDE_SPAN * 2 + blocksize.x - 1) /
                 blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height - MAX_NB_SIDE_SPAN * 2 + blocksize.y - 1) / 
                 blocksize.y;

    // 调用核函数，开始第一步细化操作。
    _clusterLocalGrayKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud,
                                                  this->getNeighborsSideSpan(),
                                                  this->getHGrayPercentTh(),
                                                  this->getLGrayPercentTh(),
                                                  this->getGrayGapTh());
    if (cudaGetLastError() != cudaSuccess) {
        // 核函数出错，结束迭代函数。
        return CUDA_ERROR;
    }

    return NO_ERROR;    
}

