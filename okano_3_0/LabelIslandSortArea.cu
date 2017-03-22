// LabelIslandSortArea.cu 
// 实现区域排序算法

#include "LabelIslandSortArea.h"


#include <iostream>
using namespace std;


// 宏：MAX_PAIRS_NUM  
//（面积值-标记值）键值对的个数。
#ifndef MAX_PAIRS_NUM
#define MAX_PAIRS_NUM 256
#endif

// 宏：SORT_ARRAY_TYPE_ASC
// 排序标识，升序排序。
#ifndef SORT_ARRAY_TYPE_ASC
#define SORT_ARRAY_TYPE_ASC  2
#endif

// 宏：SORT_ARRAY_TYPE_DESC
// 排序标识，降序排序。
#ifndef SORT_ARRAY_TYPE_DESC
#define SORT_ARRAY_TYPE_DESC 1
#endif


// Kernel 函数: _findAreasByMinMaxKer（筛选面积）
// 筛选出在最大最小面积范围之间的标记区域。
static __global__ void 
_findAreasByMinMaxKer(
        unsigned int *histogram,  // 直方图面积。
        unsigned int minArea,     // 最小面积。
        unsigned int maxArea      // 最大面积。
);

// Kernel 函数: _bitonicSortPairsByAscendKer（按照升序排序区域面积）
// 实现并行双调排序，按照升序排序区域面积。
static __global__ void 
_bitonicSortPairsByAscendKer(
        unsigned int *devarray,    // 面积数组。
        unsigned int *devareaRank  // 输出的（面积值-标记值）键值对。
);

// Kernel 函数: _bitonicSortPairsByDescendKer（按照降序排序区域面积）
// 实现并行双调排序，按照降序排序区域面积。
static __global__ void 
_bitonicSortPairsByDescendKer(
        unsigned int *devarray,    // 面积数组。
        unsigned int *devareaRank  // 输出的（面积值-标记值）键值对。
);

// Kernel 函数: _bitonicSortPairsByDescendKer（按照降序排序区域面积）
static __global__ void _bitonicSortPairsByDescendKer(
        unsigned int *devarray, unsigned int *devareaRank)
{
    // 读取线程号。
    int tid = threadIdx.x;
    int k, ixj, j;
    unsigned int tempArea, tempIndex;

    // 声明共享内存，加快数据存取速度。
    __shared__ unsigned int area[MAX_PAIRS_NUM];
    __shared__ unsigned int index[MAX_PAIRS_NUM];

    // 将面积值拷贝到共享内存中。
    area[tid] = devarray[tid];
    // 将标记值拷贝到共享内存了。
    index[tid] = tid;
    __syncthreads();

    // 并行双调排序，降序排序。
    for (k = 2; k <= MAX_PAIRS_NUM; k = k << 1) {
        // 双调合并。
        for (j = k >> 1; j > 0; j = j >> 1) {
            // ixj 是与当前位置 i 进行比较交换的位置。
            ixj = tid ^ j;
            if (ixj > tid) {
                // 如果 (tid & k) == 0，按照降序交换两项。
                if ((tid & k) == 0 && (area[tid] < area[ixj])) {
                    // 交换面积值。
                    tempArea = area[tid];
                    area[tid] = area[ixj];
                    area[ixj] = tempArea;
                    // 交换下标值。
                    tempIndex = index[tid];
                    index[tid] = index[ixj];
                    index[ixj] = tempIndex;
                // 如果 (tid & k) == 0，按照升序交换两项。
                } else if ((tid & k) != 0 && area[tid] > area[ixj]) {
                    // 交换面积值。
                    tempArea = area[tid];
                    area[tid] = area[ixj];
                    area[ixj] = tempArea;
                    // 交换下标值。
                    tempIndex = index[tid];
                    index[tid] = index[ixj];
                    index[ixj] = tempIndex;
                }
            }
            __syncthreads();
        }
    }
    // 将共享内存中的面积值拷贝到全局内存中。
    devareaRank[2 * tid] = area[tid];
    // 将共享内存中的下标值拷贝到全局内存中。
    devareaRank[2 * tid + 1] = index[tid];
}

// Kernel 函数: _bitonicSortPairsByAscendKer（按照升序排序区域面积）
static __global__ void _bitonicSortPairsByAscendKer(
        unsigned int *devarray, unsigned int *devareaRank)
{
    // 读取线程号。
    int tid = threadIdx.x;
    int k, ixj, j;
    unsigned int tempArea, tempIndex;

    // 声明共享内存，加快数据存取速度。
    __shared__ unsigned int area[MAX_PAIRS_NUM];
    __shared__ unsigned int index[MAX_PAIRS_NUM];

    // 将面积值拷贝到共享内存中。
    area[tid] = devarray[tid];
    // 将标记值拷贝到共享内存了。
    index[tid] = tid;
    __syncthreads();

    // 并行双调排序，升序排序。
    for (k = 2; k <= MAX_PAIRS_NUM; k = k << 1) {
        // 双调合并。
        for (j = k >> 1; j > 0; j = j >> 1) {
            // ixj 是与当前位置 i 进行比较交换的位置。
            ixj = tid ^ j;
            if (ixj > tid) {
                // 如果 (tid & k) == 0，按照升序交换两项。
                if ((tid & k) == 0 && (area[tid] > area[ixj])) {
                    // 交换面积值。
                    tempArea = area[tid];
                    area[tid] = area[ixj];
                    area[ixj] = tempArea;
                    // 交换下标值。
                    tempIndex = index[tid];
                    index[tid] = index[ixj];
                    index[ixj] = tempIndex;
                // 如果 (tid & k) == 0，按照降序交换两项。
                } else if ((tid & k) != 0 && area[tid] < area[ixj]) {
                    // 交换面积值。
                    tempArea = area[tid];
                    area[tid] = area[ixj];
                    area[ixj] = tempArea;
                    // 交换下标值。
                    tempIndex = index[tid];
                    index[tid] = index[ixj];
                    index[ixj] = tempIndex;
                }
            }
            __syncthreads();
        }
    }
    // 将共享内存中的面积值拷贝到全局内存中。
    devareaRank[2 * tid] = area[tid];
    // 将共享内存中的下标值拷贝到全局内存中。
    devareaRank[2 * tid + 1] = index[tid];
}

// Host 成员方法：bitonicSortPairs（对区域面积进行排序）
__host__ int LabelIslandSortArea::bitonicSortPairs(
        unsigned int *inarray, unsigned int *areaRank)
{
    // 检查 inarray 是否为空
    if (inarray == NULL)
        return NULL_POINTER;

    // 检查 areaRank 是否为空
    if (areaRank == NULL)
        return NULL_POINTER;

    if (this->sortflag == SORT_ARRAY_TYPE_ASC)
        // 升序排序区域面积。
        _bitonicSortPairsByAscendKer<<<1, MAX_PAIRS_NUM>>>(inarray, areaRank);
    else if (this->sortflag == SORT_ARRAY_TYPE_DESC)
        // 降序排序区域面积。
        _bitonicSortPairsByDescendKer<<<1, MAX_PAIRS_NUM>>>(inarray, areaRank);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;    
           
    return NO_ERROR;
}

// Kernel 函数: _findAreasByMinMaxKer（筛选面积）
static __global__ void _findAreasByMinMaxKer(
        unsigned int *histogram, unsigned int minArea, unsigned int maxArea)
{
    // 获取线程号。
    int tid = threadIdx.x;
    histogram[0] = 0;
    // 如果直方图面积不在最大最小面积范围内，则将其对应面积清0。
    if (histogram[tid] < minArea || histogram[tid] > maxArea)
        histogram[tid] = 0;
}

// Host 成员方法：labelIslandSortArea（对标记后的所有区域按照面积进行排序）
__host__ int LabelIslandSortArea::labelIslandSortArea(
        Image *inimg, unsigned int *areaRank)
{
    // 检查图像是否为 NULL。
    if (inimg == NULL)
        return NULL_POINTER;

    // 检查 areaRank 是否为空
    if (areaRank == NULL)
        return NULL_POINTER;

    // 检查参数是否合法。
    if (minarea < 0 || maxarea <  0 || (sortflag != SORT_ARRAY_TYPE_ASC && 
        sortflag != SORT_ARRAY_TYPE_DESC))
        return INVALID_DATA;

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

    // 在 Device 上分配临时空间。一次申请所有空间，然后通过偏移索引各个数组。
    cudaError_t cudaerrcode;
    unsigned int *alldevicedata;
    unsigned int *devhistogram, *devareaRank;
    cudaerrcode = cudaMalloc((void**)&alldevicedata,
                             3 * MAX_PAIRS_NUM * sizeof (unsigned int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 初始化 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldevicedata, 0,
                             3 * MAX_PAIRS_NUM * sizeof (unsigned int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 通过偏移读取 devhistogram 内存空间。
    devhistogram = alldevicedata;

    // 通过直方图计算区域面积.
    Histogram hist;
    errcode = hist.histogram(inimg, devhistogram, 0);
    if (errcode != NO_ERROR)
        return errcode;

    // 筛选出在最大最小面积范围之间的标记区域。
    _findAreasByMinMaxKer<<<1, MAX_PAIRS_NUM>>>(devhistogram, 
                                                minarea, maxarea);
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicedata);
        return CUDA_ERROR;   
    }
            
    // areaRank 在 Host 端。
    if (this->ishost == 1) {
        // 通过偏移读取 devareaRank 内存空间。
        devareaRank = alldevicedata + MAX_PAIRS_NUM;

        // 调用并行双调排序函数，对所选面积进行排序。
        bitonicSortPairs(devhistogram, devareaRank);

        //将 Device上的 devareaRank 拷贝到 Host上。
        cudaerrcode = cudaMemcpy(areaRank, devareaRank, 
                                 MAX_PAIRS_NUM  * 2 * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess)
            return cudaerrcode;

        // 计算满足条件的不同区域的个数
        int k;
        this->length = 0;
        for (k = 0; k< MAX_PAIRS_NUM; k++) {
            if (areaRank [2 * k] > 0)
                this->length++;
        }
    
        // 当 areaRank 按照升序排序时，在数组的前面会有很多无效的 0 项，因为
        // 区域的个数可能小于定义的 MAX_PAIRS_NUM，所以需要重新颠倒数组，使得
        // 有效的非零数据位于数组的前面。下面的代码就是解决此问题的。例如处理
        // 前假设 areaRank = [0, 0, 0, 0, ......, 50000, 8, 60000, 3, 70000, 6]
        // ，那么处理后 areaRank = [50000, 8, 60000, 3, 70000, 6, 0, 0, 0, 0,
        //  ......]。
        int i, j;
        if (sortflag == 2) {
            if (areaRank[0] == 0) {
                j = 0;
                for (i = 0; i < MAX_PAIRS_NUM; i++) {
                    // 如果面积大于0，则迁移。
                    if (areaRank[2 * i] > 0) {
                        areaRank[2 * j] = areaRank[2 * i];
                        areaRank[2 * j + 1] = areaRank[2 * i + 1];
                        areaRank[2 * i] = 0;
                        areaRank[2 * i + 1] = 0;
                        j++;
                    }
                }
            }
        }
    // areaRank 在 Device 端。
    } else if (this->ishost == 0) {
        // 声明 Host 端数组，为以后的处理做准备。
        unsigned int hostareaRank[MAX_PAIRS_NUM * 2];
        // 通过偏移读取 devareaRank 内存空间。
        devareaRank = alldevicedata + MAX_PAIRS_NUM;

        // 调用并行双调排序函数，对所选面积进行排序。
        bitonicSortPairs(devhistogram, areaRank);

        //将 Device上的 areaRank 拷贝到 Host上。
        cudaerrcode = cudaMemcpy(hostareaRank, areaRank, 
                                 MAX_PAIRS_NUM * 2 * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess)
            return cudaerrcode;

        // 计算满足条件的不同区域的个数
        int k;
        this->length = 0;
        for (k = 0; k< MAX_PAIRS_NUM; k++) {
            if (hostareaRank [2 * k] > 0)
                this->length++;
        }
    
        // 当 hostareaRank 按照升序排序时，在数组的前面会有很多无效的 0 项，
        // 因为区域的个数可能小于定义的 MAX_PAIRS_NUM，所以需要重新颠倒数组，
        // 使得有效的非零数据位于数组的前面。下面的代码就是解决此问题的。
        // 例如处理前假设 hostareaRank = [0, 0, 0, 0, ......, 50000, 8, 60000,
        // 3, 70000, 6]，那么处理后 hostareaRank = [50000, 8, 60000, 3, 70000,
        // 6, 0, 0, 0, 0,......]。
        int i, j;
        if (sortflag == 2) {
            if (hostareaRank[0] == 0) {
                j = 0;
                for (i = 0; i < MAX_PAIRS_NUM; i++) {
                    // 如果面积大于0，则迁移。
                    if (hostareaRank[2 * i] > 0) {
                        hostareaRank[2 * j] = hostareaRank[2 * i];
                        hostareaRank[2 * j + 1] = hostareaRank[2 * i + 1];
                        hostareaRank[2 * i] = 0;
                        hostareaRank[2 * i + 1] = 0;
                        j++;
                    }
                }
            }
        }
    }

    // 释放显存上的临时空间。
    cudaFree(alldevicedata);

    return NO_ERROR;
}
