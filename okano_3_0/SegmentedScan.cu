// SegmentedScan.cu
// 数组分段扫描

#include "SegmentedScan.h"
#include <cmath>
#include <iostream>
#include <stdio.h>
using namespace std;

// 宏：SEG_SCAN_PACK
// 定义了核函数中每段处理的数量。
#define SEG_SCAN_PACK                                        16

// 宏：SEG_SCAN_PACK_NUM
// 定义了核函数中处理的段数。
#define SEG_SCAN_PACK_NUM                                    64

// 宏：SEG_SCAN_BLOCKSIZE
// 定义了线程块大小。
#define SEG_SCAN_BLOCKSIZE  (SEG_SCAN_PACK * SEG_SCAN_PACK_NUM)

// 宏：SEG_DEBUG_CPU_PRINT（CPU 版本调试打印开关）
// 打开该开关则会在 CPU 版本运行时打印相关的信息，以参考调试程序；如果注释掉该
// 宏，则 CPU 不会打印这些信息，但这会有助于程序更快速的运行。
// #define SEG_DEBUG_CPU_PRINT

// Kernel 函数: _segmentedScanMatrixKer（数组分段扫描的矩阵方法版本）
// 矩阵方法的SegmentedScan 实现。具体的算法为核函数用 SEG_SCAN_PACK_NUM 个线程
// 进行扫描，将输入的数组分为 SEG_SCAN_PACK_NUM 段，每段的 SEG_SCAN_PACK 个元素
// 采用串行的方法进行分段扫描。
__global__ void               // Kernel 函数无返回值
_segmentedScanMatrixKer(
        float *inarray,       // 输入数组。
        int *label,           // 输入数组的分段标签值。
        int *index,           // 输入数组的位置索引。
        int n,                // 数组的长度，处理元素的个数。
        float *maxdist,       // 输出，分段扫描后的最大垂直距离数组，
        int *maxdistidx,      // 输出，分段扫描侯的最大垂距的索引值数组。
        float *blockmaxdist,  // 中间结果，每块最后位置的最大垂距。
        int *blocklabel,      // 中间结果，每块最后位置处的标签。
        int *blockmaxdistidx  // 中间结果，每块最后位置处的垂距索引。
);

// Kernel 函数: _segmentedScanBackKer（将中间结果进行返回分段扫描）
// 对一个数组进行扫描，结合中间结果的小数组，对输出数组进行返回式的分段扫描。
__global__ void                // Kernel 函数无返回值
_segmentedScanBackKer(
        float *maxdist,        // 输出的分段扫描后的最大垂直距离数组，
                               // 表示每段中的垂距最大的值。
        int *maxdistidx,       // 输出的扫描后最大垂距点的位置索引。
        int *label,            // 输入数组的分段标签值。
        float *blockmaxdist,   // 中间结果，每块最后位置的最大垂距。
        int *blocklabel,       // 中间结果，每块最后位置处的标签。
        int *blockmaxdistidx,  // 中间结果，每块最后位置处的垂距索引。
        int numelements        // 扫描数组的长度。
);

// Kernel 函数: _segmentedScanMatrixKer（数组分段扫描的矩阵方法版本）
__global__ void _segmentedScanMatrixKer(float *inarray, int *label, 
                                        int *index, int n,
                                        float *maxdist, int *maxdistidx,
                                        float *blockmaxdist, int *blocklabel,
                                        int *blockmaxdistidx)
{
    // 声明共享内存。shd开头的指针表示当前点的信息，长度为块大小。
    // shdcol开头的指针表示每段数组最后一个元素的信息，长度为 PACK_NUM。
    extern __shared__ float shdmem[];
    float *shdmaxdist = shdmem;
    float *shdcolmaxdist = &shdmaxdist[blockDim.x];
    int *shdlabel = (int*)&shdcolmaxdist[SEG_SCAN_PACK_NUM];
    int *shdcollabel = &shdlabel[blockDim.x];
    int *shdindex = &shdcollabel[SEG_SCAN_PACK_NUM];
    int *shdcolindex = &shdindex[blockDim.x];

    // 基础索引。表示每块的起始位置索引。
    int baseidx = blockIdx.x * blockDim.x;

    // 块外的数组索引。
    int idx = threadIdx.x + baseidx;

    // 采用矩阵方法分段扫描，每段的头索引。
    int packidx = SEG_SCAN_PACK * threadIdx.x;

    // 本地变量。表示目前扫描过的已知的最大垂距，区域标签，索引。
    float curmaxdist;
    int curlabel;
    int curindex;

    // 本地变量，特殊值，用来给超出数组长度的点赋值或者给不需要处理的点赋值。
    float pmaxdist = -100;
    int plabel = -1;
    int pindex = -1;

    // 将需要计算的值从输入加载到共享内存上。
    if (idx < n) {
        shdmaxdist[threadIdx.x] = inarray[idx];
        shdlabel[threadIdx.x] = label[idx];
        // 如果记录最大垂距位置索引的指针为空，用当前点的实际索引对共享内存进行
        // 赋值。
        if (index == NULL)
            shdindex[threadIdx.x] = idx;
        // 否则用输入对共享内存进行赋值。
        else
            shdindex[threadIdx.x] = index[idx];
    // 超出数组长度的点赋特殊值。
    } else {
        shdmaxdist[threadIdx.x] = pmaxdist;
        shdlabel[threadIdx.x] = plabel;
        shdindex[threadIdx.x] = pindex;
    }

    // 进行块内同步。
    __syncthreads();

    // 用 SEG_SCAN_PACK_NUM 个线程对每段进行 segmented scan，段内为
    // SEG_SCAN_PACK 个元素的串行扫描。
    if (threadIdx.x < SEG_SCAN_PACK_NUM) {
        // 记录每段 SEG_SCAN_PACK 中开始位置处的值，作为目前已知最大垂距的点的
        // 信息。
        curmaxdist = shdmaxdist[packidx];
        curlabel = shdlabel[packidx];
        curindex = shdindex[packidx];

        // 对每段 SEG_SCAN_PACK 进行串行扫描。
        for (int i = packidx + 1; i < packidx + SEG_SCAN_PACK; i++)
        {
            // 如果当前点的区域标签和目前已知的最大垂距的点的区域标签不同，
            // 那么重新记录目前已知最大垂距点的信息。
            // 或者区域标签相同，当前点的垂距大于目前已知的最大垂距点的垂距，
            // 那么重新记录目前已知最大垂距点的信息。
            if (shdlabel[i] != curlabel || shdmaxdist[i] > curmaxdist) {
                curmaxdist = shdmaxdist[i];
                curlabel = shdlabel[i];
                curindex = shdindex[i];
            // 否则就更改当前点的最大垂距点位置的索引，更新当前点记录的已知的最
            // 大垂距值。
            } else {
                shdindex[i] = curindex;
                shdmaxdist[i] = curmaxdist;
            }
        }

        // 将每段 SEG_SCAN_PACK 进行分段扫描后的记录值(目前已知的最大垂距点信
        // 息)写入列数组。
        shdcolmaxdist[threadIdx.x] = curmaxdist;
        shdcollabel[threadIdx.x] = curlabel;
        shdcolindex[threadIdx.x] = curindex;
    }

    // 进行块内同步。
    __syncthreads();

    // 用第 0 个线程对保存每段 SEG_SCAN_PACK 中已知信息的列数组 shdcol 进行串行
    // 分段扫描
    if (threadIdx.x == 0) {
        // 串行扫描，扫描长度为 PACK_NUM。
        for(int i = 1; i < SEG_SCAN_PACK_NUM; i++) {
            // 从上之下对列数组进行扫描，比较相邻的两个点的区域标签和最大垂距，
            // 如果属于同一区域， 并且最大距离小于前一个记录值，那么就改写当前
            // 的记录。
            if (shdcollabel[i] == shdcollabel[i - 1] && 
                shdcolmaxdist[i] < shdcolmaxdist[i - 1]) {
                shdcolmaxdist[i] = shdcolmaxdist[i - 1];
                shdcollabel[i] = shdcollabel[i - 1];
                shdcolindex[i] = shdcolindex[i - 1];
            }
        }
    }

    // 进行块内同步。
    __syncthreads();

    // 用 SEG_SCAN_PACK_NUM 个线程对每段 SEG_SCAN_PACK 进行回扫，将列数组经过分
    // 段扫描后的更新值，与每段 SEG_SCAN_PACK 的值进行比较，进行更新分段扫描。
    if (threadIdx.x < SEG_SCAN_PACK_NUM) {
        // 对于第一段，不需要用更新的列数组结果进行比较，故赋予特殊值。
        if (threadIdx.x == 0) {
            curmaxdist = pmaxdist;
            curlabel = plabel;
            curindex = pindex;
        // 对于之后的每段，需要跟前一段的目前扫描到的最大垂距点的信息进行更新，
        // 所以把列数组中保存的，当前段的前一段的更新信息赋值给目前所能找到的最
        // 大垂距点信息。
        } else {
            curmaxdist = shdcolmaxdist[threadIdx.x - 1];
            curlabel = shdcollabel[threadIdx.x - 1];
            curindex = shdcolindex[threadIdx.x - 1];
        }

        // 对每行进行串行的更新扫描
        for (int i = packidx; i < packidx + SEG_SCAN_PACK; i++) {
            // 比较当前点和目前已知最大垂距点的区域标签和垂距，
            // 如果属于同一区域， 并且当前点的垂距小于目前已知信息，那么就改写
            // 当前点的记录。
            if (curlabel == shdlabel[i] && shdmaxdist[i] < curmaxdist) {
                shdmaxdist[i] = curmaxdist;
                shdindex[i] = curindex;
            // 否则，就 break 掉。
            } else {
                break;
            }
        }
    }

    // 进行块内同步。
    __syncthreads();

    // 超出数组长度 n 的值不进行写入，直接返回。
    if (idx >= n)
        return;
    // 将结果从共享内存写入到输出。包括本区域内的最大垂距和本区域内的最大垂距的
    // 点的位置索引。
    maxdist[idx] = shdmaxdist[threadIdx.x];
    maxdistidx[idx] = shdindex[threadIdx.x];

    // 如果中间结果数组为空，不进行处理直接返回。
    if (blockmaxdist == NULL)
        return;
    // 如果大于一个线程块，用第 0 个线程把每块的最后一个点的信息记录到中间结果
    // 数组。
    if (blockIdx.x < gridDim.x - 1 && threadIdx.x == 0) {
        blockmaxdist[blockIdx.x] = shdcolmaxdist[SEG_SCAN_PACK_NUM - 1];
        blocklabel[blockIdx.x] = shdcollabel[SEG_SCAN_PACK_NUM - 1];
        blockmaxdistidx[blockIdx.x] = shdcolindex[SEG_SCAN_PACK_NUM - 1];
    }
}

// Kernel 函数: _segmentedScanBackKer（将中间结果进行返回分段扫描）
__global__ void _segmentedScanBackKer(float *maxdist, int *maxdistidx,
                                      int *label, float *blockmaxdist,
                                      int *blocklabel, int *blockmaxdistidx,
                                      int numelements)
{
    // 声明共享内存。用来存放中间结果小数组中的元素，也就是输入的原数组的每块最
    // 后一个元素。共包含三个信息。
    __shared__ float shdcurmaxdist[1];
    __shared__ int shdcurlabel[1];
    __shared__ int shdcurmaxdistindex[1];

            
    // 状态位，用来标记上一块的最后一个元素的标签值是否和本段第一个元素的标签值
    // 相同。
    __shared__ int state[1];

    // 计算需要进行块间累加位置索引（块外的数组索引）。
    int idx = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

    // 用每块的第一个线程来读取每块前一块的最后一个元素，从中间结果数组中读取。
    if (threadIdx.x == 0) {
        shdcurmaxdist[0] = blockmaxdist[blockIdx.x];
        shdcurlabel[0] = blocklabel[blockIdx.x];
        shdcurmaxdistindex[0] = blockmaxdistidx[blockIdx.x];
        // 用 state 来记录上一块的最后一个元素的标签值是否和本段第一个元素的
        // 标签值相同，相同则为 1，不同则为 0。
        state[0] = (label[idx] == shdcurlabel[0]);
    }

    // 块内同步。
    __syncthreads();

    // 如果状态位为 0，说明上一块和本块无关，不在一个区域内，直接返回。
    if (state[0] == 0)
        return;
    // 如果数组索引大于数组长度，直接返回。
    if (idx >= numelements)
        return;
    // 如果当前位置处的标签值和目前已知的最大垂距的标签值相同，并且垂距小于目前
    // 已知的最大垂距，那么更新当前位置处的最大垂距记录和最大垂距位置的索引。
    if (label[idx] == shdcurlabel[0] && maxdist[idx] < shdcurmaxdist[0]) {
        maxdist[idx] = shdcurmaxdist[0];
        maxdistidx[idx] = shdcurmaxdistindex[0];
    }
}

// Host 成员方法：segmentedScanBack（将中间结果进行返回分段扫描）
__host__ int SegmentedScan::segmentedScanBack(float *maxdist, int *maxdistidx,
                                              int *label, float *blockmaxdist,
                                              int *blocklabel, 
                                              int *blockmaxdistidx,
                                              int numelements)
{
    // 检查输入和输出是否为 NULL，如果为 NULL 直接报错返回。
    if (maxdist == NULL || maxdistidx == NULL || label == NULL ||
        blockmaxdist == NULL || blocklabel == NULL || blockmaxdistidx == NULL)
        return NULL_POINTER;

    // 检查处理的数组长度，如果小于 0 出错。
    if (numelements < 0)
        return INVALID_DATA;
    
    // 计算线程块大小。
    int gridsize = max(1, (numelements + SEG_SCAN_BLOCKSIZE - 1) /
                       SEG_SCAN_BLOCKSIZE);

    // 判断 gridsize 大小，如果小于 1，则不用进行加回操作。返回正确。
    if (gridsize < 1)
        return NO_ERROR;

    // 调用 _segmentedScanBackKer 核函数，将中间结果数组加回到原扫描数组。
    _segmentedScanBackKer<<<gridsize, SEG_SCAN_BLOCKSIZE>>>(
            maxdist, maxdistidx, label, blockmaxdist, blocklabel,
            blockmaxdistidx, numelements);

    // 判断是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
       
    // 处理完毕退出。
    return NO_ERROR;
}

// Host 成员方法：segmentedScan（数组分段扫描）
// 输入输出均在 CPU 端，不做 GPU 端考虑。以后可考虑添加输入输出在 device 端情况
// 不涉及中间变量和加回过程，串行代码。
__host__ int SegmentedScan::segmentedScanCpu(float *inarray, int *label,
                                             int *index, float *maxdist,
                                             int *maxdistidx, int numelements,
                                             bool hostinarray, bool hostlabel,
                                             bool hostindex, bool hostmaxdist,
                                             bool hostmaxdistidx)
{
    // 检查输入和输出是否为 NULL，如果为 NULL 直接报错返回。
    if (inarray == NULL || label == NULL || maxdist == NULL ||
        maxdistidx == NULL)
        return NULL_POINTER;

    // 本程序实现的方法可处理的数组长度，加以判断控制。
    if (numelements < 0)
        return INVALID_DATA;

    // 本地变量
    int idx;
    float curmaxdist;
    int curlabel;
    int curindex;

    // 如果记录最大垂距位置索引的指针为空，用当前点的实际索引进行赋值。
    if (index == NULL) {
        // 申请数组
        index = new int[numelements];
        // 申请失败
        if (index == NULL) {
            return OUT_OF_MEM;
        }

        // 对索引数组赋值
        for (idx = 0; idx < numelements; idx++) {
            index[idx] = idx;
#ifdef SEG_DEBUG_CPU_PRINT
            cout << "[Cpu]index " << idx << " is " << index[idx] << endl;
#endif
        }
    }

    // 当前信息初始化
    curmaxdist = inarray[0];
    curlabel = label[0];
    curindex = index[0];

    // 输出首位初始化
    maxdist[0] = curmaxdist;
    maxdistidx[0] = curindex;
#ifdef SEG_DEBUG_CPU_PRINT
    cout << "[CPU]0 maxdist is " << maxdist[0] << endl;
    cout << "[CPU]0 maxdistidx is " << maxdistidx[0] << endl;
#endif

    // 从第一个点开始处理
    for (idx = 1; idx < numelements; idx++) {
        // 如果此点标签与当前标签不符，或者此点的垂距大于当前最大垂距，
        // 更新当前最大垂距，当前标签，当前最大垂距的位置索引。
        if (label[idx] != curlabel || inarray[idx] > curmaxdist) {
            curmaxdist = inarray[idx];
            curlabel = label[idx];
            curindex = index[idx];
        }
        // 对输出赋值
        maxdist[idx] = curmaxdist;
        maxdistidx[idx] = curindex;
#ifdef SEG_DEBUG_CPU_PRINT
        cout << idx << "[CPU]maxdist is " << maxdist[idx] << endl;
        cout << idx <<  "[CPU]maxdistidx is " << maxdistidx[idx] << endl;
#endif
    }

    // 释放内存
    delete index;

    // 处理完毕退出。
    return NO_ERROR;
}

// 宏：FAIL_SEGMENTED_SCAN_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_SEGMENTED_SCAN_FREE  do {                \
        if (gridsize > 1) {                           \
            if(blockmaxdistDev != NULL)               \
                cudaFree(blockmaxdistDev);            \
            if(blocklabelDev != NULL)                 \
                cudaFree(blocklabelDev);              \
            if(blockmaxdistidxDev != NULL)            \
                cudaFree(blockmaxdistidxDev);         \
        }                                             \
        if (hostinarray && inarrayDev != NULL)        \
            cudaFree(inarrayDev);                     \
        if (hostlabel && labelDev != NULL)            \
            cudaFree(labelDev);                       \
        if (hostindex && indexDev != NULL)            \
            cudaFree(indexDev);                       \
        if (hostmaxdist && maxdistDev != NULL)        \
            cudaFree(maxdistDev);                     \
        if (hostmaxdistidx && maxdistidxDev != NULL)  \
            cudaFree(maxdistidxDev);                  \
    } while (0)

// Host 成员方法：segmentedScan（数组分段扫描）
__host__ int SegmentedScan::segmentedScan(float *inarray, int *label,
                                          int *index, float *maxdist,
                                          int *maxdistidx, int numelements,
                                          bool hostinarray, bool hostlabel,
                                          bool hostindex, bool hostmaxdist,
                                          bool hostmaxdistidx)
{
    // 检查输入和输出是否为 NULL，如果为 NULL 直接报错返回。
    if (inarray == NULL || label == NULL || maxdist == NULL ||
        maxdistidx == NULL)
        return NULL_POINTER;

    // 本程序实现的方法可处理的数组长度，加以判断控制。
    if (numelements < 0)
        return INVALID_DATA;

    // 局部变量，错误码。
    cudaError_t cuerrcode;
    int errcode;

    // 计算共享内存的长度。
    unsigned int sharedmemsize = 0; 

    // 定义设备端的输入输出数组指针，当输入输出指针在 Host 端时，在设备端申请对
    // 应大小的数组。
    float *inarrayDev = NULL;
    int *labelDev = NULL;
    int *indexDev = NULL;
    float *maxdistDev = NULL;
    int *maxdistidxDev = NULL;

    // 线程块的大小尺寸。
    int gridsize = 0;
    int blocksize;

    // 局部变量，中间结果存放数组。长度会根据线程块大小来确定。
    float *blockmaxdistDev = NULL;
    int *blocklabelDev = NULL;
    int *blockmaxdistidxDev = NULL;

    // 中间结果数组的长度。
    int blocksumsize;

    // 判断当前 inarray 数组是否存储在 Host 端。若是，则需要在 Device 端为数组
    // 申请一段空间；若该数组是在 Device 端，则直接使用。
    if (hostinarray) {
        // 为输入数组在设备端申请内存。    
        cuerrcode = cudaMalloc((void **)&inarrayDev,
                               sizeof (float) * numelements);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return cuerrcode;
        }

        // 将输入数组拷贝到设备端内存。
        cuerrcode = cudaMemcpy(inarrayDev, inarray,
                               sizeof (float) * numelements, 
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return cuerrcode;
        }
    } else {
        // 如果在设备端，则将指针传给对应的设备端统一指针。
        inarrayDev = inarray;
    }

    // 判断当前 label 数组是否存储在 Host 端。若是，则需要在 Device 端为数组
    // 申请一段空间；若该数组是在 Device 端，则直接使用。
    if (hostlabel) {
        // 为输入数组在设备端申请内存。    
        cuerrcode = cudaMalloc((void **)&labelDev, sizeof (int) * numelements);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return cuerrcode;
        }

        // 将输入数组拷贝到设备端内存。
        cuerrcode = cudaMemcpy(labelDev, label, sizeof (int) * numelements, 
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return cuerrcode;
        }
    } else {
        // 如果在设备端，则将指针传给对应的设备端统一指针。
        labelDev = label;
    }

    // 判断当前 label 数组是否存储在 Host 端。若在设备端，则直接使用。若在 Host
    // 端，则不处理。
    if (!hostindex) {
        // 如果在设备端，则将指针传给对应的设备端统一指针。
        indexDev = index;
    }

    // 判断当前 maxdist 数组是否存储在 Host 端。若是，则需要在 Device 端为数组
    // 申请一段空间；若该数组是在 Device 端，则直接使用。
    if (hostmaxdist) {
        // 为输出数组在设备端申请内存。
        cuerrcode = cudaMalloc((void **)&maxdistDev,
                               sizeof (float) * numelements);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return cuerrcode;
        }
    } else {
        // 如果在设备端，则将指针传给对应的设备端统一指针。
        maxdistDev = maxdist;
    }

    // 判断当前 maxdistidx 数组是否存储在 Host 端。若是，则需要在 Device 端为数
    // 组申请一段空间；若该数组是在 Device 端，则直接使用。
    if (hostmaxdistidx) {
        // 为输出数组在设备端申请内存。
        cuerrcode = cudaMalloc((void **)&maxdistidxDev,
                               sizeof (int) * numelements);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return cuerrcode;
        }
    } else {
        // 如果在设备端，则将指针传给对应的设备端统一指针。
        maxdistidxDev = maxdistidx;
    }

    // 针对不同的实现类型，选择不同的路径进行处理。
    switch(segmentedScanType) {
    // 使用矩阵方法的 segmentedscan 实现。
    case MATRIX_SEGMENTED_SCAN:
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        // 矩阵方法分段扫描版本线程块大小。
        blocksize = SEG_SCAN_BLOCKSIZE;

        // 计算线程块大小和共享内存长度。
        gridsize = max(1, (numelements + blocksize - 1) / blocksize);
        sharedmemsize = (sizeof (float) + 2 * sizeof (int)) *
                        (blocksize + SEG_SCAN_PACK_NUM);

        // 如果扫描所需要的线程的 grid 尺寸大于 1，就需要进行加回操作，就需要申
        // 请存放中间结果的数组。
        if (gridsize > 1) {
            // 需要将每块处理的最后一个元素取出，最后一个线程块不进行处理。
            blocksumsize = gridsize - 1;

            // 为存放中间结果的 3 个数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blockmaxdistDev,
                                   blocksumsize * sizeof(float));
            // 出错则释放申请的内存。
            if (cuerrcode != cudaSuccess) {
                // 释放之前申请的内存。
                FAIL_SEGMENTED_SCAN_FREE;
                return cuerrcode;
            }

            // 为存放区域标签的中间数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blocklabelDev,
                                   blocksumsize * sizeof(int));
            if (cuerrcode != cudaSuccess) {
                // 释放之前申请的内存。
                FAIL_SEGMENTED_SCAN_FREE;
                return cuerrcode;
            }

            // 为存放最大垂距位置索引的中间结果数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blockmaxdistidxDev,
                                   blocksumsize * sizeof(int));
            if (cuerrcode != cudaSuccess) {
                // 释放之前申请的内存。
                FAIL_SEGMENTED_SCAN_FREE;
                return cuerrcode;
            }
        }

        // 调用 Kernel 函数，完成实际的分段数组扫描。
        _segmentedScanMatrixKer<<<gridsize, blocksize, sharedmemsize>>>(
                inarrayDev, labelDev, indexDev, numelements,
                maxdistDev, maxdistidxDev,
                blockmaxdistDev, blocklabelDev, blockmaxdistidxDev);

        // 判断核函数是否出错。
        if (cudaGetLastError() != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return CUDA_ERROR;
        }
        break;

    // 其他方式情况下，直接返回非法数据错误。
    default:
        // 释放之前申请的内存。
        FAIL_SEGMENTED_SCAN_FREE;
        return INVALID_DATA;
    }

    if (gridsize > 1) {
        // 递归调用分段扫描函数。此时输入输出数组皆为中间结果数组。
        // 这里的递归调用不会调用多次，数组的规模是指数倍减小的。
        errcode = segmentedScan(blockmaxdistDev, blocklabelDev,
                                blockmaxdistidxDev, blockmaxdistDev,
                                blockmaxdistidxDev, blocksumsize,
                                false, false, false, false, false);
        if (errcode != NO_ERROR) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return errcode;
        }

        // 调用加回函数，将各块的扫描中间结果加回到输出数组。
        errcode = segmentedScanBack(maxdistDev, maxdistidxDev, labelDev,
                                    blockmaxdistDev, blocklabelDev, 
                                    blockmaxdistidxDev, numelements);
        if (errcode != NO_ERROR) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return errcode;
        }
    }

    // 如果 maxdist 数组在 Host 端，将结果拷贝到输出。
    if (hostmaxdist) {
        // 将结果从设备端内存拷贝到输出数组。
        cuerrcode = cudaMemcpy(maxdist, maxdistDev,
                               sizeof (float) * numelements,
                               cudaMemcpyDeviceToHost);
        // 出错则释放之前申请的内存。
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return cuerrcode;
        }
    }

    // 如果 maxdistidx 数组在 Host 端，将结果拷贝到输出。
    if (hostmaxdistidx) {
        // 将结果从设备端内存拷贝到输出数组。
        cuerrcode = cudaMemcpy(maxdistidx, maxdistidxDev,
                               sizeof (int) * numelements,
                               cudaMemcpyDeviceToHost);
        // 出错则释放之前申请的内存。
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SEGMENTED_SCAN_FREE;
            return cuerrcode;
        }
    }
        
    // 释放 Device 内存。需要判断输入输出参数是否在 host 端。
    FAIL_SEGMENTED_SCAN_FREE;

    // 处理完毕退出。
    return NO_ERROR;
}

// 取消前面的宏定义。
#undef FAIL_SEGMENTED_SCAN_FREE

