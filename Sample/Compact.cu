// Compact.cu
// 对数组进行压缩，只保留合法元素

#include "Compact.h"
#include <iostream>
using namespace std;

#define BLOCK_SIZE 1024

// Kernel 函数: _compactDataKer
// 每个线程处理一个元素，实现数组的压缩
__global__ void _compactDataKer(
        int *d_indices,          // exclusive 结果数组
        int *d_isValid,          // 合法性判断数组（只有 1 和 0）
        int *d_in,               // 输入数组
        int numElements,         // 数组元素个数
        int *d_out,              // 输出数组
        int *d_numValidElements  // 合法点的个数
);

// Kernel 函数: _compactDataKer（数组压缩）
__global__ void _compactDataKer(int *d_indices, int *d_isValid, int *d_in, 
                                int numElements, int *d_out,             
                                int *d_numValidElements)
{
    // 块内第一个线程处理合法元素计数
    if (threadIdx.x == 0)
        d_numValidElements[0] = d_isValid[numElements-1] + 
                                d_indices[numElements-1];

    // 线程处理的当前元素的全局索引
    unsigned int iglobal = blockIdx.x * blockDim.x + threadIdx.x;
    // 当前元素没有越界且合法时，向输出数组对应位置赋值
    if (iglobal < numElements && d_isValid[iglobal] > 0)
        d_out[d_indices[iglobal]] = d_in[iglobal];
}

// 宏：COMPACT_FREE
// 如果出错，就释放之前申请的内存。
#define COMPACT_FREE  do {              \
    if (allDev != NULL)                 \
        cudaFree(allDev);               \
    } while (0)                    

// Host 成员方法：compactDataGPU（int 型的数组压缩）
__host__ int Compact::compactDataGPU(int *indices, int *d_isValid, int *d_in, 
                                     int numElements, int *d_out,
                                     int *d_numValidElements)
{
     // 检查输入和输出以及合法判断数组是否为 NULL，如果为 NULL 直接报错返回。
    if (d_in == NULL || d_out == NULL || d_isValid == NULL || indices == NULL ||
        d_numValidElements == NULL)
        return NULL_POINTER;

    // 对数组长度必须加以判断控制。
    if (numElements < 0)
        return INVALID_DATA;

    // 定义运算类型为加法
    add_class<int> add;
    // 调用 scan exclusive 函数
    sa.scanArrayExclusive(d_isValid, indices, numElements, add);

    // 定义计算数组有效元素数量数组
    int *d_num = NULL;
    // 局部变量，错误码
    cudaError_t cuerrcode;

    // 定义设备端的输入输出数组,合法判定数组，scan-ex 数组指针以及设备端内存引用
    // 指针，当输入输出指针在 Host 端时，在设备端申请对应大小的数组。
    int *d_inDev = NULL;
    int *d_outDev = NULL;
    int *d_isValidDev = NULL; 
    int *indicesDev = NULL;
    int *allDev = NULL;

    // 这里 compact 实现只支持单个线程块的计算，这里的 gridsize 可以设置的大于
    //  1，从而让多个线程块都运行相同程序来测速。计算调用 Kernel 函数的线程块的
    // 尺寸和线程块的数量。
    int gridsize;
    int blocksize;

    // 在设备端统一申请内存
    cuerrcode = cudaMalloc((void **)&allDev,
	                       sizeof (int) * (numElements * 4 + 1));
    if (cuerrcode != cudaSuccess)
        return cuerrcode;

    // 确定各数组在设备内存上的地址。
    d_num = allDev;
    d_inDev = allDev + 1;
    d_isValidDev = d_inDev + numElements;
    indicesDev = d_isValidDev + numElements;
    d_outDev = indicesDev + numElements;

    // 为数量统计数组赋初值为 0
    cuerrcode = cudaMemset(d_num, 0, sizeof (int));
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;

    // 将输入数组拷贝到设备端内存。
    cuerrcode = cudaMemcpy(d_inDev, d_in, sizeof (int) * numElements, 
                           cudaMemcpyHostToDevice);
    if (cuerrcode != cudaSuccess) {
        COMPACT_FREE;
        return cuerrcode;
    }

    // 将判定数组拷贝到设备端内存。
    cuerrcode = cudaMemcpy(d_isValidDev, d_isValid, sizeof (int) * numElements, 
                           cudaMemcpyHostToDevice);
    if (cuerrcode != cudaSuccess) {
        COMPACT_FREE;
        return cuerrcode;
    }

    // 将 scan-ex 数组拷贝到设备端内存。
    cuerrcode = cudaMemcpy(indicesDev, indices, sizeof (int) * numElements, 
                           cudaMemcpyHostToDevice);
    if (cuerrcode != cudaSuccess) {
        COMPACT_FREE;    
        return cuerrcode;
    }

    // 为输出数组赋初值为 0
    cuerrcode = cudaMemset(d_outDev, 0, sizeof (int) * numElements);
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    // 简单版本每个线程处理一个元素。
    blocksize = BLOCK_SIZE;
    // 计算线程块大小和共享内存长度。
    gridsize = max(1, (numElements + blocksize - 1) / blocksize);

    // 调用 Kernel 函数，完成实际的数组压缩。
    _compactDataKer<<<gridsize, blocksize>>>(
            indicesDev, d_isValidDev, d_inDev, numElements, d_outDev, d_num);

    // 判断是否出错。
    if (cudaGetLastError() != cudaSuccess) {
        // 释放之前申请的内存。
        COMPACT_FREE;
        return CUDA_ERROR;
    }

    // 将结果从设备端内存拷贝到输出数组。
    cuerrcode = cudaMemcpy(d_out, d_outDev, sizeof (int) * numElements,
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        // 释放之前申请的内存。
        COMPACT_FREE;
        return cuerrcode;
    }

    // 将计数结果从设备端内存拷贝到输出数组。
    cuerrcode = cudaMemcpy(d_numValidElements, d_num, sizeof (int),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        // 释放之前申请的内存。
        COMPACT_FREE;
        return cuerrcode;
    }

    // 释放内存。
    cudaFree(allDev);

    // 处理完毕退出。
    return NO_ERROR;
} 

// 取消前面的宏定义。
#undef COMPACT_FREE
