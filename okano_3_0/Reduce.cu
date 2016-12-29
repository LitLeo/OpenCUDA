#include "ErrorCode.h"
#include "Reduce.h"
#include <iostream>
using namespace std;

// 每个线程格包含的线程数
#define THREADSPERBLOCK  256

// Device 子程序：_reduceAddDev
// 将数组内部的值归约求和，将和存储到该数组的下标 0 处
static __device__ void
_reduceAddDev(
        float *input,   // 输入数组，应当归约运算的数组
        int inputL,     // 数组长度
        int cacheIndex  // 当前的 cache 索引
);

// Kernel 核函数：_addKer（归约加法）
// 对输入的数组，进行归约求和
static __global__ void
_addKer(
        float *input,  // 输入数组
        int inputL,    // 数组长度
        float *output  // 输出
);

// Device 子程序：_reduceAddDev
static __device__ void _reduceAddDev(float *input, int inputL, int cacheIndex)
{
    int i = inputL / 2;
    while (i != 0) {
        // 当共享内存索引小于当前的归约计数 i 的值时，可以进行归约，表明当前共享
        // 内存中有 2 * i 个有效数据等待归约
        if (cacheIndex < i)
            input[cacheIndex] += input[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
}

// Kernel 核函数：_addKer（归约加法）
static __global__ void _addKer(float *input, int inputL, float *output)
{

    // 共享内存 cache 用于存储归约运算的数组
    __shared__ float cache[THREADSPERBLOCK];

    // 计算线程索引，线程 threadIdx.x 处理一个数据的下标，同时处理的数据为
    // blockDim.x 个数据，每两个数据间相隔 blockDim.x * guidDim.x 个数据
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 将共享内存中的每一个元素置 0。
    cache[threadIdx.x] = 0.0f;
    // 共享内存中的索引
    int cacheIndex = threadIdx.x;
    
    // 设置 cache 中相应位置上的值
    while (tid < inputL) {
        cache[cacheIndex] += input[tid];
        tid += blockDim.x * gridDim.x;
    }

    // 调用 Device 子程序对 cache 内数据进行归约求和
    _reduceAddDev(cache, blockDim.x, cacheIndex);

    // 对线程块中的线程进行同步
    __syncthreads();

    // 归约运算求和，将所有共享内存中的数据同时进行规约求和。
    // 初始化归约计数 i 为共享内存长度的一半
    if (cacheIndex != 0)
        return;
    // 利用原子运算求和， 用 0 号线程规约
    if (cacheIndex == 0)
        atomicAdd(output, cache[0]);
}

// 成员方法：ReduceAdd（进行加法归约运算）
__host__ int Reduce::reduceAdd(float *input, int inputL, float *output)
{ 
    // 检查输入数组和输出指针是否为空
    if (input == NULL || output == NULL)
        return NULL_POINTER;

    // 检查数组长度是否有效
    if (inputL < 1)
        return INVALID_DATA;

    // 获取线程块数
    dim3 gridsize, blocksize;
    blocksize.x = THREADSPERBLOCK;
    gridsize.x = (inputL + THREADSPERBLOCK - 1) / blocksize.x;

    // 设备端数组，存储输入值
    float *dev_input = NULL;
    float *dev_output = NULL;

    // 在 GPU 上分配内存，在 GPU 上申请一块连续的内存 temp，通过指针将内存分配给
    // dev_input 和 dev_output，其中 dev_input 指向长为 inputL 的地址的首地址，
    // dev_output 指向一个内存，这样避免了反复申请内存的耗时
    float *temp = NULL;
    cudaMalloc((void**)&temp, (inputL + 1) * sizeof(float));
    dev_input = temp;
    dev_output = temp + inputL;

    // 将输入数组 input 复制到 GPU
    cudaMemcpy(dev_input, input, inputL * sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(dev_output, output, sizeof(float), cudaMemcpyHostToDevice);

    // 运行 Kernel 核函数
    _addKer<<<gridsize, blocksize>>>(dev_input, inputL, dev_output);

    // 将结果拷贝到 CPU 端
    cudaMemcpy(output, dev_output, sizeof(float), cudaMemcpyDeviceToHost); 

    // 获取错误信息
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(temp);
        return CUDA_ERROR;
    }

    // 释放 GPU 上的内存
    cudaFree(temp);

    // 处理完毕，退出
    return NO_ERROR;
}

