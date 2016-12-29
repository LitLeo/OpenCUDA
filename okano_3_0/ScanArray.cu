// ScanArray.cu
// 找出图像中给定点集的包围矩形

#include "ScanArray.h"
#include "OperationFunctor.h"
#include <iostream>
#include <stdio.h>
using namespace std;

#define BLOCK_SIZE 1024

// 宏：NUM_BANKS
// 定义了bank的数量。
#define NUM_BANKS     16

// 宏：LOG_NUM_BANKS
// 定义了bank的对数值。
#define LOG_NUM_BANKS  4

// Kernel 函数: _scanNaiveKer（数组扫描的简单版本）
// 简单版本的 scan 实现，每个线程处理一个元素。运行 log(n) 步，
// 加 n * (log(n) - 1) 次。需要 2 * n 长度的贡献内存。
template < class T, class Operation >
__global__ void        // Kernel 函数无返回值
_scanNaiveKer(
        T *outarray,   // 输入数组
        T *inarray,    // 输出数组
        T *blocksum,   // 扫描的中间结果数组，用于存放每块扫描结果的最后一
                       // 个值，不处理最后一个线程块。少于一块时不处理。
        int n,         // 元素个数，数组长度
        Operation op,  // 运算符类型
        bool backward  // 判断是否为后序扫描
);

// Kernel 函数: _scanWorkEfficientKer（数组扫描的效率版本）
// 效率版本的 scan 实现，每个线程处理两个元素。复杂度是 O(log(n))，需要加法的次
// 数为 O(n)。共享内存长度为 n，使用 balanced tree 算法。
// 来源一为：Blelloch，1990 "Prefix Sums and Their Applications"。
// http://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
// 来源二为：Prins and Chatterjee PRAM course notes：
// https://www.cs.unc.edu/~prins/Classes/633/Handouts/pram.pdf
template < class T, class Operation >
__global__ void        // Kernel 函数无返回值
_scanWorkEfficientKer(
        T *outarray,   // 输入数组
        T *inarray,    // 输出数组
        T *blocksum,   // 扫描的中间结果数组，用于存放每块扫描结果的最后一
                       // 个值，不处理最后一个线程块。少于一块时不处理。
        int n,         // 元素个数，数组长度
        Operation op,  // 运算符类型
        bool backward  // 判断是否为后序扫描
);

// Kernel 函数: _scanOptKer（数组扫描的优化版本）
// 优化版本的 scan 实现，每个线程处理两个原色。复杂度是 O(log(n))，需要加法的次
// 数为 O(n)。共享内存不检查冲突。
// 使用 balanced tree 算法。
// 来源一为：Blelloch，1990 "Prefix Sums and Their Applications"。
// http://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
// 来源二为：Prins and Chatterjee PRAM course notes：
// https://www.cs.unc.edu/~prins/Classes/633/Handouts/pram.pdf
template < class T, class Operation >
__global__ void        // Kernel 函数无返回值
_scanOptKer(
        T *outarray,   // 输入数组
        T *inarray,    // 输出数组
        T *blocksum,   // 扫描的中间结果数组，用于存放每块扫描结果的最后一
                       // 个值，不处理最后一个线程块。少于一块时不处理。
        int n,         // 元素个数，数组长度
        Operation op,  // 运算类型
        bool backward  // 判断是否为后序扫描
);

// Kernel 函数: _scanBetterKer（数组扫描的较优版本）
// 优化版本的 scan 实现，每个线程处理两个原色。复杂度是 O(log(n))，需要加法的次
// 数为 O(n)。共享内存的长度为了避免冲突，设计为 n + n / NUM_BANKS。
// 使用 balanced tree 算法。
// 来源一为：Blelloch，1990 "Prefix Sums and Their Applications"。
// http://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
// 来源二为：Prins and Chatterjee PRAM course notes：
// https://www.cs.unc.edu/~prins/Classes/633/Handouts/pram.pdf
template < class T, class Operation >
__global__ void        // Kernel 函数无返回值
_scanBetterKer(
        T *outarray,   // 输入数组
        T *inarray,    // 输出数组
        T *blocksum,   // 扫描的中间结果数组，用于存放每块扫描结果的最后一
                       // 个值，不处理最后一个线程块。少于一块时不处理。
        int n,         // 元素个数，数组长度
        Operation op,  // 运算类型
        bool backward  // 判断是否为后序扫描
);

// 函数：scanComputeGold（CPU 端的 inclusive 类型数组计算）
// 对一个数组进行扫描，对所有元素进行某操作的遍历，操作包含自身。
template < class T, class Operation >
__host__ int                     // 返回值：函数是否正确执行，若函数正确
                                 // 执行，返回 NO_ERROR。
scanComputeGold(
        T *inarray,              // 输入数组
        T *reference,            // 输出数组
        const unsigned int len,  // 数组的长度，处理元素的个数。
        Operation op,            // 运算类型
        bool backward            // 判断是否为后序扫描
);

// Kernel 函数: _addBackKer（中间结果数组加回操作）
// 对一个数组进行初始扫描后，将中间结果的小数组（原扫描数组每段最后一个元素）加
// 回到原扫描数组。
template < class T, class Operation >
__global__ void            // Kernel 函数无返回值
_addBackKer(
        T *array,          // 初始扫描后的数组。
        T *lastelemarray,  // 中间结果数组，原扫描数组每段的最后一个元素提
                           // 取出来即为中间结果数组。
        int n,             // 元素个数，数组长度
        int packnum,       // 扫描核函数每块计算能力，即核函数每块的处理长
                           // 度与线程块大小的比值。
        Operation op,      // 运算类型
        bool backward      // 判断是否为后序扫描
);

// Kernel 函数: _scanNaiveKer（数组扫描的简单版本）
template < class T, class Operation >
__global__ void _scanNaiveKer(T *outarray, T *inarray, T *blocksum, int n, 
                              Operation op, bool backward)
{
    // 声明共享内存。
    extern __shared__ unsigned char sharedmemo[];
    // 转化为模板类型的共享内存。
    T *sharedmem = (T *)sharedmemo;

    // 数组索引（块内索引为 threadIdx.x）。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 定义共享内存中计算位置的两个变量。
    // 共享内存大小是两倍的输入长度，所以通过 pout 和 pin 指针来控制计算位置。
    T *pout = sharedmem;
    T *pin = sharedmem + blockDim.x;

    // 将需要计算的值从输入加载到共享内存上。大于数组长度的位置置为 0。
    if (idx < n)
        pout[threadIdx.x] = inarray[idx];
    else
        pout[threadIdx.x] = op.identity();

    // 扫描过程，通过偏移量的控制，在两倍输入长度的共享内存上进行切换计算。
    // 每次循环偏移量扩大 2 倍，这样能够实现不断的层次累加。最终实现 scan 的处
    // 理效果。
    if (!backward) {
        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            // 共享内存大小是两倍的输入长度，所以通过 pout 和 pin 指针交换来换位
            // 置进行处理，即计算值不覆盖原值。
            T *ptemp;
            ptemp = pout;
            pout = pin;
            pin  = ptemp;
            __syncthreads();

            // 将所有当前的 scan 计算值，从共享内存的一侧复制到另一侧
            // 一侧指共享内存采用两倍的输入数组的长度，即 double buffer。
            pout[threadIdx.x] = pin[threadIdx.x];

            // 如果线程索引大于偏移，那么要计算当前位置和当前位置减去偏移量处的
            // 加和
            if (threadIdx.x >= offset)
                pout[threadIdx.x] = op(pout[threadIdx.x], 
                                       pin[threadIdx.x - offset]);
        }

        // 进行块内同步。
        __syncthreads();

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (idx >= n)
            return;
        // 将结果从共享内存写入到输出。
        outarray[idx] = pout[threadIdx.x];

        // 如果中间结果数组为空，不进行处理直接返回。
        if (blocksum == NULL)
            return;
        // 每块的最后一个线程，将每段处理的最后一个元素写入中间结果数组。最后一
        // 个线程块不进行处理。
        if (threadIdx.x == blockDim.x - 1 &&
            blockIdx.x < gridDim.x - 1) {
            blocksum[blockIdx.x] = pout[threadIdx.x];
        }
    } else {
        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            // 共享内存大小是两倍的输入长度，所以通过 pout 和 pin 指针交换来换位
            // 置进行处理，即计算值不覆盖原值。
            T *ptemp;
            ptemp = pout;
            pout = pin;
            pin  = ptemp;
            __syncthreads();

            // 将所有当前的 scan 计算值，从共享内存的一侧复制到另一侧
            // 一侧指共享内存采用两倍的输入数组的长度，即 double buffer。
            pout[threadIdx.x] = pin[threadIdx.x];

            // 如果线程索引加上偏移量小于块长，那么要计算当前位置和当前位置加上
            // 偏移量处的加和
            if (threadIdx.x + offset < blockDim.x)
                pout[threadIdx.x] = op(pin[threadIdx.x], 
                                       pin[threadIdx.x + offset]);
        }

        // 进行块内同步。
        __syncthreads();

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (idx >= n)
            return;
        // 将结果从共享内存写入到输出。
        outarray[idx] = pout[threadIdx.x];

        // 如果中间结果数组为空，不进行处理直接返回。
        if (blocksum == NULL)
            return;
        // 每块的第一个线程，将每段处理的第一个元素写入中间结果数组。第一个线
        // 程块不进行处理。
        if (threadIdx.x == 0 && blockIdx.x > 0) {
            blocksum[blockIdx.x - 1] = pout[threadIdx.x];
        }
    }
}

// Kernel 函数: _scanWorkEfficientKer（数组扫描的效率版本）
template < class T, class Operation >
__global__ void _scanWorkEfficientKer(T *outarray, T *inarray,
                                      T *blocksum, int n, Operation op,
                                      bool backward)
{
    // 声明共享内存。
    extern __shared__ unsigned char sharedmemo[];
    // 转化为模板类型的共享内存。
    T *sharedmem = (T *)sharedmemo;

    // 定义块内索引。 
    int baseidx = 2 * blockIdx.x * blockDim.x;
    int inidx = 2 * threadIdx.x;
    int idx = baseidx + inidx;

    // 定义偏移量 offset。
    int offset = 1;

    // 定义核函数每块的处理数组长度。
    int length = blockDim.x * 2;

    // 将需要计算的值从输入加载到共享内存上。每个线程处理两个元素（相邻元素）
    sharedmem[inidx] = (idx < n) ? inarray[idx] : op.identity();
    sharedmem[inidx + 1] = (idx + 1 < n) ? inarray[idx + 1] : op.identity();

    // scan 的累加过程，自底向上进行累加操作。
    if (!backward) {
        for (int d = blockDim.x; d > 0; d >>= 1) {
            // 进行块内同步。
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行累加。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ai = offset * (inidx + 1) - 1;
                int bi = offset * (inidx + 2) - 1;

                // 累加。通过这样的过程使得最终最后一位的值是之前所有值 scan 操
                // 作的结果。
                sharedmem[bi] = op(sharedmem[bi], sharedmem[ai]);
            }

            // 偏移量每次扩大 2 倍。
            offset *= 2;
        }

        // 配合自顶向下的回扫过程，清除最后一个位置上的值。
        if (threadIdx.x == 0) {
            // 根据运算类型不同，最后一个位置赋为不同的单位元。
            sharedmem[length - 1] = op.identity();
        }   

        // 自顶向下回扫，这样能够使每一位上算出需要的 scan 值。
        // 回扫即把上一步的计算结果进行累加。
        for (int d = 1; d < length; d *= 2) {
            // 回扫过程中，每次偏移量缩小2倍。
            offset >>= 1;

            // 进行块内同步。
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行回扫累加的过程。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ai = offset * (inidx + 1) - 1;
                int bi = offset * (inidx + 2) - 1;

                // 将 ai 位置处的值拷贝出来加到 bi 处，ai 的新值为 bi 原值。
                T t = sharedmem[ai];
                sharedmem[ai] = sharedmem[bi];
                sharedmem[bi] = op(sharedmem[bi], t);
            }
        }
        // 进行块内同步。
        __syncthreads();

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (idx >= n)
            return;
       
        // 将结果从共享内存写入到输出。
        outarray[idx] = sharedmem[inidx + 1];

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (idx + 1 >= n)
            return;
        // 判断是否为当前块处理数组的最后一个元素
        if ((inidx + 1) == (2 * blockDim.x - 1))
            // 是最后一个元素的话，需要与对应输入数组位置上的元素进行运算
            outarray[idx + 1] = op(sharedmem[inidx + 1], inarray[idx + 1]);
        else
            // 每个线程处理的下一个元素，将结果从共享内存写入到输出。
            outarray[idx + 1] = sharedmem[inidx + 2];

        // 如果中间结果数组为空，不进行处理直接返回。
        if (blocksum == NULL)
            return;
        // 每块的最后一个线程，将每段处理的最后一个元素写入中间结果数组。最后一
        // 个线程块不进行处理。
        if (threadIdx.x == blockDim.x - 1 &&
            blockIdx.x < gridDim.x - 1) {
            blocksum[blockIdx.x] = outarray[idx + 1];
        }
    } else {
        for (int d = blockDim.x; d > 0; d >>= 1) {
            // 进行块内同步。
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行累加。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ai = offset * inidx;
                int bi = offset * (inidx + 1);

                // 累加。通过这样的过程使得最终第一位的值是之前所有值 scan 操
                // 作的结果。
                sharedmem[ai] = op(sharedmem[bi], sharedmem[ai]);
            }

            // 偏移量每次扩大 2 倍。
            offset *= 2;
        }

        // 配合自顶向下的回扫过程，清除第一个位置上的值。
        if (threadIdx.x == 0) {
            // 根据运算类型不同，第一个位置赋为不同的单位元。
            sharedmem[0] = op.identity();
        }   

        // 自顶向下回扫，这样能够使每一位上算出需要的 scan 值。
        // 回扫即把上一步的计算结果进行累加。
        for (int d = 1; d < length; d *= 2) {
            // 回扫过程中，每次偏移量缩小2倍。
            offset >>= 1;

            // 进行块内同步。
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行回扫累加的过程。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ai = offset * inidx;
                int bi = offset * (inidx + 1);

                // 将 bi 位置处的值拷贝出来加到 ai 处，bi 的新值为 ai 原值。
                T t = sharedmem[bi];
                sharedmem[bi] = sharedmem[ai];
                sharedmem[ai] = op(sharedmem[ai], t);
            }
        }

        // 进行块内同步。
        __syncthreads();

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (idx >= n)
            return;
        // 判断是否为当前块处理数组的第一个元素
        if (inidx == 0)
            // 是第一个元素的话，需要与对应输入数组位置上的元素进行运算
            outarray[idx] = op(sharedmem[inidx], inarray[idx]);
        else
            // 将结果从共享内存写入到输出。
            outarray[idx] = sharedmem[inidx - 1];

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (idx + 1 < n) {
            // 每个线程处理的下一个元素，将结果从共享内存写入到输出。
            outarray[idx + 1] = sharedmem[inidx];
        }
        // 如果中间结果数组为空，不进行处理直接返回。
        if (blocksum == NULL)
            return;
        // 每块的第一个线程，将每段处理的第一个元素写入中间结果数组。第一个线程
        // 块不进行处理。
        if (threadIdx.x == 0 &&
            blockIdx.x > 0) {
            blocksum[blockIdx.x - 1] = outarray[idx];
        }
    }
}

// Kernel 函数: _scanOptKer（数组扫描的优化版本）
template < class T, class Operation >
__global__ void _scanOptKer(T *outarray, T *inarray,
                            T *blocksum, int n, Operation op, bool backward)
{
    // 声明共享内存。
    extern __shared__ unsigned char sharedmemo[];
    // 转化为模板类型的共享内存。
    T *sharedmem = (T *)sharedmemo;

    // 定义块内索引。
    int baseidx = 2 * blockIdx.x * blockDim.x;

    // 定义核函数每块的处理数组长度。
    int length = blockDim.x * 2;

    // 定义计算位置的索引（块内）。
    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    
    // 定义数组索引（块外）。
    int aindex = baseidx + ai;
    int bindex = baseidx + bi;

    // 将需要计算的值从输入加载到共享内存上。每个线程处理两个元素。
    sharedmem[ai] = (aindex < n) ? inarray[aindex] : op.identity(); 
    sharedmem[bi] = (bindex < n) ? inarray[bindex] : op.identity(); 
        
    // 定义偏移值 offset。
    int offset = 1;

    if (!backward) {
        // scan 的累加过程，自底向上进行累加操作。
        for (int d = blockDim.x; d > 0; d >>= 1) {
            // 进行块内同步。
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行累加。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ai = offset * (2 * threadIdx.x + 1) - 1;
                int bi = offset * (2 * threadIdx.x + 2) - 1;

                // 累加。通过这样的过程使得最终最后一位的值是之前所有值 scan 操
                // 作的结果。
                sharedmem[bi] = op(sharedmem[bi], sharedmem[ai]);
            }

            // 偏移量每次扩大 2 倍。
            offset *= 2;
        }

        // 配合自顶向下的回扫过程，清除最后一个位置上的值。
        if (threadIdx.x == 0) {
            int index = length - 1;
            // 根据运算符类型，数组最后一个位置设为不同的单位元
            sharedmem[index] = op.identity();
        }   

        // 自顶向下回扫，这样能够使每一位上算出需要的 scan 值。
        // 回扫即把上一步的计算结果进行累加。
        for (int d = 1; d < length; d *= 2) {
            // 回扫过程中，每次偏移量缩小2倍。
            offset /= 2;

            // 进行块内同步。   
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行回扫累加的过程。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ai = offset * (2 * threadIdx.x + 1) - 1;
                int bi = offset * (2 * threadIdx.x + 2) - 1;

                // 将 ai 位置处的值拷贝出来加到 bi 处，ai 的新值为 bi 原值。
                T t = sharedmem[ai];
                sharedmem[ai] = sharedmem[bi];
                sharedmem[bi] = op(sharedmem[bi], t);
            }
        }

        // 进行块内同步。
        __syncthreads();

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (aindex >= n)
            return;
        // 将结果从共享内存写入到输出。
        outarray[aindex] = sharedmem[ai + 1];

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (bindex >= n)
            return;
        // 判断是否为当前块处理数组的最后一个元素
        if (bi == (2 * blockDim.x - 1))
            // 是最后一个元素的话，需要与对应输入数组位置上的元素进行运算
            outarray[bindex] = op(sharedmem[bi], inarray[bindex]);
        else
            // 每个线程处理的下一个元素，将结果从共享内存写入到输出。
            outarray[bindex] = sharedmem[bi + 1];

        // 如果中间结果数组为空，不进行处理直接返回。
        if (blocksum == NULL)
            return;
        // 每块的最后一个线程，将每段处理的最后一个元素写入中间结果数组。最后一个线
        // 程块不进行处理。
        if (threadIdx.x == blockDim.x - 1 &&
            blockIdx.x < gridDim.x - 1) {
            blocksum[blockIdx.x] = outarray[bindex];
        }
    } else {
        // scan 的累加过程，自底向上进行累加操作。
        for (int d = blockDim.x; d > 0; d >>= 1) {
            // 进行块内同步。
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行累加。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ai = offset * 2 * threadIdx.x;
                int bi = offset * (2 * threadIdx.x + 1);

                // 累加。通过这样的过程使得最终第一位的值是之前所有值 scan 操
                // 作的结果。
                sharedmem[ai] = op(sharedmem[bi], sharedmem[ai]);
            }

            // 偏移量每次扩大 2 倍。
            offset *= 2;
        }

        // 配合自顶向下的回扫过程，清除第一个位置上的值。
        if (threadIdx.x == 0) {
            int index = 0;
            // 根据运算符类型，数组第一个位置设为不同的单位元
            sharedmem[index] = op.identity();
        }   

        // 自顶向下回扫，这样能够使每一位上算出需要的 scan 值。
        // 回扫即把上一步的计算结果进行累加。
        for (int d = 1; d < length; d *= 2) {
            // 回扫过程中，每次偏移量缩小2倍。
            offset /= 2;

            // 进行块内同步。   
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行回扫累加的过程。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ai = offset * 2 * threadIdx.x;
                int bi = offset * (2 * threadIdx.x + 1);

                // 将 bi 位置处的值拷贝出来加到 ai 处，bi 的新值为 ai 原值。
                T t = sharedmem[bi];
                sharedmem[bi] = sharedmem[ai];
                sharedmem[ai] = op(sharedmem[ai], t);
            }
        }

        // 进行块内同步。
        __syncthreads();

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (aindex >= n)
            return;
        // 判断是否为当前块处理数组的第一个元素
        if (ai == 0)
            // 是第一个元素的话，需要与对应输入数组位置上的元素进行运算
            outarray[aindex] = op(sharedmem[ai], inarray[aindex]);
        else
            // 将结果从共享内存写入到输出。
            outarray[aindex] = sharedmem[ai - 1];

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (bindex < n) {
            // 每个线程处理的下一个元素，将结果从共享内存写入到输出。
            outarray[bindex] = sharedmem[bi - 1];
        }

        // 如果中间结果数组为空，不进行处理直接返回。
        if (blocksum == NULL)
            return;
        // 每块的第一个线程，将每段处理的第一个元素写入中间结果数组。第一个线
        // 程块不进行处理。
        if (threadIdx.x == 0 &&
            blockIdx.x > 0) {
            blocksum[blockIdx.x - 1] = outarray[aindex];
        }
    }
}

// 宏：CONFLICT_FREE_OFFSET
// 定义此是为了更严格的避免 bank conflicts，即使在树的低层上。
#ifdef ZERO_BANK_CONFLICTS
#  define CONFLICT_FREE_OFFSET(index)              \
          (((index) >> LOG_NUM_BANKS) + ((index) >> (2 * LOG_NUM_BANKS)))
#else
#  define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

// Kernel 函数: _scanBetterKer（数组扫描的Better版本）
template < class T, class Operation >
__global__ void _scanBetterKer(T *outarray, T *inarray,
                               T *blocksum, int n, Operation op, bool backward)
{
    // 声明共享内存。
    extern __shared__ unsigned char sharedmemo[];
    // 转化为模板类型的共享内存。
    T *sharedmem = (T *)sharedmemo;

    // 本地变量，基索引。
    int baseidx = 2 * blockIdx.x * blockDim.x;
    int idx = threadIdx.x + blockDim.x;

    // 定义核函数每块的处理数组长度。
    int length = blockDim.x * 2;

    // 定义计算位置的索引（块内，加上 bankOffset）。
    int ai = threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x);
    int bi = idx + CONFLICT_FREE_OFFSET(idx);
    
    // 定义数组索引（块外）。
    int aindex = baseidx + threadIdx.x;
    int bindex = aindex + blockDim.x;

    // 将需要计算的值从输入加载到共享内存上。每个线程处理两个元素。
    sharedmem[ai] = (aindex < n) ? inarray[aindex] : op.identity(); 
    sharedmem[bi] = (bindex < n) ? inarray[bindex] : op.identity();

    // 定义偏移值 offset。
    int offset = 1;

    if (!backward) {
        // scan 的累加过程，自底向上进行累加操作。
        for (int d = blockDim.x; d > 0; d >>= 1) {
            // 进行块内同步。
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行累加。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ci = offset * (2 * threadIdx.x + 1) - 1;
                int di = offset * (2 * threadIdx.x + 2) - 1;

                // 避免 bank conflicts，修改计算位置索引。
                ci += CONFLICT_FREE_OFFSET(ci);
                di += CONFLICT_FREE_OFFSET(di);

                // 累加。通过这样的过程使得最终最后一位的值是之前所有值 scan 操
                // 作的结果。
                sharedmem[di] = op(sharedmem[di], sharedmem[ci]);
            }

            // 偏移量每次扩大 2 倍。
            offset *= 2;
        }

        // 配合自顶向下的回扫过程，清除最后一个位置上的值。
        if (threadIdx.x == 0) {
            int index = length - 1;

            // 避免 bank conflicts，重新计算索引。
            index += CONFLICT_FREE_OFFSET(index);

            // 根据运算符类型，数组最后一个位置设为不同的单位元。
            sharedmem[index] = op.identity();
        }   

        // 自顶向下回扫，这样能够使每一位上算出需要的 scan 值。
        // 回扫即把上一步的计算结果进行累加。
        for (int d = 1; d < length; d *= 2) {
            // 回扫过程中，每次偏移量缩小2倍。
            offset /= 2;

            // 进行块内同步。   
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行回扫累加的过程。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ci = offset * (2 * threadIdx.x + 1) - 1;
                int di = offset * (2 * threadIdx.x + 2) - 1;

                // 避免 bank conflicts，重新计算索引。
                ci += CONFLICT_FREE_OFFSET(ci);
                di += CONFLICT_FREE_OFFSET(di);

                // 将 ai 位置处的值拷贝出来加到 bi 处，ai 的新值为 bi 原值。
                T t = sharedmem[ci];
                sharedmem[ci] = sharedmem[di];
                sharedmem[di] = op(sharedmem[di], t);
            }
        }

        // 进行块内同步。
        __syncthreads();

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (aindex >= n)
            return;
        // 将结果从共享内存写入到输出。
        outarray[aindex] = op(sharedmem[ai], inarray[aindex]);

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (bindex >= n)
            return;
        // 每个线程处理的下一个元素，将结果从共享内存写入到输出。
        outarray[bindex] = op(sharedmem[bi], inarray[bindex]);

        // 如果中间结果数组为空，不进行处理直接返回。
        if (blocksum == NULL)
            return;
        // 每块的最后一个线程，将每段处理的最后一个元素写入中间结果数组。最后一个线
        // 程块不进行处理。
        if (threadIdx.x == blockDim.x - 1 &&
            blockIdx.x < gridDim.x - 1) {
            blocksum[blockIdx.x] = outarray[bindex];
        }
    } else {
         // scan 的累加过程，自底向上进行累加操作。
        for (int d = blockDim.x; d > 0; d >>= 1) {
            // 进行块内同步。
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行累加。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ci = offset * 2 * threadIdx.x;
                int di = offset * (2 * threadIdx.x + 1);

                // 避免 bank conflicts，修改计算位置索引。
                ci += CONFLICT_FREE_OFFSET(ci);
                di += CONFLICT_FREE_OFFSET(di);

                // 累加。通过这样的过程使得最终第一位的值是之前所有值 scan 操
                // 作的结果。
                sharedmem[ci] = op(sharedmem[di], sharedmem[ci]);
            }

            // 偏移量每次扩大 2 倍。
            offset *= 2;
        }

        // 配合自顶向下的回扫过程，清除第一个位置上的值。
        if (threadIdx.x == 0) {
            int index = 0;

            // 避免 bank conflicts，重新计算索引。
            index += CONFLICT_FREE_OFFSET(index);

            // 根据运算符类型，数组第一个位置设为不同的单位元。
            sharedmem[index] = op.identity();
        }  

        // 自顶向下回扫，这样能够使每一位上算出需要的 scan 值。
        // 回扫即把上一步的计算结果进行累加。
        for (int d = 1; d < length; d *= 2) {
            // 回扫过程中，每次偏移量缩小2倍。
            offset /= 2;

            // 进行块内同步。   
            __syncthreads();

            // 当线程索引小于处理的范围 d，进行回扫累加的过程。
            if (threadIdx.x < d) {
                // 计算处理位置的索引。
                int ci = offset * 2 * threadIdx.x;
                int di = offset * (2 * threadIdx.x + 1);

                // 避免 bank conflicts，重新计算索引。
                ci += CONFLICT_FREE_OFFSET(ci);
                di += CONFLICT_FREE_OFFSET(di);

                // 将 di 位置处的值拷贝出来加到 ci 处，di 的新值为 ci 原值。
                T t = sharedmem[di];
                sharedmem[di] = sharedmem[ci];
                sharedmem[ci] = op(sharedmem[ci], t);
            }
        }

        // 进行块内同步。
        __syncthreads();

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (aindex >= n)
            return;
        // 处理的第一个元素，需要与对应输入数组位置上的元素进行运算
        outarray[aindex] = op(sharedmem[ai], inarray[aindex]);

        // 超出数组长度 n 的值不进行写入，直接返回。
        if (bindex < n) {
            // 每个线程处理的下一个元素，将结果从共享内存写入到输出。
            outarray[bindex] = op(sharedmem[bi], inarray[bindex]);
        }

        // 如果中间结果数组为空，不进行处理直接返回。
        if (blocksum == NULL)
            return;
        
        // 每块的第一个线程，将每段处理的第一个元素写入中间结果数组。第一个线
        // 程块不进行处理。     
        if (threadIdx.x == 0 &&
            blockIdx.x > 0) {
            blocksum[blockIdx.x - 1] = outarray[baseidx];
        }
        
    }
}

// 函数：scanComputeGold（CPU 端的 inclusive 类型数组计算）
template < class T, class Operation >
__host__ int scanComputeGold(T *inarray, T *reference,             
                             const unsigned int len, Operation op,
                             bool backward)
{
    // 计数器变量
    int i;

    if (!backward) {
        // 初始化第一个输出元素为 inarray[0]
        reference[0] = inarray[0];
        for (i = 1; i < len; ++i) {
            // 前序迭代累加计算
            reference[i] = op(inarray[i], reference[i-1]);
        }
    } else {
        // 初始化最后一个输出元素为 inarray[len - 1]
        reference[len - 1] = inarray[len - 1];
        for (i = len - 2; i >= 0; i--) {
            // 后序迭代累加计算
            reference[i] = op(inarray[i], reference[i+1]);
        }
    }

    // 处理完毕退出。
    return NO_ERROR;
}

// Kernel 函数: _addBackKer（中间结果数组加回操作）
template < class T, class Operation >
__global__ void _addBackKer(T *array, T *lastelemarray, int n,
                            int packnum, Operation op, bool backward)
{
    // 声明共享内存。用来存放中间结果小数组中的元素，也就是原数组的每段最后
    // 一个或第一个元素。
    __shared__ T lastelement[1];

    if (!backward) {
        // 用每块的第一个线程来读取每块前一块的最后一个元素，从中间结果数组中读
        // 取。
        if (threadIdx.x == 0)
            lastelement[0] = lastelemarray[blockIdx.x];

        // 计算需要进行块间累加位置索引（块外的数组索引）。
        unsigned int idx = (blockIdx.x + 1) * (blockDim.x * packnum) + 
                           threadIdx.x;

        // 块内同步。
        __syncthreads();
    
        // 每个线程处理两个元素，将中间结果数组中的值加回到原数组。
        for (int i = 0; i < packnum; i++) {
            // 如果索引大于处理数组长度，则退出。
            if (idx >= n)
                break;

            // 将中间结果加回。
            array[idx] = op(array[idx],lastelement[0]);

            // 计算每个线程处理的下一个元素的索引值。
            idx += blockDim.x;
        }
    } else {
        // 用每块的第一个线程来读取每块后一块的第一个元素，从中间结果数组中读
        // 取。
        if (threadIdx.x == 0) {
            lastelement[0] = lastelemarray[blockIdx.x];
        }
               
        // 计算需要进行块间累加位置索引（块外的数组索引）。
        unsigned int idx = blockIdx.x * (blockDim.x * packnum) + threadIdx.x;

        // 块内同步。
        __syncthreads();
    
        // 每个线程处理两个元素，将中间结果数组中的值加回到原数组。
        for (int i = 0; i < packnum; i++) {
            // 如果索引大于处理数组长度，则退出。
            if (idx >= n)
                break;

            // 将中间结果加回。
            array[idx] = op(array[idx], lastelement[0]);

            // 计算每个线程处理的下一个元素的索引值。
            idx += blockDim.x;
        }
    }
}

// Host 成员方法：addBack（float 型的中间结果加回）
template< class Operation >
__host__ int ScanArray::addBack(float *array, float *lastelemarray,
                                int numelements, int blocksize, int packnum, 
                                Operation op, bool backward)
{
    // 检查输入和输出是否为 NULL，如果为 NULL 直接报错返回。
    if (array == NULL || lastelemarray == NULL)
        return NULL_POINTER;

    // 检查处理的数组长度，如果小于 0 出错。
    if (numelements < 0)
        return INVALID_DATA;
    
    // 计算线程块大小。
    int gridsize = (numelements + blocksize * packnum - 1) /
                   (blocksize * packnum) - 1;

    // 判断 gridsize 大小，如果小于 1，则不用进行加回操作。返回正确。
    if (gridsize < 1)
        return NO_ERROR;

    // 调用 _addBackKer 核函数，将中间结果数组加回到原扫描数组。
    _addBackKer<<<gridsize, blocksize>>>(array, lastelemarray,
            numelements, packnum, op, backward);

    // 判断是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
       
    // 处理完毕退出。
    return NO_ERROR;
}

// Host 成员方法：addBack（int 型的中间结果加回）
template< class Operation >
__host__ int ScanArray::addBack(int *array, int *lastelemarray,
                                int numelements, int blocksize, int packnum, 
                                Operation op, bool backward)
{
    // 检查输入和输出是否为 NULL，如果为 NULL 直接报错返回。
    if (array == NULL || lastelemarray == NULL)
        return NULL_POINTER;

    // 检查处理的数组长度，如果小于 0 出错。
    if (numelements < 0)
        return INVALID_DATA;
    
    // 计算线程块大小。
    int gridsize = (numelements + blocksize * packnum - 1) /
                   (blocksize * packnum) - 1;

    // 判断 gridsize 大小，如果小于 1，则不用进行加回操作。返回正确。
    if (gridsize < 1)
        return NO_ERROR;

    // 调用 _addBackKer 核函数，将中间结果数组加回到原扫描数组。
    _addBackKer<<<gridsize, blocksize>>>(array, lastelemarray,
                                         numelements, packnum, op, backward);

    // 判断是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
       
    // 处理完毕退出。
    return NO_ERROR;
}

// 宏：SCANFREE
// 如果出错，就释放之前申请的内存。
#define SCANFREE do {                       \
        if (gridsize > 1)                   \
            cudaFree(blocksumDev);          \
        if (hostinarray)                    \
            cudaFree(inarrayDev);           \
        if (hostoutarray)                   \
            cudaFree(outarrayDev);          \
        if (!hostinarray)                   \
            delete inarrayHost;             \
        if (!hostoutarray)                  \
            delete outarrayHost;            \
    } while (0)

// Host 成员方法：scanArray（float 型的数组扫描）
template< class Operation >
__host__ int ScanArray::scanArray(float *inarray, float *outarray, 
                                  int numelements, Operation op, bool backward, 
                                  bool hostinarray, bool hostoutarray)
{
    // 检查输入和输出是否为 NULL，如果为 NULL 直接报错返回。
    if (inarray == NULL || outarray == NULL)
        return NULL_POINTER;

    // 本程序实现的 4 种方法可处理的数组长度。必须加以判断控制。
    if (numelements < 0)
        return INVALID_DATA;

    // 局部变量，错误码。
    cudaError_t cuerrcode;
    int errcode;
 
    // 局部变量。
    const unsigned int memsize = sizeof (float) * numelements;
    unsigned int extraspace;

    // 计算共享内存的长度。
    unsigned int sharedmemsize = 0; 

    // 定义设备端的输入输出数组指针，当输入输出指针在 Host 端时，在设备端申请对
    // 应大小的数组。
    float *inarrayDev = NULL;
    float *outarrayDev = NULL;

    // 定义主机端的输入输出数组指针，当输入输出指针在 Device 端时，在主机端申请
    // 对应大小的数组。
    float *inarrayHost = NULL;
    float *outarrayHost = NULL;

    // 这里 scan 实现只支持单个线程块的计算，这里的 gridsize 可以设置的大于 1，
    // 从而让多个线程块都运行相同程序来测速。计算调用 Kernel 函数的线程块的尺寸
    // 和线程块的数量。
    int gridsize;
    int blocksize;

    // 局部变量，中间结果存放数组。长度会根据线程块大小来确定。
    float *blocksumDev = NULL;

    // 中间结果数组的长度。
    int blocksumsize;

    // scan 算法中每个线程块的计算能力。核函数每块处理长度与线程块大小的比值。
    int packnum;

    // 针对 CPU 端的实现类型，选择路径进行处理。
    if (scanType == CPU_IN_SCAN) {
        // 判断当前 inarray 数组是否存储在 Host 端。若不是，则需要在 Host 端
        // 为数组申请一段空间；若该数组是在 Host 端，则直接使用。
        if (!hostinarray) {
            // 为输入数组在 Host 端申请内存。    
            inarrayHost = new float[memsize];
            // 将输入数组拷贝到主机端内存。
            cuerrcode = cudaMemcpy(inarrayHost, inarray, memsize, 
                                   cudaMemcpyDeviceToHost);
            if (cuerrcode != cudaSuccess)
                return cuerrcode;
        } else {
            // 如果在主机端，则将指针传给主机端指针。
            inarrayHost = inarray;
        }

        // 判断当前 outarray 数组是否存储在 Host 端。若不是，则需要在 Host 端
        // 为数组申请一段空间；若该数组是在 Host 端，则直接使用。
        if (!hostoutarray) {
            // 为输出数组在 Host 端申请内存。    
            outarrayHost = new float[memsize];
            // 将输出数组拷贝到主机端内存。
            cuerrcode = cudaMemcpy(outarrayHost, outarray, memsize, 
                                   cudaMemcpyDeviceToHost);
            if (cuerrcode != cudaSuccess)
                return cuerrcode;
        } else {
            // 如果在主机端，则将指针传给主机端指针。
            outarrayHost = outarray;
        }

        
        // 调用 inclusive 版的 scan 函数
        errcode = scanComputeGold<float>(inarrayHost, outarrayHost, 
                                         numelements, op, backward);
        // 出错则返回错误码。
        if (errcode != NO_ERROR) {
            // 释放内存
            SCANFREE;
            return errcode;
        }

        // 执行结束
        return NO_ERROR;
    }

    // 判断当前 inarray 数组是否存储在 Host 端。若是，则需要在 Device 端为数组
    // 申请一段空间；若该数组是在 Device端，则直接使用。
    if (hostinarray) {
        // 为输入数组在设备端申请内存。    
        cuerrcode = cudaMalloc((void **)&inarrayDev, memsize);
        if (cuerrcode != cudaSuccess)
            return cuerrcode;

        // 将输入数组拷贝到设备端内存。
        cuerrcode = cudaMemcpy(inarrayDev, inarray, memsize, 
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess)
            return cuerrcode;
    } else {
        // 如果在设备端，则将指针传给对应的设备端统一指针。
        inarrayDev = inarray;
    }

    // 判断当前 outarray 数组是否存储在 Host 端。若是，则需要在 Device 端为数组
    // 申请一段空间；若该数组是在 Device端，则直接使用。
    if (hostoutarray) {
        // 为输出数组在设备端申请内存。
        cuerrcode = cudaMalloc((void **)&outarrayDev, memsize);
        if (cuerrcode != cudaSuccess)
            return cuerrcode;
    } else {
        // 如果在设备端，则将指针传给对应的设备端统一指针。
        outarrayDev = outarray;
    }

    // 针对不同的实现类型，选择不同的路径进行处理。
    switch(scanType) {
    // 使用简单版本的 scan 实现。
    case NAIVE_SCAN:
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        // 简单版本每个线程处理一个元素。
        blocksize = BLOCK_SIZE;
        packnum = 1;

        // 计算线程块大小和共享内存长度。
        gridsize = max(1, (numelements + blocksize * packnum - 1) / blocksize);
        sharedmemsize = sizeof (float) * blocksize * packnum;

        // 如果扫描所需要的线程的 grid 尺寸大于 1，就需要进行加回操作，就需要申
        // 请存放中间结果的数组。
        if (gridsize > 1) {
            // 需要将每段处理的最后一个元素取出，最后一个线程块不进行处理。
            blocksumsize = gridsize - 1;              

            // 为存放中间结果的数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blocksumDev,
                                   blocksumsize * sizeof(float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(blocksumDev);
                return cuerrcode;
            }
        }
   
        // 调用 Kernel 函数，完成实际的数组扫描。
        // 这里需要判断输入输出指针是否在设备端。
        _scanNaiveKer<float><<<gridsize, blocksize, 2 * sharedmemsize>>>(
                outarrayDev, inarrayDev, blocksumDev, numelements, op, 
                backward);

        // 判断核函数是否出错。
        if (cudaGetLastError() != cudaSuccess) {
            // 释放之前申请的内存。
            SCANFREE;
            return CUDA_ERROR;
        }
        break;

    // 使用效率版本的 scan 实现。
    case EFFICIENT_SCAN:
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        // 效率版本每个线程处理两个元素。
        blocksize = BLOCK_SIZE;
        packnum = 2;

        // 计算线程块大小和共享内存长度。
        sharedmemsize = sizeof (float) * (blocksize * packnum);
        gridsize = max(1, (numelements + packnum * blocksize - 1) /
                       (packnum * blocksize));

        // 如果扫描所需要的线程的 grid 尺寸大于 1，就需要进行加回操作，就需要申
        // 请存放中间结果的数组。
        if (gridsize > 1) {
            // 需要将每段处理的最后一个元素取出，最后一个线程块不进行处理。
            blocksumsize = gridsize - 1;

            // 为存放中间结果的数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blocksumDev,
                                   blocksumsize * sizeof(float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(blocksumDev);
                return cuerrcode;
            }
        }

        // 调用 Kernel 函数，完成实际的数组扫描。
        // 这里需要判断输入输出指针是否在设备端。
        _scanWorkEfficientKer<float><<<gridsize, blocksize, sharedmemsize>>>(
                outarrayDev, inarrayDev, blocksumDev, numelements, op,
                backward);
        
        // 判断是否出错。
        if (cudaGetLastError() != cudaSuccess) {
            // 释放之前申请的内存。
            SCANFREE;
            return CUDA_ERROR;
        }
        break;

    // 使用优化版本的 scan 实现。
    case OPTIMIZE_SCAN:
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        // 优化版本每个线程处理两个元素。
        blocksize = BLOCK_SIZE;
        packnum = 2;

        // 计算线程块大小和共享内存长度。
        gridsize = max(1, (numelements + packnum * blocksize - 1) /
                       (packnum * blocksize));
        sharedmemsize = sizeof (float) * (blocksize * packnum);   

        // 如果扫描所需要的线程的 grid 尺寸大于 1，就需要进行加回操作，就需要申
        // 请存放中间结果的数组。
        if (gridsize > 1) {
            // 需要将每段处理的最后一个元素取出，最后一个线程块不进行处理。
            blocksumsize = gridsize - 1;

            // 为存放中间结果的数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blocksumDev,
                                   blocksumsize * sizeof(float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(blocksumDev);
                return cuerrcode;
            }
        }

        // 调用 Kernel 函数，完成实际的数组扫描。
        // 这里需要判断输入输出指针是否在设备端。
        _scanOptKer<float><<<gridsize, blocksize, sharedmemsize>>>(
                outarrayDev, inarrayDev, blocksumDev, numelements, op,
                backward);

        // 判断是否出错。
        if (cudaGetLastError() != cudaSuccess) {
            // 释放之前申请的内存。
            SCANFREE;
            return CUDA_ERROR;
        }
        break;

    // 使用优化版本的 scan 实现。
    case BETTER_SCAN:
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        // 优化版本每个线程处理两个元素。
        blocksize = BLOCK_SIZE;
        packnum = 2;

        // 计算线程块大小和共享内存长度。
        gridsize = max(1, (numelements + packnum * blocksize - 1) /
                       (packnum * blocksize));
        extraspace = blocksize * packnum / NUM_BANKS;;
        sharedmemsize = sizeof (float) * (blocksize * packnum + extraspace);

        // 如果扫描所需要的线程的 grid 尺寸大于 1，就需要进行加回操作，就需要申
        // 请存放中间结果的数组。
        if (gridsize > 1) {
            // 需要将每段处理的最后一个元素取出，最后一个线程块不进行处理。
            blocksumsize = gridsize - 1;

            // 为存放中间结果的数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blocksumDev,
                                   blocksumsize * sizeof(float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(blocksumDev);
                return cuerrcode;
            }
        }

        // 调用 Kernel 函数，完成实际的数组扫描。
        // 这里需要判断输入输出指针是否在设备端。
        _scanBetterKer<float><<<gridsize, blocksize, sharedmemsize>>>(
                outarrayDev, inarrayDev, blocksumDev, numelements, op,
                backward);
        
        // 判断是否出错。
        if (cudaGetLastError() != cudaSuccess) {
            // 释放之前申请的内存。
            SCANFREE;
            return CUDA_ERROR;
        }
        break;
        
    // 其他方式情况下，直接返回非法数据错误。
    default:
        if (hostinarray)
            cudaFree(inarrayDev);
        if (hostoutarray)
            cudaFree(outarrayDev);
        return INVALID_DATA;
    }


    // 如果处理的 gird 尺寸大于 1，那么对于扫描中间结果数组进行一次 scan
    // 操作和加回操作。
    if (gridsize > 1) {
        // 递归调用扫描函数。此时输入输出数组皆为中间结果数组。
        // 这里的递归调用不会调用多次，数组的规模是指数倍减小的。
        errcode = scanArray(blocksumDev, blocksumDev, blocksumsize, op, 
                            backward, false, false);
        if (errcode != NO_ERROR) {
            // 释放之前申请的内存。
            SCANFREE;
            return errcode;
        }
        
        // 调用加回函数，将各块的扫描中间结果加回到输出数组。
        errcode = addBack(outarrayDev, blocksumDev, numelements,
                          blocksize, packnum, op, backward);
        if (errcode != NO_ERROR) {
            // 释放之前申请的内存。
            SCANFREE;
            return errcode;
        }

        // 释放中间结果数组的设备端内存。
        cudaFree(blocksumDev);
    }

    // 如果 outarray 在 Host 端，将结果拷贝到输出。
    if (hostoutarray) {
        // 将结果从设备端内存拷贝到输出数组。
        cuerrcode = cudaMemcpy(outarray, outarrayDev, memsize,
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            if (hostinarray)            
                cudaFree(inarrayDev);
            cudaFree(outarrayDev);
            return cuerrcode;
        }
    }
        
    // 释放 Device 内存。需要判断输入输出参数是否在 host 端。
    if (hostinarray)
        cudaFree(inarrayDev);
    if (hostoutarray)
        cudaFree(outarrayDev);

    // 处理完毕退出。
    return NO_ERROR;
}

// Host 成员方法：scanArray（int 型的数组扫描）
template< class Operation >
__host__ int ScanArray::scanArray(int *inarray, int *outarray, 
                                  int numelements, Operation op, bool backward,
                                  bool hostinarray, bool hostoutarray)
{
    // 检查输入和输出是否为 NULL，如果为 NULL 直接报错返回。
    if (inarray == NULL || outarray == NULL)
        return NULL_POINTER;

    // 本程序实现的 4 种方法可处理的数组长度。必须加以判断控制。
    if (numelements < 0)
        return INVALID_DATA;

    // 局部变量，错误码。
    cudaError_t cuerrcode;
    int errcode;
 
    // 局部变量。
    int memsize = sizeof (int) * numelements;
    int extraspace;

    // 计算共享内存的长度。
    int sharedmemsize = 0; 

    // 定义设备端的输入输出数组指针，当输入输出指针在 Host 端时，在设备端申请对
    // 应大小的数组。
    int *inarrayDev = NULL;
    int *outarrayDev = NULL;

    // 定义主机端的输入输出数组指针，当输入输出指针在 Device 端时，在主机端申请
    // 对应大小的数组。
    int *inarrayHost = NULL;
    int *outarrayHost = NULL;

    // 这里 scan 实现只支持单个线程块的计算，这里的 gridsize 可以设置的大于 1，
    // 从而让多个线程块都运行相同程序来测速。计算调用 Kernel 函数的线程块的尺寸
    // 和线程块的数量。
    int gridsize;
    int blocksize;

    // 局部变量，中间结果存放数组。长度会根据线程块大小来确定。
    int *blocksumDev = NULL;

    // 中间结果数组的长度。
    int blocksumsize;

    // scan 算法中每个线程块的计算能力。核函数每块处理长度与线程块大小的比值。
    int packnum;

    // 针对 CPU 端实现类型，选择路径进行处理。
    if (scanType == CPU_IN_SCAN) {
        // 判断当前 inarray 数组是否存储在 Host 端。若不是，则需要在 Host 端
        // 为数组申请一段空间；若该数组是在 Host 端，则直接使用。
        if (!hostinarray) {
            // 为输入数组在 Host 端申请内存。    
            inarrayHost = new int[memsize];
            // 将输入数组拷贝到主机端内存。
            cuerrcode = cudaMemcpy(inarrayHost, inarray, memsize, 
                                   cudaMemcpyDeviceToHost);
            if (cuerrcode != cudaSuccess)
                return cuerrcode;
        } else {
            // 如果在主机端，则将指针传给主机端指针。
            inarrayHost = inarray;
        }

        // 判断当前 outarray 数组是否存储在 Host 端。若不是，则需要在 Host 端
        // 为数组申请一段空间；若该数组是在 Host 端，则直接使用。
        if (!hostoutarray) {
            // 为输出数组在 Host 端申请内存。    
            outarrayHost = new int[memsize];
            // 将输出数组拷贝到主机端内存。
            cuerrcode = cudaMemcpy(outarrayHost, outarray, memsize, 
                                   cudaMemcpyDeviceToHost);
            if (cuerrcode != cudaSuccess)
                return cuerrcode;
        } else {
            // 如果在主机端，则将指针传给主机端指针。
            outarrayHost = outarray;
        }

        // 调用 inclusive 版的 scan 函数
        errcode = scanComputeGold<int>(inarrayHost, outarrayHost, 
                                       numelements, op, backward);
        // 出错则返回错误码。
        if (errcode != NO_ERROR) {
            // 释放内存
            SCANFREE;
            return errcode;
        }

        // 执行结束
        return NO_ERROR;
    }

    // 判断当前 inarray 数组是否存储在 Host 端。若是，则需要在 Device 端为数组
    // 申请一段空间；若该数组是在 Device端，则直接使用。
    if (hostinarray) {
        // 为输入数组在设备端申请内存。    
        cuerrcode = cudaMalloc((void **)&inarrayDev, memsize);
        if (cuerrcode != cudaSuccess)
            return cuerrcode;

        // 将输入数组拷贝到设备端内存。
        cuerrcode = cudaMemcpy(inarrayDev, inarray, memsize, 
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess)
            return cuerrcode;
    } else {
        // 如果在设备端，则将指针传给对应的设备端统一指针。
        inarrayDev = inarray;
    }

    // 判断当前 outarray 数组是否存储在 Host 端。若是，则需要在 Device 端为数组
    // 申请一段空间；若该数组是在 Device端，则直接使用。
    if (hostoutarray) {
        // 为输出数组在设备端申请内存。
        cuerrcode = cudaMalloc((void **)&outarrayDev, memsize);
        if (cuerrcode != cudaSuccess)
            return cuerrcode;
    } else {
        // 如果在设备端，则将指针传给对应的设备端统一指针。
        outarrayDev = outarray;
    }

    // 针对不同的实现类型，选择不同的路径进行处理。
    switch(scanType) {
    // 使用简单版本的 scan 实现。
    case NAIVE_SCAN:
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        // 简单版本每个线程处理一个元素。
        blocksize = BLOCK_SIZE;
        packnum = 1;

        // 计算线程块大小和共享内存长度。
        gridsize = max(1, (numelements + blocksize * packnum - 1) / blocksize);
        sharedmemsize = sizeof (int) * blocksize * packnum;

        // 如果扫描所需要的线程的 grid 尺寸大于 1，就需要进行加回操作，就需要申
        // 请存放中间结果的数组。
        if (gridsize > 1) {
            // 需要将每段处理的最后一个元素取出，最后一个线程块不进行处理。
            blocksumsize = gridsize - 1;              

            // 为存放中间结果的数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blocksumDev,
                                   blocksumsize * sizeof(int));
            if (cuerrcode != cudaSuccess) {
                cudaFree(blocksumDev);
                return cuerrcode;
            }
        }
   
        // 调用 Kernel 函数，完成实际的数组扫描。
        // 这里需要判断输入输出指针是否在设备端。
        _scanNaiveKer<int><<<gridsize, blocksize, 2 * sharedmemsize>>>(
                outarrayDev, inarrayDev, blocksumDev, numelements, op, 
                backward);
                  
        // 判断核函数是否出错。
        if (cudaGetLastError() != cudaSuccess) {
            // 释放之前申请的内存。
            SCANFREE;
            return CUDA_ERROR;
        }
        break;

    // 使用效率版本的 scan 实现。
    case EFFICIENT_SCAN:
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        // 效率版本每个线程处理两个元素。
        blocksize = BLOCK_SIZE;
        packnum = 2;

        // 计算线程块大小和共享内存长度。
        sharedmemsize = sizeof (int) * (blocksize * packnum);
        gridsize = max(1, (numelements + packnum * blocksize - 1) /
                       (packnum * blocksize));

        // 如果扫描所需要的线程的 grid 尺寸大于 1，就需要进行加回操作，就需要申
        // 请存放中间结果的数组。
        if (gridsize > 1) {
            // 需要将每段处理的最后一个元素取出，最后一个线程块不进行处理。
            blocksumsize = gridsize - 1;

            // 为存放中间结果的数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blocksumDev,
                                   blocksumsize * sizeof(int));
            if (cuerrcode != cudaSuccess) {
                cudaFree(blocksumDev);
                return cuerrcode;
            }
        }

        // 调用 Kernel 函数，完成实际的数组扫描。
        // 这里需要判断输入输出指针是否在设备端。
        _scanWorkEfficientKer<int><<<gridsize, blocksize, sharedmemsize>>>(
                outarrayDev, inarrayDev, blocksumDev, numelements, op,
                backward);
        
        // 判断是否出错。
        if (cudaGetLastError() != cudaSuccess) {
            // 释放之前申请的内存。
            SCANFREE;
            return CUDA_ERROR;
        }
        break;

    // 使用优化版本的 scan 实现。
    case OPTIMIZE_SCAN:
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        // 优化版本每个线程处理两个元素。
        blocksize = BLOCK_SIZE;
        packnum = 2;

        // 计算线程块大小和共享内存长度。
        gridsize = max(1, (numelements + packnum * blocksize - 1) /
                       (packnum * blocksize));
        sharedmemsize = sizeof (int) * (blocksize * packnum);   

        // 如果扫描所需要的线程的 grid 尺寸大于 1，就需要进行加回操作，就需要申
        // 请存放中间结果的数组。
        if (gridsize > 1) {
            // 需要将每段处理的最后一个元素取出，最后一个线程块不进行处理。
            blocksumsize = gridsize - 1;

            // 为存放中间结果的数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blocksumDev,
                                   blocksumsize * sizeof(int));
            if (cuerrcode != cudaSuccess) {
                cudaFree(blocksumDev);
                return cuerrcode;
            }
        }

        // 调用 Kernel 函数，完成实际的数组扫描。
        // 这里需要判断输入输出指针是否在设备端。
        _scanOptKer<int><<<gridsize, blocksize, sharedmemsize>>>(
                outarrayDev, inarrayDev, blocksumDev, numelements, op,
                backward);

        // 判断是否出错。
        if (cudaGetLastError() != cudaSuccess) {
            // 释放之前申请的内存。
            SCANFREE;
            return CUDA_ERROR;
        }
        break;

    // 使用优化版本的 scan 实现。
    case BETTER_SCAN:
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        // 优化版本每个线程处理两个元素。
        blocksize = BLOCK_SIZE;
        packnum = 2;

        // 计算线程块大小和共享内存长度。
        gridsize = max(1, (numelements + packnum * blocksize - 1) /
                       (packnum * blocksize));
        extraspace = blocksize * packnum / NUM_BANKS;;
        sharedmemsize = sizeof (int) * (blocksize * packnum + extraspace);

        // 如果扫描所需要的线程的 grid 尺寸大于 1，就需要进行加回操作，就需要申
        // 请存放中间结果的数组。
        if (gridsize > 1) {
            // 需要将每段处理的最后一个元素取出，最后一个线程块不进行处理。
            blocksumsize = gridsize - 1;

            // 为存放中间结果的数组在设备端申请内存。
            cuerrcode = cudaMalloc((void **)&blocksumDev,
                                   blocksumsize * sizeof(int));
            if (cuerrcode != cudaSuccess) {
                cudaFree(blocksumDev);
                return cuerrcode;
            }
        }

        // 调用 Kernel 函数，完成实际的数组扫描。
        // 这里需要判断输入输出指针是否在设备端。
        _scanBetterKer<int><<<gridsize, blocksize, sharedmemsize>>>(
                outarrayDev, inarrayDev, blocksumDev, numelements, op,
                backward);
        
        // 判断是否出错。
        if (cudaGetLastError() != cudaSuccess) {
            // 释放之前申请的内存。
            SCANFREE;
            return CUDA_ERROR;
        }
        break;
        
    // 其他方式情况下，直接返回非法数据错误。
    default:
        if (hostinarray)
            cudaFree(inarrayDev);
        if (hostoutarray)
            cudaFree(outarrayDev);
        return INVALID_DATA;
    }


    // 如果处理的 gird 尺寸大于 1，那么对于扫描中间结果数组进行一次 scan
    // 操作和加回操作。
    if (gridsize > 1) {
        // 递归调用扫描函数。此时输入输出数组皆为中间结果数组。
        // 这里的递归调用不会调用多次，数组的规模是指数倍减小的。
        errcode = scanArray(blocksumDev, blocksumDev, blocksumsize, op,
                            backward, false, false);
        if (errcode != NO_ERROR) {
            // 释放之前申请的内存。
            SCANFREE;
            return errcode;
        }
        
        // 调用加回函数，将各块的扫描中间结果加回到输出数组。
        errcode = addBack(outarrayDev, blocksumDev, numelements,
                          blocksize, packnum, op, backward);
        if (errcode != NO_ERROR) {
            // 释放之前申请的内存。
            SCANFREE;
            return errcode;
        }

        // 释放中间结果数组的设备端内存。
        cudaFree(blocksumDev);
    }

    // 如果 outarray 在 Host 端，将结果拷贝到输出。
    if (hostoutarray) {
        // 将结果从设备端内存拷贝到输出数组。
        cuerrcode = cudaMemcpy(outarray, outarrayDev, memsize,
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            if (hostinarray)            
                cudaFree(inarrayDev);
            cudaFree(outarrayDev);
            return cuerrcode;
        }
    }
        
    // 释放 Device 内存。需要判断输入输出参数是否在 host 端。
    if (hostinarray)
        cudaFree(inarrayDev);
    if (hostoutarray)
        cudaFree(outarrayDev);

    // 处理完毕退出。
    return NO_ERROR;
}

// 函数：为了使模板运算能够连接通过，通过此函数预处理，将用到模板的函数实例化。
void example()
{
    // 定义 ScanArray 对象 
    ScanArray s;
    // 数组长度
    unsigned int num_elements = 1024;
    // 开辟空间的大小
    const unsigned int mem_size = sizeof(float) * num_elements;
    const unsigned int mem_size1 = sizeof(int) * num_elements;
    // 为 float 型输入输出指针开辟空间
    float *inarray = new float[mem_size];
    float *outarray = new float[mem_size];
    // 为 int 型输入输出指针开辟空间
    int *inarray1 = new int[mem_size1];
    int *outarray1 = new int[mem_size1];
    // 设置扫描类型为 NAIVE_SCAN
    s.setScanType(NAIVE_SCAN);
    // 默认输入输出数组均在 host 端
    bool inhost, outhost;
    inhost = true;
    outhost = true;
    // 新建加法和乘法运算对象
    add_class<float> a;
    multi_class<float> m;
    max_class<float> max;
    min_class<float> min;

    add_class<int> a1;
    multi_class<int> m1;
    max_class<int> max1;
    min_class<int> min1;
    
    // 用 float 型和 int 型分别调用加法和乘法的仿函数。
    s.scanArray(inarray, outarray, num_elements, a, false, inhost, outhost);
    s.scanArray(inarray1, outarray1, num_elements, a1, false, inhost, outhost);
    s.scanArray(inarray, outarray, num_elements, m, false, inhost, outhost);
    s.scanArray(inarray1, outarray1, num_elements, m1, false, inhost, outhost);
    s.scanArray(inarray, outarray, num_elements, max, false, inhost, outhost);
    s.scanArray(inarray1, outarray1, num_elements, max1, false, inhost, outhost);
    s.scanArray(inarray, outarray, num_elements, min, false, inhost, outhost);
    s.scanArray(inarray1, outarray1, num_elements, min1, false, inhost, outhost);
}

// 取消前面的宏定义。
#undef SCANFREE

