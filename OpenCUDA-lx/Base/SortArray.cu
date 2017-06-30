// SortArray.cu
// 实现并行排序算法

#include "SortArray.h"

#include <iostream>
#include <cmath>
using namespace std;


// Kernel 函数: _bitonicSortByAscendKer（双调升序排序）
// 实现并行双调排序，按照升序排序。套用 template 模板以便对多类型数据进行处理。
template < typename Type >
static __global__ void 
_bitonicSortByAscendKer(
        Type *inarray,   // 输入数组。
        Type *outarray,  // 排序后的输出数组。
        int length       // 数组长度，必须是 2 的整数次方。
);

// Kernel 函数: _bitonicSortByDescendKer（双调降序排序）
// 实现并行双调排序，按照降序排序。套用 template 模板以便对多类型数据进行处理。
template < typename Type >
static __global__ void 
_bitonicSortByDescendKer(
        Type *inarray,   // 输入数组。
        Type *outarray,  // 排序后的输出数组。
        int length       // 数组长度，必须是 2 的整数次方。
);

// Host 静态方法：_bitonicSort（并行双调排序模板函数）
// 并行双调排序模板函数，CLASS 成员方法调用此 Host 函数模板。
template < typename Type >
static __host__ int
_bitonicSort(
        Type *inarray,   // 输出数组
        Type *outarray,  // 输入数组
        int ishost,      // 判断输入和输出数组位置
        int sortflag,    // 排序标记
        int length       // 排序数组长度
);

// Kernel 函数: _oddEvenMergeSortByAscendKer（Batcher's 奇偶合并升序排序）
// 实现并行 Batcher's 奇偶合并排序，按照升序排序。套用 template 模板以便对
// 多类型数据进行处理。
template < typename Type >
static __global__ void 
_oddEvenMergeSortByAscendKer(
        Type *inarray,   // 输入数组。
        Type *outarray,  // 排序后的输出数组。
        int length,      // 数组长度，必须是 2 的幂次方。
        int tempp,       // 输入参数。
        int tempq        // 输入参数。
);

// Kernel 函数: _oddEvenMergeSortByDescendKer（Batcher's 奇偶合并降序排序）
// 实现并行 Batcher's 奇偶合并排序，按照降序排序。套用 template 模板以便对
// 多类型数据进行处理。
template < typename Type >
static __global__ void 
_oddEvenMergeSortByDescendKer(
        Type *inarray,   // 输入数组。
        Type *outarray,  // 排序后的输出数组。
        int length,      // 数组长度，必须是 2 的幂次方。
        int tempp,       // 输入参数。
        int tempq        // 输入参数。
);

// Host 静态方法：_oddEvenMergeSort（Batcher's 奇偶合并降序排序模板函数）
// Batcher's 奇偶合并降序排序模板函数，CLASS 成员方法调用此 Host 函数模板。
template < typename Type >
static __host__ int
_oddEvenMergeSort(
        Type *inarray,   // 输出数组
        Type *outarray,  // 输入数组
        int ishost,      // 判断输入和输出数组位置
        int sortflag,    // 排序标记
        int length       // 排序数组长度
);

// Kernel 函数: _shearSortRowAscKer（行升序排序）
// 对二维数据矩阵的每一行进行双调排序。套用 template 模板以便对多类型数据
// 进行处理。
template < typename Type >
static __global__ void 
_shearSortRowAscKer(
        Type *inarray,  // 输入数组。
        int lensec      // 矩阵行数。
);

// Kernel 函数: _shearSortRowDesKer（行降序排序）
// 对二维数据矩阵的每一行进行双调排序。套用 template 模板以便对多类型数据
// 进行处理。
template < typename Type >
static __global__ void 
_shearSortRowDesKer(
        Type *inarray,  // 输入数组。
        int lensec      // 矩阵行数。
); 

// Kernel 函数: _shearSortColAscKer（列升序排序）
// 对二维数据矩阵的每一列进行双调排序。套用 template 模板以便对多类型数据
// 进行处理。
template < typename Type >
static __global__ void 
_shearSortColAscKer(
        Type *inarray,  // 输入数组。
        int length,     // 矩阵列数。
        int lensec      // 矩阵行数。
);

// Kernel 函数: _shearSortColDesKer（列降序排序）
// 对二维数据矩阵的每一列进行双调排序。套用 template 模板以便对多类型数据
// 进行处理。
template < typename Type >
static __global__ void 
_shearSortColDesKer(
        Type *inarray,  // 输入数组。
        int length,     // 矩阵列数。
        int lensec      // 矩阵行数。
);

// Kernel 函数: _shearToPosKer（转换数据形式）
// 将一维数据转换成二维矩阵。套用 template 模板以便对多类型数据进行处理。
template < typename Type >
static __global__ void 
_shearToPosKer(
        Type *inarray,   // 输入数组。
        Type *outarray,  // 矩阵列数。
        int lensec       // 矩阵行数。
);

// Host 静态方法：_shearSortLoop（shear 排序核心函数）
// shear 排序的核心函数，需要判断是升序还是降序。
template < typename Type >
static __host__ int 
_shearSortLoop(
        Type *inarray,   // 输入数组。
        Type *outarray,  // 输出数组。
        int length,      // 矩阵列数。
        int lensec,      // 矩阵行数。
        int sortflag     // 排序标识。
);

// Host 静态方法：_shearSort（并行 shear 排序模板函数）
// 并行双调排序模板函数，CLASS 成员方法调用此 Host 函数模板。
template < typename Type >
static __host__ int
_shearSort(
        Type *inarray,   // 输出数组
        Type *outarray,  // 输入数组
        int ishost,      // 判断输入和输出数组位置
        int sortflag,    // 排序标记
        int length,      // 排序数组长度
        int lensec       // 排序矩阵的宽度
);

// Kernel 函数: _bitonicSortByAscendKer（双调升序排序）
template < typename Type >
static __global__ void _bitonicSortByAscendKer(Type *inarray, Type *outarray, 
                                               int length)
{
    // 读取线程号。
    int tid = threadIdx.x;

    // 声明共享内存，加快数据存取速度。
    extern __shared__ unsigned char sharedascend[];
    // 转化为模板类型的共享内存。
    Type *shareddata = (Type *)sharedascend;

    // 将全局内存中的数组拷贝到共享内存了。
    shareddata[tid] = inarray[tid];
    __syncthreads();

    int k, ixj, j;
    Type temp;

    // 并行双调排序，升序排序。
    for (k = 2; k <= length; k <<= 1) {
        // 双调合并。
        for (j = k >> 1; j > 0; j >>= 1) {
            // ixj 是与当前位置 tid 进行比较交换的位置。
            ixj = tid ^ j;
            if (ixj > tid) {
                // 如果 (tid & k) == 0，按照升序交换两项。
                if ((tid & k) == 0 && (shareddata[tid] > shareddata[ixj])) {
                    // 交换数组项。
                    temp = shareddata[tid];
                    shareddata[tid] = shareddata[ixj];
                    shareddata[ixj] = temp;
                // 如果 (tid & k) == 0，按照降序交换两项。
                } else if ((tid & k) != 0 &&
                           shareddata[tid] < shareddata[ixj]) {
                    // 交换数组项。
                    temp = shareddata[tid];
                    shareddata[tid] = shareddata[ixj];
                    shareddata[ixj] = temp;
                }
            }
            __syncthreads();
        }
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    outarray[tid] = shareddata[tid];
}

// Kernel 函数: _bitonicSortByDescendKer（双调降序排序）
template < typename Type >
static __global__ void _bitonicSortByDescendKer(Type *inarray, Type *outarray, 
                                                int length)
{
    // 读取线程号。
    int tid = threadIdx.x;

    // 声明共享内存，加快数据存取速度。
    extern __shared__ unsigned char shareddescend[];
    // 转化为模板类型的共享内存。
    Type *shareddata = (Type *)shareddescend;

    // 将全局内存中的数组拷贝到共享内存了。
    shareddata[tid] = inarray[tid];
    __syncthreads();

    int k, ixj, j;
    Type temp;

    // 并行双调排序，降序排序。
    for (k = 2; k <= length; k <<= 1) {
        // 双调合并。
        for (j = k >> 1; j > 0; j >>= 1) {
            // ixj 是与当前位置 tid 进行比较交换的位置。
            ixj = tid ^ j;
            if (ixj > tid) {
                // 如果 (tid & k) == 0，按照降序交换两项。
                if ((tid & k) == 0 && (shareddata[tid] < shareddata[ixj])) {
                    // 交换数组项。
                    temp = shareddata[tid];
                    shareddata[tid] = shareddata[ixj];
                    shareddata[ixj] = temp;
                // 如果 (tid & k) == 0，按照升序交换两项。
                } else if ((tid & k) != 0 &&
                           shareddata[tid] > shareddata[ixj]) {
                    // 交换数组项。
                    temp = shareddata[tid];
                    shareddata[tid] = shareddata[ixj];
                    shareddata[ixj] = temp;
                }
            }
            __syncthreads();
        }
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    outarray[tid] = shareddata[tid];
}

// Host 静态方法：_bitonicSort（并行双调排序模板函数）
template < typename Type >
static __host__ int _bitonicSort(Type *inarray, Type *outarray, int ishost,
                                 int sortflag, int length)
{
    // 检查输入输出参数是否为空。
    if (inarray == NULL || outarray == NULL)
        return NULL_POINTER;

    // 如果输入输出数组在 Host 端。
    if (ishost) {
        // 在 Device 上分配空间。一次申请所有空间，然后通过偏移索引各个数组。
        cudaError_t cudaerrcode;
        Type *alldevicedata, *devinarray, *devoutarray;
        cudaerrcode = cudaMalloc((void **)&alldevicedata,
                                 2 * length * sizeof (Type));
        if (cudaerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 通过偏移读取 Device 端内存空间。
        devinarray = alldevicedata;
        devoutarray = alldevicedata + length;

        //将 Host 上的 inarray 拷贝到 Device 上的 devinarray 中。
        cudaerrcode = cudaMemcpy(devinarray, inarray, 
                                 length * sizeof (Type),
                                 cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;
        }

        if (sortflag == SORT_ARRAY_TYPE_ASC) {
            // 双调升序排序。
            _bitonicSortByAscendKer<Type><<<
                    1, length, length * sizeof (Type)>>>(
                    devinarray, devoutarray, length);
        } else if (sortflag == SORT_ARRAY_TYPE_DESC) {
            // 双调降序排序。
            _bitonicSortByDescendKer<Type><<<
                    1, length, length * sizeof (Type)>>>(
                    devinarray, devoutarray, length);
        }
        
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;      
        } 
            
        //将 Device上的 devoutarray 拷贝到 Host上。
        cudaerrcode = cudaMemcpy(outarray, devoutarray, 
                                 length * sizeof (Type),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;
        }

        // 释放显存上的临时空间。
        cudaFree(alldevicedata);

    // 如果输入输出数组在 Device 端。
    } else {
        if (sortflag == SORT_ARRAY_TYPE_ASC) {
            // 双调升序排序。
            _bitonicSortByAscendKer<Type><<<
                    1, length, length * sizeof (Type)>>>(
                    inarray, outarray, length);
        } else if (sortflag == SORT_ARRAY_TYPE_DESC) {
            // 双调降序排序。
            _bitonicSortByDescendKer<Type><<<
                    1, length, length * sizeof (Type)>>>(
                    inarray, outarray, length);
        }
             
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;      
    }

    return NO_ERROR;
}

// 成员方法：bitonicSort（并行双调排序）
__host__ int SortArray::bitonicSort(int *inarray, int *outarray)
{
    // 调用模板函数并返回。
    return _bitonicSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// 成员方法：bitonicSort（并行双调排序）
__host__ int SortArray::bitonicSort(float *inarray, float *outarray)
{
    // 调用模板函数并返回。
    return _bitonicSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// 成员方法：bitonicSort（并行双调排序）
__host__ int SortArray::bitonicSort(unsigned char *inarray, 
                                    unsigned char *outarray)
{
    // 调用模板函数并返回。
    return _bitonicSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// 成员方法：bitonicSort（并行双调排序）
__host__ int SortArray::bitonicSort(char *inarray, char *outarray)
{
    // 调用模板函数并返回。
    return _bitonicSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// 成员方法：bitonicSort（并行双调排序）
__host__ int SortArray::bitonicSort(double *inarray, double *outarray)
{
    // 调用模板函数并返回。
    return _bitonicSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// Kernel 函数: _oddEvenMergeSortByAscendKer（Batcher's 奇偶合并升序排序）
template < typename Type >
static __global__ void _oddEvenMergeSortByAscendKer(
        Type *inarray, Type *outarray, int length, int tempp, int tempq)
{
    // 读取线程号CUDA_ERROR。
    int tid = threadIdx.x;

    // 声明共享内存，加快数据存取速度。
    extern __shared__ unsigned char sharedoddascend[];
    // 转化为模板类型的共享内存。
    Type *shared = (Type *)sharedoddascend;
    shared[tid] = inarray[tid];
    __syncthreads();
    
    // 声明临时变量。
    int p, q, r, d;
    Type temp;

    // 并行Batcher's 奇偶合并排序，升序排序。
    for (p = tempp; p >= 1; p >>= 1) {
        // r 是标记位。
        r = 0;
        // d 是步长。
        d = p;
        for (q = tempq; q >= p; q >>= 1) {
            if ((tid < length - d) && ((tid & p) == r) && 
                shared[tid] > shared[tid + d]) {
                // 交换数据项。
                temp = shared[tid];
                shared[tid] = shared[tid + d];
                shared[tid + d] = temp;
            }
            d = q - p;
            r = p;
            __syncthreads();
        }    
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    outarray[tid] = shared[tid];
}

// Kernel 函数: _oddEvenMergeSortByDescendKer（Batcher's 奇偶合并降序排序）
template < typename Type >
static __global__ void _oddEvenMergeSortByDescendKer(
        Type *inarray, Type *outarray, int length, int tempp, int tempq)
{
    // 读取线程号。
    int tid = threadIdx.x;

    // 声明共享内存，加快数据存取速度。
    extern __shared__ unsigned char sharedodddescend[];
    // 转化为模板类型的共享内存。
    Type *shared = (Type *)sharedodddescend;
    shared[tid] = inarray[tid];
    __syncthreads();

    // 声明临时变量。
    int p , q, r, d;
    Type temp;
    // 并行 Batcher's 奇偶合并排序，降序排序。
    for (p = tempp; p >= 1; p >>= 1) {
        // r 是标记位。
        r = 0;
        // d 是步长。
        d = p;
        for (q = tempq; q >= p; q >>= 1) {
            if ((tid < length - d) && ((tid & p) == r) &&
                shared[tid] < shared[tid + d]) {
                // 交换数据项。
                temp = shared[tid];
                shared[tid] = shared[tid + d];
                shared[tid + d] = temp;
            }
            d = q - p;
            r = p;
            __syncthreads();
        }
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    outarray[tid] = shared[tid];
}

// Host 静态方法：_oddEvenMergeSort（Batcher's 奇偶合并降序排序模板函数）
template < typename Type >
static __host__ int _oddEvenMergeSort(Type *inarray, Type *outarray, 
                                      int ishost, int sortflag, int length)
{
    // 检查输入输出参数是否为空。
    if (inarray == NULL || outarray == NULL)
        return NULL_POINTER;

    // 奇偶合并排序参数。
    int t, tempp, tempq;
    t = log((float)length) / log(2.0f);
    tempp = 1 << (t - 1);
    tempq = 1 << (t - 1);

    // 如果输入输出数组在 Host 端。
    if (ishost) {
        // 在 Device 上分配空间。一次申请所有空间，然后通过偏移索引各个数组。
        cudaError_t cudaerrcode;
        Type *alldevicedata, *devinarray, *devoutarray;
        cudaerrcode = cudaMalloc((void **)&alldevicedata,
                                 2 * length * sizeof (Type));
        if (cudaerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 通过偏移读取 Device 端内存空间。
        devinarray = alldevicedata;
        devoutarray = alldevicedata + length;
        
        //将 Host 上的 inarray 拷贝到 Device 上的 devinarray 中。
        cudaerrcode = cudaMemcpy(devinarray, inarray, 
                                 length * sizeof (Type),
                                 cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;
        }

        if (sortflag == SORT_ARRAY_TYPE_ASC) {
            // Batcher's 奇偶升序排序。
            _oddEvenMergeSortByAscendKer<Type><<<
                    1, length, length * sizeof (Type)>>>(
                    devinarray, devoutarray, length, tempp, tempq);
        } else if (sortflag == SORT_ARRAY_TYPE_DESC) {
            // Batcher's 奇偶降序排序。
            _oddEvenMergeSortByDescendKer<Type><<<
                    1, length, length * sizeof (Type)>>>(
                    devinarray, devoutarray, length, tempp, tempq); 
        }                                                      
        
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;      
        } 
            
        //将 Device上的 devoutarray 拷贝到 Host上。
        cudaerrcode = cudaMemcpy(outarray, devoutarray, 
                                 length * sizeof (Type),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;
        }

        // 释放显存上的临时空间。
        cudaFree(alldevicedata);
    // 如果输入输出数组在 Device 端。
    } else {
        if (sortflag == SORT_ARRAY_TYPE_ASC) {
            // Batcher's 奇偶升序排序。
            _oddEvenMergeSortByAscendKer<Type><<<
                    1, length, length * sizeof (Type)>>>(
                    inarray, outarray, length, tempp, tempq);
        } else if (sortflag == SORT_ARRAY_TYPE_DESC) {
            // Batcher's 奇偶降序排序。
            _oddEvenMergeSortByDescendKer<Type><<<
                    1, length, length * sizeof (Type)>>>(
                    inarray, outarray, length, tempp, tempq);
        }
                    
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;      
    }
    
    return NO_ERROR;
}

// 成员方法：oddEvenMergeSort（并行 Batcher's 奇偶合并排序）
__host__ int SortArray::oddEvenMergeSort(int *inarray, int *outarray)
{
    // 调用模板函数并返回。
    return _oddEvenMergeSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// 成员方法：oddEvenMergeSort（并行 Batcher's 奇偶合并排序）
__host__ int SortArray::oddEvenMergeSort(float *inarray, float *outarray)
{
    // 调用模板函数并返回。
    return _oddEvenMergeSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// 成员方法：oddEvenMergeSort（并行 Batcher's 奇偶合并排序）
__host__ int SortArray::oddEvenMergeSort(unsigned char *inarray, 
                                         unsigned char *outarray)
{
    // 调用模板函数并返回。
    return _oddEvenMergeSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// 成员方法：oddEvenMergeSort（并行 Batcher's 奇偶合并排序）
__host__ int SortArray::oddEvenMergeSort(char *inarray, char *outarray)
{
    // 调用模板函数并返回。
    return _oddEvenMergeSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// 成员方法：oddEvenMergeSort（并行 Batcher's 奇偶合并排序）
__host__ int SortArray::oddEvenMergeSort(double *inarray, double *outarray)
{
    // 调用模板函数并返回。
    return _oddEvenMergeSort(inarray, outarray, this->ishost, this->sortflag,
                        this->length);
}

// Kernel 函数: _shearSortRowAscKer（行升序排序）
template < typename Type >
static __global__ void _shearSortRowAscKer(Type *inarray, int lensec)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    // 将全局内存中的数组拷贝到共享内存了。
    extern __shared__ unsigned char sharedrowasc[];
	// 转化为模板类型的共享内存。
    Type *shared = (Type *)sharedrowasc;
    if (cid < lensec)
        shared[cid] = inarray[rid * lensec + cid];
    __syncthreads();

    // 声明临时变量。
    int ixj;
    Type temp;
    // 偶数行升序排序。
    if (rid % 2 == 0) {
        for (int k = 2; k <= lensec; k <<= 1) {
             // 双调合并。
            for (int j = k >> 1; j > 0; j >>= 1) {
                // ixj 是与当前位置 cid 进行比较交换的位置。
                ixj = cid ^ j;
                if (ixj > cid) {
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    if ((cid & k) == 0 && (shared[cid] > shared[ixj])) {
                        // 交换数组项。
                        temp = shared[cid];
                        shared[cid] = shared[ixj];
                        shared[ixj] = temp;
                    // 如果 (cid & k) == 0，按照降序交换两项。
                    } else if ((cid & k) != 0 && shared[cid] < shared[ixj]) {
                        // 交换数组项。
                        temp = shared[cid];
                        shared[cid] = shared[ixj];
                        shared[ixj] = temp;
                    }
                }
                __syncthreads();
            }
        }
    // 奇数行降序排序。
    } else {
        for (int k = 2; k <= lensec; k <<= 1) {
            // 双调合并。
            for (int j = k >> 1; j > 0; j >>= 1) {
                // ixj 是与当前位置 cid 进行比较交换的位置。
                ixj = cid ^ j;
                if (ixj > cid) {
                    // 如果 (cid & k) == 0，按照降序交换两项。
                    if ((cid & k) == 0 && (shared[cid] < shared[ixj])) {
                        // 交换数组项。
                        temp = shared[cid];
                        shared[cid] = shared[ixj];
                        shared[ixj] = temp;
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    } else if ((cid & k) != 0 && shared[cid] > shared[ixj]) {
                        // 交换数组项。
                        temp = shared[cid];
                        shared[cid] = shared[ixj];
                        shared[ixj] = temp;
                    }
                }   
                __syncthreads();
            }
        }    
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    if (cid <lensec)
        inarray[rid * lensec + cid] = shared[cid];
}

// Kernel 函数: _shearSortRowDesKer（行降序排序）
template < typename Type >
static __global__ void _shearSortRowDesKer(Type *inarray, int lensec)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    // 将全局内存中的数组拷贝到共享内存了。
    extern __shared__ unsigned char sharedrowdes[];
    // 转化为模板类型的共享内存。
    Type *shared = (Type *)sharedrowdes;
    if (cid < lensec)
        shared[cid] = inarray[rid * lensec + cid];
    __syncthreads();

    // 声明临时变量
    int ixj;
    Type temp;
    // 偶数行降序排序。
    if (rid % 2 == 0) {
        for (int k = 2; k <= lensec; k <<= 1) {
             // 双调合并。
            for (int j = k >> 1; j > 0; j >>= 1) {
                // ixj 是与当前位置 cid 进行比较交换的位置。
                ixj = cid ^ j;
                if (ixj > cid) {
                    // 如果 (cid & k) == 0，按照降序交换两项。
                    if ((cid & k) == 0 && (shared[cid] < shared[ixj])) {
                        // 交换数组项。
                        temp = shared[cid];
                        shared[cid] = shared[ixj];
                        shared[ixj] = temp;
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    } else if ((cid & k) != 0 && shared[cid] > shared[ixj]) {
                        // 交换数组项。
                        temp = shared[cid];
                        shared[cid] = shared[ixj];
                        shared[ixj] = temp;
                    }
                }
                __syncthreads();
            }
        }
    // 奇数行升序排序。
    } else {
        for (int k = 2; k <= lensec; k <<= 1) {
            // 双调合并。
            for (int j = k >> 1; j > 0; j >>= 1) {
                // ixj 是与当前位置 cid 进行比较交换的位置。
                ixj = cid ^ j;
                if (ixj > cid) {
                    // 如果 (cid & k) == 0，按照降序交换两项。
                    if ((cid & k) == 0 && (shared[cid] > shared[ixj])) {
                        // 交换数组项。
                        temp = shared[cid];
                        shared[cid] = shared[ixj];
                        shared[ixj] = temp;
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    } else if ((cid & k) != 0 && shared[cid] < shared[ixj]) {
                        // 交换数组项。
                        temp = shared[cid];
                        shared[cid] = shared[ixj];
                        shared[ixj] = temp;
                    }
                }   
                __syncthreads();
            }
        }    
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    if (cid < lensec)
        inarray[rid * lensec + cid] = shared[cid];
}

// Kernel 函数: _shearSortColAscKer（列升序排序）
template < typename Type >
static __global__ void _shearSortColAscKer(Type *inarray, 
                                           int length, int lensec)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    if (rid >= lensec)
        return;

    // 将全局内存中的数组拷贝到共享内存了。
    extern __shared__ unsigned char sharedcolasc[];
    // 转化为模板类型的共享内存。
    Type *shared = (Type *)sharedcolasc;
    if (cid < length)
        shared[cid] = inarray[rid + cid * lensec];
    __syncthreads();

    // 声明临时变量。
    int ixj;
    Type temp;
    // 并行双调排序，升序排序。
    for (int k = 2; k <= length; k <<= 1) {
        // 双调合并。
        for (int j = k >> 1; j > 0; j >>= 1) {
            // ixj 是与当前位置 cid 进行比较交换的位置。
            ixj = cid ^ j;
            if (ixj > cid) {
                // 如果 (cid & k) == 0，按照升序交换两项。
                if ((cid & k) == 0 && (shared[cid] > shared[ixj])) {
                    // 交换数组项。
                    temp = shared[cid];
                    shared[cid] = shared[ixj];
                    shared[ixj] = temp;
                // 如果 (cid & k) == 0，按照降序交换两项。
                } else if ((cid & k) != 0 && shared[cid] < shared[ixj]) {
                    // 交换数组项。
                    temp = shared[cid];
                    shared[cid] = shared[ixj];
                    shared[ixj] = temp;
                }
            }
            __syncthreads();
        }
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    if (cid < length)
        inarray[rid + cid * lensec] = shared[cid];
}
    
// Kernel 函数: _shearSortColDesKer（列降序排序）
template < typename Type >
static __global__ void _shearSortColDesKer(Type *inarray, 
                                           int length, int lensec)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    if (rid >= lensec)
        return;

    // 将全局内存中的数组拷贝到共享内存了。
    extern __shared__ unsigned char sharedcoldes[];
    // 转化为模板类型的共享内存。
    Type *shared = (Type *)sharedcoldes;
    if (cid < length)
        shared[cid] = inarray[rid + cid * lensec];
    __syncthreads();

    // 声明临时变量。
    int ixj;
    Type temp;
    // 并行双调排序，降序排序。
    for (int k = 2; k <= length; k <<= 1) {
        // 双调合并。
        for (int j = k >> 1; j > 0; j >>= 1) {
            // ixj 是与当前位置 cid 进行比较交换的位置。
            ixj = cid ^ j;
            if (ixj > cid) {
                // 如果 (cid & k) == 0，按照降序交换两项。
                if ((cid & k) == 0 && (shared[cid] < shared[ixj])) {
                    // 交换数组项。
                    temp = shared[cid];
                    shared[cid] = shared[ixj];
                    shared[ixj] = temp;
                // 如果 (cid & k) == 0，按照升序交换两项。
                } else if ((cid & k) != 0 && shared[cid] > shared[ixj]) {
                    // 交换数组项。
                    temp = shared[cid];
                    shared[cid] = shared[ixj];
                    shared[ixj] = temp;
                }
            }
            __syncthreads();
        }
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    if (cid < length)
        inarray[rid + cid * lensec] = shared[cid];
}

// Kernel 函数: _shearToPosKer（转换数据形式）
template < typename Type >
static __global__ void _shearToPosKer(Type *inarray, Type *outarray, 
                                      int lensec)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    // 将全局内存中的数组拷贝到共享内存了。
    extern __shared__ unsigned char sharedpos[];
    // 转化为模板类型的共享内存。
    Type *shared = (Type *)sharedpos;
    shared[cid] = inarray[rid * lensec + cid];
    __syncthreads();
    
    // 偶数行赋值。
    if (rid % 2 == 0)
        outarray[rid * lensec + cid] = shared[cid];
    // 奇数行赋值。
    else
        outarray[rid * lensec + cid] = shared[lensec - 1 - cid];
}

// Host 静态方法：_shearSortLoop（shear 排序核心函数）
template < typename Type >
static __host__ int _shearSortLoop(Type *inarray, Type * outarray, 
                                   int length, int lensec, int sortflag)
{
    // 计算二维数组中长和宽的较大值。
    int judge = (length > lensec) ? length : lensec;

    // 将输入的一维数组转换成二维数组，便于后面的排序操作。
    _shearToPosKer<Type><<<length, lensec, judge * sizeof (Type)>>>(
            inarray, outarray, lensec);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;       

    for (int i = length; i >= 1; i >>= 1) {
        if (sortflag == 2) {
            // 首先进行列排序。
            _shearSortColAscKer<Type><<<judge, judge, judge * sizeof (Type)>>>(
                    outarray, length, lensec);
                    
            // 若调用 CUDA 出错返回错误代码
            if (cudaGetLastError() != cudaSuccess)
                return CUDA_ERROR;      
                 
            // 然后进行行排序。
            _shearSortRowAscKer<Type><<<judge, judge, judge * sizeof (Type)>>>(
                    outarray, lensec);

            // 若调用 CUDA 出错返回错误代码
            if (cudaGetLastError() != cudaSuccess)
                return CUDA_ERROR;
        } else {
            // 首先进行列排序。
            _shearSortColDesKer<Type><<<judge, judge, judge * sizeof (Type)>>>(
                    outarray, length, lensec);
                    
            // 若调用 CUDA 出错返回错误代码
            if (cudaGetLastError() != cudaSuccess)
                return CUDA_ERROR;    
                               
            // 然后进行行排序。
            _shearSortRowDesKer<Type><<<judge, judge, judge * sizeof (Type)>>>(
                    outarray, lensec);

            // 若调用 CUDA 出错返回错误代码
            if (cudaGetLastError() != cudaSuccess)
                return CUDA_ERROR;
        }
    }
    // 整理排序后的数组。
    _shearToPosKer<Type><<<length, lensec, judge * sizeof (Type)>>>(
            outarray, outarray, lensec);
            
    // 若调用 CUDA 出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return NO_ERROR;
}

// Host 静态方法：_shearSort（并行 shear 排序模板函数）
template < typename Type >
static __host__ int _shearSort(Type *inarray, Type *outarray, bool ishost, 
                               int sortflag, int length, int lensec)
{
    // 检查输入输出数组是否为空。
    if (inarray == NULL || outarray == NULL)
        return NULL_POINTER;

    // 检查算法参数的有效性。
    if ((sortflag != 1 && sortflag != 2) || (length % 2 != 0) || lensec < 0)
        return INVALID_DATA;

    // 数据项总个数。
    int datalength = length * lensec;
    // 局部变量，错误码。
    int errcode;

    if (ishost) {
        // 在 Device 上分配空间。一次申请所有空间，然后通过偏移索引各个数组。
        cudaError_t cudaerrcode;
        Type *alldevicedata, *devinarray, *devoutarray;
        cudaerrcode = cudaMalloc((void **)&alldevicedata,
                                 2 * datalength * sizeof (Type));
        if (cudaerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 通过偏移读取 Device 端内存空间。
        devinarray = alldevicedata;
        devoutarray = alldevicedata + datalength;

        //将 Host 上的 inarray 拷贝到 Device 上的 devinarray 中。
        cudaerrcode = cudaMemcpy(devinarray, inarray, 
                                 datalength * sizeof (Type),
                                 cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;
        }


        // 调用排序核心函数。
        errcode = _shearSortLoop<Type>(devinarray, devoutarray, 
                                      length, lensec, sortflag);
        if (errcode != NO_ERROR) {
            cudaFree(alldevicedata);
            return errcode;
        }

        //将 Device上的 devoutarray 拷贝到 Host 上。
        cudaerrcode = cudaMemcpy(outarray, devoutarray, 
                                 datalength * sizeof (Type),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;
        }

        // 释放显存上的临时空间。
        cudaFree(alldevicedata);
        return NO_ERROR;

    } else {
        // 调用排序核心函数。
        errcode = _shearSortLoop<Type>(inarray, outarray, 
                                       length, lensec, sortflag);
        if (errcode != NO_ERROR)
            return errcode;
    }
    
    return NO_ERROR;
}

// 成员方法：shearSort（并行 shear 排序）
__host__ int SortArray::shearSort(int *inarray, int *outarray)
{
    // 调用模板函数并返回。
    return _shearSort(inarray, outarray, this->ishost, this->sortflag,
                      this->length, this->lensec);
}

// 成员方法：shearSort（并行 shear 排序）
__host__ int SortArray::shearSort(float *inarray, float *outarray)
{
    // 调用模板函数并返回。
    return _shearSort(inarray, outarray, this->ishost, this->sortflag,
                      this->length, this->lensec);
}

// 成员方法：shearSort（并行 shear 排序）
__host__ int SortArray::shearSort(unsigned char *inarray, 
                                  unsigned char *outarray)
{
    // 调用模板函数并返回。
    return _shearSort(inarray, outarray, this->ishost, this->sortflag,
                      this->length, this->lensec);
}

// 成员方法：shearSort（并行 shear 排序）
__host__ int SortArray::shearSort(char *inarray, char *outarray)
{
    // 调用模板函数并返回。
    return _shearSort(inarray, outarray, this->ishost, this->sortflag,
                      this->length, this->lensec);
}

// 成员方法：shearSort（并行 shear 排序）
__host__ int SortArray::shearSort(double *inarray, double *outarray)
{
    // 调用模板函数并返回。
    return _shearSort(inarray, outarray, this->ishost, this->sortflag,
                      this->length, this->lensec);
}
    
