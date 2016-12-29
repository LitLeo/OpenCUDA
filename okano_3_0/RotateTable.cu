// RotateTable.cu
// 生成指定点集在某一旋转范围内各角度下的旋转后坐标集。

#include "RotateTable.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：M_PI
// π值。对于某些操作系统，M_PI可能没有定义，这里补充定义 M_PI。
#ifndef M_PI
#define M_PI 3.14159265359
#endif

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。这里之所以定义成 512×1 的 Block 尺寸，是希望能够
// 减少重复的角度计算，并能够充分的利用全局内存的带宽。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8
#define DEF_BLOCK_Z   1

// Kernel 函数：_rotateTableKer（计算旋转表）
// 计算出原始点集对应各个旋转角度下的旋转表。在这个 Kernel 函数中，Grid 的 x 分
// 量用于计算对应的点；y 分量用于计算对应的角度。
static __global__ void  // Kernel 函数无返回值
_rotateTableKer(
        RotateTable rt  // 旋转表类实例，用于提供旋转参数
);


// Kernel 函数：rotateTableKer（计算旋转表）
static __global__ void _rotateTableKer(RotateTable rt)
{
    // 计算当前线程在 Grid 中的位置。其中，c 表示 x 分量，各列处理各自的坐标
    // 点；r 表示 y 分量，各行处理各自的角度。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int aidx = blockIdx.z * blockDim.z + threadIdx.z;

    // 共享内存数据区，该部分包含了当前 Block 的各角度的正弦和余弦值，和总的角
    // 度数量。由于未对 Kernel 的尺寸做出假设，这里使用动态申请的 Shared
    // Memory（共享内存）。
    extern __shared__ float shddata[];

    // 从共享内存中拆出当前线程所需要的那些数据。这些数据包括，当前线程处理的角
    // 度的正弦和余弦值，总的角度的数量（用于计算输出结果在旋转表中的下标）。
    float *sinangle = &(shddata[threadIdx.z * 2]);
    float *cosangle = &(shddata[threadIdx.z * 2 + 1]);
    int *angcnt = (int *)(&(shddata[blockDim.z * 2]));

    // 计算共享内存中的数据。由于这些数据需要在 Block 内的各线程共享，因此只需
    // 要几个线程计算出结果，其他线程就可以坐享其成。这里选择线程号为 0 的线程
    // 完成这些计算。
    if (threadIdx.x == 0 || threadIdx.y == 0) {
        // 根据当前线程在 Grid 中的行号，计算出当前行所要处理的角度。
        float angle = rt.getAngleVal(aidx);

        // 计算角度的正弦余弦值。由于 sin 和 cos 内建函数要求输入弧度，因此需要
        // 先将角度转化为弧度。
        angle = angle * (float)M_PI / 180.0f;
        sinangle[0] = sin(angle);
        cosangle[0] = cos(angle);

        // 计算总的角度数量。由于对于所有的线程，该值都一样，因此只需要在每个
        // Block 的第一个线程处理这个计算即可，其他线程全部可以坐享其成。
        if (threadIdx.z == 0)
            angcnt[0] = rt.getAngleCount();
    }
    // 对共享内存的写入操作到此结束，因此需要同步一下 Block 内的个线程，使得其
    // 写入的结果在其他个线程中也是可见的。
    __syncthreads();

    // 如果当前线程是一个越界的线程，则直接退出。
    if (c >= rt.getSizeX() || r >= rt.getSizeY() || aidx >= rt.getAngleCount())
        return;

    // 获得当前线程处理的坐标点。
    float srcx = c + rt.getOffsetX();
    float srcy = r + rt.getOffsetY();

    // 进行旋转，并将旋转后的结果平移回原处。
    float dstx = srcx * cosangle[0] - srcy * sinangle[0];
    float dsty = srcx * sinangle[0] + srcy * cosangle[0];

    // 将得到的旋转结果写入旋转表中。
    int outidx = (aidx * rt.getSizeY() + r) * rt.getSizeX() + c;
    rt.getRotateTableX()[outidx] = dstx;
    rt.getRotateTableY()[outidx] = dsty;
}

// Host 成员方法：calcRotateTable（计算旋转表）
__host__ int RotateTable::initRotateTable()
{
    // 如果 CLASS 实例已处于 READY_RTT 状态，则直接返回，不需要进行任何操作。
    if (this->curState == READY_RTT)
        return OP_OVERFLOW;

    // 为旋转表申请内存空间。
    cudaError_t cuerrcode;
    // 首先计算初各种数据的尺寸，包括角度的数量和旋转表的尺寸。旋转表的尺寸应该
    // 能够包括所有的范围内的坐标点在所有角度范围内旋转后的坐标。
    size_t anglecnt = this->getAngleCount();
    size_t datasize = this->sizex * this->sizey * anglecnt * sizeof (float);

    // 按照计算得到的数据尺寸为旋转表申请空间
    // 受限申请 x 分量旋转表的内存空间。
    cuerrcode = cudaMalloc((void **)&this->rttx, datasize);
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;

    // 之后申请 y 分量旋转表的内存空间。
    cuerrcode = cudaMalloc((void **)&this->rtty, datasize);
    if (cuerrcode != cudaSuccess) {
        // 如果 y 分量旋转表的内存空间申请失败需要释放掉之前申请的 x 分量旋转
        // 表，以防止内存泄漏。
        cudaFree(this->rttx);
        this->rttx = NULL;
        return CUDA_ERROR;
    }

    // 计算 Kernel 函数调用的 Grid 尺寸，根据默认的 Block 尺寸，使用最普通的线
    // 程块划分方法。
    dim3 blocksize(DEF_BLOCK_X, DEF_BLOCK_Y, DEF_BLOCK_Z);
    dim3 gridsize;
    gridsize.x = (this->sizex + blocksize.x - 1) / blocksize.x;
    gridsize.y = (this->sizey + blocksize.y - 1) / blocksize.y;
    gridsize.z = (anglecnt + blocksize.z - 1) / blocksize.z;

    // 计算所需要的共享内存尺寸，共享内存存储的内容包括前面存储角度的正弦余弦
    // 值（偶数下标用于正弦，奇数下标用于余弦）和总的角度数量。
    int shdsize = 2 * blocksize.z * sizeof (float) + sizeof (int);

    // 调用 Kernel 函数，完成并行旋转表的计算。
    _rotateTableKer<<<gridsize, blocksize, shdsize>>>(*this);
    if (cudaGetLastError() != cudaSuccess) {
        // 注意，为了防止操作失败导致内存泄露，此处需要释放先前申请的内存空间。
        cudaFree(this->rttx);
        cudaFree(this->rtty);
        return CUDA_ERROR;
    }

    // 将 CLASS 状态转为 READY_RTT
    this->curState = READY_RTT;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：calcRotateTable（销毁旋转表）
__host__ int RotateTable::disposeRotateTable()
{
    // 如果 CLASS 实例已处于 NULL_RTT 状态，则直接返回，不需要进行任何操作。
    if (this->curState == NULL_RTT)
        return NO_ERROR;

    // 将当前 CLASS 的状态转回 NULL_RTT，之所以要先转状态，是考虑了可能的多线程
    // 操作，防止释放数据空间后的访问导致意外的错误。当然如果是多线程操作这段代
    // 码需要加锁。
    this->curState = NULL_RTT;

    // 释放旋转表所占用的内容空间。
    cudaFree(this->rttx);
    cudaFree(this->rtty);
    this->rttx = NULL;
    this->rtty = NULL;

    // 处理完毕返回。
    return NO_ERROR;
}

