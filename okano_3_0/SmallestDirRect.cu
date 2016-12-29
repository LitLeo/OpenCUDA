// SmallestDirRect.cu
// 最小有向外接矩形实现。

#include "SmallestDirRect.h"
#include "CoordiSet.h"
#include <cmath>
#include <iostream>
#include <stdio.h>
using namespace std;

// 宏：SDR_BLOCKSIZE
// 定义了核函数线程块的大小。
#define DEF_BLOCK_1D 512

// 宏：SDR_LARGE_ENOUGH
// 定义了一个足够大的正整数，该整数在使用过程中被认为是无穷大。
#define SDR_LARGE_ENOUGH  ((1 << 30) - 1)

// 宏：SDR_DEBUG_KERNEL_PRINT（Kernel 调试打印开关）
// 打开该开关则会在 Kernel 运行时打印相关的信息，以参考调试程序；如果注释掉该
// 宏，则 Kernel 不会打印这些信息，但这会有助于程序更快速的运行。
//#define SDR_DEBUG_KERNEL_PRINT

// Kernel 函数: _sdrComputeBoundInfoKer（计算凸壳点集中每相邻两点的旋转矩阵
// 信息,进而计算新坐标系下凸壳的有向外接矩形的边界信息）
// 根据输入的凸壳点，计算顺时针相邻两点的构成的直线与 x 轴的角度，同时计算
// 旋转矩阵信息。在此基础上，计算新坐标系下各点的坐标。从而计算每个有向外接
// 矩形的边界点的坐标信息。
static __global__ void              // Kernel 函数无返回值。
_sdrComputeBoundInfoKer(
        CoordiSet convexcst,        // 输入凸壳点集。
        RotationInfo rotateinfo[],  // 输出，旋转矩阵信息数组。
        BoundBox bbox[]             // 输出，找出的包围矩形的边界坐标信息数组。
);

// Kernel 函数: _sdrComputeSDRKer（计算包围矩形中面积最小的）
// 根据输入的目前的每个包围矩形的长短边长度，计算最小有向外接矩形的标号索引。
static __global__ void    // Kernel 函数无返回值。
_sdrComputeSDRKer(
        int cstcnt,       // 输入，点集中点的数量。
        BoundBox bbox[],  // 输入，找出的包围矩形的边界坐标信息。
        int *index        // 输出，计算出的最小有向外接矩形的标号索引。
);

// Kernel 函数: _sdrComputeBoundInfoKer（计算凸壳点集中每相邻两点的旋转矩阵
// 信息,进而计算新坐标系下凸壳的有向外接矩形的边界信息）
static __global__ void _sdrComputeBoundInfoKer(
        CoordiSet convexcst, RotationInfo rotateinfo[], BoundBox bbox[])
{
    // 当前 block 的索引，在 x 上 block 的索引表示各个凸壳点的索引。
    int r = blockIdx.x;

    // 检查索引值是否越界。如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (r >= convexcst.count)
        return;

    // 当前凸壳点的下一个点的索引。
    int nextidx;

    // 当前点与下一点的 x、y 坐标差值。
    float deltax, deltay;

    // 当前点与下一点间距离。
    float sidelength;

    // 旋转角度的余弦值和正弦值。
    float cosalpha, sinalpha;

    // 声明 Shared Memory，并分配各个指针。
    extern __shared__ float shdmem[];
    float *shdcos = shdmem;
    float *shdsin = shdcos + 1;
    float *shdradian = shdsin + 1;

    if (threadIdx.x == 0) {
        // 当前索引值加 1，求得下一点索引值。
        nextidx = r + 1;

        // 若当前点为点集中最后一点，则下一点为起始点。
        if (nextidx == convexcst.count)
            nextidx = 0;
    
        // 计算当前点和下一点 x, y 坐标差值。
        deltax = convexcst.tplData[nextidx * 2] -
                 convexcst.tplData[r * 2];
        deltay = convexcst.tplData[nextidx * 2 + 1] -
                 convexcst.tplData[r * 2 + 1];

        // 如果解的 x 在第二或者第三象限，转化坐标到第四或者第一象限。
        if (deltax < 0) {
            deltax = -deltax;
            deltay = -deltay;
        }

        // 计算当前点和下一点间距离。
        sidelength = sqrtf(deltax * deltax + deltay * deltay);

        // 计算旋转角度的余弦、正弦值。
        cosalpha = deltax / sidelength;
        sinalpha = deltay / sidelength;

        // 根据计算得到的正弦值计算角度，将旋转矩阵信息存入到 Shared Memory
        // 和 Global Memory 参数
        rotateinfo[r].cos = shdcos[0] = cosalpha;
        rotateinfo[r].sin = shdsin[0] = sinalpha;
        rotateinfo[r].radian = shdradian[0] = asin(sinalpha);
    }

    // 同步所有线程，使初始化 Shared Memory 的结果对所有线程可见。
    __syncthreads();

    // 计算当前块内线程的下标。在 x 维上该 Kernel 计算边界值点，
    // 必须以单 Block 运行，避免跨 block 同步引发的同步问题。
    int c = threadIdx.x;

    // 声明包围矩形。
    BoundBox tmpbbox;

    // 当前 Thread 处理的若干个点中找到的局部极值点。初始化。
    tmpbbox.left = tmpbbox.bottom = SDR_LARGE_ENOUGH;
    tmpbbox.right = tmpbbox.top = -SDR_LARGE_ENOUGH;

    // 当前点在新坐标系下的新坐标。
    float curx, cury;

    // 迭代处理该线程所要处理的所有坐标点，这些坐标点是间隔 blockDim.x
    // 个的各个坐标点。
    while (c < convexcst.count) {
        // 从 Global Memory 中读取坐标值，从 Shared Memory 读取旋转信息值，
        // 并计算当前点在新坐标系下的新坐标。
        curx = convexcst.tplData[2 * c] * shdcos[0] +
               convexcst.tplData[2 * c + 1] * shdsin[0];

        cury = convexcst.tplData[2 * c] * (-shdsin[0]) +
               convexcst.tplData[2 * c + 1] * shdcos[0];

        // 判断该坐标值的大小，和已经找到的极值做比较，更新极值。
        tmpbbox.left = min(tmpbbox.left, curx);
        tmpbbox.right = max(tmpbbox.right, curx);
        tmpbbox.bottom = min(tmpbbox.bottom, cury);
        tmpbbox.top = max(tmpbbox.top, cury);

        // 更新 idx，在下一轮迭代时计算下一个点。
        c += blockDim.x;
    }

    // 至此，所有 Thread 都得到了自己的局部极值，现在需要将极值放入 
    // Shared Memory 中，以便下一步进行归约处理。

    // 分配 Shared Memory 给各个指针。
    float *shdbboxleft = shdradian + 1;
    float *shdbboxright = shdbboxleft + blockDim.x;
    float *shdbboxbottom = shdbboxright + blockDim.x;
    float *shdbboxtop = shdbboxbottom + blockDim.x;

    // 将局部结果拷贝到 Shared Memory 中。
    c = threadIdx.x;
    shdbboxleft[c] = tmpbbox.left;
    shdbboxright[c] = tmpbbox.right;
    shdbboxbottom[c] = tmpbbox.bottom;
    shdbboxtop[c] = tmpbbox.top;

    // 同步所有线程，使初始化Shared Memory 的结果对所有线程可见。
    __syncthreads();

    // 下面进行折半归约迭代。这里要求 blockDim.x 必须为 2 的整数次幂。
    int currdsize = blockDim.x / 2;
    // 和当前线程间隔 currdsize 位置处的索引。
    int inidx;
    for (/*currdsize*/; currdsize >= 1; currdsize /= 2) {
        if (c < currdsize) {
            inidx = c + currdsize;
            // 将两个局部结果归约成一个局部结果。
            shdbboxleft[c]   = min(shdbboxleft[c],   shdbboxleft[inidx]);
            shdbboxright[c]  = max(shdbboxright[c],  shdbboxright[inidx]);
            shdbboxbottom[c] = min(shdbboxbottom[c], shdbboxbottom[inidx]);
            shdbboxtop[c]    = max(shdbboxtop[c],    shdbboxtop[inidx]);
        }

        // 同步线程，使本轮迭代归约的结果对所有线程可见。
        __syncthreads();
    }

    // 打印当前的最值点，检查中间结果。
    if (c == 0)
    // 调试打印。
#ifdef SDR_DEBUG_KERNEL_PRINT
    printf("Kernel[computeBdInf]:(%3d, %3d) LRBT (%7.3f,%7.3f,%7.3f,%7.3f)\n",
           r, c, shdbboxleft[c], shdbboxright[c], shdbboxbottom[c],
           shdbboxtop[c]);
#endif

    // 将边界值传递给 Global Memory 参数，每个线程块的第一个线程会进行这个操作。
    if (c == 0) {
        bbox[r].left = shdbboxleft[c];
        bbox[r].right = shdbboxright[c];
        bbox[r].bottom = shdbboxbottom[c];
        bbox[r].top = shdbboxtop[c];
    }
}

// Host 成员方法：sdrComputeBoundInfo（计算新坐标系下凸壳的有向外接矩形的边界信
// 息）
__host__ int SmallestDirRect::sdrComputeBoundInfo(
        CoordiSet *convexcst, RotationInfo rotateinfo[],
        BoundBox bbox[])
{
    // 检查坐标集和旋转矩阵是否为空，若为空则直接返回。
    if (convexcst == NULL || rotateinfo == NULL || bbox == NULL)
        return NULL_POINTER;

    // 如果输入点集中不含有任何的坐标点，则直接退出。
    if (convexcst->count < 1 || convexcst->tplData == NULL)
        return INVALID_DATA;
    
    // 局部变量，错误码。
    int errcode;

    // 将 convexcst 拷贝到 Device 端。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算启动 Kernel 函数所需要的 Block 尺寸与数量。
    size_t blocksize;
    blocksize = DEF_BLOCK_1D;

    size_t gridsize;
    gridsize = DEF_BLOCK_1D;

    // 分配共享内存大小。
    int sharedmemsize = (4 * DEF_BLOCK_1D + 3) * sizeof (float);

    // 启动 Kernel 函数，完成计算。
    _sdrComputeBoundInfoKer<<<gridsize, blocksize, sharedmemsize>>>(
            *convexcst, rotateinfo, bbox);

    // 检查 Kernel 函数执行是否正确。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 运行完毕退出。
    return NO_ERROR;
}

// Host 成员方法：sdrComputeBoundInfoCpu（计算新坐标系下凸壳的有向外接矩形的边界
// 信息）
__host__ int SmallestDirRect::sdrComputeBoundInfoCpu(
        CoordiSet *convexcst, RotationInfo rotateinfo[],
        BoundBox bbox[])
{
    // 检查坐标集和旋转矩阵是否为空，若为空则直接返回。
    if (convexcst == NULL || rotateinfo == NULL || bbox == NULL)
        return NULL_POINTER;

    // 如果输入点集中不含有任何的坐标点，则直接退出。
    if (convexcst->count < 1 || convexcst->tplData == NULL)
        return INVALID_DATA;
    
    // 局部变量，错误码。
    int errcode;

    // 将 convexcst 拷贝到 Host 端。
    errcode = CoordiSetBasicOp::copyToHost(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    int idx;
    int cstidx;
    // 当前凸壳点的下一个点的索引。
    int nextidx;

    // 当前点与下一点的 x、y 坐标差值。
    float deltax, deltay;

    // 当前点与下一点间距离。
    float sidelength;

    // 旋转角度的余弦值和正弦值。
    float cosalpha, sinalpha;

    // 声明包围矩形。
    BoundBox tmpbbox;

    // 当前点在新坐标系下的新坐标。
    float curx, cury;

    for (idx = 0; idx < convexcst->count; idx++) {
        // 当前索引值加 1，求得下一点索引值。
        nextidx = idx + 1;

        // 若当前点为点集中最后一点，则下一点为起始点。
        if (nextidx == convexcst->count)
            nextidx = 0;
    
        // 计算当前点和下一点 x, y 坐标差值。
        deltax = convexcst->tplData[nextidx * 2] -
                 convexcst->tplData[idx * 2];
        deltay = convexcst->tplData[nextidx * 2 + 1] -
                 convexcst->tplData[idx * 2 + 1];

        // 如果解的 x 在第二或者第三象限，转化坐标到第四或者第一象限。
        if (deltax < 0) {
            deltax = -deltax;
            deltay = -deltay;
        }

        // 计算当前点和下一点间距离。
        sidelength = sqrtf(deltax * deltax + deltay * deltay);

        // 计算旋转角度的余弦、正弦值。
        cosalpha = deltax / sidelength;
        sinalpha = deltay / sidelength;

        // 根据计算得到的正弦值计算角度，将旋转矩阵信息存入到参数
        rotateinfo[idx].cos = cosalpha;
        rotateinfo[idx].sin = sinalpha;
        rotateinfo[idx].radian = asin(sinalpha);

        // 每次均初始化。
        tmpbbox.left = tmpbbox.bottom = SDR_LARGE_ENOUGH;
        tmpbbox.right = tmpbbox.top = -SDR_LARGE_ENOUGH;

        for (cstidx = 0; cstidx < convexcst->count; cstidx++) {
            // 读取坐标值和旋转信息值，
            // 并计算当前点在新坐标系下的新坐标。
            curx = convexcst->tplData[2 * cstidx] * rotateinfo[idx].cos +
                   convexcst->tplData[2 * cstidx + 1] * rotateinfo[idx].sin;

            cury = convexcst->tplData[2 * cstidx] * (-rotateinfo[idx].sin) +
                   convexcst->tplData[2 * cstidx + 1] * rotateinfo[idx].cos;

            // 判断该坐标值的大小，和已经找到的极值做比较，更新极值。
            tmpbbox.left = min(tmpbbox.left, curx);
            tmpbbox.right = max(tmpbbox.right, curx);
            tmpbbox.bottom = min(tmpbbox.bottom, cury);
            tmpbbox.top = max(tmpbbox.top, cury);
        }

        // 最值赋值
        bbox[idx].left = tmpbbox.left;
        bbox[idx].right = tmpbbox.right;
        bbox[idx].bottom = tmpbbox.bottom;
        bbox[idx].top = tmpbbox.top;
    }

    // 运行完毕退出。
    return NO_ERROR;
}

// Kernel 函数: _sdrComputeSDRKer（计算包围矩形中面积最小的）
static __global__ void _sdrComputeSDRKer(
        int cstcnt, BoundBox bbox[], int *index)
{
    // 计算当前线程的下标，该 Kernel 必须以单 Block 运行，因此不涉及到 Block 相
    // 关的变量。
    int idx = threadIdx.x;

    // 当前 Thread 处理的若干个矩形中找到的最小的矩形面积。
    float cursdrarea = SDR_LARGE_ENOUGH;

    // 当前线程计算得到的矩形面积。
    float curarea;

    // 当前线程记录的最小矩形面积的索引，初始化为 idx。
    int cursdrindex = idx;

    // 当前线程对应的点计算得到的长宽。
    float length1, length2;

    // 迭代处理该线程所要处理的所有矩形，这些矩形是间隔 blockDim.x 个索引的各个
    // 矩形。
    while (idx < cstcnt) {
        // 从 Global Memory 中读取极值，计算长宽。
        length1 = bbox[idx].right - bbox[idx].left;
        length2 = bbox[idx].top - bbox[idx].bottom;
        // 计算当前的矩形面积。
        curarea = length1 * length2;

        // 判断该面积的大小，和已经找到的最小面积做比较，更新最小面积及索引。
        cursdrindex = (curarea <= cursdrarea) ? idx : cursdrindex;
        cursdrarea = min(curarea, cursdrarea);

        // 更新 idx，在下一轮迭代时计算下一个点。
        idx += blockDim.x;
    }

    // 至此，所有 Thread 都得到了自己的局部最小面积及索引，现在需要将这些点放入 
    // Shared Memory 中，以便下一步进行归约处理。

    // 声明 Shared Memory，并分配各个指针。
    extern __shared__ float shdmem[];
    float *shdarea = shdmem;
    int *shdidx = (int *)(shdarea + blockDim.x);

    // 将局部结果拷贝到 Shared Memory 中。
    idx = threadIdx.x;
    shdarea[idx] = cursdrarea;
    shdidx[idx] = cursdrindex;

    // 同步所有线程，使初始化Shared Memory 的结果对所有线程可见。
    __syncthreads();

    // 下面进行折半归约迭代。这里要求 blockDim.x 必须为 2 的整数次幂。
    int currdsize = blockDim.x / 2;
    // 和当前线程间隔 currdsize 位置处的索引。
    int inidx;
    for (/* currdsize */; currdsize > 0; currdsize >>= 1) {
        if (idx < currdsize) {
            inidx = idx + currdsize;
            // 将两个局部结果归约成一个局部结果。
            shdidx[idx] = (shdarea[idx] <= shdarea[inidx]) ?
                          shdidx[idx] : shdidx[inidx];
            shdarea[idx] = min(shdarea[idx], shdarea[inidx]);

            // 输出结果进行验证。
#ifdef SDR_DEBUG_KERNEL_PRINT
            printf("Kernel[computeSDR]: ReduceSize %3d,"
                   "(%3d) CurSdrArea %7.3f CurSdrId %3d\n",
                   "(%3d) CurReSdrArea %7.3f CurReSdrId %3d\n", 
                   currdsize, idx, shdarea[idx], shdidx[idx],
                   inidx, shdarea[inidx], shdidx[inidx]);
#endif
        }

        // 同步线程，使本轮迭代归约的结果对所有线程可见。
        __syncthreads();
    }

    // 将最小面积的索引传递给 Global Memory 参数，第一个线程会进行这个操作。
    if (idx == 0) {
        index[0] = shdidx[idx];

        // 调试打印。
#ifdef SDR_DEBUG_KERNEL_PRINT
        printf("Kernel[computeSDR]: SDR index %5d\n", index[0]);
#endif
    }

}

// Host 成员方法：sdrComputeSDR（计算有向外接矩形中面积最小的）
__host__ int SmallestDirRect::sdrComputeSDR(
        int cstcnt, BoundBox bbox[], int *index)
{
    // 检查坐标集和旋转矩阵是否为空，若为空则直接返回。
    if (bbox == NULL || index == NULL)
        return NULL_POINTER;

    // 如果点集数量小于 1，直接退出。
    if (cstcnt < 1)
        return INVALID_DATA;

    // 计算启动 Kernel 函数所需要的 Block 尺寸与数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = 1;

    // 共享内存大小。
    int shdmemsize = DEF_BLOCK_1D * (sizeof (float) + sizeof (int));

    // 启动 Kernel 函数，完成计算。
    _sdrComputeSDRKer<<<gridsize, blocksize, shdmemsize>>>(
            cstcnt, bbox, index);

    // 检查 Kernel 函数执行是否正确。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 运行完毕退出。
    return NO_ERROR;
}

// Host 成员方法：sdrComputeSDRCpu（计算有向外接矩形中面积最小的）
__host__ int SmallestDirRect::sdrComputeSDRCpu(
        int cstcnt, BoundBox bbox[], int *index)
{
    // 检查坐标集和旋转矩阵是否为空，若为空则直接返回。
    if (bbox == NULL || index == NULL)
        return NULL_POINTER;

    // 如果点集数量小于 1，直接退出。
    if (cstcnt < 1)
        return INVALID_DATA;

    // 索引，初始化为 0。
    int idx = 0;

    // 当前 Thread 处理的若干个矩形中找到的最小的矩形面积。
    float cursdrarea = SDR_LARGE_ENOUGH;

    // 当前线程计算得到的矩形面积。
    float curarea;

    // 当前线程记录的最小矩形面积的索引，初始化为 idx。
    int cursdrindex = idx;

    // 当前线程对应的点计算得到的长宽。
    float length1, length2;

    for (idx = 0; idx < cstcnt; idx++) {
        // 读取极值，计算长宽。
        length1 = bbox[idx].right - bbox[idx].left;
        length2 = bbox[idx].top - bbox[idx].bottom;
        // 计算当前的矩形面积。
        curarea = length1 * length2;

        // 判断该面积的大小，和已经找到的最小面积做比较，更新最小面积及索引。
        cursdrindex = (curarea <= cursdrarea) ? idx : cursdrindex;
        cursdrarea = min(curarea, cursdrarea);
    }

    // 输出赋值
    index[0] = cursdrindex;

    // 运行完毕退出。
    return NO_ERROR;
}

// 宏：FAIL_SDRPARAMONCVX_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_SDRPARAMONCVX_FREE  do {             \
        if (devtemp != NULL)                      \
            cudaFree(devtemp);                    \
    } while (0)

// Host 成员方法：sdrParamOnConvex（求凸壳点集的最小有向外接矩形的参数）
__host__ int SmallestDirRect::sdrParamOnConvex(
        CoordiSet *convexcst, BoundBox *bbox, RotationInfo *rotinfo)
{
    // 检查输入，输出是否为空。
    if (convexcst == NULL || bbox == NULL || rotinfo == NULL)
        return NULL_POINTER;

    // 如果输入点集中不含有任何的坐标点，则直接退出。
    if (convexcst->count < 1 || convexcst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    cudaError_t cuerrcode;
    int errcode;

    // 用来记录最小有向外接矩形在整个结果中的索引。
    int index = 0;

    // 中间变量的设备端数组。存放旋转矩阵信息，包围盒顶点，索引。
    RotationInfo *rotateinfoDev = NULL;
    BoundBox *bboxDev = NULL;
    int *indexDev = NULL;

    // 中间变量申请 Device 内存空间，并将这些空间分配给各个中间变量。
    float *devtemp = NULL;
    size_t datasize = (sizeof (RotationInfo) + sizeof (BoundBox)) *
                      convexcst->count + sizeof (int);
    cuerrcode = cudaMalloc((void **)&devtemp, datasize);
    if (cuerrcode != cudaSuccess) {
        FAIL_SDRPARAMONCVX_FREE;
        return CUDA_ERROR;
    }

    // 为各个中间变量分配内存空间，采用这种一次申请一个大空间的做法是为了减少申
    // 请内存的开销，同时也减少因内存对齐导致的内存浪费。
    rotateinfoDev = (RotationInfo *)(devtemp);
    bboxDev       = (BoundBox *)(rotateinfoDev + convexcst->count);
    indexDev      = (int *)(bboxDev + convexcst->count);

    // 将输入坐标集拷贝到 device 端。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(convexcst); 
    if (errcode != NO_ERROR) {
        FAIL_SDRPARAMONCVX_FREE;
        return errcode;
    }

    // 调用计算凸壳点集中每相邻两点的旋转矩阵信息,进而计算新坐标系下
    // 凸壳的有向外接矩形的边界信息的函数。
    errcode = this->sdrComputeBoundInfo(convexcst, rotateinfoDev, bboxDev);
    if (errcode != NO_ERROR) {
        FAIL_SDRPARAMONCVX_FREE;
        return errcode;
    }

    // 调用计算最小有向外接矩形的函数。
    errcode = this->sdrComputeSDR(convexcst->count, bboxDev, indexDev);
    if (errcode != NO_ERROR) {
        FAIL_SDRPARAMONCVX_FREE;
        return errcode;
    }

    // 将最小有向外接矩形在所有结果中的索引，拷贝到 host 端。
    cuerrcode = cudaMemcpy(&index, indexDev, sizeof (int),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        FAIL_SDRPARAMONCVX_FREE;
        return CUDA_ERROR;
    }

    // 将最小有向外接矩形的四个顶点，拷贝到主存端。
    cuerrcode = cudaMemcpy(bbox, &bboxDev[index], sizeof (BoundBox),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        FAIL_SDRPARAMONCVX_FREE;
        return CUDA_ERROR;
    }

    // 将最小有向外接矩形的旋转信息，拷贝到主存端。
    cuerrcode = cudaMemcpy(rotinfo, &rotateinfoDev[index],
                           sizeof (RotationInfo),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        FAIL_SDRPARAMONCVX_FREE;
        return CUDA_ERROR;
    }

    // 释放内存
    cudaFree(devtemp);

    // 退出。
    return NO_ERROR;
}
#undef FAIL_SDRPARAMONCVX_FREE

// 宏：FAIL_SDRPARAMONCVXCPU_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_SDRPARAMONCVXCPU_FREE  do {       \
        if (temp != NULL)                      \
            delete temp;                       \
    } while (0)

// Host 成员方法：sdrParamOnConvexCpu（求凸壳点集的最小有向外接矩形的参数）
__host__ int SmallestDirRect::sdrParamOnConvexCpu(
        CoordiSet *convexcst, BoundBox *bbox, RotationInfo *rotinfo)
{
    // 检查输入，输出是否为空。
    if (convexcst == NULL || bbox == NULL || rotinfo == NULL)
        return NULL_POINTER;

    // 如果输入点集中不含有任何的坐标点，则直接退出。
    if (convexcst->count < 1 || convexcst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 用来记录最小有向外接矩形在整个结果中的索引。
    int index = 0;

    // 中间变量的设备端数组。存放旋转矩阵信息，包围盒顶点，索引。
    RotationInfo *rotateinfoHost = NULL;
    BoundBox *bboxHost = NULL;
    int *indexHost = NULL;

    // 中间变量申请 Host 内存空间，并将这些空间分配给各个中间变量。
    float *temp = NULL;
    size_t datasize = (sizeof (RotationInfo) + sizeof (BoundBox)) *
                      convexcst->count + sizeof (int);

    // liuyao debug
    temp = new float[datasize];
    if (temp == NULL) {
        return OUT_OF_MEM;;
    }

    // 为各个中间变量分配内存空间，采用这种一次申请一个大空间的做法是为了减少申
    // 请内存的开销，同时也减少因内存对齐导致的内存浪费。
    rotateinfoHost = (RotationInfo *)(temp);
    bboxHost       = (BoundBox *)(rotateinfoHost + convexcst->count);
    indexHost      = (int *)(bboxHost + convexcst->count);

    // 将输入坐标集拷贝到 Host 端。
    errcode = CoordiSetBasicOp::copyToHost(convexcst); 
    if (errcode != NO_ERROR) {
        FAIL_SDRPARAMONCVXCPU_FREE;
        return errcode;
    }

    // 调用计算凸壳点集中每相邻两点的旋转矩阵信息,进而计算新坐标系下
    // 凸壳的有向外接矩形的边界信息的函数。
    errcode = this->sdrComputeBoundInfoCpu(convexcst, rotateinfoHost, bboxHost);
    if (errcode != NO_ERROR) {
        FAIL_SDRPARAMONCVXCPU_FREE;
        return errcode;
    }

    // 调用计算最小有向外接矩形的函数。
    errcode = this->sdrComputeSDRCpu(convexcst->count, bboxHost, indexHost);
    if (errcode != NO_ERROR) {
        FAIL_SDRPARAMONCVXCPU_FREE;
        return errcode;
    }

    // 输出赋值
    index = indexHost[0];
    bbox[0] = bboxHost[index];
    rotinfo[0] = rotateinfoHost[index];

    // 释放内存
    delete temp;

    // 退出。
    return NO_ERROR;
}
#undef FAIL_SDRPARAMONCVXCPU_FREE

// 宏：FAIL_SDRONCVX_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_SDRONCVX_FREE  do {            \
        if (!hostrect && recthost != NULL)  \
            delete [] recthost;             \
    } while (0)

// Host 成员方法：smallestDirRectOnConvex（求凸壳点集的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRectCpuOnConvex(
        CoordiSet *convexcst, Quadrangle *outrect, bool hostrect)
{
    // 检查输入，输出是否为空。
    if (convexcst == NULL || outrect == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点或者只含 1 个坐标点，则报错退出。
    if (convexcst->count <= 1 || convexcst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    cudaError_t cuerrcode;
    int errcode;

    // 定义 Host 端的输出数组指针，这里计算量比较小，所以统一采取在 host 端
    // 根据最小有向包围矩形的顶点和旋转矩阵信息计算输出的包围矩形各个参数。
    Quadrangle *recthost = NULL;

    // 判断输出矩形是否存储在 Device 端。若不是，则需要在 Host 端为输出矩形
    // 申请一段空间；若该数组是在 Host 端，则直接使用。
    if (hostrect) {
        // 如果在 Host 端，则将指针传给对应的 Host 端统一指针。
        recthost = outrect;
    } else {
        // 为输入数组在 Host 端申请内存。    
        recthost = new Quadrangle[1];
        // 出错则报错返回。
        if (recthost == NULL) {
            return OUT_OF_MEM;
        }

        // 将输出数组拷贝到 Host 端内存。
        cuerrcode = cudaMemcpy(recthost, outrect,
                               sizeof (Quadrangle), 
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SDRONCVX_FREE;
            return cuerrcode;
        }
    }

    // 如果输入凸壳点集中只含有 2 个坐标点，则特殊处理。
    if (convexcst->count == 2) {
        // 四个顶点赋值为两个坐标点的坐标。四个顶点有两对重合。
        recthost->points[0][0] = recthost->points[1][0] = convexcst->tplData[0];
        recthost->points[0][1] = recthost->points[1][1] = convexcst->tplData[1];
        recthost->points[2][0] = recthost->points[3][0] = convexcst->tplData[2];
        recthost->points[2][1] = recthost->points[3][1] = convexcst->tplData[3];

        // 计算两点坐标差值。
        int deltax = convexcst->tplData[0] - convexcst->tplData[2];
        int deltay = convexcst->tplData[1] - convexcst->tplData[3];

        // 如果解的 x 在第二或者第三象限，转化坐标到第四或者第一象限。
        if (deltax < 0) {
            deltax = -deltax;
            deltay = -deltay;
        }

        // 计算当前点和下一点间距离。
        float sidelength = sqrtf(deltax * deltax + deltay * deltay);

        // 计算旋转角度的正弦值。
        float sinalpha = deltay / sidelength;

        // 该角度的弧度值。从而计算最小有向外接矩形的角度。
        float radian = asin(sinalpha);
        recthost->angle = RECT_RAD_TO_DEG(radian);

        // 如果输出矩形在 Device 端，将结果拷贝到输出。
        if (!hostrect) {
            // 将结果从 Host 端内存拷贝到输出。
            cuerrcode = cudaMemcpy(outrect, recthost,
                                   sizeof (Quadrangle),
                                   cudaMemcpyHostToDevice);
            // 出错则释放之前申请的内存。
            if (cuerrcode != cudaSuccess) {
                // 释放之前申请的内存。
                FAIL_SDRONCVX_FREE;
                return cuerrcode;
            }
        }

        // 释放之前申请的 Host 端内存。需要判断输出参数是否在 host 端。
        FAIL_SDRONCVX_FREE;

        // 特殊情况，不用计算下列步骤，退出。
        return NO_ERROR;
    }

    // 局部变量，用来记录面积最小的有向外接矩形和对应的旋转信息。
    BoundBox bbox;
    RotationInfo rotinfo;

    // 将输入坐标集拷贝到 Host 端。
    errcode = CoordiSetBasicOp::copyToHost(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 调用求凸壳点集的最小有向外接矩形参数的函数。
    errcode = this->sdrParamOnConvexCpu(convexcst, &bbox, &rotinfo);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算最小有向外接矩形的角度。
    recthost->angle = RECT_RAD_TO_DEG(rotinfo.radian);

    // 计算最小有向外接矩形的边界点值。
    float points[4][2];
    points[0][0] = bbox.left;
    points[0][1] = bbox.top;
    points[1][0] = bbox.right;
    points[1][1] = bbox.top;
    points[2][0] = bbox.right;
    points[2][1] = bbox.bottom;
    points[3][0] = bbox.left;
    points[3][1] = bbox.bottom;

    // 打印临时顶点信息。
#ifdef SDR_DEBUG_KERNEL_PRINT
    cout << "temprect info: " << endl;
    cout << points[0][0] << "," << points[0][1] << endl;
    cout << points[1][0] << "," << points[1][1] << endl;
    cout << points[2][0] << "," << points[2][1] << endl;
    cout << points[3][0] << "," << points[3][1] << endl;
#endif

    // 临时存放的四个顶点值。
    float tempvertex[4][2];

    // 计算旋转后最小有向外接矩形的四个顶点值。
    RECT_ROTATE_POINT(points[0], tempvertex[0], rotinfo);
    RECT_ROTATE_POINT(points[1], tempvertex[1], rotinfo);
    RECT_ROTATE_POINT(points[2], tempvertex[2], rotinfo);
    RECT_ROTATE_POINT(points[3], tempvertex[3], rotinfo);

    // 求中心坐标。
    float boxcenter[2];
    boxcenter[0] = (tempvertex[0][0] + tempvertex[1][0] + tempvertex[2][0] +
                    tempvertex[3][0]) / 4.0f;
    boxcenter[1] = (tempvertex[0][1] + tempvertex[1][1] + tempvertex[2][1] +
                    tempvertex[3][1]) / 4.0f;

    // 计算所得的包围盒的四个顶点逆时针排列，寻找右上点的索引值。
    int rightupidx;
    // 如果是垂直于坐标轴的菱形，也就是对角的 x 坐标相等，需要特殊处理。
    // 如果第 0 个和第 2 个点的 x 坐标相等。
    if (tempvertex[0][0] == tempvertex[2][0]) {
        // 如果第 0 个的 y 坐标更大。
        if (tempvertex[0][1] > boxcenter[1])
            // 右上点的索引值为 0。
            rightupidx = 0;
        // 如果第 2 个的 y 坐标更大。
        else
            // 右上点的索引值为 2。
            rightupidx = 2;
    // 如果第 1 个和第 3 个点的 x 坐标相等。
    } else if (tempvertex[1][0] == tempvertex[3][0]) {
        // 如果第 1 个的 y 坐标更大。
        if (tempvertex[1][1] > boxcenter[1])
            // 右上点的索引值为 1。
            rightupidx = 1;
        // 如果第 3 个的 y 坐标更大。
        else
            // 右上点的索引值为 3。
            rightupidx = 3;
    // 如果没有 x 或者 y 坐标相等的特殊情况。
    } else {
        // 如果第 0 个点的 x，y 坐标均大于中心点坐标。
        if (tempvertex[0][0] > boxcenter[0] && tempvertex[0][1] > boxcenter[1])
            // 右上点的索引值为 0。
            rightupidx = 0;
        // 如果第 1 个点的 x，y 坐标均大于中心点坐标。
        else if (tempvertex[1][0] > boxcenter[0] &&
                 tempvertex[1][1] > boxcenter[1])
            // 右上点的索引值为 1。
            rightupidx = 1;
        // 如果第 2 个点的 x，y 坐标均大于中心点坐标。
        else if (tempvertex[2][0] > boxcenter[0] &&
                 tempvertex[2][1] > boxcenter[1])
            // 右上点的索引值为 2。
            rightupidx = 2;
        // 如果第 3 个点的 x，y 坐标均大于中心点坐标。
        else
            // 右上点的索引值为 3。
            rightupidx = 3;
    }

    // 按照算得的右上点索引值，对四个顶点的 x，y 坐标进行分别的向下向上取整处理
    // 右上点，x 向上取整，y 向上取整。
    recthost->points[rightupidx][0] =
            (int)ceil(tempvertex[rightupidx][0]);
    recthost->points[rightupidx][1] =
            (int)ceil(tempvertex[rightupidx][1]);
    // 右下点，x 向上取整，y 向下取整。
    recthost->points[(rightupidx + 1) % 4][0] =
            (int)ceil(tempvertex[(rightupidx + 1) % 4][0]);
    recthost->points[(rightupidx + 1) % 4][1] =
            (int)floor(tempvertex[(rightupidx + 1) % 4][1]);
    // 左下点，x 向下取整，y 向下取整。
    recthost->points[(rightupidx + 2) % 4][0] =
            (int)floor(tempvertex[(rightupidx + 2) % 4][0]);
    recthost->points[(rightupidx + 2) % 4][1] =
            (int)floor(tempvertex[(rightupidx + 2) % 4][1]);
    // 左上点，x 向下取整，y 向上取整。
    recthost->points[(rightupidx + 3) % 4][0] =
            (int)ceil(tempvertex[(rightupidx + 3) % 4][0]);
    recthost->points[(rightupidx + 3) % 4][1] =
            (int)floor(tempvertex[(rightupidx + 3) % 4][1]);

    // 计算矩形的长宽。
    float length1 = bbox.right - bbox.left;
    float length2 = bbox.top - bbox.bottom;

    // 角度是跟 length1 边平行的，需求为角度的方向平行于长边。当 length1 不是
    // 长边时，做出调整。
    if (length1 < length2) {
        // 旋转角度为负时，加上 90 度。
        if (recthost->angle < 0.0f)
            recthost->angle += 90.0f;
        // 旋转角度为正时，减去 90 度。
        else
            recthost->angle -= 90.0f;
    }

    // 如果输出矩形在 Device 端，将结果拷贝到输出。
    if (!hostrect) {
        // 将结果从 Host 端内存拷贝到输出。
        cuerrcode = cudaMemcpy(outrect, recthost,
                               sizeof (Quadrangle),
                               cudaMemcpyHostToDevice);
        // 出错则释放之前申请的内存。
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SDRONCVX_FREE;
            return cuerrcode;
        }
    }

    // 释放之前申请的 Host 端内存。需要判断输出参数是否在 host 端。
    FAIL_SDRONCVX_FREE;

    // 退出。
    return NO_ERROR;
}

// Host 成员方法：smallestDirRectOnConvex（求凸壳点集的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRectOnConvex(
        CoordiSet *convexcst, Quadrangle *outrect, bool hostrect)
{
    // 检查输入，输出是否为空。
    if (convexcst == NULL || outrect == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点或者只含 1 个坐标点，则报错退出。
    if (convexcst->count <= 1 || convexcst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    cudaError_t cuerrcode;
    int errcode;

    // 定义 Host 端的输出数组指针，这里计算量比较小，所以统一采取在 host 端
    // 根据最小有向包围矩形的顶点和旋转矩阵信息计算输出的包围矩形各个参数。
    Quadrangle *recthost = NULL;

    // 判断输出矩形是否存储在 Device 端。若不是，则需要在 Host 端为输出矩形
    // 申请一段空间；若该数组是在 Host 端，则直接使用。
    if (hostrect) {
        // 如果在 Host 端，则将指针传给对应的 Host 端统一指针。
        recthost = outrect;
    } else {
        // 为输入数组在 Host 端申请内存。    
        recthost = new Quadrangle[1];
        // 出错则报错返回。
        if (recthost == NULL) {
            return OUT_OF_MEM;
        }

        // 将输出数组拷贝到 Host 端内存。
        cuerrcode = cudaMemcpy(recthost, outrect,
                               sizeof (Quadrangle), 
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SDRONCVX_FREE;
            return cuerrcode;
        }
    }

    // 如果输入凸壳点集中只含有 2 个坐标点，则特殊处理。
    if (convexcst->count == 2) {
        // 四个顶点赋值为两个坐标点的坐标。四个顶点有两对重合。
        recthost->points[0][0] = recthost->points[1][0] = convexcst->tplData[0];
        recthost->points[0][1] = recthost->points[1][1] = convexcst->tplData[1];
        recthost->points[2][0] = recthost->points[3][0] = convexcst->tplData[2];
        recthost->points[2][1] = recthost->points[3][1] = convexcst->tplData[3];

        // 计算两点坐标差值。
        int deltax = convexcst->tplData[0] - convexcst->tplData[2];
        int deltay = convexcst->tplData[1] - convexcst->tplData[3];

        // 如果解的 x 在第二或者第三象限，转化坐标到第四或者第一象限。
        if (deltax < 0) {
            deltax = -deltax;
            deltay = -deltay;
        }

        // 计算当前点和下一点间距离。
        float sidelength = sqrtf(deltax * deltax + deltay * deltay);

        // 计算旋转角度的正弦值。
        float sinalpha = deltay / sidelength;

        // 该角度的弧度值。从而计算最小有向外接矩形的角度。
        float radian = asin(sinalpha);
        recthost->angle = RECT_RAD_TO_DEG(radian);

        // 如果输出矩形在 Device 端，将结果拷贝到输出。
        if (!hostrect) {
            // 将结果从 Host 端内存拷贝到输出。
            cuerrcode = cudaMemcpy(outrect, recthost,
                                   sizeof (Quadrangle),
                                   cudaMemcpyHostToDevice);
            // 出错则释放之前申请的内存。
            if (cuerrcode != cudaSuccess) {
                // 释放之前申请的内存。
                FAIL_SDRONCVX_FREE;
                return cuerrcode;
            }
        }

        // 释放之前申请的 Host 端内存。需要判断输出参数是否在 host 端。
        FAIL_SDRONCVX_FREE;

        // 特殊情况，不用计算下列步骤，退出。
        return NO_ERROR;
    }

    // 局部变量，用来记录面积最小的有向外接矩形和对应的旋转信息。
    BoundBox bbox;
    RotationInfo rotinfo;

    // 将输入坐标集拷贝到 device 端。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 调用求凸壳点集的最小有向外接矩形参数的函数。
    errcode = this->sdrParamOnConvex(convexcst, &bbox, &rotinfo);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算最小有向外接矩形的角度。
    recthost->angle = RECT_RAD_TO_DEG(rotinfo.radian);

    // 计算最小有向外接矩形的边界点值。
    float points[4][2];
    points[0][0] = bbox.left;
    points[0][1] = bbox.top;
    points[1][0] = bbox.right;
    points[1][1] = bbox.top;
    points[2][0] = bbox.right;
    points[2][1] = bbox.bottom;
    points[3][0] = bbox.left;
    points[3][1] = bbox.bottom;

    // 打印临时顶点信息。
#ifdef SDR_DEBUG_KERNEL_PRINT
    cout << "temprect info: " << endl;
    cout << points[0][0] << "," << points[0][1] << endl;
    cout << points[1][0] << "," << points[1][1] << endl;
    cout << points[2][0] << "," << points[2][1] << endl;
    cout << points[3][0] << "," << points[3][1] << endl;
#endif

    // 临时存放的四个顶点值。
    float tempvertex[4][2];

    // 计算旋转后最小有向外接矩形的四个顶点值。
    RECT_ROTATE_POINT(points[0], tempvertex[0], rotinfo);
    RECT_ROTATE_POINT(points[1], tempvertex[1], rotinfo);
    RECT_ROTATE_POINT(points[2], tempvertex[2], rotinfo);
    RECT_ROTATE_POINT(points[3], tempvertex[3], rotinfo);

    // 求中心坐标。
    float boxcenter[2];
    boxcenter[0] = (tempvertex[0][0] + tempvertex[1][0] + tempvertex[2][0] +
                    tempvertex[3][0]) / 4.0f;
    boxcenter[1] = (tempvertex[0][1] + tempvertex[1][1] + tempvertex[2][1] +
                    tempvertex[3][1]) / 4.0f;

    // 计算所得的包围盒的四个顶点逆时针排列，寻找右上点的索引值。
    int rightupidx;
    // 如果是垂直于坐标轴的菱形，也就是对角的 x 坐标相等，需要特殊处理。
    // 如果第 0 个和第 2 个点的 x 坐标相等。
    if (tempvertex[0][0] == tempvertex[2][0]) {
        // 如果第 0 个的 y 坐标更大。
        if (tempvertex[0][1] > boxcenter[1])
            // 右上点的索引值为 0。
            rightupidx = 0;
        // 如果第 2 个的 y 坐标更大。
        else
            // 右上点的索引值为 2。
            rightupidx = 2;
    // 如果第 1 个和第 3 个点的 x 坐标相等。
    } else if (tempvertex[1][0] == tempvertex[3][0]) {
        // 如果第 1 个的 y 坐标更大。
        if (tempvertex[1][1] > boxcenter[1])
            // 右上点的索引值为 1。
            rightupidx = 1;
        // 如果第 3 个的 y 坐标更大。
        else
            // 右上点的索引值为 3。
            rightupidx = 3;
    // 如果没有 x 或者 y 坐标相等的特殊情况。
    } else {
        // 如果第 0 个点的 x，y 坐标均大于中心点坐标。
        if (tempvertex[0][0] > boxcenter[0] && tempvertex[0][1] > boxcenter[1])
            // 右上点的索引值为 0。
            rightupidx = 0;
        // 如果第 1 个点的 x，y 坐标均大于中心点坐标。
        else if (tempvertex[1][0] > boxcenter[0] &&
                 tempvertex[1][1] > boxcenter[1])
            // 右上点的索引值为 1。
            rightupidx = 1;
        // 如果第 2 个点的 x，y 坐标均大于中心点坐标。
        else if (tempvertex[2][0] > boxcenter[0] &&
                 tempvertex[2][1] > boxcenter[1])
            // 右上点的索引值为 2。
            rightupidx = 2;
        // 如果第 3 个点的 x，y 坐标均大于中心点坐标。
        else
            // 右上点的索引值为 3。
            rightupidx = 3;
    }

    // 按照算得的右上点索引值，对四个顶点的 x，y 坐标进行分别的向下向上取整处理
    // 右上点，x 向上取整，y 向上取整。
    recthost->points[rightupidx][0] =
            (int)ceil(tempvertex[rightupidx][0]);
    recthost->points[rightupidx][1] =
            (int)ceil(tempvertex[rightupidx][1]);
    // 右下点，x 向上取整，y 向下取整。
    recthost->points[(rightupidx + 1) % 4][0] =
            (int)ceil(tempvertex[(rightupidx + 1) % 4][0]);
    recthost->points[(rightupidx + 1) % 4][1] =
            (int)floor(tempvertex[(rightupidx + 1) % 4][1]);
    // 左下点，x 向下取整，y 向下取整。
    recthost->points[(rightupidx + 2) % 4][0] =
            (int)floor(tempvertex[(rightupidx + 2) % 4][0]);
    recthost->points[(rightupidx + 2) % 4][1] =
            (int)floor(tempvertex[(rightupidx + 2) % 4][1]);
    // 左上点，x 向下取整，y 向上取整。
    recthost->points[(rightupidx + 3) % 4][0] =
            (int)ceil(tempvertex[(rightupidx + 3) % 4][0]);
    recthost->points[(rightupidx + 3) % 4][1] =
            (int)floor(tempvertex[(rightupidx + 3) % 4][1]);

    // 计算矩形的长宽。
    float length1 = bbox.right - bbox.left;
    float length2 = bbox.top - bbox.bottom;

    // 角度是跟 length1 边平行的，需求为角度的方向平行于长边。当 length1 不是
    // 长边时，做出调整。
    if (length1 < length2) {
        // 旋转角度为负时，加上 90 度。
        if (recthost->angle < 0.0f)
            recthost->angle += 90.0f;
        // 旋转角度为正时，减去 90 度。
        else
            recthost->angle -= 90.0f;
    }

    // 如果输出矩形在 Device 端，将结果拷贝到输出。
    if (!hostrect) {
        // 将结果从 Host 端内存拷贝到输出。
        cuerrcode = cudaMemcpy(outrect, recthost,
                               sizeof (Quadrangle),
                               cudaMemcpyHostToDevice);
        // 出错则释放之前申请的内存。
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SDRONCVX_FREE;
            return cuerrcode;
        }
    }

    // 释放之前申请的 Host 端内存。需要判断输出参数是否在 host 端。
    FAIL_SDRONCVX_FREE;

    // 退出。
    return NO_ERROR;
}

// Host 成员方法：smallestDirRectCpuOnConvex（求凸壳点集的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRectCpuOnConvex(
        CoordiSet *convexcst, DirectedRect *outrect, bool hostrect)
{
    // 检查输入，输出是否为空。
    if (convexcst == NULL || outrect == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点或者只含 1 个坐标点，则报错退出。
    if (convexcst->count <= 1 || convexcst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    cudaError_t cuerrcode;
    int errcode;

    // 定义 Host 端的输出数组指针，这里计算量比较小，所以统一采取在 host 端
    // 根据最小有向包围矩形的顶点和旋转矩阵信息计算输出的包围矩形各个参数。
    DirectedRect *recthost = NULL;

    // 判断输出矩形是否存储在 Device 端。若不是，则需要在 Host 端为输出矩形
    // 申请一段空间；若该数组是在 Host 端，则直接使用。
    if (hostrect) {
        // 如果在 Host 端，则将指针传给对应的 Host 端统一指针。
        recthost = outrect;
    } else {
        // 为输入数组在 Host 端申请内存。    
        recthost = new DirectedRect[1];
        // 出错则报错返回。
        if (recthost == NULL) {
            return OUT_OF_MEM;
        }

        // 将输出数组拷贝到 Host 端内存。
        cuerrcode = cudaMemcpy(recthost, outrect,
                               sizeof (DirectedRect), 
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SDRONCVX_FREE;
            return cuerrcode;
        }
    }

    // 如果输入凸壳点集中只含有 2 个坐标点，则特殊处理。
    if (convexcst->count == 2) {
        // 中心点坐标为两个点的中点。
        recthost->centerPoint[0] = convexcst->tplData[0] +
                                   convexcst->tplData[2];
        recthost->centerPoint[1] = convexcst->tplData[1] +
                                   convexcst->tplData[3];

        // 计算两点坐标差值。
        int deltax = convexcst->tplData[0] - convexcst->tplData[2];
        int deltay = convexcst->tplData[1] - convexcst->tplData[3];

        // 如果解的 x 在第二或者第三象限，转化坐标到第四或者第一象限。
        if (deltax < 0) {
            deltax = -deltax;
            deltay = -deltay;
        }

        // 计算当前点和下一点间距离。
        float sidelength = sqrtf(deltax * deltax + deltay * deltay);

        // 计算旋转角度的正弦值。
        float sinalpha = deltay / sidelength;

        // 该角度的弧度值。从而计算最小有向外接矩形的角度。
        float radian = asin(sinalpha);
        recthost->angle = RECT_RAD_TO_DEG(radian);

        // 该包围矩形的边长。
        recthost->length1 = (int)sidelength;
        recthost->length2 = 0;

        // 如果输出矩形在 Device 端，将结果拷贝到输出。
        if (!hostrect) {
            // 将结果从 Host 端内存拷贝到输出。
            cuerrcode = cudaMemcpy(outrect, recthost,
                                   sizeof (DirectedRect),
                                   cudaMemcpyHostToDevice);
            // 出错则释放之前申请的内存。
            if (cuerrcode != cudaSuccess) {
                // 释放之前申请的内存。
                FAIL_SDRONCVX_FREE;
                return cuerrcode;
            }
        }

        // 释放之前申请的 Host 端内存。需要判断输出参数是否在 host 端。
        FAIL_SDRONCVX_FREE;

        // 特殊情况，不用计算下列步骤，退出。
        return NO_ERROR;
    }

    // 局部变量，用来记录面积最小的有向外接矩形和对应的旋转信息。
    BoundBox bbox;
    RotationInfo rotinfo;

    // 将输入坐标集拷贝到 Host 端。
    errcode = CoordiSetBasicOp::copyToHost(convexcst); 
    if (errcode != NO_ERROR)
        return errcode;

    // 调用求凸壳点集的最小有向外接矩形参数的函数。
    errcode = this->sdrParamOnConvexCpu(convexcst, &bbox, &rotinfo);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算最小有向外接矩形的角度。
    recthost->angle = RECT_RAD_TO_DEG(rotinfo.radian);

    // 计算中心坐标。
    float boxcenter[2];
    boxcenter[0] = (bbox.left + bbox.right) / 2.0f;
    boxcenter[1] = (bbox.top + bbox.bottom) / 2.0f;
    RECT_ROTATE_POINT(boxcenter, recthost->centerPoint, rotinfo);

    // 计算矩形的长宽。
    recthost->length1 = (int)(bbox.right - bbox.left);
    recthost->length2 = (int)(bbox.top - bbox.bottom);

    // 选择长的作为矩形的长。
    if (recthost->length1 < recthost->length2) {
        // 长短边进行交换。
        int length_temp;
        length_temp = recthost->length1;
        recthost->length1 = recthost->length2;
        recthost->length2 = length_temp;

        // 角度是跟 length1 边平行的，需求为角度的方向平行于长边。当 length1
        // 不是长边时，做出调整。
        // 旋转角度为负时，加上 90 度。
        if (recthost->angle < 0.0f)
            recthost->angle += 90.0f;
        // 旋转角度为正时，减去 90 度。
        else
            recthost->angle -= 90.0f;
    }

    // 如果输出矩形在 Device 端，将结果拷贝到输出。
    if (!hostrect) {
        // 将结果从 Host 端内存拷贝到输出。
        cuerrcode = cudaMemcpy(outrect, recthost,
                               sizeof (DirectedRect),
                               cudaMemcpyHostToDevice);
        // 出错则释放之前申请的内存。
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SDRONCVX_FREE;
            return cuerrcode;
        }
    }

    // 释放之前申请的 Host 端内存。需要判断输出参数是否在 host 端。
    FAIL_SDRONCVX_FREE;

    // 退出。
    return NO_ERROR;
}

// Host 成员方法：smallestDirRectOnConvex（求凸壳点集的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRectOnConvex(
        CoordiSet *convexcst, DirectedRect *outrect, bool hostrect)
{
    // 检查输入，输出是否为空。
    if (convexcst == NULL || outrect == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点或者只含 1 个坐标点，则报错退出。
    if (convexcst->count <= 1 || convexcst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    cudaError_t cuerrcode;
    int errcode;

    // 定义 Host 端的输出数组指针，这里计算量比较小，所以统一采取在 host 端
    // 根据最小有向包围矩形的顶点和旋转矩阵信息计算输出的包围矩形各个参数。
    DirectedRect *recthost = NULL;

    // 判断输出矩形是否存储在 Device 端。若不是，则需要在 Host 端为输出矩形
    // 申请一段空间；若该数组是在 Host 端，则直接使用。
    if (hostrect) {
        // 如果在 Host 端，则将指针传给对应的 Host 端统一指针。
        recthost = outrect;
    } else {
        // 为输入数组在 Host 端申请内存。    
        recthost = new DirectedRect[1];
        // 出错则报错返回。
        if (recthost == NULL) {
            return OUT_OF_MEM;
        }

        // 将输出数组拷贝到 Host 端内存。
        cuerrcode = cudaMemcpy(recthost, outrect,
                               sizeof (DirectedRect), 
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SDRONCVX_FREE;
            return cuerrcode;
        }
    }

    // 如果输入凸壳点集中只含有 2 个坐标点，则特殊处理。
    if (convexcst->count == 2) {
        // 中心点坐标为两个点的中点。
        recthost->centerPoint[0] = convexcst->tplData[0] +
                                   convexcst->tplData[2];
        recthost->centerPoint[1] = convexcst->tplData[1] +
                                   convexcst->tplData[3];

        // 计算两点坐标差值。
        int deltax = convexcst->tplData[0] - convexcst->tplData[2];
        int deltay = convexcst->tplData[1] - convexcst->tplData[3];

        // 如果解的 x 在第二或者第三象限，转化坐标到第四或者第一象限。
        if (deltax < 0) {
            deltax = -deltax;
            deltay = -deltay;
        }

        // 计算当前点和下一点间距离。
        float sidelength = sqrtf(deltax * deltax + deltay * deltay);

        // 计算旋转角度的正弦值。
        float sinalpha = deltay / sidelength;

        // 该角度的弧度值。从而计算最小有向外接矩形的角度。
        float radian = asin(sinalpha);
        recthost->angle = RECT_RAD_TO_DEG(radian);

        // 该包围矩形的边长。
        recthost->length1 = (int)sidelength;
        recthost->length2 = 0;

        // 如果输出矩形在 Device 端，将结果拷贝到输出。
        if (!hostrect) {
            // 将结果从 Host 端内存拷贝到输出。
            cuerrcode = cudaMemcpy(outrect, recthost,
                                   sizeof (DirectedRect),
                                   cudaMemcpyHostToDevice);
            // 出错则释放之前申请的内存。
            if (cuerrcode != cudaSuccess) {
                // 释放之前申请的内存。
                FAIL_SDRONCVX_FREE;
                return cuerrcode;
            }
        }

        // 释放之前申请的 Host 端内存。需要判断输出参数是否在 host 端。
        FAIL_SDRONCVX_FREE;

        // 特殊情况，不用计算下列步骤，退出。
        return NO_ERROR;
    }

    // 局部变量，用来记录面积最小的有向外接矩形和对应的旋转信息。
    BoundBox bbox;
    RotationInfo rotinfo;

    // 将输入坐标集拷贝到 device 端。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(convexcst); 
    if (errcode != NO_ERROR)
        return errcode;

    // 调用求凸壳点集的最小有向外接矩形参数的函数。
    errcode = this->sdrParamOnConvex(convexcst, &bbox, &rotinfo);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算最小有向外接矩形的角度。
    recthost->angle = RECT_RAD_TO_DEG(rotinfo.radian);

    // 计算中心坐标。
    float boxcenter[2];
    boxcenter[0] = (bbox.left + bbox.right) / 2.0f;
    boxcenter[1] = (bbox.top + bbox.bottom) / 2.0f;
    RECT_ROTATE_POINT(boxcenter, recthost->centerPoint, rotinfo);

    // 计算矩形的长宽。
    recthost->length1 = (int)(bbox.right - bbox.left);
    recthost->length2 = (int)(bbox.top - bbox.bottom);

    // 选择长的作为矩形的长。
    if (recthost->length1 < recthost->length2) {
        // 长短边进行交换。
        int length_temp;
        length_temp = recthost->length1;
        recthost->length1 = recthost->length2;
        recthost->length2 = length_temp;

        // 角度是跟 length1 边平行的，需求为角度的方向平行于长边。当 length1
        // 不是长边时，做出调整。
        // 旋转角度为负时，加上 90 度。
        if (recthost->angle < 0.0f)
            recthost->angle += 90.0f;
        // 旋转角度为正时，减去 90 度。
        else
            recthost->angle -= 90.0f;
    }

    // 如果输出矩形在 Device 端，将结果拷贝到输出。
    if (!hostrect) {
        // 将结果从 Host 端内存拷贝到输出。
        cuerrcode = cudaMemcpy(outrect, recthost,
                               sizeof (DirectedRect),
                               cudaMemcpyHostToDevice);
        // 出错则释放之前申请的内存。
        if (cuerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            FAIL_SDRONCVX_FREE;
            return cuerrcode;
        }
    }

    // 释放之前申请的 Host 端内存。需要判断输出参数是否在 host 端。
    FAIL_SDRONCVX_FREE;

    // 退出。
    return NO_ERROR;
}
#undef FAIL_SDRONCVX_FREE

// 宏：FAIL_SDRONCST_FREE
// 该宏用于完成下面函数运行出现错误退出前的内存清理工作。
#define FAIL_SDRONCST_FREE  do {                       \
        if (convexcst != NULL)                         \
        CoordiSetBasicOp::deleteCoordiSet(convexcst);  \
    } while (0)
// Host 成员方法：smallestDirRectCpu（求给定点集的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRectCpu(
        CoordiSet *cst, Quadrangle *outrect, bool hostrect)
{
    // 检查输入，输出是否为空。
    if (cst == NULL || outrect == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点或者只含 1 个坐标点，则报错退出。
    if (cst->count <= 1 || cst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 凸壳点集。
    CoordiSet *convexcst;

    // 创建凸壳点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 宏：SDR_USE_CPU_CONVEXHULL
    // 该开关宏用于指示是否在后续步骤中使用 CPU 版本的 ConvexHull 函数。 
#define SDR_USE_CPU_CONVEXHULL

    // 初始化 LABEL 数组。
#ifdef SDR_USE_CPU_CONVEXHULL
    // 给该凸壳点集开辟合适的内存空间。
    //errcode = CoordiSetBasicOp::makeAtCurrentDevice(convexcst, cst->count);
    errcode = CoordiSetBasicOp::makeAtHost(convexcst, cst->count);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 调用求凸壳的函数。
    errcode = this->cvHull.convexHullCpu(cst, convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }
#else
    // 给该凸壳点集开辟合适的内存空间。
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(convexcst, cst->count);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 调用求凸壳的函数。GPU 版本
    errcode = this->cvHull.convexHull(cst, convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }
#endif
#undef SDR_USE_CPU_CONVEXHULL
    // 调用求给定凸壳点集的最小有向外接矩形的函数。
    errcode = smallestDirRectCpuOnConvex(convexcst, outrect, hostrect);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 清除凸壳点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(convexcst);

    // 退出。
    return NO_ERROR;
}

// Host 成员方法：smallestDirRect（求给定点集的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRect(
        CoordiSet *cst, Quadrangle *outrect, bool hostrect)
{
    // 检查输入，输出是否为空。
    if (cst == NULL || outrect == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点或者只含 1 个坐标点，则报错退出。
    if (cst->count <= 1 || cst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 凸壳点集。
    CoordiSet *convexcst;

    // 创建凸壳点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 宏：SDR_USE_CPU_CONVEXHULL
    // 该开关宏用于指示是否在后续步骤中使用 CPU 版本的 ConvexHull 函数。 
//#define SDR_USE_CPU_CONVEXHULL

    // 初始化 LABEL 数组。
#ifdef SDR_USE_CPU_CONVEXHULL
    // 给该凸壳点集开辟合适的内存空间。
    //errcode = CoordiSetBasicOp::makeAtCurrentDevice(convexcst, cst->count);
    errcode = CoordiSetBasicOp::makeAtHost(convexcst, cst->count);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 调用求凸壳的函数。
    errcode = this->cvHull.convexHullCpu(cst, convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }
#else
    // 给该凸壳点集开辟合适的内存空间。
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(convexcst, cst->count);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 调用求凸壳的函数。GPU 版本
    errcode = this->cvHull.convexHull(cst, convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }
#endif
#undef SDR_USE_CPU_CONVEXHULL
    // 调用求给定凸壳点集的最小有向外接矩形的函数。
    errcode = smallestDirRectOnConvex(convexcst, outrect, hostrect);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 清除凸壳点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(convexcst);

    // 退出。
    return NO_ERROR;
}

// Host 成员方法：smallestDirRectCpu（求给定点集的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRectCpu(
        CoordiSet *cst, DirectedRect *outrect, bool hostrect)
{
    // 检查输入，输出是否为空。
    if (cst == NULL || outrect == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点或者只含 1 个坐标点，则报错退出。
    if (cst->count <= 1 || cst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 凸壳点集。
    CoordiSet *convexcst;

    // 创建凸壳点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 宏：SDR_USE_CPU_CONVEXHULL
    // 该开关宏用于指示是否在后续步骤中使用 CPU 版本的 ConvexHull 函数。 
//#define SDR_USE_CPU_CONVEXHULL

    // 初始化 LABEL 数组。
#ifdef SDR_USE_CPU_CONVEXHULL
    // 给该凸壳点集开辟合适的内存空间。
    //errcode = CoordiSetBasicOp::makeAtCurrentDevice(convexcst, cst->count);
    errcode = CoordiSetBasicOp::makeAtHost(convexcst, cst->count);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 调用求凸壳的函数。
    errcode = this->cvHull.convexHullCpu(cst, convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }
#else
    // 给该凸壳点集开辟合适的内存空间。
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(convexcst, cst->count);
    //errcode = CoordiSetBasicOp::makeAtHost(convexcst, cst->count);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 调用求凸壳的函数。
    errcode = this->cvHull.convexHull(cst, convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }
#endif
#undef SDR_USE_CPU_CONVEXHULL

    // 调用求给定凸壳点集的最小有向外接矩形的函数。
    errcode = smallestDirRectCpuOnConvex(convexcst, outrect, hostrect);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 清除凸壳点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(convexcst);

    // 退出。
    return NO_ERROR;
}

// Host 成员方法：smallestDirRect（求给定点集的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRect(
        CoordiSet *cst, DirectedRect *outrect, bool hostrect)
{
    // 检查输入，输出是否为空。
    if (cst == NULL || outrect == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点或者只含 1 个坐标点，则报错退出。
    if (cst->count <= 1 || cst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 凸壳点集。
    CoordiSet *convexcst;

    // 创建凸壳点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 宏：SDR_USE_CPU_CONVEXHULL
    // 该开关宏用于指示是否在后续步骤中使用 CPU 版本的 ConvexHull 函数。 
//#define SDR_USE_CPU_CONVEXHULL

    // 初始化 LABEL 数组。
#ifdef SDR_USE_CPU_CONVEXHULL
    // 给该凸壳点集开辟合适的内存空间。
    //errcode = CoordiSetBasicOp::makeAtCurrentDevice(convexcst, cst->count);
    errcode = CoordiSetBasicOp::makeAtHost(convexcst, cst->count);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 调用求凸壳的函数。
    errcode = this->cvHull.convexHullCpu(cst, convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }
#else
    // 给该凸壳点集开辟合适的内存空间。
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(convexcst, cst->count);
    //errcode = CoordiSetBasicOp::makeAtHost(convexcst, cst->count);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 调用求凸壳的函数。
    errcode = this->cvHull.convexHull(cst, convexcst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }
#endif
#undef SDR_USE_CPU_CONVEXHULL

    // 调用求给定凸壳点集的最小有向外接矩形的函数。
    errcode = smallestDirRectOnConvex(convexcst, outrect, hostrect);
    if (errcode != NO_ERROR) {
        FAIL_SDRONCST_FREE;
        return errcode;
    }

    // 清除凸壳点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(convexcst);

    // 退出。
    return NO_ERROR;
}
#undef FAIL_SDRONCST_FREE

// 宏：FAIL_SDRONIMG_FREE
// 该宏用于完成下面函数运行出现错误退出前的内存清理工作。
#define FAIL_SDRONIMG_FREE  do {                 \
        if (cst != NULL)                         \
        CoordiSetBasicOp::deleteCoordiSet(cst);  \
    } while (0)

// Host 成员方法：smallestDirRectCpu（求像素值给定的对象的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRectCpu(
        Image *inimg, Quadrangle *outrect, bool hostrect)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outrect == NULL)
        return NULL_POINTER;

    // 局部变量，错误码。
    int errcode;

    // 新建点集。
    CoordiSet *cst;

    // 构造点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&cst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 调用图像转点集的函数。
    errcode = this->imgCvt.imgConvertToCst(inimg, cst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 调用求给定凸壳点集的最小有向外接矩形的函数。
    errcode = smallestDirRectCpu(cst, outrect, hostrect);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 清除点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(cst);

    // 退出。
    return NO_ERROR;
}

__host__ int SmallestDirRect::smallestDirRect(
        Image *inimg, Quadrangle *outrect, bool hostrect)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outrect == NULL)
        return NULL_POINTER;

    // 局部变量，错误码。
    int errcode;

    // 新建点集。
    CoordiSet *cst;

    // 构造点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&cst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 调用图像转点集的函数。
    errcode = this->imgCvt.imgConvertToCst(inimg, cst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 调用求给定凸壳点集的最小有向外接矩形的函数。
    errcode = smallestDirRect(cst, outrect, hostrect);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 清除点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(cst);

    // 退出。
    return NO_ERROR;
}

// Host 成员方法：smallestDirRectCpu（求像素值给定的对象的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRectCpu(
        Image *inimg, DirectedRect *outrect, bool hostrect)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outrect == NULL)
        return NULL_POINTER;

    // 局部变量，错误码。
    int errcode;

    // 新建点集。
    CoordiSet *cst;

    // 构造点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&cst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 调用图像转点集的函数。
    errcode = this->imgCvt.imgConvertToCst(inimg, cst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 调用求给定凸壳点集的最小有向外接矩形的函数。
    errcode = smallestDirRectCpu(cst, outrect, hostrect);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 清除点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(cst);

    // 退出。
    return NO_ERROR;
}

// Host 成员方法：smallestDirRect（求像素值给定的对象的最小有向外接矩形）
__host__ int SmallestDirRect::smallestDirRect(
        Image *inimg, DirectedRect *outrect, bool hostrect)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outrect == NULL)
        return NULL_POINTER;

    // 局部变量，错误码。
    int errcode;

    // 新建点集。
    CoordiSet *cst;

    // 构造点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&cst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 调用图像转点集的函数。
    errcode = this->imgCvt.imgConvertToCst(inimg, cst);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 调用求给定凸壳点集的最小有向外接矩形的函数。
    errcode = smallestDirRect(cst, outrect, hostrect);
    if (errcode != NO_ERROR) {
        FAIL_SDRONIMG_FREE;
        return errcode;
    }

    // 清除点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(cst);

    // 退出。
    return NO_ERROR;
}
#undef FAIL_SDRONIMG_FREE

