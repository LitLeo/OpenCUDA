// CurveFluctuation.cu
// 创建人：邱孝兵

#include "CurveFluctuation.h"
#include "ErrorCode.h"
#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_1D
// 定义了默认的 1D 线程块的尺寸。
#define DEF_BLOCK_1D    256

// Kernel 函数：_calcCurveFluctuKer（计算曲线波动特征）
// 首先并行计算出每个点对应平滑后的点坐标，然后计算二者的偏移距离，块内同步
// 后再统计出偏移距离最大的若干个点，记录这些点的坐标和偏移距离，作为曲线的
// 波动特征，同时还需要统计出曲线上每个点的平均偏移距离和偏移坐标。
__global__ void                // Kernel 函数无返回值
_calcCurveFluctuKer(
        CurveCuda_st incurve,  // 输入曲线
        int smWindowSize,      // 平滑邻域宽度
        int *outdxy            // 每个点的偏移距离
);


// Kernel 函数：_calcCurveFluctuKer（计算曲线波动特征）
__global__ void _calcCurveFluctuKer(CurveCuda_st incurve,
                                    int smWindowSize, int *outdxy)
{    
    // 计算当前 Thread 所对应的坐标集中的点的位置
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;

    // 判断当前线程是否超过了曲线中点的个数
    if (idx >= incurve.crvMeta.curveLength) {
        return;
    }
    
    // 定义一些寄存器变量
    int length = incurve.crvMeta.curveLength;  // 曲线内点的个数
    float smx = 0.0;                        // 平滑后的横坐标 
    float smy = 0.0;                        // 平滑后的纵坐标
    int count = 0;                          // 参与平滑的点个数

    // 当前点邻域坐标求和
    for (int i = idx - smWindowSize; i < idx + smWindowSize; i++) {
        if (i < 0 || i >= length) 
            continue;
        smx += incurve.crvMeta.crvData[2 * i];
        smy += incurve.crvMeta.crvData[2 * i + 1];
        count++;
    }

    // 利用平均值进行平滑
    if (count > 0) {
        smx /= count;
        smy /= count;
    }

    // 计算当前点和其对应的平滑点之间的偏移距离
    float dx = incurve.crvMeta.crvData[2 * idx] - smx;
    float dy = incurve.crvMeta.crvData[2 * idx + 1] - smy;
    outdxy[idx] = (int)(sqrt(dx * dx + dy * dy) + 0.5);    
}


// 宏：FREE_LOCAL_MEMORY_CALC_FLUCTU（清理局部申请的设备端或者主机端内存）
// 该宏用于清理在 calcCurveFluctu 过程中申请的设备端或者主机端内存空间。
#define FREE_LOCAL_MEMORY_CALC_FLUCTU do {  \
        if ((dxyhost) != NULL)              \
            delete [] (dxyhost);            \
        if ((dxydevice) != NULL)            \
            cudaFree((dxydevice));          \
    } while (0)

// Host 成员方法：calcCurveFluctu（计算曲线波动特征）
__host__ int CurveFluctuation::calcCurveFluctu(Curve *incurve,
                                               CurveFluctuPropers *inoutcfp)
{
    // 检查输入和输出是否为 NULL，如果为 NULL 直接报错返回
    if (incurve == NULL || inoutcfp == NULL) 
        return NULL_POINTER;

    // 检查输入和输出的数据是否合法，如果不合法直接返回报错
    if (incurve->curveLength <= 0 || inoutcfp->maxFluctuNum <= 0 || 
        inoutcfp->smNieghbors <= 0) {
        return INVALID_DATA;
    }

    // 将 Curve 拷贝到当前设备中
    CurveBasicOp::copyToCurrentDevice(incurve);
    
    // 根据输入 Curve 指针，获取 Curve_st 类型
    CurveCuda *cucurve = CURVE_CUDA(incurve);

    // 申请存储每个点和平滑点距离的数组 Host 内存空间
    int length = incurve->curveLength;  // 曲线中点的个数
    int *dxyhost = new int[length];

    // 申请存储每个点和平滑点距离的数组的 Device 内存空间
    int *dxydevice;
    cudaError_t cuerrcode = cudaMalloc((void **)&dxydevice, 
                                       sizeof(int) * length);

    // 申请失败，返回错误
    if (cuerrcode != cudaSuccess) {
        FREE_LOCAL_MEMORY_CALC_FLUCTU;
        return CUDA_ERROR;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_1D;
    blocksize.y = 1;
    gridsize.x = length / blocksize.x + 1;
    gridsize.y = 1;

    // 调用核函数，计算每个点和其对应平滑点的距离
    _calcCurveFluctuKer<<<gridsize, blocksize>>>(*cucurve,
                                                 inoutcfp->smNieghbors,
                                                 dxydevice);

    // 将标记值拷贝到主机端。
    cuerrcode = cudaMemcpy(dxyhost, dxydevice, length * sizeof(int),
                         cudaMemcpyDeviceToHost);
    if (cuerrcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_CALC_FLUCTU;
        return CUDA_ERROR;
    }
    
    int maxnum = inoutcfp->maxFluctuNum;  // 需要统计的最大距离点的个数 
    int sumdxy = 0;                       // 所有点偏移距离的和 

    // 将 cfp 中的 maxFluctu 均初始化为 0
    for (int i = 0; i < maxnum; i++){
        inoutcfp->maxFluctu[i] = 0;
        inoutcfp->maxFluctuX[i] = 0;
        inoutcfp->maxFluctuY[i] = 0;
    }

    // 将 Curve 拷贝到 Host 端
    CurveBasicOp::copyToHost(incurve);

    // 遍历所有距离，找出最大的 maxnum 个
    for (int i = 0; i < length; i++){
        // 将距离进行加和
        sumdxy += dxyhost[i];

        // 比较当前距离和 maxFluctu 中最小的（第 maxnum 个）
        if (dxyhost[i] > inoutcfp->maxFluctu[maxnum - 1]) {
            // 将 maxFluctu 最后一个值赋为当前的距离值
            inoutcfp->maxFluctu[maxnum - 1] = dxyhost[i];

            // 同时将对应的横纵坐标也赋值
            inoutcfp->maxFluctuX[maxnum - 1] = incurve->crvData[2 * i];
            inoutcfp->maxFluctuY[maxnum - 1] = incurve->crvData[2 * i + 1];

            // 重新调整 maxFluctu 数组
            for (int j = maxnum - 2; j >= 0; j--){
                // 如果 j + 1 对应的 maxFluctu 值 大于 j 则交换二者的位置
                if (inoutcfp->maxFluctu[j + 1] > inoutcfp->maxFluctu[j]) {
                    int tmp = inoutcfp->maxFluctu[j + 1];
                    int tmpx = inoutcfp->maxFluctuX[j + 1];
                    int tmpy = inoutcfp->maxFluctuY[j + 1];
                    inoutcfp->maxFluctu[j + 1] = inoutcfp->maxFluctu[j]; 
                    inoutcfp->maxFluctuX[j + 1] = inoutcfp->maxFluctuX[j]; 
                    inoutcfp->maxFluctuY[j + 1] = inoutcfp->maxFluctuY[j]; 
                    inoutcfp->maxFluctu[j] = tmp; 
                    inoutcfp->maxFluctuX[j] = tmpx;
                    inoutcfp->maxFluctuY[j] = tmpy;
                } else {
                    break;
                }

            }
        }
    }
    // 计算偏移均值 aveFluctu 
    inoutcfp->aveFluctu = (int)(sumdxy / length + 0.5f);

    // 计算坐标偏移均值 xyAveFluctu
    inoutcfp->xyAveFluctu = inoutcfp->aveFluctu;

    // 处理完毕返回
    return NO_ERROR;
}
 