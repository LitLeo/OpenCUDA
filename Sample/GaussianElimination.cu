// GaussianElimination.cu
// 实现矩阵的高斯消元法，求矩阵的上三角矩阵形式

#include "GaussianElimination.h"
#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_1D
// 定义一维线程块尺寸。
#define DEF_BLOCK_1D  512 

// Kernel 函数：_pretreatmentKer （对矩阵数据进行预处理）
// 对矩阵 ROI 区域的第一行和 ROI 区域内第一列元素绝对值最大的一行进行行对换变
// 换，使 ROI 第一行第一列元素在同列元素中的绝对值最大，这样既可以使该元素值
// 不为 0，又可以提高精确度。
static __global__ void 
_pretreatmentKer(
        MatrixCuda matcuda  // 源矩阵，要求为方阵
);

// Kernel 函数：_gaussianEliminationKer（实现矩阵某一行的高斯消元）
// 对矩阵 ROI 区域内的除第一行以外的某一行进行处理，通过行倍加变换使该行第一列
// 的元素值为 0，具体过程是用该行每一列的元素值减去 ROI 区域内第一行对应列元素
// 值的 multiple 倍，multiple 值为该行第一列元素值除以 ROI 区域内第一行第一列
// 元素值的商。通过对 ROI 区域内第一行之外的每一行依次进行处理，使第一行之外的
// 每一行元素值均为 0。
static __global__ void 
_gaussianEliminateKer(
        MatrixCuda matcuda,  // 源矩阵，要求为方阵
        int row              // 要进行计算的矩阵的行数
);

// Kernel 函数：_pretreatmentKer（对矩阵数据进行预处理）
static __global__ void _pretreatmentKer(MatrixCuda matcuda) 
{
    // cindex 表示当前线程的索引，col 表示当前线程计算的矩阵元素坐标的 x 分量。
    int cindex = threadIdx.x + blockIdx.x * blockDim.x;
    int col = cindex + matcuda.matMeta.roiY1;

    // 检查矩阵元素是否越界，如果越界，则不进行处理。
    if (col >= matcuda.matMeta.width) 
        return;

    // 局部变量 maxrow, 标记第一列元素绝对值最大的一行的行数。
    int maxrow = matcuda.matMeta.roiX1;

    // 计算找出当前 ROI 区域内第一列元素绝对值最大的一行。
    for (int r = matcuda.matMeta.roiX1; r <= matcuda.matMeta.roiX2; r++) { 
        // 如果 maxrow 行第一列元素的绝对值小于第 r 行第一列的元素的绝对值，则
        // 把 r 的值赋给 maxrow。
        if (abs(matcuda.matMeta.matData[matcuda.matMeta.roiY1 + 
                    maxrow * matcuda.pitchWords]) < 
                    abs(matcuda.matMeta.matData[matcuda.matMeta.roiY1 + 
                    r * matcuda.pitchWords]))
            maxrow = r;
    }

    // 交换 ROI 区域内第一行和第一列元素最大的一行的每一列元素的值。
    // 局部变量 temp, 作为交换的中间值。
    float temp = matcuda.matMeta.matData[col + 
            (matcuda.matMeta.roiX1) * matcuda.pitchWords];
    matcuda.matMeta.matData[col + (matcuda.matMeta.roiX1) * 
            matcuda.pitchWords] = matcuda.matMeta.matData[col + 
            maxrow * matcuda.pitchWords];
    matcuda.matMeta.matData[col + maxrow * matcuda.pitchWords] = temp;
}

// Kernel 函数：_gaussianEliminationKer（实现矩阵某一行的高斯消元）
static __global__ void _gaussianEliminateKer(MatrixCuda matcuda, int row) 
{
    // cindex 表示当前线程的索引。
    // col 表示当前线程计算的矩阵元素坐标的 x 分量，即列数。
    int cindex = threadIdx.x + blockIdx.x * blockDim.x;
    int col = cindex + matcuda.matMeta.roiY1;
    
    // 检查矩阵元素是否越界或者该行在 ROI 区域内第一列元素值为 0，如果越界，则
    // 不进行处理。如果该行第一列元素值为 0，则不用进行处理。
    if (col >= matcuda.matMeta.width || 
        matcuda.matMeta.matData[matcuda.matMeta.roiY1 + 
                row * matcuda.pitchWords] == 0) 
        return;
    
    // 局部变量 multiple，行倍加变换的倍数，值为该行第一列元素值除以 ROI 区域内
    // 第一行第一列元素值的商。
    double multiple = matcuda.matMeta.matData[matcuda.matMeta.roiY1 + 
            row * matcuda.pitchWords] / 
            matcuda.matMeta.matData[matcuda.matMeta.roiY1 + 
            (matcuda.matMeta.roiX1) * matcuda.pitchWords];

    // 用该行每一列的元素值减去 ROI 区域内第一行对应列元素值的 multiple 倍，使
    // 该行第一列元素值为 0。
    matcuda.matMeta.matData[col + row * matcuda.pitchWords] -= 
            multiple * matcuda.matMeta.matData[col + 
            (matcuda.matMeta.roiX1) * matcuda.pitchWords];
}

// Host 成员方法：gaussianEliminate（实现矩阵的高斯消元法）
__host__ int GaussianElimination::gaussianEliminate(Matrix *inmat,
                                                    Matrix *outmat)
{
    // 检查输入指针是否为 NULL。
    if (inmat == NULL)
        return NULL_POINTER;

    // 检查矩阵是否为方阵。
    if (inmat->height != inmat->width)
        return UNMATCH_IMG;

    // 局部变量，错误码。
    int errcode;  

    // 把 inmat 的矩阵数据拷贝到当前 Device 内存上，在这里调用  Out-Place 形式
    // 的拷贝，把数据传给 outmat。
    errcode = MatrixBasicOp::copyToCurrentDevice(inmat, outmat); 
    if (errcode != NO_ERROR)
        return errcode;

    // 获取 outmat 对应的 MatrixCuda 型指针。
    MatrixCuda *outmatcuda = MATRIX_CUDA(outmat);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。调用一维核函数，在这里
    // 设置线程块内的线程数为 256，用 DEF_BLOCK_1D 表示。
    size_t blocksize, gridsize;
    blocksize = DEF_BLOCK_1D;
    gridsize = (inmat->width + blocksize - 1) / blocksize; 

    // 对矩阵进行高斯消元，循环中 ROI 区域内的矩阵依次变为原矩阵减去第一行和第
    // 一列的子矩阵，然后通过初等行变换使该子矩阵第一列除第一个元素之外的元素值
    // 变为 0，这样每次循环依次使矩阵每一列主对角线以下的元素值为 0，从而使矩阵
    // 化为上三角矩阵。
    for (int i = 0; i < inmat->height - 1; i++) {

        // 设置 ROI 区域的范围，每执行一次循环，ROI 范围都会减少一行一列。
        outmatcuda->matMeta.roiX1 = i;
        outmatcuda->matMeta.roiY1 = i;
        outmatcuda->matMeta.roiX2 = outmatcuda->matMeta.height - 1;
        outmatcuda->matMeta.roiY2 = outmatcuda->matMeta.width - 1;

        // 修改线程格内的线程块数，因为每执行一次循环，ROI 区域范围都会减少，所
        // 以要修改减少线程块数，以节省计算资源。
        gridsize = (inmat->width - outmatcuda->matMeta.roiY1 + 
            blocksize - 1) / blocksize;

        // 对矩阵 ROI 区域内元素进行预处理。
        _pretreatmentKer<<<gridsize, blocksize>>>(*outmatcuda);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束函数。
            return CUDA_ERROR;
        }

        // 对 ROi 区域内每一行依次进行高斯消元。
        for (int r = i + 1; r < inmat->height; r++) {
            _gaussianEliminateKer<<<gridsize, blocksize>>>(*outmatcuda, r);
            if (cudaGetLastError() != cudaSuccess) {
                // 核函数出错，结束函数。
                return CUDA_ERROR;
            }
        }
    }

    // 处理完毕，退出。
    return NO_ERROR;
}

