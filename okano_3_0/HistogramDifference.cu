// HistogramDifference.cu
// 直方图差异

#include "HistogramDifference.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32 
#define DEF_BLOCK_Y   8 

// 宏：IN_LABEL 和 OUT_LABEL
// 定义了曲线内的点和曲线外的点标记值
#define IN_LABEL  255
#define OUT_LABEL   0 

// Kernel 函数：_labelCloseAreaKer（标记封闭区域）
// 在 HistogramDifference 算法中，需要对曲线封闭的区域统计直方图，
// 因此需要首先确定图像中每个点是否位于曲线封闭的区域当中，该核函数
// 使用著名的射线法确定点和一个封闭曲线的位置关系，即如果由当前点
// 引射线，与曲线有奇数个交点则在内部，如果有偶数个交点，则在曲线
// 外部（ 0 属于偶数），为了提高准确度，在实际实现的时候，我们使用了
// 由当前点向上下左右四个方向引射线来确定位置关系。
__global__ void             // Kernel 函数无返回值
_labelCloseAreaKer(
        CurveCuda incurve,  // 输入曲线
        ImageCuda inimg,    // 输入图像
        ImageCuda outlabel  // 输出标记结果
);

// Kernel 函数：_closeAreaHistKer（统计封闭区域的直方图）
// 该核函数，根据输入图像和输入图像的标记图像，统计由封闭曲线划分
// 闭合区域的直方图。在块内使用共享内存，末尾利用原子加操作将结果累加
// 到全局内存。
__global__ void  // Kernel 函数无返回值
_closeAreaHistKer(
        ImageCuda inimg,     // 输入图像
        ImageCuda labelimg,  // 标记图像
        int *histogram       // 直方图
);


// Kernel 函数：_labelCloseAreaKer（标记封闭区域）
__global__ void _labelCloseAreaKer(CurveCuda incurve,ImageCuda outlabelimg)
{
    // 计算当前线程的索引
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断当前线程是否越过输入图像尺寸
    if (xidx >= outlabelimg.imgMeta.width || yidx >= outlabelimg.imgMeta.height)
        return;

    // 定义部分寄存器变量   
    int downcount = 0;                         // 向下引射线和曲线的交点个数
    int length = incurve.crvMeta.curveLength;  // 曲线上的点的个数 
    int outpitch = outlabelimg.pitchBytes;     // 输出标记图像的 pitch

    // 首先将所有点标记为曲线外的点
    outlabelimg.imgMeta.imgData[yidx *  outpitch+ xidx] = OUT_LABEL;

    int flag = 0; // 判断是否进入切线区域

    // 遍历曲线，统计上述各个寄存器变量的值
    for (int i = 0; i < length; i++) {
        int x = incurve.crvMeta.crvData[2 * i];
        int y = incurve.crvMeta.crvData[2 * i + 1];

        // 曲线中的下一个点的位置
        int j = (i + 1) % length;
        int x2 = incurve.crvMeta.crvData[2 * j];

        // 曲线中上一个点的位置
        int k = (i - 1 + length) % length;
        int x3 = incurve.crvMeta.crvData[2 * k];

        // 曲线上的第 i 个点与当前点在同一列上
        if (x == xidx) {
            if (y == yidx) {
                // 当前点在曲线上，此处把曲线上的点也作为曲线内部的点
                outlabelimg.imgMeta.imgData[yidx *  outpitch+ xidx] = IN_LABEL;
                return;
            }

            // 交点在当前点的下方
            if (y > yidx) {
                // 曲线上下一个也在射线上时，避免重复统计，同时设置 flag 
                // 标记交点行开始。如果下一个点不在射线上，通过 flag 判断到
                // 底是交点行结束还是单点相交，如果是单点相交判断是否为突出点
                // 如果是交点行结束判断是否曲线在交点行同侧，以上都不是统计值
                // 加一.
                if (x2 == xidx) {
                    if (flag == 0) 
                        flag = x3 - x;   
                } else {
                    if (flag == 0) {
                        if ((x3 - x) * (x2 - x) <= 0) 
                            downcount++;                        
                    } else {
                        if (flag * (x2 - x) < 0) 
                            downcount++;
                        flag = 0;
                    }
                }
            }
        }
    }

    // 交点数均为奇数则判定在曲线内部
    if (downcount % 2 == 1) {
        outlabelimg.imgMeta.imgData[yidx *  outpitch + xidx] = IN_LABEL;
    }    
}

// Kernel 函数：_closeAreaHistKer（统计封闭区域的直方图）
__global__ void _closeAreaHistKer(ImageCuda inimg, ImageCuda labelimg,
                                  int *histogram)
{
    // 计算当前线程的索引
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    // 定义寄存器变量
    int imgpitch = inimg.pitchBytes;
    int labelpitch = labelimg.pitchBytes;

    // 判断当前线程是否越过输入图像尺寸
    if (xidx >= labelimg.imgMeta.width || yidx >= labelimg.imgMeta.height)
        return;

    // 如果当前点在闭合区域外，直接返回
    if (labelimg.imgMeta.imgData[yidx * labelpitch + xidx] == OUT_LABEL) 
        return;

    // 申请大小为灰度图像灰度级 256 的共享内存，其中下标代表图像的灰度值，数
    // 组用来累加等于该灰度值的像素点个数。
    __shared__ unsigned int temp[256];

    // 计算该线程在块内的相对位置。
    int inindex = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 若线程在块内的相对位置小于 256，即灰度级大小，则用来给共享内存赋初值 0。
    if (inindex < 256)
        temp[inindex] = 0;

    // 进行块内同步，保证执行到此处，共享内存的数组中所有元素的值都为 0。
    __syncthreads();
    
    int curgray = inimg.imgMeta.imgData[yidx * imgpitch + xidx];
    atomicAdd(&temp[curgray], 1);

    // 块内同步。此处保证图像中所有点的像素值都被统计过。
    __syncthreads();
    
    // 用每一个块内前 256 个线程，将共享内存 temp 中的结果保存到输出数组中。
    if (inindex < 256)
        atomicAdd(&histogram[inindex], temp[inindex]);

}

// 宏：FREE_LOCAL_MEMORY_HIST_DIFF（清理局部申请的设备端或者主机端内存）
// 该宏用于清理在 histogramDiff 过程中申请的设备端或者主机端内存空间。
#define FREE_LOCAL_MEMORY_HIST_DIFF do {         \
        if ((labelimg) != NULL)                  \
            ImageBasicOp::deleteImage(labelimg); \
        if ((histogramhost) != NULL)             \
            delete []histogramhost;              \
        if ((histogramdevice) != NULL)           \
            cudaFree(histogramdevice);           \
    } while (0)

// 成员函数：histogramDiff （直方图差异）
__host__ int  HistogramDifference::histogramDiff(Curve *incurve, Image *inimg,
                                                 float *referhistogram, 
                                                 float &chisquarehd,
                                                 float &intersecthd)
{
    // 检查输入指针是否为 NULL
    if (incurve == NULL || inimg == NULL || referhistogram == NULL)
        return NULL_POINTER;

    // 检查输入参数是否有数据
    if (incurve->curveLength <= 0 || inimg -> width <= 0 || inimg->height <= 0)
        return INVALID_DATA;

    // 检查输入曲线是否为封闭曲线，如果不是封闭曲线返回错误
    if (!incurve->closed)
        return INVALID_DATA;

    int *histogramhost = NULL;  // host 端直方图指针
    int *histogramdevice;       // device 端直方图指针
    int errcode;                // 局部变量，错误码

    // 将曲线拷贝到 Device 内存中
    errcode = CurveBasicOp::copyToCurrentDevice(incurve);

    if (errcode != NO_ERROR)
        return errcode;

    // 获取 CurveCuda 指针
    CurveCuda *incurvecud = CURVE_CUDA(incurve);
 
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgcud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgcud);
    if (errcode != NO_ERROR) 
        return errcode;

    // 获取输入图像的尺寸
    int width = inimg -> width;
    int height = inimg -> height;

    // 创建标记图像
    Image *labelimg;
    ImageBasicOp::newImage(&labelimg);
    ImageBasicOp::makeAtHost(labelimg, width, height);

    // 将标记图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(labelimg);
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_HIST_DIFF;
        return errcode;
    }

    // 提取标记图像的 ROI 子图像。
    ImageCuda labelsubimgcud;
    errcode = ImageBasicOp::roiSubImage(labelimg, &labelsubimgcud);
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_HIST_DIFF;
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y  - 1) / blocksize.y;
    
    // 调用核函数，标记曲线内部和曲线外部的点
    _labelCloseAreaKer<<<gridsize, blocksize>>>( 
            *incurvecud, labelsubimgcud);    
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        FREE_LOCAL_MEMORY_HIST_DIFF;
        return CUDA_ERROR;
    }
    
    // 申请 host 端直方图存储空间
    histogramhost = new int [256];

    errcode = cudaMalloc((void**)&histogramdevice,
                         256 * sizeof (unsigned int));
    if (errcode != cudaSuccess) {
        FREE_LOCAL_MEMORY_HIST_DIFF;
        return CUDA_ERROR;
    }

    // 初始化 Device 上的内存空间。
    errcode = cudaMemset(histogramdevice, 0, 256 * sizeof (int));
    if (errcode != cudaSuccess) {
        FREE_LOCAL_MEMORY_HIST_DIFF;
        return CUDA_ERROR;
    }

    // 调用核函数统计闭合区域内直方图
    _closeAreaHistKer<<<gridsize, blocksize>>>(insubimgcud,labelsubimgcud,
                                               histogramdevice);

    // 拷贝结果到 Host 
    errcode = cudaMemcpy(histogramhost, histogramdevice, 
                         256 * sizeof (int), cudaMemcpyDeviceToHost);

    if (errcode != cudaSuccess) {
        FREE_LOCAL_MEMORY_HIST_DIFF;
        return CUDA_ERROR;
    }

    // 定义 C(H1,H2) 和 I(H1,H2) 差异的统计量
    float csum = 0.0f;
    float isum = 0.0f;  

    // 计算两个差异统计量
    for (int i = 0; i < 256 ; i++) {
        csum += (referhistogram[i] - histogramhost[i]) * 
                (referhistogram[i] - histogramhost[i]) / 
                referhistogram[i];
        isum += (referhistogram[i] < histogramhost[i]) ? 
                 referhistogram[i] : histogramhost[i];
    }

    // 将结果赋值给引用参数
    chisquarehd = csum;
    intersecthd = isum;

    // 释放空间
    FREE_LOCAL_MEMORY_HIST_DIFF;

    return NO_ERROR;
}