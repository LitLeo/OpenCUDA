// CurveTopology
// 曲线间的相位关系

#include "CurveTopology.h"
#include "Image.h"
#include <stdio.h>
#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X    32
#define DEF_BLOCK_Y     8

// 宏：IN_LABEL 和 OUT_LABEL
// 定义了曲线内的点和曲线外的点标记值
#define IN_LABEL  255
#define OUT_LABEL   0 

// Kernel 函数：_setCloseAreaKer（将封闭曲线包围的内部区域的值变为白色）
// 引用邱孝兵实现的 _labelCloseAreaKer（标记封闭区域），详见 HistogramDifference
// 该核函数使用著名的射线法确定点和一个封闭曲线的位置关系，即如果由当前点引射线，
// 与曲线有奇数个交点则在内部，如果有偶数个交点，则在曲线外部（ 0 属于偶数），
// 为了提高准确度，在实际实现的时候，我们使用了由当前点向上下左右四个方向引射线
// 来确定位置关系。引用该算法实现将封闭曲线包围的内部区域的值变为白色，并且需要
// 得到闭合曲线包围的点的个数，用于后续处理
static __global__ void      // Kernel 函数无返回值
_setCloseAreaKer(
        CurveCuda curve,    // 输入曲线
        ImageCuda maskimg,  // 输出标记结果
        int *count          // 闭合曲线包围点的个数
);

// Kernel 函数：_matdotKer（得到两幅图像数据的点积）
// 该核函数实现了两幅图像的点积，两图像其实可以看做两个矩阵，这样就转化成两矩阵
// 的点积，由于得到是 0-255 的二值图像，当对应位置灰度值都为 255 时候，结果可以
// 认为 1 * 1，返回一个 1。最后把得到的所有 1 相加得到点积
static __global__ void     // Kernel 函数无返回值
_matdotKer(
        ImageCuda inimg1,  // 标记图像1
        ImageCuda inimg2,  // 标记图像2
        int *partial_sum   // 点积结果
);

// Kernel 函数：_intersecNumtKer（得到两个曲线的交点个数）
// 输入两个曲线，得到两个曲线的交点，根据第一条曲线进行并行划分，并行得到第一条
// 曲线的坐标点，根据这个坐标点去循环查询第二条曲线是否有相等的坐标点，最终得到
// 两个曲线的交点个数，返回在下面的部分和中。
static __global__ void     // Kernel 函数无返回值
_intersecNumtKer(
        CurveCuda curve1,    // 输入曲线1
        CurveCuda curve2,    // 输入曲线2
        int *sectnum         // 部分和
);


// Kernel 函数：_setCloseAreaKer（将封闭曲线包围的内部区域的值变为白色）
static __global__ void _setCloseAreaKer(CurveCuda curve, ImageCuda maskimg,
                                          int *count)
{
    // 计算当前线程的索引
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断当前线程是否越过输入图像尺寸
    if (xidx >= maskimg.imgMeta.width || yidx >= maskimg.imgMeta.height)
        return;

    // 定义部分寄存器变量   
    int downcount = 0;                         // 向下引射线和曲线的交点个数
    int length = curve.crvMeta.curveLength;  // 曲线上的点的个数 
    int outpitch = maskimg.pitchBytes;     // 输出标记图像的 pitch

    // 首先将所有点标记为曲线外的点
    maskimg.imgMeta.imgData[yidx *  outpitch+ xidx] = OUT_LABEL;

    int flag = 0; // 判断是否进入切线区域

    // 遍历曲线，统计上述各个寄存器变量的值
    for (int i = 0; i < length; i++) {
        int x = curve.crvMeta.crvData[2 * i];
        int y = curve.crvMeta.crvData[2 * i + 1];

        // 曲线中的下一个点的位置
        int j = (i + 1) % length;
        int x2 = curve.crvMeta.crvData[2 * j];

        // 曲线中上一个点的位置
        int k = (i - 1 + length) % length;
        int x3 = curve.crvMeta.crvData[2 * k];

        // 曲线上的第 i 个点与当前点在同一列上
        if (x == xidx) {
            if (y == yidx) {
                // 当前点在曲线上，此处把曲线上的点也作为曲线内部的点
                maskimg.imgMeta.imgData[yidx *  outpitch+ xidx] = IN_LABEL;
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
        maskimg.imgMeta.imgData[yidx *  outpitch + xidx] = IN_LABEL;
        atomicAdd(count, 1);
    }    
}

// Kernel 函数：_matdotKer（得到两幅图像数据的点积）
static __global__ void _matdotKer(ImageCuda inimg1, ImageCuda inimg2,
                                  int *partial_sum)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg1.imgMeta.width || r >= inimg1.imgMeta.height)
        return;
    
    // 计算输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg1.pitchBytes + c;
    
    // 如果图像矩阵对应位置的像素值都不为 0，则给点积加 1 处理
    if (inimg1.imgMeta.imgData[inidx] && inimg2.imgMeta.imgData[inidx]) {
        atomicAdd(partial_sum, 1);
    }
}

// Kernel 函数：_intersecNumtKer（得到两个曲线的交点个数）
static __global__ void _intersecNumtKer(CurveCuda curve1, CurveCuda curve2,
                                        int *sectnum)
{
    // index 表示线程处理的像素点的坐标。
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int length1 = curve1.crvMeta.curveLength;    // 曲线1上的点的个数 
    
    // 检查坐标点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (index >= length1)
        return;
        
    int length2 = curve2.crvMeta.curveLength;   // 曲线2上的点的个数 
    
    // 得到该线程第一条曲线的坐标点数据
    int x1 = curve1.crvMeta.crvData[2 * index];
    int y1 = curve1.crvMeta.crvData[2 * index + 1];
    
    int x2, y2;  // 临时变量，存储第二条曲线的坐标点
    
    // 循环查找第二条曲线的坐标点是否有相等的点
    for (int i = 0; i < length2; i++) {
        // 得到该线程第二条曲线的坐标点数据
        x2 = curve2.crvMeta.crvData[2 * i];
        y2 = curve2.crvMeta.crvData[2 * i + 1];
        // 如果找到，则交点加 1
        if ((x1 == x2) && (y1 == y2)) {
            atomicAdd(sectnum, 1);
        }
    }
}

// 宏：FREE_CURVE_TOPOLOGY（清理局部申请的设备端或者主机端内存）
// 该宏用于清理在 curveTopology 过程中申请的设备端或者主机端内存空间。
#define FREE_CURVE_TOPOLOGY do {                  \
        if (maskimg1 != NULL)                     \
            ImageBasicOp::deleteImage(maskimg1);  \
        if (maskimg2 != NULL)                     \
            ImageBasicOp::deleteImage(maskimg2);  \
        if (temp_dev != NULL)                     \
            cudaFree(temp_dev);                   \
    } while (0)


// Host 成员方法：curveTopology（曲线相位关系）
__host__ int CurveTopology::curveTopology(Curve *curve1, Curve *curve2, 
                                          CurveRelation *crvrelation,
                                          int width, int height)
{
    // 判断输入曲线是否为空
    if (curve1 == NULL || curve2 == NULL)
        return NULL_POINTER;

    // 检查输入参数是否有数据
    if (curve1->curveLength <= 0 || curve2->curveLength <= 0 ||
        width <= 0 || height <= 0)
        return INVALID_DATA;
    
    // 检查输入曲线是否为封闭曲线，如果不是封闭曲线返回错误
    if (!curve1->closed || !curve2->closed)
         return INVALID_DATA;
        
    // 局部变量，错误码。
    int errcode;
    cudaError_t cuerrcode;
    
    // 将曲线拷贝到 Device 内存中
    errcode = CurveBasicOp::copyToCurrentDevice(curve1);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 将曲线拷贝到 Device 内存中
    errcode = CurveBasicOp::copyToCurrentDevice(curve2);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 获取 CurveCuda 指针
    CurveCuda *curvecud1 = CURVE_CUDA(curve1);
    CurveCuda *curvecud2 = CURVE_CUDA(curve2);
    
    // 定义临时变量，统计两个曲线包围的点数，包括曲线上的点
    int count1;
    int count2;
    // 定义点积
    int result;
    // 定义交点个数
    int sectnum;
    
    // 定义局部变量，用于多份数据的一份申请
    int *temp_dev = NULL;

    // 定义临时标记图像指针
    Image *maskimg1 = NULL;
    Image *maskimg2 = NULL;
    
    // 给 temp_dev 在设备申请空间
    cuerrcode = cudaMalloc((void**)&temp_dev, sizeof (int) * 4);
    if (cuerrcode != cudaSuccess) {
        FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }

    // 给 temp_dev 的内容初始化为 0
    cuerrcode = cudaMemset(temp_dev, 0, sizeof (int) * 4);
    if (cuerrcode != cudaSuccess) {
        FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }
    
    // 定义设备指针，存储两个曲线包围的点数、点积和交点个数
    int *dev_count1 = temp_dev;
    int *dev_count2 = dev_count1 + 1;
    int *dev_sum = dev_count2 + 1;
    int *dev_sectnum = dev_sum + 1;
  
    // 给临时标记图像1在设备申请空间
    ImageBasicOp::newImage(&maskimg1);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(maskimg1, width, height);
    if (errcode != NO_ERROR) {
        FREE_CURVE_TOPOLOGY;
        return errcode;
    }

    // 给临时标记图像2在设备申请空间
    ImageBasicOp::newImage(&maskimg2);
    if (errcode != NO_ERROR) {
        FREE_CURVE_TOPOLOGY;
        return errcode;
    }
    errcode = ImageBasicOp::makeAtCurrentDevice(maskimg2, width, height);
    if (errcode != NO_ERROR) {
        FREE_CURVE_TOPOLOGY;
        return errcode;
    }
    
    // 获取 ImageCuda 指针
    ImageCuda *maskimgcud1 = IMAGE_CUDA(maskimg1);
    ImageCuda *maskimgcud2 = IMAGE_CUDA(maskimg2);
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y  - 1) / blocksize.y;

    // 调用核函数，将封闭曲线包围的内部区域的值变为白色，并且得到包围点的个数
    _setCloseAreaKer<<<gridsize, blocksize>>>( 
            *curvecud1, *maskimgcud1, dev_count1);  
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }

    // 拷贝 dev_count1 到 Host 端
    cuerrcode = cudaMemcpy(&count1, dev_count1, sizeof (int),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }
    
    // 得到第一条曲线包围的点的个数，包括曲线上的点
    count1 += curvecud1->capacity;

    // 调用核函数，将封闭曲线包围的内部区域的值变为白色，并且得到包围点的个数
    _setCloseAreaKer<<<gridsize, blocksize>>>( 
            *curvecud2, *maskimgcud2, dev_count2);
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }

    // 拷贝 dev_count2 到 Host 端
    cuerrcode = cudaMemcpy(&count2, dev_count2, sizeof (int),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }
    
    // 得到第二条曲线包围的点的个数，包括曲线上的点
    count2 += curvecud2->capacity;
    // 调用核函数，得到两幅图像矩阵的点积
    _matdotKer<<<gridsize, blocksize>>>(*maskimgcud1, *maskimgcud2, dev_sum);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }
    
    // 拷贝 dev_sum 到 Host 端
    cuerrcode = cudaMemcpy(&result, dev_sum, sizeof (int),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }

    // 如果两个图像矩阵点积为 0，则表示两个曲线没有包含、被包含和相交关系，是
    // 属于除上述三种外的其他关系，设置曲线关系为其他关系
    if (result == 0) {
        crvrelation->relation = CURVE_OTHERSHIP;
        crvrelation->internum = 0;
    }
    
    // 如果两个图像矩阵点积为第一条曲线的包围点的个数，则曲线1被包含在曲线2中，
    // 设置曲线关系为被包含
    else if (result == count1) {
        crvrelation->relation = CURVE_INCLUDED;
        crvrelation->internum = 0;
    }
    
    // 如果两个图像矩阵点积为第二条曲线的包围点的个数，则曲线1包含在曲线2中，设
    // 置曲线关系为包含
    else if (result == count2) {
        crvrelation->relation = CURVE_INCLUDE;
        crvrelation->internum = 0;
    }
    
    // 除上面三种情况外，则两个曲线相交，设置曲线相位关系为相交
    else 
        crvrelation->relation = CURVE_INTERSECT; 

    // 如果曲线是相交关系，则开始求交点个数
    if (crvrelation->relation == CURVE_INTERSECT) {
        
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。这里使用一维线程块
        int blocksize1, gridsize1;
        blocksize1 = DEF_BLOCK_X * DEF_BLOCK_Y;
        gridsize1 = (curvecud1->capacity + blocksize1 - 1) / blocksize1;

        // 调用核函数，得到两条曲线的交点个数
        _intersecNumtKer<<<gridsize1, blocksize1>>>( 
                *curvecud1, *curvecud2, dev_sectnum);

        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess) {
            FREE_CURVE_TOPOLOGY;
            return CUDA_ERROR;
        }
        
        // 拷贝 dev_sectnum 到 Host 端
        cuerrcode = cudaMemcpy(&sectnum, dev_sectnum, sizeof (int),
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            FREE_CURVE_TOPOLOGY;
            return CUDA_ERROR;
        }
        
        // 得到曲线的交点数目
        crvrelation->internum = sectnum;
    }

    // 释放临时申请的空间
    FREE_CURVE_TOPOLOGY;

    // 程序执行结束，返回
    return NO_ERROR;
}  
