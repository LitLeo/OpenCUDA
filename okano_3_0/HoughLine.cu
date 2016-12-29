   // HoughLine.cu
// 实现 Hough 变换检测直线

#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "HoughLine.h"
#include "Image.h" 
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#include "ErrorCode.h"
#include "CoordiSet.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32 
#define DEF_BLOCK_Y   8

// 宏：M_PI
// π 值。对于某些操作系统，M_PI 可能没有定义，这里补充定义 M_PI。
#ifndef M_PI
#define M_PI 3.14159265359
#endif    
// ==========================全局函数声明==============================
// 根据输入点集的坐标，找到最上、最下、最左、最右的点，从而确定图像的宽和高。
static __host__ int _findMinMaxCoordinates(CoordiSet *guidingset, 
                                           int *xmin, int *ymin,
                                           int *xmax, int *ymax);
// ==========================Kernel函数声明==============================
static __global__ void     // Kernel 函数无返回值
_houghlineImgKer(
        ImageCuda inimg,   // 输入图像。
        int *bufhoughdev,  // 得票数矩阵。
        int numtheta,      // theta 的递增次数。
        int numrho,        // rho 的递增次数。
        double detheta,    // 每一次的角度增量。
        double derho       // 每一次的距离增量。
);

// Kernel 函数：_houghlineCorKer（根据输入坐标集计算得票数）
// 根据输入坐标集，通过计算角度，距离等参数，计算最终的得票数。
static __global__ void 
_houghlineCorKer(
        CoordiSet guidingset,  // 输入坐标集。
        int *bufhoughdev,      // 得票数矩阵。
        int numtheta,          // theta 的递增次数。
        int numrho,            // rho 的递增次数。
        double detheta,        // 每一次的角度增量。
        double derho           // 每一次的距离增量。
);

// Kernel 函数：_findlocalMaxKer（计算局部最大值）
static __global__ void 
_findlocalMaxKer(
        int *bufhoughdev,  // 得票数矩阵。
        int *bufsortdev,   // 局部最值矩阵。
        int *sumdev,       // 存在的直线数。
        int numtheta,      // theta 的递增次数。
        int threshold      // 直线的阈值。
);

// Kernel 函数：_houghoutKer（画出已检测到的直线）
// 根据计算得到的直线的参数，得到输出图像。所有检测出来的直线，
// 在输出图像中用像素值 128 将其画出。
static __global__ void 
_houghoutKer(
        ImageCuda outimg,         // 输出图像。
        LineParam *lineparamdev,  // 计算得到的直线参数。
        int linenum,              // 最大待检测直线数量。
        int derho                 // 每一次的距离增量。
);
// Kernel 函数：_realLineKer（检测给出线段的真实性）在inimg中检测参数给出的线段
// 是真实线段，还是某个线段的延长线，
static __global__ void _realLineKer(ImageCuda inimg,
                                    int x1, int y1, int x2, int y2,
                                    int xmax, int xmin, int ymax, int ymin,
                                    int delta, int *pointnumdev);
// ==========================Kernel函数定义==============================

// Kernel 函数：_houghlineImgKer（根据输入图像计算得票数）
static __global__ void _houghlineImgKer(ImageCuda inimg, int *bufhoughdev, 
                                        int numtheta, int numrho, 
                                        double detheta, double derho)
{
    // 处理当前线程对应的图像点(c,r)，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量,
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
	 
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if(c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;
 
    // 定义局部变量。
    unsigned char intemp;
    int k, rho, bufidx;
    float theta;
    float irho = 1.0f / derho;
    float tempr;
        
    // 计算输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;    
    // 读取第一个输入坐标点对应的像素值。
    intemp = inimg.imgMeta.imgData[inidx];

    // 如果当前像素值为 255，即有效像素值，则对该像素点进行直线检测。
    if (intemp == 255) {
        for (k = 0; k < numtheta; k++) {
            // 计算当前的角度 theta。
            theta = k * detheta;
            // 计算该角度 theta 对应直线另一个参数 rho 的值。
            tempr = (int)(c * cos(theta) * irho + r * sin(theta) * irho);
            // 根据上一步结果进行四舍五入。
            rho = (int)(tempr + (tempr >= 0 ? 0.5f : -0.5f));
            rho += (numrho - 1) / 2;

            // 计算得到当前直线的两个参数 theta 和 rho 对应的累加器
            // bufferHough 中的索引。使用原子操作，统计得票数。
            bufidx = (rho + 1) * (numtheta + 2) + k + 1; 
            atomicAdd(&bufhoughdev[bufidx], 1);
        }
    }
}

// Kernel 函数：_houghlineCorKer（根据输入坐标集计算得票数）
static __global__ void _houghlineCorKer(CoordiSet guidingset, int *bufhoughdev, 
                                        int numtheta, int numrho, 
                                        double detheta, double derho)
{
    // 计算计算当前线程的索引。         
    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    
    // 处理coordiset中的点（dx,dy）
    int dx = guidingset.tplData[2 * idx];
    int dy = guidingset.tplData[2 * idx + 1];
 
    // 定义局部变量。
    int k, rho, bufidx;
    float theta;
    float irho = 1.0f / derho;
    float tempr;
    
    // 计算得票数。 
    for (k = 0; k < numtheta; k ++) {
        // 计算当前的角度 theta。
        theta = k * detheta;

        // 计算该角度 theta 对应直线另一个参数 rho 的值。
        tempr = (int)(dx * cos(theta) * irho + dy * sin(theta) * irho);
        rho = (int)(tempr + (tempr >= 0 ? 0.5f : -0.5f));
        rho += (numrho - 1) / 2;
        
        // 计算得到当前直线的两个参数 theta 和 rho 对应的累加器
        // bufferHough 中的索引。使用原子操作，统计得票数。
        bufidx = (rho + 1) * (numtheta + 2) + k + 1; 
        atomicAdd(&bufhoughdev[bufidx], 1);
    } 
}

// Kernel 函数：_findlocalMaxKer（计算局部最大值）
static __global__ void _findlocalMaxKer(
        int *bufhoughdev, int *bufsortdev, int *sumdev, 
        int numtheta, int threshold)
{        
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 计算该线程在块内的相对位置。
    int inindex = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 申请共享内存，存该块内符合条件的局部最大值个数，
    // 即存在的直线数。
    __shared__ int totalsum[1];
    
    // 初始化所有块内的共享内存。
    if (inindex == 0)
        totalsum[0] = 0;
    // 块内同步。
    __syncthreads();
    
    // 计算当前线程在 bufHough 矩阵中的对应索引值。
    int index = (r + 1) * (numtheta + 2) + (c + 1);  

    // 当前线程的得票数大于直线阈值，并且大于邻域中的值时，
    // 认为他是局部最大值，即可能是直线。
    if (bufhoughdev[index] > threshold &&
        bufhoughdev[index] > bufhoughdev[index - 1] &&
        bufhoughdev[index] >= bufhoughdev[index + 1] &&
        bufhoughdev[index] > bufhoughdev[index - numtheta - 2] &&
        bufhoughdev[index] >= bufhoughdev[index + numtheta + 2]) 
	{
        bufsortdev[r * numtheta + c] = index;
        // 使用原子操作对局部最大值进行统计。
        atomicAdd(&totalsum[0], 1);
    } else {
        bufsortdev[r * numtheta + c] = 0;
    }
    // 块内同步。
    __syncthreads();
    
    // （0，0）号线程负责将本块（共32*8个线程）统计出的直线的存在数统计到 sumdev 中。
    if (inindex == 0 && totalsum[0] != 0)
        atomicAdd(&sumdev[0], totalsum[0]);       
}

// Kernel 函数：_houghoutKer（画出已检测到的直线）
static __global__ void _houghoutKer(ImageCuda outimg, LineParam *lineparamdev,
                                    int linenum, int derho)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省
    // 计算资源，一方面防止由于段错误导致的程序崩溃。
    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;
    
    // 计算当前坐标点对应的图像数据数组下标。
    unsigned char *outptr;
    outptr = outimg.imgMeta.imgData + c + r * outimg.pitchBytes;
    
    // 声明局部变量
    int i, temp;
    float theta; 
    float irho = 1.0f / derho;
    
    // 对所有已经检测出的直线进行循环，找到输入图像中该点所在的直线，
    // 并赋值 128。
    for (i = 0; i < linenum; i++) {
        // 得到直线的参数 rho，theta。
        theta = lineparamdev[i].angle;
        temp = (int)(c * cos(theta) * irho + r * sin(theta) * irho);
        if (temp == lineparamdev[i].distance)
            {*outptr = 255;
			 break;
			}
    }
}       
// Kernel 函数：_realLineKer（检测给出线段的真实性）在inimg中检测参数给出的线段
// 是真实线段，还是某个线段的延长线，
static __global__ void _realLineKer(ImageCuda inimg,
                                    int x1, int y1, int x2, int y2,
                                    int xmax, int xmin, int ymax, int ymin,
                                    int delta, int *pointnumdev)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int cx=blockIdx.x * blockDim.x+threadIdx.x;
    int ry=blockIdx.y * blockDim.y+threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if(cx >= inimg.imgMeta.width || ry >= inimg.imgMeta.height)
        return ;

    int inidx, temp=0;

    // 检查当前点是否在线段带容许误差的包围盒范围内，不在的话，直接返回
    if( cx <= xmax+delta && cx >= xmin-delta && 
        ry <= ymax+delta && ry >= ymin-delta ) {
        // 计算输入坐标点对应的图像数据数组下标。
        inidx=ry * inimg.pitchBytes+cx;
        // 读取第一个输入坐标点对应的像素值。
        temp=inimg.imgMeta.imgData[inidx];
        // 如果当前点在线段上，或者误差小于门限值，计数器加一
        if(temp == 255){
            float dis=abs( (x2-x1) * (ry-y1)-(cx-x1) * ( y2-y1) ) / sqrt( (x2-x1) * (x2-x1) * 1.0+(y2-y1) * (y2-y1));
            if(dis<delta)
                atomicAdd(pointnumdev, 1);
        }
    }// end of if;
    return ;
}
// ==========================全局函数定义==============================
// 根据输入点集的坐标，找到最上、最下、最左、最右的点，从而确定图像的宽和高。
static __host__ int _findMinMaxCoordinates(CoordiSet *guidingset, 
                                            int *xmin, int *ymin,
                                           int *xmax, int *ymax)
{
    // 声明局部变量。
    int i;
    int errcode;

    // 在 host 端申请一个新的 CoordiSet 变量。
    CoordiSet *tmpcoordiset;
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpcoordiset);
    if (errcode != NO_ERROR) 
        return errcode;
    
    errcode = CoordiSetBasicOp::makeAtHost(tmpcoordiset, guidingset->count);
    if (errcode != NO_ERROR) 
        return errcode;
    
    // 将坐标集拷贝到 Host 端。
    errcode = CoordiSetBasicOp::copyToHost(guidingset, tmpcoordiset);
    if (errcode != NO_ERROR) 
        return errcode;

    // 初始化 x 和 y 方向上的最小最大值。
    xmin[0] = xmax[0] = tmpcoordiset->tplData[0];
    ymin[0] = ymax[0] = tmpcoordiset->tplData[1]; 
    // 循环寻找坐标集最左、最右、最上、最下的坐标。   
    for (i = 1;i < tmpcoordiset->count;i++) {
        //　寻找 x 方向上的最小值。
        if (xmin[0] > tmpcoordiset->tplData[2 * i])
            xmin[0] = tmpcoordiset->tplData[2 * i];
        //　寻找 x 方向上的最大值    
        if (xmax[0] < tmpcoordiset->tplData[2 * i])
            xmax[0] = tmpcoordiset->tplData[2 * i];
            
        //　寻找 y 方向上的最小值。
        if (ymin[0] > tmpcoordiset->tplData[2 * i + 1])
            ymin[0] = tmpcoordiset->tplData[2 * i + 1];
        //　寻找 y 方向上的最大值
        if (ymax[0] < tmpcoordiset->tplData[2 * i + 1])
            ymax[0] = tmpcoordiset->tplData[2 * i + 1];
    }
    
    // 释放临时坐标集变量。
    CoordiSetBasicOp::deleteCoordiSet(tmpcoordiset);
    
    return errcode;
}

// ==========================成员函数定义==============================

// 宏：FAIL_HOUGH_LINE_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_HOUGH_LINE_FREE  do {       \
        if (alldatadev != NULL)          \
            cudaFree(alldatadev);        \
        if (alldata != NULL)             \
            delete[] alldata;            \
        if (linedata != NULL)            \
            delete[] linedata;           \
        if (line != NULL)                \
            delete[] line;               \
    } while (0)

// Host 成员方法：houghlineCor（Hough 变换检测直线）
__host__ int HoughLine::houghLineCor(CoordiSet *guidingset,
                                      int *linesmax, 
                                     LineParam *lineparam)
{

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 得到输入图像的宽和高。
    int width, height;
    int xmin, ymin, xmax, ymax; 

    if (guidingset != NULL) {
        // 输入图像为空，则根据输入点集得到最左、
        // 最右、最上、最下的坐标值。
        errcode = _findMinMaxCoordinates(guidingset, &xmin, &ymin,
                                         &xmax, &ymax);
        if (errcode != NO_ERROR)
            return errcode;
            
        // 计算得票数矩阵的宽和高。
        width = xmax-xmin ;
        height = ymax-ymin;
        
    }

    // 计算rho 和 theta 的递增次数。为减少计算，numrho用了近似值的距离最大值。
    int numrho = (int)((width + height) * 2 + 1) / derho;
    int numtheta = (int)(M_PI / detheta);

    // 声明需要的指针变量。
    int *alldatadev = NULL;
    int *alldata = NULL;
    int *linedata = NULL;   
    LineParam *line = NULL;      
    LineParam *lineparamdev = NULL;   
   
    // 一次性申请 Device 端需要的所有空间。
    int *bufhoughdev = NULL, *bufsortdev = NULL, *sumdev = NULL; 
    cudaError_t cudaerrcode;                      
    cudaerrcode = cudaMalloc((void **)&alldatadev,
                             (1 + (numtheta + 2) * (numrho + 2) + 
                              numtheta * numrho) * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 通过偏移得到各指针的地址。
    sumdev = alldatadev;
    bufhoughdev = alldatadev + 1;
    bufsortdev = alldatadev + 1 + (numtheta + 2) * (numrho + 2);
    
    // 初始化 Hough 变换累加器在 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldatadev, 0, 
                             (1 + (numtheta + 2) * (numrho + 2) + 
                              numtheta * numrho) * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize; 

    
    // 根据输入坐标集是否为空，分两种情况进行：
    if (guidingset != NULL) {
        // 若输入坐标集不为空，则将该点集拷贝入 Device 内存。   
        errcode = CoordiSetBasicOp::copyToCurrentDevice(guidingset);
        if (errcode != NO_ERROR)
            return errcode; 
            

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。      
        blocksize.x = 16;
        blocksize.y = 1;
        gridsize.x = (guidingset->count+15)/16;
        gridsize.y = 1;
                
        // 调用核函数，对输入坐标集 guidingset 计算 Hough 累加矩阵。
        _houghlineCorKer<<<gridsize, blocksize>>>(*guidingset, bufhoughdev,  
                                                  numtheta, numrho, 
                                                  detheta, derho);
        if (cudaGetLastError() != cudaSuccess) {
            // 释放内存空间。
            FAIL_HOUGH_LINE_FREE;
            return CUDA_ERROR;
        }         
    } 
    
    // 在 Host 端一次性申请全部所需的空间。
    int *bufHough = NULL, *bufsort = NULL;  
    int sum;
    alldata = new int [(numtheta + 2) * (numrho + 2) + numtheta * numrho];
    if (alldata == NULL)
        return OUT_OF_MEM;
  
    // 通过偏移得到各指针的地址。
    bufHough = alldata;
    bufsort = alldata + (numtheta + 2) * (numrho + 2);
    
    // 将 Kernel 函数中计算的得票数矩阵 bufHoughDev 拷贝至 Host 端。
    cudaerrcode = cudaMemcpy(bufHough, bufhoughdev, 
                             (numtheta + 2) * (numrho + 2) * sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;        
        return cudaerrcode;
    }   
 
    // 计算调用计算局部最大值的 kernel 函数的线程块的尺寸和线程块的数量。  
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (numtheta + blocksize.x - 1) / blocksize.x;
    gridsize.y = (numrho + blocksize.y - 1) / blocksize.y;
    
    // 调用计算局部最大值的 kernel 函数。
    _findlocalMaxKer<<<gridsize, blocksize>>>(bufhoughdev, bufsortdev, sumdev,
                                             numtheta, threshold);   
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return CUDA_ERROR;  
    }   
    
    // 将 Kernel 函数中计算的得票数 sumdev 拷贝至 Host 端。
    cudaerrcode = cudaMemcpy(&sum, sumdev, sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 将 Kernel 函数中计算的得票数矩阵 bufsortdev 拷贝至 Host 端。
    cudaerrcode = cudaMemcpy(bufsort, bufsortdev, 
                             numtheta * numrho * sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }

    // 在 Host 端申请存放直线得票数和索引的数组。
    int *linevote = NULL, *lineindex = NULL;
    
    // 根据计算出存在的直线数，一次性申请所需空间。
    linedata = new int [sum * 2];
    if (linedata == NULL)
        return OUT_OF_MEM;
    linevote = linedata;
    lineindex = linedata + sum;
    // 局部变量。
    int k = 0, temp;
    
    // 统计可能存在的直线的得票数和索引值。
    for (int j = 0; j < numrho; j++) {
        for (int i = 0; i < numtheta; i++){
            temp = j * numtheta + i;
            if (bufsort[temp] != 0) {
                // 将直线的索引值赋值到 lineindex 数组。
                lineindex[k] = bufsort[temp];
                // 将直线的得票数赋值到 linevote 数组。
                linevote[k] = bufHough[bufsort[temp]];
                k++;
            }
        }
    }
    
    // 使用希尔排序，以得票数递减的顺序，为存在的直线排序。
    int i, j, tempvote, tempindex;
    // 希尔排序的增量。
    int gap = sum >> 1;
    
    while(gap > 0) { 
        // 对所有相隔 gap 位置的所有元素采用直接插入排序。
        for (i = gap; i < sum;i++) {
            tempvote = linevote[i];
            tempindex = lineindex[i];
            j = i - gap;
            // 对相隔 gap 位置的元素进行排序。
            while (j >= 0 && tempvote > linevote[j]) {
                linevote[j + gap] = linevote[j];
                lineindex[j + gap] = lineindex[j];
                j = j - gap;
            }
            linevote[j + gap] = tempvote;
            lineindex[j + gap] = tempindex;
            j = j - gap;
        }
        // 减小增量。
        gap = gap >> 1;
    }

    // 申请直线返回参数结构体，保存找到的可能直线。  
    line = new LineParam[sum];
    if (line == NULL)
        return OUT_OF_MEM;

    // 计算检测出的直线的参数：rho 以及 theta 的值，并
    // 保存在参数结构体中。
    float scale;
    scale = 1.0 / (numtheta + 2);
    for (int i = 0; i < sum; i++) {
        int idx = lineindex[i];
        int rho = (int)(idx * scale) - 1;
        // 根据原始计算方法反计算出 theta 的值。
        int theta = idx - (rho + 1) * (numtheta + 2) - 1;
        line[i].angle = theta * detheta;
        // 计算出直线参数 rho 的值。
        rho =(int)(rho - ((numrho - 1) / 2) - ((rho >= 0) ? -0.5f : 0.5f));
        line[i].distance = rho;
        // 将得票数保存在直线参数结构体中。
        line[i].votes = linevote[i];
    }

    // 统计最终检测的直线的个数。
    int linenum = 0;
    int diffdis,diffdis2;
    float diffang,diffang2;
    for (int i = 0; i < sum; i++) {
        // 若当前直线的参数结构体的得票数为 0，0是认为重复的直线
        // 则直接进行下次循环。
        if (line[i].votes <= 0)
            continue;
        for (int j = i + 1; j < sum; j++) {
            // 计算两条直线距离和角度的差值。
            diffang=abs(line[i].angle-line[j].angle);
            diffdis=abs(line[i].distance-line[j].distance);
            // 角度为1度和179度也很相似
            diffang2=abs(M_PI-line[i].angle-line[j].angle);
            // 角度相差180时，dis值异号，相加才相当于他们之差
            diffdis2=abs(line[i].distance+line[j].distance);
            // 若距离和角度的差值均小于设定的阈值，
            // 则认为这两条直线实质上是一条直线。
            if  ( (diffdis<thresdis && diffang<this->thresang)
                    || (diffdis2<thresdis && diffang2<this->thresang)){
                line[j].angle = 0.0f;
                line[j].distance = 0;
                line[j].votes = 0;
            }
        }
        // 检测出的直线数加 1。
        linenum++;
    }

    // 检测出的最大直线数。
    // 检测出的最大直线数是期望检测的最大直线数 linenum 和
    // 存在的直线数 linesmax[0] 二者中的较小值。
    linesmax[0] = (linenum < linesmax[0]) ? linenum : linesmax[0];

    // 将最终检测的直线的参数赋值到需要返回的直线参数结构体中。
    int n = 0;
    for (int i = 0; i < sum; i++) {
        // 若得票数不为 0，说明是检测出的直线，
        // 赋值到直线参数返回结构体中。 
        if (n == linesmax[0])
            break;
        if (line[i].votes > 0) {
            lineparam[n].angle = line[i].angle;
            lineparam[n].distance = line[i].distance;
            lineparam[n].votes = line[i].votes;
            // 标记加 1。
            n++;
        }
    }


    // 释放内存空间。
    FAIL_HOUGH_LINE_FREE;
    cudaFree(lineparamdev);

    // 处理完毕，退出。 
    return NO_ERROR;
}

// Host 成员方法：houghline（Hough 变换检测直线）
__host__ int HoughLine::houghLine(Image *inimg, CoordiSet *guidingset,
                                      int *linesmax, 
                                     LineParam *lineparam)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL && guidingset == NULL)
        return NULL_POINTER;
    
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 得到输入图像的宽和高。   
    int width, height;
    int xmin, ymin, xmax, ymax; 

    if (guidingset != NULL) {     
        // 输入图像为空，则根据输入点集得到最左、
        // 最右、最上、最下的坐标值。
        errcode = _findMinMaxCoordinates(guidingset, &xmin, &ymin,
                                         &xmax, &ymax);
        if (errcode != NO_ERROR)
            return errcode;
            
        // 计算得票数矩阵的宽和高。
        width = xmax-xmin ;
        height = ymax-ymin;
        
    } else {
        // 输入图像不为空，则根据输入图像的尺寸得到图像需要处理部分的宽和高。
        width = inimg->roiX2-inimg->roiX1;
        height = inimg->roiY2-inimg->roiY1;
    }

    // 计算rho 和 theta 的递增次数。为减少计算，numrho用了近似值的距离最大值。
    int numrho = (int)((width + height) * 2 + 1) / derho;
    int numtheta = (int)(M_PI / detheta);

    // 声明需要的指针变量。
    int *alldatadev = NULL;
    int *alldata = NULL;
    int *linedata = NULL;   
    LineParam *line = NULL;      
    LineParam *lineparamdev = NULL;   
   
    // 一次性申请 Device 端需要的所有空间。
    int *bufhoughdev = NULL, *bufsortdev = NULL, *sumdev = NULL; 
    cudaError_t cudaerrcode;                      
    cudaerrcode = cudaMalloc((void **)&alldatadev,
                             (1 + (numtheta + 2) * (numrho + 2) + 
                              numtheta * numrho) * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 通过偏移得到各指针的地址。
    sumdev = alldatadev;
    bufhoughdev = alldatadev + 1;
    bufsortdev = alldatadev + 1 + (numtheta + 2) * (numrho + 2);
    
    // 初始化 Hough 变换累加器在 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldatadev, 0, 
                             (1 + (numtheta + 2) * (numrho + 2) + 
                              numtheta * numrho) * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize; 

    
    // 根据输入坐标集是否为空，分两种情况进行：
    if (guidingset != NULL) {
        // 若输入坐标集不为空，则将该点集拷贝入 Device 内存。   
        errcode = CoordiSetBasicOp::copyToCurrentDevice(guidingset);
        if (errcode != NO_ERROR)
            return errcode; 
            

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。      
        blocksize.x = 16;
        blocksize.y = 1;
        gridsize.x = (guidingset->count+15)/16;
        gridsize.y = 1;
                
        // 调用核函数，对输入坐标集 guidingset 计算 Hough 累加矩阵。
        _houghlineCorKer<<<gridsize, blocksize>>>(*guidingset, bufhoughdev,  
                                                  numtheta, numrho, 
                                                  detheta, derho);
        if (cudaGetLastError() != cudaSuccess) {
            // 释放内存空间。
            FAIL_HOUGH_LINE_FREE;
            return CUDA_ERROR;
        }         
    } else {
        // 将输入图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(inimg);
            if (errcode != NO_ERROR)
                return errcode;
                

        // 提取输入图像的 ROI 子图像。
        ImageCuda insubimgCud;
        errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
        if (errcode != NO_ERROR)
            return errcode;
        


        // 若输入坐标集guidingset为空           
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。      
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / 
                     blocksize.x;
        gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / 
                     blocksize.y;
            
        // 调用核函数，对输入图像计算 Hough 累加矩阵。
        _houghlineImgKer<<<gridsize, blocksize>>>(insubimgCud, bufhoughdev,  
                                                  numtheta, numrho, 
                                                  detheta, derho);
        if (cudaGetLastError() != cudaSuccess) {
            // 释放内存空间。
            FAIL_HOUGH_LINE_FREE;
            return CUDA_ERROR;
        }     
    }      
    
    // 在 Host 端一次性申请全部所需的空间。
    int *bufHough = NULL, *bufsort = NULL;  
    int sum;
    alldata = new int [(numtheta + 2) * (numrho + 2) + numtheta * numrho];
    if (alldata == NULL)
        return OUT_OF_MEM;
  
    // 通过偏移得到各指针的地址。
    bufHough = alldata;
    bufsort = alldata + (numtheta + 2) * (numrho + 2);
    
    // 将 Kernel 函数中计算的得票数矩阵 bufHoughDev 拷贝至 Host 端。
    cudaerrcode = cudaMemcpy(bufHough, bufhoughdev, 
                             (numtheta + 2) * (numrho + 2) * sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;        
        return cudaerrcode;
    }   
 
    // 计算调用计算局部最大值的 kernel 函数的线程块的尺寸和线程块的数量。  
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (numtheta + blocksize.x - 1) / blocksize.x;
    gridsize.y = (numrho + blocksize.y - 1) / blocksize.y;
    
    // 调用计算局部最大值的 kernel 函数。
    _findlocalMaxKer<<<gridsize, blocksize>>>(bufhoughdev, bufsortdev, sumdev,
                                             numtheta, threshold);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return CUDA_ERROR;  
    }   
    
    // 将 Kernel 函数中计算的得票数 sumdev 拷贝至 Host 端。
    cudaerrcode = cudaMemcpy(&sum, sumdev, sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 将 Kernel 函数中计算的得票数矩阵 bufsortdev 拷贝至 Host 端。
    cudaerrcode = cudaMemcpy(bufsort, bufsortdev, 
                             numtheta * numrho * sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }

    // 在 Host 端申请存放直线得票数和索引的数组。
    int *linevote = NULL, *lineindex = NULL;
    
    // 根据计算出存在的直线数，一次性申请所需空间。
    linedata = new int [sum * 2];
    if (linedata == NULL)
        return OUT_OF_MEM;
    linevote = linedata;
    lineindex = linedata + sum;
    // 局部变量。
    int k = 0, temp;
    
    // 统计可能存在的直线的得票数和索引值。
    for (int j = 0; j < numrho; j++) {
        for (int i = 0; i < numtheta; i++){
            temp = j * numtheta + i;
            if (bufsort[temp] != 0) {
                // 将直线的索引值赋值到 lineindex 数组。
                lineindex[k] = bufsort[temp];
                // 将直线的得票数赋值到 linevote 数组。
                linevote[k] = bufHough[bufsort[temp]];
                k++;
            }
        }
    }
    
    // 使用希尔排序，以得票数递减的顺序，为存在的直线排序。
    int i, j, tempvote, tempindex;
    // 希尔排序的增量。
    int gap = sum >> 1;
    
    while(gap > 0) { 
        // 对所有相隔 gap 位置的所有元素采用直接插入排序。
        for (i = gap; i < sum;i++) {
            tempvote = linevote[i];
            tempindex = lineindex[i];
            j = i - gap;
            // 对相隔 gap 位置的元素进行排序。
            while (j >= 0 && tempvote > linevote[j]) {
                linevote[j + gap] = linevote[j];
                lineindex[j + gap] = lineindex[j];
                j = j - gap;
            }
            linevote[j + gap] = tempvote;
            lineindex[j + gap] = tempindex;
            j = j - gap;
        }
        // 减小增量。
        gap = gap >> 1;
    }



    // 申请直线返回参数结构体，保存找到的可能直线。  
    line = new LineParam[sum];
    if (line == NULL)
        return OUT_OF_MEM;

    // 计算检测出的直线的参数：rho 以及 theta 的值，并
    // 保存在参数结构体中。
    float scale;
    scale = 1.0 / (numtheta + 2);
    for (int i = 0; i < sum; i++) {
        int idx = lineindex[i];
        int rho = (int)(idx * scale) - 1;
        // 根据原始计算方法反计算出 theta 的值。
        int theta = idx - (rho + 1) * (numtheta + 2) - 1;
        line[i].angle = theta * detheta;
        // 计算出直线参数 rho 的值。
        rho =(int)(rho - ((numrho - 1) / 2) - ((rho >= 0) ? -0.5f : 0.5f));
        line[i].distance = rho;
        // 将得票数保存在直线参数结构体中。
        line[i].votes = linevote[i];
    }

    // 统计最终检测的直线的个数。
    int linenum = 0;
    int diffdis,diffdis2;
    float diffang,diffang2;
    for (int i = 0; i < sum; i++) {
        // 若当前直线的参数结构体的得票数为 0，0是认为重复的直线
        // 则直接进行下次循环。
        if (line[i].votes <= 0)
            continue;
        for (int j = i + 1; j < sum; j++) {
            // 计算两条直线距离和角度的差值。
            diffang=abs(line[i].angle-line[j].angle);
            diffdis=abs(line[i].distance-line[j].distance);
            // 角度为1度和179度也很相似
            diffang2=abs(M_PI-line[i].angle-line[j].angle);
            // 角度相差180时，dis值异号，相加才相当于他们之差
            diffdis2=abs(line[i].distance+line[j].distance);
            // 若距离和角度的差值均小于设定的阈值，
            // 则认为这两条直线实质上是一条直线。
            if  ( (diffdis<thresdis && diffang<this->thresang)
                    || (diffdis2<thresdis && diffang2<this->thresang)){
                line[j].angle = 0.0f;
                line[j].distance = 0;
                line[j].votes = 0;
            }
        }
        // 检测出的直线数加 1。
        linenum++;
    }

    // 检测出的最大直线数。
    // 检测出的最大直线数是期望检测的最大直线数 linenum 和
    // 存在的直线数 linesmax[0] 二者中的较小值。
    linesmax[0] = (linenum < linesmax[0]) ? linenum : linesmax[0];

    // 将最终检测的直线的参数赋值到需要返回的直线参数结构体中。
    int n = 0;
    for (int i = 0; i < sum; i++) {
        // 若得票数不为 0，说明是检测出的直线，
        // 赋值到直线参数返回结构体中。 
        if (n == linesmax[0])
            break;
        if (line[i].votes > 0) {
            lineparam[n].angle = line[i].angle;
            lineparam[n].distance = line[i].distance;
            lineparam[n].votes = line[i].votes;
            // 标记加 1。
            n++;
        }
    }


    // 释放内存空间。
    FAIL_HOUGH_LINE_FREE;
    cudaFree(lineparamdev);

    // 处理完毕，退出。 
    return NO_ERROR;
}

// Host 成员方法：houghlineimg（Hough 变换检测直线）
__host__ int HoughLine::houghLineImg(Image *inimg, CoordiSet *guidingset,
                                     Image *outimg, int *linesmax, 
                                     LineParam *lineparam)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL && guidingset == NULL)
        return NULL_POINTER;
    
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 得到输入图像的宽和高。   
    int width, height;
    int xmin, ymin, xmax, ymax; 

    if (guidingset != NULL) {     
        // 输入图像为空，则根据输入点集得到最左、
        // 最右、最上、最下的坐标值。
        errcode = _findMinMaxCoordinates(guidingset, &xmin, &ymin,
                                         &xmax, &ymax);
        if (errcode != NO_ERROR)
            return errcode;
            
        // 计算得票数矩阵的宽和高。
        width = xmax-xmin ;
        height = ymax-ymin;
        
    } else {
        // 输入图像不为空，则根据输入图像的尺寸得到图像需要处理部分的宽和高。
        width = inimg->roiX2-inimg->roiX1;
        height = inimg->roiY2-inimg->roiY1;
    }

    // 计算rho 和 theta 的递增次数。为减少计算，numrho用了近似值的距离最大值。
    int numrho = (int)((width + height) * 2 + 1) / derho;
    int numtheta = (int)(M_PI / detheta);

    // 声明需要的指针变量。
    int *alldatadev = NULL;
    int *alldata = NULL;
    int *linedata = NULL;   
    LineParam *line = NULL;      
    LineParam *lineparamdev = NULL;   
   
    // 一次性申请 Device 端需要的所有空间。
    int *bufhoughdev = NULL, *bufsortdev = NULL, *sumdev = NULL; 
    cudaError_t cudaerrcode;                      
    cudaerrcode = cudaMalloc((void **)&alldatadev,
                             (1 + (numtheta + 2) * (numrho + 2) + 
                              numtheta * numrho) * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 通过偏移得到各指针的地址。
    sumdev = alldatadev;
    bufhoughdev = alldatadev + 1;
    bufsortdev = alldatadev + 1 + (numtheta + 2) * (numrho + 2);
    
    // 初始化 Hough 变换累加器在 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldatadev, 0, 
                             (1 + (numtheta + 2) * (numrho + 2) + 
                              numtheta * numrho) * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize; 
    ImageCuda outsubimgCud;
    
    // 根据输入坐标集是否为空，分两种情况进行：
    if (guidingset != NULL) {
        // 若输入坐标集不为空，则将该点集拷贝入 Device 内存。   
        errcode = CoordiSetBasicOp::copyToCurrentDevice(guidingset);
        if (errcode != NO_ERROR)
            return errcode; 

        outimg->width = xmax;
        outimg->height = ymax;
        // 将输出图片拷贝至 Device 端。
        ImageBasicOp::copyToCurrentDevice(outimg);
        
        // 提取输出图像的 ROI 子图像。
        errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
        if (errcode != NO_ERROR)
            return errcode;
        outsubimgCud.imgMeta.width = xmax;
        outsubimgCud.imgMeta.height = ymax;
    
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。      
        blocksize.x = 16;
        blocksize.y = 1;
        gridsize.x = (guidingset->count+15)/16;
        gridsize.y = 1;
                
        // 调用核函数，对输入坐标集 guidingset 计算 Hough 累加矩阵。
        _houghlineCorKer<<<gridsize, blocksize>>>(*guidingset, bufhoughdev,  
                                                  numtheta, numrho, 
                                                  detheta, derho);
        if (cudaGetLastError() != cudaSuccess) {
            // 释放内存空间。
            FAIL_HOUGH_LINE_FREE;
            return CUDA_ERROR;
        }         
    } else {
        // 将输入图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(inimg);
            if (errcode != NO_ERROR)
                return errcode;
                
        // 将输出图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(outimg);
        if (errcode != NO_ERROR) {
            // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
            // 像的 ROI 子图像尺寸相同的图像。
            errcode = ImageBasicOp::makeAtCurrentDevice(
                    outimg, inimg->roiX2 - inimg->roiX1, 
                    inimg->roiY2 - inimg->roiY1);
            // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
            if (errcode != NO_ERROR)
                return errcode;
        }
            
        // 提取输入图像的 ROI 子图像。
        ImageCuda insubimgCud;
        errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
        if (errcode != NO_ERROR)
            return errcode;
        
        // 提取输出图像的 ROI 子图像。
        errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
        if (errcode != NO_ERROR)
            return errcode;
            
        // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一   
        if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width)
            insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
        else
            outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;
        
        if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height)
            insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
        else
            outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;

        // 若输入坐标集guidingset为空           
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。      
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / 
                     blocksize.x;
        gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / 
                     blocksize.y;
            
        // 调用核函数，对输入图像计算 Hough 累加矩阵。
        _houghlineImgKer<<<gridsize, blocksize>>>(insubimgCud, bufhoughdev,  
                                                  numtheta, numrho, 
                                                  detheta, derho);
        if (cudaGetLastError() != cudaSuccess) {
            // 释放内存空间。
            FAIL_HOUGH_LINE_FREE;
            return CUDA_ERROR;
        }     
    }      
    
    // 在 Host 端一次性申请全部所需的空间。
    int *bufHough = NULL, *bufsort = NULL;  
    int sum;
    alldata = new int [(numtheta + 2) * (numrho + 2) + numtheta * numrho];
    if (alldata == NULL)
        return OUT_OF_MEM;
  
    // 通过偏移得到各指针的地址。
    bufHough = alldata;
    bufsort = alldata + (numtheta + 2) * (numrho + 2);
    
    // 将 Kernel 函数中计算的得票数矩阵 bufHoughDev 拷贝至 Host 端。
    cudaerrcode = cudaMemcpy(bufHough, bufhoughdev, 
                             (numtheta + 2) * (numrho + 2) * sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;        
        return cudaerrcode;
    }   
 
    // 计算调用计算局部最大值的 kernel 函数的线程块的尺寸和线程块的数量。  
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (numtheta + blocksize.x - 1) / blocksize.x;
    gridsize.y = (numrho + blocksize.y - 1) / blocksize.y;
    
    // 调用计算局部最大值的 kernel 函数。
    _findlocalMaxKer<<<gridsize, blocksize>>>(bufhoughdev, bufsortdev, sumdev,
                                             numtheta, threshold);   
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return CUDA_ERROR;  
    }   
    
    // 将 Kernel 函数中计算的得票数 sumdev 拷贝至 Host 端。
    cudaerrcode = cudaMemcpy(&sum, sumdev, sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 将 Kernel 函数中计算的得票数矩阵 bufsortdev 拷贝至 Host 端。
    cudaerrcode = cudaMemcpy(bufsort, bufsortdev, 
                             numtheta * numrho * sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }

    // 在 Host 端申请存放直线得票数和索引的数组。
    int *linevote = NULL, *lineindex = NULL;
    
    // 根据计算出存在的直线数，一次性申请所需空间。
    linedata = new int [sum * 2];
    if (linedata == NULL)
        return OUT_OF_MEM;
    linevote = linedata;
    lineindex = linedata + sum;
    // 局部变量。
    int k = 0, temp;
    
    // 统计可能存在的直线的得票数和索引值。
    for (int j = 0; j < numrho; j++) {
        for (int i = 0; i < numtheta; i++){
            temp = j * numtheta + i;
            if (bufsort[temp] != 0) {
                // 将直线的索引值赋值到 lineindex 数组。
                lineindex[k] = bufsort[temp];
                // 将直线的得票数赋值到 linevote 数组。
                linevote[k] = bufHough[bufsort[temp]];
                k++;
            }
        }
    }
    
    // 使用希尔排序，以得票数递减的顺序，为存在的直线排序。
    int i, j, tempvote, tempindex;
    // 希尔排序的增量。
    int gap = sum >> 1;
    
    while(gap > 0) { 
        // 对所有相隔 gap 位置的所有元素采用直接插入排序。
        for (i = gap; i < sum;i++) {
            tempvote = linevote[i];
            tempindex = lineindex[i];
            j = i - gap;
            // 对相隔 gap 位置的元素进行排序。
            while (j >= 0 && tempvote > linevote[j]) {
                linevote[j + gap] = linevote[j];
                lineindex[j + gap] = lineindex[j];
                j = j - gap;
            }
            linevote[j + gap] = tempvote;
            lineindex[j + gap] = tempindex;
            j = j - gap;
        }
        // 减小增量。
        gap = gap >> 1;
    }

    // 申请直线返回参数结构体，保存找到的可能直线。  
    line = new LineParam[sum];
    if (line == NULL)
        return OUT_OF_MEM;

    // 计算检测出的直线的参数：rho 以及 theta 的值，并
    // 保存在参数结构体中。
    float scale;
    scale = 1.0 / (numtheta + 2);
    for (int i = 0; i < sum; i++) {
        int idx = lineindex[i];
        int rho = (int)(idx * scale) - 1;
        // 根据原始计算方法反计算出 theta 的值。
        int theta = idx - (rho + 1) * (numtheta + 2) - 1;
        line[i].angle = theta * detheta;
        // 计算出直线参数 rho 的值。
        rho =(int)(rho - ((numrho - 1) / 2) - ((rho >= 0) ? -0.5f : 0.5f));
        line[i].distance = rho;
        // 将得票数保存在直线参数结构体中。
        line[i].votes = linevote[i];
    }

    // 统计最终检测的直线的个数。
    int linenum = 0;
    int diffdis,diffdis2;
    float diffang,diffang2;
    for (int i = 0; i < sum; i++) {
        // 若当前直线的参数结构体的得票数为 0，0是认为重复的直线
        // 则直接进行下次循环。
        if (line[i].votes <= 0)
            continue;
        for (int j = i + 1; j < sum; j++) {
            // 计算两条直线距离和角度的差值。
            diffang=abs(line[i].angle-line[j].angle);
            diffdis=abs(line[i].distance-line[j].distance);
            // 角度为1度和179度也很相似
            diffang2=abs(M_PI-line[i].angle-line[j].angle);
            // 角度相差180时，dis值异号，相加才相当于他们之差
            diffdis2=abs(line[i].distance+line[j].distance);
            // 若距离和角度的差值均小于设定的阈值，
            // 则认为这两条直线实质上是一条直线。
            if  ( (diffdis<thresdis && diffang<this->thresang)
                    || (diffdis2<thresdis && diffang2<this->thresang)){
                line[j].angle = 0.0f;
                line[j].distance = 0;
                line[j].votes = 0;
            }
        }
        // 检测出的直线数加 1。
        linenum++;
    }

    // 检测出的最大直线数。
    // 检测出的最大直线数是期望检测的最大直线数 linenum 和
    // 存在的直线数 linesmax[0] 二者中的较小值。
    linesmax[0] = (linenum < linesmax[0]) ? linenum : linesmax[0];

    // 将最终检测的直线的参数赋值到需要返回的直线参数结构体中。
    int n = 0;
    for (int i = 0; i < sum; i++) {
        // 若得票数不为 0，说明是检测出的直线，
        // 赋值到直线参数返回结构体中。 
        if (n == linesmax[0])
            break;
        if (line[i].votes > 0) {
            lineparam[n].angle = line[i].angle;
            lineparam[n].distance = line[i].distance;
            lineparam[n].votes = line[i].votes;
            
            // 标记加 1。
            n++;
        }
    }
    
    // 在 Device 端申请内存空间用于存储直线的返回参数。                   
    cudaerrcode = cudaMalloc((void **)&lineparamdev, 
                             linesmax[0] * sizeof (LineParam));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        return cudaerrcode;
    }
    
    // 将计算得到的直线返回参数从 Host 端拷贝到 Device 端。
    cudaerrcode = cudaMemcpy(lineparamdev, lineparam, 
                             linesmax[0] * sizeof (LineParam),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        cudaFree(lineparamdev);
        return cudaerrcode;
    }
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;
    
    // 调用 kernel函数，得出最终输出图像。
    _houghoutKer<<<gridsize, blocksize>>>(outsubimgCud, lineparamdev, 
                                          linesmax[0], derho);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_HOUGH_LINE_FREE;
        cudaFree(lineparamdev);
        return CUDA_ERROR; 
    }

    // 释放内存空间。
    FAIL_HOUGH_LINE_FREE;
    cudaFree(lineparamdev);

    // 处理完毕，退出。 
    return NO_ERROR;
}

// Host 成员方法：getGlobalParam（ROI局部直线参数转换成为全局坐标下的参数）
// 把 hough 变换检测到的直线参数转换成为全局坐标下的参数
__host__ int                     // 返回值：函数是否正确执行，若函数
                                 // 否则返回假
    HoughLine::getGlobalParam(
    Image *inimg,             // 输入图像
    int *linesmax,          // 检测直线的最大数量
    LineParam *lineparam    // 直线返回参数结构体
){
    int rx=inimg->roiX1;
    int ry=inimg->roiY1;
    if(rx==0 && ry==0)
        return NO_ERROR;

    for (int i=0; i<*linesmax; i++) 
        lineparam[i].distance=lineparam[i].distance+
                              rx*cos(lineparam[i].angle)+
                              ry*sin(lineparam[i].angle);
    return NO_ERROR;
}

// Host 成员方法：realLine（判断给出线段的真实性）
__host__ bool  	HoughLine::realLine(
    Image *inimg,                   // 输入图像
    int x1,                         // 要判断的线段两端点坐标
    int y1,
    int x2,
    int y2,
    float threshold,                // 点是否在线段上的误差范围参数，1-3
    float thresperc                 // 线段真实性判定阈值，线段上有效点和线段理论上应该有的
                                    // 点的比值超过此阈值，认为线段真实存在
){
    // 对端点x、y坐标进行排序，方便判断范围
    int xmax, xmin, ymax, ymin;
    if(x1>x2){
        xmax=x1;
        xmin=x2;
    }
    else{
        xmax=x2;
        xmin=x1;
    }

    if(y1>y2){
        ymax=y1;
        ymin=y2;
    }
    else{
        ymax=y2;
        ymin=y1;
    }

    // 正常线段应该的有效点个数
    int pointnumfull=sqrt(0.0+(x1-x2) * (x1-x2)+(y1-y2) * (y1-y2));
    // 显存空间，用来存储内核函数计算出来的线段有效点个数
    int *pointnumdev=NULL;
    int cudaerrcode=cudaMalloc((void **)&pointnumdev,  sizeof (int));
    if (cudaerrcode != cudaSuccess)
    {
        // 释放内存空间。
        cudaFree(pointnumdev);
        return cudaerrcode;
    }
    cudaerrcode=cudaMemset(pointnumdev, 0, sizeof (int));
    if (cudaerrcode != cudaSuccess)
    {
        // 释放内存空间。
        cudaFree(pointnumdev);
        return cudaerrcode;
    }
    // 将输入图像拷贝入 Device 内存。
    int errcode=ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode=ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x=DEF_BLOCK_X;
    blocksize.y=DEF_BLOCK_Y;
    gridsize.x=(insubimgCud.imgMeta.width+blocksize.x-1) / blocksize.x;
    gridsize.y=(insubimgCud.imgMeta.height+blocksize.y-1) / blocksize.y;
    // 调用 kernel函数，计算线段上有效点个数
    _realLineKer <<< gridsize, blocksize>>>(insubimgCud, x1, y1, x2, y2,
                                            xmax, xmin, ymax, ymin,
                                            threshold, pointnumdev);
    // 把显存数据复制到内存中
    int pointnum=0;
    cudaerrcode=cudaMemcpy(&pointnum, pointnumdev,
                             sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess){
        // 释放内存空间。
        cudaFree(pointnumdev);
        return cudaerrcode;
    }
    // 判断计算得到的有效点数是否合理，从而判断线段真实性
#ifdef DEBUG
    cout << endl << "pointnum=" << pointNum << ", pointnumfull=" << pointNumfull << endl;
#endif
    if(pointnum>pointnumfull *thresperc)
        return true;
    else
        return false;
}

// 取消前面的宏定义。
#undef FAIL_HOUGH_LINE_FREE

