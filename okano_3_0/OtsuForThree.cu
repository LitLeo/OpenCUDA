// OtsuForThree.cu
// 根据图像像素的分散程度,自动找到两个最佳分割阈值，得到
// 图像的三值化结果。

#include "OtsuForThree.h"
#include "Histogram.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏： DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y  8

// 存储在常量内存中的概率集和均值集
    __constant__ float dev_W[256];
    __constant__ float dev_U[256];
	
// Kernel 函数：_OtsuForThree_ForwardKer（前向三值化）
static __global__ void      // Kernel 函数无返回值
_OtsuForThree_ForwardKer(
        ImageCuda inimg,             // 输入图像
        ImageCuda outimg,            // 输出图像
        unsigned char thresholda,    // 阈值1
        unsigned char thresholdb	 // 阈值2
);

// Kernel 函数：_OtsuForThree_ForwardKer（前向三值化）
static __global__ void _OtsuForThree_ForwardKer(ImageCuda inimg, 
                                        ImageCuda outimg,        
                                        unsigned char thresholda,
										unsigned char thresholdb)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线 程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像 素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理， 一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >=  inimg.imgMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp;
    intemp = inimg.imgMeta.imgData[inidx];
    
    // 一个线程处理四个像素。
    // 判断当前像素点的灰度值处于哪个阈值区间，并将该点的像素值设为阈值区间的
    // 前向阈值。线程中处理的第一个点。
    if (intemp <= thresholda) {
        outimg.imgMeta.imgData[outidx] = thresholda;
    }
    else if (intemp > thresholda && intemp <= thresholdb) {
        outimg.imgMeta.imgData[outidx] = thresholdb;
    }
    else if (intemp > thresholdb && intemp <= 255) {
        outimg.imgMeta.imgData[outidx] = 255;
    }

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += outimg.pitchBytes;
        intemp = inimg.imgMeta.imgData[inidx];

        // 判断当前像素点的灰度值处于哪个阈值区间，并将该点的像素值设为阈值区间的
        // 前向阈值。
        if (intemp <= thresholda) {
            outimg.imgMeta.imgData[outidx] = thresholda;
        }
        else if (intemp > thresholda && intemp <= thresholdb) {
	    outimg.imgMeta.imgData[outidx] = thresholdb;
	}
	else if (intemp > thresholdb && intemp <= 255) {
	    outimg.imgMeta.imgData[outidx] = 255;
	}
    }
}

// Kernel 函数：_OtsuForThree_BackwardKer（后向三值化）
static __global__ void      // Kernel 函数无返回值
_OtsuForThree_BackwardKer(
        ImageCuda inimg,             // 输入图像
        ImageCuda outimg,            // 输出图像
        unsigned char thresholda,    // 阈值1
        unsigned char thresholdb	 // 阈值2
);

// Kernel 函数：_OtsuForThree_BackwardKer（后向三值化）
static __global__ void _OtsuForThree_BackwardKer(ImageCuda inimg, 
                                        ImageCuda outimg,        
                                        unsigned char thresholda,
										unsigned char thresholdb)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线 程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像 素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理， 一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >=  inimg.imgMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp;
    intemp = inimg.imgMeta.imgData[inidx];
    
    // 一个线程处理四个像素。
    // 判断当前像素点的灰度值处于哪个阈值区间，并将该点的像素值设为阈值区间的
    // 前向阈值。线程中处理的第一个点。
    if (intemp < thresholda) {
        outimg.imgMeta.imgData[outidx] = 0;
    }
    else if (intemp >= thresholda && intemp < thresholdb) {
        outimg.imgMeta.imgData[outidx] = thresholda;
    }
    else if (intemp >= thresholdb && intemp <= 255) {
        outimg.imgMeta.imgData[outidx] = thresholdb;
    }

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += outimg.pitchBytes;
        intemp = inimg.imgMeta.imgData[inidx];

        // 判断当前像素点的灰度值处于哪个阈值区间，并将该点的像素值设为阈值区间的
        // 前向阈值。
        if (intemp < thresholda) {
            outimg.imgMeta.imgData[outidx] = 0;
        }
        else if (intemp >= thresholda && intemp < thresholdb) {
            outimg.imgMeta.imgData[outidx] = thresholda;
        }
        else if (intemp >= thresholdb && intemp <= 255) {
            outimg.imgMeta.imgData[outidx] = thresholdb;
        }
    }
}

// Kernel 函数：_CalcuVarianceKer （计算最小类内方差）
static __global__ void      // Kernel 函数无返回值
_CalcuVarianceKer(
        float * thres
);
// Kernel 函数：_CalcuVarianceKer （计算最小类内方差）
static __global__ void _CalcuVarianceKer(float * thres)
{	
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线 程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
	// 检查第一个像素点是否越界，如果越界，则不进行处理， 一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= 128 || r >=  128)
        return;
    int index = c * 128 + r;
    int counti = c;
    int countj = r + 128;
    float vara, varb, varc;
	
    // 每个线程计算一种分割情况下的类内方差总和，并通过对应关系，存储在相应下标的
    // 数组元素中。计算时，分别计算（0-t1）、（t1-t2）、（t2-255）三个类内方差。

    // 计算（0-t1）的类内方差 vara
    float Wk, Uk;
    Wk = dev_W[counti] - dev_W[0];
    if (Wk == 0.0)
        Uk = 0.0;
    else
	Uk = (dev_U[counti] - dev_U[0]) / Wk;
    vara = 0.0;
    for (int count = 1; count <= counti; count++) {
        vara += abs(count - Uk) * abs(count - Uk) * 
	                       (dev_W[count] - dev_W[count - 1]);
    }    
    // 计算（t1-t2）的类内方差 varb
    Wk = dev_W[countj] - dev_W[counti];
    if (Wk == 0.0)
    	Uk = 0.0;
    else
 	Uk = (dev_U[countj] - dev_U[counti]) / Wk;
    varb = 0.0;
    for (int count = counti; count <= countj; count++) {
        if (count < 1)
 	    continue;
        varb += abs(count - Uk) * abs(count - Uk) * 
                               (dev_W[count] - dev_W[count - 1]);
    }
    // 计算（t2-255）的类内方差varc
    Wk = dev_W[255] - dev_W[countj];
    if (Wk == 0.0)
 	Uk = 0.0;
    else
	Uk = (dev_U[255] - dev_U[countj]) / Wk;
    varc = 0.0;
    for (int count = countj; count <= 255; count++) {
 	varc += abs(count - Uk) * abs(count - Uk) * 
	                       (dev_W[count] - dev_W[count - 1]);
    }	
    // 将计算得到的方差和存储在数组中。
    thres[index] = vara + varb + varc;
}

// Host 成员方法：OtsuForThree（最佳二值化自动生成）
__host__ int OtsuForThree::otsuForThree(Image *inimg, Image *outimg)
{
    // 检查输入图像和输出图像是否为 NULL，如果为 NULL 直接报错返回 。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;
   
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
		
    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice (outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败）， 则会创建一个和输入图
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
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 根据子图像的大小对长，宽进行调整，选择长度小的长， 宽进行子图像的统一	
    if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width)
        insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
    else
        outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;
		
    if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height)
        insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
    else
        outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;
         
    // 调用直方图，获取图像的像素信息        
    Histogram h;
    
    // 图像的像素信息
    unsigned int his[256];
    h.histogram(inimg, his, true);
    
    // 图像总像素数 
    int sumpixel = 0;                    
    for (int i = 0; i < 256; i++) {
        sumpixel += his[i];
    }
    
    // 计算图像的概率信息、有聚合度的概率集和有聚合度的均值集合。
    float P[256];
    float W[256];
    float U[256];
	
    P[0] = (float)his[0] / (float)sumpixel;
    W[0] = P[0];
    U[0] = 0.0;
    for(int i = 1; i < 256; i++) {
        P[i] = (float)his[i] / (float)sumpixel;
 	W[i] = P[i] + W[i-1];
	U[i] = i * P[i] + U[i-1];
    }
	
    // 将概率集和均值集复制到常量内存中
    cudaMemcpyToSymbol(dev_W, W, sizeof(float) * 256);
    cudaMemcpyToSymbol(dev_U, U, sizeof(float) * 256);
	
    // 存储128×128个类内方差总和的数组
    float *hostthresholds = new float[16384];
    float *devthreshlods;

    // 为标记数组分配大小。
    errcode = cudaMalloc((void **)&devthreshlods, 16384 * sizeof (float));
    if (errcode != cudaSuccess) {
        cudaFree(devthreshlods);
        return errcode;
    }
	
    // 为标记数组设定初值。
    errcode = cudaMemset(devthreshlods, 0, 16384 * sizeof (float));
    if (errcode != cudaSuccess) {
        cudaFree(devthreshlods);
        return errcode;
    }

    // 将数组复制至 device 端。
    errcode = cudaMemcpy(devthreshlods, hostthresholds, 16384 * sizeof (float),
                             cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess) {
        cudaFree(devthreshlods);
        return errcode;
    }
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (128 + blocksize.x - 1) / blocksize.x;
    gridsize.y = (128 + blocksize.y - 1) / blocksize.y;
                 
    // 调用核函数，计算128×128种分割方式下的方差集合
    _CalcuVarianceKer<<<gridsize, blocksize>>>(devthreshlods);
	
    // 将数组复制至 host 端。
    errcode = cudaMemcpy(hostthresholds, devthreshlods, 16384 * sizeof (float),
                             cudaMemcpyDeviceToHost);
    if (errcode != cudaSuccess) {
        cudaFree(devthreshlods);
        return errcode;
    }
	
    // 串行计算，找出128×128个方差元素中的最小值	
    float min = 10000.0;
    int thresa = 0;
    int thresb = 0;
    
	// 计算数组的最小值
    for (int i = 0; i < 16384; i++) {
        if (min > hostthresholds[i]) {
            min = hostthresholds[i];
            // 通过对应成二维数组，得到两个对应的阈值
            thresa = i / 128;
            thresb = i % 128 + 128;
        }
    }
	        
    // 将阈值进行类型转换。
    unsigned char thresholda = (unsigned char)thresa;
    unsigned char thresholdb = (unsigned char)thresb;
	
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);
				 
    // 调用核函数，使用最佳阈值对图像进行二值化
    if (this-> isForward) {
	    _OtsuForThree_ForwardKer<<<gridsize, blocksize>>>(insubimgCud,  
                                        outsubimgCud,thresholda, thresholdb);
	}
    else {
	    _OtsuForThree_BackwardKer<<<gridsize, blocksize>>>(insubimgCud,  
                                        outsubimgCud,thresholda, thresholdb);
    }
	
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
        
    // 处理完毕，退出。	
    return NO_ERROR;
}

