#include "ContourMatch.h"
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X   4
#define DEF_BLOCK_Y   2

// Device 子程序： _getDistanceDev(得到距离)
// 试算两点间的距离的平方值。
static __device__ int  // 返回值： 两点间的距离的平方值
_getDistanceDev(
        int x1,        // 第一个点的横坐标
        int y1,        // 第一个点的纵坐标
        int x2,        // 第二个点的横坐标
        int y2         // 第二个点的纵坐标
);

// Device 子程序： _ifInCurDev(是否在闭合轮廓内)
// 判断一个点是否在闭合轮廓曲线内。
static __device__ bool    // 返回值： 是否在闭合轮廓曲线内
_ifInCurDev(
        Curve curve,      // 输入的闭合轮廓曲线
        int x,            // 被判断点的横坐标
        int y             // 被判断点的纵坐标
);

// Kernel 函数： _conMatKer(轮廓匹配)
// 按宽度objCurve->curveBandWidth对轮廓曲线周围设定，设定值为
// objCurve->ownerObjectsIndices[0] +100，如果曲线是闭合轮廓，则将内部设定为
// objCurve->ownerObjectsIndices[0]的值。
static __global__ void    // 无返回值
_conMatKer(
        Curve curve,      // 输入曲线
        Image outimg,     // 输出图像
        ImageCuda inimg   // 输入图像
);


// Device 子程序： _getDistanceDev(得到距离)
static __device__ int _getDistanceDev(int x1, int y1, int x2, int y2)
{
    // 返回两个点间距离的平方值
    return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

}

// Device 子程序： _ifInCurDev(是否在闭合轮廓内)
static __device__ bool _ifInCurDev(Curve curve,int xidx,int yidx)
{

    // 寄存器变量，用来记录该点上方做出的射线与曲线的交点数。
    int downcount1 = 0; 
    // 寄存器变量，用来记录该点下方做出的射线与曲线的交点数。
    int downcount2 = 0;

    // 曲线上点的个数。
    int length = curve.curveLength;

    // 遍历曲线上的各点，和该点做出的射线做比较。
    for (int i = 0; i < length; i++) {
        // 取得当前要比较的曲线上的点的横坐标。
        int x = curve.crvData[2 * i];
        // 取得当前要比较的曲线上的点的纵坐标。
        int y = curve.crvData[2 * i + 1];

        // 曲线中的下一个点的位置。
        int j = (i + 1) % length;
        int x2 = curve.crvData[2 * j];

        // 曲线中上一个点的位置。
        int k = (i - 1 + length) % length;
        int x3 = curve.crvData[2 * k];

        // 曲线上的第 i 个点与当前点在同一列上
        if (x == xidx) {
            // 向下方竖起做一条射线，
            // 当曲线的上一个点和下一点在两侧时，或是上一个点在一侧，下一个点在一条线上时
            // 寄存器变量加1。
            if (y  < yidx) {
                if (((x3 - x) * (x2 - x) < 0) || ((x2 != x)&& (x3 == x))) {
                    downcount1++;
                }
            }

            // 向上方竖起做一条射线，
            // 当曲线的上一个点和下一点在两侧时，或是下一个点在一侧，上一个点在一条线上时
            // 寄存器变量加1。
            if (y > yidx) {
                if (((x3 - x) * (x2 - x) < 0)|| ((x3 != x)&& (x2 == x))) {
                    downcount2++;
                }
            }
        }

    }                  

    // 交点数均为奇数则判定在曲线内部
    if ((downcount1 % 2 == 1)&&(downcount2 % 2 == 1)) {
        return true;
    }

    // 如果交点数有偶数，说明不在轮廓内。
    return false;
}

// Kernel 函数： _conMatKer(轮廓匹配)
static __global__ void _conMatKer(Curve curve,Image outimg, ImageCuda inimg)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= outimg.width || r >= outimg.height)
        return;

    // 计算对应的图像数据数组下标。
    int idx = r * inimg.pitchBytes + c;

    // 曲线上点的个数。
    int length = curve.curveLength; 

    // 全局变量，当i值大于等于曲线点的个数时，说明已经遍历所有曲线上的点，且该点符合条件。
    int i;
    
    
    outimg.imgData[idx] = 0;

    // 遍历曲线上的各点判断该点是否符合条件，即该点与轮廓曲线的最短距离小于 curveBandWidth。
    for (i = 0; i < length; i++) {
        
        // 当发现该点距离某个曲线上的点满足条件时，对该位置设定值
        // ownerObjectsIndices[0] +100，并跳出循环。
        if(_getDistanceDev(c, r, curve.crvData[2 * i],
                           curve.crvData[2 * i + 1]) < 100/* curve.curveBandWidth * curve.crvMeta.curveBandWidth*/) {
            outimg.imgData[idx] = 200/*curve.ownerObjectsIndices[0] + 100*/;
            break;
        }
    }

    // 当曲线是闭合轮廓曲线时，则对轮廓内且不在轮廓边缘处的点设定值ownerObjectsIndices[0]。
    if (i >= length && (curve.closed == 1)) {
        if(_ifInCurDev(curve, c, r)) {
            outimg.imgData[idx] = 100/*curve.ownerObjectsIndices[0]*/;
        }
        
    }
}

__host__ int ContourMatch::contourMatch(Image *inimg, Image **outimg, Curve** curve, int num)
{
    // 检查输入输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;

    // 判断输入曲线是否为空
    if (curve == NULL)
        return NULL_POINTER;

    // 检查输入参数是否有数据
    //if (curve->curveLength <= 0)
        //return INVALID_DATA;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;


    for (int i = 0; i < num; ++i) {
        // 将输出图像拷贝到 Device 内存中。
        errcode = ImageBasicOp::copyToCurrentDevice(outimg[i]);
        if (errcode != NO_ERROR)
        {
            cout << "errcode = ImageBasicOp::copyToCurrentDevice(outimg[i]);" << endl;
            return CUDA_ERROR;
        }

        // 将输入曲线拷贝到 Device 内存中。
        errcode = CurveBasicOp::copyToCurrentDevice(curve[i]);
        if (errcode != NO_ERROR)
        {
            cout << "CurveBasicOp::copyToCurrentDevice(curve[i]);" << endl;
            return CUDA_ERROR;
        }
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 创建流
    cudaStream_t *stream = new cudaStream_t[num];
    for (int i = 0; i < num; ++i) {
        cudaStreamCreate(&stream[i]);
    } 

    // 在流中执行核函数。
    for (int i = 0; i < num; ++i) {
        _conMatKer<<<gridsize, blocksize,0,stream[i]>>>(*curve[i], *outimg[i],insubimgCud);

        if (cudaGetLastError() != cudaSuccess) {

            cout << "kernel error" << endl;
        return CUDA_ERROR;
        }
   }

    // 销毁流
   for (int i = 0; i < num; ++i) {
       cudaStreamDestroy(stream[i]);
   }
        
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。	
    return NO_ERROR;

}
