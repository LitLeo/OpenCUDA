   // HoughCircle.cu
// 实现 Hough 变换检测圆

#include "HoughCircle.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"
#include "CoordiSet.h"

//#define DEBUG

// 宏：HOUGH_INF_GREAT
// 定义了一个足够大的正整数，该整数在使用过程中被认为是无穷大。
#define HOUGH_INF_GREAT  ((1 << 30) - 1)

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_YI
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：DEF_BLOCK_1D
// 定义了默认的一维线程块的尺寸。
#define DEF_BLOCK_1D  512

//----------------------------内核函数声明（10个）-----------------------------


// Kernel 函数：_houghcirImgKer（根据输入图像 inimg 计算得票数）
// 对输入图像的每一个有效像素点，寻找每一个该点可能在的圆，计算圆心以及半径，
// 统计得票数。
static __global__ void      // Kernel 函数无返回值
_houghcirImgKer(
        ImageCuda inimg,    // 输入图像。
        int bufHoughDev[],  // 得票数矩阵。
        int radiusMin,      // 最小检测的圆半径。
        int bufwidth,       // 得票数矩阵 bufHough 的宽度。
        int bufheight       // 得票数矩阵 bufHough 的高度。
);



// Kernel 函数：_findpartmaxKer（计算局部最大值）
// 在得票数矩阵 bufHough 中寻找局部最大值，当该位置的得票数大于其邻域的得票
// 数，并且大于圆的阈值，被认为是一个局部最大值，即一个可能圆。
static __global__ void 
_findpartmaxKer(
        int bufHoughDev[],  // 得票数矩阵。
        int bufsortDev[],   // 局部最值矩阵。
        int sumdev[],       // 存在的圆的个数。
        int threshold,      // 圆的阈值。
        int bufwidth,       // 得票数矩阵 bufHough 的宽度。
        int bufheights,     // 得票数矩阵 bufHough 的高度。
        int numperRDev[]    // 不同半径的圆的个数。
);



// Kernel 函数：_countcirbyRKer（按照不同的半径计算圆的个数）
// 根据局部最大值矩阵，按照不同的半径，统计每个局部最大值（可能圆）是当前半径
// 的第几个局部最大值，并将索引值保存在 bufsort 数组中。
static __global__ void 
_countcirbyRKer(
        int bufsortDev[],  // 局部最值矩阵。
        int bufwidth,      // 矩阵的宽度。
        int bufheight      // 矩阵的高度。
);

// Kernel 函数：_getcirinfKer（获得圆的得票数和索引信息）
// 将得到的可能圆的得票数和索引值保存在对应的数组中。
static __global__ void 
_getcirinfKer(
        int bufHoughDev[],  // 得票数矩阵。
        int bufsortDev[],   // 局部最值矩阵。
        int numperRDev[],   // 不同半径的圆的个数。
        int cirvoteDev[],   // 圆的得票数
        int cirindexDev[],  // 圆的索引值。
        int bufwidth,       // 矩阵的宽度。
        int bufheight       // 矩阵的高度。
);

// Kernel 函数: _shearToPosKer（转换数据形式）
// 对排序后的数组进行整理。
static __global__ void 
_shearToPosKer(
        int cirvoteDev[],   // 圆的得票数。
        int cirindexDev[],  // 圆的索引值。
        int lensec,         // 矩阵行数。
        int judge           // 块内共享内存的大小。
);

// Kernel 函数: _shearSortRowDesKer（行降序排序）
// 对待排序矩阵的每一行进行双调排序。
static __global__ void 
_shearSortRowDesKer(
        int cirvoteDev[],   // 圆的得票数。
        int cirindexDev[],  // 圆的索引值。
        int lensec,         // 矩阵行数。
        int judge           // 块内共享内存的大小。
);

// Kernel 函数: _shearSortColDesKer（列降序排序）
// 对待排序矩阵的每一列进行双调排序。
static __global__ void 
_shearSortColDesKer(
        int cirvoteDev[],   // 圆的得票数。
        int cirindexDev[],  // 圆的索引值。
        int length,         // 矩阵列数。
        int lensec,         // 矩阵行数。
        int judge           // 块内共享内存的大小。
);

// Kernel 函数：_calcirparamKer（计算圆的返回参数）
// 对按照得票数进行排序之后的圆，重新恢复参数，并保存在圆的返回参数结构体中。
static __global__ void 
_calcirparamKer(
        int cirvoteDev[],         // 圆的得票数。
        int cirindexDev[],        // 圆的索引值。
        CircleParam circleDev[],  // 圆的返回参数。
        int bufwidth,             // 矩阵的宽度。
        int bufheight,            // 矩阵的高度。
        int radiusMin             // 最小检测的圆半径。
);



// Kernel 函数：_houghoutKer（画出已检测到的圆）
// 根据圆的参数返回结构体，对最终已经检测到的圆，输出到 outimg 中。
static __global__ void 
_houghoutKer(
        ImageCuda outimg,           // 输出图像
        CircleParam cirparamdev[],  // 圆的参数结构体
        int circlenum               // 圆的个数
);

//----------------------------全局函数声明（3个）-------------------------------------
// Device 静态方法：_findcirsumDev（计算小于半径 radius 的圆的总个数）
// 根据上一步计算结果，按照不同半径存在的圆的个数进行累加，统计小于给定半径的
// 圆的总个数。
static __device__ int 
_findcirsumDev(
        int radius,       // 圆的半径。
        int numperRDev[]  // 不同半径的圆的个数。
);

// Host 函数：_recalCirParam（确定最终检测的圆和参数）
// 根据可能圆之间的距离信息，确定最终检测到的圆的个数，并还原圆的参数。
static __host__ int 
_recalCirParam(
        CircleParam circle[],      // 可能圆的参数结构体
        CircleParam *circleparam,  // 圆的参数结构体
        int *circleMax,            // 检测圆的最大数量
        int sum,                   // 可能圆的数量
        float distThres,             // 两个不同圆之间的最小距离
        int rThres                 // 区别两个圆的最小半径差别。
);

// Host 函数：_houghcirByImg（根据输入图像进行 Hough 圆检测）
// 根据输入图像，通过 Hough 变换进行圆检测。
static __host__ int 
_houghcirByImg(
        Image *inimg,              // 输入图像
        int *circleMax,            // 检测的圆的最大数量
        CircleParam *circleparam,  // 圆的参数结构体 
        int radiusMin,             // 最小检测的圆半径
        int radiusMax,             // 最大检测的圆半径
        int cirThreshold,          // 圆的阈值
        float distThres,             // 区别两个圆的最小距离。
        int rThres                 // 区别两个圆的最小半径差别。
);



//-----------------------------内核函数实现-------------------------------------------

// 宏：VOTE(x,y,z)
// 在三维投票空间中投票。
#define VOTE(x,y,z)  \
        if(x>=0 && x<inimgCud.imgMeta.width && y>=0 && y<inimgCud.imgMeta.height )\
        {int index=(z) * (bufwidth + 2) * (bufheight + 2)+((y) + 1) * (bufwidth + 2) + (x) + 1;\
        atomicAdd(&bufHoughDev[index], 1);}
// Kernel 函数：_houghcirImgKer（根据输入图像计算得票数）
// 采用bresenham画圆投票，效率提高
static __global__ void _houghcirImgKer(
        ImageCuda inimgCud, int bufHoughDev[], int radiusMin,
        int bufwidth, int bufheight)
{
    // 计算线程对应的输出点的位置，其中 x 和 y 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，x 表示 column；y 表示 row）。
    // r 代表当前线程处理的圆的半径的大小。
    // (x0,y0)点，对应的是（inimg->ROX1+x0,inimg->ROY1+y0),以ROI区域的左上角
    // （inimg->ROX1,inimg->ROY1）为原点
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (x0 >= inimgCud.imgMeta.width || y0 >= inimgCud.imgMeta.height)
        return;
 
    // 定义局部变量。
    unsigned char intemp;
    int radius;
        
    // 计算输入坐标点对应的图像数据数组下标。
    int inidx = y0 * inimgCud.pitchBytes + x0;
    // 读取第一个输入坐标点对应的像素值。
    intemp = inimgCud.imgMeta.imgData[inidx];
    // 根据当前 block 的 z 方向坐标计算需要计算的半径的值。r是z坐标，radius是
    radius = z + radiusMin;
    // 若当前像素点(x0, y0)是前景点（即像素值为 255)，以他为中心，radisu半径，
    // 投票空间画圆

    // 如果当前像素值为 255，即有效像素值，则在投票空间进行bresenham法画圆投票。
    if (intemp == 255) {
        int x, y,d;
        x = 0;
        y = radius;
        d = 3-2*radius;
        while(x < y){
            // 注意:x,y是以(0,0)为圆心得到的坐标，需要偏移到(x0,x0)坐标系中，z是
            // 投票空间第三维坐标，不是圆的半径，注意和radius区别。
            VOTE(x0+x,y0+y,z);
            VOTE(x0+x,y0-y,z);
            VOTE(x0-x,y0+y,z);
            VOTE(x0-x,y0-y,z);
            VOTE(x0+y,y0+x,z);
            VOTE(x0+y,y0-x,z);
            VOTE(x0-y,y0+x,z);
            VOTE(x0-y,y0-x,z);
            if(d < y)
                d += 4*x+6;
            else{
                d += 4*(x-y)+10;
                y--;
            }
            x++;
        }// while
    }// if (intemp == 255)
}// end of kernel
#undef VOTE

/*
// Kernel 函数：_houghcirImgKer（根据输入图像计算得票数）
static __global__ void _houghcirImgKer(
        ImageCuda inimg, int bufHoughDev[], int radiusMin,
        int bufwidth, int bufheight)
{
    // 计算线程对应的输出点的位置，其中 x 和 y 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，x 表示 column；y 表示 row）。
    // r 代表当前线程处理的圆的半径的大小。
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.z;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (x0 >= inimg.imgMeta.width || y0 >= inimg.imgMeta.height)
        return;
 
    // 定义局部变量。
    unsigned char intemp;
    int bufidx;
    int y, ymin, ymax;
    float tempx;
    int x, radius;
        
    // 计算输入坐标点对应的图像数据数组下标。
    int inidx = y0 * inimg.pitchBytes + x0;
    // 读取第一个输入坐标点对应的像素值。
    intemp = inimg.imgMeta.imgData[inidx];
    
    // 根据当前 block 的 z 方向坐标计算需要计算的半径的值。
    radius = r + radiusMin;
    // 若当前像素点(x, y)是前景点（即像素值为 1），则经过该点的半径为 r 的圆
    // 心纵坐标范围为（y - r, y + r）。
    // 计算该线程需要处理的圆心纵坐标的范围。
    ymin = max(0, (int)y0 - radius);
    ymax = min(y0 + radius, (int)inimg.imgMeta.height);

    // 如果当前像素值为 255，即有效像素值，则对该像素点进行圆检测。
    if (intemp == 255) {
        // 圆心纵坐标从 bmin 循环到 bmax，对于每一个可能的纵坐标值，
        // 计算其对应的圆心横坐标 a。若 a 在图像范围内，进行投票。
		//i是当前像素点高度坐标，a是当前点宽度坐标
        for (y = ymin; y < ymax + 1; y++){
            // 计算圆心横坐标 a的值。
            tempx = sqrtf((float)(radius * radius - (y0 - y) * (y0 - y)));

            // 左半圆投票
            x = (int)(fabs(x0 - tempx) + 0.5f);
            // 若 a 不在范围内，则跳出循环。 
            if (x <= 0 || x > inimg.imgMeta.width)
                continue;
            // 计算当前 (x, y, r) 在得票数矩阵中的索引值。
            bufidx = r * (bufwidth + 2) * (bufheight + 2) +
                     (y + 1) * (bufwidth + 2) + x + 1;
            // 使用原子操作进行投票。
            atomicAdd(&bufHoughDev[bufidx], 1);
            
            // 右半圆投票
            x = (int)(fabs(x0 + tempx) + 0.5f);
            // 若 a 不在范围内，则跳出循环。 
            if (x <= 0 || x > inimg.imgMeta.width)
                continue;
            // 计算当前 (x, y, r) 在得票数矩阵中的索引值。
            bufidx = r * (bufwidth + 2) * (bufheight + 2) +
                     (y + 1) * (bufwidth + 2) + x + 1;
            // 使用原子操作进行投票。
            atomicAdd(&bufHoughDev[bufidx], 1);
        }
    }
}
*/

// Kernel 函数：_findpartmaxKer（计算局部最大值）
static __global__ void _findpartmaxKer(
        int bufHoughDev[], int bufsortDev[], int sumdev[],
        int threshold, int bufwidth, int bufheight, int numperRDev[])
{        
    // 计算线程对应的输出点的位置，其中 x 和 y 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，x 表示 column；y 表示 row）。
    // r 代表当前线程处理的圆的半径的大小。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;    
    // 计算该线程在块内的相对位置。
    int inindex = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 申请共享内存，存该块内符合条件的局部最大值个数，即存在的圆的个数。
    __shared__ int totalsum[1];
    
    // 初始化所有块内的共享内存。
    if (inindex == 0)
        totalsum[0] = 0;
    // 块内同步。
    __syncthreads();
    
    // 计算当前线程在 bufHough 矩阵中的对应索引值。
    int index = z * (bufwidth + 2) * (bufheight + 2) +
                (r + 1) * (bufwidth + 2) + c + 1;
    int idx = z * bufwidth * bufheight + r * bufwidth + c;

    // 当前线程的得票数大于圆的阈值，并且大于邻域中的值时，认为是局部最大值，
    // 即可能是圆。
    if (bufHoughDev[index] > threshold &&
        bufHoughDev[index] > bufHoughDev[index - 1] &&
        bufHoughDev[index] >= bufHoughDev[index + 1] &&
        bufHoughDev[index] > bufHoughDev[index - bufwidth - 2] &&
        bufHoughDev[index] >= bufHoughDev[index + bufwidth + 2]) {
        bufsortDev[idx] = bufHoughDev[index];
        // 使用原子操作对局部最大值进行统计。
        atomicAdd(&numperRDev[z], 1);
        atomicAdd(&totalsum[0], 1);
    } else {
        bufsortDev[idx] = 0;
    }

    // 块内同步。
    __syncthreads();
    
    // 将统计出的圆的个数统计到 sumdev 中。
    if (inindex == 0 && totalsum[0] != 0) {
        atomicAdd(&sumdev[0], totalsum[0]); 
    }
}

// Kernel 函数：_countcirbyRKer（按照不同的半径计算圆的个数）
static __global__ void _countcirbyRKer(
        int bufsortDev[], int bufwidth, int bufheight)
{
    // 计算线程的索引，即圆的半径。
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    // 初始化圆的个数为 1。
    int count = 1;
    // 计算该线程对应的局部最大值矩阵 bufsort 中的索引值。
    int idx = r * bufwidth * bufheight;
    int index;  
    
    // 半径为 r，对矩阵 bufsort 进行统计，得到该半径的圆的个数。
    for (int j = 0; j < bufheight; j++) {
        index = idx;
        for (int i = 0; i < bufwidth; i++) {
            // 若矩阵 bufsort 当前位置的值不为 0，则为其赋值 count，表示该
            // 局部最大值是半径为 r 的圆中的第 count 个。
            if (bufsortDev[index] != 0) {
                bufsortDev[index] = count;
                count++; 
            }
            index += 1; 
        }
        idx += bufwidth;       
    }
}

// Kernel 函数：_getcirinfKer（获得圆的得票数和索引信息）
static __global__ void _getcirinfKer(
        int bufHoughDev[], int bufsortDev[],  int numperRDev[], 
        int cirvoteDev[], int cirindexDev[], int bufwidth, int bufheight)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    // z 代表当前线程处理的圆的半径的大小。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    
    // 计算当前线程在 bufsort 矩阵中的对应索引值。
    int index = z * bufwidth * bufheight + r * bufwidth + c;
    int ciridx, idx;
    
    // 如果 bufsort 矩阵当前位置的值不为 0，则说明是局部最大值。
    if (bufsortDev[index] != 0) {
        // 计算该局部最大值的输出位置。该位置等于半径小于 z 的所有圆数
        // 加上该局部最大值在当前半径中的位置。
        ciridx = bufsortDev[index] + _findcirsumDev(z, numperRDev) - 1;
        
        // 将该局部最大值的索引信息赋值到 cirindex 数组中。
        // 该索引是不加边框的bufSortDev中的索引。
        cirindexDev[ciridx] = index;

        // 计算在得票数矩阵 bufHough 中的索引值。
        idx = z * (bufwidth + 2) * (bufheight + 2) +
              (r + 1) * (bufwidth + 2) + c + 1;
        // 将该局部最大值的得票数赋值到 cirvote 数组中。
        cirvoteDev[ciridx] = bufHoughDev[idx];
    }  
}

// Kernel 函数: _shearToPosKer（转换数据形式）
static __global__ void _shearToPosKer(
        int cirvoteDev[], int cirindexDev[], int lensec, int judge)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    extern __shared__ int shared[];
    // 通过偏移，获得存放得票数和索引值的两部分共享内存空间。
    int *vote, *index;
    vote = shared;
    index = shared + judge;
    // 为得票数和索引值赋初始值。
    vote[cid] = cirvoteDev[rid * lensec + cid];
    index[cid] = cirindexDev[rid * lensec + cid];
    // 块内同步。
    __syncthreads();
    
    // 偶数行赋值。
    if (rid % 2 == 0) {
        cirvoteDev[rid * lensec + cid] = vote[cid];
        cirindexDev[rid * lensec + cid] = index[cid];
    } else {
        // 奇数行赋值。
        cirvoteDev[rid * lensec + cid] = vote[lensec - 1 - cid];
        cirindexDev[rid * lensec + cid] = index[lensec - 1 - cid];
    }
}

// Kernel 函数: _shearSortRowDesKer（行降序排序）
static __global__ void _shearSortRowDesKer(
        int cirvoteDev[], int cirindexDev[], int lensec, int judge)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    extern __shared__ int shared[];
    // 通过偏移，获得存放得票数和索引值的两部分共享内存空间。
    int *vote, *index;
    vote = shared;
    index = shared + judge;
    
    // 为共享内存赋初始值。
    if (cid < lensec) {
        vote[cid] = cirvoteDev[rid * lensec + cid];
        index[cid] = cirindexDev[rid * lensec + cid];
    }
    // 块内同步。
    __syncthreads();

    // 声明临时变量
    int ixj, tempvote, tempindex;
    // 偶数行降序排序。
    if (rid % 2 == 0) {
        for (int k = 2; k <= lensec; k <<= 1) {
             // 双调合并。
            for (int j = k >> 1; j > 0; j >>= 1) {
                // ixj 是与当前位置 cid 进行比较交换的位置。
                ixj = cid ^ j;
                if (ixj > cid) {
                    // 如果 (cid & k) == 0，按照降序交换两项。
                    if ((cid & k) == 0 && (vote[cid] < vote[ixj])) {
                        // 交换得票数。                        
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = index[cid];
                        index[cid] = index[ixj];
                        index[ixj] = tempindex; 
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    } else if ((cid & k) != 0 && vote[cid] > vote[ixj]) {
                        // 交换得票数。                     
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = index[cid];
                        index[cid] = index[ixj];
                        index[ixj] = tempindex; 
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
                    if ((cid & k) == 0 && (vote[cid] > vote[ixj])) {
                        // 交换得票数。                        
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = index[cid];
                        index[cid] = index[ixj];
                        index[ixj] = tempindex; 
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    } else if ((cid & k) != 0 && vote[cid] < vote[ixj]) {
                        // 交换得票数。 
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = index[cid];
                        index[cid] = index[ixj];
                        index[ixj] = tempindex; 
                    }
                }   
                __syncthreads();
            }
        }    
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    if (cid <lensec) {
        cirvoteDev[rid * lensec + cid] = vote[cid];
        cirindexDev[rid * lensec + cid] = index[cid];
    }
}

// Kernel 函数: _shearSortColDesKer（列降序排序）
static __global__ void _shearSortColDesKer(
        int cirvoteDev[], int cirindexDev[], 
        int length, int lensec, int judge)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    // 判断是否越界。
    if (rid >= lensec)
        return;

    extern __shared__ int shared[];
    // 通过偏移，获得存放得票数和索引值的两部分共享内存空间。
    int *vote, *index;
    vote = shared;
    index = shared + judge;
    
    // 为共享内存赋初始值。
    if (cid < length) {
        vote[cid] = cirvoteDev[rid + cid * lensec];
        index[cid] = cirindexDev[rid + cid * lensec];
    }
    // 块内同步。
    __syncthreads();

    // 声明临时变量。
    int ixj, tempvote, tempindex;
    // 并行双调排序，降序排序。
    for (int k = 2; k <= length; k <<= 1) {
        // 双调合并。
        for (int j = k >> 1; j > 0; j >>= 1) {
            // ixj 是与当前位置 cid 进行比较交换的位置。
            ixj = cid ^ j;
            if (ixj > cid) {
                // 如果 (cid & k) == 0，按照降序交换两项。
                if ((cid & k) == 0 && (vote[cid] < vote[ixj])) {
                    // 交换得票数。
                    tempvote = vote[cid];
                    vote[cid] = vote[ixj];
                    vote[ixj] = tempvote;
                    // 交换索引值。
                    tempindex = index[cid];
                    index[cid] = index[ixj];
                    index[ixj] = tempindex; 
                // 如果 (cid & k) == 0，按照升序交换两项。
                } else if ((cid & k) != 0 && vote[cid] > vote[ixj]) {
                    // 交换得票数。
                    tempvote = vote[cid];
                    vote[cid] = vote[ixj];
                    vote[ixj] = tempvote;
                    // 交换索引值。
                    tempindex = index[cid];
                    index[cid] = index[ixj];
                    index[ixj] = tempindex;
                }
            }
            __syncthreads();
        }
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    if (cid < length) {
        cirvoteDev[rid + cid * lensec] = vote[cid];
        cirindexDev[rid + cid * lensec] = index[cid];
    }
}

// Kernel 函数：_calcirparamKer（计算圆的返回参数）
static __global__ void _calcirparamKer(
        int cirvoteDev[], int cirindexDev[], CircleParam circleDev[],
        int bufwidth, int bufheight, int radiusMin)
{
    // 获取线程号。
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 声明局部变量。
    int radius, x, y, temp;
    // 获取当前圆在 cirindex 矩阵中的索引值。
    int idx = cirindexDev[index];
    // 矩阵的大小。
    // 矩阵的大小。
    int size = (bufwidth) * (bufheight);
    // 计算当前圆的半径。
    radius = idx / size;
    temp = idx - radius * size;
    // 计算当前圆的圆心的纵坐标。
    y = temp / (bufwidth);
    // 计算当前圆的圆心的横坐标。
    x = temp % (bufwidth); 
    
    // 为当前圆的返回参数进行赋值。
    circleDev[index].a = x;
    circleDev[index].b = y;
    circleDev[index].radius = radius + radiusMin;
    circleDev[index].votes = cirvoteDev[index];
}


// Kernel 函数：_houghoutKer（画出已检测到的圆）
static __global__ void _houghoutKer(
        ImageCuda outimg, CircleParam cirparamdev[], int circlenum)
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
    *outptr = 0;

    // 声明局部变量
    int i, temp, radius, a, b;

    // 对所有已经检测出的圆进行循环，找到输入图像中对应的点，并赋值 128。
    for (i = 0; i < circlenum; i++) {
        // 得到圆的参数，圆心 (a, b)， 圆的半径 radius。
        radius = cirparamdev[i].radius;
        a = cirparamdev[i].a;
        b = cirparamdev[i].b;

        // 计算当前像素点 (c, r) 到该圆心 (a, b) 的距离。
        temp = (c - a) * (c - a) + (r - b) * (r - b);

        // 若该距离小于 20，则认为是该圆上的点，在输出图像中赋值 128。
        if (abs(temp - radius * radius) < 50)
            *outptr = 255;
    }
}

//-----------------------------全局函数实现-------------------------------------------
// 函数：_findMinMaxCoordinates(根据输入点集的坐标，找到最上、最下、最左、最右
// 的点，从而确定图像的宽和高)
static __host__ int _findMinMaxCoordinates(CoordiSet *guidingset, 
                                           int *xmin, int *ymin,
                                           int *xmax, int *ymax)
{
    // 声明局部变量。
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
    for (int i = 1;i < tmpcoordiset->count;i++) {
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

// Device 静态方法：_findcirsumDev（计算小于半径 radius 的圆的总个数）
static __device__ int _findcirsumDev(int radius, int numperRDev[])
{
    int n = radius;
    // 将圆的总个数初始化为 0。
    int cirsum = 0;

    // 计算小于所给半径 radius 的圆的总个数，并赋值给 cirsum。
    while (--n >= 0) {
        cirsum += numperRDev[n];
    }
    // 返回计算所得的圆的总个数。
    return cirsum;
}


// Host 静态方法：_cirSortLoop（shear 排序核心函数）
static __host__ int _cirSortLoop(int cirvoteDev[], int cirindexDev[],
                                 int length, int lensec)
{
    // 检查数组是否为 NULL，如果为 NULL 直接报错返回。
    if (cirvoteDev == NULL || cirindexDev == NULL)
        return NULL_POINTER;

    // 计算二维数组中长和宽的较大值。
    int judge;
    if (length > 0 && lensec > 0)
        judge = (length > lensec) ? length : lensec;

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;       

    for (int i = length; i >= 1; i >>= 1) {
        // 首先进行列排序。
        _shearSortColDesKer<<<judge, judge, 2 * judge * sizeof (int)>>>
                (cirvoteDev, cirindexDev, length, lensec, judge);
                    
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;    
                               
        // 然后进行行排序。
        _shearSortRowDesKer<<<judge, judge, 2 * judge * sizeof (int)>>>
                (cirvoteDev, cirindexDev, lensec, judge);

        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;
    }
    // 整理排序后的数组。
    _shearToPosKer<<<length, lensec, 2 * judge * sizeof (int)>>>
            (cirvoteDev, cirindexDev, lensec, judge);            
    // 若调用 CUDA 出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return NO_ERROR;
}

// Host 函数：_recalCirParam（确定最终检测的圆和参数）
// 从circle中选出有效的、不重复的，放入circleparam中
static __host__ int _recalCirParam(
        CircleParam *circle, CircleParam *circleparam, 
        int *circleMax, int sum, float distThres, int rThres)
{
    // 根据两个圆的距离关系，确定最终检测出的圆。
    int a1, a2, b1, b2, diffr;
    float distance;
    // 统计最终检测的圆的个数。
    int circlenum = 0;
    for (int i = 0; i < sum; i++) {
        // 若当前圆的参数结构体的得票数为 0，直接进行下次循环。
        if (circle[i].votes == 0 || circle[i].radius == 0)
            continue;
        for (int j = i + 1; j < sum; j++) {
            // 得到两个圆的圆心的坐标。
            a1 = circle[i].a;
            b1 = circle[i].b;
            a2 = circle[j].a;
            b2 = circle[j].b;
            // 计算两个圆半径差值
            diffr = abs(circle[i].radius - circle[j].radius);
            // 计算两个圆的圆心 (a1, b1), (a2, b2) 的距离。
            distance = (float)(a1 - a2) * (a1 - a2) + (b1 - b2) * (b1 - b2);
            // 圆心和半径都相近的，认为是同一个圆
            if (distance < distThres * distThres && diffr < rThres) {
                // 合并后的圆参数取平均值，票数合并 ，放入i中
                circle[i].a = (circle[i].a+circle[j].a)/2;
                circle[i].b = (circle[i].b+circle[j].b)/2;
                circle[i].radius = (circle[i].radius+circle[j].radius)/2;
                circle[i].votes = circle[i].votes+circle[j].votes;
                // j中另一圆取消
                circle[j].a = 0;
                circle[j].b = 0;
                circle[j].radius = 0;
                circle[j].votes = 0;
            }
        }
        // 检测出的圆的个数加 1。
        circlenum++;
    }
    
    // 根据circlenum以及期望检测出的圆的个数circleMax，确定最终圆的个数。
    circleMax[0] = (circlenum < circleMax[0]) ? circlenum : circleMax[0];
    
    // 将最终检测的圆的参数赋值到需要返回的圆的参数结构体中。
    int k = 0;
    for (int i = 0; i < sum; i++) {        
        // 赋值到最后一个圆时，结束循环。
        if (k >= circleMax[0])
            break;
        // 若得票数不为 0，说明是检测出的圆，赋值到圆的返回参数结构体中。
        // 票数为零 则说明是被合并到其他圆中，直接跳过。
        if (circle[i].votes != 0) {
            circleparam[k].a = circle[i].a;
            circleparam[k].b = circle[i].b;
            circleparam[k].radius = circle[i].radius;
            circleparam[k].votes = circle[i].votes;
            // 标记加 1。
            k++;
        }
    }
    return NO_ERROR;
}


// 宏：FAIL_CIRCLE_IMG_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_CIRCLE_IMG_FREE  do {        \
        if (alldataDev != NULL)           \
            cudaFree(alldataDev);         \
        if (cirdataDev != NULL)           \
            cudaFree(cirdataDev);         \
        if (circleDev != NULL)            \
            cudaFree(circleDev);          \
        if (circle != NULL)               \
            delete[] circle;              \
    } while (0)

// Host 静态方法：_houghcirByImg（根据输入图像进行 Hough 圆检测）
// 根据输入图像 inimg，通过 Hough 变换进行圆检测。
static __host__ int _houghcirByImg(
        Image *inimg, int *circleMax, CircleParam *circleparam, 
        int radiusMin, int radiusMax, int cirThreshold, float distThres,int rThres)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;

    int errcode; // 局部变量，错误码  
    int bufwidth, bufheight;
    int rangeR = radiusMax - radiusMin + 1;

    // 输入图像不为空，则根据输入图像的ROI区域得到ROI区域的宽和高。
    bufwidth = inimg->roiX2-inimg->roiX1;
    bufheight = inimg->roiY2-inimg->roiY1;
    
    // 定义设备端的输入输出数组指针，当输入输出指针在 Host 端时，在设备端申请对
    // 应大小的数组。
    int *alldataDev = NULL;
    int *cirdataDev = NULL;
    CircleParam *circleDev = NULL;
    CircleParam *circle = NULL; 
    
    // 声明 Device 端需要的所有空间。
    int *bufHoughDev = NULL, *bufsortDev = NULL;
    int *sumdev = NULL, *numperRDev = NULL;
    cudaError_t cudaerrcode;  
    
    // 一次性申请 Device 端需要的所有空间。   
    cudaerrcode = cudaMalloc((void **)&alldataDev,
                             (1 + rangeR + bufwidth * bufheight * rangeR +
                              (bufwidth + 2) * (bufheight + 2) * rangeR) *
                             sizeof (int));
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

    // 通过偏移得到各指针的地址。
    sumdev = alldataDev;
    numperRDev = alldataDev + 1;
    bufsortDev = alldataDev + 1 + rangeR;
    bufHoughDev = alldataDev + 1 +rangeR+ bufwidth * bufheight * rangeR;

    // 初始化 Hough 变换累加器在 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldataDev, 0,
                             (1 + rangeR + bufwidth * bufheight * rangeR +
                              (bufwidth + 2) * (bufheight + 2) * rangeR) *
                             sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return CUDA_ERROR;
    }

    // 将输入图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return errcode;
    }
            
    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return errcode;
    }

    // 调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。      
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = 1;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / 
                 blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / 
                 blocksize.y;
    gridsize.z = rangeR;

    // 调用核函数，对输入图像计算 Hough 累加矩阵。
    _houghcirImgKer<<<gridsize, blocksize>>>(
            insubimgCud, bufHoughDev, radiusMin, bufwidth, bufheight);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return CUDA_ERROR;
    }

    // 重新计算调用 Kernel 函数的线程块的尺寸和线程块的数量。 
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = 1;
    gridsize.x = (bufwidth + blocksize.x - 1) / blocksize.x;
    gridsize.y = (bufheight + blocksize.y - 1) / blocksize.y;
    gridsize.z = rangeR;

    // 调用核函数，对 bufHough 矩阵寻找局部最大值。
    _findpartmaxKer<<<gridsize, blocksize>>>(
            bufHoughDev, bufsortDev, sumdev, cirThreshold,
            bufwidth, bufheight, numperRDev);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;     
        return CUDA_ERROR;
    }

    // 可能存在的圆的数量。
    int sum;
    // 将计算得到的可能存在的圆的数量 sum 拷贝到 Host 端。
    cudaerrcode = cudaMemcpy(&sum, sumdev, sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return CUDA_ERROR;
    }
    if(sum<=0){
        *circleMax=0;
        return NO_ERROR;
    }
    // 重新计算调用 Kernel 函数的线程块的尺寸和线程块的数量。 
    blocksize.x = (rangeR > DEF_BLOCK_1D) ? DEF_BLOCK_1D : rangeR;
    blocksize.y = 1;
    blocksize.z = 1;
    gridsize.x = (rangeR + blocksize.x - 1) / blocksize.x;
    gridsize.y = 1;
    gridsize.z = 1;

    // 调用核函数，统计不同半径的圆的个数。
    _countcirbyRKer<<<blocksize, gridsize>>>(bufsortDev, bufwidth, bufheight);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;                
        return CUDA_ERROR;
    }

    // 对统计出的可能存在的圆的总数 sum,
    // 取大于或者等于它的最小的 2 的幂次方数。
    int index = (int)ceil(log(sum*1.0) / log(2.0f));
    if (index > sizeof (int) * 8 - 1)
        return OP_OVERFLOW;
    int sortlength = (1 << index);

    // 声明 Device 端需要的所有空间。
    int *cirvoteDev = NULL, *cirindexDev = NULL;
    // 一次性申请 Device 端需要的所有空间。   
    cudaerrcode = cudaMalloc((void **)&cirdataDev,
                             (2 * sortlength) * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return CUDA_ERROR;
    }

    // 初始化 Device 上的内存空间。
    cudaerrcode = cudaMemset(cirdataDev, 0, (2 * sortlength) * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return CUDA_ERROR;
    }

    // 通过偏移获得数组的地址。
    cirindexDev = cirdataDev;
    cirvoteDev = cirdataDev + sortlength;
   
    // 重新计算调用 Kernel 函数的线程块的尺寸和线程块的数量。 
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = 1;
    gridsize.x = (bufwidth + blocksize.x - 1) / blocksize.x;
    gridsize.y = (bufheight + blocksize.y - 1) / blocksize.y;
    gridsize.z = rangeR;
 
    // 调用核函数，计算可能圆的索引以及得票数值。
    _getcirinfKer<<<gridsize, blocksize>>>(
            bufHoughDev, bufsortDev, numperRDev,
            cirvoteDev, cirindexDev, bufwidth, bufheight);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return CUDA_ERROR;
    }

    // 需要使用并行的 Shear Sort 对可能圆按照得票数大小
    // 排序，定义排序时需要的数组宽度和高度。    
    int sortwidth = (sortlength > 256) ? 256 : sortlength;
    int sortheight = (sortlength + sortwidth - 1) / sortwidth;
    
    // 调用并行 Shear Sort 算法，对可能圆按照得票数排序。
    errcode = _cirSortLoop(cirvoteDev, cirindexDev, sortwidth, sortheight);
    if (errcode != NO_ERROR) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return errcode;
    }

    #ifdef DEBUG
    int *cirIndex=(int*)malloc(sortlength*sizeof (int));
    int *cirVote=(int*)malloc(sortlength*sizeof (int));
    // 将计算得到的极值点数量 sum 拷贝到 Host 端。
    cudaerrcode = cudaMemcpy(cirIndex, cirindexDev, sortlength*sizeof (int),
                             cudaMemcpyDeviceToHost);
    cudaerrcode = cudaMemcpy(cirVote, cirvoteDev, sortlength*sizeof (int),
                             cudaMemcpyDeviceToHost);
    int size = (bufwidth) * (bufheight);
    for(int n=0;n<sum;n++){
        // 计算当前圆的半径。
        int radius = cirIndex[n] / size;
        int temp = cirIndex[n] - radius * size;
        // 计算当前圆的圆心的纵坐标。
        int b = temp / (bufwidth+2);
        // 计算当前圆的圆心的横坐标。
        int a = temp % (bufwidth+2); 
         printf("[%2d] index=%10d  vote=%5d (%3d,%3d) r=%3d\n",n,cirIndex[n],cirVote[n],a,b,radius);
    }
    delete[]cirIndex;
    delete[]cirVote;
    #endif

    // 申请 Device 端需要的存放圆的返回参数的空间。         
    cudaerrcode = cudaMalloc((void **)&circleDev,
                             sum * sizeof (CircleParam));
    if (cudaerrcode != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;    
        return CUDA_ERROR;
    }

    // 重新计算调用 Kernel 函数的线程块的尺寸和线程块的数量。 
    blocksize.x = (sum > DEF_BLOCK_1D) ? DEF_BLOCK_1D : sum;
    blocksize.y = 1;
    blocksize.z = 1;
    gridsize.x = (sum + blocksize.x - 1) / blocksize.x;
    gridsize.y = 1;
    gridsize.z = 1; 

    // 调用核函数，计算圆的返回参数。
    _calcirparamKer<<<gridsize, blocksize>>>(
            cirvoteDev, cirindexDev, circleDev, bufwidth, bufheight, 
            radiusMin);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;               
        return CUDA_ERROR;
    }

    // 为圆的参数返回结构体分配空间。
    circle = new CircleParam[sum];

    // 将核函数计算出的圆的返回参数复制到 Host 端中。
    cudaerrcode = cudaMemcpy(circle, circleDev,
                             sum * sizeof (CircleParam),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return CUDA_ERROR;
    }

    // 调用函数 _recalCirParam 计算最终检测的圆的数量以及参数。
    errcode = _recalCirParam(circle, circleparam, circleMax, sum, distThres,rThres);
    if (errcode != NO_ERROR) {
        // 释放之前申请的内存。
        FAIL_CIRCLE_IMG_FREE;
        return errcode;
    }

    // 释放之前申请的内存。
    cudaFree(alldataDev);
    cudaFree(cirdataDev);
    cudaFree(circleDev);
    delete[] circle;

    // 处理完毕，退出。 
    return NO_ERROR;
}

// 取消前面的宏定义。
#undef FAIL_CIRCLE_IMG_FREE

// 全局方法：_drawCircle（把圆参数数组绘制到图像上）
__host__ int _drawCircle(Image *resultimg,
                         int *circleMax,
                          CircleParam *circleparam
                        ){

    int errcode;  // 局部变量，错误码
    cudaError_t cudaerrcode;

        CircleParam *circleDev = NULL;
        // 为 device 端圆返回参数数组申请空间。
        cudaerrcode = cudaMalloc((void **)&circleDev,
                             circleMax[0] * sizeof (CircleParam));
        if (cudaerrcode != cudaSuccess) {
            cudaFree(circleDev);
             return CUDA_ERROR;
        }

        // 将计算得到的参数从 Host 端拷贝到 Device 端。
        cudaerrcode = cudaMemcpy(circleDev, circleparam,
                             circleMax[0] * sizeof (CircleParam),
                             cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            cudaFree(circleDev);
            return CUDA_ERROR;
        }
        // 将结果图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(resultimg);
        if (errcode != NO_ERROR)
            return errcode;
        // 提取结果图像的 ROI 子图像。
        ImageCuda resultimgCud;
        errcode = ImageBasicOp::roiSubImage(resultimg, &resultimgCud);
        if (errcode != NO_ERROR)
            return errcode;

        dim3 blocksize, gridsize;
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        blocksize.z = 1;
        gridsize.x = (resultimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
        gridsize.y = (resultimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;
        gridsize.z = 1;
        
        // 调用 kernel函数，得出最终输出图像。
        _houghoutKer<<<gridsize, blocksize>>>(
                resultimgCud, circleDev, circleMax[0]);
        if (cudaGetLastError() != cudaSuccess) 
            return CUDA_ERROR; 


        return NO_ERROR;
}

//-----------------------------成员函数实现-------------------------------------------

// Host 成员方法：houghcircle（Hough 变换检测圆）
__host__ int HoughCircle::houghcircle(Image *inimg, CoordiSet *guidingset,
                                      int *circleMax, CircleParam *circleparam,
                                      bool writetofile)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL && guidingset == NULL)
        return NULL_POINTER;

    int errcode;  // 局部变量，错误码


    // 声明输出图像
    Image *resultimg;
    ImageBasicOp::newImage(&resultimg);

    if (guidingset != NULL) {
        // 若输入坐标集不为空，则将该点集拷贝入 Device 内存。   
        errcode = CoordiSetBasicOp::copyToCurrentDevice(guidingset);
        if (errcode != NO_ERROR)
            return errcode;
        // 计算坐标集的最大坐标位置，为创建图像做准备
        int minx,miny,maxx,maxy;
        int errorcode=_findMinMaxCoordinates(guidingset,&minx,&miny,&maxx,&maxy);
        if(errorcode!=NO_ERROR)
            return INVALID_DATA;

        // 根据输入 coordiset 创建输入图像 coorimg
        Image *coorimg;
        ImageBasicOp::newImage(&coorimg);
        //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1，保证轮廓外连通
        ImageBasicOp::makeAtHost(coorimg,maxx+2 ,maxy+2);
        // coordiset 转成coorimg ，把坐标集绘制到图像上,前景255，背景0
        ImgConvert imgcvt(255,0);
        imgcvt.cstConvertToImg(guidingset,coorimg);
        #ifdef DEBUG
            // 把填充前的图像coorimg保存到文件
            ImageBasicOp::copyToHost(coorimg);
            ImageBasicOp::writeToFile("coorimg.bmp",coorimg);
         #endif

        // 根据输入图像 coorimg 进行 Hough 变换圆检测。
        errcode = _houghcirByImg(coorimg, circleMax, circleparam, radiusMin, 
                                 radiusMax, cirThreshold, distThres,rThres);
        if (errcode != NO_ERROR)
            return errcode;
        // 清除输入图像 coorimg 
        ImageBasicOp::deleteImage(coorimg);
        // 如果需要输出图像,分配空间
        if(writetofile)
            ImageBasicOp::makeAtHost(resultimg,maxx+2 ,maxy+2);
    } else {

        // 输入图像不为空，则根据输入图像进行 Hough 变换圆检测。
        errcode = _houghcirByImg(inimg, circleMax, circleparam, radiusMin, 
                                 radiusMax, cirThreshold, distThres,rThres);
        if (errcode != NO_ERROR)
            return errcode;

     // 分片局部坐标转化成全局坐标
      for(int i=0; i< *circleMax; i++) {
            circleparam[i].a+=inimg->roiX1;
            circleparam[i].b+=inimg->roiY1;
          }

        // 如果需要输出图像,分配空间
        if(writetofile)
            ImageBasicOp::makeAtHost(resultimg,
                                    inimg->width,
                                    inimg->height);
    }


    // 如果需要输出图像，调用 kernel 写入到“result.bmp”中
    if(writetofile)
        _drawCircle(resultimg, circleMax, circleparam);

    // 检测结果写入文件
    ImageBasicOp::copyToHost(resultimg);
    ImageBasicOp::writeToFile("result.bmp",resultimg);
    // 删除输出图像img
    ImageBasicOp::deleteImage(resultimg);
    // 处理完毕，退出。 
    return NO_ERROR;
}



// Host 成员方法：pieceCircle(分片检测inimg中的圆形，放入数组返回)
__host__ int 
HoughCircle:: pieceCircle(
    Image *inimg,                   // 输入待检测的图形
    int piecenum,                   // 每个维度上分块数量
    int *circleMax,                 // 返回检测到的圆数量
    CircleParam *circleparam,       // 返回检测到的圆参数
    bool writetofile                // 是否把检测结果写到文件中
){
    int pointer=0;

    // 计算分片的大小
    int cell_x=inimg->width/piecenum;
    int cell_y=inimg->height/piecenum;
    #ifdef DEBUG
        printf("cell_x=%d cell_y=%d\n",cell_x,cell_y);
    #endif

    // 开始分块处理
    for(int y=0;y<piecenum;y++)
        for(int x=0;x<piecenum;x++)
        {//.......................分块第一阶段..........................
            #ifdef DEBUG
             printf(" \n----------------- y=[%d] x=[%d]\n",y,x);
            #endif
            // 每个分片中圆上限
            int piececirmax=10;
             CircleParam *piececirparam= new CircleParam[piececirmax];
             for(int i=0;i<piececirmax;i++){
                  piececirparam[i].a=-1;
                  piececirparam[i].b=-1;
                  piececirparam[i].radius=-1;
                  piececirparam[i].votes=-1;
             }
             inimg->roiX1=x*cell_x;
             inimg->roiX2=x*cell_x+cell_x-1;
             inimg->roiY1=y*cell_y;
             inimg->roiY2=y*cell_y+cell_y-1; 

             #ifdef DEBUG
             printf("x1=%d x2=%d y1=%d y2=%d \n",
                 inimg->roiX1,inimg->roiX2,
                 inimg->roiY1,inimg->roiY2);
             #endif
             // 下面函数运行后，piececirmax中放的是检测到的圆的个数。
             // houghcircle（）返回后得到的是全局坐标
             houghcircle(inimg, NULL, &piececirmax, piececirparam,false);

             // 分片圆结果放入全局数组
             for(int i=0; i< piececirmax; i++) {
                    if(pointer>=*circleMax)break;
                    circleparam[pointer]=piececirparam[i];
                    pointer++;

                 }
            // 循环内声明的局部动态内存，循环内回收
            if(piececirparam!=NULL)
                {delete[] piececirparam;piececirparam=NULL;}

        //.........................分块第二阶段........................
        if(x<piecenum-1 && y<piecenum-1){
            }
        }// end of for x,for y

     // 回收资源

     // 返回真实矩形的个数
     *circleMax=pointer;
     if(*circleMax<=0)
        return NO_ERROR;

    // 如果需要输出图像，调用 kernel 写入到“result.bmp”中
    if(writetofile)
    {   
        Image *resultimg;
        ImageBasicOp::newImage(&resultimg);
        ImageBasicOp::makeAtHost(resultimg,inimg->width,inimg->height);

        _drawCircle(resultimg, circleMax, circleparam);

        // 检测结果写入文件
        ImageBasicOp::copyToHost(resultimg);
        ImageBasicOp::writeToFile("result.bmp",resultimg);
        ImageBasicOp::deleteImage(resultimg);
        }
    return NO_ERROR;
  }




