// LocalHistogramEqualization.cu
// 对指定图像进行直方图均衡化处理。

#include <iostream> 
#include <math.h>
#include "LocalHistogramEqualization.h"
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 定义图像灰度值个数
#define HISTO_NUM  256     


// Kernel 函数：_hisEqualHistoKer（并行实现所有窗口的直方图统计）
// 根据分割数将图像分割成窗口后，对每个窗口进行直方图统计工作。
static __global__ void
_hisEqualHistoKer(
    ImageCuda inimg,  // 输入图像 
    int *his,         // 各个窗口的直方图
    int ww,           // 窗口宽度
    int wh,           // 窗口长度
    int blockperwinw, // 窗每个窗口横向对应的线程块数目
    int blockperwinh  // 窗每个窗口纵向对应的线程块数目
);

// Kernel 函数：_hisEqualKer（并行实现归一化直方图的计算）
// 各个窗口的直方图统计完成后，对每个直方图进行归一化工作。
static __global__ void
_hisEqualKer(
    int *his,         // 各个窗口的直方图
    float *norhis,    // 各个窗口归一化直方图
    int *max,         // 各个窗口最大灰度
    int *min,         // 各个窗口最小灰度
    int total         // 窗口总像素点数
);

// Kernel 函数：_hisEqualLastKer（实现直方图均衡化操作）
// 根据归一化直方图和原始图像，计算cumucounter和cumugray。
static __global__ void
_hisEqualLastKer(
    ImageCuda inimg,  // 输入图像
    float *norhis,    // 各个窗口归一化直方图
    int *max,         // 各个窗口最大灰度
    int *min,         // 各个窗口最小灰度
    int ww,           // 窗口宽度
    int wh,           // 窗口长度
    int blockperwinw, // 窗每个窗口横向对应的线程块数目
    int blockperwinh, // 窗每个窗口纵向对应的线程块数目
    int *cumucounter, // 窗口重叠数
    int *cumugray     // 均衡化结果
);


// Kernel 函数：_hisEqualSecKer（实现将各窗口重新整合成输出图像）
// 各个窗口直方图均衡化完成后，此函数将根据所有窗口的均衡化结果重新整合成输出
// 图像。
static __global__ void
_hisEqualSecKer(
    ImageCuda inimg,   // 输入图像
    ImageCuda outimg,  // 输出图像
    int *cumucounter,  // 窗口重叠数量
    int *cumugray,     // 均衡化结果
    unsigned char t0,  // 外部参数
    float c1,          // 外部参数
    float c2,          // 外部参数
    float weight       // 外部参数
);

// Kernel 函数：_hisEqualHistoKer（并行实现所有窗口的直方图统计）
static __global__ void _hisEqualHistoKer(ImageCuda inimg, int *his, 
                                         int ww, int wh,
                                         int blockperwinw, int blockperwinh)
{
    // 申请大小为灰度图像灰度级 256 的共享内存，其中下标代表图像的灰度值，数
    // 组用来累加等于该灰度值的像素点个数。
    __shared__ unsigned int temp[HISTO_NUM];
    
    // 标记当前线程块对应的窗口的横纵索引以及总索引数
    __shared__ int winnumx, winnumy, winidx;
    
    // dstc 和 dstr 分别表示线程坐标的 x 和 y 分量（其中，c 表示
    // column，r 表示 row）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算该线程在块内的相对位置。
    int inindex = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 每块的第一个线程负责计算当前线程块对应的窗口的横纵索引以及总索引数。
    if(inindex == 0)
    {
        // 当前线程块对应的窗口的横纵索引。
        winnumx = blockIdx.x / blockperwinw ;
        winnumy = blockIdx.y / blockperwinh ;
        // 窗口索引
        winidx = (gridDim.x / blockperwinw) * winnumy + winnumx; 
       
    }
   
    // 若线程在块内的相对位置小于 256，即灰度级大小，则用来给共享内存赋初值 0。
    if (inindex < HISTO_NUM)
        temp[inindex] = 0;
   
    // 进行块内同步，保证执行到此处，共享内存的数组中所有元素的值都为 0。
    __syncthreads();
    
    // 计算窗口内相对坐标。
    int inidxc = dstc - winnumx * blockDim.x * blockperwinw;
    int inidxr = dstr - winnumy * blockDim.y * blockperwinh;

    // 计算线程对应于图像数据的坐标。
    int inidx = (winnumx * ww / 2 + inidxc) + inimg.pitchBytes * 
                (winnumy * wh / 2 + inidxr);
    
    // 输入坐标点对应的像素值。
    int curgray ;
    
    // 检查像素点是否越界，如果不越界，则进行统计。 
    if(inidxc < ww && inidxr < wh){
        curgray = inimg.imgMeta.imgData[inidx];
        // 使用原子操作实现+1操作，可以防止多个线程同时更改数据而发生的写错误。
        // 灰度值统计数组对应数目+1
        atomicAdd(&temp[curgray], 1);
    }
    
    // 块内同步。此处保证图像中所有点的像素值都被统计过。
    __syncthreads();
    
    // 用每一个块内前 256 个线程，将共享内存 temp 中的结果保存到输出数组中。
    // 每个窗口直方图对应his的一段
    if (inindex < HISTO_NUM && temp[inindex] != 0)
        atomicAdd(&his[inindex + winidx * HISTO_NUM], temp[inindex]);
        
}

// Kernel 函数：_hisEqualKer（并行实现归一化直方图的计算）
static __global__ void _hisEqualKer(int *his, float *norhis, 
                                    int *max, int *min, int total)
{
    // 计算当前线程块的索引，即对应的窗口索引。
    int winindx = blockIdx.y * gridDim.x + blockIdx.x;
    
    // 计算当前线程在块内的索引。
    int inindex = threadIdx.y * blockDim.x + threadIdx.x;

    // 计算当前线程块负责处理的直方图的起始下标
    int inxstart = winindx * HISTO_NUM;
    
    // 统计各个灰度值出现的概率
    // 起始下标加上块内索引即为对应直方图数组的下标。
    norhis[inindex + inxstart] = 1.0 *  his [inindex + inxstart] / total;

    // 线程同步，保证概率统计完毕
     __syncthreads();
     
     // 块内第一个线程计算累计归一化直方图，
     if(inindex == 0){
        // 数组各个位置存在计算先后顺序关系，故串行实现。
        // 起始下标加上索引i即为对应直方图数组的下标。
        for (int i = 1;i < HISTO_NUM;i++){
            norhis[i + inxstart] += norhis[i - 1 + inxstart] ;
        }   
    }
    
    // 块内第2个线程查找最大最小灰度值max ,min
    // 由于对于有序短数组并行查找最值效率不高，故对于每个直方图分别直接串行计算
    // 起始下标加上索引i即为对应直方图数组的下标。
    if(inindex == 1){
        // 第一个不为零的位置即为最小灰度值
        for (int i = 0; i < HISTO_NUM; i++) { 
            if(his[i + inxstart] != 0) {
                 min[winindx] = i;
                break;
            }
        }          
        
        // 最后一个不为零的位置即为最大灰度值
        for (int i = HISTO_NUM - 1; i >= 0; i--) {  
            if (his[i + inxstart] != 0) {
                max[winindx] = i;
                break;
            }
         }

    }
   
}

// Kernel 函数：_hisEqualLastKer（实现直方图均衡化操作）
 static __global__ void _hisEqualLastKer(ImageCuda inimg, float *norhis,    
                                         int *max, int *min, int ww, int wh, 
                                         int blockperwinw, int blockperwinh, 
                                         int *cumucounter, int *cumugray )
{    
     // 标记当前线程块对应的窗口的横纵索引以及总索引数
    int winnumx, winnumy, winidx;
     // 计算当前线程块对应的窗口的横纵索引以及总索引数。
    winnumx = blockIdx.x / blockperwinw ;
    winnumy = blockIdx.y / blockperwinh ;
    // 窗口索引
    winidx = (gridDim.x / blockperwinw) * winnumy + winnumx; 
    
    // dstc 和 dstr 分别表示线程坐标的 x 和 y 分量
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算窗口内相对坐标。
    int inidxc = dstc - winnumx * blockDim.x * blockperwinw;
    int inidxr = dstr - winnumy * blockDim.y * blockperwinh;

     // 检查像素点是否越界，如果不越界，则进行统计。
    if(inidxc < ww && inidxr < wh){
    
        // 计算线程对应于图像数据的坐标。
        int inidx = (winnumx * ww / 2 + inidxc) + inimg.pitchBytes * 
                    (winnumy * wh / 2 + inidxr);
                    
        // 计算与图像同维度的数组（ cumugray 和 cumucounter ）的对应下标 。
        int inidx2 = (winnumx * ww / 2 + inidxc) + inimg.imgMeta.width * 
                     (winnumy * wh / 2 + inidxr);
                     
        // 线程对应的直方图的起始下标，即对应的窗口索引乘以256
        int inxstart = winidx * HISTO_NUM;    
        
        // 计算最大最小灰度值之差
        int sub =  max[winidx]- min[winidx];
        
        // 将均衡化结果累加于 cumugray 数组，同时 cumucounter++  
        // 通过原子操作完成累加，避免内存写竞争引起的错误
        atomicAdd(&cumugray[inidx2], norhis[inimg.imgMeta.imgData[inidx] +
                                     inxstart] * sub + min[winidx]);  
        atomicAdd(&cumucounter[inidx2], 1);
    }
     
}

// Kernel 函数：_hisEqualSecKer（实现将各窗口计算结果重新整合成输出图像）
static __global__ void _hisEqualSecKer(ImageCuda inimg, ImageCuda outimg, 
                                       int *cumucounter, int *cumugray, 
                                       unsigned char t0,
                                       float c1, float c2, float weight)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量（其中，c 表示
    // column，r 表示 row）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= outimg.imgMeta.width || dstr >= outimg.imgMeta.height )
        return;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * inimg.pitchBytes + dstc;
    
    // 获取重叠计数数组和灰度数组下标。
    int curpos2 = dstr * inimg.imgMeta.width+ dstc;
    
    // 获得当前像素点的像素值。
    unsigned char g = inimg.imgMeta.imgData[curpos];

    // 对原图像的像素值进行处理。
    float gray;
    if (g <= t0) {
        gray = 0.0f;
    } else if(g >= 250) {
        gray = 255.0f;
    } else {
        gray = c1 * (logf(g) - c2);
    }
    
    // 根据原图像和均衡化的结果计算输出图像。 
    outimg.imgMeta.imgData[curpos] = (unsigned char)((cumugray[curpos2] /
                                      cumucounter[curpos2] - gray) * weight + 
                                      gray);  
}

// 成员方法：localHistEqual（图像直方图均衡化）
__host__ int LocalHistEqual::localHistEqual(Image *inimg, Image* outimg)
{
    // 局部变量，错误码。
    int errcode;  
    cudaError_t cudaerrcode;
 
    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 将输出图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 提取输入图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 提取输出图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        return errcode;
    }
    // 计算窗口的宽度和长度。
    int ww = inimg->width / divNum;
    int wh = inimg->height / divNum;

    // 获得图像内像素点的总数量。
    int totalnum = inimg->width * inimg->height;


    // 为核函数所需的 global 内存在 device 端开辟空间。
    // 为均衡化结果和窗口重叠数量一次性申请所有空间，然后通过偏移索引
    // 各个数组。 
    int *devdata;
    cudaerrcode = cudaMalloc((void **)&devdata, 
                             sizeof (int) * totalnum * 2);
    // 开辟失败，释放内存空间。
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devdata);
        return CUDA_ERROR;
    }
    int *devcumucounter = devdata;
    int *devcumugray = &devdata[totalnum];


    // 初始化为 0 
     cudaerrcode = cudaMemset(devdata, 0, sizeof (int) * totalnum * 2);
        if (cudaerrcode != cudaSuccess) {   
            cudaFree(devdata);
            return CUDA_ERROR;
        }
        
    // 图像窗口的总数
    int winnum = (divNum * 2 - 1) * (divNum * 2 - 1);

    // 每个窗口的直方图和最大值最小值三个数组一次性申请所有空间，
    // 然后通过偏移索引各个数组。 
    int *his, *max, *min;
    cudaerrcode = cudaMalloc((void **)&his, sizeof (int) * winnum * 
                             (HISTO_NUM + 2));               
    
    // 开辟失败，释放内存空间。
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devdata);
        cudaFree(his);
        return CUDA_ERROR;
    }
    max = &his[winnum * HISTO_NUM];
    min = &max[winnum];
    
    // 初始化为0
     cudaerrcode = cudaMemset(his, 0, sizeof (int) * winnum * (HISTO_NUM + 2));                
        if (cudaerrcode != cudaSuccess) {   
            cudaFree(devdata);
            cudaFree(his);
            return CUDA_ERROR;
        }
        
    // 归一化直方图
    float *norhis;
    cudaerrcode = cudaMalloc((void **)& norhis, 
                             sizeof(float) * winnum * HISTO_NUM );
    
    // 开辟失败，释放内存空间。
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devdata);
        cudaFree(his);
        cudaFree(norhis);
        return CUDA_ERROR;
    }
    // 初始化为0
    cudaerrcode = cudaMemset(norhis, 0.0f, sizeof (float) * winnum * HISTO_NUM);
     // 开辟失败，释放内存空间。
    if (cudaerrcode != cudaSuccess) { 
        cudaFree(devdata);
        cudaFree(his);        
        cudaFree(norhis);
        return CUDA_ERROR;
    }

    // 根据外部参数 t0 计算 c1 和 c2 两个参数的值。
    float c1 = 255 / log(1.0f * 250 / (t0 + 1)); 
    float c2 = log((t0 + 1) * 1.0f);

    // 计算调用均衡化函数的线程块的尺寸和线程块的数量。  
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X ;
    blocksize.y = DEF_BLOCK_Y ;

    // 每个窗口横向线程块数目
    int blockperwinw = (ww + blocksize.x - 1) / blocksize.x,
    // 每个窗口纵向线程块数目
    blockperwinh = (wh + blocksize.y - 1) / blocksize.y;
    
    // 线程总规模
    gridsize.x = blockperwinw * (divNum * 2 - 1) ;
    gridsize.y = blockperwinh * (divNum * 2 - 1) ;
     
    // 统计直方图
    _hisEqualHistoKer<<<gridsize, blocksize>>>(insubimgCud, his, ww, wh,
                                               blockperwinw, blockperwinh );
    if (cudaGetLastError() != cudaSuccess) {
        // 核函数出错。
        cudaFree(his);
        cudaFree(norhis);
        cudaFree(devdata);
        return CUDA_ERROR;
    }
  
    // 窗口内总像素点数
     int total = ww * wh;
     
     // 线程块数目即为窗口数目
     dim3 gridsize2;
     gridsize2.x = divNum * 2 - 1;
     gridsize2.y = divNum * 2 - 1;
     
     // 计算归一化直方图和最大最小值
    _hisEqualKer<<<gridsize2, blocksize>>>(his, norhis, max, min, total);
    if (cudaGetLastError() != cudaSuccess) {
        // 核函数出错。
        cudaFree(his);
        cudaFree(norhis);
        cudaFree(devdata);
        return CUDA_ERROR;
    }  
       
   
    // 计算各个窗口均衡化最终结果，gridsize与统计直方图时相同
    // 由于窗口长、宽、每个窗口横纵向线程块数目等信息已经计算过,
    // 在核函数中直接使用会减少一定的计算量，所以作为参数传入
    _hisEqualLastKer<<<gridsize, blocksize>>>(insubimgCud, norhis, 
                                              max, min, ww, wh, 
                                              blockperwinw,  blockperwinh,
                                              devcumucounter, devcumugray );
    if (cudaGetLastError() != cudaSuccess) {
        // 核函数出错。
        cudaFree(his);
        cudaFree(norhis);
        cudaFree(devdata);
        return CUDA_ERROR;
    }
        
    // 计算调用整合 Kernel 函数的线程块的数量。
    dim3 gridsize3;
    gridsize3.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize3.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 调用核函数，将各窗口重新整合成输出图像。
    _hisEqualSecKer<<<gridsize3, blocksize>>>(insubimgCud, outsubimgCud, 
                                              devcumucounter, devcumugray,  
                                              t0, c1, c2, weight);

    if (cudaGetLastError() != cudaSuccess) {
        // 核函数出错。
        cudaFree(his);
        cudaFree(norhis);
        cudaFree(devdata);
        return CUDA_ERROR;
    }

    // 释放申请空间，避免内存泄露
    cudaFree(devdata);
    cudaFree(his);
    cudaFree(norhis);

    return NO_ERROR;
}

