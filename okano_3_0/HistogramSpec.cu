// HistogramSpec.cu
// 实现直方图规定化算法

#include "HistogramSpec.h"
#include "Histogram.h"

#include <iostream>
using namespace std;

#include "ErrorCode.h"


// 宏：HISTOGRAM_LEVEL  
// 输入图像直方图的灰度级，默认为 256。
#ifndef HISTOGRAM_LEVEL
#define HISTOGRAM_LEVEL 256
#endif

// 宏：HISTSPEC_LEVEL  
// 规定化后输出图像的灰度级，默认为 256。
#ifndef HISTSPEC_LEVEL
#define HISTSPEC_LEVEL 256
#endif

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8  


// Kernel 函数: _calDiffMatrixKer（根据原始和目标累积直方图建立差值矩阵）
// 根据输入图像的原始累积直方图，以及目标累积直方图计算差值矩阵
// diffmatrix，并且初始化映射矩阵 maptable，将其各元素其初始值置为
// HISTSPEC_LEVEL.
static __global__ void          // Kernel 函数无返回值。
_calDiffMatrixKer(
        float *diffmatrix,      // 差值矩阵。
        float *cumhist,         // 原始累积直方图。
        float *cumhistequ,      // 目标累积直方图。
        unsigned int *maptable  // 映射矩阵。
);

// Kernel 函数: _findColumnMinKer（查找列最小值）
// 根据差值矩阵 diffmatrix，查找每一列的最小值，并将每一列出现最小
// 值的行号保存在数组 colmin 中。
static __global__ void        // Kernel 函数无返回值。
_findColumnMinKer(
        float *diffmatrix,    // 差值矩阵。
        unsigned int *colmin  // 列最小值矩阵。
);

// Kernel 函数: _findRowMinKer（查找行最小值）
// 根据差值矩阵 diffmatrix，查找每一行的最小值，并将每一行出现最小
// 值的行列号保存在数组 rowmin 中。
static __global__ void        // Kernel 函数无返回值。
_findRowMinKer(
        float *diffmatrix,    // 差值矩阵。
        unsigned int *rowmin  // 行最小值矩阵。
);

// Kernel 函数: _groupMappingLawKer（计算灰度级之间的映射矩阵）
// 根据组映射规则，通过行、列最小值矩阵，计算原始图像和目标图像灰度级
// 之间的映射关系。
static __global__ void          // Kernel 函数无返回值。
_groupMappingLawKer(
        unsigned int *rowmin,   // 行最小值矩阵。
        unsigned int *colmin,   // 列最小值矩阵。
        unsigned int *maptable  // 灰度级之间的映射矩阵。
);

// Kernel 函数: _maptableJudgeKer（整理灰度级之间的映射矩阵）
// 根据原始累积直方图 devcumhist，以及向后匹配原则，整理灰度级之间的
// 映射矩阵。
static __global__ void           // Kernel 函数无返回值。
_maptableJudgeKer(
        unsigned int *maptable,  // 灰度级之间的映射矩阵。 
        float *devcumhist        // 原始累积直方图。   
);

// Kernel 函数: _mapToOutimgKer（计算输出图像）
// 对于每个像素点，查找原始灰度级，并根据灰度级映射矩阵 maptable，得
// 到变化后灰度级，从而极端得到输出图像。
static __global__ void          // Kernel 函数无返回值。
_mapToOutimgKer(
        ImageCuda inimg,        // 输入图像。
        ImageCuda outimg,       // 输出图像。
        unsigned int *maptable  // 灰度级之间的映射矩阵。
);

// Kernel 函数: _calDiffMatrixKer（根据原始和目标累积直方图建立差值矩阵）
static __global__ void _calDiffMatrixKer(
        float *diffmatrix, float *cumhist, float *cumhistequ,
        unsigned int *maptable)
{
    // 申请大小为目标直方图灰度级 HISTSPEC_LEVEL 的共享内存。
    __shared__ float shared_cumhistequ[HISTSPEC_LEVEL];
    
    // 获取当前线程的块号。
    int blocktid = blockIdx.x;
    // 获取当前线程的线程号。
    int threadtid = threadIdx.x;
    // 计算差值矩阵中对应的输出点的位置。
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // 申请局部变量，用于临时存储当前线程计算得到的差值。
    float temp = 0.0;     
    
    // 初始化灰度级匹配矩阵。
    maptable[blocktid] = HISTSPEC_LEVEL;
    
    // 将目标累积直方图对应的存储在共享内存中，方便同一块内的线程共享，从而
    // 提高读取速度。
    shared_cumhistequ[threadtid] = cumhistequ[threadtid];
    // 计算差值。
    temp = shared_cumhistequ[threadtid] - cumhist[blocktid];
    // 进行块内同步。
    __syncthreads();
   
    // 将计算所得的差值写入差值矩阵相应位置上，若临时变量 temp 大于或者等
    // 于 0，直接赋值；若 temp 小于 0，则取其相反数之后再赋值。
    *(diffmatrix + index) = (temp >= 0.0 ? temp : -temp);
}

// Kernel 函数: _findColumnMinKer（查找列最小值）
static __global__ void _findColumnMinKer(
        float *diffmatrix, unsigned int *colmin)
{
    // 获取当前线程的块号，即对应差值矩阵中当前的列号。
    int blocktid = blockIdx.x;
    // 获取当前线程的线程号，即对应差值矩阵中当前的行号。
    int threadtid = threadIdx.x;
    // 计算当前线程在差值矩阵中的偏移。
    int tid = threadIdx.x * gridDim.x + blockIdx.x;
    int k;
    float tempfloat;
    unsigned int tempunint; 
    
    // 申请一个大小等于原始直方图灰度级的 float 型共享内存，用于存储每
    // 一列中待比较的差值。
    __shared__ float shared[HISTOGRAM_LEVEL];
    // 申请一个大小等于原始直方图灰度级的 unsigned int 型共享内存，用于
    // 存储待比较差值对应的行号。 
    __shared__ unsigned int index[HISTOGRAM_LEVEL];
    
    // 将当前线程对应的差值矩阵中的差值以及其对应的索引（即行号）保存
    // 在该块的共享内存中。
    shared[threadtid] = *(diffmatrix + tid);
    index[threadtid] = threadtid;
    // 块内同步，为了保证一个块内的所有线程都已经完成了上述操作，即存
    // 储该列的差值以及索引到共享内存中。
    __syncthreads();
    
    // 使用双调排序的思想，找到该列的最小值。
    for (k = 1; k < HISTOGRAM_LEVEL; k = k << 1) {
        // 对待排序的元素进行分组，每次都将差值较小的元素交换到数组中
        // 较前的位置，然后改变分组大小，进而在比较上一次得到的较小值
        // 并做相应的交换，以此类推，最终数组中第 0 号元素存放的是该列
        // 的最小值。
        if (((threadtid % (k << 1)) == 0) &&
            shared[threadtid] > shared[threadtid + k] ) {
            // 两个差值进行交换。
            tempfloat = shared[threadtid];
            shared[threadtid] = shared[threadtid + k];
            shared[threadtid + k] = tempfloat;
            
            // 交换相对应的索引 index 值。
            tempunint = index[threadtid];  
            index[threadtid] = index[threadtid + k];
            index[threadtid + k] = tempunint;
        } 
        // 块内同步
        __syncthreads();
    }
    
    // 将当前列最小值出现的行号保存在数组 colmin 中。
    colmin[blocktid] = index[0];
}

// Kernel 函数: _findRowMinKer（查找行最小值）
static __global__ void _findRowMinKer(
        float *diffmatrix, unsigned int *rowmin)
{
    // 获取当前线程的块号。
    int blocktid = blockIdx.x;
    // 获取当前线程的线程号。
    int threadtid = threadIdx.x;
    // 计算当前线程在差值矩阵中的偏移。
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k;
    float tempfloat;
    unsigned int tempunint; 
    
    // 申请一个大小等于原始直方图灰度级的 float 型共享内存，用于存储每
    // 一行中待比较的差值。
    __shared__ float shared[HISTSPEC_LEVEL];
    // 申请一个大小等于原始直方图灰度级的 unsigned int 型共享内存，用于
    // 存储待比较差值对应的列号。 
    __shared__ unsigned int index[HISTSPEC_LEVEL];
    
    // 将当前线程对应的差值矩阵中的差值以及其对应的索引（即列号）保存
    // 在该块的共享内存中。 
    shared[threadtid] = *(diffmatrix + tid);
    index[threadtid] = threadtid;
    // 块内同步，为了保证一个块内的所有线程都已经完成了上述操作，即存
    // 储该行的差值以及索引到共享内存中。
    __syncthreads();
    
    // 使用双调排序的思想，找到该行的最小值。
    for (k = 1; k < HISTSPEC_LEVEL; k = k << 1) {
        // 对待排序的元素进行分组，每次都将差值较小的元素交换到数组中
        // 较前的位置，然后改变分组大小，进而在比较上一次得到的较小值
        // 并做相应的交换，以此类推，最终数组中第 0 号元素存放的是该行
        // 的最小值。
        if (((threadtid % (k << 1)) == 0) &&
            shared[threadtid] > shared[threadtid + k] ) {
            // 两个差值进行交换。
            tempfloat = shared[threadtid];
            shared[threadtid] = shared[threadtid + k];
            shared[threadtid + k] = tempfloat;
            
            // 交换相对应的索引index值。
            tempunint = index[threadtid];  
            index[threadtid] = index[threadtid + k];
            index[threadtid + k] = tempunint;
        } 
        // 块内同步。
        __syncthreads();
    }
    
    // 将当前行最小值出现的列号保存在数组 rowmin 中。
    rowmin[blocktid] = index[0];
}

// Kernel 函数: _groupMappingLawKer（计算灰度级之间的映射矩阵）
static __global__ void _groupMappingLawKer(
        unsigned int *rowmin, unsigned int *colmin, 
        unsigned int *maptable)
{
    // 获取当前的线程号。
    int tid = threadIdx.x;
    
    // 通过行列最小值的关系，计算 group mapping law（GML）映射关系。
    // 可得到初始的不完整的映射表。
    maptable[colmin[tid]] = rowmin[colmin[tid]];   
}

// Kernel 函数: _maptableJudgeKer（整理灰度级之间的映射矩阵）
static __global__ void _maptableJudgeKer(
        unsigned int *maptable, float *devcumhist)
{
    // 获取当前的线程号。
    int tid = threadIdx.x;
    int temp, i = tid;
  
    // 通过向高灰度匹配的原则，整理灰度级映射关系表。
    while (devcumhist[tid] >= 0) 
    {
        // 暂存映射表中的值。
        temp = maptable[i];
        
        // 判断如果当前映射表中的值是无效值，则向后进行匹配，直到
        // 符合灰度级要求。
        if (temp == HISTSPEC_LEVEL) {
            i++;
        } else {
            // 更新灰度级映射表中映射关系。
            maptable[tid] = temp;
            break;
        }       
    }
}

// Kernel 函数: _mapToOutimgKer（计算输出图像）
static __global__ void _mapToOutimgKer(
        ImageCuda inimg, ImageCuda outimg, unsigned int *maptable)
{
    // 计算想成对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻
    // 4 行上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;    
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp;
    intemp = inimg.imgMeta.imgData[inidx];

    // 一个线程处理四个像素点.
    // 通过灰度级匹配矩阵，得到输入图像当前点所对应的变换后的灰度值，并赋值
    // 给输出图像的对应位置。
    // 线程中处理的第一个点。
    outimg.imgMeta.imgData[outidx] = maptable[intemp];

    // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += outimg.pitchBytes;
        intemp = inimg.imgMeta.imgData[inidx];

        // 通过灰度级匹配矩阵，得到输入图像当前点所对应的变换后的灰度值，并赋值
        // 给输出图像的对应位置。
        outimg.imgMeta.imgData[outidx] = maptable[intemp];
    }
}

// Host 成员方法：HistogramEquilibrium（直方图均衡化）
__host__ int HistogramSpec::HistogramEquilibrium(Image *inimg, Image *outimg)
{
    // 检查输入和输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
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
    ImageCuda outsubimgCud;
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

    // 调用类 Histogram 中的方法，计算输入图像的直方图。	
    Histogram hist;
    unsigned int histogram[HISTOGRAM_LEVEL] = {0};
    hist.histogram(inimg, histogram, true);

    // 计算原始直方图的累积直方图。
    float cumsum = 0;
    float cumhist[HISTOGRAM_LEVEL] = {0};
    for (int i = 0; i < HISTOGRAM_LEVEL; i++) {
         cumsum += histogram[i];
         cumhist[i] = (float)cumsum / (inimg->height * inimg->width);
    }

    // 计算均衡化后的累积直方图，均衡化之后每个灰度值的概率相等。
    float cumhistequ[HISTSPEC_LEVEL]={0};
    for (int j = 0; j < HISTSPEC_LEVEL; j++) {
         cumhistequ[j] = (float)(j+1) / HISTSPEC_LEVEL;
    }	
    	
    // 在 Device 上分配临时空间。一次申请所有空间，然后通过偏移索引各个数组。
    // 包括原始累积直方图 devcumhist，均衡化后累积直方图 devcumhistequ，行最
    // 小值矩阵 devrowmin，列最小值矩阵 colmin，映射矩阵 devmaptable，差值矩
    // 阵 devdiffmatrix。
    cudaError_t cudaerrcode;
    float *alldevicepointer;
    float *devcumhist, *devcumhistequ, *devdiffmatrix;
    unsigned int *devcolmin, *devrowmin, *devmaptable;
    cudaerrcode = cudaMalloc(
            (void **)&alldevicepointer, 
            (3 * HISTOGRAM_LEVEL + 2 * HISTOGRAM_LEVEL + 
            HISTOGRAM_LEVEL * HISTOGRAM_LEVEL) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        return cudaerrcode;
    }
    
    // 初始化所有 Device 上的内存空间。
    cudaerrcode = cudaMemset(
            alldevicepointer, 0, 
            (3 * HISTOGRAM_LEVEL + 2 * HISTOGRAM_LEVEL +
            HISTOGRAM_LEVEL * HISTOGRAM_LEVEL) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 通过偏移读取 devcumhist 内存空间。
    devcumhist = alldevicepointer;
    // 将 Host 端计算的累积直方图 cumhist 拷贝到 Device 端。
    cudaerrcode = cudaMemcpy(devcumhist, cumhist, 
                             HISTOGRAM_LEVEL * sizeof (float),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 通过偏移读取 devcumhistequ 内存空间。
    devcumhistequ = alldevicepointer + HISTOGRAM_LEVEL;
    // 将 Host 端计算的累积直方图 cumhistequ 拷贝到 Device 端。
    cudaerrcode = cudaMemcpy(devcumhistequ, cumhistequ, 
                             HISTSPEC_LEVEL * sizeof (float),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 通过偏移读取差值矩阵 devdiffmatrix 内存空间。
    devdiffmatrix = alldevicepointer + 3 * HISTOGRAM_LEVEL + 
                    2 * HISTOGRAM_LEVEL;
    
    // 通过偏移读取映射矩阵 devmaptable 内存空间，并将转换指针类型。
    devmaptable = (unsigned int *)(alldevicepointer +
                  2 * (HISTOGRAM_LEVEL + HISTSPEC_LEVEL));

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    gridsize.x = HISTOGRAM_LEVEL;
    gridsize.y = 1;
    blocksize.x = HISTSPEC_LEVEL;
    blocksize.y = 1;

    // 调用核函数，计算差值矩阵。
    _calDiffMatrixKer<<<gridsize, blocksize>>>(
            devdiffmatrix, devcumhist, devcumhistequ, devmaptable); 
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 通过偏移读取映射矩阵 devrowmin 内存空间，并将转换指针类型。
    devrowmin = (unsigned int *)(alldevicepointer +
                HISTOGRAM_LEVEL + HISTSPEC_LEVEL);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    gridsize.x = HISTOGRAM_LEVEL;
    gridsize.y = 1;
    blocksize.x = HISTSPEC_LEVEL;
    blocksize.y = 1;

    // 调用核函数，计算行最小值。
    _findRowMinKer<<<gridsize, blocksize>>>(
            devdiffmatrix, (unsigned int *)devrowmin);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 通过偏移读取映射矩阵 devcolmin 内存空间，并将转换指针类型。
    devcolmin = (unsigned int *)(alldevicepointer + 2 * HISTOGRAM_LEVEL
                                 + HISTSPEC_LEVEL);
 
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    gridsize.x = HISTSPEC_LEVEL;
    gridsize.y = 1;
    blocksize.x = HISTOGRAM_LEVEL;
    blocksize.y = 1;

    // 调用核函数，计算列最小值。
    _findColumnMinKer<<<gridsize, blocksize>>>(
            devdiffmatrix, (unsigned int *)devcolmin);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
 
    // 计算灰度级之间映射关系。
    _groupMappingLawKer<<<1, HISTSPEC_LEVEL>>>(
            (unsigned int *)devrowmin, (unsigned int *)devcolmin, devmaptable);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 整理映射关系
    _maptableJudgeKer<<<1, HISTOGRAM_LEVEL>>>(
            (unsigned int *)devmaptable, devcumhist);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);  
        		
    // 通过映射矩阵，得到输出图像。
    _mapToOutimgKer<<<gridsize, blocksize>>>(
            insubimgCud, outsubimgCud,(unsigned int *)devmaptable);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
    
    // 释放 Device 上的临时空间 alldevicedata。
    cudaFree(alldevicepointer);
    return NO_ERROR;
}

// Host 成员方法：HistogramSpecByImage（根据参考图像进行规定化）
__host__ int HistogramSpec::HistogramSpecByImage(Image *inimg, Image *outimg)
{
    // 检查输入图像,参考图像和输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL || this->refimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
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
    ImageCuda outsubimgCud;
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
 
    // 调用类 Histogram 中的方法，计算输入图像的直方图。	
    Histogram hist;
    unsigned int histogram[HISTOGRAM_LEVEL] = {0};
    hist.histogram(inimg, histogram, true);
 
    // 调用类 Histogram 中的方法，计算参考图像的直方图。
    unsigned int histspec[HISTSPEC_LEVEL] = {0};
    hist.histogram(this->refimg, histspec, true);

    // 计算原始直方图的累积直方图。
    unsigned int cumsum = 0;
    float cumhist[HISTOGRAM_LEVEL] = {0.0};
    for (int i = 0; i < HISTOGRAM_LEVEL; i++) {
         cumsum += histogram[i];
         cumhist[i] = (float)cumsum / (inimg->height * inimg->width);
    }
 
    // 计算参考图像的累积直方图，均衡化之后每个灰度值的概率相等。
    float cumhistspec[HISTSPEC_LEVEL]={0.0};
    cumsum = 0;
    for (int i = 0; i < HISTSPEC_LEVEL; i++) {
         cumsum += histspec[i];
         cumhistspec[i] = (float)cumsum / (this->refimg->width 
                                           * this->refimg->height);
    }
    	
    // 在 Device 上分配临时空间。一次申请所有空间，然后通过偏移索引各个数组。
    // 包括原始累积直方图 devcumhist，均衡化后累积直方图 devcumhistequ，行最
    // 小值矩阵 devrowmin，列最小值矩阵 colmin，映射矩阵 devmaptable，差值矩
    // 阵 devdiffmatrix。
    cudaError_t cudaerrcode;
    float *alldevicepointer;
    float *devcumhist, *devcumhistspec, *devdiffmatrix;
    unsigned int *devcolmin, *devrowmin, *devmaptable;
    cudaerrcode = cudaMalloc(
            (void **)&alldevicepointer, 
            (3 * HISTOGRAM_LEVEL + 2 * HISTOGRAM_LEVEL + 
            HISTOGRAM_LEVEL * HISTOGRAM_LEVEL) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }
    
    // 初始化所有 Device 上的内存空间。
    cudaerrcode = cudaMemset(
            alldevicepointer, 0, 
            (3 * HISTOGRAM_LEVEL + 2 * HISTOGRAM_LEVEL +
            HISTOGRAM_LEVEL * HISTOGRAM_LEVEL) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 通过偏移读取 devcumhist 内存空间。
    devcumhist = alldevicepointer;
    // 将 Host 端计算的累积直方图 cumhist 拷贝到 Device 端。
    cudaerrcode = cudaMemcpy(devcumhist, cumhist, 
                             HISTOGRAM_LEVEL * sizeof (float),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 通过偏移读取 devcumhistequ 内存空间。
    devcumhistspec = alldevicepointer + HISTOGRAM_LEVEL;
    // 将 Host 端计算的累积直方图 cumhistspec 拷贝到 Device 端。
    cudaerrcode = cudaMemcpy(devcumhistspec, cumhistspec, 
                             HISTSPEC_LEVEL * sizeof (float),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 通过偏移读取差值矩阵 devdiffmatrix 内存空间。
    devdiffmatrix = alldevicepointer + 3 * HISTOGRAM_LEVEL + 
                    2 * HISTOGRAM_LEVEL;
    
    // 通过偏移读取映射矩阵 devmaptable 内存空间，并将转换指针类型。
    devmaptable = (unsigned int *)(alldevicepointer +
                  2 * (HISTOGRAM_LEVEL + HISTSPEC_LEVEL));
 
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    gridsize.x = HISTOGRAM_LEVEL;
    gridsize.y = 1;
    blocksize.x = HISTSPEC_LEVEL;
    blocksize.y = 1;
 
    // 调用核函数，计算差值矩阵。
    _calDiffMatrixKer<<<gridsize, blocksize>>>(
            devdiffmatrix, devcumhist, devcumhistspec, devmaptable); 
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
 
    // 通过偏移读取映射矩阵 devrowmin 内存空间，并将转换指针类型。
    devrowmin = (unsigned int *)(alldevicepointer +
                HISTOGRAM_LEVEL + HISTSPEC_LEVEL);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    gridsize.x = HISTOGRAM_LEVEL;
    gridsize.y = 1;
    blocksize.x = HISTSPEC_LEVEL;
    blocksize.y = 1;
 
    // 调用核函数，计算行最小值。
    _findRowMinKer<<<gridsize, blocksize>>>(
            devdiffmatrix, (unsigned int *)devrowmin);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 通过偏移读取映射矩阵 devcolmin 内存空间，并将转换指针类型。
    devcolmin = (unsigned int *)(alldevicepointer + 2 * HISTOGRAM_LEVEL
                                 + HISTSPEC_LEVEL);
 
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    gridsize.x = HISTSPEC_LEVEL;
    gridsize.y = 1;
    blocksize.x = HISTOGRAM_LEVEL;
    blocksize.y = 1;
 
    // 调用核函数，计算列最小值。
    _findColumnMinKer<<<gridsize, blocksize>>>(
            devdiffmatrix, (unsigned int *)devcolmin);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
 
    // 计算灰度级之间映射关系。
    _groupMappingLawKer<<<1, HISTSPEC_LEVEL>>>(
            (unsigned int *)devrowmin, (unsigned int *)devcolmin, devmaptable);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 整理映射关系
    _maptableJudgeKer<<<1, HISTOGRAM_LEVEL>>>(
            (unsigned int *)devmaptable, devcumhist);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
 
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);  
        		
    // 通过映射矩阵，得到输出图像。
    _mapToOutimgKer<<<gridsize, blocksize>>>(
            insubimgCud, outsubimgCud,(unsigned int *)devmaptable);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
    
    // 释放 Device 上的临时空间 alldevicedata。
    cudaFree(alldevicepointer);
    return NO_ERROR;
}

// Host 成员方法：HistogramSpecByHisto（根据参考直方图进行规定化）
__host__ int HistogramSpec::HistogramSpecByHisto(Image *inimg, Image *outimg)
{
    // 检查输入和输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL || refHisto == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
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
    ImageCuda outsubimgCud;
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
 
    // 调用类 Histogram 中的方法，计算输入图像的直方图。	
    Histogram hist;
    unsigned int histogram[HISTOGRAM_LEVEL] = {0};
    hist.histogram(inimg, histogram, true);

    // 计算原始直方图的累积直方图。
    unsigned int  cumsum = 0;
    float cumhist[HISTOGRAM_LEVEL] = {0};
    for (int i = 0; i < HISTOGRAM_LEVEL; i++) {
         cumsum += histogram[i];
         cumhist[i] = (float)cumsum / (inimg->height * inimg->width);
    }
    
    // 计算参考直方图的累积直方图。
    float cumhistspec[HISTSPEC_LEVEL] = {0};
    cumhistspec[0] = this->refHisto[0];
    for (int i = 1; i < HISTSPEC_LEVEL; i++) {
         cumhistspec[i] = cumhistspec[i - 1] + this->refHisto[i];
    }
    	
    // 在 Device 上分配临时空间。一次申请所有空间，然后通过偏移索引各个数组。
    // 包括原始累积直方图 devcumhist，均衡化后累积直方图 devcumhistequ，行最
    // 小值矩阵 devrowmin，列最小值矩阵 colmin，映射矩阵 devmaptable，差值矩
    // 阵 devdiffmatrix。
    cudaError_t cudaerrcode;
    float *alldevicepointer;
    float *devcumhist, *devcumhistspec, *devdiffmatrix;
    unsigned int *devcolmin, *devrowmin, *devmaptable;
    cudaerrcode = cudaMalloc(
            (void **)&alldevicepointer, 
            (3 * HISTOGRAM_LEVEL + 2 * HISTOGRAM_LEVEL + 
            HISTOGRAM_LEVEL * HISTOGRAM_LEVEL) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }
    
    // 初始化所有 Device 上的内存空间。
    cudaerrcode = cudaMemset(
            alldevicepointer, 0, 
            (3 * HISTOGRAM_LEVEL + 2 * HISTOGRAM_LEVEL +
            HISTOGRAM_LEVEL * HISTOGRAM_LEVEL) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 通过偏移读取 devcumhist 内存空间。
    devcumhist = alldevicepointer;
    // 将 Host 端计算的累积直方图 cumhist 拷贝到 Device 端。
    cudaerrcode = cudaMemcpy(devcumhist, cumhist, 
                             HISTOGRAM_LEVEL * sizeof (float),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 通过偏移读取 devcumhistequ 内存空间。
    devcumhistspec = alldevicepointer + HISTOGRAM_LEVEL;
    // 将 Host 端计算的参考直方图拷贝到 Device 端。
    cudaerrcode = cudaMemcpy(devcumhistspec, cumhistspec, 
                             HISTSPEC_LEVEL * sizeof (float),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 通过偏移读取差值矩阵 devdiffmatrix 内存空间。
    devdiffmatrix = alldevicepointer + 3 * HISTOGRAM_LEVEL + 
                    2 * HISTOGRAM_LEVEL;
    
    // 通过偏移读取映射矩阵 devmaptable 内存空间，并将转换指针类型。
    devmaptable = (unsigned int *)(alldevicepointer +
                  2 * (HISTOGRAM_LEVEL + HISTSPEC_LEVEL));
 
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    gridsize.x = HISTOGRAM_LEVEL;
    gridsize.y = 1;
    blocksize.x = HISTSPEC_LEVEL;
    blocksize.y = 1;

    // 调用核函数，计算差值矩阵。
    _calDiffMatrixKer<<<gridsize, blocksize>>>(
            devdiffmatrix, devcumhist, devcumhistspec, devmaptable); 
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
    
    // 通过偏移读取映射矩阵 devrowmin 内存空间，并将转换指针类型。
    devrowmin = (unsigned int *)(alldevicepointer +
                HISTOGRAM_LEVEL + HISTSPEC_LEVEL);
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    gridsize.x = HISTOGRAM_LEVEL;
    gridsize.y = 1;
    blocksize.x = HISTSPEC_LEVEL;
    blocksize.y = 1;
 
    // 调用核函数，计算行最小值。
    _findRowMinKer<<<gridsize, blocksize>>>(
            devdiffmatrix, (unsigned int *)devrowmin);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
    
    // 通过偏移读取映射矩阵 devcolmin 内存空间，并将转换指针类型。
    devcolmin = (unsigned int *)(alldevicepointer + 2 * HISTOGRAM_LEVEL
                                 + HISTSPEC_LEVEL);
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    gridsize.x = HISTSPEC_LEVEL;
    gridsize.y = 1;
    blocksize.x = HISTOGRAM_LEVEL;
    blocksize.y = 1;
 
    // 调用核函数，计算列最小值。
    _findColumnMinKer<<<gridsize, blocksize>>>(
            devdiffmatrix, (unsigned int *)devcolmin);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
    
    // 计算灰度级之间映射关系。
    _groupMappingLawKer<<<1, HISTSPEC_LEVEL>>>(
            (unsigned int *)devrowmin, (unsigned int *)devcolmin, 
            devmaptable);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
    
    // 整理映射关系
    _maptableJudgeKer<<<1, HISTOGRAM_LEVEL>>>(
            (unsigned int *)devmaptable, devcumhist);			
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);  
        		
    // 通过映射矩阵，得到输出图像。
    _mapToOutimgKer<<<gridsize, blocksize>>>(
            insubimgCud, outsubimgCud,(unsigned int *)devmaptable);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
        
    // 释放 Device 上的临时空间 alldevicedata。
    cudaFree(alldevicepointer);
    return NO_ERROR;
}

