// SelectShape.cu
// 实现形状选择算法

#include "SelectShape.h"

#include <iostream>
using namespace std;


// 宏：MAX_LABEL  
// 特征值数组中标记的最大值，默认为 256。
#ifndef MAX_LABEL
#define MAX_LABEL 256
#endif

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8  


// Kernel 函数: _selectShapeByIndexKer（根据索引值进行形状区域拷贝）
// 查询输入图像中像素值（标记值）等于 label 的像素，拷贝至输出图像中；
// 否则将输出图像中该位置清0。
static __global__ void     // Kernel 函数无返回值。
_selectShapeByIndexKer(
        ImageCuda inimg,   // 输入图像。
        ImageCuda outimg,  // 输出图像。
        int label          // 待查询的标记值。
);

// Kernel 函数: _setLabelByValueKer（根据特征值进行形状区域拷贝）
// 查询 rank 数组中特征值等于 value 的项，将其对应 flag 标记设为 1；
// 否则为0。
static __global__ void            // Kernel 函数无返回值。
_setLabelByValueKer(
        int *rank,                // 特征值数组。
        int value,                // 待查询的特征值。
        unsigned char *flaglabel  // 标记数组。
);

// Kernel 函数: _selectShapeByValueKer（根据特征值进行形状区域拷贝）
// 查询输入图像中区域的特征值等于 value，则将该区域拷贝至输出图像中；
// 否则将输出图像中该位置清 0。
static __global__ void            // Kernel 函数无返回值。
_selectShapeByValueKer(
        ImageCuda inimg,          // 输入图像。
        ImageCuda outimg,         // 输出图像。
        unsigned char *flaglabel  // 标记数组。
);

// Kernel 函数: _shapeClearByLabelKer（清除区域标记）
// 如果 flag 等于 0， 则设置其对应区域的像素值为 0；否则不做任何改变。
static __global__ void            // Kernel 函数无返回值。
_shapeClearByLabelKer(
        ImageCuda inimg,          // 输入图像。
        unsigned char *flaglabel  // 标记数组。
);

// Kernel 函数: _setLabelByMinMaxKer（根据最大最小范围进行形状区域拷贝）
// 查询 rank 数组中特征值在最大最小范围内的项，将其对应 flag 标记设为 1；
// 否则为 0。
static __global__ void            // Kernel 函数无返回值。
_setLabelByMinMaxKer(
        int *rank,                // 特征值数组。
        int minvalue,             // 最小特征值。
        int maxvalue,             // 最大特征值。
        unsigned char *flaglabel  // 标记数组。
);

// Kernel 函数: _selectShapeByIndexKer（根据索引值进行形状区域拷贝）
static __global__ void _selectShapeByIndexKer(
        ImageCuda inimg, ImageCuda outimg, int label)
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
    // 如果输入图像的该像素值等于 label, 则将其拷贝到输出图像中；
    // 否则将输出图像中该位置清 0。
    // 线程中处理的第一个点。
    outimg.imgMeta.imgData[outidx] = (intemp == label ? intemp : 0);

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

        // 如果输入图像的该像素值等于 label, 则将其拷贝到输出图像中；
        // 否则将输出图像中该位置清 0。
        outimg.imgMeta.imgData[outidx] = (intemp == label ? intemp : 0);
    }
}

// Host 成员方法：selectShapeByIndex（根据标记索引形状）
__host__ int SelectShape::selectShapeByIndex(Image *inimg, Image *outimg)
{
    // 检查图像是否为 NULL。
    if (inimg == NULL)
        return NULL_POINTER;

    // 检查 rank 数组是否为空。
    if (this->rank == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
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

    // 通过输入的索引值找到相对应的形状区域标记值。
    int label = this->rank[2 * this->index + 1];

    // 如果输入图像不等于输出图像，并且输出图像不为空。
    if (inimg != outimg && outimg != NULL) {
        // 将输出图像拷贝到 Device 内存中。
        errcode = ImageBasicOp::copyToCurrentDevice(outimg);
        if (errcode != NO_ERROR) {
            // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入
            // 图像的 ROI 子图像尺寸相同的图像。
            errcode = ImageBasicOp::makeAtCurrentDevice(
                    outimg, inimg->roiX2 - inimg->roiX1, 
                    inimg->roiY2 - inimg->roiY1);
            // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
            if (errcode != NO_ERROR)
                return errcode;
        }

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

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 blocksize, gridsize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / 
                      blocksize.x;
        gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                     (blocksize.y * 4);

        // 调用核函数，根据标记 label 进行形状区域拷贝。
        _selectShapeByIndexKer<<<gridsize, blocksize>>>(
                insubimgCud, outsubimgCud, label);

       // 若调用 CUDA 出错返回错误代码
       if (cudaGetLastError() != cudaSuccess)
           return CUDA_ERROR;
    // 如果输入图像等于输出图像，或者输出图像为空。
    } else {
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 blocksize, gridsize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / 
                      blocksize.x;
        gridsize.y = (insubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                     (blocksize.y * 4);

        // 调用核函数根据标记 label 进行形状区域拷贝。
        _selectShapeByIndexKer<<<gridsize, blocksize>>>(
                insubimgCud, insubimgCud, label);
                
       // 若调用 CUDA 出错返回错误代码
       if (cudaGetLastError() != cudaSuccess)
           return CUDA_ERROR;
    }

    return NO_ERROR;
}

// Kernel 函数: _setLabelByValueKer（根据特征值进行形状区域拷贝）
static __global__ void _setLabelByValueKer(
        int *rank, int value, unsigned char *flaglabel)
{
    // 获取线程号。
    int tid = threadIdx.x;

    // 如果特征值等于 value，将其对应 flag 标记设为1。
    if (rank[2 * tid] == value)
        flaglabel[rank[2 * tid + 1]] = 1;
}

// Kernel 函数: _selectShapeByValueKer（根据特征值进行形状区域拷贝）
static __global__ void _selectShapeByValueKer(
        ImageCuda inimg, ImageCuda outimg, unsigned char *flaglabel)
{
    // 计算想成对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻
    // 4 行上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 将 flag 数组保存到 share 内存中，加快读取速度。 
    __shared__ unsigned char shared[MAX_LABEL];
    // 将 flag 数组拷贝到 share 内存中。
    shared[tid & ((1 << 8)-1)] = flaglabel[tid & ((1 << 8)-1)];
    __syncthreads();

    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;    
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp;
    intemp = inimg.imgMeta.imgData[inidx];

    // 一个线程处理四个像素点.
    // 如果输入图像的该像素值对应的 flag 等于1, 则将其拷贝到输出图像中；
    // 否则将输出图像的该像素值设为 0。
    // 线程中处理的第一个点。
    outimg.imgMeta.imgData[outidx] = (shared[intemp] == 1 ? intemp : 0);

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

        // 如果输入图像的该像素值对应的 flag 等于1, 则将其拷贝到输出图像中；
        // 否则将输出图像的该像素值设为0。
        outimg.imgMeta.imgData[outidx] = (shared[intemp] == 1 ? intemp : 0);
    }
}

// Kernel 函数: _shapeClearByLabelKer（清除区域标记）
static __global__ void _shapeClearByLabelKer(
        ImageCuda inimg, unsigned char *flaglabel)
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

    // 一个线程处理四个像素点.
    // 如果输入图像的该像素值对应的 flag 等于 0, 则将像素值设为 0；
    // 否则保持原来的像素值。
    // 线程中处理的第一个点。
    if (flaglabel[inimg.imgMeta.imgData[inidx]] == 0)
        inimg.imgMeta.imgData[inidx] = 0;

    // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++r >= inimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;

        // 如果输入图像的该像素值对应的 flag 等于 0, 则将像素值设为 0；
        // 否则保持原来的像素值。
        if (flaglabel[inimg.imgMeta.imgData[inidx]] == 0)
            inimg.imgMeta.imgData[inidx] = 0;
    }
}

// Host 成员方法：selectShapeByValue（根据特征值查找形状）
__host__ int SelectShape::selectShapeByValue(Image *inimg, Image *outimg)
{
    // 检查图像是否为 NULL。
    if (inimg == NULL)
        return NULL_POINTER;

    // 检查 rank 数组是否为空。
    if (this->rank == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
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

    // 在 Device 上分配临时空间。一次申请所有空间，然后通过偏移索引各个数组。
    cudaError_t cudaerrcode;
    int *alldevicedata;
    unsigned char *devflaglabel;
    int *devRank;
    cudaerrcode = cudaMalloc((void** )&alldevicedata,
                             (2 * this->pairsnum + MAX_LABEL) * sizeof (int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 初始化 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldevicedata, 0,
                             (2 * this->pairsnum + MAX_LABEL)
                             * sizeof (int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 通过偏移读取 devRank 内存空间。
    devRank = alldevicedata;
    // 将 Host 上的 rank 拷贝到 Device 上的 devRank 中。
    cudaerrcode = cudaMemcpy(devRank, this->rank,
                             2 * this->pairsnum * sizeof (int),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 通过偏移读取 devflaglabel 内存空间。
    devflaglabel = (unsigned char*)(alldevicedata + 2 * this->pairsnum);
    
    // 调用核函数，在 devRank数组中查询 value 值，并获取其标记值。
    _setLabelByValueKer<<<1, this->pairsnum>>>(
            devRank, this->value, devflaglabel);
       
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicedata);
        return CUDA_ERROR;
    }
           
    // 如果输入图像不等于输出图像，并且输出图像不为空。
    if (inimg != outimg && outimg != NULL) {
        // 将输出图像拷贝到 Device 内存中。
        errcode = ImageBasicOp::copyToCurrentDevice(outimg);
        if (errcode != NO_ERROR) {
            // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入
            // 图像的 ROI 子图像尺寸相同的图像。
            errcode = ImageBasicOp::makeAtCurrentDevice(
                    outimg, inimg->roiX2 - inimg->roiX1, 
                    inimg->roiY2 - inimg->roiY1);
            // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
            if (errcode != NO_ERROR)
                return errcode;
        }

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

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 blocksize, gridsize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / 
                      blocksize.x;
        gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                     (blocksize.y * 4);

        // 调用核函数，根据特征值 value 进行形状区域拷贝。
        _selectShapeByValueKer<<<gridsize, blocksize>>>(
                insubimgCud, outsubimgCud, devflaglabel); 
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;
        }           
        // 如果输入图像等于输出图像，或者输出图像为空。
    } else {
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 blocksize, gridsize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / 
                      blocksize.x;
        gridsize.y = (insubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                     (blocksize.y * 4);

        // 调用核函数，根据特征值 value 进行形状区域拷贝。
        _shapeClearByLabelKer<<<gridsize, blocksize>>>(
                insubimgCud, devflaglabel);
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;      
        }               
    }
    // 释放 Device 上的临时空间 alldevicedata。
    cudaFree(alldevicedata);
    return NO_ERROR;
}

// Kernel 函数: _setLabelByMinMaxKer（根据最大最小范围进行形状区域拷贝）
static __global__ void _setLabelByMinMaxKer(
        int *rank, int minvalue, int maxvalue,
        unsigned char *flaglabel)
{
    // 获取线程号。
    int tid = threadIdx.x;

    // 如果特征值在最小最大范围内，将其对应 flag 标记设为 1。
    if (rank[2 * tid] >= minvalue && rank[2 * tid] <= maxvalue)
        flaglabel[rank[2 * tid + 1]] = 1;
}

// Host 成员方法：selectShapeByMinMax（根据特征值最大最小范围查找形状）
__host__ int SelectShape::selectShapeByMinMax(Image *inimg, Image *outimg)
{
    // 检查图像是否为 NULL。
    if (inimg == NULL)
        return NULL_POINTER;

    // 检查 rank 数组是否为空。
    if (this->rank == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
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

    // 在 Device 上分配临时空间。一次申请所有空间，然后通过偏移索引各个数组。
    cudaError_t cudaerrcode;
    int *alldevicedata;
    unsigned char *devflaglabel;
    int *devRank;
    cudaerrcode = cudaMalloc((void** )&alldevicedata,
                             (2 * this->pairsnum + MAX_LABEL)
                             * sizeof (int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 初始化 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldevicedata, 0,
                             (2 * this->pairsnum + MAX_LABEL)
                             * sizeof (int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 通过偏移读取 devRank 内存空间。
    devRank = alldevicedata;
    // 将 Host 上的 rank 拷贝到 Device 上的 devRank 中。
    cudaerrcode = cudaMemcpy(devRank, this->rank,
                             2 * this->pairsnum * sizeof (int),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 通过偏移读取 devflaglabel 内存空间。
    devflaglabel = (unsigned char*)(alldevicedata + 2 * this->pairsnum);
    
    // 调用核函数，在 devRank数组中查询 在最大最小范围内的 value 值，
    // 并获取其标记值。
    _setLabelByMinMaxKer<<<1, this->pairsnum>>>(
            devRank, this->minvalue, this->maxvalue, devflaglabel);
            
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicedata);
        return CUDA_ERROR;   
    }
            
    // 如果输入图像不等于输出图像，并且输出图像不为空。
    if (inimg != outimg && outimg != NULL) {
        errcode = ImageBasicOp::copyToCurrentDevice(outimg);
        if (errcode != NO_ERROR) {
            // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入
            // 图像的 ROI 子图像尺寸相同的图像。
            errcode = ImageBasicOp::makeAtCurrentDevice(
                    outimg, inimg->roiX2 - inimg->roiX1, 
                    inimg->roiY2 - inimg->roiY1);
            // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
            if (errcode != NO_ERROR)
                return errcode;
        }

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

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 blocksize, gridsize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / 
                      blocksize.x;
        gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                     (blocksize.y * 4);

        // 调用核函数，根据特征值标记进行形状区域拷贝。
        _selectShapeByValueKer<<<gridsize, blocksize>>>(
                insubimgCud, outsubimgCud, devflaglabel);
                
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;      
        } 
                            
       // 如果输入图像等于输出图像，或者输出图像为空。
    } else {
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 blocksize, gridsize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / 
                      blocksize.x;
        gridsize.y = (insubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                     (blocksize.y * 4);

        // 调用核函数，根据特征值标记进行形状区域拷贝。
        _shapeClearByLabelKer<<<gridsize, blocksize>>>(
                insubimgCud, devflaglabel);
                
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(alldevicedata);
            return CUDA_ERROR;      
        }                 
    }
    // 释放 Device 上的临时空间 alldevicedata。
    cudaFree(alldevicedata);

    return NO_ERROR;
}
