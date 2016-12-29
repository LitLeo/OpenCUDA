// LinearEnhancement.cu
// 实现图像的线性增强
 
#include "LinearEnhancement.h" 

#include <iostream> 
using namespace std;
 
#include "ErrorCode.h" 
 
 
// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32 
#define DEF_BLOCK_Y   8 


// 核函数：_inPlaceLinearEnhanceKer （线性增强）   
// 根据给定的参数对图像进行增前处理，这是一个 In-place 形式的处理。
// 对于每一个像素，根据四个参数确定的函数进行变化。
static __global__ void        // Kernel 函数无返回值
_inPlaceLinearEnhanceKer(
        ImageCuda inimgCud,   // 待处理图像
        LinearEnhancement le  // 图像增强类
);

// 核函数：_linearEnhanceKer （线性增强）   
// 根据给定的参数对图像进行增前处理，这是一个 Out-place 形式的处理。
// 对于每一个像素，根据四个参数确定的函数进行变化。
static __global__ void        // Kernel 函数无返回值
_linearEnhanceKer( 
        ImageCuda inimgCud,   // 输入图像
        ImageCuda outimgCud,  // 输出图像
        LinearEnhancement le  // 图像增强类
); 


// 核函数：_linearEnhanceKer （ 线性增强）   
static __global__ void _linearEnhanceKer( 
        ImageCuda inimgCud, ImageCuda outimgCud, LinearEnhancement le) 
{ 
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
 
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (dstc >= inimgCud.imgMeta.width || dstr >= inimgCud.imgMeta.height) 
        return;
 
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = dstr * inimgCud.pitchBytes + dstc;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = dstr * outimgCud.pitchBytes + dstc;
    
    // 查转置表得到输出图像对应位置的像素     
    outimgCud.imgMeta.imgData[outidx] = 
            le.getPixelTable()[inimgCud.imgMeta.imgData[inidx]];
 
    // 处理剩下的三个点
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++dstr >= inimgCud.imgMeta.height)
            return;
 
        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimgCud.pitchBytes;
        outidx += outimgCud.pitchBytes;
        
        // 查转置表得到输出图像对应位置的像素
        outimgCud.imgMeta.imgData[outidx] = 
                le.getPixelTable()[inimgCud.imgMeta.imgData[inidx]];
    } 
} 
  
// 成员函数：linearEnhance （线性增强）
__host__ int LinearEnhancement::linearEnhance(Image *inimg, Image *outimg) 
{ 
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL) 
        return NULL_POINTER;
 
    // 如果输出图像为 NULL，直接调用 In—Place 版本的成员方法。  
    if (outimg == NULL || inimg == outimg) 
        return linearEnhance(inimg);
 
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
       
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) / 
                 (blocksize.y * 4);
     
    _linearEnhanceKer<<<gridsize, blocksize>>>( 
            insubimgCud, outsubimgCud, *this);     
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) 
        return CUDA_ERROR;

    return NO_ERROR;
}  
 
// 核函数：_inPlaceLinearEnhanceKer （ 线性增强）   
static __global__ void _inPlaceLinearEnhanceKer(
        ImageCuda inimgCud, LinearEnhancement le) 
{ 

    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
 
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (dstc >= inimgCud.imgMeta.width || dstr >= inimgCud.imgMeta.height) 
        return;
 
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int dstidx = dstr * inimgCud.pitchBytes + dstc;
    
    // 查转置表得到输出图像对应位置的像素    
    inimgCud.imgMeta.imgData[dstidx] = 
            le.getPixelTable()[inimgCud.imgMeta.imgData[dstidx]];
 
    // 处理剩下的三个点
    for (int i = 0; i < 3; i++) { 
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++dstr >= inimgCud.imgMeta.height) 
            return;
 
        // 计算输入坐标点以及输出坐标点，由于只有 y 分量增加 1，所以下标只需
        // 要加上对应的 pitch 即可，不需要在进行乘法计
        dstidx += inimgCud.pitchBytes;
         
        // 查转置表得到输出图像对应位置的像素    
        inimgCud.imgMeta.imgData[dstidx] = 
                le.getPixelTable()[inimgCud.imgMeta.imgData[dstidx]];
    } 
}

// 成员函数：linearEnhance（ 线性增强）
__host__ int LinearEnhancement::linearEnhance(Image *inimg) 
{ 
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL) 
        return NULL_POINTER;
 
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
 
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y * 4 - 1) / 
                 (blocksize.y * 4);
 
    _inPlaceLinearEnhanceKer<<<gridsize, blocksize>>>(insubimgCud, *this); 
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) 
        return CUDA_ERROR;

    return NO_ERROR;
} 
 
