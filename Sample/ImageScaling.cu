// ImageScaling.cu
// 实现图像的旋转变换

#include "ImageScaling.h"

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 全局变量：_imageScalingTex（作为输入图像的纹理内存引用）
// 纹理内存只能用于全局变量，因此将图像扩缩的 Kernel 函数的输入图像列
// 于此处。
static texture<unsigned char, 2, cudaReadModeElementType> _imageScalingTex;

// Kernel 函数：_imageScalingKer（实现图像扩缩）
// 利用纹理内存提供的硬件插值功能实现图像扩缩。没有输入图像的参数，是因
// 为输入图像通过纹理内存来读取数据，纹理内存只能声明为全局变量。
static __global__ void         // Kernel 函数无返回值。
_imageScalingKer(
        ImageCuda outimg,         // 输出图像
        int x,                    // 扩缩中心 x 分量
        int y,                    // 扩缩中心 y 分量
        float tmpScalCoefficient  // 扩缩系数的倒数，用于扩缩计算
);


// Kernel 函数：_imageScalingKer（利用硬件插值实现图像扩缩）
static __global__ void _imageScalingKer(ImageCuda outimg, int x, int y, 
                                        float tmpScalCoefficient)
{
    // 计算想成对应的输出点的位置，其中 dstc 和 dstr 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻
    // 4 行上，因此，对于 dstr 需要进行乘 4 计算。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (dstc >= outimg.imgMeta.width || dstr >= outimg.imgMeta.height)
        return;
    
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int dstidx = dstr * outimg.pitchBytes + dstc;
    
    // 声明目标图像输出像素对应的源图像中的坐标点，由于计算会得到小数结果，因此
    // 使用浮点型存储该做标。
    float srcc, srcr;
   
    // 通过目标坐标点反推回源图像中的坐标点
    srcc = (dstc - x) * tmpScalCoefficient + x;
    srcr = (dstr - y) * tmpScalCoefficient + y;
    
    // 通过上面的计算，求出了第一个输出坐标对应的源图像坐标。这里利用纹理内存的
    // 硬件插值功能，直接使用浮点型的坐标读取相应的源图像“像素”值，并赋值给目
    // 标图像。这里没有进行对源图像读取的越界检查，这是因为纹理内存硬件插值功能
    // 可以处理越界访问的情况，越界访问会按照事先的设置得到一个相对合理的像素颜
    // 色值，不会引起错误。
    outimg.imgMeta.imgData[dstidx] = tex2D(_imageScalingTex, srcc, srcr);
   
    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++dstr >= outimg.imgMeta.height)
            return;
        
        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        dstidx += outimg.pitchBytes;
        
        // 计算当前的源坐标位置。
        srcr += tmpScalCoefficient;
        
        // 将对应的源坐标位置出的插值像素写入到目标图像的当前像素点中。
        outimg.imgMeta.imgData[dstidx] = tex2D(_imageScalingTex, srcc, srcr);
    }
}


// Host 成员方法：scaling（图像扩缩）
__host__ int ImageScaling::scaling(Image *inimg, Image *outimg)
{
    int errcode;  // 局部变量，错误码

    // 检查输入和输出图像，若有一个为 NULL，则报错。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;
           
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图像
        // 尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->width, inimg->height);
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

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    cudaError_t cuerrcode;
    // 使用硬件插值实现图像扩缩
    // 设置数据通道描述符，因为只有一个颜色通道（灰度图），因此描述符中只有
    // 第一个分量含有数据。概述据通道描述符用于纹理内存的绑定操作。
    struct cudaChannelFormatDesc chndesc;
    chndesc = cudaCreateChannelDesc(sizeof (unsigned char) * 8, 0, 0, 0,
                                    cudaChannelFormatKindUnsigned);

     // 将输入图像的 ROI 子图像绑定到纹理内存。
    cuerrcode = cudaBindTexture2D(
            NULL, &_imageScalingTex, inimg->imgData, &chndesc, 
            inimg->width, inimg->height, insubimgCud.pitchBytes);
                  
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;

    // 调用 Kernel 函数，完成实际的图像旋转变换。
    _imageScalingKer<<<gridsize, blocksize>>>(outsubimgCud, x, y,
                                              1 / scalCoefficient);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
    // 处理完毕退出。
    return NO_ERROR;
}

