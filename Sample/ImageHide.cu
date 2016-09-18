// ImageHide.cu
// 实现二值图隐藏于正常图片中

#include "ImageHide.h"

#include <iostream>
#include <fstream>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数: _imageHideKer（二值图像隐藏）
// 根据给定的载体图像和要隐藏的二值图像进行二值隐藏处理，
// 这是一个 Out-Place 形式的处理。
// 本二值图像的灰度值只有 0 和 255，可以把所有灰度值看成一个 01 序列，
// 隐藏这个二值序列，如果是 0，则载体图像的灰度值最低位置 0，否则最低位置 1。
static __global__ void     // Kernel 函数无返回值
_imageHideKer(
        ImageCuda inimg1,  // 原图像
        ImageCuda inimg2,  // 待隐藏的二值图
        ImageCuda outimg   // 输出图像
);

// Kernel 函数: _imageHideInKer（二值图像隐藏）
// 根据给定的载体图像和要隐藏的二值图像进行二值隐藏处理，
// 这是一个 In-Place 形式的处理。
// 本二值图像的灰度值只有 0 和 255，可以把所有灰度值看成一个 01 序列，
// 隐藏这个二值序列，如果是 0，则载体图像的灰度值最低位置 0，否则最低位置1。
static __global__ void     // Kernel 函数无返回值
_imageHideInKer(
        ImageCuda inimg1,  // 原图像
        ImageCuda inimg2   // 待隐藏的二值图
);


// Kernel 函数：_imageHideKer（二值图像隐藏）
static __global__ void _imageHideKer(ImageCuda inimg1, ImageCuda inimg2,
                                     ImageCuda outimg)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg1.imgMeta.width || r >= inimg1.imgMeta.height)
        return;

    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx1 = r * inimg1.pitchBytes + c;
    int inidx2 = r * inimg2.pitchBytes + c;

    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;

    // 读取两图像的第一个输入坐标点和对应的像素值。
    unsigned char intemp1;
    unsigned char intemp2;
    intemp1 = inimg1.imgMeta.imgData[inidx1];
    intemp2 = inimg2.imgMeta.imgData[inidx2];

    if (c < inimg2.imgMeta.width && r < inimg2.imgMeta.height) { 
        // 当二值图的灰度值为 0 时 提取原图像对应位置的灰度值，
        // 把该灰度值的最低位置为 0 把生成的新的灰度值赋值给输出图像。
        // 当二值图的灰度值为 255 时 提取原图像对应位置的灰度值，
        // 把该灰度值的最低位置为 1 把生成的新的灰度值赋值给输出图像。
        outimg.imgMeta.imgData[outidx] = ((intemp1 & 0xFE) | (intemp2 != 0));
    } else {
        // 当二值图所有像素点处理完后，载体图像剩余的像素值直接赋值到输出
        // 图像。
        outimg.imgMeta.imgData[outidx] = intemp1;
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
        inidx1 += inimg1.pitchBytes;
        inidx2 += inimg2.pitchBytes;
        outidx += outimg.pitchBytes;
        intemp1 = inimg1.imgMeta.imgData[inidx1];
        intemp2 = inimg2.imgMeta.imgData[inidx2];

        if(r < inimg2.imgMeta.height) {
            // 当二值图的灰度值为 0 时 提取原图像对应位置的灰度值，
            // 把该灰度值的最低位置为 0 把生成的新的灰度值赋值给输出图像。
            // 当二值图的灰度值为 255 时 提取原图像对应位置的灰度值，
            // 把该灰度值的最低位置为 1 把生成的新的灰度值赋值给输出图像。
            outimg.imgMeta.imgData[outidx] = ((intemp1 & 0xFE) | (
                                              intemp2 != 0));
        } else {
            // 当二值图所有像素点处理完后，载体图像剩余的像素值直接赋值到输出
            //图像。
            outimg.imgMeta.imgData[outidx] = intemp1;
        }
    }
}

// Host 成员方法：imageHide（二值图像隐藏处理）
__host__ int ImageHide::imageHide(Image *inimg1, Image *inimg2, Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg1 == NULL || inimg2 == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像 1 拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg1);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入图像 2 拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg2);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像1的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg1->roiX2 - inimg1->roiX1, 
                inimg1->roiY2 - inimg1->roiY1);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 提取输入图像 1 的 ROI 子图像。
    ImageCuda inimg1Cud;
    errcode = ImageBasicOp::roiSubImage(inimg1, &inimg1Cud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输入图像 2 的 ROI 子图像。
    ImageCuda inimg2Cud;
    errcode = ImageBasicOp::roiSubImage(inimg2, &inimg2Cud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
	
    // 如果要隐藏的二值图像的子图像长度或者宽度大于载体子图像就错误返回
    if (inimg1Cud.imgMeta.width < inimg2Cud.imgMeta.width || 
        inimg1Cud.imgMeta.height < inimg2Cud.imgMeta.height)
        return INVALID_DATA;

    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一	
    if (inimg1Cud.imgMeta.width > outsubimgCud.imgMeta.width)
        inimg1Cud.imgMeta.width = outsubimgCud.imgMeta.width;
    else
        outsubimgCud.imgMeta.width = inimg1Cud.imgMeta.width;
		
    if (inimg1Cud.imgMeta.height > outsubimgCud.imgMeta.height)
        inimg1Cud.imgMeta.height = outsubimgCud.imgMeta.height;
    else
        outsubimgCud.imgMeta.height = inimg1Cud.imgMeta.height;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize(DEF_BLOCK_X,DEF_BLOCK_Y);
    dim3 gridsize;
    gridsize.x = (inimg1Cud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (inimg1Cud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);
    
    // 调用核函数，根据阈值 threshold 进行二值化处理。
    _imageHideKer<<<gridsize, blocksize>>>(inimg1Cud, inimg2Cud, 
                                           outsubimgCud);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
			
    // 处理完毕，退出。	
    return NO_ERROR;
}

// Kernel 函数：_imageHideInKer（二值图像隐藏）
static __global__ void _imageHideInKer(ImageCuda inimg1, ImageCuda inimg2)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg2.imgMeta.width || r >= inimg2.imgMeta.height)
        return;

    // 计算输入坐标点对应的图像数据数组下标。
    int inidx1 = r * inimg1.pitchBytes + c;
    int inidx2 = r * inimg2.pitchBytes + c;

    // 读取两图像的输入坐标点对应的像素值。
    unsigned char intemp1;
    unsigned char intemp2;
    intemp1 = inimg1.imgMeta.imgData[inidx1];
    intemp2 = inimg2.imgMeta.imgData[inidx2];

    // 当二值图的灰度值为 0 时 提取原图像对应位置的灰度值，
    // 把该灰度值的最低位置为 0 把生成的新的灰度值赋值给输出图像。
    // 当二值图的灰度值为 255 时 提取原图像对应位置的灰度值，
    // 把该灰度值的最低位置为 1 把生成的新的灰度值赋值给输出图像。
    inimg1.imgMeta.imgData[inidx1] = ((intemp1 & 0xFE) | (intemp2 != 0));

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= inimg2.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx1 += inimg1.pitchBytes;
        inidx2 += inimg2.pitchBytes;
        intemp1 = inimg1.imgMeta.imgData[inidx1];
        intemp2 = inimg2.imgMeta.imgData[inidx2];

        // 当二值图的灰度值为 0 时 提取原图像对应位置的灰度值，
        // 把该灰度值的最低位置为 0 把生成的新的灰度值赋值给输出图像。
        // 当二值图的灰度值为 255 时 提取原图像对应位置的灰度值，
        // 把该灰度值的最低位置为 1 把生成的新的灰度值赋值给输出图像。
        inimg1.imgMeta.imgData[inidx1] = ((intemp1 & 0xFE) | (intemp2 != 0));
    }
}

// Host 成员方法：imageHide（二值图像隐藏处理）
__host__ int ImageHide::imageHide(Image *inimg1, Image *inimg2)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg1 == NULL || inimg2 == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像 1 拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg1);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入图像 2 拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg2);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取图像的 ROI 子图像。
    ImageCuda inimg1Cud;
    errcode = ImageBasicOp::roiSubImage(inimg1, &inimg1Cud);
    if (errcode != NO_ERROR)
        return errcode;
    
    ImageCuda inimg2Cud;
    errcode = ImageBasicOp::roiSubImage(inimg2, &inimg2Cud);
    if (errcode != NO_ERROR)
        return errcode;
	
    // 如果要隐藏的二值图像的子图像长度或者宽度大于载体子图像就错误返回
    if (inimg1Cud.imgMeta.width < inimg2Cud.imgMeta.width || 
        inimg1Cud.imgMeta.height < inimg2Cud.imgMeta.height)
        return INVALID_DATA;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize(DEF_BLOCK_X,DEF_BLOCK_Y);
    dim3 gridsize;
    gridsize.x = (inimg2Cud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (inimg2Cud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);
    
    // 调用核函数，根据阈值 threshold 进行二值化处理。
    _imageHideInKer<<<gridsize, blocksize>>>(inimg1Cud, inimg2Cud);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
			
    // 处理完毕，退出。	
    return NO_ERROR;
}

