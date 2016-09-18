// Zoom.cu
// 实现图像的放大镜操作

#include "Zoom.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：放大率变换函数
// 定义了放大率变换函数（现为二次函数）
#define fac(r, mR, mag) (- 1 * (mag) / ((mR) * (mR)) * ((r) * (r)) + (mag) + 1)

// 纹理内存只能用于全局变量，使用全局存储时需要加入边界判断，经测试效率不及
// 纹理内存，纹理拾取返回的数据类型 unsigned char 型，维度为2，返回类型不转换
static texture<unsigned char, 2, cudaReadModeElementType> _bilateralInimgTex;

// Host 函数：initTexture（初始化纹理内存）
// 将输入图像数据绑定到纹理内存
static __host__ int      // 返回值：若正确执行返回 NO_ERROR
_initTexture(
        Image *insubimg  // 输入图像
);

// Kernel 函数：_zoomKer（放大镜变化）
// 对在给定半径范围内的区域进行放大处理。
static __global__ void           // Kernel 函数无返回值
_zoomKer(
        ImageCuda outimg,     // 输出图像
        int centreX,          // 放大中心横坐标
        int centreY,          // 放大中心纵坐标
        unsigned int radius,  // 区域半径
        float magnifyMul      // 放大倍数
);

// Host 函数：initTexture（初始化纹理内存）
static __host__ int _initTexture(Image *inimg)
{
    cudaError_t cuerrcode;
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

    // 设置数据通道描述符，因为只有一个颜色通道（灰度图），因此描述符中只有第一
    // 个分量含有数据。概述据通道描述符用于纹理内存的绑定操作。
    struct cudaChannelFormatDesc chndesc;
    chndesc = cudaCreateChannelDesc(sizeof (unsigned char) * 8, 0, 0, 0,
                                    cudaChannelFormatKindUnsigned);

    // 将输入图像数据绑定至纹理内存（texture） 
    cuerrcode = cudaBindTexture2D(
            NULL, &_bilateralInimgTex, insubimgCud.imgMeta.imgData, &chndesc, 
            insubimgCud.imgMeta.width, insubimgCud.imgMeta.height, 
            insubimgCud.pitchBytes);
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;
    return NO_ERROR;
}

// Kernel 函数：_zoomKer（放大镜变化）
static __global__ void _zoomKer(ImageCuda outimg, 
                                 int centreX, int centreY, 
                                 unsigned int radius, float magnifyMul)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x; 
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    float realMagnifyMul;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;

    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;

    // 获取当前处理点的像素值
    unsigned char inimg;
    // 第一个目标点
    inimg = tex2D(_bilateralInimgTex, c, r);

    // 给第一个输出图像点的像素值设定为输入图像点的像素值
    outimg.imgMeta.imgData[outidx] = inimg;

    // 检查第一个像素点是否在圆内，若在，则进行放大处理，否则，退出判定
    if ((c - centreX) * (c - centreX) + (r - centreY) *
        (r - centreY) <= radius * radius) {
        // 计算第一个像素点坐标，是放大中心点与当前点中间的最靠近圆心
        // 的 magnifyMul 等分点
        float rouge = sqrt(float(c - centreX) * (c - centreX) + 
                      (r - centreY) * (r - centreY));
        // 获得当前距离下的放大率
        realMagnifyMul = fac(rouge, radius, magnifyMul);
        // 获得放大目标放大后的对应点位置
        int inc = floor((c + (realMagnifyMul - 1) * centreX) / realMagnifyMul);
        int inr = floor((r + (realMagnifyMul - 1) * centreY) / realMagnifyMul);

        inimg = tex2D(_bilateralInimgTex, inc, inr);

        // 把输出图像在修改半径之内的像素点用输入图像中放大中心点与当前点中间的
        // 最靠近圆心的magnifyMul等分点代替
        outimg.imgMeta.imgData[outidx] = inimg;
    }

    // 处理剩下的三个像素点。
    for(int i = 1; i <= 3; i++){
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        outidx += outimg.pitchBytes;
        // 给输出图像点的像素值设定为输入图像点的像素值
        inimg = tex2D(_bilateralInimgTex, c, r);
        outimg.imgMeta.imgData[outidx] = inimg;

        // 检查像素点是否在圆内，若在，则进行放大处理，否则，退出判定
        if ((c - centreX) * (c - centreX) + (r - centreY) *
            (r - centreY) <= radius * radius) {
            // 计算目标像素点坐标，是放大中心点与当前点中间的最靠近圆心
            // 的magnifyMul等分点
            float rouge = sqrt(float(c - centreX) * (c - centreX) + 
                          (r - centreY) * (r - centreY));
            // 获得当前距离下的放大率
            realMagnifyMul = fac(rouge, radius, magnifyMul);
            // 获得放大目标放大后的对应点位置
            int inc = floor((c + (realMagnifyMul - 1) * centreX) / 
                      realMagnifyMul);
            int inr = floor((r + (realMagnifyMul - 1) * centreY) / 
                      realMagnifyMul);

            inimg = tex2D(_bilateralInimgTex, inc, inr);

            // 把输出图像在修改半径之内的像素点用输入图像中放大中心点与当前点中
            // 间的最靠近圆心的magnifyMul等分点代替
            outimg.imgMeta.imgData[outidx] = inimg;
        }
    }
}

// Host 成员方法：zoom（放大镜处理）
__host__ int Zoom::zoom(Image *inimg, Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    // 初始化纹理内存，将输入图像与之绑定
    _initTexture(inimg);

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

    // 调用核函数，进行放大镜处理。
    _zoomKer<<<gridsize, blocksize>>>(
            outsubimgCud, centreX, centreY, circleRadius, magnifyMul);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}
