// Tattoo.cu
// 贴图（Tattoo）

#include "Tattoo.h"

#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel函数：_tattooKer（贴图）
// 输入两幅图像，一幅为前景图，一幅为背景图，输出一幅图像，其中输出图像满足当前
// 景图灰度值与指定透明像素相同时，则输出背景图对应的灰度值，否则输出前景图灰度
// 值。
static __global__ void            // Kernel 函数无返回值
_tattooKer(
        ImageCuda frimg,          // 前景图像
        ImageCuda baimg,          // 背景图像
        ImageCuda outimg,         // 输出图像
        unsigned char dummypixel  // 透明像素
);

// Kernel 函数：_tattooKer（贴图）
static __global__ void _tattooKer(ImageCuda frimg, ImageCuda baimg,
                                  ImageCuda outimg, unsigned char dummypixel)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标的 
    // x 和 y 分量（其中，c 表示 column; r 表示 row ）。由于我们采用了并行度缩减
    // 的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻4行上，因
    // 此，对于 r 需要乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= frimg.imgMeta.width  || r >= frimg.imgMeta.height)
        return;

    // 转化为图像数组下标。
    // 计算第一个前景图坐标点对应的图像数据数组下标。
    int fridx = r * frimg.pitchBytes + c;
    // 计算第一个背景图坐标点对应的图像数据数组下标。
    int baidx = r * baimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;

    // 一个线程处理四个像素点。
    // 如果前景图像的该位置的像素值等于 dummypixel, 则输出图像对应位置为背景图像
    // 对应位置的像素值，否则为前景图像对应位置的像素值。
    // 线程中处理的第一个点。
    // 判断前景图中的灰度值是否等于某一透明像素。
    if (frimg.imgMeta.imgData[fridx] == dummypixel)
        outimg.imgMeta.imgData[outidx] = baimg.imgMeta.imgData[baidx];
    else
        outimg.imgMeta.imgData[outidx] = frimg.imgMeta.imgData[fridx];

    // 处理剩下的三个像素点。
    for (int i = 1; i < 4; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因此，
        // 需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点之间没
        //有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y 分
        // 量增加 1，所以下标只需要加上一个 pitch 即可，不需要再进行乘法计算。
        fridx += frimg.pitchBytes;
        baidx += baimg.pitchBytes;
        outidx += outimg.pitchBytes;

        // 判断前景图中的灰度值是否等于某一透明像素。
        if (frimg.imgMeta.imgData[fridx] == dummypixel)
            outimg.imgMeta.imgData[outidx] = baimg.imgMeta.imgData[baidx];
        else
            outimg.imgMeta.imgData[outidx] = frimg.imgMeta.imgData[fridx];
    }
}

// Host 成员方法：tattoo（贴图）
__host__ int Tattoo::tattoo(Image *frimg, Image *baimg, Image *outimg)
{
    // 检查输入、输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (frimg == NULL || baimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 如果两幅输入图像的 ROI 区域大小不一样，返回 UNMATCH_IMG。
    if (frimg->roiX2 - frimg->roiX1 != baimg->roiX2 - baimg->roiX1 ||
        frimg->roiY2 - frimg->roiY1 != baimg->roiY2 - baimg->roiY1)
        return UNMATCH_IMG;

    // 将前景图复制到 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(frimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将背景图复制到 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(baimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将 outimg 复制到 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    // 如果输出图像无数据 (故上面的拷贝函数会失败)，则会创建一个大小和输入图像尺
    // 寸相同的图像。
    if (errcode != NO_ERROR) {
        errcode = ImageBasicOp::makeAtCurrentDevice(
                          outimg, 
                          frimg->roiX2 - frimg->roiX1,
                          frimg->roiY2 - frimg->roiY1);
        // 如果创建图像也操作失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }

    //提取输入图像的 ROI 子图像。
    ImageCuda frsubimgCud, basubimgCud;   
    errcode = ImageBasicOp::roiSubImage(frimg, &frsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::roiSubImage(baimg, &basubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 对输入图像和输出图像的 ROI 区域大小进行调整，调整为一致。
    if (frsubimgCud.imgMeta.width < outsubimgCud.imgMeta.width) {
        outsubimgCud.imgMeta.width = frsubimgCud.imgMeta.width;
    } else {
        frsubimgCud.imgMeta.width = outsubimgCud.imgMeta.width; 
        basubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
    }
    if (frsubimgCud.imgMeta.height < outsubimgCud.imgMeta.height) {
        outsubimgCud.imgMeta.height = frsubimgCud.imgMeta.height;
    } else {
        frsubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
        basubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (frsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (frsubimgCud.imgMeta.height + blocksize.y * 4 - 1) / 
                 (blocksize.y * 4);

    // 调用核函数，根据阈值进行贴图处理。
    _tattooKer<<<gridsize, blocksize>>>(frsubimgCud, basubimgCud, outsubimgCud, 
                                        dummyPixel);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}
