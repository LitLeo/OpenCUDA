// CombineImage.cu
// 将若干幅图像融合成一幅图像。要求这些图像的 ROI 子区域的尺寸完全相同。

#include "CombineImage.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_combineImageMaxKer（以最大值方式合并图像）
// 输出图像的对应坐标位置的像素值为各个输入图像对应位置的像素的最大值。
static __global__ void           // Kernel 函数无返回值
_combineImageMaxKer(
        ImageCuda inimg[],       // 输入图像，多幅图像
        unsigned imgcnt,         // 输入图像的数量
        ImageCuda outimg         // 输出图像
);


// Kernel 函数：_combineImageMaxKer（以最大值方式合并图像）
__global__ void _combineImageMaxKer(
        ImageCuda inimg[], unsigned imgcnt, ImageCuda outimg)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;

    // 计算第一个输入坐标点对应的第一幅输入图像数据数组下标。
    int inidx0 = r * inimg[0].pitchBytes + c;
    // 计算第一个输出坐标点对应的输出图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值作为最大值的初始化值。
    unsigned char curmax = inimg[0].imgMeta.imgData[inidx0];

    // 迭代其他的输入图像，求取这些图像对应位置的最大值。
    for (int j = 1; j < imgcnt; j++) {
        // 读取第 j 幅图像当前坐标位置下的像素值。
        int inidxj = r * inimg[j].pitchBytes + c;
        unsigned char curval = inimg[j].imgMeta.imgData[inidxj];

        // 如果第 j 幅图像当前坐标下的像素值大于当前的最大值，则更新最大值。
        if (curval > curmax)
            curmax = curval;
    }
    // 将当前最大值输出到输出图像的对应位置。
    outimg.imgMeta.imgData[outidx] = curmax;

    // 处理剩下的三个像素点。
    for (int i = 1; i < 4; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx0 += inimg[0].pitchBytes;
        outidx += outimg.pitchBytes;
        curmax = inimg[0].imgMeta.imgData[inidx0];
        
        // 迭代其他的输入图像，求取这些图像对应位置的最大值。
        for (int j = 1; j < imgcnt; j++) {
            // 读取第 j 幅图像当前坐标位置下的像素值。
            int inidxj = r * inimg[j].pitchBytes + c;
            unsigned char curval = inimg[j].imgMeta.imgData[inidxj];

            // 如果第 j 幅图像当前坐标下的像素值大于当前的最大值，则更新最大值。
            if (curval > curmax)
                curmax = curval;
        }
        // 将当前最大值输出到输出图像的对应位置。
        outimg.imgMeta.imgData[outidx] = curmax;
    }
}

// Host 成员方法：combineImageMax（以最大值的方式合并图像）
__host__ int CombineImage::combineImageMax(Image **inimg, int imgcnt, 
                                           Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 如果待合并的图像数量为 0，则直接报错返回。
    if (imgcnt < 1)
        return INVALID_DATA;

    // 这个迭代完成三件事情，第一个检查是否存在 NULL 作为输入图像的情况，第二个
    // 将输入图像的数据部分拷贝到设备端，第三个计算出最大可能的输出图像 ROI 尺
    // 寸。
    // 首先利用第 0 幅图像的信息初始化迭代数据。
    if (inimg[0] == NULL)
        return NULL_POINTER;
    int errcode = ImageBasicOp::copyToCurrentDevice(inimg[0]);
    if (errcode != NO_ERROR)
        return errcode;
    int maxoutroiw = inimg[0]->roiX2 - inimg[0]->roiX1;
    int maxoutroih = inimg[0]->roiY2 - inimg[0]->roiY1;
    // 之后迭代其他的输入图像，完成判断和计算。
    for (int i = 1; i < imgcnt; i++) {
        // 如果当前输入图像为一个空指针，则报错退出。
        if (inimg[i] == NULL)
            return NULL_POINTER;

        // 拷贝输入图像的数据到设备端。
        errcode = ImageBasicOp::copyToCurrentDevice(inimg[i]);
        if (errcode != NO_ERROR)
            return errcode;

        // 迭代计算得到各个输入图像中最小的 ROI 尺寸，这个尺寸是输出图像最大合
        // 法的 ROI 区域尺寸。
        int inroiw = inimg[i]->roiX2 - inimg[i]->roiX1;
        int inroih = inimg[i]->roiY2 - inimg[i]->roiY1;
        if (inroiw < maxoutroiw)
            maxoutroiw = inroiw;
        if (inroih < maxoutroih)
            maxoutroih = inroih;
    }
    
    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和具有最大
        // 合法 ROI 区域尺寸的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, maxoutroiw, maxoutroih);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    } else {
        // 如果拷贝成功，则说明 outimg 原来已有数据，这是需要判断其 ROI 区域是
        // 否合法。
        if (outimg->roiX2 - outimg->roiX1 > maxoutroiw ||
            outimg->roiY2 - outimg->roiY1 > maxoutroih)
            return UNMATCH_IMG;
    }

    // 后面的内容涉及到在出错时的内存释放，这里定义一个宏简化后面的代码。
#define COMBINE_IMAGE_MAX_ERRFREE(errcode)  do {                   \
            if (insubimgCud != NULL)    delete [] insubimgCud;     \
            if (insubimgCudDev != NULL) cudaFree(insubimgCudDev);  \
            return (errcode);                                      \
        } while (0)

    // 提取输入图像和输出图像的子图像。
    // 由于输入图像由不定个数的多幅图像，因此需要动态申请内存。
    ImageCuda *insubimgCud = NULL, *insubimgCudDev = NULL;
    insubimgCud = new ImageCuda[imgcnt];
    cudaMalloc((void **)&insubimgCudDev, imgcnt * sizeof (ImageCuda));
    if (insubimgCud == NULL || insubimgCudDev == NULL)
        COMBINE_IMAGE_MAX_ERRFREE(OUT_OF_MEM);

    // 迭代提取所有输入图像的子图像
    for (int i = 0; i < imgcnt; i++) {
        errcode = ImageBasicOp::roiSubImage(inimg[i], insubimgCud + i);
        if (errcode != NO_ERROR)
            COMBINE_IMAGE_MAX_ERRFREE(errcode);
    }

    // 将提取出来的输入图像的子图像拷贝集中拷贝到设备内存中。
    cudaError_t cuerrcode = cudaMemcpy(insubimgCudDev, insubimgCud, 
                                       imgcnt * sizeof (ImageCuda), 
                                       cudaMemcpyHostToDevice);
    if (cuerrcode != cudaSuccess)
        COMBINE_IMAGE_MAX_ERRFREE(CUDA_ERROR);

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        COMBINE_IMAGE_MAX_ERRFREE(errcode);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 调用核函数。
    _combineImageMaxKer<<<gridsize, blocksize>>>(
            insubimgCudDev, imgcnt, outsubimgCud);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，清除临时申请的内存，退出。
    delete [] insubimgCud;
    cudaFree(insubimgCudDev);
    return NO_ERROR;
#undef COMBINE_IMAGE_MAX_ERRFREE
}

