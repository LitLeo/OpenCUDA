// InnerDigger.cu
// 实现区域抠心

#include "InnerDigger.h"

#include "Image.h"
#include "ErrorCode.h"

// 宏 DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸
#define DEF_BLOCK_X    32
#define DEF_BLOCK_Y     8


// Kernel 函数：_InnerDiggerKer（区域抠心）
// 根据输入的灰度图，对其每个像素的四领域或八领域做判断如果都为白色，则将其设为
// 黑色
static __global__ void     // 无返回值
_InnerDiggerKer(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 输出图像
        int mode           // 识别是四领域还是八领域抠心
);

// Kernel 函数：_InnerDigger（区域抠心）
static __global__ void _InnerDiggerKer(
        ImageCuda inimg, ImageCuda outimg, int mode)
{
    // 获取线程索引，图像索引采用线程索引
    int tidc = blockIdx.x * blockDim.x + threadIdx.x;
    int tidr = blockIdx.y * blockDim.y + threadIdx.y;

    // 转化为图像下标
    int idin = tidr * inimg.pitchBytes + tidc;
    int idout = tidr * outimg.pitchBytes + tidc;

    // 判断是否越界
    if (tidc >= inimg.imgMeta.width || tidr >= inimg.imgMeta.height)
        return;

    // 处理边界
    if (tidc == 0 || tidc == inimg.imgMeta.width - 1 ||
        tidr == 0 || tidr == inimg.imgMeta.height - 1) {
        // 对于边界的像素直接复制源图像的像素值
        outimg.imgMeta.imgData[idout] = inimg.imgMeta.imgData[idin];
        return;
    }

    // 对是四领域抠心还是八领域抠心做出判断如果是四领域抠心
    if (mode == CR_FOUR_AREAS) {
        // 判断其四领域像素是否都为白色如果不是白色区域的中心，则复制输入图像的
        // 像素值
        if (inimg.imgMeta.imgData[idin - inimg.pitchBytes] == 255 &&
            inimg.imgMeta.imgData[idin - 1] == 255 &&
            inimg.imgMeta.imgData[idin + 1] == 255 &&
            inimg.imgMeta.imgData[idin + inimg.pitchBytes] == 255)
            // 设置为黑色
            outimg.imgMeta.imgData[idout] = 0;
        else
            outimg.imgMeta.imgData[idout] = inimg.imgMeta.imgData[idin];
    } else {
        // 判断其八领域像素是否都为白色，如不是白色区域的中心，则复制输入图像的
        // 像素值
        if (inimg.imgMeta.imgData[idin - 1 - inimg.pitchBytes] == 255 && 
            inimg.imgMeta.imgData[idin - inimg.pitchBytes] == 255 && 
            inimg.imgMeta.imgData[idin + 1 - inimg.pitchBytes] == 255 && 
            inimg.imgMeta.imgData[idin - 1] == 255 && 
            inimg.imgMeta.imgData[idin + 1] == 255 && 
            inimg.imgMeta.imgData[idin - 1 + inimg.pitchBytes] == 255 && 
            inimg.imgMeta.imgData[idin + inimg.pitchBytes] == 255 && 
            inimg.imgMeta.imgData[idin + 1 + inimg.pitchBytes] == 255)
            // 设置为黑色
            outimg.imgMeta.imgData[idout] = 0;
        else
            outimg.imgMeta.imgData[idout] = inimg.imgMeta.imgData[idin];
    }
}

// Host 成员方法：innerDigger（区域抠心）
__host__ int InnerDigger::innerDigger(Image *inimg, Image *outimg)
{
    // 检查输入输出图像是否为空
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;
    
    // 检查图像是否为空
    if (inimg->imgData == NULL)
        return UNMATCH_IMG;

    // 将输入图像复制到 Device
    int errcode;
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 将输出图像复制到 Device，如果复制 outimg 失败，则创建一个图像
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        int outw = inimg->roiX2 - inimg->roiX1;
        int outh = inimg->roiY2 - inimg->roiY1;
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, outw, outh);
        // 如果创建失败，则退出
        if (errcode != NO_ERROR)
            return errcode;
    }
    
    // 提取输入图像的 ROI 子图
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输出图像的 ROI 子图
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

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X; 
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;
    
    // 调用 Kernel 函数：_InnerDigger（区域抠心）
    _InnerDiggerKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud, mode);

    // 调用 cudaGetLastError 判断程序是否出错
    cudaError_t cuerrcode;
    cuerrcode = cudaGetLastError();
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出
    return NO_ERROR;
}

