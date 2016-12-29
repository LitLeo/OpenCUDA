// RoiCopy.cu
// 拷贝图片的 ROI 子图

#include "ErrorCode.h"
#include "RoiCopy.h"

// 成员函数：roiCopyAtHost（拷贝图像的 ROI 子图 Host 版本）
__host__ int RoiCopy::roiCopyAtHost(Image *inimg, Image *outimg)
{
    int errcode;          // 局部变量，错误码
    cudaError_t cudaerr;  // CUDA 错误码

    // 判断 inimg 和 outimg 是否为空，若为空，返回错误
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER; 

    // 将输入图像拷贝到 Host 端
    errcode = ImageBasicOp::copyToHost(inimg);
    if (errcode != NO_ERROR)
        return errcode;
    // 将输出图像拷贝到 Host 端
    errcode = ImageBasicOp::copyToHost(outimg);
    // 若输出图像无数据，则根据输入图像子图的大小创建输出图像的数据
    if (errcode != NO_ERROR) {
        // 计算输入图像子图的宽和高
        int roiwidth = inimg->roiX2 - inimg->roiX1;
        int roiheight = inimg->roiY2 - inimg->roiY1;
        // 为输出图像申请在 Host 端申请空间
        errcode = ImageBasicOp::makeAtHost(outimg, roiwidth, roiheight);
        if (errcode != NO_ERROR)
            return errcode;
    }

    ImageCuda outcuda, incuda;
    // 提取输入图像的子图
    errcode = ImageBasicOp::roiSubImage(inimg, &incuda);
    if (errcode != NO_ERROR)
        return errcode;
    // 提取输出图像的子图
    errcode = ImageBasicOp::roiSubImage(outimg, &outcuda);
    if (errcode != NO_ERROR)
        return errcode; 

    // 调整输入图像和输出图像的大小，是两个图像的大小相同
    if (incuda.imgMeta.width > outcuda.imgMeta.width)
        incuda.imgMeta.width = outcuda.imgMeta.width;
    else
        outcuda.imgMeta.width = incuda.imgMeta.width;

    if (incuda.imgMeta.height > outcuda.imgMeta.height)
        incuda.imgMeta.height = outcuda.imgMeta.height;
    else
        outcuda.imgMeta.height = incuda.imgMeta.height;

    // 将输入图像的子图拷贝到输出图像的子图中
    cudaerr = cudaMemcpy2D((void *)outcuda.imgMeta.imgData, outcuda.pitchBytes,
                           (void *)incuda.imgMeta.imgData, incuda.pitchBytes,
                           incuda.imgMeta.width, incuda.imgMeta.height,
                           cudaMemcpyHostToHost);
    // 若拷贝失败，则返回错误
    if (cudaerr != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}

// 成员函数：roiCopyAtDevice（拷贝图像的 ROI 子图 Device 版本）
__host__ int RoiCopy::roiCopyAtDevice(Image *inimg, Image *outimg)
{
    int errcode;          // 局部变量，错误码
    cudaError_t cudaerr;  // CUDA 错误码

    // 判断 inimg 和 outimg 是否为空，若为空，返回错误
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER; 

    // 将输入图像拷贝到当前 Device 端
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
    // 将输出图像拷贝到当前 Device 端
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    // 若输出图像无数据，则根据输入图像子图的大小创建输出图像的数据
    if (errcode != NO_ERROR) {
        // 计算输入图像子图的宽和高
        int roiwidth = inimg->roiX2 - inimg->roiX1;
        int roiheight = inimg->roiY2 - inimg->roiY1;
        // 为输出图像申请在当前 Device 端申请空间
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, 
                                                    roiwidth, roiheight);
        if (errcode != NO_ERROR)
            return errcode;
    }

    ImageCuda outcuda, incuda;
    // 提取输入图像的子图
    errcode = ImageBasicOp::roiSubImage(inimg, &incuda);
    if (errcode != NO_ERROR)
        return errcode;
    // 提取输出图像的子图
    errcode = ImageBasicOp::roiSubImage(outimg, &outcuda);
    if (errcode != NO_ERROR)
        return errcode; 

    // 调整输入图像和输出图像的大小，是两个图像的大小相同
    if (incuda.imgMeta.width > outcuda.imgMeta.width)
        incuda.imgMeta.width = outcuda.imgMeta.width;
    else
        outcuda.imgMeta.width = incuda.imgMeta.width;

    if (incuda.imgMeta.height > outcuda.imgMeta.height)
        incuda.imgMeta.height = outcuda.imgMeta.height;
    else
        outcuda.imgMeta.height = incuda.imgMeta.height;

    // 将输入图像的子图拷贝到输出图像的子图中
    cudaerr = cudaMemcpy2D((void *)outcuda.imgMeta.imgData, outcuda.pitchBytes,
                           (void *)incuda.imgMeta.imgData, incuda.pitchBytes,
                           incuda.imgMeta.width, incuda.imgMeta.height,
                           cudaMemcpyDeviceToDevice);
    // 若拷贝失败，则返回错误
    if (cudaerr != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}
