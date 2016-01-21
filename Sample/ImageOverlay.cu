// ImageOverlay.cu
// 实现 n 幅图像叠加

#include "ImageOverlay.h"
#include "ErrorCode.h"
#include "RoiCopy.h"
#include <cmath>

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_imageoverlayKer（图像叠加）
// 将 n 幅输入图像的像素灰度值逐点与相应权重相乘，权重值范围为[0, 1]且总和为 1，
// 然后求和，结果去尾为整数，且范围为[0, 255]，得到图像 outimg。
static __global__ void      // Kernel 函数无返回值
_imageOverlayKer(
        ImageCuda inimg[],  // 输入图像集合
        int n,              // 输入图像个数
        float weight[],     // 存储输入图像对应权重。
        int weightLength,   // 权重数组个数
        ImageCuda outimg    // 输出图像。
);

// Kernel 函数: _imageOverlayKer（图像叠加）
static __global__ void _imageOverlayKer(ImageCuda *inimg, int n, float *weight,
                                        int weightLength, ImageCuda outimg)
{
    // 循环语句局部变量。
    int i;

    // 获取线程索引，c 代表行，r 代表列。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断 r 和 c 是否超过了图像的尺寸。
    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;

    // 声明 Shared Memory，用来计算该次计算的权重值总和。
    __shared__ float sum[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // 将总和的初始值设为 0。
        sum[0] = 0.0f;
        // 如果权重数组不为空且权重数组长度不为 0，则循环按输入图像数目加权。
        if (weight != NULL && weightLength != 0)
            // 求 n 幅输入图像的权重和。
            for (i = 0; i < n; i++)
                // 第 i 幅图对应权重数组中第 i % weightLength 个权重值。
                sum[0] += weight[i % weightLength];
    }

    // 同步所有线程，使初始化 Shared Memory 的结果对所有线程可见。
    __syncthreads();

    // 转化为图像数组下标。
    int outidx = r * outimg.pitchBytes + c;

    // 第 i 个输入图像的数组下标，随相加时变化。
    int inidx;

    // 对应 n 幅图像灰度值乘以对应图像权重并相加，然后除以权重和 sum。
    // 用 result 暂时存储数据和。
    int result = 0;
    if (weight == NULL || fabs(sum[0] - 0.0f) < 1.0e-6f) {
        // 若 weight 未设置或值全为 0，则将权重看做都相等。
        for (i = 0; i < n; i++) {
            inidx = r * inimg[i].pitchBytes + c;
            result += (float)inimg[i].imgMeta.imgData[inidx];
        }
        result /= n;
    } else {
        for (i = 0; i < n; i++) {
            inidx = r * inimg[i].pitchBytes + c;
            // 第 i 幅图对应权重数组中第 i % weightLength 个权重值。
            result += (float)inimg[i].imgMeta.imgData[inidx] *
                      weight[i % weightLength];
        }
        result /= sum[0];
    } 

    // 给输出图像赋值。
    outimg.imgMeta.imgData[outidx] = (unsigned char)result;
}

// Host 成员方法：imageOverlay（图像叠加）
__host__ int ImageOverlay::imageOverlay(Image *inimg[], int n, Image *outimg)
{
    // 错误码。
    int errcode;

    // 提取输入图像的 ROI 子图像的变量。
    ImageCuda *insubimgCud, *insubimgCudDev = NULL;

    // 检查 n 值是否有效。
    if (n < 1)
        return INVALID_DATA;

    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    for (int i = 0; i < n; i++)
        if (inimg[i] == NULL)
            return NULL_POINTER;

    // 检查图像是否为空。
    for (int i = 0; i < n; i++)
        if (inimg[i]->imgData == NULL)
            return UNMATCH_IMG;

    // 当 n == 1 时，可直接将输入图像拷贝到输出中。
    if (n == 1) {
        errcode = this->roiCopy.roiCopyAtDevice(inimg[0], outimg);
        if (errcode != NO_ERROR)
            return errcode;
        // 处理完毕，退出。
        return NO_ERROR;
    }

    // 输出图像 ROI 区域大小默认为第一幅图像的 ROI 区域大小。
    int outw = inimg[0]->roiX2 - inimg[0]->roiX1;
    int outh = inimg[0]->roiY2 - inimg[0]->roiY1;

    // n 幅输入图像的 ROI 区域大小不一样，那么将输出图像的 ROI 大小设定为所有
    // 图像的公共 ROI 区域。
    for (int i = 0; i < n; i++) {
        if (inimg[i]->roiX2 - inimg[i]->roiX1 < outw)
            outw = inimg[i]->roiX2 - inimg[i]->roiX1;
        if (inimg[i]->roiY2 - inimg[i]->roiY1 < outh)
            outh = inimg[i]->roiY2 - inimg[i]->roiY1;
    }

    // 检查输出图像是否为空，如果不为空则将输出图像的 roi 区域也考虑在内。
    if (outimg != NULL && outimg->imgData != NULL) {
        if (outimg->roiX2 - outimg->roiX1 < outw)
            outw = outimg->roiX2 - outimg->roiX1;
        if (outimg->roiY2 - outimg->roiY1 < outh)
            outh = outimg->roiY2 - outimg->roiY1;
    }

    // 如果所有图像的公共 ROI 区域长或宽小于 1，则报错返回。
    if (outw < 1 || outh < 1 )
        return INVALID_DATA;

    // 将输出图像复制到 device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和
        // 最小输入图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, outw, outh);
        // 如果创建图像操作失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入图像集合复制到 device 端。
    for (int i = 0; i < n; i++) {
        errcode = ImageBasicOp::copyToCurrentDevice(inimg[i]);
        if (errcode != NO_ERROR)
            return errcode;
    }

    insubimgCud = new ImageCuda[n];
    if (insubimgCud == NULL)
        return OUT_OF_MEM;

    // 对 n 幅输入图像分别进行提取子图像。
    for (int i = 0; i < n; i++) {
        errcode = ImageBasicOp::roiSubImage(inimg[i], &insubimgCud[i]);
        if (errcode != NO_ERROR) {
            delete [] insubimgCud;
            return errcode;
        }
    }

    // 为 insubimgCud 中 n 幅输入图像设置新的高度和宽度。
    for (int i = 0; i < n; i++) {
        insubimgCud[i].imgMeta.width = outw;
        insubimgCud[i].imgMeta.height = outh;
    }

    // 为 insubimgCudDev 分配内存空间。
    errcode = cudaMalloc((void **)&insubimgCudDev,
                         n * sizeof (ImageCuda));
    if (errcode != NO_ERROR) {
        delete [] insubimgCud;
        return errcode;
    }

    // 将 Host 上的 insubimgCud 拷贝到 Device 上。
    errcode = cudaMemcpy(insubimgCudDev, insubimgCud,
                         n * sizeof (ImageCuda), cudaMemcpyHostToDevice);
    delete [] insubimgCud;
    if (errcode != NO_ERROR) {
        // 释放之前申请的内存。
        cudaFree(insubimgCudDev);
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outw + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outh + blocksize.y - 1) / blocksize.y;

    // 调用 kernel 函数 _imageOverlayKer
    _imageOverlayKer<<<gridsize, blocksize>>>(
            insubimgCudDev, n, this->weight,this->weightLength, outsubimgCud);

    // 调用 cudaGetLastError 判断程序是否出错
    if (cudaGetLastError() != cudaSuccess) {
        // 释放之前申请的内存。
        cudaFree(insubimgCudDev);
        return CUDA_ERROR;
    }

    // 释放之前申请的内存。
    cudaFree(insubimgCudDev);

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：setValue（设置权重值）
__host__ int ImageOverlay::setValue(
        float *weight, int weightLength, bool onhostweight)
{
    // 局部变量，错误码。
    cudaError_t cudaerrcode;

    // 检查输入的权重数组 weight 数值是否为 NULL，若是则直接赋值并返回。
    if (weight == NULL) {
        // 检查成员的权重数组 weight 数值是否为 NULL，若不是则释放掉。
        if (this->weight != NULL)
            cudaFree(this->weight);
        this->weight = NULL;
        this->weightLength = 0;
        return NO_ERROR;
    }

    // 为 host 端权重申请内存，作为检验用。
    float *weighthost = new float[weightLength];
    // 出错则报错返回。
    if (weighthost == NULL)
        return OUT_OF_MEM;

    // 判断当前 weight 数组是否存储在 Host 端。
    if (!onhostweight) {
        // 将 Device 权重数组拷贝到 Host 上的内存空间。
        cudaerrcode = cudaMemcpy(weighthost, weight,
                                 weightLength * sizeof (float),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            delete [] weighthost;
            return cudaerrcode;
        }
    } else {
        cudaerrcode = cudaMemcpy(weighthost, weight,
                                 weightLength * sizeof (float),
                                 cudaMemcpyHostToHost);
        if (cudaerrcode != cudaSuccess) {
            // 释放之前申请的内存。
            delete [] weighthost;
            return cudaerrcode;
        }
    }

    // 检查权重数组 weighthost 数值是否为负数，若是则报错返回。
    for (int i = 0; i < weightLength; i++)
        if (weighthost[i] < 0.0f) {
            // 释放之前申请的内存。
            delete [] weighthost;
            return INVALID_DATA;
        }

    // 为 device 端权重申请内存,作为赋值用。
    float *weightdev;

    // 为 weightdev 分配内存空间。
    cudaerrcode = cudaMalloc((void **)&weightdev,
                             weightLength * sizeof (float));
    if (cudaerrcode != NO_ERROR) {
        // 释放之前申请的内存。
        delete [] weighthost;
        return cudaerrcode;
    }

    // 将 Host 上的 weighthost 拷贝到 Device 上。
    cudaerrcode = cudaMemcpy(weightdev, weighthost,
                             weightLength * sizeof (float),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != NO_ERROR) {
        // 释放之前申请的内存。
        delete [] weighthost;
        cudaFree(weightdev);
        return cudaerrcode;
    }

    // 检查成员的权重数组 weight 数值是否为 NULL，若不是则释放掉。
    if (this->weight != NULL)
        cudaFree(this->weight);
    this->weight = weightdev;
    this->weightLength = weightLength;

    // 释放之前申请的内存。
    delete [] weighthost;

    // 处理完毕，退出。
    return NO_ERROR;
}
