// ImageToText.cu
// 实现图像转文本的算法

#include "ImageToText.h"

#include "Image.h"
#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_imageToTextKer（图像转文本）
// 将输入的灰度图像转成制定大小的文本，文本中用一个字符代表特定的灰度级。首先将
// 原图缩放到和文本同样大小，然后按照灰度级对应找到文本，写入字符串中。
static __global__ void      // 无返回值
_imageToTextKer(
        ImageCuda inimg,    // 输入图像
        char *outstr,       // 输出字符串
        char *ascii,        // 用于存储图像的各个像素点转换成字符的标准。
        unsigned int level  // 一个 ASCII 码代表的灰度值的个数，默认值为 8
); 

// Kernel 函数：_imageToTextKer（图像转文本）
static __global__ void _imageToTextKer(ImageCuda inimg, char *outstr,
                                       char *ascii, unsigned int level)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于同一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 判断 c 和 r 是否超过了文本的尺寸
    if (c > inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算将像素点转化为字符时将其存储在数组 outstr 中对应的下标，由于在每行像
    // 素点转化后的字符后面要添加一个换行符，所以使用 inimg.imgMeta.width + 1
    int outidx = r * (inimg.imgMeta.width + 1) + c;

    // 计算正在处理的像素点对应的图像数据的数组下标。
    int inidx = r * inimg.pitchBytes + c;

    // 处理第一个点，若正在处理的点位于行尾，则添加换行符，若位于文本尾部，
    // 则添加终止符号，否则用 ASCII 码表示灰度值
    outstr[outidx] = ascii[inimg.imgMeta.imgData[inidx] / level];

    // 处理统一列上剩下的三个点   
    for (int i = 1; i < 4; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查
        if (r++ >= inimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输入图像的下标和输出字符串的
        // 下标。由于只有 r 分量增加 1，所以下标 inidx 只需要加上一个
        // pitchBytes，下标 outidx 只需要加上一个 inimg.imgMeta.width + 1 即可，
        // 不需要再进行乘法计算。
        inidx += inimg.pitchBytes;
        outidx += inimg.imgMeta.width + 1;

        // 若正在处理的点位于行尾，则添加换行符，若位于文本尾部，则添加终止
        // 符号，否则用 ASCII 码表示灰度值
        outstr[outidx] = ascii[inimg.imgMeta.imgData[inidx] / level];
    }
}

// Host 成员方法：imageToText（图像转文本）
__host__ int ImageToText::imageToText(Image *inimg, char *outstr,
                                      size_t width, size_t height,
                                      bool onhostarray)
{
    // 检查输入图像和输出字符串 outstr 是否为 NULL
    if (inimg == NULL || outstr == NULL)
        return NULL_POINTER;

    // 检查图像是否为空
    if (inimg->imgData == NULL)
        return UNMATCH_IMG;

    // 将处理后的图像复制到 device
    int errcode;  // 局部变量，错误码
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    ImageCuda insubimgCud;  // 存储输入图像的 ROI 子图像。

    // 将输入图像的 ROI 区域大小与文本大小进行比较，若不相同则调用图像拉伸算法
    // 进行缩放。
    if (inimg->width != width || inimg->height != height) {
        Image * outimg;
        errcode = ImageBasicOp::newImage(&outimg);
        if (errcode != NO_ERROR)
            return errcode;
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, width, height);
        if (errcode != NO_ERROR) {
            // 释放申请的空间，防止内存泄露
            ImageBasicOp::deleteImage(outimg);
            return errcode;
        }

        // 设置缩放的倍数。
	    this->imageStretch.setTimesWidth((float)width / inimg->width);
	    this->imageStretch.setTimesHeight((float)height / inimg->height);

        // 调用 performImgStretch 函数对 inimg 进行缩放，存放到 outimg 中。
        errcode = this->imageStretch.performImgStretch(inimg, outimg);
        if (errcode != NO_ERROR) {
            // 释放申请的空间，防止内存泄露
            ImageBasicOp::deleteImage(outimg);
            return errcode;
        }

        // 提取输入图像的 ROI 子图像。
        errcode = ImageBasicOp::roiSubImage(outimg, &insubimgCud);
        if (errcode != NO_ERROR)
            return errcode;
    } else {
        // 提取输入图像的 ROI 子图像。
        errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
        if (errcode != NO_ERROR)
           return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y * 4 - 1) / (blocksize.y * 4);

    // 局部变量，错误码
    cudaError_t cuerrcode;

    // 判断当前 outstr 数组是否存储在 Host 端。若是，则需要在 Device 端为文本
    // 申请一段空间，否则直接调用核函数。
    if (onhostarray) {
        // 在 Device 上分配存储临时输出字符串的空间。
        char *outstrDev;
        cuerrcode = cudaMalloc((void **)&outstrDev,
                               (width + 1) * height * sizeof (char));
        if (cuerrcode != cudaSuccess) {
            cudaFree(outstrDev);
            return CUDA_ERROR;
        }

        // 调用 kernel 函数 _imageToTextKer
        _imageToTextKer<<<gridsize, blocksize>>>(insubimgCud, outstrDev,
                                                 this->ascii, this->level);

        // 调用 cudaGetLastError 判断程序是否出错
        cuerrcode = cudaGetLastError();
        if (cuerrcode != cudaSuccess) {
            cudaFree(outstrDev);
            return CUDA_ERROR;
        }

        // 将 Device 上的 outstr 拷贝回 Host 上
        cuerrcode = cudaMemcpy(outstr, outstrDev,
                               (width + 1) * height * sizeof (char),
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            cudaFree(outstrDev);
            return CUDA_ERROR;
        }

        // 为 outstr 添加换行符和终止符号
        for (int i = 0; i < height - 1; i++)
            outstr[(width + 1) * i + width] = '\n';
        outstr[(width + 1) * height - 1] = '\0';

        // 释放在GPU上分配的内存，避免内存泄露
        cudaFree(outstrDev);
    } else {
        // 调用 kernel 函数 _imageToTextKer
        _imageToTextKer<<<gridsize, blocksize>>>(insubimgCud, outstr,
                                                 this->ascii, this->level);
        cuerrcode = cudaGetLastError();
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 在 Host 端分配存储临时输出字符串的空间。
        char *outstrHost = new char[(width + 1) * height];

        // 将 Device 端的 outstr 拷贝到 Host 端
        cuerrcode = cudaMemcpy(outstrHost, outstr,
                               (width + 1) * height * sizeof (char),
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            cudaFree(outstrHost);
            return CUDA_ERROR;
        }

        // 为 outstrHost 添加换行符和终止符号
        for (int i = 0; i < height - 1; i++)
            outstrHost[(width + 1) * i + width] = '\n';
        outstrHost[(width + 1) * height - 1] = '\0';

        // 将 Host 端的 outstrHost 拷贝回 Device 端
        cuerrcode = cudaMemcpy(outstr, outstrHost,
                               (width + 1) * height * sizeof (char),
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            cudaFree(outstrHost);
            return CUDA_ERROR;
        }
    }

    // 处理完毕，退出
    return NO_ERROR;    
}

