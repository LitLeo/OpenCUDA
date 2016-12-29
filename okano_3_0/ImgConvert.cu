// ImgConvert.cu
// 实现图像与坐标记得互相转化。

#include "ImgConvert.h"
#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X    32
#define DEF_BLOCK_Y     8

// 宏：DEF_BLOCK_1D
// 定义一维线程块尺寸。
#define DEF_BLOCK_1D  512

// Kernel 函数：_cstConvertImgKer（实现将坐标集转化为图像算法）
// 将坐标集内的坐标映射到输入图像中并将目标点置为 highpixel 从而实现
// 坐标集与图像的转化。
static __global__ void           // Kernel 函数无返回值。
_cstConvertImgKer(
        CoordiSet incst,         // 输入坐标集
        ImageCuda inimg,         // 输入图像 
        unsigned char highpixel  // 高像素
);

// Kernel 函数：_markImageFlagKer（将图像转化为标志数组）
// 将图像转化为标志数组，其中若像素需要记录到坐标集时，对应的标志位为 1，否则为
// 0。
static __global__ void     // Kernel 函数无返回值。
_markImageFlagKer(
        ImageCuda inimg,   // 输入图像
        int imgflag[],     // 输出的图像标志位数组。
        ImgConvert imgcvt  // 转换算法 CLASS，主要使用其中的转换标志位。
);

// Kernel 函数：_arrangeCstKer（重组坐标点集）
// 在计算处图像标志位数组和对应的累加数组之后，将图像的信息转换为坐标点集的信息
// 写入输出坐标点集中。
static __global__ void        // Kernel 函数无返回值。
_arrangeCstKer(
        ImageCuda inimg,      // 输入图像
        int imgflag[],        // 图像标志位数组
        int imgacc[],         // 图像标志位数组的累加数组
        int ptscnt,           // 有效坐标点的数量
        CoordiSetCuda outcst  // 输出坐标集
);

// Kernel 函数：_curConvertImgKer（实现将曲线转化为图像算法）
// 将曲线内的坐标映射到输入图像中并将目标点置为 highpixel 从而实现
// 坐标集与图像的转化。
static __global__ void           // Kernel 函数无返回值。
_curConvertImgKer(
        Curve incur,             // 输入曲线
        ImageCuda inimg,         // 输入图像 
        unsigned char highpixel  // 高像素
);

// Kernel 函数：_cstConvertImgKer（实现将坐标集转化为图像算法）
static __global__ void _cstConvertImgKer(CoordiSet incst, ImageCuda inimg,
                                         unsigned char highpixel)
{
    // index 表示线程处理的像素点的坐标。
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查坐标点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (index >= incst.count)
        return;
    
    // 获得目标点在图像中的对应位置。
    int curpos = incst.tplData[2 * index + 1] * inimg.pitchBytes + 
                 incst.tplData[2 * index];  
    
    // 将坐标集中坐标在图像中对应的像素点的像素值置为 higpixel。
    inimg.imgMeta.imgData[curpos] = highpixel;
}

// 成员方法：cstConvertToImg（坐标集转化为图像算法）
__host__ int ImgConvert::cstConvertToImg(CoordiSet *incst, Image *outimg)
{
    // 局部变量，错误码。
    int errcode;

    // 检查输入坐标集，输出图像是否为空。
    if (incst == NULL || outimg == NULL)
        return NULL_POINTER;

    // 将输出图像拷贝到 device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 将输入坐标集拷贝到 device 端。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(incst);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 初始化输入图像内所有的像素点的的像素值为 lowpixel，为转化做准备。
    cudaError_t cuerrcode;
    cuerrcode = cudaMemset(outsubimgCud.imgMeta.imgData, this->lowPixel, 
                           sizeof(unsigned char) * outsubimgCud.imgMeta.width * 
                           outsubimgCud.imgMeta.height);  
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;


    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。调用一维核函数，
    // 在这里设置线程块内的线程数为 256，用 DEF_BLOCK_1D 表示。
    size_t blocksize1, gridsize1;
    blocksize1 = DEF_BLOCK_1D;
    gridsize1 = (incst->count + blocksize1 - 1) / blocksize1;

    // 将输入坐标集转化为输入图像图像，即将坐标集内点映射在图像上点的
    // 像素值置为 highpixel。
    _cstConvertImgKer<<<gridsize1, blocksize1>>>(*incst, outsubimgCud,
                                                 this->highPixel);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Kernel 函数：_markImageFlagKer（将图像转化为标志数组）
static __global__ void _markImageFlagKer(ImageCuda inimg, int imgflag[], 
                                         ImgConvert imgcvt)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * inimg.imgMeta.width + c;
    // 根据输入像素下标读取输入像素值。
    unsigned char curpixel = inimg.imgMeta.imgData[inidx];

    // 如果当前像素点为有效像素点，则图像标志位置位，否则置零。
    if (imgcvt.getConvertFlag(curpixel)) {
        imgflag[outidx] = 1;
    } else {
        imgflag[outidx] = 0;
    }

    // 处理余下的三个点。
    for (int i = 1; i < 4; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= inimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += inimg.imgMeta.width;
        curpixel = inimg.imgMeta.imgData[inidx];

        // 如果当前像素点为有效像素点，则图像标志位置位，否则置零。
        if (imgcvt.getConvertFlag(curpixel)) {
            imgflag[outidx] = 1;
        } else {
            imgflag[outidx] = 0;
        }
    }
}

// Kernel 函数：_arrangeCstKer（重组坐标点集）
static __global__ void _arrangeCstKer(
        ImageCuda inimg, int imgflag[], int imgacc[], 
        int ptscnt, CoordiSetCuda outcst)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;


    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算当前 Thread 所对应的原图像数据下标和图像标志位数组的下标。
    int inidx = r * inimg.pitchBytes + c;
    int flagidx = r * inimg.imgMeta.width + c;

    // 如果当前计算的像素点是一个无效像素点，则直接退出。
    if (imgflag[flagidx] == 0)
        return;

    // 获取当前像素点所对应的坐标集中的下标。
    int outidx = imgacc[flagidx];

    // 计算坐标点集中的附属数据，这里直接使用其像素值所对应的浮点亮度值来作为附
    // 属数据。
    float curval = inimg.imgMeta.imgData[inidx] / 255.0f;

    // 完成输出操作。这里使用 while 的原因是，如果坐标点集可容纳的坐标点数量超
    // 过了实际的坐标点数量，这样需要重复的防止现有的坐标点（这样做的考虑时为了
    // 防止防止了不是有效像素点的坐标点，导致凸壳等算法出错。
    while (outidx < outcst.tplMeta.count) {
        // 将有效像素点存放到坐标点集中，顺带输出了附属数据。
        outcst.tplMeta.tplData[2 * outidx] = c;
        outcst.tplMeta.tplData[2 * outidx + 1] = r;
        outcst.attachedData[outidx] = curval;

        // 更新输出下标，如果输出坐标点集容量过大，则循环输出有效坐标点。
        outidx += ptscnt;
    }
}

// 宏：FAIL_IMGCONVERTTOCST_FREE
// 当下面函数运行出错时，使用该宏清除内存，防止内存泄漏。
#define FAIL_IMGCONVERTTOCST_FREE  do {  \
        if (tmpdata != NULL)             \
            cudaFree(tmpdata);           \
    } while (0)

// 成员方法：imgConvertToCst（图像转化成坐标集算法）
__host__ int ImgConvert::imgConvertToCst(Image *inimg, CoordiSet *outcst)
{
    // 检查输入图像，输出坐标集是否为空。
    if (inimg == NULL || outcst == NULL)
        return NULL_POINTER;

    // 局部变量，错误码。
    int errcode;
    cudaError_t cuerrcode;

    // 定义加法运算类型
    add_class<int> add;

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算数据尺寸。
    int flagsize = inimg->width * inimg->height;
    int datasize = (2 * flagsize + 1) * sizeof (int);
    // 申请存放图像标志位数组何其对应的累加数组的内存空间。
    int *tmpdata = NULL;
    cuerrcode = cudaMalloc((void **)&tmpdata, datasize);
    if (cuerrcode != cudaSuccess) {
        FAIL_IMGCONVERTTOCST_FREE;
        return CUDA_ERROR;
    }

    // 将申请的临时内存空间分配给各个指针。
    int *imgflagDev = tmpdata;
    int *imgaccDev = imgflagDev + flagsize;

    // 提取输入图像对应的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用第一个 Kernel 所需要的线程块尺寸。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y * 4 - 1) / 
                 (blocksize.y * 4);

    // 调用第一个 Kernel 生成图像标志位数组。
    _markImageFlagKer<<<gridsize, blocksize>>>(insubimgCud, imgflagDev, *this);
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_IMGCONVERTTOCST_FREE;
        return CUDA_ERROR;
    }

    // 通过扫描算法计算图像标志位数组对应的累加数组。
    errcode = this->aryScan.scanArrayExclusive(
            imgflagDev, imgaccDev, flagsize, add, false, false, false);
    if (errcode != NO_ERROR) {
        FAIL_IMGCONVERTTOCST_FREE;
        return errcode;
    }
    
    // 将累加数组中最后一个元素拷贝到 Host，该数据表示整个图像中有效像素点的数
    // 量。
    int ptscnt;
    cuerrcode = cudaMemcpy(&ptscnt, &imgaccDev[flagsize], sizeof (int), 
                           cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_IMGCONVERTTOCST_FREE;
        return CUDA_ERROR;
    }

    // 将输出坐标点集拷贝入 Device 内存。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(outcst);
    if (errcode != NO_ERROR) {
        // 如果输出坐标点击无数据（故上面的拷贝函数会失败），则会创建一个和有效
        // 像素点等量的坐标点集。
        errcode = CoordiSetBasicOp::makeAtCurrentDevice(outcst, ptscnt);
        // 如果创建坐标点集也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR) {
            FAIL_IMGCONVERTTOCST_FREE;
            return errcode;
        }
    }

    // 获取输出坐标点集对应的 CUDA 型数据。
    CoordiSetCuda *outcstCud = COORDISET_CUDA(outcst);

    // 计算调用第二个 Kernel 所需要的线程块尺寸。
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 调用第二个 Kernel 得到输出坐标点集。
    _arrangeCstKer<<<gridsize, blocksize>>>(
            insubimgCud, imgflagDev, imgaccDev, ptscnt, *outcstCud);
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_IMGCONVERTTOCST_FREE;
        return CUDA_ERROR;
    }

    // 释放临时内存空间。
    cudaFree(tmpdata);

    // 处理完毕退出。
    return NO_ERROR;
}

// Kernel 函数：_curConvertImgKer（实现将曲线转化为图像算法）
static __global__ void _curConvertImgKer(Curve incur, ImageCuda inimg, 
                                         unsigned char highpixel)
{
    // index 表示线程处理的像素点的坐标。
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查坐标点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (index >= incur.curveLength)
        return;
    
    // 获得目标点在图像中的对应位置。
    int curpos = incur.crvData[2 * index + 1] * inimg.pitchBytes + 
                 incur.crvData[2 * index];  
    
    // 将曲线中坐标在图像中对应的像素点的像素值置为 higpixel。
    inimg.imgMeta.imgData[curpos] = highpixel;
}

// Host 成员方法：curConvertToImg（曲线转化为图像算法）
__host__ int ImgConvert::curConvertToImg(Curve *incur, Image *outimg)
{
    // 局部变量，错误码。
    int errcode;

    // 检查输入曲线，输出图像是否为空。
    if (incur == NULL || outimg == NULL)
        return NULL_POINTER;

    // 将输出图像拷贝到 device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 将输入曲线拷贝到 device 端。
    errcode = CurveBasicOp::copyToCurrentDevice(incur);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 初始化输入图像内所有的像素点的的像素值为 lowpixel，为转化做准备。
    cudaError_t cuerrcode;
    cuerrcode = cudaMemset(outsubimgCud.imgMeta.imgData, this->lowPixel, 
                           sizeof(unsigned char) * outsubimgCud.imgMeta.width * 
                           outsubimgCud.imgMeta.height);  
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。调用一维核函数，
    // 在这里设置线程块内的线程数为 256，用 DEF_BLOCK_1D 表示。
    size_t blocksize1, gridsize1;
    blocksize1 = DEF_BLOCK_1D;
    gridsize1 = (incur->curveLength + blocksize1 - 1) / blocksize1;

    // 将输入曲线转化为输入图像图像，即将曲线内点映射在图像上点的
    // 像素值置为 highpixel。
    _curConvertImgKer<<<gridsize1, blocksize1>>>(*incur, outsubimgCud,
                                                 this->highPixel);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}