// BilateralFilter.cu
// 实现图像的双边滤波

#include "BilateralFilter.h"

#include <iostream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 纹理内存只能用于全局变量，使用全局存储时需要加入边界判断，经测试效率不及
// 纹理内存
static texture<unsigned char, 2, cudaReadModeElementType> _bilateralInimgTex;

// Host 函数：initTexture（初始化纹理内存）
// 将输入图像数据绑定到纹理内存
static __host__ int      // 返回值：若正确执行返回 NO_ERROR
_initTexture(
        Image *insubimg  // 输入图像
);

// Kernel 函数：_bilateralFilterKer（使用 ImageCuda 实现的双边滤波）
// 空域参数只影响高斯表，在调用该方法前初始化高斯表即可
static __global__ void         // kernel 函数无返回值
_bilateralFilterKer(
        ImageCuda outimg,     // 输出图像
        int radius,           // 双边滤波半径
        TemplateCuda gauCud,  // 高斯表
        TemplateCuda euCud    // 欧氏距离表
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

// Kernel 函数：_bilateralFilterKer（使用 ImageCuda 实现的双边滤波）
static __global__ void _bilateralFilterKer(
        ImageCuda outimg, int radius, TemplateCuda gauCud, TemplateCuda euCud)
{
    // 给定半径不在范围内时直接跳出
    if (radius <= 0|| radius > DEF_FILTER_RANGE)
        return;
    // 半径对应的高斯表数组的下标
    int gi = (2 * radius + 1) * (2 * radius + 1);

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
    
    // 邻域像素与参数乘积的累加值
    float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    // 存储参数的临时变量
    float factor;
    // 邻域参数的累加值
    float t[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // 获取当前处理点的像素值，即为中心点，取同一列的四个点
    unsigned char center[4];
    // 第一个中心点
    center[0] = tex2D(_bilateralInimgTex, dstc, dstr);
    // 第二个中心点，位于第一个中心点下方
    center[1] = tex2D(_bilateralInimgTex, dstc, dstr + 1);
    // 处于同列的第三个中心点
    center[2] = tex2D(_bilateralInimgTex, dstc, dstr + 2);
    // 处于同列的第四个中心点
    center[3] = tex2D(_bilateralInimgTex, dstc, dstr + 3);

    for (int col = 0; col <= gi; col++)
    {
        // 获取当前处理点的横纵坐标
        int i = gauCud.tplMeta.tplData[2 * col], 
            j = gauCud.tplMeta.tplData[2 * col + 1];
        // 获取当前处理点的像素值
        unsigned char curPix = tex2D(_bilateralInimgTex,
                                     dstc + j, dstr + i);
        // 计算当前点与中心点的像素差值
        unsigned char euindex = curPix > center[0] ? curPix - center[0] : 
                                 center[0] - curPix;
        // 欧氏距离与高斯距离的乘积
        factor =  gauCud.attachedData[col] * euCud.attachedData[euindex];
        t[0] += factor * curPix;
        sum[0] += factor;
        
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点不变，由于是位于循环体内部不可直接进行 ++ 运算，且当列超出时也
        // 不能进行 return，否则邻域扫描将终止，且输出图像不能赋值
        if (dstr + 1 >= outimg.imgMeta.height)
            continue;

        // 获取当前处理点的像素值
        curPix = tex2D(_bilateralInimgTex, dstc + j, dstr + i);
        // 计算当前点与中心点的像素差值
        euindex = curPix > center[1] ? curPix - center[1] : 
                  center[1] - curPix;
        // 欧氏距离与高斯距离的乘积
        factor =  gauCud.attachedData[col] * euCud.attachedData[euindex];
        t[1] += factor * curPix;
        sum[1] += factor;

        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点不变，由于是位于循环体内部不可直接进行 ++ 运算，且当列超出时也
        // 不能进行 return，否则邻域扫描将终止，且输出图像不能赋值
        if (dstr + 2 >= outimg.imgMeta.height)
            continue;
    
        // 获取当前处理点的像素值
        curPix = tex2D(_bilateralInimgTex, dstc + j, dstr + i);
        // 计算当前点与中心点的像素差值
        euindex = curPix > center[2] ? curPix - center[2] : 
                  center[2] - curPix;
        // 欧氏距离与高斯距离的乘积
        factor =  gauCud.attachedData[col] * euCud.attachedData[euindex];
        t[2] += factor * curPix;
        sum[2] += factor;
            
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点不变，由于是位于循环体内部不可直接进行 ++ 运算，且当列超出时也
        // 不能进行 return，否则邻域扫描将终止，且输出图像不能赋值
        if (dstr + 3 >= outimg.imgMeta.height)
            continue;

        // 获取当前处理点的像素值
        curPix = tex2D(_bilateralInimgTex, dstc + j, dstr + i);
        // 计算当前点与中心点的像素差值
        euindex = curPix > center[3] ? curPix - center[3] : 
                  center[3] - curPix;
        // 欧氏距离与高斯距离的乘积
        factor =  gauCud.attachedData[col] * euCud.attachedData[euindex];
        t[3] += factor * curPix;
        sum[3] += factor;
    }
    // 对第一列的点进行赋值
    outimg.imgMeta.imgData[dstidx] = (unsigned char)(t[0] / sum[0]);
    // 若列超出范围，此处可直接使用 return 直接跳出
    if (++dstr >= outimg.imgMeta.height)
        return;
    // 将对应数据的下标加一行
    dstidx += outimg.pitchBytes;
    // 对第二列的点进行赋值
    outimg.imgMeta.imgData[dstidx] = (unsigned char)(t[1] / sum[1]);
    // 准备处理第三列
    if (++dstr >= outimg.imgMeta.height)
        return;
    dstidx += outimg.pitchBytes;
    outimg.imgMeta.imgData[dstidx] = (unsigned char)(t[2] / sum[2]);
    // 处理第四列
    if (++dstr >= outimg.imgMeta.height)
        return;
    dstidx += outimg.pitchBytes;
    outimg.imgMeta.imgData[dstidx] = (unsigned char)(t[3] / sum[3]);
}

// Host 成员方法：doFilter（执行滤波）
__host__ int BilateralFilter::doFilter(Image *inoutimg)
{
    // 给定半径不在范围内时直接跳出 
    if (radius <= 0 && radius > DEF_FILTER_RANGE)
        return INVALID_DATA;
    // 若滤波的重复次数为 0 ，则不进行滤波返回正确执行
    if (repeat <= 0)
        return INVALID_DATA;
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inoutimg == NULL)
        return NULL_POINTER;
    // 检查模板数据
    if (gaussian == NULL || euclid == NULL)
        return INVALID_DATA;
    
    int errcode;  // 局部变量，错误码
    // 初始化纹理内存，将输入图像与之绑定
    _initTexture(inoutimg);

    // 将高斯模板数据拷贝至 Device 端避免核函数中无法访问
    errcode = TemplateBasicOp::copyToCurrentDevice(gaussian);
    if (errcode != NO_ERROR)
        return errcode;

    // 将欧式距离模板数据拷贝至 Device 端避免核函数中无法访问
    errcode = TemplateBasicOp::copyToCurrentDevice(euclid);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输入图像的 ROI 子图像。
    ImageCuda inoutsubimgCud;
    errcode = ImageBasicOp::roiSubImage(inoutimg, &inoutsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (inoutsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (inoutsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);
    
    // 进行重复滤波以提高质量 
    for (int i = 0; i < repeat; i++) {
        // 调用核函数进行滤波
        _bilateralFilterKer<<<gridsize, blocksize>>>(
                inoutsubimgCud, radius, *TEMPLATE_CUDA(gaussian), 
                *TEMPLATE_CUDA(euclid));
    }
    
    return NO_ERROR;
}

// Host 成员方法：doFilter（执行滤波）
__host__ int BilateralFilter::doFilter(Image *inimg, Image *outimg)
{
    // 给定半径不在范围内时直接跳出 
    if (radius <= 0 && radius > DEF_FILTER_RANGE)
        return INVALID_DATA;
    // 若滤波的重复次数为 0 ，则不进行滤波返回正确执行
    if (repeat <= 0)
        return INVALID_DATA;
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;
    // 检查模板数据
    if (gaussian == NULL || euclid == NULL)
        return INVALID_DATA;

    int errcode;  // 局部变量，错误码
    
    // 初始化纹理内存，将输入图像与之绑定，需将第一次运行结果保存至 outimg ，
    // 之后的重复则相当于在 outimg 上的 inplace 版本，这样保证了 inimg 中的数据
    // 一致性
    _initTexture(inimg);
    
    // 将高斯模板数据拷贝至 Device 端避免核函数中无法访问
    errcode = TemplateBasicOp::copyToCurrentDevice(gaussian);
    if (errcode != NO_ERROR)
        return errcode;

    // 将欧式距离模板数据拷贝至 Device 端避免核函数中无法访问
    errcode = TemplateBasicOp::copyToCurrentDevice(euclid);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->roiX2 - inimg->roiX1, 
                inimg->roiY2 - inimg->roiY1);
        if (errcode != NO_ERROR)
            return errcode;
    }
        
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
    
    // 调用核函数进行滤波
    _bilateralFilterKer<<<gridsize, blocksize>>>(
            outsubimgCud, radius, *TEMPLATE_CUDA(gaussian), 
            *TEMPLATE_CUDA(euclid));

    // 进行重复滤波以提高质量 
    for (int i = 1; i < repeat; i++) {
        // 调用核函数进行滤波
        _bilateralFilterKer<<<gridsize, blocksize>>>(
                outsubimgCud, radius, *TEMPLATE_CUDA(gaussian), 
                *TEMPLATE_CUDA(euclid));
    }
    
    return NO_ERROR;
}

