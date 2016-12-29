// IsolatedPoints.cu
// 实现图像的孤立点检测

#include "IsolatedPoints.h"

#include <iostream>
#include <cmath>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：DEF_TPL_SIZE_X 和 DEF_TPL_SIZE_Y
// 定义了模板的横向和纵向尺寸
#define DEF_TPL_SIZE_X  3
#define DEF_TPL_SIZE_Y  3

// 宏：DEF_TPL_OFFSET_X 和 DEF_TPL_OFFSET_Y
// 定义了模板的横向和纵向偏移
#define DEF_TPL_OFFSET_X  (DEF_TPL_SIZE_X / 2)
#define DEF_TPL_OFFSET_Y  (DEF_TPL_SIZE_Y / 2)

// 宏：DEF_TPL_DATA_SIZE
// 定义了模板的大小
#define DEF_TPL_DATA_SIZE  (DEF_TPL_SIZE_X * DEF_TPL_SIZE_Y)

// 宏：DEF_TPL_POINT_CNT
// 定义了模板邻域的大小
#define DEF_TPL_POINT_CNT  (DEF_TPL_DATA_SIZE - 1)

// Device 全局常量：_laplasDev（拉普拉斯模板权重）
// 拉普拉斯模板为 3 * 3 模板，具体权重比例为中心为 -8，四周为 1。
// 应用此模板可以使孤立点更加突出明显。
static __device__ int _laplasDev[DEF_TPL_SIZE_Y][DEF_TPL_SIZE_X] = {
    { 1,   1,  1 },
    { 1,  -8,  1 },
    { 1,   1,  1 }
};

// Kernel 函数：_isopointDetectKer（孤立点检测）
// 根据给定的模版对图像进行孤立点检测。
static __global__ void           // Kernel 函数无返回值
_isopointDetectKer(
        ImageCuda inimg,         // 输入图像
        ImageCuda outimg,        // 输出图像
        unsigned char threshold  // 灰度差值
);

// Kernel 函数: _isopointDetectKer（孤立点检测）
static __global__ void _isopointDetectKer(ImageCuda inimg, ImageCuda outimg,
                                          unsigned char threshold)
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

    // 计算第一个输入坐标点对应的图像数据指针位置。
    unsigned char *curinptr = inimg.imgMeta.imgData + r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据指针位置。
    unsigned char *curoutptr = outimg.imgMeta.imgData + 
                               r * outimg.pitchBytes + c;
    // 因为一个线程处理四个点，这里定义一个长度为 4 的数组用来保存滤波后各点值
    int total[4] = { 0, 0, 0, 0 };
    // 计算邻域合法点个数变量，初始化为 -1
    int lepoint[4] = { -1, -1, -1, -1 };
    // 根据图像点位置不同，阈值判断有差异，此为实际阈值，定义长度为 4 的数组
    int curthres[4];
    // 记录输入图像处理模板像素点位置
    unsigned char *curtplptr;
    // 记录图像邻域点横纵坐标
    int inx, iny;

    // 进行滤波迭代运算
    for (int dy = -DEF_TPL_OFFSET_Y; dy <= DEF_TPL_OFFSET_Y; dy++) {
        for (int dx = -DEF_TPL_OFFSET_X; dx <= DEF_TPL_OFFSET_X; dx++) {
            // 计算对应模板数组中实际横坐标
            int lapx = dx + DEF_TPL_OFFSET_X;
            // 计算对应模板数组中实际纵坐标
            int lapy = dy + DEF_TPL_OFFSET_Y; 
            // 计算图像邻域点横坐标
            inx = c + dx;
            // 计算图像邻域点纵坐标
            iny = r + dy;
            // 判断是否越界
            if (inx < 0 && inx >= inimg.imgMeta.width)
                continue;
            // 计算输入图像第一个模板点真实坐标
            curtplptr = inimg.imgMeta.imgData + inx + 
                        iny * inimg.pitchBytes;
            if (iny >= 0 && iny < inimg.imgMeta.height) {
                // 合法点数加一
                lepoint[0]++;
                // 计算滤波迭代值            
                total[0] += _laplasDev[lapy][lapx] * (*curtplptr);
            }

            // 一个线程处理四个像素点.
            // 处理剩下的三个像素点。
            for (int i = 1; i < 4; i++) {
                // 获取当前列的下一行的像素的位置
                curtplptr += inimg.pitchBytes;
                // 使 iny 加一，得到当前要处理的像素的 y 分量
                iny++;
                // 判断是否越界，若越界，则继续处理下一个模板点
                if (iny >= 0 && iny < inimg.imgMeta.height) {
                    // 合法点数加一
                    lepoint[i]++;
                    // 计算滤波迭代值            
                    total[i] += _laplasDev[lapy][lapx] * (*curtplptr);
                }
            } 
        }
    }

    // total[0] 减去不合法邻域点的权重
    total[0] += (DEF_TPL_POINT_CNT - lepoint[0]) * (*curinptr);
    // 根据点位置不同计算实际阈值
    curthres[0] = threshold * lepoint[0];
    // 通过对滤波后的图像进行二值化处理实现孤立点检测
    // 线程中处理的第一个点。       
    // 滤波后值大于阈值设为白色，小于阈值设为黑色
    *curoutptr = (abs(total[0]) >= curthres[0]) ? 255 : 0;

    // 一个线程处理四个像素点.
    // 处理剩下的三个像素点，为输出图像对应点赋值
    for (int i = 1; i < 4; i++) {
        // 先判断 y 分量是否越界,如果越界，则可以确定后面的点也会越界，所以
        // 直接返回
        if (++r >= outimg.imgMeta.height) 
            return; 

        // 获取当前输出列的下一行的位置指针
        curoutptr += outimg.pitchBytes;
        // 获取当前输入列的下一行的位置指针
        curinptr += inimg.pitchBytes;
        // total[i] 减去不合法邻域点的权重
        total[i] += (DEF_TPL_POINT_CNT - lepoint[i]) * (*curinptr);
        // 根据点位置不同计算实际阈值
        curthres[i] = threshold * lepoint[i];
            
        // 一个线程处理四个像素点。
        // 通过对滤波后的图像进行二值化处理实现孤立点检测
        // 线程中处理的后三个点。
        // 滤波后值大于阈值设为白色，小于阈值设为黑色
        *curoutptr = (abs(total[i]) >= curthres[i]) ? 255 : 0;
    }
}

// Host 成员方法：isopointDetect（孤立点检测）
__host__ int IsolatedPoints::isopointDetect(Image *inimg, Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;    

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

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

    // 调用核函数，根据模版进行孤立点检测。
    _isopointDetectKer<<<gridsize,blocksize>>>(insubimgCud, outsubimgCud,
                                               threshold);  
           
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

