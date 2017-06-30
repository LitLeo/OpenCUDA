// learningFilter.cu
// 实现图像的learning滤波操作

#include "LearningFilter.h"
#include "Matrix.h"
#include "LinearFilter.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_matToImg（类型转换）
// 将 MatrixCuda 数据写入到 ImageCuda 类之中
static __global__ void _matToImg(
        ImageCuda img,        // 目标 ImageCuda 类
        MatrixCuda matrixImg  // 目标 MatrixCuda 类
);

// Kernel 函数：_imgToMat（类型转换）
// 将 MatrixCuda 数据写入到 ImageCuda 类之中
static __global__ void _imgToMat(
        ImageCuda img,        // 目标 ImageCuda 类
        MatrixCuda matrixImg  // 目标 MatrixCuda 类
);

// Kernel 函数：_subtract（Matrix 减法）
// 矩阵 img、p 之间做减法，结果放在矩阵 out 中
static __global__ void _subtract(
        MatrixCuda img,  // 被减数
        MatrixCuda p,    // 减数
        MatrixCuda out   // 结果
);

// Kernel 函数：_add（Matrix 加法）
// 矩阵 img、p 之间做加法，结果放在矩阵 out 中
static __global__ void _add(
        MatrixCuda img,  // 被加数
        MatrixCuda p,    // 加数
        MatrixCuda out   // 结果
);

// Kernel 函数：_multiply（Matrix 乘法）
// 矩阵 img、p 之间做乘法，结果放在矩阵 out 中
static __global__ void _multiply(
        MatrixCuda img,  // 被乘数
        MatrixCuda p,    // 乘数
        MatrixCuda out   // 结果
);

// Kernel 函数：_divide（Matrix 除法）
// 矩阵 img、p 之间做除法，结果放在矩阵 out 中
static __global__ void _divide(
        MatrixCuda img,  // 被除数
        MatrixCuda p,    // 除数
        MatrixCuda out   // 结果
);

// Kernel 函数：_addWeighted 
// 矩阵 img 的每个元素增加 eps 大小的值
static __global__ void _addWeighted(
        MatrixCuda img,  // 目标矩阵
        float eps,       // 调整值
        MatrixCuda out   // 输出值
);

// Kernel 函数：_linearFilter（归一化滤波）
// 根据率比半径进行滤波计算
static __global__ void _linearFilter(
        MatrixCuda in,  // 输入图像 
        MatrixCuda out, // 输出图像
        float ra        // 滤波半径
);

// Kernel 函数：_beTop（像素置白）
// 通过操作使图片的像素全都变成255
static __global__ void _beTop(
        MatrixCuda img  // 进行操作的图片 
);

static __global__ void _imgToMat(ImageCuda img, MatrixCuda matrixImg){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= img.imgMeta.width || r >= img.imgMeta.height)
        return;
    int inidx = r * img.pitchBytes + c;
	int indexM = r * matrixImg.pitchWords + c;

    matrixImg.matMeta.matData[indexM] = (float)(img.imgMeta.imgData)[inidx];

    for(int i = 1; i <= 3; i++){
        if (++r >= img.imgMeta.height)
            return;
		inidx = r * img.pitchBytes + c;
		indexM = r * matrixImg.pitchWords + c;
        matrixImg.matMeta.matData[indexM] = (float)(img.imgMeta.imgData)[inidx];
    }
}

// 从矩阵写回到图像
static __global__ void _matToImg(ImageCuda img, MatrixCuda matrixImg){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= img.imgMeta.width || r >= img.imgMeta.height)
        return;

    int inidx = r * img.pitchBytes + c;
    int indexM = r * matrixImg.pitchWords + c;

    (img.imgMeta.imgData)[inidx] = (int)matrixImg.matMeta.matData[indexM];

    for(int i = 1; i <= 3; i++){
        if (++r >= img.imgMeta.height)
            return;
        inidx = r * img.pitchBytes + c;
        indexM = r * matrixImg.pitchWords + c;
        (img.imgMeta.imgData)[inidx] = (int)matrixImg.matMeta.matData[indexM];
    }
}

static __global__ void _subtract(MatrixCuda img, MatrixCuda p, MatrixCuda out){
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= img.matMeta.width || r >= img.matMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx1 = r * img.pitchWords + c;
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx2 = r * p.pitchWords + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * out.pitchWords + c;


    // 关于float的大小判断需要重新写，写成减法形式
    out.matMeta.matData[outidx] = img.matMeta.matData[inidx1] - p.matMeta.matData[inidx2];
    if(out.matMeta.matData[outidx] < 0){
        out.matMeta.matData[outidx] = 0;
    }
    
   // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= out.matMeta.height)
            return;

        inidx1 = r * img.pitchWords + c;
        inidx2 = r * p.pitchWords + c;
        outidx = r * out.pitchWords + c;
        
        out.matMeta.matData[outidx] = img.matMeta.matData[inidx1] - p.matMeta.matData[inidx2];
        if(out.matMeta.matData[outidx] < 0){
            out.matMeta.matData[outidx] = 0;
        }
    }
	
}

static __global__ void _add(MatrixCuda img, MatrixCuda p, MatrixCuda out){
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= img.matMeta.width || r >= img.matMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx1 = r * img.pitchWords + c;
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx2 = r * p.pitchWords + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * out.pitchWords + c;


    // 关于float的大小判断需要重新写，写成减法形式
    out.matMeta.matData[outidx] = img.matMeta.matData[inidx1] + p.matMeta.matData[inidx2];
    if(out.matMeta.matData[outidx] > 255){
        out.matMeta.matData[outidx] = 255;
    }

   // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= out.matMeta.height)
            return;

        inidx1 = r * img.pitchWords + c;
        inidx2 = r * p.pitchWords + c;
        outidx = r * out.pitchWords + c;
        
        out.matMeta.matData[outidx] = img.matMeta.matData[inidx1] + p.matMeta.matData[inidx2];
        if(out.matMeta.matData[outidx] > 255){
            out.matMeta.matData[outidx] = 255;
        }
    }
	
}

static __global__ void _multiply(MatrixCuda img, MatrixCuda p, MatrixCuda out){
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= img.matMeta.width || r >= img.matMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx1 = r * img.pitchWords + c;
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx2 = r * p.pitchWords + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * out.pitchWords + c;

    float ratio1, ratio2, ratioO;

    ratio1 = img.matMeta.matData[inidx1] / 255;
    ratio2 = p.matMeta.matData[inidx2] / 255;

    // 关于float的大小判断需要重新写，写成减法形式
    ratioO = ratio1 * ratio2;
    out.matMeta.matData[outidx] = 255 * ratioO;

   // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= out.matMeta.height)
            return;

        inidx1 = r * img.pitchWords + c;
        inidx2 = r * p.pitchWords + c;
        outidx = r * out.pitchWords + c;
        
        ratio1 = img.matMeta.matData[inidx1] / 255;
        ratio2 = p.matMeta.matData[inidx2] / 255;

        ratioO = ratio1 * ratio2;
        out.matMeta.matData[outidx] = 255 * ratioO;
    }
	
}

static __global__ void _divide(MatrixCuda img, MatrixCuda p, MatrixCuda out){
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= img.matMeta.width || r >= img.matMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx1 = r * img.pitchWords + c;
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx2 = r * p.pitchWords + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * out.pitchWords + c;

    float ratio1, ratio2, ratioO;

    ratio1 = img.matMeta.matData[inidx1] / 255;
    ratio2 = p.matMeta.matData[inidx2] / 255;
 
    // 关于float的大小判断需要重新写，写成减法形式
    if(ratio2 != 0.0)
        ratioO = ratio1 * ratio2;
    else
        ratioO = 0.0;
    out.matMeta.matData[outidx] = 255 * ratioO;

   // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= out.matMeta.height)
            return;

        inidx1 = r * img.pitchWords + c;
        inidx2 = r * p.pitchWords + c;
        outidx = r * out.pitchWords + c;
        
        ratio1 = img.matMeta.matData[inidx1] / 255;
        ratio2 = p.matMeta.matData[inidx2] / 255;

        if(ratio2 != 0.0)
            ratioO = ratio1 * ratio2;
        else
            ratioO = 0.0;
        out.matMeta.matData[outidx] = 255 * ratioO;
    }
	
}

static __global__ void _addWeighted(MatrixCuda img, float eps, MatrixCuda out){
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= img.matMeta.width || r >= img.matMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx1 = r * img.pitchWords + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * out.pitchWords + c;

    out.matMeta.matData[outidx] = out.matMeta.matData[inidx1] + eps;
    if(out.matMeta.matData[outidx] > 255)
        out.matMeta.matData[outidx] = 255;
    
   // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= out.matMeta.height)
            return;

        inidx1 = r * img.pitchWords + c;
        outidx = r * out.pitchWords + c;
        
        out.matMeta.matData[outidx] = out.matMeta.matData[inidx1] + eps;
        if(out.matMeta.matData[outidx] > 255)
            out.matMeta.matData[outidx] = 255;
    }
	
}

static __global__ void _linearFilter(MatrixCuda in, MatrixCuda out, float ra){
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= in.matMeta.width || r >= in.matMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    //int inidx1 = r * in.pitchWords + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * out.pitchWords + c;

    int zhouweiSum = 0;
    int zhouweizuobiao = 0;
    int count = 0;
    int cNow = 0, rNow = 0;
    int cEnd = c + ra / 2, rEnd = r + ra / 2;
    
    if(ra / 2 >= c){
        cNow = 0;
    }
    else{
        cNow = c - ra / 2;
    }
    if(ra / 2 >= r){
        rNow = 0;
    }
    else{
        rNow = r - ra / 2;
    }
    count = (cEnd - cNow + 1) * (rEnd - rNow + 1);
    for(int i = cNow; i != cEnd; i++){
        for( int j=rNow; j != rEnd ;j++ ){
            zhouweizuobiao = j * in.pitchWords + i;
            zhouweiSum += in.matMeta.matData[zhouweizuobiao];
        }      
    }

    out.matMeta.matData[outidx] = zhouweiSum / count;

   // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
		
        // 算法可以改进点：把上一次的计算结果应用上
        if (++r >= out.matMeta.height)
            return;

        //inidx1 = r * in.pitchWords + c;
        outidx = r * out.pitchWords + c;
        zhouweiSum = 0;
        zhouweizuobiao = 0;
        count = 0;

        if(ra / 2 >= c){
            cNow = 0;
        }
        else{
            cNow = c - ra / 2;
        }
        if(ra / 2 >= r){
            rNow = 0;
        }
        else{
            rNow = r - ra / 2;
        }
        count = (cEnd - cNow + 1) * (rEnd - rNow + 1);

        for(int i = cNow; i != cEnd; i++){
            for( int j=rNow; j != rEnd ;j++ ){
                zhouweizuobiao = j * in.pitchWords + i;
                zhouweiSum += in.matMeta.matData[zhouweizuobiao];
            }      
        }
        out.matMeta.matData[outidx] = zhouweiSum / count;

    }
}

static __global__ void _beTop(MatrixCuda img){
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= img.matMeta.width || r >= img.matMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx1 = r * img.pitchWords + c;
	
    // 颜色致白
    img.matMeta.matData[inidx1] = 255;
	
    // 处理剩下的三个像素点。
    for (int i =0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= img.matMeta.height)
            return;

        inidx1 = r * img.pitchWords + c;      
        img.matMeta.matData[inidx1] = 255;
    }
}


__host__ int LearningFilter::learningFilter(Image *inimg1, Image *inimg2, Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg1 == NULL || inimg2==NULL || outimg==NULL)
        return NULL_POINTER;
		
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg1);
    if (errcode != NO_ERROR)
        return errcode;
		
    errcode = ImageBasicOp::copyToCurrentDevice(inimg2);
    if (errcode != NO_ERROR)
        return errcode;
		
    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg1->roiX2 - inimg1->roiX1,
                inimg1->roiY2 - inimg1->roiY1);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }
	
    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud1;
    errcode = ImageBasicOp::roiSubImage(inimg1, &insubimgCud1);
    if (errcode != NO_ERROR)
        return errcode;
		
    ImageCuda insubimgCud2;
    errcode = ImageBasicOp::roiSubImage(inimg2, &insubimgCud2);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
	
    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一
    if (insubimgCud1.imgMeta.width > outsubimgCud.imgMeta.width)
        insubimgCud1.imgMeta.width = outsubimgCud.imgMeta.width;
    else
        outsubimgCud.imgMeta.width = insubimgCud1.imgMeta.width;
		
    if (insubimgCud1.imgMeta.height > outsubimgCud.imgMeta.height){
        insubimgCud1.imgMeta.height = outsubimgCud.imgMeta.height;
        insubimgCud2.imgMeta.height = outsubimgCud.imgMeta.height;
        }
    else
        outsubimgCud.imgMeta.height = insubimgCud1.imgMeta.height;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    int width = insubimgCud1.imgMeta.width;
    int height = insubimgCud1.imgMeta.height;
//  
    Matrix *matrixImg1, *matrixImg2, *matrixOut;     // 特征值矩阵

    // 创建特征值 matrix 指针
    errcode = MatrixBasicOp::newMatrix(&matrixImg1);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    errcode = MatrixBasicOp::newMatrix(&matrixImg2);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    errcode = MatrixBasicOp::newMatrix(&matrixOut);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    // 在设备端申请 matrix 空间
    errcode = MatrixBasicOp::makeAtCurrentDevice(matrixImg1, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    errcode = MatrixBasicOp::makeAtCurrentDevice(matrixImg2, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    errcode = MatrixBasicOp::makeAtCurrentDevice(matrixOut, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    // 创建 MatrixCuda 指针
    MatrixCuda *matrixImg1cuda, *matrixImg2cuda, *matrixOutcuda;     // 特征值
                                                                     // 设备端矩阵
    // 通过预定义的宏将 Matrix 指针转化为 MatrixCuda 类型的指针
    matrixImg1cuda = MATRIX_CUDA(matrixImg1);
    matrixImg2cuda = MATRIX_CUDA(matrixImg2);
    matrixOutcuda = MATRIX_CUDA(matrixOut);

    // 矩阵赋值
    _imgToMat<<<gridsize, blocksize>>>(insubimgCud1, *matrixImg1cuda);
    _imgToMat<<<gridsize, blocksize>>>(insubimgCud2, *matrixImg2cuda);
    _imgToMat<<<gridsize, blocksize>>>(outsubimgCud, *matrixOutcuda);
    
    // 所有变量声明：
    Matrix *tBI, *tBp, *Ip, *tBIp, *mean_I, *mean_p, *mean_mean_Ip, *mean_Ip, *cov_Ip, 
    *tBII, *mean_II, *mean_I_mean_I, *var_I, *a, *b, *var_Ieps, *var_Ieps2, *tt, *mean_a,
    *mean_b, *tBa, *tBb, *ttt, *N;
	
    //MatrixCuda *tBICuda, *tBpCuda, *IpCuda, *tBIpCuda, *mean_ICuda, *mean_pCuda, *mean_mean_IpCuda, *mean_IpCuda, *cov_IpCuda, 
    //*tBIICuda, *mean_IICuda, *mean_I_mean_ICuda, *var_ICuda, *aCuda, *bCuda, *var_IepsCuda, *var_Ieps2Cuda, *ttCuda, *mean_aCuda,
    //*mean_bCuda, *tBaCuda, *tBbCuda, *tttCuda;
    MatrixCuda *NCuda;
// 初始化变量   
    
    errcode = MatrixBasicOp::newMatrix(&N);
    errcode = MatrixBasicOp::makeAtCurrentDevice(N, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    
    errcode = MatrixBasicOp::newMatrix(&tBI);
    errcode = MatrixBasicOp::makeAtCurrentDevice(tBI, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    
    errcode = MatrixBasicOp::newMatrix(&tBp);
    errcode = MatrixBasicOp::makeAtCurrentDevice(tBp, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    errcode = MatrixBasicOp::newMatrix(&Ip);
    errcode = MatrixBasicOp::makeAtCurrentDevice(Ip, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

	
    errcode = MatrixBasicOp::newMatrix(&tBIp);
    errcode = MatrixBasicOp::makeAtCurrentDevice(tBIp, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    errcode = MatrixBasicOp::newMatrix(&mean_I);
    errcode = MatrixBasicOp::makeAtCurrentDevice(mean_I, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    
    errcode = MatrixBasicOp::newMatrix(&mean_p);
    errcode = MatrixBasicOp::makeAtCurrentDevice(mean_p, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    errcode = MatrixBasicOp::newMatrix(&mean_mean_Ip);
    errcode = MatrixBasicOp::makeAtCurrentDevice(mean_mean_Ip, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
	
    errcode = MatrixBasicOp::newMatrix(&mean_Ip);
    errcode = MatrixBasicOp::makeAtCurrentDevice(mean_Ip, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
		
    errcode = MatrixBasicOp::newMatrix(&cov_Ip);
    errcode = MatrixBasicOp::makeAtCurrentDevice(cov_Ip, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    
    errcode = MatrixBasicOp::newMatrix(&tBII);
    errcode = MatrixBasicOp::makeAtCurrentDevice(tBII, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    errcode = MatrixBasicOp::newMatrix(&mean_II);
    errcode = MatrixBasicOp::makeAtCurrentDevice(mean_II, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
		
    errcode = MatrixBasicOp::newMatrix(&mean_I_mean_I);
    errcode = MatrixBasicOp::makeAtCurrentDevice(mean_I_mean_I, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    
    errcode = MatrixBasicOp::newMatrix(&var_I);
    errcode = MatrixBasicOp::makeAtCurrentDevice(var_I, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    errcode = MatrixBasicOp::newMatrix(&a);
    errcode = MatrixBasicOp::makeAtCurrentDevice(a, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
		
    errcode = MatrixBasicOp::newMatrix(&b);
    errcode = MatrixBasicOp::makeAtCurrentDevice(b, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    
    errcode = MatrixBasicOp::newMatrix(&var_Ieps);
    errcode = MatrixBasicOp::makeAtCurrentDevice(var_Ieps, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    errcode = MatrixBasicOp::newMatrix(&var_Ieps2);
    errcode = MatrixBasicOp::makeAtCurrentDevice(var_Ieps2, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
		
    errcode = MatrixBasicOp::newMatrix(&tt);
    errcode = MatrixBasicOp::makeAtCurrentDevice(tt, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    
    errcode = MatrixBasicOp::newMatrix(&mean_a);
    errcode = MatrixBasicOp::makeAtCurrentDevice(mean_a, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    errcode = MatrixBasicOp::newMatrix(&mean_b);
    errcode = MatrixBasicOp::makeAtCurrentDevice(mean_b, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
		
    errcode = MatrixBasicOp::newMatrix(&tBa);
    errcode = MatrixBasicOp::makeAtCurrentDevice(tBa, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    
    errcode = MatrixBasicOp::newMatrix(&tBb);
    errcode = MatrixBasicOp::makeAtCurrentDevice(tBb, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    errcode = MatrixBasicOp::newMatrix(&ttt);
    errcode = MatrixBasicOp::makeAtCurrentDevice(ttt, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;		
    
    //tBICuda = MATRIX_CUDA(tBI);
    //tBpCuda = MATRIX_CUDA(tBp);
    //IpCuda = MATRIX_CUDA(Ip);
    //tBIpCuda = MATRIX_CUDA(tBIp);
    //mean_ICuda = MATRIX_CUDA(mean_I);
    //mean_pCuda = MATRIX_CUDA(mean_p);
    //mean_mean_IpCuda = MATRIX_CUDA(mean_mean_Ip);
    //mean_IpCuda = MATRIX_CUDA(mean_Ip);
    //cov_IpCuda = MATRIX_CUDA(cov_Ip);
    //tBIICuda = MATRIX_CUDA(tBII);
    //mean_IICuda = MATRIX_CUDA(mean_II);
    //mean_I_mean_ICuda = MATRIX_CUDA(mean_I_mean_I);
    //var_ICuda = MATRIX_CUDA(var_I);
    //aCuda = MATRIX_CUDA(a);
    //bCuda = MATRIX_CUDA(b);
    //var_IepsCuda = MATRIX_CUDA(var_Ieps);
    //var_Ieps2Cuda = MATRIX_CUDA(var_Ieps2);
    //ttCuda = MATRIX_CUDA(tt);
    //mean_aCuda = MATRIX_CUDA(mean_a);
    //mean_bCuda = MATRIX_CUDA(mean_b);
    //tBaCuda = MATRIX_CUDA(tBa);
    //tBbCuda = MATRIX_CUDA(tBb);
    //tttCuda = MATRIX_CUDA(ttt);
    NCuda = MATRIX_CUDA(N);

    // 初始化 N
    // cv::Mat N = cv::Mat::ones (hei,wid,CV_32FC1)
    _beTop<<<gridsize, blocksize>>>(*NCuda);
    // cv::boxFilter(N,N,N.depth(),cv::Size(r,r),cv::Point(-1,-1),true,cv::BORDER_REFLECT);//boxfiter;N-->the size of each local patch 
    _linearFilter<<<gridsize, blocksize>>>(*NCuda, *NCuda, r);
    // _subtract<<<gridsize, blocksize>>> _add<<<gridsize, blocksize>>> 
    // _multiply<<<gridsize, blocksize>>> _divide<<<gridsize, blocksize>>> 
    // _addWeighted<<<gridsize, blocksize>>> _linearFilter<<<gridsize, blocksize>>>
    // /* End define variables*/
            // // 输入，输出， 像素位数，  内核尺寸，        锚点，   数值是否正规化 ，
    // cv::boxFilter(image,tBI,image.depth(),cv::Size(r, r),cv::Point(-1,-1),true,cv::BORDER_REFLECT);//box fitering 
    _linearFilter<<<gridsize, blocksize>>>(*matrixImg1cuda, *matrixOutcuda, r);
    // cv::boxFilter(p,tBp,p.depth(),cv::Size(r, r),cv::Point(-1,-1),true,cv::BORDER_REFLECT);
/*    _linearFilter<<<gridsize, blocksize>>>(*matrixImg2cuda, *tBpCuda, r);
    // cv::multiply(image,p,Ip);
    _multiply<<<gridsize, blocksize>>>(*matrixImg1cuda, *matrixImg2cuda, *IpCuda);
    // cv::boxFilter(Ip,tBIp,Ip.depth(),cv::Size(r, r),cv::Point(-1,-1),true,cv::BORDER_REFLECT);
    _linearFilter<<<gridsize, blocksize>>>(*IpCuda, *tBIpCuda, r);
    // cv::divide(tBI,N,mean_I);
    _divide<<<gridsize, blocksize>>>(*tBICuda, *NCuda, *mean_ICuda);
    // cv::divide(tBp,N,mean_p);
    _divide<<<gridsize, blocksize>>>(*tBpCuda, *NCuda, *mean_pCuda);
    // cv::divide(tBIp,N,mean_Ip); // mean_Ip = tBIp/N (前面数组中的每个个元素除以后面数组中的每个元素）
    _divide<<<gridsize, blocksize>>>(*tBIpCuda, *NCuda, *mean_IpCuda);
    // cv::multiply(mean_I,mean_p,mean_mean_Ip); 
    _multiply<<<gridsize, blocksize>>>(*mean_ICuda, *mean_pCuda, *mean_mean_IpCuda);
    // cv::subtract(mean_Ip,mean_mean_Ip,cov_Ip); //this is the covariance of (image, p) in each local patch
    _subtract<<<gridsize, blocksize>>>(*mean_IpCuda, *mean_mean_IpCuda, *cov_IpCuda);
    // cv::multiply(image,image,tBII); //Ip = image*p;
    _multiply<<<gridsize, blocksize>>>(*matrixImg1cuda, *matrixImg1cuda, *tBIICuda);
    // cv::boxFilter(tBII,tBII,tBII.depth(),cv::Size(r, r),cv::Point(-1,-1),true,cv::BORDER_REFLECT);
    _linearFilter<<<gridsize, blocksize>>>(*tBIICuda, *tBIICuda, r);
    // cv::divide(tBII,N,mean_II);
    _divide<<<gridsize, blocksize>>>(*tBIICuda, *NCuda, *mean_IICuda);	
    // cv::multiply(mean_I,mean_I,mean_I_mean_I);
    _multiply<<<gridsize, blocksize>>>(*mean_ICuda, *mean_ICuda, *mean_I_mean_ICuda);	
    // cv::subtract(mean_II,mean_I_mean_I,var_I);
    _subtract<<<gridsize, blocksize>>>(*mean_IICuda, *mean_I_mean_ICuda, *var_ICuda);
    // cv::addWeighted(var_I, 1 ,var_I, 0,eps,var_Ieps);
    // _addWeighted(MatrixCuda img, float eps, MatrixCuda out)
    _addWeighted<<<gridsize, blocksize>>>(*var_ICuda, eps, *var_ICuda);
    // cv::divide(cov_Ip,var_Ieps,a);
    _divide<<<gridsize, blocksize>>>(*cov_IpCuda, *var_IepsCuda, *aCuda);	
    // cv::multiply(a,mean_I,tt);
    _multiply<<<gridsize, blocksize>>>(*aCuda, *mean_ICuda, *ttCuda);	
    // cv::subtract(mean_p,tt,b);
    _subtract<<<gridsize, blocksize>>>(*mean_pCuda, *ttCuda, *bCuda);
    // cv::boxFilter(a,tBa,a.depth(),cv::Size(r, r),cv::Point(-1,-1),true,cv::BORDER_REFLECT);
    _linearFilter<<<gridsize, blocksize>>>(*aCuda, *tBaCuda, r);
    // cv::boxFilter(b,tBb,b.depth(),cv::Size(r, r),cv::Point(-1,-1),true,cv::BORDER_REFLECT);
    _linearFilter<<<gridsize, blocksize>>>(*bCuda, *tBbCuda, r);
    // cv::divide(tBa,N,mean_a);
    _divide<<<gridsize, blocksize>>>(*tBaCuda, *NCuda, *mean_aCuda);	
    // cv::divide(tBb,N,mean_b);
    _divide<<<gridsize, blocksize>>>(*tBbCuda, *NCuda, *mean_bCuda);	
    // cv::multiply(mean_a,image,ttt);
    _multiply<<<gridsize, blocksize>>>(*mean_aCuda, *matrixImg1cuda, *tttCuda);	
    // cv::add(ttt, mean_b, Out); 
    _add<<<gridsize, blocksize>>>(*tttCuda, *mean_bCuda, *matrixOutcuda);
*/
    _matToImg<<<gridsize, blocksize>>>(outsubimgCud, *matrixOutcuda);
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。	
    return NO_ERROR;
}

void __learningFilter_Dummy()
{
    MatrixCuda *tmp = new MatrixCuda;
    _subtract<<<1,1>>>(*tmp, *tmp, *tmp);
    _add<<<1,1>>>(*tmp, *tmp, *tmp);
    _multiply<<<1,1>>>(*tmp, *tmp, *tmp);
    _divide<<<1,1>>>(*tmp, *tmp, *tmp);
    _addWeighted<<<1,1>>>(*tmp, 0.0f, *tmp);
}
