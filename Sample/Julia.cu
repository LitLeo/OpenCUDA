// Julia.cu
// Julia集生成算法

#include "Julia.h"
#include "Image.h"
#include "ErrorCode.h"
#include "Complex.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数：_juliaKer（Mandelbrot集生成）
// 以显示区域 W 内所有的点 (mc, mr) 作为初始迭代值 z = z * z + p 进行 times 次
// 迭代，z 的初始值为 mc + mr。迭代完成后比较 z 的模和逃逸半径 radius 的大小，
// 根据比较结果进行着色。
static __global__ void      // 无返回值 
_juliaKer(
        ImageCuda outimg,   // 输出图像
        Julia param    // 参数
);

// Kernel 函数: _juliaKer（Julia 集生成）
static __global__ void _juliaKer(ImageCuda outimg, Julia param)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标的
    // x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算输出图像的宽和高
    int width = outimg.imgMeta.width;
    int height = outimg.imgMeta.height;

    // 读取各个参数的值。
    float radius = param.getRadius();
    float radius2 = radius * radius;
    int times = param.getTimes();
    int colorcnt = param.getColorCount();
    int exp = param.getExponent();
    float fromc = param.getScopeFrom().getReal();
    float fromr = param.getScopeFrom().getImaginary();
    float toc = param.getScopeTo().getReal();
    float tor = param.getScopeTo().getImaginary();

    // 计算各个颜色阶的颜色值。
    extern __shared__ unsigned char colorbuf[];
    int inidx = threadIdx.y * blockDim.x + threadIdx.x;
    if (inidx <= colorcnt) {
        colorbuf[inidx] = (unsigned char)((256 * (colorcnt - inidx) /
                                          colorcnt)/* & 0xFF*/);
    }
    __syncthreads();

    // 将坐标的原点定位到图像中心，将图像的范围设定为 -1.5 到 1.5
    float mc = (toc - fromc) * c / width + fromc;
    float mr = (tor - fromr) * r / height + fromr;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 一方面防止由于段错误导致的程序崩溃。
    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;    
 
    // 以显示区域 W 内所有的点(mc, mr)作为初始迭代值 p = mc + mri 对
    // z = z * z + p 进行 times 次迭代，z 的初始值为 0。迭代完成后比较 z 的模和
    // 逃逸半径 radius 的大小，根据比较结果进行着色。

    // 设定 z 的初始值为 0
    Complex z(mc, mr);

    // 计算第一个坐标点对应的图像数据数组下标。
    int idx = r * outimg.pitchBytes + c;

    // 将像素坐标转换为复数空间的坐标，根据复数空间的坐标生成初始迭代值 p  
    Complex p = param.getConstP();

    for (int i = 0; i < times; i++) {
        // 调用复数类的乘号重载函数，计算 z 的 exp 次幂
        Complex z2 = z;
        for (int j = 2; j <= exp; j++)
            z2 = z2 * z;

        // 调用复数类的加号重载函数，实现 z 和 p相加
        z = z2 + p;

        // 调用复数类的求模函数
        float zmod2 = z.modulus2();

        if (zmod2 > radius2) {
            // 如果复数的模的平方大于逃逸半径的平方

            // 计算当前对应的颜色下标
            int coloridx = i * colorcnt / times;
            
            // 为当前像素涂上其对应的颜色。
            outimg.imgMeta.imgData[idx] = colorbuf[coloridx];

            // 跳出此次循环
            break;
        } else if (/*zmodulus <= radius2 && */i == times - 1 ) {
            // 如果复数的模的平方小于或等于逃逸半径的平方，并且迭代次数为
            // times，设定像素值的大小为 255
            outimg.imgMeta.imgData[idx] = 255;
            //break;
        }
   }
}

// Host 成员方法：julia（Julia 集生成）
__host__ int Julia::julia(Image *outimg)
{
    // 检查图像是否为 NULL，如果为 NULL 直接报错返回。
    if (outimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取图像的 ROI 子图像。
    ImageCuda subimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &subimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (subimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (subimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 调用核函数
    _juliaKer<<<gridsize, blocksize, colorCount + 1>>>(subimgCud, *this);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：juliaHost（Julia 集生成）
__host__ int Julia::juliaHost(Image *outimg)
{
    // 检查图像是否为 NULL，如果为 NULL 直接报错返回。
    if (outimg == NULL)
        return NULL_POINTER;

    int errcode;  // 局部变量，错误码

    // 将图像拷贝到 Host 内存中。
    errcode = ImageBasicOp::copyToHost(outimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取图像的 ROI 子图像。
    ImageCuda subimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &subimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    ////
    // 计算输出图像的宽和高
    int width = subimgCud.imgMeta.width;
    int height = subimgCud.imgMeta.height;

    // 读取各个参数的值。
    const float &radius = this->getRadius();
    float radius2 = radius * radius;
    const int &times = this->getTimes();
    const int &colorcnt = this->getColorCount();
    const int &exp = this->getExponent();
    const float &fromc = this->getScopeFrom().getReal();
    const float &fromr = this->getScopeFrom().getImaginary();
    const float &toc = this->getScopeTo().getReal();
    const float &tor = this->getScopeTo().getImaginary();

    // 计算各个颜色阶的颜色值。
    static unsigned char colorbuf[256];
    for (int i = 0 ;i < colorcnt; i++) {
        colorbuf[i] = (unsigned char)((256 * (colorcnt - i) /
                                       colorcnt)/* & 0xFF*/);
    }

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            // 将坐标的原点定位到图像中心，将图像的范围设定为 -1.5 到 1.5
            float mc = (toc - fromc) * c / width + fromc;
            float mr = (tor - fromr) * r / height + fromr;

            // 以显示区域 W 内所有的点(mc, mr)作为初始迭代值 p = mc + mri 对
            // z = z * z + p 进行 times 次迭代，z 的初始值为 0。迭代完成后比较
            // z 的模和逃逸半径 radius 的大小，根据比较结果进行着色。

            // 设定 z 的初始值为 0
            Complex z(mc, mr);

            // 计算第一个坐标点对应的图像数据数组下标。
            int idx = r * subimgCud.pitchBytes + c;

            // 将像素坐标转换为复数空间的坐标，根据复数空间的坐标生成初始迭代值
            // p  
            Complex p = this->getConstP();

            for (int i = 0; i < times; i++) {
                // 调用复数类的乘号重载函数，计算 z 的 exp 次幂
                Complex z2 = z;
                for (int j = 2; j <= exp; j++)
                    z2 = z2 * z;

                // 调用复数类的加号重载函数，实现 z 和 p相加
                z = z2 + p;

                // 调用复数类的求模函数
                float zmod2 = z.modulus2();

                if (zmod2 > radius2) {
                    // 如果复数的模的平方大于逃逸半径的平方

                    // 计算当前对应的颜色下标
                    int coloridx = i * colorcnt / times;

                    // 为当前像素涂上其对应的颜色。
                    subimgCud.imgMeta.imgData[idx] = colorbuf[coloridx];

                    // 跳出此次循环
                    break;
                } else if (/*zmodulus <= radius2 && */i == times - 1 ) {
                    // 如果复数的模的平方小于或等于逃逸半径的平方，并且迭代次数
                    // 为 times，设定像素值的大小为 255
                    subimgCud.imgMeta.imgData[idx] = 255;
                    //break;
                }
            } // end of for (i, times)
        } // end of for (c, width)
    } // end of for (r, height)

    return NO_ERROR;
}
