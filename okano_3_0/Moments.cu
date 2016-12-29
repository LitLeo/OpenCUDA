// Moments.cu
// 几何矩的计算

#include "Moments.h"

#include <iostream>
#include <stdio.h>
#include <cmath>
using namespace std;

#include "ErrorCode.h"


// 宏：M_PI
// π值。对于某些操作系统，M_PI可能没有定义，这里补充定义 M_PI。
#ifndef M_PI
#define M_PI 3.14159265359
#endif

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  256
#define DEF_BLOCK_Y    1

// Kernel 函数：_accumulateImageKer（原图像的累进求和）
// 利用差分矩因子算法计算几何矩。对原图像的每一行做 5 次累进求和，保留结果
// g1(j, 1), g2(j, 2), g3(j, 1) + g3(j, 2), g4(j, 1) + g4(j, 2) * 4 + g4(j, 3),
// g5(j, 1) + g5(j, 2) * 11 + g5(j, 3) * 11 + g5(j, 4),j 等于图像的高度 height，
// 所以输出数组 accimg 大小为 5 * height。每个线程处理一行的元素。
static __global__ void  // Kernel 函数无返回值
_accumulateImageKer(
        ImageCuda img,  // 输入图像
        double *accimg  // 原图像的 5 次累进求和
);

// Kernel 函数：_accumulateConstantOneKer（原图像的累进求和,乘积项设置恒为 1）
// 利用差分矩因子算法计算几何矩。对原图像的每一行做 5 次累进求和，保留结果
// g1(j, 1), g2(j, 2), g3(j, 1) + g3(j, 2), g4(j, 1) + g4(j, 2) * 4 + g4(j, 3),
// g5(j, 1) + g5(j, 2) * 11 + g5(j, 3) * 11 + g5(j, 4),j 等于图像的高度 height，
// 所以输出数组 accimg 大小为 5 * height。每个线程处理一行的元素。注意，计算矩
// 过程中的乘积项设置为 1。
static __global__ void  // Kernel 函数无返回值
_accumulateConstantOneKer(
        ImageCuda img,  // 输入图像
        double *accimg  // 原图像的 5 次累进求和
);

// Host 方法：complexMultiply（复数乘法运算）
// 输入两个复数的实部和虚部，输出结果的实虚部。
__host__ int complexMultiply(double real1, double imag1, double real2, 
                             double imag2, double *realout, double *imagout);

// Kernel 函数：_accumulateImageKer（原图像的累进求和）
static __global__ void _accumulateImageKer(ImageCuda img, double *accimg)
{
    // 计算线程对应的位置。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > img.imgMeta.height)
        return;

    // 原图像的每一行做 5 次累进求和的结果。
    double g11, g21, g31, g41, g51;

    // 因为是从右向左累进，所以初始化为每一行最右边的元素。
    int index = img.pitchBytes * idx + (img.imgMeta.width - 1);
    g11 = g21 = g31 = g41 = g51 = img.imgMeta.imgData[index];

    // 循环累加每一行元素。该循环暂且计算到 img.imgMeta.width - 3 个元素，
    // 因为需要保存 g54 的值。
    int tempidx = img.pitchBytes * idx;
    //int tempidx = img.imgMeta.width * idx;
    for (int i = img.imgMeta.width - 2; i >= 3; i--) {
        // 对原图像进行 5 次 累进求和。
        g11 += img.imgMeta.imgData[tempidx + i];
        g21 += g11;
        g31 += g21;
        g41 += g31;
        g51 += g41;
    }
    // 计算 g54 的值。
    double g54 = g51;

    // 继续累进一个像素，即总体计算到 img.imgMeta.width - 2 个元素。
    g11 += img.imgMeta.imgData[tempidx + 2];
    g21 += g11;
    g31 += g21;
    g41 += g31;
    g51 += g41;

    // 计算 g43, g53 的值。
    double g43 = g41;
    double g53 = g51;


    // 继续累进一个像素，即总体计算到 img.imgMeta.width - 1 个元素。
    g11 += img.imgMeta.imgData[tempidx + 1];
    g21 += g11;
    g31 += g21;
    g41 += g31;
    g51 += g41;

    // 计算 g32, g42, g52 的值。
    double g32 = g31;
    double g42 = g41;
    double g52 = g51;

    // 继续累进一个像素，即总体计算到 img.imgMeta.width 个元素，累进求和结束。
    g11 += img.imgMeta.imgData[tempidx];
    g21 += g11;
    g31 += g21;
    g41 += g31;
    g51 += g41;

    // 将累进结果保存到输出数组中。
    accimg[idx] = g11;
    accimg[idx += img.imgMeta.height] = g21;
    accimg[idx += img.imgMeta.height] = g31 + g32;
    accimg[idx += img.imgMeta.height] = g41 + g42 * 4.0f + g43;
    accimg[idx += img.imgMeta.height] = g51 + g52 * 11.0f + g53 * 11.0f + g54;
}

// Kernel 函数：_accumulateConstantOneKer（原图像的累进求和,乘积项设置恒为 1）
static __global__ void _accumulateConstantOneKer(ImageCuda img, double *accimg)
{
    // 计算线程对应的位置。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > img.imgMeta.height)
        return;

    // 原图像的每一行做 5 次累进求和的结果。
    double g11, g21, g31, g41, g51;

    // 因为是从右向左累进，所以初始化为每一行最右边的元素。
    g11 = g21 = g31 = g41 = g51 = 1.0f;

    // 循环累加每一行元素。该循环暂且计算到 img.imgMeta.width - 3 个元素，
    // 因为需要保存 g54 的值。
    int tempidx = img.pitchBytes * idx;
    //int tempidx = img.imgMeta.width * idx;
    for (int i = img.imgMeta.width - 2; i >= 3; i--) {
        // 对原图像进行 5 次 累进求和。
        g11 += 1.0f;
        g21 += g11;
        g31 += g21;
        g41 += g31;
        g51 += g41;
    }
    // 计算 g54 的值。
    double g54 = g51;

    // 继续累进一个像素，即总体计算到 img.imgMeta.width - 2 个元素。
    g11 += img.imgMeta.imgData[tempidx + 2];
    g21 += g11;
    g31 += g21;
    g41 += g31;
    g51 += g41;

    // 计算 g43, g53 的值。
    double g43 = g41;
    double g53 = g51;


    // 继续累进一个像素，即总体计算到 img.imgMeta.width - 1 个元素。
    g11 += img.imgMeta.imgData[tempidx + 1];
    g21 += g11;
    g31 += g21;
    g41 += g31;
    g51 += g41;

    // 计算 g32, g42, g52 的值。
    double g32 = g31;
    double g42 = g41;
    double g52 = g51;

    // 继续累进一个像素，即总体计算到 img.imgMeta.width 个元素，累进求和结束。
    g11 += img.imgMeta.imgData[tempidx];
    g21 += g11;
    g31 += g21;
    g41 += g31;
    g51 += g41;

    // 将累进结果保存到输出数组中。
    accimg[idx] = g11;
    accimg[idx += img.imgMeta.height] = g21;
    accimg[idx += img.imgMeta.height] = g31 + g32;
    accimg[idx += img.imgMeta.height] = g41 + g42 * 4.0f + g43;
    accimg[idx += img.imgMeta.height] = g51 + g52 * 11.0f + g53 * 11.0f + g54;
}

// Host 成员方法：spatialMoments（计算空间矩）
__host__ int Moments::spatialMoments(Image *img, MomentSet *momset)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (img == NULL)
        return NULL_POINTER;

    // 检查 momset 是否为空。
    if (momset == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(img);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda subimgCud;
    errcode = ImageBasicOp::roiSubImage(img, &subimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 在 Device 端申请空间，并将其数据初始化为 0。
    cudaError_t cudaerrcode;
    double *accimgdev;
    int datasize = 5 * subimgCud.imgMeta.height * sizeof (double);
    cudaerrcode = cudaMalloc((void**)&accimgdev, datasize);
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 初始化 Device 上的内存空间。
    cudaerrcode = cudaMemset(accimgdev, 0, datasize);
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (subimgCud.imgMeta.height + DEF_BLOCK_X - 1) / DEF_BLOCK_X;
    gridsize.y = DEF_BLOCK_Y;

    // 计算矩的过程中正常乘以原图像的灰度值。
    if (!this->isconst) {
        // 调用核函数，对原图像每一行计算 5 次累进求和，每个线程计算一行数据。
        _accumulateImageKer<<<gridsize, blocksize>>>(subimgCud, accimgdev);
    // 计算矩的过程中图像的灰度值部分恒等于 1。
    } else {
        // 调用核函数，对原图像每一行计算 5 次累进求和，每个线程计算一行数据。
        _accumulateConstantOneKer<<<gridsize, blocksize>>>(subimgCud, 
                                                           accimgdev);
    }
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(accimgdev);
        return CUDA_ERROR;
    }

    // 将 Device 端数据拷贝到 Host 端。
    double *accimg = new double[5 * subimgCud.imgMeta.height];
    cudaerrcode = cudaMemcpy(accimg, accimgdev, datasize, 
                             cudaMemcpyDeviceToHost);

    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 对 g1(i, j) 的第一列做 5 次迭代累进求和。
    int tempidx;
    double r11, r21, r31, r41, r51,  r32 = 0.0f, r42 = 0.0f, 
           r43 = 0.0f, r52 = 0.0f, r53 = 0.0f, r54 = 0.0f;
    tempidx = subimgCud.imgMeta.height - 1;
    r11 = r21 = r31 = r41 = r51 = accimg[tempidx];

    // 对 g2(i, j) 的第一列做 4 次迭代累进求和。
    double s11, s21, s31, s41, s32 = 0.0f, s42 = 0.0f, s43 = 0.0f;
    tempidx += subimgCud.imgMeta.height;
    s11 = s21 = s31 = s41 = accimg[tempidx];

    // 对 t0(j) = g3(j, 1) + g3(j, 2)做 3 次累进求和。
    double t11, t21, t31, t32 = 0.0f;
    tempidx += subimgCud.imgMeta.height;
    t11 = t21 = t31 = accimg[tempidx];

    // 对 u0(j) = g4(j, 1) + g4(j, 2) * 4 + g4(j, 3) 做 2 次累进求和。
    double u11, u21;
    tempidx += subimgCud.imgMeta.height;
    u11 = u21 = accimg[tempidx];

    // 对 p0(j) = g5(j, 1) + g5(j, 2) * 11 + g5(j, 3) * 11 + g5(j, 4) 做
    // 1 次累进求和。
    tempidx += subimgCud.imgMeta.height;
    double p11 = accimg[tempidx];

    // 循环累加每一行元素。中间过程中需要计算一些额外值，
    // 包括 r54, r53, r52, r43, r42, s43, s42, s32, t32。
    for (int i = subimgCud.imgMeta.height - 2; i >= 0; i--) {
        // 对 g1(j) 进行 5 次累进求和。
        r11 += accimg[i];
        r21 += r11;
        r31 += r21;
        r41 += r31;
        r51 += r41;

        // 对 g2(j) 进行 4 次累进求和。
        tempidx = i + subimgCud.imgMeta.height;
        s11 += accimg[tempidx];
        s21 += s11;
        s31 += s21;
        s41 += s31;

        // 对 t0(j) 进行 3 次累进求和。
        tempidx += subimgCud.imgMeta.height;
        t11 += accimg[tempidx];
        t21 += t11;
        t31 += t21;

        // 对 u0(j) 进行 2 次累进求和。
        tempidx += subimgCud.imgMeta.height;
        u11 += accimg[tempidx];
        u21 += u11;

        // 对 p0(j) 进行 1 次累进求和。
        tempidx += subimgCud.imgMeta.height;
        p11 += accimg[tempidx];

        // 计算 r54。
        if (i == 3) {
            r54 = r51;
        }

        // 计算 r53, r43, s43。
        if (i == 2) {
            r43 = r41;
            r53 = r51;
            s43 = s41;
        }

        // 计算 r32, r42, r52, s42, s32, t32。
        if (i == 1) {
            r32 = r31;
            r42 = r41;
            r52 = r51;
            s32 = s31;
            s42 = s41;
            t32 = t31;
        }
    }

    // 根据上面计算的变量，对 MomentSet 中的 15 个空间矩进行赋值。
    momset->m00 = r11;
    momset->m10 = s11;
    momset->m01 = r21;
    momset->m20 = t11;
    momset->m11 = s21;
    momset->m02 = r31 + r32;
    momset->m30 = u11;
    momset->m21 = t21;
    momset->m12 = s31 + s32;
    momset->m03 = r41 + r42 * 4.0f + r43;
    momset->m22 = t31 + t32;
    momset->m31 = u21;
    momset->m13 = s41 + s42 * 4.0f + s43;
    momset->m40 = p11;
    momset->m04 = r51 + r52 * 11.0f + r53 * 11.0f +r54;

    // 释放 Device 端空间。
    cudaFree(accimgdev);
    // 释放 Host 端空间。
    delete [] accimg;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：centralMoments（计算中心矩）
__host__ int Moments::centralMoments(Image *img, MomentSet *momset)
{
    // 局部变量，错误码
    int errcode;

    // 首先计算空间矩。
    errcode = spatialMoments(img, momset);
    if (errcode != NO_ERROR)
        return errcode;

    // 对 MomentSet 中的中心距 mu00, mu10, mu01 进行赋值。
    momset->mu00 = momset->m00;
    momset->mu10 = 0.0f;
    momset->mu01 = 0.0f;

    // 如果 mu00 不为 0 的话，继续计算其他中心矩的数值。
    if (!(fabs(momset->m00) < 0.000001)) {
        // meanX 和 meanY 是形状的分布重心。
        double meanX = momset->m10 / momset->m00;
        double meanY = momset->m01 / momset->m00;

        // 判断是否需要调整中心坐标。
        if (this->adjustcenter == true) {
            momset->mu20 = momset->m20 - meanX * momset->m10;
            momset->mu02 = momset->m02 - meanY * momset->m01;

            // 计算偏差。
            double xs = sqrt(momset->mu20 / momset->mu00);
            double ys = sqrt(momset->mu02 / momset->mu00);

            // 重新定义中心。
            meanX = meanX - xs;
            meanY = meanY - ys;
        }

        // 定义中间变量。
        double meanX2 = meanX * meanX;
        double meanY2 = meanY * meanY;

        // 计算其余的中心矩的数值。
        momset->mu20 = momset->m20 - meanX * momset->m10;
        momset->mu11 = momset->m11 - meanY * momset->m10;
        momset->mu02 = momset->m02 - meanY * momset->m01;
        momset->mu30 = momset->m30 - 3.0f * meanX * momset->m20 + 
                       2.0f * meanX2 * momset->m10;
        momset->mu21 = momset->m21 - 2.0f * meanX * momset->m11 + 
                       2.0f * meanX2 * momset->m01- meanY * momset->m20;
        momset->mu12 = momset->m12 - 2.0f * meanY * momset->m11 + 
                       2.0f * meanY2 * momset->m10 - meanX * momset->m02;
        momset->mu03 = momset->m03 - 3.0f * meanY * momset->m02 + 
                       2.0f * meanY2 * momset->m01;
        momset->mu22 = momset->m22 - 2.0f * meanY * momset->m21 + meanY2 *
                       momset->m20 - 2.0f * meanX * momset->m12 + 4.0f * 
                       meanX * meanY * momset->m11 - 2.0f * meanX * meanY *
                       momset->m10 + meanX2 * momset->m02 - 3.0f * meanX2 *
                       meanY * momset->m01;
        momset->mu31 = momset->m31 - meanY * momset->m30 - 3.0f * meanX * 
                       momset->m21 + 3.0f * meanX * meanY * momset->m20 +
                       3.0f * meanX2 * momset->m11 - 3.0f * meanX2 * meanY * 
                       momset->m10;
        momset->mu13 = momset->m13 - 3.0f * meanY * momset->m12 + 3.0f * 
                       meanY2 * momset->m11 - meanX * momset->m03 + 3.0f *
                       meanX * meanY * momset->m02 - 3.0f * meanX * meanY2 *
                       momset->m01;
        momset->mu40 = momset->m40 - 4.0f * meanX * momset->m30 +
                       6.0f * meanX2 * momset->m20 - 3.0f * meanX2 * 
                       meanX * momset->m10;
        momset->mu04 = momset->m04 - 4.0f * meanY * momset->m03 +
                       6.0f * meanY2 * momset->m02 - 3.0f * meanY2 *
                       meanY * momset->m01;
                      
    // 否则为了避免产生除 0 的错误，直接退出。
    } else {
        return OP_OVERFLOW;
    }

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：centralMoments（计算形状的分布重心和方向）
__host__ int Moments::centralMoments(Image *img, double centers[2],
                                     double *angle)
{
    // 检查输入参数是否为空。
    if (centers == NULL || angle == NULL)
        return NULL_POINTER;

    // 局部变量，错误码
    int errcode;

    // 声明 MomentSet 结构体变量。
    MomentSet momset;

    // 计算中心矩的各项值。
    errcode = centralMoments(img, &momset);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 计算几何分布重心。
    centers[0] = momset.m10 / momset.m00;
    centers[1] = momset.m01 / momset.m00;

    // 计算几何分布方向。
    double u_20 = momset.mu20 / momset.mu00;
    double u_02 = momset.mu02 / momset.mu00;
    double u_11 = momset.mu11 / momset.mu00;

    // 如果 u_11 不等于 0，并且 u_20 不等于 u_02的话，计算方向角度。
    if (!(fabs(u_11) < 0.000001) && !(fabs(u_20 - u_02) < 0.000001)) {
        // 计算角度大小。
        *angle = (atan(2.0f * u_11 / (u_20 - u_02)) / 2.0f);

    // 否则为了避免产生除 0 的错误，直接退出。
    } else {
        // 特殊标识。
        *angle = -2 * M_PI;
    }

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：huMoments（计算 Hu 矩）
__host__ int Moments::huMoments(Image *img, MomentSet *momset)
{
    // 局部变量，错误码
    int errcode;

    // 首先计算中心矩。
    errcode = centralMoments(img, momset);
    if (errcode != NO_ERROR)
        return errcode;

    // 标准化中心距。
    double p1 = pow(momset->mu00, 2.0f);
    double p2 = pow(momset->mu00, 2.5f);
    double n11 = momset->mu11 / p1;
    double n02 = momset->mu02 / p1;
    double n20 = momset->mu20 / p1;
    double n12 = momset->mu12 / p2;
    double n21 = momset->mu21 / p2;
    double n03 = momset->mu03 / p2;
    double n30 = momset->mu30 / p2;

    // 声明中间变量，方便简化后面的语句。
    double temp1 = n20 - n02;
    double temp2 = n30 - 3.0f * n12;
    double temp3 = 3.0f * n21 - n03;
    double temp4 = n30 + n12;
    double temp5 = n21 + n03;

    // 计算 Hu moments 的 8 个值。
    momset->hu1 = n20 + n02;
    momset->hu2 = temp1 * temp1 + 4.0f * n11 * n11;
    momset->hu3 = temp2 * temp2 + temp3 * temp3;
    momset->hu4 = temp4 * temp4 + temp5 * temp5;
    momset->hu5 = temp2 * temp4 * (temp4 * temp4 - 3.0f * temp5 * temp5) + 
                  temp3 * temp5 * (3.0f * temp4 * temp4 - temp5 * temp5);
    momset->hu6 = temp1 * (temp4 * temp4 - temp5 * temp5) + 
                  4.0f * n11 * temp4 * temp5;
    momset->hu7 = temp3 * temp4 * (temp4 * temp4 - 3.0f * temp5 * temp5) -
                  temp2 * temp5 * (3.0f * temp4 * temp4 - temp5 * temp5);

    // Hu 矩的扩展，增加了一个不变量。
    momset->hu8 = n11 * (temp4 * temp4 - temp5 * temp5) - 
                  temp1 * temp4 * temp5;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：affineMoments（计算 affine 矩）
__host__ int Moments::affineMoments(Image *img, MomentSet *momset)
{
    // 局部变量，错误码
    int errcode;

    // 首先计算中心矩。
    errcode = centralMoments(img, momset);
    if (errcode != NO_ERROR)
        return errcode;

    // 获得中心矩。
    double u11 = momset->mu11;
    double u20 = momset->mu20;
    double u02 = momset->mu02;
    double u12 = momset->mu12;
    double u21 = momset->mu21;
    double u30 = momset->mu30;
    double u03 = momset->mu03;
    double u13 = momset->mu13;
    double u31 = momset->mu31;
    double u22 = momset->mu22;
    double u40 = momset->mu40;
    double u04 = momset->mu04;

/*
    // 计算 9 个 affine moment invariants
    double s = momset->mu00;
    momset->ami1 = (u20 * u02 − u11 * u11) / pow(s, 4);
    momset->ami2 = (-u30 * u30 * u03 * u03 + 6 * u30 * u21 * u12 * u03 − 
                    4 * u30 * u12 * u12 * u12 − 4 * u21 * u21 * u21 * u03 + 
                    3 * u21 * u21 * u12 * u12) / pow(s, 10);
    momset->ami3 = (u20 * u21 * u03 − u20 * u12 * u12− u11 * u30 * u03 + u11 *
                    u21 * u12 + u02 * u30 * u12− u02 * u21 * u21) / pow(s, 7);
    momset->ami4 = (−u20 * u20 * u20 * u03 * u03 + 6 * u20 * u20 * u11 * u12 * 
                    u03 – 3 * u20 * u20 * u02 * u12 * u12 − 6 * u20 * u11 * 
                    u11 * u21 * u03 – 6 * u20 * u11 * u11 * u12 * u12 + 12 * 
                    u20 * u11 * u02 * u21 * u12 – 3 * u20 * u02 * u02 * u21 * 
                    u21 + 2 * u11 * u11 * u11 * u30 * u03 + 6 * u11 * u11 *
                    u11 * u21 * u12 – 6 * u11 * u11 * u02 * u30 * u12 – 
                    6 * u11 * u11 * u02 * u21 * u21 + 6 * u11 * u02 * u02 * 
                    u30 * u21 − u02 * u02 * u02 * u30 * u30) / pow(s, 11);
    momset->ami6 = (u40 * u04 – 4 * u31 * u13 + 3 * u22 * u22 ) / pow(s, 6);
    momset->ami7 = (u40 * u22 * u04 − u40 * u13 * u13 − u31 * u31 * u04 + 
                    2 * u31 * u22 * u13 − u22 * u22 * u22) / pow(s, 9);
    momset->ami8 = (u20 * u20 * u04 – 4 * u20 * u11 * u13 + 2 * u20 * u02 * 
                    u22 + 4 * u11 * u11 * u22 − 4 * u11 * u02 * u31 + u02 * 
                    u02 * u40) / pow(s, 7);
    momset->ami9 = (u20 * u20 * u22 * u04 − u20 * u20 * u13 * u13− 2 * u20 * 
                    u11 * u31 * u04 + 2 * u20 * u11 * u22 * u13 + u20 * u02 * 
                    u40 * u04 – 2 * u20 * u02 * u31 * u13 + u20 * u02 * u22 * 
                    u22 + 4 * u11 * u11 * u31 * u13 – 4 * u11 * u11 * u22 * 
                    u22 − 2 * u11 * u02 * u40 * u13 + 2 * u11 * u02 * u31 * 
                    u22 + u02 * u02 * u40 * u22 − u02 * u02 * u31 * u31) / 
                    pow(s, 10);
    momset->ami19 = (u20 * u30 * u12 * u04 − u20 * u30 * u03 * u13 − u20 * 
                     u21 * u21 * u04 + u20 * u21 * u12 * u13 + u20 * u21 * 
                     u03 * u22 − u20 * u12 * u12 * u22 – 2 * u11 * u30 * u12 *
                     u13 + 2 * u11 * u30 * u03 * u22 + 2 * u11 * u21 * u21 * 
                     u13 – 2 * u11 * u21 * u12 * u22 – 2 * u11 * u21 * u03 * 
                     u31 + 2 * u11 * u12 * u12 * u31 + u02 * u30 * u12 * u22 − 
                     u02 * u30 * u03 * u31 – u02 * u21 * u21 * u22 + u02 * 
                     u21 * u12 * u31 + u02 * u21 * u03 * u40 − u02 * u12 * 
                     u12 * u40) / pow(s, 10);
*/

    // 定义一些中间变量。
    double temp1 = u20 * u02;
    double temp2 = u11 * u11;
    double temp3 = u30 * u30;
    double temp4 = u03 * u03;
    double temp5 = u03 * u30;
    double temp6 = u21 * u12;
    double temp7 = u12 * u12 * u12;
    double temp8 = u21 * u21 * u21;
    double temp9 = u21 * u21;
    double temp10 = u12 * u12;
    double temp11 = u20 * u20 * u20;
    double temp12 = u20 * u20;
    double temp13 = u11 * u11 * u11;
    double temp14 = u02 * u02;
    double temp15 = u02 * u02 * u02;
    double temp16 = u40 * u04;
    double temp17 = u31 * u13;
    double temp18 = u22 * u22;
    double temp19 = u13 * u13;
    double temp20 = u31 * u31;

    // 计算 9 个 affine moment invariants
    double s4 = pow(momset->mu00, 4);
    double s6 = pow(momset->mu00, 6);
    double s7 = s6 * momset->mu00;
    double s9 = pow(momset->mu00, 9);
    double s10 = s9 * momset->mu00;
    double s11 = s10 * momset->mu00;

    momset->ami1 = (temp1 - temp2) / s4;

    momset->ami2 = (-temp3 * temp4 + 6 * temp5 * temp6 - 4 * u30 * temp7 -
                    4 * temp8 * u03 + 3 * temp9 * temp10) / s10;

    momset->ami3 = (u20 * u21 * u03 - u20 * temp10 - u11 * temp5 + u11 * 
                    temp6 + u02 * u30 * u12 - u02 * temp9) / s7;

    momset->ami4 = (-temp11 * temp4 + 6 * temp12 * u11 * u12 * u03 - 
                    3 * temp12 * u02 * temp10 - 6 * u20 * temp2 * u21 * 
                    u03 - 6 * u20 * temp2 * temp10 + 12 * temp1 * u11 * temp6 -
                    3 * temp1 * u02 * temp9 + 2 * temp13 * temp5 + 6 * temp13 * 
                    temp6 - 6 * temp2 * u02 * u30 * u12 - 6 * temp2 * u02 * 
                    temp9 + 6 * u11 * temp14 * u30 * u21 - temp15 * temp3) / 
                    s11;

    momset->ami6 = (temp16 - 4 * temp17 + 3 * temp18) / s6;


    momset->ami7 = (temp16 * u22 - u40 * temp19 - temp20 * u04 + 2 * u22 * 
                    temp17 - temp18 * u22) / s9;


    momset->ami8 = (temp12 * u04 - 4 * u20 * u11 * u13 + 2 * temp1 * u22 + 
                    4 * temp2 * u22 - 4 * u11 * u02 * u31 + temp14 * u40) /
                    s7;

    momset->ami9 = (temp12 * u22 * u04 - temp12 * temp19 - 2 * u20 * u11 * 
                    u31 * u04 + 2 * u20 * u11 * u22 * u13 + temp1 * temp16 -
                    2 * temp1 * temp17 + temp1 * temp18 + 4 * temp2 * temp17 -
                    4 * temp2 * temp18 - 2 * u11 * u02 * u40 * u13 + 2 * u11 * 
                    u02 * u31 * u22 + temp14 * u40 * u22 - temp14 * temp20) /
                    s10;

    momset->ami19 = (u20 * u30 * u12 * u04 - u20 * temp5 * u13 - u20 * temp9 *
                     u04 + u20 * temp6 * u04 + u20 * temp6 * u13 + u20 * u21 *
                     u03 * u22 - u20 * temp10 * u22 - 2 * u11 * u30 * u12 * 
                     u13 + 2 * u11 * temp5 * u22 + 2 * u11 * temp9 * u13 - 
                     2 * u11 * temp6 * u22 - 2 * u11 * u21 * u03 * u31 +
                     2 * u11 * temp10 * u31 + u02 * u30 * u12 * u22 - u20 * 
                     temp5 * u31 - u02 * temp9 * u22 + u02 * temp6 * u31 +
                     u02 * u21 * u03 * u40 - u02 * temp10 * u40) / s10;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 方法：complexMultiply（复数乘法运算）
__host__ int complexMultiply(double real1, double imag1, double real2, 
                             double imag2, double *realout, double *imagout)
{
    // 两复数的乘法运算。
    *realout = real1 * real2 - imag1 * imag2;
    *imagout = imag1 * real2 + real1 * imag2;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：flusserMoments（计算 flusser 矩）
__host__ int Moments::flusserMoments(Image *img, MomentSet *momset)
{
    // 局部变量，错误码。
    int errcode;

    // 使用调整的中心距。
    errcode = setAdjustcenter(true);
    if (errcode != NO_ERROR)
        return errcode;

    // 首先计算中心矩。
    errcode = centralMoments(img, momset);
    if (errcode != NO_ERROR)
        return errcode;

    // 获得中心矩。
    double u11 = momset->mu11;
    double u20 = momset->mu20;
    double u02 = momset->mu02;
    double u12 = momset->mu12;
    double u21 = momset->mu21;
    double u30 = momset->mu30;
    double u03 = momset->mu03;
    double u13 = momset->mu13;
    double u31 = momset->mu31;
    double u22 = momset->mu22;
    double u40 = momset->mu40;
    double u04 = momset->mu04;

    // 归一化中心距。
    double temp1 = pow(momset->mu00, 2);
    double temp2 = pow(momset->mu00, 2.5f);
    double temp3 = pow(momset->mu00, 3);
    double n11 = u11 / temp1;
    double n20 = u20 / temp1;
    double n02 = u02 / temp1;
    double n12 = u12 / temp2;
    double n21 = u21 / temp2;
    double n30 = u30 / temp2;
    double n03 = u03 / temp2;
    double n13 = u13 / temp3;
    double n31 = u31 / temp3;
    double n22 = u22 / temp3;
    double n40 = u40 / temp3;
    double n04 = u04 / temp3;

    // 计算 11 个 flusser moments。
    // 计算 flu1。
    momset->flu1 = n20 + n02;

    // 计算 flu2。
    double c21[2];
    c21[0] = n12 + n30;
    c21[1] = n03 + n21;
    double c12[2];
    c12[0] = n12 + n30;
    c12[1] = -n03 - n21;
    // c21 乘以 c12。
    double c21c12[2];
    complexMultiply(c21[0], c21[1], c21[0], c21[1], 
                    &c21c12[0], &c21c12[1]);
    momset->flu2 = c21c12[0];

    // 计算 flu3, flu4。
    double c20[2];
    c20[0] = n20 - n02;
    c20[1] = 2 * n11;
    // c12 的平方。
    double c12p2[2];
    complexMultiply(c12[0], c12[1], c12[0], c12[1], 
                    &c12p2[0], &c12p2[1]);
    // c20 乘以 c12 的平方。
    double c20c12p2[2];
    complexMultiply(c20[0], c20[1], c12p2[0], c12p2[1], 
                    &c20c12p2[0], &c20c12p2[1]);
    momset->flu3 = c20c12p2[0];
    momset->flu4 = c20c12p2[1];

    // 计算 flu5, flu6。
    double c30[2];
    c30[0] = n30 - 3 * n12;
    c30[1] = 3 * n21 - n03;
    // c12 的 3 次方。
    double c12p3[2];
    complexMultiply(c12[0], c12[1], c12p2[0], c12p2[1], 
                    &c12p3[0], &c12p3[1]);
    // c30 乘以 c12 的 3 次方。
    double c30c12p3[2];
    complexMultiply(c30[0], c30[1], c12p3[0], c12p3[1], 
                    &c30c12p3[0], &c30c12p3[1]);
    momset->flu5 = c30c12p3[0];
    momset->flu6 = c30c12p3[1];

    // 计算 flu7。
    momset->flu7 = n04 + 2 * n22 + n40;

    // 计算 flu8, flu9。
    double c31[2];
    c31[0] = n40 - n04;
    c31[1] = 2 * n13 + 2 * n31;
    // c31 乘以 c12 的平方。
    double c31c12p2[2];
    complexMultiply(c31[0], c31[1], c12p2[0], c12p2[1], 
                    &c31c12p2[0], &c31c12p2[1]);
    momset->flu8 = c31c12p2[0];
    momset->flu9 = c31c12p2[1];

    // 计算 flu10, flu11。
    double c40[2];
    c40[0] = n04 + n40 - 6 * n22;
    c40[1] = 4 * n13 + 4 * n31;
    // c12 的 4 次方。
    double c12p4[2];
    complexMultiply(c12[0], c12[1], c12p3[0], c12p3[1], 
                    &c12p4[0], &c12p4[1]);
    // c40 乘以 c12 的 4 次方。
    double c40c12p4[2];
    complexMultiply(c40[0], c40[1], c12p4[0], c12p4[1], 
                    &c40c12p4[0], &c40c12p4[1]);
    momset->flu10 = c40c12p4[0];
    momset->flu11 = c40c12p4[1];

    // 处理完毕，退出。
    return NO_ERROR;
}

