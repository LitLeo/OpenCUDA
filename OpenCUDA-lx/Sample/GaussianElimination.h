// GaussianElimination.h
//
// 高斯消元法（GaussianElimination）
// 功能说明：通过高斯消元法，求出输入行数与列数相等的方阵的上三角方阵
            
#ifndef __GAUSSIANELIMINATION_H__
#define __GAUSSIANELIMINATION_H__                                              

#include "Matrix.h"
#include "ErrorCode.h"

// 类：GaussianElimination（求上三角矩阵）
// 继承自：无。
// 实现了矩阵的高斯消元法，将矩阵主对角线以下的所有元素值变成 0，即将矩阵转换成
// 上三角矩阵形式。
class GaussianElimination {

public:

    // 构造函数：GaussianElimination
    // 无参数版本的构造函数。
    __host__ __device__
    GaussianElimination() 
    { }

    // 成员方法：gaussianEliminate（求上三角矩阵 - 高斯消元法）
    // 对 n 维矩阵进行 n - 1 次循环，通过行矩阵变换依次把矩阵每一列主对角线以下
    // 的元素的值变为 0 ，从而将矩阵转换成上三角矩阵。
    __host__ int            // 返回值：函数是否正确执行，若函数正确执行，返回 
                            // NO_ERROR。
    gaussianEliminate(
            Matrix *inmat,  // 源矩阵，要求为方阵。
            Matrix *outmat  // 目标矩阵，源矩阵的上三角矩阵形式。
    );
};

#endif

