// GaussianSmoothImage.h
// 创建人：高新凯
//
// 高斯平滑
// 功能说明：定义了高斯平滑。
//
// 修订历史：
// 2013年10月10日（高新凯）
//    初始版本。
//
// 2013年12月25日（高新凯）
//    优化并行策略
//
// 2014年10月11日（高新凯）
//    修改了文件编码格式
//

#ifndef __GAUSSIANSMOOTNIMAGE_H__
#define __GAUSSIANSMOOTNIMAGE_H__

#include "ErrorCode.h"
#include "Image.h"
#include "stdlib.h"
#include "stdio.h"
#include <iostream>
using namespace std;

//#include "Common.h"        
class GaussSmoothImage{

public:
    
    // 构造函数：GaussSmoothImage
    // 无参数版本的构造函数，因为该类没有成员变量，所以该构造函数为空。
    // 没有需要设置默认值的成员变量。
     __host__ __device__ 
     GaussSmoothImage(){
     }
    
    // Host 成员方法：gaussSmoothImage（普通高斯平滑函数）
    // 根据输入图像和平滑区域、平滑窗口尺寸等信息对图像进行高斯平滑
    __host__ int              // 返回值：函数是否正确执行，若函数正确执行，返回
                              // NO_ERROR。
    gaussSmoothImage(
        Image* origiImage,    // 输入图像
        int smWidth,          // 需要平滑的特定区域的宽度
        int smHeight,         // 需要平滑的特定区域的高度
        int smLocatX,         // 需要平滑的特定区域的左上角pixel在
        int smLocatY,         // 原始图像中的横、纵坐标.             
        int smWindowSize,     // 平滑窗口尺寸
        Image* gaussSmImage   // 平滑结果
    );
    
    // Host 成员方法：gaussSmoothImage（带mask的高斯平滑函数）
    // 此函数与普通高斯平滑图像同名，使用时应以参数进行区分
    // 根据输入图像和平滑区域、平滑窗口尺寸、mask图像等信息进行高斯平滑
    __host__ int              // 返回值：函数是否正确执行，若函数正确执行，返回
                              // NO_ERROR。
    gaussSmoothImage(
        Image* origiImage,    // 输入图像
        int smWidth,          // 需要平滑的特定区域的宽度
        int smHeight,         // 需要平滑的特定区域的高度
        int smLocatX,         // 需要平滑的特定区域的左上角pixel在
        int smLocatY,         // 原始图像中的横、纵坐标. 
        int smWindowSize,     // 平滑窗口尺寸
        Image* gaussSmImage,  // 平滑结果
        Image* maskImage,     // mask图像
        unsigned char mask    // mask值
    );

    //增加的GPU代码

    __host__ int              // 返回值：函数是否正确执行，若函数正确执行，返回
                              // NO_ERROR。
    gaussSmoothImageMultiGPU(
        Image* origiImage,    // 输入图像
        int smWidth,          // 需要平滑的特定区域的宽度
        int smHeight,         // 需要平滑的特定区域的高度
        int smLocatX,         // 需要平滑的特定区域的左上角pixel在
        int smLocatY,         // 原始图像中的横、纵坐标.             
        int smWindowSize,     // 平滑窗口尺寸
        Image* gaussSmImage   // 平滑结果
    );
    

    __host__ int 

    gaussSmoothImageMultiGPU(
        Image* origiImage,    // 输入图像
        int smWidth,          // 需要平滑的特定区域的宽度
        int smHeight,         // 需要平滑的特定区域的高度
        int smLocatX,         // 需要平滑的特定区域的左上角pixel在
        int smLocatY,         // 原始图像中的横、纵坐标. 
        int smWindowSize,     // 平滑窗口尺寸
        Image* gaussSmImage,  // 平滑结果
        Image* maskImage,     // mask图像
        unsigned char mask    // mask值  
    );

    //图像分割函数
    __host__ ImageCuda* imageCut(
        Image *img,
        int *realheight,
        int *realLocatY,
        int roiLocatY,
        int roiHeight
    );

};

#endif
