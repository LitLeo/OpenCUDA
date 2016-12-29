// SimpleRegionDetect.cu
// 实现图像的阈值分割

#include "SimpleRegionDetect.h"
#include "BilateralFilter.h"
#include "BilinearInterpolation.h"
#include "SmallestDirRect.h"
#include "ConnectRegion.h"
#include "FreckleFilter.h"
#include "Morphology.h"
#include "Image.h"
#include "ImageDiff.h"
#include "DownSampleImage.h"
#include "LocalCluster.h"
#include "TemplateFactory.h"
#include "Histogram.h"
#include "Threshold.h"

#include <iostream>
#include <stdio.h>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X    32
#define DEF_BLOCK_Y     8

// 宏：DEF_WRITE_IMG
// 定义是否将标记分割后的图像写入文件，默认关闭
// #define DEF_WRITE_IMG

// 宏：DEF_MAX_RATIO
// 定义长短径的最大比例，默认为 20，超出则认为不是所需的区域
#define DEF_MAX_RATIO  20


// Host 成员方法：hasValidRect（检测区域长短径函数）
__host__ int SimpleRegionDetect::hasValidRect(Image *inimg, bool *result) 
{
    // 若指针未初始化，返回空指针异常
    if (result == NULL)
        return NULL_POINTER;

    // 提取输入图像的子图像
    ImageCuda insubimgCud;
    int errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    // 若图像数据在设备端，直接退出
    if (insubimgCud.deviceId >= 0) {
        errcode = ImageBasicOp::copyToHost(inimg);
        if (errcode != NO_ERROR)
            return errcode;
    }
    int x1 = insubimgCud.imgMeta.width, x2 = 0, 
        y1 = insubimgCud.imgMeta.height, y2 = 0;
    for (int i = 0; i < insubimgCud.imgMeta.width; i++) {
        for (int j = 0; j < insubimgCud.imgMeta.height;j++) {
            // 计算坐标对应点的位置
            int inidx = j * insubimgCud.pitchBytes + i;
            // 像素值不为 0 时，比较其坐标信息
            if (insubimgCud.imgMeta.imgData[inidx]) {
                x1 = i < x1 ? i : x1;
                x2 = i > x2 ? i : x2;
                y1 = j < y1 ? j : y1;
                y2 = j > y2 ? j : y2;
            }
        }
    }
    // 矩形的宽度，保证为非负整数
    int width = x2 > x1 ? x2 - x1 : x1 - x2;
    // 矩形的高度
    int height = y2 > y1 ? y2 - y1 : y1 - y2;
    // 二者之中有一个为 0 时，找不到符合要求的矩形
    if (width == 0 || height == 0) {
        *result = false;
        return NO_ERROR;
    }
    
    // 计算长径和短径的比例
    int ratio = width > height ? width / height : height / width;
    // 返回最后的比较结果
    *result = (ratio <= DEF_MAX_RATIO) ? true : false;
        return NO_ERROR;
}

// Host 成员方法：cutImages（图像分割函数）
__host__ int SimpleRegionDetect::cutImages(Image *inimg, Image *outimg, 
        int closeSize, DirectedRect **rects, int *count) 
{
    // 直方图统计表
    unsigned int gram[256];
    // 错误代码
    int errcode;
    // 外接矩形的检出数量
    *count = 0;

    errcode = hist.histogram(inimg, gram);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 图像有效区域的实际宽度
    int actualwidth = outimg->roiX2 - outimg->roiX1;
    // 图像有效区域的实际高度
    int actualheight = outimg->roiY2 - outimg->roiY1;
    // 灰度分割的存储临时数据的图像
    Image* temp;
    errcode = ImageBasicOp::newImage(&temp);
    if (errcode != NO_ERROR) 
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(temp, actualwidth, 
                                                actualheight);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(temp);
        return errcode;
    }

    for (int i = 1; i < 256; i++) {
        // 若像素值对应的统计数大于 0，进行阈值分割
        if (gram[i] > 0) {
            // 将最大最小阈值设置为同一值，进行单阈值分割
            errcode = th.setMaxPixelVal(i);
            if (errcode != NO_ERROR) {
                // 释放申请的空间，防止内存泄露
                ImageBasicOp::deleteImage(temp);
                return errcode;
            }
            errcode = th.setMinPixelVal(i);
            if (errcode != NO_ERROR) {
                // 释放申请的空间，防止内存泄露
                ImageBasicOp::deleteImage(temp);
                return errcode;
            }
            errcode = th.threshold(inimg, temp, 0, 255);
            if (errcode != NO_ERROR) {
                // 释放申请的空间，防止内存泄露
                ImageBasicOp::deleteImage(temp);
                return errcode;
            }
            
            // 对分割后的图像进行 CLOSE 运算
            errcode = morph.close(temp, outimg);
            // 若运算失败，将模板放回
            if (errcode != NO_ERROR) {
                // 释放申请的空间，防止内存泄露
                ImageBasicOp::deleteImage(temp);
                TemplateFactory::putTemplate(tpl);
                return errcode;
            }

            // 检测包含区域矩形是否符合要求
            bool isvalid = false;
            hasValidRect(outimg, &isvalid);
            
#ifdef DEF_WRITE_IMG
            // 将符合要求的图像写入文件
            if (isvalid) {           
                // 格式化文件名，并将其保存至当前目录的 imgs 目录下
                char filename[128];
                sprintf(filename, "imgs/th_%d.bmp", i);
                ImageBasicOp::writeToFile(filename,  outimg); 
            }  
#endif
        
            // 对符合要求的图像检测其最小外接矩形
            if (isvalid) {
                // 使用最小外接矩形类检出，由于是二值图像，对前景区域检出即可
                errcode = sdr.smallestDirRect(outimg, rects[*count], true);
                if (errcode != NO_ERROR) {
                    // 释放申请的空间，防止内存泄露
                    ImageBasicOp::deleteImage(temp);
                    TemplateFactory::putTemplate(tpl);
                    return errcode;
                }
    
                // 外接矩形数目增加
                (*count)++;
            }
        }    
    }
    // 正常结束时也需将模板放回
    TemplateFactory::putTemplate(tpl);
    // 释放申请的空间，防止内存泄露
    ImageBasicOp::deleteImage(temp);
    return  NO_ERROR;
}

// Host 成员方法：detectRegion（区域检测）
__host__ int SimpleRegionDetect::detectRegion(
        Image *inimg, DirectedRect **regionsdetect, int *regioncount)
{
    // 若指针未初始化，返回空指针异常
    if (regionsdetect == NULL || regioncount == NULL)
        return NULL_POINTER;
    
    // 错误代码    
    int errcode;

    // 保存中间数据的临时图像，其中 tempimg1 和 tempimg2 交替使用
    Image *tempimg1, *tempimg2;
    
    errcode = ImageBasicOp::newImage(&tempimg1);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg1, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg1);
        return errcode;
    }

    errcode = ImageBasicOp::newImage(&tempimg2);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg2, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg2);
        return errcode;
    }

    int scale = this->getScale();
    // 若缩小倍数不合法，返回错误码退出
    if (scale < 1) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg1);
        ImageBasicOp::deleteImage(tempimg2);
        return INVALID_DATA;
    }

    // 调整大图的 roi 以接收缩小后的图像
    tempimg1->roiX2 = inimg->width / scale;
    tempimg1->roiY2 = inimg->height / scale;

    // 初始化图像缩小类，并调用其中的 Dominance 缩小方法
    errcode = downimg.dominanceDownSImg(inimg, tempimg1);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg1);
        ImageBasicOp::deleteImage(tempimg2);
        return errcode;
    }
   
    // 若 FreckleFilter 的滤波半径合法，则调用 FreckFilter 对缩小图像优化，需要
    // 注意的是若设置了 radius，需保证其 varThreshold 和 matchErrThreshold 也已
    // 正确设置
    if (this->getRadius() > 0) {
        // 此处使用 tempimg2 来接收可选算法的输出
        tempimg2->roiX2 = inimg->width / scale;
        tempimg2->roiY2 = inimg->height / scale;
        // 若调用出错则返回出错代码并退出
        errcode = freckle.freckleFilter(tempimg1, tempimg2);
        if (errcode != NO_ERROR) {
            // 释放申请的空间，防止内存泄露
            ImageBasicOp::deleteImage(tempimg1);
            ImageBasicOp::deleteImage(tempimg2);
            return errcode;
        }

        // 交换图像指针，这样不会对下一步的调用产生影响 
        Image *tempimg = tempimg1;
        tempimg1 = tempimg2;
        tempimg2 = tempimg;
    }
    // 将 roi 设置为正常大小，以接收上一步放大的图像
    tempimg2->roiX2 = inimg->width;
    tempimg2->roiY2 = inimg->height;
    // 使用双线性插值方式进行图像的放大
    errcode = biInterpo.setScale(scale);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg1);
        ImageBasicOp::deleteImage(tempimg2);
        return errcode;
    }
    // 需要根据上一步的参数来决定输入图像
    errcode = biInterpo.doInterpolation(tempimg1, tempimg2);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg1);
        ImageBasicOp::deleteImage(tempimg2);
        return errcode;
    }

    // 以插值放大后的图像为背景做图像的差分
    tempimg1->roiX2 = inimg->width;
    tempimg1->roiY2 = inimg->height;
    errcode = diff.imageDiff(inimg, tempimg2, tempimg1);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg1);
        ImageBasicOp::deleteImage(tempimg2);
        return errcode;
    }    

    // 对差分图像 调用 local clustering 算法去除其中的不明显区域
    errcode = cluster.localCluster(tempimg1, tempimg2);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg1);
        ImageBasicOp::deleteImage(tempimg2);
        return errcode;
    }

    // 若滤波半径以及重复次数均合法，调用双边滤波进行优化
    if (this->getFilterRadius() > 0 && this->getRepeat() > 0) {
        errcode = biFilter.doFilter(tempimg2, tempimg1);
        if (errcode != NO_ERROR) {
            // 释放申请的空间，防止内存泄露
            ImageBasicOp::deleteImage(tempimg1);
            ImageBasicOp::deleteImage(tempimg2);
            return errcode;
        }
        
        // 交换图像指针，这样不会对下一步的调用产生影响 
        Image *tempimg = tempimg1;
        tempimg1 = tempimg2;
        tempimg2 = tempimg;
    }
    
    // 调用连通域标记算法进行标记
    errcode = conn.connectRegion(tempimg2, tempimg1);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg1);
        ImageBasicOp::deleteImage(tempimg2);
        return errcode;
    }

    // 对标记后的图像进行分割并检出其最小有向外接矩形
    errcode = cutImages(tempimg1, tempimg2, closeSize, regionsdetect, 
                        regioncount);
    if (errcode != NO_ERROR) {
        // 释放申请的空间，防止内存泄露
        ImageBasicOp::deleteImage(tempimg1);
        ImageBasicOp::deleteImage(tempimg2);
        return errcode;
    }

    // 释放申请的空间，防止内存泄露
    ImageBasicOp::deleteImage(tempimg1);
    ImageBasicOp::deleteImage(tempimg2);
   
    return NO_ERROR;
}
