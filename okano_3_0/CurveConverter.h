// CurveConverter.h
// 创建人：曹建立
//
// 结构体 curve 和 Image 之间相互转换（CurveConverter）

// 修订历史
// 2013年9月20日（曹建立）
//     初始版本
#ifndef __CURVECONVERTER_H__
#define __CURVECONVERTER_H__

#include "Image.h"
#include "Curve.h"
#include "ErrorCode.h"



class CurveConverter
{
public:
    int bordercolor;
    int bkcolor;
    CurveConverter(int border,int bk){
        bordercolor=border;
        bkcolor=bk;
    }
    // 成员方法：Curve2Img(把 Curve 中的点集合绘制到指定图像上)
    __host__ int                     // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    curve2Img(
        Curve *curve,                   // 输入的结构体
        Image *img                   // 输出的图像
    );

    // 成员方法：Img2Curve（把图像上的点保存到 Curve 结构体中）
    __host__ int                     // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
        img2Curve(
        Image *img,                    // 输入的图像
        Curve *curve                     // 输出的结构体
        );
};

#endif
