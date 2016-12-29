// FillCurve.h
// 创建人：曹建立
//
// 填充 coordiset 集合围起的封闭区域（FillCoor）
// 功能说明：参数给出的 CoordiSet 集合是一条或多条闭合曲线，本类功能为将闭合
// 曲线围成的封闭区域内部用 FILL_COLOR 像素值填充。
// 使用并行算法时，输入的外轮廓集合和内轮廓集合要放入两个coordiset中。如果没
// 有内轮廓，内轮廓坐标集设为NULL。

// 修订历史
// 2013年9月20日（曹建立）
//     初始版本
// 2013年11月19日（曹建立）
//     设计共享内存版本的填充内核函数，提高填充速度
// 2014年10月9日 （曹建立)
//      将全局内存、共享内存两个版本放入一个类中整合
#ifndef __FILLCURVE_H__
#define __FILLCURVE_H__

#include "Image.h"
#include "CoordiSet.h"
#include "ErrorCode.h"
#include "Curve.h"
#include "CurveConverter.h"
#include "ImgConvert.h" 

// 宏：BORDER_COLOR
// 定义边界颜色
#define BORDER_COLOR 255
// 宏：BK_COLOR
// 定义背景颜色
#define BK_COLOR 0

class FillCurve
{
    // 成员变量：maxThreadsPerBlock
    int maxThreadsPerBlock;
public:
    // 构造函数：FillCurveShared
    // 有参数版本的构造函数
    __host__ __device__
        FillCurve(
        int maxThreads
        ) {
            maxThreadsPerBlock=1024;
            maxThreadsPerBlock=maxThreads;
    }

    // 构造函数：FillCurveShared
    // 无参数版本的构造函数
    __host__ __device__
        FillCurve() {
            maxThreadsPerBlock=1024;
    }
    // 成员方法：getMaxThreads（获取 block 中最大线程数）
    // 获取成员变量 maxThreadsPerBlock 的值。
    __host__ int  // 返回值：成员变量 maxThreadsPerBlock 的值
        getMaxThreads(){
            // 返回 detheta 成员变量的值。
            return this->maxThreadsPerBlock;
    } 

    // 成员方法：setMaxThreads（设置 block 中最大线程数）
    // 设置成员变量 maxThreadsPerBlock 的值。
    __host__ int     // 返回值：函数是否正确执行，若函数正确执
        // 行，返回 NO_ERROR。
        setMaxThreads(int n){
            // 返回 detheta 成员变量的值。
            this->maxThreadsPerBlock=n;
            return NO_ERROR;
    } 

    // 成员方法：seedScanLineImgShr（并行种子扫描线算法填充 coordiset 集合围起的区域）
    __host__ int                        // 返回值：函数是否正确执行，若函数正确执
        // 行，返回 NO_ERROR。
        seedScanLineImgShr(
        Image *outborderimg,          // 外轮廓闭合曲线图像,同时也是输出结果
        Image *inborderimg            // 内轮廓闭合曲线图像，没有内轮廓设为空
        );


    // 成员方法：seedScanLineCoorShr（并行种子扫描线算法填充 coordiset 集合围起的区域）
    // 使用本并行算法时，内外轮廓要放入不同的coordiset中。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确执
        // 行，返回 NO_ERROR。
        seedScanLineCoorShr(
        CoordiSet *outbordercoor,          // 输入的 coordiset ，内容为封闭区域
        // 外轮廓闭合曲线
        CoordiSet *inbordercoor,           // 输入的 coordiset ，内容为封闭区域
        // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
        CoordiSet *fillcoor                // 输出填充过的的 coordiset 
        );


    // 成员方法：seedScanLineCurveShr（并行种子扫描线算法填充 Curve 集合围起的区域）
    // 使用本并行算法时，内外轮廓要放入不同的 Curve 中。该方法得到的填充后Curve
    // 类型结构fillcurve中仅curveX、curveY、minX、minY、maxX、maxY域为有效数据，
    // 其他域并未赋值。
    __host__ int                        // 返回值：函数是否正确执行，若函数正确执
        // 行，返回 NO_ERROR。
        seedScanLineCurveShr(
        Curve *outbordercurve,          // 输入的 Curve ，内容为封闭区域
        // 外轮廓闭合曲线
        Curve *inbordercurve,           // 输入的 Curve ，内容为封闭区域
        // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
        Curve *fillcurve                // 输出填充过的的 Curve 
        );

    // 成员方法：seedScanLineImgGlo（并行种子扫描线算法填充 coordiset 集合围起的区域）
    __host__ int                        // 返回值：函数是否正确执行，若函数正确执
                                      // 行，返回 NO_ERROR。
        seedScanLineImgGlo(
        Image *outborderimg,          // 外轮廓闭合曲线图像,同时也是输出结果
        Image *inborderimg            // 内轮廓闭合曲线图像，没有内轮廓设为空
        );


    // 成员方法：seedScanLineCoorGlo（并行种子扫描线算法填充 coordiset 集合围起的区域）
    // 使用本并行算法时，内外轮廓要放入不同的coordiset中。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确执
                                    // 行，返回 NO_ERROR。
    seedScanLineCoorGlo(
        CoordiSet *outbordercoor,          // 输入的 coordiset ，内容为封闭区域
                                           // 外轮廓闭合曲线
        CoordiSet *inbordercoor,           // 输入的 coordiset ，内容为封闭区域
                                           // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
        CoordiSet *fillcoor                // 输出填充过的的 coordiset 
        );


// 成员方法：seedScanLineCurveGlo（并行种子扫描线算法填充 Curve 集合围起的区域）
// 使用本并行算法时，内外轮廓要放入不同的 Curve 中。该方法得到的填充后Curve
//类型结构fillcurve中仅curveX、curveY、minX、minY、maxX、maxY域为有效数据，
//其他域并未赋值。
__host__ int                        // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    seedScanLineCurveGlo(
    Curve *outbordercurve,          // 输入的 Curve ，内容为封闭区域
                                           // 外轮廓闭合曲线
    Curve *inbordercurve,           // 输入的 Curve ，内容为封闭区域
                                           // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
    Curve *fillcurve                // 输出填充过的的 Curve 
    );

}; // end of class


#endif
