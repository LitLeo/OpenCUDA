// FillCoor.h
// 创建人：曹建立
//
// 填充 coordiset 集合围起的封闭区域（FillCoor）
// 功能说明：参数给出的 CoordiSet 集合是一条或多条闭合曲线，本类功能为将闭合
// 曲线围成的封闭区域内部用 FILL_COLOR 像素值填充。
// 使用并行算法时，输入的外轮廓集合和内轮廓集合要放入两个coordiset中。如果没
// 有内轮廓，内轮廓坐标集设为NULL。

// 修订历史:

#ifndef __FillCoor_H__
#define __FillCoor_H__

#include "Image.h"
#include "CoordiSet.h"
#include "ErrorCode.h"
#include "ImageDrawer.h"
#include "ImgConvert.h" 

#include<string>
using namespace std;

// 宏：BORDER_COLOR
// 定义边界颜色
#define BORDER_COLOR 255
// 宏：BK_COLOR
// 定义背景颜色
#define BK_COLOR 0

class FillCoor
{

public:


    // 成员方法：fillCoordiSetSeri（填充 coordiset 集合围起的区域）
    __host__ int                // 返回值：函数是否正确执行，若函数正确执
                                // 行，返回 NO_ERROR。
    seedScanLineSeri(
    CoordiSet *incoor,          // 输入的coordiset，内容为围起一个封闭区域的一条
                                // 或多条封闭曲线，允许区域有孔洞
    CoordiSet *outcoor,         // 输出填充过的的coordiset，内容为一个填充区域
    int x,                      // 种子x坐标
    int y                       // 种子y坐标
    );

    // 成员方法：isInCoordiSetSeri（判断当前点是否在 coordiset 集合围起的区域中
    //                          的串行算法）
    __host__ bool                // 返回值：在内部返回真，否则返回假
        isInCoordiSetSeri(
        CoordiSet *incoor,          // 输入的coordiset，内容为围起封闭区域的一
                                    // 条或多条封闭曲线，允许区域有孔洞
        int x,                      // 坐标点x坐标
        int y                       // 坐标点y坐标
        );

    // 成员方法：seedScanLineCon（并行种子扫描线算法填充 coordiset 集合围起的区域）
    // 使用本并行算法时，内外轮廓要放入不同的coordiset中。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确执
                                    // 行，返回 NO_ERROR。
    seedScanLineCon(
        CoordiSet *outbordercoor,          // 输入的 coordiset ，内容为封闭区域
                                           // 外轮廓闭合曲线
        CoordiSet *inbordercoor,           // 输入的 coordiset ，内容为封闭区域
                                           // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
        CoordiSet *fillcoor                // 输出填充过的的 coordiset 
        );

// 成员方法：seedScanLineCon（并行种子扫描线算法填充 coordiset 集合围起的区域）
__host__ int                // 返回值：函数是否正确执行，若函数正确执
                            // 行，返回 NO_ERROR。
    seedScanLineCon(
        Image *outborderimg,          // 外轮廓闭合曲线图像,同时也是输出结果
        Image *inborderimg            // 内轮廓闭合曲线图像，没有内轮廓设为空
    );
};
#endif
