// HistogramDifference.h
// 创建人：邱孝兵
//
// 直方图差异（HistogramDifference）
// 功能说明：根据一个闭合的输入曲线，确定图像上在闭合曲线
// 内部的点，统计这些点的直方图，和输入的标准直方图之间计算差异。
// 
// 修订历史：
// 2013年10月08日（邱孝兵）
//     初始版本
// 2013年11月08日（邱孝兵）
//     修复切线上的点造成的 bug

#ifndef __HISTOGRAMDIFFERENCE_H__
#define __HISTOGRAMDIFFERENCE_H__

#include "ErrorCode.h"
#include "Curve.h"
#include "Image.h"

// 类：HistogramDifference
// 继承自：无
// 根据一个闭合的输入曲线，确定图像上在闭合曲线
// 内部的点，统计这些点的直方图，和输入的标准直方图之间计算差异。
class HistogramDifference {

public:

    // Host 成员方法：histogramDiff（直方图差异）
    // 根据一个闭合的输入曲线，确定图像上在闭合曲线
    // 内部的点，统计这些点的直方图，和输入的标准直方图之间计算差异。
    __host__ int                    // 返回值：函数是否正确执行，若正确
                                    // 执行，返回 NO_ERROR
    histogramDiff(
            Curve *incurve,         // 输入闭合曲线
            Image *inimg,           // 输入图像
            float *referhistogram,  // 参考直方图
            float &chisquarehd,     // 卡方直方图差异   
            float &intersecthd      // 交叉直方图差异
	);
};

#endif
 