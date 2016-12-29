// CurveFluctuation.h
// 创建人：邱孝兵
//
// 曲线波动计算（CurveFluctuation）
// 功能说明：对一个曲线（Curve）的坐标集，根据外部设定参数，计算其波动特征。
// 该波动特征由偏离其平滑坐标最远的几个坐标点来确定。
//
// 修订历史：
// 2013年08月26日（邱孝兵）
//     初始版本

#ifndef __CURVEFLUCTUATION_H__
#define __CURVEFLUCTUATION_H__


#include "Curve.h"
#include "CurveFluctuPropers.h"


// 类：CurveFluctuation
// 继承自：无
// 对于给定的曲线（Curve）计算其波动特征，该波动特征使用曲线上偏离平滑曲线最远的
// 的若干个点来表征，计算结果包括这若干个点的横纵坐标和偏移的距离，同时还统计出
// 了所有点的偏移平均距离以及平均的偏移坐标等。
class CurveFluctuation {

public:

    // Host 成员方法：calcCurveFluctu（计算曲线波动特征）
    // 算法的主函数，首先并行计算出每个点偏离平滑后的曲线上对应点的偏移距离
    // 然后统计出偏移距离最大的若干个点，记录这些点的坐标和偏移距离，作为曲线的
    // 波动特征，同时还需要统计出曲线上每个点的平均偏移距离和偏移坐标。
    __host__ int                      // 返回值，函数是否正确执行，若函数
                                      // 正确执行返回 NO_ERROR
    calcCurveFluctu(
        Curve *incurve,               // 输入曲线 
        CurveFluctuPropers *inoutcfp  // 曲线波动特征属性
    );
};

#endif
