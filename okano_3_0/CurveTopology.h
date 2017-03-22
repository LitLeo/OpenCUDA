// CurveTopology.h
// 创建者：欧阳翔
// 
// 曲线间的相位关系（CurveTopology）
// 功能说明：判断两个闭合曲线的关系，其中包括包含关系、被包含关系、相交关系和
//           无上述三种关系的其他关系，如果相交得到交点个数。

// 修订历史：
// 2013年10月27日（欧阳翔）
//     初始版本，曲线间的相位关系初步实现。
// 2013年11月10日（欧阳翔）
//     修复了设置曲线内部为白色的核函数的一处 Bug

#ifndef __CURVETOPOLOGY_H__
#define __CURVETOPOLOGY_H__

#include "Curve.h"
#include "ErrorCode.h"
#include "Image.h"

// 宏：CURVE_INCLUDE
// 表示包含关系
#define CURVE_INCLUDE          0

// 宏：CURVE_INCLUDE
// 表示被包含关系
#define CURVE_INCLUDED         1

// 宏：CURVE_INTERSECT
// 表示相交关系
#define CURVE_INTERSECT        2

// 宏：CURVE_OTHERSHIP
// 表示除上述三种外的其他关系
#define CURVE_OTHERSHIP        3


// 结构体：CurveRelation（曲线间的相位关系）
// 该结构体包含了两曲线之间的相位关系，如果相交还含有交点个数描述
typedef struct CurveRelation_st {
    int relation;  // 表示两条曲线的相位关系，通过计算得到的曲线之间的相位关系，
                   // 用宏来区分不同的关系，当最后关系是包含时设置 relation 为 
                   // CURVE_INCLUDE，是被包含时设置为 CURVE_INCLUDED，是相交
                   // 关系时设置为 CURVE_INTERSECT，没有上述三种关系时设置为 
                   // CURVE_OTHERSHIP，并且程序开始运行时，默认设置为 
                   // CURVE_OTHERSHIP。
    int internum;  // 如果曲线相交，得到交点个数，如果不是相交关系，则设置为 0，
                   // 程序初始默认设置为 0
} CurveRelation;


// 类：CurveTopology（曲线间的相位关系）
// 继承自：无
// 曲线间的相位关系：判断两个闭合曲线的关系，其中包括包含关系、被包含关系、
// 相交关系和无上述三种关系的其他关系，如果相交得到交点个数。默认是相对于第二条
// 曲线，第一条曲线的关系，表示包含关系时是第一条曲线包含第二条曲线；表示被包含
// 关系时是第二条曲线包含第一条曲线
class CurveTopology {

public:

    // 构造函数：CurveTopology
    // 无参数版本的构造函数，因为该类没有成员变量，所以该构造函数为空。
    // 没有需要设置默认值的成员变量。
    __host__ __device__
    CurveTopology() {}

    // Host 成员方法：curveTopology（曲线相位关系）
    // 对图像进行曲线跟踪，得到非闭合曲线和闭合曲线的有序序列
    __host__ int                 // 返回值：函数是否正确执行，若函数正确执行，返回
                                 // NO_ERROR。
    curveTopology(
            Curve *curve1,               // 输入第一条曲线
            Curve *curve2,               // 输入第二条曲线
            CurveRelation *crvrelation,  // 曲线相位关系
            int width,                   // 曲线对应的图像宽度
            int height                   // 曲线对应的图像高度
    ); 
};

#endif
