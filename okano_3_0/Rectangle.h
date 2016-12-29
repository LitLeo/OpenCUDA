// Rectangle.h 
// 创建人：刘瑶
//
// 矩形（Rectangle）
// 功能说明：定义 BoundingRect 和 SmallestDirRect 里用到的公共结构体。
//
// 修订历史：
// 2013年01月09日（刘瑶）
//     初始版本
// 2013年03月21日（刘瑶）
//     将公共结构体的命名由 BR_XXX 统一为 RECT_XXX

#ifndef __RECTANGLE_H__
#define __RECTANGLE_H__

#include "Image.h"
#include "ErrorCode.h"

// 结构体：Quadrangle（有向矩形）
// 该结构体定了有向矩形的数据结构，其中包含了矩形的方向角度，4个角的坐标。
typedef struct Quadrangle_st {
    float angle;       // 有向矩形的角度，即矩形的长边与水平方向的顺时针夹角
    int points[4][2];  // 有向矩形的四个角坐标，为左上，右上，右下，左下四个点   
} Quadrangle;

// 结构体：DirectedRect（有向矩形）
// 该结构体定了有向矩形的另一种数据结构，其中包含了矩形的方向角度，中心点坐标，
// 长边长度，短边长度。
typedef struct DirectedRect_st {
    float angle;           // 有向矩形的角度，即矩形的长边与水平方向的顺时针夹
                           // 角
    int centerPoint[2];    // 有向矩形的中心点坐标
    int length1, length2;  // 有向矩形的长边，短边长度，length1 ≧ length2
} DirectedRect;

// 结构体：RotationInfo（旋转矩阵信息）
// 该结构体定义了旋转矩阵的信息，包含了旋转角度，余弦值，正弦值。
typedef struct RotationInfo_st
{
    float radian;  // 旋转角度对应的弧度。
    float cos;     // 旋转角度的余弦值。
    float sin;     // 旋转角度的正弦值。
} RotationInfo;

// 结构体：BoundBox（包围矩形的边界坐标）
// 该结构体定义了包围矩形的边界坐标信息。坐标的数据类型为 float。
typedef struct BoundBox_st
{
    float left;    // 左边界坐标。
    float right;   // 右边界坐标。
    float top;     // 上边界坐标。
    float bottom;  // 下边界坐标。
} BoundBox;

// 结构体：BoundBoxInt（包围矩形的边界坐标）
// 该结构体定义了包围矩形的边界坐标信息。坐标的数据类型为 int。
typedef struct BoundBoxInt_st
{
    int left;    // 左边界坐标。
    int right;   // 右边界坐标。
    int top;     // 上边界坐标。
    int bottom;  // 下边界坐标。
} BoundBoxInt;

// 宏：M_PI
// π值。对于某些操作系统，M_PI可能没有定义，这里补充定义 M_PI。
#ifndef M_PI
#define M_PI 3.14159265359
#endif

// 宏：RECT_ROTATE_POINT
// 根据给定的旋转信息，计算旋转后的点坐标。
#define RECT_ROTATE_POINT(ptsor, pttar, rtinfo) do {  \
        (pttar)[0] = (ptsor)[0] * (rtinfo).cos -    \
                     (ptsor)[1] * (rtinfo).sin;     \
        (pttar)[1] = (ptsor)[0] * (rtinfo).sin +    \
                     (ptsor)[1] * (rtinfo).cos;     \
    } while (0)

#endif

// 宏：RECT_RAD_TO_DEG
// 从弧度转换为角度。
#define RECT_RAD_TO_DEG(rad) ((rad) * 180.0f / M_PI)

// 宏：RECT_DEG_TO_RAD
// 从角度转换为弧度。
#define RECT_DEG_TO_RAD(deg) ((deg) * M_PI / 180.0f)