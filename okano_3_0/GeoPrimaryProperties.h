// GeoPrimaryProperties.h
// 创建人：杨伟光
//
// 曲线的最小有向外接矩形定义（GeoPrimaryProperties）
// 功能说明：定义了曲线的最小有向外接矩形的数据结构。
//                                                                          
// 修订历史：
// 2013年10月06日（杨伟光）
//     初始版本。
// 2014年09月28日（杨伟光）
//     修改了 .cu 文件的规范。

#ifndef __GEOPRIMARYPROPERTIES_H__
#define __GEOPRIMARYPROPERTIES_H__

#include "ErrorCode.h"
#include "Moments.h"
                                                                                
                                                                                                                                                  
// 结构体：GeoPrimaryProperties（曲线的最小有向外接矩形）
// 该结构体定义了曲线的最小有向外接矩形的数据结构，其中包含了曲线的最小有向外接
// 矩形的数据和和曲线的最小有向外接矩形相关的逻辑参数，
typedef struct GeoPrimaryProperties_st {
    int mdrCenterX;        // MDR 的几何中心的 x 坐标。
    int mdrCenterY;        // MDR 的几何中心的 y 坐标。

    int contourArea;       // 轮廓所围面积，轮廓的曲线时为 0。
    int mdrLS;             // MDR 的长边长。
    int mdrSS;             // MDR 的短边长。
    float mdrAngle;        // MDR 的走向角。
    int *mdrVertexesX;     // MDR 的 4 个顶点坐标。 
    int *mdrVertexesY;     // MDR 的 4 个顶点坐标。

    int vertexesNum;       // convex hull 坐标对的个数。
    int *chCordiX;         // convex hull 顶点的 x 坐标集。 
    int *chCordiY;         // convex hull 顶点的 y 坐标集。

    int chDomainPixelNum;  // convex hull 所围领域的 pixel 个数。
    int *chDomainX;        // convex hull 所围领域的 x 坐标集。
    int *chDomainY;        // convex hull 所围领域的 y 坐标集。
    int chArea;            // convex hull 所围面积。

    Moments  moments;      // curve /contour moment.
} GeoPrimaryProperties;                                                                       
                                                                                
#endif

                                                                         