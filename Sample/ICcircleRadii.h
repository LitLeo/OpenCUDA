// ICcircleRadii
// 
// 最远距离最小的点 与 最近距离最大的点
// 功能说明：对一个给定轮廓的坐标集，求其所包围领域内的一点
//           最远距离最小的点-----外接圆的半径
//           最近距离最大的点-----内接圆的半径


#ifndef __ICCIRCLERADII_H__
#define __ICCIRCLERADII_H__

#include"Image.h"
#include"ErrorCode.h"

// 类：ICcircleRadii
// 继承自：无
// 该类包含了求出对一个给定轮廓的坐标集，求其所包围领域内的一点的操作。包含两个
// 方法，分别为求最远距离最小的点（外接圆的半径）和最近距离最大的点（内接圆的半
// 径
class ICcircleRadii {

public:

    // 构造函数：ICcircleRadii
    // 无参数构造函数
    __host__ __device__
    ICcircleRadii()
    {
        // 空构造函数
    }

    // 成员方法：minMax
    // 对给定轮廓的坐标集，求最远距离最小的点（外接圆的半径）
    __host__ int               // 返回值：若寒数执行正确，返回 NO_ERROR，否则返回
                               // 相应的错误代码
    minMax(
            Image *inimg,      // 输入图像，即给定坐标集轮廓
            int pickNum,       // 输入选择的距离最远最小的点的数量
            int *minMaxDist,   // 输出所选择的点
			int *minMaxIndexX, // 输出所选择的点对应的 X 坐标
			int *minMaxIndexY  // 输出所选择的点对应的 Y 坐标
	);

    // 成员方法：maxMin
    // 对给定轮廓的坐标集，求最近距离最大的点（内接圆的半径）
    __host__ int               // 返回值：若寒数执行正确，返回 NO_ERROR，否则返回
                               // 相应的错误代码
    maxMin(
            Image *inimg,      // 输入图像，即给定坐标集轮廓
            int pickNum,       // 输入选择的最近距离最大的点的数量
            int *minMaxDist,   // 输出所选择的点
			int *minMaxIndexX, // 输出所选择的点对应的 X 坐标
			int *minMaxIndexY  // 输出所选择的点对应的 Y 坐标
   );
};

#endif

