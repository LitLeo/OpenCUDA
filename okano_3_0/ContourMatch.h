// ContourMatch.h
// 创建者：孙慧琪
// 
// 轮廓匹配（ContourMatch）
// 功能说明：根据曲线对轮廓进行匹配，按宽度objCurve->curveBandWidth对轮廓曲线周围设定，设
// 定值为 objCurve->ownerObjectsIndices[0] +100，如果曲线是闭合轮廓，则将内部设定为
// objCurve->ownerObjectsIndices[0]的值。


// 修订历史：
// 2014年8月28日（孙慧琪）
//     初始版本，函数方法类初步实现。
// 2014年9月02日（孙慧琪）
//     实现对轮廓周围曲线的设定。
// 2014年9月07日 (孙慧琪)
//     添加判断点在曲线轮廓内的函数功能。
// 2014年9月12日 (孙慧琪)
//     修改完善关于判断一个点是否在曲线轮廓内的算法的实现。
// 2014年9月18日 (孙慧琪)
//     使用多曲线测试，并尝试加入对多曲线处理的功能。
// 2014年9月22日 (孙慧琪)
//     增加流操作，实现多曲线的操作处理。


#ifndef __CONTOURMATCH_H__
#define __CONTOURMATCH_H__

#include "Image.h"
#include "ErrorCode.h"
#include "Curve.h"


// 类：ContourMatch(轮廓匹配)
// 继承自：无
// 根据曲线对轮廓进行匹配，按宽度objCurve->curveBandWidth对轮廓曲线周围设定，设
// 定值为 objCurve->ownerObjectsIndices[0] +100，如果曲线是闭合轮廓，则将内部设定为
// objCurve->ownerObjectsIndices[0]的值。

class ContourMatch {

public:

    // 构造函数： ContourMatch
    // 无参数版本的构造函数，因为该类没有成员变量，所以该构造函数为空。
    // 没有需要设置默认值的成员变量。
    __host__ __device__
    ContourMatch() {}

    // Host 成员方法：contourMatch（轮廓匹配）
    // 根据曲线对轮廓进行匹配。

 
    __host__ int              // 返回值：函数是否正确执行，若函数正确执行，返回
                              // NO_ERROR。
   contourMatch(
            Image *inimg,    // 输入图像
            Image **outimg,  // 输出图像组
			Curve **curve,   // 输入曲线组
			int num          // 曲线个数
    );
};

#endif

