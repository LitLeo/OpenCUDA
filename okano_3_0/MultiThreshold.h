// MultiThreshold.h 
// 创建人：仲思惠
//
// 多阈值二值化图像生成（MultiThreshold）
// 功能说明：根据设定的阈值，对灰度图像进行二值化处理，得到二值图像。
//
// 修订历史：
// 2012年10月19日（仲思惠）
//     初始版本
// 2012年10月20日（王媛媛、仲思惠)
//     去除了多余的成员变量
// 2012年10月25日（于玉龙、王媛媛）
//     解决了同时生成 254 幅图像时图像无法输出的问题
// 2012年10月27日（仲思惠）
//     修正了代码的格式，添加了一些注释

#ifndef __MULTITHRESHOLD_H__
#define __MULTITHRESHOLD_H__

#include "Image.h"

// 类：MultiThreshold
// 继承自：无
// 对灰度图像进行多阈值二值化处理，以 1 - 254 内的所有的灰度为阈值,同时生成
// 254 个 2 值化结果(0 - 1)图像。
class MultiThreshold {

public:

    // 构造函数：Multithreshold
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。该函数没有任何内
    // 容。
    __host__ __device__
    MultiThreshold(){}    

    // Host 成员方法：multithreshold（多阈值二值化处理）
    // 对输入图像进行多阈值二值化处理。以 1 - 254 内的所有的灰度为阈值,同时
    // 生成 254 个 2 值化结果(0 - 1)图像。
    __host__ int                // 返回值：函数是否正确执行，若函数正
                                // 确执行，返回 NO_ERROR。
    multithreshold(
            Image *inimg,       // 输入图像
            Image *outimg[254]  // 输出图像集合
    );
};

#endif

