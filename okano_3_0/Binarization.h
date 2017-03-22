// Binarization.h
// 创建人：仲思惠
//
// 多阈值二值化图像处理（Binarization）
// 功能说明：对灰度图像进行多阈值二值化处理，以 1 - 254 内的所有的灰度为阈值,
// 同时生成 254 个 2 值化结果(0 - 1)图像，并根据一定的基准选择最佳二值化结果。
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
// 2012年11月12日（仲思惠）
//     添加了对254幅图像进行选择的功能
// 2012年11月20日（侯怡婷、王媛媛）
//     修改了图像复制的问题
// 2012年12月15日（仲思惠）
//     修改了文件名称，由 MultiThreshold 改为 Binarization。
// 2013年01月03日（仲思惠）
//     根据新版本的需求分析，修改了选择最佳面积比的方法。
// 2013年01月04日（王媛媛、仲思惠）
//     修正了选择最佳面积比的方法。
// 2013年01月07日（侯怡婷、王媛媛）
//     使用共享内存和原子操作获取面积值，提高了算法的效率。
// 2013年05月16日（仲思惠）
//     添加了代码的注释，修改了格式错误。
#ifndef __BINARIZATION_H__
#define __BINARIZATION_H__

#include "Image.h"
#include "Binarize.h"

// 类：Binarization
// 继承自：无
// 对灰度图像进行多阈值二值化处理，以 1 - 254 内的所有的灰度为阈值,同时生成
// 254 个 2 值化结果(0 - 1)图像，并根据一定的基准选择最佳二值化结果。
class Binarization {

protected:

    // 成员变量：area（最佳面积）
    // 求得的最佳二值化结果的面积值。
    long long  area;

    // 成员变量：normalArea（标准面积）
    // 由用户指定的标准面积值，选择最佳面积比结果的基准。
    long long normalArea;

public:

    // 构造函数：Binarization
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。该函数没有任何内
    // 容。
    __host__ __device__
    Binarization(){       // 使用默认值为类的各个成员变量赋值。

       // 标准面积的默认值为 70000。
       this->normalArea = 70000;
    }

    // 构造函数：Binarization
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    Binarization(
            long long normalArea
    ) {
        this->normalArea = 70000;  // 标准面积（具体解释见成员变量）

        // 根据参数列表中的值设定成员变量的初值
        setNormalArea(normalArea);
    }

    // 成员方法：getArea（读取最佳面积）
    // 读取 area 成员变量的值。
    __host__ __device__ long long
    getArea() const
    {
        return this->area;
    }

    // 成员方法：getNormalArea（读取标准面积）
    // 读取 normalArea 成员变量的值。
    __host__ __device__ long long
    getNormalArea() const
    {
        return this->normalArea;
    }

    // 成员方法：setNormalArea（设置标准面积）
    // 设置 normalArea 成员变量的值。
    __host__ __device__ int
    setNormalArea(
            long long normalArea
    ) {
        // 检查输入参数是否合法
        if (normalArea < 0)
            return INVALID_DATA;
        this->normalArea = normalArea;
        return NO_ERROR;
    }

    // Host 成员方法：binarization（多阈值二值化处理flag=1）
    // 对输入图像进行多阈值二值化处理。以 1 - 254 内的所有的灰度为阈值,同时
    // 生成 254 个 2 值化结果(0 - 1)图像,并在254个结果中选择只满足
    // |标准面积-TEST图像上的OBJECT面积| / 标准面积 < areaRatio的二值化结果
    __host__ int                // 返回值：函数是否正确执行，若函数正
                                // 确执行，返回 NO_ERROR。
    binarization(
            Image *inimg,       // 输入图像
            Image *outimg,      // 输出图像
            float areaRatio     // 面积比
    );
};

#endif
