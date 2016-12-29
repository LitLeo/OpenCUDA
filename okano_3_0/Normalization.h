// Normalization.h
// 创建人：罗劼
//
// 图像正规化（Image Normalization）
// 功能说明：对一张图像进行正规化
//
// 修订历史：
// 2012年10月31日（罗劼）
//     初始版本
// 2012年10月31日（罗劼，于玉龙）
//     修改了多处代码不规范的地方以及 normalize 函数的设计
// 2012年11月19日（罗劼）
//     修改了将正规化结果从 Device 拷贝到 Host 上的一处错误
// 2012年11月29日（罗劼）
//     修改了一处计算方差的错误
// 2013年05月07日（罗劼）
//     修改了一处计算块数量的错误

#ifndef __NORMALIZATION_H__
#define __NORMALIZATION_H__ 

#include "ErrorCode.h"
#include "Image.h"

// 类：Normalization
// 继承自：无
// 对一张图像进行正规化操作，对每一个像素值，以该像素为中心，求出该邻域内的平均
// 值和标准差，将该像素值减平均值，得到的差再除以标准差，得到的结果就是该点的正
// 规化值
class Normalization {

protected:

    // 成员变量：k（邻域的大小）
    // 用来指定每个点邻域的大小
    int k;

public:

    // 构造函数：Normalization
    // 无参版本的构造函数，所有成员初始化为默认值
    __host__ __device__
    Normalization()
    {
        k = 3;             // 邻域大小默认为 3
    }

    // 构造函数：Normalization
    // 有参版本的构造函数，根据需要，参数可以在程序的运行过程中改变
    __host__ __device__
    Normalization(
            int k  // 邻域的大小
    ) {
        // 使用默认值初始化成员
        this->k = 3;

        // 根据参数列表中的值设定成员变量的初值
        this->setK(k);
    }

    // 成员方法：getK（获取邻域的大小）
    // 获取 k 的值
    __host__ __device__ int  // 返回值：成员变量 k 的值
    getK() const
    {
        // 返回设置的领域的大小
        return this->k;
    }

    // 成员方法：setK（设置邻域的大小）
    // 设置 k 的值
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR 
    setK(
            int k            // 邻域的大小
    ) {
        // 判断 k 是否是负数，是则返回错误
        if (k < 0)
            return INVALID_DATA;

        // 设置邻域大小的值
        this->k = k;

        return NO_ERROR;
    }

    // 成员方法：normalize（对输入图像进行正规化）
    // 对输入图像进行正规化，取 inimg 和 out 中范围更小的，结果存放在 Host 
    // 内存中
    inline __host__ int    // 返回值：函数是否正确执行，如果正确执行，返
                           // 回 NO_ERROR
    normalize(
            Image *inimg,  // 输入图像
            float *out,    // 输出每个点正规化的画像值
            int width,     // out 的宽
            int height,    // out 的高
            bool ishost    // 标记 out 是否指向 host 内存区域
    ) {
        // 将 pitch 设置为 width * sizeof (flaot)，此时 out 是不带 pitch 的
        return this->normalize(inimg, out, width * sizeof (float),
                               width,height,ishost);
    }

    // 成员方法：normalize（对输入图像进行正规化）
    // 对输入图像进行正规化，取 inimg 和 out 中范围更小的，结果存放在
    // Device 内存中
    __host__ int           // 返回值：函数是否正确执行，如果正确执行，返回
                           // NO_ERROR
    normalize(
            Image *inimg,  // 输入图像
            float *out,    // 输出每个点正规化像素值
            size_t pitch,  // out 的 pitch 值
            int width,     // out 的宽
            int height,    // out 的高
            bool ishost    // 标记 out 是否指向 host 内存区域
    );
};

#endif

