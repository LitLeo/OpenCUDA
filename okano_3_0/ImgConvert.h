// ImgConvert.h
// 创建人：杨伟光
//
// 图像与坐标集的互相转化，曲线转化成图像（ImgConvert）
// 功能说明：实现了图像与坐标集互相转化和曲线转化成图像的一些接口。
//  
// 修订历史：
// 2013年03月18日（杨伟光）
//     初始版本。
// 2013年03月20日（杨伟光）
//     增加了识别图像中高像素的四种模式。
// 2013年03月23日（于玉龙）
//     修改了算法 CLASS 的设计，使其能够适应更加复杂的情况。
// 2013年03月24日（于玉龙）
//     实现了并行的图像到坐标点集的转化算法。修正了代码实现中的潜在 Bug。
// 2013年09月21日（于玉龙）
//     修正了调用 SCAN 算法时的一处 BUG。
// 2013年10月13日（杨伟光）
//     添加了曲线转化成图像功能。
// 2014年09月20日（杨伟光）
//      改进了算法中初始化图像的逻辑。


#ifndef __IMGCONVERT_H__
#define __IMGCONVERT_H__

#include "ErrorCode.h"
#include "Image.h"
#include "CoordiSet.h"
#include "ScanArray.h"
#include "Curve.h"


// 宏：IC_FLAG_CNT
// 转换标志位数组的尺寸，该值为 unsigned char 型数据所能表示的数据数量的总和。
#define IC_FLAG_CNT  (1 << (sizeof (unsigned char) * 8))


// 类：ImgConvert（图像与坐标集的互相转化算法）
// 继承自：无。
// 实现了图像与坐标集的互相转化算法。
class ImgConvert {

protected:

    // 成员变量：highPixel（高像素）
    // 图像内高像素的像素值，可由用户定义。
    unsigned char highPixel;
    
    // 成员变量：lowPixel（低像素）
    // 图像内低像素的像素值，可由用户定义。
    unsigned char lowPixel;

    // 成员变量：convertFlags（转换标志位）
    // 表示那些像素值可以在图像转换为坐标集时记录到坐标集中。该数组中对应位为 
    // true 的像素值在图像转换成坐标集时会被记录到坐标集中。
    bool convertFlags[IC_FLAG_CNT];

    // 成员变量：aryScan（扫描累加器）
    // 完成非分段扫描，主要在迭代计算凸壳点的过程中用于计算各个标记值所对应的累
    // 加值。
    ScanArray aryScan;

public:

    // 构造函数：ImgConvert
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    ImgConvert()
    {
        this->highPixel = 255;  // 高像素值默认为 255。
        this->lowPixel = 0;     // 低像素值默认为 0。

        // 初始化转换标志位，其中大于等于 50% 灰度的像素值被置为有效像素值。
        for (int i = 0; i < IC_FLAG_CNT / 2; i++)
            convertFlags[i] = false;
        for (int i = IC_FLAG_CNT / 2; i < IC_FLAG_CNT; i++)
            convertFlags[i] = true;

        // 配置扫描累加器。
        this->aryScan.setScanType(NAIVE_SCAN);
    }
    
    // 构造函数：ImgConvert
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    ImgConvert(
            unsigned char highpixel,  // 高像素 
            unsigned char lowpixel    // 低像素
    ) {
        this->highPixel = 255;        // 高像素值默认为 255。
        this->lowPixel = 0;           // 低像素值默认为 0。

        // 初始化转换标志位，其中大于等于 50% 灰度的像素值被置为有效像素值。
        for (int i = 0; i < IC_FLAG_CNT / 2; i++)
            convertFlags[i] = false;
        for (int i = IC_FLAG_CNT / 2; i < IC_FLAG_CNT; i++)
            convertFlags[i] = true;

        // 配置扫描累加器。
        this->aryScan.setScanType(NAIVE_SCAN);

        // 根据参数列表中的值设定成员变量的初值。
        this->setHighPixel(highpixel);
        this->setLowPixel(lowpixel);
    }

    // 成员方法：getHighPixel（获取高像素的值）
    // 获取成员变量 highPixel 的值。
    __host__ __device__ unsigned char  // 返回值：返回 hignPixel 的值。
    getHighPixel() const
    { 
        // 返回 highPixel 成员变量的值。
        return this->highPixel;   
    }

    // 成员方法：setHighPixel（设置高像素）
    // 设置成员变量 highPixel 的值。
    __host__ __device__ int          // 返回值：若函数正确执行，返回 NO_ERROR。
    setHighPixel(                                                       
            unsigned char highpixel  // 高像素的像素值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (highpixel == this->lowPixel)
            return INVALID_DATA;

        // 将 highPixel 成员变量赋成新值
        this->highPixel = highpixel;

        return NO_ERROR;
    }

    // 成员方法：getLowPixel（获取低像素的值）
    // 获取成员变量 lowPixel 的值。
    __host__ __device__ unsigned char  // 返回值：返回 lowPixel 的值。
    getLowPixel() const
    { 
        // 返回 lowPixel 成员变量的值。
        return this->lowPixel;   
    }

    // 成员方法：setLowPixel（设置低像素）
    // 设置成员变量 lowPixel 的值。
    __host__ __device__ int         // 返回值：若函数正确执行，返回 NO_ERROR。
    setLowPixel(
            unsigned char lowpixel  // 低像素的像素值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (lowpixel == this->highPixel)
            return INVALID_DATA;

        // 将 lowPixel 成员变量赋成新值。
        this->lowPixel = lowpixel;

        return NO_ERROR;
    }

    // 成员方法：setHighLowPixel（设置高低像素）
    // 设置成员变量 highPixel 和 lowPixel 的值。
    __host__ __device__ int           // 返回值：函数正确执行，返回 NO_ERROR。
    setHighLowPixel(
            unsigned char highpixel,  // 高像素的像素值
            unsigned char lowpixel    // 低像素的像素值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (highpixel == lowpixel)
            return INVALID_DATA;

        // 将 highPixel 和 lowPixel 成员变量赋成新值。
        this->highPixel = highpixel;
        this->lowPixel = lowpixel;

        return NO_ERROR;
    }

    // 成员方法：getConvertFlag（获取某像素值是否为有效像素）
    // 获取指定像素值是否在转化为坐标集时写入坐标集中。
    __host__ __device__ bool     // 返回值：该像素值是否为有效像素。
    getConvertFlag(
            unsigned char pixel  // 给定的像素值
    ) const
    {
        // 返回成员变量 convertFlags 中对应的元素。
        return this->convertFlags[pixel];
    }

    // 成员方法：clearAllConvertFlags（清楚所有转换标志位）
    // 清除所有转换标志位。
    __host__ __device__ int  // 返回值：函数正确执行，返回 NO_ERROR。
    clearAllConvertFlags()
    {
        // 将所有的转换标志位都设为 false。
        for (int i = 0; i < IC_FLAG_CNT; i++)
            this->convertFlags[i] = false;

        // 处理完毕，退出。
        return NO_ERROR;
    }

    // 成员方法：setAllConvertFlags（置位所有转换标志位）
    // 设置所有的转换标志位。
    __host__ __device__ int  // 返回值：函数正确执行，返回 NO_ERROR。
    setAllConvertFlags()
    {
        // 将所有的转换标志位都设为 true。
        for (int i = 0; i < IC_FLAG_CNT; i++)
            this->convertFlags[i] = true;

        // 处理完毕，退出。
        return NO_ERROR;
    }

    // 成员方法：clearConvertFlag（清除某个转换标志位）
    // 清除指定的某一个转换标志位。
    __host__ __device__ int     // 返回值：函数正确执行，返回 NO_ERROR。
    clearConvertFlag(
            unsigned char pixel // 给定的像素值
    ) {
        // 将指定的转换标志位置为 false。
        this->convertFlags[pixel] = false;

        // 处理完毕退出。
        return NO_ERROR;
    }

    // 成员方法：setConvertFlag（置位某个转换标志位）
    // 设置指定的某一个转换标志位。
    __host__ __device__ int     // 返回值：函数正确执行，返回 NO_ERROR。
    setConvertFlag(
            unsigned char pixel // 给定的像素值
    ) {
        // 将指定的转换标志位置为 true。
        this->convertFlags[pixel] = true;

        // 处理完毕退出。
        return NO_ERROR;
    }

    // 成员方法：clearConvertFlags（清除某范围内转换标志位）
    // 清除指定的某范围内转换标志位。
    __host__ __device__ int     // 返回值：函数正确执行，返回 NO_ERROR。
    clearConvertFlags(
            unsigned char low,  // 范围下界（含）
            unsigned char high  // 范围上界（含）
    ) {
        // 如果给定的下界大于上界，则报错退出
        if (high < low)
            return INVALID_DATA;

        // 将指定范围内的转换标志位置为 false。
        for (int i = low; i <= high; i++)
            this->convertFlags[i] = false;

        // 处理完毕退出。
        return NO_ERROR;
    }

    // 成员方法：setConvertFlags（置位某范围内转换标志位）
    // 设置指定的某范围内转换标志位。
    __host__ __device__ int     // 返回值：函数正确执行，返回 NO_ERROR。
    setConvertFlags(
            unsigned char low,  // 范围下界（含）
            unsigned char high  // 范围上界（含）
    ) {
        // 如果给定的下界大于上界，则报错退出
        if (high < low)
            return INVALID_DATA;

        // 将指定范围内的转换标志位置为 true。
        for (int i = low; i <= high; i++)
            this->convertFlags[i] = true;
        
        // 处理完毕退出。
        return NO_ERROR;
    }

    // Host 成员方法：imgConvertToCst（图像转化成坐标集算法）
    // 将图像中的高像素值点存储到坐标集中从而实现图像与坐标集的转化
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    imgConvertToCst(
            Image *inimg,      // 输入图像
            CoordiSet *outcst  // 输出坐标集
    );

    // Host 成员方法：cstConvertToImg（坐标集转化为图像算法）
    // 将坐标集内的坐标映射到输出图像中并将目标点置为 highpixel 从而实现
    // 坐标集与图像的转化。
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回NO_ERROR。
    cstConvertToImg(
            CoordiSet *incst,  // 输入坐标集
            Image *outimg      // 输出图像
    );

    // Host 成员方法：curConvertToImg（曲线转化为图像算法）
    // 将曲线内的坐标映射到输出图像中并将目标点置为 highpixel 从而实现
    // 曲线与图像的转化。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，
                           // 返回NO_ERROR。
    curConvertToImg(
            Curve *incur,  // 输入曲线
            Image *outimg  // 输出图像
    );
};

#endif

