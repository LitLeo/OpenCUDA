// Morphology.h
// 创建人：罗劼
//
// 形态学图像算法（morphology algorithm）
// 功能说明：包含四种形态学图像算法，分别为：腐蚀，膨胀，开操作，闭操作
//
// 修订历史：
// 2012年09月03日（罗劼）
//     初始版本。
// 2012年09月05日（罗劼，于玉龙）
//     修改了一些注释和代码中的错误规范
// 2012年09月06日（罗劼，于玉龙）
//     增加了 ROI 子图的处理和代码中的错误规范
// 2012年09月23日（于玉龙,罗劼）
//     修改了计算默认模版的一处错误
// 2012年09月27日（罗劼，于玉龙）
//     修改了代码中的一处错误规范
// 2012年10月26日（罗劼）
//     gridsize 中的一处不合理的位置
// 2012年10月29日（于玉龙）
//     为开运算和闭运算增加了中间图像缓冲区。
//     修正了代码中部分潜在的 Bug。
// 2013年05月28日（于玉龙）
//     优化了算法实现技巧，提高了性能近。修正了代码中几处不合理的设计。

#ifndef __MORPHOLOGY_H__
#define __MORPHOLOGY_H__

#include "Image.h"
#include "Template.h"


// 类：Morphology（形态学图像算法）
// 继承自：无
// 包含四种形态学图像算法，分别为：腐蚀，膨胀，开操作，闭操作
class Morphology {

protected:

    // 成员变量：tpl（模板指针）
    // 在腐蚀和膨胀操作中需要通过它来指定图像中要处理的像素范围
    Template *tpl;

    // 成员变量：intermedImg（中间图像）
    // 该图像作为缓冲区，用于开闭运算，可以避免每次调用开闭运算都需要重新申请和
    // 释放内存空间，提高运算性能。该图像通过调整 ROI 尺寸来适应不同尺寸的图
    // 像。
    Image *intermedImg;

public:

    // 构造函数：Morphology
    // 传递模板指针，如果不传，则默认为空
    __host__ 
    Morphology(
            Template *tp = NULL  //腐蚀和膨胀操作需要使用到模板，默认为空
    );

    // 析构函数：~Morphology
    // 用于释放中间图像。
    __host__
    ~Morphology();
    
    // 成员方法：getTemplate
    // 获取模板指针，如果模板指针和默认模板指针相同，则返回空
    __host__ Template*           // 返回值：如果模板和默认模板指针相同，则返
                                 // 回空，否则返回模板指针
    getTemplate() const;

    // 成员方法：setTemplate
    // 设置模板指针，如果参数 tp 为空，这使用默认的模板
    __host__ int                 // 返回值：若函数正确执行，返回 NO_ERROR
    setTemplate(
            Template *tp         // 腐蚀和膨胀操作需要使用的模板
    );

    // 成员方法：erode
    // 对图像进行腐蚀操作，根据模板，找到附近最小的像素，作为当前像素的输出
    __host__ int             // 返回值：若函数正确执行，返回 NO_ERROR，否则返回
                             // 相应的错误码
    erode(
            Image *inimg,    // 输入图像
            Image *outimg    // 输出图像
    );

    // 成员方法：dilate
    // 对图像进行膨胀操作，根据模板，找到附近最大的像素，当作当前像素的输出
    __host__ int             // 返回值：若函数正确执行，返回 NO_ERROR，否则返回
                             // 相应的错误码
    dilate(
            Image *inimg,    // 输入图像
            Image *outimg    // 输出图像
    );

    // 成员方法：open
    // 对图像进行开操作，先对图像进行腐蚀，然后再进行膨胀
    __host__ int             // 返回值：若函数正确执行，返回 NO_ERROR，否则返回
                             // 相应的错误码
    open(
            Image *inimg,    // 输入图像
            Image *outimg    // 输出图像
    );

    // 成员方法：close
    // 对图像进行关操作，先对图像进行膨胀，然货再进行腐蚀
    __host__ int             // 返回值：若函数正确执行，返回 NO_ERROR，否则返回
                             // 相应的错误码
    close(
            Image *inimg,    // 输入图像
            Image *outimg    // 输出图像
    );
};

#endif

