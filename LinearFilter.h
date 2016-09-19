//LinearFilter.h
// 创建人：王雪菲
//
// 线性滤波滤波（LinearFilter）
// 功能说明：根据给定模板，对邻域像素进行卷积运算，实现图像的线性滤波操作
//
// 修订历史：
// 2012年12月06日（王雪菲）
//     初始版本。
// 2012年12月12日（王雪菲）
//     规范了代码，修改了程序中的越界错误
// 2012年12月13日（王雪菲）
//     规范了代码，增加了对函数运行结果的检查,完成了部分计算简化
// 2012年12月14日（王雪菲）
//     增加了函数的中文名称，规范了代码       
// 2012年12月17日（王雪菲）
//     增加了滤波操作的两种实现方式
// 2012年12月18日（王雪菲）
//     改正了一处数据类型错误

#ifndef __LinearFilter_H__
#define __LinearFilter_H__

#include "Image.h"
#include "Template.h"
#include "Common.h"

// 宏：LNFT_COUNT_DIV
// 用于设置 LinearFilter 类中的 impType 成员变量，告知类的实例选用邻域像素总和
// 除以像素点个数的运算方法实现线性滤波
#define LNFT_COUNT_DIV   1

// 宏：LNFT_WEIGHT_DIV
// 用于设置 LinearFilter 类中的 impType 成员变量，告知类的实例选用邻域像素总和
// 除以像素点权重之和的运算方法实现线性滤波
#define LNFT_WEIGHT_DIV  2

// 宏：LNFT_NO_DIV
// 用于设置 LinearFilter 类中的 impType 成员变量，告知类的实例选用邻域像素
// 直接带权加和的运算方法实现线性滤波
#define LNFT_NO_DIV      3


// 类：LinearFilter（线性滤波算法）
// 继承自：无
// 根据给定模板，对邻域像素进行卷积运算，实现图像的线性滤波操作
class LinearFilter: public cudaCommon {

protected:

    // 成员变量：impType（实现类型）
    // 设定三种实现类型中的一种，在调用滤波函数的时候，使用对应的实现方式
    int impType;

    // 成员变量：tpl（模板指针）
    // 在滤波操作中需要通过它来指定图像中要处理的像素范围和滤波模板值
    Template *tpl;

public:

    // 构造函数：LinearFilter
    // 根据需要给定各个参数，若不传参数，
    // 默认滤波实现方式为除以像素点个数，默认模板为空
    __host__ 
    LinearFilter(
            int imptype = LNFT_COUNT_DIV,  // 线性滤波实现类型，
                                           // 默认为除以像素点个数 
            Template *tp = NULL            // 线性滤波操作需要使用
                                           // 到模板，默认为空              
    );
    
    // 成员方法：getImpType（读取实现类型）
    // 获取实现滤波操作的类型
     
    __host__ int                // 返回值：当前实现滤波操作的类型
    getImpType() const;
    
    // 成员方法：setImpType（设置实现类型）
    // 设置实现滤波操作的类型
    __host__ int                // 返回值：若函数正确执行，返回 NO_ERROR
    setImpType(   
            int imptype         // 新的滤波操作实现类型
    );
    
    // 成员方法：getTemplate（读取模板）
    // 获取模板指针，如果模板指针和默认模板指针相同，则返回空
     
    __host__ Template *         // 返回值：如果模板和默认模板指针相同，则返
                                // 回空，否则返回模板指针
    getTemplate() const;

    // 成员方法：setTemplate（设置模板）
    // 设置模板指针，如果参数 tp 为空，则使用默认的模板
    __host__ int                // 返回值：若函数正确执行，返回 NO_ERROR
    setTemplate(   
            Template *tp        // 滤波操作需要使用的模板
    );

    // 成员方法：linearFilter（执行线性滤波）
    // 对图像进行线性滤波操作，根据模板，计算出当前点的像素值并输出
    __host__ int                // 返回值：函数正确执行，返回 NO_ERROR，
                                // 否则返回相应的错误码
    linearFilter(
            Image *inimg,       // 输入图像
            Image *outimg       // 输出图像
    );

    // 成员方法：linearFilterMultiGPU（多GPU执行线性滤波）
    // 将图像划分给多个GPU进行线性滤波操作
    __host__ int                // 返回值：函数正确执行，返回 NO_ERROR，
                                // 否则返回相应的错误码
    linearFilterMultiGPU(
            Image *inimg,       // 输入图像
            Image *outimg       // 输出图像
    );
};

#endif
