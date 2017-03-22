// FreckleFilter.h
// 创建者：欧阳翔
// 
// 广义的中值滤波（Freckle Filter）
// 功能说明：具有两种不同的方法实现：（1）根据给定半径 radius 和
//           radius + 4 生成的拱形邻域内的方差大小，比较方差与外部给定的
//           varTh 大小，如果方差小于给定阈值，则进行领域灰度平均
//           赋值给新图像对应位置；（2）根据圆内和圆外灰度值序列的子段相
//           似度匹配得到最小匹配差进行中值滤波。如果最小匹配差小于外部给
//           定的 matchErrTh 的阈值，则实现另一种方式的中值平均。
//
// 修订历史：
// 2012年10月29日（欧阳翔）
//     初始版本，方差阈值法的实现。
// 2012年11月01日（欧阳翔，于玉龙）
//     更正了代码的一些注释规范。
// 2012年11月29日（欧阳翔）
//     更正了方差迭代算法的一处 bug
// 2012年12月03日（欧阳翔）
//     修改了一些注释错误
// 2012年12月04日（欧阳翔）
//     初步完成了第二种方法，相似度匹配法
// 2012年12月05日（欧阳翔）
//     修正了相似度计算的一处 Bug
// 2012年12月10日（欧阳翔，于玉龙）
//     借用了 TemplateFactory 中于玉龙的得到圆形模板算法，改进了
//     本程序中得到圆形模板的算法
// 2012年12月11日（欧阳翔）
//     通过模板工厂得到圆形和环形模板，不再使用自己写的圆形和环形模板
// 2012年12月12日（欧阳翔）
//     增加了宏 FRECKLE_VAR_TH 和 FRECKLE_MATCH_ERRTH，用于区分滤波
//     实现的两种方法，减少了相同代码量，增加了代码可维护性
// 2012年12月13日（欧阳翔，于玉龙）
//     更正了部分格式错误
// 2012年12月18日（欧阳翔）
//     修正了相似度计算的部分 Bug
// 2012年12月19日（欧阳翔）
//     改正了通过模板工厂得到环形模板的调用
// 2012年12月21日（欧阳翔）
//     更正了相似度匹配法的一处严重 Bug，修改了相似度计算过程
// 2012年12月28日（欧阳翔）
//     增加了开关宏 FRECKEL_OPEN 和 FRECKLE_CLOSE ，用于最后当累加次数为 0
//     赋值时是赋值原图像对应灰度值还是赋值 0。
// 2012年12月29日（欧阳翔）
//     修正了类的设计，增加了成员 method 和 select 属性
// 2013年01月02日（欧阳翔）
//     相似度匹配法中原来的一个线程处理四个点改成处理一个点，修正了重复计算
//     问题，减少了计算量，避免了两处可能除 0 的情况。
// 2013年03月24日（张丽洁）
//     修正了一处潜在的 bug。

#ifndef __FRECKLEFILTER_H__
#define __FRECKLEFILTER_H__

#include "Image.h"
#include "ErrorCode.h"
#include "Template.h"
#include "TemplateFactory.h"

// 宏：VAR_TH
// 表示该滤波处理将用方差阈值法
#define FRECKLE_VAR_TH          0

// 宏：MATCH_ERRTH
// 表示该滤波处理将用相似度匹配法
#define FRECKLE_MATCH_ERRTH     1

// 宏：FRECKLE_OPEN
// 打开宏，表示在最后输出赋值时赋原图像的对应灰度值
#define FRECKLE_OPEN            1

// 宏：FRECKLE_CLOSE 
// 关闭宏，表示在最后输出赋值时赋值赋灰度值 0
#define FRECKLE_CLOSE           0


// 类：FreckleFilter（广义的中值滤波）
// 继承自：无
// 广义的图像中值滤波。具有两种不同的方法实现：（1）根据给定半径 radius
// 和 radius + 4 生成的拱形邻域内的方差大小，比较方差与外部给定的 varTh
// 大小，如果方差小于给定阈值，则进行领域灰度平均赋值给新图像对应位置；（2）根
// 据圆内和圆外灰度值序列的子段相似度匹配得到最小匹配差进行中值滤波。如果最小配
// 配差大于外部给定的 matchErrTh 的阈值，则实现另一种方式的中值平均。
class FreckleFilter {

protected:

    // 成员变量：radius（圆形邻域参数）
    // 外部设定的圆形领域的半径大小。
    int radius;

    // 成员变量：varTh（方差阈值参数）
    // 外部设定的用于圆周上灰度值方差大小的比较阈值。
    float varTh;

    // 成员变量：matchErrTh（匹配差阈值参数）
    // 外部设定的用于序列的子段间最小匹配差的阈值。
    float matchErrTh;

    // 成员变量：length（匹配长度参数）
    // 外部设定的用于序列匹配计算长度。
    int length;

    // 成员变量：method （区分两种方法的调用参数）
    // 外部设定的用于选择方差阈值法还是相似度匹配法
    // 若值为 FRECKLE_VAR_TH 则表示方差阈值法的实现，
    // 若为 FRECKLE_MATCH_ERRTH 则表示相似度匹配法的滤波实现
    int method;

    // 成员变量：select（最后给输出图像赋值时的区分参数）
    // 外部设定，当最后计算结果是 0，给输出图像赋值时，如果 select
    // 值为 FRECKLE_OPEN 则应该赋值为原图像对应灰度值，如果为
    // FRECKLE_CLOSE 则赋值为 0
    int select;

public:

    // 构造函数：FreckleFilter
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    FreckleFilter()
    {
        // 使用默认值为类的各个成员变量赋值。
        radius = 1;               // 默认圆的半径大小为 1
        varTh = 0.0f;             // 圆周上灰度值方差大小的阈值默认为 0
        matchErrTh = 0.0f;        // 序列的子段间最小匹配差的阈值的默认值为 0.0
        length = 2;               // 匹配计算长度的默认值为 2
        method = FRECKLE_VAR_TH;  // 默认调用第一种方法方差阈值法
        select = FRECKLE_OPEN;    // 最后给输出图像赋值时遇到赋 0 时默认赋
                                  // 原图像对应灰度值
    }
   
    // 构造函数：FreckleFilter
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    FreckleFilter (
            int radius,        // 圆形邻域参数（具体解释见成员变量）
            float varTh,       // 方差阈值参数（具体解释见成员变量）
            float matchErrTh,  // 匹配差阈值参数（具体解释见成员变量）
            int length,        // 匹配长度参数（具体解释见成员变量）
            int method,        // 区分两种方法的调用参数（具体解释见成员变量）
            int select         // 在最后给输出图像赋值时的区分参数（具体解释见
                               // 成员变量）
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->radius = 1;               // 默认圆的半径大小为 1
        this->varTh = 0.0f;             // 圆周上灰度值方差大小的阈值默认为 0
        this->matchErrTh = 0.0f;        // 序列的子段间最小匹配差的阈值的默认
                                        // 值为 0.0
        this->length = 2;               // 匹配计算长度的默认值为 2
        this->method = FRECKLE_VAR_TH;  // 默认调用第一种方法方差阈值法
        this->select = FRECKLE_OPEN;    // 最后给输出图像赋值时遇到赋 0 时默认赋
                                        // 原图像对应灰度值

        // 根据参数列表中的值设定成员变量的初值
        this->setRadius(radius);
        this->setVarTh(varTh);
        this->setMatchErrTh(matchErrTh);
        this->setLength(length);
        this->setMethod(method);
        this->setSelect(select);
    }

    // 成员方法：getRadius（读取圆形邻域参数）
    // 读取 radius 成员变量的值。
    __host__ __device__ int  // 返回值：当前 radius 成员变量的值。
    getRadius() const
    {
        // 返回 radius 成员变量的值。
        return this->radius;
    }

    // 成员方法：setRadius（设置圆形邻域参数）
    // 设置 radius 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setRadius(
            int radius       // 新的圆形邻域参数
    ) {
        // 将 radius 成员变量赋成新值
        this->radius = radius;
        return NO_ERROR;
    }

    // 成员方法：getvarTh（读取方差阈值参数）
    // 读取 varTh 成员变量的值。
    __host__ __device__ float  // 返回值：当前 varTh 成员变量的值。
    getVarTh() const
    {
        // 返回 varTh 成员变量的值。
        return this->varTh;
    }

    // 成员方法：setvarTh（设置方差阈值参数）
    // 设置 varTh 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setVarTh(
            float varTh      // 新的方差阈值参数
    ) {
        // 将 varTh 成员变量赋成新值
        this->varTh = varTh;
        return NO_ERROR;
    }

    // 成员方法：getmatchErrTh（读取匹配差阈值参数）
    // 读取 matchErrTh 成员变量的值。
    __host__ __device__ float  // 返回值：当前 matchErrTh 成员变量的值。
    getMatchErrTh() const
    {
        // 返回 matchErrTh 成员变量的值。
        return this->matchErrTh;
    }

    // 成员方法：setmatchErrTh（设置匹配差阈值参数）
    // 设置 matchErrTh 成员变量的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数
                              // 正确执行，返回 NO_ERROR。
    setMatchErrTh(
            float matchErrTh  // 新的匹配差阈值参数
    ) {
        // 将 matchErrTh 成员变量赋成新值
        this->matchErrTh = matchErrTh;
        return NO_ERROR;
    }

    // 成员方法：getLength（读取匹配长度参数）
    // 读取 length 成员变量的值。
    __host__ __device__ int  // 返回值：当前 length 成员变量的值。
    getLength() const
    {
        // 返回 length 成员变量的值。
        return this->length;
    }

    // 成员方法：setLength（设置匹配长度参数）
    // 设置 length 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setLength(
            int length       // 新的匹配长度参数
    ) {
        // 将 length 成员变量赋成新值
        this->length = length;
        return NO_ERROR;
    }

    // 成员方法：getMethod（读取区分两种方法的调用参数）
    // 读取 method 成员变量的值。
    __host__ __device__ int  // 返回值：当前 method 成员变量的值。
                             // 若为 0（也即 FRECKLE_VAR_TH），表示调用
                             // 第一种方法（方差阈值法），若为 1（也即
                             // FRECKLE_MATCH_ERRTH)，则表示调用第二种方法
    getMethod() const
    {
        // 返回 method 成员变量的值。
        return this->method;
    }

    // 成员方法：setMethod（设置最后给输出图像赋值时的区分参数）
    // 设置 method 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMethod(
            int method       // 新的区分两种方法的调用参数
    ) {
        // 将 method 成员变量赋成新值
        this->method = method;
        return NO_ERROR;
    }

    // 成员方法：getSelect（读取最后给输出图像赋值时的区分参数）
    // 读取 method 成员变量的值。
    __host__ __device__ int  // 返回值：当前 length 成员变量的值。
                             // 若为 0（也即 FRECKLE_CLOSE），表示最后赋 0 值，
                             // 若为 1（也即 FRECKLE_OPEN)，则表示最后赋原图像
                             // 对应灰度值
    getSelect() const
    {
        // 返回 select 成员变量的值。
        return this->select;
    }

    // 成员方法：setSelect（设置最后给输出图像赋值时的区分参数）
    // 设置 select 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setSelect(
            int select       // 新的最后给输出图像赋值时的区分参数
    ) {
        // 将 select 成员变量赋成新值
        this->select = select;
        return NO_ERROR;
    }

    // Host 成员方法：freckleFilter（广义的中值滤波）
    // 对图像进行广义的中值滤波，可以实现方差阈值的实现方法，也可以实现相似度
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    freckleFilter(
            Image *inimg,  // 输入图像 
            Image *outimg  // 输出图像
    );
};

#endif

