// BilateralFilter.h
// 创建人：邓建平
//
// 双边滤波（BilateralFilter）
// 功能说明：根据给定的空域参数求出高斯距离表，同时根据范围参数（颜色域参数）求
//           出 0 到 255 的欧氏距离表。对处于滤波半径内的邻域像素进行扫描实现双
//           边滤波
//
// 修订历史：
// 2012年10月22日（邓建平）
//     初始版本
// 2012年11月06日（邓建平、于玉龙）
//     增加了滤波半径参数，加入了表空间释放函数，并修正了一些格式错误
// 2012年12月27日（邓建平、于玉龙）
//     将高斯表和欧氏距离存储在模板中，减少开销，提高模板数据的复用性

#ifndef __BILATERALFILTER_H__
#define __BILATERALFILTER_H__

#include "ErrorCode.h"
#include "Image.h"
#include "TemplateFactory.h"

// 宏：DEF_FILTER_RADIUS
// 定义了默认的双边滤波半径
#define DEF_FILTER_RANGE 20

// 类：BilateralFilter（双边滤波）
// 继承自：无
// 根据给定的空域参数求出高斯距离表，同时根据范围参数（颜色域参数）求出 0 到 
// 255 的欧氏距离表。对处于滤波半径内的邻域像素进行扫描实现双边滤波
class BilateralFilter {

protected:
    // 成员变量：gaussian（高斯表），euclid（欧氏距离表）
    // 高斯表和欧氏距离表数据均保存在模板中
    Template *gaussian, *euclid;

    // 成员变量：sigmaSpace（空域参数），sigmaRange（颜色域参数）
    // 空域参数采用高斯核来计算，颜色域参数是基于像素与中心像素的亮度差的差值的
    // 加权，相似的像素赋给较大的权值，不相似的赋予较小的权值
    float sigmaSpace, sigmaRange;
    
    // 成员变量：radius（滤波半径）
    int radius;

    // 成员变量：repeat（重复次数）
    // 该参数默认为 0 ，当设置的参数大于 0 时务必确保空域参数与颜色域参数均设置
    // 正确，若设置为非正数，则不对图像进行双边滤波
    int repeat;

public:

    // 构造函数：BilateralFilter
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    BilateralFilter()
    {
        // 使用默认值为类的各个成员变量赋值。
        sigmaSpace = 0.0f;  // 空域参数的默认值为 0
        sigmaRange = 0.0f;  // 颜色域参数的默认值为 0
        radius = 0;         // 滤波半径的默认值为 0
        repeat = 0;         // 滤波重复次数默认为 0
        gaussian = NULL;    // 高斯表模板指针默认为 NULL
        euclid = NULL;      // 欧氏距离表模板指针默认为 NULL
    }

    // 构造函数：BilateralFilter
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__
    BilateralFilter(
            float sigmaspace,  // 空域参数（具体解释见成员变量）
            float sigmarange,  // 颜色域参数（具体解释见成员变量）
            int radius,        // 滤波半径（具体解释见成员变量）
            int repeat         // 重复次数（具体解释见成员变量）
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->sigmaSpace = 0.0f;  // 空域参数的默认值为 0
        this->sigmaRange = 0.0f;  // 颜色域参数的默认值为 0
        this->radius = 0;         // 滤波半径的默认值为 0
        this->repeat = 0;         // 滤波重复次数默认为 0
        this->gaussian = NULL;    // 高斯表模板指针默认为 NULL
        this->euclid = NULL;      // 欧氏距离表模板指针默认为 NULL
        
        // 根据参数列表中的值设定成员变量的初值
        this->setSigmaSpace(sigmaspace);
        this->setSigmaRange(sigmarange);
        this->setRadius(radius);
        this->setRepeat(repeat);
    }

    // 析构函数：~BilateralFilter
    // 类的析构函数，在类销毁时被调用，在类销毁时将模板数据放回模板池
    __host__
    ~BilateralFilter()
    {
        // 若模板数据不为空，将其放回
        if (gaussian != NULL)
            TemplateFactory::putTemplate(gaussian);
        
        // 若模板数据不为空，将其放回
        if (euclid != NULL)
            TemplateFactory::putTemplate(euclid);
    }

    // 成员方法：getSigmaSpace（读取空域参数）
    // 读取 sigmaSpace 成员变量的值。
    __host__ float  // 返回: 当前 sigmaSpace 成员变量的值
    getSigmaSpace() const
    {
        // 返回 sigmaSpace 成员变量的值
        return this->sigmaSpace;  
    }

    // 成员方法：setSigmaSpace（设置空域参数）
    // 设置 sigmaSpace 成员变量的值。
    __host__ int   // 返回值：函数是否正确执行，若函数正确
                              // 执行，返回 NO_ERROR。
    setSigmaSpace(
            float sigmaspace  // 空域参数
    ) {

        // 若参数不合法，返回对应的错误代码并退出
        if (sigmaspace <= 0)
            return INVALID_DATA;
        // 将 sigmaSpace 成员变量赋成新值
        this->sigmaSpace = sigmaspace;

        // 若模板数据不为空，先将其放回再获取新模板
        if (gaussian != NULL)
            TemplateFactory::putTemplate(gaussian);
        // 获取高斯模板数据，由于模板采用环形存储，一次性获取最大可能的数据
        int errcode = TemplateFactory::getTemplate(&gaussian, TF_SHAPE_GAUSS, 
                                                   2 * DEF_FILTER_RANGE + 1, 
                                                   &sigmaSpace);
        if (errcode != NO_ERROR)
            return errcode;

        return NO_ERROR;                
    }

    // 成员方法：getSigmaRange（读取颜色域参数）
    // 读取 sigmaRange 成员变量的值。
    __host__ float  // 返回: 当前 sigmaRange 成员变量的值
    getSigmaRange() const
    {
        // 返回 sigmaRange 成员变量的值
        return this->sigmaRange;  
    }

    // 成员方法：setSigmaRange（设置颜色域参数）
    // 设置 sigmaRange 成员变量的值。
    __host__ int   // 返回值：函数是否正确执行，若函数正确
                              // 执行，返回 NO_ERROR。
    setSigmaRange(
            float sigmarange  // 颜色域参数
    ) {

        // 若参数不合法，返回对应的错误代码并退出
        if (sigmarange <= 0)
            return INVALID_DATA;
        // 将 sigmaRange 成员变量赋成新值 
        this->sigmaRange = sigmarange;

        // 若模板数据不为空，先将其放回再获取新模板
        if (euclid != NULL)
            TemplateFactory::putTemplate(euclid);
        
        // 获取欧式距离模板数据，由于像素值的差值在 0 - 255 范围内，其大小固定为
        // 256，该模板中的点信息即表示所有可能的像素值差值
        int errcode = TemplateFactory::getTemplate(&euclid, TF_SHAPE_EUCLIDE,
                                                   256, &sigmaRange); 
        if (errcode != NO_ERROR)
            return errcode;

        return NO_ERROR;               
    }

    // 成员方法：getRadius（读取滤波半径）
    // 读取 radius 成员变量的值。
    __host__ __device__ int  // 返回: 当前 radius 成员变量的值
    getRadius() const
    {   
        // 返回 radius 成员变量的值
        return this->radius;
    }

    // 成员方法：setRadius（设置滤波半径）
    // 设置 radius 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setRadius(
            int radius       // 滤波半径
    ) { 

        // 若半径超出允许范围不赋值直接退出
        if (radius <= 0 || radius > DEF_FILTER_RANGE)
            return INVALID_DATA;
        // 将 radius 成员变量赋成新值                       
        this->radius = radius;
        return NO_ERROR;
    }

    // 成员方法：getRepeat（读取迭代次数）
    // 读取 repeat 成员变量的值。
    __host__ __device__ int  // 返回: 当前 repeat 成员变量的值
    getRepeat() const
    {   
        // 返回 repeat 成员变量的值
        return this->repeat;
    }

    // 成员方法：setRepeat（设置迭代次数）
    // 设置 repeat 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setRepeat(
            int repeat       // 迭代次数
    ) { 
        // 若迭代次数超出允许范围不赋值直接退出
        if (repeat <= 0)
            return INVALID_DATA;
        // 将 repeat 成员变量赋成新值                       
        this->repeat = repeat;
        return NO_ERROR;
    }

    // Host 成员方法：doFilter（执行滤波）
    // 对图像进行双边滤波，inplace 版本
    __host__ int             // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    doFilter(
            Image *inoutimg  // 输入输出图像 
    );

    // Host 成员方法：doFilter（执行滤波）
    // 对图像进行双边滤波，outplace 版本
    __host__ int             // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    doFilter(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像 
    );

};

#endif

