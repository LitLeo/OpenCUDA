// HistogramSpec.h
// 创建人：侯怡婷
// 
// 直方图规定化（HistogramSpec）
// 功能说明：实现输入图像的直方图规定化，对于输入图像，规定化之后不改变原图
//           像尺寸大小。该算法有三种不同的实现：（1）直方图均衡化；（2）根
//           据参考图像对输入图像进行规定化；（3）根据给定直方图对输入图像
//           进行规定化。
//
// 修订历史：
// 2012年09月01日（侯怡婷）
//     初始版本。
// 2012年09月05日（刘瑶、侯怡婷）
//     更正了代码中对 ROI 处理的一些错误。
// 2012年10月25日（侯怡婷）
//     更正了代码的一些注释规范。
// 2012年11月15日（侯怡婷）
//     在核函数执行后添加 cudaGetLastError 判断语句。
// 2012年11月23日（侯怡婷）
//     增加对类成员变量 RefHisto 有效性判断。

#ifndef __HISTOGRAMSPEC_H__
#define __HSITOGRAMSPEC_H__

#include "Image.h"
#include "Histogram.h"
#include "ErrorCode.h"


// 类：HistogramSpec（直方图规定化类）
// 继承自：无
// 实现输入图像的直方图规定化，对于输入图像，规定化之后不改变原图像尺寸大小。
// 该算法有三种不同的实现：（1）直方图均衡化；（2）根据参考图像对输入图像进行
// 规定化；（3）根据给定直方图对输入图像进行规定化。
class HistogramSpec{

protected:

    // 成员变量：refimg（参考图像）
    // 参考图像，根据参考图像的直方图对输入图像进行规定化。
    Image *refimg;              

    // 成员变量：refhistogram（规定直方图）
    // 规定直方图，根据该给定的直方图对输入图像进行规定化。
    float *refHisto;  

public:

    // 构造函数：HistogramSpec
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__  __device__     
    HistogramSpec()
    {
        // 使用默认值为类的各个成员变量赋值。
        refimg = NULL;    // 参考图像默认为空。
        refHisto = NULL;  // 参考直方图数组默认为空。
    }  

    // 构造函数：HistogramSpec
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。 
    __host__  __device__  
    HistogramSpec(
            Image *refimg,   // 形状特征值数组
            float *refHisto  // 特征数组中键值对的个数
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法的初
        // 始值而使系统进入一个未知的状态。
        this->refimg = NULL;    // 参考图像默认为空。
        this->refHisto = NULL;  // 参考直方图数组默认为空。

        // 根据参数列表中的值设定成员变量的初值。
        setRefimg(refimg);
        setRefHisto(refHisto);
    }

    // 成员方法：getRefimg（读取参考图像）
    // 读取 refimg 成员变量的值。
    __host__ __device__ Image *  // 返回值：当前 refimg 成员变量的值。
    getRefimg() const
    {
        // 返回 refimg 成员变量的值。
        return this->refimg;
    }	

    // 成员方法：setRefimg（设置参考图像）
    // 设置 refimg成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setRefimg(                
            Image *refimg    // 指定的特征值数组。
    ) {
        // 检查输入参数是否合法
        if (refimg == NULL)
            return INVALID_DATA;

        // 将 refimg 成员变量赋成新值
        this->refimg = refimg;

        return NO_ERROR;
    }
                                                                                               
    // 成员方法：getRefHisto（读取给定的参考直方图）
    // 读取 refHisto 成员变量的值。
    __host__ __device__ float *  // 返回值：当前 refHisto 成员变量的值。
    getRefHisto() const
    {
        // 返回 refHisto 成员变量的值。
        return this->refHisto;
    }

    // 成员方法：setRefHisto（设置特征值数组中键值对的个数）
    // 设置 refHisto 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setRefHisto(            
            float *refHisto  // 给定的参考直方图。
    ) {
        // 检查输入参数是否合法
        if (refHisto == NULL)
            return INVALID_DATA;

        // 将 refHisto 成员变量赋成新值
        this->refHisto = refHisto;

        return NO_ERROR;
    }

    // Host 成员方法：HistogramEquilibrium（直方图均衡化）
    // 根据组映射方法，将输入图像的直方图进行均衡化，得到输出图像，使得输出图像
    // 的直方图分布极大可能的接近等概率分布。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    HistogramEquilibrium(
            Image *inimg,  // 输入图像。
            Image *outimg  // 输出图像。
    );

    // Host 成员方法：HistogramSpecByImage（根据参考图像进行直方图规定化）
    // 根据参考图像refimg的直方图分布情况，利用组映射方法，对输入图像inimg直方图
    // 进行规定化处理，使得最终输出图像outimg的直方图接近参考图像refimg的直方
    // 图。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    HistogramSpecByImage(
            Image *inimg,  // 输入图像。
            Image *outimg  // 输出图像。
    );

    // Host 成员方法：HistogramSpecByHisto（根据参考直方图进行规定化）
    // 根据参考直方图refHisto，利用组映射方法，对输入图像inimg直方图进行规定化
    // 处理，使得最终输出图像outimg的直方图接近参考直方图refHisto。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    HistogramSpecByHisto(  
            Image *inimg,  // 输入图像。
            Image *outimg  // 输出图像。
    );
};

#endif

