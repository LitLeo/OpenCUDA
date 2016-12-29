// PriFeatureCheckC.h
// 创建人：刘婷

// 任意形状曲线基本特征检测（Primary Feature Check for arbitrarily shaped-Curve）
// 功能说明：对于任意形状的曲线，计算该曲线的几种特征值：（1）曲线点数与曲线最小
//           有向外接矩形（Smallest Direction Rectangle，简称 MDR）周长比；
//          （2）MDR 短边与长边比；（3）AMI（AFFINE MOMENT INVARIANTS）平均值；
//          （4）曲线角点个数与曲线的点数比；（5）曲线多边形面积和曲线 MDR 面积
//           比。并判断这些特征值是否在给定指标范围内，从而给出错误判断。
// 修订历史：
// 2013年08月03日（刘婷）
//     编写头文件。
// 2013年09月16日（刘婷）
//     完成求闭合曲线的面积和中心坐标函数。
// 2013年10月16日（刘婷）
//     实现除角点检测的其他功能。
// 2013年10月17日（刘婷）
//     修改 BUG，更改代码规范。

#ifndef __PRIFEATURECHECKC_H__
#define __PRIEFATURECHECKC_H__

#include "ErrorCode.h"
#include "Image.h"
#include "Curve.h"
#include "ImgConvert.h"
#include "Moments.h"
#include "SmallestDirRect.h"

// 宏变量，默认设置 AMI 的长度为 9
#define AMISIZE  9

// 类：PriFeatureCheckC（任意形状曲线基本特征检测）
// 继承自：无
// 实现任意形状曲线的基本特征检测。对于任意形状的曲线，计算该曲线的几种特征值：
//（1）曲线点数与曲线最小有向外接矩形（Smallest Direction Rectangle，简称 MDR）
// 周长比；（2）MDR 短边与长边比；（3）AMI 平均值；（4）曲线角点个数与曲线的点数
// 比；（5）曲线多边形面积和曲线 MDR 面积比。并判断这些特征值是否在给定指标范围
// 内，从而给出错误判断。
class PriFeatureCheckC {

protected:

    // 成员变量：minLengthRatio，maxLengthRatio（给定指标范围）
    // 曲线的点数与曲线的最小有向外接矩形（MDR）周长比值的下限、上限。
    float minLengthRatio, maxLengthRatio;

    // 成员变量：minMDRSideRatio，maxMDRSideRatio（给定指标范围）
    // MDR的短边与长边的比值的下限、上限。
    float minMDRSideRatio, maxMDRSideRatio;

    // 成员变量：minPolygonAreaRatio，maxPloygonArearRatio（给定指标范围）
    // 曲线的多边形面积与曲线的 MDR 面积比值的下限、上限。
    float minContourAreaRatio, maxContourAreaRatio;

    // 成员变量：minVertexNumRatio，maxVertexNumRatio（给定指标范围）
    // 曲线角点个数与曲线点数比值的下限、上限。
    float minVertexNumRatio, maxVertexNumRatio;

    // 成员变量：avgAMIs（平均 AMI 值）
    // 按照需求，它是一个长度为 9 的数组。
    double *avgAMIs;

    // 成员变量：maxAMIsError（AMI 容差向量）
    // 按照需求，它是一个长度为 9 的数组。
    double *maxAMIsError;

    // 成员变量：imgconvert（ImgConvert 对象）
    // 在本算法中需要调用 ImgConvert 算法将曲线转为图像，因此此处将 ImgConvert
    // 对象作为该类的成员变量，方便使用。
    ImgConvert imgconvert;

    // 调用最小外接矩形算法得到最小外接矩形的长短边。
    SmallestDirRect sdr;

    // 成员变量：ami（Moments 对象）
    // 在本算法中需要调用 Moments 算法，计算图像的 AMIS。
    Moments ami;


public:

    // 构造函数：PriFeatureCheckC
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    PriFeatureCheckC() :imgconvert(), ami(), sdr()
    {
        // 使用默认值为成员变量赋值。
        minLengthRatio = 0.0f;
        maxLengthRatio = 1.0f;
        minMDRSideRatio = 0.0f;
        maxMDRSideRatio = 1.0f;
        minVertexNumRatio = 0.0f;
        maxVertexNumRatio = 1.0f;
        minContourAreaRatio = 0.0f;
        maxContourAreaRatio = 1.0f;
        avgAMIs = NULL;
        maxAMIsError = NULL;
    }

    // 构造函数：PriFeatureCheckC
    // 有参数版本的构造函数，这些值在运行过程中可以改变。
    __host__ __device__
    PriFeatureCheckC(
            // 各参数含义见上。
            float minlengthratio, float maxlengthratio,
            float minmdrsideratio, float maxmdrsideratio,
            float minvertexnumratio, float maxvertexnumratio,
            float mincontourarearatio, float maxcontourarearatio,
            double *avgamis, double *maxamiserror
    ):imgconvert(), ami(), sdr()
    {
        // 使用默认值为成员变量赋值。
        minLengthRatio = 0.0f;       // 各参数含义见上
        maxLengthRatio = 1.0f;       // 各参数含义见上
        minMDRSideRatio = 0.0f;      // 各参数含义见上
        maxMDRSideRatio = 1.0f;      // 各参数含义见上
        minVertexNumRatio = 0.0f;    // 各参数含义见上
        maxVertexNumRatio = 1.0f;    // 各参数含义见上
        minContourAreaRatio = 0.0f;  // 各参数含义见上
        maxContourAreaRatio = 1.0f;  // 各参数含义见上
        avgAMIs = NULL;              // 各参数含义见上
        maxAMIsError = NULL;         // 各参数含义见上

        // 根据参数列表中的值设定成员变量的初值
        setMinLengthRatio(minlengthratio);
        setMaxLengthRatio(maxlengthratio);
        setMinMDRSideRatio(minmdrsideratio);
        setMaxMDRSideRatio(maxmdrsideratio);
        setMinContourAreaRatio(mincontourarearatio);
        setMaxContourAreaRatio(maxcontourarearatio);
        setMinVertexNumRatio(minvertexnumratio);
        setMaxVertexNumRatio(maxvertexnumratio);
        setAvgAMIs(avgamis);
        setMaxAMIsError(maxamiserror);
    }

    // 成员方法：getMinLengthRatio（获取指标 minLengthRatio）
    // 获取曲线的点数与曲线的最小有向外接矩形（MDR）周长比值的下限。
    __host__ __device__ float  // 返回值：获取指标 minLengthRatio。
    getMinLengthRatio() const
    {
        // 返回指标 minLengthRatio。
        return this->minLengthRatio;    
    } 

    // 成员方法：setMinLengthRatio（设置指标 minLengthRatio 的值）
    // 设置曲线的点数与曲线的最小有向外接矩形（MDR）周长比值的下限。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    setMinLengthRatio(
            float minlengthratio  // 新的 minLengthRatio 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (minlengthratio <= 0 || minlengthratio >= this->maxLengthRatio)
            return INVALID_DATA;

        // 为成员变量 minLengthRatio 赋新值。
        this->minLengthRatio = minlengthratio;

        return NO_ERROR;
    }

    // 成员方法：getMaxLengthRatio（获取指标 maxLengthRatio）
    // 获取曲线的点数与曲线的最小有向外接矩形（MDR）周长比值的上限。
    __host__ __device__ float  // 返回值：获取指标 maxLengthRatio。
    getMaxLengthRatio() const
    {
        // 返回指标 maxLengthRatio。
        return this->maxLengthRatio;    
    } 

    // 成员方法：setMaxLengthRatio（设置指标 maxLengthRatio 的值）
    // 设置曲线的点数与曲线的最小有向外接矩形（MDR）周长比值的上限。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    setMaxLengthRatio(
            float maxlengthratio  // 新的 maxLengthRatio 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (maxlengthratio <= 0 || maxlengthratio <= this->minLengthRatio)
            return INVALID_DATA;

        // 为成员变量 maxLengthRatio 赋新值。
        this->maxLengthRatio = maxlengthratio;

        return NO_ERROR;
    }

    // 成员方法：getMinMDRSideRatio（获取指标 minMDRSideRatio）
    // 获取MDR的短边与长边的比值的下限。
    __host__ __device__ float  // 返回值：获取指标 minMDRSideRatio。
    getMinMDRSideRatio() const
    {
        // 返回指标 minMDRSideRatio。
        return this->minMDRSideRatio;    
    } 

    // 成员方法：setMinMDRSideRatio（设置指标 minMDRSideRatio 的值）
    // 设置MDR的短边与长边的比值的下限。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    setMinMDRSideRatio(
            float minmdrsideratio  // 新的 minMDRSideRatio 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (minmdrsideratio <= 0 || minmdrsideratio >= this->minMDRSideRatio)
            return INVALID_DATA;

        // 为成员变量 minMDRSideRatio 赋新值。
        this->minMDRSideRatio = minmdrsideratio;

        return NO_ERROR;
    }

    // 成员方法：getMaxMDRSideRatio（获取指标 maxMDRSideRatio）
    // 获取MDR的短边与长边的比值的上限。
    __host__ __device__ float  // 返回值：获取指标 maxMDRSideRatio。
    getMaxMDRSideRatio() const
    {
        // 返回指标 maxMDRSideRatio。
        return this->maxMDRSideRatio;    
    } 

    // 成员方法：setMaxMDRSideRatio（设置指标 maxMDRSideRatio 的值）
    // 设置MDR的短边与长边的比值的上限。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    setMaxMDRSideRatio(
            float maxmdrsideratio  // 新的 maxMDRSideRatio 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (maxmdrsideratio <= 0 || maxmdrsideratio <= this->minMDRSideRatio)
            return INVALID_DATA;

        // 为成员变量 maxMDRSideRatio 赋新值。
        this->maxMDRSideRatio = maxmdrsideratio;

        return NO_ERROR;
    }

    // 成员方法：getMinContourAreaRatio（获取指标 minContourAreaRatio）
    // 曲线的多边形面积与曲线的 MDR 面积比值的下限。
    __host__ __device__ float  // 返回值：获取指标 minContourAreaRatio。
    getMinContourAreaRatio() const
    {
        // 返回指标 minContourAreaRatio。
        return this->minContourAreaRatio;    
    } 

    // 成员方法：setMinContourAreaRatio（设置指标 minContourAreaRatio的值）
    // 设置曲线的多边形面积与曲线的 MDR 面积比值的下限。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    setMinContourAreaRatio(
            float mincontourarearatio  // 新的 minContourAreaRatio 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (mincontourarearatio <= 0 ||
            mincontourarearatio >= this->maxContourAreaRatio)
            return INVALID_DATA;

        // 为成员变量 minContourAreaRatio 赋新值。
        this->minContourAreaRatio = mincontourarearatio;

        return NO_ERROR;
    }

    // 成员方法：getMaxContourAreaRatio（获取指标 maxContourAreaRatio）
    // 获取曲线的多边形面积与曲线的 MDR 面积比值的上限。
    __host__ __device__ float  // 返回值：获取指标 maxContourAreaRatio。
    getMaxContourAreaRatio() const
    {
        // 返回指标 maxContourAreaRatio。
        return this->maxContourAreaRatio;    
    } 

    // 成员方法：setMaxContourAreaRatio（设置指标 maxContourAreaRatio 的值）
    // 设置曲线的多边形面积与曲线的 MDR 面积比值的上限。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    setMaxContourAreaRatio(
            float maxcontourarearatio  // 新的 maxContourAreaRatio 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (maxcontourarearatio <= 0 ||
            maxcontourarearatio <= this->minContourAreaRatio)
            return INVALID_DATA;

        // 为成员变量 maxContourAreaRatio 赋新值。
        this->maxContourAreaRatio = maxcontourarearatio;

        return NO_ERROR;
    }

    // 成员方法：getMinVertexNumRatioo（获取指标 minVertexNumRatio）
    // 获取曲线的角点个数与曲线的点数比值的下限。
    __host__ __device__ float  // 返回值：获取指标 minVertexNumRatio。
    getMinVertexNumRatio() const
    {
        // 返回指标 minVertexNumRatio。
        return this->minVertexNumRatio;    
    } 

    // 成员方法：setMinVertexNumRatio（设置指标 minVertexNumRatio 的值）
    // 设置曲线的角点个数与曲线的点数比值的下限。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    setMinVertexNumRatio(
            float minvertexnumratio  // 新的 minVertexNumRatio 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (minvertexnumratio <= 0 ||
            minvertexnumratio >= this->maxVertexNumRatio)
            return INVALID_DATA;

        // 为成员变量 minVertexNumRatio 赋新值。
        this->minVertexNumRatio = minvertexnumratio;

        return NO_ERROR;
    }

    // 成员方法：getMaxVertexNumRatioo（获取指标 maxVertexNumRatio）
    // 获取曲线的角点个数与曲线的点数比值的上限。
    __host__ __device__ float  // 返回值：获取指标 maxVertexNumRatio。
    getMaxVertexNumRatio() const
    {
        // 返回指标 maxVertexNumRatio。
        return this->maxVertexNumRatio;    
    } 

    // 成员方法：setMaxVertexNumRatio（设置指标 maxVertexNumRatio 的值）
    // 设置曲线的角点个数与曲线的点数比值的上限。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    setMaxVertexNumRatio(
            float maxvertexnumratio  // 新的 maxVertexNumRatio 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (maxvertexnumratio <= 0 ||
            maxvertexnumratio <= this->minVertexNumRatio)
            return INVALID_DATA;

        // 为成员变量 maxVertexNumRatio 赋新值。
        this->maxVertexNumRatio = maxvertexnumratio;

        return NO_ERROR;
    }

    // 成员方法：getAvgAMIs（获取平均 AMI）
    // 获取平均 AMI 值。
    __host__ __device__ double*  // 返回值：获取平均 AMI 值。
    getAvgAMIs() const
    {
        // 返回 avgAMIs。
        return this->avgAMIs;    
    } 

    // 成员方法：setAvgAMIs（设置平均 AMI 值）
    // 设置平均 AMI 值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，
                             // 返回 NO_ERROR。
    setAvgAMIs(
            double *avgamis  // 新的 AvgAMIs 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (avgamis == NULL)
            return INVALID_DATA;

        for (int i = 0; i < AMISIZE; i++) {
            // 若新的 AvgAMIs 包含负值，则报错。
            if (avgamis[i] < 0)
                return INVALID_DATA;
        }

        // 为成员变量 avgAMIs 赋新值。
        this->avgAMIs = avgamis;

        return NO_ERROR;
    }

    // 成员方法：getMaxAMIsError（获取 AMI 最大差值向量）
    // 获取 AMI 最大差值向量，该向量的每一个值都是一个指标。
    __host__ __device__ double*  // 返回值：获取平均 AMI 最大差值向量。
    getMaxAMIsError() const
    {
        // 返回 maxAMIsError。
        return this->maxAMIsError;    
    } 

    // 成员方法：setMaxAMIsError（设置 AMI 最大差值向量）
    // 设置 AMI 最大差值向量。
    __host__ __device__ int    // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    setMaxAMIsError(
            double *maxamiserror  // 新的 maxAMIsError 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (maxamiserror == NULL)
            return INVALID_DATA;

        for (int i = 0; i < AMISIZE; i++) {
            // 若新的 maxamiserror 包含负值，则报错。
            if (maxamiserror[i] < 0)
                return INVALID_DATA;
        }

        // 为成员变量 MaxAMIsError 赋新值。
        this->maxAMIsError = maxamiserror;

        return NO_ERROR;
    }

    // 成员方法：priFeatureCheckC（任意曲线的基本特征值检查）
    // 根据给定 Curve 信息，提取出来曲线的特征值进而检查这些特征值是否在给定指
    // 标范围内。
    __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回 NO_ERROR。
    priFeatureCheckC(
            Curve *curve,      // 输入 Curve 类对象
            float *result,     // 结果数组，按照需求，它是一个长度为 5 的数组，
                               // 存储的是本算法计算出来的特征值（即 length
                               // Ratio，MDR Side Ratio，AIM Errors，Vertex Num
                               // Ratio，Contour Area Ratio）。

            bool *errorJudge   // 存放 bool 值的结果数组，按照需求，长度为 5 的
                               // 数组，初始化均为 false，存放的是特征值与给定指
                               // 标之间的关系，若在指标范围内则为 true，反之为 
                               // false。
    ); 

};

#endif

