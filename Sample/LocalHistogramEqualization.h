// LocalHistogramEqualization.h
// 创建人：杨伟光
//
// 自适应直方图均衡化算法（LocalHistEqual）
// 功能说明：对指定图像进行直方图均衡化处理。
//
// 修订历史：
// 2013年06月28日（杨伟光）
//     初始版本。
// 2013年09月17日（高新凯）
//     实现算法基本功能，并行处理窗口。  
// 2013年10月9日（高新凯）
//     实现窗口均衡化的完全并行化。 

#ifndef __LOCALHISTOGRAMEQUALIZATION_H__
#define __LOCALHISTOGRAMEQUALIZATION_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：LocalHistEqual（自适应直方图均衡化）
// 继承自：无
// 功能说明：对指定图像进行直方图均衡化处理。
class LocalHistEqual {

protected:

    // 成员变量：divNum（分割数）
    // 一般为1， 2， 3， 4， 5 
    int divNum;

    // 成员变量：t0（外部参数）
    // 范围为5，10，15,20, 25, 30,35,40,45,50  
    unsigned char t0;

    // 成员变量：weight（外部参数）
    // 0 <= weight < 1  
    float weight;

public:

    // 构造函数：LocalHistEqual
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    LocalHistEqual() {     
        // 分割数默认为 1。 
        this->divNum = 1;  
        // 默认值为最小值。
        this->t0 = 5;      
        // 默认值为最小值。
        this->weight = 0;  
    }

    // 成员函数：getDivNum（获取分割率的值）
    // 获取成员变量 divNum 的值。
    __host__ __device__ int 
    getDivNum() const
    {
        // 返回 divNum 成员变量的值。
        return this->divNum;
    }

    // 成员函数：setDivNum（设置分割率）
    // 设置成员变量 divNum 的值。
    __host__ __device__ int  // 返回值：若函数正确执行，返回 NO_ERROR。
    setDivNum(
            int divnum       // 分割率
    ) {
        // 如果分割率小于等于０，则报错。 
        if(divNum <= 0)
            return INVALID_DATA;

        // 将 divNum 成员变量赋成新值。
        this->divNum = divnum;

        return NO_ERROR;
    }

    // 成员函数：getT0（获取 T0 的值）
    // 获取成员变量 t0 的值。
    __host__ __device__ int 
    getT0() const
    {
        // 返回 t0 成员变量的值。 
        return this->t0;
    }

    // 成员函数：setT0（设置 t0）
    // 设置成员变量 t0 的值。
    __host__ __device__ int  // 返回值：若函数正确执行，返回 NO_ERROR。
    setT0(
            int t0           // t0
    ) {
        //　如果 t0 的值超出取值范围，则报错。 
        if(t0 < 5 || t0 > 50)
            return INVALID_DATA;

        // 将 t0 成员变量赋成新值。
        this->t0 = t0;

        return NO_ERROR;
    }

    // 成员函数：getWeight（获取 weight 的值）
    // 获取成员变量 weight 的值。
    __host__ __device__ float 
    getWeight() const
    {
        // 返回 weight 成员变量的值。
        return this->weight;
    }

    // 成员函数：setWeight（设置 weight ）
    // 设置成员变量 weight 的值。
    __host__ __device__ int  // 返回值：若函数正确执行，返回 NO_ERROR。
    setWeight(
            float weight       // 权值
    ) {
        //　如果 weight 的值超出取值范围，则报错。 
        if(weight < 0 || weight > 1)
            return INVALID_DATA;

        // 将 weight 成员变量赋成新值。
        this->weight = weight;

        return NO_ERROR;
    }

    // 成员方法：localHistEqual（图像直方图均衡化）
    // 功能说明：对指定图像inImg分块进行直方图均衡化处理，图像分割率为divNum，
    // 处理结果保存在outimg中。

    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，
                           // 返回NO_ERROR
    localHistEqual(
            Image *inimg,  // 输入图像 
            Image *outimg  // 输出图像 
    );
};

#endif

