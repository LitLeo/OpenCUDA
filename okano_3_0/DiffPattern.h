// DiffPattern.h
// 创建人：邓建平
//
// 局部特异检出（DiffPattern）
// 功能说明：按照给定的 pattern 检测图像的局部区域的像素点是否有异常。若有异常，
//           则将出现异常的 pattern 信息以数组形式返回
//
// 修订历史：
// 2013年07月12日（邓建平）
//     初始版本。

#ifndef __DIFFPATTERN_H__
#define __DIFFPATTERN_H__

#include "ErrorCode.h"
#include "Image.h"


// 类：DiffPattern（局部特异检出）
// 继承自：无
// 该类通过一个数组来指定局部特异部分检测需要用到的 pattern 种类，一共包含 19 
// 个共 10 对 pattern 且都是静态的，数组中的元素即 pattern 对的标号。若 pattern
// 对中的像素点平均值都满足同一个不等关系，则将这个 pattern 对的信息返回。
class DiffPattern {

protected:
    // 成员变量：centerX（中心点横坐标），centerY（中心点纵坐标）
    // 图像的中心点坐标是与 pattern 中的 anchor 对齐的像素点的坐标
    int centerX, centerY;

    // 成员变量：patCount（pattern 对的数目）
    // 检测异常需要的差分 pattern 的对数
    int patCount;

    // 成员变量：indice（pattern 对的序号数组）
    // 差分 pattern 的序号集合，一共包含 patCount 个不同的标号
    int *indice;

public:
    // 构造函数：DiffPattern
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    DiffPattern()
    {
        // 使用默认值为类的各个成员变量赋值。 
        centerX = 0;    // 中心点横坐标默认为 0
        centerY = 0;    // 中心点纵坐标默认为 0
        patCount = 0;   // pattern 对的数目默认为 0
        indice = NULL;  // pattern 对序号数组默认为 NULL
    }

    // 构造函数：DiffPattern
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    DiffPattern(
            int centerx,   // 中心点横坐标（具体解释见成员变量）
            int centery,   // 中心点纵坐标（具体解释见成员变量）
            int patcount,  // pattern 对的数目（具体解释见成员变量）
            int *indice    // pattern 对的序号数组（具体解释见成员变量）
    ) {
	// 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
	// 的初始值而使系统进入一个未知的状态。
	this->centerX = 0;    // 中心点横坐标默认为 0
	this->centerY = 0;    // 中心点纵坐标默认为 0
	this->patCount = 0;   // pattern 对的数目默认为 0
	this->indice = NULL;  // pattern 对序号数组默认为 NULL

        // 根据参数列表中的值设定成员变量的初值
        this->setCenterX(centerx);
        this->setCenterY(centery);
        this->setPatCount(patcount);
        this->setIndice(indice);
    }

    // 成员方法：getCenterX（读取中心点横坐标）
    // 读取 centerX 成员变量的值。
    __host__ __device__ int  // 返回: 当前 centerX 成员变量的值
    getCenterX() const
    {
        // 返回 centerX 成员变量的值
        return this->centerX;
    }

    // 成员方法：setCenterX（设置中心点横坐标）
    // 设置 centerX 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setCenterX(
            int centerx      // 中心点横坐标
    ) {
        // 若参数不合法，返回对应的错误代码并退出
        if (centerX <= 0)
            return INVALID_DATA;
        // 将 centerX 成员变量赋成新值 
        this->centerX = centerx;
        return NO_ERROR;
    }

    // 成员方法：getCenterY（读取中心点纵坐标）
    // 读取 centerY 成员变量的值。
    __host__ __device__ int  // 返回: 当前 centerY 成员变量的值
    getCenterY() const
    {
        // 返回 centerY 成员变量的值
        return this->centerY;
    }

    // 成员方法：setCenterY（设置中心点纵坐标）
    // 设置 centerY 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setCenterY(
            int centery      // 中心点纵坐标
    ) {
        // 若参数不合法，返回对应的错误代码并退出
        if (centerY <= 0)
            return INVALID_DATA;
        // 将 centerX 成员变量赋成新值 
        this->centerY = centery;
        return NO_ERROR;
    }

    // 成员方法：getPatCount（读取 pattern 对的数目）
    // 读取 patCount 成员变量的值。
    __host__ __device__ int  // 返回: 当前 patCount 成员变量的值
    getPatCount() const
    {
        // 返回 centerY 成员变量的值
        return this->patCount;
    }

    // 成员方法：setPatCount（设置 pattern 对的数目）
    // 设置 patCount 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setPatCount(
            int patcount     // pattern 对的数目
    ) {
        // 若参数不合法，返回对应的错误代码并退出
        if (patcount <= 0)
            return INVALID_DATA;
        // 将 patCount 成员变量赋成新值 
        this->patCount = patcount;
        return NO_ERROR;
    }

    // 成员方法：setIndice（设置 pattern 对的序号数组）
    // 设置 indice 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setIndice(
            int *indice      // pattern 对的序号数组
    ) {
        // 若参数不合法，返回对应的错误代码并退出
        if (indice == NULL)
            return NULL_POINTER;
        // 将 indice 成员变量赋成新值 
        this->indice = indice;
        return NO_ERROR;
    }

    // Host 成员方法：doDiffPattern（检测异常）
    // 根据输入图像，计算异常 pattern 对的数目，并将这些 pattern 的信息返回
    __host__ int     // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR。
    doDiffPattern(
            Image *inimg,  // 输入图像
            int *counter,  // 异常 pattern 计数器
            float *result  // 返回 pattern 信息的指针	    
    );

};

#endif

