// LearningFilter.h 
//
// 学习型滤波（LearningFilter）
// 功能说明：根据设定的阈值，对灰度图像进行二值化处理，得到二值图像。

#ifndef __LEARNINGFILTER_H__
#define __LEARNINGFILTER_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：LearningFilter
// 继承自：无
// 暂无
class LearningFilter {

protected:
    // 成员变量：eps(变化量)
    float eps;
	float r;
public:
    // 构造函数：LearningFilter
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    LearningFilter()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->eps = 0.001;  // 微变化值值默认为 0.001
		this->r = 5.0;      // 核尺寸默认为 5.0
    }

    // 构造函数：LearningFilter
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    LearningFilter(
            float eps, float r
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->eps = 0.001;  
		this->r = 5.0;
	
        // 根据参数列表中的值设定成员变量的初值
        setEps(eps);
		setR(r);
    }
   
    // 成员方法：getEps（获取灰度阈值）
    // 获取成员变量 eps 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 eps 的值
    getEps() const
    {
        // 返回 eps 成员变量的值。
        return this->eps;
    } 

    // 成员方法：setEps（设置灰度阈值）
    // 设置成员变量 eps 的值。
    __host__ __device__ int          // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    setEps(
            float eps  // 设定新的灰度阈值
    ) {
        // 将 eps 成员变量赋成新值
        this->eps = eps;

        return NO_ERROR;
    }
	
	// 成员方法：getR（获取灰度阈值）
    // 获取成员变量 r 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 r 的值
    getR() const
    {
        // 返回 r 成员变量的值。
        return this->r;
    } 

    // 成员方法 r（设置灰度阈值）
    // 设置成员变量 r 的值。
    __host__ __device__ int          // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    setR(
            float r  // 设定新的灰度阈值
    ) {
        // 将 r 成员变量赋成新值
        this->r = r;

        return NO_ERROR;
    }

    // Host 成员方法：LearningFilter（二值化处理）
    // 根据阈值对图像进行二值化处理。这是一个 Out-Place 形式的。如果像素的灰度
    // 值大于等于阈值，此像素的灰度值赋值为 255。如果像素的灰度值小于阈值，此
    // 像素的灰度值赋值为 0。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    learningFilter(
            Image *inimg1,  // 输入图像
			Image *inimg2,  // 输入图像2
            Image *outimg  // 输出图像
    );
};

#endif

