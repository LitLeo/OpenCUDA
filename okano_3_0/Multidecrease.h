// Multidecrease.h
// 创建者：仲思惠
// N值化（Multidecrease）

// 功能说明：将图像的 256 个灰度级，降低为 N 个
// 修订历史：
// 2013年07月03日 （仲思惠）
// 初始版本
// 2013年07月19日 （仲思惠）
// 实现了获取图像粗分割阈值功能。
// 2013年07月21日（仲思惠）
// 实现了根据粗分割阈值，将图像灰度级降低功能。
// 2013年07月28日（仲思惠）
// 实现了使用 OTSU 法，在粗分割阈值的基础上搜索最佳阈值。
// 2013年09月12日（仲思惠）
// 修改了前向N值化和后向N值化核函数中的bug。
// 2013年10月25日（仲思惠）
// 增加了对区间顶点值的边界情况检测，取消了两个宏定义，提供了参数
// 设定的新接口。

#ifndef __MULTIDECREASE_H__
#define __MULTIDECREASE_H__

#include "Image.h"
#include "ErrorCode.h"
#include "Histogram.h"

#define MD_FRONT 0
#define MD_BACK 1

// 类：Multidecrease
// 继承自：无
// 将图像的 256 个灰度级，降低为 N 个。
// 使用双峰法求出图像的粗分割阈值，再在一定的松弛余量范围内搜索
// 图像的最佳分割阈值。最够根据求得的阈值集合，将图像灰度级降低。
class Multidecrease {

protected:
    
	// 成员变量：threshholds（灰度阈值集合）
    unsigned char *threshold;

    // 成员变量：stateflag（状态标记）
    unsigned int stateflag;
	
	// 成员变量：widthrange（宽度范围阈值）
	unsigned int widthrange;
	
	// 成员变量：pixelrange（像素数范围阈值）
	unsigned int pixelrange;

public: 
    // 构造函数：Multidecrease
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Multidecrease()
    {
		threshold = 0;      
        stateflag = MD_FRONT;
		widthrange = 30;
		pixelrange = 1000;
    }
	
	// 获取图像的分割阈值集合
	__host__ unsigned char* getthreshold () {
	    return this->threshold;
	}
    
	// 设置图像的状态标记
    __host__ int setsateflag (unsigned int flag) {       
        if(flag != MD_FRONT &&flag != MD_BACK)           
             return 0;
        this->stateflag = flag;
        return 1;   
    }
	
	// 获取图像的宽度范围阈值
	__host__ unsigned int getwidthrange () {
	    return this->widthrange;
	}
	
	// 设置图像的宽度范围阈值
    __host__ int setwidthrange (unsigned int range) {       
        if(range <= 0 || range >= 255)           
             return 0;
        this->widthrange = range;
        return 1;   
    }
	
	// 获取图像的像素数范围阈值
	__host__ unsigned int getpixelrange () {
	    return this->pixelrange;
	}
	
	// 设置图像的像素数范围阈值
    __host__ int setpixelrange (unsigned int range) {       
        if(range <= 0 || range >= 255)            
             return 0;
        this->pixelrange = range;
        return 1;   
    }
	// Host 成员方法：multidecrease（N值化处理）  
    __host__ int multidecrease(
               Image *inimg,    // 输入图像
               Image *outimg    // 输出图像
    );
};

#endif

