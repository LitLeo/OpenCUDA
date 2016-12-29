// RegionGrow.h 
// 创建人：王媛媛
//
// 
// 修订历史：
// 2012年09月13日（王媛媛）
//     初始版本

#ifndef __REGIONGROW_H__
#define __REGIONGROW_H__

#include "Image.h"
#include "ErrorCode.h"

class RegionGrow {

protected:

    unsigned char seed;
    unsigned char threshold;

public:
    // 构造函数：RegionGrow
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    RegionGrow()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->seed = 255;  // 灰度阈值默认为 128。
        this->threshold = 225;
    }

    // 构造函数：RegionGrow
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    RegionGrow(
            unsigned char seed,
            unsigned char threshold
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->seed = 255;  // 灰度阈值默认为 128。
		this->threshold = 225;
	
        // 根据参数列表中的值设定成员变量的初值
        setT(seed);
		setD(threshold);
    }
   
    // 成员方法：getT（获取灰度阈值）
    // 获取成员变量 threshold 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 threshold 的值
    getT() const
    {
        // 返回 threshold 成员变量的值。
        return this->seed;
    } 

    // 成员方法：setT（设置灰度阈值）
    // 设置成员变量 threshold 的值。
    __host__ __device__ int          // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    setT(
            unsigned char seed  // 设定新的灰度阈值
    ) {
        // 将 threshold 成员变量赋成新值
        this->seed = seed;

        return NO_ERROR;
    }

    // 成员方法：getD（获取灰度阈值）
    // 获取成员变量 threshold 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 threshold 的值
    getD() const
    {
        // 返回 threshold 成员变量的值。
        return this->threshold;
    } 

    // 成员方法：setD（设置灰度阈值）
    // 设置成员变量 threshold 的值。
    __host__ __device__ int          // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    setD(
            unsigned char threshold  // 设定新的灰度阈值
    ) {
        // 将 threshold 成员变量赋成新值
        this->threshold = threshold;

        return NO_ERROR;
    }
	

    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    regionGrow_parallel(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
	
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    regionGrow_serial(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
};

#endif

