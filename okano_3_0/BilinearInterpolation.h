// BilinearInterpolation.h
// 创建人：邓建平
//
// 双线性插值（BilinearInterpolation）
// 功能说明：根据给定的放大倍数对放大后的图像的未知点进行双线性插值,保
//           证放大后的图像比较平滑
//
// 修订历史：
// 2012年10月30日（邓建平）
//     初始版本
// 2102年11月07日（邓建平、于玉龙）
//     修正了 texture 的使用方式，修正了一些格式错误

#ifndef __BILINEARINTERPOLATION_H__
#define __BILINEARINTERPOLATION_H__

#include "ErrorCode.h"
#include "Image.h"


// 类：BilinearInterpolation（双线性插值）
// 继承自：无
// 根据给定的放大倍数对放大后的图像的未知点进行双线性插值,保证放大后的
// 图像比较平滑
class BilinearInterpolation {

protected:

    // 成员变量：scale（放大倍数）
    // 该参数默认为 0 ，当设置的参数大于 0 时则对图像进行对应倍数的放大
    int scale;

public:

    // 构造函数：BilinearInterpolation
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    BilinearInterpolation()
    {
        // 使用默认值为类的各个成员变量赋值。
        scale = 0;  // 放大倍数默认为 0
    }

    // 构造函数：BilinearInterpolation
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    BilinearInterpolation(
            int scale  // 放大倍数（具体解释见成员变量）
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->scale = 0;  // 放大倍数默认为 0
        
        // 根据参数列表中的值设定成员变量的初值
        this->setScale(scale);
    }

    // 成员方法：getScale（读取放大倍数）
    // 读取 scale 成员变量的值。
    __host__ __device__ int  // 返回: 当前 scale 成员变量的值
    getScale() const
    {   
        // 返回 scale 成员变量的值
        return this->scale;
    }

    // 成员方法：setScale（设置迭代次数）
    // 设置 scale 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setScale(
            int scale        // 放大倍数
    ) { 
        // 将 scale 成员变量赋成新值                       
        this->scale = scale;
        return NO_ERROR;
    }

    // Host 成员方法：doInterpolation（执行滤波）
    // 对图像进行双边滤波，outplace 版本，由于输出图像与输入图像大小不一，只提
    // 供 outplace 版本
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    doInterpolation(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像 
    );

};

#endif

