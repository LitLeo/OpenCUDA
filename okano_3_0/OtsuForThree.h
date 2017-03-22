// OtsuForThree.h
// 创建人：仲思惠
//
// 图像三值化（OtsuForThree.h）
// 根据图像像素的分散程度,自动找到两个最佳分割阈值，得到
// 图像的三值化结果。
//
// 修订历史：
// 2014年09月09日（仲思惠）
//     初始版本。
// 2014年09月19日（仲思惠）
//     修改了一处计算错误。
// 2014年09月25日（仲思惠）
//     并行实现128×128个方差元素的计算。

#ifndef __OTSUFORTHREE_H__
#define __OTSUFORTHREE_H__

#include "Image.h"
#include "Histogram.h"

// 类：OtsuBinarize
// 继承自：无
// 根据图像像素的分散程度,自动找到两个最佳分割阈值，得到
// 图像的三值化结果。
class OtsuForThree {

protected:
    
    // 成员变量：isForward（三值化类型）
    bool isForward;
	
public:

    // 构造函数：OtsuForThree
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    OtsuForThree(){
        isForward = true;
    }

	// 获取图像的三值化类型
	__host__ bool getisForward () {
	    return this->isForward;
	}
    
	// 设置图像的三值化类型
    __host__ int set_isForward (bool type) {       
        this->isForward = type;
        return 1; 
    }
	
    // Host 成员方法：otsuForThree（图像三值化）
    // 根据图像像素的分散程度,自动找到两个最佳分割阈值，得到
    // 图像的三值化结果。
    __host__ int                // 返回值：函数是否正确执行，若函数正
                                // 确执行，返回 NO_ERROR。
    otsuForThree(
            Image *inimg,       // 输入图像
            Image *outimg       // 输出图像
    );
};

#endif

