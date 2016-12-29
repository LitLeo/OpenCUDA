// EdgeDetection.h
// 创建者：焦利颖
//  
// 算法名称：简单亮度渐变（SimpleBrightnessGradient）
// 功能说明：模拟图像左上角增加一光源，　
//           使得图像从左上角到右下角亮度逐渐变化
//           G2 = W * 255 + （1-W）* G1；(0<=W<=1)
//           此处 G2 为目标灰度值，G1 为原始灰度值； 
//           W 为权重，离光源越远权重越小
//          （核函数内开方函数： sqrtf（x）；）
//
// 修订历史：
// 2014年09月23日（焦利颖）
//     初始版本。

#ifndef __SIMPLEBRIGHTNESSGRADIENT_H__ 
#define __SIMPLEBRIGHTNESSGRADIENT_H__ 

#include "Image.h" 

// 类：SimpleBrightnessGradient （亮度渐变类） 
// 继承自：无
// 该类定义了简单亮度渐变的方法
class SimpleBrightnessGradient { 
public: 

    // 构造函数：SimpleBrightnessGradient
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值
    __host__ __device__ 
    SimpleBrightnessGradient() 
    { 
        // 无任何操作。
    } 

    // Host 成员方法：SimpleBrightnessGradient（亮度渐变）
    // 实现图像的亮度渐变
    __host__ int           // 返回值：若函数正确执行，返回 NO_ERROR，
                           // 　　　　否则返回相应的错误码
    simpleBrightnessGradient(
            Image *inimg,  // 输入图像  
            Image *outimg  // 输出图像
    );
};

#endif  

