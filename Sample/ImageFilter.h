// ImageFilter.h
// 
// 多阈值图像过滤（image filter）
// 功能说明：给定一幅图像，根据用户输入的阈值数组，取出每个阈值区间对应的图像，如阈值数组为如[0,50,100,150,200],
//           则该算法输出5幅图像，第一幅保留原图像的（0,50）灰度值，其余灰度值设为0，第二幅保留原图像的（50,100）
//           灰度值，以此类推，
//           最后一幅保留原图像的（200，255）灰度值。用户输入阈值时应注意，若您选定的阈值为50,100,150,200,
//           则您输入的阈值数组应为[0,50,100,150,200]，前面加0，后面不加255。之所以这么处理，使阈值数组元素个数
//           和输出图像函数个数一致。
//           


#ifndef __IMAGEFILTER_H__
#define __IMAGEFILTER_H__
#include "Image.h"
#include "ErrorCode.h"

// 类：ImageFilter
// 继承自：无
// 给定一幅图像，根据用户输入的阈值数组，取出每个阈值区间对应的图像，如阈值数组
// 为如[0,50,100,150,200],则该算法输出5幅图像，第一幅保留原图像的（0,50）灰度值
// ，其余灰度值设为0，第二幅保留原图像的（50,100）灰度值，以此类推，最后一幅
// 保留原图像的（200，255）灰度值。用户输入阈值时应注意，若您选定的阈值为
// 50,100,150,200,则您输入的阈值数组应为[0,50,100,150,200]，前面加0，后面
// 不加255。之所以这么处理，使阈值数组元素个数和输出图像函数个数一致。
class ImageFilter {
protected:

    // 成员变量：threshold
    // 阈值数组
    int *threshold;

    // 成员变量：thresholdNum
    // 阈值数组大小
    int thresholdNum;

public:

    // 无参构造函数：ImageFilter
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值
    // 阈值数组：[0,50,100,150,200]
    // 阈值数组大小：5
    __host__ 
    ImageFilter() {

        // 阈值数组大小初始化
        thresholdNum = 5;

        // 阈值数组开辟空间
        threshold = new int[5];

        // 初始化阈值数组
        for(int i = 0; i < thresholdNum; i++) {
            threshold[i] = 50 * i;
        }
    }

    // 有参构造函数：ImageFilter
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 可以改变的。
    __host__
    ImageFilter(
            int *threshold,   //阈值数组 
            int thresholdnum  //阈值数组大小
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给
        // 了非法的初始值而使系统进入一个未知的状态。
        this->thresholdNum = 5;
        this->threshold = new int[5];
        for(int i = 0; i < thresholdNum; i++) {
            this->threshold[i] = 50 * i;
        }

        // 根据参数列表中的值设定成员变量的初值
        setThresholdAndNum(threshold, thresholdnum);
    }

    // 成员方法： setThresholdAndNum
    // 设置成员变量 threshold 和 thresholdNum
    __host__ int              // 返回值：函数是否正确执行，若
                              // 函数正确执行，返回 NO_ERROR  
    setThresholdAndNum(
            int *threshold,   // 阈值数组 
            int thresholdnum  // 阈值数组大小
    ) {
        // 判断用户输入是否合法
        if(threshold == NULL || thresholdnum <= 0) {
            return INVALID_DATA;
        }
        else {

            // 释放原阈值数组空间
            delete []this->threshold;

            // 设置 thresholdNum
            this->thresholdNum = thresholdnum;

            // 为 threshold 开辟空间
            this->threshold = new int[thresholdNum];

            // 初始化 threshold 数组 
            for(int i = 0; i < thresholdNum; i++) {
                this->threshold[i] = threshold[i];
            }

            return NO_ERROR;
        }

    }

    // 成员方法：getThreshold
    // 获取 threshold 成员变量的首地址
    __host__ int *         // 返回值：当前 threshold 成员变量的首地址  
    getThreshold() const   //
    {
        // 返回 threshold 成员变量的首地址
        return threshold;  
    }

    // 成员方法：getThresholdNum
    // 获取 thresholdNum 成员变量的值
    __host__ int              // 返回值： 当前 thresholdNum 成员变量的值
    getThresholdNum() const   //
    {
        // 返回 thresholdNum 成员变量的值
        return thresholdNum;  
    }

   // void calImageFilter(Image *inimg, Image ***outimg);

    // 成员方法：calImageFilterOpt (多阈值图像过滤)
    // 给定一幅图像，根据用户输入的阈值数组，取出每个阈值区间对应的图像
    __host__ int             // 返回值：函数是否正确执行，若函数正确执行，
                             // 返回 NO_ERROR 
    calImageFilterOpt(
            Image *inimg,    // 输入图像   
            Image ***outimg  // 输出图像数组  
    );                                                    

    // 析构函数：~ImageFilter
    // 释放 threshold 成员变量
    __host__
    ~ImageFilter() {
        // 释放 threshold 成员变量
        delete []threshold;
    }
};
#endif
