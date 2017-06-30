// DownSampleImage.h
//
// 拉伸图像（ImageStretch）
// 功能说明：根据给定的长宽拉伸倍数 timesWidth 和 timesHeight，将输入图像拉伸，
//           将其尺寸从 width * height 变成(width * timesWidth) * 
//           (height * timesHeight)。

#ifndef __IMAGESTRETCH_H__
#define __IMAGESTRETCH_H__

#include "Image.h"

// 类：ImageStretch
// 继承自：无
// 根据给定的宽高拉伸倍数 timesWidth 和 timesHeight，将输入图像拉伸，将其尺寸从
// width * height 变成 (width * timesWidth) * (height * timesHeight)。
class ImageStretch {

protected:

    // 成员变量：timesWidth（图像宽度拉伸倍数）
    // 图像宽度拉伸倍数，图像宽拉伸 timesWidth 倍。
    float timesWidth;
 
    // 成员变量：timesHeight（图像高度拉伸倍数）
    // 图像高度拉伸倍数，图像高度拉伸 timesHeight 倍。
    float timesHeight;
  
public:

    // 构造函数：ImageStretch
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    ImageStretch()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->timesWidth = 2.0f;   // 宽度拉伸倍数默认为 2.0f。
        this->timesHeight = 2.0f;  // 高度拉伸倍数默认为 2.0f。
    }
    
    // 构造函数：ImageStretch
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    ImageStretch(
           float timeswidth,      // 宽度拉伸倍数。
           float timesheight      // 高度拉伸倍数。
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->timesWidth = 2.0f;   // 宽度拉伸倍数默认为 2.0f。
        this->timesHeight = 2.0f;  // 高度拉伸倍数默认为 2.0f。
    
        // 根据参数列表中的值设定成员变量的初值。
        setTimesWidth(timeswidth);
        setTimesHeight(timesheight);
    }
   
    // 成员方法：getTimesWidth（获取宽度拉伸倍数）
    // 获取成员变量 timesWidth 的值。
    __host__ __device__ float  // 返回值：成员变量 timesWidth 的值。
    getTimesWidth() const
    {
        // 返回 timesWidth成员变量的值。
        return this->timesWidth;
    }

    // 成员方法：setTimesWidth（设置宽度拉伸倍数）
    // 设置成员变量 timesWidth 的值。
    __host__ __device__ float  // 返回值：函数是否正确执行，若函数正确执行，返回
                               // NO_ERROR。
    setTimesWidth(
            float timeswidth   // 设定新的 timesWidth 的值。
    ) {
        // 判断拉伸倍数是否合理 
        if (timeswidth <= 0)
            return INVALID_DATA; 
  
        // 将 timesWidth 成员变量赋成新值。
        this->timesWidth = timeswidth;
        return NO_ERROR;
    }

    // 成员方法：getTimesHeight（获取高度拉伸倍数）
    // 获取成员变量 timesHeight 的值。
    __host__ __device__ float  // 返回值：成员变量 timesHeight 的值。
    getTimesHeight() const
    {
        // 返回 timesHeight成员变量的值。
        return this->timesHeight;
    } 

    // 成员方法：setTimesHeight（设置高度拉伸倍数）
    // 设置成员变量 timesHeight 的值。
    __host__ __device__ float  // 返回值：函数是否正确执行，若函数正确执行，返回
                               // NO_ERROR。
    setTimesHeight(
            float timesheight  // 设定新的 timesHeight 的值。
    ) {        
        // 判断拉伸倍数是否合理 
        if (timesheight <= 0)
            return INVALID_DATA;   
    
        // 将 timesHeight 成员变量赋成新值。
        this->timesHeight = timesheight;
        return NO_ERROR; 
    }    
    
    // Host 成员方法：performImgStretch（图像拉伸处理）
    // 根据给定的拉伸倍数 timesWidth 和 timesHeight，将输入图像缩小，将其尺寸从
    // width * height 变成(width * timesWidth) * (height * timesHeight)。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    performImgStretch(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    ); 
};

#endif
