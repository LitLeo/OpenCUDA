// DownSampleImage.h
// 创建人：李冬
//
// 缩小图像（DownSampleImage）
// 功能说明：根据给定的缩小倍数 N，将输入图像缩小，将其尺寸从  
// width * height 变成 (width / N) * (height / N)

#ifndef __DOWNSAMPLEIMAGE_H__
#define __DOWNSAMPLEIMAGE_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：DownSampleImage
// 继承自：无
// 根据给定的缩小倍数 N，将输入图像缩小，将其尺寸从 width * height 变成 
// (width / N) * (height / N)
class DownSampleImage {

protected:

    // 成员变量：times（缩小倍数 N）
    // 图像缩小倍数，图像长和宽各缩小times倍。
    int times;

public:

    // 构造函数：DownSampleImage
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    DownSampleImage()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->times = 2;  // 缩小倍数默认为 2。
    }
    
    // 构造函数：DownSampleImage
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在
    // 程序运行过程中还是可以改变的。
    __host__ __device__  
    DownSampleImage(
           int times     // 缩小倍数 N。
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的
        // 参数中给了非法的初始值而使系统进入一个未知的状态。
        this->times = 2;  // 缩小倍数默认为 2。
    
        // 根据参数列表中的值设定成员变量的初值
        setTimes(times);
    }
   
    // 成员方法：getTimes（获取缩小倍数 N）
    // 获取成员变量 times 的值。
    __host__ __device__ int  // 返回值：成员变量 times 的值
    getTimes() const
    {
        // 返回 times成员变量的值。
        return this->times;
    }  

    // 成员方法：setTimes（设置缩小倍数 N）
    // 设置成员变量 times 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setTimes(
            int times        // 设定新的 times 的值
    ) {
        // 判断缩小倍数是否合理 
        if (times <= 1)
            return INVALID_DATA; 
        
        // 将 times 成员变量赋成新值
        this->times = times;
        return NO_ERROR;
        
    }

    // Host 成员方法：dominanceDownSImg（图像缩小处理）
    // 根据给定的缩小倍数 N，将输入图像缩小，将其尺寸从 width * height 变成 
    // (width / N) * (height / N)
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    dominanceDownSImg(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    ); 
    
    // Host 成员方法：probabilityDownSImg（图像缩小处理）
    // 根据给定的缩小倍数 N，用概率法将输入图像缩小，将其尺寸从  
    // width * height 变成(width / N) * (height / N)。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    probabilityDownSImg(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    ); 
};

#endif

