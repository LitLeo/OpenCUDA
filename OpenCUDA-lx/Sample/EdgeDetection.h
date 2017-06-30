// EdgeDetection.h

// 边缘检测（EdgeDetection）
// 功能说明：实现画出两种颜色的边界。


#ifndef __EDGEDETECTION_H__
#define __EDGEDETECTION_H__

#include "Image.h"

// 类：EdgeDetection（边缘检测）
// 继承自：无
// 一幅图像中有两块区域，颜色不同。画出两种颜色的边界。
class EdgeDetection {

protected:

    // 成员变量：drawcolor (边界颜色)
    // 边界的颜色，范围是 [0, 255]。
    unsigned char drawcolor;

public:

    // 构造函数：EdgeDetection
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    EdgeDetection()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->drawcolor = 155;  // 边界颜色默认为 155。
    }

    // 构造函数：EdgeDetection
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    EdgeDetection(
            unsigned char drawcolor  // 边界颜色
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->drawcolor = 155;  // 边界颜色默认为 155。
    
        // 根据参数列表中的值设定成员变量的初值
        setDrawcolor(drawcolor);
    }
   
    // 成员方法：getDrawcolor（获取边界颜色）
    // 获取成员变量 drawcolor 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 drawcolor 的值
    getDrawcolor() const
    {
        // 返回 drawcolor 成员变量的值。
        return this->drawcolor;
    } 

    // 成员方法：setDrawcolor（设置边界颜色）
    // 设置成员变量 drawcolor 的值。
    __host__ __device__ int          // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    setDrawcolor(
            unsigned char drawcolor  // 设定新的边界颜色
    ) {
        // 将 drawcolor 成员变量赋成新值
        this->drawcolor = drawcolor;

        return NO_ERROR;
    }

    // Host 成员方法：edgeDetection（边缘检测）
    // 实现画出两种颜色的边界。
    __host__ int               // 返回值：若函数正确执行，返回 NO_ERROR，否则返回
                               // 相应的错误码
    edgeDetection(
                Image *inimg,  // 输入图像
                Image *outimg  // 输出图像
    );
};

#endif

