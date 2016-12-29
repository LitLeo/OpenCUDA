//IsolatedPoints.h
//创建人：王春峰
//
// 孤立点检测（IsolatedPoints）
// 功能说明：根据给定模板，利用拉普拉斯二阶导数对邻域像素进行卷积运算，
//           然后进行二值化操作，实现图像的孤立点检测
//
// 修订历史：
// 2013年03月22日（王春峰）
//    初始版本。
// 2013年03月28日（王春峰）
//    修正了一些代码注释规范，修改了模板的 set 和 get 函数。
// 2013年03月30日（王春峰）
//    添加了无参构造函数
// 2013年04月01日（王春峰）
//    修改了模板的初始化，删除了模板的 get 和 set 函数，修改了构造函数为 host 
//    端，修正了一些代码格式
// 2013年04月02日（王春峰）
//    删除了模板的成员变量，改为 device 端的静态数组。
// 2013年04月03日（王春峰）
//    修改了部分变量名和部分代码的结构。加入了一些宏定义
// 2013年04月05日（王春峰）
//    修改了核函数算法
// 2013年04月11日（王春峰）
//    修正了成员变量的意义，由灰度阈值变为灰度差值（孤立点检测的灰度差值标准）

#ifndef __ISOLATEDPOINTS_H__
#define __ISOLATEDPOINTS_H__

#include "Image.h"
#include "ErrorCode.h"


// 类：IsolatedPoints（孤立点检测算法）
// 继承自：无。
// 该类用于根据给定模板，利用拉普拉斯二阶导数对邻域像素进行卷积运算，
// 然后进行二值化操作，实现图像的孤立点检测。
class IsolatedPoints {

protected:

    // 成员变量：threshold（灰度差值）
    // 二值化判断的标准。
    unsigned char threshold;

public:

    // 构造函数：IsolatedPoints
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__
    IsolatedPoints()
    {
        // 使用默认值为类的成员变量赋值。
        this->threshold = 50;
    }

    // 构造函数：IsolatedPoints
    // 有参数版本的构造函数，根据需要给定参数，参数值在程序运行过程中
    // 还是可以改变的。
    __host__
    IsolatedPoints(
            unsigned char threshold  // 灰度差值
    ) {
        // 使用默认值为类的成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        // 根据参数列表中的值设定成员变量的初值
        this->threshold = 50;
 
        // 根据参数列表中的值设定成员变量的初值
        setThreshold(threshold);
    }   
       
    // 成员方法：getThreshold（获取灰度差值）
    // 获取孤立点检测二值化差值   
    __host__ __device__ unsigned char  // 返回值：当前灰度差值
    getThreshold() const
    {
        // 返回 threshold 成员变量的值。
        return this->threshold;   
    }
    
    // 成员方法：setThreshold（设置灰度差值）
    // 设置灰度差值
    __host__ __device__ int             // 返回值：若函数正确执行，返回 NO_ERROR
    setThreshold(   
            unsigned char impthreshold  // 新的灰度差值
    ) {
        // 将 threshold 成员变量赋成新值
        this->threshold = impthreshold;

        return NO_ERROR;
    }   

    // Host 成员方法：isopointDetect（执行孤立点检测）
    // 对图像进行孤立点检测操作，根据模板，计算出当前点的像素值，二值化并输出
    __host__ int           // 返回值：函数正确执行，返回 NO_ERROR，
                           // 否则返回相应的错误码
    isopointDetect(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );

};

#endif

