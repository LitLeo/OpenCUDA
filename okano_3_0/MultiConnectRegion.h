// MultiConnectRegion.h
// 创建人：丁燎原
// 
// 多图连通区域 （MultiConnectRegion）
// 功能说明：给定一幅图像，首先根据用户输入的阈值数组，取出每个阈值区间
//           对应的图像（详见 ImageFilter ），然后调用 ConnectRegion 对
//           每一幅图像做连通区域操作。
//           
// 修订历史：
// 2014年10月10日（丁燎原）
//     初始版本

#ifndef __MULTICONNECTREGION_H__
#define __MULTICONNECTREGION_H__

#include "Image.h"
#include "ErrorCode.h"
#include "ImageFilter.h"
#include "ConnectRegion.h"

// 类：MultiConnectRegion
// 继承自：无
// 给定一幅图像，首先根据用户输入的阈值数组，取出每个阈值区间对应的图像  
// （详见 ImageFilter ），
// 然后调用 ConnectRegion 对每一幅图像做连通区域操作。
class MultiConnectRegion {
protected:
    // 成员变量：thresholdArr
    // 阈值数组
    int *thresholdArr;

    // 成员变量：thresholdNum
    // 阈值数组大小
    int thresholdNum;

    // 成员变量：threshold（给定阈值）
    // 进行区域连通的给定值，当两个点满足八邻域关系，
    // 且灰度值之差的绝对值小于该值时，这两点属于同一区域。
    int threshold;

    // 成员变量：maxArea 和 minArea（区域面积的最小和最大值）
    // 进行区域面积判断时的面积值最大最小的范围。
    int maxArea, minArea; 

public:
    // 构造函数：ConnectRegion
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值
    __host__
    MultiConnectRegion() {
        // 阈值数组大小初始化
        thresholdNum = 5;

        // 阈值数组开辟空间
        thresholdArr = new int[5];

        // 初始化阈值数组
        for(int i = 0; i < thresholdNum; i++) {
            thresholdArr[i] = 50 * i;
        }
        this->threshold = 0;    // 给定阈值默认为0
        this->maxArea = 100000; // 区域最大面积默认为100000
        this->minArea = 0;    // 区域最小面积默认为10
    }

    // 构造函数：ConnectRegion
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    MultiConnectRegion(
            int *thresholdArr,  // 阈值数组 
            int thresholdnum,   // 阈值数组大小 
            int threshold,      // 给定的标记阈值 
            int maxArea,        // 区域面积的最大值
            int minArea         // 区域面积的最小值
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给
        // 了非法的初始值而使系统进入一个未知的状态。
        this->thresholdNum = 5;
        this->thresholdArr = new int[5];
        for(int i = 0; i < thresholdNum; i++) {
            this->thresholdArr[i] = 50 * i;
        }
        this->threshold = 0;
        this->maxArea = 100000;
        this->minArea = 0;

        // 根据参数列表中的值设定成员变量的初值
        setThresholdArrAndNum(thresholdArr, thresholdnum);
        setThreshold(threshold);
        setMaxArea(maxArea);
        setMinArea(minArea);
    }

    // 成员方法： setThresholdArrAndNum
    // 设置成员变量 thresholdArr 和 thresholdNum
    __host__ int              // 返回值：函数是否正确执行，若
                              // 函数正确执行，返回 NO_ERROR  
    setThresholdArrAndNum(
            int *thresholdArr,   // 阈值数组 
            int thresholdnum  // 阈值数组大小
    ) {
        // 判断用户输入是否合法
        if(thresholdArr == NULL || thresholdnum <= 0) {
            return INVALID_DATA;
        }
        else {

            // 释放原阈值数组空间
            delete []this->thresholdArr;

            // 设置 thresholdNum
            this->thresholdNum = thresholdnum;

            // 为 thresholdArr 开辟空间
            this->thresholdArr = new int[thresholdNum];

            // 初始化 thresholdArr 数组 
            for(int i = 0; i < thresholdNum; i++) {
                this->thresholdArr[i] = thresholdArr[i];
            }

            return NO_ERROR;
        }

    }

    // 成员方法：setThreshold（设置阈值）
    // 设置 threshold 成员变量的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，返回
                              // NO_ERROR。
    setThreshold( 
            int threshold     // 指定的阈值大小。
    ) {
        // 将 threshold 成员变量赋值成新值
        this->threshold = threshold;
    
        return NO_ERROR;
    }    

    // 成员方法：setMinArea（设置进行区域面积判断时的最小面积值）
    // 设置 minArea 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMinArea( 
            int minArea      // 指定的进行区域面积判断时的最小面积值。
    ) {
        // 将 minArea 成员变量赋成新值
        this->minArea = minArea;
    
        // 处理完毕，返回。
        return NO_ERROR;
    }

    // 成员方法：setMaxArea（设置进行区域面积判断时的最大面积值）
    // 设置 maxArea 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMaxArea( 
            int maxArea      // 指定的进行区域面积判断时的最大面积值。
    ) {
        // 将 maxArea 成员变量赋成新值
        this->maxArea = maxArea;
    
        // 处理完毕，返回。
        return NO_ERROR;
    }

    // 成员方法：getThresholdArr
    // 获取 thresholdArr 成员变量的首地址
    __host__ int *         // 返回值：当前 thresholdArr 成员变量的首地址  
    getThresholdArr() const   //
    {
        // 返回 thresholdArr 成员变量的首地址
        return thresholdArr;  
    }

    // 成员方法：getThresholdNum
    // 获取 thresholdNum 成员变量的值
    __host__ int              // 返回值： 当前 thresholdNum 成员变量的值
    getThresholdNum() const   //
    {
        // 返回 thresholdNum 成员变量的值
        return thresholdNum;  
    }

    // 成员方法：getThreshold（读取阈值）
    // 读取 threshold 成员变量的值。
    __host__ __device__ int   // 返回值：当前 threshold 成员变量的值。
    getThreshold() const
    {
        // 返回 threshold 成员变量的值。
        return this->threshold;
    } 

    // 成员方法：getMinArea（读取进行区域面积判断时的最小面积值）
    // 读取 minArea 成员变量的值。
    __host__ __device__ int  // 返回值：当前 minArea 成员变量的值。
    getMinArea() const
    {
        // 返回 minArea 成员变量的值。
        return this->minArea;
    }

    // 成员方法：getMaxArea（读取进行区域面积判断时的最小面积值）
    // 读取 maxArea 成员变量的值。
    __host__ __device__ int  // 返回值：当前 maxArea 成员变量的值。
    getMaxArea() const
    {
        // 返回 maxArea 成员变量的值。
        return this->maxArea;
    }

    // 成员方法：multiConnectRegion (多图连通区域)
    // 给定一幅图像，首先根据用户输入的阈值数组，取出每个阈值区间对应的图像
    // （详见 ImageFilter ），
    // 然后调用 ConnectRegion 对每一幅图像做连通区域操作。注：outimg 不需要用户
    // 开辟空间，但需用户释放空间。
    __host__ int             // 返回值：函数是否正确执行，若函数正确执行，
                             // 返回 NO_ERROR 
    multiConnectRegion(
            Image *inimg,    // 输入图像   
            Image ***outimg  // 输出图像数组  
    );

    // 析构函数：~ImageFilter
    // 释放 threshold 成员变量
    __host__
    ~MultiConnectRegion() {
        // 释放 thresholdArr 成员变量
        delete []thresholdArr;
    }
};

#endif
