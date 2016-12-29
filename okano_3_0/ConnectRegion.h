// ConnectRegion.h
// 创建人：王媛媛
//
// 连通区域（ConnectRegion）
// 功能说明：根据指定阈值将输入图像分割成若干区域，并根据给定最大最小面积
//           将满足条件的区域按序进行标记（1,2……255），最多标记255个区域，
//           其中未被标记的区域的标记值将被置为0。
//
// 修订历史：
// 2012年08月31日（王媛媛）
//     初始版本
// 2012年09月12日（王媛媛）
//     实现了并行的旋转表计算。
// 2012年09月13日（王媛媛）
//     更正了代码的一些错误和注释规范。
// 2012年10月25日（王媛媛）
//     更正了代码的一些注释规范。

#ifndef __CONNECTREGION_H__
#define __CONNECTREGION_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：ConnectRegion
// 继承自：无
// 根据参数 threshold，给定区域面积的最大最小值，将满足条件的形状区域进行按序标记
// 后拷贝到输出图像中。
class ConnectRegion {

protected:

    // 成员变量：threshold（给定阈值）
    // 进行区域连通的给定值，当两个点满足八邻域关系，
    // 且灰度值之差的绝对值小于该值时，这两点属于同一区域。
    int threshold;
	
    // 成员变量：maxArea和 minArea（区域面积的最小和最大值）
    // 进行区域面积判断时的面积值最大最小的范围。
    int maxArea, minArea;
	
public:
    // 构造函数：ConnectRegion
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值
    __host__ __device__
    ConnectRegion() {
        // 使用默认值为类的各个成员变量赋值
        this->threshold = 0;    // 给定阈值默认为0
        this->maxArea = 100000; // 区域最大面积默认为100000
        this->minArea = 100;    // 区域最小面积默认为100
    }
	
    // 构造函数：ConnectRegion
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    ConnectRegion(
	    int threshold,            // 给定的标记阈值
            int maxArea, int minArea  // 区域面积的最大值和最小值
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法的初
        // 始值而使系统进入一个未知的装填
        this->threshold = 0;
        this->maxArea = 100000;
        this->minArea = 100;
    
        // 根据参数列表中的值设定成员变量的初值
        setThreshold(threshold);
        setMaxArea(maxArea);
        setMinArea(minArea);
    }
	
    // 成员方法：getThreshold（读取阈值）
    // 读取 threshold 成员变量的值。
    __host__ __device__ int   // 返回值：当前 threshold 成员变量的值。
    getThreshold() const
    {
        // 返回 threshold 成员变量的值。
        return this->threshold;
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

    // 成员方法：getMinArea（读取进行区域面积判断时的最小面积值）
    // 读取 minArea 成员变量的值。
    __host__ __device__ int  // 返回值：当前 minArea 成员变量的值。
    getMinArea() const
    {
        // 返回 minArea 成员变量的值。
        return this->minArea;
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

    // 成员方法：getMaxArea（读取进行区域面积判断时的最小面积值）
    // 读取 maxArea 成员变量的值。
    __host__ __device__ int  // 返回值：当前 maxArea 成员变量的值。
    getMaxArea() const
    {
        // 返回 maxArea 成员变量的值。
        return this->maxArea;
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

    // Host 成员方法：connectRegion（连通区域的提取与标记）
    // 根据参数 threshold，给定区域面积的最大最小值，将满足条件的形状区域进行按序标记
    // 后拷贝到输出图像中。如果输出图像 outimg 为空，则返回错误。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    connectRegion(
            Image *inimg,  // 输入图像。
            Image *outimg  // 输出图像。
    );
};

#endif

