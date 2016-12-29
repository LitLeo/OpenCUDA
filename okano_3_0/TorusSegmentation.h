// TorusSegmentation.h
// 创建人：邱孝兵
//
// 圆环分割（TorusSegmentation）
// 功能说明：这个类用于进行圆环分割。其输入为一个图像，和代表该图像中一个圆环形
// 区域的坐标集，目标为将该环形区域划分为两个部分，其中一个分割闭合包围该圆环
// 内部区域。分割是根据当前坐标集的点一定邻域范围内是否有坐标集外的点，如果有即
// 标记为同一个类别，这样就可以同时找出一个包围环和一个被包围环，实现目标。
//
// 修订历史：
// 2013年03月28日（邱孝兵）
//     初始版本
// 2013年04月01日（邱孝兵）
//     修改部分代码格式问题


#include "Image.h"
#include "ErrorCode.h"
#include "CoordiSet.h"


#ifndef __TORUS_SEGMENTATION_H__
#define __TORUS_SEGMENTATION_H__


// 类：TorusSegmentation
// 继承自：无
// 这个类用于进行圆环分割。其输入为一个图像，和代表该图像中一个圆环形
// 区域的坐标集，目标为将该环形区域划分为两个部分，其中一个分割闭合包围该圆环
// 内部区域。分割是根据当前坐标集的点一定邻域范围内是否有坐标集外的点，如果有即
// 标记为同一个类别，这样就可以同时找出一个包围环和一个被包围环，实现目标。
class TorusSegmentation {

protected:

    // 成员变量：neighborSize（邻域宽度）
    // 定义用于检查是否边缘的宽度，超过当前处理的坐标集的点 
    // neighborSize 范围内有超出坐标集范围的点就确定为边缘。
    int neighborSize;    

public:

    // 构造函数：TorusSegmentation
    // 无参数版本的构造函数，所有成员变量均初始化为默认值。
    __host__ __device__
    TorusSegmentation() {
        // 无参数的构造函数，使用默认值初始化各个变量。
        this->neighborSize = 1;
    }
    
    // 构造函数：TorusSegmentation
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    TorusSegmentation(
        int neighborsize  // 邻域宽度
    ) {
        // 使用默认值初始化各个变量。
        this->neighborSize = 1;

        // 调用 seters 给各个成员变量赋值。
        this->setNeighborSize(neighborsize);
    }

    // 成员方法：getnNighborSize（获取 neighborSize 的值）
    // 获取成员变量  neighborSize 的值。
    __host__ __device__ int  // 返回值：成员变量 neighborSize 的值。
    getNeighborSize() const
    {
        // 返回成员变量 neighborSize 的值。
        return this->neighborSize;
    }
    
    // 成员方法：setNeighborSize（设置 neighborSize 的值）
    // 设置成员变量 neighborSize 的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，返回
                              // NO_ERROR。
    setNeighborSize(
            int neighborSize  // 外部参数 neighborSize
    ) {
        // 检查数据有效性。
        if (neighborSize <= 0) 
            return INVALID_DATA;

        // 设置成员变量 beta 的值。
        this->neighborSize = neighborSize;        
        return NO_ERROR;
    }    
    
    // 成员函数：torusSegregate（对圆环进行二分割）
    // 该方法实现圆环分割。其输入为一个图像，和代表该图像中一个圆环形
    // 区域的坐标集，目标为将该环形区域划分为两个部分，其中一个分割闭合
    // 包围该圆环内部区域。分割是根据当前坐标集的点一定邻域范围内是否
    // 有坐标集外的点，如果有即标记为同一个类别，剩下的标记为另一个类别。
    __host__ int                 // 返回值：函数是否正确执行，若函数
                                 // 正确执行，返回 NO_ERROR。
    torusSegregate(
        int width,               // 坐标集所在图像的宽度
        int height,              // 坐标集所在图像的高度
        CoordiSet *incoordiset,  // 输入坐标集
        unsigned char *outlbl    // 输出特征向量
    );

    // 成员函数：torusSegregateToImg（对圆环进行二分割，结果体现到图像上）
    // 在 torusSegregate 方法中，标记结果存储在一个线性数组中，这里为了直观
    // 重新将结果体现到二维图像中去。
    __host__ int                 // 返回值：函数是否正确执行，若函数
                                 // 正确执行，返回 NO_ERROR。
    torusSegregateToImg(
        int width,               // 坐标集所在图像的宽度
        int height,              // 坐标集所在图像的高度
        CoordiSet *incoordiset,  // 输入坐标集
        Image *outimg            // 输出图像
    );
};

#endif

