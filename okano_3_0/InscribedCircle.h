// InscribedCircle.h
// 创建者：欧阳翔
// 
// 内接圆（InscribedCircle）
// 功能说明：闭合曲线的最大内接圆，此算法要求得到闭合曲线的多个内接圆，按半径
//           长度排序，得到前 num 个最大内接圆的半径长度和圆心坐标

// 修订历史：
// 2014年8月10日（欧阳翔）
//     初始版本，曲线跟踪初步实现。

#ifndef __INSCRIBEDCIRCLE_H__
#define __INSCRIBEDCIRCLE_H__

#include "Curve.h"
#include "Graph.h"
#include "Image.h"
#include "DynamicArrays.h"
#include "ErrorCode.h"
#include "Template.h"
#include "TemplateFactory.h"
#include "SortArray.h"

// 类：InscribedCircle（内接圆）
// 继承自：无
// 内接圆：闭合曲线的最大内接圆，此算法要求得到闭合曲线的多个内接圆，按半径长度
// 排序，得到前 num 个最大内接圆的半径长度和圆心坐标

class InscribedCircle {

protected:

    // 成员变量：num（内接圆个数参数）
    // 外部设定的要得到的内接圆个数。一般是1~5之间的数
    int num;
    
    // 成员变量：disTh（距离参数阈值）
    // 外部设定的两个圆心之间的距离参数
    int disTh;

public:

    // 构造函数：InscribedCircle
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    InscribedCircle() {
        // 使用默认值为类的各个成员变量赋值。
        num = 1;               // 默认内接圆个数为 1
        disTh = 1;             // 默认两个圆心之间的距离默认为 1
    }

    // 构造函数：InscribedCircle
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    InscribedCircle(
            int num,   // 内接圆个数参数（具体解释见成员变量）
            int disTh  // 距离参数阈值（具体解释见成员变量）
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->num = 1;               // 默认内接圆个数为 1
        this->disTh = 1;             // 默认两个圆心之间的距离默认为 1
        
        // 根据参数列表中的值设定成员变量的初值
        this->setNum(num);
        this->setDisTH(disTh);
    }
    
    // 成员方法：getNum（得到内接圆个数）
    // 读取 num 成员变量的值。
    __host__ __device__ int  // 返回值：当前 num 成员变量的值。
    getNum() const
    {
        // 返回 num 成员变量的值。
        return this->num;
    }

    // 成员方法：setnum（设置内接圆个数参数）
    // 设置 num 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setNum(
            int num          // 新的内接圆个数参数
    ) {
        // 将 num 成员变量赋成新值
        this->num = num;
        return NO_ERROR;
    }
    
    // 成员方法：getDisTh（读取圆形邻域参数）
    // 读取 disTh 成员变量的值。
    __host__ __device__ int  // 返回值：当前 disTh 成员变量的值。
    getDisTh() const
    {
        // 返回 num 成员变量的值。
        return this->disTh;
    }

    // 成员方法：setDisTH（设置圆形邻域参数）
    // 设置 disTh 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setDisTH(
            int disTh        // 新的距离参数阈值参数
    ) {
        // 将 num 成员变量赋成新值
        this->disTh = disTh;
        return NO_ERROR;
    }
    
    // Host 成员方法：inscribedCircle（曲线最大内接圆）
    // 对图像进行曲线跟踪，得到非闭合曲线和闭合曲线的有序序列，根据输入参数设置
    // 大小具有一定的断点连接功能。
    __host__ int                 // 返回值：函数是否正确执行，若函数正确执行，返回
                                 // NO_ERROR。
    inscribedCircle(
            Curve *curve,     // 输入的闭合曲线
            int width,        // 曲线对应的图像宽度
            int height,       // 曲线对应的图像高度
            int &count,       // 得到的真正的符合要求的内接圆个数
            int *inscirDist,  // 得到的最大的 count 个内接圆半径序列
            int *inscirX,     // 得到的最大的 count 个内接圆的 X 坐标
            int *inscirY      // 得到的最大的 count 个内接圆的 Y 坐标
    ); 
};

#endif
