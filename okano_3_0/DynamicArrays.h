// DynamicArrays.h
// 创建者：欧阳翔
// 
// 动态数组（DynamicArrays）
// 功能说明：辅助性数据结构，为了便于管理内存分配和释放，主要是为了实现曲线跟踪
//           设计的。其中也实现了栈的部分功能，当作为曲线跟踪存储点的坐标时，
//           是成对存在的，也就是说两个坐标时同时添加进去，任何时候，当前数组
//           大小是偶数个。当作为栈时，不限制数组大小是奇数还是偶数。
//
// 修订历史：
// 2013年08月01日（欧阳翔）
//     初始版本，动态数组的实现
// 2013年09月10日（欧阳翔）
//     增加了 delElemXY 函数，用于删除动态数组中坐标为 x y 的点。

#ifndef _DYNAMICARRAYS_H
#define _DYNAMICARRAYS_H

#include <iostream>
#include "ErrorCode.h"


// 类：DynamicArrays（动态数组）
// 继承自：无
// 广义的图像中值滤波。具有两种不同的方法实现：（1）根据给定半径 radius
// 辅助性数据结构，为了便于管理内存分配和释放，主要是为了实现曲线跟踪设计的。
// 其中也实现了栈的部分功能，当作为曲线跟踪存储点的坐标时，成对存在的，
// 也就是说两个坐标时同时添加进去，任何时候，当前数组大小是偶数个。当作为栈时， 
// 不限制数组大小是奇数还是偶数。
class DynamicArrays {

protected:

    // 成员变量：array（整型指针参数）
    // 动态数组自己的内部指针
    int *array;
    
    // 成员变量：size（动态数组元素个数参数）
    // 表示当前动态数组元素个数
    int size;
    
    // 成员变量：maxsize（动态数组最大容量参数）
    // 表示当前动态数组最大容量数
    int maxsize;
 
public:

    // 构造函数： DynamicArrays
    // 当外界没有设置数组最大容量数时，选择默认值，否则设置最大容量数，并初始化
    // 其他成员变量
    __host__ __device__
    DynamicArrays(
            int sz = 50            // 默认设置为 50 否则设置为外界给定的 sz
    ) {
        this->size  = 0;           // 初始时，动态数组没有元素，当前元素
                                   // 个数为 0
        this->maxsize = sz;        // 设置当前容量大小
        array = new int[maxsize];  // 动态申请大小
    }

    // 复制构造函数： DynamicArrays
    // 由于涉及到对象当做参数传递的情况，并且动态数组需要动态管理内存，需要实现
    // 复制构造函数，防止出现意外的错误
    __host__ __device__
    DynamicArrays(const DynamicArrays &object) {
        // 复制一份新的内存空间，并且拷贝出一份完全一样的数据
        size = object.size;
        maxsize = object.maxsize;
        array = new int[maxsize];
        memcpy(array, object.array, size * sizeof (int));
    }

    // 成员方法：getSize（得到当前动态数组的元素个数）
    // 读取 size 成员变量的值
    __host__ __device__ int
    getSize() const
    {
        // 返回 size 成员变量的值
        return size;
    }

    // 成员方法：getMaxsize（得到当前动态数组的最大容量）
    // 读取 maxsize 成员变量的值
    __host__ __device__ int
    getMaxsize() const
    {
        // 返回 maxsize 成员变量的值
        return maxsize;
    }

    // 成员方法：array（得到当前动态数组的指针）
    // 读取 array 成员变量的值
    __host__ __device__ int *
    getCrvDatap() const
    {
        // 返回 array 成员变量的值
        return array;
    }

    // 成员方法：operator[] （重载中括号）
    // 得到下标为 i 的元素
    __host__ int &
    operator[](int i) {
        // 得到下标为 i 的元素
        return array[i];
    }
 
    // 成员方法：addElem（往动态数组增加元素）
    // 往动态数组末尾增加一个元素
    __host__ int      // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR
    addElem(
            int elem  // 增加的元素 elem
    );
 
    // 成员方法：addTail（往动态数组末尾增加两个元素）
    // 往动态数组末尾同时增加两个元素，赋值曲线坐标的 x 轴和 y 轴坐标
    __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR
    addTail(
            int x,  // 末尾添加的元素 x
            int y   // 末尾添加的元素 y
    );

    // 成员方法：addHead（往动态数组头部增加两个元素）
    // 往动态数组首部同时增加两个元素，赋值曲线坐标的 x 轴和 y 轴坐标
    __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR
    addHead(
            int x,  // 首部添加的元素 x
            int y   // 首部添加的元素 y
    );

    // 成员方法：delElem（删除数组中为相邻值 x，y 的点）
    // 曲线跟踪辅助函数，删除曲线中值为 x，y 的点，这是一种快速删除方式
    __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR
    delElem(
            int x,  // 删除值为 x 的数
            int y   // 删除值为 y 的数
    );

    // 成员方法：delElemXY（删除数组中为相邻值 x，y 的点）
    // 曲线跟踪辅助函数，删除曲线中值为 x，y 的点，这个删除方式相对前一种缓慢
    __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR
    delElemXY(
            int x,  // 删除值为 x 的数
            int y   // 删除值为 y 的数
    );

    // 成员方法：delTail（删除末尾最后一个数）
    // 实现动态数组实现栈的 pop 方式，并且得到栈顶元素
    __host__ int       // 返回值：函数是否正确执行，若函数正确执行，
                       // 返回 NO_ERROR
    delTail(
            int &elem  // 得到删除的末尾元素
    );

    // 成员方法：reverse（动态数组以成对坐标反转）
    // 实现动态数组得到的曲线坐标进行点坐标的反转
    __host__ int
    reverse();

    // 成员方法：findElem（查找动态数组里是否有元素 elem）
    // 查找动态数组里是否有外界给定的元素 elem
    __host__ bool     // 返回值：如果找到返回 true，否则返回 false
    findElem(
            int elem  // 要查找的元素
    );

    // 成员方法：findElem（查找动态数组里是否有要查找的坐标对）
    // 查找动态数组里是否有外界给定的坐标对
    __host__ bool     // 返回值：如果找到返回 true，否则返回 false
    findElemXY(
            int x,    // 要查找的横坐标 x
            int y     // 要查找的纵坐标 y
    );

    // 成员方法：addArray（动态数组的连接）
    // 连接两个动态数组，曲线跟踪辅助函数，实现两个曲线的连接
    __host__ int
    addArray(
            DynamicArrays &object  // 连接在后边的动态数组
    );

    // 成员方法：clear（动态数组内容的清空）
    // 辅助实现栈函数，用于栈的清空
    __host__ __device__ int   // 返回值：函数执行正确，返回 NO_ERROR
    clear()
    {
        // 设置 size 为 0
        size = 0;
        // 无错返回
        return NO_ERROR;
    }

    // 析构函数：~DynamicArrays
    // 释放内存空间
    __host__ __device__
    ~DynamicArrays() 
    {
        // 如果不为空就释放空间
        if (array != NULL)
            delete []array; 
    } 
};

#endif
