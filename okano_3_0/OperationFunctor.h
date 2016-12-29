// OperationFunctor.h 
// 创建人：王春峰
//
// 仿函数模板运算（OperationFunctor）
// 功能说明：实现各运算类型的模板运算。
//
// 修订历史：
// 2013年05月13日（王春峰）
//     初始版本
// 2013年05月13日（王春峰）
//     在每个类中增加了单位元
// 2013年05月16日（王春峰）
//     增加了 max 运算和 min 运算，同时宏定义了它们的单位元，无穷小与无穷大。

#ifndef __OPERATIONFUNCTOR_H__
#define __OPERATIONFUNCTOR_H__

#include<iostream>
using namespace std;

// 宏：OPR_LARGE_ENOUGH
// 定义了一个足够大的正整数，该整数在使用过程中被认为是无穷大。
#define OPR_LARGE_ENOUGH  ((1 << 30) - 1)

// 宏：OPR_SMALL_ENOUGH
// 定义了一个足够小的整数，该整数在使用过程中被认为是无穷小。
#define OPR_SMALL_ENOUGH  (-(1 << 30))

// 类：add_class
// 继承自：无
// 加法模板类，实现对运算符 () 的重载，以及获取加法运算的单位元。
template< class T >
class add_class {

public:
    
    // 成员方法：operator()（对 () 的重载）
    __host__ __device__ T 
    operator()(
               T A,  // 操作数 A
               T B   // 操作数 B
    ) const {
        // 返回 A、B 加和
        return A + B;
    }

    // 成员方法：identity（获取运算类型单位元）
    __host__ __device__ T 
    identity() 
    { 
        // 返回加法单位元 0。
        return (T)0; 
    }
};

// 类：multi_class
// 继承自：无
// 乘法模板类，实现对运算符 () 的重载。
template< class T >
class multi_class {

public:

    // 成员方法：operator()（对 () 的重载）
    __host__ __device__ T 
    operator()(
               T A,  // 操作数 A
               T B   // 操作数 B
    ) const {
        // 返回 A、B 乘积
        return A * B;
    }

    // 成员方法：identity（获取运算类型单位元）
    __host__ __device__ T 
    identity() 
    { 
        // 返回乘法单位元 1。
        return (T)1; 
    }
};

// 类：max_class
// 继承自：无
// 最大值运算模板类，实现对运算符 () 的重载。
template< class T >
class max_class {

public:

    // 成员方法：operator()（对 () 的重载）
    __host__ __device__ T 
    operator()(
               T A,  // 操作数 A
               T B   // 操作数 B
    ) const {
        // 返回 A、B 最大值
        return (A > B ? A : B);
    }

    // 成员方法：identity（获取运算类型单位元）
    __host__ __device__ T 
    identity() 
    { 
        // 返回最大值运算单位元。
        return (T)OPR_SMALL_ENOUGH; 
    }
};

// 类：min_class
// 继承自：无
// 最小值运算模板类，实现对运算符 () 的重载。
template< class T >
class min_class {

public:

    // 成员方法：operator()（对 () 的重载）
    __host__ __device__ T 
    operator()(
               T A,  // 操作数 A
               T B   // 操作数 B
    ) const {
        // 返回 A 和 B 中较小的值
        return (A < B ? A : B);
    }

    // 成员方法：identity（获取运算类型单位元）
    __host__ __device__ T 
    identity() 
    { 
        // 返回最小值运算单位元。
        return (T)OPR_LARGE_ENOUGH; 
    }
};

#endif