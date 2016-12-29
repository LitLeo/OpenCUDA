// ScanArray.h 
// 创建人：刘瑶
//
// 数组扫描（ScanArray）
// 功能说明：对一个数组进行扫描，对所有元素进行某操作的遍历。
// 举例：数组 1, 4, 5, 7, 10 进行 exclusive scan 的结果是：0, 1, 5, 10, 17。
// 参照 Mark Harris 的论文 “Parallel Prefix Sum (Scan) with CUDA” 进进行实现。
// 声明只支持单个一维线程块规模的数组扫描，不支持多块扫描和非满块扫描（要求单个
// 线程块必须为满块）。
//
// 修订历史：
// 2012年11月24日（刘瑶）
//     初始版本
// 2012年11月30日（刘瑶）
//     添加了完整的代码注释，进行了部分修改。
// 2012年12月17日（刘瑶）
//     添加了算法对于多个线程块处理的实现，完成部分代码。
// 2012年12月19日（刘瑶）
//     添加了扫描后的中间结果加回到扫描数组的功能，添加了部分注释。
// 2012年12月24日（刘瑶）
//     修改了部分代码，规范了代码格式及注释。
// 2012年01月05日（刘瑶）
//     修改数组扫描的成员方法，支持函数重载，支持 int 和 float 型的数组扫描。
//     核函数采用模板，支持 int 和 float 型的数组扫描。修改了部分注释。
// 2012年01月09日（刘瑶，王雪菲）
//     增加了 Scan 的 int 类型的 Exclusive 扫描，即不包含自身的扫描。
// 2012年03月14日（刘瑶）
//     修正了 OPTIMIZE_SCAN 和 BETTER_SCAN 两种类型的扫描，之前调用 scanArray
//     却得到 scanArrayExclusive 的结果，调用 scanArrayExclusive 结果错位。现在
//     更正了这两个问题。
// 2012年03月14日（刘瑶）
//     再次修正了 OPTIMIZE_SCAN 和 BETTER_SCAN 两种类型的扫描，结合凸壳的扫描
//     重新发现了这个错误。现在更正了这两个问题。
// 2013年05月10日（王春峰）
//     增加了 CPUIN_SCAN 和 CPUEX_SCAN 两种类型的扫描，增加了 CPU 端校验的函数
// 2013年05月11日（王春峰）
//     保留了 CPU_IN_SCAN，删除了 CPUEX_SCAN。scanArray 可以调用 包括 
//     CPU_IN_SCAN 在内的五种 inclusive 类型，scanArrayExclusive 可以通过调用
//     scanArray 来实现。为 scanArrayExclusive 函数增加了模板类。修改了一些代码
//     规范。
// 2013年05月13日（王春峰）
//     增加了仿函数的的模板运算，支持加法和乘法的 scan 运算。修改了一些运算单位
//     元的赋值。
// 2013年05月24日（王春峰）
//     增加了仿函数的的最大值和最小值运算。增加了 Naive 版本和 workefficient 版
//     本的后序扫描操作。
// 2013年06月18日（王春峰）
//     增加了五种方法扫描的 backward 版本。相应的修改了 addback 核函数。修改了一
//     些代码规范。

#ifndef __SCANARRAY_H__
#define __SCANARRAY_H__

#include "ErrorCode.h"
#include "OperationFunctor.h"
#include<iostream>
using namespace std;

// 宏：NAIVE_SCAN
// 用于设置 ScanArray 类中 scanType 成员变量，告知类的实例选用简单版本的
// ScanArray 实现。
#define NAIVE_SCAN      1

// 宏：EFFICIENT_SCAN
// 用于设置 ScanArray 类中 scanType 成员变量，告知类的实例选用效率版本的
// ScanArray 实现。
#define EFFICIENT_SCAN  2

// 宏：OPTIMIZE_SCAN
// 用于设置 ScanArray 类中 scanType 成员变量，告知类的实例选用优化版本的
// ScanArray 实现。
#define OPTIMIZE_SCAN   3

// 宏：BETTER_SCAN
// 用于设置 ScanArray 类中 scanType 成员变量，告知类的实例选用较优版本的
// ScanArray 实现。
#define BETTER_SCAN     4

// 宏：CPU_IN_SCAN
// 用于设置 ScanArray 类中 scanType 成员变量，告知类的实例选用 CPU inclusive 版
// 本的 ScanArray 实现。
#define CPU_IN_SCAN      5

// 类：ScanArray
// 继承自：无
// 对一个数组进行扫描，对所有元素进行某操作的遍历。根据论文中提到的三种方法，实
// 现 4 种方法。1:简单版本的扫描实现，每个线程处理一个元素。2:效率版本的扫描实
// 现，每个线程处理两个元素。3：优化版本的扫描实现，每个线程处理两个元素。
// 4：消除 bank conflicts 版本的扫描实现，每个线程处理两个元素。这三种方法都支持
// 多块任意长度的计算。
class ScanArray {

protected:

    // 成员变量：scanType（实现类型）
    // 设定四种扫描方法实现类型中的一种，在调用 scan 扫描函数的时候，使用对应的
    // 实现方式。
    int scanType;

    // Host 成员方法：addBack（float 型的中间结果加回）
    // 对一个数组进行扫描，将中间结果的小数组加回到扫描数组。
    template< class Operation >
    __host__ int 
    addBack(
            float *array,          // 初始扫描后的数组，float 型。
            float *lastelemarray,  // 中间结果数组，原扫描数组每段的最后一个元
                                   // 素提取出来即为中间结果数组，float 型。
            int numelements,       // 扫描数组的长度。
            int blocksize,         // 线程块的大小。
            int packnum,           // 扫描核函数每块计算能力，即核函数每块的处
                                   // 理长度与线程块大小的比值。
            Operation op,          // 运算类型
            bool backward          // 判断是否为后序扫描
    );

    // Host 成员方法：addBack（int 型的中间结果加回）
    // 对一个数组进行扫描，将中间结果的小数组加回到扫描数组。
    template< class Operation >
    __host__ int 
    addBack(
            int *array,          // 初始扫描后的数组，int 型
            int *lastelemarray,  // 中间结果数组，原扫描数组每段的最后一个元
                                 // 素提取出来即为中间结果数组，int 型
            int numelements,     // 扫描数组的长度。
            int blocksize,       // 线程块的大小。
            int packnum,         // 扫描核函数每块计算能力，即核函数每块的处
                                 // 理长度与线程块大小的比值。
            Operation op,        // 运算类型
            bool backward        // 判断是否为后序扫描
    );

public:

    // 构造函数：ScanArray
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    ScanArray()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->scanType = NAIVE_SCAN;  // 实现类型的默认值为 NAIVE_CAN，
                                      // 优化扫描。
    }

    // 构造函数：ScanArray
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    ScanArray(
            int scanType  // 实现类型
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->scanType = NAIVE_SCAN;  // 实现类型的默认值为 NAIVE_SCAN，
                                      // 优化扫描。

        // 根据参数列表中的值设定成员变量的初值
        this->setScanType(scanType);
    }

    // 成员方法：getScanType（读取实现类型）
    // 读取 scanType 成员变量的值。
    __host__ __device__ int  // 返回值：当前 scanType 成员变量的值。
    getScanType() const
    {
        // 返回 ScanType 成员变量的值。
        return this->scanType;
    }
    
    // 成员方法：setScanType（设置实现类型）
    // 设置 scanType 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setScanType(
            int scanType     // 实现类型
    ) {
        // 检查输入参数是否合法
        if (scanType != NAIVE_SCAN && scanType != EFFICIENT_SCAN && 
            scanType != OPTIMIZE_SCAN && scanType != BETTER_SCAN && 
            scanType != CPU_IN_SCAN)
            return INVALID_DATA;

        // 将 scanType 成员变量赋成新值
        this->scanType = scanType;
        return NO_ERROR;
    }

    // Host 成员方法：scanArray（float 型的数组扫描）
    // 对一个 float 型数组进行扫描，对所有元素进行某操作的遍历。
    template< class Operation >
    __host__ int                      // 返回值：函数是否正确执行，若函数正确
                                      // 执行，返回NO_ERROR。
    scanArray(
            float *inarray,           // 输入数组，float 型。
            float *outarray,          // 输出数组，float 型。
            int numelements,          // 数组的长度，处理元素的个数。
            Operation op,             // 运算类型
            bool backward,            // 判断是否为后序扫描
            bool hostinarray = true,  // 判断 inarray 是否是 Host 端的指针，
                                      // 默认为“是”。
            bool hostoutarray = true  // 判断 outarray 是否是 Host 端的指针，
                                      // 默认为“是”。
    );

    // Host 成员方法：scanArray（int 型的数组扫描）
    // 对一个 int 型数组进行扫描，对所有元素进行某操作的遍历。
    template< class Operation >
    __host__ int                      // 返回值：函数是否正确执行，若函数正确
                                      // 执行，返回NO_ERROR。
    scanArray(
            int *inarray,             // 输入数组，int 型
            int *outarray,            // 输出数组，int 型
            int numelements,          // 数组的长度，处理元素的个数。
            Operation op,             // 运算类型
            bool backward,            // 判断是否为后序扫描
            bool hostinarray = true,  // 判断 inarray 是否是 Host 端的指针，
                                      // 默认为“是”。
            bool hostoutarray = true  // 判断 outarray 是否是 Host 端的指针，
                                      // 默认为“是”。
    );
    
    // Host 成员方法：scanArrayExclusive（Exclusive 类型数组扫描）
    // 对一个数组进行扫描，对所有元素进行某操作的遍历，操作不包含自身。
    // 相当于在 Scan Inclusive 的版本上将输出后移一位，然后第一位赋值为 0。
    // 需要输出数组在申请时长度比输入数组加 1 个单位。用来存放第一个位置的 0。
    template < class T, class Operation >
    inline __host__ int               // 返回值：函数是否正确执行，若函数正确
                                      // 执行，返回NO_ERROR。
    scanArrayExclusive(
            T *inarray,               // 输入数组
            T *outarray,              // 输出数组
            int numelements,          // 数组的长度，处理元素的个数。
            Operation op,             // 运算类型
            bool backward = false,    // 判断是否为后序扫描
            bool hostinarray = true,  // 判断 inarray 是否是 Host 端的指针，
                                      // 默认为“是”。
            bool hostoutarray = true  // 判断 outarray 是否是 Host 端的指针，
                                      // 默认为“是”。
    ) {
        // 局部变量，错误码。
        cudaError_t cuerrcode;
        int errcode;

        if (!backward) {
            // 调用 Scan Inclusive 的版本。输出数组传指针为第一个位置处开始。
            errcode = ScanArray::scanArray(inarray, (outarray + 1),
                                           numelements, op, backward,
                                           hostinarray, hostoutarray);

            // 判断当前 outarray 数组是否存储在 Host 端。若是，直接在 Host 端操作
            if (hostoutarray) {
                // 首位赋值为 0。
                outarray[0] = op.identity();
            // 如果该数组是在 Device 端，则直接在 Device 端使用。
            } else {
                // 首位赋值为 0。
                cuerrcode = cudaMemset(outarray, op.identity(), sizeof(T));
                if (cuerrcode != cudaSuccess)
                    return CUDA_ERROR;
            }
        } else {
            // 调用 Scan Inclusive 的版本。输入数组传指针为第一个位置处开始。
            errcode = ScanArray::scanArray(inarray + 1, outarray,
                                           numelements, op, backward,
                                           hostinarray, hostoutarray);

            // 判断当前 outarray 数组是否存储在 Host 端。若是，直接在 Host 端操作
            if (hostoutarray) {
                // 末位赋值为 0。
                outarray[numelements - 1] = op.identity();
            // 如果该数组是在 Device 端，则直接在 Device 端使用。
            } else {
                // 末位赋值为 0。
                cuerrcode = cudaMemset(outarray + numelements - 1,
                                       op.identity(), sizeof(T));
                if (cuerrcode != cudaSuccess)
                    return CUDA_ERROR;
            }
        }

        // 出错则返回错误码。
            if (errcode != NO_ERROR)
                return errcode;

        // 处理完毕退出。
        return NO_ERROR;
    }

    // Host 成员方法：scanCorrectCheck（GPU 端的 scan 扫描数组结果检测）
    // 对两个数组进行扫描，对所有元素依次进行比较，得出检测结果。
    template < class T, class Operation >
    __host__ int                       // 返回值：函数是否正确执行，若函数正确
                                       // 执行，返回 -1。若检测结果出错，返
                                       // 回错误位置。
    scanCorrectCheck(
            T *inarray,                // 输入数组
            T *array_g,                // gpu 端计算数组
            T *array_c,                // cpu 端计算数组
            const unsigned int len,    // 数组的长度，处理元素的个数。
            Operation op,              // 运算类型
            bool backward,             // 判断是否为后序扫描
            bool hostinarray = true,   // 判断 inarray 是否是 Host 端的指针，
                                       // 默认为“是”。
            bool hostoutarray = true,  // 判断 outarray 是否是 Host 端的指针，
                                       // 默认为“是”。
            bool isinclusive =true     // 判断是否为 inclusive 版本
    ) {
        // 设置实现类型为 CPU_IN_SCAN
        this->setScanType(CPU_IN_SCAN);

        int errcode;
        if (isinclusive)
            // 调用 cpu 端计算 scan 数组的函数
            errcode = ScanArray::scanArray(inarray, array_c, len, op, backward,
                                           hostinarray, hostoutarray);
        else
            // 调用 cpu 端计算 exclusive 版本的 scan 数组的函数
            errcode = ScanArray::scanArrayExclusive(inarray, array_c, len, op,
                                                    backward, hostinarray, 
                                                    hostoutarray);

        // 出错则返回错误码。
        if (errcode != NO_ERROR)
            return errcode;

        // 计数器变量
        unsigned int i;
        // 两端的数组进行一一比对，得出检测结果。
        for (i = 0; i < len; ++i) {
            if (array_g[i] != array_c[i])
               return i + 1; 
        }
        // 执行结束
        return -1;
    }
};

#endif

