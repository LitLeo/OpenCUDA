// Compact.h 
// 创建人：王春峰
//
// 数组压缩（Compact）
// 功能说明：对一个数组进行压缩，删除不合法元素，按顺序输出合法元素。
// 举例：数组 A, x, B, C, x, D 进行压缩的结果是：A, B, C, D。
//
// 修订历史：
// 2013年05月20日（王春峰）
//     初始版本
// 2013年05月21日（王春峰）
//     添加了数组压缩的 GPU 版本。

#ifndef __COMPACT_H__
#define __COMPACT_H__

#include "ErrorCode.h"
#include "OperationFunctor.h"
#include "ScanArray.h"
#include <iostream>
using namespace std;

// 类：Compact
// 继承自：无
// 对一个数组进行压缩，对所有元素进行合法性判断。分别实现了 CPU 端和 GPU 端的压
// 缩函数。
class Compact {

protected:

    // 成员变量：ScanArray 类对象 sa（用以调用 exclusive 版本函数）。
    ScanArray sa;

public:

    // 构造函数：Compact
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Compact() 
    {
        // 设置 sa 默认扫描类型
        sa.setScanType(NAIVE_SCAN);
    }

    // 构造函数：Compact
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Compact(
            int scanType  // 扫描类型
    ) {
        // 设置 sa 默认扫描类型
        sa.setScanType(NAIVE_SCAN);
        // 根据参数列表中的值设定成员变量的初值
        this->setSaType(scanType);
    }

    // 成员方法：getSa（读取 ScanArray 对象）
    // 读取 sa 成员变量的值。
    __host__ __device__ ScanArray  // 返回值：当前 sa 成员变量的值。
    getSa() const
    {
        // 返回 ScanArray 成员变量的值。
        return this->sa;
    }

    // 成员方法：setSaType（设置扫描类型）
    // 设置 sa 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setSaType(
            int scanType     // 扫描类型
    ) {
        // 检查输入参数是否合法
        if (scanType != NAIVE_SCAN && scanType != EFFICIENT_SCAN && 
            scanType != OPTIMIZE_SCAN && scanType != BETTER_SCAN && 
            scanType != CPU_IN_SCAN)
            return INVALID_DATA;       

        // 将 sa 成员变量赋成新值
        sa.setScanType(scanType);
        return NO_ERROR;
    }

    // Host 成员方法：compactDataCPU
    // CPU 端对数组进行串行压缩。
    __host__ int 
    compactDataCPU(
            int *indices,            // scan-ex 结果数组
            int *d_isValid,          // 合法性判断数组（只有 1 和 0）
            int *d_in,               // 输入数组
            int numElements,         // 数组元素个数
            int *d_out,              // 输出数组
            int *d_numValidElements  // 合法点的个数
    ) {
        // 定义运算类型为加法
        add_class<int> add;
        // 调用 scan exclusive 函数
        sa.scanArrayExclusive(d_isValid, indices, numElements, add);
        
        // 根据 scan exclusive 结果进行输出数组的赋值。
        for(int i = 0; i < numElements; i++) {
            // 如果不合法，则跳过判断下一元素
            if (d_isValid[i] == 0)
                continue;
            // 将输入数组中对应的合法元素输出到输出数组的对应位置上。
            d_out[indices[i]] = d_in[i];
            // 合法点数加一
            d_numValidElements[0]++;
        }

        // 运行结束
        return NO_ERROR;
    }

    // Host 成员方法：compactDataGPU
    // GPU 端对数组进行并行压缩。
    __host__ int 
    compactDataGPU(
            int *indices,            // scan-ex 结果数组
            int *d_isValid,          // 合法性判断数组（只有 1 和 0）
            int *d_in,               // 输入数组
            int numElements,         // 数组元素个数
            int *d_out,              // 输出数组
            int *d_numValidElements  // 合法点的个数
    );

    // Host 成员方法：compactCorrectCheck（GPU 端的数组压缩结果检测）
    // 对两个数组进行压缩，对所有元素依次进行比较，得出检测结果。
    inline __host__ int                // 返回值：函数是否正确执行，若函数正确
                                       // 执行，返回 -1。若检测结果出错，返
                                       // 回错误位置。若合法点计数结果不一致返回
                                       // -2。
    compactCorrectCheck(
            int *indices,              // scan-ex 结果数组
            int *d_isValid,            // 合法性判断数组（只有 1 和 0）
            int *d_in,                 // 输入数组
            int numElements,           // 数组元素个数
            int *d_outC,               // CPU 输出数组
            int *d_outG,               // GPU 输出数组
            int *d_numValidElementsC,  // CPU 端合法点的个数
            int *d_numValidElementsG   // GPU 端合法点的个数
    ) {
        // 错误码
        int errcode;

        // 调用 CPU 端压缩数组的函数
        errcode = Compact::compactDataCPU(indices, d_isValid, d_in, numElements,
                                          d_outC, d_numValidElementsC);

        // 出错则返回错误码。
        if (errcode != NO_ERROR)
            return errcode;

        // 调用 GPU 端压缩数组的函数
        errcode = Compact::compactDataGPU(indices, d_isValid, d_in, numElements,
                                          d_outG, d_numValidElementsG);

        // 出错则返回错误码。
        if (errcode != NO_ERROR)
            return errcode;

        // 首先比较合法点个数是否一致
        if (d_numValidElementsC[0] != d_numValidElementsG[0])
            return -2;

        // 计数器变量
        unsigned int i;

        // 两端的数组进行一一比对，得出检测结果。
        for (i = 0; i < d_numValidElementsC[0]; ++i) {
            if (d_outC[i] != d_outG[i])
               return i + 1; 
        }

        // 执行结束
        return -1;
    }
};

#endif
    
