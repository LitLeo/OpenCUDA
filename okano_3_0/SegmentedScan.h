// SegmentedScan.h 
// 创建人：刘瑶
//
// 数组分段扫描（SegmentedScan）
// 功能说明：对一个数组进行分段扫描，对所有元素进行某操作的遍历。
// 举例：数组 1, 7, 5, 13, 10 的标签值为 0, 0, 0, 3, 3, 进行 inclusive
// segmented scan 的结果是：1, 7, 7, 13, 13。
// 参照 Y Dotsenko 的论文 "fast scan algorithms on graphics processors" 实现。
// 支持任意长度的数组分段扫描。
//
// 修订历史：
// 2012年12月29日（刘瑶）
//     初始版本
// 2012年12月30日（刘瑶）
//     修改了核函数的部分错误。完善了代码注释。
// 2012年12月31日（刘瑶）
//     添加了将中间结果进行返回分段扫描的函数。修改了部分代码。
// 2013年12月23日（刘瑶）
//     添加了 CPU 串行版本分段扫描接口。

#ifndef __SEGMENTEDSCAN_H__
#define __SEGMENTEDSCAN_H__

#include "ErrorCode.h"
#include <stdio.h>

// 宏：MATRIX_SEGMENTED_SCAN
// 用于设置 SegmentedScan 类中 segmentedScanType 成员变量，告知类的实例选用矩阵
// 方法的 SegmentedScan 实现。
// 参考文献：Y Dotsenko, NK Govindaraju, PP Sloan, and etc. Fast scan
// algorithms on graphics processors. ICS, 2008, pp. 205-213. 
// 论文被引用次数：60
#define MATRIX_SEGMENTED_SCAN                  1

// 用于设置 SegmentedScan 类中 segmentedScanType
// 成员变量，告知类的实例选用运算转换法的分段扫描，未实现
// 参考文献：S Sengupta, M Harris, and M Garland. Efficient parallel scan 
// algorithms for GPUs. NVR, 2008.
// 论文被引用次数：55
#define OPERATOR_TRANSFORMATION_SEGMENTED_SCAN 2

// 用于设置 SegmentedScan 类中 segmentedScanType
// 成员变量，告知类的实例选用直接法的分段扫描，未实现
// 参考文献：S Sengupta, M Harris, and M Garland. Efficient parallel scan 
// algorithms for GPUs. NVR, 2008.
// 论文被引用次数：55
#define DIRECT_SEGMENTED_SCAN                  3

// 类：SegmentedScan
// 继承自：无
// 对一个数组进行分段扫描，对所有元素进行某操作的遍历。
// 输入数组包括 3 个，当前点的垂距，当前点的所属区域的标签，当前区域内最大垂距
// 点的位置索引。输出数组包括 2 个，当前所属区域内的最大垂距，当前区域内的最大
// 垂距的位置索引。目前只实现了矩阵方法的分段扫描。
class SegmentedScan {

protected:

    // 成员变量：segmentedScanType（实现类型）
    // 设定分段扫描方法实现类型中的一种，在调用 segmentedscan 扫描函数的时候，
    // 使用对应的实现方式。
    int segmentedScanType;

    // Host 成员方法：segmentedScanBack（将中间结果进行返回分段扫描）
    // 对一个数组进行扫描，结合中间结果的小数组，对输出数组进行返回式的分段
    // 扫描。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确
                                   // 执行，返回NO_ERROR。
    segmentedScanBack(
            float *maxdist,        // 输出的分段扫描后的最大垂直距离数组，
                                   // 表示每段中的垂距最大的值。
            int *maxdistidx,       // 输出的扫描后最大垂距点的位置索引。
            int *label,            // 输入数组的分段标签值。
            float *blockmaxdist,   // 中间结果，每块最后位置的最大垂距。
            int *blocklabel,       // 中间结果，每块最后位置处的标签。
            int *blockmaxdistidx,  // 中间结果，每块最后位置处的垂距索引。
            int numelements        // 扫描数组的长度。
    );

public:

    // 构造函数：SegmentedScan
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    SegmentedScan()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->segmentedScanType = MATRIX_SEGMENTED_SCAN;  // 实现类型的默认值为 
                                                          // 矩阵法的分段扫描
    }

    // 构造函数：SegmentedScan
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    SegmentedScan(
            int segmentedScanType                         // 实现类型
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->segmentedScanType = MATRIX_SEGMENTED_SCAN;  // 实现类型的默认值为
                                                          // 矩阵分段扫描法。
        // 根据参数列表中的值设定成员变量的初值
        this->setSegmentedScanType(segmentedScanType);
    }

    // 成员方法：getScanType（读取实现类型）
    // 读取 segmentedScanType 成员变量的值。
    __host__ __device__ int  // 返回值：当前 segmentedScanType 成员变量的值。
    getSegmentedScanType() const
    {
        // 返回 ScanType 成员变量的值。
        return this->segmentedScanType;
    }
    
    // 成员方法：setScanType（设置实现类型）
    // 设置 segmentedScanType 成员变量的值。
    __host__ __device__ int        // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    setSegmentedScanType(
            int segmentedScanType  // 实现类型
    ) {
        // 检查输入参数是否合法
        if (segmentedScanType != MATRIX_SEGMENTED_SCAN)
            return INVALID_DATA;

        // 将 segmentedScanType 成员变量赋成新值
        this->segmentedScanType = segmentedScanType;
        return NO_ERROR;
    }

    // Host 成员方法：SegmentedScan（数组分段扫描）
    // 对一个数组进行分段扫描，对所有元素进行某操作的遍历。
    __host__ int                 // 返回值：函数是否正确执行，若函数正确
                                 // 执行，返回NO_ERROR。
    segmentedScan(
            float *inarray,      // 输入数组
            int *label,          // 输入数组的分段标签值。
            int *index,          // 输入数组的位置索引。
            float *maxdist,      // 输出的分段扫描后的最大垂直距离数组，
                                 // 表示每段中的垂距最大的值。
            int *maxdistidx,     // 输出的分段扫描后的最大垂距的点的位置
                                 // 索引。
            int numelements,     // 数组的长度，处理元素的个数。
            bool hostinarray,    // 判断 inarray 是否是 Host 端的指针，
                                 // 默认为“是”。
            bool hostlabel,      // 判断 label 是否是 Host 端的指针，
                                 // 默认为“是”。
            bool hostindex,      // 判断 index 是否是 Host 端的指针，
                                 // 默认为“是”。
            bool hostmaxdist,    // 判断 hostmaxdist 是否是 Host 端的指
                                 // 针，默认为“是”。
            bool hostmaxdistidx  // 判断 hostmaxdistidx 是否是 Host 端的
                                 // 指针，默认为“是”。
    );

    // Host 成员方法：SegmentedScans（数组分段扫描）
    // 对一个数组进行分段扫描，对所有元素进行某操作的遍历。
    inline __host__ int          // 返回值：函数是否正确执行，若函数正确
                                 // 执行，返回NO_ERROR。
    segmentedScan(
            float *inarray,      // 输入数组
            int *label,          // 输入数组的分段标签值。
            int *index,          // 输入数组的位置索引。
            float *maxdist,      // 输出的分段扫描后的最大垂直距离数组，
                                 // 表示每段中的垂距最大的值。
            int *maxdistidx,     // 输出的扫描后最大垂距点的位置索引。
            int numelements,     // 数组的长度，处理元素的个数。
            bool hostall = true  // 判断所有参数是否是 Host 端指针。
    ) {
        // 调用分段扫描的成员方法。输入输出参数全部在 host 端的判断变量一致。
        return segmentedScan(inarray, label, index,
                             maxdist, maxdistidx, numelements,
                             hostall, hostall, hostall, hostall, hostall);
    }

    // Host 成员方法：SegmentedScan（数组分段扫描）
    // 对一个数组进行分段扫描，对所有元素进行某操作的遍历。
    __host__ int                 // 返回值：函数是否正确执行，若函数正确
                                 // 执行，返回NO_ERROR。
    segmentedScan(
            float *inarray,      // 输入数组
            int *label,          // 输入数组的分段标签值。
            float *maxdist,      // 输出的分段扫描后的最大垂直距离数组，
                                 // 表示每段中的垂距最大的值。
            int *maxdistidx,     // 输出的分段扫描后的最大垂距的点的位置
                                 // 索引。
            int numelements,     // 数组的长度，处理元素的个数。
            bool hostinarray,    // 判断 inarray 是否是 Host 端的指针，
                                 // 默认为“是”。
            bool hostlabel,      // 判断 label 是否是 Host 端的指针，
                                 // 默认为“是”。
            bool hostmaxdist,    // 判断 hostmaxdist 是否是 Host 端的指
                                 // 针，默认为“是”。
            bool hostmaxdistidx  // 判断 hostmaxdistidx 是否是 Host 端的
                                 // 指针，默认为“是”。
    ) {
        // 调用分段扫描的成员方法。输入输出参数全部在 host 端的判断变量一致。
        return segmentedScan(inarray, label, NULL,
                             maxdist, maxdistidx, numelements,
                             hostinarray, hostlabel, false, hostmaxdist, 
                             hostmaxdistidx);
    }

    // Host 成员方法：SegmentedScan（数组分段扫描）
    // 对一个数组进行分段扫描，对所有元素进行某操作的遍历。
    inline __host__ int          // 返回值：函数是否正确执行，若函数正确
                                 // 执行，返回NO_ERROR。
    segmentedScan(
            float *inarray,      // 输入数组
            int *label,          // 输入数组的分段标签值。
            float *maxdist,      // 输出的分段扫描后的最大垂直距离数组，
                                 // 表示每段中的垂距最大的值。
            int *maxdistidx,     // 输出的扫描后最大垂距点的位置索引。
            int numelements,     // 数组的长度，处理元素的个数。
            bool hostall = true  // 判断所有参数是否是 Host 端指针。
    ) {
        // 调用分段扫描的成员方法。输入输出参数全部在 host 端的判断变量一致。
        return segmentedScan(inarray, label, NULL,
                             maxdist, maxdistidx, numelements,
                             hostall, hostall, hostall, hostall, hostall);
    }

    // Host 成员方法：SegmentedScan（数组分段扫描）
    // 对一个数组进行分段扫描，对所有元素进行某操作的遍历。
    __host__ int                 // 返回值：函数是否正确执行，若函数正确
                                 // 执行，返回NO_ERROR。
    segmentedScanCpu(
            float *inarray,      // 输入数组
            int *label,          // 输入数组的分段标签值。
            int *index,          // 输入数组的位置索引。
            float *maxdist,      // 输出的分段扫描后的最大垂直距离数组，
                                 // 表示每段中的垂距最大的值。
            int *maxdistidx,     // 输出的分段扫描后的最大垂距的点的位置
                                 // 索引。
            int numelements,     // 数组的长度，处理元素的个数。
            bool hostinarray,    // 判断 inarray 是否是 Host 端的指针，
                                 // 默认为“是”。
            bool hostlabel,      // 判断 label 是否是 Host 端的指针，
                                 // 默认为“是”。
            bool hostindex,      // 判断 index 是否是 Host 端的指针，
                                 // 默认为“是”。
            bool hostmaxdist,    // 判断 hostmaxdist 是否是 Host 端的指
                                 // 针，默认为“是”。
            bool hostmaxdistidx  // 判断 hostmaxdistidx 是否是 Host 端的
                                 // 指针，默认为“是”。
    );

    // Host 成员方法：segmentedScanCpu（数组分段扫描）
    // 对一个数组进行分段扫描，对所有元素进行某操作的遍历。
    inline __host__ int          // 返回值：函数是否正确执行，若函数正确
                                 // 执行，返回NO_ERROR。
    segmentedScanCpu(
            float *inarray,      // 输入数组
            int *label,          // 输入数组的分段标签值。
            int *index,          // 输入数组的位置索引。
            float *maxdist,      // 输出的分段扫描后的最大垂直距离数组，
                                 // 表示每段中的垂距最大的值。
            int *maxdistidx,     // 输出的分段扫描后的最大垂距的点的位置
                                 // 索引。
            int numelements,     // 数组的长度，处理元素的个数。
            bool hostall = true  // 判断所有参数是否是 Host 端指针。
    ) {
        // 调用分段扫描的成员方法。输入输出参数全部在 host 端的判断变量一致。
        return segmentedScanCpu(inarray, label, index,
                                maxdist, maxdistidx, numelements,
                                hostall, hostall, hostall, hostall, hostall);
    }

    // Host 成员方法：segmentedScanCpu（数组分段扫描）
    // 对一个数组进行分段扫描，对所有元素进行某操作的遍历。
    inline __host__ int          // 返回值：函数是否正确执行，若函数正确
                                 // 执行，返回NO_ERROR。
    segmentedScanCpu(
            float *inarray,      // 输入数组
            int *label,          // 输入数组的分段标签值。
            float *maxdist,      // 输出的分段扫描后的最大垂直距离数组，
                                 // 表示每段中的垂距最大的值。
            int *maxdistidx,     // 输出的扫描后最大垂距点的位置索引。
            int numelements,     // 数组的长度，处理元素的个数。
            bool hostall = true  // 判断所有参数是否是 Host 端指针。
    ) {
        // 调用分段扫描的成员方法。输入输出参数全部在 host 端的判断变量一致。
        return segmentedScanCpu(inarray, label, NULL,
                                maxdist, maxdistidx, numelements,
                                hostall, hostall, hostall, hostall, hostall);
    }
};

#endif

