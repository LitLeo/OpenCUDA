// DouglasPeucker.h

// 创建者：刘婷

// 道格拉斯普克算法（DouglasPeucker）
// 功能说明：利用 GPU CUDA 并行实现 DouglasPeucker 算法，利用迭代求角点。首先曲
// 线的起始点和最末一个点自动视为结果点集中的点，利用这两个点生成直线，计算其他
// 点到该直线的距离，如果最大的距离大于了阈值（参数）则将其加入到结果点集中，并
// 将原曲线分解为两条曲线，再以此迭代下去，直到找不到符合条件的点。

// 修订历史：

// 2013年11月02日（刘婷）
//     完成头文件和核函数的设计。
// 2013年11月04日（刘婷）
//     实现核函数 _initLabelAryKer
// 2013年11月05日（刘婷）
//     实现核函数 _updateDistKer
// 2013年11月06日（刘婷）
//     实现核函数 _updateFoundInfoKer
// 2013年11月07日（刘婷）
//     实现核函数 _updateCornerCstKer
// 2013年11月09日（刘婷）
//     实现核函数 _markLeftPointsKer
// 2013年11月10日（刘婷）
//     实现核函数 _updatePropertyKer
// 2013年11月12日（刘婷）
//     实现核函数 _arrangeCstKer
// 2013年11月15日（刘婷）
//     调试检查代码
// 2013年11月15日（刘婷）
//     测试代码，代码规范
// 2013年11月23日（刘婷）
//     更改结果点集初始化方式，直接将存放起点和终点的数组拷贝至 GPU 端的结果点集
//     中。
// 2013年12月05日（刘婷）
//     整理代码。
// 2013年12月06日（刘婷）
//     将主函数的参数改为 Curve 类型。

#ifndef __DOUGLASPEUCKER_H__
#define __DOUGLASPEUCKER_H__

#include "Curve.h"

#include "ErrorCode.h"

#include "ScanArray.h"

#include "SegmentedScan.h"

#include "OperationFunctor.h"



// 类：DouglasPeucker

// 继承自：无

// 功能说明：利用 GPU CUDA 并行实现 DouglasPeucker 算法，利用迭代求角点。首先曲
// 线的起始点和最末一个点自动视为结果点集中的点，利用这两个点生成直线，计算其他
// 点到该直线的距离，如果最大的距离大于了阈值（参数）则将其加入到结果点集中，并
// 将原曲线分解为两条曲线，再以此迭代下去，直到找不到符合条件的点。

class DouglasPeucker {

protected:


    // 成员变量：segScan（分段扫描器）
    // 完成分段扫描，主要在迭代过程中用于计算每个 LABEL 区域的最大垂距点。
    SegmentedScan segScan;

    // 成员变量：aryScan（扫描累加器）

    // 完成非分段扫描，主要在迭代过程中用于计算各个标记值所对应的累加值。
    ScanArray aryScan;

    // 成员变量：threshold（与垂距相关的阈值）
    // 用于与 label 内最大垂距做比较的阈值
    float threshold;

    // Host 成员方法：initLabelAry（初始化 LABEL 数组）
    // 在迭代之前初始化 LABEL 数组，初始化后该数组最后一个元素为 1，其余元素皆为
    // 0， 用此来划分区域，及将曲线分解为多条曲线。
    __host__ int          // 返回值：函数是否正确执行，如果函数能够正确执行，返
                          // 回 NO_ERROR。

    initLabelAry(

            int label[],  // 待初始化的 LABEL 数组。

            int cstcnt    // 数组长度。

    );

    // Host 成员方法：updateDist（计算各点垂距）

    // 根据目前已知结果点集和区域的标签值，根据点到直线的垂距公式，计算点

    // 集的附带数据：点到当前所在区域的最左最右点构成的直线的垂直距离。

    __host__ int                   // 返回值：函数是否正确执行，如果函数能够正

                                   // 确执行，返回 NO_ERROR。

    updateDist(

            int *cst,        // 输入点集坐标

            int *cornercst,  // 目前已知结果点集，即每段的最值点信息

            int label[],     // 输入，当前点集的区域标签值数组。

            int cstcnt,      // 输入，当前点的数量。

            float dis[]      // 记录每一个点的垂距


    );

    // Host 成员方法: updateFoundInfo（更新新发现的结果点集）

    // 根据分段扫描后得到的点集信息，更新各个区域是否有新发现的点，更新这些点的

    // 位置索引。

    __host__ int               // 返回值：函数是否正确执行，如果函数能够正确执

                               // 行，返回 NO_ERROR。

    updateFoundInfo(

            int label[],       // 输入，当前点集的区域标签值数组。

            float dist[],      // 输入数组，分段扫描后，当前位置记录的本段目前

                               // 已知的最大垂距。

            int maxdistidx[],  // 输入，分段扫描后，当前位置记录的本段目前已知

                               // 的最大垂距点的位置索引数组。

            int cstcnt,        // 当前点的数量。

            int foundflag[],   // 输出数组，如当前区域内找到新的凸壳上的点，标

                               // 志位置 1。

            int startidx[],    // 输出，目前已知的凸壳上的点的位置索引数组，也

                               // 相当于当前每段上的起始位置的索引数组。
            float thread,
            int foundidx[]     // 记录找到的点的一维索引

    );

    // Host 成员方法：updateCornerCst（生成新结果点集，以下用角点简称）

    // 根据分段扫描后得到的点集信息，和每段上是否发现新凸壳点的信息，构造新的结

    // 果点集。

    __host__ int                     // 返回值：函数是否正确执行，如果函数能够

                                     // 正确执行，返回 NO_ERROR。

    updateCornerCst(

            int *cst,          // 输入点集

            int *cornercst,    // 输入，现有的结果点集。

            int foundflag[],   // 输入，当前区域内有新发现点的标志数组，

                               // 如果当前区域内找到新的点，标志位置 1。

            int foundacc[],    // 输入，偏移量数组，当前区域内有新发现点

                               // 的标志位的累加值。用来计算新添加的点的存

                               // 放位置的偏移量。

            int startidx[],    // 输入，目前已知的结果点集的点索引数组，

                               // 也相当于当前每段上的起始位置的索引数组

            int maxdistidx[],  // 输入，分段扫描后，当前位置记录的本段目

                               // 前已知的最大垂距点的位置索引数组。

            int num,           // 当前找到的角点的数量。

            int *newcornercst  // 输出，更新后目前已知角点的点集坐标

    );



    // Host 成员方法：updateLabel（更新标签值）

    // 根据目前每段上是否有新发现角点的标志，更新点的标签值。

    __host__ int                      // 返回值：函数是否正确执行，如果函数能够

                                      // 正确执行，返回 NO_ERROR。

    updateLabel(

            int label[],     // 输入，当前点集的区域标签值数组。


            int cstcnt,      // 输入，当前输入点集的数量。
            int foundidx[],  // 输入，已经找到的角点在原输入点集中的索引
            int foundacc[],  // 输入，用来找到当前点之前有多少个已经找到的角点。
            int tmplabel[]   // 输出，新的标签值。

    );



    // Host 成员方法：douglasIter（迭代法求曲线点）

    // 采用迭代的方法找到曲线的角点

    __host__ int              // 返回值：函数是否正确执行，如果函数能够

                              // 正确执行，返回 NO_ERROR。

    douglasIter(

            int *inputcst,    // 输入点集。
            int *corner ,     // 输出角点点集
            float threshold,  // 用来判断是否为新发现点的阈值
            int count,        // 输入点集的点的个数
            int *cornerpnt    // 得到的结果点的个数
    );


public:
    // 构造函数：DouglasPeucker

    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。

    __host__ __device__

    DouglasPeucker()

    {

        // 配置扫描器。

        this->aryScan.setScanType(NAIVE_SCAN);



        // 使用默认值为类的各个成员变量赋值。

        this->threshold = 0.3f;  // 设置垂距阈值

    }

    // 成员方法：getThreshold（获取垂距阈值）

    // 获取与最大垂距相关的阈值。

    __host__ __device__ float  // 返回值：与最大垂距相关的阈值。

    getThreshold() const

    {

        // 返回与最大垂距相关的阈值。

        return this->threshold;    

    } 



    // 成员方法：setThreshold（设置垂距阈值）

    // 设置垂距阈值。

    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，

                             // 返回 NO_ERROR。

    setThreshold(

            float threshold  // 设定新的垂距阈值。

    ) {

        // 根据阈值设置转换标志位。

        if (threshold < 0)

            return INVALID_DATA;

        

        // 将图像转换点集的像素阈值赋成新值。

        this->threshold = threshold;
        return NO_ERROR;

    }


    // Host 成员方法：douglasPeucker（利用 Douglas Peucker 算法求解角点）

    // 利用 Douglas Peucker 算法求解角点

    __host__ int           // 返回值：函数是否正确执行，如果函数能够正确

                           // 执行，返回 NO_ERROR。

    douglasPeucker(

            Curve *incur,  // 输入曲线

            Curve *outcur  // 结果曲线

    );

};

#endif
