// ConvexHull.h
// 创建者：刘瑶
//
// 计算点集的凸壳（ConvexHull）
// 功能说明：对于已知点集，计算点集的凸壳，并最终输出凸壳上的点集。
// 核心算法采用 QuickHull 算法，通过迭代求凸壳。
// 参考文献：[1] S Srungarapu, DP Reddy, K Kothapalli, and etc. Fast Two
// Dimensional Convex Hull on the GPU. WAINA, 2011, pp. 7-12.
// [2] S Srungarapu, DP Reddy, K Kothapalli and etc. Parallelizing two
// dimensional convex hull on NVIDIA GPU and Cell BE. HIPC, 2009.
// [3] CB Barber, DP Dobkin, and H Huhdanpaa. The quickhull algorithm
// for convex hulls. ACM TOMS, pp. 469-483.
//
// 修订历史：
// 2013年01月02日（刘瑶）
//     初始版本。搭建框架。
// 2013年01月03日（刘瑶）
//     修改部分接口和注释。
// 2013年01月06日（刘瑶）
//     增加了更新垂距的核函数和成员方法。
// 2013年01月07日（刘瑶）
//     增加了更新区域发现新凸壳点的函数和更新新凸壳点集的函数。
// 2013年01月08日（刘瑶，刘婷）
//     增加了标记新发现凸壳点左侧标志的函数。
// 2013年01月09日（刘瑶，王雪菲）
//     增加了图像转成点集的函数。
// 2013年01月11日（刘瑶，王雪菲）
//     增加了图像初始化函数和点集转成图像的函数。
// 2013年01月14日（刘瑶，刘婷）
//     增加了更新点集属性的函数和重新排列点集的函数。完成半凸壳的首次迭代过程。
// 2013年03月06日（刘瑶）
//     修改了迭代过程的部分变量，修正了内存越界访问的错误。
// 2013年03月09日（于玉龙）
//     翻新了更新垂距的 Kernel，修正了其中多处注释错误，修正了一处严重的 Bug。
// 2013年03月10日（于玉龙）
//     翻新了两个 Kernel。修正了其中多处注释错误，修正了多处严重的 Bug。
// 2013年03月11日（于玉龙）
//     修正了代码中所有的逻辑错误，可以完成下半凸壳的求解工作，目前正在修改代码
//     格式。
// 2013年03月12日（于玉龙）
//     修正了代码中的格式问题，以及代码中的一处潜在的 Bug。
// 2013年03月13日（于玉龙）
//     修正了代码中的格式问题。
//     翻新了主函数，修正了其中潜在的 Bug。
// 2013年03月14日（于玉龙）
//     增加了寻找最左最右点的函数。
//     增加了初始化 LABEL 数组的函数，进一步整理了凸壳迭代函数。
//     增加了翻转坐标点集的函数，为求解上凸壳点做准备。
//     完成了上半凸壳的求解。
// 2013年03月15日（于玉龙）
//     增加了合并凸壳点的函数，为求解整个凸壳点做准备。
//     实现了完整的凸壳求解。
// 2013年03月16日（于玉龙）
//     完善了算法中的一个步骤。
// 2013年03月22日（刘瑶）
//     根据图像转换点集的函数接口，修改了部分代码，修改了构造函数，添加了 get
//     与 set 函数。添加了输入图像求凸壳的函数接口实现。
// 2013年03月24日（刘瑶）
//     根据修改的图像转换点集的函数接口，修改对应的调用代码。
// 2013年05月14日（刘瑶）
//     根据 ScanArray 的修改，修改了成员方法中对于 ScanArray 的调用。
// 2013年09月21日（于玉龙）
//     修正了代码中调用 SCAN 算法的 BUG。
//     修正了计算 VALUE 区间的 BUG。
// 2013年12月23日（刘瑶）
//     添加了串行版本的凸壳算法接口及对应的各个函数串行接口。

#ifndef __CONVEXHULL_H__
#define __CONVEXHULL_H__

#include "Image.h"
#include "ErrorCode.h"
#include "CoordiSet.h"
#include "ScanArray.h"
#include "SegmentedScan.h"
#include "ImgConvert.h"
#include "OperationFunctor.h"

// 类：ConvexHull
// 继承自：无
// 功能说明：对于输入的已知点集，计算点集的凸壳，并最终输出凸壳上的点集。
// 核心算法采用 QuickHull 算法，通过迭代求凸壳。即在目前已知凸壳的每个边外侧，
// 找垂距最大的点，记录下来，并更新凸壳点集。
class ConvexHull {

protected:

    // 成员变量：segScan（分段扫描器）
    // 完成分段扫描，主要在迭代计算凸壳点的过程中用于计算每个 LABEL 区域的最大
    // 垂距点。
    SegmentedScan segScan;

    // 成员变量：aryScan（扫描累加器）
    // 完成非分段扫描，主要在迭代计算凸壳点的过程中用于计算各个标记值所对应的累
    // 加值。
    ScanArray aryScan;

    // 成员变量：imgCvt（图像与坐标集转换）
    // 根据给定的图像和阈值，转换成坐标集形式。
    ImgConvert imgCvt;

    // 成员变量：value（图像转换点集的像素阈值）
    // 用于图像转换点集的像素阈值。
    unsigned char value;

    // Host 成员方法：initLabelAry（初始化 LABEL 数组）
    // 在进行凸壳点迭代之前初始化 LABEL 数组，初始化后该数组最后一个元素为 1，
    // 其余元素皆为 0。
    __host__ int          // 返回值：函数是否正确执行，如果函数能够正确执行，返
                          // 回 NO_ERROR。
    initLabelAry(
            int label[],  // 待初始化的 LABEL 数组。
            int cstcnt    // 数组长度。
    );

    // Host 成员方法：initLabelAryCpu（初始化 LABEL 数组）
    // 在进行凸壳点迭代之前初始化 LABEL 数组，初始化后该数组最后一个元素为 1，
    // 其余元素皆为 0。
    __host__ int          // 返回值：函数是否正确执行，如果函数能够正确执行，返
                          // 回 NO_ERROR。
    initLabelAryCpu(
            int label[],  // 待初始化的 LABEL 数组。
            int cstcnt    // 数组长度。
    );

    // Host 成员方法：swapEdgePoint（寻找最左最右点）
    // 寻找 cst 中的最左和最右点，并将最左点交换到 cst 的首部，最右点交换到 cst
    // 的尾部，同时将最左最右点放入 convexcst 的头两个坐标点中，但还函数不会改
    // 变两个坐标点集的 count 域。
    __host__ int                  // 返回值：函数是否正确执行，如果函数能够正确
                                  // 执行，返回 NO_ERROR。
    swapEdgePoint(
            CoordiSet *cst,       // 输入点集，从函数退出后其首个坐标点为最左
                                  // 点，末个坐标点为最右点。原来存放于首末位置
                                  // 的点分别被移位到原来放置最左最右点的位置。
            CoordiSet *convexcst  // 存放输出最左最右点的集合，这个点集在求凸壳
                                  // 的过程中被用来作为初始凸壳点集。
    );

    // Host 成员方法：swapEdgePointCpu（寻找最左最右点）
    // 寻找 cst 中的最左和最右点，并将最左点交换到 cst 的首部，最右点交换到 cst
    // 的尾部，同时将最左最右点放入 convexcst 的头两个坐标点中，但还函数不会改
    // 变两个坐标点集的 count 域。
    __host__ int                  // 返回值：函数是否正确执行，如果函数能够正确
                                  // 执行，返回 NO_ERROR。
    swapEdgePointCpu(
            CoordiSet *cst,       // 输入点集，从函数退出后其首个坐标点为最左
                                  // 点，末个坐标点为最右点。原来存放于首末位置
                                  // 的点分别被移位到原来放置最左最右点的位置。
            CoordiSet *convexcst  // 存放输出最左最右点的集合，这个点集在求凸壳
                                  // 的过程中被用来作为初始凸壳点集。
    );

    // Host 成员方法：updateDist（计算各点垂距）
    // 根据目前已知的凸壳上的点集和区域的标签值，根据点到直线的垂距公式，计算点
    // 集的附带数据：点到当前所在区域的最左最右点构成的直线的垂直距离。并且标记
    // 垂距为负的点。
    __host__ int                   // 返回值：函数是否正确执行，如果函数能够正
                                   // 确执行，返回 NO_ERROR。
    updateDist(
            CoordiSet *cst,        // 输入点集，也是输出点集，更新点集的
                                   // attachData，也就是垂距的信息。
            CoordiSet *convexcst,  // 目前已知凸壳上的点集，即每段的最值点信息
            int label[],           // 输入，当前点集的区域标签值数组。
            int cstcnt,            // 输入，当前点的数量。
            int negdistflag[]      // 输出，当前点垂距为负的标志数组。
                                   // 如果当前点垂距为负，则对应的标志位为 1。
    );

    // Host 成员方法：updateDistCpu（计算各点垂距）
    // 根据目前已知的凸壳上的点集和区域的标签值，根据点到直线的垂距公式，计算点
    // 集的附带数据：点到当前所在区域的最左最右点构成的直线的垂直距离。并且标记
    // 垂距为负的点。
    __host__ int                   // 返回值：函数是否正确执行，如果函数能够正
                                   // 确执行，返回 NO_ERROR。
    updateDistCpu(
            CoordiSet *cst,        // 输入点集，也是输出点集，更新点集的
                                   // attachData，也就是垂距的信息。
            CoordiSet *convexcst,  // 目前已知凸壳上的点集，即每段的最值点信息
            int label[],           // 输入，当前点集的区域标签值数组。
            int cstcnt,            // 输入，当前点的数量。
            int negdistflag[]      // 输出，当前点垂距为负的标志数组。
                                   // 如果当前点垂距为负，则对应的标志位为 1。
    );

    // Host 成员方法: updateFoundInfo（更新新发现凸壳点信息）
    // 根据分段扫描后得到的点集信息，更新各个区域是否有新发现的凸壳上的点，更新
    // 目前已知的凸壳上的点的位置索引。
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
            int startidx[]     // 输出，目前已知的凸壳上的点的位置索引数组，也
                               // 相当于当前每段上的起始位置的索引数组。
    );

    // Host 成员方法: updateFoundInfoCpu（更新新发现凸壳点信息）
    // 根据分段扫描后得到的点集信息，更新各个区域是否有新发现的凸壳上的点，更新
    // 目前已知的凸壳上的点的位置索引。
    __host__ int               // 返回值：函数是否正确执行，如果函数能够正确执
                               // 行，返回 NO_ERROR。
    updateFoundInfoCpu(
            int label[],       // 输入，当前点集的区域标签值数组。
            float dist[],      // 输入数组，分段扫描后，当前位置记录的本段目前
                               // 已知的最大垂距。
            int maxdistidx[],  // 输入，分段扫描后，当前位置记录的本段目前已知
                               // 的最大垂距点的位置索引数组。
            int cstcnt,        // 当前点的数量。
            int foundflag[],   // 输出数组，如当前区域内找到新的凸壳上的点，标
                               // 志位置 1。
            int startidx[]     // 输出，目前已知的凸壳上的点的位置索引数组，也
                               // 相当于当前每段上的起始位置的索引数组。
    );

    // Host 成员方法：updateConvexCst（生成新的凸壳点集）
    // 根据分段扫描后得到的点集信息，和每段上是否发现新凸壳点的信息，构造新的凸
    // 壳点集。
    __host__ int                     // 返回值：函数是否正确执行，如果函数能够
                                     // 正确执行，返回 NO_ERROR。
    updateConvexCst(
            CoordiSet *cst,          // 输入点集
            CoordiSet *convexcst,    // 输入，现有的凸壳上的点集。
            int foundflag[],         // 输入，当前区域内有新发现点的标志数组，
                                     // 如果当前区域内找到新的凸壳上的点，
                                     // 标志位置 1。
            int foundacc[],          // 输入，偏移量数组，当前区域内有新发现点
                                     // 的标志位的累加值。用来计算新添加的凸壳
                                     // 点的存放位置的偏移量。
            int startidx[],          // 输入，目前已知的凸壳上的点的索引数组，
                                     // 也相当于当前每段上的起始位置的索引数组
            int maxdistidx[],        // 输入，分段扫描后，当前位置记录的本段目
                                     // 前已知的最大垂距点的位置索引数组。
            int num,                 // 当前凸壳点的数量。
            CoordiSet *newconvexcst  // 输出，更新后目前已知凸壳上的点集，即每
                                     // 段的最值点信息。
    );

    // Host 成员方法：updateConvexCstCpu（生成新的凸壳点集）
    // 根据分段扫描后得到的点集信息，和每段上是否发现新凸壳点的信息，构造新的凸
    // 壳点集。
    __host__ int                     // 返回值：函数是否正确执行，如果函数能够
                                     // 正确执行，返回 NO_ERROR。
    updateConvexCstCpu(
            CoordiSet *cst,          // 输入点集
            CoordiSet *convexcst,    // 输入，现有的凸壳上的点集。
            int foundflag[],         // 输入，当前区域内有新发现点的标志数组，
                                     // 如果当前区域内找到新的凸壳上的点，
                                     // 标志位置 1。
            int foundacc[],          // 输入，偏移量数组，当前区域内有新发现点
                                     // 的标志位的累加值。用来计算新添加的凸壳
                                     // 点的存放位置的偏移量。
            int startidx[],          // 输入，目前已知的凸壳上的点的索引数组，
                                     // 也相当于当前每段上的起始位置的索引数组
            int maxdistidx[],        // 输入，分段扫描后，当前位置记录的本段目
                                     // 前已知的最大垂距点的位置索引数组。
            int num,                 // 当前凸壳点的数量。
            CoordiSet *newconvexcst  // 输出，更新后目前已知凸壳上的点集，即每
                                     // 段的最值点信息。
    );

    // Host 成员方法：markLeftPoints（标记左侧标点）
    // 根据目前每段上是否有新发现凸壳点的标志，标记在新发现的凸壳点点左侧的点，
    // 记录到标记数组。左侧标记在其后的计算过程中可以用来计算在下一轮迭代中坐标
    // 点的新位置。
    __host__ int                      // 返回值：函数是否正确执行，如果函数能够
                                      // 正确执行，返回 NO_ERROR。
    markLeftPoints(
            CoordiSet *cst,           // 输入点集，也是输出点集
            CoordiSet *newconvexcst,  // 输入，更新后的目前已知凸壳上的点集，即
                                      // 每段的最值点信息。
            int negdistflag[],        // 输入，负垂距标记值。
            int label[],              // 输入，当前点集的区域标签值数组。
            int foundflag[],          // 输入，当前区域内有新发现点的标志位数
                                      // 组，如果当前区域内找到新的凸壳上的点，
                                      // 标志位置 1。
            int foundacc[],           // 输入，偏移量数组，即当前区域内有新发现
                                      // 点的标志位的累加值。用来计算新添加的凸
                                      // 壳点的存放位置的偏移量。
            int cstcnt,               // 当前点的数量。
            int leftflag[]            // 输出，当前点在目前区域中新发现凸壳点的
                                      // 左侧的标志数组，如果在左侧，则置为 1。
    );

    // Host 成员方法：markLeftPointsCpu（标记左侧标点）
    // 根据目前每段上是否有新发现凸壳点的标志，标记在新发现的凸壳点点左侧的点，
    // 记录到标记数组。左侧标记在其后的计算过程中可以用来计算在下一轮迭代中坐标
    // 点的新位置。
    __host__ int                      // 返回值：函数是否正确执行，如果函数能够
                                      // 正确执行，返回 NO_ERROR。
    markLeftPointsCpu(
            CoordiSet *cst,           // 输入点集，也是输出点集
            CoordiSet *newconvexcst,  // 输入，更新后的目前已知凸壳上的点集，即
                                      // 每段的最值点信息。
            int negdistflag[],        // 输入，负垂距标记值。
            int label[],              // 输入，当前点集的区域标签值数组。
            int foundflag[],          // 输入，当前区域内有新发现点的标志位数
                                      // 组，如果当前区域内找到新的凸壳上的点，
                                      // 标志位置 1。
            int foundacc[],           // 输入，偏移量数组，即当前区域内有新发现
                                      // 点的标志位的累加值。用来计算新添加的凸
                                      // 壳点的存放位置的偏移量。
            int cstcnt,               // 当前点的数量。
            int leftflag[]            // 输出，当前点在目前区域中新发现凸壳点的
                                      // 左侧的标志数组，如果在左侧，则置为 1。
    );

    // Host 成员方法：updateProperty（更新点集的属性）
    // 根据已知信息，更新当前点集的区域标签和位置索引。
    __host__ int                      // 返回值：函数是否正确执行，如果函数能够
                                      // 正确执行，返回 NO_ERROR。
    updateProperty(
            int leftflag[],           // 输入，当前点在目前区域中新发现凸壳点的
                                      // 左侧的标志数组，如在左侧，则置为 1
            int leftacc[],            // 输入，偏移量数组，即当前点在目前区域中
                                      // 新发现凸壳点的左侧的标志的累加值。
            int negdistflag[],        // 输入，垂距为负的标志数组。如果当前点垂
                                      // 距为负，则对应的标志位为 1。
            int negdistacc[],         // 输入，垂距为正的标志的累加值数组。
            int startidx[],           // 输入，目前已知的凸壳上的点的位置索引数
                                      // 组，也相当于当前每段上的起始位置的索引
                                      // 数组。
            int label[],              // 输入，当前点集的区域标签值数组。
            int foundacc[],           // 输入，偏移量数组，即当前区域内有新发现
                                      // 凸壳点的标志位的累加值。用来计算新添加
                                      // 的凸壳点的存放位置的偏移量。
            int cstcnt,               // 坐标点的数量。 
            int newidx[],             // 输出，每个点的新的索引值数组。
            int tmplabel[]            // 输出，当前点集更新后的区域标签值数组。
    );

    // Host 成员方法：updateProperty（更新点集的属性）
    // 根据已知信息，更新当前点集的区域标签和位置索引。
    __host__ int                      // 返回值：函数是否正确执行，如果函数能够
                                      // 正确执行，返回 NO_ERROR。
    updatePropertyCpu(
            int leftflag[],           // 输入，当前点在目前区域中新发现凸壳点的
                                      // 左侧的标志数组，如在左侧，则置为 1
            int leftacc[],            // 输入，偏移量数组，即当前点在目前区域中
                                      // 新发现凸壳点的左侧的标志的累加值。
            int negdistflag[],        // 输入，垂距为负的标志数组。如果当前点垂
                                      // 距为负，则对应的标志位为 1。
            int negdistacc[],         // 输入，垂距为正的标志的累加值数组。
            int startidx[],           // 输入，目前已知的凸壳上的点的位置索引数
                                      // 组，也相当于当前每段上的起始位置的索引
                                      // 数组。
            int label[],              // 输入，当前点集的区域标签值数组。
            int foundacc[],           // 输入，偏移量数组，即当前区域内有新发现
                                      // 凸壳点的标志位的累加值。用来计算新添加
                                      // 的凸壳点的存放位置的偏移量。
            int cstcnt,               // 坐标点的数量。 
            int newidx[],             // 输出，每个点的新的索引值数组。
            int tmplabel[]            // 输出，当前点集更新后的区域标签值数组。
    );

    // Host 成员方法：arrangeCst（生成下一轮迭代的坐标点集）
    // 根据所求出的新下标，生成下一轮迭代中的新的坐标点集。
    __host__ int                // 返回值：函数是否正确执行，如果函数能够
                                // 正确执行，返回 NO_ERROR。
    arrangeCst(
            CoordiSet *cst,     // 输入点集。
            int negdistflag[],  // 输入，垂距为负的标志数组。 如果当前点垂距为
                                // 负，则对应的标志位为1。
            int newidx[],       // 输入，每个点的新的索引值数组。
            int tmplabel[],     // 输入，当前点集的区域标签值数组。
            int cstcnt,         // 坐标点数量。
            CoordiSet *newcst,  // 输出，更新元素位置后的新点集。
            int newlabel[]      // 输出，当前点集更新后的区域标签值数组。    
    );

    // Host 成员方法：arrangeCstCpu（生成下一轮迭代的坐标点集）
    // 根据所求出的新下标，生成下一轮迭代中的新的坐标点集。
    __host__ int                // 返回值：函数是否正确执行，如果函数能够
                                // 正确执行，返回 NO_ERROR。
    arrangeCstCpu(
            CoordiSet *cst,     // 输入点集。
            int negdistflag[],  // 输入，垂距为负的标志数组。 如果当前点垂距为
                                // 负，则对应的标志位为1。
            int newidx[],       // 输入，每个点的新的索引值数组。
            int tmplabel[],     // 输入，当前点集的区域标签值数组。
            int cstcnt,         // 坐标点数量。
            CoordiSet *newcst,  // 输出，更新元素位置后的新点集。
            int newlabel[]      // 输出，当前点集更新后的区域标签值数组。    
    );

    // Host 成员方法：flipWholeCst（整体翻转坐标点集）
    // 将坐标点集由第一象限翻转到第四象限，原来 (x, y) 坐标反转后为 (-x, -y)。
    // 该步骤用来求解上半凸壳，因为翻转后的点集的下半凸壳恰好是源点集的下半凸壳
    // 的相反数。
    __host__ int               // 返回值：函数是否正确执行，如果函数能够正确执
                               // 行，返回 NO_ERROR。
    flipWholeCst(
            CoordiSet *incst,  // 输入坐标点集，该坐标点集为只读点集
            CoordiSet *outcst  // 输出坐标点集，该坐标点集可以和输入坐标点集相
                               // 同，可进行 In-place 操作。
    );

    // Host 成员方法：flipWholeCstCpu（整体翻转坐标点集）
    // 将坐标点集由第一象限翻转到第四象限，原来 (x, y) 坐标反转后为 (-x, -y)。
    // 该步骤用来求解上半凸壳，因为翻转后的点集的下半凸壳恰好是源点集的下半凸壳
    // 的相反数。
    __host__ int               // 返回值：函数是否正确执行，如果函数能够正确执
                               // 行，返回 NO_ERROR。
    flipWholeCstCpu(
            CoordiSet *incst,  // 输入坐标点集，该坐标点集为只读点集
            CoordiSet *outcst  // 输出坐标点集，该坐标点集可以和输入坐标点集相
                               // 同，可进行 In-place 操作。
    );

    // Host 成员方法：convexHullIter（迭代法求凸壳点）
    // 采用 Quick Hull 迭代的方法，输出给定点集的上半或下半凸壳点。
    __host__ int                     // 返回值：函数是否正确执行，如果函数能够
                                     // 正确执行，返回 NO_ERROR。
    convexHullIter(
            CoordiSet *inputcst,     // 输入点集。
            CoordiSet *convexcst,    // 凸壳点集。
            bool lowerconvex = true  // 开关变量，用来确定是求解下半凸壳
                                     // （true）还是上半凸壳（false）
    );

    // Host 成员方法：convexHullIterCpu（迭代法求凸壳点）
    // 采用 Quick Hull 迭代的方法，输出给定点集的上半或下半凸壳点。
    __host__ int                     // 返回值：函数是否正确执行，如果函数能够
                                     // 正确执行，返回 NO_ERROR。
    convexHullIterCpu(
            CoordiSet *inputcst,     // 输入点集。
            CoordiSet *convexcst,    // 凸壳点集。
            bool lowerconvex = true  // 开关变量，用来确定是求解下半凸壳
                                     // （true）还是上半凸壳（false）
    );

    // Host 成员方法：joinConvex（合并凸壳点）
    // 将通过迭代求得的两个凸壳点集（下半凸壳点集和上半凸壳点集）合并成一个完整
    // 的凸壳点集。合并过程中两侧若有重复点需要去掉。
    __host__ int                 // 返回值：函数是否正确执行，如果函数能够正确
                                 // 执行，返回 NO_ERROR。
    joinConvex(
            CoordiSet *lconvex,  // 下半凸壳
            CoordiSet *uconvex,  // 上半凸壳
            CoordiSet *convex    // 整合后的凸壳
    );

    // Host 成员方法：joinConvexCpu（合并凸壳点）
    // 将通过迭代求得的两个凸壳点集（下半凸壳点集和上半凸壳点集）合并成一个完整
    // 的凸壳点集。合并过程中两侧若有重复点需要去掉。
    __host__ int                 // 返回值：函数是否正确执行，如果函数能够正确
                                 // 执行，返回 NO_ERROR。
    joinConvexCpu(
            CoordiSet *lconvex,  // 下半凸壳
            CoordiSet *uconvex,  // 上半凸壳
            CoordiSet *convex    // 整合后的凸壳
    );

public:

    // 构造函数：ConvexHull
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    ConvexHull()
    {
        // 配置扫描器。
        this->aryScan.setScanType(NAIVE_SCAN);

        // 使用默认值为类的各个成员变量赋值。
        this->imgCvt.clearAllConvertFlags();             // 清除所有的转换标志位
        this->value = 128;                               // 设置转换阈值。
        this->imgCvt.setConvertFlags(this->value, 255);  // 置位某范围
                                                         // 内的转换标志位。
    }

    // 成员方法：getValue（获取图像转换点集的像素阈值）
    // 获取图像转换点集的像素阈值。
    __host__ __device__ unsigned char  // 返回值：图像转换点集的像素阈值。
    getValue() const
    {
        // 返回图像转换点集的像素阈值。
        return this->value;    
    } 

    // 成员方法：setValue（设置图像转换点集的像素阈值）
    // 设置图像转换点集的像素阈值。
    __host__ __device__ int      // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    setValue(
            unsigned char value  // 设定新的图像转换点集的像素阈值。
    ) {
        // 根据阈值设置转换标志位。
        if (value < this->value)
            this->imgCvt.setConvertFlags(value, this->value - 1);
        else if (value > this->value)
            this->imgCvt.clearConvertFlags(this->value, value - 1);
        
        // 将图像转换点集的像素阈值赋成新值。
        this->value = value;

        return NO_ERROR;
    }

    // Host 成员方法：convexHull（求一个点集对应的凸壳点集）
    // 求出一个点集所对应的凸壳点集，所求得的凸壳点集从左下角点其，按逆时针顺序
    // 排列。
    __host__ int                  // 返回值：函数是否正确执行，如果函数能够正确
                                  // 执行，返回 NO_ERROR。
    convexHull(
            CoordiSet *inputcst,  // 输入点集
            CoordiSet *convex     // 凸壳点集
    );

    // Host 成员方法：convexHullCpu（求一个点集对应的凸壳点集）
    // 求出一个点集所对应的凸壳点集，所求得的凸壳点集从左下角点其，按逆时针顺序
    // 排列。
    __host__ int                  // 返回值：函数是否正确执行，如果函数能够正确
                                  // 执行，返回 NO_ERROR。
    convexHullCpu(
            CoordiSet *inputcst,  // 输入点集
            CoordiSet *convex     // 凸壳点集
    );

    // Host 成员方法：convexHull（求图像中阈值给定的对象对应的凸壳点集）
    // 根据给定的阈值在图像中找出对象，所求得的凸壳点集从左下角点起，按逆时针
    // 顺序排列。
    __host__ int               // 返回值：函数是否正确执行，如果函数能够正确
                               // 执行，返回 NO_ERROR。
    convexHull(
            Image *inimg,      // 输入图像。
            CoordiSet *convex  // 凸壳点集
    );

    // Host 成员方法：convexHullCpu（求图像中阈值给定的对象对应的凸壳点集）
    // 根据给定的阈值在图像中找出对象，所求得的凸壳点集从左下角点起，按逆时针
    // 顺序排列。
    __host__ int               // 返回值：函数是否正确执行，如果函数能够正确
                               // 执行，返回 NO_ERROR。
    convexHullCpu(
            Image *inimg,      // 输入图像。
            CoordiSet *convex  // 凸壳点集
    );
};

#endif

