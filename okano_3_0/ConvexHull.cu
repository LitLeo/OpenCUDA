// ConvexHull.cu
// 凸壳算法实现。

#include "ConvexHull.h"

#include <stdio.h>

// 宏：CH_DEBUG_KERNEL_PRINT（Kernel 调试打印开关）
// 打开该开关则会在 Kernel 运行时打印相关的信息，以参考调试程序；如果注释掉该
// 宏，则 Kernel 不会打印这些信息，但这会有助于程序更快速的运行。
//#define CH_DEBUG_KERNEL_PRINT

// 宏：CH_DEBUG_CPU_PRINT（CPU版本 调试打印开关）
// 打开该开关则会在 CPU 版本运行时打印相关的信息，以参考调试程序；如果注释掉该
// 宏，则 CPU 不会打印这些信息，但这会有助于程序更快速的运行。
//#define CH_DEBUG_CPU_PRINT

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的 2D Block 尺寸。
#define DEF_BLOCK_X    32
#define DEF_BLOCK_Y     8

// 宏：DEF_BLOCK_1D
// 定义了默认的 1D Block 尺寸。
#define DEF_BLOCK_1D  512

// 宏：DEF_WARP_SIZE
// 定义了 Warp 尺寸
#define DEF_WARP_SIZE  32

// 宏：CH_LARGE_ENOUGH
// 定义了一个足够大的正整数，该整数在使用过程中被认为是无穷大。
#define CH_LARGE_ENOUGH  ((1 << 30) - 1)

// Kernel 函数：_initiateImgKer（实现将图像初始化为 threshold 算法）
// 图像的 threshold 由用户决定，所以在初始化输入图像时
// 将输入图像内各个点的像素值置为 threshold。
static __global__ void
_initiateImgKer(
        ImageCuda inimg,         // 输入图像
        unsigned char threshold  // 阈值
);

// Kernel 函数：_initLabelAryKer（初始化 LABEL 数组）
// 在凸壳迭代前初始化 LABEL 数组，初始化后的 LABEL 数组要求除最后一个元素为 1 
// 以外，其他的元素皆为 0。
static __global__ void  // Kernel 函数无返回值
_initLabelAryKer(
        int label[],    // 待初始化的 LABEL 数组。
        int cstcnt      // LABEL 数组长度。
);

// Kernel 函数：_swapEdgePointKer（寻找最左最右点）
// 寻找一个坐标点集中的最左和最右两侧的点，并将结果存入一个整型数组中。如果存在
// 多个最左点或者多个最右点的情况时，则以 y 值最大者为准。找到最左最右点后将直
// 接修改输入坐标点集，使最左点处于坐标点集的第一个点，输出点处于坐标点集的最末
// 一点。
static __global__ void      // Kernel 函数无返回值
_swapEdgePointKer(
        CoordiSetCuda cst,  // 待寻找的坐标点集
        int edgecst[4],     // 寻找到的最左最右点的坐标。其中下标 0 和 1 表示最
                            // 左点的下标，下标 2 和 3 表示最右点的下标。该参数
                            // 可以为 NULL。
        int edgeidx[2]      // 寻找到的最左点和最右点对应于输入坐标点集的下标。
                            // 该参数可以为 NULL。
);

// Kernel 函数: _updateDistKer（更新点集的垂距信息）
// 根据目前已知的凸壳上的点集和区域的标签值，找出当前每个点所在区域的最左最右
// 点，根据点到直线的垂距公式，计算点集的附带数据：点到当前所在区域的最左最右点
// 构成的直线的垂直距离。并且将垂距为负的点标记一下。
static __global__ void            // Kernel 函数无返回值
_updateDistKer(
        CoordiSetCuda cst,        // 输入点集，也是输出点集，更新点集的 
                                  // attachData，也就是垂距的信息。
        CoordiSetCuda convexcst,  // 目前已知凸壳上的点集，即每段的最值点信息。
        int label[],              // 输入，当前点集的区域标签值数组。
        int cstcnt,               // 输入，当前点的数量。
        int negdistflag[]         // 输出，当前点垂距为负的标志数组。如果当前点
                                  // 垂距为负，则对应的标志位为 1。
);

// Kernel 函数: _updateFoundInfoKer（更新新发现凸壳点信息）
// 根据分段扫描后得到的点集信息，更新当前区域是否有新发现的凸壳上的点，更新目前
// 已知的凸壳上的点的位置索引。
static __global__ void     // Kernel 函数无返回值
_updateFoundInfoKer(
        int label[],       // 输入，当前点集的区域标签值数组。   
        float dist[],      // 输入数组，所有点的垂距，即坐标点集数据结构中的 
                           // attachedData 域。
        int maxdistidx[],  // 输入，分段扫描后，当前位置记录的本段目前已知的最
                           // 大垂距点的位置索引数组。
        int cstcnt,        // 坐标点的数量。
        int foundflag[],   // 输出数组，如果当前区域内找到新的凸壳上的点，标志
                           // 位置 1。
        int startidx[]     // 输出，目前已知的凸壳上的点的位置索引数组，也相当
                           // 于当前每段上的起始位置的索引数组。
);

// Kernel 函数: _updateConvexCstKer（生成新凸壳点集）
// 根据分段扫描后得到的点集信息，和每段上是否发现新凸壳点的信息，生成新的凸壳点
// 集。
static __global__ void              // Kernel 函数无返回值
_updateConvexCstKer(
        CoordiSetCuda cst,          // 输入点集
        CoordiSetCuda convexcst,    // 输入，现有的凸壳上的点集。
        int foundflag[],            // 输入，当前区域内有新发现点的标志位数组，
                                    // 如果当前区域内找到新的凸壳上的点，标志位
                                    // 置 1。
        int foundacc[],             // 输入，偏移量数组，即当前区域内有新发现点
                                    // 的标志位的累加值。用来计算新添加的
                                    // 凸壳点的存放位置的偏移量。
        int startidx[],             // 输入，目前已知的凸壳上点的位置索引数组，
                                    // 也相当于当前每段上的起始位置的索引数组。
        int maxdistidx[],           // 输入，分段扫描后，当前位置记录
                                    // 的本段目前已知的最大垂距点的位置索引数组
        int convexcnt,              // 当前凸壳点的数量。
        CoordiSetCuda newconvexcst  // 输出，更新后的目前已知凸壳上的点集，
                                    // 即每段的最值点信息。
);

// Kernel 函数: _markLeftPointsKer（标记左侧点）
// 根据目前每段上是否有新发现凸壳点的标志，标记在新发现点左侧的点，记录到标记数
// 组。
static __global__ void               // Kernel 函数无返回值
_markLeftPointsKer(
        CoordiSetCuda cst,           // 输入点集，也是输出点集
        CoordiSetCuda newconvexcst,  // 输入，更新后的目前已知凸壳上的点集，即
                                     // 每段的最值点信息。
        int negdistflag[],           // 输入，负垂距标记值。
        int label[],                 // 输入，当前点集的区域标签值数组。
        int foundflag[],             // 输入，当前区域内有新发现点的标志位数
                                     // 组，如果当前区域内找到新的凸壳上的点，
                                     // 标志位置 1
        int foundacc[],              // 输入，偏移量数组，即当前区域内有新发现
                                     // 点的标志位的累加值。用来计算新添加的凸
                                     // 壳点的存放位置的偏移量。
        int cstcnt,                  // 坐标点的数量。
        int leftflag[]               // 输出，当前点在目前区域中新发现凸壳点的
                                     // 左侧的标志数组，如果在左侧，则置为 1。
);

// Kernel 函数: _updatePropertyKer（计算新下标）
// 计算原坐标点集中各个坐标点的新位置和新的 LABEL 值。这些值将在下一步中用来并
// 行生成新的坐标点集。
static __global__ void      // Kernel 函数无返回值
_updatePropertyKer(
        int leftflag[],     // 输入，当前点在目前区域中新发现凸壳点的左侧的标志
                            // 数组，如果在左侧，则置为 1。
        int leftacc[],      // 输入，偏移量数组，即当前点在目前区域中新发现凸壳
                            // 点的左侧的标志的累加值。
        int negdistflag[],  // 输入，垂距为负的标志数组。如果当前点垂距为负，则
                            // 对应的标志位为 1。
        int negdistacc[],   // 输入，垂距为正的标志的累加值数组。
        int startidx[],     // 输入，目前已知的凸壳上的点的位置索引数组，也相当
                            // 于当前每段上的起始位置的索引数组。
        int label[],        // 输入，当前点集的区域标签值数组。
        int foundacc[],     // 输入，偏移量数组，即当前区域内有新发现点的标志位
                            // 的累加值。用来计算新添加的凸壳点的存放位置的偏移
                            // 量。
        int cstcnt,         // 坐标点的数量。 
        int newidx[],       // 输出，每个点的新的索引值数组。
        int tmplabel[]      // 输出，当前点集更新后的区域标签值数组。该数组可以
                            // 和 label 数组是同一个数组，即可以进行 In-place 
                            // 操作。
);

// Kernel 函数: _arrangeCstKer（形成新坐标点集）
// 根据已知信息，形成新的坐标点集和对应的 LABEL 数组，这些数据将在下一轮迭代中
// 作为输入信息。
static __global__ void         // Kernel 函数无返回值
_arrangeCstKer(
        CoordiSetCuda cst,     // 输入点集。
        int negdistflag[],     // 输入，垂距为负的标志数组。如果当前点垂距为
                               // 负，则对应的标志位为 1。 
        int newidx[],          // 输入，每个点的新的索引值数组。
        int tmplabel[],        // 输入，当前点集的区域标签值数组。
        int cstcnt,            // 输入，坐标点的数量。
        CoordiSetCuda newcst,  // 输出，更新元素位置后的新点集。
        int newlabel[]         // 输出，当前点集更新后的区域标签值数组。
);

// Kernel 函数：_flipWholeCstKer（整体翻转坐标点集）
// 将坐标点集由第一象限翻转到第四象限，原来 (x, y) 坐标反转后为 (-x, -y)。该步
// 骤用来求解上半凸壳，因为翻转后的点集的下半凸壳恰好是源点集的下半凸壳的相反
// 数。
static __global__ void        // Kernel 函数无返回值
_flipWholeCstKer(
        CoordiSetCuda incst,  // 输入坐标点集，该坐标点集为只读点集
        CoordiSetCuda outcst  // 输出坐标点集，该坐标点集可以和输入坐标点集相
                              // 同，可进行 In-place 操作。
);

// Kernel 函数：_joinConvexKer（合并凸壳点）
// 将通过迭代求得的两个凸壳点集（下半凸壳点集和上半凸壳点集）合并成一个完整的凸
// 壳点集。合并过程中两侧若有重复点需要去掉。
static __global__ void          // Kernel 函数无返回值
_joinConvexKer(
        CoordiSetCuda lconvex,  // 下半凸壳
        CoordiSetCuda uconvex,  // 上半凸壳
        CoordiSetCuda convex,   // 整合后的凸壳
        int *convexcnt          // 整合后凸壳点集的数量
);

// Kernel 函数：_initLabelAryKer（初始化 LABEL 数组）
static __global__ void _initLabelAryKer(int label[], int cstcnt)
{
    // 计算当前 Thread 对应的数组下标。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果当前下标处理的是越界数据，则直接退出。
    if (idx >= cstcnt)
        return;

    // 在 LABEL 数组中，将最后一个变量写入 1，其余变量写入 0。
    if (idx == cstcnt - 1)
        label[idx] = 1;
    else
        label[idx] = 0;
}

// Host 成员方法：initLabelAry（初始化 LABEL 数组）
__host__ int ConvexHull::initLabelAry(int label[], int cstcnt)
{
    // 检查输入的数组是否为 NULL。
    if (label == NULL)
        return NULL_POINTER;

    // 检查数组长度是否大于等于 2。
    if (cstcnt < 2)
        return INVALID_DATA;

    // 计算启动 Kernel 函数所需要的 Block 尺寸与数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (cstcnt + blocksize - 1) / blocksize;

    // 启动 Kernel 函数，完成计算。
    _initLabelAryKer<<<gridsize, blocksize>>>(label, cstcnt);

    // 检查 Kernel 函数执行是否正确。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 成员方法：initLabelAryCpu（初始化 LABEL 数组）
__host__ int ConvexHull::initLabelAryCpu(int label[], int cstcnt)
{
    // 检查输入的数组是否为 NULL。
    if (label == NULL)
        return NULL_POINTER;

    // 检查数组长度是否大于等于 2。
    if (cstcnt < 2)
        return INVALID_DATA;

    // 在 LABEL 数组中，将最后一个变量写入 1，其余变量写入 0。
    for (int i = 0; i < cstcnt - 1; i++) {
        label[i] = 0;
        // 打印信息检查
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[initLabelAryCpu]label " << i << " is " << label[i] << endl;
#endif
    }
    label[cstcnt - 1] = 1;

#ifdef CH_DEBUG_CPU_PRINT
    cout << "[initLabelAryCpu]label " << cstcnt - 1 << " is " <<
            label[cstcnt - 1] << endl;
    cout << endl;
#endif  

    // 处理完毕，返回。
    return NO_ERROR;
}

// Kernel 函数：_swapEdgePointKer（寻找最左最右点）
static __global__ void _swapEdgePointKer(
        CoordiSetCuda cst, int edgecst[4], int edgeidx[2])
{
    // 计算当前线程的下标，该 Kernel 必须以单 Block 运行，因此不涉及到 Block 相
    // 关的变量。
    int idx = threadIdx.x;

    // 当前 Thread 处理的若干个点中找到的局部最左最右点。
    int curleftx  = CH_LARGE_ENOUGH,  curlefty  = -CH_LARGE_ENOUGH;
    int currightx = -CH_LARGE_ENOUGH, currighty = -CH_LARGE_ENOUGH;

    // 当前 Thread 处理的若干个点中找到的局部最左做有点的下标。
    int curleftidx = -1, currightidx = -1;

    // 处于下标为 idx 处的坐标点坐标值。
    int curx, cury;

    // 迭代处理该线程所要处理的所有坐标点，这些坐标点是间隔 blockDim.x 个的各个
    // 坐标点。
    while (idx < cst.tplMeta.count) {
        // 从 Global Memory 中读取坐标值。
        curx = cst.tplMeta.tplData[2 * idx];
        cury = cst.tplMeta.tplData[2 * idx + 1];

        // 判断该坐标值的大小，和已经找到的最左最优值做比较，更新最左最右点。
        // 首先对最左点进行更新。
        if (curx < curleftx || (curx == curleftx && cury > curlefty)) {
            curleftx = curx;
            curlefty = cury;
            curleftidx = idx;
        }

        // 然后对最右点进行更新。
        if (curx > currightx || (curx == currightx && cury > currighty)) {
            currightx = curx;
            currighty = cury;
            currightidx = idx;
        }

        // 更新 idx，在下一轮迭代时计算下一个点。
        idx += blockDim.x;
    }

    // 至此，所有 Thread 都得到了自己的局部最左最右点，现在需要将这些点放入 
    // Shared Memory 中，以便下一步进行归约处理。

    // 声明 Shared Memory，并分配各个指针。
    extern __shared__ int shdmem[];
    int *leftxShd    = &shdmem[0];
    int *leftyShd    = &leftxShd[blockDim.x];
    int *leftidxShd  = &leftyShd[blockDim.x];
    int *rightxShd   = &leftidxShd[blockDim.x];
    int *rightyShd   = &rightxShd[blockDim.x];
    int *rightidxShd = &rightyShd[blockDim.x];

    // 将局部结果拷贝到 Shared Memory 中。
    idx = threadIdx.x;
    leftxShd[idx] = curleftx;
    leftyShd[idx] = curlefty;
    leftidxShd[idx] = curleftidx;
    rightxShd[idx] = currightx;
    rightyShd[idx] = currighty;
    rightidxShd[idx] = currightidx;

    // 同步所有线程，使初始化Shared Memory 的结果对所有线程可见。
    __syncthreads();

    // 下面进行折半归约迭代。这里要求 blockDim.x 必须为 2 的整数次幂。
    int curstep = blockDim.x / 2;
    for (/*curstep*/; curstep >= 1; curstep /= 2) {
        // 每一轮迭代都只有上一轮一半的点在参与。直到最后剩下一个线程。
        if (idx < curstep) {
            // 将两个局部结果归约成一个局部结果。
            // 首先处理最左点。
            if (leftxShd[idx] > leftxShd[idx + curstep] ||
                (leftxShd[idx] == leftxShd[idx + curstep] &&
                 leftyShd[idx] < leftyShd[idx + curstep])) {
                leftxShd[idx] = leftxShd[idx + curstep];
                leftyShd[idx] = leftyShd[idx + curstep];
                leftidxShd[idx] = leftidxShd[idx + curstep];
            }

            // 之后处理最右点。
            if (rightxShd[idx] < rightxShd[idx + curstep] ||
                (rightxShd[idx] == rightxShd[idx + curstep] &&
                 rightyShd[idx] < rightyShd[idx + curstep])) {
                rightxShd[idx] = rightxShd[idx + curstep];
                rightyShd[idx] = rightyShd[idx + curstep];
                rightidxShd[idx] = rightidxShd[idx + curstep];
            }
        }

        // 同步线程，使本轮迭代归约的结果对所有线程可见。
        __syncthreads();
    }

    // 下面进行一些零碎的收尾工作。由于访问 Shared Memory 不同部分，造成 Bank 
    // Conflict 的概率很大，因此没有采用并行处理（此时即便是并行代码，硬件上也
    // 会串行处理）
    if (idx == 0) {
        // 如果 edgecst 不为 NULL，则将找到的最左最右点坐标拷贝其中。
        if (edgecst != NULL) {
            edgecst[0] = leftxShd[0];
            edgecst[1] = leftyShd[0];
            edgecst[2] = rightxShd[0];
            edgecst[3] = rightyShd[0];
        }
    } else if (idx == DEF_WARP_SIZE) {
        // 如果 edgeidx 不为 NULL，则将找到的最左最右点下标拷贝其中。
        if (edgeidx != NULL) {
            edgeidx[0] = leftidxShd[0];
            edgeidx[1] = rightidxShd[0];
        }
    } else if (idx == DEF_WARP_SIZE * 2) {
        // 将最左点交换到坐标点集的首部。
        if (leftidxShd[0] > 0) {
            curx = cst.tplMeta.tplData[0];
            cury = cst.tplMeta.tplData[1];
            cst.tplMeta.tplData[0] = leftxShd[0];
            cst.tplMeta.tplData[1] = leftyShd[0];
            cst.tplMeta.tplData[2 * leftidxShd[0]] = curx;
            cst.tplMeta.tplData[2 * leftidxShd[0] + 1] = cury;
        }
    } else if (idx == DEF_WARP_SIZE * 3) {
        // 将最右点交换到坐标点集的尾部。
        if (rightidxShd[0] < cst.tplMeta.count - 1) {
            curx = cst.tplMeta.tplData[2 * (cst.tplMeta.count - 1)];
            cury = cst.tplMeta.tplData[2 * (cst.tplMeta.count - 1) + 1];
            cst.tplMeta.tplData[2 * (cst.tplMeta.count - 1)] = rightxShd[0];
            cst.tplMeta.tplData[2 * (cst.tplMeta.count - 1) + 1] = 
                    rightyShd[0];
            cst.tplMeta.tplData[2 * rightidxShd[0]] = curx;
            cst.tplMeta.tplData[2 * rightidxShd[0] + 1] = cury;
        }
    }
}

// Host 成员方法：swapEdgePoint（寻找最左最右点）
__host__ int ConvexHull::swapEdgePoint(CoordiSet *cst, CoordiSet *convexcst)
{
    // 判断函数参数是否为 NULL。
    if (cst == NULL || convexcst == NULL)
        return NULL_POINTER;

    // 判断参数是否包含了足够多的坐标点。
    if (cst->count < 2 || cst->tplData == NULL ||
        convexcst->count < 2 || convexcst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    errcode = CoordiSetBasicOp::copyToCurrentDevice(cst);
    if (errcode != NO_ERROR)
        return errcode;

    errcode = CoordiSetBasicOp::copyToCurrentDevice(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    //CoordiSetCuda *convexcstCud = COORDISET_CUDA(convexcst);

    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = 1;
    size_t sharedsize = 6 * blocksize * sizeof (int);

    _swapEdgePointKer<<<gridsize, blocksize, sharedsize>>>(
            *cstCud, convexcst->tplData, NULL);

    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return NO_ERROR;
}

// Host 成员方法：swapEdgePointCpu（寻找最左最右点）
__host__ int ConvexHull::swapEdgePointCpu(CoordiSet *cst, CoordiSet *convexcst)
{
    // 判断函数参数是否为 NULL。
    if (cst == NULL || convexcst == NULL)
        return NULL_POINTER;

    // 判断参数是否包含了足够多的坐标点。
    if (cst->count < 2 || cst->tplData == NULL ||
        convexcst->count < 2 || convexcst->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 拷贝到 Host 端
    errcode = CoordiSetBasicOp::copyToHost(cst);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = CoordiSetBasicOp::copyToHost(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);

    // 记录当前最左最右点，初始化为第一个点。
    int curleftx  = (*cstCud).tplMeta.tplData[2 * 0];
    int curlefty  = (*cstCud).tplMeta.tplData[2 * 0 + 1];
    int currightx = curleftx; 
    int currighty = curlefty;
    int curleftidx = 0;
    int currightidx = 0;

    int id;
    int curx, cury;

    // 寻找最左最右点
    for(id = 1; id < (*cstCud).tplMeta.count; id++) {
        // 当前点
        curx = (*cstCud).tplMeta.tplData[2 * id];
        cury = (*cstCud).tplMeta.tplData[2 * id + 1];

        // 首先对最左点进行更新。
        if (curx < curleftx || (curx == curleftx && cury > curlefty)) {
            curleftx = curx;
            curlefty = cury;
            curleftidx = id;
        }

        // 然后对最右点进行更新。
        if (curx > currightx || (curx == currightx && cury > currighty)) {
            currightx = curx;
            currighty = cury;
            currightidx = id;
        }
    }

#ifdef CH_DEBUG_CPU_PRINT
    cout << "[swapEdgePointCpu]the left point is " << curleftidx <<
            " the right point is " << currightidx << endl;
#endif

    // 下面进行一些零碎的收尾工作。
    // 将找到的最左最右点坐标拷贝到凸壳点集。
    convexcst->tplData[0] = curleftx;
    convexcst->tplData[1] = curlefty;
    convexcst->tplData[2] = currightx;
    convexcst->tplData[3] = currighty;

    // 将最左点交换到坐标点集的首部。
    if (curleftidx > 0) {
        curx = (*cstCud).tplMeta.tplData[0];
        cury = (*cstCud).tplMeta.tplData[1];
        (*cstCud).tplMeta.tplData[0] = curleftx;
        (*cstCud).tplMeta.tplData[1] = curlefty;
        (*cstCud).tplMeta.tplData[2 * curleftidx] = curx;
        (*cstCud).tplMeta.tplData[2 * curleftidx + 1] = cury;
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[swapEdgePointCpu]first cst x is " <<
                (*cstCud).tplMeta.tplData[0] << endl;
        cout << "[swapEdgePointCpu]first cst y is " <<
                (*cstCud).tplMeta.tplData[1] << endl;
        cout << "[swapEdgePointCpu]former leftest cst x now is " <<
                (*cstCud).tplMeta.tplData[2 * curleftidx] << endl;
        cout << "[swapEdgePointCpu]former leftest cst y now is " <<
                (*cstCud).tplMeta.tplData[2 * curleftidx + 1] << endl;
#endif 
    }

    // 将最右点交换到坐标点集的尾部。
    if (currightidx < cst->count - 1) {
        curx = (*cstCud).tplMeta.tplData[2 * (cst->count - 1)];
        cury = (*cstCud).tplMeta.tplData[2 * (cst->count - 1) + 1];
        (*cstCud).tplMeta.tplData[2 * (cst->count - 1)] = currightx;
        (*cstCud).tplMeta.tplData[2 * (cst->count - 1) + 1] = currighty;
        (*cstCud).tplMeta.tplData[2 * currightidx] = curx;
        (*cstCud).tplMeta.tplData[2 * currightidx + 1] = cury;
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[swapEdgePointCpu]last cst x is " <<
                (*cstCud).tplMeta.tplData[2 * (cst->count - 1)] << endl;
        cout << "[swapEdgePointCpu]last cst y is " <<
                (*cstCud).tplMeta.tplData[2 * (cst->count - 1) + 1] << endl;
        cout << "[swapEdgePointCpu]former rightest cst x now is " <<
                (*cstCud).tplMeta.tplData[2 * currightidx] << endl;
        cout << "[swapEdgePointCpu]former rightest cst y now is " <<
                (*cstCud).tplMeta.tplData[2 * currightidx + 1] << endl;
#endif
    }

    return NO_ERROR;
}

// Kernel 函数: _updateDistKer（更新点集的垂距信息）
static __global__ void _updateDistKer(
        CoordiSetCuda cst, CoordiSetCuda convexcst, int label[], 
        int cstcnt, int negdistflag[])
{
    // 记录了本 Kernel 所使用到的共享内存中各个下标所存储的数据的含义。其中，
    // SIDX_BLK_CNT 表示当前 Block 所需要处理的坐标点的数量，由于坐标点的数量不
    // 一定能够被 BlockDim 整除，因此最后一个 Block 所处理的坐标点的数量要小于 
    // BlockDim。
    // SIDX_BLK_LABEL_LOW 和 SIDX_BLK_LABEL_UP 用来存当前 Block 中所加载的点集
    // 的区域标签值的上下界。根据这个上下界，可以计算出当前点所在区域的最左最右
    // 点，从而根据这两点确定的直线计算当前点的垂距。
    // 从下标为 SIDX_BLK_CST 开始的其后的所有共享内存空间存储了当前 Block 中的
    // 点集坐标。坐标集中第 i 个点对应的数组下标为 2 * i 和 2 * i + 1，其中下标
    // 为 2 * i 的数据表示该点的横坐标，下标为 2 * i + 1 的数据表示该点的纵坐
    // 标。
#define SIDX_BLK_CNT        0
#define SIDX_BLK_LABEL_LOW  1
#define SIDX_BLK_LABEL_UP   2
#define SIDX_BLK_CONVEX     3    

    // 共享内存的声明。
    extern __shared__ int shdmem[];

    // 基准索引。表示当前 Block 的起始位置索引。
    int baseidx = blockIdx.x * blockDim.x;
    // 全局索引。
    int idx = baseidx + threadIdx.x;

    // 当前 Block 的第 0 个线程来处理共享内存中彼此共享的数据的初始化工作。
    if (threadIdx.x == 0) {
        // 计算当前 Block 所要处理的坐标点的数量。默认情况下该值等于 BlockDim，
        // 但对于最后一个 Block 来说，在坐标点总数量不能被 BlockDim 所整除的时
        // 候，需要处理的坐标点数量会小于 BlockDim。
        if (baseidx + blockDim.x <= cstcnt)
            shdmem[SIDX_BLK_CNT] = blockDim.x;
        else
            shdmem[SIDX_BLK_CNT] = cstcnt - baseidx;

        // 计算当前 Block 所处理的坐标点中起始的 LABEL 编号。
        shdmem[SIDX_BLK_LABEL_LOW] = label[baseidx];

        // 计算当前 Block 索要处理的坐标点中最大的 LABEL 编号。由于考虑到根据两
        // 点计算直线方程，因此所谓的最大 LABEL 编号其实是
        if (baseidx + shdmem[SIDX_BLK_CNT] <= cstcnt)
            shdmem[SIDX_BLK_LABEL_UP] = 
                    label[baseidx + shdmem[SIDX_BLK_CNT] - 1] + 1;
        else
            shdmem[SIDX_BLK_LABEL_UP] = label[cstcnt - 1];
    }

    // Block 内部同步，使得上一步的初始化对 Block 内的所有 Thread 可见。
    __syncthreads();

    // 将当前 Block 处理的 LABEL 值上下界加载到寄存器，该步骤没有逻辑上的含义，
    // 只是为了 GPU 处理速度更快。
    int labellower = shdmem[SIDX_BLK_LABEL_LOW];
    int labelupper = shdmem[SIDX_BLK_LABEL_UP];

    // 为了方便代码编写，这里单独提出一个 blockcstShd 指针，指向当前 Block 所对
    // 应的点集数据的共享内存空间。
    int *convexShd = &shdmem[SIDX_BLK_CONVEX];

    // 加载当前 Block 中所用到的 LABEL 所谓应的凸壳点，两个相邻 LABEL 的凸壳点
    // 构成的直线可用来衡量各点的垂距并以此推算出下一轮的凸壳点。将所用到的凸壳
    // 点加载的 Shared Memory 中也没有逻辑上的目的，仅仅是为了下一步计算时访存
    // 时间的缩短。
    if (threadIdx.x < labelupper - labellower + 1) {
        convexShd[2 * threadIdx.x] =
                convexcst.tplMeta.tplData[2 * (labellower + threadIdx.x)];
        convexShd[2 * threadIdx.x + 1] =
                convexcst.tplMeta.tplData[2 * (labellower + threadIdx.x) + 1];
    }

    // Block 内部同步，使得上面所有的数据加载对 Block 内的所有 Thread 可见。下
    // 面的代码就正式的投入计算了。
    __syncthreads();

    // 如果当前线程的全局下标越界，则直接返回，因为他没有对应的所要处理坐标点。
    if (idx >= cstcnt)
        return;

    // 对于最后一个点（其实是坐标集中的最右点）是一个特殊的点，它独自处于一个
    // LABEL，因此对于它不需要进行计算，直接赋值就行了。
    if (idx == cstcnt - 1) {
        cst.attachedData[idx] = 0.0f;
        negdistflag[idx] = 0;
        return;
    }

    // 计算当前点的坐标和区域标签值。
    int curx = cst.tplMeta.tplData[2 * idx];
    int cury = cst.tplMeta.tplData[2 * idx + 1];
    int curlabelidx = 2 * (label[idx] - labellower);

    // 计算当前 LABEL 区域的最左点的坐标。
    int leftx = convexShd[curlabelidx++];
    int lefty = convexShd[curlabelidx++];

    // 计算当前 LABEL 区域的最右点的坐标。
    int rightx = convexShd[curlabelidx++];
    int righty = convexShd[curlabelidx  ];

    // 如果当前点就是凸壳点，那么不需要计算直接赋值退出就可以了。
    if ((curx == leftx && cury == lefty) || 
        (curx == rightx && cury == righty)) {
        cst.attachedData[idx] = 0.0f;
        negdistflag[idx] = 0;
        return;
    }

    // 计算垂距，该计算通过最左点和最右点形成的直线作为垂距求解的依据，但实际求
    // 解过程中并不是真正的垂距，而是垂直于 y 轴的距离。当点在直线之下时，具有
    // 正垂距，当点在直线之上时，具有负垂距。
    // y ^        right
    //   |          +
    //   |         /
    //   |        / --
    //   |       /|  ^
    //   |      / |  | dist
    //   |     /  |  v
    //   |    /   * -- 
    //   |   +   cur
    //  O| left           x
    // --+----------------->
    //   |
    float s = (float)(righty - lefty) / (rightx - leftx);
    float dist = (cury - righty) - (curx - rightx) * s;

    // 将垂距信息更新到 Global 内存中作为输出。
    cst.attachedData[idx] = dist;

    // 当垂距为负值时，在负数标记数组中标记之，因为这样的点将在下一轮迭代的时候
    // 删除，以加快处理速度。
    negdistflag[idx] = ((dist < 1.0e-6f) ? 1 : 0);

    // 调试打印
#ifdef CH_DEBUG_KERNEL_PRINT
    printf("Kernel[updateDist]: (%3d, %3d) Dist %7.3f, "
           "Line: (%3d, %3d) - (%3d, %3d) Label %2d\n",
           curx, cury, dist, leftx, lefty, rightx, righty, label[idx]);
#endif

    // 清除对 Shared Memory 下标含义的定义，因为在其他的函数中不同的下标会有不
    // 同的含义。
#undef SIDX_BLK_CNT
#undef SIDX_BLK_LABEL_LOW
#undef SIDX_BLK_LABEL_UP
#undef SIDX_BLK_CONVEX    
}

// 成员方法：updateDist（更新坐标点集垂距）
__host__ int ConvexHull::updateDist(
        CoordiSet *cst, CoordiSet *convexcst, int label[],
        int cstcnt, int negdistflag[])
{
    // 检查输入坐标集，输出坐标集是否为空。
    if (convexcst == NULL || cst == NULL || label == NULL ||
        negdistflag == NULL)
        return NULL_POINTER;

    // 检查当前点的数量，小于等于 0 则无效数据。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 将输入坐标集拷贝到 device 端。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入输出坐标集拷贝到 device 端。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 坐标集的 CUDA 相关数据
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    CoordiSetCuda *convexcstCud = COORDISET_CUDA(convexcst);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量，以及所需要的 Shared 
    // Memory 的数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (cstcnt + blocksize - 1) / blocksize;
    size_t sharedsize = (3 + 2 * blocksize) * sizeof (int);

    // 调用更新点集的垂距信息的核函数，计算每个点的垂距，更新负垂距标志数组。
    _updateDistKer<<<gridsize, blocksize, sharedsize>>>(
            *cstCud, *convexcstCud, label, cstcnt, negdistflag);

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// 成员方法：updateDistCpu（更新坐标点集垂距）
__host__ int ConvexHull::updateDistCpu(
        CoordiSet *cst, CoordiSet *convexcst, int label[],
        int cstcnt, int negdistflag[])
{
    // 检查输入坐标集，输出坐标集是否为空。
    if (convexcst == NULL || cst == NULL || label == NULL ||
        negdistflag == NULL)
        return NULL_POINTER;

    // 检查当前点的数量，小于等于 0 则无效数据。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 将输入坐标集拷贝到 host 端。
    errcode = CoordiSetBasicOp::copyToHost(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入输出坐标集拷贝到 host 端。
    errcode = CoordiSetBasicOp::copyToHost(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 坐标集的 CUDA 相关数据
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    CoordiSetCuda *convexcstCud = COORDISET_CUDA(convexcst);

    // 初始化末位
    (*cstCud).attachedData[cstcnt - 1] = 0.0f;
    negdistflag[cstcnt - 1] = 0;

#ifdef CH_DEBUG_CPU_PRINT
    cout << "[updateDistCpu]id " << cstcnt - 1 << " dist is "<<
            (*cstCud).attachedData[cstcnt - 1] << endl;
    cout << "[updateDistCpu]id " << cstcnt - 1 << " negdistflag is " <<
            negdistflag[cstcnt - 1] << endl;
#endif

    // 本地变量
    int curx, cury;
    int leftx, lefty;
    int rightx, righty;
    float s, dist;
    int id;

#ifdef CH_DEBUG_CPU_PRINT
    cout << "[updateDistCpu]update cstcnt is " << cstcnt << endl;
#endif

    // 计算每个点对应的垂距
    for (id = 0; id  < cstcnt - 1; id++) {
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[updateDistCpu]update dist id " << id << endl;
#endif
        // 读取当前点坐标
        curx = (*cstCud).tplMeta.tplData[2 * id];
        cury = (*cstCud).tplMeta.tplData[2 * id + 1];
    
        // 记录当前区域的最左点最右点
        if (id == 0 || label[id] != label[id - 1]) {
            leftx = (*convexcstCud).tplMeta.tplData[2 * label[id]];
            lefty = (*convexcstCud).tplMeta.tplData[2 * label[id] + 1];
            rightx = (*convexcstCud).tplMeta.tplData[2 * (label[id] + 1)];
            righty = (*convexcstCud).tplMeta.tplData[2 * (label[id] + 1) + 1];
#ifdef CH_DEBUG_CPU_PRINT
            cout << "[updateDistCpu]leftest x is " << leftx << endl;
            cout << "[updateDistCpu]leftest y is " << lefty << endl; 
            cout << "[updateDistCpu]rightest x is " << rightx << endl; 
            cout << "[updateDistCpu]rightest x is " << righty << endl;
#endif   
        }

        // 如果当前点就是凸壳点，那么不需要计算直接赋值退出就可以了。
        if ((curx == leftx && cury == lefty) || 
            (curx == rightx && cury == righty)) {
            (*cstCud).attachedData[id] = 0.0f;
            negdistflag[id] = 0;
#ifdef CH_DEBUG_CPU_PRINT
            cout << "[updateDistCpu]id " << id << " dist is "<<
                    (*cstCud).attachedData[id] << endl;
            cout << "[updateDistCpu]id " << id << " negdistflag is " <<
                    negdistflag[id] << endl;
#endif
        // 计算垂距，该计算通过最左点和最右点形成的直线作为垂距求解的依据，但实
        // 际求解过程中并不是真正的垂距，而是垂直于 y 轴的距离。当点在直线之下
        // 时，具有正垂距，当点在直线之上时，具有负垂距。
        // y ^        right
        //   |          +
        //   |         /
        //   |        / --
        //   |       /|  ^
        //   |      / |  | dist
        //   |     /  |  v
        //   |    /   * -- 
        //   |   +   cur
        //  O| left           x
        // --+----------------->
        //   |
        } else {
            s = (float)(righty - lefty) / (rightx - leftx);
            dist = (cury - righty) - (curx - rightx) * s;

            // 将垂距信息更新到 输出。
            (*cstCud).attachedData[id] = dist;

            // 当垂距为负值时，在负数标记数组中标记之，因为这样的点将在下一轮
            // 迭代的时候删除，以加快处理速度。
            negdistflag[id] = ((dist < 1.0e-6f) ? 1 : 0);
#ifdef CH_DEBUG_CPU_PRINT
            cout << "[updateDistCpu]id " << id << " dist is "<<
                    (*cstCud).attachedData[id] << endl;
            cout << "[updateDistCpu]id " << id << " negdistflag is " <<
                    negdistflag[id] << endl;
#endif
        }
    }

    return NO_ERROR;
}

// Kernel 函数: _updateFoundInfoKer（更新新发现凸壳点信息）
static __global__ void _updateFoundInfoKer(
        int *label, float *dist, int *maxdistidx, int cstcnt,
        int *foundflag, int *startidx)
{
    // 共享内存，用来存放当前 Block 处理的 LABEL 值，其长度为 BlockDim + 1，因
    // 为需要加载下一 Blcok 的第一个 LABEL 值。
    extern __shared__ int labelShd[];

    // 基准索引。表示当前 Block 的起始位置索引
    int baseidx = blockIdx.x * blockDim.x;

    // 全局索引。
    int idx = baseidx + threadIdx.x;

    // 初始化 Shared Memory，将当前 Block 所对应的坐标点的 LABEL 值赋值给 
    // Shared Memroy，为了程序健壮性的考虑，我们将处理越界数据的那些 Thread 所
    // 对应的 LABEL 值赋值为最后一个点的 LABEL 值。
    if (idx < cstcnt)
        labelShd[threadIdx.x] = label[idx];
    else
        labelShd[threadIdx.x] = label[cstcnt - 1];

    // 使用每个 Block 中第 0 个 Thread 来初始化多出来的那个 LABEL 值，初始化的
    // 规则同上面的规则一样，也做了健壮性的考量。
    if (threadIdx.x == 0) {
        if (baseidx + blockDim.x < cstcnt)
            labelShd[blockDim.x] = label[baseidx + blockDim.x];
        else
            labelShd[blockDim.x] = label[cstcnt - 1];

        // 如果是第一块的话，起始索引更新。
        if (blockIdx.x == 0)
            startidx[0] = 0;
    }
    
    // 块内的线程同步
    __syncthreads();

    // 对于处理越界数据的 Thread 直接进行返回操作，不进行任何处理。
    if (idx >= cstcnt)
        return;

    // 当前 Thread 处理坐标点的 LABEL 值。
    int curlabel = labelShd[threadIdx.x];

    // 对于单独处于一个 LABEL 区域的最后一个点，该点不需要做任何查找操作，直接
    // 赋值为未找到新的凸壳点。
    if (idx == cstcnt - 1) {
        foundflag[curlabel] = 0;
        return;
    }
    
    // 本函数只针对处于 LABEL 区域边界的点进行处理，对于不处于区域边界的点则直
    // 接返回。
    if (curlabel == labelShd[threadIdx.x + 1])
        return;

    // 读取当前 LABEL 区域的最大垂距和最大垂距所对应的下标和该最大垂距的值。
    int curmaxdistidx = maxdistidx[idx]; 
    float curmaxdist = dist[curmaxdistidx];
           
    // 如果当前 LABEL 区域的最大垂距点的垂距值大于 0，则说明了在当前的 LABEL 区
    // 域内发现了凸壳点。为了健壮性的考虑，这里将 0 写为 1.0e-6。
    foundflag[curlabel] = (curmaxdist > 1.0e-6f) ? 1 : 0;

    // 更新下一个 LABEL 区域的起始下标。由于当前 Thread 是当前 LABEL 区域的最后
    // 一个，因此下一个 LABEL 区域的起始下标为当前 Thread 全局索引加 1。
    startidx[curlabel + 1] = idx + 1;

    // 调试打印
#ifdef CH_DEBUG_KERNEL_PRINT
    printf("Kernel[FoundInfo]: Label %2d - Found %1d (%7.3f at %3d), "
           "End Idx %3d\n",
           curlabel, foundflag[curlabel], curmaxdist, curmaxdistidx, idx);
#endif
}
  
// 成员方法: updateFoundInfo（更新新发现凸壳点信息）
__host__ int ConvexHull::updateFoundInfo(
        int label[], float dist[], int maxdistidx[],
        int cstcnt, int foundflag[], int startidx[])
{
    // 检查所有的输入指针或数组是否为 NULL，如果存在一个为 NULL 则报错退出。
    if (label == NULL || dist == NULL || maxdistidx == NULL ||
        foundflag == NULL || startidx == NULL)
        return NULL_POINTER;

    // 检查坐标点的数量是否小于等于 0，若是则报错推出。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量，以及所需要的 Shared 
    // Memory 的数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (cstcnt + blocksize - 1) / blocksize;
    size_t sharedsize = (blocksize + 1) * sizeof (int);

    // 调用 Kernel 函数，完成计算。
    _updateFoundInfoKer<<<gridsize, blocksize, sharedsize>>>(
            label, dist, maxdistidx, cstcnt, foundflag, startidx);

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// 成员方法: updateFoundInfoCpu（更新新发现凸壳点信息）
__host__ int ConvexHull::updateFoundInfoCpu(
        int label[], float dist[], int maxdistidx[],
        int cstcnt, int foundflag[], int startidx[])
{
    // 检查所有的输入指针或数组是否为 NULL，如果存在一个为 NULL 则报错退出。
    if (label == NULL || dist == NULL || maxdistidx == NULL ||
        foundflag == NULL || startidx == NULL)
        return NULL_POINTER;

    // 检查坐标点的数量是否小于等于 0，若是则报错推出。
    if (cstcnt <= 0)
        return INVALID_DATA;

    int id;
    // 如果是首末位，起始索引更新。
    startidx[0] = 0;

    // 对于单独处于一个 LABEL 区域的最后一个点，该点不需要做任何查找操作，直接
    // 赋值为未找到新的凸壳点。
    foundflag[label[cstcnt - 1]] = 0;

#ifdef CH_DEBUG_CPU_PRINT
    cout << "[updateFoundInfoCpu]label " << label[cstcnt - 1] << " found " <<
            foundflag[label[cstcnt - 1]] << endl;
    cout << "[updateFoundInfoCpu]startidx " << label[0] << " is " <<
            startidx[label[0]] << endl;
#endif       

    // 循环，更新新发现凸壳点信息，不处理第一个点
    for (id = 1; id < cstcnt; id++) {
        // 处理新区域
        if (label[id] != label[id - 1]) {
#ifdef CH_DEBUG_CPU_PRINT
            cout << "[updateFoundInfoCpu]label different " << endl;
#endif
            // 记录新区域的起始位置
            startidx[label[id]] = id;
            
            // 如果当前 LABEL 区域的最大垂距点的垂距值大于 0，则说明了在当前的
            // LABEL 区域内发现了凸壳点。为了健壮性的考虑，这里将 0 写为 1.0e-6
            foundflag[label[id - 1]] =
                    (dist[maxdistidx[id - 1]] > 1.0e-6f) ? 1 : 0;
#ifdef CH_DEBUG_CPU_PRINT
            cout << "[updateFoundInfoCpu]label " << label[id - 1] <<
                    " found " << foundflag[label[id - 1]] << endl;
            cout << "[updateFoundInfoCpu]startidx " << label[id] << " is " <<
                    startidx[label[id]] << endl;
#endif
        } 
    }

    // 处理完毕退出。
    return NO_ERROR;
}

// Kernel 函数: _updateConvexCstKer（生成新凸壳点集）
static __global__ void _updateConvexCstKer(
        CoordiSetCuda cst, CoordiSetCuda convexcst, int foundflag[],
        int foundacc[], int startidx[], int maxdistidx[], int convexcnt,
        CoordiSetCuda newconvexcst)
{
    // 计算当前 Thread 的全局索引。本 Kernel 中，每个线程都对应于一个 LABEL 区
    // 域，对于发现了新凸壳点的 LABEL 区域，则需要将原来这个 LABEL 区域内的凸壳
    // 点和新发现的凸壳点同时拷贝到新的凸壳点集中。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果该 Thread 对应的时越界数据，则直接返回，不进行任何处理。
    if (idx >= convexcnt)
        return;

    // 计算原来的凸壳点在新凸壳点集中的下标，由于前面的 LABEL 区域共产生了 
    // foundacc[idx] 个凸壳点，因此，下标应相较于原来的下标（idx）增加了相应的
    // 数量。
    int newidx = idx + foundacc[idx];

    // 将这个凸壳点的坐标从原来的凸壳点集中拷贝到新的凸壳点集中。
    newconvexcst.tplMeta.tplData[2 * newidx] =
            convexcst.tplMeta.tplData[2 * idx];
    newconvexcst.tplMeta.tplData[2 * newidx + 1] =
            convexcst.tplMeta.tplData[2 * idx + 1];

    // 调试打印
#ifdef CH_DEBUG_KERNEL_PRINT
    printf("Kernel[UpdateConvex]: Add Old (%3d, %3d) - "
           "%3d => %3d, Label %2d\n",
           convexcst.tplMeta.tplData[2 * idx], 
           convexcst.tplMeta.tplData[2 * idx + 1],
           idx, newidx, idx);
#endif

    // 如果当前 LABEL 区域中没有发现新的凸壳点，则只需要拷贝原有的凸壳点到新的
    // 凸壳点集中就可以了，不需要再进行后面的操作。
    if (foundflag[idx] == 0)
        return;

    // 计算新发现的凸壳点在凸壳点集中的下标（就把它放在原来凸壳点集的后面）和该
    // 凸壳点对应的坐标点集中的下标（就是该 LABEL 区域最大垂距点的下标）。由于
    // 最大垂距点下标数组是记录的 Scanning 操作的结果，因此正确的结果存放再该 
    // LABEL 区域最后一个下标处。
    newidx++;
    int cstidx = maxdistidx[startidx[idx + 1] - 1];

    // 将新发现的凸壳点从坐标点集中拷贝到新的凸壳点集中。
    newconvexcst.tplMeta.tplData[2 * newidx] =
            cst.tplMeta.tplData[2 * cstidx];
    newconvexcst.tplMeta.tplData[2 * newidx + 1] =
            cst.tplMeta.tplData[2 * cstidx + 1];

    // 调试打印
#ifdef CH_DEBUG_KERNEL_PRINT
    printf("Kernel[UpdateConvex]: Add New (%3d, %3d) - "
           "%3d => %3d, Label %2d\n",
           cst.tplMeta.tplData[2 * cstidx], 
           cst.tplMeta.tplData[2 * cstidx + 1],
           cstidx, newidx, idx);
#endif
}

// Host 成员方法：updateConvexCst（生成新凸壳点集）
__host__ int ConvexHull::updateConvexCst(
        CoordiSet *cst, CoordiSet *convexcst, int foundflag[],
        int foundacc[], int startidx[], int maxdistidx[], int convexcnt,
        CoordiSet *newconvexcst)
{
    // 检查参数中所有的指针和数组是否为空。
    if (cst == NULL || convexcst == NULL || foundacc == NULL ||
        foundflag == NULL || startidx == NULL || maxdistidx == NULL ||
        newconvexcst == NULL)
        return NULL_POINTER;

    // 检查当前凸壳点的数量，小于等于 0 则无效数据。
    if (convexcnt <= 0)
        return INVALID_DATA;

    int errcode;

    // 将输入坐标集拷贝到当前 Device。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入凸壳点集拷贝到当前 Device。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出的新的凸壳集拷贝到当前 Device 端。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(newconvexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 获取各个坐标集的 CUDA 型数据
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    CoordiSetCuda *convexcstCud = COORDISET_CUDA(convexcst);
    CoordiSetCuda *newconvexcstCud = COORDISET_CUDA(newconvexcst);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    // 矩阵方法分段扫描版本线程块大小。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (convexcnt + blocksize - 1) / blocksize;

    // 调用 Kernel 函数完成计算。
    _updateConvexCstKer<<<gridsize, blocksize>>>(
            *cstCud, *convexcstCud, foundflag, foundacc, startidx,
            maxdistidx, convexcnt, *newconvexcstCud);

    // 判断 Kernel 函数是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// Host 成员方法：updateConvexCstCpu（生成新凸壳点集）
__host__ int ConvexHull::updateConvexCstCpu(
        CoordiSet *cst, CoordiSet *convexcst, int foundflag[],
        int foundacc[], int startidx[], int maxdistidx[], int convexcnt,
        CoordiSet *newconvexcst)
{
    // 检查参数中所有的指针和数组是否为空。
    if (cst == NULL || convexcst == NULL || foundacc == NULL ||
        foundflag == NULL || startidx == NULL || maxdistidx == NULL ||
        newconvexcst == NULL)
        return NULL_POINTER;

    // 检查当前凸壳点的数量，小于等于 0 则无效数据。
    if (convexcnt <= 0)
        return INVALID_DATA;

    // 错误码
    int errcode;

    // 将输入坐标集拷贝到 Host。
    errcode = CoordiSetBasicOp::copyToHost(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入凸壳点集拷贝到当前 Host。
    errcode = CoordiSetBasicOp::copyToHost(convexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出的新的凸壳集拷贝到当前 Host。
    errcode = CoordiSetBasicOp::copyToHost(newconvexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 获取各个坐标集的 CUDA 型数据
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    CoordiSetCuda *convexcstCud = COORDISET_CUDA(convexcst);
    CoordiSetCuda *newconvexcstCud = COORDISET_CUDA(newconvexcst);

    // 本地变量
    int newid;
    int cstidx;

    // 循环处理凸壳点集
    for (int id = 0; id < convexcnt; id++) {
        // 计算原来的凸壳点在新凸壳点集中的下标，由于前面的 LABEL 区域共产生了 
        // foundacc[idx] 个凸壳点，因此，下标应相较于原来的下标（idx）增加了相
        // 应的数量。
        newid = id + foundacc[id];

        // 将这个凸壳点的坐标从原来的凸壳点集中拷贝到新的凸壳点集中。
        (*newconvexcstCud).tplMeta.tplData[2 * newid] =
                (*convexcstCud).tplMeta.tplData[2 * id];
        (*newconvexcstCud).tplMeta.tplData[2 * newid + 1] =
                (*convexcstCud).tplMeta.tplData[2 * id + 1];

#ifdef CH_DEBUG_CPU_PRINT
        printf("[updateConvexCstCpu]: Add Old (%3d, %3d) - "
               "%3d => %3d\n",
               (*convexcstCud).tplMeta.tplData[2 * id], 
               (*convexcstCud).tplMeta.tplData[2 * id + 1],
               id, newid);
#endif
        
        // 计算新发现的凸壳点在凸壳点集中的下标（就把它放在原来凸壳点集的后面）
        // 和该凸壳点对应的坐标点集中的下标（就是该 LABEL 区域最大垂距点的下标）
        // 由于最大垂距点下标数组是记录的 Scanning 操作的结果，因此正确的结果存
        // 放再该 LABEL 区域最后一个下标处。
        if (foundflag[id]) {
            newid++;
            cstidx = maxdistidx[startidx[id + 1] - 1];
        

            // 将新发现的凸壳点从坐标点集中拷贝到新的凸壳点集中。
            (*newconvexcstCud).tplMeta.tplData[2 * newid] =
                    (*cstCud).tplMeta.tplData[2 * cstidx];
            (*newconvexcstCud).tplMeta.tplData[2 * newid + 1] =
                    (*cstCud).tplMeta.tplData[2 * cstidx + 1];
#ifdef CH_DEBUG_CPU_PRINT
            printf("[updateConvexCstCpu]: Add New (%3d, %3d) - "
                   "%3d => %3d\n",
                   (*cstCud).tplMeta.tplData[2 * cstidx], 
                   (*cstCud).tplMeta.tplData[2 * cstidx + 1],
                   cstidx, newid);
#endif
        }
    }

    // 处理完毕退出。
    return NO_ERROR;
}

// Kernel 函数: _markLeftPointsKer（标记左侧点）
static __global__ void _markLeftPointsKer(
        CoordiSetCuda cst, CoordiSetCuda newconvexcst, int negdistflag[], 
        int label[], int foundflag[], int foundacc[], int cstcnt, 
        int leftflag[])
{
    // 记录了本 Kernel 所使用到的共享内存中各个下标所存储的数据的含义。其中，
    // SIDX_BLK_LABEL_LOW 和 SIDX_BLK_LABEL_UP 用来存当前 Block 中所加载的点集
    // 的区域标签值的上下界。根据这个上下界，可以计算出当前点所在区域的最左最右
    // 点，从而根据这两点确定的直线计算当前点的垂距。
    // 从下标为 SIDX_BLK_CONVEX_X 开始的其后的所有共享内存空间存储了当前 Block 
    // 所处理的所有的新的凸壳点的 X 坐标。
#define SIDX_BLK_LABEL_LOW  0
#define SIDX_BLK_LABEL_UP   1
#define SIDX_BLK_CONVEX_X   2

    // 共享内存的声明。
    extern __shared__ int shdmem[]; 

    // 基准下标。表示当前 Block 第一个 Thread 所处理的下标。
    int baseidx = blockIdx.x * blockDim.x;

    // 当前 Thread 的全局下标。
    int idx = baseidx + threadIdx.x;

    // 初始化共享内存中的公共数据，为了防止写入冲突，这里只使用每个 Block 的第
    // 一个 Thread 处理初始化工作。
    if (threadIdx.x == 0) {
        // 读取当前 Block 所处理的所有坐标点中最小的 LABEL 值。
        shdmem[SIDX_BLK_LABEL_LOW] = label[baseidx];

        // 计算当前 Block 所处理的所有坐标点中最大的 LABEL 值。
        if (baseidx + blockDim.x <= cstcnt)
            shdmem[SIDX_BLK_LABEL_UP] = label[baseidx + blockDim.x - 1];
        else
            shdmem[SIDX_BLK_LABEL_UP] = label[cstcnt - 1];
    }

    // 同步 Block 内的所有 Thread，使得上述初始化对所有的 Thread 都可见。
    __syncthreads();

    // 从 Shared Memory 中读取当前 Block 所处理的 LABEL 值范围。这一步骤没有实
    // 际的逻辑含义，将数据从共享内存搬入寄存器仅仅是为了加快处理速度。
    int labellower = shdmem[SIDX_BLK_LABEL_LOW];
    int labelupper = shdmem[SIDX_BLK_LABEL_UP];

    // 定义中心点（即新增加的凸壳点）的横坐标的哑值。这是由于并不是所有的 LABEL
    // 区域都会在该论迭代中发现新的凸壳点。该值要求非常的大，因为没有发现新凸壳
    // 点的区域，相当于所有的坐标点放在左侧。
#define LP_DUMMY_CVXX  CH_LARGE_ENOUGH

    // 将新凸壳点的 X 坐标存储 Shared Memory 提取出，用一个指针来表示，这样的写
    // 法是为了代码更加易于理解。
    int *newcvxxShd = &shdmem[SIDX_BLK_CONVEX_X];

    // 在 Shared Memory 中初始化新凸壳点（中心点）的 X 坐标。
    if (threadIdx.x < labelupper - labellower + 1) {
        // 计算新凸壳点在新的凸壳点集中的下标。
        int labelidx = threadIdx.x + labellower;
        int newconvexidx = labelidx + foundacc[labelidx] + 1;

        // 初始化 Shared Memory 中的数据，对于没有产生新的凸壳点的 LABEL 区域来
        // 说，该值直接赋哑值。
        if (foundflag[labelidx])
            newcvxxShd[threadIdx.x] =
                    newconvexcst.tplMeta.tplData[2 * newconvexidx];
        else
            newcvxxShd[threadIdx.x] = LP_DUMMY_CVXX;
    }

    // 同步 Block 内的所有 Thread，是的上述所有初始化计算对所有 Thread 可见。
    __syncthreads();

    // 如果当前 Thread 处理的是越界范围，则直接返回不进行任何处理。
    if (idx >= cstcnt)
        return;

    // 读取当前坐标点所对应的 LABEL 值（经过校正的，表示 Shared Memory 中的下
    // 标）。
    int curlabel = label[idx] - labellower;

    // 读取当前坐标点的 x 坐标和该点的垂距值。
    int curx = cst.tplMeta.tplData[2 * idx];
    int curnegflag = negdistflag[idx];

    // 对于所有垂距大于等于 0，且 x 坐标小于中心点坐标时认为该点在中心点左侧。
    // （因为所有垂距小于 0 的点将在下一轮迭代中被排除，因此，这里没有将垂距小
    // 于 0 的点设置左侧标志位）
    if (curx < newcvxxShd[curlabel] && curnegflag == 0)
        leftflag[idx] = 1;
    else
        leftflag[idx] = 0;

    // 调试打印
#ifdef CH_DEBUG_KERNEL_PRINT
    printf("Kernel[LeftPoint]: (%3d, %3d) d=%8.3f, "
           "Label %2d ( NC.x %3d ) Left %1d\n",
           cst.tplMeta.tplData[2 * idx], cst.tplMeta.tplData[2 * idx + 1],
           cst.attachedData[idx], curlabel + labellower, 
           newcvxxShd[curlabel], leftflag[idx]);
#endif

    // 清除函数内部的宏定义，防止同后面的函数造成冲突。
#undef LP_TMPX_DUMMY 
#undef SIDX_BLK_LABEL_LOW
#undef SIDX_BLK_LABEL_UP
#undef SIDX_BLK_CONVEX_X
}

// Host 成员方法：markLeftPoints（标记左侧点）
__host__ int ConvexHull::markLeftPoints(
        CoordiSet *cst, CoordiSet *newconvexcst, int negdistflag[], 
        int label[], int foundflag[], int foundacc[], int cstcnt, 
        int leftflag[])
{
    // 检查参数中所有的指针和变量是否为空。
    if (cst == NULL || newconvexcst == NULL || label == NULL ||
        foundacc == NULL || foundflag == NULL || leftflag == NULL)
        return NULL_POINTER;

    // 检查当前点的数量，小于等于 0 则无效数据。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 将输入坐标点集拷贝到当前 Device。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将新的凸壳点集拷贝到当前 Device。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(newconvexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 获取坐标点集和凸壳点集的 CUDA 相关数据
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    CoordiSetCuda *newconvexcstCud = COORDISET_CUDA(newconvexcst);

    // 计算 Kernel 函数所需要的 Block 尺寸和数量，以及每个 Block 所使用的 
    // Shared Memory 的数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (cstcnt + blocksize - 1) / blocksize;
    size_t sharedsize = (2 + blocksize) * sizeof (int);

    // 调用 Kernel 函数，完成计算。
    _markLeftPointsKer<<<gridsize, blocksize, sharedsize>>>(
            *cstCud, *newconvexcstCud, negdistflag, label, 
            foundflag, foundacc, cstcnt, leftflag);

    // 判断 Kernel 函数运行是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// Host 成员方法：markLeftPointsCpu（标记左侧点）
__host__ int ConvexHull::markLeftPointsCpu(
        CoordiSet *cst, CoordiSet *newconvexcst, int negdistflag[], 
        int label[], int foundflag[], int foundacc[], int cstcnt, 
        int leftflag[])
{
    // 检查参数中所有的指针和变量是否为空。
    if (cst == NULL || newconvexcst == NULL || label == NULL ||
        foundacc == NULL || foundflag == NULL || leftflag == NULL)
        return NULL_POINTER;

    // 检查当前点的数量，小于等于 0 则无效数据。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 将输入坐标点集拷贝到 Host。
    errcode = CoordiSetBasicOp::copyToHost(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将新的凸壳点集拷贝到当前 Host。
    errcode = CoordiSetBasicOp::copyToHost(newconvexcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 获取坐标点集和凸壳点集的 CUDA 相关数据
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    CoordiSetCuda *newconvexcstCud = COORDISET_CUDA(newconvexcst);

    // 本地变量
    int newconvexcstx;
    int newconvexidx;

    // 定义中心点（即新增加的凸壳点）的横坐标的哑值。这是由于并不是所有的 LABEL
    // 区域都会在该论迭代中发现新的凸壳点。该值要求非常的大，因为没有发现新凸壳
    // 点的区域，相当于所有的坐标点放在左侧。
#define LP_DUMMY_CVXX_CPU  CH_LARGE_ENOUGH

    for (int id = 0; id < cstcnt; id++) {
        // 计算新凸壳点在新的凸壳点集中的下标。
        newconvexidx = label[id] + foundacc[label[id]] + 1;

        // 初始化新凸壳点 x 坐标，对于没有产生新的凸壳点的 LABEL 区域来
        // 说，该值直接赋哑值。
        if (foundflag[label[id]])
            newconvexcstx =
                    (*newconvexcstCud).tplMeta.tplData[2 * newconvexidx];
        else
            newconvexcstx = LP_DUMMY_CVXX_CPU;

        // 对于所有垂距大于等于 0，且 x 坐标小于中心点坐标时认为该点在中心点左侧。
        // （因为所有垂距小于 0 的点将在下一轮迭代中被排除，因此，这里没有将垂距小
        // 于 0 的点设置左侧标志位）
        if ((*cstCud).tplMeta.tplData[2 * id] < newconvexcstx &&
            negdistflag[id] == 0)
            leftflag[id] = 1;
        else
            leftflag[id] = 0;

#ifdef CH_DEBUG_CPU_PRINT
        printf("[markLeftPointsCpu]: (%3d, %3d) d=%8.3f, "
               "Label %2d ( NC.x %3d ) Left %1d\n",
               (*cstCud).tplMeta.tplData[2 * id],
               (*cstCud).tplMeta.tplData[2 * id + 1],
               (*cstCud).attachedData[id], label[id], 
               newconvexidx, leftflag[id]);
#endif
    }

    // 处理完毕退出。
    return NO_ERROR;
}

// Kernel 函数: _updatePropertyKer（计算新下标）
static __global__ void _updatePropertyKer(
        int leftflag[], int leftacc[], int negdistflag[], int negdistacc[],
        int startidx[], int label[], int foundacc[], int cstcnt,
        int newidx[], int tmplabel[])
{
    // 记录了本 Kernel 所使用到的共享内存中各个下标所存储的数据的含义。其中，
    // SIDX_BLK_LABEL_LOW 和 SIDX_BLK_LABEL_UP 用来存当前 Block 中所加载的点集
    // 的区域标签值的上下界。根据这个上下界，可以计算出当前点所在区域的最左最右
    // 点，从而根据这两点确定的直线计算当前点的垂距。
    // SIDX_BLK_START_IDX 表示当前 Block 所囊括的所有 LABEL 区域对应的起始下
    // 标。
    // SIDX_BLK_NEG_ACC 表示当前 Block 所囊括的所有 LABEL 区域对应的负垂距累加
    // 值，该值用于计算坐标点在下一轮迭代中所对应的新下标值。
    // SIDX_BLK_LEFT_ACC 表示当前 Block 所囊括的所有 LABEL 区域对应的左侧点累加
    // 值，该值用于计算坐标点在下一轮迭代中所对应的新下标值。
    // SIDX_BLK_NEG_ACC 表示当前 Block 所囊括的所有 LABEL 区域对应的新发现凸壳
    // 点的累加值。该值用来计算本轮结束后目前所有找到的凸壳点在新凸壳点集中的下
    // 标值。
#define SIDX_BLK_LABEL_LOW   0
#define SIDX_BLK_LABEL_UP    1
#define SIDX_BLK_START_IDX  (2 + 0 * blockDim.x)
#define SIDX_BLK_NEG_ACC    (2 + 1 * blockDim.x)
#define SIDX_BLK_LEFT_ACC   (2 + 2 * blockDim.x)
#define SIDX_BLK_FOUND_ACC  (2 + 3 * blockDim.x)

    // 共享内存的声明。
    extern __shared__ int shdmem[];

    // 基准下标。当前 Block 所有线程的起始全局下标。
    int baseidx = blockIdx.x * blockDim.x;

    // 当前 Thread 的全局下标
    int idx = baseidx + threadIdx.x;

    // 初始化 Shared Memory 公用部分。只需要一个线程来做这件事情即可。
    if (threadIdx.x == 0) {
        // 计算当前 Block 处理的最小的 LABEL 区域。
        shdmem[SIDX_BLK_LABEL_LOW] = label[baseidx];

        // 计算当前 Block 处理的最大的 LABEL 区域。这里针对最后一个 Block，需要
        // 考虑越界读取数据的情况。
        if (baseidx + blockDim.x < cstcnt)
            shdmem[SIDX_BLK_LABEL_UP] = label[baseidx + blockDim.x - 1] + 1;
        else
            shdmem[SIDX_BLK_LABEL_UP] = label[cstcnt - 1];
    }

    // 针对上面的初始化进行同步，使其结果对所有 Thread 可见。
    __syncthreads();

    // 将共享内存的各个数组指针取出。这一步骤没有逻辑上的实际意义，只是为了后续
    // 步骤表达方便。
    int *startidxShd = &shdmem[SIDX_BLK_START_IDX];
    int *negdistaccShd = &shdmem[SIDX_BLK_NEG_ACC];
    int *leftaccShd = &shdmem[SIDX_BLK_LEFT_ACC];
    int *foundaccShd = &shdmem[SIDX_BLK_FOUND_ACC];

    // 将存放于 Shared Memory 中的 LABEL 区域上下界转存到寄存器中。该步骤也没有
    // 实际的逻辑意义，目的在于使程序运行更加高效。
    int labellower = shdmem[SIDX_BLK_LABEL_LOW];
    int labelupper = shdmem[SIDX_BLK_LABEL_UP];

    // 初始化 Shared Memory 中的各个数组的值。
    if (threadIdx.x < labelupper - labellower + 1) {
        // 从 Global Memory 中读取各个 LABEL 的起始下标。
        startidxShd[threadIdx.x] = startidx[threadIdx.x + labellower];

        // 根据起始下标，计算各个 LABEL 区域所对应的负垂距和左侧点累加值。
        negdistaccShd[threadIdx.x] = negdistacc[startidxShd[threadIdx.x]];
        leftaccShd[threadIdx.x] = leftacc[startidxShd[threadIdx.x]];

        // 从 Global Memory 中读取新凸壳点的累加值。
        foundaccShd[threadIdx.x] = foundacc[threadIdx.x + labellower];
    }

    // 针对上面的初始化进行 Block 内部的同步，是的这些初始化结果对所有的 
    // Thread 可见。
    __syncthreads();

    // 若当前 Thread 处理的是越界数据，则直接退出。
    if (idx >= cstcnt)
        return;
 
    // 若当前 Thread 处理的坐标点具有负垂距，则直接退出，因为负垂距坐标点再下一
    // 轮迭代的过程中则不再使用。
    if (negdistflag[idx] == 1)
        return;

    // 读取当前坐标点的 LABEL 值，由于后面只使用 Shared Memory 中的数据，因此这
    // 里直接将其转换为 Shared Memory 所对应的下标。
    int curlabel = label[idx] - labellower;

    // 宏：CHNI_ENABLE_FAST_CALC（下标值快速计算开关）
    // 在下面的代码中新下标值的计算可分为快速和普通两种方式。如果开启该定义，则
    // 使用快速下标值计算；如果关闭，则是用普通下标值计算。无论是快速计算还是普
    // 通计算，两者在计算公式上归根结底是一样的，只是为了减少计算量，某些变量被
    // 合并同类项后消掉了。因此快速下标值计算的公式不易理解，普通计算的公式易于
    // 理解。快速计算仅仅是普通计算的推导结果。
#define CHNI_ENABLE_FAST_CALC

    // 针对当前在新发现的凸壳点的左侧还有右侧，需要进行不同的计算公式来确定其在
    // 下一轮迭代中的 LABEL 值和在坐标点集中的下标值。
    if (leftflag[idx] == 1) {
        // 对于当前点在新的凸壳点的左侧，计算新的 LABEL 值。这里 foundacc 的物
        // 理含义是当前 LABEL 值之前的各个 LABEL 区域中总共找到的新增凸壳点的数
        // 量。
        tmplabel[idx] = label[idx] + foundaccShd[curlabel];

        // 对于当前点在新的凸壳点的左侧，计算坐标点在新一轮迭代中的下标值。这里
        // 首先确定当前 LABEL 的新的起始下标值，然后再加上该点在其 LABEL 区域内
        // 其前面的左侧点的数量，就得到了其新的下标值。
#ifndef CHNI_ENABLE_FAST_CALC
        int basenewidx = startidxShd[curlabel] - negdistaccShd[curlabel];
        int innernewidx = leftacc[idx] - leftaccShd[curlabel];
        newidx[idx] = basenewidx + innernewidx;
#else
        newidx[idx] = startidxShd[curlabel] - negdistaccShd[curlabel] +
                      leftacc[idx] - leftaccShd[curlabel];
#endif
    } else {
        // 对于当前点在新的凸壳点的右侧，计算新的 LABEL 值。这里 foundacc 的物
        // 理含义是当前 LABEL 值之前的各个 LABEL 区域中总共找到的新增凸壳点的数
        // 量。
        tmplabel[idx] = label[idx] + foundaccShd[curlabel] + 1; 

        // 对于当前点在新的凸壳点的右侧，计算坐标点在新一轮迭代中的下标值。计算
        // 该值，首先计算右侧构成的新的 LABEL 区域的起始位置，这部分只需要在左
        // 侧起始下标处加上当前 LABEL 区域总共检出的左侧坐标的数量即可；之后，
        // 需要计算该坐标点在新的区域内部的偏移量，即内部的原来下标值，减去其前
        // 面的负垂距点数量和左侧点数量。
#ifndef CHNI_ENABLE_FAST_CALC
        int leftcnt = leftaccShd[curlabel + 1] - leftaccShd[curlabel];
        int basenewidx = startidxShd[curlabel] - negdistaccShd[curlabel] + 
                         leftcnt;
        int inidx = idx - startidxShd[curlabel];
        int innegacc = negdistacc[idx] - negdistaccShd[curlabel];
        int inleftacc = leftacc[idx] - leftaccShd[curlabel];
        int innernewidx = inidx - innegacc - inleftacc;
        newidx[idx] = basenewidx + innernewidx;
#else
        newidx[idx] = idx - negdistacc[idx] +
                      leftaccShd[curlabel + 1] - leftacc[idx];
#endif
    }

    // 调试打印
#ifdef CH_DEBUG_KERNEL_PRINT
    printf("Kernel[NewLabel]: Label %2d => %2d, "
           "Idx %2d => %2d, Left %1d\n",
           label[idx], tmplabel[idx], idx, newidx[idx], leftflag[idx]);
#endif

    // 消除本 Kernel 函数内部的宏定义，防止后面的函数使用造成冲突。
#ifdef CHNI_ENABLE_FAST_CALC
#  undef CHNI_ENABLE_FAST_CALC
#endif

#undef SIDX_BLK_LABEL_LOW
#undef SIDX_BLK_LABEL_UP
#undef SIDX_BLK_START_IDX
#undef SIDX_BLK_NEG_ACC
#undef SIDX_BLK_LEFT_ACC
#undef SIDX_BLK_FOUND_ACC
}

// 成员方法：updateProperty（计算新下标）
__host__ int ConvexHull::updateProperty(
        int leftflag[], int leftacc[], int negdistflag[], int negdistacc[], 
        int startidx[], int label[], int foundacc[], int cstcnt, 
        int newidx[], int newlabel[])
{
    // 检查所有参数中的指针和数组，是否为 NULL。
    if (leftflag == NULL || leftacc == NULL || negdistflag == NULL ||
        negdistacc == NULL || startidx == NULL || label == NULL ||
        foundacc == NULL || newidx == NULL || newlabel == NULL)
        return NULL_POINTER;

    // 如果坐标点集的数量小于等于 0，则报错退出。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 计算调用 Kernel 函数所需要的 Grid 和 Block 尺寸，以及每个 Block 所使用的
    // Shared Memory 的字节数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (cstcnt + blocksize - 1) / blocksize;
    size_t sharedsize = (2 + blocksize * 4) * sizeof (int);

    // 调用 Kernel 函数, 完成计算。
    _updatePropertyKer<<<gridsize, blocksize, sharedsize>>>(
            leftflag, leftacc, negdistflag, negdistacc, startidx,
            label, foundacc, cstcnt, newidx, newlabel);

    // 判断 Kernel 函数的执行是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// 成员方法：updatePropertyCpu（计算新下标）
__host__ int ConvexHull::updatePropertyCpu(
        int leftflag[], int leftacc[], int negdistflag[], int negdistacc[], 
        int startidx[], int label[], int foundacc[], int cstcnt, 
        int newidx[], int newlabel[])
{
    // 检查所有参数中的指针和数组，是否为 NULL。
    if (leftflag == NULL || leftacc == NULL || negdistflag == NULL ||
        negdistacc == NULL || startidx == NULL || label == NULL ||
        foundacc == NULL || newidx == NULL || newlabel == NULL)
        return NULL_POINTER;

    // 如果坐标点集的数量小于等于 0，则报错退出。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 宏：CHNI_ENABLE_FAST_CALC（下标值快速计算开关）
    // 在下面的代码中新下标值的计算可分为快速和普通两种方式。如果开启该定义，则
    // 使用快速下标值计算；如果关闭，则是用普通下标值计算。无论是快速计算还是普
    // 通计算，两者在计算公式上归根结底是一样的，只是为了减少计算量，某些变量被
    // 合并同类项后消掉了。因此快速下标值计算的公式不易理解，普通计算的公式易于
    // 理解。快速计算仅仅是普通计算的推导结果。
#define CHNI_ENABLE_FAST_CALC

    for (int idx = 0; idx < cstcnt; idx++) {
        if (negdistflag[idx] == 0) {
            // 针对当前在新发现的凸壳点的左侧还有右侧，需要进行不同的计算公式来
            // 确定其在下一轮迭代中的 LABEL 值和在坐标点集中的下标值。
            if (leftflag[idx] == 1) {
                // 对于当前点在新的凸壳点的左侧，计算新的 LABEL 值。这里
                // foundacc 的物理含义是当前 LABEL 值之前的各个 LABEL 区域中总
                // 共找到的新增凸壳点的数量。
                newlabel[idx] = label[idx] + foundacc[label[idx]];

                // 对于当前点在新的凸壳点的左侧，计算坐标点在新一轮迭代中的下标
                // 值。这里首先确定当前 LABEL 的新的起始下标值，然后再加上该点在
                // 其 LABEL 区域内其前面的左侧点的数量，就得到了其新的下标值。
#ifndef CHNI_ENABLE_FAST_CALC
                int basenewidx = startidx[label[idx]] -
                                 negdistacc[startidx[label[idx]]];
                int innernewidx = leftacc[idx] - leftacc[startidx[label[idx]]];
                newidx[idx] = basenewidx + innernewidx;
#else
                newidx[idx] = startidx[label[idx]] -
                              negdistacc[startidx[label[idx]]] + leftacc[idx] -
                              leftacc[startidx[label[idx]]];
#endif
            } else {
                // 对于当前点在新的凸壳点的右侧，计算新的 LABEL 值。这里
                // foundacc 的物理含义是当前 LABEL 值之前的各个 LABEL 区域中总共
                // 找到的新增凸壳点的数量。
                newlabel[idx] = label[idx] + foundacc[label[idx]] + 1; 

                // 对于当前点在新的凸壳点的右侧，计算坐标点在新一轮迭代中的下标
                // 值。计算该值，首先计算右侧构成的新的 LABEL 区域的起始位置，这
                // 部分只需要在左侧起始下标处加上当前 LABEL 区域总共检出的左侧坐
                // 标的数量即可；之后，需要计算该坐标点在新的区域内部的偏移量，
                // 即内部的原来下标值，减去其前 面的负垂距点数量和左侧点数量。
#ifndef CHNI_ENABLE_FAST_CALC
                int leftcnt = leftacc[startidx[label[idx] + 1]] -
                              leftacc[startidx[label[idx]]];
                int basenewidx = startidx[label[idx]] -
                                 negdistacc[startidx[label[idx]]] + 
                                 leftcnt;
                int inidx = idx - startidx[label[idx]];
                int innegacc = negdistacc[idx] -
                               negdistacc[startidx[label[idx]]];
                int inleftacc = leftacc[idx] - leftacc[startidx[label[idx]]];
                int innernewidx = inidx - innegacc - inleftacc;
                newidx[idx] = basenewidx + innernewidx;
#else
                newidx[idx] = idx - negdistacc[idx] +
                              leftacc[startidx[label[idx] + 1]] - leftacc[idx];
#endif
            }

        }
    }

// 消除本 Kernel 函数内部的宏定义，防止后面的函数使用造成冲突。
#ifdef CHNI_ENABLE_FAST_CALC
#  undef CHNI_ENABLE_FAST_CALC
#endif

    return NO_ERROR;
}

// Kernel 函数: _arrangeCstKer（生成新坐标点集）
static __global__ void _arrangeCstKer(
        CoordiSetCuda cst, int negdistflag[], int newidx[], int tmplabel[],
        int cstcnt, CoordiSetCuda newcst, int newlabel[])
{
    // 计算当前 Thread 的全局索引。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 若当前 Thread 处理的是越界数据，则直接返回
    if (idx >= cstcnt)
        return;

    // 如果当前 Thread 对应的坐标点是应该在下一轮计算中被排除的负垂距点，那么
    // 该 Thread 直接退出。
    if (negdistflag[idx] == 1)
        return;

    // 读取当前线程所处理的
    int newindex = newidx[idx];

    // 将坐标集按照新的位置拷贝到新的坐标集中。由于并行拷贝，无法保证顺序，因此
    // 这里使用了两个数组。
    newcst.tplMeta.tplData[2 * newindex] = cst.tplMeta.tplData[2 * idx];
    newcst.tplMeta.tplData[2 * newindex + 1] = 
            cst.tplMeta.tplData[2 * idx + 1];

    // 将新的 LABEL 标记从原来的下标处拷贝到新的下标处，由于并行拷贝，无法保证
    // 顺序，因此这里使用了两个数组。
    newlabel[newindex] = tmplabel[idx];
}

// 成员方法：arrangeCst（生成新坐标点集）
__host__ int ConvexHull::arrangeCst(
            CoordiSet *cst, int negdistflag[], int newidx[], int tmplabel[],
            int cstcnt, CoordiSet *newcst, int newlabel[])
{
    // 检查参数中所有的指针和数组是否为空。
    if (cst == NULL || newcst == NULL || negdistflag == NULL ||
        tmplabel == NULL || newidx == NULL || newlabel == NULL)
        return NULL_POINTER;

    // 检查坐标点数量，必须要大于 0。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 将输入坐标集拷贝到当前 Device。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出坐标集拷贝到当前 Device。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(newcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 坐标集的 CUDA 相关数据
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    CoordiSetCuda *newcstCud = COORDISET_CUDA(newcst);

    // 计算调用 Kernel 函数 Block 尺寸和 Block 数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (cstcnt + blocksize - 1) / blocksize;

    // 调用 Kernel 函数，完成计算。
    _arrangeCstKer<<<gridsize, blocksize>>>(
            *cstCud, negdistflag, newidx, tmplabel, cstcnt, 
            *newcstCud, newlabel);

    // 判断 Kernel 函数执行是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// 成员方法：arrangeCstCpu（生成新坐标点集）
__host__ int ConvexHull::arrangeCstCpu(
            CoordiSet *cst, int negdistflag[], int newidx[], int tmplabel[],
            int cstcnt, CoordiSet *newcst, int newlabel[])
{
    // 检查参数中所有的指针和数组是否为空。
    if (cst == NULL || newcst == NULL || negdistflag == NULL ||
        tmplabel == NULL || newidx == NULL || newlabel == NULL)
        return NULL_POINTER;

    // 检查坐标点数量，必须要大于 0。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 将输入坐标集拷贝到当前 Host。
    errcode = CoordiSetBasicOp::copyToHost(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出坐标集拷贝到当前 Host。
    errcode = CoordiSetBasicOp::copyToHost(newcst);
    if (errcode != NO_ERROR)
        return errcode;

    // 坐标集的 CUDA 相关数据
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    CoordiSetCuda *newcstCud = COORDISET_CUDA(newcst);

    for (int idx = 0; idx < cstcnt; idx++) {
        if (negdistflag[idx] == 0) {
            int newindex = newidx[idx];

            // 将坐标集按照新的位置拷贝到新的坐标集中。由于并行拷贝，无法保证顺
            // 序，因此这里使用了两个数组。
            (*newcstCud).tplMeta.tplData[2 * newindex] =
                    (*cstCud).tplMeta.tplData[2 * idx];
            (*newcstCud).tplMeta.tplData[2 * newindex + 1] = 
                    (*cstCud).tplMeta.tplData[2 * idx + 1];

            // 将新的 LABEL 标记从原来的下标处拷贝到新的下标处，由于并行拷贝，
            // 无法保证顺序，因此这里使用了两个数组。
            newlabel[newindex] = tmplabel[idx];
        }
    }
    return NO_ERROR;
}

// Kernel 函数：_flipWholeCstKer（整体翻转坐标点集）
static __global__ void _flipWholeCstKer(
        CoordiSetCuda incst, CoordiSetCuda outcst)
{
    // 计算当前 Thread 的全局下标
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果当前 Thread 处理的是越界数据，则直接退出。
    if (idx >= incst.tplMeta.count)
        return;

    // 将 x 和 y 坐标的相反数赋值给输出坐标点集。
    outcst.tplMeta.tplData[2 * idx] = 
            -incst.tplMeta.tplData[2 * idx];
    outcst.tplMeta.tplData[2 * idx + 1] = 
            -incst.tplMeta.tplData[2 * idx + 1];
}

// Host 成员方法：flipWholeCstCpu（整体翻转坐标点集）
__host__ int ConvexHull::flipWholeCstCpu(CoordiSet *incst, CoordiSet *outcst)
{
    // 检查输入坐标点集是否为 NULL。
    if (incst == NULL)
        return NULL_POINTER;

    // 检查输入坐标点集是否包含有效的坐标点。
    if (incst->count <= 0 || incst->tplData == NULL)
        return INVALID_DATA;

    // 如果输出点集为 NULL，则函数会进行 In-Place 操作，即将输出点集赋值为输入
    // 点集。
    if (outcst == NULL)
        outcst = incst;

    // 声明局部变量，错误码。
    int errcode;

    // 将输入坐标点集拷贝到当前 Host 中。
    errcode = CoordiSetBasicOp::copyToHost(incst);
    if (errcode != NO_ERROR)
        return errcode;

    // 对于 Out-Place 方法还需要对输出坐标点集进行初始化操作。
    if (incst != outcst) {
        // 将输出坐标集拷贝入 Host 内存。
        errcode = CoordiSetBasicOp::copyToHost(outcst);
        if (errcode != NO_ERROR) {
            // 如果输出坐标集无数据（故上面的拷贝函数会失败），则会创建一个和
            // 输入坐标集寸相同的图像。
            errcode = CoordiSetBasicOp::makeAtHost(
                    outcst, incst->count); 
            // 如果创建坐标集也操作失败，则说明操作彻底失败，报错退出。
            if (errcode != NO_ERROR)
                return errcode;
        }
    }

    // 取出两个坐标点集对应的 CUDA 型变量。
    CoordiSetCuda *incstCud = COORDISET_CUDA(incst);
    CoordiSetCuda *outcstCud = COORDISET_CUDA(outcst);

    // 为了防止越界访存，这里临时将输入点集的尺寸切换为输入和输出点集中较小的那
    // 个。当然，在操作后还需要将点集的数量恢复，因此，通过另一个变量保存原始的
    // 坐标点数量。
    int incstcntorg = incst->count;
    if (incst->count > outcst->count)
        incst->count = outcst->count;

    for (int idx = 0; idx < (*incstCud).tplMeta.count; idx++) {
        // 将 x 和 y 坐标的相反数赋值给输出坐标点集。
        (*outcstCud).tplMeta.tplData[2 * idx] = 
                -(*incstCud).tplMeta.tplData[2 * idx];
        (*outcstCud).tplMeta.tplData[2 * idx + 1] = 
                -(*incstCud).tplMeta.tplData[2 * idx + 1];
    }

    // 回复输入坐标点集中坐标点的数量。
    incst->count = incstcntorg;

    // 处理完毕退出。
    return NO_ERROR;
}


// Host 成员方法：flipWholeCst（整体翻转坐标点集）
__host__ int ConvexHull::flipWholeCst(CoordiSet *incst, CoordiSet *outcst)
{
    // 检查输入坐标点集是否为 NULL。
    if (incst == NULL)
        return NULL_POINTER;

    // 检查输入坐标点集是否包含有效的坐标点。
    if (incst->count <= 0 || incst->tplData == NULL)
        return INVALID_DATA;

    // 如果输出点集为 NULL，则函数会进行 In-Place 操作，即将输出点集赋值为输入
    // 点集。
    if (outcst == NULL)
        outcst = incst;

    // 声明局部变量，错误码。
    int errcode;

    // 将输入坐标点集拷贝到当前 Device 中。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(incst);
    if (errcode != NO_ERROR)
        return errcode;

    // 对于 Out-Place 方法还需要对输出坐标点集进行初始化操作。
    if (incst != outcst) {
        // 将输出坐标集拷贝入 Device 内存。
        errcode = CoordiSetBasicOp::copyToCurrentDevice(outcst);
        if (errcode != NO_ERROR) {
            // 如果输出坐标集无数据（故上面的拷贝函数会失败），则会创建一个和
            // 输入坐标集寸相同的图像。
            errcode = CoordiSetBasicOp::makeAtCurrentDevice(
                    outcst, incst->count); 
            // 如果创建坐标集也操作失败，则说明操作彻底失败，报错退出。
            if (errcode != NO_ERROR)
                return errcode;
        }
    }

    // 取出两个坐标点集对应的 CUDA 型变量。
    CoordiSetCuda *incstCud = COORDISET_CUDA(incst);
    CoordiSetCuda *outcstCud = COORDISET_CUDA(outcst);

    // 为了防止越界访存，这里临时将输入点集的尺寸切换为输入和输出点集中较小的那
    // 个。当然，在操作后还需要将点集的数量恢复，因此，通过另一个变量保存原始的
    // 坐标点数量。
    int incstcntorg = incst->count;
    if (incst->count > outcst->count)
        incst->count = outcst->count;

    // 计算启动 Kernel 函数所需要的 Thread 数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (incst->count + blocksize - 1) / blocksize;

    // 启动 Kernel 函数完成计算。
    _flipWholeCstKer<<<gridsize, blocksize>>>(*incstCud, *outcstCud);

    // 回复输入坐标点集中坐标点的数量。
    incst->count = incstcntorg;

    // 检查 Kernel 函数是否执行正确。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// 宏：FAIL_CONVEXHULL_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_CONVEXHULL_FREE  do {                            \
        if (tmpmem != NULL)                                   \
            delete tmpmem;                                    \
        if (tmpcstin != NULL)                                 \
            CoordiSetBasicOp::deleteCoordiSet(tmpcstin);      \
        if (tmpcstout != NULL)                                \
            CoordiSetBasicOp::deleteCoordiSet(tmpcstout);     \
        if (tmpconvexin != NULL)                              \
            CoordiSetBasicOp::deleteCoordiSet(tmpconvexin);   \
        if (tmpconvexout != NULL)                             \
            CoordiSetBasicOp::deleteCoordiSet(tmpconvexout);  \
    } while (0)

// 成员方法：convexHullIter（迭代法求凸壳上的点集）
__host__ int ConvexHull::convexHullIterCpu(
        CoordiSet *inputcst, CoordiSet *convexcst, bool lowerconvex)
{
    // 检查输入坐标集，输出坐标集是否为空。
    if (inputcst == NULL || convexcst == NULL)
        return NULL_POINTER;

    // 如果输入点集中不含有任何的坐标点，则直接退出。
    if (inputcst->count < 1 || inputcst->tplData == NULL)
        return INVALID_DATA;

    // 如果输入点集中点的数量少于 2 个点时，则不需要任何求解过程，直接将输入点
    // 集拷贝到输出点集即可。虽然当坐标点集中仅包含两个点时也可以直接判定为凸壳
    // 点，但考虑到顺序问题，代码还是让仅有两个点的情况走完整个流程。
    if (inputcst->count < 2)
        return CoordiSetBasicOp::copyToHost(inputcst, convexcst);

    // 局部变量
    int errcode;

    // 定义扫描所用的二元操作符。
    add_class<int> add;

    // 采用 CPU scan
    this->aryScan.setScanType(CPU_IN_SCAN);

    int cstcnt = inputcst->count;  // 坐标点集中点的数量。这里之所以将其使用另
                                   // 外一个变量保存出来是因为这个值随着迭代会
                                   // 变化，如果直接使用 CoordiSet 中的 count 
                                   // 域会带来内存管理上的不便。
    int convexcnt = 2;             // 当前凸壳点的数量，由于迭代开始时，已经实
                                   // 现找到了点集中的最左和最有两点作为凸壳
                                   // 点，因此这里直接赋值为 2。
    int foundcnt;                  // 当前迭代时找到的新凸壳点的数量，这一数量
                                   // 并不包含往次所找到的凸壳点。
    int negdistcnt;                // 当前负垂距点的数量。
    //int itercnt = 0;             // 迭代次数记录器。

    int *tmpmem = NULL;              // 存放中间变量的内存空间。
    CoordiSet *tmpcstin = NULL;      // 每次迭代中作为输入坐标点集的临时坐标点
                                     // 集。
    CoordiSet *tmpcstout = NULL;     // 每次迭代中作为输出坐标点击的临时坐标点
                                     // 集。
    CoordiSet *tmpconvexin = NULL;   // 每次迭代中作为输入凸壳点集的临时坐标点
                                     // 集。
    CoordiSet *tmpconvexout = NULL;  // 每次迭代中作为输出凸壳点集（新凸壳点
                                     // 集）的临时坐标点集。

    size_t datacnt = 0;   // 所需要的数据元素的数量。
    size_t datasize = 0;  // 书需要的数据元素的字节尺寸。

    // 宏：CHI_DATA_DECLARE（中间变量声明器）
    // 为了消除中间变量声明过程中大量的重复代码，这里提供了一个宏，使代码看起来
    // 整洁一些。
#define CHI_DATA_DECLARE(dataname, type, count)  \
    type *dataname = NULL;                       \
    size_t dataname##cnt = (count);              \
    datacnt += dataname##cnt;                    \
    datasize += dataname##cnt * sizeof (type)

    // 声明各个中间变量的 Device 数组。
    CHI_DATA_DECLARE(label, int,            // 记录当前迭代中每个像素点所在的
                     inputcst->count);      // LABEL 区域。
    CHI_DATA_DECLARE(negdistflag, int,      // 记录当前迭代中每个像素点是否具有
                     inputcst->count);      // 负垂距。
    CHI_DATA_DECLARE(negdistacc, int,       // 记录当前迭代中具有负垂距点的累加
                     inputcst->count + 1);  // 和，其物理含义是在当前点之前存在
                                            // 多少个负垂距点，其最后一个元素表
                                            // 示当前迭代共找到了多少个负垂距
                                            // 点。
    CHI_DATA_DECLARE(maxdistidx, int,       // 记录当前迭代中每个坐标点前面的所
                     inputcst->count);      // 有点中和其在同一个 LABEL 区域的
                                            // 所有点中具有最大垂距的下标。
    CHI_DATA_DECLARE(foundflag, int,        // 记录当前迭代中各个 LABEL 区域是
                     inputcst->count);      // 否找到了新的凸壳点。
    CHI_DATA_DECLARE(foundacc, int,         // 记录当前迭代中每个 LABEL 区域其
                     inputcst->count + 1);  // 前面的所有 LABEL 区域共找到的新
                                            // 的凸壳点的数量。该值用于计算各个
                                            // 凸壳点（无论是旧的还是新的）在新
                                            // 的凸壳点集中的新下标。
    CHI_DATA_DECLARE(leftflag, int,         // 记录当前的坐标点是否处于新发现的
                     inputcst->count);      // 坐标点的左侧
    CHI_DATA_DECLARE(leftacc, int,          // 记录当前的坐标点之前的左侧点的数
                     inputcst->count + 1);  // 量。该数组用于计算坐标点在下一轮
                                            // 计算中的下标。
    CHI_DATA_DECLARE(startidx, int,         // 记录每个 LABEL 区域在坐标点集中
                     inputcst->count);      // 的起始下标
    CHI_DATA_DECLARE(newidx, int,           // 记录当前坐标点集中各个坐标点在下
                     inputcst->count);      // 一轮迭代中的新的下标。
    CHI_DATA_DECLARE(tmplabel, int,
                     inputcst->count);
    CHI_DATA_DECLARE(newlabel, int,         // 记录当前坐标点集中各个坐标点在下
                     inputcst->count);      // 一轮迭代中新的 LABEL 区域。

    // 消除中间变量声明器这个宏，防止后续步骤的命名冲突。
#undef CHI_DATA_DECLARE

    // 中间变量申请 Host 内存空间，并将这些空间分配给各个中间变量。
    tmpmem = new int[datasize];

    // 为各个中间变量分配内存空间，采用这种一次申请一个大空间的做法是为了减少申
    // 请内存的开销，同时也减少因内存对齐导致的内存浪费。
    label       = tmpmem;
    negdistflag = label       + labelcnt;
    negdistacc  = negdistflag + negdistflagcnt;
    maxdistidx  = negdistacc  + negdistacccnt;
    foundflag   = maxdistidx  + maxdistidxcnt;
    foundacc    = foundflag   + foundflagcnt;
    leftflag    = foundacc    + foundacccnt;
    leftacc     = leftflag    + leftflagcnt;
    startidx    = leftacc     + leftacccnt;
    newidx      = startidx    + startidxcnt;
    newlabel    = newidx      + newidxcnt;
    tmplabel    = newlabel    + newlabelcnt;

    // 宏：CHI_USE_SYS_FUNC
    // 该开关宏用于指示是否在后续步骤中尽量使用 CUDA 提供的函数，而不是启动由开
    // 发方自行编写的 Kernel 函数完成操作。
//#define CHI_USE_SYS_FUNC

    // 初始化 LABEL 数组。
#ifdef CHI_USE_SYS_FUNC
    // 首先将 LABEL 数组中所有内存元素全部置零。
    memset(label, 0, labelcnt * sizeof (int));

    // 将 LABEL 数组中最后一个元素置 1。
    label[cstcnt - 1] = 1;
#else
    // 调用 LABEL 初始化函数，完成 LABEL 初始化。初始化后，除最后一个元素为 1 
    // 外，其余元素皆为 0。
    errcode = this->initLabelAryCpu(label, cstcnt);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }
#endif

    // 初始化迭代过程中使用的坐标点集，这里一共需要使用到两个坐标点集，为了不破
    // 坏输入坐标点集，这里在迭代过程中我们使用内部申请的坐标点集。

#ifdef CH_DEBUG_CPU_PRINT
    cout << "[convexHullIterCpu]init CoordiSet" << endl;
#endif
    // 初始化第一个坐标点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpcstin);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 将输入坐标点集中的数据从输入点集中拷贝到第一个坐标点集中。此后所有的操作
    // 仅在临时坐标点集中处理，不再碰触输入坐标点集。这里如果是求解上半凸壳，则
    // 直接调用翻转坐标点的函数。
    if (lowerconvex)
        errcode = CoordiSetBasicOp::copyToHost(inputcst, tmpcstin);
    else
        errcode = this->flipWholeCst(inputcst, tmpcstin);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 初始化第二个坐标点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpcstout);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 在 Device 内存中初始化第二个坐标点集，为其申请足够长度的内存空间。
    errcode = CoordiSetBasicOp::makeAtHost(tmpcstout, inputcst->count);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }   

    // 初始化迭代过程中使用到的凸壳点集，这里一共需要两个凸壳点集。我们不急于更
    // 新输出参数 convexcst，是因为避免不必要的麻烦，等到凸壳计算完毕后，再将凸
    // 壳内容拷贝到输出参数中。

    // 初始化第一个凸壳点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpconvexin);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 在 Device 内存中初始化第一个凸壳点集，为其申请足够长度的内存空间。
    errcode = CoordiSetBasicOp::makeAtHost(tmpconvexin,
                                           inputcst->count);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 初始化第二个凸壳点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpconvexout);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 在 Device 内存中初始化第二个凸壳点集，为其申请足够长度的内存空间。
    errcode = CoordiSetBasicOp::makeAtHost(tmpconvexout,
                                           inputcst->count);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }   

#ifdef CH_DEBUG_CPU_PRINT
    cout << "[convexHullIterCpu]init CoordiSet finish" << endl;
#endif

    // 寻找最左最右点，并利用这两个点初始化输入点集和凸壳点集。初始化后，输入点
    // 集的第一个点为最左点，最后一个点为最右点；凸壳点集中仅包含最左最右两个
    // 点。
    errcode = swapEdgePointCpu(tmpcstin, tmpconvexin);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }
#ifdef CH_DEBUG_CPU_PRINT
    cout << "[convexHullIterCpu]swap finish" << endl;
    cout << "[convexHullIterCpu]interation begin" << endl;
#endif
    // 所有的初始化过程至此全部完毕，开始进行迭代。每次迭代都需要重新计算坐标点
    // 在其 LABEL 区域内的垂距，然后根据垂距信息判断每个 LABEL 区域内是否存在新
    // 的凸壳点（如果有需要确定是哪一个点），之后根据这个新发现的凸壳点，计算所
    // 有坐标点在下一轮迭代中的下标。计算后的下标要求属于一个 LABEL 的点都在一
    // 起，并且排除所有具有负垂距的点，因为这些点在下一轮迭代中已经毫无意义。迭
    // 代的过程知道无法在从当前所有的 LABEL 区域内找到新的凸壳点为止。此处循环
    // 的判断条件只是一个防护性措施，若坐标点集的数量同凸壳点相等，那就说明没有
    // 任何可能在找到新的凸壳点了。
    while (cstcnt >= convexcnt) {
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]new: convexcnt is " << convexcnt << endl;
        cout << endl;
        cout << "[convexHullIterCpu]updatedist begin" << endl;
        cout << "[convexHullIterCpu]cstcnt is " << cstcnt << endl;
#endif
        // 调用更新垂距函数。更新点集中每个点的垂距值和负垂距标志数组。
        errcode = this->updateDistCpu(tmpcstin, tmpconvexin, label, 
                                      cstcnt, negdistflag);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

#ifdef CH_DEBUG_CPU_PRINT
        cout << endl;
        cout << "[convexHullIterCpu]segscan begin" << endl;
#endif
        // 利用分段扫描得到各个 LABEL 区域的最大垂距，记忆最大垂距坐标点的下标
        // 值。
        errcode = this->segScan.segmentedScanCpu(
                ATTACHED_DATA(tmpcstin), label, 
                ATTACHED_DATA(tmpcstout), maxdistidx, cstcnt, false);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]segscan end" << endl;
        cout << "[convexHullIterCpu]updateFoundInfoCpu begin" << endl;
#endif
        // 根据所求出来的垂距信息判断各个 LABEL 区域是否有新的凸壳点存在。
        errcode = this->updateFoundInfoCpu(
                label, ATTACHED_DATA(tmpcstin), maxdistidx,
                cstcnt, foundflag, startidx);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]updateFoundInfoCpu end" << endl;
        cout << "[convexHullIterCpu]scan foudnd begin" << endl;
#endif
        // 通过扫描，计算出 LABEL 区域新发现凸壳点标记值对应的累加值。
        errcode = this->aryScan.scanArrayExclusive(foundflag, foundacc,
                                                   convexcnt, add,
                                                   false, true, true);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]scan found end" << endl;
#endif
        // 将新凸壳点标记累加值的最后一个拷贝到 Host 内存中，这个累加值的含义是
        // 当前迭代下所有新发现的凸壳点的数量。 
        foundcnt = foundacc[convexcnt];
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]foundcnt now is " << foundcnt << endl;
#endif

        // 如果新发现的凸壳点的数量小于等于 0，则说明说有的凸壳点都已经被找到，
        // 没有必要在继续做下去了，因此退出迭代。
        if (foundcnt <= 0)
            break;
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]updateConvexCstCpu begin" << endl;
#endif
        // 更新凸壳点集，将新发现的凸壳点集更新到凸壳点集中。
        errcode = this->updateConvexCstCpu(
                tmpcstin, tmpconvexin, foundflag, foundacc, startidx, 
                maxdistidx, convexcnt, tmpconvexout);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]updateConvexCstCpu end" << endl;
#endif
        // 更新凸壳点集中点的数量。
        convexcnt += foundcnt;
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]convexcnt now is " << convexcnt << endl;
        cout << "[convexHullIterCpu]markLeftPointsCpu begin" << endl;
#endif
        // 标记左侧点。所谓左侧点是在某 LABEL 区域内处于新发现的凸壳点左侧的
        // 点。
        errcode = this->markLeftPointsCpu(
                tmpcstin, tmpconvexout, negdistflag, label, 
                foundflag, foundacc, cstcnt, leftflag);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]markLeftPointsCpu end" << endl;
        cout << "[convexHullIterCpu]scanArrayExclusive neg begin" << endl;
#endif
        // 通过扫描，计算出负垂距点标记数组对应的累加数组。negdistflagDev 实在
        // 第一步更新垂距的时候获得的，之所以这么晚才计算其对应的累加数组，是因
        // 为在前面检查 foundcnt 退出循环之前不需要这个数据，这样，如果真的在该
        // 处退出，则程序进行了多余的计算，为了避免这一多余计算，我们延后计算 
        // negdistaccDev 至此处。
        errcode = this->aryScan.scanArrayExclusive(
                negdistflag, negdistacc, cstcnt, add, 
                false, true, true);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]scanArrayExclusive neg end" << endl;
#endif

        // 将负垂距点累加总和拷贝出来，用来更新下一轮循环的坐标点数量值。
        negdistcnt = negdistacc[cstcnt];
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]negdistcnt now is " << negdistcnt << endl;
        cout << "[convexHullIterCpu]scanArrayExclusive left begin" << endl;
#endif
        // 通过扫描计算处左侧点标记数组对应的累加数组。
        errcode = this->aryScan.scanArrayExclusive(
                leftflag, leftacc, cstcnt, add, 
                false, true, true);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]scanArrayExclusive left end" << endl;
        cout << "[convexHullIterCpu]updatePropertyCpu begin" << endl;
#endif
        // 计算各个坐标点在下一轮迭代中的新下标。
        errcode = this->updatePropertyCpu(
                leftflag, leftacc, negdistflag, negdistacc,
                startidx, label, foundacc, cstcnt,
                newidx, tmplabel);

        // Merlin debug
        cudaDeviceSynchronize();
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]updatePropertyCpu end" << endl;
        cout << "[convexHullIterCpu]arrangeCstCpu begin" << endl;
#endif
        // 根据上一步计算得到的新下标，生成下一轮迭代所需要的坐标点集。
        errcode = this->arrangeCstCpu(
                tmpcstin, negdistflag, newidx, tmplabel, 
                cstcnt, tmpcstout, newlabel);
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]arrangeCstCpu end" << endl;
#endif

        // 交还部分中间变量，将本轮迭代得到的结果给到下一轮迭代的参数。
        int *labelswptmp = label;
        label = newlabel;
        newlabel = labelswptmp;
    
        CoordiSet *cstswptmp = tmpcstin;
        tmpcstin = tmpcstout;
        tmpcstout = cstswptmp;

        cstswptmp = tmpconvexin;
        tmpconvexin = tmpconvexout;
        tmpconvexout = cstswptmp;

        cstcnt -= negdistcnt;
#ifdef CH_DEBUG_CPU_PRINT
        cout << "[convexHullIterCpu]cstcnt now is " << cstcnt << endl;
#endif
        // 一轮迭代到此结束。
    }

    // 将计算出来的凸壳点拷贝到输出点集中。迭代完成后，tmpconvexin 保存有最后的
    // 结果。如果在 while 判断条件处退出迭代，则上一轮求出的凸壳点集是最终结
    // 果，此时在上一轮末，由于交换指针，使得原本存放在tmpconvexout 的最终结果
    // 变为了存放在 tmpconvexin 中；如果迭代实在判断有否新发现点处退出，则说明
    // 当前并未发现新的凸壳点，那么 tmpconvexin 和 tmpconvexout 内容应该是一致
    // 的，但本着稳定的原则，应该取更早形成的变量，即 tmpconvexin。

    // 首先临时将这个存放结果的点集的点数量修改为凸壳点的数量。
    tmpconvexin->count = convexcnt;

    // 然后，将计算出来的凸壳点拷贝到输出参数中。如果是求解上半凸壳点，则需要将
    // 结果翻转后输出，但是由于翻转函数不能改变输出点集的点的数量，因此，这里还
    // 需要先使用拷贝函数，调整输出点的数量（好在，通常凸壳点的数量不错，这一步
    // 骤不会造成太能的性能下降，若日后发现有严重的性能下降，还需要额外写一个更
    // 加复杂一些的翻转函数。）
    errcode = CoordiSetBasicOp::copyToHost(tmpconvexin, convexcst);
    if (errcode != NO_ERROR) {
        tmpconvexin->count = inputcst->count;
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 最后，为了程序稳定性的考虑，回复其凸壳点的数量。
    tmpconvexin->count = inputcst->count;

    // 释放内存
    delete tmpmem;
    // cudaFree(tmpmemDev);
    CoordiSetBasicOp::deleteCoordiSet(tmpcstin);
    CoordiSetBasicOp::deleteCoordiSet(tmpcstout);
    CoordiSetBasicOp::deleteCoordiSet(tmpconvexin);
    CoordiSetBasicOp::deleteCoordiSet(tmpconvexout);

    // 最后，如果所求点是上半凸壳，则还需要翻转所有凸壳点。
    if (!lowerconvex) {
        errcode = this->flipWholeCstCpu(convexcst, convexcst);
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 操作完毕，退出。
    return NO_ERROR;
}
#undef FAIL_CONVEXHULL_FREE

// 宏：FAIL_CONVEXHULL_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_CONVEXHULL_FREE  do {                            \
        if (tmpmemDev != NULL)                                \
            cudaFree(tmpmemDev);                              \
        if (tmpcstin != NULL)                                 \
            CoordiSetBasicOp::deleteCoordiSet(tmpcstin);      \
        if (tmpcstout != NULL)                                \
            CoordiSetBasicOp::deleteCoordiSet(tmpcstout);     \
        if (tmpconvexin != NULL)                              \
            CoordiSetBasicOp::deleteCoordiSet(tmpconvexin);   \
        if (tmpconvexout != NULL)                             \
            CoordiSetBasicOp::deleteCoordiSet(tmpconvexout);  \
    } while (0)

// 成员方法：convexHullIter（迭代法求凸壳上的点集）
__host__ int ConvexHull::convexHullIter(
        CoordiSet *inputcst, CoordiSet *convexcst, bool lowerconvex)
{
    // 检查输入坐标集，输出坐标集是否为空。
    if (inputcst == NULL || convexcst == NULL)
        return NULL_POINTER;

    // 如果输入点集中不含有任何的坐标点，则直接退出。
    if (inputcst->count < 1 || inputcst->tplData == NULL)
        return INVALID_DATA;

    // 如果输入点集中点的数量少于 2 个点时，则不需要任何求解过程，直接将输入点
    // 集拷贝到输出点集即可。虽然当坐标点集中仅包含两个点时也可以直接判定为凸壳
    // 点，但考虑到顺序问题，代码还是让仅有两个点的情况走完整个流程。
    if (inputcst->count < 2)
        return CoordiSetBasicOp::copyToCurrentDevice(inputcst, convexcst);

    // 局部变量
    cudaError_t cuerrcode;  // CUDA 函数调用返回的错误码
    int errcode;            // 调用函数返回的错误码

    // 定义扫描所用的二元操作符。
    add_class<int> add;

    int cstcnt = inputcst->count;  // 坐标点集中点的数量。这里之所以将其使用另
                                   // 外一个变量保存出来是因为这个值随着迭代会
                                   // 变化，如果直接使用 CoordiSet 中的 count 
                                   // 域会带来内存管理上的不便。
    int convexcnt = 2;             // 当前凸壳点的数量，由于迭代开始时，已经实
                                   // 现找到了点集中的最左和最有两点作为凸壳
                                   // 点，因此这里直接赋值为 2。
    int foundcnt;                  // 当前迭代时找到的新凸壳点的数量，这一数量
                                   // 并不包含往次所找到的凸壳点。
    int negdistcnt;                // 当前负垂距点的数量。
    //int itercnt = 0;             // 迭代次数记录器。

    int *tmpmemDev = NULL;           // 存放中间变量的 Device 内存空间。
    CoordiSet *tmpcstin = NULL;      // 每次迭代中作为输入坐标点集的临时坐标点
                                     // 集。
    CoordiSet *tmpcstout = NULL;     // 每次迭代中作为输出坐标点击的临时坐标点
                                     // 集。
    CoordiSet *tmpconvexin = NULL;   // 每次迭代中作为输入凸壳点集的临时坐标点
                                     // 集。
    CoordiSet *tmpconvexout = NULL;  // 每次迭代中作为输出凸壳点集（新凸壳点
                                     // 集）的临时坐标点集。

    size_t datacnt = 0;   // 所需要的数据元素的数量。
    size_t datasize = 0;  // 书需要的数据元素的字节尺寸。

    // 宏：CHI_DATA_DECLARE（中间变量声明器）
    // 为了消除中间变量声明过程中大量的重复代码，这里提供了一个宏，使代码看起来
    // 整洁一些。
#define CHI_DATA_DECLARE(dataname, type, count)  \
    type *dataname##Dev = NULL;                  \
    size_t dataname##cnt = (count);              \
    datacnt += dataname##cnt;                    \
    datasize += dataname##cnt * sizeof (type)

    // 声明各个中间变量的 Device 数组。
    CHI_DATA_DECLARE(label, int,            // 记录当前迭代中每个像素点所在的
                     inputcst->count);      // LABEL 区域。
    CHI_DATA_DECLARE(negdistflag, int,      // 记录当前迭代中每个像素点是否具有
                     inputcst->count);      // 负垂距。
    CHI_DATA_DECLARE(negdistacc, int,       // 记录当前迭代中具有负垂距点的累加
                     inputcst->count + 1);  // 和，其物理含义是在当前点之前存在
                                            // 多少个负垂距点，其最后一个元素表
                                            // 示当前迭代共找到了多少个负垂距
                                            // 点。
    CHI_DATA_DECLARE(maxdistidx, int,       // 记录当前迭代中每个坐标点前面的所
                     inputcst->count);      // 有点中和其在同一个 LABEL 区域的
                                            // 所有点中具有最大垂距的下标。
    CHI_DATA_DECLARE(foundflag, int,        // 记录当前迭代中各个 LABEL 区域是
                     inputcst->count);      // 否找到了新的凸壳点。
    CHI_DATA_DECLARE(foundacc, int,         // 记录当前迭代中每个 LABEL 区域其
                     inputcst->count + 1);  // 前面的所有 LABEL 区域共找到的新
                                            // 的凸壳点的数量。该值用于计算各个
                                            // 凸壳点（无论是旧的还是新的）在新
                                            // 的凸壳点集中的新下标。
    CHI_DATA_DECLARE(leftflag, int,         // 记录当前的坐标点是否处于新发现的
                     inputcst->count);      // 坐标点的左侧
    CHI_DATA_DECLARE(leftacc, int,          // 记录当前的坐标点之前的左侧点的数
                     inputcst->count + 1);  // 量。该数组用于计算坐标点在下一轮
                                            // 计算中的下标。
    CHI_DATA_DECLARE(startidx, int,         // 记录每个 LABEL 区域在坐标点集中
                     inputcst->count);      // 的起始下标
    CHI_DATA_DECLARE(newidx, int,           // 记录当前坐标点集中各个坐标点在下
                     inputcst->count);      // 一轮迭代中的新的下标。
    CHI_DATA_DECLARE(tmplabel, int,
                     inputcst->count);
    CHI_DATA_DECLARE(newlabel, int,         // 记录当前坐标点集中各个坐标点在下
                     inputcst->count);      // 一轮迭代中新的 LABEL 区域。

    // 消除中间变量声明器这个宏，防止后续步骤的命名冲突。
#undef CHI_DATA_DECLARE

    // 中间变量申请 Device 内存空间，并将这些空间分配给各个中间变量。
    cuerrcode = cudaMalloc((void **)&tmpmemDev, datasize);
    if (cuerrcode != cudaSuccess) {
        FAIL_CONVEXHULL_FREE;
        return CUDA_ERROR;
    }

    // 为各个中间变量分配内存空间，采用这种一次申请一个大空间的做法是为了减少申
    // 请内存的开销，同时也减少因内存对齐导致的内存浪费。
    labelDev       = tmpmemDev;
    negdistflagDev = labelDev       + labelcnt;
    negdistaccDev  = negdistflagDev + negdistflagcnt;
    maxdistidxDev  = negdistaccDev  + negdistacccnt;
    foundflagDev   = maxdistidxDev  + maxdistidxcnt;
    foundaccDev    = foundflagDev   + foundflagcnt;
    leftflagDev    = foundaccDev    + foundacccnt;
    leftaccDev     = leftflagDev    + leftflagcnt;
    startidxDev    = leftaccDev     + leftacccnt;
    newidxDev      = startidxDev    + startidxcnt;
    newlabelDev    = newidxDev      + newidxcnt;
    tmplabelDev    = newlabelDev    + newlabelcnt;

    // 宏：CHI_USE_SYS_FUNC
    // 该开关宏用于指示是否在后续步骤中尽量使用 CUDA 提供的函数，而不是启动由开
    // 发方自行编写的 Kernel 函数完成操作。
//#define CHI_USE_SYS_FUNC

    // 初始化 LABEL 数组。
#ifdef CHI_USE_SYS_FUNC
    // 首先将 LABEL 数组中所有内存元素全部置零。
    cuerrcode = cudaMemset(labelDev, 0, labelcnt * sizeof (int));
    if (cuerrcode != cudaSuccess) {
        FAIL_CONVEXHULL_FREE;
        return CUDA_ERROR;
    }

    // 将 LABEL 数组中最后一个元素置 1。
    int tmp_one = 1;
    cuerrcode = cudaMemcpy(&labelDev[cstcnt - 1], &tmp_one, 
                           sizeof (int), cudaMemcpyHostToDevice);
    if (cuerrcode != cudaSuccess) {
        FAIL_CONVEXHULL_FREE;
        return CUDA_ERROR;
    }
#else
    // 调用 LABEL 初始化函数，完成 LABEL 初始化。初始化后，除最后一个元素为 1 
    // 外，其余元素皆为 0。
    errcode = this->initLabelAry(labelDev, cstcnt);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }
#endif

    // 初始化迭代过程中使用的坐标点集，这里一共需要使用到两个坐标点集，为了不破
    // 坏输入坐标点集，这里在迭代过程中我们使用内部申请的坐标点集。

    // 初始化第一个坐标点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpcstin);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 将输入坐标点集中的数据从输入点集中拷贝到第一个坐标点集中。此后所有的操作
    // 仅在临时坐标点集中处理，不再碰触输入坐标点集。这里如果是求解上半凸壳，则
    // 直接调用翻转坐标点的函数。
    if (lowerconvex)
        errcode = CoordiSetBasicOp::copyToCurrentDevice(inputcst, tmpcstin);
    else
        errcode = this->flipWholeCst(inputcst, tmpcstin);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 初始化第二个坐标点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpcstout);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 在 Device 内存中初始化第二个坐标点集，为其申请足够长度的内存空间。
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(tmpcstout, 
                                                    inputcst->count);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }   

    // 初始化迭代过程中使用到的凸壳点集，这里一共需要两个凸壳点集。我们不急于更
    // 新输出参数 convexcst，是因为避免不必要的麻烦，等到凸壳计算完毕后，再将凸
    // 壳内容拷贝到输出参数中。

    // 初始化第一个凸壳点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpconvexin);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 在 Device 内存中初始化第一个凸壳点集，为其申请足够长度的内存空间。
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(tmpconvexin,
                                                    inputcst->count);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 初始化第二个凸壳点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpconvexout);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 在 Device 内存中初始化第二个凸壳点集，为其申请足够长度的内存空间。
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(tmpconvexout,
                                                    inputcst->count);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }   

    // 寻找最左最右点，并利用这两个点初始化输入点集和凸壳点集。初始化后，输入点
    // 集的第一个点为最左点，最后一个点为最右点；凸壳点集中仅包含最左最右两个
    // 点。
    errcode = swapEdgePoint(tmpcstin, tmpconvexin);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 所有的初始化过程至此全部完毕，开始进行迭代。每次迭代都需要重新计算坐标点
    // 在其 LABEL 区域内的垂距，然后根据垂距信息判断每个 LABEL 区域内是否存在新
    // 的凸壳点（如果有需要确定是哪一个点），之后根据这个新发现的凸壳点，计算所
    // 有坐标点在下一轮迭代中的下标。计算后的下标要求属于一个 LABEL 的点都在一
    // 起，并且排除所有具有负垂距的点，因为这些点在下一轮迭代中已经毫无意义。迭
    // 代的过程知道无法在从当前所有的 LABEL 区域内找到新的凸壳点为止。此处循环
    // 的判断条件只是一个防护性措施，若坐标点集的数量同凸壳点相等，那就说明没有
    // 任何可能在找到新的凸壳点了。
    while (cstcnt >= convexcnt) {
        // 调用更新垂距函数。更新点集中每个点的垂距值和负垂距标志数组。
        errcode = this->updateDist(tmpcstin, tmpconvexin, labelDev, 
                                   cstcnt, negdistflagDev);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 利用分段扫描得到各个 LABEL 区域的最大垂距，记忆最大垂距坐标点的下标
        // 值。
        errcode = this->segScan.segmentedScan(
                ATTACHED_DATA(tmpcstin), labelDev, 
                ATTACHED_DATA(tmpcstout), maxdistidxDev, cstcnt, false);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 根据所求出来的垂距信息判断各个 LABEL 区域是否有新的凸壳点存在。
        errcode = this->updateFoundInfo(
                labelDev, ATTACHED_DATA(tmpcstin), maxdistidxDev,
                cstcnt, foundflagDev, startidxDev);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 通过扫描，计算出 LABEL 区域新发现凸壳点标记值对应的累加值。
        errcode = this->aryScan.scanArrayExclusive(foundflagDev, foundaccDev,
                                                   convexcnt, add,
                                                   false, false, false);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 将新凸壳点标记累加值的最后一个拷贝到 Host 内存中，这个累加值的含义是
        // 当前迭代下所有新发现的凸壳点的数量。 
        cuerrcode = cudaMemcpy(&foundcnt, &foundaccDev[convexcnt], 
                               sizeof (int), cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
             FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 如果新发现的凸壳点的数量小于等于 0，则说明说有的凸壳点都已经被找到，
        // 没有必要在继续做下去了，因此退出迭代。
        if (foundcnt <= 0)
            break;

        // 更新凸壳点集，将新发现的凸壳点集更新到凸壳点集中。
        errcode = this->updateConvexCst(
                tmpcstin, tmpconvexin, foundflagDev, foundaccDev, startidxDev, 
                maxdistidxDev, convexcnt, tmpconvexout);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 更新凸壳点集中点的数量。
        convexcnt += foundcnt;

        // 标记左侧点。所谓左侧点是在某 LABEL 区域内处于新发现的凸壳点左侧的
        // 点。
        errcode = this->markLeftPoints(
                tmpcstin, tmpconvexout, negdistflagDev, labelDev, 
                foundflagDev, foundaccDev, cstcnt, leftflagDev);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 通过扫描，计算出负垂距点标记数组对应的累加数组。negdistflagDev 实在
        // 第一步更新垂距的时候获得的，之所以这么晚才计算其对应的累加数组，是因
        // 为在前面检查 foundcnt 退出循环之前不需要这个数据，这样，如果真的在该
        // 处退出，则程序进行了多余的计算，为了避免这一多余计算，我们延后计算 
        // negdistaccDev 至此处。
        errcode = this->aryScan.scanArrayExclusive(
                negdistflagDev, negdistaccDev, cstcnt, add, 
                false, false, false);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 将负垂距点累加总和拷贝出来，用来更新下一轮循环的坐标点数量值。
        cuerrcode = cudaMemcpy(&negdistcnt, &negdistaccDev[cstcnt],
                               sizeof (int), cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
             FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 通过扫描计算处左侧点标记数组对应的累加数组。
        errcode = this->aryScan.scanArrayExclusive(
                leftflagDev, leftaccDev, cstcnt, add, 
                false, false, false);
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 计算各个坐标点在下一轮迭代中的新下标。
        errcode = this->updateProperty(
                leftflagDev, leftaccDev, negdistflagDev, negdistaccDev,
                startidxDev, labelDev, foundaccDev, cstcnt,
                newidxDev, tmplabelDev);
        cudaDeviceSynchronize();
        if (errcode != NO_ERROR) {
            FAIL_CONVEXHULL_FREE;
            return errcode;
        }

        // 根据上一步计算得到的新下标，生成下一轮迭代所需要的坐标点集。
        errcode = this->arrangeCst(
                tmpcstin, negdistflagDev, newidxDev, tmplabelDev, 
                cstcnt, tmpcstout, newlabelDev);

        // 交还部分中间变量，将本轮迭代得到的结果给到下一轮迭代的参数。
        int *labelswptmp = labelDev;
        labelDev = newlabelDev;
        newlabelDev = labelswptmp;
    
        CoordiSet *cstswptmp = tmpcstin;
        tmpcstin = tmpcstout;
        tmpcstout = cstswptmp;

        cstswptmp = tmpconvexin;
        tmpconvexin = tmpconvexout;
        tmpconvexout = cstswptmp;

        cstcnt -= negdistcnt;

        // 一轮迭代到此结束。
    }

    // 将计算出来的凸壳点拷贝到输出点集中。迭代完成后，tmpconvexin 保存有最后的
    // 结果。如果在 while 判断条件处退出迭代，则上一轮求出的凸壳点集是最终结
    // 果，此时在上一轮末，由于交换指针，使得原本存放在tmpconvexout 的最终结果
    // 变为了存放在 tmpconvexin 中；如果迭代实在判断有否新发现点处退出，则说明
    // 当前并未发现新的凸壳点，那么 tmpconvexin 和 tmpconvexout 内容应该是一致
    // 的，但本着稳定的原则，应该取更早形成的变量，即 tmpconvexin。

    // 首先临时将这个存放结果的点集的点数量修改为凸壳点的数量。
    tmpconvexin->count = convexcnt;

    // 然后，将计算出来的凸壳点拷贝到输出参数中。如果是求解上半凸壳点，则需要将
    // 结果翻转后输出，但是由于翻转函数不能改变输出点集的点的数量，因此，这里还
    // 需要先使用拷贝函数，调整输出点的数量（好在，通常凸壳点的数量不错，这一步
    // 骤不会造成太能的性能下降，若日后发现有严重的性能下降，还需要额外写一个更
    // 加复杂一些的翻转函数。）
    errcode = CoordiSetBasicOp::copyToCurrentDevice(tmpconvexin, convexcst);
    if (errcode != NO_ERROR) {
        tmpconvexin->count = inputcst->count;
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 最后，为了程序稳定性的考虑，回复其凸壳点的数量。
    tmpconvexin->count = inputcst->count;

    // 释放内存
    cudaFree(tmpmemDev);
    CoordiSetBasicOp::deleteCoordiSet(tmpcstin);
    CoordiSetBasicOp::deleteCoordiSet(tmpcstout);
    CoordiSetBasicOp::deleteCoordiSet(tmpconvexin);
    CoordiSetBasicOp::deleteCoordiSet(tmpconvexout);

    // 最后，如果所求点是上半凸壳，则还需要翻转所有凸壳点。
    if (!lowerconvex) {
        errcode = this->flipWholeCst(convexcst, convexcst);
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 操作完毕，退出。
    return NO_ERROR;
}
#undef FAIL_CONVEXHULL_FREE

// Kernel 函数：_joinConvexKer（合并凸壳点集）
static __global__ void _joinConvexKer(
        CoordiSetCuda lconvex, CoordiSetCuda uconvex,
        CoordiSetCuda convex, int *convexcnt)
{
    // 共享内存，用来记录上下凸壳在最左最右点处是否重合，如果重合，应该在整合后
    // 的坐标点中排除重合的点。其中，[0] 表示最左点，[1] 表示最右点。用 1 表示
    // 有重合的点，用 0 表示没有重合的点。
    __shared__ int sameedge[2];

    // 为了代码中的简化表示，这里将比较长的变量换成了比较短的变量。该语句在编译
    // 中不会带来额外的运行性能下降。
    int *ldata = lconvex.tplMeta.tplData;
    int *udata = uconvex.tplMeta.tplData;
    int lcnt = lconvex.tplMeta.count;
    int ucnt = uconvex.tplMeta.count;

    // 由每个 Block 的第一个 Thread 计算是否存在重合的最左最右点。
    if (threadIdx.x == 0) {
        // 判断最左点是否重合，对于上半凸壳，最左点存放在其首部，对于下半凸壳，
        // 最左点存放在其尾部。
        if (ldata[0] == udata[2 * (ucnt - 1)] && 
            ldata[1] == udata[2 * (ucnt - 1) + 1]) {
            sameedge[0] = 1;
        } else {
            sameedge[0] = 0;
        }

        // 判断最右点是否重合，对于上半凸壳，最右点存放在其尾部，对于下半凸壳，
        // 最右点存放在其首部。
        if (ldata[2 * (lcnt - 1)] == udata[0] &&
            ldata[2 * (lcnt - 1) + 1] == udata[1]) {
            sameedge[1] = 1;
        } else {
            sameedge[1] = 0;
        }

        // 根据对最左最右点的判断，就可以得到最终凸壳点集的数量，这里用整个 
        // Kernel 的第一个 Thread 写入最终凸壳点集的数量。
        if (blockIdx.x == 0)
            *convexcnt = lcnt + ucnt - sameedge[0] - sameedge[1];
    }

    // 同步 Block 内部的所有线程，使得求解结果对所有的 Thread 可见。
    __syncthreads();

    // 计算当前线程的全局下标。该下标对应于输出凸壳点的下标。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 判断当前线程是对应于下半凸壳和上半凸壳。将上半凸壳放在输出凸壳的前半部
    // 分，下半凸壳放在其后半部分。
    if (idx >= lcnt) {
        // 对于处理上半凸壳，首先计算出当前线程对应的上半凸壳下标，这里还需要经
        // 过最右点重合的校正。
        int inidx = idx - lcnt + sameedge[1];

        // 如果对应的下标是越界的，则直接返回，这里仍需要经过最左点重合的校正。
        if (inidx >= ucnt - sameedge[0])
            return;

        // 将上半凸壳拷贝到整体的凸壳中。
        convex.tplMeta.tplData[2 * idx] = udata[2 * inidx];
        convex.tplMeta.tplData[2 * idx + 1] = udata[2 * inidx + 1];
    } else {
        // 将下半凸壳拷贝到整体的凸壳中。由于上半凸壳内部坐标和整体凸壳的坐标是
        // 一致的，且越界情况通过上面的 if 语句已经屏蔽，故没有进行下标的计算和
        // 判断。
        convex.tplMeta.tplData[2 * idx] = ldata[2 * idx];
        convex.tplMeta.tplData[2 * idx + 1] = ldata[2 * idx + 1];
    }
}

// 宏：FAIL_JOINCONVEX_FREE
// 该宏用于完成下面函数运行出现错误退出前的内存清理工作。
#define FAIL_JOINCONVEX_FREE  do {                         \
        if (tmpconvex != NULL && tmpconvex != convex)      \
            CoordiSetBasicOp::deleteCoordiSet(tmpconvex);  \
        if (convexcntDev != NULL)                          \
            cudaFree(convexcntDev);                        \
    } while (0)

// Host 成员方法：joinConvex（合并凸壳点）
__host__ int ConvexHull::joinConvex(
        CoordiSet *lconvex, CoordiSet *uconvex, CoordiSet *convex)
{
    // 检查指针性参数是否为 NULL。
    if (lconvex == NULL || uconvex == NULL || convex == NULL)
        return NULL_POINTER;

    // 检查输入坐标点是否包含了有效的坐标点数量，如果输入坐标点中点数小于 2，则
    // 无法完成相应的计算工作。
    if (lconvex->count < 2 || lconvex->tplData == NULL || 
        uconvex->count < 2 || uconvex->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;
    cudaError_t cuerrcode;

    // 局部变量，输出凸壳点数量上限，即上下凸壳点数量加和。
    int tmptotal = lconvex->count + uconvex->count;

    // 将下半凸壳点数据拷贝到当前 Device。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(lconvex);
    if (errcode != NO_ERROR)
        return errcode;

    // 将上半凸壳点数据拷贝到当前 Device。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(uconvex);
    if (errcode != NO_ERROR)
        return errcode;

    // 如果输出凸壳点是一个空点集，则为其开辟适当的内存空间，以用于存放最后的凸
    // 壳点。
    if (convex->tplData == NULL) {
        errcode = CoordiSetBasicOp::makeAtCurrentDevice(convex, tmptotal);
        if (errcode != NO_ERROR)
           return errcode;
    }

    // 局部变量。
    CoordiSet *tmpconvex = NULL;  // 临时输出凸壳点集。由于参数给定输出用凸壳点
                                  // 集不一定具有合适数量的存储空间，因此，先用
                                  // 一个临时的凸壳点集存放 Kernel 返回的结果，
                                  // 然后在归放到参数所对应的凸壳点集中。
    int *convexcntDev = NULL;     // 用于存放 Kernel 返回的最终凸壳点数量。

    // 给临时输出凸壳点集初始化。
    if (convex->count < tmptotal || convex->count >= tmptotal * 2) {
        // 如果给定的输出凸壳点集点的数量不合适，则需要重新申请一个凸壳点集并赋
        // 值给临时凸壳点集。
        errcode = CoordiSetBasicOp::newCoordiSet(&tmpconvex);
        if (errcode != NO_ERROR)
            return errcode;

        // 申请后，还需要给该凸壳点集开辟合适的内存空间。
        errcode = CoordiSetBasicOp::makeAtCurrentDevice(tmpconvex, tmptotal);
        if (errcode != NO_ERROR) {
            FAIL_JOINCONVEX_FREE;
            return errcode;
        }
    } else {
        // 如果输出土克点击中点的数量合适，则直接使用输出凸壳点集承接 Kernel 的
        // 输出。
        tmpconvex = convex;
    }

    // 取出坐标点集对应的 CUDA 型数据。
    CoordiSetCuda *lconvexCud = COORDISET_CUDA(lconvex);
    CoordiSetCuda *uconvexCud = COORDISET_CUDA(uconvex);
    CoordiSetCuda *tmpconvexCud = COORDISET_CUDA(tmpconvex);

    // 为 Kernel 输出的凸壳点集数量开辟 Device 内存空间。
    cuerrcode = cudaMalloc((void **)&convexcntDev, sizeof (int));
    if (cuerrcode != cudaSuccess) {
        FAIL_JOINCONVEX_FREE;
        return CUDA_ERROR;
    }

    // 计算启动 Kernel 所需要的 Block 尺寸和数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (tmptotal + blocksize - 1) / blocksize;

    // 启动 Kernel 完成计算。
    _joinConvexKer<<<gridsize, blocksize>>>(*lconvexCud, *uconvexCud, 
                                            *tmpconvexCud, convexcntDev);

    // 检查 Kernel 函数是否执行正确。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_JOINCONVEX_FREE;
        return CUDA_ERROR;
    }

    // 从 Device 内存中读取 Kernel 返回的凸壳点数量，并将其赋值到临时凸壳点集的
    // 坐标点数量中。
    cuerrcode = cudaMemcpy(&(tmpconvex->count), convexcntDev, sizeof (int), 
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        FAIL_JOINCONVEX_FREE;
        return CUDA_ERROR;
    }

    // 如果使用的是临时凸壳点集，则需要将点集从临时凸壳点集中拷贝到输出凸壳点集
    // 中，在拷贝的过程中，输出凸壳点集的坐标点数量会被安全的重新定义。
    if (tmpconvex != convex) {
        errcode = CoordiSetBasicOp::copyToCurrentDevice(tmpconvex, convex);
        if (errcode != NO_ERROR) {
            FAIL_JOINCONVEX_FREE;
            return errcode;
        }

        // 至此，临时凸壳点集的使命完成，清除其占用的内存空间。
        CoordiSetBasicOp::deleteCoordiSet(tmpconvex);
    }

    // 释放 Device 内存空间。
    cudaFree(convexcntDev);

    // 操作结束，返回。
    return NO_ERROR;
}
#undef FAIL_JOINCONVEX_FREE

// FAIL_JOINCONVEXCPU_FREE
// 该宏用于完成下面函数运行出现错误退出前的内存清理工作。
#define FAIL_JOINCONVEXCPU_FREE  do {                         \
        if (tmpconvex != NULL && tmpconvex != convex)      \
            CoordiSetBasicOp::deleteCoordiSet(tmpconvex);  \
    } while (0)

// Host 成员方法：joinConvexCpu（合并凸壳点）
__host__ int ConvexHull::joinConvexCpu(
        CoordiSet *lconvex, CoordiSet *uconvex, CoordiSet *convex)
{
    // 检查指针性参数是否为 NULL。
    if (lconvex == NULL || uconvex == NULL || convex == NULL)
        return NULL_POINTER;

    // 检查输入坐标点是否包含了有效的坐标点数量，如果输入坐标点中点数小于 2，则
    // 无法完成相应的计算工作。
    if (lconvex->count < 2 || lconvex->tplData == NULL || 
        uconvex->count < 2 || uconvex->tplData == NULL)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 局部变量，输出凸壳点数量上限，即上下凸壳点数量加和。
    int tmptotal = lconvex->count + uconvex->count;

    // 将下半凸壳点数据拷贝到当前 Host。
    errcode = CoordiSetBasicOp::copyToHost(lconvex);
    if (errcode != NO_ERROR)
        return errcode;

    // 将上半凸壳点数据拷贝到 Host。
    errcode = CoordiSetBasicOp::copyToHost(uconvex);
    if (errcode != NO_ERROR)
        return errcode;

    // 如果输出凸壳点是一个空点集，则为其开辟适当的内存空间，以用于存放最后的凸
    // 壳点。
    if (convex->tplData == NULL) {
        errcode = CoordiSetBasicOp::makeAtHost(convex, tmptotal);
        if (errcode != NO_ERROR)
           return errcode;
    }

    // 局部变量。
    CoordiSet *tmpconvex = NULL;  // 临时输出凸壳点集。由于参数给定输出用凸壳点
                                  // 集不一定具有合适数量的存储空间，因此，先用
                                  // 一个临时的凸壳点集存放 Kernel 返回的结果，
                                  // 然后在归放到参数所对应的凸壳点集中。
    int convexcnt = 0;     // 用于存放最终凸壳点数量。

    // 给临时输出凸壳点集初始化。
    if (convex->count < tmptotal || convex->count >= tmptotal * 2) {
        // 如果给定的输出凸壳点集点的数量不合适，则需要重新申请一个凸壳点集并赋
        // 值给临时凸壳点集。
        errcode = CoordiSetBasicOp::newCoordiSet(&tmpconvex);
        if (errcode != NO_ERROR)
            return errcode;

        // 申请后，还需要给该凸壳点集开辟合适的内存空间。
        errcode = CoordiSetBasicOp::makeAtHost(tmpconvex, tmptotal);
        if (errcode != NO_ERROR) {
            FAIL_JOINCONVEXCPU_FREE;
            return errcode;
        }
    } else {
        // 如果输出土克点击中点的数量合适，则直接使用输出凸壳点集承接 Kernel 的
        // 输出。
        tmpconvex = convex;
    }

    // 取出坐标点集对应的 CUDA 型数据。
    CoordiSetCuda *lconvexCud = COORDISET_CUDA(lconvex);
    CoordiSetCuda *uconvexCud = COORDISET_CUDA(uconvex);
    CoordiSetCuda *tmpconvexCud = COORDISET_CUDA(tmpconvex);

    // 共享内存，用来记录上下凸壳在最左最右点处是否重合，如果重合，应该在整合后
    // 的坐标点中排除重合的点。其中，[0] 表示最左点，[1] 表示最右点。用 1 表示
    // 有重合的点，用 0 表示没有重合的点。
    int sameedge[2];

    // 为了代码中的简化表示，这里将比较长的变量换成了比较短的变量。该语句在编译
    // 中不会带来额外的运行性能下降。
    int *ldata = (*lconvexCud).tplMeta.tplData;
    int *udata = (*uconvexCud).tplMeta.tplData;
    int lcnt = (*lconvexCud).tplMeta.count;
    int ucnt = (*uconvexCud).tplMeta.count;

    // 判断最左点是否重合，对于上半凸壳，最左点存放在其首部，对于下半凸壳，
    // 最左点存放在其尾部。
    if (ldata[0] == udata[2 * (ucnt - 1)] && 
        ldata[1] == udata[2 * (ucnt - 1) + 1]) {
        sameedge[0] = 1;
    } else {
        sameedge[0] = 0;
    }

    // 判断最右点是否重合，对于上半凸壳，最右点存放在其尾部，对于下半凸壳，
    // 最右点存放在其首部。
    if (ldata[2 * (lcnt - 1)] == udata[0] &&
        ldata[2 * (lcnt - 1) + 1] == udata[1]) {
        sameedge[1] = 1;
    } else {
        sameedge[1] = 0;
    }

    // 根据对最左最右点的判断，就可以得到最终凸壳点集的数量
    convexcnt = lcnt + ucnt - sameedge[0] - sameedge[1];

    for (int idx = 0; idx < tmptotal; idx++) {
        // 判断当前线程是对应于下半凸壳和上半凸壳。将上半凸壳放在输出凸壳的前半
        // 部分，下半凸壳放在其后半部分。
        if (idx >= lcnt) {
            // 对于处理上半凸壳，首先计算出当前线程对应的上半凸壳下标，这里还需
            // 要经过最右点重合的校正。
            int inidx = idx - lcnt + sameedge[1];

            // 如果对应的下标是不越界的
            if (inidx < ucnt - sameedge[0]) {
                // 将上半凸壳拷贝到整体的凸壳中。
                (*tmpconvexCud).tplMeta.tplData[2 * idx] = udata[2 * inidx];
                (*tmpconvexCud).tplMeta.tplData[2 * idx + 1] =
                        udata[2 * inidx + 1];
            }
        } else {
            // 将下半凸壳拷贝到整体的凸壳中。由于上半凸壳内部坐标和整体凸壳的坐
            // 标是一致的，且越界情况通过上面的 if 语句已经屏蔽，故没有进行下标
            // 的计算和判断。
            (*tmpconvexCud).tplMeta.tplData[2 * idx] = ldata[2 * idx];
            (*tmpconvexCud).tplMeta.tplData[2 * idx + 1] = ldata[2 * idx + 1];
        }
    }

    // 从 Device 内存中读取 Kernel 返回的凸壳点数量，并将其赋值到临时凸壳点集的
    // 坐标点数量中。
    tmpconvex->count = convexcnt;

    // 如果使用的是临时凸壳点集，则需要将点集从临时凸壳点集中拷贝到输出凸壳点集
    // 中，在拷贝的过程中，输出凸壳点集的坐标点数量会被安全的重新定义。
    if (tmpconvex != convex) {
        errcode = CoordiSetBasicOp::copyToHost(tmpconvex, convex);
        if (errcode != NO_ERROR) {
            FAIL_JOINCONVEXCPU_FREE;
            return errcode;
        }

        // 至此，临时凸壳点集的使命完成，清除其占用的内存空间。
        CoordiSetBasicOp::deleteCoordiSet(tmpconvex);
    }

    // 操作结束，返回。
    return NO_ERROR;
}
#undef FAIL_JOINCONVEXCPU_FREE

// 宏：FAIL_CONVEXHULL_FREE
// 该宏用于完成下面函数运行出现错误退出前的内存清理工作。
#define FAIL_CONVEXHULL_FREE  do {                       \
        if (lconvex != NULL)                             \
            CoordiSetBasicOp::deleteCoordiSet(lconvex);  \
        if (uconvex != NULL)                             \
            CoordiSetBasicOp::deleteCoordiSet(uconvex);  \
    } while (0)

// Host 成员方法：convexHullCpu（求一个点集对应的凸壳点集）
__host__ int ConvexHull::convexHullCpu(CoordiSet *inputcst, CoordiSet *convex)
{
    // 检查指针性参数是否为 NULL。
    if (inputcst == NULL || convex == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点，则报错退出。
    if (inputcst->count < 1 || inputcst->tplData == NULL)
        return INVALID_DATA;

    // 如果输入点集中只有一个点，那么该点直接输出，作为凸壳点。
    if (inputcst->count == 1)
        return CoordiSetBasicOp::copyToHost(inputcst, convex);

    // 如果输入点集中只有两个点，则直接将其下半凸壳输出（显然此时上半凸壳也是这
    // 两个点）
    if (inputcst->count == 2)
        return this->convexHullIterCpu(inputcst, convex, true);

    // 局部变量，错误码。
    int errcode;

    // 局部变量，下半凸壳和上半凸壳点集变量。
    CoordiSet *lconvex = NULL;
    CoordiSet *uconvex = NULL;

    // 申请一个临时点集，用来存放下半凸壳。
    errcode = CoordiSetBasicOp::newCoordiSet(&lconvex);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 申请一个临时点集，用来存放上半凸壳。
    errcode = CoordiSetBasicOp::newCoordiSet(&uconvex);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    cout << endl;
#ifdef CH_DEBUG_CPU_PRINT
    cout << "[convexHullCpu]convexHullIterCpu upper begin" << endl;
#endif
    // 调用凸壳迭代，求输入点集的下半凸壳。
    errcode = this->convexHullIterCpu(inputcst, lconvex, true);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }
#ifdef CH_DEBUG_CPU_PRINT
    cout << "[convexHullCpu]convexHullIterCpu upper end" << endl;
    cout << endl;
    cout << "[convexHullCpu]convexHullIterCpu lower begin" << endl;
#endif

    // 调用凸壳迭代，求输入点集的上半凸壳。
    errcode = this->convexHullIterCpu(inputcst, uconvex, false);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

#ifdef CH_DEBUG_CPU_PRINT
    cout << "[convexHullCpu]convexHullIterCpu lower end" << endl;
    cout << endl;
#endif

    // 调用合并两个凸壳的函数，将下半凸壳和上半凸壳粘在一起。
    errcode = this->joinConvexCpu(lconvex, uconvex, convex);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 清除临时申请的两个坐标点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(lconvex);
    CoordiSetBasicOp::deleteCoordiSet(uconvex);

    // 处理完毕，退出
    return NO_ERROR;
}

// Host 成员方法：convexHull（求一个点集对应的凸壳点集）
__host__ int ConvexHull::convexHull(CoordiSet *inputcst, CoordiSet *convex)
{
    // 检查指针性参数是否为 NULL。
    if (inputcst == NULL || convex == NULL)
        return NULL_POINTER;

    // 如果输入点集中不包含任何点，则报错退出。
    if (inputcst->count < 1 || inputcst->tplData == NULL)
        return INVALID_DATA;

    // 如果输入点集中只有一个点，那么该点直接输出，作为凸壳点。
    if (inputcst->count == 1)
        return CoordiSetBasicOp::copyToCurrentDevice(inputcst, convex);

    // 如果输入点集中只有两个点，则直接将其下半凸壳输出（显然此时上半凸壳也是这
    // 两个点）
    if (inputcst->count == 2)
        return this->convexHullIter(inputcst, convex, true);

    cout << "GPU convex 1" << endl;
    // 局部变量，错误码。
    int errcode;

    // 局部变量，下半凸壳和上半凸壳点集变量。
    CoordiSet *lconvex = NULL;
    CoordiSet *uconvex = NULL;

    // 申请一个临时点集，用来存放下半凸壳。
    errcode = CoordiSetBasicOp::newCoordiSet(&lconvex);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 申请一个临时点集，用来存放上半凸壳。
    errcode = CoordiSetBasicOp::newCoordiSet(&uconvex);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    cout << "GPU convex lower" << endl;
    // 调用凸壳迭代，求输入点集的下半凸壳。
    errcode = this->convexHullIter(inputcst, lconvex, true);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }
    cout << "GPU convex lower cnt is " << lconvex->count << endl;

    cout << "GPU convex up" << endl;
    // 调用凸壳迭代，求输入点集的上半凸壳。
    errcode = this->convexHullIter(inputcst, uconvex, false);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }
    cout << "GPU convex up cnt is " << uconvex->count << endl;

    cout << "GPU joinConvex" << endl;
    // 调用合并两个凸壳的函数，将下半凸壳和上半凸壳粘在一起。
    errcode = this->joinConvex(lconvex, uconvex, convex);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULL_FREE;
        return errcode;
    }

    // 清除临时申请的两个坐标点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(lconvex);
    CoordiSetBasicOp::deleteCoordiSet(uconvex);

    // 处理完毕，退出
    return NO_ERROR;
}
#undef FAIL_CONVEXHULL_FREE

// 宏：FAIL_CONVEXHULLONIMG_FREE
// 该宏用于完成下面函数运行出现错误退出前的内存清理工作。
#define FAIL_CONVEXHULLONIMG_FREE  do {              \
        if (cst != NULL)                             \
            CoordiSetBasicOp::deleteCoordiSet(cst);  \
    } while (0)

// Host 成员方法：convexHullCpu（求图像中阈值给定的对象对应的凸壳点集）
__host__ int ConvexHull::convexHullCpu(Image *inimg, CoordiSet *convex)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || convex == NULL)
        return NULL_POINTER;

    // 局部变量，错误码。
    int errcode;

    // 新建点集。
    CoordiSet *cst;

    // 构造点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&cst);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULLONIMG_FREE;
        return errcode;
    }

    // 调用图像转点集的函数。
    errcode = this->imgCvt.imgConvertToCst(inimg, cst);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULLONIMG_FREE;
        return errcode;
    }

    // 调用求给定点集的凸壳函数。
    errcode = convexHullCpu(cst, convex);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULLONIMG_FREE;
        return errcode;
    }

    // 清除点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(cst);

    // 退出。
    return NO_ERROR;
}

// Host 成员方法：convexHull（求图像中阈值给定的对象对应的凸壳点集）
__host__ int ConvexHull::convexHull(Image *inimg, CoordiSet *convex)
{
    // 检查输入图像和输出包围矩形是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || convex == NULL)
        return NULL_POINTER;

    // 局部变量，错误码。
    int errcode;

    // 新建点集。
    CoordiSet *cst;

    // 构造点集。
    errcode = CoordiSetBasicOp::newCoordiSet(&cst);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULLONIMG_FREE;
        return errcode;
    }

    // 调用图像转点集的函数。
    errcode = this->imgCvt.imgConvertToCst(inimg, cst);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULLONIMG_FREE;
        return errcode;
    }

    // 调用求给定点集的凸壳函数。
    errcode = convexHull(cst, convex);
    if (errcode != NO_ERROR) {
        FAIL_CONVEXHULLONIMG_FREE;
        return errcode;
    }

    // 清除点集所占用的内容空间。
    CoordiSetBasicOp::deleteCoordiSet(cst);

    // 退出。
    return NO_ERROR;
}
#undef FAIL_CONVEXHULLONIMG_FREE

