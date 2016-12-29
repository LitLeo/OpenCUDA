// DouglasPeucker.cu
// 道格拉斯普克算法简化曲线上的点。

#include "DouglasPeucker.h"

// 宏：DEF_BLOCK_1D
// 定义了默认的 1D Block 尺寸。
#define DEF_BLOCK_1D  512

// 宏：CH_LARGE_ENOUGH
// 定义了一个足够大的正整数，该整数在使用过程中被认为是无穷大。
#define CH_LARGE_ENOUGH  ((1 << 30) - 1)

// Kernel 函数：_initLabelAryKer（初始化 LABEL 数组）
// 在迭代前初始化 LABEL 数组，初始化后的 LABEL 数组要求除最后一个元素为 1 
// 以外，其他的元素皆为 0。
static __global__ void  // Kernel 函数无返回值
_initLabelAryKer(
        int label[],    // 待初始化的 LABEL 数组。
        int cstcnt      // LABEL 数组长度。
);

// Kernel 函数: _updateDistKer（更新点集的垂距信息）
// 根据目前已知的结果集上的点集和区域的标签值，找出当前每个点所在区域的起止点，
// 根据点到直线的垂距公式，计算点集的附带数据：点到当前所在区域的起止点构成的直
// 线的垂直距离。
static __global__ void    // Kernel 函数无返回值
_updateDistKer(
        int cst[],        // 输入点集，也是输出点集，更新点集的 
                                  // attachData，也就是垂距的信息。
        int cornercst[],  // 目前已知结果点集，即每段的最值点信息。
        int label[],      // 输入，当前点集的区域标签值数组。
        int cstcnt,       // 输入，当前点的数量。
        float dis[]       // 记录每一个点的垂距
);

// Kernel 函数: _updateFoundInfoKer（更新新发现角点信息）
// 根据分段扫描后得到的点集信息，更新当前区域是否有新发现的角点，更新目前
// 已知的结果集的点的位置索引。
static __global__ void     // Kernel 函数无返回值
_updateFoundInfoKer(
        int label[],       // 输入，当前点集的区域标签值数组。   
        float dist[],      // 输入数组，所有点的垂距，即坐标点集数据结构中的 
                           // attachedData 域。
        int maxdistidx[],  // 输入，分段扫描后，当前位置记录的本段目前已知的最
                           // 大垂距点的位置索引数组。
        int cstcnt,        // 坐标点的数量。
        int foundflag[],   // 输出数组，如果当前区域内找到新的点，标志位置 1。
        int startidx[],    // 输出，目前已知的结果点集中点的位置索引数组，也相当
                           // 于当前每段上的起始位置的索引数组。
        float threshold,   // 垂距的阈值
        int foundidx[]     // 新找到的角点的一维索引
);

// Kernel 函数: _updateCornerCstKer（生成新结果点集）
// 根据分段扫描后得到的点集信息，和每段上是否发现新点的信息，生成新点集。
static __global__ void      // Kernel 函数无返回值
_updateCornerCstKer(
        int cst[],          // 输入点集
        int cornercst[],    // 目前已知结果点集，即每段的最值点信息。
        int foundflag[],    // 输入，当前区域内有新发现点的标志位数组，如果当前
                            // 区域内找到新的点，标志位置 1。
        int foundacc[],     // 输入，偏移量数组，即当前区域内有新发现点的标志位
                            // 的累加值。
        int startidx[],     // 输入，目前已知的点的位置索引数组，
                            // 也相当于当前每段上的起始位置的索引数组。
        int maxdistidx[],   // 输入，分段扫描后，当前位置记录的本段目前已知的最
                            // 大垂距点的位置索引数组
        int cornercnt,      // 当前点的数量。
        int newcornercst[]  // 输出，更新后的目前已知结果点集即每段的最值点信息。
);

// Kernel 函数: _updateLabelKer（更新 Label 值）
// 根据已得到的结果点，更新 Label 值。
static __global__ void   // Kernel 函数无返回值
_updateLabelKer(
        int label[],     // 输入，当前点集的区域标签值数组。
        int cstcnt,      // 坐标点的数量。
        int foundidx[],  // 新找到的点的一维索引
        int foundacc[],  // 输入，偏移量数组即当前区域内新发现点的标志位累加值。 
        int tmplabel[]   // 新的标签值
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
__host__ int DouglasPeucker::initLabelAry(int label[], int cstcnt)
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


// Kernel 函数: _updateDistKer（更新点集的垂距信息）
static __global__ void _updateDistKer(
        int cst[], int cornercst[], int label[], 
        int cstcnt, /*int foundflagDev[], int oldlabelDev[], */float dis[])
{
    // 记录了本 Kernel 所使用到的共享内存中各个下标所存储的数据的含义。其中，
    // SIDX_BLK_CNT 表示当前 Block 所需要处理的坐标点的数量，由于坐标点的数量不
    // 一定能够被 BlockDim 整除，因此最后一个 Block 所处理的坐标点的数量要小于 
    // BlockDim。
    // SIDX_BLK_LABEL_LOW 和 SIDX_BLK_LABEL_UP 用来存当前 Block 中所加载的点集
    // 的区域标签值的上下界。根据这个上下界，可以计算出当前点所在区域的起止
    // 点，从而根据这两点确定的直线计算当前点的垂距。
    // 从下标为 SIDX_BLK_CST 开始的其后的所有共享内存空间存储了当前 Block 中的
    // 点集坐标。坐标集中第 i 个点对应的数组下标为 2 * i 和 2 * i + 1，其中下标
    // 为 2 * i 的数据表示该点的横坐标，下标为 2 * i + 1 的数据表示该点的纵坐
    // 标。
#define SIDX_BLK_CNT        0
#define SIDX_BLK_LABEL_LOW  1
#define SIDX_BLK_LABEL_UP   2
#define SIDX_BLK_CORNER     3    

    // 共享内存的声明。
    extern __shared__ int shdmem[];

    // 基准索引。表示当前 Block 的起始位置索引。
    int baseidx = blockIdx.x * blockDim.x;
    // 全局索引。
    int idx = baseidx + threadIdx.x;
    // 如果当前线程的全局下标越界，则直接返回，因为他没有对应的所要处理坐标点。
    if (idx >= cstcnt)
        return;

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
    int *cornerShd = &shdmem[SIDX_BLK_CORNER];

    // 加载当前 Block 中所用到的 LABEL 所对应的起止点，这两个点构成的直线可用来
    // 衡量各点的垂距并以此推算出下一轮的角点。将所用到的点加载的 Shared Memory 
    // 中也没有逻辑上的目的，仅仅是为了下一步计算时访存时间的缩短。
    if (threadIdx.x < labelupper - labellower + 1) {
        cornerShd[2 * threadIdx.x] =
                cornercst[2 * (labellower + threadIdx.x)];
        cornerShd[2 * threadIdx.x + 1] =
                cornercst[2 * (labellower + threadIdx.x) + 1];
    }

    // Block 内部同步，使得上面所有的数据加载对 Block 内的所有 Thread 可见。下
    // 面的代码就正式的投入计算了。
    __syncthreads();

    if (idx == cstcnt - 1) {
        dis[idx] = 0.0f;
        return;
    }
    // 计算当前点的坐标和区域标签值。
    int curx = cst[2 * idx];
    int cury = cst[2 * idx + 1];
    int curlabelidx = 2 * (label[idx] - labellower);

    // 计算当前 LABEL 区域的最左点的坐标。
    int leftx = cornerShd[curlabelidx++];
    int lefty = cornerShd[curlabelidx++];

    // 计算当前 LABEL 区域的最右点的坐标。
    int rightx = cornerShd[curlabelidx++];
    int righty = cornerShd[curlabelidx  ];

    // 如果当前点就是角点，那么不需要计算直接赋值退出就可以了。
    if ((curx == leftx && cury == lefty) || 
        (curx == rightx && cury == righty)) {
        dis[idx] = 0.0f;
        return;
    }

   // 计算垂距，该计算通过起止形成的直线作为垂距求解的依据
    float k, dist,b, temp;
   if (rightx == leftx) {
        dist = fabsf(curx -  leftx);
    } else {
        k = (righty - lefty) * 1.0f / (rightx - leftx);
        b = lefty - k * leftx;
        temp = fabsf(k * curx - cury + b);
        dist = temp / sqrtf(k * k + 1);
    }

    // 将垂距信息更新到 Global 内存中作为输出。
    dis[idx] = dist;

#undef SIDX_BLK_CNT
#undef SIDX_BLK_LABEL_LOW
#undef SIDX_BLK_LABEL_UP
#undef SIDX_BLK_CORNER    
}

// 成员方法：updateDist（更新坐标点集垂距）
__host__ int DouglasPeucker::updateDist(
        int *cst, int *cornercst, int label[], int cstcnt, float dis[])
{
    // 检查输入坐标集，输出坐标集是否为空。
    if (cornercst == NULL || cst == NULL || label == NULL || dis == NULL)
        return NULL_POINTER;

    // 检查当前点的数量，小于等于 0 则无效数据。
    if (cstcnt <= 0)
        return INVALID_DATA;


    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量，以及所需要的 Shared 
    // Memory 的数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (cstcnt + blocksize - 1) / blocksize;
    size_t sharedsize = (3 + 2 * blocksize) * sizeof (int);

    // 调用更新点集的垂距信息的核函数，计算每个点的垂距，更新负垂距标志数组。
    _updateDistKer<<<gridsize, blocksize, sharedsize>>>(
            cst, cornercst, label, cstcnt,/* foundflagDev, oldlabelDev, */dis);

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// Kernel 函数: _updateFoundInfoKer（更新新发现点信息）
static __global__ void _updateFoundInfoKer(
        int *label, float *dist, int *maxdistidx, int cstcnt,
        int *foundflag, int *startidx, float threshold, int *foundidx)
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
        foundidx[curlabel] = CH_LARGE_ENOUGH;
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
    // 域内发现了点。为了健壮性的考虑，这里将 0 写为 1.0e-6。
    foundflag[curlabel] = (curmaxdist >= threshold) ? 1 : 0;
    foundidx[curlabel] = (foundflag[curlabel] == 1 ?
                          curmaxdistidx : CH_LARGE_ENOUGH);

    // 更新下一个 LABEL 区域的起始下标。由于当前 Thread 是当前 LABEL 区域的最后
    // 一个，因此下一个 LABEL 区域的起始下标为当前 Thread 全局索引加 1。
    startidx[curlabel + 1] = idx + 1;
}

// 成员方法: updateFoundInfo（更新新发现点信息）
__host__ int DouglasPeucker::updateFoundInfo(
        int label[], float dist[], int maxdistidx[],
        int cstcnt, int foundflag[], int startidx[],
        float threshold, int foundidx[])
{
    // 检查所有的输入指针或数组是否为 NULL，如果存在一个为 NULL 则报错退出。
    if (label == NULL || dist == NULL || maxdistidx == NULL ||
        foundflag == NULL || startidx == NULL || foundidx == NULL)
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
            label, dist, maxdistidx, cstcnt, foundflag,
            startidx, threshold, foundidx);

    // 判断核函数是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}


// Kernel 函数: _updateCornerCstKer（生成新点集）
static __global__ void _updateCornerCstKer(
        int cst[], int cornercst[], int foundflag[],
        int foundacc[], int startidx[], int maxdistidx[], int cornercnt,
        int newcornercst[])
{
    // 计算当前 Thread 的全局索引。本 Kernel 中，每个线程都对应于一个 LABEL 区
    // 域，对于发现了新点的 LABEL 区域，则需要将原来这个 LABEL 点和新发现的点
    // 同时拷贝到新的点集中。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果该 Thread 对应的时越界数据，则直接返回，不进行任何处理。
    if (idx >= cornercnt)
        return;

    // 计算原来的角点在新角点集中的下标，由于前面的 LABEL 区域共产生了 
    // foundacc[idx] 个角点，因此，下标应相较于原来的下标（idx）增加了相应的
    // 数量。
    int newidx = idx + foundacc[idx];

    // 将这个点的坐标从原来的点集中拷贝到新的点集中。
    newcornercst[2 * newidx] = cornercst[2 * idx];
    newcornercst[2 * newidx + 1] = cornercst[2 * idx + 1];

    // 如果当前 LABEL 区域中没有发现新的点，则只需要拷贝原有的点到新的点集中。
    if (foundflag[idx] == 0)
        return;

    // 计算新发现的点在点集中的下标和该点对应的坐标点集中的下标。由于最大垂距点
    // 下标数组是记录的 Scanning 操作的结果，因此正确的结果存放再该LABEL 区域 
    // 最后一个下标处。
    newidx++;
    int cstidx = maxdistidx[startidx[idx + 1] - 1];

    // 将新发现的凸壳点从坐标点集中拷贝到新的凸壳点集中。
    newcornercst[2 * newidx] = cst[2 * cstidx];
    newcornercst[2 * newidx + 1] = cst[2 * cstidx + 1];
}

// Host 成员方法：updateCornerCst（生成新点集）
__host__ int DouglasPeucker::updateCornerCst(
        int *cst, int *cornercst, int foundflag[],
        int foundacc[], int startidx[], int maxdistidx[], int cornercnt,
        int *newcornercst)
{
    // 检查参数中所有的指针和数组是否为空。
    if (cst == NULL || cornercst == NULL || foundacc == NULL ||
        foundflag == NULL || startidx == NULL || maxdistidx == NULL ||
        newcornercst == NULL)
        return NULL_POINTER;

    // 检查当前角点的数量，小于等于 0 则无效数据。
    if (cornercnt <= 0)
        return INVALID_DATA;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    // 矩阵方法分段扫描版本线程块大小。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (cornercnt + blocksize - 1) / blocksize;

    // 调用 Kernel 函数完成计算。
    _updateCornerCstKer<<<gridsize, blocksize>>>(
            cst, cornercst, foundflag, foundacc, startidx,
            maxdistidx, cornercnt, newcornercst);
    // 判断 Kernel 函数是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// Kernel 函数: _updateLabelKer（更新标签值）
static __global__ void _updateLabelKer(
        int label[], int cstcnt, 
        int foundidx[],  int foundacc[], int tmplabel[])
{
    // 记录了本 Kernel 所使用到的共享内存中各个下标所存储的数据的含义。其中，
    // SIDX_BLK_LABEL_LOW 和 SIDX_BLK_LABEL_UP 用来存当前 Block 中所加载的点集
    // 的区域标签值的上下界。根据这个上下界，可以计算出当前点所在区域的起止
    // 点，从而根据这两点确定的直线计算当前点的垂距。
    // 从下标为 SIDX_BLK_CORNER_X 开始的其后的所有共享内存空间存储了当前 Block 
    // 所处理的所有的新点的 X 坐标。
#define SIDX_BLK_LABEL_LOW  0
#define SIDX_BLK_LABEL_UP   1
#define SIDX_BLK_CORNER_X   2
#define SIDX_BLK_FOUND_ACC  2 + blockDim.x
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

    // 并不是所有的 LABEL 区域都会在该论迭代中发现新点。该值要求非常的大，因为没
    // 有发现新凸壳点的区域，相当于所有的坐标点放在左侧。
#define LP_DUMMY_CVXX  CH_LARGE_ENOUGH

    // 将新点的 X 坐标存储 Shared Memory 提取出，用一个指针来表示，这样的写
    // 法是为了代码更加易于理解。
    int *newidx = &shdmem[SIDX_BLK_CORNER_X];
    int *foundaccShd = &shdmem[SIDX_BLK_FOUND_ACC];
    // 在 Shared Memory 中初始化新点（中心点）的 X 坐标。
    if (threadIdx.x < labelupper - labellower + 1) {
        // 计算新点在新的点集中的下标。
        int labelidx = threadIdx.x + labellower;
        newidx[threadIdx.x] = foundidx[labelidx];
        // 从 Global Memory 中读取新点的累加值。
        foundaccShd[threadIdx.x] = foundacc[threadIdx.x + labellower];
    }

    // 同步 Block 内的所有 Thread，是的上述所有初始化计算对所有 Thread 可见。
    __syncthreads();

    // 如果当前 Thread 处理的是越界范围，则直接返回不进行任何处理。
    if (idx >= cstcnt)
        return;

    // 读取当前坐标点所对应的 LABEL 值（经过校正的，表示 Shared Memory 中的下
    // 标）。
    int curlabel = label[idx] - labellower;

    // 对于所有垂距大于等于 0，且 x 坐标小于中心点坐标时认为该点在中心点左侧。
    if (idx < newidx[curlabel]) {
        tmplabel[idx] = label[idx] + foundaccShd[curlabel];
    }
    else
        tmplabel[idx] = label[idx] + foundaccShd[curlabel] + 1;

    // 清除函数内部的宏定义，防止同后面的函数造成冲突。
#undef LP_TMPX_DUMMY 
#undef SIDX_BLK_LABEL_LOW
#undef SIDX_BLK_LABEL_UP
#undef SIDX_BLK_CORNER_X
#undef SIDX_BLK_FOUND_ACC
}

// Host 成员方法：updateLabel（标记左侧点）
__host__ int DouglasPeucker::updateLabel(
        int label[], int cstcnt, int foundidx[],  int foundacc[], int tmplabel[])
{
    // 检查参数中所有的指针和变量是否为空。
    if (label == NULL || foundacc == NULL || foundidx == NULL || tmplabel == NULL)
        return NULL_POINTER;

    // 检查当前点的数量，小于等于 0 则无效数据。
    if (cstcnt <= 0)
        return INVALID_DATA;

    // 计算 Kernel 函数所需要的 Block 尺寸和数量，以及每个 Block 所使用的 
    // Shared Memory 的数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (cstcnt + blocksize - 1) / blocksize;
    size_t sharedsize = (2 + 2 * blocksize) * sizeof (int);

    // 调用 Kernel 函数，完成计算。
    _updateLabelKer<<<gridsize, blocksize, sharedsize>>>(
            label, cstcnt, foundidx, foundacc, tmplabel);

    // 判断 Kernel 函数运行是否出错。
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕退出。
    return NO_ERROR;
}

// 宏：FAIL_CORNER_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_CORNER_FREE  do {                                \
        if (tmpmemDev != NULL)                                \
            cudaFree(tmpmemDev);                              \
    } while (0)

// 成员方法：cornerHullIter（迭代法求凸壳上的点集）
__host__ int DouglasPeucker::douglasIter(
        int *inputcst, int *cornercst, float threshold, int count, int *cornerpnt)
{
    // 检查输入坐标集，输出坐标集是否为空。
    if (inputcst == NULL || cornercst == NULL)
        return NULL_POINTER;

    // 局部变量
    cudaError_t cuerrcode;  // CUDA 函数调用返回的错误码
    int errcode;            // 调用函数返回的错误码

    // 定义扫描所用的二元操作符。
    add_class<int> add;

    int cornercnt = 2;             // 当前角点的数量，由于迭代开始时，已经实
                                   // 现找到了点集中的最左和最有两点作为角
                                   // 点，因此这里直接赋值为 2。
    int foundcnt;                  // 当前迭代时找到的新点的数量，这一数量
                                   // 并不包含往次所找到的点。

    int *tmpmemDev = NULL;           // 存放中间变量的 Device 内存空间。
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
    CHI_DATA_DECLARE(label, int,         // 记录当前迭代中每个像素点所在的
                     count);             // LABEL 区域。
    CHI_DATA_DECLARE(maxdistidx, int,    // 记录当前迭代中每个坐标点前面的所
                     count);             // 有点中和其在同一个 LABEL 区域的
                                         // 所有点中具有最大垂距的下标。
    CHI_DATA_DECLARE(foundflag, int,     // 记录当前迭代中各个 LABEL 区域是
                     count);             // 否找到了新点。
    CHI_DATA_DECLARE(foundidx, int,      // 记录当前迭代中各个 LABEL 区域是
                     count);             // 否找到了新点。
    CHI_DATA_DECLARE(foundacc, int,      // 记录当前迭代中每个 LABEL 区域其
                     count + 1);         // 前面的所有 LABEL 区域共找到的新
                                         // 点的数量。该值用于计算各个点（无
                                         // 论是旧的还是新的）在新点集中的新下标。

    CHI_DATA_DECLARE(startidx, int,      // 记录每个 LABEL 区域在坐标点集中
                     count);             // 的起始下标
    CHI_DATA_DECLARE(tmplabel, int,      // 记录新的标签值
                     count);
    CHI_DATA_DECLARE(tmpcstin, int,      // 迭代过程中的临时数组，存放输入点集。
                     count * 2);
    CHI_DATA_DECLARE(tmpcornerin, int,   // 迭代过程中的临时数组，存放截止上次迭
                     count * 2);         // 代已经找到的结果点集。
    CHI_DATA_DECLARE(tmpcornerout, int,  // 迭代过程中的临时数组，存放本次找到的
                     count * 2);         // 结果点集。
    CHI_DATA_DECLARE(dist, float,        // 存放垂距的数组
                     count);
    CHI_DATA_DECLARE(tmpmaxdist, float,  // 存放分段扫描的结果值。
                     count * 2);
    // 消除中间变量声明器这个宏，防止后续步骤的命名冲突。
#undef CHI_DATA_DECLARE
    // 中间变量申请 Device 内存空间，并将这些空间分配给各个中间变量。
    cuerrcode = cudaMalloc((void **)&tmpmemDev, datasize);
    if (cuerrcode != cudaSuccess) {
        FAIL_CORNER_FREE;
        return CUDA_ERROR;
    }

    // 为各个中间变量分配内存空间，采用这种一次申请一个大空间的做法是为了减少申
    // 请内存的开销，同时也减少因内存对齐导致的内存浪费。
    labelDev        = tmpmemDev;
    maxdistidxDev   = labelDev + labelcnt;
    foundflagDev    = maxdistidxDev + maxdistidxcnt;
    foundidxDev     = foundflagDev + foundflagcnt;
    foundaccDev     = foundidxDev + foundidxcnt;
    startidxDev     = foundaccDev + foundacccnt;
    tmplabelDev     = startidxDev + startidxcnt;
    tmpcstinDev     = tmplabelDev + tmplabelcnt;
    tmpcornerinDev  = tmpcstinDev + tmpcstincnt;     
    tmpcorneroutDev = tmpcornerinDev + tmpcornerincnt;
    distDev         = (float *)tmpcorneroutDev + tmpcorneroutcnt;
    tmpmaxdistDev   = distDev + distcnt;

    // 调用 LABEL 初始化函数，完成 LABEL 初始化。初始化后，除最后一个元素为 1 
    // 外，其余元素皆为 0。
    errcode = this->initLabelAry(labelDev, count);
    if (errcode != NO_ERROR) {
        FAIL_CORNER_FREE;
        return errcode;
    }

    // 初始化迭代过程中使用的坐标点集，这里一共需要使用到两个坐标点集，为了不破
    // 坏输入坐标点集，这里在迭代过程中我们使用内部申请的坐标点集。

    // 一个临时数组，存放的是曲线的首尾坐标，用来初始化结果点集。
    int temp[4]= {inputcst[0], inputcst[1],
                  inputcst[2 * (count - 1)],
                  inputcst[2 * (count - 1) + 1]};
    // 为 tmpcstinDev 赋初值。
    cudaMemcpy(tmpcstinDev, inputcst, count * sizeof(int) * 2,
               cudaMemcpyHostToDevice);

    // 初始化结果点集。
    cuerrcode = cudaMemcpy(tmpcornerinDev, temp, 
                           sizeof (int) * 4, cudaMemcpyHostToDevice);
    if (cuerrcode != cudaSuccess) {
        FAIL_CORNER_FREE;
        return CUDA_ERROR;
    }
    // 所有的初始化过程至此全部完毕，开始进行迭代。每次迭代都需要重新计算坐标点
    // 在其 LABEL 区域内的垂距，然后根据垂距信息判断每个 LABEL 区域内是否存在新
    // 的凸壳点（如果有需要确定是哪一个点），之后根据这个新发现的角点点，计算所
    // 有坐标点在下一轮迭代中的下标。迭代的过程知道无法在从当前所有的 LABEL 区域
    // 内找到新的点为止。

    while (count >= cornercnt) {
        // 调用更新垂距函数。更新点集中每个点的垂距值。
        errcode = this->updateDist(tmpcstinDev, tmpcornerinDev, labelDev, count,
                                   distDev);
        if (errcode != NO_ERROR) {
            FAIL_CORNER_FREE;
            return errcode;
        }

        // 利用分段扫描得到各个 LABEL 区域的最大垂距，记忆最大垂距坐标点的下标
        // 值
        errcode = this->segScan.segmentedScan(
                distDev, labelDev, tmpmaxdistDev, maxdistidxDev, count, false);
        if (errcode != NO_ERROR) {
            FAIL_CORNER_FREE;
            return errcode;
        }

       // 根据所求出来的垂距信息判断各个 LABEL 区域是否有新的点存在。
        errcode = this->updateFoundInfo(
                labelDev, distDev, maxdistidxDev,
                count, foundflagDev, startidxDev, threshold, foundidxDev);
        if (errcode != NO_ERROR) {
            FAIL_CORNER_FREE;
            return errcode;
        }

        // 通过扫描，计算出 LABEL 区域新发现点标记值对应的累加值。
        errcode = this->aryScan.scanArrayExclusive(foundflagDev, foundaccDev,
                                                   cornercnt, add,
                                                   false, false, false);
        if (errcode != NO_ERROR) {
            FAIL_CORNER_FREE;
            return errcode;
        }

        // 将新点标记累加值的最后一个拷贝到 Host 内存中，这个累加值的含义是
        // 当前迭代下所有新发现点的数量。 
        cuerrcode = cudaMemcpy(&foundcnt, &foundaccDev[cornercnt], 
                               sizeof (int), cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
             FAIL_CORNER_FREE;
            return errcode;
        }

        // 如果新发现点的数量小于等于 0，则说明所有的角点都已经被找到，
        // 没有必要在继续做下去了，因此退出迭代。
       if (foundcnt <= 0)
            break;

        // 更新点集
        errcode = this->updateCornerCst(
                tmpcstinDev, tmpcornerinDev, foundflagDev, foundaccDev, startidxDev, 
                maxdistidxDev, cornercnt, tmpcorneroutDev);
        if (errcode != NO_ERROR) {
            FAIL_CORNER_FREE;
            return errcode;
        }
        // 更新角点点集中点的数量。
        cornercnt += foundcnt;
        *cornerpnt = cornercnt;
       // 标记左侧点。所谓左侧点是在某 LABEL 区域内处于新发现的点左侧的点。
        errcode = this->updateLabel(labelDev, count, foundidxDev,
                                    foundaccDev, tmplabelDev);
        if (errcode != NO_ERROR) {
            FAIL_CORNER_FREE;
            return errcode;
        }

        // 交还部分中间变量，将本轮迭代得到的结果给到下一轮迭代的参数。
        labelDev = tmplabelDev;   
        int *cstswptmp = tmpcornerinDev;
        tmpcornerinDev = tmpcorneroutDev;
        tmpcorneroutDev = cstswptmp;
        // 一轮迭代到此结束。
    }

    // 将结果点集拷贝到 cornercst中
    cuerrcode = cudaMemcpy(
            cornercst, tmpcornerinDev, cornercnt * sizeof(int) * 2,
            cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(tmpmemDev);

    // 操作完毕，退出。
    return NO_ERROR;
}
#undef FAIL_CORNER_FREE

// Host 成员方法：douglasPeucker（道格拉斯算法简化曲线）
__host__ int DouglasPeucker::douglasPeucker(
        Curve *incur, Curve *outcur)
{
    // 检查指针性参数是否为 NULL。
    if (incur == NULL || outcur == NULL)
        return NULL_POINTER;

    // 局部变量，错误码。
    int errcode;
    int point = 0; 
    // 调用凸壳迭代，求输入点集的下半凸壳。
    errcode = this->douglasIter(incur->crvData, outcur->crvData, this->threshold,
                                incur->curveLength, &point);
    outcur->curveLength = point;
    if (errcode != NO_ERROR)
        return errcode;
    // 处理完毕，退出
    return NO_ERROR;
}



