// GetContourSet.cu
// 实现有连接性的闭合轮廓的获得算法

#include "GetContourSet.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：GET_CONTOUR_SET_INI_IFI
// 定义了一个无穷大
#define GET_CONTOUR_SET_INI_IFI    0x7fffffff

// 宏：OUTER_ORIGIN_CONTOUR
// 定义了输入闭曲线外部区域的标记值。
#define OUTER_ORIGIN_CONTOUR    0

// 宏：INNER_ORIGIN_CONTOUR
// 定义了输入闭曲线内部区域的标记值。
#define INNER_ORIGIN_CONTOUR   10

// 宏：DILATE_CIRCLE
// 定义了膨胀后得到环上点的标记值。
#define DILATE_CIRCLE          50

// 宏：OUTER_CONTOUR
// 定义了膨胀后得到环上外轮廓点的标记值。
#define OUTER_CONTOUR         100

// 宏：OUTER_CIRCLE
// 定义了经过二分类后外部环状物上的点的标记值。
#define OUTER_CIRCLE          150

// 宏：INNER_CONTOUR
// 定义了经过二分类后内外环状物交界处的点的标记值。
#define INNER_CONTOUR         200

// Device 全局变量：_eightNeighDev（八邻域点索引下标）
// 存放当前点八邻域范围内对应点的索引下标。
// 应用此数组可以便于进行八邻域像素点的遍历。
static __device__ int _eightNeighDev[8][2] = {
        { -1, -1 }, { 0, -1 }, {  1, -1 }, {  1,  0 },
        {  1,  1 }, { 0,  1 }, { -1,  1 }, { -1,  0 }
};

// Device 子程序：_findRootDev （查找根节点标记值）
// 查找根节点标记值算法，根据给定的 label 数组和坐标值
// 返回该坐标对应的根节点坐标值。该函数是为了便于其他 Kernel 函数调用。
static __device__ int  // 返回值：根节点标记值
_findRootDev(
        int label[],   // 输入的标记数组
        int idx        // 输入点的标记值
);

// Device 子程序：_unionDev （合并两个像素点使其位于同一区域）
// 合并两个不同像素点以使它们位于同一连通区域中
static __device__ void          // 该函数无返回值
_unionDev(                 
        int label[],            // 标记值数组
        unsigned char elenum1,  // 第一个像素点灰度值
        unsigned char elenum2,  // 第二个像素点灰度值
        int elelabel1,          // 第一个像素点标记值
        int elelabel2,          // 第二个像素点标记值
        int *flag               // 变换标记，当这两个输入像素点被合并到一个
                                // 区域后，该标记值将被设为 1。
);

// Kernel 函数：_imginitKer（初始化输入图像第一步）
// 将输入图像所有点的灰度值置为 0
static __global__ void   // Kernel 函数无返回值
_imginitKer(
        ImageCuda inimg  // 输入图像
);

// Kernel 函数：_initInimgKer（初始化输入图像第二步）
// 根据输入闭曲线的坐标值，将输入图像对应位置的灰度值修改为 255，
// 从而得到对应的输入图像。
static __global__ void          // Kernel 函数无返回值
_initInimgKer(
        CoordiSet incoordiset,  // 输入闭曲线
        int xmin, int ymin,     // 输入闭曲线的最上，最左点坐标值
        int radius,             // 半径
        ImageCuda inimg         // 输入图像
);

// Kernel 函数：_initLabelPerBlockKer (初始化每个块内像素点的标记值)
// 初始化每个线程块内点的标记值。该过程主要分为两个部分，首先，
// 每个节点的标记值为其在源图像中的索引值，如对于坐标为 (c, r) 点，
// 其初始标记值为 r * width + c ，其中 width 为图像宽;
// 然后，将各点标记值赋值为该点满足阈值关系的八邻域点中的最小标记值。
// 该过程在一个线程块中进行。
static __global__ void    // Kernel 函数无返回值
_initLabelPerBlockKer(
        ImageCuda inimg,  // 输入图像
        int label[]       // 输入标记数组
);

// Kernel 函数：_mergeBordersKer （合并不同块内像素点的标记值）
// 不同线程块的合并过程。该过程主要合并每两个线程块边界的点，
// 在这里我们主要采用每次合并 4 × 4 个线程块的策略。
static __global__ void    // Kernel 函数无返回值
_mergeBordersKer(
        ImageCuda inimg,  // 输入图像
        int *label,       // 输入标记数组
        int blockw,       // 应合并线程块的长度
        int blockh,       // 应合并线程块的宽度
        int threadz_z,    // 合并水平方向线程块时，z 向线程最大值
        int threadz_y     // 合并竖直方向线程块时，z 向线程最大值
);

// Kernel 函数：_findFinalLabelKer (找到每个点对应的根节点标记值)
// 找出每个点对应的根节点标记值，并将该值修改为当前点标记值。
static __global__ void  // Kernel 函数无返回值
_findFinalLabelKer(
        int *label,     // 输入标记值数组
        int width,      // 宽度
        int height      // 高度
);

// Kernel 函数：_initFlagSetKer（初始化标记值数组）
// 初始化标记值数组，将环外点置为 0，环内点置为 10，
// 同时使得生成的圆环上点的标记值置为 50
static __global__ void    // Kernel 函数无返回值
_initFlagSetKer(
        ImageCuda inimg,  // 输入图像
        int inflagset[],  // 输入标记值数组
        int *outflagset   // 输出标记值数组
);

// Kernel 函数：_findOuterContourKer（标记外轮廓点）
// 将外环轮廓点标记值置为 100，同时将点的 class 值存入 classNum 变量中。
static __global__ void          // Kernel 函数无返回值
_findOuterContourKer(
        int inflagset[],        // 输入标记值数组
        int *outflagset,        // 输出标记值数组（存储外环轮廓点标记值）
        int devclassarr[],      // 输入 class 数组
        int width, int height,  // 图像宽和图像高
        int *classnum           // 输出外环的 class 值
);

// Kernel 函数：_fillOuterCircleKer(将外环上点标记值置为 150)
// 将属于外环的所有点的标记值置为 150。
static __global__ void         // Kernel 函数无返回值
_fillOuterCircleKer(
        int inflagset[],       // 输入标记值数组
        int *outflagset,       // 中间标记值数组（存储外环所有点标记值）
        int devclassarr[],     // 输入 class 数组
        int classnum[],        // 输入外环 class 值
        int width, int height  // 图像宽和高
);

// Kernel 函数：_findInnerContourKer （修改内外环交界处点标记值，第一步）
// 将内外环交界处点标记值置为 200，第一步
static __global__ void         // Kernel 函数无返回值
_findInnerContourKer(
        int outflagset[],      // 输入标记值数组
        int *devflagset,       // 输出标记值数组
        int width, int height  // 图像宽和高
);

// Kernel 函数：_findInnerContourSecondKer（修改内外环交界处点标记值，第二步）
// 将内外环交界处点标记值置为 200，第二步
static __global__ void         // Kernel 函数无返回值
_findInnerContourSecondKer(
        int outflagset[],      // 输入标记值数组
        int *devflagset,       // 输出标记值数组
        int width, int height  // 图像宽和高
);

// Kernel 函数：_contourSetToimgKer （输出轮廓坐标至图像中）
// 将轮廓坐标输出到图像中。
static __global__ void        // Kernel 函数无返回值
_contourSetToimgKer(
        int inflagset[],      // 输入标记值数组
        ImageCuda contourimg  // 输出轮廓图像
);

// Kernel 函数：_innerConSetToimgKer （输出内环内点坐标至图像中）
// 将内环内点坐标输出到图像中。
static __global__ void      // Kernel 函数无返回值
_innerConSetToimgKer(
        int inflagset[],    // 输入标记值数组
        ImageCuda innerimg  // 输出内环内点图像
);

// Kernel 函数：_imginitKer（初始化输入图像第一步）
static __global__ void _imginitKer(ImageCuda inimg)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    int inidx = r * inimg.pitchBytes + c;
    // 将输入图像所有点的灰度值初始化为 0 。
    inimg.imgMeta.imgData[inidx] = 0;
}

// Kernel 函数：_initInimgKer（初始化输入图像第二步）
static __global__ void _initInimgKer(CoordiSet incoordiset,
                                     int xmin, int ymin,
                                     int radius,
                                     ImageCuda inimg)
{
    // 计算该线程在块内的相对位置。
    int inidx = blockIdx.x * blockDim.x + threadIdx.x;

    // 若线程在块内的相对位置大于输入坐标集大小，即点个数，
    // 则不执行任何操作，返回。
    if (inidx >= incoordiset.count)
        return;

    // 计算坐标集中每个点的横纵坐标
    int x = incoordiset.tplData[2 * inidx];
    int y = incoordiset.tplData[2 * inidx + 1];
    // 计算在新坐标系下的每个点的横纵坐标值
    x = x - xmin + radius + 1;
    y = y - ymin + radius + 1;

    // 计算坐标点对应的图像数据数组下标。
    int outidx = y * inimg.pitchBytes + x;
    // 将图像对应点的灰度值置为255。
    inimg.imgMeta.imgData[outidx] = 255;
}

// Device 子程序：_findRootDev (查找根节点标记值)
static __device__ int _findRootDev(int label[], int idx)
{
    // 在 label 数组中查找 idx 下标对应的最小标记值，
    // 并将该值作为返回值。
    int nexidx;
    do {
        nexidx = idx;
        idx = label[nexidx];
    } while (idx < nexidx);
    
    // 处理完毕，返回根节点标记值。
    return idx;
}

// Kernel 函数：_initLabelPerBlockKer (初始化各线程块内像素点的标记值)
static __global__ void _initLabelPerBlockKer(
        ImageCuda inimg, int label[])
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量 (其中, c 表示 column; r 表示 row)。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    int i, j, k;
    // 计算输入坐标点在label数组中对应的数组下标
    int idx = r * inimg.imgMeta.width + c;
    // 计算输入坐标点对应的图像数据数组下标
    int inidx = r * inimg.pitchBytes + c, newidx;

    // 计算应申请的 shared memory 的步长
    int spitch = blockDim.x + 2;
    // 计算当前坐标点在 shared memory 中对应的下标
    int localidx = (threadIdx.y + 1) * spitch + threadIdx.x + 1;

    // oldlabel 用来记录当前点未经过八邻域判断前的标记值，
    // newlabel 用来记录经过一轮判断后当前点的最新标记值，
    // 当一个点的 oldlabel 与 newlabel 一致时，当前点对应的标记值为最终标记
    // 初始时，每个点的标记值设为其在 shared memory 中的对应下标
    int oldlabel, newlabel = localidx;
    // curvalue 用来记录当前点的灰度值，newvalue 用来记录其八邻域点的灰度值
    unsigned char curvalue, newvalue;
    curvalue = inimg.imgMeta.imgData[inidx];

    // 共享内存数据区,该部分包含了存储在共享内存中的像素点的标记值。
    // 由于未对 Kernel 的尺寸做出假设，这里使用动态申请的 Shared
    // Memory（共享内存）。
    extern __shared__ int slabel[];
    // 共享变量 sflag 数组用来存储是否应停止循环信息。
    // 当 sflag[0] 的值为 0 时，表示块内的迭代已经完成。
    __shared__ int sflag[1];

    // 由于 shared memory 的大小为 (blockDim.x + 2) * (blockDim.y + 2)
    // 在这里将 shared memory 的边界点（即 shared memory 中超出线程块的点）
    // 的标记值设为无穷大。
    if (threadIdx.x == 0)
        slabel[localidx - 1] = GET_CONTOUR_SET_INI_IFI;
    if (threadIdx.x == blockDim.x - 1)
        slabel[localidx + 1] = GET_CONTOUR_SET_INI_IFI;
    if (threadIdx.y == 0) {
        slabel[localidx - spitch] = GET_CONTOUR_SET_INI_IFI;
        if (threadIdx.x == 0)
            slabel[localidx - spitch - 1] = GET_CONTOUR_SET_INI_IFI;
        if (threadIdx.x == blockDim.x - 1)
            slabel[localidx - spitch + 1] = GET_CONTOUR_SET_INI_IFI;
    }
    if (threadIdx.y == blockDim.y - 1) {
        slabel[localidx + spitch] = GET_CONTOUR_SET_INI_IFI;
        if (threadIdx.x == 0)
            slabel[localidx + spitch - 1] = GET_CONTOUR_SET_INI_IFI;
        if (threadIdx.x == blockDim.x - 1)
            slabel[localidx + spitch + 1] = GET_CONTOUR_SET_INI_IFI;
    }

    while (1) {
        // 将当前点的标记值设为其在 shared memory 中的数组下标
        slabel[localidx] = newlabel;
        // 将 sflag[0] 标记值设为 0
        if ((threadIdx.x | threadIdx.y) == 0)
            sflag[0] = 0;
        // 初始时，将 newlabel 值赋给 oldlabel
        oldlabel = newlabel;
        __syncthreads();

        // 在当前点的八邻域范围内查找与其灰度值之差的绝对值小于阈值的点，
        // 并将这些点的最小标记值赋予记录在 newlabel 中
        for (i = r - 1; i <= r + 1; i++) {
            for (j = c - 1; j <= c + 1; j++) {
                if (j == c && i == r)
                    continue;
                newidx = i * inimg.pitchBytes + j;
                newvalue = inimg.imgMeta.imgData[newidx];
                if ((i >= 0 && i < inimg.imgMeta.height
                     && j >= 0 && j < inimg.imgMeta.width)
                    && (curvalue == newvalue)) {
                    k = localidx + (i - r) * spitch + j - c;
                    newlabel = min(newlabel, slabel[k]);
                }
            }
        }
        __syncthreads();

        // 若当前点的 oldlabel 值大于 newlabel 值，
        // 表明当前点的标记值不是最终的标记值
        // 则将 sflag[0] 值设为 1，来继续进行循环判断，并通过原子操作
        // 将 newlabel 与 slabel[oldlabel] 的较小值赋予 slabel[oldlabel]
        if (oldlabel > newlabel) {
            atomicMin(&slabel[oldlabel], newlabel);
            sflag[0] = 1;
        }
        __syncthreads();

        // 当线程块内所有像素点对应的标记值不再改变,
        // 即 sflag[0] 的值为 0 时，循环结束。
        if (sflag[0] == 0) break;

        // 计算 newlabel 对应的根节点标记值，并将该值赋给 newlabel
        newlabel = _findRootDev(slabel, newlabel);
        __syncthreads();
    }

    // 将 newlabel 的值转换为其在 label 数组中的数组下标
    j = newlabel / spitch;
    i = newlabel % spitch;
    i += blockIdx.x * blockDim.x - 1;
    j += blockIdx.y * blockDim.y - 1;
    newlabel = j * inimg.imgMeta.width + i;
    label[idx] = newlabel;
}

// Device 子程序：_unionDev (合并两个不同像素点以使它们位于同一连通区域中)
static __device__ void _unionDev(
        int label[], unsigned char elenum1, unsigned char elenum2,
        int label1, int label2, int *flag)
{
    int newlabel1, newlabel2;

    // 比较两个输入像素点的灰度值是否满足给定的阈值范围
    if (elenum1 == elenum2) {
        // 若两个点满足指定条件，则分别计算这两个点的根节点标记值
        // 计算第一个点的根节点标记值
        newlabel1 = _findRootDev(label, label1);
        // 计算第二个点的根节点标记值
        newlabel2 = _findRootDev(label, label2);
        // 将较小的标记值赋值给另一点在标记数组中的值
        // 并将 flag[0] 置为 1
        if (newlabel1 > newlabel2) {
            // 使用原子操作以保证操作的唯一性与正确性
            atomicMin(&label[newlabel1], newlabel2);
            flag[0] = 1;
        } else if (newlabel2 > newlabel1) {
            atomicMin(&label[newlabel2], newlabel1);
            flag[0] = 1;
        }
    }
}

// Kernel 函数：_mergeBordersKer（合并不同块内像素点的标记值）
static __global__ void _mergeBordersKer(
        ImageCuda inimg, int *label, int blockw, int blockh,
        int threadz_x, int threadz_y)
{
    int idx, iterateTimes, i;
    int x, y;
    int curidx, newidx;
    unsigned char curvalue, newvalue;

    // 在这里以每次合并 4 * 4 = 16 个线程块的方式合并线程块
    // 分别计算待合并线程块在 GRID 中的 x 和 y 向分量
    int threadidx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int threadidx_y = blockDim.y * blockIdx.y + threadIdx.y;

    // 共享数组变量，只含有一个元素，每当有两个像素点合并时，该数组
    // 变量值置为 1。
    __shared__ int sflag[1];

    while (1) {
        // 设置 sflag[0] 的值为 0。
        if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0)
            sflag[0] = 0;
        __syncthreads();

        // 合并上下相邻线程块的水平方向边界点
        // 由于位于 GRID 中最后一行的线程块向下没有待合并的线程块
        // 因而这里不处理最后一行的线程块
        if ((threadIdx.y < blockDim.y - 1)) {
            // 计算为了合并一行线程块的迭代次数
            iterateTimes = blockw / threadz_x;

            // 计算待合并像素点在源图像中的像素点坐标
            x = threadidx_x * blockw + threadIdx.z;
            y = threadidx_y * blockh + blockh - 1;

            // 根据迭代次数合并块内线程标记值
            for (i = 0; i < iterateTimes; i++) {
                if (threadIdx.z < threadz_x && x < inimg.imgMeta.width &&
                    y < inimg.imgMeta.height) {
                    idx = y * inimg.imgMeta.width + x;

                    // 计算当前像素点灰度值
                    curidx = y * inimg.pitchBytes + x;
                    curvalue = inimg.imgMeta.imgData[curidx];
                    // 计算位于当前像素点下方像素点的灰度值，
                    // 其坐标值为 (x, y + 1)。
                    newidx = curidx + inimg.pitchBytes;
                    newvalue = inimg.imgMeta.imgData[newidx];

                    // 合并这两个像素点
                    _unionDev(label, curvalue, newvalue,
                              idx, idx + inimg.imgMeta.width, sflag);

                    // 若当前像素点不为最左侧像素点时，即 x ！= 0 时，合并
                    // 位于当前像素点左下方像素点，其坐标值为 (x - 1, y + 1)。
                    if (x - 1 >= 0) {
                        newidx -= 1;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue,
                                  idx, idx + inimg.imgMeta.width - 1,
                                  sflag);
                    }

                    // 若当前像素点不为最右侧像素点时，x ！= inimg.imgMeta.width
                    // 时,合并位于当前像素点右下方像素点，其坐标值为
                    // (x + 1, y + 1)。
                    if (x + 1 < inimg.imgMeta.width) {
                        newidx += 2;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue,
                                  idx, idx + inimg.imgMeta.width + 1,
                                  sflag);
                    }
                }
                // 计算下次迭代的起始像素点的 x 坐标
                x += threadz_x;
            }
        }

        // 合并左右相邻线程块的竖直方向边界点
        // 由于位于 GRID 中最后一列的线程块向右没有待合并的线程块
        // 因而这里不处理最后一列的线程块
        if ((threadIdx.x < blockDim.x - 1)) {
            // 计算为了合并一列线程块的迭代次数
            iterateTimes = blockh / threadz_y;

            // 计算待合并像素点在源图像中的像素点坐标，
            // 由于处理的是每个线程块的最右一列像素点，
            // 因此 x 坐标值因在原基础上加上线程块宽度 - 1
            x = threadidx_x * blockw + blockw - 1;
            y = threadidx_y * blockh + threadIdx.z;

            // 根据迭代次数合并块内线程标记值
            for (i = 0; i < iterateTimes; i++) {
                if (threadIdx.z < threadz_y && x < inimg.imgMeta.width &&
                    y < inimg.imgMeta.height) {
                    idx = y * inimg.imgMeta.width + x;

                    // 计算当前像素点灰度值
                    curidx = y * inimg.pitchBytes + x;
                    curvalue = inimg.imgMeta.imgData[curidx];
                    // 计算位于当前像素点右侧像素点的灰度值，
                    // 其坐标值为 (x + 1, y)。
                    newidx = curidx + 1;
                    newvalue = inimg.imgMeta.imgData[newidx];
                    // 合并这两个像素点
                    _unionDev(label, curvalue, newvalue,
                              idx, idx + 1, sflag);

                    // 若当前像素点不为最上侧像素点时，即 y ！= 0 时，合并
                    // 位于当前像素点右上方像素点，其坐标值为 (x + 1, y - 1)。
                    if (y - 1 >= 0) {
                        newidx -= inimg.pitchBytes;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue, idx, 
                                  idx - inimg.imgMeta.width + 1,
                                  sflag);
                    }

                    // 若当前像素点不为最下侧像素点时，
                    // 即 y ！= inimg.imgMeta.height时，合并位于当前像素点
                    // 右下方像素点，其坐标值为(x + 1, y + 1)。
                    if (y + 1 < inimg.imgMeta.height) {
                        newidx = curidx + inimg.pitchBytes + 1;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue,
                                  idx, idx + inimg.imgMeta.width + 1,
                                  sflag);
                    }
                }
                // 计算下次迭代的起始像素点的 y 坐标
                y += threadz_y;
            }
        }
        __syncthreads();
        if (sflag[0] == 0) break;
    }
}

// Kernel 函数：_findFinalLabelKer (找到每个点对应的根节点标记值)
static __global__ void _findFinalLabelKer(int *label, int width, int height)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量 (其中, c 表示 column; r 表示 row)。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= width || r >= height)
        return;

    // 计算输入坐标点在label数组中对应的数组下标
    int inidx = r * width + c;

    // 计算当前像素点的标记值
    int curlabel = label[inidx];
    // 将当前像素点标记值的根节点值赋给原像素点
    int newlabel = _findRootDev(label, curlabel);
    label[inidx] = newlabel;
}

// Kernel 函数：_initFlagSetKer（初始化标记值数组）
static __global__ void _initFlagSetKer(
        ImageCuda inimg, int inflagset[], int *outflagset)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算输入坐标点在图像中对应的数组下标
    int inidx = r * inimg.pitchBytes + c;
    // 计算输入坐标点在输入数组中对应的数组下标
    int flagidx = r * inimg.imgMeta.width + c;
    unsigned char intemp;
    // 读取坐标点对应的像素值
    intemp = inimg.imgMeta.imgData[inidx];
    if (inflagset[flagidx] != OUTER_ORIGIN_CONTOUR)
        inflagset[flagidx] = INNER_ORIGIN_CONTOUR;

    // 若当前点像素点不为 0，则将标记值数组对应位置的值置为 50。
    if (intemp) {        
        inflagset[flagidx] = DILATE_CIRCLE;
    }
    outflagset[flagidx] = inflagset[flagidx];
}

// Kernel 函数：_findOuterContourKer（修改外轮廓点标记值）
static __global__ void _findOuterContourKer(
        int inflagset[], int *outflagset, int devclassarr[],
        int width, int height, int *classnum)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= width || r >= height)
        return;

    // 计算输入坐标点在输入数组中对应的数组下标
    int flagidx = r * width + c, newidx;
    if (inflagset[flagidx] == DILATE_CIRCLE) {
        for (int i = 0; i < 8; i++) {
            newidx = (_eightNeighDev[i][1] + r) * width +
                      _eightNeighDev[i][0] + c;
            if (!inflagset[newidx]) {
                outflagset[flagidx] = OUTER_CONTOUR;
                *classnum = devclassarr[flagidx];
                break;
            }
        }
    }
}

// Kernel 函数：_fillOuterCircleKer（将外环内所有点标记值置为 150）
static __global__ void _fillOuterCircleKer(
        int inflagset[], int *outflagset, int devclassarr[], int classnum[],
        int width, int height)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= width || r >= height)
        return;

    // 计算输入坐标点在输入数组中对应的数组下标
    int flagidx = r * width + c, newidx;
    // num 用来存储当前点八邻域范围内标记值为 50 的点个数
    // classtotalnum 用来存储当前点八邻域范围内 class 值为 *classNum 的点个数
    int num = 0, classtotalnum = 0;
    if (inflagset[flagidx] == DILATE_CIRCLE) {
        for (int i = 0; i < 8; i++) {
            newidx = (_eightNeighDev[i][1] + r) * width + 
                      _eightNeighDev[i][0] + c;
            if (inflagset[newidx] == DILATE_CIRCLE)
                num++;
            if (devclassarr[newidx] == *classnum)
                classtotalnum++;
        }
    }

    // 若当前点八邻域范围内皆为膨胀环内点时，则设置当前点为外环上点。
    if (num == 8 && classtotalnum == 8) {
        outflagset[flagidx] = OUTER_CIRCLE;
    }
}

// Kernel 函数：_findInnerContourKer（修改内外环交界处点标记值，第一步）
static __global__ void _findInnerContourKer(
        int outflagset[], int *devflagset, int width, int height)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= width || r >= height)
        return;

    // 计算输入坐标点在输入数组中对应的数组下标
    int flagidx = r * width + c, newidx = 0;
    if (outflagset[flagidx]== DILATE_CIRCLE) {
        for (int i = 0; i < 8; i++) {
            newidx = (_eightNeighDev[i][1] + r) * width +
                     _eightNeighDev[i][0] + c;
            if (outflagset[newidx] == OUTER_CIRCLE ||
                outflagset[newidx] == OUTER_CONTOUR) {
                devflagset[flagidx] = INNER_CONTOUR;
                break;
            }
        }
    }
}

// Kernel 函数：_findInnerContourSecondKer（修改内外环交界处点标记值，第二步）
static __global__ void _findInnerContourSecondKer(
        int outflagset[], int *devflagset, int width, int height)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= width || r >= height)
        return;

    // 计算输入坐标点在输入数组中对应的数组下标
    int flagidx = r * width + c;
    if (devflagset[flagidx] == INNER_CONTOUR)
        outflagset[flagidx] = INNER_CONTOUR;
}

// Kernel 函数：_contourSetToimgKer（将轮廓坐标输出到图像中）
static __global__ void _contourSetToimgKer(
        int inflagset[], ImageCuda contourimg)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= contourimg.imgMeta.width || r >= contourimg.imgMeta.height)
        return;

    // 计算输入坐标点在图像中对应的数组下标
    int inidx = r * contourimg.pitchBytes + c;
    // 计算输入坐标点在输入数组中对应的数组下标
    int flagidx = r * contourimg.imgMeta.width + c;
    contourimg.imgMeta.imgData[inidx] = 0;
    if (inflagset[flagidx] == INNER_CONTOUR)
        contourimg.imgMeta.imgData[inidx] = 255;
}

// Kernel 函数：_innerConSetToimgKer（将内环内点坐标输出到图像中）
static __global__ void _innerConSetToimgKer(int inflagset[], ImageCuda innerimg)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= innerimg.imgMeta.width || r >= innerimg.imgMeta.height)
        return;

    // 计算输入坐标点在图像中对应的数组下标
    int inidx = r * innerimg.pitchBytes + c;
    // 计算输入坐标点在输入数组中对应的数组下标
    int flagidx = r * innerimg.imgMeta.width + c;
    innerimg.imgMeta.imgData[inidx] = 0;
    if (inflagset[flagidx] == INNER_ORIGIN_CONTOUR ||
        inflagset[flagidx] == DILATE_CIRCLE)
        innerimg.imgMeta.imgData[inidx] = 255;
}

// Host 成员方法：findMinMaxCoordinates （计算最左最右最上最下点坐标值）
// 根据输入闭曲线分别找到曲线最左，最右，最上，最下点的对应坐标值
__host__ int GetContourSet::findMinMaxCoordinates(CoordiSet *incoordiset,
                                                  int *resultcs)
{
    // 声明局部变量。
    int i;
    int errcode = NO_ERROR;

    // 将 incoordiSet 坐标集拷贝至 host 端。
    errcode = CoordiSetBasicOp::copyToHost(incoordiset);
    if (errcode != NO_ERROR) 
        return errcode;

    // 初始化 x 和 y 方向上的最小最大值。
    resultcs[0] = resultcs[1] = incoordiset->tplData[0];
    resultcs[2] = resultcs[3] = incoordiset->tplData[1];
    for (i = 1; i < incoordiset->count; i++) {
        // resultcs[0] 存储坐标点横坐标最大值
        if (resultcs[0] > incoordiset->tplData[2 * i])
            resultcs[0] = incoordiset->tplData[2 * i];
        // resultcs[1] 存储坐标点横坐标最小值
        if (resultcs[1] < incoordiset->tplData[2 * i])
            resultcs[1] = incoordiset->tplData[2 * i];
        // resultcs[2] 存储坐标点纵坐标最大值
        if (resultcs[2] > incoordiset->tplData[2 * i + 1])
            resultcs[2] = incoordiset->tplData[2 * i + 1];
        // resultcs[3] 存储坐标点纵坐标最小值
        if (resultcs[3] < incoordiset->tplData[2 * i + 1])
            resultcs[3] = incoordiset->tplData[2 * i + 1];
    }
    return errcode;
}

// Host 成员方法：sortContour （按序输出坐标点集）
// 根据输入 inArray 按顺时针方向顺序输出有序的点集，并将结果
// 输出到一个坐标集 outcoordiset 中。
__host__ int GetContourSet::sortContour(int inarray[], CoordiSet *outcoordiset,
                                        int width, int height)
{
    int errcode = NO_ERROR;  // 局部变量，错误码

    CoordiSetBasicOp::makeAtHost(outcoordiset, width * height);
    // bFindStartPoint 表示是否是否找到起始点及回到起始点
    // bFindPoint 表示是否扫描到一个边界点
    bool bFindStartPoint, bFindPoint;
    // startPW 和 startPH 分别表示起始点对应横坐标以及纵坐标
    //   curPW 和   curPH 分别表示当前点对应横坐标以及纵坐标
    int startPW, startPH, curPW, curPH;
    // 定义扫描的八邻域方向坐标
    static int direction[8][2] = {
            { -1, -1 }, { 0, -1 }, {  1, -1 }, {  1, 0 },
            {  1,  1 }, { 0,  1 }, { -1,  1 }, { -1, 0 }
    };
    int beginDirect;
    bFindStartPoint = false;
    int index = 0;
    int curvalue;

    // 找到最左下方的边界点
    for (int j = 1; j < height - 1 && !bFindStartPoint; j++) {
        for (int i = 1; i < width - 1 && !bFindStartPoint; i++) {
            curvalue = inarray[j * width + i];
            if (curvalue == INNER_CONTOUR) {
                bFindStartPoint = true;
                startPW = i;
                startPH = j;
            }
        }
    }

    // 由于起始点是在左下方，故起始扫描沿左上方向
    beginDirect = 0;
    bFindStartPoint = false;
    curPW = startPW;
    curPH = startPH;
    while (!bFindStartPoint) {
        // 从起始点一直找边界，直到再次找到起始点为止
        bFindPoint = false;
        while (!bFindPoint) {
            // 沿扫描方向，获取左上方向像素点灰度值
            curvalue = inarray[(curPH + direction[beginDirect][1]) * width
                                      + curPW + direction[beginDirect][0]];
            if (curvalue == INNER_CONTOUR) {
                bFindPoint = true;
                outcoordiset->tplData[2 * index] = curPW;
                outcoordiset->tplData[2 * index + 1] = curPH;
                index++;
                curPW = curPW + direction[beginDirect][0];
                curPH = curPH + direction[beginDirect][1];
                if (curPH == startPH && curPW == startPW)
                    bFindStartPoint = true;
                // 扫描的方向逆时针旋转两格
                beginDirect--;
                if (beginDirect == -1)
                    beginDirect = 7;
                beginDirect--;
                if (beginDirect == -1)
                    beginDirect = 7;
            } else {
                // 扫描的方向顺时针旋转一格
                beginDirect++;
                if (beginDirect == 8)
                    beginDirect = 0;
            }
        }
    }
    // 修改输出坐标集的大小为轮廓点个数，即 index
    outcoordiset->count = index;
    return errcode;
}

// 宏：FAIL_GET_CONTOUR_SET_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_GET_CONTOUR_SET_FREE  do {                       \
        if (alldatadev != NULL)                               \
            cudaFree(alldatadev);                             \
        if (resultcs != NULL)                                 \
            delete [] resultcs;                               \
        if (tmpimg != NULL)                                   \
            ImageBasicOp::deleteImage(tmpimg);                \
        if (outimg != NULL)                                   \
            ImageBasicOp::deleteImage(outimg);                \
        if (coorForTorus != NULL)                             \
            CoordiSetBasicOp::deleteCoordiSet(coorForTorus);  \
        if (torusClass != NULL)                               \
            delete [] torusClass;                             \
        if (classArr != NULL)                                 \
            delete [] classArr;                               \
    } while (0)

// Host 成员方法：getContourSet（执行有连接性的封闭轮廓的获得算法）
// 根据输入输入闭曲线坐标点集，生成内环内所有点坐标点集到 innerCoordiset 中，
// 生成内外环交界处坐标点集并输出到 contourCoordiset 中。
__host__ int GetContourSet::getContourSet(CoordiSet *incoordiset,
                                          CoordiSet *innercoordiset,
                                          CoordiSet *contourcoordiset)
{
    // 检查输入输出坐标集是否为 NULL，如果为 NULL 直接报错返回。
    if (incoordiset == NULL || innercoordiset == NULL ||
        contourcoordiset == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    int *resultcs;
    // 声明需要的指针变量。
    int *alldatadev = NULL;
    // 声明 Device 端需要的变量。
    int *devFlagset = NULL, *outFlagset = NULL;
    int *devNum = NULL, *devClassArr = NULL;

    //  声明需要使用的中间图像 tmpimg 和 outimg
    Image *tmpimg, *outimg;

    // 声明调用二分类方法需要使用的临时变量，
    // 分别为 coorForTorus，torusClass 以及 classArr。
    CoordiSet *coorForTorus;
    unsigned char *torusClass = NULL;
    int *classArr = NULL;

    // 为 resultcs 分配大小，用来存储输入闭曲线的最左最右最上最下点的坐标值。
    resultcs = new int[4];
    if (resultcs == NULL) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return OUT_OF_MEM;
    }

    // 在 host 端分配 tmpimg 以及 outimg
    errcode = ImageBasicOp::newImage(&tmpimg);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }
    errcode = ImageBasicOp::newImage(&outimg);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    // 分别找到输入闭曲线的最左最右最上最下点的坐标值，
    // 并将其输出到一个大小为 4 的数组中。
    errcode = GetContourSet::findMinMaxCoordinates(incoordiset, resultcs);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    // 计算应生成图像的长和宽
    int width = resultcs[1] - resultcs[0] + 2 * radiusForCurve + 3;
    int height = resultcs[3] - resultcs[2] + 2 * radiusForCurve + 3;

    // 将图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::makeAtCurrentDevice(tmpimg, width, height);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    // 将图像拷贝到 Host 内存中。
    errcode = ImageBasicOp::makeAtHost(outimg, width, height);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    // 一次性申请 Device 端需要的所有空间。 
    errcode = cudaMalloc((void **)&alldatadev,
                         (1 + 3 * width * height) * sizeof (int));
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 初始化 GetContourSet 累加器在 Device 上的内存空间。
    errcode = cudaMemset(alldatadev, 0, 
                         (1 + 3 * width * height) * sizeof (int));
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 通过偏移得到各指针的地址。
    devNum      = alldatadev;
    devFlagset  = alldatadev + 1;
    outFlagset  = alldatadev + 1 + width * height;
    devClassArr = alldatadev + 1 + 2 * width * height;

    // 将 incoordiset 拷贝到 Device 内存中。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(incoordiset);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    // 提取图像的 ROI 子图像。
    ImageCuda tmpsubimgCud;
    errcode = ImageBasicOp::roiSubImage(tmpimg, &tmpsubimgCud);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }
    tmpsubimgCud.imgMeta.width = width;
    tmpsubimgCud.imgMeta.height = height;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 grid, block;
    block.x = DEF_BLOCK_X;
    block.y = DEF_BLOCK_Y;
    block.z = 1;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (height + block.y - 1) / block.y;
    grid.z = 1;
    _imginitKer<<<grid, block>>>(tmpsubimgCud);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = 1;
    blocksize.z = 1;
    gridsize.x = (incoordiset->count + blocksize.x - 1) / blocksize.x;
    gridsize.y = 1;
    gridsize.z = 1;
    _initInimgKer<<<gridsize, blocksize>>>(*incoordiset,
                                           resultcs[0], resultcs[2],
                                           radiusForCurve, tmpsubimgCud);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 将 tmpimg 拷贝至 host 端，便于执行后续的膨胀操作。
    errcode = ImageBasicOp::copyToHost(tmpimg);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    // 调用已封装好的膨胀操作
    morForCurve.dilate(tmpimg, outimg);

    // 声明 ImgConvert 类变量，进行图像转坐标集调用。
    ImgConvert imgconForCurve;
    CoordiSetBasicOp::newCoordiSet(&coorForTorus);
    // 根据输入图像生成坐标集 coorForTorus
    errcode = imgconForCurve.imgConvertToCst(outimg, coorForTorus);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    // 为 torusClass 分配大小，用以存储生成的 class 数组。
    torusClass = new unsigned char[coorForTorus->count];    
    if (torusClass == NULL) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return OUT_OF_MEM;
    }
    // 调用 TorusSegmentation 的 torusSegregate 操作，生成内外环
    errcode = tsForCurve.torusSegregate(width, height,
                                        coorForTorus, torusClass);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    errcode = CoordiSetBasicOp::copyToHost(coorForTorus);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    // 为 classArr 分配大小，大小为 width * height 的整型数组，
    // 用以保存 torusClass 对应于图像索引的值
    //（由于 torusClass 的大小为 coorForTorus->count）
    classArr = new int[width * height];
    if (classArr == NULL) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return OUT_OF_MEM;
    }

    int x, y;
    for (int i = 0; i < coorForTorus->count; i++) {
        // 计算输入坐标集对应的横坐标与纵坐标
        x = coorForTorus->tplData[2 * i];
        y = coorForTorus->tplData[2 * i + 1];
        // 将 torusClass 中的值赋值给 classArr 数组
        classArr[y * width + x] = (int)torusClass[i];
    }
    // 将 classArr 数组的值拷贝至 Device 端，便于进行 Device 端处理。
    errcode = cudaMemcpy(devClassArr, classArr,
                         width * height * sizeof (int),
                         cudaMemcpyHostToDevice);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 将膨胀生成的图像 outimg 拷贝至 Device 端，便于进行 Device 端处理。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }
    // 提取生成图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }
    outsubimgCud.imgMeta.width = width;
    outsubimgCud.imgMeta.height = height;
  
    // 计算初始化块内内存时，共享内存的大小。
    int smsize = sizeof (int) * (block.x + 2) * (block.y + 2);

    // 调用核函数，初始化每个线程块内标记值
    _initLabelPerBlockKer<<<grid, block, smsize>>>(
            outsubimgCud, devFlagset);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 合并线程块时每次合并线程块的长、宽和高
    int blockw, blockh, blockz;
    // 计算第一次合并时，应合并线程块的长、宽和高
    // 第一次合并时，应合并线程块的长应为初始线程块长，宽为初始线程块宽
    blockw = block.x;
    blockh = block.y;
    // 由于这里采用的是 3 维线程块，线程块的高设为初始线程块长和宽的较大者。
    blockz = blockw;
    if (blockw < blockh)
        blockz = blockh;

    // 计算每次合并的线程块个数，在这里我们采用的是每次合并 4 × 4 的线程块，
    // 由于采用这种方式合并所需的迭代次数最少。
    int xtiles = 4, ytiles = 4;
    // 计算合并线程块前 GRID 的长
    int tilesizex = grid.x;
    // 计算合并线程块前 GRID 的宽
    int tilesizey = grid.y;
    // 定义为进行线程块合并而采用的线程块与网格。
    dim3 mrgblocksize, mrggridsize;

    // 由于每个线程块的大小限制为 1024，而 tilesizex * tilesizey * blockz 的值
    // 为每次用来进行合并操作的三维线程块的最大大小，因此当该值不大于 1024 时，
    // 可将所有线程块放在一个三维线程块中合并，这样，我们就可以以该值是否
    // 不大于 1024 来作为是否终止循环的判断条件。
    while (tilesizex * tilesizey * blockz > 1024) {
        // 计算每次合并线程块时 GRID 的长，这里采用向上取整的方式
        tilesizex = (tilesizex - 1) / xtiles + 1;
        // 计算每次合并线程块时 GRID 的宽，这里采用向上取整的方式
        tilesizey = (tilesizey - 1) / ytiles + 1;

        // 设置为了合并而采用的三维线程块大小，这里采用的是 4 × 4 的方式，
        // 因此线程块的长为 4，宽也为 4，高则为 32。
        mrgblocksize.x = xtiles; mrgblocksize.y = ytiles;
        mrgblocksize.z = blockz;        
        // 设置为了合并而采用的二维网格的大小。
        mrggridsize.x = tilesizex; mrggridsize.y = tilesizey;
        mrggridsize.z = 1;
        // 调用核函数，每次合并4 × 4 个线程块内的标记值
        _mergeBordersKer<<<mrggridsize, mrgblocksize>>>(
                outsubimgCud, devFlagset, blockw, blockh,
                block.x, block.y);
        if (cudaGetLastError() != cudaSuccess) {
            FAIL_GET_CONTOUR_SET_FREE;
            return CUDA_ERROR;
        }
        // 在每次迭代后，修改应合并线程块的长和宽，因为每次合并 4 * 4 个线程块，
        // 因此，经过迭代后，应合并线程块的长和宽应分别乘 4。
        blockw *= xtiles;
        blockh *= ytiles;
    }

    // 进行最后一轮线程块的合并
    // 计算该轮应采用的三维线程块大小
    mrgblocksize.x = tilesizex; mrgblocksize.y = tilesizey;
    mrgblocksize.z = blockz; 
    // 设置该论应采用的网格大小，长宽高分别为1。
    mrggridsize.x = 1; mrggridsize.y = 1;mrggridsize.z = 1;
    // 调用核函数，进行最后一轮线程块合并
    _mergeBordersKer<<<mrggridsize, mrgblocksize>>>(
            outsubimgCud, devFlagset, blockw, blockh,
            block.x, block.y);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 调用核函数，即找出每个结点对应的标记值，
    // 其中根节点的标记值与其自身在数组中的索引值一致
    _findFinalLabelKer<<<grid, block>>>(devFlagset, width, height);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 调用核函数，初始化标记值数组，将环外点置为 0，环内点置为 10
    _initFlagSetKer<<<grid, block>>>(outsubimgCud, devFlagset, outFlagset);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 调用核函数，将外环轮廓点标记值置为 100，
    // 同时将点的 class 值存入 classNum 变量中。
    _findOuterContourKer<<<grid, block>>>(devFlagset, outFlagset,
                                          devClassArr,width, height, devNum);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 调用核函数，将外环上点标记值置为 150
    _fillOuterCircleKer<<<grid, block>>>(devFlagset, outFlagset, devClassArr,
                                         devNum, width, height);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 调用核函数，将内外环交界处点标记值置为 200，第一步
    _findInnerContourKer<<<grid, block>>>(outFlagset, devFlagset, 
                                          width, height);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 调用核函数，将内外环交界处点标记值置为 200，第二步
    _findInnerContourSecondKer<<<grid, block>>>(outFlagset, devFlagset,
                                                width, height);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    // 调用核函数，将内环内点坐标输出到图像中。
    _innerConSetToimgKer<<<grid, block>>>(outFlagset, outsubimgCud);
    if (cudaGetLastError() != cudaSuccess) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return CUDA_ERROR;
    }

    errcode = ImageBasicOp::copyToHost(outimg);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }
    // 调用已封装好的图像转坐标集函数，生成内环内点坐标集。
    errcode = imgconForCurve.clearAllConvertFlags();
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }
    errcode = imgconForCurve.setConvertFlag(255);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }
    errcode = imgconForCurve.imgConvertToCst(outimg, innercoordiset);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }
    errcode = CoordiSetBasicOp::copyToHost(innercoordiset);
    if (errcode != NO_ERROR) {
        // 释放内存空间。
        FAIL_GET_CONTOUR_SET_FREE;
        return errcode;
    }

    // 当 inorder 参数表示为有序时，调用 host 端 SortContour 函数输出数组，
    // 否则调用 reindex 实现乱序输出。
    if (inorder) {
        int *hostFlagset = new int[width * height];
        if (hostFlagset == NULL) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            return OUT_OF_MEM;
        }
        errcode = cudaMemcpy(hostFlagset, outFlagset, 
                                 width * height * sizeof (int),
                                 cudaMemcpyDeviceToHost);
        if (errcode != NO_ERROR) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            return CUDA_ERROR;
        }
        errcode = sortContour(hostFlagset, contourcoordiset, width, height);
        if (errcode != NO_ERROR) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            delete [] hostFlagset;
            return errcode;
        }
        if (hostFlagset != NULL) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            delete [] hostFlagset;
            return errcode;
        }
    } else {
        ImageBasicOp::copyToCurrentDevice(outimg);
        // 调用核函数，将轮廓点坐标输出到图像中。
        _contourSetToimgKer<<<grid, block>>>(outFlagset, outsubimgCud);
        if (cudaGetLastError() != cudaSuccess) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            return CUDA_ERROR;
        }
        errcode = ImageBasicOp::copyToHost(outimg);
        if (errcode != NO_ERROR) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            return errcode;
        }
        errcode = imgconForCurve.clearAllConvertFlags();
        if (errcode != NO_ERROR) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            return errcode;
        }
        errcode = imgconForCurve.setConvertFlag(255);
        if (errcode != NO_ERROR) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            return errcode;
        }
        // 调用已封装好的图像转坐标集函数，生成内环内点坐标集。
        errcode = imgconForCurve.imgConvertToCst(outimg, contourcoordiset);
        if (errcode != NO_ERROR) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            return errcode;
        }
        errcode = CoordiSetBasicOp::copyToHost(contourcoordiset);
        if (errcode != NO_ERROR) {
            // 释放内存空间。
            FAIL_GET_CONTOUR_SET_FREE;
            return errcode;
        }
    }

    // 释放内存
    FAIL_GET_CONTOUR_SET_FREE;
    // 处理完毕，退出。
    return NO_ERROR;
}

