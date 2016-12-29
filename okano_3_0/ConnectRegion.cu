// ConnectRegion.cu
// 实现图像的连通区域操作

#include "ConnectRegion.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：CONNREGION_PACK_LEVEL
// 定义了一个线程中计算的像素点个数，若该值为4，则在一个线程中计算2 ^ 4 = 16
// 个像素点
#define CONNREGION_PACK_LEVEL 5

#define CONNREGION_PACK_NUM   (1 << CONNREGION_PACK_LEVEL)
#define CONNREGION_PACK_MASK  (CONNREGION_PACK_LEVEL - 1)

#if (CONNREGION_PACK_LEVEL < 1 || CONNREGION_PACK_LEVEL > 5)
#  error Unsupport CONNREGION_PACK_LEVEL Value!!!
#endif

// 宏：CONNREGION_DIFF_INT
// 比较两个值的绝对值之差是否小于给定值，若是返回1，若不是，返回0
#define CONNREGION_DIFF_INT(p1, p2, t1,t2)  ((p1 >= t1 && p1 <= t2) && (p2 >= t1 && p2 <=t2))

// 宏：CONNREGION_INI_IFI
// 定义了一个无穷大
#define CONNREGION_INI_IFI              0x7fffffff

// Device 子程序: _findRootDev
// 查找根节点标记值算法，根据给定的 label 数组和坐标值
// 返回该坐标对应的根节点坐标值。该函数是为了便于其他 Kernel 函数调用。
static __device__ int  // 返回值:根节点标记值
_findRootDev(
        int *label,    // 输入的标记数组
        int idx        // 输入点的标记值
);

// Device 子程序: _unionDev
// 合并两个不同像素点以使它们位于同一连通区域中
static __device__ void          // Device 程序无返回值
_unionDev(                 
        int *label,             // 标记值数组
        unsigned char elenum1,  // 第一个像素点灰度值
        unsigned char elenum2,  // 第二个像素点灰度值
        int elelabel1,          // 第一个像素点标记值
        int elelabel2,          // 第二个像素点标记值
        int mingray, int maxgray,          // 给定阈值
        int *flag               // 变换标记，当这两个输入像素点被合并到一个
	                        // 区域后，该标记值将被设为 1。
);                                                                               

// Device 子程序: _findePreNumDev
// 计算当前行前所有行中的根节点个数。
static __device__ int  // 返回值：根节点个数
_findPreNumDev(
        int rownum,    // 当前行号
        int *elenum    // 一个长度与总行数一致的一维数组，用来记录每行中根节
                       // 点个数。
);

// Kernel 函数: _initLabelPerBlockKer (初始化每个块内像素点的标记值)
// 初始化每个线程块内点的标记值。该过程主要分为两个部分，首先，每个节点的标记值为
// 其在源图像中的索引值,如对于坐标为 (c, r) 点,其初始标记值为 r * width + c ,
// 其中 width 为图像宽;然后，将各点标记值赋值为该点满足阈值关系的八邻域点中的最
// 小标记值。该过程在一个线程块中进行。
static __global__ void    // Kernel 函数无返回值
_initLabelPerBlockKer(
        ImageCuda inimg,  // 输入图像
        int *label,       // 输入标记数组
        int mingray, int maxgray     // 指定阈值
);

// Kernel 函数: _mergeBordersKer (合并不同块内像素点的标记值)
// 不同线程块的合并过程。该过程主要合并每两个线程块边界的点，
// 在这里我们主要采用每次合并 4 × 4 个线程块的策略。
static __global__ void    // Kernel 函数无返回值
_mergeBordersKer(
        ImageCuda inimg,  // 输入图像
        int *label,       // 输入标记数组
        int blockw,       // 应合并线程块的长度
        int blockh,       // 应合并线程块的宽度
        int threadz_z,    // 合并水平方向线程块时，z 向线程最大值
        int threadz_y,    // 合并竖直方向线程块时，z 向线程最大值
        int mingray, int maxgray    // 指定阈值
);

// Kernel 函数: _preComputeAreaKer (计算面积预处理)
// 为每个节点找到其对应的根节点标记值。为下一步计算面积做准备。
static __global__ void  // Kernel 函数无返回值
_perComputeAreaKer(
        int *label,     // 输入标记数组
        int width,      // 输入图像长度
        int height      // 输入图像宽度
);

// Kernel 函数: _computeAreaKer (计算区域面积)
// 计算各个区域的面积值。
static __global__ void  // Kernel 函数无返回值
_computeAreaKer(
        int *label,     // 输入标记数组
        int *area,      // 输出各区域面积值数组
        int width,      // 输入图像长度
        int height      // 输入图像宽度
);

// Kernel 函数: _areaAnalysisKer (区域面积大小判断)
// 进行区域面积大小判断。其中不满足给定范围的区域的根节点标记值将被赋值为 -1。
static __global__ void  // Kernel 函数无返回值
_areaAnalysisKer(
        int *label,     // 输入标记数组
        int *area,      // 输入面积数组
        int width,      // 输入图像长度
        int height,     // 输入图像宽度
        int minArea,    // 区域最小面积
        int maxArea     // 区域最大面积
);

// Kernel 函数: _findRootLabelKer (寻找根节点标记值)
// 经过面积判断后，为每个节点找到其根节点。其中区域面积超出范围的
// 所有点标记值将被置为 -1。
static __global__ void  // Kernel 函数无返回值
_findRootLabelKer(
        int *label,     // 输入标记数组
        int *tmplabel,  // 输入存储临时标记数组
        int width,      // 输入图像长度
        int height      // 输入图像宽度		
);

// Kernel 函数: _reIndexKer (根据最终结果重新标记图像)
// 将输入标记数组中每行中的根节点个数输出到 elenum 数组中。
static __global__ void  // Kernel 函数无返回值
_reIndexKer(
        int *label,     // 输入标记数组
        int *labelri,   // 记录重新标记前标记值的数组
        int *elenum,    // 记录各行根节点个数的数组
        int width,      // 输入图像长度
        int height      // 输入图像宽度
);

// Kernel 函数: _reIndexFinalKer ()
// 进行区域标记值的重新赋值。
static __global__ void  // Kernel 函数无返回值
_reIndexFinalKer(
        int *label,     // 输入标记数组
        int *labelri,   // 记录重新标记前标记值的数组
        int *elenum,    // 记录各行根节点个数的数组
        int width,      // 输入图像长度
        int height      // 输入图像宽度
);

// Kernel 函数: _markFinalLabelKer (将最终标记结果输出到一幅灰度图像上)
// 将最终标记值输出到目标图像上。
static __global__ void     // Kernel 函数无返回值
_markFinalLabelKer(
        ImageCuda outimg,  // 输出图像
        int *label,        // 标记数组
        int *tmplabel      // 临时标记数组
);

// Device 子程序：_findRootDev (查找根节点标记值)
static __device__ int _findRootDev(int *label, int idx)
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
static __global__ void _initLabelPerBlockKer(ImageCuda inimg, int *label,
                                             int mingray, int maxgray)
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
        slabel[localidx - 1] = CONNREGION_INI_IFI;
    if (threadIdx.x == blockDim.x - 1)
        slabel[localidx + 1] = CONNREGION_INI_IFI;
    if (threadIdx.y == 0) {
        slabel[localidx - spitch] = CONNREGION_INI_IFI;
        if (threadIdx.x == 0)
            slabel[localidx - spitch - 1] = CONNREGION_INI_IFI;
        if (threadIdx.x == blockDim.x - 1)
            slabel[localidx - spitch + 1] = CONNREGION_INI_IFI;
    }
    if (threadIdx.y == blockDim.y - 1) {
        slabel[localidx + spitch] = CONNREGION_INI_IFI;
        if (threadIdx.x == 0)
            slabel[localidx + spitch - 1] = CONNREGION_INI_IFI;
        if (threadIdx.x == blockDim.x - 1)
            slabel[localidx + spitch + 1] = CONNREGION_INI_IFI;
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
        for (i = r - 1;i <= r + 1;i++) {
            for (j = c - 1;j <= c + 1;j++) {
                if (j == c && i == r)
                    continue;
                newidx = i * inimg.pitchBytes + j;
                newvalue = inimg.imgMeta.imgData[newidx];
                if ((i >= 0 && i < inimg.imgMeta.height
                     && j >= 0 && j < inimg.imgMeta.width)
                    && (CONNREGION_DIFF_INT(curvalue, newvalue, mingray, maxgray))) {
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
        int *label, unsigned char elenum1, unsigned char elenum2,
        int label1, int label2, int mingray, int maxgray, int *flag)
{
    int newlabel1, newlabel2;
    
    // 比较两个输入像素点的灰度值是否满足给定的阈值范围
    if (CONNREGION_DIFF_INT(elenum1, elenum2, mingray, maxgray)) {
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

static __global__ void _mergeBordersKer(
        ImageCuda inimg, int *label,
        int blockw, int blockh,
        int threadz_x, int threadz_y, int mingray, int maxgray)
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
                              idx, idx + inimg.imgMeta.width, mingray, maxgray, sflag);
                    
                    // 若当前像素点不为最左侧像素点时，即 x ！= 0 时，合并
                    // 位于当前像素点左下方像素点，其坐标值为 (x - 1, y + 1)。
                    if (x - 1 >= 0) {
                        newidx -= 1;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue,
                                  idx, idx + inimg.imgMeta.width - 1,
                                  mingray, maxgray, sflag);
                    }
                    
                    // 若当前像素点不为最右侧像素点时，x ！= inimg.imgMeta.width
                    // 时,合并位于当前像素点右下方像素点，其坐标值为
                    // (x + 1, y + 1)。
                    if (x + 1 < inimg.imgMeta.width) {
                        newidx += 2;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue,
                                  idx, idx + inimg.imgMeta.width + 1,
                                  mingray, maxgray, sflag);
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
			
            for (i = 0;i < iterateTimes;i++) {
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
                              idx, idx + 1, mingray, maxgray, sflag);
                              
                    // 若当前像素点不为最上侧像素点时，即 y ！= 0 时，合并
                    // 位于当前像素点右上方像素点，其坐标值为 (x + 1, y - 1)。
                    if (y - 1 >= 0) {
                        newidx -= inimg.pitchBytes;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue, idx, 
                                  idx - inimg.imgMeta.width + 1,
                                  mingray, maxgray, sflag);
                    }
                    
                    // 若当前像素点不为最下侧像素点时，y ！= inimg.imgMeta.height
                    // 时,合并位于当前像素点右下方像素点，其坐标值为
                    // (x + 1, y + 1)。
                    if (y + 1 < inimg.imgMeta.height) {
                        newidx = curidx + inimg.pitchBytes + 1;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue,
                                  idx, idx + inimg.imgMeta.width + 1,
                                  mingray, maxgray, sflag);
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

static __global__ void _perComputeAreaKer(
    int *label, int width, int height)
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

static __global__ void _computeAreaKer(
        int *label, int *area, int width, int height)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并
    // 行度缩减的策略，默认令一个线程处理 16 个输出像素，这四个像素位于统一列
    // 的相邻 16 行上，因此，对于 r 需要进行右移计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) << CONNREGION_PACK_LEVEL;
    int inidx = r * width + c;
    int curlabel, nexlabel;
    int cursum = 0;

    do {
        // 线程中处理第一个点。
        // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资
        // 源，一方面防止由于段错误导致的程序崩溃。
        if (r >= height || c >= width)
            break;
        // 得到第一个输入坐标点对应的标记值。
        curlabel = label[inidx];
        cursum = 1;

        // 处理第二个点。
        // 此后的像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各点
        // 之间没有变化，故不用检查。
        if (++r >= height)
            break;
        // 得到第二个点的像素值。			
        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计算。
        inidx += width;
        nexlabel = label[inidx];
        // 若当前第二个点的标记值不等于前一个，把当前临时变量 cursum 中的统计结
        // 果增加到共享内存中的相应区域；若该值等于前一个点的标记值，则临时变量
        // cursum 加 1，继续检查下一个像素点。
        if (curlabel != nexlabel) {
            atomicAdd(&area[curlabel], cursum);
            curlabel = nexlabel;
        } else {
            cursum++;
        }

        // 宏：CONNREGION_KERNEL_MAIN_PHASE
        // 定义计算下一个像素点的程序片段。使用这个宏可以实现获取下一个点的像素
        // 值，并累加到共享内存，并且简化编码量
#define CONNREGION_KERNEL_MAIN_PHASE                                 \
        if (++r >= height)                                           \
            break;                                                   \
        inidx += width;                                              \
        nexlabel = label[inidx];                                     \
        if (curlabel != nexlabel) {                                  \
            atomicAdd(&area[curlabel], cursum);                      \
            curlabel = nexlabel;                                     \
            cursum = 1;                                              \
        } else {                                                     \
            cursum++;                                                \
        }

#define CONNREGION_KERNEL_MAIN_PHASEx2                           \
        CONNREGION_KERNEL_MAIN_PHASE                             \
        CONNREGION_KERNEL_MAIN_PHASE

#define CONNREGION_KERNEL_MAIN_PHASEx4                           \
        CONNREGION_KERNEL_MAIN_PHASEx2                           \
        CONNREGION_KERNEL_MAIN_PHASEx2

#define CONNREGION_KERNEL_MAIN_PHASEx8                           \
        CONNREGION_KERNEL_MAIN_PHASEx4                           \
        CONNREGION_KERNEL_MAIN_PHASEx4

#define CONNREGION_KERNEL_MAIN_PHASEx16                          \
        CONNREGION_KERNEL_MAIN_PHASEx8                           \
        CONNREGION_KERNEL_MAIN_PHASEx8

// 对于不同的 CONNREGION_PACK_LEVEL ，定义不同的执行次数，从而使一个线程内部
// 实现对多个点的像素值的统计。
#if (CONNREGION_PACK_LEVEL >= 2)
        CONNREGION_KERNEL_MAIN_PHASEx2
#  if (CONNREGION_PACK_LEVEL >= 3)
        CONNREGION_KERNEL_MAIN_PHASEx4
#    if (CONNREGION_PACK_LEVEL >= 4)
        CONNREGION_KERNEL_MAIN_PHASEx8
#      if (CONNREGION_PACK_LEVEL >= 5)
        CONNREGION_KERNEL_MAIN_PHASEx16
#      endif
#    endif
#  endif
#endif
	
// 取消前面的宏定义。
#undef CONNREGION_KERNEL_MAIN_PHASEx16
#undef CONNREGION_KERNEL_MAIN_PHASEx8
#undef CONNREGION_KERNEL_MAIN_PHASEx4
#undef CONNREGION_KERNEL_MAIN_PHASEx2
#undef CONNREGION_KERNEL_MAIN_PHASE
    } while (0);

    // 使用原子操作来保证操作的正确性
    if (cursum != 0)
        atomicAdd(&area[curlabel], cursum);
}

static __global__ void _areaAnalysisKer(
        int *label, int *area, int width, int height,
        int minArea, int maxArea)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= width || r >= height)
        return;
		
    // 计算坐标点对应的 label 数组下标
    int idx = r * width + c;
	
    // 若面积值大于最大面积值或小于指定最小面积值，则将当前点的标记值设为 -1
    if (area[idx]) {
        if (area[idx] < minArea || area[idx] > maxArea)
            label[idx] = -1;
    }
}

static __global__ void _findRootLabelKer(
        int *label, int *tmplabel, int width, int height)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= width || r >= height)
        return;
		
    // 计算坐标点对应的 label 数组下标
    int idx = r * width + c;
    // 计算当前点根节点的标记值
    int nexidx = label[idx];

    // 将根节点值为 -1 的点的标记值设为 -1，表明该点不应被标记
    if (nexidx >= 0 && label[nexidx] == -1)
        label[idx] = -1;
    // 将像素点最终标记值赋值到 tmplabel 数组中
    tmplabel[idx] = label[idx];
}

static __device__ int _findPreNumDev(int rownum, int *elenum)
{
    int n = rownum;
    // 将最终值初始化为 0
    int finalnum = 0;
    // 计算由第 0 行至第 n-1 行内根节点的总数和，并将其值赋值给 finalnum。
    while (--n >= 0) {
        finalnum += elenum[n];
    }
    return finalnum;
}

static __global__ void _reIndexKer(
        int *label, int *labelri, int *elenum, int width, int height)
{
    // 计算线程对应的点位置，其中 colnum 和 rownum 分别表示线程处理的像素点的
    // 列号和行号。
    int rownum = blockIdx.y, colnum = threadIdx.x;
	
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (colnum >= width || rownum >= height)
        return;

    // 当输入图像的宽度大于 1024，即给定线程块时，
    // 应采取迭代的方式完成该步骤标记值的重新赋值
    // 计算迭代次数，采用向上取整的方式
    int iteratetimes = (width - 1) / blockDim.x + 1;
    int idx, rankidx = 0, i, j;
    
    // 共享内存数据区,该部分包含了记录点是否为根节点信息，当一个点是根节点时，
    // 对应元素值为 1，否则值为 0。由于未对 Kernel 的尺寸做出假设，
    // 这里使用动态申请的 Shared Memory（共享内存）。
    extern __shared__ int srowinfo[];
    // 用来记录每个线程块（即一行）中根节点的总个数
    __shared__ int selenum[1];

    if(colnum == 0)
        selenum[0] = 0;
    __syncthreads();
    
    i = colnum;
    // 计算每行中根节点的个数
    for (j = 0;j < iteratetimes;j++) {
        if (i < width) {
            // 计算 labelri 数组下标
            idx = rownum * width + i;
            // 将 labelri 中所有元素值为 -1
            labelri[idx] = -1;
            // 将当前点是否为根节点的信息返回至 srowinfo 数组中
            srowinfo[i] = (label[idx] == idx);
            // 若当前点为根节点则使用原子操作使得 selenum[0] 值加 1
            if (srowinfo[i] == 1)
                atomicAdd(&selenum[0], 1);
        }
        i += 1024;
    }
    __syncthreads();
    // 将根节点信息存入 elenum 数组中
    if (colnum == 0)
        elenum[rownum] = selenum[0];
    //__syncthreads();
    
    // 计算每个根节点在它所在行中属于第几个根节点
    for (j = 0;j < iteratetimes;j++) {
        // 若当前点是根节点则进行如下判断
        if ((colnum < width) && (srowinfo[colnum] == 1)) {
            rankidx = 1;
            idx = rownum * width + colnum;
            // 计算在当前行，根节点前的其他根节点的总个数
            for (i = 0;i < colnum;i++) {
                // 若点为根节点，则使得 rankidx 值加 1。
                if (srowinfo[i] == 1)
                    rankidx++;
            }
            // 将根节点索引值返回至数组 labelri 中
            labelri[idx] = rankidx - 1;
        }
        colnum += 1024;
    }
}

static __global__ void _reIndexFinalKer(
    int *label, int *labelri, int *elenum, int width, int height)
{
    // 计算线程对应的点位置，其中 colnum 和 rownum 分别表示线程处理的像素点的
    // 列号和行号。
    int rownum = blockIdx.y, colnum = threadIdx.x;
	
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (colnum >= width || rownum >= height)
        return;
		
    int idx, i, j;
    // 当输入图像的宽度大于 1024，即给定线程块时，
    // 应采取迭代的方式完成该步骤标记值的重新赋值
    // 计算迭代次数，采用向上取整的方式
    int iteratetimes = (width - 1) / blockDim.x + 1;
    i = colnum;
    // 将第 256 及以后根节点的标记值设为 -1。
    for (j = 0;j < iteratetimes;j++) {
        if (i < width) {
            idx = rownum * width + i;
            if (labelri[idx] >= 0) {
                // 计算根节点的标记值
                label[idx] = labelri[idx] + _findPreNumDev(rownum, elenum);
                // 若标记值大于 256，则将根节点标记值设为 -1，表示该点不应被标记
                if (label[idx] >= 256)
                    label[idx] = -1;
            }
        }
        i += 1024;
    }
}

static __global__ void _markFinalLabelKer(
        ImageCuda outimg, int *label, int *tmplabel)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的
    // 坐标的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;

    // 计算坐标点对应的 label 数组下标
    int inidx = r * outimg.imgMeta.width + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
	
    // 计算每个像素点的最终标记值
    if (tmplabel[inidx] != -1) {
        tmplabel[inidx] = label[tmplabel[inidx]];
    }

    // 由于标记值应由 1 开始，而在 tmplabel 中未标记区域的标记值为 -1。
    // 因此输出图像的标记值为 tmplabel 在该位置的标记值加 1。
    outimg.imgMeta.imgData[outidx] = tmplabel[inidx] + 1;
}

// Host 成员方法：connectRegion（连通区域）
__host__ int  ConnectRegion::connectRegion(Image *inimg, Image *outimg)
{
    int mingray = 10;
    int maxgray = 250;
    // 检查输入输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;
		
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
		
    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->roiX2 - inimg->roiX1, 
                inimg->roiY2 - inimg->roiY1);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }
	
    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
	
    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一	
    if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width)
        insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
    else
        outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;
		
    if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height)
        insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
    else
        outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;
	
    // 计算初始化块内内存时，共享内存的大小。
    int smsize = sizeof (int) * (blocksize.x + 2) * (blocksize.y + 2);
    // 计算各标记数组的存储数据大小
    int data_size = insubimgCud.imgMeta.width * insubimgCud.imgMeta.height *
                    sizeof (int);
	
    // 存储中间标记值的数组，其大小与输入图像大小一致
    int *devtmplabel;
    // 存储输入图像各行根节点数目的数组，其下标表示图像行号，元素值为
    // 各行根节点个数，数组大小与图像总行数一致
    int *devElenumPerRow;
    // 存储最终标记值的数组，其大小与输入图像大小一致
    int *devLabel;
    // 存储记录重新标记前标记值的数组，其大小与输入图像大小一致
    int *devlabelri;
    // 存储各区域面积大小的数组，其大小与输入图像大小一致，其中各区域
    // 面积记录在根节点对应元素中。
    int *devArea;
    
    cudaError_t cudaerrcode;
    
    // 为标记数组分配大小。
    cudaerrcode = cudaMalloc((void **)&devLabel, data_size);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devLabel);
        return cudaerrcode;
    }

    // 为记录各行根节点个数的数组分配大小。
    cudaerrcode = cudaMalloc((void **)&devElenumPerRow,
	                     insubimgCud.imgMeta.height * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devElenumPerRow);
        return cudaerrcode;
    }

    // 为临时标记数组分配大小。
    cudaerrcode = cudaMalloc((void **)(&devtmplabel), data_size);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devtmplabel);
        return cudaerrcode;
    }

    // 为记录重新标记前的标记数组分配大小。
    cudaerrcode = cudaMalloc((void **)(&devlabelri), data_size);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devlabelri);
        return cudaerrcode;
    }

    // 为面积数组分配大小。
    cudaerrcode = cudaMalloc((void **)(&devArea), data_size);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devArea);
        return cudaerrcode;
    }

    // 将面积数组中所有面积值初始化为 0.
    cudaerrcode = cudaMemset(devArea, 0, data_size);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devArea);
        return cudaerrcode;
    }
	
    // 调用核函数，初始化每个线程块内标记值
    _initLabelPerBlockKer<<<gridsize, blocksize, smsize>>>(
            insubimgCud, devLabel, mingray, maxgray);
		
    // 合并线程块时每次合并线程块的长、宽和高
    int blockw, blockh, blockz;
    // 计算第一次合并时，应合并线程块的长、宽和高
    // 第一次合并时，应合并线程块的长应为初始线程块长，宽为初始线程块宽
    blockw = blocksize.x;
    blockh = blocksize.y;
    // 由于这里采用的是 3 维线程块，线程块的高设为初始线程块长和宽的较大者。
    blockz = blockw;
    if (blockw < blockh)
        blockz = blockh;
    
    // 计算每次合并的线程块个数，在这里我们采用的是每次合并 4 × 4 的线程块，
    // 由于采用这种方式合并所需的迭代次数最少。
    int xtiles = 4, ytiles = 4;
    // 计算合并线程块前 GRID 的长
    int tilesizex = gridsize.x;
    // 计算合并线程块前 GRID 的宽
    int tilesizey = gridsize.y;
                           
    // 定义为进行线程块合并而采用的线程块与网格。
    dim3 blockformerge, gridformerge;
    
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
        blockformerge.x = xtiles; blockformerge.y = ytiles;
        blockformerge.z = blockz;        
        // 设置为了合并而采用的二维网格的大小。
        gridformerge.x = tilesizex; gridformerge.y = tilesizey;
        gridformerge.z = 1;
        // 调用核函数，每次合并4 × 4 个线程块内的标记值
        _mergeBordersKer<<<gridformerge, blockformerge>>>(
                insubimgCud, devLabel, blockw, blockh,
	        blocksize.x, blocksize.y, mingray, maxgray);
        // 在每次迭代后，修改应合并线程块的长和宽，因为每次合并 4 * 4 个线程块，
        // 因此，经过迭代后，应合并线程块的长和宽应分别乘 4。
        blockw *= xtiles;
        blockh *= ytiles;
    }

    // 进行最后一轮线程块的合并
    // 计算该轮应采用的三维线程块大小
    blockformerge.x = tilesizex; blockformerge.y = tilesizey;
    blockformerge.z = blockz; 
    // 设置该论应采用的网格大小，长宽高分别为1。
    gridformerge.x = 1; gridformerge.y = 1;gridformerge.z = 1;
    // 调用核函数，进行最后一轮线程块合并
    _mergeBordersKer<<<gridformerge, blockformerge>>>(
            insubimgCud, devLabel, blockw, blockh,
            blocksize.x, blocksize.y, mingray,maxgray);
				
    // 调用核函数，进行计算面积前的预处理，即找出每个结点对应的标记值，
    // 其中根节点的标记值与其自身在数组中的索引值一致
    _perComputeAreaKer<<<gridsize, blocksize>>>(
            devLabel, insubimgCud.imgMeta.width, insubimgCud.imgMeta.height);	

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blockforcalarea, gridforcalarea;
    int height = (insubimgCud.imgMeta.height + CONNREGION_PACK_MASK) /
                  CONNREGION_PACK_NUM;
    blockforcalarea.x = DEF_BLOCK_X;
    blockforcalarea.y = DEF_BLOCK_Y;
    gridforcalarea.x = (insubimgCud.imgMeta.width + blockforcalarea.x - 1) /
                        blockforcalarea.x;
    gridforcalarea.y = (height + blockforcalarea.y - 1) / blockforcalarea.y;
	
    // 调用核函数，计算各个区域的面积大小
    _computeAreaKer<<<gridforcalarea, blockforcalarea>>>(
            devLabel, devArea,
            insubimgCud.imgMeta.width, insubimgCud.imgMeta.height);
            
    // 调用核函数，进行面积大小判断，面积大小超过给定范围的区域
    // 标记值将被置为 -1。
    _areaAnalysisKer<<<gridsize, blocksize>>>(
            devLabel, devArea,
            insubimgCud.imgMeta.width, insubimgCud.imgMeta.height,
            minArea, maxArea);
            
    // 调用核函数，进行各区域的最终根节点查找
    _findRootLabelKer<<<gridsize, blocksize>>>(
            devLabel, devtmplabel,
            insubimgCud.imgMeta.width, insubimgCud.imgMeta.height);
	
    // 为标记值的重新排序预设线程块与网格的大小，这里采用一个线程块处理一行
    // 的方式进行计算，由于线程块的大小限制为 1024，因此线程块的长设为 1024
    blockforcalarea.x = 1024; blockforcalarea.y = 1; blockforcalarea.z = 1;
    gridforcalarea.x = 1; gridforcalarea.y = insubimgCud.imgMeta.height;
    gridforcalarea.z = 1;
    
    // 调用核函数，计算各行所含根节点个数，并将结果返回 devElenumPerRow 数组中
    _reIndexKer<<<gridforcalarea, blockforcalarea,
                  insubimgCud.imgMeta.width * sizeof (int)>>>(
            devLabel, devlabelri, devElenumPerRow,
            insubimgCud.imgMeta.width, insubimgCud.imgMeta.height);
            
    // 调用核函数，计算各个区域的最终有序标记值
    _reIndexFinalKer<<<gridforcalarea, blockforcalarea>>>(
            devLabel, devlabelri, devElenumPerRow,
            insubimgCud.imgMeta.width, insubimgCud.imgMeta.height);

    // 调用核函数，将最终标记值传到输出图像中
    _markFinalLabelKer<<<gridsize, blocksize>>>(
            outsubimgCud, devLabel, devtmplabel);
    
    // 释放已分配的数组内存，避免内存泄露
    cudaFree(devtmplabel);
    cudaFree(devArea);
    cudaFree(devLabel);
    cudaFree(devlabelri);

    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
  
    // 处理完毕，退出。	
    return NO_ERROR;
}

