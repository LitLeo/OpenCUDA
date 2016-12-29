// ConnectRegionNew.cu
// 实现图像的连通区域操作

#include "ConnectRegionNew.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_H
// 定义了默认的一个线程块处理图像的行数。
#define DEF_BLOCK_H   4

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X   4
#define DEF_BLOCK_Y   2

// 宏：CONNREGION_INI_IFI
// 定义了一个无穷大
#define CONNREGION_INI_IFI              0x7fffffff

// 宏：CONNREGION_INI_SMA
// 定义了一个无穷小
#define CONNREGION_INI_SMA             0 
// 宏：BEFORESCAN
// 定义开始扫描前的线程块状态
#define BEFORESCAN   0
 
// 宏：AFTERFIRSTSCAN
// 定义进行第一次扫描后的线程块状态
#define AFTERFIRSTSCAN   1

// 宏：AFTERUNIFIED
// 定义统一标记后的线程块状态
#define AFTERUNIFIED   2

// 宏：CONNREGION_PACK_LEVEL
// 定义了一个线程中计算的像素点个数，若该值为4，则在一个线程中计算2 ^ 4 = 16
// 个像素点
#define CONNREGION_PACK_LEVEL 4

#define CONNREGION_PACK_NUM   (1 << CONNREGION_PACK_LEVEL)
#define CONNREGION_PACK_MASK  (CONNREGION_PACK_LEVEL - 1)

#if (CONNREGION_PACK_LEVEL < 1 || CONNREGION_PACK_LEVEL > 5)
#  error Unsupport CONNREGION_PACK_LEVEL Value!!!
#endif


// Device 子程序：_findMinLabel
// 将标记值设为连通数组中的值
static __device__ int    // 返回值：根标记，即最小标记值
_findMinLabel(
        int *connector,  // 连通数组
        int blockx,      // 线程块索引在本程序中代表处理灰度数组的索引
        int pitch,       // 等距步长
        int label        // 标记值
);

// Device 子程序：_spin
// 保证各线程块进行同步
static __device__ void  // 无返回值
_spin(
        int n,          // 线程块要同步的状态标记
        int m,          // 线程块的数量
        int *state      // 线程块标记数组
);

// Kernel 函数: _labelingConnectRegionKer (标记连通区域)
// 标记连通区域。该过程主要分为三个部分，首先，扫描第一遍，从第一个图像的标记值
// 为其在源图像中的索引值,其后标记值为对其邻域对比后得到的标记值，如果存在领域，
// 更新为最小的邻域标记值，否则标记值加1，然后，把每个线程块的标记结果统一，最后
// 通过找到根标记，实现最终标记。
// Kernel 函数：_labelingConnectRegionKer (标记连通区域)
static __global__ void                                // 无返回值
_labelingConnectRegionKer(ImageCuda inimg,            // 输入图像
                          int *devIndGray,            // 需要处理的灰度范围数组
                          int indGrayNum,             // 需要处理的灰度组数
                          LabelMaps *devLabelM,       // 输出区域集
                          int *lastLineLabelofBlock,  // 存储每一个处理线程块的
                                                      // 最后一行标记值
                          int *connector,             // 连通数组
                          int *bState                 // 线程块状态数组
);

// Kernel 函数：_computeAreaKer
// 计算连通区域的面积
static __global__ void        // 无返回值
_computeAreaKer(
        ImageCuda inimg,      // 输入图像
        LabelMaps *devLabelM  // 输出区域集
);

// Kernel 函数：_filterAreaKer (筛选出面积大小符合要求的连通区域)
static __global__ void          // Kernel 函数无返回值
_filterAreaKer(
        ImageCuda inimg,        // 输入图像
        LabelMaps * devLabelM,  // 输入区域集
        int *devIndGray,        // 需要处理的灰度范围数组
        int *frSize,            // 各区域集中连通区域的个数数组
        int minArea,            //区域面积的最小值
        int maxArea             //区域面积的最大值
);   

// Kernel 函数：_getFilteredRegionKer (得到符合要求的面积区域的相关信息）  
static __global__ void
_getFilteredRegionKer(
        ImageCuda inimg,        // 输入图像
        LabelMaps * devLabelM,  // 输入区域集
        int *devIndGray,        // 需要处理的灰度范围数组
        int *frSize,            // 各区域集中连通区域的个数数组
        int minArea,            //区域面积的最小值
        int maxArea,            //区域面积的最大值
        int indGrayNum          // 需要处理的灰度组数
);   

// Device 子程序：_findMinLabel（中继处理）
static __device__ int _findMinLabel(int* connector, int blockx, int pitch,
                                    int label) 
{
    int  lab1 = label,lab2;
    // 循环查找并查集的根，即最小标记值。
    while ((lab2 = connector[blockx * pitch + lab1]) >= 0) {
        lab1 = lab2;
    }
    return lab1;
}

// Device 子程序：_spin (线程块同步)
static __device__ void _spin(int n, int m, int *state)
{
    int counter = 0;
    do {
        for (int i = 0; i < m; i++) {
            if (state[i] >= n)
                counter++; 
        }
    } while (counter < m);
}

// Kernel 函数：_labelingConnectRegionKer (标记连通区域)
static __global__ void _labelingConnectRegionKer(ImageCuda inimg,
                                                 int *devIndGray,
                                                 int indGrayNum,
                                                 LabelMaps *devLabelM, 
                                                 int *lastLineLabelofBlock, 
                                                 int *connector, int *bState)
{
    // 用来表示线程块的大小。
    int bSize = DEF_BLOCK_H * inimg.imgMeta.width;
    // 动态申请共享内存，使用时共享内存大小为线程块的大小，用于保存扫描后的标记
    // 值,并且数组的第一个元素初始化为所处理的第一个像素在图像中的索引值。
    extern __shared__ int bLabel[];
    // 每个线程块中的每一个元素对应的索引值。
    int bMinLabel = blockIdx.y * (bSize - inimg.imgMeta.width);
    int min = devIndGray[2 * blockIdx.x];
    int max = devIndGray[2 * blockIdx.x + 1];
    int pitch = inimg.imgMeta.width * inimg.imgMeta.height;
    
    // 记录连通数目，初始为所处理的第一个像素在图像中的索引值。
    int labelCounter = bMinLabel + 1;
    // 标记数组的第一个元素初始化为所处理的第一个像素在图像中的索引值。
    bLabel[0] = 0;
    if (inimg.imgMeta.imgData[blockIdx.y * (DEF_BLOCK_H - 1) * inimg.pitchBytes]
        >= min &&
        inimg.imgMeta.imgData[blockIdx.y * (DEF_BLOCK_H - 1) * inimg.pitchBytes]
        <= max )
        bLabel[0] = labelCounter++;

    // 线程块数。
    int countofBlock = gridDim.x * gridDim.y;
    // 标记线程块的状态，便于同步
    bState[blockIdx.y * gridDim.x + blockIdx.x] = BEFORESCAN;
    // 实现各线程块同步。
    _spin(BEFORESCAN, countofBlock, bState);

    int i;
    // 当前处理图像的像素的下标。
    int curidx;
    // 标记图像位置的列坐标和行坐标。
    int cur_x, cur_y;

    // 从图像的第 blockIdx.y BLOCK的第一行的第二个PIXEL开始，由左向右进行扫描。
    for (i = 1; i < inimg.imgMeta.width; i++) {
        cur_x = i;
        cur_y = blockIdx.y * (DEF_BLOCK_H - 1);
        curidx = cur_y * inimg.pitchBytes + cur_x;
        bLabel[i] = 0;
        // 如果该图像像素和左侧像素点连通，则将本线程块中对就位置的标记值设为左
        // 侧像素点对应位置的标记值。
        if (inimg.imgMeta.imgData[curidx] >= min &&
            inimg.imgMeta.imgData[curidx] <= max) {
            if (inimg.imgMeta.imgData[curidx - 1] >= min &&
                inimg.imgMeta.imgData[curidx - 1] <= max) {
                bLabel[i] = bLabel[i - 1];
            }else 
                bLabel[i] = labelCounter++;
        }
    }

    // 从对应图像BLOCK的第二行的第一个PIXEL开始(最左)，
    // 由左向右，由上向下地进行扫描。
    for (; i < bSize; i++) {
        // 得到对应像素在图中的位置索引值。
        cur_x = i % inimg.imgMeta.width;
        cur_y = i / inimg.imgMeta.width + blockIdx.y * (DEF_BLOCK_H - 1);
        curidx = cur_y * inimg.pitchBytes + cur_x;
        // 对应的正上方的像素的索引值。
        int upidx = i - inimg.imgMeta.width;
        // 初始化标志值以区分该像素是否已被处理过。
        bLabel[i] = 0;

        // 如果该位置的像素在所要处理的灰度范围内，则对该像素进行处理并进行标记
        // 标记原则为：从左侧的对应像素开始处理，比对这个左侧像素是否也在所要处
        // 理的灰度范围内，如果也在，将该像素位置的对应的标记值更新为较小的标记
        // 值，并更新连通数组。右上方和正上方以及左上方的处理方式类似。
        if (inimg.imgMeta.imgData[curidx] >= min &&
            inimg.imgMeta.imgData[curidx] <= max) {
            // 先处理左方像素。
            int leftLabel;
            if (cur_x > 0 &&
                inimg.imgMeta.imgData[curidx - 1] >= min &&
                inimg.imgMeta.imgData[curidx - 1] <= max) {
                leftLabel = bLabel[i - 1];
            }else
                leftLabel = -2;
            // 依次处理右上方，正上方，左上方的像素，并检查是否也在所要处理的
            // 灰度范围内判断是否和该像素连通。
            for (int times = 0;times < 4;times++) {
                // 处理右上方的像素，检查是否也在所要处理的灰度范围内以判断是否
                // 和该像素连通。
                if (times == 0 && cur_x < inimg.imgMeta.width - 1) {
                    if (inimg.imgMeta.imgData[curidx - inimg.pitchBytes + 1] >=
                        min &&
                        inimg.imgMeta.imgData[curidx - inimg.pitchBytes + 1] <=
                        max)
                        bLabel[i] = bLabel[upidx + 1];
                }
                // 处理正上方的像素，检查是否也在所要处理的灰度范围内以判断是否
                // 和该像素连通。
                if (times == 1) {
                    if (inimg.imgMeta.imgData[curidx - inimg.pitchBytes] >= min
                        &&
                        inimg.imgMeta.imgData[curidx - inimg.pitchBytes] <=
                        max) {
                        if (bLabel[i] != 0) {
                            int a = _findMinLabel(connector, blockIdx.x, pitch,
                                                  bLabel[i]);
                            int b = _findMinLabel(connector, blockIdx.x, pitch,
                                                  bLabel[upidx]);
                            if (a != b) {
                                int c = (a > b) ? a : b;
                                connector[blockIdx.x * pitch + c] = a + b - c;
                            }
                        }else
                            bLabel[i] = bLabel[upidx];
                    }
                }
                // 处理左上方的像素，检查是否也在所要处理的灰度范围内以判断是否
                // 和该像素连通。
                if (times == 2 && cur_x > 0) {
                    if (inimg.imgMeta.imgData[curidx - inimg.pitchBytes - 1] >=
                        min &&
                        inimg.imgMeta.imgData[curidx - inimg.pitchBytes - 1] <=
                        max) {
                        if (bLabel[i] != 0) {
                            int a = _findMinLabel(connector, blockIdx.x, pitch,
                                                  bLabel[i]);
                            int b = _findMinLabel(connector, blockIdx.x, pitch,
                                                  bLabel[upidx - 1]);
                            if (a != b) {
                                int c = (a > b) ? a : b;
                                connector[blockIdx.x * pitch + c] = a + b - c;
                            }
                        }else
                            bLabel[i] = bLabel[upidx - 1];
                    }
                }
                // 如果该像素的左侧像素和右上方像素都连通，则更新连通区域使左侧
                // 像素对应位置的标志值与右上方保持一致。左侧像素与正上方像素及
                // 左上方像素和右上方像素，左侧像素与正上方像素同理保持一致。
                if (times == 3) {
                    if (leftLabel != -2) {
                        if (bLabel[i] != 0) {
                            int a = _findMinLabel(connector, blockIdx.x, pitch,
                                                  bLabel[i]);
                            int b = _findMinLabel(connector, blockIdx.x, pitch,
                                                  leftLabel);
                            if (a != b) {
                                int c = (a > b) ? a : b;
                                connector[blockIdx.x * pitch + c] = a + b - c;
                            }
                        }else
                            bLabel[i] = leftLabel;
                    }
                }
            }
            if (bLabel[i] == 0)
                bLabel[i] = (leftLabel != -2) ? leftLabel : labelCounter++;
        }
    }

    // 进行bh行扫描后，将bLabel的最后一行的内容copy到
    // lastLineLabelofBlock [blockIdx.y][]中。
    for (i = 0; i < inimg.imgMeta.width; i++) {
        if (blockIdx.y < gridDim.y - 1) {
            lastLineLabelofBlock[blockIdx.x * inimg.imgMeta.width *
                    (gridDim.y - 1) + blockIdx.y * inimg.imgMeta.width + i] =
                    bLabel[bSize - inimg.imgMeta.width + i];
        }
    }

    // 更新各大线程块的状态。
    bState[blockIdx.y * gridDim.x + blockIdx.x] = AFTERFIRSTSCAN;
    // 实现各线程块同步。
    _spin(AFTERFIRSTSCAN, countofBlock, bState);
  
    // 根据连通数组更新标记值实现不同的线程块处理的同一行的像素值对应的标记值相
    // 同。
    if (blockIdx.y > 0) {
        for (i = 0; i < inimg.imgMeta.width; i++) {
            if (bLabel[i] != 0) {
                int a = _findMinLabel(connector, blockIdx.x, pitch, bLabel[i]);
                int b = _findMinLabel(connector, blockIdx.x, pitch,
                                      lastLineLabelofBlock[blockIdx.x *
                                      inimg.imgMeta.width * (gridDim.y - 1) +
                                      (blockIdx.y - 1) * inimg.imgMeta.width +
                                      i]);
                if (a != b) {
                    int c = (a > b) ? a : b;
                    connector[blockIdx.x * pitch + c] = a + b - c;
                }
            }
        }
    }

    // 更新各大线程块的状态。
    bState[blockIdx.y * gridDim.x + blockIdx.x] = AFTERUNIFIED;
    // 实现各线程块同步。
    _spin(AFTERUNIFIED, countofBlock, bState);

    // 将最终结果输出到输出标记值数组中。
    int gbMinLabel = blockIdx.y * (DEF_BLOCK_H - 1) * inimg.imgMeta.width;

    for (i = 0; i < bSize; i++) {
        // 找到最小的标记值。
        int  trueLabel = _findMinLabel(connector, blockIdx.x, pitch, bLabel[i]);
        devLabelM[blockIdx.x].gLabel[gbMinLabel + i] = trueLabel; 
    }
}

// Kernel 函数：_computeAreaKer (计算连通区域的面积)
static __global__ void
_computeAreaKer(ImageCuda inimg, LabelMaps *devLabelM)
{
    // 所要处理的图们大小对应的宽度。
    int width = inimg.imgMeta.width;
    // 所要处理的图们大小对应的高度。
    int height = inimg.imgMeta.height;
    // 初始化区域个数。
    devLabelM[blockIdx.z].regionCount = 0;

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
        curlabel = devLabelM[blockIdx.z].gLabel[inidx];
       // if (blockIdx.z == 1)
//printf("curlabel = %d\n", curlabel);
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
        nexlabel = devLabelM[blockIdx.z].gLabel[inidx];

        // 若当前第二个点的标记值不等于前一个，把当前临时变量 cursum 中的统计结
        // 果增加到共享内存中的相应区域；若该值等于前一个点的标记值，则临时变量
        // cursum 加 1，继续检查下一个像素点。
        if (curlabel != nexlabel) {
            atomicAdd(&devLabelM[blockIdx.z].area[curlabel], cursum);
            curlabel = nexlabel;
        } else {
            cursum++;
        }

        // 宏：CONNREGION_KERNEL_MAIN_PHASE
        // 定义计算下一个像素点的程序片段。使用这个宏可以实现获取下一个点的像素
        // 值，并累加到共享内存，并且简化编码量
#define CONNREGION_KERNEL_MAIN_PHASE                                      \
        if (++r >= height)                                                \
            break;                                                        \
        inidx += width;                                                   \
        nexlabel = devLabelM[blockIdx.z].gLabel[inidx];                   \
        if (curlabel != nexlabel) {                                       \
            atomicAdd(&devLabelM[blockIdx.z].area[curlabel], cursum);     \
            curlabel = nexlabel;                                          \
            cursum = 1;                                                   \
        } else {                                                          \
            cursum++;                                                     \
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
        atomicAdd(&devLabelM[blockIdx.z].area[curlabel], cursum);
}

// Kernel 函数：_filterAreaKer (筛选出面积大小符合要求的连通区域)
static __global__ void
_filterAreaKer(ImageCuda inimg, LabelMaps * devLabelM, int *devIndGray,
               int *frSize, int minArea, int maxArea)   
{
    int imgSize = inimg.imgMeta.width * inimg.imgMeta.height;
    // 初始化 regionCount 的大小为0.
    devLabelM[blockIdx.x].regionCount = 0;

    for (int i = 0; i < imgSize; i++) {
        // 遍历每一个标记值对应的连通区域对应的面积的大小，如果面积大小在要求范
        // 围内， regionCount 的大小加1，否则，将面积值置为0，得到区域的个数
        if (devLabelM[blockIdx.x].area[i] > minArea &&
            devLabelM[blockIdx.x].area[i] < maxArea)
            devLabelM[blockIdx.x].regionCount++;
        else devLabelM[blockIdx.x].area[i] = 0;
    }

   // 把 regionCount (区域个数) 赋给 frSize 数组，用于开辟保存区域的具体信息的结
   // 构体的空间
   frSize[blockIdx.x] = devLabelM[blockIdx.x].regionCount;
}

// Kernel 函数：_getFilteredRegionKer (得到符合要求的面积区域的相关信息）  
static __global__ void
_getFilteredRegionKer(ImageCuda inimg, LabelMaps * devLabelM, int *devIndGray,
                      int *frSize, int minArea, int maxArea, int indGrayNum)   
{   
    // 图像大小。
    int imgSize = inimg.imgMeta.width * inimg.imgMeta.height;
    int currentLabel;
    // 要筛选的连通区域的最小值。
    int min = devIndGray[2 * blockIdx.x];
    // 要筛选的连通区域的最大值。
    int max = devIndGray[2 * blockIdx.x + 1];
    // 对应的 LABEL MEMORY 号(BLOCK列index)。
    int index = (min + max) / 2;

    // 初始化每个连通区域的索引值和标记值，索引值即为对应的 LABEL MEMORY 号。
    for (int i = 0,k = 0; k < frSize[blockIdx.x]; i++)  
        if (devLabelM[blockIdx.x].area[i] != 0) {
           devLabelM[blockIdx.x].fr[k].index = index;
           devLabelM[blockIdx.x].fr[k++].labelMapNum = i;
        }

    // 初始化每个连通区域的左上角坐标和右下角坐标。(左上角坐标初始化为无限大，右
    // 下角坐标初始化为无限小)
    for (int i = 0;i < devLabelM[blockIdx.x].regionCount;i++) {
    
       devLabelM[blockIdx.x].fr[i].regionX1 = CONNREGION_INI_IFI;
       devLabelM[blockIdx.x].fr[i].regionY1 = CONNREGION_INI_IFI;
       devLabelM[blockIdx.x].fr[i].regionX2 = CONNREGION_INI_SMA;
       devLabelM[blockIdx.x].fr[i].regionY2 = CONNREGION_INI_SMA;
   }

    // 遍历每一个标记值，找到其对应的连通区域，通过比较得到左上角坐标和右下角坐
    // 标(因为已经将每一个连通区域的左上角和右下角初始化，在遍历的过程对，将每一
    // 个标记值对应的在图像中的坐标找到，如果当前坐标值的横坐标小于 regionX1，则
    // 更新为当前的坐标的横坐标，regionY1,regionX1,regionY2 的更新同理。)
    for (int i = 0; i < imgSize; i++) {
        currentLabel = devLabelM[blockIdx.x].gLabel[i];
        if (devLabelM[blockIdx.x].gLabel[i] != 0 ) {
            // 得到当前位置对应的图像中的纵坐标值。
            int y = i / inimg.imgMeta.width;
            // 得到当前位置对应的图像中的横坐标值。
            int x = i % inimg.imgMeta.width;
            // 得到当前位置对应的标记值 。
            currentLabel = devLabelM[blockIdx.x].gLabel[i];
            for (int i = 0;i < devLabelM[blockIdx.x].regionCount;i++) {
                // 找到存储对应的连通区域信息的结构体。
                if (currentLabel ==  devLabelM[blockIdx.x].fr[i].labelMapNum) {
                    // 更新左上角的横坐标，取最小值。
                    if (x < devLabelM[blockIdx.x].fr[i].regionX1)
                        devLabelM[blockIdx.x].fr[i].regionX1 = x;
                    // 更新左上角的纵坐标，取最小值。
                    if (x > devLabelM[blockIdx.x].fr[i].regionX2)
                        devLabelM[blockIdx.x].fr[i].regionX2 = x;
                    // 更新右下角的横坐标，取最大值。
                    if (y <devLabelM[blockIdx.x].fr[i].regionY1)
                        devLabelM[blockIdx.x].fr[i].regionY1 = y;
                    // 更新右下角的纵坐标，取最大值。
                    if (y > devLabelM[blockIdx.x].fr[i].regionY2)
                        devLabelM[blockIdx.x].fr[i].regionY2 = y;
                }
            }
        }
    }
}

// 宏：FAIL_CONNECT_REGION_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_CONNECT_REGION_NEW_FREE  do {  \
        if (devtmplabel != NULL)            \
            cudaFree(devtmplabel);          \
        if (devLabel != NULL)               \
            cudaFree(devLabel);             \
        if (connector != NULL)              \
            cudaFree(connector);            \
        if (bState != NULL)                 \
            cudaFree(bState);               \
        if (devFrsize != NULL)              \
            cudaFree(devFrsize);            \
        if (devIndGray != NULL)             \
            cudaFree(devIndGray);           \
        if (devLabelM != NULL)              \
            cudaFree(devLabelM);            \
    } while (0)

// Host 成员方法：connectRegionNew（连通区域新方法）
__host__ int  ConnectRegionNew::connectRegionNew(Image *inimg, int * indGray,
        int indGrayNum, LabelMaps *labelM)
{
    // 检查输入输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    cudaError_t cudaerrcode;
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算各标记数组的存储数据大小
    int data_size = insubimgCud.imgMeta.width * insubimgCud.imgMeta.height *
                    sizeof (int);

    // 存储中间标记值的数组，其大小与输入图像大小一致
    int *devtmplabel;
    // 存储最终标记值的数组，其大小与输入图像大小一致
    int *devLabel;
    // 存储连通数组，其大小与输入图像大小 一致。
    int *connector;
    // 存储线程块状态数组，其大小等于线程块数。
    int *bState;
    // 存储连通区域个数的数组，其大小等于需要处理的灰度组数。
    int *hstFrsize = new int[indGrayNum];
    // 在设备端存储连通区域个数的数组。
    int *devFrsize;
    // 需要处理的灰度范围数组。
    int * devIndGray;
    // device 端区域集，用于处理信息。
    LabelMaps *devLabelM;
    // 临时的区域集，用于在 host 端开辟空间。
    LabelMaps *tmpMaps = new LabelMaps[indGrayNum];

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = 1;
    blocksize.y = 1;
    gridsize.x = indGrayNum;
    // 每个线程格中的总线程块数为把图像以 DEF_BLOCK_H 为一个线程块分割，并保证上
    // 下两个线程处理的数据有一个重合的行。
    gridsize.y = (insubimgCud.imgMeta.height + DEF_BLOCK_H - 3) /
                 (DEF_BLOCK_H - 1);

    // 给设备端连通区域个数的数组分配空间。
    cudaerrcode = cudaMalloc((void **)&devFrsize, indGrayNum * sizeof(int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_CONNECT_REGION_NEW_FREE;
        return cudaerrcode;
    }
    
    // 为标记数组分配大小。
    cudaerrcode = cudaMalloc((void **)&devLabel, data_size);
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_CONNECT_REGION_NEW_FREE;
        return cudaerrcode;
    }

    // 为临时标记数组分配大小。
    cudaerrcode = cudaMalloc((void **)(&devtmplabel),
            indGrayNum * insubimgCud.imgMeta.width * (gridsize.y - 1) *
            sizeof(int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_CONNECT_REGION_NEW_FREE;
        return cudaerrcode;
    }

    // 为连通数组分配大小。
    cudaerrcode = cudaMalloc((void **)(&connector),
            indGrayNum * insubimgCud.imgMeta.width * 
            insubimgCud.imgMeta.height * sizeof(int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_CONNECT_REGION_NEW_FREE;
        return cudaerrcode;
    }

    // 为线程块状态数组分配大小。
    cudaerrcode = cudaMalloc((void **)(&bState),
                             gridsize.x * gridsize.y * sizeof(int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_CONNECT_REGION_NEW_FREE;
        return cudaerrcode;
    }
 
    // 将线程块状态数组中所有值初始化为 0。
    cudaerrcode = cudaMemset(bState, 0, gridsize.x * gridsize.y * sizeof(int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_CONNECT_REGION_NEW_FREE;
        return cudaerrcode;
    }
    
    // 将连通数组初始化为 -1。
    cudaerrcode = cudaMemset(connector, -1, indGrayNum *
                                            insubimgCud.imgMeta.width * 
                             insubimgCud.imgMeta.height * sizeof(int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_CONNECT_REGION_NEW_FREE;
        return cudaerrcode;
    }

    // 核函数中使用的共享内存的大小。
    int bsize = sizeof (int) * DEF_BLOCK_H * insubimgCud.imgMeta.width;

    // 为要处理的灰度范围数组开辟空间，大小为 2 * indGrayNum。
    cudaerrcode = cudaMalloc((void **)&devIndGray,
                             2 * indGrayNum * sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_CONNECT_REGION_NEW_FREE;
        return cudaerrcode;
    }

    // 为 device 端区域集开辟空间，大小为 indGryaNum 个区域集结构体。
    cudaerrcode = cudaMalloc((void **)&devLabelM,
                              indGrayNum * sizeof (LabelMaps));
    if (cudaerrcode != cudaSuccess) {
        // 释放内存空间。
        FAIL_CONNECT_REGION_NEW_FREE;
        return cudaerrcode;
    }

    // 用于方向区域集的下标变量。
    int lmsize;

    // 为每一个区域集中的标记值数组和面积数组开辟空间。大小和图像大小一致。
    for (lmsize = 0; lmsize < indGrayNum; lmsize++) {
        cudaerrcode = cudaMalloc((void **)&(tmpMaps[lmsize].gLabel), data_size);
        cudaerrcode = cudaMalloc((void **)&(tmpMaps[lmsize].area), data_size);
        cudaerrcode = cudaMemset(tmpMaps[lmsize].area, 0, data_size);
        if (cudaerrcode != cudaSuccess) {
            // 释放内存空间。
            FAIL_CONNECT_REGION_NEW_FREE;
            return cudaerrcode;
        }
    }

    // 把输入的灰度数组拷贝到设备端。
    cudaMemcpy(devIndGray, indGray, 2 * indGrayNum * sizeof (int),
               cudaMemcpyHostToDevice);
    // 把输入区域集拷贝到设备端。
    cudaMemcpy(devLabelM, tmpMaps, indGrayNum * sizeof (LabelMaps),
               cudaMemcpyHostToDevice);

    // 调用核函数，初始化每个线程块内标记值。并将结果计算出来，划分出连通区域。
    _labelingConnectRegionKer<<<gridsize, blocksize, bsize>>>(
            insubimgCud, devIndGray, indGrayNum, devLabelM, devtmplabel,
            connector, bState);

    // 计算调用计算面积的 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blockforcalarea, gridforcalarea;
    int height = (insubimgCud.imgMeta.height + CONNREGION_PACK_MASK) /
                  CONNREGION_PACK_NUM;
    blockforcalarea.x = DEF_BLOCK_X;
    blockforcalarea.y = DEF_BLOCK_Y;
    gridforcalarea.x = (insubimgCud.imgMeta.width + blockforcalarea.x - 1) /
                        blockforcalarea.x;
    gridforcalarea.y = (height + blockforcalarea.y - 1) / blockforcalarea.y;
    gridforcalarea.z = indGrayNum;
    // 计算每一个区域的面积
    _computeAreaKer<<<gridforcalarea, blockforcalarea>>>(insubimgCud,
                                                         devLabelM);
    if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
           cout << "error" << endl;
            
            return CUDA_ERROR;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 mblocksize, mgridsize;
    mblocksize.x = 1;
    mblocksize.y = 1;
    mgridsize.x = indGrayNum;
    mgridsize.y = 1;

   // 根据各区域的大小筛选出符合面积要求的连通区域，并将区域个数存储在 devFrsize
   // 数组中，用于开辟新的空间存放连通区域的具体信息
    _filterAreaKer<<<mgridsize,mblocksize>>>(insubimgCud,devLabelM, devIndGray,
                                             devFrsize, minArea, maxArea);
    if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
           cout << "error" << endl;
            
            return CUDA_ERROR;
    }

    // 将连通区域的数组个数数组数据从 device 端拷到 host 端。
    cudaMemcpy(hstFrsize, devFrsize, indGrayNum * sizeof (int),
               cudaMemcpyDeviceToHost);

    // 将连通区域集的结果从设备端拷到主机端。
    cudaMemcpy(tmpMaps, devLabelM, indGrayNum * sizeof (LabelMaps),
               cudaMemcpyDeviceToHost);

    // 为记录连通区域具体作息的结构体分配空间。
    for (lmsize = 0; lmsize < indGrayNum; lmsize++) {
        cudaerrcode = cudaMalloc((void **)&(tmpMaps[lmsize].fr),
                                 hstFrsize[lmsize] * sizeof(FilteredRegions));
        if (cudaerrcode != cudaSuccess) {
            // 释放内存空间。
            FAIL_CONNECT_REGION_NEW_FREE;
            return cudaerrcode;
        }
    }

    // 把分配好空间的区域集从 host 端拷到 device端。
    cudaMemcpy(devLabelM, tmpMaps, indGrayNum * sizeof (LabelMaps),
               cudaMemcpyHostToDevice);

    // 得到符合条件的连通区域的相关信息：左上角坐标值、右下角坐标值、标记值、对
    // 应的 LABEL MEMORY 号(BLOCK列index)
    _getFilteredRegionKer<<<indGrayNum,1>>>(insubimgCud,devLabelM, devIndGray,
                                             devFrsize, minArea, maxArea,
                                             indGrayNum);
    if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
           cout << "error" << endl;
            
            return CUDA_ERROR;
    }

    // 进行最后拷贝，把区域集的完整信息从 device 端拷贝到 host 端
    cudaMemcpy(hstFrsize, devFrsize, indGrayNum * sizeof (int),
               cudaMemcpyDeviceToHost);

    // 保存标记值数组的指针值。
    int *devGlabel;
    // 保存区域结构体数组的指针值。
    FilteredRegions *devFr;
    // 将区域集的指针从 device 端拷贝到 host 端。
    cudaMemcpy(labelM, devLabelM, indGrayNum * sizeof (LabelMaps),
               cudaMemcpyDeviceToHost);
    // 通过拷贝得到的指针得到区域集的完整信息。
    for (lmsize = 0; lmsize < indGrayNum; lmsize++) {
        devGlabel =labelM[lmsize].gLabel;
        devFr = labelM[lmsize].fr;
        labelM[lmsize].gLabel = new int[insubimgCud.imgMeta.width *
                                        insubimgCud.imgMeta.height];
        labelM[lmsize].fr = new FilteredRegions[hstFrsize[lmsize]];
        labelM[lmsize].regionCount = hstFrsize[lmsize];
        cudaMemcpy(labelM[lmsize].gLabel, devGlabel, insubimgCud.imgMeta.width *
                   insubimgCud.imgMeta.height * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(labelM[lmsize].fr, devFr, hstFrsize[lmsize] *
                   sizeof(FilteredRegions), cudaMemcpyDeviceToHost);
        cudaFree(devFr);
        cudaFree(devGlabel);
    }

// 以下程序段用于检测输出结果
    for (lmsize = 0; lmsize < indGrayNum; lmsize++) {
        printf("labelM[%d].regionCount = %d\n", lmsize, labelM[lmsize].regionCount);
        for (int i = 0; i < insubimgCud.imgMeta.width *
             insubimgCud.imgMeta.height; i++) {
            if (i % insubimgCud.imgMeta.width == 0)
                printf("\n");
            printf("%4d",labelM[lmsize].gLabel[i]);
        }
        printf("\n");
        for (int i = 0; i < labelM[lmsize].regionCount; i++) {
            printf("labelM[%d}.fr[%d].index = %d\n",lmsize,
                   i,labelM[lmsize].fr[i].index);
            printf("labelM[%d}.fr[%d].labelMapNum = %d\n", lmsize, i,
                   labelM[lmsize].fr[i].labelMapNum);
            printf("labelM[%d}.fr[%d].X1 = %d\n",lmsize, i,
                   labelM[lmsize].fr[i].regionX1);
            printf("labelM[%d}.fr[%d].Y1 = %d\n",lmsize, i,
                   labelM[lmsize].fr[i].regionY1);
            printf("labelM[%d}.fr[%d].X2 = %d\n",lmsize, i,
                   labelM[lmsize].fr[i].regionX2);
            printf("labelM[%d}.fr[%d].Y2 = %d\n",lmsize, i,
                   labelM[lmsize].fr[i].regionY2);
        }
    }
    
    // 释放已分配的数组内存，避免内存泄露。
    delete []tmpMaps;
    cudaFree(devFrsize);
    cudaFree(devIndGray);
    cudaFree(devLabelM);
    cudaFree(devtmplabel);
    cudaFree(devLabel);
    cudaFree(connector);
    cudaFree(bState);
        
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。	
    return NO_ERROR;
}

