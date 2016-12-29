// RegionGrow.cu
// 实现图像的区域生长操作，串行算法 regionGrow_serial，并行 regionGrow_parallel

#include "RegionGrow.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：REGIONGROW_INI_IFI
// 定义了一个无穷大
#define REGIONGROW_INI_IFI              0x7fffffff

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
static __device__ void            // Device 程序无返回值
_unionDev(                 
        int *label,               // 标记值数组
        unsigned char elenum1,    // 第一个像素点灰度值
        unsigned char elenum2,    // 第二个像素点灰度值
        int elelabel1,            // 第一个像素点标记值
        int elelabel2,            // 第二个像素点标记值
		unsigned char threshold,  // 输入生长规则
        int *flag                 // 变换标记，当这两个输入像素点被合并到一个
	                              // 区域后，该标记值将被设为 1。
);

// Kernel 函数: _initLabelPerBlockKer (初始化每个块内像素点的标记值)
// 初始化每个线程块内点的标记值。该过程主要分为两个部分，首先，若当前节点的灰度
// 值为种子点，则标记值设为 -1，否则节点的标记值为其在源图像中的索引值，
// 如对于坐标为 (c, r) 点，其初始标记值为 r * width + c，其中 width 为图像宽;
// 然后，将各点标记值赋值为该点满足阈值关系的八邻域点中的最小标记值。
// 该过程在一个线程块中进行。
static __global__ void           // Kernel 函数无返回值
_initLabelPerBlockKer(
        ImageCuda inimg,         // 输入图像
        int *label,              // 输入标记数组
		unsigned char seed,      // 输入种子点标记值
		unsigned char threshold  // 输入生长规则
);

// Kernel 函数: _mergeBordersKer (合并不同块内像素点的标记值)
// 不同线程块的合并过程。该过程主要合并每两个线程块边界的点，
// 在这里我们主要采用每次合并 4 × 4 个线程块的策略。
static __global__ void           // Kernel 函数无返回值
_mergeBordersKer(
        ImageCuda inimg,         // 输入图像
        int *label,              // 输入标记数组
        int blockw,              // 应合并线程块的长度
        int blockh,              // 应合并线程块的宽度
        int threadz_z,           // 合并水平方向线程块时，z 向线程最大值
        int threadz_y,           // 合并竖直方向线程块时，z 向线程最大值
		unsigned char threshold  // 输入生长规则
);

// Device 子程序：_findRootDev (查找根节点标记值)
static __device__ int _findRootDev(int *label, int idx)
{
    // 在 label 数组中查找 idx 下标对应的最小标记值，
    // 并将该值作为返回值。
    int nexidx;
    do {
        nexidx = idx;
		if (nexidx == -1)
            return -1;
        idx = label[nexidx];
    } while (idx < nexidx);
    
    // 处理完毕，返回根节点标记值。
    return idx;
}

// Kernel 函数：_initLabelPerBlockKer (初始化各线程块内像素点的标记值)
static __global__ void _initLabelPerBlockKer(ImageCuda inimg, int *label,
                                             unsigned char seed,
											 unsigned char threshold)
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

	// 若当前点的灰度值与种子点灰度值 seed 相同，则标记值设为 -1。
	if (curvalue == seed)
		newlabel = -1;

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
        slabel[localidx - 1] = REGIONGROW_INI_IFI;
    if (threadIdx.x == blockDim.x - 1)
        slabel[localidx + 1] = REGIONGROW_INI_IFI;
    if (threadIdx.y == 0) {
        slabel[localidx - spitch] = REGIONGROW_INI_IFI;
        if (threadIdx.x == 0)
            slabel[localidx - spitch - 1] = REGIONGROW_INI_IFI;
        if (threadIdx.x == blockDim.x - 1)
            slabel[localidx - spitch + 1] = REGIONGROW_INI_IFI;
    }
    if (threadIdx.y == blockDim.y - 1) {
        slabel[localidx + spitch] = REGIONGROW_INI_IFI;
        if (threadIdx.x == 0)
            slabel[localidx + spitch - 1] = REGIONGROW_INI_IFI;
        if (threadIdx.x == blockDim.x - 1)
            slabel[localidx + spitch + 1] = REGIONGROW_INI_IFI;
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
        
        // 若当前点灰度值满足生长规则，则在八邻域范围内查找也满足生长规则的点，
        // 并将这些点的最小标记值赋予记录在 newlabel 中
		if (curvalue > threshold && curvalue != seed) {
			for (i = r - 1;i <= r + 1;i++) {
				for (j = c - 1;j <= c + 1;j++) {
					if (j == c && i == r)
						continue;
					newidx = i * inimg.pitchBytes + j;
					newvalue = inimg.imgMeta.imgData[newidx];
					if ((i >= 0 && i < inimg.imgMeta.height
						&& j >= 0 && j < inimg.imgMeta.width)
						&& newvalue > threshold) {
						k = localidx + (i - r) * spitch + j - c;
						newlabel = min(newlabel, slabel[k]);
					}
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
	if (newlabel != -1) {
		j = newlabel / spitch;
		i = newlabel % spitch;
		i += blockIdx.x * blockDim.x - 1;
		j += blockIdx.y * blockDim.y - 1;
		newlabel = j * inimg.imgMeta.width + i;
	}
    label[idx] = newlabel;
}

// Device 子程序：_unionDev (合并两个不同像素点以使它们位于同一连通区域中)
static __device__ void _unionDev(
        int *label, unsigned char elenum1, unsigned char elenum2,
        int label1, int label2, unsigned char threshold, int *flag)
{
    int newlabel1, newlabel2;
    
    // 比较两个输入像素点的灰度值是否均大于等于 threshold
    if ((elenum1 > threshold && elenum2 > threshold)) {
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
        int threadz_x, int threadz_y, unsigned char threshold)
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
                              idx, idx + inimg.imgMeta.width, threshold, sflag);
                    
                    // 若当前像素点不为最左侧像素点时，即 x ！= 0 时，合并
                    // 位于当前像素点左下方像素点，其坐标值为 (x - 1, y + 1)。
                    if (x - 1 >= 0) {
                        newidx -= 1;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue,
                                  idx, idx + inimg.imgMeta.width - 1,
                                  threshold, sflag);
                    }
                    
                    // 若当前像素点不为最右侧像素点时，x ！= inimg.imgMeta.width
                    // 时,合并位于当前像素点右下方像素点，其坐标值为
                    // (x + 1, y + 1)。
                    if (x + 1 < inimg.imgMeta.width) {
                        newidx += 2;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue,
                                  idx, idx + inimg.imgMeta.width + 1,
                                  threshold, sflag);
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
                              idx, idx + 1, threshold, sflag);
                              
                    // 若当前像素点不为最上侧像素点时，即 y ！= 0 时，合并
                    // 位于当前像素点右上方像素点，其坐标值为 (x + 1, y - 1)。
                    if (y - 1 >= 0) {
                        newidx -= inimg.pitchBytes;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue, idx, 
                                  idx - inimg.imgMeta.width + 1,
                                  threshold, sflag);
                    }
                    
                    // 若当前像素点不为最下侧像素点时，y ！= inimg.imgMeta.height
                    // 时,合并位于当前像素点右下方像素点，其坐标值为
                    // (x + 1, y + 1)。
                    if (y + 1 < inimg.imgMeta.height) {
                        newidx = curidx + inimg.pitchBytes + 1;
                        newvalue = inimg.imgMeta.imgData[newidx];
                        _unionDev(label, curvalue, newvalue,
                                  idx, idx + inimg.imgMeta.width + 1,
                                  threshold, sflag);
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

static __global__ void _markFinalLabelKer(
        ImageCuda outimg, int *label)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量 (其中, c 表示 column; r 表示 row)。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;

    // 计算输入坐标点在label数组中对应的数组下标
    int inidx = r * outimg.imgMeta.width + c;
    // 计算输入坐标点在图像数组中对应的数组下标
    int outidx = r * outimg.pitchBytes + c;
	label[inidx] = _findRootDev(label, label[inidx]);
	outimg.imgMeta.imgData[outidx] = 0;

    if (label[inidx] == -1)
		outimg.imgMeta.imgData[outidx] = 255;
}

// Host 成员方法：regionGrow_parallel
__host__ int  RegionGrow::regionGrow_parallel(Image *inimg, Image *outimg)
{
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

    // 存储最终标记值的数组，其大小与输入图像大小一致
    int *devLabel;
    
    cudaError_t cudaerrcode;
    
    // 为标记数组分配大小。
    cudaerrcode = cudaMalloc((void **)&devLabel, data_size);
    if (cudaerrcode != cudaSuccess) {
        cudaFree(devLabel);
        return cudaerrcode;
    }

    // 调用核函数，初始化每个线程块内标记值
    _initLabelPerBlockKer<<<gridsize, blocksize, smsize>>>(
            insubimgCud, devLabel, seed, threshold);

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
				blocksize.x, blocksize.y, threshold);
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
				blocksize.x, blocksize.y, threshold);

    // 调用核函数，将最终标记值传到输出图像中
    _markFinalLabelKer<<<gridsize, blocksize>>>(
            outsubimgCud, devLabel);

    // 释放已分配的数组内存，避免内存泄露
    cudaFree(devLabel);

    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。
    return NO_ERROR;
}

/*__host__ int RegionGrow::regionGrow_serial(Image *inimg, Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

	static int nDx[] = {-1,  0,  1, -1, 1, -1, 0, 1};
    static int nDy[] = {-1, -1, -1,  0, 0,  1, 1, 1};
    // 定义堆栈，存储坐标
	int * pnGrowQueX ;
	int * pnGrowQueY ;
	int * pUnRegion;
	pUnRegion = new int [inimg->width * inimg->height];

	// 分配空间
	pnGrowQueX = new int [inimg->width * inimg->height];
	pnGrowQueY = new int [inimg->width * inimg->height];
	
	// 定义堆栈的起点和终点
	// 当nStart=nEnd, 表示堆栈中只有一个点
	int nStart = 0;
	int nEnd = 0;
	int index;
	int nCurrX, nCurrY, xx, yy;
    int originFlag = 1;                             
	memset(pUnRegion,0,sizeof(int)*inimg->width*inimg->height);
	int i, j;

	for (i = 0; i < inimg->width; i++)
	    for (j = 0; j < inimg->height; j++) {
        index = j * inimg->width + i;
        outimg->imgData[index] = 0;
    }

	for (int i = 0; i < inimg->width; i++)
	    for (int j = 0; j < inimg->height; j++)
		{
		    index = j * inimg->width + i;
		    if (inimg->imgData[index] == seed) {
			    // 把种子点的坐标压入栈
				outimg->imgData[index] = seed;
                nStart = nEnd = 0;
	            pnGrowQueX[nEnd] = i;
				pnGrowQueY[nEnd] = j;
                originFlag++;

				while (nStart <= nEnd) {
					nCurrX = pnGrowQueX[nStart];
					nCurrY = pnGrowQueY[nStart];
					for (int k = 0;k < 8;k++) {
						// 4邻域象素的坐标
						xx = nCurrX + nDx[k];
						yy = nCurrY + nDy[k];
						// 判断象素(xx，yy) 是否在图像内部
						// 判断象素(xx，yy) 是否已经处理过
						if (xx >= 0 && xx < inimg->width
						    && yy >= 0 && yy < inimg->height
							&& pUnRegion[yy * inimg->width + xx] != originFlag
							&& inimg->imgData[yy * inimg->width + xx] > threshold) {
							// 堆栈的尾部指针后移一位
							nEnd++;
							// 象素(xx，yy) 压入栈
							pnGrowQueX[nEnd] = xx;
							pnGrowQueY[nEnd] = yy;
							// 把象素(xx，yy)设置成逻辑（）
							// 同时也表明该象素处理过
							pUnRegion[yy * inimg->width + xx] = originFlag;
                            outimg->imgData[yy * outimg->width + xx] = seed;
						}
					}
					nStart++;
				}
			}
		}
	// 释放内存
	delete []pnGrowQueX;
	delete []pnGrowQueY;
	delete []pUnRegion;
	pnGrowQueX = NULL ;
	pnGrowQueY = NULL ;
	return NO_ERROR;
}*/

__host__ int RegionGrow::regionGrow_serial(Image *inimg, Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

	static int nDx[] = {-1,  0,  1, -1, 1, -1, 0, 1};
    static int nDy[] = {-1, -1, -1,  0, 0,  1, 1, 1};
    // 定义堆栈，存储坐标
	int * pnGrowQueX ;
	int * pnGrowQueY ;
	int * pUnRegion;
	pUnRegion = new int [inimg->width * inimg->height];

	// 分配空间
	pnGrowQueX = new int [inimg->width * inimg->height];
	pnGrowQueY = new int [inimg->width * inimg->height];
	
	// 定义堆栈的起点和终点
	// 当nStart=nEnd, 表示堆栈中只有一个点
	int nStart = 0;
	int nEnd = 0;
	int index;
	int nCurrX, nCurrY, xx, yy;                   
	memset(pUnRegion,0,sizeof(int)*inimg->width*inimg->height);
	int i, j;

	for (i = 0; i < inimg->width; i++)
	    for (j = 0; j < inimg->height; j++) {
        index = j * inimg->width + i;
        outimg->imgData[index] = 0;
    }
	for (int i = 0; i < inimg->width; i++)
	    for (int j = 0; j < inimg->height; j++)
		{
		    index = j * inimg->width + i;
		    if (inimg->imgData[index] == seed && pUnRegion[index] == 0) {
			    // 把种子点的坐标压入栈
				outimg->imgData[index] = seed;
                nStart = nEnd = 0;
	            pnGrowQueX[nEnd] = i;
				pnGrowQueY[nEnd] = j;

				while (nStart <= nEnd) {
					nCurrX = pnGrowQueX[nStart];
					nCurrY = pnGrowQueY[nStart];
					for (int k = 0;k < 8;k++) {
						// 4邻域象素的坐标
						xx = nCurrX + nDx[k];
						yy = nCurrY + nDy[k];
						// 判断象素(xx，yy) 是否在图像内部
						// 判断象素(xx，yy) 是否已经处理过
						if (xx >= 0 && xx < inimg->width
						    && yy >= 0 && yy < inimg->height
							&& pUnRegion[yy * inimg->width + xx] == 0
							&& inimg->imgData[yy * inimg->width + xx] > threshold) {
							// 堆栈的尾部指针后移一位
							nEnd++;
							// 象素(xx，yy) 压入栈
							pnGrowQueX[nEnd] = xx;
							pnGrowQueY[nEnd] = yy;
							// 把象素(xx，yy)设置成逻辑（）
							// 同时也表明该象素处理过
							pUnRegion[yy * inimg->width + xx] = 1;
                            outimg->imgData[yy * outimg->width + xx] = seed;
						}
					}
					nStart++;
				}
			}
		}
	// 释放内存
	delete []pnGrowQueX;
	delete []pnGrowQueY;
	delete []pUnRegion;
	pnGrowQueX = NULL ;
	pnGrowQueY = NULL ;
	return NO_ERROR;
}
