// ICcircleRadii.cu
// 实现对一个给定轮廓的坐标集，求其所包围领域内的一点的最远距离最小的点（外接圆
// 的半径）和最近距离最大的点（内接圆的半径）

#include "ICcircleRadii.h"

#include "Image.h"
#include "ErrorCode.h"

// 宏 IC_BLOCK_X 和 IC_BLOCK_Y
#define IC_BLOCK_X       32
#define IC_BLOCK_Y        8
#define IC_BLOCK_Z        8

// 宏 IC_MAX_INT 和 IC_MIN_INT
// 定义距离的最大值和最小值
#define IC_MAX_INT    65536
#define IC_MIN_INT       -1


// Kernel 函数：_getContourKer（获得轮廓上每个点的坐标）
// 根据给定的轮廓图像，将其上轮廓的每一个点的坐标输出
static __global__ void    // 无返回值
_getContourKer(
        ImageCuda inimg,  // 输入坐标集
        int *lenth,       // 轮廓上点的个数
        int *contourX,    // 轮廓上每个点的 X 坐标
        int *contourY     // 轮廓上每个点的 Y 坐标
);

// Kernel 函数：_minMaxKer（最远距离最小的点）
// 根据给定轮廓的坐标集，其所包围领域内的一点，以及需要的点的数量，返回最远距离
// 最小的点的集合
static __global__ void    // 无返回值
_minMaxKer(
        ImageCuda inimg,  // 输入坐标集
        int *minMaxDist,  // 输出点的集合
        int lenth,        // 轮廓上点的个数
        int *contourX,    // 轮廓上点的 X 坐标
        int *contourY,    // 轮廓上点的 Y 坐标
		int *indexX,      // 结果点集的 X 坐标
		int *indexY,      // 结果点集的 Y 坐标
		int *lensec       // 结果点集的数量
);


// Kernel 函数：_maxMinKer（最近距离最大的点）
// 根据给定轮廓的坐标集，其所包围领域内的一点，以及需要的点的数量，返回最近距离
// 最大的点的集合
static __global__ void    // 无返回值
_maxMinKer(
        ImageCuda inimg,  // 输入坐标集
        int *minMaxDist,  // 输出点的集合
        int lenth,        // 轮廓上点的数量
        int *contourX,    // 轮廓上点的 X 坐标
        int *contourY,    // 轮廓上点的 Y 坐标
		int *indexX,      // 结果点集的 X 坐标
		int *indexY,      // 结果点集的 Y 坐标
		int *lensec       // 结果点集的数量
);

// Kernel 函数: _shearSortRowDesKer（行降序排序）
// 对待排序矩阵的每一行进行双调排序。
static __global__ void 
_shearSortDesKer(
        int distDev[],      // 得票数。
        int indexXDev[],    // 索引值。
		int indexYDev[],
        int lensec,         // 矩阵行数。
        int judge           // 块内共享内存的大小。
);

// Kernel 函数: _shearSortRowDesKer（行升序排序）
// 对待排序矩阵的每一行进行双调排序。
static __global__ void 
_shearSortAscKer(
        int distDev[],   // 得票数。
        int indexXDev[],    // 索引值。
		int indexYDev[],
        int lensec,         // 矩阵行数。
        int judge           // 块内共享内存的大小。
);

// Kernel 函数：_getContourKer（获得轮廓上每个点的坐标）
static __global__ void _getContourKer(ImageCuda inimg, int *lenth,
        int *contourX, int *contourY)
{
    // 获取线程索引，图像索引采用线程索引
    int tidc = blockIdx.x * blockDim.x + threadIdx.x;
    int tidr = blockIdx.y * blockDim.y + threadIdx.y;

    // 转化为图像下标
    int id = tidr * inimg.pitchBytes + tidc;

    // 判断是否越界
    if (tidc >= inimg.imgMeta.width || tidr >= inimg.imgMeta.height)
        return;

    // 判断是否为轮廓上的点，如果在轮廓上，将其坐标记录
    if (inimg.imgMeta.imgData[id] == 0) {
        contourX[*lenth] = tidc;
        contourY[*lenth] = tidr;
        *lenth = *lenth + 1;
    }
}

// Kernel 函数：_minMaxKer（最远距离最小的点）
static __global__ void _minMaxKer(
        ImageCuda inimg, int *minMaxDist, int lenth, int *contourX,
        int *contourY, int *indexX, int *indexY, int *lensec)
{
    // 获取线程索引
    int tidc = blockIdx.x * blockDim.x + threadIdx.x;
    int tidr = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // 转化为图像下标
    int id = tidr * inimg.pitchBytes + tidc;

    // 判断是否越界
    if (tidc >= inimg.imgMeta.width || tidr >= inimg.imgMeta.height)
        return;
    if (z >= lenth) return;

    // 记录两点之间的坐标差值
    int dx = 0;
    int dy = 0;
    int dist = 0;

    // 是否在轮廓上
    if (inimg.imgMeta.imgData[id] == 0) {
        dx = contourX[z] - tidc;
        dy = contourY[z] - tidr;
        dist = dx * dx + dy * dy;

        // 统计轮廓内部坐标到某一点的最小距离
        if (dist < minMaxDist[z]) {
            minMaxDist[z] = dist;
			indexX[z] = tidc;
			indexY[z] = tidr;
			if (z > *lensec)
			    *lensec = z;
        }
    }
}

// Kernel 函数：_maxMinKer（最近距离最大的点）
static __global__ void _maxMinKer(
        ImageCuda inimg, int *maxMinDist, int lenth, int *contourX,
        int *contourY, int *indexX, int *indexY, int *lensec)
{
    // 获取线程索引
    int tidc = blockIdx.x * blockDim.x + threadIdx.x;
    int tidr = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // 转化为图像下标
    int id = tidr * inimg.pitchBytes + tidc;

    // 判断是否越界
    if (tidc >= inimg.imgMeta.width || tidr >= inimg.imgMeta.height)
        return;
    if (z >= lenth) return;

    // 记录两点之间的坐标差值
    int dx = 0;
    int dy = 0;
    int dist = 0;

    // 是否在轮廓上
    if (inimg.imgMeta.imgData[id] == 0) {
        dx = contourX[z] - tidc;
        dy = contourY[z] - tidr;
        dist = dx * dx + dy * dy;

        // 统计轮廓内部坐标到某一点的最大距离
        if (dist > maxMinDist[z]) {
            maxMinDist[z] = dist;
			indexX[z] = tidc;
			indexY[z] = tidr;
			if (z > *lensec)
			    *lensec = z;
        }
    }
}

// Kernel 函数: _shearSortRowDesKer（行降序排序）
static __global__ void _shearSortDesKer(
        int distDev[], int indexXDev[], int indexYDev[], int lensec, int judge)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    extern __shared__ int shared[];
    // 通过偏移，获得存放得票数和索引值的两部分共享内存空间。
    int *vote, *indexX, *indexY;
    vote = shared;
    indexX = shared + judge;
    indexY = shared + judge * 2;
    
    // 为共享内存赋初始值。
    if (cid < lensec) {
        vote[cid] = distDev[rid * lensec + cid];
        indexX[cid] = indexXDev[rid * lensec + cid];
        indexY[cid] = indexYDev[rid * lensec + cid];
    }
    // 块内同步。
    __syncthreads();

    // 声明临时变量
    int ixj, tempvote, tempindex;
    // 偶数行降序排序。
    if (rid % 2 == 0) {
        for (int k = 2; k <= lensec; k <<= 1) {
             // 双调合并。
            for (int j = k >> 1; j > 0; j >>= 1) {
                // ixj 是与当前位置 cid 进行比较交换的位置。
                ixj = cid ^ j;
                if (ixj > cid) {
                    // 如果 (cid & k) == 0，按照降序交换两项。
                    if ((cid & k) == 0 && (vote[cid] < vote[ixj])) {
                        // 交换得票数。                        
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = indexX[cid];
                        indexX[cid] = indexX[ixj];
                        indexX[ixj] = tempindex; 
                        tempindex = indexY[cid];
                        indexY[cid] = indexY[ixj];
                        indexY[ixj] = tempindex;
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    } else if ((cid & k) != 0 && vote[cid] > vote[ixj]) {
                        // 交换得票数。                     
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = indexX[cid];
                        indexX[cid] = indexX[ixj];
                        indexX[ixj] = tempindex; 
                        tempindex = indexY[cid];
                        indexY[cid] = indexY[ixj];
                        indexY[ixj] = tempindex;
                    }
                }
                __syncthreads();
            }
        }
    // 奇数行升序排序。
    } else {
        for (int k = 2; k <= lensec; k <<= 1) {
            // 双调合并。
            for (int j = k >> 1; j > 0; j >>= 1) {
                // ixj 是与当前位置 cid 进行比较交换的位置。
                ixj = cid ^ j;
                if (ixj > cid) {
                    // 如果 (cid & k) == 0，按照降序交换两项。
                    if ((cid & k) == 0 && (vote[cid] > vote[ixj])) {
                        // 交换得票数。                        
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = indexX[cid];
                        indexX[cid] = indexX[ixj];
                        indexX[ixj] = tempindex;
                        tempindex = indexY[cid];
                        indexY[cid] = indexY[ixj];
                        indexY[ixj] = tempindex;						
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    } else if ((cid & k) != 0 && vote[cid] < vote[ixj]) {
                        // 交换得票数。 
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = indexX[cid];
                        indexX[cid] = indexX[ixj];
                        indexX[ixj] = tempindex;
                        tempindex = indexY[cid];
                        indexY[cid] = indexY[ixj];
                        indexY[ixj] = tempindex;						
                    }
                }   
                __syncthreads();
            }
        }    
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    if (cid <lensec) {
        distDev[rid * lensec + cid] = vote[cid];
        indexXDev[rid * lensec + cid] = indexX[cid];
		indexYDev[rid * lensec + cid] = indexY[cid];
    }
}

// Kernel 函数: _shearSortRowAscKer（行升序排序）
static __global__ void _shearSortAscKer(
        int distDev[], int indexXDev[], int indexYDev[], int lensec, int judge)
{
    // 读取线程号和块号。
    int cid = threadIdx.x;
    int rid = blockIdx.x;

    extern __shared__ int shared[];
    // 通过偏移，获得存放得票数和索引值的两部分共享内存空间。
    int *vote, *indexX, *indexY;
    vote = shared;
    indexX = shared + judge;
	indexY = shared + judge * 2;
    
    // 为共享内存赋初始值。
    if (cid < lensec) {
        vote[cid] = distDev[rid * lensec + cid];
        indexX[cid] = indexXDev[rid * lensec + cid];
		indexY[cid] = indexYDev[rid * lensec + cid];
    }
    // 块内同步。
    __syncthreads();

    // 声明临时变量
    int ixj, tempvote, tempindex;
    // 偶数行降序排序。
    if (rid % 2 == 0) {
        for (int k = 2; k <= lensec; k <<= 1) {
             // 双调合并。
            for (int j = k >> 1; j > 0; j >>= 1) {
                // ixj 是与当前位置 cid 进行比较交换的位置。
                ixj = cid ^ j;
                if (ixj > cid) {
                    // 如果 (cid & k) == 0，按照降序交换两项。
                    if ((cid & k) == 0 && (vote[cid] > vote[ixj])) {
                        // 交换得票数。                        
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = indexX[cid];
                        indexX[cid] = indexX[ixj];
                        indexX[ixj] = tempindex; 
						tempindex = indexY[cid];
						indexY[cid] = indexY[ixj];
						indexY[ixj] = tempindex;
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    } else if ((cid & k) != 0 && vote[cid] < vote[ixj]) {
                        // 交换得票数。                     
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = indexX[cid];
                        indexX[cid] = indexX[ixj];
                        indexX[ixj] = tempindex; 
						tempindex = indexY[cid];
						indexY[cid] = indexY[ixj];
						indexY[ixj] = tempindex;
                    }
                }
                __syncthreads();
            }
        }
    // 奇数行升序排序。
    } else {
        for (int k = 2; k <= lensec; k <<= 1) {
            // 双调合并。
            for (int j = k >> 1; j > 0; j >>= 1) {
                // ixj 是与当前位置 cid 进行比较交换的位置。
                ixj = cid ^ j;
                if (ixj > cid) {
                    // 如果 (cid & k) == 0，按照降序交换两项。
                    if ((cid & k) == 0 && (vote[cid] < vote[ixj])) {
                        // 交换得票数。                        
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = indexX[cid];
                        indexX[cid] = indexX[ixj];
                        indexX[ixj] = tempindex;
                        tempindex = indexY[cid];
                        indexY[cid] = indexY[ixj];
                        indexY[ixj] = tempindex;						
                    // 如果 (cid & k) == 0，按照升序交换两项。
                    } else if ((cid & k) != 0 && vote[cid] > vote[ixj]) {
                        // 交换得票数。 
                        tempvote = vote[cid];
                        vote[cid] = vote[ixj];
                        vote[ixj] = tempvote;
                        // 交换索引值。
                        tempindex = indexX[cid];
                        indexX[cid] = indexX[ixj];
                        indexX[ixj] = tempindex;
                        tempindex = indexY[cid];
                        indexY[cid] = indexY[ixj];
                        indexY[ixj] = tempindex;						
                    }
                }   
                __syncthreads();
            }
        }    
    }
    // 将共享内存中的排序后的数组拷贝到全局内存中。
    if (cid <lensec) {
        distDev[rid * lensec + cid] = vote[cid];
        indexXDev[rid * lensec + cid] = indexX[cid];
		indexYDev[rid * lensec + cid] = indexY[cid];
    }
}

// Host 成员方法：minMax（最远距离最小的点）
__host__ int ICcircleRadii::minMax(Image *inimg, int picNum, int *minMaxDist,
        int *minMaxIndexX, int *minMaxIndexY)
{
    // 检查输入图像是否为空
    if (inimg == NULL )
        return NULL_POINTER;

    // 检查图像是否为空
    if (inimg->imgData == NULL)
        return UNMATCH_IMG;	
	
    // 将输入图像复制到 Device
    int errcode;
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
		
	// 提取输入图像的 ROI 子图
	ImageCuda inSubimgCud;
	errcode = ImageBasicOp::roiSubImage(inimg, &inSubimgCud);
	if (errcode != NO_ERROR)
	    return errcode;

	// 定义点集数量
	int jugde = inSubimgCud.imgMeta.height * inSubimgCud.imgMeta.width;
	
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 gridsize, blocksize;
    blocksize.x = IC_BLOCK_X;
    blocksize.y = IC_BLOCK_Y;
    blocksize.z = IC_BLOCK_Z;
    gridsize.x = (inSubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (inSubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;
    gridsize.z = (jugde + blocksize.z - 1) / blocksize.z;
	
	// host 端数据，用于初始化
	int *minMax = NULL;
	int lenth = -1;
	int lensec = -1;
    
	// host 端数组分配空间以及初始化
	minMax = (int*)malloc(jugde * sizeof(int));
	for (int i = 0; i < jugde; i++)
        minMax[i] = IC_MAX_INT;
	
    // 设备端数组，用于存储轮廓上的点和轮廓上点的数量
    int *dev_contourX = NULL;
    int *dev_contourY = NULL;
    int *dev_lenth = NULL;
	
	// 设备端数组，用于存储结果点的坐标以及数量
	int *dev_indexX = NULL;
	int *dev_indexY = NULL;
	int *dev_lensec = NULL;
	int *dev_minMax = NULL;

    // 在 GPU 上分配内存，在 GPU 上申请一块连续的内存，通过指针将内存分配给
    // contourX，contourY，lenth，minMaxDist，indexX，intdexY，lensec，minMax
    cudaMalloc((void**)&dev_contourX, (jugde * 5 + 3) * sizeof(int));
    dev_contourY = dev_contourX + jugde;
    dev_lenth = dev_contourY + jugde;
	dev_indexX = dev_lenth + 1;
	dev_indexY = dev_indexX + jugde;
	dev_lensec = dev_indexY + jugde;
	dev_minMax = dev_lensec + 1;
	
		
	// 将数据拷贝到设备端
    cudaMemcpy(dev_minMax, minMaxDist, jugde * sizeof(int),
            cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lenth, &lenth, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lensec, &lensec, sizeof(int), cudaMemcpyHostToDevice);

	
    // 运行 Kernel 核函数 _getContourKer 求出轮廓上点的坐标
    _getContourKer<<<gridsize, blocksize>>>(inSubimgCud, dev_lenth, dev_contourX,
	        dev_contourY);

    // 运行 Kernel 核函数 _minMaxKer 求出距离最远的点
    _minMaxKer<<<gridsize, blocksize>>>(inSubimgCud, dev_minMax, *dev_lenth,
	        dev_contourX, dev_contourY, dev_indexX, dev_indexY, dev_lensec);

    // 运行 Kernel 核函数 _shearSortAecKer 对距离最远的点升序排序
    _shearSortAscKer<<<1, jugde, 2*jugde*sizeof(int)>>>(dev_minMax, dev_indexX,
	        dev_indexY, *dev_lensec, jugde);
			
	// 将结果拷贝至输出数组
	cudaMemcpy(minMaxDist, dev_minMax, picNum * sizeof(int),
	        cudaMemcpyDeviceToHost);
			
	// 调用 cudaGetLastError 判断程序是否出错
	cudaError_t cuerrcode;
	cuerrcode = cudaGetLastError();
	if (cuerrcode != cudaSuccess)
	    return CUDA_ERROR;
			
	// 处理完毕，退出
	return NO_ERROR;
}


// Host 成员方法：maxMin（最近距离最大的点）
__host__ int ICcircleRadii::maxMin(Image *inimg, int picNum, int *maxMinDist,
        int *maxMinIndexX, int *maxMinIndexY)
{
    // 检查输入图像是否为空
    if (inimg == NULL )
        return NULL_POINTER;

    // 检查图像是否为空
    if (inimg->imgData == NULL)
        return UNMATCH_IMG;
	
    // 将输入图像复制到 Device
    int errcode;
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图
	ImageCuda inSubimgCud;
	errcode = ImageBasicOp::roiSubImage(inimg, &inSubimgCud);
	if (errcode != NO_ERROR)
	    return errcode;

	// 定义点集数量
	int jugde = inSubimgCud.imgMeta.height * inSubimgCud.imgMeta.width;
	
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 gridsize, blocksize;
    blocksize.x = IC_BLOCK_X;
    blocksize.y = IC_BLOCK_Y;
    blocksize.z = IC_BLOCK_Z;
    gridsize.x = (inSubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (inSubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;
    gridsize.z = (jugde + blocksize.z - 1) / blocksize.z;
	
	// host 端数据，用于初始化
	int *maxMin = NULL;
	int lenth = -1;
	int lensec = -1;
    
	// host 端数组分配空间以及初始化
	maxMin = (int*)malloc(jugde * sizeof(int));
	
    // 设备端数组，用于存储轮廓上的点和轮廓上点的数量
    int *dev_contourX = NULL;
    int *dev_contourY = NULL;
    int *dev_lenth = NULL;
	
	// 设备端数组，用于存储结果点的坐标以及数量
	int *dev_indexX = NULL;
	int *dev_indexY = NULL;
	int *dev_lensec = NULL;
	int *dev_maxMin = NULL;

    // 在 GPU 上分配内存，在 GPU 上申请一块连续的内存，通过指针将内存分配给
    // contourX，contourY，lenth，minMaxDist，indexX，intdexY，lensec，minMax
    cudaMalloc((void**)&dev_contourX, (jugde * 5 + 3) * sizeof(int));
    dev_contourY = dev_contourX + jugde;
    dev_lenth = dev_contourY + jugde;
	dev_indexX = dev_lenth + 1;
	dev_indexY = dev_indexX + jugde;
	dev_lensec = dev_indexY + jugde;
	dev_maxMin = dev_lensec + 1;
	
	// 初始化
    for (int i = 0; i < jugde; i++)
        maxMin[i] = IC_MIN_INT;
		
	// 将数据拷贝到设备端
    cudaMemcpy(dev_maxMin, maxMinDist, jugde * sizeof(int),
            cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lenth, &lenth, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lensec, &lensec, sizeof(int), cudaMemcpyHostToDevice);

	
    // 运行 Kernel 核函数 _getContourKer 求出轮廓上点的坐标
    _getContourKer<<<gridsize, blocksize>>>(inSubimgCud, dev_lenth, dev_contourX,
	        dev_contourY);

    // 运行 Kernel 核函数 _minMaxKer 求出距离最远的点
    _maxMinKer<<<gridsize, blocksize>>>(inSubimgCud, dev_maxMin, *dev_lenth,
	        dev_contourX, dev_contourY, dev_indexX, dev_indexY, dev_lensec);

    // 运行 Kernel 核函数 _shearSortDesKer 对距离最远的点降序排序
    _shearSortDesKer<<<1, jugde, 2*jugde*sizeof(int)>>>(dev_maxMin, dev_indexX,
	        dev_indexY, *dev_lensec, jugde);
			
	// 将结果拷贝至输出数组
	cudaMemcpy(maxMinDist, dev_maxMin, picNum * sizeof(int),
	        cudaMemcpyDeviceToHost);
			
	// 调用 cudaGetLastError 判断程序是否出错
	cudaError_t cuerrcode;
	cuerrcode = cudaGetLastError();
	if (cuerrcode != cudaSuccess)
	    return CUDA_ERROR;
	
	// 处理完毕，退出
	return NO_ERROR;
}
