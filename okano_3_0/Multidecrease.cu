// Multidecrease.cu
// 实现图像的多阈值N值化图像生成操作

#include "Multidecrease.h"
#include "Histogram.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ErrorCode.h"

// 宏：DEF_BLOCK_X ºÍ DEF_BLOCK_Y
// 定义了默认的线程块尺寸
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// Kernel 函数： _multidecrease_frontKer(前向 N 值化)
// 根据给定的阈值集合对图像进行 N 值化处理。判断当前像素点灰度值所在的阈值区间，
// 并将处于某个阈值区间的像素点设定为该阈值区间的前向阈值（即阈值区间的左端点）。
static __global__ void      // Kernel 函数无返回值
_multidecrease_frontKer(
        ImageCuda inimg,            // 输入图像
        ImageCuda outimg,           // 输出图像
        unsigned char *thresholds,  // 阈值集合
        int thresnum                // 阈值个数
);

// Kernel 函数： _multidecrease_backKer(后向 N 值化)
// 根据给定的阈值集合对图像进行 N 值化处理。判断当前像素点灰度值所在的阈值区间，
// 并将处于某个阈值区间的像素点设定为该阈值区间的后向阈值（即阈值区间的右端点）。
static __global__ void      // Kernel 函数无返回值
_multidecrease_backKer(
        ImageCuda inimg,            // 输入图像
        ImageCuda outimg,           // 输出图像
        unsigned char *thresholds,  // 阈值集合
        int thresnum                // 阈值个数
);

// Kernel 函数: _multidecrease_frontKer（前向 N 值化）
static __global__ void _multidecrease_frontKer(ImageCuda inimg, ImageCuda outimg,
                                    unsigned char *thresholds, int thresnum)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;
    
    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp;
    intemp = inimg.imgMeta.imgData[inidx];
	
    // 一个线程处理四个像素。
	// 判断当前像素点的灰度值处于哪个阈值区间，并将该点的像素值设为阈值区间的
	// 前向阈值。线程中处理的第一个点。
    for (int i = 1; i < thresnum; i++) {
        if (intemp == 255) {
            outimg.imgMeta.imgData[outidx] = thresholds[thresnum - 2];
            break;
        }
        if (intemp >= thresholds[i-1] && intemp < thresholds[i]) {
            outimg.imgMeta.imgData[outidx] = thresholds[i-1];
            break;
        }
    }
	
    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += outimg.pitchBytes;
        intemp = inimg.imgMeta.imgData[inidx];

        // 判断当前像素点的灰度值处于哪个阈值区间，并将该点的像素值设为阈值区间的
	    // 前向阈值。
        for (int j = 1; j < thresnum; j++) {
            if (intemp == 255) {
                outimg.imgMeta.imgData[outidx] = thresholds[thresnum - 2];
                break;
            }
            if (intemp >= thresholds[j-1] && intemp < thresholds[j]) {
                outimg.imgMeta.imgData[outidx] = thresholds[j-1];
                break;
            }
        }
    }
}


// Kernel 函数: _multidecrease_backKer（后向 N 值化）
static __global__ void _multidecrease_backKer(ImageCuda inimg, ImageCuda outimg,
                                    unsigned char *thresholds, int thresnum)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算第一个输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算第一个输出坐标点对应的图像数据数组下标。
    int outidx = r * outimg.pitchBytes + c;
    // 读取第一个输入坐标点对应的像素值。
    unsigned char intemp;
    intemp = inimg.imgMeta.imgData[inidx];

    // 一个线程处理四个像素。
	// 判断当前像素点的灰度值处于哪个阈值区间，并将该点的像素值设为阈值区间的
	// 后向阈值。线程中处理的第一个点。线程中处理的第一个点。
    for (int i = 1; i < thresnum; i++) {
        if (intemp == 0) {
            outimg.imgMeta.imgData[outidx] = thresholds[1];
            break;
        }
        if (intemp > thresholds[i-1] && intemp <= thresholds[i]) {
            outimg.imgMeta.imgData[outidx] = thresholds[i];
            break;
        }
    }

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        inidx += inimg.pitchBytes;
        outidx += outimg.pitchBytes;
        intemp = inimg.imgMeta.imgData[inidx];

        // 判断当前像素点的灰度值处于哪个阈值区间，并将该点的像素值设为阈值区间的
	    // 后向阈值。
        for (int j = 1; j < thresnum; j++) {
            if (intemp == 0) {
                outimg.imgMeta.imgData[outidx] = thresholds[1];
                break;
            }
            if (intemp > thresholds[j-1] && intemp <= thresholds[j]) {
                outimg.imgMeta.imgData[outidx] = thresholds[j];
                break;
            }
        }
    }
}

// 存储阈值的结构 ThresNode
class ThresNode {

public:
    int thres;              // 当前阈值节点的阈值
    int rangeA;             // 当前阈值节点的前向范围
    int rangeB;             // 当前阈值节点的后向范围
    ThresNode *leftchild;   // 当前阈值节点的左子节点
    ThresNode *rightchild;  // 当前阈值节点的右子节点
    ThresNode(){};
};

// 给定直方图信息及搜索范围，在此范围内搜索双峰之间的谷值
__host__ int gettrough(int rangea, int rangeb, unsigned int *his,
                       unsigned int widthrange, unsigned int pixelrange) {
    // 判断区间的值是否正确，当范围出错时返回 -1
    if(rangea < 0 || rangeb < 0 || rangea > 255 || rangeb > 255
            || rangea >= rangeb ||abs(rangea - rangeb) < widthrange) {
        return -1;
    }
    int minnum = 0;  // 范围内像素点数目最少的灰度值
    for(int i = rangea; i <= rangeb; i++){
        if (his[i] < his[minnum])
            minnum = i;
    }
    int firstpeak = minnum;   // 范围内的第一峰值
    int secondpeak = minnum;  // 范围内的第二峰值
    int secondfront = 1;      // 判断坐标轴上的位置，第二峰值是否在第一峰值的前方

	// 搜索第一峰值
    for (int i = rangea; i <= rangeb; i++) {
        if (his[i] > his[firstpeak]) {
            firstpeak = i;
        }
    }
    int trough = firstpeak;  // 双峰之间的谷值
	
    // 分别将区间的左右顶点值与第一峰值做差，与 widthrange 做比较，判断能否求得
	// 第二峰值。若大于 widthrange，则可以求得；若小于，则不可求得。
	if ((firstpeak - rangea) >= widthrange) {
        for (int i = rangea; i < (firstpeak - widthrange); i++) {
            if (his[i] > his[secondpeak] && his[i] < his[firstpeak])
                secondpeak = i;
        }
	}
	else {
	    // 不在范围内，则第二峰值不可能在第一峰值的前方。
	    secondfront = 0;
	}
	if (rangeb - firstpeak >= widthrange) {
        for (int i = (firstpeak + widthrange); i < rangeb; i++) {
            if (his[i] > his[secondpeak] && his[i] < his[firstpeak]) {
                secondpeak = i;
                // 此时代表第二峰值在第一峰值的后方
                secondfront = 0;
			}
        }
	}
	else if (secondfront == 0) {
	    // 第一峰值的前后均不可求得第二峰值，故无法求得谷值，故退出计算。
	    return -1;
	}
	// 第一峰值在前，第二峰值在后
    if (secondfront == 0) {
        for (int i = firstpeak + 1; i < secondpeak; i++){
            if (his[i] < his[trough])
                trough = i;
        }
    }
	// 第一峰值在后，第二峰值在前
    else {
        for (int i = secondpeak + 1; i < firstpeak; i++){
            if (his[i] < his[trough])
                trough = i;
        }
	}
    
	// 若第二峰值与谷值的像素值的差比设定的范围小，则说明该谷值无效，返回 -1
    if ((his[secondpeak] - his[trough]) < pixelrange ) {
        return -1;
    }
    else 
        return trough;
};

// 根据直方图信息，创建一棵存储阈值信息的二叉树
void createtree(ThresNode *tree, unsigned int * his,
                unsigned int widthrange, unsigned int pixelrange) {
    // 获取当前节点的阈值

    int thres = gettrough(tree->rangeA, tree->rangeB, his, widthrange, pixelrange);
	// 判断阈值节点是否合法
    if (thres == -1) {
        tree->thres = -1;
    }
    else{
        tree->thres = thres;
        // 为阈值节点创建其左子节点，并设定左子节点的搜索范围 
        ThresNode * leftc = new ThresNode();
        tree->leftchild = leftc;
        leftc->rangeA = tree->rangeA;
        leftc->rangeB = tree->thres;
        createtree(tree->leftchild, his, widthrange, pixelrange);
		
        // 为阈值节点创建其右子节点，并设定其右子节点的搜索范围
        ThresNode *rightc = new ThresNode();
        tree->rightchild = rightc;
        rightc->rangeA = tree->thres + 1;
        rightc->rangeB = tree->rangeB;
        createtree(tree->rightchild, his, widthrange, pixelrange);
    }
};

// 获取所有阈值
void  searchtree(ThresNode *tree, unsigned char *thresholds, int &thresnum) {
    if (tree -> thres != -1) {
	        // 通过对二叉树进行中序遍历，按照从小到大的顺序存储所有阈值
            searchtree(tree->leftchild, thresholds, thresnum);
            thresholds[thresnum++] = (unsigned char)tree->thres;
            searchtree(tree->rightchild, thresholds, thresnum);
    }
};

// Host 成员方法：multidecrease（多阈值N值化处理）
__host__ int Multidecrease::multidecrease(Image *inimg, Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL)
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
	
	// 获取当前图像的直方图
    Histogram h;
    unsigned int his[256];
    h.histogram(inimg, his, true);

	// 存储当前图像的所有粗分割阈值
    unsigned char *devthreshlods;
    unsigned char *hostthresholds = new unsigned char [20];
    // 存储当前图像的所有粗分割阈值个数。
	// 为了计算方便将第一个阈值设为 0 。
    int thresnum = 1;
    hostthresholds[0] = 0;
    
	// 创建阈值的根节点信息，并设定其搜索范围
    ThresNode * tn = new ThresNode();
    tn->rangeA = 0;
    tn->rangeB = 255;
	// 根据当前图像直方图信息建立阈值二叉树
    createtree(tn, his, this->getwidthrange(), this->getpixelrange());
	// 对二叉树进行搜索和存储
    searchtree(tn, hostthresholds, thresnum);

    // 图像总像素数
    int sumpixel = 0;
    for (int i = 0; i < 256; i++) {
        sumpixel += his[i];
    }

	// 数组 pix_w[i] 记录图像中值为i的像素占总像素数的比例
    float pix_w[256];    
    // 计算 pix_w[i] 的值
    for (int i = 0; i < 256; i++) {
        pix_w[i] = ((float)his[i])/sumpixel;
    }

    // 使用 OTSU 方法对各个粗分割阈值进行搜索，找到使类内方差最小的最佳阈值。
    for (int countthres = 0; countthres < thresnum - 1; countthres++){
        float min = 100000.0;  // 类内方差的最小值
        int threshold = 0;     // 最佳阈值
        float Wk = 0.0;        // 第 k 个类的概率的累加和
        float Ukp = 0.0;       // 第 k 个类的各个像素值与概率乘积的累加和
        float Uk = 0.0;        // 第 k 个类的均值
        float Qk = 0.0;        // 第 k 个类的方差

		// 在每个粗分割阈值的松弛余量范围内，进行 OTSU 法搜索，直到找到最
		// 小类内方差，此时对应的即是最佳阈值。
        for (int j = -5; j < 6; j++) {
            for (int i = hostthresholds[countthres] + 1;
       			i <= (hostthresholds[countthres + 1] + j); i++){
				
                // 计算类的概率的累加和
                Wk+=pix_w[i];
				
                // 计算像素值与其概率乘积的累加和
                Ukp+=i * pix_w[i];
            }
			
			// 计算类的均值
            Uk = Ukp/Wk;
			
            for (int i = hostthresholds[countthres] + 1;
     			i <= (hostthresholds[countthres + 1] + j); i++){
				// 再次搜索，计算类的方差
                Qk = (i - Uk)*(i - Uk) * pix_w[i];
            }

			// 判断当前方差是否小于 min，若是则覆盖最小值，并存储当前阈值。
            if (min > Qk) {
                min = Qk;
                threshold = hostthresholds[countthres + 1] + j;
            }
        }
        // 更新阈值集合的值为求得的最佳阈值。
        hostthresholds[countthres + 1] = threshold;
    }

    // 返回最佳阈值集合至 threshold，并跳过第一个阈值（0）。
    threshold = new unsigned char[thresnum -1];
    for (int i = 1; i < thresnum; i++) {
        this->threshold[i -1] = hostthresholds[i];
    }

    // 将阈值集合的最后一个阈值设定为255。
    hostthresholds[thresnum] = 255;

    // 为标记数组分配大小。
    errcode = cudaMalloc((void **)&devthreshlods, (thresnum + 1) * sizeof (unsigned char));
    if (errcode != cudaSuccess) {
        cudaFree(devthreshlods);
        return errcode;
    }
	
	// 为标记数组设定初值。
    errcode = cudaMemset(devthreshlods, 0, (thresnum + 1) * sizeof (unsigned char));
    if (errcode != cudaSuccess) {
        cudaFree(devthreshlods);
        return errcode;
    }

	// 将数组复制至 device 端。
    errcode = cudaMemcpy(devthreshlods, hostthresholds, (thresnum + 1) * sizeof (unsigned char),
                             cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess) {
        cudaFree(devthreshlods);
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);
	
	// 判断当前的 stateflag 是前向标记还是后向标记，并根据其值调用对应的 Kernel 函数。
    if (this->stateflag == MD_FRONT) {
        _multidecrease_frontKer<<<gridsize, blocksize>>>(
           insubimgCud, outsubimgCud, devthreshlods, (thresnum + 1));
    }
    else {
        _multidecrease_backKer<<<gridsize, blocksize>>>(
           insubimgCud, outsubimgCud, devthreshlods, (thresnum + 1));
    }

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
			
    // 处理完毕，退出。	
    return NO_ERROR;
}

