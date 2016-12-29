// DiffPattern.cu
// 图像局部特异检出

#include "DiffPattern.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  8
#define DEF_BLOCK_Y  8

// 结构体：PatternData（ 每个 pattern 的详细数据）
// 描述每个 pattern 的数据集合，包括走向角、中心坐标、和最小外接有向矩形的长边
// 及短边等
typedef struct PatternData_st {
    float angel;  // 走向角
    float ss;     // 短边 
    float ls;     // 长边
    float csX;    // 中心点横坐标
    float csY;    // 中心点纵坐标
} PatternData;


// 结构体：PatternDesc（ 每个 pattern 中区域的分布信息）
// 描述每个 pattern 中红色区域和紫色区域的分布信息，由于 19 个 pattern 的大小
// 均在 8 x 8 的方形区域内，因此使用位图记录，r 表示红色区域，p 表示紫色区域
// r 和 p 都是单字节，字节中二进制位为 1 的区域代表被使用，0 表示未使用。
// 包围每个 pattern 的矩形的左上角顶点与 8 x 8 方形区域左上顶点对齐，坐标方向
// 与 pattern 方向一致
typedef struct PatternDesc_st {
    unsigned char r[8];  // 红色区域，每个字节表示一行的 8 个像素点位置
    unsigned char p[8];  // 紫色区域，每个字节表示一行的 8 个像素点位置
    
    int pCount;          // 紫色区块数目
    int rCount;          // 红色区块数目
    int xinCord;         // 区域中心点相对横坐标
    int yinCord;         // 区域中心点相对纵坐标
} PatternDesc;

// 各个 pattern 的数据 
PatternData _patData[19] = { { 0,          1,      1,   0,    0 },  // pattern1
                             { 0.25f, 1.414f, 4.242f,   0,    0 },  // pattern2
                             { 0.75f, 1.414f, 4.242f,   0,    0 },  // pattern3
                             { 0,          1,      3,   0,    0 },  // pattern4
                             { 0.5f,       1,      3,   0,    0 },  // pattern5
                             { 0.25f, 1.414f, 2.828f,  0.5, 0.5 },  // pattern6
                             { 0.75f, 1.414f, 2.828f, -0.5, 0.5 },  // pattern7
                             { 0,          1,      2,  0.5,   0 },  // pattern8
                             { 0.5,        1,      2,    0, 0.5 },  // pattern9
                             { 0,          2,      2,  0.5, 0.5 },  // pattern10
                             { 0,          2,      2,  0.5, 0.5 },  // pattern11
                             { 0,          2,      3,    0, 0.5 },  // pattern12
                             {0.5,         2,      3,  0.5,   0 },  // pattern13
                             {0.25,        2,      5,    0, 0.5 },  // pattern14
                             {0.75,        2,      5,    0, 0.5 },  // pattern15
                             {0,           3,      3,    0,   0 },  // pattern16
                             {0,           3,      3,    0,   0 },  // pattern17
                             {0.25,    4.242,  4.242,    0,   0 },  // pattern18
                             {0.75,    4.242,  4.242,    0,    0}   // pattern19
                           };

// 每个 pattern 的区域分布数据
static __device__ PatternDesc _pd[19] =  {
    { // [0] 
        // r[8]
        { 0x0E, 0x1F, 0x1B, 0x1F, 0x0E, 0x00, 0x00, 0x00 }, 
        // p[8] 
        { 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount       
        1, 20,
        // xinCord, yinCord                                           
        2, 2                                             
    },
    { // [1]
        // r[8] 
        { 0x1B, 0x36, 0x6C, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x04, 0x08, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount             
        3, 12,
        // xinCord, yinCord 
        3, 1
    },
    { // [2]
        // r[8] 
        { 0x6C, 0x36, 0x1B, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x10, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount             
        3, 12,
        // xinCord, yinCord 
        3, 1
    },
    { // [3]
        // r[8] 
        { 0x07, 0x07, 0x00, 0x07, 0x07, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount             
        3, 12,
        // xinCord, yinCord 
        1, 2
    },
    { // [4]
        // r[8] 
        { 0x1B, 0x1B, 0x1B, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount             
        3, 12,
        // xinCord, yinCord 
        2, 1
    },
    { // [5]
        // r[8]
        { 0x1B, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x04, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount             
        3, 12,
        // xinCord, yinCord 
        2, 1
    },
    { // [6]
        // r[8]
        { 0x6C, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x10, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount 
        2, 8,
        // xinCord, yinCord
        3, 0
    },
    { // [7]
        // r[8]
        { 0x03, 0x03, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount 
        2, 8,
        // xinCord, yinCord
        0, 2
    },
    { // [8]
        // r[8]
        { 0x1B, 0x1B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount 
        2, 8,
        // xinCord, yinCord
        2, 0
    },
    { // [9]
        // r[8]
        { 0x03, 0x03, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00 },
        // p[8]
        { 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount 
        4, 8,
        // xinCord, yinCord
        0, 2
    },
    { // [10]
        // r[8]
        { 0x33, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x0C, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount               
        4, 8,
        // xinCord, yinCord
        2, 0
    },
    { // [11]
        // r[8]
        { 0x07, 0x07, 0x00, 0x00, 0x07, 0x07, 0x00, 0x00 },
        // p[8]
        { 0x00, 0x00, 0x07, 0x07, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount 
        6, 12,
        // xinCord, yinCord
        1, 2
    },
    { // [12]
        // r[8]
        { 0x33, 0x33, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x0C, 0x0C, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount 
        6, 12,
        // xinCord, yinCord
        2, 1
    },
    { // [13]
        // r[8]
        { 0x01, 0x03, 0x06, 0x04, 0x01, 0x03, 0x06, 0x04 },
        // p[8]
        { 0x00, 0x00, 0x01, 0x03, 0x06, 0x04, 0x00, 0x00 },
        // pCount, rCount 
        6, 12,
        // xinCord, yinCord
        1, 3
    },
    { // [14]
        // r[8]
        { 0x04, 0x06, 0x03, 0x01, 0x04, 0x06, 0x03, 0x01 },
        // p[8]
        { 0x00, 0x00, 0x04, 0x06, 0x03, 0x01, 0x00, 0x00 },
        // pCount, rCount 
        6, 12,
        // xinCord, yinCord
        1, 3
    },
    { // [15]
        // r[8]
        { 0x07, 0x07, 0x00, 0x00, 0x00, 0x07, 0x07, 0x00 },
        // p[8]
        { 0x00, 0x00, 0x07, 0x07, 0x07, 0x00, 0x00, 0x00 },
        // pCount, rCount 
        9, 12,
        // xinCord, yinCord
        1, 3
    },
    { // [16]
        // r[8]
        { 0x63, 0x63, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // p[8]
        { 0x1B, 0x1B, 0x1B, 0x00, 0x00, 0x00, 0x00, 0x00 },
        // pCount, rCount 
        9, 12,
        // xinCord, yinCord
        3, 1
    },
    { // [17]
        // r[8]
        { 0x38, 0x70, 0x60, 0x41, 0x03, 0x07, 0x0E, 0x00 },
        // p[8]
        { 0x00, 0x08, 0x1C, 0x3E, 0x1C, 0x08, 0x00, 0x00 },
        // pCount, rCount 
        13, 18,
        // xinCord, yinCord
        3, 3
    },
    { // [18]
        // r[8]
        { 0x0E, 0x07, 0x03, 0x41, 0x60, 0x70, 0x38, 0x00 },
        // p[8]
        { 0x00, 0x08, 0x1C, 0x3E, 0x1C, 0x08, 0x00, 0x00 },
        // pCount, rCount 
        13, 18,
        // xinCord, yinCord
        3, 3
    }
};

// 宏：GET_BIT
// 获取某一行数据的中指定列的 2 进制位
#define GET_BIT(row, x) row == 0 ? 0 : (row >> x) % 2 

// Kernel 函数：_diffPatternKer（根据各个 pattern 对检查是否局部特异）
// 根据 patterns 参数指定的 pattern 序号，计算对应的 pattern 是否特异，若特异则
// 修改 patterns 数组中的值进行标记
static __global__ void    // kernel 函数无返回值
_diffPatternKer(
        ImageCuda inimg,  // 输入图像
        int centerx,      // 中心点横坐标
        int centery,      // 中心点纵坐标
        int patcount,     // 差分 pattern 对的数目
        int *patterns,    // 差分 pattern 序号数组
        float *avgs       // 每个 pattern 中紫色区域像素平均值  
);


// Kernel 函数：_diffPatternKer（根据各个 pattern 对检查是否局部特异）
static __global__ void _diffPatternKer(
        ImageCuda inimg, int centerX, int centerY, int patcount, int *patterns,
        float *avgs)
{
    // 如果 z 序号超出差分 pattern 总对数则直接退出
    if (threadIdx.z >= patcount)
        return;
    // 将 pattern 对编号与数组编号对应
    int couple = patterns[threadIdx.z] - 1;
    if (couple < 0)
        return;
    
    // 申明动态共享内存
    extern __shared__ unsigned short pixels[];
    // 获取第 1 个 pattern 红色区域数据指针
    unsigned short *red1 = &pixels[256 * threadIdx.z];
    // 获取第 1 个 pattern 紫色区域数据指针
    unsigned short *pur1 = &pixels[256 * threadIdx.z + 64];
    // 获取第 2 个 pattern 红色区域数据指针
    unsigned short *red2 = &pixels[256 * threadIdx.z + 128];
    // 获取第 2 个 pattern 紫色区域数据指针
    unsigned short *pur2 = &pixels[256 * threadIdx.z + 192];

    // 计算对应的图像位置下标
    int pidx1 = couple == 0 ? 0 : 2 * couple - 1, pidx2 = 2 * couple;
    int idx1 = (centerY - _pd[pidx1].yinCord) * inimg.pitchBytes + centerX + 
              threadIdx.x - _pd[pidx1].xinCord;
    int idx2 = (centerY - _pd[pidx2].yinCord) * inimg.pitchBytes + centerX + 
              threadIdx.x - _pd[pidx2].xinCord;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 将对应的区域的图像数据复制到共享内存
    red1[tid] = GET_BIT(_pd[pidx1].r[threadIdx.y], threadIdx.x) * 
                inimg.imgMeta.imgData[idx1];
    pur1[tid] = GET_BIT(_pd[pidx1].p[threadIdx.y], threadIdx.x) * 
                inimg.imgMeta.imgData[idx1];
    red2[tid] = GET_BIT(_pd[pidx2].r[threadIdx.y], threadIdx.x) * 
                inimg.imgMeta.imgData[idx2];
    pur2[tid] = GET_BIT(_pd[pidx2].p[threadIdx.y], threadIdx.x) * 
                inimg.imgMeta.imgData[idx2];
    __syncthreads();
   
    // 使用 reduction 对各个区域内进行求和
    if (tid < 32) {
        red1[tid] += red1[tid + 32];
        pur1[tid] += pur1[tid + 32];
        red2[tid] += red2[tid + 32];
        pur2[tid] += pur2[tid + 32];
        __syncthreads();

        red1[tid] += red1[tid + 16];
        pur1[tid] += pur1[tid + 16];
        red2[tid] += red2[tid + 16];
        pur2[tid] += pur2[tid + 16];
        __syncthreads();

        red1[tid] += red1[tid + 8];
        pur1[tid] += pur1[tid + 8];
        red2[tid] += red2[tid + 8];
        pur2[tid] += pur2[tid + 8];
        __syncthreads();

        red1[tid] += red1[tid + 4];
        pur1[tid] += pur1[tid + 4];
        red2[tid] += red2[tid + 4];
        pur2[tid] += pur2[tid + 4];
        __syncthreads();

        red1[tid] += red1[tid + 2];
        pur1[tid] += pur1[tid + 2];
        red2[tid] += red2[tid + 2];
        pur2[tid] += pur2[tid + 2];
        __syncthreads();

        red1[tid] += red1[tid + 1];
        pur1[tid] += pur1[tid + 1];
        red2[tid] += red2[tid + 1];
        pur2[tid] += pur2[tid + 1];
        __syncthreads();

    }

    // 计算最终结果 
    if (tid == 0) {
        // 记录第一个 pattern 的紫色区域像素平均值
        avgs[pidx1] = pur1[0] * 1.0f / _pd[pidx1].pCount;
        // 保存第二个 pattern 的紫色区域像素平均值
        avgs[pidx2] = pur2[0] * 1.0f / _pd[pidx2].pCount;
	// 计算第 1 个 pattern 红色区域像素平均值和紫色区域像素平均值的差值
        float comp1 = red1[0] * 1.0f / _pd[pidx1].rCount - avgs[pidx1];
	// 计算第 2 个 pattern 红色区域像素平均值和紫色区域像素平均值的差值
        float comp2 = red2[0] * 1.0f / _pd[pidx2].rCount - avgs[pidx2];
	// 若两个 pattern 都满足同样的不等关系则将该 pattern 对序号标记为 0
        if ((comp1 > 0 && comp2 > 0) || (comp1 < 0 && comp2 < 0)) {
            patterns[threadIdx.z] = 0;
        }
    }
}


// Host 方法：doDiffPattern（检出图像特异的 pattern 信息）
__host__ int DiffPattern::doDiffPattern(Image *inimg, int *counter, 
                                        float *result)
{
     // 数据指针为空时返回空指针异常
     if (inimg == NULL || counter == NULL || result == NULL || indice == NULL)
        return NULL_POINTER;
     // 差分 pattern 对数为 0 返回数据异常
     if (patCount == 0 )
        return INVALID_DATA;
     
    int errcode;  // 局部变量，错误码
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 差分 pattern 对的序号数组，设备端使用的指针
    int *patterns;
    errcode = cudaMalloc(&patterns, patCount * sizeof (int));
    if (errcode != cudaSuccess)
        return errcode;
    // 初始数据与 indice 数组中的相同
    errcode = cudaMemcpy(patterns, indice, patCount * sizeof (int), 
                         cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess)
        return errcode;

    // 所有 19 个 pattern 的紫色区域像素平均值
    float *avgs = new float[19], *dev_avgs;
    // 数组置 0
    memset(avgs, 0, 19 * sizeof(float));
    errcode = cudaMalloc(&dev_avgs, 19 * sizeof (float));
    if (errcode != cudaSuccess)
        return errcode;
    errcode = cudaMemcpy(dev_avgs, avgs, 19 * sizeof (float), 
                         cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = patCount;
    gridsize.x = 1;
    gridsize.y = 1;
    int sharedSize = 256 * patCount * sizeof (unsigned short);

    // 调用核函数
    _diffPatternKer<<<gridsize, blocksize, sharedSize>>>
        (insubimgCud, 10, 10, patCount, patterns, dev_avgs);

    // 保存运算后的 pattern 对数组
    int *comp = new int[patCount];
    errcode = cudaMemcpy(comp, patterns, patCount * sizeof (int), 
                         cudaMemcpyDeviceToHost);
    if (errcode != cudaSuccess)
        return errcode;
    errcode = cudaMemcpy(avgs, dev_avgs, 19 * sizeof (float), 
                         cudaMemcpyDeviceToHost);
    if (errcode != cudaSuccess)
        return errcode;

    // 差异 pattern 对的数目的计数器
    *counter = 0;
    for (int i = 0; i < patCount; i++) {
        if (comp[i] != indice[i]) {
            int idx = 2 * indice[i] - 3;
            if (idx < 0)
                continue;

	    // 把有差异的 pattern 信息保存至数据指针
            result[6 * (*counter)] = _patData[idx].csX;
            result[6 * (*counter) + 1] = _patData[idx].csY;
            result[6 * (*counter) + 2] = _patData[idx].angel;
            result[6 * (*counter) + 3] = _patData[idx].ss;
            result[6 * (*counter) + 4] = _patData[idx].ls;
            result[6 * (*counter) + 5] = avgs[idx];
            (*counter)++;

            idx = 2 * indice[i] - 1;
            if (idx < 0)
                continue;
            result[6 * (*counter)] = _patData[idx].csX;
            result[6 * (*counter) + 1] = _patData[idx].csY;
            result[6 * (*counter) + 2] = _patData[idx].angel;
            result[6 * (*counter) + 3] = _patData[idx].ss;
            result[6 * (*counter) + 4] = _patData[idx].ls;
            result[6 * (*counter) + 5] = avgs[idx];
            (*counter)++;
        }
    }

    // 释放 Host 端内存空间
    delete []comp;
    delete []avgs;
    // 释放 Device 端内存空间
    cudaFree(patterns);
    cudaFree(dev_avgs);
    return NO_ERROR;
}

