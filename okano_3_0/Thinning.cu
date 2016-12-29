// Thinning.cu
// 实现二值图像的细化算法

#include "Thinning.h"
#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：DEF_PATTERN_SIZE
// 定义了 PATTERN 表的默认大小。
#define DEF_PATTERN_SIZE  512

// 宏：DEF_PATTERN_LEN
// 定义了 PATTERN 表的个数。
#define DEF_PATTERN_LEN  4

// 宏：CST_IMG_WIDTH 和 CST_IMG_HEIGHT
// 定义了当输入参数为坐标集时，坐标集转化图像的大小。
#define CST_IMG_WIDTH   1024
#define CST_IMG_HEIGHT  1024

// static 变量：lutthin（PATTERN 表）
// 此一个 unsigned char 型数组的大小为 DEF_PATTERN_SIZE * 4 ， 为 4 个
// PATTERN 表，是 PATTERN 表细化算法里的四种判数断组。在这里将四个表合 
// 并为一个，便于将数据拷贝到 device 端。
unsigned char lutthin[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 
                            1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
                            0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 
                            1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 
                            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 
                            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
                            1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                            1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 
                            0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
                            0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 
                            0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
                            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
                            1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
                            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
                            1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 
                            1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 
                            0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
                            1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 
                            0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 
                            1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 
                            0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 
                            1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 
                            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
                            0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 
                            1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 
                            0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 
                            1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 
                            0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 
                            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 
                            0, 0 
                          };

// Kernel 函数：_thinMatSubFirKer（实现 PATTERN 表删除算法）
// 根据算法、PATTERN 表 1、PATTERN 表 2 和 PATTERN 表 3 对图像进行第一步细化
// 处理，对输出图像（已将输入图像完全拷贝到输出图像中）进行遍历，如果目标点的 
// 8 邻域满足 PATTERN 表 1、PATTERN 表 2 和 PATTERN 表 3 条件，则将 highpixel
// 置为 lowpixel 表示删除，否则不删除。并将第一步细化的结果存储在暂存图像 
// tempimg 中。
static __global__ void 
_thinMatSubFirKer(
        ImageCuda outimg,            // 输出图像
        ImageCuda tempimg,           // 暂存图像（暂时存储 PATTERN 表删除算法
                                     // 第一步操作的结果）
        unsigned char *devlutthin1,  // PATTERN 表 1
        unsigned char *devlutthin2,  // PATTERN 表 2
        unsigned char *devlutthin3,  // PATTERN 表 3
        unsigned char highpixel,     // 高像素 
        unsigned char lowpixel       // 低像素
);

// Kernel 函数：_thinMatSubSecKer（实现 PATTERN 表删除算法）
// 根据算法、PATTERN 表 1、PATTERN 表 2 和 PATTERN 表 4 对暂存图像进行第二次
// 细化处理,遍历暂存图像，如果目标点的 8 邻域满足 PATTERN 表 1、PATTERN 表 2
// 和 PATTERN 表 4 条件，将 highpixel 置为 lowpixel 表示删除，否则不删除。并
// 将第二步细化的结果重新存储到输出图像 outimg 中。同时将细化的点数存储在 
// devchangecount 中。在核函数外对 devchangecount 进行判断，如果其值 为 0 时，
// 即图像中没有点可以删除时，停止迭代。
static __global__ void  
_thinMatSubSecKer(
        ImageCuda tempimg,           // 暂存图像（暂时存储 PATTERN 表删除算法
                                     // 第一步操作的结果）
        ImageCuda outimg,            // 输出图像
        unsigned char *devlutthin1,  // PATTERN 表 1
        unsigned char *devlutthin2,  // PATTERN 表 2
        unsigned char *devlutthin4,  // PATTERN 表 4
        int *devchangecount,         // 经过一次细化后细化点的个数（用于判断
                                     // 是否继续迭代）
        unsigned char lowpixel       // 低像素
);

// Kernel 函数：_thinMatSubFirKer（实现 PATTERN 表删除算法的第一步操作）
static __global__ void _thinMatSubFirKer(ImageCuda outimg,
                                         ImageCuda tempimg,
                                         unsigned char *devlutthin1,
                                         unsigned char *devlutthin2,
                                         unsigned char *devlutthin3,
                                         unsigned char highpixel, 
                                         unsigned char lowpixel)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= outimg.imgMeta.width - 1 || 
        dstr >= outimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * outimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != lowpixel) {
        // 根据目标像素点 8 邻域的特性获取其在 PATTERN 表内的索引。
        int index = 0;

        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * outimg.pitchBytes;
        int posColumn2 = posColumn1 + outimg.pitchBytes;
        int posColumn3 = posColumn2 + outimg.pitchBytes;

        // 根据算法描述，对 8 邻域内的像素点赋予权重，对 8 邻域内的像素点进行
        // 遍历，将邻域内像素值为 highpixel 的像素点的权重值相加，即获得 8 邻 
        // 域内像素特性在 PATTERN 表内对应的索引值。以此获得目标像素在 PATTERN 
        // 表中对应的值。
        if (outimg.imgMeta.imgData[dstc - 1 + posColumn1] != lowpixel) 
            index += 1;
        if (outimg.imgMeta.imgData[dstc - 1 + posColumn2] != lowpixel) 
            index += 2;
        if (outimg.imgMeta.imgData[dstc - 1 + posColumn3] != lowpixel) 
            index += 4;
        if (outimg.imgMeta.imgData[dstc + posColumn1] != lowpixel) 
            index += 8;
        if (outimg.imgMeta.imgData[dstc + posColumn2] != lowpixel) 
            index += 16;
        if (outimg.imgMeta.imgData[dstc + posColumn3] != lowpixel) 
            index += 32;
        if (outimg.imgMeta.imgData[dstc + 1 + posColumn1] != lowpixel) 
            index += 64;
        if (outimg.imgMeta.imgData[dstc + 1 + posColumn2] != lowpixel) 
            index += 128;
        if (outimg.imgMeta.imgData[dstc + 1 + posColumn3] != lowpixel) 
            index += 256;

        // 获得索引值在 PATTERN 表 1 、2 、3 内对应的值
        unsigned char replacedPix1 = devlutthin1[index];
        unsigned char replacedPix2 = devlutthin2[index];
        unsigned char replacedPix3 = devlutthin3[index];
        
        // 根据获取的值得出初步细化结果，将结果中存储到暂存图像中。
        if (replacedPix1 == 1 && replacedPix2 == 1 && replacedPix3 == 1)
            tempimg.imgMeta.imgData[curpos] = lowpixel;
        else
            tempimg.imgMeta.imgData[curpos] = highpixel;
    } else {
        // 如果目标点的像素点为 lowpixel ，则直接将目标点的像素值赋给暂存图像。
        tempimg.imgMeta.imgData[curpos] = lowpixel; 
    }  
}

// Kernel 函数：_thinMatSubSecKer（实现 PATTERN 表删除算法的第二步操作）
static __global__ void _thinMatSubSecKer(ImageCuda tempimg,
                                         ImageCuda outimg,
                                         unsigned char *devlutthin1,
                                         unsigned char *devlutthin2,
                                         unsigned char *devlutthin4,
                                         int *devchangecount,
                                         unsigned char lowpixel)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量（其中，
    // c 表示 column， r 表示 row）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 1 || 
        dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
        return;

    // 定义目标点在暂存图像中位置的指针。
    unsigned char *temptr;

    // 获取当前像素点在暂存图像中的相对位置。
    int curpos = dstr * outimg.pitchBytes + dstc;
    
    // 获取当前像素点在图像中的绝对位置。
    temptr = tempimg.imgMeta.imgData + curpos;

    // 判断目标像素点是否删除，初始化为 false 。
    bool niv = false;

    // 如果暂存图像内目标像素点的像素值为 lowpixel , 则不进行第二步细化处理
    if (*temptr != lowpixel) {
        // 根据暂存图像内目标像素点 8 邻域的特性获取其在 PATTERN 表内的索引。
        int index = 0;

        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;

        // 根据算法描述，对 8 邻域内的像素点赋予权重，对 8 邻域内的像素点进行
        // 遍历，将邻域内像素值为 highpixel 的像素点的权重值相加，即获得 8 
        // 邻域内像素特性在 PATTERN 表内对应的索引值。以此获得目标像素在 
        // PATTERN 表中对应的值。
        if (tempimg.imgMeta.imgData[dstc - 1 + posColumn1] != lowpixel) 
            index += 1;
        if (tempimg.imgMeta.imgData[dstc - 1 + posColumn2] != lowpixel) 
            index += 2;
        if (tempimg.imgMeta.imgData[dstc - 1 + posColumn3] != lowpixel) 
            index += 4;
        if (tempimg.imgMeta.imgData[dstc + posColumn1] != lowpixel) 
            index += 8;
        if (tempimg.imgMeta.imgData[dstc + posColumn2] != lowpixel) 
            index += 16;
        if (tempimg.imgMeta.imgData[dstc + posColumn3] != lowpixel) 
            index += 32;
        if (tempimg.imgMeta.imgData[dstc + 1 + posColumn1] != lowpixel) 
            index += 64;
        if (tempimg.imgMeta.imgData[dstc + 1 + posColumn2] != lowpixel) 
            index += 128;
        if (tempimg.imgMeta.imgData[dstc + 1 + posColumn3] != lowpixel) 
            index += 256;

        // 获得索引值在 PATTERN 表 1、 2 、4 内对应的值
        unsigned char replacedPix1 = devlutthin1[index];
        unsigned char replacedPix2 = devlutthin2[index];
        unsigned char replacedPix4 = devlutthin4[index];

        // 根据从 PATTERN 表获取的值判断目标点是否删除。
        niv = !(replacedPix1 == 1 && replacedPix2 == 1 && replacedPix4 == 1);
    }

    // 经过细化处理后，如果输出图像对应的像素点的像素值为 highpixel 而 niv 的 
    // 值为 false，则将目标像素点置为 lowpixel。
    if (niv != (outimg.imgMeta.imgData[curpos] != lowpixel)) {
        // 删除目标像素点。 
        outimg.imgMeta.imgData[curpos] = lowpixel;

        // 记录删除点数的 devchangecount 值加 1 。
        atomicAdd(devchangecount, 1);
    }    
}

// 宏：FAIL_THIN_IMAGE_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_THIN_IMAGE_FREE  do {               \
        if (devlutthin != NULL)                  \
            cudaFree(devlutthin);               \
        if (tempimg != NULL)                    \
            ImageBasicOp::deleteImage(tempimg);  \
        if (devchangecount != NULL)              \
            cudaFree(devchangecount);            \
    } while (0)

// 成员方法：thinMatlabLike（细化边界 - PATTERN 表法）
__host__ int Thinning::thinMatlabLike(Image *inimg, Image *outimg)
{
    // 局部变量，错误码。
    int errcode;  
    cudaError_t cudaerrcode; 

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 声明所有中间变量并初始化为空。
    unsigned char *devlutthin = NULL;
    Image *tempimg = NULL;
    int *devchangecount = NULL;

    // 在 Device 上为 4 个 PATTERN 表 分配空间。并且因为 PATTERN 表在所给的代码
    // 中已给出，且大小都为 DEF_PATTERN_SIZE .在这里直接一次申请所有空间
    //（DEF_PATTERN_SIZE * 4），然后通过偏移索引各个 PATTERN 表。
    unsigned char *devlutthin1, *devlutthin2, *devlutthin3, *devlutthin4;
    cudaerrcode = cudaMalloc((void **)&devlutthin, 
                             DEF_PATTERN_SIZE * DEF_PATTERN_LEN * 
                             sizeof (unsigned char));
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

    // 通过偏移读取 Device 端内存空间。每次偏移一个 PATTERN 的长度，即 
    // DEF_PATTERN_SIZE。
    devlutthin1 = devlutthin;
    devlutthin2 = devlutthin1 + DEF_PATTERN_SIZE;
    devlutthin3 = devlutthin2 + DEF_PATTERN_SIZE;
    devlutthin4 = devlutthin3 + DEF_PATTERN_SIZE;

    // 将位于 host 的 PATTERN 表数据拷贝到 device 端。
    cudaerrcode = cudaMemcpy(devlutthin, lutthin, DEF_PATTERN_SIZE * 
                             DEF_PATTERN_LEN * sizeof (unsigned char), 
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) {
        FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 记录细化点数的变量，位于 host 端。
    int changeCount;

    // 记录细化点数的变量，位于 device 端。并为其申请空间。
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 将输入图像 inimg 完全拷贝到输出图像 outimg ，并将 outimg 拷贝到 
    // device 端。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR) {
        FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取输出图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 提取暂存图像
    ImageCuda tempsubimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
    if (errcode != NO_ERROR) {
        FAIL_THIN_IMAGE_FREE;
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 赋值为 1，以便开始第一次迭代。
    changeCount = 1;

    // 开始迭代，当不可再被细化，即记录细化点数的变量 changeCount 的值为 0 时，
    // 停止迭代。 
    while (changeCount > 0) {
        // 将 host 端的变量赋值为 0 ，并将值拷贝到 device 端的 devchangecount。
        changeCount = 0;
        cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                                 cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }
        
        // 调用核函数，开始第一步细化操作。
        _thinMatSubFirKer<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud,
                                                   devlutthin1, devlutthin2,
                                                   devlutthin3, highPixel,
                                                   lowPixel);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
            FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        // 调用核函数，开始第二步细化操作。
        _thinMatSubSecKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                                                   devlutthin1, devlutthin2,
                                                   devlutthin4, devchangecount,
                                                   lowPixel);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间 。
            FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }     
        
        // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount 
        // 变量，进行迭代判断。
        cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                                 cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

   }

    // 细化结束后释放申请的变量空间。
    cudaFree(devlutthin);
    cudaFree(devchangecount);
    ImageBasicOp::deleteImage(tempimg);

    return NO_ERROR;
}

// 取消前面的宏定义。
#undef FAIL_THIN_IMAGE_FREE

// 成员方法：thinMatlabLike（细化边界- PATTERN 表法）
__host__ int Thinning::thinMatlabLike(CoordiSet *incst, 
                                      CoordiSet *outcst)
{
    // 局部变量，错误码。
    int errcode;

    // 检查输入坐标集，输出坐标集是否为空。
    if (incst == NULL || outcst == NULL)
        return NULL_POINTER;

    // 生成输入图像。
    Image *inimg;
    errcode = ImageBasicOp::newImage(&inimg);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 图像大小默认为 1024 * 1024。
    errcode = ImageBasicOp::makeAtHost(inimg, CST_IMG_WIDTH, 
                                       CST_IMG_HEIGHT);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放 Device 端空间。
        ImageBasicOp::deleteImage(inimg);
        return errcode;
    }

    // 生成输出图像。
    Image *outimg;
    errcode = ImageBasicOp::newImage(&outimg);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放 Device 端空间。
        ImageBasicOp::deleteImage(inimg);
        return errcode;
    }

    // 图像大小默认为 1024 * 1024。
    errcode = ImageBasicOp::makeAtHost(outimg, CST_IMG_WIDTH, 
                                       CST_IMG_HEIGHT);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放 Device 端空间。
        ImageBasicOp::deleteImage(inimg);
        ImageBasicOp::deleteImage(outimg);
        return errcode;
    }

    // 利用转换器将输入坐标集转化为输入图像。
    errcode = this->imgCon.cstConvertToImg(incst, inimg);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放 Device 端空间。
        ImageBasicOp::deleteImage(inimg);
        ImageBasicOp::deleteImage(outimg);
        return errcode;
    }

    // 细化图像。
    errcode = this->thinMatlabLike(inimg, outimg);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放 Device 端空间。
        ImageBasicOp::deleteImage(inimg);
        ImageBasicOp::deleteImage(outimg);
        return errcode;
    }
    
    // 利用转换器将输出图像转化为输出坐标集。
    errcode = this->imgCon.imgConvertToCst(outimg, outcst);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放 Device 端空间。
        ImageBasicOp::deleteImage(inimg);
        ImageBasicOp::deleteImage(outimg);
        return errcode;
    }

    // 细化结束后释放申请的变量空间。
    ImageBasicOp::deleteImage(inimg);
    ImageBasicOp::deleteImage(outimg);
        
    return NO_ERROR;
}

