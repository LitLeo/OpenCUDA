// LocalCluster.cu
// 局部聚类

#include "LocalCluster.h"


#include <iostream>
#include <cmath>
#include "stdio.h"
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y 以及 DEF_BLOCK_Z
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   1
#define DEF_BLOCK_Z   8

// Device 全局常量：
// 用于计算点之间的坐标关系。
const int static __device__ _idxDev[8][2] = { 
    // [0][ ], [1][ ]
    { 1,  0},  { 1, -1},
    // [2][ ], [3][ ]
    { 0, -1},  {-1, -1},
    // [4][ ], [5][ ]
    {-1,  0},  {-1,  1},
    // [6][ ], [7][ ]
    { 0,  1},  { 1,  1}
};
    
// Host 函数：_adjustRoiSize（调整 ROI 子图的大小）
// 调整 ROI 子图的大小，使输入和输出的子图大小统一。
static __host__ void       // 无返回值
_adjustRoiSize( 
        ImageCuda *inimg,  // 输入图像
        ImageCuda *outimg  // 输出图像
);
    
// Kernel 函数：_localCluKer（对每一个点进行局部聚类处理） 
// 给定一张图像略去边缘部分，在每一个点的八个方向上各求出 pntRange
// 个点的像素平均值并存放在共享变量 temp 中(根据河边老师发来的串行 
// 实现代码，pntRange 不能超过 100）。通过各个方向平均值与当前
// 像素值做差，从差值中选取满足条件的 pntCount 个点求平均值(根据
// 河边老师发来的串行实现代码，pntCount 不能超过 8），将该平均值
// 与当前像素值相加得出点的新像素值。
static __global__ void             // Kernel 函数无返回值
_localCluKer(
        ImageCuda inimg,           // 待处理图像
        ImageCuda outimg,          // 输出图像       
        const int pntrange,        // 在当前像素点的八个方向上，
                                   // 每个方向上取得点的个数，
                                   // 据河边老师发来的串行实现代码，
                                   // 不超过 100。
        unsigned char gapthred,    // 当前像素点和相邻点的灰度差
                                   // 的阈值。
        unsigned char diffethred,  // 当前点两侧各两个点的像素和
                                   // 的差的阈值。
        unsigned char problack,    // 像素值，默认为 0
        unsigned char prowhite,    // 像素值，默认为 250 
        int pntCount,              // 利用循环在 temp 数组中寻找
                                   // 最接近 0 的值循环次数的上界，
                                   // 据河边老师发来的串行实现代码，
                                   // 不超过 8。 
        int sx,                    // 处理边界，横坐标小于该值的点保留原值
        int ex,                    // 处理边界，横坐标大于该值的点保留原值
        int sy,                    // 处理边界，纵坐标小于该值的点保留原值
        int ey                     // 处理边界，纵坐标大于该值的点保留原值
); 

// Kernel 函数：_localCluKer（对每一个点进行局部聚类处理）
static __global__ void _localCluKer(
        ImageCuda inimg, ImageCuda outimg, const int pntrange,       
        unsigned char gapthred, unsigned char diffethred,
        unsigned char problack, unsigned char prowhite,
        int pntcount, int sx, int ex, int sy, int ey)
{  
    // 声明一个 extern 共享数组，存放每一个点的八个方向上的平均值。
    extern __shared__ float temp[];

    // 计算线程对应像素点的坐标位置，坐标的 x 和 y 分量。
    // z 表示计算方向
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    // 计算当前线程需要用到的共享内存地址
    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * 8 + z;
    float *curtemp = temp + idx;

    // 检查第像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    
    if (x >= inimg.imgMeta.width || y >= inimg.imgMeta.height) 
        return;
  
    // 当前线程对应的像素点一维索引和像素值
    int idxcv;
    unsigned char cv;
    idxcv = y * inimg.pitchBytes + x;
    cv = inimg.imgMeta.imgData[idxcv];

    // 对非边缘部分的点进行计算
    // 对于黑白比较明显的点保留原值 
    // 图像边缘上的点保持原像素值
    if (x < sx || x > ex || y < sy || y > ey ||
        cv <= problack || cv >= prowhite) {
        outimg.imgMeta.imgData[idxcv] = cv;
        return;
    }
    
    // 正在处理点像素值和索引
    // 注：正在处理点于当前点不同，处理点是当前计算点某个方向上的点。
    unsigned char cuv;
    int idxcuv = idxcv;
        
    // 处理点两侧各取两个点的像素和
    float side1, side2;
        
    // sum 用于累计 pntrange 个点的图像值
    // flag 为标记，标记循环次数
    float sum = cv;
    int i = 0;

    // 计算中涉及到的点的坐标
    int dx = x, dy = y;
    
    // 取出第一个点的像素值
    idxcuv = (y + _idxDev[z][1]) * inimg.pitchBytes + 
             (x + _idxDev[z][0]);
    cuv = inimg.imgMeta.imgData[idxcuv];
    
    // 在具体的某一个处理方向，处理的第一个点的左侧两个点像素值
    unsigned char pre1, pre2;
    pre1 = inimg.imgMeta.imgData[(y - _idxDev[z][1]) * inimg.pitchBytes + 
                                 (x - _idxDev[z][0])];
    pre2 = cv;
    
    // 在具体的某一个处理方向，处理的第一个点的右侧两个点像素值
    unsigned char latt1, latt2;
    latt1 = inimg.imgMeta.imgData[_idxDev[z][0] + idxcuv +  
                                  _idxDev[z][1] * inimg.pitchBytes];
                                   
    // 每一个线程循环处理某一个方向 pntrange 个点
    for (i = 1; i < pntrange; i++) {
       
        // 正在处理点和计算点的做差，如果超过 gapthred，
        // 则停止在该该方向上的累加，即跳出循环。
        if (abs((float)cv -(float) cuv) > gapthred) 
            break;
              
        // 两侧各取两个点：
        // 在具体的某一个处理方向，正在处理点的右侧第二个点的坐标，
        // 计算该点的索引，并取出该值
        dx += _idxDev[z][0] * 2;
        dy += _idxDev[z][1] * 2; 
        int idxsid;        
        idxsid = dy * inimg.pitchBytes + dx;                    
                
        // 取出该方向上右侧第二个点的像素值并把每侧的两个值相加               
        latt2 = inimg.imgMeta.imgData[idxsid];
        side1 = pre1 + pre2;
        side2 = latt1 + latt2;
                        
        // side1 与 side2 做差，如果超过 diffethred，
        // 则停止在该该方向上的累加，即跳出循环。
        if (abs(side1 - side2) > diffethred) 
            break;
        
        // 更新 pre1， pre2， cuv， latt1 为计算下个点做准备
        // 在计算下一个点时，它的当前值变为 latt1，
        // 其他三个变量也顺势向右移动一个点
        pre1 = pre2;
        pre2 = cuv;
        cuv = latt1;
        latt1 = latt2;

        // 将满足条件的像素值累加到 sum 中
        sum += (float)cuv;
    }
                
    // 当前计算方向上的像素平均值
    *curtemp = sum / i;

    // 设置线程块里面的线程同步    
    __syncthreads();
    
    // 对每一个点接下来的处理只需要用一个线程来处理，
    // 此时我们选择 z 等于 0 来处理，因此对 z 不等于 
    // 0 的线程 return。
    if (z != 0)
        return;
            
    float sumag = 0.0f;
    curtemp = &(temp[(threadIdx.y * blockDim.x  + threadIdx.x) * 8]);
    
    // 处理 temp，寻找 pntCount 个较小值（即最小，次小，以此类推），
    // 此时，该算法并未采取排序，而是当找到一个最大值时，并将该值重新
    // 设为 cv（作为哑值处理），再接着找次大值，并也设为 0，直到找完 
    // 8 - pntCount 个次大值，然后再将数组里面的所有值求平均值。
    int mark = 0; 
    for (int j = 8 - pntcount; j > 0; j--) {
        for (int i = 0; i < 8; i++) {
            if (abs(curtemp[i] - cv) >= abs(curtemp[mark] - cv) )
                mark = i;  
        }
        curtemp[mark] = cv;     
    } 
    
    for (int j = 0; j < 8; j ++)    
        sumag += curtemp[j]; 
    
    // 因为在处理时设定的哑值为 cv（即多加了 8 - pntcount 个 cv） ，
    // 故此处需要将改值减去     
    sumag -= (8 - pntcount) * cv;

    // 防止累加值越界并将结果写入输出图片
    outimg.imgMeta.imgData[idxcv] = 
            (sumag / pntcount > 255) ? 255 :
            (unsigned char)(sumag / pntcount);
}

// Host 函数：_adjustRoiSize（调整输入和输出图像的 ROI 的大小）
inline static __host__ void _adjustRoiSize(ImageCuda *inimg, 
                                           ImageCuda *outimg)
{
    if (inimg->imgMeta.width > outimg->imgMeta.width)
        inimg->imgMeta.width = outimg->imgMeta.width;
    else
        outimg->imgMeta.width = inimg->imgMeta.width;

    if (inimg->imgMeta.height > outimg->imgMeta.height)
        inimg->imgMeta.height = outimg->imgMeta.height;
    else
        outimg->imgMeta.height = inimg->imgMeta.height;
}

// Host 成员方法： localCluster（局部聚类）
__host__ int LocalCluster::localCluster(Image *inimg, Image *outimg)
{
    // 检查输入图像是否为 NULL
    if (inimg == NULL || inimg->imgData == NULL)
        return NULL_POINTER;
        
    // 输入图像的 ROI 区域尺寸
    int imgroix = inimg->roiX2 - inimg->roiX1;
    int imgroiy = inimg->roiY2 - inimg->roiY1; 
        
    // 将输入图像复制到 device
    int errcode;
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
        
    // 将 outimg 复制到 device
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和
    // 输入图像尺寸相同的图像             
    if (errcode != NO_ERROR) {
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, imgroix, imgroiy);
        // 如果创建图像也操作失败，报错退出。
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
        
    // 调整输入和输出图像的 ROI 子图，使大小统一
    _adjustRoiSize(&insubimgCud, &outsubimgCud);
    
    // 为 kernel 分配线程
    dim3 blockdim;	
    dim3 griddim;
    blockdim.x = DEF_BLOCK_X;
    blockdim.y = DEF_BLOCK_Y;
    blockdim.z = DEF_BLOCK_Z;
    griddim.x = (insubimgCud.imgMeta.width + blockdim.x - 1) /
                blockdim.x;
    griddim.y = (insubimgCud.imgMeta.height + blockdim.y - 1) /
                blockdim.y;
    griddim.z = 1;
    
    // 计算处理边界，处理的点如果不在这四个值的包围范围内部，则保留原值
    int sx, ex, sy, ey;
    sx = this->pntRange + 2;
    ex = insubimgCud.imgMeta.width - this->pntRange - 2;
    sy = this->pntRange + 2;
    ey = insubimgCud.imgMeta.height - this->pntRange - 2;
    
    // 计算共享内存大小
    int size = DEF_BLOCK_X * DEF_BLOCK_Y * 
               DEF_BLOCK_Z * sizeof (float);
               
    // 调用 kernel       
    _localCluKer<<<griddim, blockdim, size>>>(
            insubimgCud,outsubimgCud,
            this->pntRange, this->gapThred, this->diffeThred,
            this->proBlack, this->proWhite, this->pntCount,
            sx, ex, sy, ey);
    
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR; 
         
    // 处理完毕退出。
    return NO_ERROR;
}

