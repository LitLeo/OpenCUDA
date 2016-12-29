// EdgeCheck.h
// 边缘异常点和片段检查

#include "EdgeCheck.h"

#include <iostream>
#include <cmath>
using namespace std;


// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：DEF_BLOCK_1D
// 定义一维块大小。
#define DEF_BLOCK_1D  256

// 宏：DEF_COL_MAX
// 定义欧式距离最大列数。
#define DEF_COL_MAX  1024

// 宏：DEF_HUMOM_SIZE
// 定义 Hu 矩大小。
#define DEF_HUMOM_SIZE  7

// 宏：ERR_EDEFORM 和 ERR_EHUMOM
// 定义错误码。
#define ERR_EDEFORM  200
#define ERR_EHUMOM   150

// 宏：DEF_ERR_COUNT
// errmap 中每个点的邻域大小中错误点的个数。
#define DEF_ERR_COUNT  3

// 宏：DEF_INVALID_FLOAT
// 定义 float 类型的无效数据。
#define DEF_INVALID_FLOAT  100000000.0f


// Kernel 函数：_edgeMatchKer（边缘匹配算法）
// 计算测试图像每个参考图像的相关系数。
static __global__ void     // Kernel 函数无返回值
_edgeMatchKer(
        ImageCuda teimg,   // 测试图像
        ImageCuda *reimg,  // 参考图像数组
        int recount,       // 参考图像数量
        float *cormapsum   // 相关系数数组
);

// Kernel 函数：_getCormapMaxIndexKer（获取 cormapsum 中最大的值的索引）
// 在匹配得到的结果中找到最大的值。
static __global__ void    // Kernel 函数无返回值
_getCormapMaxIndexKer(
        float *cormap,    // cormapsum 的数据
        int count,        // cormapsum 中数据的数量
        int *maxindx      // 最大值索引
);

// Kernel 函数：_imgConvertCstKer（实现将图像转化为坐标集算法）
// 当输入参数为坐标集时，此算法将细化后的图像转化为输出坐标集。
static __global__ void            // Kernel 函数无返回值
_imgConvertCstKer(
        ImageCuda outimg,         // 输出图像
        CoordiSet outcst,         // 输出坐标 
        unsigned char highpixel,  // 高像素
        int *outcstcount          // 坐标集索引
);

// Kernel 函数：_localMomentsKer（计算边缘点的 local moments）
// 计算边缘坐标集合上的每个点的 local moments。
static __global__ void    // Kernel 函数无返回值
_localMomentsKer(
        CoordiSet cdset,  // 边缘的坐标集合
        int width,        // 顺逆时针跟踪的宽度
        float *moments    // local moments 特征矩阵
);

// Kernel 函数：_euclidMatKer（计算边缘点间的欧式距离）
// 计算参考边缘和测试边缘的所有点的欧式距离，输出到矩阵中。
static __global__ void      // Kernel 函数无返回值
_euclidMatKer(
        CoordiSet recdset,  // 参考边缘的坐标集合
        CoordiSet tecdset,  // 测试边缘的坐标集合
        int group,          // 参考边缘的分段大小
        float *eudmat,      // 欧式距离矩阵
        float *indexmat     // 索引下标矩阵
);

// Kernel 函数: _findRowMinKer（查找行最小值）
// 根据差值矩阵 diffmatrix，查找每一行的最小值，并将每一行出现最小
// 值的行列号保存在数组 rowmin 中。
static __global__ void    // Kernel 函数无返回值
_findRowMinKer(
        float *eudmat,    // 欧式距离矩阵
        float *indexmat,  // 索引下标矩阵
        int matwidth,     // 距离列数大小
        int rowlen,       // 最大列数
        float maxdis2,    // 欧式距离阈值平方
        int *rowmin       // 行最小值矩阵
);
                                      
// Kernel 函数： _relateCoeffKer（计算特征向量间的相关系数）
// 根据测试边缘点和参考边缘点的对应关系，计算 local moments 特征向量间的标准
// 相关系数。
static __global__ void     // Kernel 函数无返回值
_relateCoeffKer(
        float *temoments,  // 测试边缘点的 local moments 特征矩阵
        float *remoments,  // 测试边缘点的 local moments 特征矩阵
        float mincor,      // 标准相关系数的阈值大小
        int *rowmin        // 相关点的对应索引值
);

// Kernel 函数： _makeErrmapKer（错误码标记输入到 errmap 图像中）
// 根据 rowmin 中的错误码标记，输入到 errmap 图像中。
static __global__ void      // Kernel 函数无返回值
_makeErrmapKer(
        CoordiSet tecdset,  // 边缘的坐标集合
        int *rowmin,        // 相关点的对应索引值
        ImageCuda errmap    // 错误码图像
);

// Kernel 函数： _confirmErrmapKer（确定 errmap 中的错误点为异常）
// 根据 errmap 图像中每个错误点，如果其 3 * 3 邻域内包括 3 个以上的异常点，
// 则确定当前错误点为异常点。
static __global__ void      // Kernel 函数无返回值
_confirmErrmapKer(
        ImageCuda errmap,   // 错误码图像
        ImageCuda errpoint  // 异常点图像
);

// Kernel 函数：_edgeMatchKer（边缘匹配算法）
static __global__ void _edgeMatchKer(ImageCuda teimg, ImageCuda *reimg,
                                     int recount, float *cormapsum)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标的
    // x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 8 个输出像素，这八个像素位于统一列的相邻行上，
    // 因此，对于 r 需要进行乘 8 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 8;
    int z = blockIdx.z;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    // 同时检查 z 的值，若大于参考图像数量则 z 值无效，直接返回。
    if (c >= teimg.imgMeta.width || r >= teimg.imgMeta.height || z >= recount)
        return; 
                  
    // 申请一个 float 型共享内存，用于存储每每个块内线程所计算相关系数总和。
    __shared__ float cormap;
   
    // 读取线程号。
    int threadid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 局部变量，存储当前线程内像素点的相关系数和。
    float tcormap = 0.0f;  
    
    // 存储测试图像和参考图像的像素值。
    unsigned char tepix,repix;
    
    // 用每个块内的0号线程给共享内存赋初值。
    if (threadid == 0 )
        cormap = 0.0f;
        
    // 块内同步。
    __syncthreads();
  
    // 计算测试图像第一个输入坐标点对应的图像数据数组下标。
    int  teimgidx = r * teimg.pitchBytes + c;
    
    // 计算参考图像第一个输入坐标点对应的图像数据数组下标。
    int  reimgidx = r * reimg[z].pitchBytes + c;
 
    // 读取第一个输入坐标点对应的像素值。
    tepix = teimg.imgMeta.imgData[teimgidx];
    repix = reimg[z].imgMeta.imgData[reimgidx];
  
    // 计算当前两个点的相关系数。
    tcormap = tepix * repix;
   
    // 处理后 7 个点。  
    for(int j = 1; j < 8; j++) {
        // y 分量加 1 。 
        r++;
        // 获取当前像素点坐标。
        teimgidx += teimg.pitchBytes;
        reimgidx += reimg[z].pitchBytes;
        
        // 若当前像素点越界，则跳过该点，处理下一点；否则计算当前点的相关系数。
        if (r < teimg.imgMeta.height) {
            // 读取第 j 个输入坐标点对应的测试图像和参考图像像素值。
            tepix = teimg.imgMeta.imgData[teimgidx];
            repix = reimg[z].imgMeta.imgData[reimgidx];
            
            // 计算当前两个点的相关系数并累加。
            tcormap += tepix * repix;    
        }
    }
      
    // 原子操作将当前线程所计算相关系数和累加到共享内存中。 
    tcormap = tcormap / (255 * 255); 
    atomicAdd(&cormap, tcormap);
  
    // 块内同步。 
    __syncthreads();

    // 每个块内 0 号线程将该块所计算的相关系数和累加入
    // 每个参考图像总的相关系数和中。  
    if (threadid == 0 )  
        // 原子操作将该块所计算的相关系数和累加入每个参考图像总的相关系数和中。   
        atomicAdd(&cormapsum[z], cormap);
}

// Kernel 函数：_imgConvertCstKer（实现将图像转化为坐标集算法）
static __global__ void _imgConvertCstKer(ImageCuda outimg, CoordiSet outcst,
                                         unsigned char highpixel,
                                         int *outcstcount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，
    // dstc 表示 column， dstr 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (dstc >= outimg.imgMeta.width || dstr >= outimg.imgMeta.height)
        return ;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * outimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos;
    
    // 若当前像素值等于 highpixel 值。
    if (*outptr == highpixel) {
        // 原子操作获得当前坐标点的索引值。
        int idx = atomicAdd(outcstcount, 1);
        
        // 保存图像的横纵坐标到输出坐标集中。
        outcst.tplData[idx * 2] = dstc;
        outcst.tplData[idx * 2 + 1] = dstr;
    }
}

// Kernel 函数：_getCormapMaxIndexKer（获取 cormapsum 中最大的值的索引）
static __global__ void _getCormapMaxIndexKer(float *cormapcpu, 
                                             int count,int *maxindex)
{
    // 获取当前线程的线程号。
    int threadtid = threadIdx.x;
         
    // 声明共享内存，保存当前块内的相关系数矩阵和索引矩阵。
    extern __shared__ float datashare[];
    
    if (threadtid < count) {
        // 将当前线程对应的相关系数矩阵中的值以及其对应的索引（即列号）保存
        // 在该块的共享内存中。 
        datashare[threadtid] = *(cormapcpu + threadtid);
        datashare[threadtid + count] = threadtid;
    } else {
        datashare[threadtid] = DEF_INVALID_FLOAT;
        datashare[threadtid + count] = DEF_INVALID_FLOAT;
    }
      
    // 块内同步，为了保证一个块内的所有线程都已经完成了上述操作，即存
    // 储该行的欧式距离和索引到共享内存中。
    __syncthreads();

    // 使用双调排序的思想，找到该行的最小值。
    for (int k = 1; k < count; k <<= 1) {
        // 对待排序的元素进行分组，每次都将较大的元素交换到数组中
        // 较前的位置，然后改变分组大小，进而在比较上一次得到的较大值
        // 并做相应的交换，以此类推，最终数组中第 0 号元素存放的是该行
        // 的最大值。
        if (((threadtid % (k << 1)) == 0) &&
            datashare[threadtid] < datashare[threadtid + k] ) {
            // 两个值进行交换。
            float temp1 = datashare[threadtid];
            datashare[threadtid] = datashare[threadtid + k];
            datashare[threadtid + k] = temp1;
            
            // 交换相对应的索引 index 值。
            float temp2 = datashare[threadtid + count];  
            datashare[threadtid + count] = datashare[threadtid + k + count];
            datashare[threadtid + k + count] = temp2;
        } 
        // 块内同步。
        __syncthreads();
    }
    
    // 将最大值的索引保存在 maxindex 中。
    *maxindex = (int)datashare[count];   
}

// Kernel 函数：_localMomentsKer（计算边缘点的 local moments）
static __global__ void _localMomentsKer(CoordiSet cdset, int width,
                                        float *moments)
{
    // 读取线程号。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 获得坐标集大小。
    int count = cdset.count;

    // 多余的线程直接退出。
    if (idx >= count)
        return;

    // 声明空间矩。
    float m00 = 0.0f, m01 = 0.0f, m10 = 0.0f, m11 = 0.0f, m02 = 0.0f, 
          m20 = 0.0f, m12 = 0.0f, m21 = 0.0f, m03 = 0.0f, m30 = 0.0f;
    
    // 计算当前边缘点的空间矩。
    for (int i = -width; i <= width; i++) {
        // 判断邻域是否越界，注意坐标集合的首尾显示上是相连的。
        int temp = idx + i;
        if (temp < 0)
            temp = count + temp;
        if (temp > count - 1)
            temp = temp - count;
   
        // 获得当前边缘点的横纵坐标。
        int xdata = cdset.tplData[2 * temp];
        int ydata = cdset.tplData[2 * temp + 1];

        // 计算当前边缘点的空间矩。
        m00 += 1;
        m01 += ydata;
        m10 += xdata;
        m11 += xdata * ydata;
        m02 += ydata * ydata;
        m20 += xdata * xdata;
        m12 += xdata * ydata * ydata;
        m21 += xdata * xdata * ydata;
        m03 += ydata * ydata * ydata;
        m30 += xdata * xdata * xdata;
    }

    // 声明矩中心。
    float centerx = 0.0f, centery = 0.0f;
    centerx = m10 / m00;
    centery = m01 / m00;

    // 声明中心矩。
    float u00, /*u01, u10,*/ u11, u02, u20, u12, u21, u03, u30;
    float centerx2 = centerx * centerx;
    float centery2 = centery * centery;

    // 计算当前边缘点的中心矩。
    u00 = m00;
    // u01 = 0.0f;
    // u10 = 0.0f;
    u11 = m11 - centerx * m01;
    u20 = m20 - centerx * m10;
    u02 = m02 - centery * m01;
    u21 = m21 - 2 * centerx * m11 - centery * m20 + 2 * centerx2 * m01;
    u12 = m12 - 2 * centery * m11 - centerx * m02 + 2 * centery2 * m10;
    u30 = m30 - 3 * centerx * m20 + 2 * centerx2 * m10;
    u03 = m03 - 3 * centery * m02 + 2 * centery2 * m01;

    // 声明正规矩。
    float /*n00, n01, n10,*/ n11, n02, n20, n12, n21, n03, n30;
    float temp1= pow(u00, 2.0f), temp2 = pow(u00, 2.5f);

    // 计算当前边缘点的中心矩。
    n11 = u11 / temp1;
    n20 = u20 / temp1;
    n02 = u02 / temp1;
    n21 = u21 / temp2;
    n12 = u12 / temp2;
    n30 = u30 / temp2;
    n03 = u03 / temp2;

    // 计算当前边缘点在特征矩阵中对应的行数。
    float * humoments = moments + idx * DEF_HUMOM_SIZE;

    // 声明临时变量，减少重复计算。
    float t0 = n30 + n12, t1 = n21 + n03;
    float q0 = t0 * t0, q1 = t1 * t1;
    float n4 = 4 * n11;
    float s = n20 + n02, d = n20 - n02;

    // 计算 Hu 矩值 0, 1, 3, 5。
    humoments[0] = s;
    humoments[1] = d * d + n4 * n11;
    humoments[3] = q0 + q1;
    humoments[5] = d * (q0 - q1) + n4 * t0 * t1;

    // 改变临时变量。
    t0 *= q0 - 3 * q1;
    t1 *= 3 * q0 - q1;
    q0 = n30 - 3 * n12;
    q1 = 3 * n21 - n03;

    // 计算 Hu 矩值 2, 4, 6。
    humoments[2] = q0 * q0 + q1 * q1;
    humoments[4] = q0 * t0 + q1 * t1;
    humoments[6] = q1 * t0 - q0 * t1;
}

// Kernel 函数：_euclidMatKer（计算边缘点间的欧式距离）
static __global__ void _euclidMatKer(CoordiSet recdset, CoordiSet tecdset,
                                     int group, float *eudmat, float *indexmat)
{
    // 获取当前线程的块号。
    int blocktid = blockIdx.x;
    // 获取当前线程的线程号。
    int threadtid = threadIdx.x;
    // 计算矩阵中对应的输出点的位置。
    int inidx = blockIdx.x * blockDim.x + threadIdx.x;

    // 获得当前线程对应的测试边缘点的坐标。
    int tecurx = tecdset.tplData[2 * blocktid];
    int tecury = tecdset.tplData[2 * blocktid + 1];

    // 保存分段内 group 个距离的最小值。
    float mindis = DEF_INVALID_FLOAT;
    // 记录最小值下标。
    int minindex = 0;

    // 测试边缘的每个坐标点与分段内 group 个参考边缘的坐标点计算欧式距离。这样
    // 可以减少欧式矩阵的大小，便于后续算法操作。 
    for (int i = 0; i < group; i++) {
        // 获得参考边缘点的坐标。
        int tempidx = threadtid * group + i;
        int recurx = recdset.tplData[2 * tempidx];
        int recury = recdset.tplData[2 * tempidx + 1];
        // 计算欧式距离。
        float distemp = (float)((tecurx - recurx) * (tecurx - recurx) + 
                                (tecury - recury) * (tecury - recury));
        
        // 记录最小值和下标。
        if (distemp < mindis) {
            mindis = distemp;
            minindex = tempidx;
        }
    }

    // 将最小值保存到矩阵的当前元素中。
    *(eudmat + inidx) = mindis;
    *(indexmat + inidx) = (float)minindex;
}

// Kernel 函数: _findRowMinKer（查找行最小值）
static __global__ void _findRowMinKer(float *eudmat, float *indexmat, 
                                      int matwidth, int rowlen, 
                                      float maxdis2, int *rowmin)
{ 
    // 获取当前线程的块号。
    int blocktid = blockIdx.x;
    // 获取当前线程的线程号。
    int threadtid = threadIdx.x;
    // 计算当前线程在矩阵中的偏移。
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
    // 声明共享内存，保存当前块内的欧式距离矩阵和索引矩阵。
    extern __shared__ float datashare[];
    
    if (threadtid < matwidth) {
        // 将当前线程对应的差值矩阵中的值以及其对应的索引（即列号）保存
        // 在该块的共享内存中。 
        datashare[threadtid] = *(eudmat + tid);
        datashare[threadtid + rowlen] = *(indexmat + tid);
    } else {
        datashare[threadtid] = DEF_INVALID_FLOAT;
        datashare[threadtid + rowlen] = DEF_INVALID_FLOAT;
    }

    // 块内同步，为了保证一个块内的所有线程都已经完成了上述操作，即存
    // 储该行的欧式距离和索引到共享内存中。
    __syncthreads();

    // 使用双调排序的思想，找到该行的最小值。
    for (int k = 1; k < rowlen; k <<= 1) {
        // 对待排序的元素进行分组，每次都将距离较小的元素交换到数组中
        // 较前的位置，然后改变分组大小，进而在比较上一次得到的较小值
        // 并做相应的交换，以此类推，最终数组中第 0 号元素存放的是该行
        // 的最小值。
        if (((threadtid % (k << 1)) == 0) &&
            datashare[threadtid] > datashare[threadtid + k] ) {
            // 两个欧式距离进行交换。
            float temp1 = datashare[threadtid];
            datashare[threadtid] = datashare[threadtid + k];
            datashare[threadtid + k] = temp1;
            
            // 交换相对应的索引 index 值。
            float temp2 = datashare[threadtid + rowlen];  
            datashare[threadtid + rowlen] = datashare[threadtid + k + rowlen];
            datashare[threadtid + k + rowlen] = temp2;
        } 
        // 块内同步。
        __syncthreads();
    }
    
    // 将当前行最小值出现的列号保存在数组 rowmin 中。如果最小距离大于指定阈值，
    // 则设置错误码 ERR_EDEFORM。
    if (datashare[0] < maxdis2)
        rowmin[blocktid] = (int)datashare[rowlen];
    else
        rowmin[blocktid] = ERR_EDEFORM;      
}

// Kernel 函数： _relateCoeffKer（计算特征向量间的相关系数）
static __global__ void _relateCoeffKer(
        float *temoments, float *remoments, float mincor, int *rowmin)
{
    // 读取线程号。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 获得当前测试边缘点对应的参考边缘点坐标。
    int corr = rowmin[idx];

    // 如果没有对应的参考边缘点，则不处理当前的测试边缘点。
    if (corr == ERR_EDEFORM)
        return;

    // 获得当前测试边缘点的 local moments。
    float *tepoint = temoments + idx * DEF_HUMOM_SIZE;

    // 获得对应的参考边缘点的 local moments。
    float *repoint = remoments + idx * DEF_HUMOM_SIZE;

    // 计算标准相关系数。
    float sum = 0.0f;
    for (int i = 0; i < 7; i++) {
        sum += tepoint[i] * repoint[i];
    }
    sum /= 7;

    if (sum < mincor)
        rowmin[idx] = ERR_EHUMOM;
}

// Kernel 函数： _makeErrmapKer（错误码标记输入到 errmap 图像中）
static __global__ void _makeErrmapKer(CoordiSet tecdset, int *rowmin, 
                                      ImageCuda errmap)
{
    // 读取线程号。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 判断线程是否超出界限。
    if (idx >= tecdset.count)
        return;

    // 获得当前点在 errmap 图像中的位置。
    int curerr = tecdset.tplData[2 * idx + 1] * errmap.pitchBytes + 
                 tecdset.tplData[2 * idx];
                     
    // 如果当前标记是错误码的话，则是异常点。
    if (rowmin[idx] == ERR_EDEFORM || rowmin[idx] == ERR_EHUMOM) {
        // 将错误码输出到 errmap 图像中，表示该点是异常点。
        errmap.imgMeta.imgData[curerr] = (unsigned char)rowmin[idx];
    } else {
        // 否则设置值为 0，表示非异常点。
        errmap.imgMeta.imgData[curerr] = 0;
    }
}

// Kernel 函数： _confirmErrmapKer（确定 errmap 中的错误点为异常）
static __global__ void _confirmErrmapKer(ImageCuda errmap, ImageCuda errpoint)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，
    // c 表示 column， r 表示 row）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃
    if (dstc >= errmap.imgMeta.width || dstr >= errmap.imgMeta.height)
        return;

    // 记录输入图像对应位置。
    unsigned char *curinptr;
    curinptr = errmap.imgMeta.imgData + dstc + dstr * errmap.pitchBytes;

    // 如果当前点不是错误点则直接退出。
    if (*curinptr != ERR_EDEFORM && *curinptr != ERR_EHUMOM)
        return;
    
    // 记录输出图像对应位置。
    unsigned char *curoutptr;
    curoutptr = errpoint.imgMeta.imgData + dstc + dstr * errpoint.pitchBytes;

    // 因为是存放邻域内错误点的个数，所以先初始化为最小值 0x00。
    unsigned char count = 0;
    
    // 保存邻域的像素值。
    unsigned char neighpixel;

    for (int j = dstr - 1; j <= dstr + 1; j++) {
        for (int i = dstc - 1; i <= dstc + 1; i++) {
            // 判断当前像素是否越界。
            if (j >= 0 && j < errmap.imgMeta.height && 
                i >= 0 && i < errmap.imgMeta.width) {
                // 循环计算每个邻域内错误点的个数。
                neighpixel = *(errmap.imgMeta.imgData + i + 
                               j * errmap.pitchBytes);         
                if (neighpixel == ERR_EDEFORM || neighpixel == ERR_EHUMOM)
                    count++;
                
                // 如果计数个数大于 DEF_ERR_COUNT 个，则确定当前点为异常点，
                // 并结束循环。
                if (count >= DEF_ERR_COUNT) {
                    *curoutptr = *curinptr;
                    return;
                }
            }
        }    
    }
}

// 宏：FAIL_EDGECHECK_SPACE_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_EDGECHECK_SPACE_FREE  do {               \
     if (reimgCud != NULL)                            \
         delete []reimgCud;                           \
     if (tecdset != NULL)                             \
         CoordiSetBasicOp::deleteCoordiSet(tecdset);  \
     if (recdset != NULL)                             \
         CoordiSetBasicOp::deleteCoordiSet(recdset);  \
     if (reimgcudDev != NULL)                         \
         cudaFree(reimgcudDev);                       \
     if (alldevpointer != NULL)                       \
         cudaFree(alldevpointer);                     \
     if (alldevpointermat != NULL)                    \
         cudaFree(alldevpointermat);                  \
     if (errmap != NULL)                              \
         ImageBasicOp::deleteImage(errmap);           \
    } while (0)
    
// Host 成员方法：edgeCheckPoint（边缘的异常点检查）
__host__ int EdgeCheck::edgeCheckPoint(Image *teimg, Image *errpoint)
{
    // 检查测试图像， errpoint 和参考图像是否为空，若为空则直接返回。
    if (teimg == NULL || errpoint == NULL || reImages == NULL)
        return NULL_POINTER;
     
    // 检查每幅参考图像是否为空，若为空则直接返回。   
    for(int i = 0; i < reCount; i++) {
        if(reImages[i] == NULL)
        return NULL_POINTER;
    }
    
    // 局部变量，错误码。
    int errcode;  
    
    // 声明所有中间变量并初始化为空。
    ImageCuda *reimgCud = NULL;     
    CoordiSet *tecdset = NULL;    
    CoordiSet *recdset = NULL; 
    ImageCuda *reimgcudDev = NULL;
    float *alldevpointer = NULL;
    float *alldevpointermat = NULL;
    Image *errmap = NULL;

    // 将测试图像数据拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(teimg); 
    if (errcode != NO_ERROR)
        return errcode;
        
    // 提取测试图像的 ROI 子图。
    ImageCuda teimgCud;
    errcode = ImageBasicOp::roiSubImage(teimg, &teimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 将所有参考图像拷贝入 Device 内存。
    for (int i = 0; i < reCount; i++) {
        errcode = ImageBasicOp::copyToCurrentDevice(reImages[i]); 
        if (errcode != NO_ERROR)       
            return errcode;
    }
    
    // 提取所有参考图像的 ROI 子图。
    reimgCud = new ImageCuda[reCount];
    
    for (int i = 0; i < reCount; i++) {
        errcode = ImageBasicOp::roiSubImage(reImages[i], &reimgCud[i]);
        if (errcode != NO_ERROR)
            return errcode;
    }
    
    // 为 reimgcudDev 分配内存空间。
    errcode = cudaMalloc((void **)&reimgcudDev, 
                         reCount * sizeof (ImageCuda));
    if (errcode != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }
    
    // 将 Host 上的 reimgCud 拷贝到 Device 上。
    errcode = cudaMemcpy(reimgcudDev, reimgCud,
                         reCount * sizeof (ImageCuda), cudaMemcpyHostToDevice);
    // 判断是否拷贝成功，若失败，释放之前的空间，防止内存泄漏，然后返回错误。
    if (errcode != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }
        
    // 计算坐标集初始大小。
    int count = teimg->height * teimg->width;
        
    // 一次申请 Device 端所有空间。
    cudaError_t cudaerrcode;  
    
    // 申明所有指针变量。
    float *cormapsumDev;
    int *maxindexDev, *tecountdev, *recountdev;
    float *deveudmat, *devindexmat;
    int *devrowmin;
    
    // 为 alldevpointer 分配空间。
    cudaerrcode = cudaMalloc((void **)&alldevpointer, 
                             (reCount + 3) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return cudaerrcode;
    }
   
    // 初始化所有 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldevpointer, 0,
                             (reCount + 3) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return cudaerrcode;
    }
    // 获得 cormapsumDev 位置指针。
    cormapsumDev = alldevpointer; 
     
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = 1;
    gridsize.x = (teimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (teimgCud.imgMeta.height + blocksize.y * 8 - 1) /
                 (blocksize.y * 8);
    gridsize.z = reCount;
                             
    // 调用匹配函数对每个参考图像进行匹配。
    _edgeMatchKer<<<gridsize, blocksize>>>(teimgCud,reimgcudDev,
                                           reCount,cormapsumDev);
    
    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return CUDA_ERROR;
    }
   
    // 获得 maxindexDev 位置指针。 
    maxindexDev = (int *)(cormapsumDev + reCount);
             
    size_t blocksize1D, gridsize1D;      
    blocksize1D = 1;
    // 调用 _getCormapMaxIndexKer 函数，找到相关系数最大的参考图像，
    // 即为匹配图像。
    _getCormapMaxIndexKer<<<blocksize1D, reCount, 2 * reCount>>>(cormapsumDev,
                                                                 reCount,
                                                                 maxindexDev);
                                                           
    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return CUDA_ERROR;
    }
    
    // 将 Device 端的 maxindexDev 拷贝到 Host 端。
    int maxindex;
    errcode = cudaMemcpy(&maxindex, maxindexDev,
                         sizeof (int), cudaMemcpyDeviceToHost);
    // 判断是否拷贝成功，若失败，释放之前的空间，防止内存泄漏，然后返回错误。
    if (errcode != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }

    // 获得 tecountdev 和 recountdev 位置指针。 
    tecountdev = maxindexDev + 1;    
    recountdev = tecountdev + 1;  
      
    // 创建测试边缘的坐标集合和匹配得到的参考边缘的坐标集合。    
    CoordiSetBasicOp::newCoordiSet(&tecdset);
    if (errcode != NO_ERROR) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }
    CoordiSetBasicOp::newCoordiSet(&recdset);
    if (errcode != NO_ERROR) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }
     
    // 在 Device 端创建测试边缘的坐标集合。
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(tecdset,count);
    if (errcode != NO_ERROR) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }

    // 在 Device 端创建匹配参考边缘的坐标集合。
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(recdset,count);
    if (errcode != NO_ERROR) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。   
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y; 
    gridsize.x = (teimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (teimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;
    gridsize.z = 1;
    
    // 将测试图像转化为坐标集。
    _imgConvertCstKer<<<gridsize, blocksize>>>(teimgCud, *tecdset, 
                                               this->highPixel, 
                                               tecountdev);
    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return CUDA_ERROR;
    }
    
    // 将匹配参考图像转化为坐标集。                                             
    _imgConvertCstKer<<<gridsize, blocksize>>>(reimgCud[maxindex], *recdset,
                                               this->highPixel, 
                                               recountdev);
    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return CUDA_ERROR;
    }
                                                                                    
    // 将 Device 上的 recountdev 拷贝到 Host 上。
    int recount; 
    errcode = cudaMemcpy(&recount,recountdev,
                         sizeof (int), cudaMemcpyDeviceToHost);
    // 判断是否拷贝成功，若失败，释放之前的空间，防止内存泄漏，然后返回错误。
    if (errcode != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }  

    // 将 Device 上的 tecountdev 拷贝到 Host 上。
    int tecount;
    errcode = cudaMemcpy(&tecount,tecountdev,
                         sizeof (int), cudaMemcpyDeviceToHost);
    // 判断是否拷贝成功，若失败，释放之前的空间，防止内存泄漏，然后返回错误。
    if (errcode != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }
    
    // 测试边缘的每个点与参考边缘的 group 个点计算欧式距离。
    int group = (recount + DEF_COL_MAX - 1) / DEF_COL_MAX;
   
    // 计算欧式距离矩阵的宽度和高度。
    int matwidth = (recount + group - 1) / group;
    int matheight = tecount;
   
    // 为 alldevpointermat 分配空间。
    cudaerrcode = cudaMalloc((void **)&alldevpointermat, 
                              (recount * tecount + 
                               2 * matheight * matwidth) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return cudaerrcode;
    }

    // 初始化所有 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldevpointermat, 0,
                             (recount * tecount + 2 * matheight * matwidth) * 
                             sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return cudaerrcode;
    }  
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize1D = matwidth;
    gridsize1D = matheight;

    // 获得欧式距离矩阵和索引矩阵位置指针。
    deveudmat = alldevpointermat;
    devindexmat = deveudmat + matheight * matwidth;

    // 调用核函数，计算参考边缘和测试边缘的所有点的欧式距离的矩阵。
    _euclidMatKer<<<gridsize1D, blocksize1D>>>(*recdset, *tecdset, 
                                               group, deveudmat, devindexmat);

    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return CUDA_ERROR;
    }
    
    // 获得行最小值数组指针。
    devrowmin = (int *)(devindexmat + matheight * matwidth);

    // 获得最小 2 的幂次数，使得排序的长度满足 2 的幂次方。
    int exponent = (int)ceil(log((float)matwidth) / log(2.0f));
    int length = (1 << exponent);
   
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize1D = length;
    gridsize1D = matheight;

    // 调用核函数，计算行最小值，即找到点对应关系。
    _findRowMinKer<<<gridsize1D, blocksize1D, 2 * length * sizeof (float)>>>(
            deveudmat, devindexmat, matwidth, length, 
            maxDisPoint * maxDisPoint, devrowmin);
            
    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return CUDA_ERROR;
    }

    // 申请中间错误码图像。
    errcode = ImageBasicOp::newImage(&errmap);
    if (errcode != NO_ERROR) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }
    // 大小和输入的 errpoint 一致。
    errcode = ImageBasicOp::makeAtCurrentDevice(errmap, errpoint->width, 
                                                errpoint->height);
    if (errcode != NO_ERROR) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }

    // 提取错误码子图像。
    ImageCuda errmapcud;
    errcode = ImageBasicOp::roiSubImage(errmap, &errmapcud);
    if (errcode != NO_ERROR) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize1D = DEF_BLOCK_1D;
    gridsize1D = (tecount + DEF_BLOCK_1D - 1) / DEF_BLOCK_1D;

    // 调用核函数，将错误码标记输出到 errmap 图像中。
    _makeErrmapKer<<<gridsize1D, blocksize1D>>>(*tecdset, devrowmin, errmapcud);

    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return CUDA_ERROR;
    }

    // 将错误码图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(errpoint);
    if (errcode != NO_ERROR) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }

    // 提取错误码子图像。
    ImageCuda errpointcud;
    errcode = ImageBasicOp::roiSubImage(errpoint, &errpointcud);
    if (errcode != NO_ERROR) {
        FAIL_EDGECHECK_SPACE_FREE;
        return errcode;
    }

    // 计算核函数调用的分块大小。
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = 1;
    gridsize.x = (errmapcud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (errmapcud.imgMeta.height + blocksize.y - 1) / blocksize.y;
    gridsize.z = 1;

    // 调用核函数，根据错误码图像确定最终的异常点。
    _confirmErrmapKer<<<gridsize, blocksize>>>(errmapcud, errpointcud);

    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_EDGECHECK_SPACE_FREE;
        return CUDA_ERROR;
    }

    // 释放 Device 端内存。
    FAIL_EDGECHECK_SPACE_FREE;
    return NO_ERROR;    
}

// 取消前面的宏定义。
#undef FAIL_EDGECHECK_SPACE_FREE

// Host 成员方法：edgeCheckFragment（边缘的异常片段检查）
__host__ int EdgeCheck::edgeCheckFragment(CoordiSet *recdset, 
                                          CoordiSet *tecdset,
                                          Image *errpoint)
{
    // 检查输入坐标集合和图像是否为 NULL，如果为 NULL 直接报错返回。
    if (recdset == NULL || tecdset == NULL || errpoint == NULL)
        return NULL_POINTER;

    // 局部变量，错误码
    int errcode;

    // 将参考边缘的坐标集拷贝到 Device 内存中。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(recdset);
    if (errcode != NO_ERROR)
        return errcode;

    // 将测试边缘的坐标集拷贝到 Device 内存中。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(tecdset);
    if (errcode != NO_ERROR)
        return errcode;

    // 获取参考边缘和测试边缘点的个数。
    int recount = recdset->count;
    int tecount = tecdset->count;

    // 测试边缘的每个点与参考边缘的 group 个点计算欧式距离。
    int group = (recount + DEF_COL_MAX - 1) / DEF_COL_MAX;

    // 计算欧式距离矩阵的宽度和高度。
    int matwidth = (recount + group - 1) / group;
    int matheight = tecount;
    
    // 一次申请 Device 端所有空间。
    cudaError_t cudaerrcode;
    float *alldevicepointer, *devremat, *devtemat, *deveudmat, *devindexmat;
    int *devrowmin;
    cudaerrcode = cudaMalloc((void **)&alldevicepointer, 
                             (recount * DEF_HUMOM_SIZE + 
                             tecount * DEF_HUMOM_SIZE + 2 * matheight * 
                             matwidth + matheight) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }
  
    // 初始化所有 Device 上的内存空间。
    cudaerrcode = cudaMemset(alldevicepointer, 0,
                             (recount * DEF_HUMOM_SIZE + 
                              tecount * DEF_HUMOM_SIZE + 2 * matheight * 
                              matwidth + matheight) * sizeof (float));
    if (cudaerrcode != cudaSuccess) {
        cudaFree(alldevicepointer);
        return cudaerrcode;
    }

    // 获得参考边缘的 local moments 特征矩阵 devremat。
    devremat = alldevicepointer;
               
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    size_t blocksize1D, gridsize1D;
    blocksize1D = DEF_BLOCK_1D;
    gridsize1D = (recount + blocksize1D - 1) / blocksize1D;

    // 调用核函数，计算参考边缘的 local moments。
    _localMomentsKer<<<gridsize1D, blocksize1D>>>(*recdset, this->followWidth, 
                                              devremat);
  
    // 若调用核函数出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 获得测试边缘的 local moments 特征矩阵 devremat。
    devtemat = devremat + recount * DEF_HUMOM_SIZE;
               
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize1D = DEF_BLOCK_1D;
    gridsize1D = (tecount + blocksize1D - 1) / blocksize1D;

    // 调用核函数，计算测试边缘的 local moments。
    _localMomentsKer<<<gridsize1D, blocksize1D>>>(*tecdset, this->followWidth, 
                                              devtemat);

    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize1D = matwidth;
    gridsize1D = matheight;

    // 获得欧式距离矩阵和索引矩阵位置指针。
    deveudmat = devtemat + tecount * DEF_HUMOM_SIZE;
    devindexmat = deveudmat + matheight * matwidth;

    // 调用核函数，计算参考边缘和测试边缘的所有点的欧式距离的矩阵。
    _euclidMatKer<<<gridsize1D, blocksize1D>>>(*recdset, *tecdset, 
                                               group, deveudmat, devindexmat);

    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 获得行最小值数组指针。
    devrowmin = (int *)(devindexmat + matheight * matwidth);

    // 获得最小 2 的幂次数，使得排序的长度满足 2 的幂次方。
    int exponent = (int)ceil(log((float)matwidth) / log(2.0f));
    int length = (1 << exponent);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize1D = length;
    gridsize1D = matheight;
 
    // 调用核函数，计算行最小值，即找到点对应关系。
    _findRowMinKer<<<gridsize1D, blocksize1D, 2 * length * sizeof (float)>>>(
            deveudmat, devindexmat, matwidth, length, 
            maxDis * maxDis, devrowmin);

    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }
           
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize1D = DEF_BLOCK_1D;
    gridsize1D = (tecount + blocksize1D - 1) / blocksize1D;

    // 调用核函数，计算对应点间的特征向量的标准相关系数。
    _relateCoeffKer<<<gridsize1D, blocksize1D>>>(devtemat, devremat,
                                                 this->minCor, devrowmin);

    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        return CUDA_ERROR;
    }

    // 申请中间错误码图像。
    Image *errmap;
    errcode = ImageBasicOp::newImage(&errmap);
    if (errcode != NO_ERROR) {
        cudaFree(alldevicepointer);
        return errcode;
    }
    // 大小和输入的 errpoint 一致。
    errcode = ImageBasicOp::makeAtCurrentDevice(errmap, errpoint->width, 
                                                errpoint->height);
    if (errcode != NO_ERROR) {
        cudaFree(alldevicepointer);
        ImageBasicOp::deleteImage(errmap);
        return errcode;
    }

    // 提取错误码子图像。
    ImageCuda errmapcud;
    errcode = ImageBasicOp::roiSubImage(errmap, &errmapcud);
    if (errcode != NO_ERROR) {
        cudaFree(alldevicepointer);
        ImageBasicOp::deleteImage(errmap);
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    blocksize1D = DEF_BLOCK_1D;
    gridsize1D = (tecount + blocksize1D - 1) / blocksize1D;

    // 调用核函数，将错误码标记输出到 errmap 图像中。
    _makeErrmapKer<<<gridsize1D, blocksize1D>>>(*tecdset, devrowmin, errmapcud);

    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        ImageBasicOp::deleteImage(errmap);
        return CUDA_ERROR;
    }

    // 将输出的异常点图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(errpoint);
    if (errcode != NO_ERROR) {
        cudaFree(alldevicepointer);
        ImageBasicOp::deleteImage(errmap);
        return errcode;
    }

    // 提取错误码子图像。
    ImageCuda errpointcud;
    errcode = ImageBasicOp::roiSubImage(errpoint, &errpointcud);
    if (errcode != NO_ERROR) {
        cudaFree(alldevicepointer);
        ImageBasicOp::deleteImage(errmap);
        return errcode;
    }

    // 计算核函数调用的分块大小。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (errmapcud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (errmapcud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 调用核函数，根据错误码图像确定最终的异常点。
    _confirmErrmapKer<<<gridsize, blocksize>>>(errmapcud, errpointcud);

    // 若调用核函数出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(alldevicepointer);
        ImageBasicOp::deleteImage(errmap);
        return CUDA_ERROR;
    }

    // 释放中间图像。
    errcode = ImageBasicOp::deleteImage(errmap);
    if (errcode != NO_ERROR) {
        cudaFree(alldevicepointer);
        return errcode;
    }

    // 释放 Device 端内存。
    cudaFree(alldevicepointer);
    
    return NO_ERROR;
}

