// Segmentation.cu
// 实现二分类

#include "Segmentation.h"


// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  256 
#define DEF_BLOCK_Y    1 


// 核函数：_countW1Ker（统计各个向量 W1 近旁的向量个数）
// 该方法首先计算当前向量与其他各个向量之间的三个距离度量（坐标的欧式距离，
// 特征值的欧式距离以及向量的夹角），如果在 W1 范围内，则将当前 index 的
// count 个数加一，最终得到统计结果
static __global__ void                       // Kernel 函数没有返回值
_countW1Ker(
        Segmentation segmentation,           // 分割操作类
        FeatureVecArray infeaturevecarray,   // 输入特征向量
        int *w1counts                        // 记录 W1 范围内的向量个数     
);

// 核函数：_labelVectorsKer（标记各个向量的分类）
// 该函数根据传入的已标记的向量的 index ，比较各个向量如他们的距离如果在
// 指定的 W2 范围内，则将该向量标记为同一类的。在启动这个核函数的时候
// 横向的坐标表示各个待标记的向量，纵向的坐标表示各个已经标记的向量
static __global__ void                       // Kernel 函数没有返回值
_labelVectorsKer(
        Segmentation segmentation,           // 分割操作类
        FeatureVecArray infeaturevecarray,   // 输入特征向量
        unsigned char *tmpbl,                // 需要标记的值
        int *tmpvecs,                        // 当前已经标记的向量的 index 数组
        int tmpsize                          // 当前已经标记的向量的数组的大小
);

// 核函数：_countAppointW1Ker（统计指定向量 W1 近旁的向量个数）
// 该函数根据传入的标记值数组，选择性地计算未标记的向量和其它未标记向量之间的
// 三个距离度量（坐标的欧式距离，特征值的欧式距离以及向量夹角），如果在 W1 
// 范围内，则将当前 index 的 cout 个数加一，最终得到统计结果
static __global__ void                       // Kernel 函数没有返回值
_countAppointW1Ker(
        Segmentation segmentation,           // 分割操作类
        FeatureVecArray infeaturevecarray,   // 输入特征向量
        unsigned char *tmplbl,               // 临时标记数组
        int *w1counts                        // 记录 W1 范围内的向量个数     
);

// 核函数：_segregateKer（对向量进行最终的分割）
// 该核函数，根据之前初步的分类结果，对每一个向量，统计其 W2 范围内的
// 类别1和类别2的向量的个数，根据二者个数来判定当前向量最终被划分到哪个类别中
static __global__ void                      // Kernel 函数没有返回值
_segregateKer(
        Segmentation segmentation,          // 分割操作类
        FeatureVecArray infeaturevecarray,  // 输入特征向量
        unsigned char *tmp1lbl,             // 类别1的临时标记数组
        unsigned char *tmp2lbl,             // 类别2的临时标记数组
        int *outlabel                       // 用于输出的分类结果数组
);

// 核函数：_countW1Ker（统计各个向量 W1 近旁的向量个数）
static __global__ void _countW1Ker(
        Segmentation segmentation, FeatureVecArray infeaturevecarray,
        int *w1counts)
{
    // 计算当前 Thread 所对应的坐标集中的点的位置
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果 index 超过了处理的点的个数不做处理直接返回
    if(index >= infeaturevecarray.count)
        return;
  
    int x1 = infeaturevecarray.x[index];      // 当前处理向量的横坐标
    int y1 = infeaturevecarray.y[index];      // 当前处理向量的纵坐标
    float cv1 = infeaturevecarray.CV[index];  // 当前处理向量的 CV
    float sd1 = infeaturevecarray.SD[index];  // 当前处理向量的 SD
    float nc1 = infeaturevecarray.NC[index];  // 当前处理向量的 NC
    int count = 0;                            // 当前向量 W1 范围内的向量个数 

    // 统计当前向量 W1 范围内的向量个数
    for (int i = 0; i < infeaturevecarray.count; i++) {
        int x2 = infeaturevecarray.x[i];      // 当前比较向量的横坐标
        int y2 = infeaturevecarray.y[i];      // 当前比较向量的纵坐标
        float cv2 = infeaturevecarray.CV[i];  // 当前比较向量的 CV
        float sd2 = infeaturevecarray.SD[i];  // 当前比较向量的 SD
        float nc2 = infeaturevecarray.NC[i];  // 当前比较向量的 NC  

        // 计算两个向量之间坐标的欧式距离
        float d1 = sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

        // 计算两个向量之间特征值的欧式距离
        float d2 = sqrt((cv2 - cv1) * (cv2 - cv1) + (sd2 -sd1) * (sd2 -sd1) + 
                        (nc2 - nc1) * (nc2 - nc1));

        // 计算两个向量之间夹角的 cos 值
        float d3 = (x1 * x2 + y1 * y2 + cv1 * cv2 + sd1 *  sd2 + nc1 * nc2) / 
                   sqrt(x1 * x1 + y1 * y1 + cv1 * cv1 + sd1 * sd1 + nc1 * nc1) / 
                   sqrt(x2 * x2 + y2 * y2 + cv2 * cv2 + sd2 * sd2 + nc2 * nc2);

        // 判断两个向量之间的距离是否在 W1 范围内
        if (d1 < segmentation.getBw1().spaceWidth && 
            d2 < segmentation.getBw1().rangeWidth &&
            d3 < segmentation.getBw1().angleWidth) 
            count++;   
    }
    
    // 将统计值写入统计值数组
    w1counts[index] = count;
}

// 核函数：_labelVectorsKer（标记各个向量的分类）
static __global__ void _labelVectorsKer(
        Segmentation segmentation, FeatureVecArray infeaturevecarray, 
        unsigned char *tmplbl, int *tmpvecs, int tmpsize) 
{
    // 计算当前 Thread 所对应的坐标集中的点的位置
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果 index 超过了处理的点的个数不做处理直接返回
    if(index >= infeaturevecarray.count)
        return;

    // 计算当前需要寻找的已标记向量的 index
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    // 如果纵向坐标超过了 tmpsize则不做处理直接返回
    if (index_y >= tmpsize)
        return;

    // 获取当前种子向量的 index
    int seedindex = tmpvecs[index_y];

    int x1 = infeaturevecarray.x[index];          // 当前处理向量的横坐标
    int y1 = infeaturevecarray.y[index];          // 当前处理向量的纵坐标
    float cv1 = infeaturevecarray.CV[index];      // 当前处理向量的 CV
    float sd1 = infeaturevecarray.SD[index];      // 当前处理向量的 SD
    float nc1 = infeaturevecarray.NC[index];      // 当前处理向量的 NC

    int x2 = infeaturevecarray.x[seedindex];      // 当前种子向量的横坐标
    int y2 = infeaturevecarray.y[seedindex];      // 当前种子向量的纵坐标
    float cv2 = infeaturevecarray.CV[seedindex];  // 当前种子向量的 CV
    float sd2 = infeaturevecarray.SD[seedindex];  // 当前种子向量的 SD
    float nc2 = infeaturevecarray.NC[seedindex];  // 当前种子向量的 NC

    // 计算两个向量之间坐标的欧式距离
    float d1 = sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

    // 计算两个向量之间特征值的欧式距离
    float d2 = sqrt((cv2 - cv1) * (cv2 - cv1) + (sd2 -sd1) * (sd2 -sd1) + 
                    (nc2 - nc1) * (nc2 - nc1));

    // 计算两个向量之间夹角的 cos 值
    float d3 = (x1 * x2 + y1 * y2 + cv1 * cv2 + sd1 *  sd2 + nc1 * nc2) / 
               sqrt(x1 * x1 + y1 * y1 + cv1 * cv1 + sd1 * sd1 + nc1 * nc1) / 
               sqrt(x2 * x2 + y2 * y2 + cv2 * cv2 + sd2 * sd2 + nc2 * nc2);

    // 判断两个向量之间的距离是否在 W2 范围内，如果是则标记该向量
    if (d1 < segmentation.getBw2().spaceWidth && 
        d2 < segmentation.getBw2().rangeWidth &&
        d3 < segmentation.getBw2().angleWidth) 
        tmplbl[index] = 1;  
}

// 核函数：_countAppointW1Ker（统计指定向量 W1 近旁的向量个数）
static __global__ void _countAppointW1Ker(
        Segmentation segmentation, FeatureVecArray infeaturevecarray,  
        unsigned char *tmplbl, int *w1counts)
{
    // 计算当前 Thread 所对应的坐标集中的点的位置
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果 index 超过了处理的点的个数不做处理直接返回
    if(index >= infeaturevecarray.count)
        return;
   
    // 将每个向量近旁个数初始化为0
    w1counts[index] = 0;

    // 如果当前 index 已经被标记，不做处理直接返回
    if (tmplbl[index] == 1)
        return;
    
    int x1 = infeaturevecarray.x[index];      // 当前处理向量的横坐标
    int y1 = infeaturevecarray.y[index];      // 当前处理向量的纵坐标
    float cv1 = infeaturevecarray.CV[index];  // 当前处理向量的 CV
    float sd1 = infeaturevecarray.SD[index];  // 当前处理向量的 SD
    float nc1 = infeaturevecarray.NC[index];  // 当前处理向量的 NC
    int count = 0;                            // 当前向量 W1 范围内的向量个数 

    // 统计当前向量 W1 范围内的向量个数
    for (int i = 0; i < infeaturevecarray.count; i++) {
        // 忽略已经被标记的向量
        if (tmplbl[i] == 1) 
            continue;

        int x2 = infeaturevecarray.x[i];      // 当前比较向量的横坐标
        int y2 = infeaturevecarray.y[i];      // 当前比较向量的纵坐标
        float cv2 = infeaturevecarray.CV[i];  // 当前比较向量的 CV
        float sd2 = infeaturevecarray.SD[i];  // 当前比较向量的 SD
        float nc2 = infeaturevecarray.NC[i];  // 当前比较向量的 NC  

        // 计算两个向量之间坐标的欧式距离
        float d1 = sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

        // 计算两个向量之间特征值的欧式距离
        float d2 = sqrt((cv2 - cv1) * (cv2 - cv1) + (sd2 -sd1) * (sd2 -sd1) + 
                        (nc2 - nc1) * (nc2 - nc1));

        // 计算两个向量之间夹角的 cos 值
        float d3 = (x1 * x2 + y1 * y2 + cv1 * cv2 + sd1 *  sd2 + nc1 * nc2) / 
                   sqrt(x1 * x1 + y1 * y1 + cv1 * cv1 + sd1 * sd1 + nc1 * nc1) / 
                   sqrt(x2 * x2 + y2 * y2 + cv2 * cv2 + sd2 * sd2 + nc2 * nc2);

        // 判断两个向量之间的距离是否在 W1 范围内
        if (d1 < segmentation.getBw1().spaceWidth && 
            d2 < segmentation.getBw1().rangeWidth &&
            d3 < segmentation.getBw1().angleWidth) 
            count++;   
    }
    
    // 将统计值写入统计值数组
    w1counts[index] = count;
}

// 核函数：_segregateKer（对向量进行最终的分割）
static __global__ void _segregateKer(
        Segmentation segmentation, FeatureVecArray infeaturevecarray,  
        unsigned char *tmp1lbl, unsigned char *tmp2lbl, int *outlabel)
{
    // 计算当前 Thread 所对应的坐标集中的点的位置
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果 index 超过了处理的点的个数不做处理直接返回
    if(index >= infeaturevecarray.count)
        return;
   
    // 将每个向量近旁个数初始化为0
    outlabel[index] = 0;

    int x1 = infeaturevecarray.x[index];      // 当前处理向量的横坐标
    int y1 = infeaturevecarray.y[index];      // 当前处理向量的纵坐标
    float cv1 = infeaturevecarray.CV[index];  // 当前处理向量的 CV
    float sd1 = infeaturevecarray.SD[index];  // 当前处理向量的 SD
    float nc1 = infeaturevecarray.NC[index];  // 当前处理向量的 NC
    long sprtcount1 = 0;                      // W2 范围内暂定1类别的向量个数
    long sprtcount2 = 0;                      // W2 范围内暂定2类别的向量个数
    float distsum1 = 0.0;                     // W2 范围内当前向量和暂定1类别
                                              // 的平方差之和
    float distsum2 = 0.0;                     // W2 范围内当前向量和暂定2类别
                                              // 的平方差之和

    // 统计当前向量 W2 范围内的属于类别1和类别2的向量个数
    for (int i = 0; i < infeaturevecarray.count; i++) {
        
        int x2 = infeaturevecarray.x[i];      // 当前比较向量的横坐标
        int y2 = infeaturevecarray.y[i];      // 当前比较向量的纵坐标
        float cv2 = infeaturevecarray.CV[i];  // 当前比较向量的 CV
        float sd2 = infeaturevecarray.SD[i];  // 当前比较向量的 SD
        float nc2 = infeaturevecarray.NC[i];  // 当前比较向量的 NC  

        // 计算两个向量之间坐标的欧式距离
        float d1 = sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

        // 计算两个向量之间特征值的欧式距离
        float d2 = sqrt((cv2 - cv1) * (cv2 - cv1) + (sd2 -sd1) * (sd2 -sd1) + 
                        (nc2 - nc1) * (nc2 - nc1));

        // 计算两个向量之间夹角的 cos 值
        float d3 = (x1 * x2 + y1 * y2 + cv1 * cv2 + sd1 *  sd2 + nc1 * nc2) / 
                   sqrt(x1 * x1 + y1 * y1 + cv1 * cv1 + sd1 * sd1 + nc1 * nc1) / 
                   sqrt(x2 * x2 + y2 * y2 + cv2 * cv2 + sd2 * sd2 + nc2 * nc2);

        // 判断两个向量之间的距离是否在 W2 范围内
        if (d1 < segmentation.getBw2().spaceWidth && 
            d2 < segmentation.getBw2().rangeWidth &&
            d3 < segmentation.getBw2().angleWidth) {
            if (tmp1lbl[i] == 1) {
                sprtcount1++;
                distsum1 += d2 * d2;
            }

            if (tmp2lbl[i] == 1) {
                sprtcount2++;
                distsum2 += d2 * d2;
            }
        }
    }

    // 如果当前向量周围只有类别1的向量，则当前向量最终标记为1
    if (sprtcount1 > 0 && sprtcount2 == 0) {
        outlabel[index] = 1;
        return;
    }

    // 如果当前向量周围只有类别1的向量，则当前向量最终标记为1
    if (sprtcount2 > 0 && sprtcount1 == 0) {
        outlabel[index] = 2;
        return;
    }
    
    // 当前向量周围类别1相对于类别2占优势地位，则当前向量最终标记为1
    if (sprtcount1 > segmentation.getBeta() * sprtcount2) {
        outlabel[index] = 1;
        return;
    }

    // 当前向量周围类别2相对类别1占优势地位，则当前向量最终标记为2
    if (sprtcount2 > segmentation.getBeta() * sprtcount1) {
        outlabel[index] = 2;
        return;
    }

    // 标记特征值平方差的值
    if (distsum1 / powf(sprtcount1, segmentation.getAlpha()) <=
        distsum2 / powf(sprtcount2, segmentation.getAlpha()))
        outlabel[index] = 1;
    else
        outlabel[index] = 2;
}


// 宏：FREE_LOCAL_MEMORY_SEGREGATE（清理局部申请的设备端或者主机端内存）
// 该宏用于清理在 segregate 过程中申请的设备端或者主机端内存空间
#define FREE_LOCAL_MEMORY_SEGREGATE do {  \
    if ((w1counts) != NULL)               \
        delete [] (w1counts);             \
    if ((w1countsdev) != NULL)            \
        cudaFree((w1countsdev));          \
    if ((tmp1vecs) != NULL)               \
        delete [] (tmp1vecs);             \
    if ((tmp2vecs) != NULL)               \
        delete [] (tmp2vecs);             \
    if ((tmp1vecsdev) != NULL)            \
        cudaFree((tmp1vecsdev));          \
    if ((tmp2vecsdev) != NULL)            \
        cudaFree((tmp2vecsdev));          \
    if ((tmp1lbl) != NULL)                \
        delete [] (tmp1lbl);              \
    if ((tmp1lbldev) != NULL)             \
        cudaFree((tmp1lbldev));           \
    if ((tmp2lbl) != NULL)                \
        delete [] (tmp2lbl);              \
    if ((tmp2lbldev) != NULL)             \
        cudaFree((tmp2lbldev));           \
    if ((outlabeldev) != NULL)            \
        cudaFree((outlabeldev));          \
}while (0)


// Host 成员方法：segregate（图像分割）
__host__ int Segmentation::segregate(FeatureVecArray *featurevecarray, 
        int *outlabel)
{
    // 检查输入的参数是否为 NULL ，如果为 NULL 直接报错返回
    if (featurevecarray == NULL || outlabel == NULL)
        return NULL_POINTER;

    // 检查输入的参数是否为合法，如果不合法直接报错返回
    if (featurevecarray->count <= 0) 
        return INVALID_DATA;

    int errcode;                         // 局部变量，错误码
    int *w1counts = NULL;                // 每个向量 W1 范围内的向量个数
    int *w1countsdev = NULL;             // w1counts 对应的设备端指针
    int count = featurevecarray->count;  // 向量的个数

    // 定义类别1和类别2的暂定数组，该数组存储暂时被标记为1和2的向量的 index
    int *tmp1vecs = NULL;     // 存储暂时被划分到类别1的向量的 index
    int *tmp2vecs = NULL;     // 存储暂时被划分到类别2的向量的 index
    int *tmp1vecsdev = NULL;  // 设备端存储暂时被划分到类别1的向量的 index
    int *tmp2vecsdev = NULL;  // 设备端存储暂时被划分到类别2的向量的 index

    // 定义暂定类别1和类别2的标记数组，该数组标记对应的 index 是否属于某类别
    unsigned char *tmp1lbl = NULL;     // 主机端类别1标记数组
    unsigned char *tmp1lbldev = NULL;  // 设备端类别1标记数组
    unsigned char *tmp2lbl = NULL;     // 主机端类别2标记数组
    unsigned char *tmp2lbldev = NULL;  // 设备端类别2标记数组

    // 定义设备端最终标记数组
    int *outlabeldev = NULL;  // 设备端最终标记数组

    // 在主机端申请 W1 统计数组空间
    w1counts = new int[count];
    if (w1counts == NULL) 
        return UNKNOW_ERROR;

    // 在设备端申请 W1 统计数组空间
    errcode = cudaMalloc((void **)&w1countsdev, count * sizeof(int));
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (count + blocksize.x - 1) / blocksize.x;
    gridsize.y = 1;

    // 调用核函数，统计各个向量 W1 范围内的向量个数
    _countW1Ker<<<gridsize, blocksize>>>(*this, *featurevecarray, 
                                         w1countsdev);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 将统计结果拷回到 Host 端
    errcode = cudaMemcpy(w1counts, w1countsdev, 
                          count * sizeof(int),
                          cudaMemcpyDeviceToHost);

    // 若拷贝失败则返回报错
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    int seed1index = 0;  // 种子点1对应的 index
    int maxcounts = 0;   // W1 距离内向量最多的向量所对应的个数

    // 遍历统计数组，找出最大统计值对应的 index 即为种子点1的 index
    for (int i = 0; i < count; i++) {
        if (w1counts[i] > maxcounts) {
            maxcounts = w1counts[i];
            seed1index = i;
        }            
    }
    
    int tmp1size = 0;      // 记录当前类别暂存数组中元素的个数，默认值为0
    int tmp2size = 0;      // 记录当前类别暂存数组中元素的个数，默认值为0

    // 为1类别的暂存数组申请 host 空间
    tmp1vecs = new int[count];
    if (tmp1vecs == NULL) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return UNKNOW_ERROR;
    }

    // 将种子点1加入到类别1的暂存数组中, 同时将 tmp1size 加1 
    tmp1vecs[tmp1size++] = seed1index;

    // 为1类别的暂存数组申请 device 空间
    errcode = cudaMalloc((void **)&tmp1vecsdev, sizeof(int) * count);
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 将主机端的暂存数组（有意义的部分）拷贝到 device 端
    errcode = cudaMemcpy(tmp1vecsdev, tmp1vecs, sizeof(int) * tmp1size,
                         cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 为1类别的标记数组申请 Host 空间
    tmp1lbl = new unsigned char[count];
    if (tmp1lbl == NULL) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return UNKNOW_ERROR;
    }

    // 初始化1类别标记数组的值    
    for (int i = 0; i < count; i++) {
        tmp1lbl[i] = 0;
    }
    tmp1lbl[seed1index] = 1;

    // 为1类别的标记数组申请 device 空间
    errcode = cudaMalloc((void **)&tmp1lbldev, sizeof(unsigned char) * count);
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 将 host 端的标记数组拷贝到 device 端
    errcode = cudaMemcpy(tmp1lbldev, tmp1lbl, sizeof(unsigned char) * count,
                         cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 循环标记暂时属于类别1的向量，直到暂时属于类别1的向量的个数不再增加
    while (1) {
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (count + blocksize.x - 1) / blocksize.x;
        gridsize.y = (tmp1size + blocksize.y - 1) / blocksize.y;

        // 调用核函数，标记类别1的向量
        _labelVectorsKer<<<gridsize,blocksize>>>(*this, *featurevecarray,
                                                 tmp1lbldev, tmp1vecsdev,
                                                 tmp1size);

        // 检查核函数调用是否出错
        errcode = cudaGetLastError();
        if (errcode != cudaSuccess) {
            // 释放申请的内存，防止内存泄漏
            FREE_LOCAL_MEMORY_SEGREGATE;
            return CUDA_ERROR;
        }
        
        // 拷贝标记值数组到 Host 端
        errcode = cudaMemcpy(tmp1lbl, tmp1lbldev, sizeof(unsigned char) * count,
                             cudaMemcpyDeviceToHost);
        if (errcode != cudaSuccess) {
            // 释放申请的内存，防止内存泄漏
            FREE_LOCAL_MEMORY_SEGREGATE;
            return CUDA_ERROR;
        }

        // 定义当前已经被标记的数组大小
        int tmp1sizenow = 0;

        // 遍历标记值数组，将已经标记的值添加到已标记数组中
        for (int i = 0; i < count; i++) {
            if (tmp1lbl[i] == 1) 
                tmp1vecs[tmp1sizenow++] = i;
        }

        // 如果两次的大小没有发生变化，则跳出循环
        if (tmp1sizenow == tmp1size) 
            break;

        // 将当前 size 赋给原 size
        tmp1size = tmp1sizenow;

        // 将已标记数组拷贝到 Device 端
        errcode = cudaMemcpy(tmp1vecsdev, tmp1vecs, tmp1size * sizeof(int),
                             cudaMemcpyHostToDevice);
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (count + blocksize.x - 1) / blocksize.x;
    gridsize.y = 1;

    // 调用核函数，寻找类别2的种子点，在不需要重新申请新的统计数组空间，
    // 可以直接使用上一步中的统计数组，覆盖掉其数据，因为其数据已经不需要了
    _countAppointW1Ker<<<gridsize, blocksize>>>(*this, *featurevecarray,
                                                tmp1lbldev, w1countsdev);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 将统计结果拷回到 Host 端
    errcode = cudaMemcpy(w1counts, w1countsdev, 
                          count * sizeof(int),
                          cudaMemcpyDeviceToHost);

    // 若拷贝失败则返回报错
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    int seed2index = 0;  // 种子点2对应的 index
    maxcounts = 0;       // W1 距离内向量最多的向量所对应的个数

    // 遍历统计数组，找出最大统计值对应的 index 即为种子点2的 index
    for (int i = 0; i < count; i++) {
        if (w1counts[i] > maxcounts) {
            maxcounts = w1counts[i];
            seed2index = i;
        }            
    }

    // 为2类别的暂存数组申请 host 空间
    tmp2vecs = new int[count];
    if (tmp2vecs == NULL) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return UNKNOW_ERROR;
    }

    // 将种子点2加入到类别2的暂存数组中, 同时将 tmp2size 加1 
    tmp2vecs[tmp2size++] = seed2index;

    // 为2类别的暂存数组申请 device 空间
    errcode = cudaMalloc((void **)&tmp2vecsdev, sizeof(int) * count);
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 将主机端的暂存数组（有意义的部分）拷贝到 device 端
    errcode = cudaMemcpy(tmp2vecsdev, tmp2vecs, sizeof(int) * tmp2size,
                         cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 为2类别的标记数组申请 Host 空间
    tmp2lbl = new unsigned char[count];
    if (tmp2lbl == NULL) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return UNKNOW_ERROR;
    }

    // 初始化2类别标记数组的值    
    for (int i = 0; i < count; i++) {
        tmp2lbl[i] = 0;
    }
    tmp2lbl[seed2index] = 1;

    // 为2类别的标记数组申请 device 空间
    errcode = cudaMalloc((void **)&tmp2lbldev, sizeof(unsigned char) * count);
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 将 host 端的标记数组拷贝到 device 端
    errcode = cudaMemcpy(tmp2lbldev, tmp2lbl, sizeof(unsigned char) * count,
                         cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 循环标记暂时属于类别2的向量，直到暂时属于类别2的向量的个数不再增加
    while (1) {
        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (count + blocksize.x - 1) / blocksize.x;
        gridsize.y = (tmp1size + blocksize.y - 1) / blocksize.y;

        // 调用核函数，标记类别2的向量
        _labelVectorsKer<<<gridsize, blocksize>>>(*this, *featurevecarray, 
                                                  tmp2lbldev, tmp2vecsdev,
                                                  tmp2size);

        // 检查核函数调用是否出错
        errcode = cudaGetLastError();
        if (errcode != cudaSuccess){
            // 释放申请的内存，防止内存泄漏
            FREE_LOCAL_MEMORY_SEGREGATE;
            return CUDA_ERROR;
        }
        
        // 拷贝标记值数组到 Host 端
        errcode = cudaMemcpy(tmp2lbl, tmp2lbldev, sizeof(unsigned char) * count,
                             cudaMemcpyDeviceToHost);
        if (errcode != cudaSuccess){
            // 释放申请的内存，防止内存泄漏
            FREE_LOCAL_MEMORY_SEGREGATE;
            return CUDA_ERROR;
        }

        // 定义当前已经被标记的数组大小
        int tmp2sizenow = 0;

        // 遍历标记值数组，将已经标记的值添加到已标记数组中
        for (int i = 0; i < count; i++) {
            if (tmp2lbl[i] == 1) 
                tmp2vecs[tmp2sizenow++] = i;
        }

        // 如果两次的大小没有发生变化，则跳出循环
        if (tmp2sizenow == tmp2size) 
            break;

        // 将当前 size 赋给原 size
        tmp2size = tmp2sizenow;

        // 将已标记数组拷贝到 Device 端
        errcode = cudaMemcpy(tmp2vecsdev, tmp2vecs, tmp2size * sizeof(int),
                             cudaMemcpyHostToDevice);
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (count + blocksize.x - 1) / blocksize.x;
    gridsize.y = 1;

    // 为设备端的最终标记数组申请空间
    errcode = cudaMalloc((void **)&outlabeldev, sizeof(int) * count);
    if (errcode != cudaSuccess) {
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }
    
    // 调用核函数，完成最终分类
    _segregateKer<<<gridsize, blocksize>>>(*this,  *featurevecarray,  
                                           tmp1lbldev, tmp2lbldev, outlabeldev);

    // 检查核函数调用是否出错
    errcode = cudaGetLastError();
    if (errcode != cudaSuccess){
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 将最终分类标记数组拷贝到 Host 端
    errcode = cudaMemcpy(outlabel, outlabeldev, sizeof(int) * count, 
                         cudaMemcpyDeviceToHost);
    if (errcode != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏
        FREE_LOCAL_MEMORY_SEGREGATE;
        return CUDA_ERROR;
    }

    // 清理内存空间，防止内存泄漏
    FREE_LOCAL_MEMORY_SEGREGATE;
    
    return NO_ERROR;
}
 