// SmoothVector.cu
// 实现特征向量平滑

#include "SmoothVector.h"

#include "ErrorCode.h" 


// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  256 
#define DEF_BLOCK_Y    1 


// 核函数： _initFeatureKer（初始化三个特征值图像）
// 该方法根据给定的 FeatureVecArray 在设备端初始化三个特征值图像
static __global__ void                       // Kernel 函数无返回值
_initFeatureKer(
        FeatureVecArray infeaturevecarray,   // 输入特征向量
        MatrixCuda cvmatrixcuda,             // CV 的暂存图像
        MatrixCuda sdmatrixcuda,             // SD 的暂存图像
        MatrixCuda ncmatrixcuda              // NC 的暂存图像
);

// 核函数： _meanShiftKer（平滑特征向量）
// 该方法以指定坐标集所规定的图像范围内的各 PIXEL 的初始特征向量为基础，进行
// meanshift 操作，每一个像素对应一个核函数，在核函数中针对该像素对应的特征值
// 进行，根据给定的操作参数进行若干次的迭代运算，得到一个收敛特征向量
static __global__ void                       // Kernel 函数无返回值
_meanShiftKer(        
        SmoothVector smoothvector,           // 平滑操作类
        FeatureVecArray infeaturevecarray,   // 输入特征向量
        FeatureVecArray outfeaturevecarray,  // 输出特征向量
        MatrixCuda cvmatrixcuda,             // CV 的暂存图像
        MatrixCuda sdmatrixcuda,             // SD 的暂存图像
        MatrixCuda ncmatrixcuda              // NC 的暂存图像
);

// 核函数： _initFeatureKer（初始化三个特征值图像）
static __global__ void _initFeatureKer(
        FeatureVecArray infeaturevecarray, MatrixCuda cvmatrixcuda,
        MatrixCuda sdmatrixcuda, MatrixCuda ncmatrixcuda)
{
    // 计算当前 Thread 所对应的坐标集中的点的位置
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果 index 超过了处理的点的个数不做处理直接返回
    if(index >= infeaturevecarray.count)
        return;

    // 取出当前的像素对应特征向量的 X, Y 坐标
    int x = infeaturevecarray.x[index];
    int y = infeaturevecarray.y[index];

    // 使用记号变量便于代码书写
    int pitch = cvmatrixcuda.pitchWords;

    // 将当前点的特征值赋给三个特征值暂存向量中
    cvmatrixcuda.matMeta.matData[x + y * pitch] = 
            infeaturevecarray.CV[index];
    sdmatrixcuda.matMeta.matData[x + y * pitch] = 
            infeaturevecarray.SD[index];
    ncmatrixcuda.matMeta.matData[x + y * pitch] = 
            infeaturevecarray.NC[index];
}


// 核函数： _meanShiftKer（平滑特征向量）
static __global__ void _meanShiftKer(
        SmoothVector smoothvector, FeatureVecArray infeaturevecarray, 
        FeatureVecArray outfeaturevecarray, MatrixCuda cvmatrixcuda, 
        MatrixCuda sdmatrixcuda, MatrixCuda ncmatrixcuda)
{

    // 计算当前 Thread 所对应的坐标集中的点的位置
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果 index 超过了处理的点的个数不做处理直接返回
    if(index >= infeaturevecarray.count)
        return;
        
    // 获取 matrix 的 width 和 height
    int width = cvmatrixcuda.matMeta.width;
    int height = cvmatrixcuda.matMeta.height;

    // 取出当前的像素对应特征向量的 X, Y 坐标
    int x = infeaturevecarray.x[index];
    int y = infeaturevecarray.y[index];

    // 获取当前 matrixcuda 的 pitchWords 方便后面定位 pixel
    int pitch = cvmatrixcuda.pitchWords;
    
    // 获取当前操作类 smoothvector 中的 relativeWeight 参数
    float weight = smoothvector.getRelativeWeight();
   
    // 使用三个 float 型的临时变量将运算的中间结果暂存
    float cv, sd, nc; 
        
    for (int i = 0; i < smoothvector.getShiftArraySize(); i++) {
        // 获取当前操作类 smoothvector 中三个 操作数组中的值
        int count = smoothvector.getShiftCounts()[i];
        int space = smoothvector.getSpaceBands()[i];  
        float range = smoothvector.getRangeBands()[i];       
        
        // 为暂存变量赋值
        cv = cvmatrixcuda.matMeta.matData[x + y * pitch];
        sd = sdmatrixcuda.matMeta.matData[x + y * pitch];
        nc = ncmatrixcuda.matMeta.matData[x + y * pitch]; 
 
        // 每次迭代 ShiftCounts 次
        for (int j = 0; j < count; j++) {
            // 定义统计求和存储变量
            float xsum = 0.0f;   // 五维向量 A 中 Y 坐标的迭代结果
            float ysum = 0.0f;   // 五维向量 A 中 X 坐标的迭代结果
            float cvsum = 0.0f;  // 五维向量 A 中特征值 CV 的迭代结果
            float sdsum = 0.0f;  // 五维向量 A 中特征值 SD 的迭代结果
            float ncsum = 0.0f;  // 五维向量 A 中特征值 NC 的迭代结果
            float bjsum = 0.0f;  // 标量 B 的迭代结果
 
            // 每次迭代中要遍历当前点的 space 邻域
            for (int k = y - space; k <= y + space; k++) {
                for (int l = x - space; l <= x + space; l++) {                
                    // 使用记号变量便于代码书写
                    float cvtmp = 
                            cvmatrixcuda.matMeta.matData[l + k * pitch];
                    float sdtmp = 
                            sdmatrixcuda.matMeta.matData[l + k * pitch];
                    float nctmp = 
                            ncmatrixcuda.matMeta.matData[l + k * pitch];
                            
                    // 根据算法文档，计算邻域向量和当前对应向量的欧式距离
                    // 由于在计算向量 A 和标量 B 的时候都需要用到这个欧式距离
                    // 故提取公共运算部分，减少重复计算
                    float tmp = expf(((x - l) * (x - l) + (y - k) * (y - k)) / 
                                     (space * space) * (-weight) - 
                                     ((cv - cvtmp) * (cv - cvtmp) + 
                                      (sd - sdtmp) * (sd - sdtmp) + 
                                      (sd - sdtmp) * (sd - sdtmp)) / 
                                     (range * range) * (1 - weight)); 
                               
                    // 计算五维向量 Aj
                    xsum += l * tmp;
                    ysum += k * tmp;
                    cvsum += cvtmp * tmp;
                    sdsum += sdtmp * tmp;
                    ncsum += nctmp * tmp;
 
                    // 计算 Bj
                    bjsum += tmp;  
                }
            }
            // 根据公式处理 Bj
            bjsum += 0.001f;
            
            // 根据公式处理 Aj
            xsum /= bjsum;
            ysum /= bjsum;
            cvsum /= bjsum;
            sdsum /= bjsum;
            ncsum /= bjsum;
            
            // 更新 feature 
            x = (int) xsum;
            y = (int) ysum;
            cv = cvsum;
            sd = sdsum;
            nc = ncsum;
            
            // 如果超过边界则自动取边界上的值
            if (x < space)
                x = space;
            else if (x >= width - space)
                x = width - space - 1;
            if (y < space)
                y = space;            
            else if (y >= height - space)
                y = height - space - 1;
        }
    }
    
    // 把运算结果赋值给 outfeaturevecarray 
    outfeaturevecarray.x[index] = x;
    outfeaturevecarray.y[index] = y;
    outfeaturevecarray.CV[index] = cv;
    outfeaturevecarray.SD[index] = sd;
    outfeaturevecarray.NC[index] =  nc;
}

// Host 成员方法： meanshift（均值偏移）
__host__ int SmoothVector::meanshift(FeatureVecArray *infeaturevecarray,
        FeatureVecArray *outfeaturevecarray, int width, int height)
{
    // 检查输入参数是否为 NULL， 如果为 NULL 直接报错返回
    if (infeaturevecarray == NULL || outfeaturevecarray == NULL) 
        return NULL_POINTER;

    // 检查输入参数是否合法，如果不合法直接报错返回
    if (width < 0 || height < 0) 
        return INVALID_DATA;

    // 检查操作类参数是否有正确的值，如果没有直接返回错误
    if (this->shiftArraySize == 0 || this->spaceBands == NULL || 
        this->rangeBands == NULL || this->shiftCounts == NULL)
        return INVALID_DATA;
   
    int errcode;                                // 局部变量，错误码
    Matrix *cvmatrix, *sdmatrix, *ncmatrix;     // 特征值矩阵

    // 创建特征值 matrix 指针
    errcode = MatrixBasicOp::newMatrix(&cvmatrix);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    errcode = MatrixBasicOp::newMatrix(&sdmatrix);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    errcode = MatrixBasicOp::newMatrix(&ncmatrix);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    // 在设备端申请 matrix 空间
    errcode = MatrixBasicOp::makeAtCurrentDevice(cvmatrix, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    errcode = MatrixBasicOp::makeAtCurrentDevice(sdmatrix, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;
    errcode = MatrixBasicOp::makeAtCurrentDevice(ncmatrix, width, height);
    if (errcode != NO_ERROR)
        return CUDA_ERROR;

    // 创建 MatrixCuda 指针
    MatrixCuda *cvmatrixcuda, *sdmatrixcuda, *ncmatrixcuda;     // 特征值
                                                                // 设备端矩阵
    // 通过预定义的宏将 Matrix 指针转化为 MatrixCuda 类型的指针
    cvmatrixcuda = MATRIX_CUDA(cvmatrix);
    sdmatrixcuda = MATRIX_CUDA(sdmatrix);
    ncmatrixcuda = MATRIX_CUDA(ncmatrix);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = infeaturevecarray->count / blocksize.x + 1;
    gridsize.y = 1;

    // 调用核函数，初始化三个特征值图像
    _initFeatureKer<<<gridsize, blocksize>>>(
            *infeaturevecarray, *cvmatrixcuda, *sdmatrixcuda, *ncmatrixcuda);

    // 调用核函数，进行 meanshift
    _meanShiftKer<<<gridsize, blocksize>>>(
            *this, *infeaturevecarray, *outfeaturevecarray, 
            *cvmatrixcuda, *sdmatrixcuda, *ncmatrixcuda);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 内存清理
    MatrixBasicOp::deleteMatrix(cvmatrix);
    MatrixBasicOp::deleteMatrix(sdmatrix);
    MatrixBasicOp::deleteMatrix(ncmatrix);
    
    return NO_ERROR;
}
