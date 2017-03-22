// GaussianSmoothxy.cu
// 实现对curve的高斯平滑

#include "GaussianSmoothxy.h"


// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
//#define DEF_BLOCK_Y   8

// 宏,定义了五个高斯平滑尺度对应的系数
// Gaussian3,5,7,9,11时分别为: 4,16,64,256,1024. 
#define GAUSS_THREE  4
#define GAUSS_FIVE   16  
#define GAUSS_SEVEN  64
#define GAUSS_NINE   256 
#define GAUSS_ELEVEN 1024

// 平滑窗口大小为7的核函数
static __global__ void 
gauss7SmoothXY(
    int n,              // 曲线上点的数量
    int* ringCordiXY,   // 辅助参数
    float* gSmCordiXY   // 平滑结果
);

// 平滑窗口大小为5的核函数                           
static __global__ void 
gauss5SmoothXY(
    int n,              // 曲线上点的数量
    int* ringCordiXY,   // 辅助参数
    float* gSmCordiXY   // 平滑结果
);
   
// 平滑窗口大小为9的核函数 
static __global__ void 
gauss9SmoothXY(
    int n,              // 曲线上点的数量
    int* ringCordiXY,   // 辅助参数
    float* gSmCordiXY   // 平滑结果
);

// 平滑窗口大小为11的核函数，
static __global__ void 
gauss11SmoothXY(
    int n,              // 曲线上点的数量
    int* ringCordiXY,   // 辅助参数
    float* gSmCordiXY   // 平滑结果
);   
                  
// 平滑窗口大小为3的核函数
static __global__ void 
gauss3SmoothXY(
    int n,              // 曲线上点的数量
    int* ringCordiXY,   // 辅助参数
    float* gSmCordiXY   // 平滑结果
);
 
// 平滑窗口大小为7的核函数的具体实现
static __global__ void gauss7SmoothXY(int n, int* ringCordiXY, 
                                      float* gSmCordiXY )
{
     // 计算当前线程下标
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查线程是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (t >= n)
        return;
    
    // 计算当前线程对应的数组下标   
    int i =  2 * (n + t);


    // 计算横纵坐标对应的数值
    int x = ringCordiXY[i - 6] + 6 * ringCordiXY[i - 4] +
            15 * ringCordiXY[i - 2] + 20 * ringCordiXY[i] +
            15 * ringCordiXY[i + 2] + 6 * ringCordiXY[i + 4] +
            ringCordiXY[i + 6];
    int y = ringCordiXY[i - 5] + 6 * ringCordiXY[i - 3] +
            15 * ringCordiXY[i - 1] + 20 * ringCordiXY[i + 1] +
            15 * ringCordiXY[i + 3] + 6 * ringCordiXY[i + 5] +
            ringCordiXY[i + 7];
    
    // 计算平滑后的横纵坐标，写入gSmCordiXY中
    gSmCordiXY[2 * t] = 1.0 * x / GAUSS_SEVEN;  
    gSmCordiXY[2 * t + 1] = 1.0 * y / GAUSS_SEVEN;

}

// 平滑窗口大小为5的核函数的具体实现
static __global__ void gauss5SmoothXY(int n, int* ringCordiXY, 
                                      float* gSmCordiXY)
{
    // 计算当前线程下标
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查线程是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (t >= n)
        return;
    
    // 计算当前线程对应的数组下标   
    int i =  2 * (n + t);


    // 计算横纵坐标对应的数值
    int x = ringCordiXY[i - 4] + 4 * ringCordiXY[i - 2] + 
            6 * ringCordiXY[i] + 4 * ringCordiXY[i + 2] + ringCordiXY[i + 4] ;
    int y = ringCordiXY[i - 3] + 4 * ringCordiXY[i - 1] +
            6 * ringCordiXY[i + 1] + 4 * ringCordiXY[i + 3] + 
            ringCordiXY[i + 5];
            
    // 计算平滑后的横纵坐标，写入gSmCordiXY中
    gSmCordiXY[2 * t] = 1.0 * x / GAUSS_FIVE;  
    gSmCordiXY[2 * t + 1] = 1.0 * y / GAUSS_FIVE;

}

// 平滑窗口大小为9的核函数的具体实现
static __global__ void gauss9SmoothXY(int n, int* ringCordiXY, 
                                      float* gSmCordiXY)
{
     // 计算当前线程下标
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查线程是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (t >= n)
        return;
    
    // 计算当前线程对应的数组下标   
    int i =  2 * (n + t);


    // 计算横纵坐标对应的数值
    int x = ringCordiXY[i - 8] + 8 * ringCordiXY[i - 6] + 
            28 * ringCordiXY[i - 4] + 56 * ringCordiXY[i - 2] + 
            70 * ringCordiXY[i] + 56 * ringCordiXY[i + 2] +
            28 * ringCordiXY[i + 4] + 8 * ringCordiXY[i + 6] + 
            ringCordiXY[i + 8];
    int y = ringCordiXY[i - 7] + 8 * ringCordiXY[i - 5] + 
            28 * ringCordiXY[i - 3] + 56 * ringCordiXY[i - 1] +
            70 * ringCordiXY[i + 1] + 56 * ringCordiXY[i + 1] +
            28 * ringCordiXY[i + 3] + 8 * ringCordiXY[i + 5] + 
            ringCordiXY[i + 7];
            
    // 计算平滑后的横纵坐标，写入gSmCordiXY中
    gSmCordiXY[2 * t] = 1.0 * x / GAUSS_NINE;  
    gSmCordiXY[2 * t + 1] = 1.0 * y / GAUSS_NINE;


}

//平滑窗口大小为11的核函数的具体实现
static __global__ void gauss11SmoothXY(int n, int* ringCordiXY, 
                                       float* gSmCordiXY)
{
    // 计算当前线程下标
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查线程是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (t >= n)
        return;
    
    // 计算当前线程对应的数组下标   
    int i =  2 * (n + t);

    // 计算横纵坐标对应的数值
    int x = ringCordiXY[i - 10]  + 10 * ringCordiXY[i - 8] + 
            45 * ringCordiXY[i - 6] + 120 * ringCordiXY[i - 4] + 
            210 * ringCordiXY[i - 2] + 252 * ringCordiXY[i] +
            210 * ringCordiXY[i + 2] + 120 * ringCordiXY[i + 4] +
            45 * ringCordiXY[i + 6] + 10 * ringCordiXY[i + 8] +
            ringCordiXY[i + 10];
    int y = ringCordiXY[i - 9] + 10 * ringCordiXY[i - 7] + 
            45 * ringCordiXY[i - 5] + 120 * ringCordiXY[i - 3] + 
            210 * ringCordiXY[i - 1] + 252 * ringCordiXY[i + 1] + 
            210 * ringCordiXY[i + 3] + 120 * ringCordiXY[i + 5] +
            45 * ringCordiXY[i + 7] + 10 * ringCordiXY[i + 9] + 
            ringCordiXY[i + 11];
            
    // 计算平滑后的横纵坐标，写入gSmCordiXY中
    gSmCordiXY[2 * t] = 1.0 * x / GAUSS_ELEVEN;  
    gSmCordiXY[2 * t + 1] = 1.0 * y / GAUSS_ELEVEN;
}

// 平滑窗口大小为3的核函数的具体实现
static __global__ void gauss3SmoothXY(int n, int* ringCordiXY,  
                                      float* gSmCordiXY)
{
     // 计算当前线程下标
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查线程是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (t >= n)
        return;
    
    // 计算当前线程对应的数组下标   
    int i =  2 * (n + t);
    
    // 计算横纵坐标对应的数值
    int x = ringCordiXY[i - 2] + 2 * ringCordiXY[i] + ringCordiXY[i + 2];
    int y = ringCordiXY[i - 1] + 2 * ringCordiXY[i + 1] + ringCordiXY[i + 3];
    
    // 计算平滑后的横纵坐标，写入gSmCordiXY中
    gSmCordiXY[2 * t] = 1.0 * x / GAUSS_THREE;  
    gSmCordiXY[2 * t + 1] = 1.0 * y / GAUSS_THREE;

}

// curve高斯平滑的具体实现
__host__ int GaussSmoothXY::gaussSmoothCurveXY(Curve *curve, int smWindowSize) 
{
    // 局部变量，错误码。
    int state;  
   
    // 获取曲线长度
    int cLength = curve->curveLength;
    
    // 曲线长度不足3
    // 按照委托方的需求返回错误值，为了避免和现有的errorcode冲突
    // 以下返回值均为自定义
    if (cLength < 3 )
        return -11;
        
    // 平滑窗口大小不在有效范围之内则报错
    // smWindowSize应仅限于3、5、7、9、11五种，防止因误输入其他数据
    // 导致用户在不知情的情况下采用了默认值3，从而出现和期望不一致的平滑结果
    if (smWindowSize != 3 && smWindowSize != 5 && smWindowSize != 7 &&
        smWindowSize != 9 && smWindowSize != 11) 
        return -12;
        
    // 如果曲线长度过短导致不大于平滑区间长度，则对平滑区间做缩小处理
    // 以取得正确的平滑结果，每次缩小长度为2
    if (cLength <= smWindowSize ) {
    
        // 如果缩小smWindowSize后不在处理范围则返回
        if ((smWindowSize -= 2) < 3) 
            return -13;
            
        // 缩小后曲线长度大于平滑区间长度，则处理结束
        if (cLength > smWindowSize) 
            curve->smWindowSize = smWindowSize;
        
        // 仍然不符合要求，继续缩小
        else {
            // 如果缩小smWindowSize后不在处理范围则返回
            if ((smWindowSize -= 2) < 3) 
                return -14;
                
            // 缩小后曲线长度大于平滑区间长度，则处理结束 
            if (cLength > smWindowSize)
                curve->smWindowSize = smWindowSize;
                
            else {
                // 如果缩小smWindowSize后不在处理范围则返回
                if ((smWindowSize -= 2) < 3)
                    return -15;
                
                // 缩小后曲线长度大于平滑区间长度，则处理结束                
                if (cLength > smWindowSize)
                    curve->smWindowSize = smWindowSize;
                    
                else {
                    // 如果缩小smWindowSize后不在处理范围则返回
                    if ((smWindowSize -= 2) < 3)
                        return -16;
                    
                    // 曲线长度大于平滑区间长度，则处理结束                    
                    if (cLength > smWindowSize)
                        curve->smWindowSize = smWindowSize;
                    
                    // 处理窗口取最小值
                    else {
                        curve->smWindowSize = 3;
                    }
                }                
            } 
        }
    }
    else {
        // 设置curve的成员变量smWindowSize的值
        curve->smWindowSize = smWindowSize;
    }
    
    // 如果平滑坐标数组为空，则开辟空间
    if (curve->smCurveCordiXY == NULL){
        // 为gSmCordiXY开辟host内存。
        curve->smCurveCordiXY = new float[2 * cLength];
    }
    
    // 启动平滑函数
    state = gaussSmoothXY(cLength, curve->crvData, curve->closed,
                          curve->smCurveCordiXY, smWindowSize);
    
    // 平滑出错，清除平滑相关数据
    if(state != NO_ERROR)
    {
        curve->smWindowSize = 0;
        delete curve->smCurveCordiXY;
        curve->smCurveCordiXY = NULL;
    }

    return  state;

}

// curve高斯平滑核心函数
__host__ int GaussSmoothXY::gaussSmoothXY(int n, int* origiCordiXY, bool closed, 
                                          float* gSmCordiXY, int smWindowSize)
{
    // 局部变量，错误码。 
    cudaError_t cudaerrcode;
    
    // 高斯平滑用到的辅助数据
    static int ringCordiLength = 0;
    static int reverseCordiLength = 0;
    static int* reverseCordiXY = NULL;
    int n3 = n * 3;

    // 计算设置ringCordiLength
    if (ringCordiLength <= n3) {
        if (ringCordiXY != NULL) {
            cudaFree(ringCordiXY);  
        } 

        n3 += 3;
        
        // 在GPGPU GLOBAL memory中取得一个长度为n3 * 2的memory -> ringCordiXY;
        cudaerrcode = cudaMalloc((void **)&ringCordiXY, sizeof (int) * n3 * 2);               
    
        // 开辟失败，释放内存空间。
        if (cudaerrcode != cudaSuccess) {
            cudaFree(ringCordiXY);
            return CUDA_ERROR;
        }
        ringCordiLength = n3;
    }

    // 处理闭合曲线
    if (closed) {
        // 由HOST memory 向GPGPU memory copy:
        // copy origiCordiXY to ringCordiXY; copy size = 2 * n
        cudaerrcode = cudaMemcpy(ringCordiXY, origiCordiXY, 
                                 2 * n * sizeof(int), cudaMemcpyHostToDevice); 
        // 拷贝函数出错
        if (cudaerrcode != cudaSuccess) {
            cudaFree(ringCordiXY);
            return CUDA_ERROR;
        }

        // copy origiCordiXY to ringCordiXY + 2 * n;  copy size = 2 * n
        cudaerrcode = cudaMemcpy(ringCordiXY + 2 * n, origiCordiXY , 
                                 2 * n * sizeof(int) , cudaMemcpyHostToDevice); 
        // 拷贝函数出错
        if (cudaerrcode != cudaSuccess) {
            cudaFree(ringCordiXY);
            return CUDA_ERROR;
        }
        
        // copy origiCordiXY to ringCordiXY + 2 * n;  copy size = 2 * n
        cudaerrcode = cudaMemcpy(ringCordiXY + 4 * n, origiCordiXY , 
                                 2 * n * sizeof(int), cudaMemcpyHostToDevice); 
        // 拷贝函数出错
        if (cudaerrcode != cudaSuccess) {
            cudaFree(ringCordiXY);
            return CUDA_ERROR;
        }
       
    }
    // 处理非闭合曲线
    else {
    
        // 为reverseCordiXY开辟空间
        if (reverseCordiLength <= n ) {
            delete reverseCordiXY;
            reverseCordiXY = new int[2 * n + 6];
            reverseCordiLength = n + 3;
        }
        
        // 将x、y坐标分别反转后存入reverseCordiXY中
        #pragma unroll
        for (int i = 0; i < 2 * n; i += 2) {
            reverseCordiXY[i + 1] = origiCordiXY[2 * n - i - 1];
        }
        #pragma unroll
        for (int i = 1; i < 2 * n; i += 2) {
            reverseCordiXY[i - 1] = origiCordiXY[2 * n - i - 1];
        }
        
        // 由HOST memory 向GPGPU memory copy:
        // copy reverseCordiXY to ringCordiXY; copy size = 2 * n
        cudaerrcode = cudaMemcpy(ringCordiXY, reverseCordiXY, 
                                 2 * n * sizeof(int), cudaMemcpyHostToDevice); 
        // 拷贝函数出错
        if (cudaerrcode != cudaSuccess) {
            delete reverseCordiXY;
            cudaFree(ringCordiXY);
            return CUDA_ERROR;
        }
        
        //copy origiCordiXY to ringCordiXY + 2 * n;   
        // copy size = 2 * n  这个必须是origiCordiXY
        cudaerrcode = cudaMemcpy(ringCordiXY + 2 * n, origiCordiXY, 
                                 2 * n * sizeof(int), cudaMemcpyHostToDevice); 
        // 拷贝函数出错
        if (cudaerrcode != cudaSuccess) {
            delete reverseCordiXY;
            cudaFree(ringCordiXY);
            return CUDA_ERROR;
        }
        
        // copy reverseCordiXY to ringCordiXY + 4 * n;   copy size = 2 * n
        cudaerrcode = cudaMemcpy(ringCordiXY + 4 * n, reverseCordiXY,
                                 2 * n * sizeof(int), cudaMemcpyHostToDevice); 
        // 拷贝函数出错
        if (cudaerrcode != cudaSuccess) {
            delete reverseCordiXY;
            cudaFree(ringCordiXY);
            return CUDA_ERROR;
        }
        
    }
   
    // 为成员变量gSmCordiXY在device端开辟空间
    cudaerrcode = cudaMalloc((void **)&this->gSmCordiXY, sizeof(float) * 2 * n);               
    
    // 开辟失败，释放内存空间。
    if (cudaerrcode != cudaSuccess) {
        cudaFree(ringCordiXY);
        cudaFree(this->gSmCordiXY);
        delete reverseCordiXY;
        return CUDA_ERROR;
    }

    // 根据平滑窗口大小选择合适的平滑函数
    // 按照委托方要求的顺序，核函数按照7,5,9,11,3排列
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridSize,blockSize;
    
    blockSize.x = DEF_BLOCK_X;
    blockSize.y = 1;
    gridSize.x = (n + blockSize.x - 1) / blockSize.x;
    gridSize.y = 1;
    
    switch (smWindowSize) {     
    case 7:
        // 启动平滑窗口大小为7的核函数
        gauss7SmoothXY<<<gridSize, blockSize>>>(n, ringCordiXY, 
                                                this->gSmCordiXY );
        //核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(ringCordiXY); 
            cudaFree(this->gSmCordiXY);
            delete reverseCordiXY;
            return CUDA_ERROR;
        }
            
        break;
         
    case 5:
        // 启动平滑窗口大小为5的核函数
        gauss5SmoothXY<<<gridSize, blockSize>>>(n, ringCordiXY,
                                                this->gSmCordiXY );
        //核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(ringCordiXY); 
            cudaFree(this->gSmCordiXY);
            delete reverseCordiXY;
            return CUDA_ERROR;
        }
        
        break;
         
    case 9:
        // 启动平滑窗口大小为9的核函数
        gauss9SmoothXY<<<gridSize, blockSize>>>(n, ringCordiXY, 
                                                this->gSmCordiXY);
        //核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(ringCordiXY); 
            cudaFree(this->gSmCordiXY);
            delete reverseCordiXY;
            return CUDA_ERROR;
        }
            
        break;
         
    case 11:
       
         // 启动平滑窗口大小为11的核函数
        gauss11SmoothXY<<<gridSize, blockSize>>>(n, ringCordiXY, 
                                                 this->gSmCordiXY);
        //核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(ringCordiXY); 
            cudaFree(this->gSmCordiXY);
            delete reverseCordiXY;
            return CUDA_ERROR;
        }

        break;
         
    default:
        // 启动平滑窗口大小为3的核函数
        gauss3SmoothXY<<<gridSize, gridSize>>>(n, ringCordiXY, 
                                               this->gSmCordiXY);
        //核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(ringCordiXY);  
            cudaFree(this->gSmCordiXY);
            delete reverseCordiXY;
            return CUDA_ERROR;
        }  
        
        break;
    }
    
    // 将计算结果拷贝回gSmCordiXY中
    cudaerrcode = cudaMemcpy(gSmCordiXY, this->gSmCordiXY, 
                             sizeof (float) * 2 * n, cudaMemcpyDeviceToHost); 
    // 拷贝函数出错 
    if (cudaerrcode != cudaSuccess) {
        cudaFree(ringCordiXY); 
        cudaFree(this->gSmCordiXY);
        delete reverseCordiXY;
        return CUDA_ERROR;
    }
    
    return NO_ERROR;
}

