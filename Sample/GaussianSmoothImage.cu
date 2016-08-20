// GaussianSmoothImage.cu
// 图像高斯平滑操作，包括普通高斯平滑和带mask的高斯平滑

#include "GaussianSmoothImage.h"
#include "ErrorCode.h"

// 宏定义，定义了五个高斯平滑尺度对应的权重总和
#define GAUSS_THREE  16
#define GAUSS_FIVE   256
#define GAUSS_SEVEN  4096
#define GAUSS_NINE   65536 
#define GAUSS_ELEVEN 1048576 

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 下列五个核函数为普通高斯平滑核函数

// 1.平滑窗口大小为3*3的高斯平滑函数
static __global__ void
gauss3SmoothImage(
    ImageCuda origiImageGPU,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标
    int smWidth,                // 平滑窗口宽度
    int smHeight                // 平滑窗口高度
); 

// 2.平滑窗口大小为5*5的高斯平滑函数                       
static __global__ void
gauss5SmoothImage(
    ImageCuda origiImageGPU,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标
    int smWidth,                // 平滑窗口宽度
    int smHeight                // 平滑窗口高度
); 
// 3.平滑窗口大小为7*7的高斯平滑函数                                   
static __global__ void
gauss7SmoothImage(
    ImageCuda origiImageGPU,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标
    int smWidth,                // 平滑窗口宽度
    int smHeight                // 平滑窗口高度
); 

// 4.平滑窗口大小为9*9的高斯平滑函数 
static __global__ void
gauss9SmoothImage(
    ImageCuda origiImageGPU,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标
    int smWidth,                // 平滑窗口宽度
    int smHeight                // 平滑窗口高度
); 

// 5.平滑窗口大小为11*11的高斯平滑函数                       
static __global__ void
gauss11SmoothImage(
    ImageCuda origiImageGPU,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标
    int smWidth,                // 平滑窗口宽度
    int smHeight                // 平滑窗口高度
); 

//  下列五个核函数为带mask的高斯平滑函数

// 1.平滑窗口大小为3*3，带mask的高斯平滑函数    
static __global__ void
gauss3SmoothImage(
    ImageCuda origiImageGPU,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标 
    int smWidth,                // 平滑窗口宽度
    int smHeight,               // 平滑窗口高度
    ImageCuda maskImageGPU,     // mask图像
    unsigned char mask          // mask值
);

// 2.平滑窗口大小为5*5，带mask的高斯平滑函数    
static __global__ void
gauss5SmoothImage(
    ImageCuda origiImageGPU,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标 
    int smWidth,                // 平滑窗口宽度
    int smHeight,               // 平滑窗口高度
    ImageCuda maskImageGPU,     // mask图像
    unsigned char mask          // mask值
);

// 3.平滑窗口大小为7*7，带mask的高斯平滑函数    
static __global__ void
gauss7SmoothImage(
    ImageCuda origiImageGPU,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标 
    int smWidth,                // 平滑窗口宽度
    int smHeight,               // 平滑窗口高度
    ImageCuda maskImageGPU,     // mask图像
    unsigned char mask          // mask值
);

// 4.平滑窗口大小为9*9，带mask的高斯平滑函数    
static __global__ void
gauss9SmoothImage(
    ImageCuda const origiImageGPU ,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标 
    int smWidth,                // 平滑窗口宽度
    int smHeight,               // 平滑窗口高度
    ImageCuda maskImageGPU,     // mask图像
    unsigned char mask          // mask值
);

// 5.平滑窗口大小为11*11，带mask的高斯平滑函数    
static __global__ void
gauss11SmoothImage(
    ImageCuda origiImageGPU,    // 原始图像
    ImageCuda gaussSmImageGPU,  // 平滑后图像
    int smLocatX,               // 平滑起始横坐标
    int smLocatY,               // 平滑起始纵坐标 
    int smWidth,                // 平滑窗口宽度
    int smHeight,               // 平滑窗口高度 
    ImageCuda maskImageGPU,     // mask图像
    unsigned char mask          // mask值
);



 // 平滑窗口大小为7*7的高斯平滑函数实现    
static __global__ void gauss7SmoothImage(ImageCuda origiImageGPU, 
                                          ImageCuda gaussSmImageGPU, 
                                          int smLocatX, int smLocatY,
                                          int smWidth, int smHeight)
{
      // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    //w=1024;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    //printf("smLocatX + smWidth = %d \n smLocatY + smHeight =  %d",smLocatX + smWidth,smLocatY + smHeight);
    //if(x==0&&y==0)
    //printf("x = %d y = %d \n",x,y);
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
    if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 高斯平滑系数数组
     int GF[7] = {1, 6, 15, 20, 15, 6, 1};
     
    // 高斯卷积累加和
    int c = 0;
    /*if(y-3<0||x-3<0){
        printf("%d\n",origiImageGPU.imgMeta.imgData[(y - 3) * w + (x - 3)]);
    }*/
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
    for(int i = 0; i < 7;i++) {
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
        for(int j = 0; j < 7;j++)
            c += GF[i] * GF[j] *
                 origiImageGPU.imgMeta.imgData[(y + i - 3) * w + (x + j - 3)];
    }
    
    // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / GAUSS_SEVEN + 0.5f; 
    //printf("1");
}

 // 平滑窗口大小为5*5的高斯平滑函数实现
static __global__ void gauss5SmoothImage(ImageCuda origiImageGPU, 
                                          ImageCuda gaussSmImageGPU, 
                                          int smLocatX, int smLocatY,
                                          int smWidth, int smHeight)
{
    // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
    if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 高斯平滑系数数组
    int GF[5] = {1, 4, 6, 4, 1};

    // 高斯卷积累加和
    int c = 0;
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
    for(int i = 0; i < 5;i++) {
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
        for(int j = 0; j < 5;j++)
            c += GF[i] * GF[j] *
                 origiImageGPU.imgMeta.imgData[(y + i - 2) * w + (x + j - 2)];
    }
    
    // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / GAUSS_FIVE  + 0.5f; 
}

// 平滑窗口大小为9*9的高斯平滑函数实现
static __global__ void gauss9SmoothImage(ImageCuda origiImageGPU, 
                                          ImageCuda gaussSmImageGPU, 
                                          int smLocatX, int smLocatY,
                                          int smWidth, int smHeight)
{
    // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
    if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 高斯平滑系数数组
    const int GF[9] = {1, 8, 28, 56, 70, 56, 28, 8, 1};

    // 高斯卷积累加和
    int c = 0;
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
    for(int i = 0; i < 9;i++) {
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
         for(int j = 0; j < 9; j++)
            c += GF[i] * GF[j] *
                 origiImageGPU.imgMeta.imgData[(y + i - 4) * w + (x + j - 4)];
    }
    
    // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / GAUSS_NINE  + 0.5f; 
}

// 平滑窗口大小为11*11的高斯平滑函数实现
static __global__ void gauss11SmoothImage(ImageCuda origiImageGPU, 
                                          ImageCuda gaussSmImageGPU, 
                                          int smLocatX, int smLocatY,
                                          int smWidth, int smHeight)
{
    // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
    if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 高斯平滑系数数组
    int GF[11] = {1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1};

    // 高斯卷积累加和
    int c = 0;
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
    for(int i = 0; i < 11;i++) {
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll

         for(int j = 0; j < 11;j++)
            c += GF[i] * GF[j] *
                 origiImageGPU.imgMeta.imgData[(y + i - 5) * w + (x + j - 5)];
    }
    
     // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / GAUSS_ELEVEN  + 0.5f; 
}

 // 平滑窗口大小为3*3的高斯平滑函数实现
static __global__ void gauss3SmoothImage(ImageCuda origiImageGPU, 
                                          ImageCuda gaussSmImageGPU, 
                                          int smLocatX, int smLocatY,
                                          int smWidth, int smHeight)
{

    // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
    if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 高斯平滑系数数组
    int GF[3] = {1, 2, 1};

    // 高斯卷积累加和
    int c = 0;
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
    for(int i = 0; i < 3;i++) {
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll

         for(int j = 0; j < 3; j++)
            c += GF[i] * GF[j] *
                 origiImageGPU.imgMeta.imgData[(y + i - 1) * w + (x + j - 1)];
    }
    
    // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / GAUSS_THREE  + 0.5f ; 
}


// 平滑窗口大小为7*7的，带mask高斯平滑函数实现
static __global__ void gauss7SmoothImage(ImageCuda origiImageGPU, 
                                         ImageCuda gaussSmImageGPU, 
                                         int smLocatX, int smLocatY, 
                                         int smWidth, int smHeight,
                                         ImageCuda maskImageGPU,
                                         unsigned char mask)
{
    // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    
    // 如果mask图像像素值不等于mask则不处理
    if (maskImageGPU.imgMeta.imgData[y * w + x]  != mask) 
        return ;
        
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
     if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 获取mask图像数据
    unsigned char * maskImg = maskImageGPU.imgMeta.imgData;
    
    // 高斯平滑系数数组
    int gf[7] = {1, 6, 15, 20, 15, 6, 1};
    
    // 高斯卷积累加和 
    int c = 0;
    
    // 参加计算的像素点权重总和wsum,当前权重wgh
    int wsum = 0, wgh;
    
    // 图像像素索引
    int mIdx;
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll 
 
    for(int i = 0; i < 7; i++){
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
          for(int j = 0; j < 7; j++) {  
            // 获取图像像素索引
            mIdx=(y + i - 3) * w + (x + j - 3);
            
            // 只处理mask图像像素值等于mask值的像素点
            if (maskImg[mIdx] == mask) {
            
                // 计算当前像素点的权重
                wgh = gf[i] * gf[j];
                
                // 当前像素点的权重累加到总权重中
                wsum += wgh ;
                
                // 计算像素值加权累加和
                c += wgh * origiImageGPU.imgMeta.imgData[mIdx]; 
            }
        }
    }
    
    // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / wsum  + 0.5f; 
}

 // 平滑窗口大小为5*5的，带mask高斯平滑函数实现
static __global__ void gauss5SmoothImage(ImageCuda origiImageGPU, 
                                         ImageCuda gaussSmImageGPU, 
                                         int smLocatX, int smLocatY, 
                                         int smWidth, int smHeight,
                                         ImageCuda maskImageGPU,
                                         unsigned char mask)
{
    // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    
    // 如果mask图像像素值不等于mask则不处理
    if (maskImageGPU.imgMeta.imgData[y * w + x]  != mask) 
        return ;
        
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
     if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 获取mask图像数据
    unsigned char * maskImg = maskImageGPU.imgMeta.imgData;
    
    // 高斯平滑系数数组
    int gf[5] = {1, 4, 6, 4, 1};
    
    // 高斯卷积累加和 
    int c = 0;
    
    // 参加计算的像素点权重总和wsum,当前权重wgh
    int wsum = 0, wgh;
    
    // 图像像素索引
    int mIdx;
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll 
 
    for(int i = 0; i < 5; i++){
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
          for(int j = 0; j < 5; j++) {  
            // 获取图像像素索引
            mIdx=(y + i - 2) * w + (x + j - 2);
            
            // 只处理mask图像像素值等于mask值的像素点
            if (maskImg[mIdx] == mask) {
            
                // 计算当前像素点的权重
                wgh = gf[i] * gf[j];
                
                // 当前像素点的权重累加到总权重中
                wsum += wgh ;
                
                // 计算像素值加权累加和
                c += wgh * origiImageGPU.imgMeta.imgData[mIdx]; 
            }
        }
    }
    
    // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / wsum  + 0.5f; 
}

 // 平滑窗口大小为9*9的，带mask高斯平滑函数实现
static __global__ void gauss9SmoothImage(ImageCuda origiImageGPU, 
                                         ImageCuda gaussSmImageGPU, 
                                         int smLocatX, int smLocatY, 
                                         int smWidth, int smHeight,
                                         ImageCuda maskImageGPU,
                                         unsigned char mask)
{
    // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    
    // 如果mask图像像素值不等于mask则不处理
    if (maskImageGPU.imgMeta.imgData[y * w + x]  != mask) 
        return ;
        
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
    if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 获取mask图像数据
    unsigned char * maskImg = maskImageGPU.imgMeta.imgData;
    
    // 高斯平滑系数数组
    int gf[9] =  {1, 8, 28, 56, 70, 56, 28, 8, 1};
    
    // 高斯卷积累加和 
    int c = 0;
    
    // 参加计算的像素点权重总和wsum,当前权重wgh
    int wsum = 0, wgh;
    
    // 图像像素索引
    int mIdx;
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll 
 
    for(int i = 0; i < 9; i++){
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
          for(int j = 0; j < 9; j++) {  
            // 获取图像像素索引
            mIdx=(y + i - 4) * w + (x + j - 4);
            
            // 只处理mask图像像素值等于mask值的像素点
            if (maskImg[mIdx] == mask) {
            
                // 计算当前像素点的权重
                wgh = gf[i] * gf[j];
                
                // 当前像素点的权重累加到总权重中
                wsum += wgh ;
                
                // 计算像素值加权累加和
                c += wgh * origiImageGPU.imgMeta.imgData[mIdx]; 
            }
        }
    }
    
    // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / wsum  + 0.5f; 
}

 // 平滑窗口大小为11*11的，带mask高斯平滑函数实现
static __global__ void gauss11SmoothImage(ImageCuda origiImageGPU,              
                                          ImageCuda gaussSmImageGPU, 
                                          int smLocatX, int smLocatY, 
                                          int smWidth, int smHeight,
                                          ImageCuda maskImageGPU,
                                          unsigned char mask)
{
    // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    
    // 如果mask图像像素值不等于mask则不处理
    if (maskImageGPU.imgMeta.imgData[y * w + x]  != mask) 
        return ;
        
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
     if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 获取mask图像数据
    unsigned char * maskImg = maskImageGPU.imgMeta.imgData;
    
    // 高斯平滑系数数组
    int gf[11] = {1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1};
    
    // 高斯卷积累加和 
    int c = 0;
    
    // 参加计算的像素点权重总和wsum,当前权重wgh
    int wsum = 0, wgh;
    
    // 图像像素索引
    int mIdx;
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll 
 
    for(int i = 0; i < 11; i++) {
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
         for(int j = 0; j < 11; j++) {  
            // 获取图像像素索引
            mIdx=(y + i - 5) * w + (x + j - 5);
            
            // 只处理mask图像像素值等于mask值的像素点
            if (maskImg[mIdx] == mask) {
            
                // 计算当前像素点的权重
                wgh = gf[i] * gf[j];
                
                // 当前像素点的权重累加到总权重中
                wsum += wgh ;
                
                // 计算像素值加权累加和
                c += wgh * origiImageGPU.imgMeta.imgData[mIdx]; 
            }
        }
    }
    
    // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / wsum  + 0.5f; 
}

 // 平滑窗口大小为3*3的，带mask高斯平滑函数实现
static __global__ void gauss3SmoothImage(ImageCuda origiImageGPU, 
                                         ImageCuda gaussSmImageGPU, 
                                         int smLocatX, int smLocatY, 
                                         int smWidth, int smHeight,
                                         ImageCuda maskImageGPU,
                                         unsigned char mask)
{
    // 获取pixel在原图像中的位置 
    int w = origiImageGPU.pitchBytes;
    int x = blockIdx.x * blockDim.x + threadIdx.x + smLocatX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + smLocatY;
    
    // 如果mask图像像素值不等于mask则不处理
    if (maskImageGPU.imgMeta.imgData[y * w + x]  != mask) 
        return ;
        
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资
    // 源，一方面防止由于段错误导致的程序崩溃。
     if(x >= smLocatX + smWidth || y >= smLocatY + smHeight)
        return ;

    // 获取mask图像数据
    unsigned char * maskImg = maskImageGPU.imgMeta.imgData;
    
    // 高斯平滑系数数组
    int gf[3] = {1, 2, 1};
    
    // 高斯卷积累加和 
    int c = 0;
    
    // 参加计算的像素点权重总和wsum,当前权重wgh
    int wsum = 0, wgh;
    
    // 图像像素索引
    int mIdx;
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll 
 
    for(int i = 0; i < 3; i++) {
    
    // 编译预处理，在编译阶段将循环展开，节约循环跳转时间
    #pragma unroll
    
         for(int j = 0; j < 3; j++) {  
            // 获取图像像素索引
            mIdx=(y + i - 1) * w + (x + j - 1);
            
            // 只处理mask图像像素值等于mask值的像素点
            if (maskImg[mIdx] == mask) {
            
                // 计算当前像素点的权重
                wgh = gf[i] * gf[j];
                
                // 当前像素点的权重累加到总权重中
                wsum += wgh ;
                
                // 计算像素值加权累加和
                c += wgh * origiImageGPU.imgMeta.imgData[mIdx]; 
            }
        }
    }
    
    // 计算平滑后像素值，结果四舍五入
    gaussSmImageGPU.imgMeta.imgData[y * w + x] = 1.0 * c / wsum  + 0.5f; 

}

// 普通高斯平滑函数
__host__ int GaussSmoothImage::gaussSmoothImage(Image* origiImage, int smWidth, 
                                                int smHeight, int smLocatX, 
                                                int smLocatY, int smWindowSize, 
                                                Image* gaussSmImage)
{
    // 局部变量，错误码。
    int errcode;  

    // 输入输出图像指针不能为空
    if (origiImage == NULL || gaussSmImage == NULL)
        return NULL_POINTER;

    // 获取图像尺寸信息
    int  imgWidth = origiImage->width;
    int  imgHeight = origiImage->height;

    // 图像小于平滑范围
    if (imgWidth < smWidth || imgHeight < smHeight) 
        return -11; 
        
    // 平滑范围小于最大平滑窗口大小
    if (smWidth < 11 || smHeight < 11) 
        return -12;
        
    // 输入的平滑窗口大小不在处理范围之内
    if (smWindowSize < 3 || smWindowSize > 11) 
        return -13;

    // 平滑计算所涉及data位置或范围不能超出原始图像的物理范围，
    // 故应根据smWindowSize作适当调整。
    int  marginOff = (smWindowSize + 1) >> 1;
    int  leftMargin = smLocatX - marginOff;
    int  rightMargin = imgWidth - smLocatX - smWidth - marginOff;

    int  topMargin = smLocatY - marginOff;
    int  bottomMargin = imgHeight - smLocatY - smHeight - marginOff;
 
    // 平滑时将发生左侧出界
    if (leftMargin < 0) {
        smLocatX -= leftMargin;
        smWidth += leftMargin;
    }

    // 平滑时将发生右侧出界
    if (rightMargin < 0) {
        smWidth += rightMargin;
    }
    
    // 平滑宽度小于1
    if (smWidth < 1)
        return -14; 

    // 平滑时将发生上方出界
    if (topMargin < 0) {
        smLocatY -= topMargin;
        smHeight += topMargin;
    }

    // 平滑时将发生下方出界
    if (bottomMargin < 0) {
        smHeight += bottomMargin;
    }

    // 平滑高度小于1
    if (smHeight < 1) 
        return -15;  
        
    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(origiImage);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 将输出图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(gaussSmImage);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 提取输入图像。
    ImageCuda origiImageGPU;
    errcode = ImageBasicOp::roiSubImage(origiImage, &origiImageGPU);
    if (errcode != NO_ERROR) {
        return errcode;
    }
    
    // 提取输出图像。
    ImageCuda gaussSmImageGPU;
    errcode = ImageBasicOp::roiSubImage(gaussSmImage, &gaussSmImageGPU);
    if (errcode != NO_ERROR) {
        return errcode;
    }
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridSize,blockSize;
    
    blockSize.x = DEF_BLOCK_X;
    blockSize.y = DEF_BLOCK_Y;
    gridSize.x = (smWidth + blockSize.x - 1) / blockSize.x;
    gridSize.y = (smHeight + blockSize.y - 1) / blockSize.y;
    
    // 根据平滑窗口大小选择对应的核函数
    // 按照委托方要求，顺序为7、5、9、11、3
    switch (smWindowSize) { 
    
    case 7: 
    // 启动平滑窗口大小为7的核函数
        gauss7SmoothImage<<<gridSize, blockSize>>>(origiImageGPU, 
                                                   gaussSmImageGPU,
                                                   smLocatX, smLocatY,
                                                   smWidth, smHeight);
        // 核函数出错                                         
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }    
         
        break;
 
    case 5:
        // 启动平滑窗口大小为5的核函数
        gauss5SmoothImage<<<gridSize, blockSize>>>(origiImageGPU, 
                                                   gaussSmImageGPU,
                                                   smLocatX, smLocatY,
                                                   smWidth, smHeight);
        // 核函数出错                                         
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
            
        break;
     
    case 9:
           
        // 启动平滑窗口大小为9的核函数
        gauss9SmoothImage<<<gridSize, blockSize>>>(origiImageGPU, 
                                                   gaussSmImageGPU,
                                                   smLocatX, smLocatY,
                                                   smWidth, smHeight);               
        // 核函数出错                                         
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
      
        break;
         
    case 11:
        // 启动平滑窗口大小为11的核函数
        gauss11SmoothImage<<<gridSize, blockSize>>>(origiImageGPU, 
                                                    gaussSmImageGPU,
                                                    smLocatX, smLocatY,
                                                    smWidth, smHeight);
        // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
            
        break;
         
    default:
        // 启动平滑窗口大小为3的核函数
        gauss3SmoothImage<<<gridSize,blockSize>>>(origiImageGPU, 
                                                  gaussSmImageGPU,
                                                  smLocatX, smLocatY,
                                                  smWidth, smHeight);
        // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
            
        break;
    }

    return NO_ERROR;

}

// 带mask的高斯平滑函数
__host__ int GaussSmoothImage::gaussSmoothImage(Image* origiImage, 
                                                int smWidth, int smHeight, 
                                                int smLocatX, int smLocatY, 
                                                int smWindowSize,
                                                Image* gaussSmImage, 
                                                Image* maskImage, 
                                                unsigned char mask)
{
    // 局部变量，错误码。
    int errcode;  

    // 获取图像尺寸信息
    int  imgWidth = origiImage->width;
    int  imgHeight = origiImage->height;

    // 图像小于平滑范围
    if (imgWidth < smWidth || imgHeight < smHeight) 
        return -11; 
        
    // 平滑范围小于最大平滑窗口
    if (smWidth < 11 || smHeight < 11) 
        return -12;
        
    // 输入的平滑窗口大小不在可处理范围之内
    if (smWindowSize < 3 || smWindowSize > 11) 
        return -13;

    // 平滑计算所涉及data位置或范围不能超出原始图像的物理范围，
    // 故应根据smWindowSize作适当调整。
    int  marginOff = (smWindowSize + 1) >> 1;
    int  leftMargin = smLocatX - marginOff;
    int  rightMargin = imgWidth - smLocatX - smWidth - marginOff;

    int  topMargin = smLocatY - marginOff;
    int  bottomMargin = imgHeight - smLocatY - smHeight - marginOff;

    // 平滑时将发生左侧出界
    if (leftMargin < 0) {
        smLocatX -= leftMargin;
        smWidth += leftMargin;
    }

    // 平滑时将发生右侧出界
    if (rightMargin < 0) {
        smWidth += rightMargin;
    }

    // 平滑宽度小于1
    if (smWidth < 1) 
       return -14; 

    // 平滑时将发生上方出界
    if (topMargin < 0) {
        smLocatY -= topMargin;
        smHeight += topMargin;
    }

    // 平滑时将发生下方出界
    if (bottomMargin < 0) {
        smHeight += bottomMargin;
    }

    // 平滑高度小于1
    if (smHeight < 1)
        return -15; 

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(origiImage);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 将输出图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(gaussSmImage);
    if (errcode != NO_ERROR) {
        return errcode;
    }
    
    // 将mask图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(maskImage);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 提取输入图像。
    ImageCuda origiImageGPU;
    errcode = ImageBasicOp::roiSubImage(origiImage, &origiImageGPU);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 提取输出图像。
    ImageCuda gaussSmImageGPU;
    errcode = ImageBasicOp::roiSubImage(gaussSmImage, &gaussSmImageGPU);
    if (errcode != NO_ERROR) {
        return errcode;
    }
    
    // 提取mask图像。
    ImageCuda maskImageGPU;
    errcode = ImageBasicOp::roiSubImage(maskImage, &maskImageGPU);
    if (errcode != NO_ERROR) {
        return errcode;
    }
    
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridSize,blockSize;
    
    blockSize.x = DEF_BLOCK_X;
    blockSize.y = DEF_BLOCK_Y;
    gridSize.x = (smWidth + blockSize.x - 1) / blockSize.x;
    gridSize.y = (smHeight + blockSize.y - 1) / blockSize.y;

    // 根据平滑窗口大小选择对应的核函数
    // 按照委托方要求，顺序为7、5、9、11、3
    switch (smWindowSize)  { 
    
    case 7:
        // 启动平滑窗口大小为7的核函数
        gauss7SmoothImage<<<gridSize, blockSize>>>(origiImageGPU,
                                                   gaussSmImageGPU, 
                                                   smLocatX, smLocatY,
                                                   smWidth, smHeight,
                                                   maskImageGPU, mask); 
        // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
        break;
         
    case 5:
        // 启动平滑窗口大小为5的核函数
        gauss5SmoothImage<<<gridSize, blockSize>>>(origiImageGPU,
                                                   gaussSmImageGPU,
                                                   smLocatX, smLocatY,
                                                   smWidth, smHeight,
                                                   maskImageGPU, mask);
        // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
        break;
         
    case 9:
       
    // 每个窗口纵向线程块数目
 
        gauss9SmoothImage<<<gridSize, blockSize>>>(origiImageGPU, 
                                                   gaussSmImageGPU,
                                                   smLocatX, smLocatY,
                                                   smWidth, smHeight,
                                                   maskImageGPU, mask);
            // 核函数出错
            if (cudaGetLastError() != cudaSuccess) {
                return CUDA_ERROR;
            }
        break;
         
    case 11:
        // 启动平滑窗口大小为11的核函数
        gauss11SmoothImage<<<gridSize, blockSize>>>(origiImageGPU,
                                                    gaussSmImageGPU,
                                                    smLocatX, smLocatY,
                                                    smWidth, smHeight,
                                                    maskImageGPU, mask);
            // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
        break;
         
    default:
        // 启动平滑窗口大小为3的核函数
        gauss3SmoothImage<<<gridSize, blockSize>>>(origiImageGPU, 
                                                   gaussSmImageGPU,
                                                   smLocatX, smLocatY,
                                                   smWidth, smHeight,
                                                   maskImageGPU, mask);
        // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
        break;
    }

    return NO_ERROR;

}
//不带mask的多GPU滤波函数
//相较之前的滤波，不会出现原先留出的空白
__host__ int GaussSmoothImage::gaussSmoothImageMultiGPU(Image* origiImage, 
                                                int smWidth, int smHeight, 
                                                int smLocatX, int smLocatY, 
                                                int smWindowSize,
                                                Image* gaussSmImage
                                                
                                                )
 {
    // 局部变量，获取原本感兴趣区域
    int roiWidth = smWidth;
    int roiHeight = smHeight;
    int roiLocatX = smLocatX;
    int roiLocatY = smLocatY;


    // 局部变量，错误码。
    int errcode;
    cudaError_t cuerrcode;  

    // 获取图像尺寸信息
    int  imgWidth = origiImage->width;
    int  imgHeight = origiImage->height;

    // 图像小于平滑范围
    if (imgWidth < smWidth || imgHeight < smHeight) 
        return -11; 
        
    // 平滑范围小于最大平滑窗口,暂时不需要此代码
    if (smWidth < 11 || smHeight < 11) 
        return -12;
        
    // 输入的平滑窗口大小不在可处理范围之内
    if (smWindowSize < 3 || smWindowSize > 11) 
        return -13;

    // 平滑计算所涉及data位置或范围不能超出原始图像的物理范围，
    // 故应根据smWindowSize作适当调整。
    int  marginOff = (smWindowSize - 1) >> 1;
    int  leftMargin = smLocatX - marginOff;
    int  rightMargin = imgWidth - smLocatX - smWidth - marginOff;

    int  topMargin = smLocatY - marginOff;
    int  bottomMargin = imgHeight - smLocatY - smHeight - marginOff;

    // 平滑时将发生左侧出界
    if (leftMargin < 0) {
        smLocatX -= leftMargin;
        smWidth += leftMargin;
    }

    // 平滑时将发生右侧出界
    if (rightMargin < 0) {
        smWidth += rightMargin;
    }

    // 平滑宽度小于1
    if (smWidth < 1) 
       return -14; 

    // 平滑时将发生上方出界
    if (topMargin < 0) {
        smLocatY -= topMargin;
        smHeight += topMargin;
    }

    // 平滑时将发生下方出界
    if (bottomMargin < 0) {
        smHeight += bottomMargin;
    }

    // 平滑高度小于1
    if (smHeight < 1)
        return -15; 


    
    
    int deviceCount = 1;
    cudaGetDeviceCount(&deviceCount);

    //实际分割后的高度
    int *realHeight = new int [deviceCount];
    //实际分割后的起始Y点坐标
    int *realLocatY = new int [deviceCount];
    ImageCuda *devicefrimg,*deviceoutimg;
    //进行图像切割，以及高度与坐标的分配
    devicefrimg = imageCut(origiImage,realHeight,realLocatY,roiLocatY,roiHeight);
    deviceoutimg = imageCut(gaussSmImage,realHeight,realLocatY,roiLocatY,roiHeight);
    if(roiLocatX+roiWidth>=imgWidth){
        smWidth = imgWidth - roiLocatX;
    }


    
    //开启多个流
    cudaStream_t *stream = new cudaStream_t[deviceCount];





    
    for(int i=0;i<deviceCount;++i){
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
        

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridSize, blockSize;
    blockSize.x = DEF_BLOCK_X;
    blockSize.y = DEF_BLOCK_Y;
    gridSize.x = (devicefrimg[i].imgMeta.width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (devicefrimg[i].imgMeta.height + blockSize.y - 1) / blockSize.y;
    
        


    //切割的图像足够大，才有必要使用此函数
    if(devicefrimg[i].imgMeta.height>=smWindowSize){
    switch (smWindowSize) { 
    
    case 7: 
    // 启动平滑窗口大小为7的核函数
        gauss7SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i]);
        // 核函数出错                                         
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }    
        
        break;
 
    case 5:
        // 启动平滑窗口大小为5的核函数
        gauss5SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i]);
        // 核函数出错                                         
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
            
        break;
     
    case 9:
           
        // 启动平滑窗口大小为9的核函数
        gauss9SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i]);           
        // 核函数出错                                         
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
      
        break;
         
    case 11:
        // 启动平滑窗口大小为11的核函数
        gauss11SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i]);
        // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
            
        break;
         
    default:
        // 启动平滑窗口大小为3的核函数
        gauss3SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i]);
        // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
            
        break;
    }
    }


  }

  
  //关闭流以及清除空间
  for(int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        //Wait for all operations to finish
        cudaStreamSynchronize(stream[i]);
        
        cudaStreamDestroy(stream[i]);
        cudaFree(devicefrimg[i].d_imgData);
        cudaFree(deviceoutimg[i].d_imgData);
    }
    

    return NO_ERROR;
 }
 // 带mask的高斯平滑多GPU函数
__host__ int GaussSmoothImage::gaussSmoothImageMultiGPU(Image* origiImage, 
                                                int smWidth, int smHeight, 
                                                int smLocatX, int smLocatY, 
                                                int smWindowSize,
                                                Image* gaussSmImage, 
                                                Image* maskImage, 
                                                unsigned char mask)
{
        // 局部变量，获取原本感兴趣区域
    int roiWidth = smWidth;
    int roiHeight = smHeight;
    int roiLocatX = smLocatX;
    int roiLocatY = smLocatY;


    // 局部变量，错误码。
    int errcode;
    cudaError_t cuerrcode;  

    // 获取图像尺寸信息
    int  imgWidth = origiImage->width;
    int  imgHeight = origiImage->height;

    // 图像小于平滑范围
    if (imgWidth < smWidth || imgHeight < smHeight) 
        return -11; 
        
    // 平滑范围小于最大平滑窗口,暂时不需要此代码
    if (smWidth < 11 || smHeight < 11) 
        return -12;
        
    // 输入的平滑窗口大小不在可处理范围之内
    if (smWindowSize < 3 || smWindowSize > 11) 
        return -13;

    // 平滑计算所涉及data位置或范围不能超出原始图像的物理范围，
    //而在运算过程中边界条件即使超出逻辑范围其值将被0所替代，
    //因此不用担心无效问题
    // 故应根据smWindowSize作适当调整。
    int  marginOff = (smWindowSize - 1) >> 1;
    int  leftMargin = smLocatX - marginOff;
    int  rightMargin = imgWidth - smLocatX - smWidth - marginOff;

    int  topMargin = smLocatY - marginOff;
    int  bottomMargin = imgHeight - smLocatY - smHeight - marginOff;

    // 平滑时将发生左侧出界
    if (leftMargin < 0) {
        smLocatX -= leftMargin;
        smWidth += leftMargin;
    }

    // 平滑时将发生右侧出界
    if (rightMargin < 0) {
        smWidth += rightMargin;
    }

    // 平滑宽度小于1
    if (smWidth < 1) 
       return -14; 

    // 平滑时将发生上方出界
    if (topMargin < 0) {
        smLocatY -= topMargin;
        smHeight += topMargin;
    }

    // 平滑时将发生下方出界
    if (bottomMargin < 0) {
        smHeight += bottomMargin;
    }

    // 平滑高度小于1
    if (smHeight < 1)
        return -15; 


    
    //获取GPU个数
    int deviceCount = 1;
    cudaGetDeviceCount(&deviceCount);

    //获得分割图像实际需要的的高度
    int *realHeight = new int [deviceCount];

    //获得分割图像实际起始的Y值
    int *realLocatY = new int [deviceCount];

    //分割图像，并赋予高度，与起始Y值
    ImageCuda *devicefrimg,*deviceoutimg,*devicemask;
    deviceoutimg = imageCut(gaussSmImage,realHeight,realLocatY,roiLocatY,roiHeight);
    devicemask = imageCut(maskImage,realHeight,realLocatY,roiLocatY,roiHeight);
    devicefrimg = imageCut(origiImage,realHeight,realLocatY,roiLocatY,roiHeight);
    if(roiLocatX+roiWidth>=imgWidth){
        smWidth = imgWidth - roiLocatX;
    }


    
    //开启多个流
    cudaStream_t *stream = new cudaStream_t[deviceCount];


    for(int i=0;i<deviceCount;++i){
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
        
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 gridSize, blockSize;
    blockSize.x = DEF_BLOCK_X;
    blockSize.y = DEF_BLOCK_Y;
    gridSize.x = (devicefrimg[i].imgMeta.width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (devicefrimg[i].imgMeta.height + blockSize.y - 1) / blockSize.y;
        

    //切割的图像足够大，才有必要使用此函数
    if(devicefrimg[i].imgMeta.height>=smWindowSize){
    switch (smWindowSize) { 
    
    case 7: 
    // 启动平滑窗口大小为7的核函数
        gauss7SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i],
                                                   devicemask[i],mask);
        // 核函数出错                                         
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }    
         //printf("get real success9\n");
        break;
 
    case 5:
        // 启动平滑窗口大小为5的核函数
        gauss5SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i],
                                                   devicemask[i],mask);
        // 核函数出错                                         
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
            
        break;
     
    case 9:
           
        // 启动平滑窗口大小为9的核函数
        gauss9SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i],
                                                   devicemask[i],mask);          
        // 核函数出错                                         
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
      
        break;
         
    case 11:
        // 启动平滑窗口大小为11的核函数
        gauss11SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i],
                                                   devicemask[i],mask);
        // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
            
        break;
         
    default:
        // 启动平滑窗口大小为3的核函数
        gauss3SmoothImage<<<gridSize, blockSize,0,stream[i]>>>(devicefrimg[i], 
                                                   deviceoutimg[i],
                                                   roiLocatX, realLocatY[i],
                                                   smWidth,realHeight[i],
                                                   devicemask[i],mask);
        // 核函数出错
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }
            
        break;
    }
    }


  }

  

  for(int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        //Wait for all operations to finish
        cudaStreamSynchronize(stream[i]);
        
        cudaStreamDestroy(stream[i]);
        cudaFree(devicefrimg[i].d_imgData);
        cudaFree(deviceoutimg[i].d_imgData);
    }
    

    return NO_ERROR;



}


 
 


  __host__ ImageCuda* GaussSmoothImage::imageCut(
        Image *img,
        int *realheight,
        int *realLocatY,
        int roiLocatY,
        int roiHeight
    )
 {
    int deviceCount=1;
    cudaGetDeviceCount(&deviceCount);

    //有两个输入一个输出，所以需要3个ImgaeCuda数组
    ImageCuda* deviceimg;
    deviceimg = new ImageCuda[deviceCount];
    ImageCuda* imgCud = IMAGE_CUDA(img); 
    size_t pitch = imgCud->pitchBytes;
    int nowHeight = 0; 
    for(int i=0;i<deviceCount;++i){
        // cudaSetDevice(i);

        //为成员变量赋值
        if (nowHeight+(img->height)/deviceCount<roiLocatY){
            realheight[i] = 0;
            realLocatY[i]=0;
        }
        else if (nowHeight+(img->height)/deviceCount>roiLocatY+roiHeight){
               realheight[i] = roiLocatY+roiHeight-nowHeight;
        }else {
            realheight[i] = (img->height)/deviceCount;
        }
        if (nowHeight>=roiLocatY){
            realLocatY[i] = 0;
        }
        else if(nowHeight+(img->height)/deviceCount>=roiLocatY){
            realLocatY[i]= roiLocatY - nowHeight;
        }
        nowHeight =nowHeight + (img->height)/deviceCount;
        deviceimg[i].imgMeta.width = img->width;
        deviceimg[i].imgMeta.height = (img->height)/deviceCount;
        deviceimg[i].pitchBytes = pitch;
        deviceimg[i].imgMeta.imgData = img->imgData + i*deviceimg[i].imgMeta.width*
                                 deviceimg[i].imgMeta.height;
    }
    return deviceimg;
}


