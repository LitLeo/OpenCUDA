// GaussianSmoothxy.h
// 创建人：高新凯
//
// 曲线高斯平滑
// 功能说明：定义了高斯平滑类GaussSmoothXY。
// 用于曲线的高斯平滑
//
// 修订历史：
// 2013年11月14日（高新凯）
//     初始版本。
// 2013年12月25日（高新凯）
//     优化了并行策略
//
// 2014年10月11日（高新凯）
//    修改了文件编码格式
//
#ifndef __GAUSSIANSMOOTNXY_H__
#define __GAUSSIANSMOOTNXY_H__

#include "ErrorCode.h"
#include "Curve.h"

class GaussSmoothXY{

protected:

    // 成员变量，存储高斯平滑辅助数据。类型为GPGPU GLOBAL memory
    int* ringCordiXY; 
    float* gSmCordiXY;
    
public:

    // 默认构造函数GaussSmoothXY，将成员变量都初始化为NULL
     __host__ __device__ 
     GaussSmoothXY(){
        ringCordiXY = NULL;
        gSmCordiXY = NULL;      
    }
    
    // 析构函数，防止内存泄漏
    __host__ __device__ 
    ~GaussSmoothXY(){
     
        // 释放ringCordiXY
        if (ringCordiXY != NULL) {
            cudaFree(ringCordiXY);  
        }  
        
        // 释放gSmCordiXY
        if (gSmCordiXY != NULL) {
            cudaFree(gSmCordiXY);  
        }  
        
    }
    
    // Host 成员方法：gaussSmoothCurveXY（curve高斯平滑函数）
    // 高斯平滑HOST侧函数，输入curve，根据平滑窗口大小进行平滑
    // 平滑窗口大小smWindowSize应在3、5、7、9、11中选取
    __host__ int                  // 返回值：函数是否正确执行，若函数正确执行，返回
                                  // NO_ERROR。
    gaussSmoothCurveXY(
                Curve *curve,     // 待平滑曲线
                int smWindowSize  // 平滑窗口大小
    ) ;
    
    // Host 成员方法：gaussSmoothXY（高斯平滑计算函数）
    // 对曲线进行高斯平滑
    __host__ int                 // 返回值：函数是否正确执行，若函数正确执行，返回
                                 // NO_ERROR。
    gaussSmoothXY(
            int n,               // 曲线上点的数量
            int* origiCordiXY,   // 原始的坐标
            bool closed,         // 曲线是否是闭合的
            float* gSmCordiXY,   // 平滑后的坐标
            int smWindowSize     // 平滑窗口大小
    );  
    
};

#endif
