// CurveFluctuPropers.cu
// 创建人：邱孝兵

#include "CurveFluctuPropers.h"
#include  "ErrorCode.h"


// 宏 DELETE_CURVEFLUCTUPROPERS_HOST（删除 Host 端曲线波动特征）
// 根据给定的 CurveFluctuPropers 分别清除它在主机端包含的五个指针空间
// 所占用的内存
#define DELETE_CURVEFLUCTUPROPERS_HOST(cfp) do {              \
    if (((CurveFluctuPropers *)(cfp))->maxFluctu != NULL)     \
        delete [] ((CurveFluctuPropers *)(cfp))->maxFluctu;   \
    if (((CurveFluctuPropers *)(cfp))->maxFluctuX != NULL)    \
        delete [] ((CurveFluctuPropers *)(cfp))->maxFluctuX;  \
    if (((CurveFluctuPropers *)(cfp))->maxFluctuY != NULL)    \
        delete [] ((CurveFluctuPropers *)(cfp))->maxFluctuY;  \
} while (0)

// 宏 DELETE_CURVEFLUCTUPROPERS_DEVICE（删除 Device 端曲线波动特征）
// 根据给定的 CurveFluctuPropers 分别清除它在设备端包含的五个指针空间
// 所占用的内存
#define DELETE_CURVEFLUCTUPROPERS_DEVICE(cfp) do {            \
    if (((CurveFluctuPropers *)(cfp))->maxFluctu != NULL)     \
        cudaFree(((CurveFluctuPropers *)(cfp))->maxFluctu);   \
    if (((CurveFluctuPropers *)(cfp))->maxFluctuX != NULL)    \
        cudaFree(((CurveFluctuPropers *)(cfp))->maxFluctuX);  \
    if (((CurveFluctuPropers *)(cfp))->maxFluctuY != NULL)    \
        cudaFree(((CurveFluctuPropers *)(cfp))->maxFluctuY);  \
} while (0)


// Host 静态方法：makeAtHost（在 Host 内存中构件数据）
__host__ int CurveFluctuPropersBasicOp::makeAtHost(CurveFluctuPropers 
        *cfp, int maxFluctuNum)
{
    // 检查输入波动特征是否为 NULL
    if (cfp == NULL)
        return NULL_POINTER;

    // 检查参数的合法性
    if (maxFluctuNum < 1)
        return INVALID_DATA;

    // 为目标曲线波动特征申请 Host 内存空间
    cfp->maxFluctu = new int[maxFluctuNum];
    cfp->maxFluctuX = new int[maxFluctuNum];
    cfp->maxFluctuY = new int[maxFluctuNum];

    // 对于波动特征的元数据进行赋值
    cfp->maxFluctuNum = maxFluctuNum;

    // 处理完毕，返回
    return NO_ERROR;
}

// Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构件数据）
__host__ int CurveFluctuPropersBasicOp::makeAtCurrentDevice(CurveFluctuPropers 
        *cfp, int maxFluctuNum)
{
    // 检查输入波动特征是否为 NULL
    if (cfp == NULL)
        return NULL_POINTER;

    // 检查参数的合法性
    if (maxFluctuNum < 1)
        return INVALID_DATA;

    cudaError_t cuerrcode;  // CUDA Error Code

    // 在当前的 Device 上申请存储 maxFluctu 的空间    
    cuerrcode = cudaMalloc((void **)&(cfp->maxFluctu), 
                           maxFluctuNum * sizeof(int));
    if (cuerrcode != cudaSuccess) {
        DELETE_CURVEFLUCTUPROPERS_DEVICE(cfp);
        return CUDA_ERROR;
    }

    // 在当前的 Device 上申请存储 maxFluctuX 的空间
    cuerrcode = cudaMalloc((void **)&(cfp->maxFluctuX), 
                           maxFluctuNum * sizeof(int));
    if (cuerrcode != cudaSuccess) {
        DELETE_CURVEFLUCTUPROPERS_DEVICE(cfp);
        return CUDA_ERROR;
    }

    // 在当前的 Device 上申请存储 maxFluctuY 的空间
    cuerrcode = cudaMalloc((void **)&(cfp->maxFluctuY), 
                           maxFluctuNum * sizeof(int));
    if (cuerrcode != cudaSuccess) {
        DELETE_CURVEFLUCTUPROPERS_DEVICE(cfp);
        return CUDA_ERROR;
    }

    // 对于波动特征的元数据进行赋值
    cfp->maxFluctuNum = maxFluctuNum;
    
    // 返回
    return NO_ERROR;
}

// Host 静态方法：copyToHost（将当前 Device 内存中数据拷贝到 Host 端）
__host__ int CurveFluctuPropersBasicOp::copyToHost(CurveFluctuPropers *srccfp,
        CurveFluctuPropers *dstcfp)
{
    // 检查输入参数是否为 NULL
    if (srccfp == NULL || dstcfp == NULL)
        return NULL_POINTER;

    // 如果源曲线特征为空，则不进行任何操作，直接报错
    if (srccfp->maxFluctuNum == 0)
        return INVALID_DATA;

    // 将目标曲线特征单值数据设为源曲线特征的单值数据
    dstcfp->smNieghbors = srccfp->smNieghbors;
    dstcfp->maxFluctuNum = srccfp->maxFluctuNum;
    dstcfp->aveFluctu = srccfp->aveFluctu;
    dstcfp->xyAveFluctu = srccfp->xyAveFluctu; 

    // 为目标曲线波动特征申请 Host 内存空间
    dstcfp->maxFluctu = new int[dstcfp->maxFluctuNum];
    dstcfp->maxFluctuX = new int[dstcfp->maxFluctuNum];
    dstcfp->maxFluctuY = new int[dstcfp->maxFluctuNum];

    cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

    // 拷贝 maxFluctu 
    cuerrcode = cudaMemcpy(dstcfp->maxFluctu, srccfp->maxFluctu,       
                           srccfp->maxFluctuNum *  sizeof (int),
                           cudaMemcpyDeviceToHost);
    
    // 判断是否拷贝成功
    if (cuerrcode != cudaSuccess) {
        // 如果拷贝失败，清除申请的空间，并返回报错
        DELETE_CURVEFLUCTUPROPERS_HOST(dstcfp);
        return CUDA_ERROR;
    }

    // 拷贝 maxFluctuX 
    cuerrcode = cudaMemcpy(dstcfp->maxFluctuX, srccfp->maxFluctuX,       
                           srccfp->maxFluctuNum *  sizeof (int),
                           cudaMemcpyDeviceToHost);
    
    // 判断是否拷贝成功
    if (cuerrcode != cudaSuccess) {
        // 如果拷贝失败，清除申请的空间，并返回报错
        DELETE_CURVEFLUCTUPROPERS_HOST(dstcfp);
        return CUDA_ERROR;
    }

    // 拷贝 maxFluctuY 
    cuerrcode = cudaMemcpy(dstcfp->maxFluctuY, srccfp->maxFluctuY,       
                           srccfp->maxFluctuNum *  sizeof (int),
                           cudaMemcpyDeviceToHost);
    
    // 判断是否拷贝成功
    if (cuerrcode != cudaSuccess) {
        // 如果拷贝失败，清除申请的空间，并返回报错
        DELETE_CURVEFLUCTUPROPERS_HOST(dstcfp);
        return CUDA_ERROR;
    }
  
    // 处理完毕退出
    return NO_ERROR;
}

// Host 静态方法：copytToCurrentDevice 方法（将 Host 中的数据拷贝到 device 端）
 __host__ int copyToCurrentDevice(CurveFluctuPropers *srccfp, 
        CurveFluctuPropers *dstcfp)
{
    // 检查输入参数是否为 NULL
    if (srccfp == NULL || dstcfp == NULL)
        return NULL_POINTER;

    // 如果源曲线特征为空，则不进行任何操作，直接报错
    if (srccfp->maxFluctuNum == 0)
        return INVALID_DATA;

    // 将目标曲线特征单值数据设为源曲线特征的单值数据
    dstcfp->smNieghbors = srccfp->smNieghbors;
    dstcfp->maxFluctuNum = srccfp->maxFluctuNum;
    dstcfp->aveFluctu = srccfp->aveFluctu;
    dstcfp->xyAveFluctu = srccfp->xyAveFluctu;

    cudaError_t cuerrcode;  // CUDA ERROR CODE 

    // 为 maxFluctu 申请设备端内存    
    cuerrcode = cudaMalloc((void **)&(dstcfp->maxFluctu), 
                           dstcfp->maxFluctuNum * sizeof(int));

    // 如果申请失败返回 CUDA_ERROR
    if (cuerrcode != cudaSuccess) {
        DELETE_CURVEFLUCTUPROPERS_DEVICE(dstcfp);
        return CUDA_ERROR;
    }

    // 拷贝 maxFluctu 数据到设备端内存
    cuerrcode = cudaMemcpy(dstcfp->maxFluctu, srccfp->maxFluctu,       
                           srccfp->maxFluctuNum *  sizeof (int),
                           cudaMemcpyHostToDevice);

    // 如果拷贝失败返回 CUDA_ERROR
    if (cuerrcode != cudaSuccess) {
        DELETE_CURVEFLUCTUPROPERS_DEVICE(dstcfp);
        return CUDA_ERROR;
    }

    // 为 maxFluctuX 申请设备端内存    
    cuerrcode = cudaMalloc((void **)&(dstcfp->maxFluctuX), 
                           dstcfp->maxFluctuNum * sizeof(int));

    // 如果申请失败返回 CUDA_ERROR
    if (cuerrcode != cudaSuccess) {
        DELETE_CURVEFLUCTUPROPERS_DEVICE(dstcfp);
        return CUDA_ERROR;
    }

    // 拷贝 maxFluctuX 数据到设备端内存
    cuerrcode = cudaMemcpy(dstcfp->maxFluctuX, srccfp->maxFluctuX,       
                           srccfp->maxFluctuNum *  sizeof (int),
                           cudaMemcpyHostToDevice);

    // 如果拷贝失败返回 CUDA_ERROR
    if (cuerrcode != cudaSuccess) {
        DELETE_CURVEFLUCTUPROPERS_DEVICE(dstcfp);
        return CUDA_ERROR;
    }

    // 为 maxFluctuY 申请设备端内存    
    cuerrcode = cudaMalloc((void **)&(dstcfp->maxFluctuY), 
                           dstcfp->maxFluctuNum * sizeof(int));

    // 如果申请失败返回 CUDA_ERROR
    if (cuerrcode != cudaSuccess) {
        DELETE_CURVEFLUCTUPROPERS_DEVICE(dstcfp);
        return CUDA_ERROR;
    }

    // 拷贝 maxFluctuY 数据到设备端内存
    cuerrcode = cudaMemcpy(dstcfp->maxFluctuY, srccfp->maxFluctuY,       
                           srccfp->maxFluctuNum *  sizeof (int),
                           cudaMemcpyHostToDevice);

    // 如果拷贝失败返回 CUDA_ERROR
    if (cuerrcode != cudaSuccess) {
        DELETE_CURVEFLUCTUPROPERS_DEVICE(dstcfp);
        return CUDA_ERROR;
    }
  
    // 操作完毕返回
    return NO_ERROR;
}

// Host 静态方法：deleteFromHost（销毁 Host 端曲线波动特征属性）
__host__ int deleteFromHost(CurveFluctuPropers *srccfp)
{
    // 检查输入参数是否为 NULL，如果是直接返回
    if (srccfp == NULL) 
        return NULL_POINTER;

    // 调用宏删除 Host 端曲线波动特征属性
    DELETE_CURVEFLUCTUPROPERS_HOST(srccfp);

    return NO_ERROR;
}

// Host 静态方法：deleteFromDevice（销毁当前设备端曲线波动特征属性）
__host__ int deleteFromCurrentDevice(CurveFluctuPropers *srccfp )
{
    // 检查输入参数是否为 NULL，如果是直接返回
    if (srccfp == NULL) 
        return NULL_POINTER;

    // 调用宏删除 Device 端曲线波动特征属性
    DELETE_CURVEFLUCTUPROPERS_DEVICE(srccfp);

    return NO_ERROR;
}

    