// FeatureVecArray.cu
// 创建人：邱孝兵

#include "FeatureVecArray.h"
#include  "ErrorCode.h"


// 宏 DELETE_FEATUREVECARRAY_HOST （删除 Host 端特征向量）
// 根据给定的 FeatureVecArray 分别清除它 对应 x, y, CV, SD, NC
// 五个数组所占的内存
#define DELETE_FEATUREVECARRAY_HOST(featurevecarray) do {   \
    delete  [] ((FeatureVecArray *)(featurevecarray))->x;   \
    delete  [] ((FeatureVecArray *)(featurevecarray))->y;   \
    delete  [] ((FeatureVecArray *)(featurevecarray))->CV;  \
    delete  [] ((FeatureVecArray *)(featurevecarray))->SD;  \
    delete  [] ((FeatureVecArray *)(featurevecarray))->NC;  \
} while(0)


// Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
__host__ int FeatureVecArrayBasicOp::makeAtCurrentDevice(
        FeatureVecArray *fvp, size_t count)
{
    // 检查输入特征向量组是否为NULL
    if (fvp == NULL)
        return NULL_POINTER;

    // 检查给定的特征向量组长度
    if (count < 1)
        return INVALID_DATA;

    // 在当前的 Device 上申请存储指定坐特征向量 X 坐标所需要的内存空间
    cudaError_t cuerrcode;
    cuerrcode = cudaMalloc((void **)(&fvp->x), count * sizeof (int));
    if (cuerrcode != cudaSuccess) {
        fvp->x = NULL;
        return CUDA_ERROR;
    }

    // 在当前的 Device 上申请存储指定坐特征向量 Y 坐标所需要的内存空间
    cuerrcode = cudaMalloc((void **)(&fvp->y), count * sizeof (int));
    if (cuerrcode != cudaSuccess) {
        fvp->y = NULL;
        return CUDA_ERROR;
    }

    // 在当前的 Device 上申请存储指定坐特征向量 CV 所需要的内存空间
    cuerrcode = cudaMalloc((void **)(&(fvp->CV)), count * sizeof (float));
    if (cuerrcode != cudaSuccess) {
        fvp->CV = NULL;
        return CUDA_ERROR;
    }

    // 在当前的 Device 上申请存储指定坐特征向量 SD 所需要的内存空间
    cuerrcode = cudaMalloc((void **)(&(fvp->SD)), count * sizeof (float));
    if (cuerrcode != cudaSuccess) {
        fvp->SD = NULL;
        return CUDA_ERROR;
    }

    // 在当前的 Device 上申请存储指定坐特征向量 NC 所需要的内存空间
    cuerrcode = cudaMalloc((void **)(&(fvp->NC)), count * sizeof (float));
    if (cuerrcode != cudaSuccess) {
        fvp->NC = NULL;
        return CUDA_ERROR;
    }

    // 修改模板的元数据
    fvp->count = count;

    // 处理完毕，退出
    return NO_ERROR;
}

// Host 静态方法：copyToHost（将当前 Device 内存中数据拷贝到 Host 端）
__host__ int FeatureVecArrayBasicOp::copyToHost(FeatureVecArray *srcfvp, 
        FeatureVecArray *dstfvp)
{
    // 检查输入特征向量是否为 NULL
    if (srcfvp == NULL)
        return NULL_POINTER;
      
    // 如果源特征向量组为空，则不进行任何操作，直接报错
    if (srcfvp->count == 0)
        return INVALID_DATA;
    
    // 将目标特征向量组的尺寸修改为源特征向量组的尺寸。
    dstfvp->count = srcfvp->count;

    // 为目标特征向量组申请空间
    dstfvp->x = new int[dstfvp->count];
    dstfvp->y = new int[dstfvp->count];
    dstfvp->CV = new float[dstfvp->count];
    dstfvp->SD = new float[dstfvp->count];
    dstfvp->NC = new float[dstfvp->count];
    
    cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

    // 拷贝 x 坐标
    cuerrcode = cudaMemcpy(dstfvp->x, srcfvp->x,                                
                           srcfvp->count * sizeof (int),
                           cudaMemcpyDeviceToHost);
    
    // 判断是否拷贝成功
    if (cuerrcode != cudaSuccess) {
        // 如果拷贝失败，清除申请的空间，并返回报错
        DELETE_FEATUREVECARRAY_HOST(dstfvp);
        return CUDA_ERROR;
    }
    
    // 拷贝 y 坐标
    cuerrcode = cudaMemcpy(dstfvp->y, srcfvp->y,                                
                           srcfvp->count * sizeof (int),
                           cudaMemcpyDeviceToHost);
    
    // 判断是否拷贝成功
    if (cuerrcode != cudaSuccess) {
        // 如果拷贝失败，清除申请的空间，并返回报错
        DELETE_FEATUREVECARRAY_HOST(dstfvp);
        return CUDA_ERROR;
    }

    // 拷贝特征值 CV
    cuerrcode = cudaMemcpy(dstfvp->CV, srcfvp->CV,                    
                           srcfvp->count * sizeof (float),
                           cudaMemcpyDeviceToHost);
    
    // 判断是否拷贝成功
    if (cuerrcode != cudaSuccess) {
        // 如果拷贝失败，清除申请的空间，并返回报错
        DELETE_FEATUREVECARRAY_HOST(dstfvp);
        return CUDA_ERROR;
    }

    // 拷贝特征值 SD
    cuerrcode = cudaMemcpy(dstfvp->SD, srcfvp->SD,                        
                           srcfvp->count * sizeof (float),
                           cudaMemcpyDeviceToHost);
    
    // 判断是否拷贝成功
    if (cuerrcode != cudaSuccess) {
        // 如果拷贝失败，清除申请的空间，并返回报错
        DELETE_FEATUREVECARRAY_HOST(dstfvp);
        return CUDA_ERROR;
    }

    // 拷贝特征值 NC
    cuerrcode = cudaMemcpy(dstfvp->NC, srcfvp->NC,      
                           srcfvp->count * sizeof (float),
                           cudaMemcpyDeviceToHost);
    
    // 判断是否拷贝成功
    if (cuerrcode != cudaSuccess) {
        // 如果拷贝失败，清除申请的空间，并返回报错
        DELETE_FEATUREVECARRAY_HOST(dstfvp);
        return CUDA_ERROR;
    }
    
    // 处理完毕，退出
    return NO_ERROR;
}

// Host 静态方法：deleteFeatureVecArray（删除不再需要的特征向量组）
__host__ int FeatureVecArrayBasicOp::deleteFeatureVecArray(FeatureVecArray *fvp)
{
    // 检查输入特征向量组是否为NULL
    if (fvp == NULL)
        return NULL_POINTER;

    // 检查是否有数据
    if (fvp->count == 0)
    {
        // do nothing
    } else {
        // 释放 Device 内存
        cudaFree(fvp->x);
        cudaFree(fvp->y);
        cudaFree(fvp->CV);
        cudaFree(fvp->SD);
        cudaFree(fvp->NC);
    }

    return NO_ERROR;
}

    

