// SmoothVector.h
// 创建人：邱孝兵
//
// 平滑向量的生成（Smooth Vector）
// 功能说明：以 FeatureVecCalc 中计算出来的特征向量的第一个为初始中心，
//           根据给定的参数对其进行均值偏移操作,从而得到一个收束点，由于每个像素
//           都可以得到一个收束点，最后我们得到了一个收束点集，即为所求。
//
// 修订历史：
// 2012年10月16日（邱孝兵）
//     初始版本。
// 2012年10月25日（邱孝兵）
//     将求初始特征向量这部分功能分离到 FeatureVecCalc 类中。
// 2012年11月22日（邱孝兵）
//     修改一些格式问题，修改构造函数赋值方式
// 2012年12月21日（邱孝兵）
//     根据河边需求的进一步确认，修改了 host 函数的参数，增加输出参数
// 2012年12月28日（邱孝兵）
//     修改部分格式问题

#include "Image.h"
#include "ErrorCode.h"
#include "FeatureVecCalc.h"
#include "FeatureVecArray.h"
#include "Matrix.h"

#ifndef __SMOOTHVECTOR_H__
#define __SMOOTHVECTOR_H__


// 类：SmoothVector
// 继承自：无
// 以 FeatureVecCalc 中计算出来的特征向量的第一个为初始中心，根据给定的
// 参数对其进行均值偏移操作,从而得到一个收束点，由于每个像素都可以得到
// 一个收束点，最后我们得到了一个收束点集，即为所求。
class SmoothVector {
    
protected:

    // 成员变量：relativeWeight（相对权重）
    // 在均值偏移（mean shift）的过程中，space 和 range 的相对权重
    float relativeWeight;
       
    // 成员变量：shiftArraySize（控制 shift 过程的数组的大小）
    // 在均值偏移（mean shift）的过程中，处理操作序列有三个长度相同的控制
    // 数组，该变量表示这三个数组的大小
    int shiftArraySize;
   
    // 成员变量：spaceBands（每次 shift 的邻域宽度）
    // 均值偏移过程中，操作序列每次邻域选择的大小为元素的数组。
    int *spaceBands;
   
    // 成员变量：rangeBands（每次 shift 的外部参数）
    // 均值偏移过程中，操作序列每次运算的参数为元素的数组。
    float *rangeBands;
    
    // 成员变量：shiftCounts（每组 band pair shift 的次数）
    // 均值偏移过程中，每组 band pair 进行反复迭代计算的次数为
    // 元素的数组，即原需求文档中的 iter 。
    int *shiftCounts;
   
public:
    
    // 构造函数：SmoothVector
    // 无参数版本的构造函数，所有成员变量均初始化为默认值
    __host__ __device__
    SmoothVector()
    {
        // 给各个成员变量赋默认值
        this->relativeWeight = 0.1f;
        this->shiftArraySize = 0;
        this->spaceBands = NULL;
        this->rangeBands = NULL;
        this->shiftCounts = NULL;
    }
    
    // 构造函数：SmoothVector
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    SmoothVector(
            float relativeWeight,  // 相对权重
            int shiftArraySize,    // 操作控制数组大小
            int *spaceBands,       // shift 邻域大小
            float *rangeBands,     // shift 参数
            int *shiftCounts       // shift 重复次数
    ) {
        // 根据参数给各个成员变量赋值
        this->relativeWeight = 0.1f;
        this->shiftArraySize = 0;
        this->spaceBands = NULL;
        this->rangeBands = NULL;
        this->shiftCounts = NULL;

        // 根据参数列表中的值设定成员变量的初值
        this->setRelativeWeight(relativeWeight);
        this->setShiftArraySize(shiftArraySize);
        this->setSpaceBands(spaceBands, shiftArraySize);
        this->setRangeBands(rangeBands, shiftArraySize);
        this->setShiftCounts(shiftCounts, shiftArraySize);
    }

    // 成员方法：getRelativeWeight （获取 relativeWeight 的值）
    // 获取成员变量 relativeWeight 的值
    __host__ __device__ float  // 返回值：成员变量 relativeWeight 的值
    getRelativeWeight() const
    {
        // 返回成员变量 relativeWeight 的值
        return relativeWeight;
    }

    // 成员方法：setRelativeWeight （设置 relativeWeight 的值）
    // 设置成员变量 relativeWeight 的值
    __host__ __device__ int       // 返回值：函数是否正确执行，若函数正确执行，
                                  // 返回 NO_ERROR。
    setRelativeWeight(
            float relativeWeight  // 相对权重的大小
    ) {
        // 判断传入 relativeWeight 是是0和1之间的小数
        // 如果不是返回 INVALID_DATA 。
        if (relativeWeight <= 0.0 || relativeWeight > 1.0)
            return INVALID_DATA;

        // 设置成员变量 relativeWeight 的值
        this->relativeWeight = relativeWeight;

        return NO_ERROR;
    }
    
    // 成员方法：getShiftArraySize （获取 shiftArraySize 的值）
    // 获取成员变量 shiftArraySize 的值
    __host__ __device__ int  // 返回值：成员变量 shiftArraySize 的值 
    getShiftArraySize() const
    {
        // 返回成员变量 shiftArraySize 的值
        return shiftArraySize;
    }
    
    // 成员方法：setShiftArraySize （设置 shiftArraySize 的值）
    // 设置成员变量 shiftArraySize 的值，在这里有个约定，就是先设定数组大小，再
    // 去设定各个操作数组，因为在设定操作数组的时候需要检查新设定的操作数组大小
    // 是否满足要求这样在就必须保证数组大小提前设定一个正确的值
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR 。
    setShiftArraySize(
            int shiftArraySize  // 操作控制数组大小
    ) {
        // 如果新的 arraySize 比 原来的 arraySize 大，
        // 则将三个数组的指针设置为 NULL ，防止出现不一致状态
        if (this->shiftArraySize < shiftArraySize) {
            this->spaceBands = NULL;
            this->rangeBands = NULL;
            this->shiftCounts = NULL;
        }

        // 设置成员变量 shiftArraySize 的值
        this->shiftArraySize = shiftArraySize;
               
        return NO_ERROR;
    }
    
    // 成员方法：getSpaceBands（获取 spaceBands 的值）
    // 获取成员变量 spaceBands 的值
    __host__ __device__ int *  // 返回值：成员变量 spaceBands 的值
    getSpaceBands() const
    {
        // 返回成员变量 spaceBands 的值 
        return this->spaceBands;
    }
    
    // 成员方法：setSpaceBands（设置 spaceBands 的值）
    // 设置成员变量 spaceBands 的值，同时传入的参数还有该数组的长度
    // shiftArraySize 设置的时候会检查该值，和当前的数组长度是否匹配，若不匹配则
    // 不设置
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR 。
    setSpaceBands(
            int *spaceBands,    // shift 邻域大小
            int shiftArraySize  // 操作控制数组大小
     ) {
        // 判断传入 shiftArraySize 和当前 shiftArraySize 是否匹配
        // 如果不匹配返回 INVALID_DATA 。
        if (this->shiftArraySize != shiftArraySize)
            return INVALID_DATA;
       
        // 设置 spaceBands
        this->spaceBands = spaceBands;
                        
        return NO_ERROR; 
    }
     
    // 成员方法：getRangeBands（获取 rangeBands 的值）
    // 获取成员变量 rangeBands 的值
    __host__ __device__ float *  // 返回值：成员变量 RangeBands 的值
    getRangeBands() const
    {
        // 返回成员变量 rangeBands 的值
        return this->rangeBands;
    }
    
    // 成员方法：setRangeBands（设置 rangeBands 的值）
    // 设置成员变量 rangeBands 的值，同时传入的参数还有该数组的长度 
    // shiftArraySize 设置的时候会检查该值，和当前的数组长度是否匹配，若不匹配则
    // 不设置
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR 。
    setRangeBands(
            float *rangeBands,  // shift 参数数组
            int shiftArraySize  // 操作控制数组大小
     ) {
        // 判断传入 shiftArraySize 和当前 shiftArraySize 是否匹配
        // 如果不匹配返回 INVALID_DATA 。
        if (this->shiftArraySize != shiftArraySize)
            return INVALID_DATA;

        // 设置 rangeBands 
        this->rangeBands = rangeBands;
                
        return NO_ERROR;
    }
            
     
    // 成员方法：getShiftCounts（获取 shiftCounts 的值）
    // 获取成员变量 shiftCounts 的值
    __host__ __device__ int *  // 返回值：成员变量 shiftCounts 的值
    getShiftCounts() const
    {
        // 返回成员变量 shiftCounts 的值
        return this->shiftCounts;
    }
    
    // 成员方法：setShiftCounts（设置 shiftCounts 的值）
    // 设置成员变量 shiftCounts 的值，同时传入的参数还有该数组的长度 
    // shiftArraySize 设置的时候会检查该值，和当前的数组长度是否匹配，若不匹配则
    // 不设置
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR 。
    setShiftCounts(
            int *shiftCounts,   // shift 操作次数
            int shiftArraySize  // 操作控制数组大小
    ) {
        // 判断传入 shiftArraySize 和当前 shiftArraySize 是否匹配
        // 如果不匹配返回 INVALID_DATA 。
        if (this->shiftArraySize != shiftArraySize)
            return INVALID_DATA;

        // 设置 shiftCounts
        this->shiftCounts = shiftCounts;
                
        return NO_ERROR;
    }
        
    // 成员方法：meanshift（均值偏移）
    // 该方法以 FeatureVecCalc 中计算出来的特征向量为初始输入值，针对每一个像素
    // 根据给的外部参数 spaceBands（shift 邻域大小）、rangeBands（shift 参数）、
    // shiftCounts（shift 重复次数）进行指定次数的均值偏移得到一个收束点。
    __host__ int                                  // 返回值：函数是否正确执行，
                                                  // 若函数正确执行，返回 
                                                  // NO_ERROR 。
    meanshift(
            FeatureVecArray *infeaturevecarray,   // 输入特征向量
            FeatureVecArray *outfeaturevecarray,  // 输出特征向量
            int width,                            // 图像的宽度
            int height                            // 图像的高度
    );
}; 
     
#endif
