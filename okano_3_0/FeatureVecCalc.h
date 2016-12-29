// FeatureVecCalc.h
// 创建人：邱孝兵
//
// 特征向量的计算（FeatureVecCalc）
// 功能说明：计算指定坐标集所规定的图像范围内的各像素的一个特征向量
//
// 修订历史：
// 2012年10月25日 （邱孝兵）
//     初始版本。
// 2012年11月02日 （邱孝兵）
//     增加了 sortNeighbor，calAvgPixel，calPixelSd 三个 device 函数
// 2012年11月04日 （邱孝兵）
//     增加了 sortFeatureValue，calFeatureVecCalc 两个 host 函数
// 2012年11月22日 （邱孝兵）
//     修改一些格式问题，修改构造函数赋值方式
// 2012年11月23日 （邱孝兵）
//     将类名由 FeatureVecCalc 修改为 FeatureVecCalc
// 2012年12月26日 （邱孝兵）
//     修改了三处程序逻辑错误。
// 2012年12月31日 （邱孝兵）
//     删除原有的 host 端排序方法，并换成快速排序，大幅度提升了性能

#include "Image.h"
#include "ErrorCode.h"
#include "FeatureVecArray.h"
#include "CoordiSet.h"

#ifndef __FEATUREVECTOR_H__
#define __FEATUREVECTOR_H__

// 定义像素范围
#define PIXELRANGE 256


// 类：FeatureVecCalc
// 继承自：无
// 根据设定的外部参数计算指定坐标集所规定的图像范围内的各像素的一个特征向量
class FeatureVecCalc {
    
protected:
    
    // 成员变量：alpha（外部指定系数）
    // 计算三个特征值时需要用到的一个系数。
    float alpha;        
    
    // 成员变量：beta（外部指定系数）
    // 计算三个特征值时需要用到的一个系数。
    float beta;
    
    // 成员变量：neighborWidth（邻域宽度）
    // 在利用每个像素 n 邻域的点的像素计算该像素特征值的时候，指定的
    // 邻域宽度 n 。
    int  neighborWidth;
   
    // 成员变量：pitch（间距）
    // 在计算每个像素的最大非共起系数 NC 的时候需要用到的一个外部参数
    int pitch;   
   
public:
    
    // 构造函数：SmoothVector
    // 无参数版本的构造函数，所有成员变量均初始化为默认值
    __host__ __device__
    FeatureVecCalc()
    {
        // 无参数的构造函数，使用默认值初始化各个变量
        this->alpha = 0;
        this->beta = 0;
        this->neighborWidth = 1;
        this->pitch = 1;
    }
    
    // 构造函数：SmoothVector
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    FeatureVecCalc(
            float alpha,         // 外部参数阿尔法
            float beta,          // 外部参数贝塔
            int neighborwidth,   // 邻域宽度
            int pitch            // 间距
    ) {
        // 初始化各个变量
        this->alpha = alpha;
        this->beta = beta;
        this->neighborWidth = neighborwidth;
        this->pitch = pitch;
    }

    // 成员方法：getAlpha （获取 alpha 的值）
    // 获取成员变量  alpha 的值
    __host__ __device__ float  // 返回值：成员变量 alpha 的值
    getAlpha() const
    {
        // 返回成员变量 alpha 的值
        return this->alpha;
    }
    
    // 成员方法：setAlpha（设置 alpha 的值）
    // 设置成员变量 alpha 的值
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setAlpha(
            float alpha      // 外部参数 alpha
    ) {
        // 设置成员变量 beta 的值
        this->alpha = alpha;
        
        return NO_ERROR;
    }
    
    // 成员方法：getBeta （获取 beta 的值）
    // 获取成员变量  beta 的值
    __host__ __device__ float  // 返回值：成员变量 beta 的值
    getBeta() const
    {
        // 返回成员变量 beta 的值
        return this->beta;
    }
    
    // 成员方法：setBeta（设置 beta 的值）
    // 设置成员变量 beta 的值
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setBeta(
            float beta       // 外部参数 beta
    ) {
        // 设置成员变量 beta 的值
        this->beta = beta;
        
        return NO_ERROR;
    }
    
    // 成员方法：getNeighborWidth （获取 neighborWidth 的值）
    // 获取成员变量  neighborWidth 的值
    __host__ __device__ int  // 返回值：成员变量 neighborWidth 的值
    getNeighborWidth() const
    {
        // 返回成员变量 neighborWidth 的值
        return this->neighborWidth;
    }
    
    // 成员方法：setNeighborWidth（设置 neighborWidth 的值）
    // 设置成员变量 neighborWidth 的值
    __host__ __device__ int    // 返回值：函数是否正确执行，若函数正确执行，
                               // 返回NO_ERROR。
    setNeighborWidth(
            int neighborwidth  // 邻域大小
    ) {
        // 设置成员变量 neighborWidth 的值
        this->neighborWidth = neighborwidth;
        
        return NO_ERROR;
    }
    
    // 成员方法：getPitch （获取 pitch 的值）
    // 获取成员变量  pitch 的值
    __host__ __device__ int  // 返回值：成员变量 pitch 的值
    getPitch() const
    {
        // 返回成员变量 pitch 的值
        return this->pitch;
    }
    
    // 成员方法：setPitch（设置 pitch 的值）
    // 设置成员变量 pitch 的值
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setPitch(
            int pitch        // 间距
    ) {
        // 设置成员变量 pitch 的值
        this->pitch = pitch;
        
        return NO_ERROR;
    }
    
    // 成员方法：sortNeighbor（对邻域像素进行排序）
    // 对一个线性数组的像素点，根据像素值进行重新排序
    // 使用统计排序法可以实现线性时间排序
    __host__ __device__ int               // 返回值：函数是否正确执行，
                                          // 若函数正确执行，返回 NO_ERROR。
    sortNeighbor(
            unsigned char *pixels,        // 像素存储指针
            unsigned char *pixelssorted,  // 排序后的数组存储位置
            int *pixelcount,              // 像素统计
            int length                    // 数组长度          
    ) {
        int i, j;

        // 临时存储像素统计
        int tmpcount[PIXELRANGE];

        // 像素统计置0
        for (i = 0; i < PIXELRANGE; i++) {
            pixelcount[i] = 0;
            tmpcount[i] = 0;
        }

        // 统计像素值等于 i 的个数
        for (j = 0; j < length; j++) {
            pixelcount[pixels[j]]++;
            tmpcount[pixels[j]]++;
        }

        // 统计像素值小于或等于 i 的个数
        for (i = 1; i < PIXELRANGE; i++)
            tmpcount[i] = tmpcount[i] + tmpcount[i - 1];        

        // 将各个元素放到正确的位置
        for (j = length - 1; j >= 0; j--) {
            pixelssorted[tmpcount[pixels[j]] - 1] = pixels[j];
            tmpcount[pixels[j]]--;
        }
    
        return NO_ERROR;
    }

    // 成员方法：calAvgPixel（计算灰度平均值）
    // 对一个像素数组，在指定的下标范围内，计算其平均值
    __host__ __device__ float       // 计算得到的平均值
    calAvgPixel(
            unsigned char *pixels,  // 像素存储指针
            int low,                // 下标左值
            int high                // 下标右值
    ) {
        // 求和
        int sum = 0;
        for (int i = low; i < high; i++) 
            sum += pixels[i];        
        
        // 返回均值
        return sum * 1.0f / (high - low);
    }

    // 成员方法：calPixelSd（计算灰度标准差）
    // 对一个像素数组，根据其平均值计算标准差
    __host__ __device__ float       // 计算得到的标准差
    calPixelSd(
            unsigned char *pixels,  // 像素存储指针
            int length,             // 数组长度
            float cv                // 平均值
    ) {
        // 求和
        float sum = 0;
        for (int i = 0; i < length ; i++)
            sum += powf((pixels[i] - cv), 2);        

        // 返回标准差
        return sqrtf(sum / length);
    }      

    // 成员方法：calAvgFeatureValue（计算特征向量平均值）
    // 计算某一特征值数组在特定下标范围内的平均值
    __host__ float                // 返回值，计算得到的平均值
    calAvgFeatureValue(
            float *featurevalue,  // 特征值数组
            int minborder,        // 下标下限
            int maxborder         // 下标上限
    ) {
        float avgfeaturevalue = 0;
        for (int i = minborder; i < maxborder; i++)
            avgfeaturevalue += featurevalue[i];        

        // 返回平均值
        return avgfeaturevalue / (maxborder - minborder);
    }

    // 成员方法：calFeatureVector（计算特征向量）
    // 该方法用于计算指定坐标集所规定的图像范围内的各 PIXEL 的初始特征向量。
    // 具体可以分为三个步骤：
    // 1. 利用需求文档中给出的公式，计算每个像素的三个特征值：灰度中心値 float 
    // CV 、灰度値标准差 float SD 、最大灰度非共起系数 float NC 。
    // 2. 针对三个特征值的数组进行排序，取极值，求均值等操作。
    // 3. 根据上个步骤中，求得的极值、均值以及外部参数 α 、β 求出我们所需要的
    // 每个像素的初始特征向量。
    __host__ int                                 // 返回值：函数是否正确执行，
                                                 // 若函数正确执行返回 
                                                 // NO_ERROR。
    calFeatureVector(
            Image *inimg,                        // 输入图像
            CoordiSet *incoordiset,              // 输入坐标集
            FeatureVecArray *outfeaturevecarray  // 输出特征向量
    );   
}; 
     
#endif

