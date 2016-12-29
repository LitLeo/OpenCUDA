// SalientRegionDetect.h 
// 创建人：刘宇
//
// 显著区域检测（SalientRegionDetect）
// 功能说明：根据图像中的领域灰度值计算像素的显著性值，分割得到显著性区域，
//           用二值图形式显示结果。
//
// 修订历史：
// 2012年12月05日（刘宇）
//     初始版本
// 2013年02月24日（刘宇）
//     修正一些成员变量

#ifndef __SALIENTREGIONDETECT_H__
#define __SALIENTREGIONDETECT_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：SalientRegionDetect
// 继承自：无
// 首先计算 saliency map ，包括两种算法，一种是根据领域的灰度值差值排序并平均，
// 一种是根据高斯平滑计算，将两种方法的结果加权平均就得到最终的 saliency map，
// 然后分割 saliency map 得到多个显著性区域，最后根据区域平均显著性阈值和区域
// 面积，筛选出最终需要的显著性区域，并用二值化显示，即显著性区域设置为白色，
// 其它区域为黑色。
class SalientRegionDetect {

protected:

    // 成员变量：highPercent（数组的高位段）
    // 指定砍掉排序后数组的高位段的 highPercent，范围是 [0, 1]。
    float highPercent;

    // 成员变量：lowPercent（数组的低位段）
    // 指定砍掉排序后数组的低位段的 lowPercent，范围是 [0, 1]。
    float lowPercent;

    // 成员变量：radius（模版半径）
    // 模版半径数组，例如 radius = {5, 8, 13,......}。
    int *radius;

    // 成员变量：iterationSM1（radius 的迭代次数）
    // 改变模版半径 radius 的值，迭代计算的次数。
    int iterationSM1;
 
    // 成员变量：isSelect（筛选数组标识位）
    // 如果其值为 1，则通过 highPercent 和 lowPercent 筛选数组，否则不筛选。
    bool isSelect;

    // 成员变量：smoothWidth（平滑模版大小）
    // 模版半径数组，例如 smoothWidth = {5, 8, 13,......}。
    int *smoothWidth;

    // 成员变量：iterationSM2（iterationSM2 的迭代次数）
    // 改变模版半径 iterationSM2 的值，迭代计算的次数。
    int iterationSM2;

    // 成员变量：weightSM1（sm1 的权重）
    // saliency map1 的权重。
    float weightSM1;

    // 成员变量：weightSM2（sm2 的权重）
    // saliency map2 的权重。
    float weightSM2;

    // 成员变量：threshold（给定阈值）
    // 进行区域连通的给定值，当两个点满足八邻域关系，
    // 且灰度值之差的绝对值小于该值时，这两点属于同一区域。
    int threshold;

    // 成员变量：minRegion（区域最小面积的阈值）
    // 显著性区域面积的最小阈值。
    int minRegion;

    // 成员变量：maxRegion（区域最大面积的阈值）
    // 显著性区域面积的最大阈值。
    int maxRegion;

    // 成员变量：saliencyThred（区域平均显著性值的阈值大小）
    // 显著性区域平均显著性值的阈值大小。
    int saliencyThred;

public:
    // 构造函数：SalientRegionDetect
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    SalientRegionDetect()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->highPercent = 0.1;   // 数组的高位段默认为 0.1
        this->lowPercent = 0.3;    // 数组的低位段默认为 0.3
        this->radius = NULL;       // 模版半径默认为空
        this->iterationSM1 = 0;    // radius 迭代次数默认为 0
        this->isSelect = false;    // 筛选数组标识位默认为 false
        this->smoothWidth = NULL;  // 平滑模版大小默认为空
        this->iterationSM2 = 0;    // smoothWidth 迭代次数默认为 0
        this->weightSM1 = 0.5;     // SM1 的权重默认为 0.5
        this->weightSM2 = 0.5;     // SM2 的权重默认为 0.5
        this->threshold = 3;       // 给定阈值默认为0
        this->minRegion = 100;     // 区域最小面积的阈值大小默认为 100
        this->maxRegion = 500000;  // 区域最大面积的阈值大小默认为 500000
        this->saliencyThred = 1;   // 区域平均显著性值的阈值大小默认为 1
    }


    // 构造函数：SalientRegionDetect
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    SalientRegionDetect(
            float highpercent,  // 数组的高位段
            float lowpercent,   // 数组的低位段
            int *radius,        // 模版半径
            int iterationsm1,   // radius 迭代次数
            bool isselect,      // 筛选数组标识位
            int *smoothwidth,   // 平滑模版大小
            int iterationsm2,   // smoothWidth 迭代次数
            float weightsm1,    // SM1 的权重
            float weightsm2,    // SM2 的权重
            int threshold,      // 给定的标记阈值
            int minregion,      // 区域最小面积的阈值大小
            int maxregion,      // 区域最大面积的阈值大小
            int saliencythred   // 区域平均显著性值的阈值大小
    ) {
        // 使用默认值为类的各个成员变量赋值。
        this->highPercent = 0.1;   // 数组的高位段默认为 0.1
        this->lowPercent = 0.3;    // 数组的低位段默认为 0.3
        this->radius = NULL;       // 模版半径默认为空
        this->iterationSM1 = 0;    // radius 迭代次数默认为 0
        this->isSelect = false;    // 筛选数组标识位默认为 false
        this->smoothWidth = NULL;  // 平滑模版大小默认为空
        this->iterationSM2 = 0;    // smoothWidth 迭代次数默认为 0
        this->weightSM1 = 0.5;     // SM1 的权重默认为 0.5
        this->weightSM2 = 0.5;     // SM2 的权重默认为 0.5
        this->threshold = 3;       // 给定的标记阈值
        this->minRegion = 100;     // 区域最小面积的阈值大小默认为 100
        this->maxRegion = 500000;  // 区域最大面积的阈值大小默认为 500000
        this->saliencyThred = 1;   // 区域平均显著性值的阈值大小默认为 1

        // 根据参数列表中的值设定成员变量的初值
        setHighPercent(highpercent);
        setLowPercent(lowpercent);
        setRadius(radius);
        setIterationSM1(iterationsm1);
        setIsSelect(isselect);
        setSmoothWidth(smoothwidth);
        setIterationSM2(iterationsm2);
        setWeightSM1(weightsm1);
        setWeightSM2(weightsm2);
        setThreshold(threshold);
        setMinRegion(minregion);
        setMaxRegion(maxregion);
        setSaliencyThred(saliencythred);
    }
   
    // 成员方法：getHighPercent（获取数组的高位段）
    // 获取成员变量 highPercent 的值。
    __host__ __device__ float  // 返回值：成员变量 highPercent 的值
    getHighPercent() const
    {
        // 返回 highPercent 成员变量的值。
        return this->highPercent;
    }

    // 成员方法：setHighPercent（设置数组的高位段）
    // 设置成员变量 highPercent 的值。
    __host__ __device__ int    // 返回值：函数是否正确执行，若函数正确执
                               // 行，返回 NO_ERROR。
    setHighPercent(
            float highpercent  // 设定新的数组的高位段
    ) {
        // 将 highPercent 成员变量赋成新值
        this->highPercent = highpercent;

        return NO_ERROR;

    }

    // 成员方法：getLowPercent（获取数组的低位段）
    // 获取成员变量 lowPercent 的值。
    __host__ __device__ float  // 返回值：成员变量 lowPercent 的值
    getLowPercent() const
    {
        // 返回 lowPercent 成员变量的值。
        return this->lowPercent;
    }

    // 成员方法：setLowPercent（设置数组的低位段）
    // 设置成员变量 lowPercent 的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执
                              // 行，返回 NO_ERROR。
    setLowPercent(
            float lowpercent  // 设定新的数组的低位段
    ) {
        // 将 lowPercent 成员变量赋成新值
        this->lowPercent = lowpercent;

        return NO_ERROR;
    }

    // 成员方法：getRadius（获取模版半径）
    // 获取成员变量 radius 的值。
    __host__ __device__ int *  // 返回值：成员变量 radius 的值
    getRadius() const
    {
        // 返回 radius 成员变量的值。
        return this->radius;
    }

    // 成员方法：setRadius（设置模版半径）
    // 设置成员变量 radius 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setRadius(
            int *radius      // 设定新的模版半径
    ) {
        // 将 radius 成员变量赋成新值
        this->radius = radius;

        return NO_ERROR;
    }

    // 成员方法：getIterationSM1（获取 radius 迭代次数）
    // 获取成员变量 iterationSM1 的值。
    __host__ __device__ int  // 返回值：成员变量 iterationSM1 的值
    getIterationSM1() const
    {
        // 返回 iterationM1 成员变量的值。
        return this->iterationSM1;
    }

    // 成员方法：setIterationSM1（设置 radius 迭代次数）
    // 设置成员变量 iterationSM1 的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执
                              // 行，返回 NO_ERROR。
    setIterationSM1(
            int iterationsm1  // 设定新的迭代次数
    ) {
        // 将 iterationM1 成员变量赋成新值
        this->iterationSM1 = iterationsm1;

        return NO_ERROR;
    }

    // 成员方法：getIsSelect（获取 isSelect 筛选标识位）
    // 获取成员变量 isSelect 的值。
    __host__ __device__ bool  // 返回值：成员变量 isSelect 的值
    getIsSelect() const
    {
        // 返回 isSelect 成员变量的值。
        return this->isSelect;
    }
      

    // 成员方法：setIsSelect（设置 isSelect 筛选标识位）
    // 设置成员变量 isSelect 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setIsSelect(
            bool isselect    // 设定新的筛选标识位
    ) {
        // 将 isSelect 成员变量赋成新值
        this->isSelect = isselect;
        
        return NO_ERROR;
    }    

    // 成员方法：getSmoothWidth（获取平滑模版大小）
    // 获取成员变量 smoothWidth 的值。
    __host__ __device__ int *  // 返回值：成员变量 smoothWidth 的值
    getSmoothWidth() const
    {
        // 返回 smoothWidth 成员变量的值。
        return this->smoothWidth;
    }

    // 成员方法：setSmoothWidth（设置平滑模版大小）
    // 设置成员变量 smoothWidth 的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执
                              // 行，返回 NO_ERROR。
    setSmoothWidth(
            int *smoothwidth  // 设定新的平滑模版大小
    ) {
        // 将 smoothWidth 成员变量赋成新值
        this->smoothWidth = smoothwidth;

        return NO_ERROR;
    }

    // 成员方法：getIterationSM2（获取 iterationSM2 迭代次数）
    // 获取成员变量 iterationSM2 的值。
    __host__ __device__ int  // 返回值：成员变量 iterationSM2 的值
    getIterationSM2() const
    {
        // 返回 iterationM2 成员变量的值。
        return this->iterationSM2;
    }

    // 成员方法：setIterationSM2（设置 iterationSM2 迭代次数）
    // 设置成员变量 iterationSM2 的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执
                              // 行，返回 NO_ERROR。
    setIterationSM2(
            int iterationsm2  // 设定新的迭代次数
    ) {
        // 将 iterationSM2 成员变量赋成新值
        this->iterationSM2 = iterationsm2;

        return NO_ERROR;
    }
    
    // 成员方法：getWeightSM1（获取 SM1 的权重）
    // 获取成员变量 weightSM1 的值。
    __host__ __device__ float  // 返回值：成员变量 weightSM1 的值
    getWeightSM1() const
    {
        // 返回 weightSM1 成员变量的值。
        return this->weightSM1;
    }

    // 成员方法：setWeightSM1（设置 SM1 的权重）
    // 设置成员变量 weightSM1 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setWeightSM1(
            float weightsm1  // 设定新的 SM1 的权重
    ) {
        // 将 weightSM1 成员变量赋成新值
        this->weightSM1 = weightsm1;

        return NO_ERROR;
    }

    // 成员方法：getWeightSM2（获取 SM2 的权重）
    // 获取成员变量 weightSM2 的值。
    __host__ __device__ float  // 返回值：成员变量 weightSM2 的值
    getWeightSM2() const
    {
        // 返回 weightSM2 成员变量的值。
        return this->weightSM2;
    }

    // 成员方法：setWeightSM2（设置 SM2 的权重）
    // 设置成员变量 weightSM2 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setWeightSM2(
            float weightsm2  // 设定新的 SM2 的权重
    ) {
        // 将 weightSM2 成员变量赋成新值
        this->weightSM2 = weightsm2;

        return NO_ERROR;
    }

    // 成员方法：getThreshold（读取阈值）
    // 读取 threshold 成员变量的值。
    __host__ __device__ int  // 返回值：当前 threshold 成员变量的值。
    getThreshold() const
    {
        // 返回 threshold 成员变量的值。
        return this->threshold;
    } 

    // 成员方法：setThreshold（设置阈值）
    // 设置 threshold 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setThreshold( 
            int threshold    // 指定的阈值大小。
    ) {
        // 将 threshold 成员变量赋值成新值
        this->threshold = threshold;
    
        return NO_ERROR;
    }

    // 成员方法：getMinRegion（获取区域最小面积的阈值）
    // 获取成员变量 minRegion 的值。
    __host__ __device__ int  // 返回值：成员变量 minRegion 的值
    getMinRegion() const
    {
        // 返回 minRegion 成员变量的值。
        return this->minRegion;
    }

    // 成员方法：setMinRegion（设置区域最小面积的阈值）
    // 设置成员变量 minRegion 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setMinRegion(
            int minregion  // 设定新的区域最小面积的阈值
    ) {
        // 将 minRegion 成员变量赋成新值
        this->minRegion = minregion;

        return NO_ERROR;
    }

    // 成员方法：getMaxRegion（获取区域最大面积的阈值）
    // 获取成员变量 maxRegion 的值。
    __host__ __device__ int  // 返回值：成员变量 maxRegion 的值
    getMaxRegion() const
    {
        // 返回 maxRegion 成员变量的值。
        return this->maxRegion;
    }

    // 成员方法：setMaxRegion（设置区域最大面积的阈值）
    // 设置成员变量 maxRegion 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setMaxRegion(
            int maxregion    // 设定新的区域最大面积的阈值
    ) {
        // 将 maxRegion 成员变量赋成新值
        this->maxRegion = maxregion;

        return NO_ERROR;
    }

    // 成员方法：getSaliencyThred（获取区域平均显著性值的阈值大小）
    // 获取成员变量 saliencyThred 的值。
    __host__ __device__ int  // 返回值：成员变量 saliencyThred 的值
    getSaliencyThred() const
    {
        // 返回 saliencyThred 成员变量的值。
        return this->saliencyThred;
    }

    // 成员方法：setSaliencyThred（设置区域平均显著性值的阈值大小）
    // 设置成员变量 saliencyThred 的值。
    __host__ __device__ int    // 返回值：函数是否正确执行，若函数正确执
                               // 行，返回 NO_ERROR。
    setSaliencyThred(
            int saliencythred  // 设定新的区域平均显著性值的阈值大小
    ) {
        // 将 saliencyThred 成员变量赋成新值
        this->saliencyThred = saliencythred;

        return NO_ERROR;
    }
    
    // Host 成员方法：saliencyMapByDiffValue（差值法计算显著值）
    // 计算图像中每个像素值的显著性值。以每个像素为中心，计算其与邻域r内所有
    // 像素的灰度差值；对所有差值按照降序进行排序，去掉排序中先头的若干值和
    // 末尾的若干值（通过设置 highPercent 和 lowPercent），只保留中间部分的
    // 排序结果。对所有的像素进行这样的计算，形成一个初期的 saliency map。
    // 然后改变 r 值，重复上述计算，得到若干个初期 saliency map，将所有的 
    // saliency map 进行累加平均，就得到最终的平均 saliency map，设为 SM1。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    saliencyMapByDiffValue(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );

    // Host 成员方法：saliencyMapBySmooth（高斯平滑法计算显著值）
    // 计算图像中每个像素值的显著性值。利用高斯平滑滤波对原始图像进行处理，
    // 设置 sigma 表示平滑尺度大小，将平滑结果与算数几何平均的图像进行整体
    // 差分，就得到一个初期的 saliency map。改变 sigma 的值，重复上述计算，
    // 得到若干个初期 saliency map，将所有的 saliency map 进行累加平均，
    // 就得到最终的平均 saliency map，设为 SM2。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    saliencyMapBySmooth(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
    
       
    // Host 成员方法：saliencyMapBySmooth（高斯平滑法计算显著值）
    // 显著性区域的总方法。首先计算 saliency map ，包括两种算法，一种是根据
    // 领域的灰度值差值排序并平均，一种是根据高斯平滑计算，将两种方法的结果
    // 加权平均就得到最终的 saliency map，然后分割 saliency map 得到多个显著
    // 性区域，最后根据区域平均显著性阈值和区域面积，筛选出最终需要的显著性区
    // 域，并用二值化显示，即显著性区域设置为白色，其它区域为黑色。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    saliencyRegionDetect(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
};

#endif
