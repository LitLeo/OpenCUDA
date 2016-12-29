// RobustEdgeDetection.h
// 创建者：张丽洁
//
// 健壮的边缘检测图像算法（RobustEdgeDetection）
// 功能说明：根据图像中每个像素的两个对象领域间的差分对图像进行 Robust 边缘
//           检测。共有两种算法实现，第一种为单纯平均法，第二种为特征向量法。
// 修订历史：

// 2012年10月13日（张丽洁）
//     初始版本。
// 2012年10月17日（张丽洁）
//     修改了成员方法声明范围有误而产生的 bug。
// 2012年10月22日（张丽洁）
//     根据 ver2.01 版项目书以及 v3.41 版编码规范对代码进行适当修改。
// 2012年10月30日（张丽洁）
//     根据 ver2.03 版项目书及问题反馈文档对代码进行修改，删除 percent 参数。
// 2012年11月10日（张丽洁）
//     运用模板完成对象临域的定义的实现。
// 2012年11月20日（张丽洁）
//     改进对象临域的实现方法，用函数将所要参加计算的点的索引一一算出，然后
//     进行差分计算。
// 2012年11月27日（张丽洁）
//     将非极大值抑制方法初步实现。
// 2012年12月04日（张丽洁）
//     添加差分计算时的边界溢出检测，修改了差分计算时没进行绝对值处理的的 bug。
// 2012年12月09日（张丽洁）
//     修改了对向临域索引值计算的一处误将真实坐标当做相对坐标计算索引逻辑错误。
// 2012年12月10日（张丽洁）
//     代码规范化。
// 2012年12月28日（张丽洁）
//     修改了非极大值抑制处的一个 DP 号匹配的 bug，同时在代码中添加了控制是否将
//     差分结果进行求平均的开关宏 RED_DIV_AVERAGE。
// 2013年3月13日（刘婷，张丽洁）
//     修改了单纯平均法中对向邻域求索引在计算有重叠的对向邻域出重叠部分大小时的
//     一处 bug。
// 2013年03月16日（刘婷）
//     修改了对向邻域坐标计算错误。
//     修改了非最大值抑制判断条件的错误。
//     getMinInterval 方法返回值类型错误。
// 2013年03月18日（刘婷)
//     添加了 host 函数 callHistogram，
//     调用 Histogram 算法计算EMAV, EMMD, ESGM，
//     将 EMAV, EMMD, ESGM 的计算区域改为了整幅图片，
//     而非每一个对向邻域对应求出四个 EMAV (EMMD, ESGM)。
//     添加了 host 函数 findmaxid 找出出现次数最多的像素点值。
// 2013年03月19日（刘婷）
//     更改了 minInterval 的默认值，根据对河边老师需求文档的分
//     析，该值在计算 MMD 时代表的是两个峰值之间的最小距离，所
//     以该值得默认值设为 255 是不合适，更改为 5。
//     更改了图像的边缘条件判断。
// 2013年03月20日（刘婷，张丽洁）
//     代码规范化。
//     修改一系列 set 函数返回值类型错误。
// 2013年03月21日（张丽洁）
//     由于需求分析文档有所更改，所以将现在已用不上的原来的成员变量
//     complexWeight 删除，增加 diffsize（对向邻域的大小）为成员变量。
// 2013年03月21日（张丽洁，刘婷）
//     增加了对输入参数的有效性的判断。
// 2013年03月22日（刘婷）
//     将非极大值部分单纯整理为一个 device 方法，减少了代码的冗余。
//     对 host 函数里面的 guidingset 设默认参数为NULL。
//     将计算 8 个邻域内像素点值的累加和计算每个邻域的直方图
//     数据整理成单独的函数。
// 2013年03月28日（张丽洁）
//     修改了个别变量的名称，增加 Histogram 型成员变量 histogram，增加对输出图像
//     outimg 是否为空的检查，以及代码规范化。
// 2013年03月28日（刘婷，张丽洁）
//     增加对坐标点集 guidingset 是否为空的检查，并调用 ImgConvert 函数，将边缘
//     检测所要操作的指导区域的点集转换为图像。
// 2013年03月29日（刘婷）
//     添加对除数是否为 0 的判断。
//     将源代码改成三维，并且将 8 个对向邻域做成一个模板，将模板传入 kernel 中，
//     便于并行处理，使用一个线程处理一个邻域的方式。
// 2013年03月30日（刘婷）
//     将 MAV 等三个量的求解分解成两个 device 函数。
// 2013年04月01日（刘婷）
//     通过使用 device 端的数组的方式，去掉了非极大值抑制处理中的 if 语句，
//     减少线程的分支。
//     代码规范化。
// 2013年04月02日（刘婷）
//     更改一处计算重叠区域的一处逻辑错误。
//     在 kernel 中增加了一个临时数组，大大减少了求解三个特征量的时间，在保证正
//     确性的基础上大大提高了效率。
// 2013年04月03日（刘婷）
//     代码规范化。
// 2013年04月06日（刘婷）
//     删除了 newipixel[256] 这个数组，减少了寄存器的使用，新设置了一个 unsigned
//     char 型数组，用来记录邻域内部的值，该数组的长度目前通过宏来设置，初始值设
//     为 11。因为通过调试发现 kernel 中寄存器溢出导致性能瓶颈，因此通过减少寄存
//     的使用使得性能出现了显著的改善。
// 2013年04月07日（刘婷）
//     将单纯平均法改为三维，将原来的一个点处理四个对象邻域改为一个点处理一个对
//     象邻域，增加了共享内存的使用。
//     增加了对宏来回收空间。
// 2013年04月08日（刘婷）
//     修改边界处理的一处错误。
// 2013年04月09日（刘婷）
//     整理代码规范。
// 2013年04月11日（刘婷）
//     整理代码规范。
// 2013年05月02日（刘婷）
//     更改求解 mmd 时的一处逻辑错误。
//     整理代码规范。
// 2013年05月20日（张丽洁）
//     改写原来的 Device 函数：_computeMavSgmDev 函数，来计算差分正规化需要的中
//     间变量中值平均 mav 和 方差 sgm。
// 2013年06月05日（张丽洁）
//     修改了核函数，用差分正规化替代原来的对向差分计算。 
// 2013年06月24日（张丽洁）
//     修改原来的非极大抑制方法为最大增强法。
// 2013年07月02日（张丽洁）
//     第二期健壮的边缘检测算法（RobustEdgeDetection）初始版。程序能正常运行，但
//     是图像的结果还不正确。
// 2013年09月17日（张丽洁）
//     在成员方法中分别增加了 sobel 算子边缘检测方法的串行实现和并行实现，分别为
//     sobelHost 函数和 sobel 函数。
// 2013年09月25日（张丽洁）
//     修正特征向量法中编译时会产生 warning 的两处 bug。
// 2013年10月09日（张丽洁）
//     改写原来的 Device 函数：_computeMavSgmDev 函数，简化其接口。
// 2013年10月17日（张丽洁）
//     第二期健壮的边缘检测算法（RobustEdgeDetection）单纯平均法的初始完成版。将
//     原来的核函数中的申请存放像素点的数组由一个变为两个，修改了最大增强法中存
//     在的一处计算 bug。
// 2013年10月20日（张丽洁）
//     添加开关宏 RED_CODE_CHOICE，来决定注释掉原来算法的代码。
//     整理代码规范。
// 2013年11月02日（张丽洁）
//     添加 Device 函数：_computeMavMaxDev，计算各个对向邻域的3个统计量。
// 2013年11月11日（张丽洁）
//     修改特征向量法的核函数 _detectEdgeFVKer。
// 2013年11月15日（张丽洁）
//     第二期健壮的边缘检测算法（RobustEdgeDetection）特征向量法的初始完成版。
// 2013年11月18日（张丽洁）
//     完整注释，整理代码规范。
// 2013年11月28日（张丽洁）
//     完整注释，整理代码规范。
// 2014年09月28日（于玉龙）
//     修正了检测模版生成算法中的一处下标计算错误。


#ifndef __ROBUSTEDGEDETECTION_H__
#define __ROBUSTEDGEDETECTION_H__

#include "cmath"
#include "CoordiSet.h"
#include "ErrorCode.h"
#include "FreckleFilter.h"
#include "Image.h"
#include "Thinning.h"
#include "Histogram.h"

// 类：RobustEdgeDetection（健壮的边缘检测图像算法）
// 继承自：无
// 应用两种不同的算法，将输入图像的边缘检测出来。第一种为单纯平均法，第二种为特
// 征向量法。第一种方法是直接通过差分计算，将边缘像素点计算出来，第二种是通过公
// 求出平均值和标准差等一些中间量，最后得到边缘，然后经过精化处理得到准确的图像
// 的边缘点。
class RobustEdgeDetection {

protected:

    // 成员变量：diffsize（对向邻域的大小）
    // 通过此成员变量可从外部设定算法中计算涉及到的对向邻域的大小。
    int diffsize;

    // 成员变量：mmdWeight 和 sgmWeight（计算量的权重）
    // mmdWeight 和 sgmWeight 分别表示计算两个对向域的三个特征量间几何距离时，
    // 两个个数最多的图像值之差 MMD 的权重和各个领域图像值的标准差 SGM 的权重。
    float mmdWeight, sgmWeight;

    // 成员变量：minInterval（控制参数）
    // 在计算两个个数最多的图像值之差 MMD 时，要先对各个辉度的出现次数
    // 进行统计，但这两个峰值之间的最小距离是由参数 minInterval 来控制的。
    int minInterval;

    // 成员变量：searchScope（搜索范围）
    // 对差分图像，在 DP 方向上进行搜索的范围。可由外部输入。
    int searchScope;

    // 成员变量：thinning（细化对象）
    // 图像进行菲最大抑制之后进行细化处理，得到单像素边缘。
    Thinning thinning;

    // 成员变量：frecklefilter（frecklefilter 对象）
    // 利用 FreckleFilter 去除像素孤岛。
    FreckleFilter frecklefilter;

    // 成员变量：histogram（直方图对象）
    // 调用 Histogram 函数，求出图像的直方图数据。
    Histogram histogram;

public:

    // 构造函数：RobustEdgeDetection
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    RobustEdgeDetection() :
            thinning(), histogram(),
            frecklefilter(1, 100.5f, 0.2f, 2, 0, 1)
   {
        diffsize = 3;        // 权重参数 complexWeight 的默认值为 3
        mmdWeight = 500.0f;  // 权重参数 mmdWeight 的默认值范围为
                             // 0 ~ 1000，这里默认为 500
        sgmWeight = 500.0f;  // 权重参数 sgmWeight 的默认值范围为
                             // 0 ~ 1000，这里默认为 500
        minInterval = 5;     // 像素值两个峰值之间的最小距离参数，
                             // 这里默认值设为 5
        searchScope = 10;    // dp 方向上进行搜索范围的参数，
                             // 默认值为 10
    }

    // 构造函数：RobustEdgeDetection
    // 有参数版的构造函数，根据需要给定各个参数，这些参数值在程序运行
    // 过程中是可以改变的。
    __host__ __device__
    RobustEdgeDetection (
            int diffsize,     // 控制参数（具体解释见成员变量）
            float mmdWeight,  // 权重参数（具体解释见成员变量）
            float sgmWeight,  // 权重参数（具体解释见成员变量）
            int minInterval,  // 控制参数（具体解释见成员变量）
            int searchScope   // 搜索参数（具体解释见成员变量）
    ) :
            thinning(), histogram(),
            frecklefilter(1, 100.5f, 0.2f, 10, 0, 1)
    {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数
        // 的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->diffsize = 3;        // 控制参数 diffsize 的默认值为 3
        this->mmdWeight = 500.0f;  // 权重参数 mmdWeight 的默认值范围
                                   // 为 0 ~ 1000，这里默认为 500
        this->sgmWeight = 500.0f;  // 权重参数 sgmWeight 的默认值范围
                                   // 为 0 ~ 1000，这里默认为 500
        this->minInterval = 5;     // 像素值两个峰值间的最小距离参数，
                                   // 这里默认值设为 5
        this->searchScope = 10;    // dp 方向上进行搜索范围的参数，默
                                   // 认值为 10

        // 根据参数列表中的值设定成员变量的初值
        setDiffsize(diffsize);
        setMmdWeight(mmdWeight);
        setSgmWeight(sgmWeight);
        setMinInterval(minInterval);
        setSearchScope(searchScope);
    }

    // 成员方法：getDiffsize（获取控制参数）
    // 获取成员变量 diffsize 的值。
    __host__ __device__ int  // 返回值：控制对向邻域的大小。
    getDiffsize() const
    {
        // 返回 diffsize 成员变量值。
        return this->diffsize;
    }

    // 成员方法：setDiffsize（设定控制参数）
    // 设置成员变量 Diffsize 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，函数正确执行，返回
                             // NO_ERROR。
    setDiffsize(
            int diffsize     // 新的对向邻域的大小
    ) {
        // 判断参数是否合法
        if (diffsize < 3 || (diffsize % 2) == 0)
            return INVALID_DATA;       

        // 将 diffsize 成员变量赋成新值。
        this->diffsize = diffsize;
        return NO_ERROR;
    }

    // 成员方法：getMmdWeight（获取权重参数）
    // 获取成员变量 mmdWeight 的值。
    __host__ __device__ float  // 返回值：图像中的权重参数。
    getMmdWeight() const
    {
        // 返回 mmdWeight 成员变量值。
        return this->mmdWeight;
    }

    // 成员方法：setMmdWeight（获取权重参数）
    // 获取成员变量 mmdWeight 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，如果函数正确执行，
                             // 返回 NO_ERROR。
    setMmdWeight(
            float mmdWeight  // 新的两个个数最多的图像值之差 MMD 的权重
    ) {
        // 判断输入参数的有效性
        if (mmdWeight < 0 || mmdWeight > 1000)
            return INVALID_DATA;

        // 将 mmdWeight 成员变量赋成新值。
        this->mmdWeight = mmdWeight;
        return NO_ERROR;
    }

    // 成员方法：getSgmWeight（获取权重参数）
    // 获取成员变量 sgmWeight 的值。
    __host__ __device__ float  // 返回值：图像中的权重参数。
    getSgmWeight() const
    {
        // 返回 sgmWeight 成员变量值。
        return this->sgmWeight;
    }

    // 成员方法：setSgmWeight（获取权重参数）
    // 获取成员变量 sgmWeight 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，如果函数正确执行，
                             // 返回 NO_ERROR。
    setSgmWeight(
            float sgmWeight  // 新的各个领域的图像值的标准差 SGM 的权重
    ) {
        // 判断输入参数的有效性
        if (mmdWeight < 0 || mmdWeight > 1000)
            return INVALID_DATA;

        // 将 sgmWeight 成员变量赋成新值。
        this->sgmWeight = sgmWeight;
        return NO_ERROR;
    }

    // 成员方法：getMinInterval（获取控制参数）
    // 获取成员变量 minInterval 的值。
    __host__ __device__ int  // 返回值：图像中的控制参数。
    getMinInterval() const
    {
        // 返回 minInterval 成员变量值。
        return this->minInterval;
    }

    // 成员方法：setMinInterval（获取控制参数）
    // 获取成员变量 minInterval 的值。
    __host__ __device__ int    // 返回值：函数是否正确执行
                               // 如果函数正确执行返回 NO_ERROR。
    setMinInterval(
            float minInterval  // 新的控制参数
    ) {
        // 判断参数是否合法
        if (minInterval < 0.0f || minInterval > 255.0f)
            return INVALID_DATA;

        // 将 minInterval 成员变量赋成新值。
        this->minInterval = minInterval;
        return NO_ERROR;
    }

    // 成员方法：getSearchScope（获取搜索参数）
    // 获取成员变量 searchScope 的值。
    __host__ __device__ int  // 返回值：图像中的搜索参数。
    getSearchScope() const
    {
        // 返回 searchScope 成员变量值。
        return this->searchScope;
    }

    // 成员方法：setSearchScope（获取搜索参数）
    // 获取成员变量 searchScope 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，如果函数正确执行，
                             // 返回 NO_ERROR。
    setSearchScope(
            int searchScope  // 新的搜索参数
    ) {
        // 判断参数是否合法
        if (searchScope <= 0)
            return INVALID_DATA;       

        // 将 searchScope 成员变量赋成新值。
        this->searchScope = searchScope;
        return NO_ERROR;
    }

    // Host 成员方法：detectEdgeSA（单纯平均法）
    // 利用单纯平均法，直接进行对向差分计算。
    __host__ int                   // 返回值：函数是否正确执行，如果函数能够
                                   // 正确执行，返回 NO_ERROR。
    detectEdgeSA(
            Image *image,          // 输入图像
            Image *outimg,         // 输出图像
            CoordiSet *guidingset  // 边缘检测所要操作的指导区域
    );

    // Host 成员方法：detectEdgeFV（特征向量法）
    // 利用特征向量法，通过计算一些权重值以及对图像进行
    // 非最大抑制处理进行检测。
    __host__ int                   // 返回值：函数是否正确执行，如果函数能够
                                   // 正确执行，返回 NO_ERROR。
    detectEdgeFV(
            Image *inimg,          // 输入图像
            Image *outimg,         // 输出图像
            CoordiSet *guidingset  // 边缘检测所要操作的指导区域
    );

    // Host 成员方法：sobelHost（串行版 sobel 边缘检测）
    // CPU 端的 sobel 算子边缘检测。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确执行，
                                   // 返回 NO_ERROR。
    sobelHost(
            Image *src,            // 输入图像
            Image *out             // 输出图像
    );

    // Host 成员方法：sobel（并行版 sobel 边缘检测）
    // GPU 端的 sobel 算子边缘检测。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确执行，
                                   // 返回 NO_ERROR。
    sobel(
            Image *inimg,          // 输入图像
            Image *outimg          // 输出图像
    );

};

#endif

