// EdgeCheck.h
// 创建人：刘宇
//
// 边缘异常点和片段检查（EdgeCheck）
// 功能说明：与标准边缘模板相比较，检查出测试边缘的异常点和片段。
//
// 修订历史：
// 2012年12月10日（刘宇）
//     初始版本
// 2013年1月6日（王雪菲）
//     添加测试边缘异常点检查
// 2013年1月8日（刘宇、王雪菲）
//     改正一些逻辑错误，修改代码规范
// 2013年02月25日（刘宇、王雪菲）
//     添加错误码图像的领域判断

#ifndef __EDGECHECK_H__
#define __EDGECHECK_H__

#include "Image.h"
#include "CoordiSet.h"
#include "ErrorCode.h"

// 类：EdgeCheck
// 继承自：无
// 与标准边缘模板相比较，检查出测试边缘的异常点和片段。
class EdgeCheck {

protected:

    // 成员变量：followWidth（顺逆时针跟踪的宽度）
    // 曲线进行顺逆时针双向跟踪的宽度。
    int followWidth;

    // 成员变量：maxDis（异常片段检查的欧式距离阈值大小）
    // 边缘点间的欧式距离的阈值大小。
    float maxDis;

    // 成员变量：maxDisPoint（异常点检查的欧式距离阈值大小）
    // 边缘点间的欧式距离的阈值大小。
    float maxDisPoint;
    
    // 成员变量：minCor（标准相关系数的阈值大小）
    // 对应边缘点间的标准相关系数的阈值大小。
    float minCor;
  
    // 成员变量：reImages（参考图像数组）
    // 存储指向参考图像的指针。
    Image **reImages;
     
    // 成员变量：reCount（参考图像数量）
    // 记录参考图像的数量。
    int reCount;
        
    // 成员变量：highPixel（高像素）
    // 将图像转化成坐标集时，图像的高像素值。
    unsigned char highPixel;

public:
    // 构造函数：EdgeCheck
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    EdgeCheck()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->followWidth = 3;      // 顺逆时针跟踪的宽度设为 3。
        this->maxDis = 10.0f;       // 异常片段检查欧式距离阈值大小设为 10.0f。
        this->maxDisPoint = 10.0f;  // 异常点检查欧式距离阈值大小设为 10.0f。
        this->minCor = 10000.0f;    // 标准相关系数的阈值设为 10000.0f。
        this->reImages = NULL;      // 参考图像设置为空。
        this->reCount = 0;          // 参考图像数量设置为 0。
        this->highPixel = 255;      // 高像素设为 255。
    }

    // 构造函数：EdgeCheck
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    EdgeCheck(
            int followwidth,
            int maxdis,
            int maxdispoint,
            int mincor,
            Image **reimages,
            int recount,
            unsigned char highpixel
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->followWidth = 3;      // 顺逆时针跟踪的宽度设为 3。
        this->maxDis = 10.0f;       // 异常片段检查欧式距离阈值大小设为 10.0f。
        this->maxDisPoint = 10.0f;  // 异常点检查欧式距离阈值大小设为 10.0f。
        this->minCor = 10000.0f;    // 标准相关系数的阈值设为 10000.0f。
        this->reImages = NULL;      // 参考图像设置为空。
        this->reCount = 0;          // 参考图像数量设置为 0。
        this->highPixel = 255;      // 高像素设为 255。

        setFollowWidth(followwidth);
        setMaxDis(maxdis);
        setMaxDisPoint(maxdispoint);
        setMinCor(mincor);
        setReImages(reimages,recount);
        setHighPixel(highpixel);
    }

    // 成员方法：getFollowWidth（获取顺逆时针跟踪的宽度）
    // 获取成员变量 followWidth 的值。
    __host__ __device__ int  // 返回值：成员变量 followWidth 的值
    getFollowWidth() const
    {
        // 返回 followWidth 成员变量的值。
        return this->followWidth;
    }

    // 成员方法：setFollowWidth（设置顺逆时针跟踪的宽度）
    // 设置成员变量 followWidth 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setFollowWidth(
            int followwidth  // 设定新的顺逆时针跟踪的宽度
    ) {
        // 将 followWidth 成员变量赋成新值。
        this->followWidth = followwidth;

        return NO_ERROR;
    }

    // 成员方法：getMaxDis（获取异常片段检查欧式距离阈值大小）
    // 获取成员变量 maxDis 的值。
    __host__ __device__ float  // 返回值：成员变量 maxDis 的值
    getMaxDis() const
    {
        // 返回 maxDis 成员变量的值。
        return this->maxDis;
    }

    // 成员方法：setMaxDis（设置异常片段检查欧式距离阈值大小）
    // 设置成员变量 maxDis 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setMaxDis(
            float maxdis     // 设定新的欧式距离阈值大小
    ) {
        // 将 maxDis 成员变量赋成新值。
        this->maxDis = maxdis;

        return NO_ERROR;
    }

    // 成员方法：getMaxDisPoint（获取异常点检查欧式距离阈值大小）
    // 获取成员变量 maxDisPoint 的值。
    __host__ __device__ float  // 返回值：成员变量 maxDisPoint 的值
    getMaxDisPoint() const
    {
        // 返回 maxDisPoint 成员变量的值。
        return this->maxDisPoint;
    }

    // 成员方法：setMaxDisPoint（设置异常点检查欧式距离阈值大小）
    // 设置成员变量 maxDisPoint 的值。
    __host__ __device__ int    // 返回值：函数是否正确执行，若函数正确执
                               // 行，返回 NO_ERROR。
    setMaxDisPoint(
            float maxdispoint  // 设定新的欧式距离阈值大小
    ) {
        // 将 maxDisPoint 成员变量赋成新值。
        this->maxDisPoint = maxdispoint;

        return NO_ERROR;
    }

    // 成员方法：getMinCor（获取标准相关系数的阈值大小）
    // 获取成员变量 minCor 的值。
    __host__ __device__ float  // 返回值：成员变量 minCor 的值
    getMinCor() const
    {
        // 返回 minCor 成员变量的值。
        return this->minCor;
    }

    // 成员方法：setMinCor（设置标准相关系数的阈值大小）
    // 设置成员变量 minCor 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setMinCor(
            float mincor     // 设定新的标准相关系数的阈值大小
    ) {
        // 将 minCor 成员变量赋成新值。
        this->minCor = mincor;

        return NO_ERROR;
    }
    
    // 成员方法：getReImages（获取 reImages）
    // 获取成员变量 reImages 的值。
    __host__ __device__ Image **  // 返回值：成员变量 reImages 的值
    getReImages() const
    {
        // 返回 reImages 成员变量的值。
        return this->reImages;
    }

    // 成员方法：getReCount（获取参考图像数目）
    // 获取成员变量 reCount 的值。
    __host__ __device__ int  // 返回值：成员变量 reCount 的值
    getReCount() const
    {
        // 返回 reCount 成员变量的值。
        return this->reCount;
    }
    
    // 成员方法：setReImages（设置 reImages 的数据）
    // 设置所有的 reImages 的数据。
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回 NO_ERROR。
    setReImages(
            Image **reimages,   // 要设置的一组参考图像
            int count           // 参考图像的数量
    ) {
        // 判断 reimages 是否为空，若为空，直接返回错误。
        if (reimages == NULL)
            return NULL_POINTER;

        // 判断 count 是否合法，若不合法，直接返回错误。
        if (count <= 0)
            return INVALID_DATA;

        // 扫描 reimages 的每一个成员，检查每个成员是否为空。
        for (int i = 0; i < count; i++) {
            // 若某个成员为空，则直接返回错误。
            if (reimages[i] == NULL) 
                return NULL_POINTER;
        }

        // 更新参考图像数据。
        this->reImages = reimages;
        // 更新 reCount 数据。
        this->reCount = count;

        // 处理完毕，返回 NO_ERROR。
        return NO_ERROR;
    }
       
    // 成员方法：getHighPixel（获取高像素）
    // 获取成员变量 highPixel 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 highPixel 的值
    getHighPixel() const
    {
        // 返回 highPixel 成员变量的值。
        return this->highPixel;
    }
    
    // 成员方法：setHighPixel（设置高像素）
    // 设置成员变量 highPixel 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setHighPixel(
            int highpixel    // 设定新的高像素
    ) {     
        // 将 highPixel 成员变量赋成新值。
        this->highPixel = highpixel;

        // 处理完毕，返回 NO_ERROR。
        return NO_ERROR;
    }
    
    // Host 成员方法：edgeCheckPoint（边缘的异常点检查）
    // 首先计算测试图像与各参考图像的相关系数并找到最匹配的参考图像，
    // 然后将测试图像和匹配图像转化成坐标集并计算参考图像和测试图像
    // 所有点的欧式距离，若欧式距离值不满足规定阈值，则认为是异常点。
    __host__ int            // 返回值：函数是否正确执行，若函数正确执行，
                            // 返回 NO_ERROR。
    edgeCheckPoint(
           Image *teimg,    // 测试图像
           Image *errpoint  // 异常点图像
    );
    
    // Host 成员方法：edgeCheckFragment（边缘的异常片段检查）
    // 首先分别计算参考边缘和测试边缘的 local moment 特征，然后根据欧式距离找到
    // 对应点关系，计算对应点关系的标准相关系数，如果系数值不满足规定阈值，则认
    // 为是异常点。
    __host__ int                 // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    edgeCheckFragment(
            CoordiSet *recdset,  // 参考边缘坐标集
            CoordiSet *tecdset,  // 测试边缘坐标集
            Image *errpoint      // 异常点图像
    );
};

#endif

