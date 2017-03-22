// SimpleRegionDetect.h
//  创建人：邓建平
//
// 简单的区域检测（SimpleRegionDetect）
// 功能说明：根据给出的区域大小以及指定的长短径要求，对输入图像进行一些预处理
//           后使用连通域合并与标记，对于符合面积要求的已标记连通域求出其最小
//           外接矩形，最终返回符合长短径要求的最小外接矩形的结构化表示，在图
//           像的预处理过程中，可以通过设定对应参数来确定是否调用某些算法来对
//           前一个处理步骤的输出图像进行优化，例如通过设置repeat参数可以设置
//           是否使用双边滤波，还可以设置radius来确定是否使用中值滤波
//
// 修订历史：
// 2012年09月22日（邓建平）
//     初始版本
// 2012年10月12日（邓建平）
//     使用全局存储保存高斯表，加入运行时间测试
// 2012年10月17日（邓建平）
//     将欧氏距离使用表结构存储在全局存储中，时间提高明显，同时将滤波半径设置
// 2013年01月07日（邓建平）
//     算法整合完毕，可输出有效数据，完成了简易区域检测的初始版本
// 2013年04月17日（邓建平）
//     加入了新的 SmallestDirRect 算法（原 BoundingRect ）,检测的结果准确性提
//     高，稳定性更强
// 2013年04月19日（邓建平）
//     修改了代码的一部分格式错误，将中间类的声明放在主类中，同时对一些函数的
//     返回值以及命名进行了修改，提高了代码的可读性与健壮性
// 2013年04月22日（邓建平）
//     修改了代码的格式错误，完善了错误检查
// 2013年05月25日（邓建平）
//     解决了编译时的 warning 问题

#ifndef __SIMPLEREGIONDETECT_H__
#define __SIMPLEREGIONDETECT_H__

#include "Image.h"
#include "Rectangle.h"
#include "SimpleRegionDetect.h"
#include "BilateralFilter.h"
#include "BilinearInterpolation.h"
#include "SmallestDirRect.h"
#include "ConnectRegion.h"
#include "FreckleFilter.h"
#include "Morphology.h"
#include "Image.h"
#include "ImageDiff.h"
#include "DownSampleImage.h"
#include "LocalCluster.h"
#include "TemplateFactory.h"
#include "Histogram.h"
#include "Threshold.h"
#include "ErrorCode.h"


// 类：SimpleRegionDetect（简单的区域检测）
// 继承自：无
// 根据给出的区域大小以及指定的长短径要求，对输入图像进行一些预处理后
// 使用连通域合并与标记，对于符合面积要求的已标记连通域求出其最小外接
// 矩形，最终返回符合长短径要求的最小外接矩形的结构化表示，在图像的预
// 处理过程中，可以通过设定对应参数来确定是否调用某些算法来对前一个处
// 理步骤的输出图像进行优化，例如通过设置 repeat 参数可以设置是否使用双
// 边滤波，还可以设置 radius 来确定是否使用中值滤波
class SimpleRegionDetect {

protected:

    // 成员变量：sdr（最小外接矩形）
    // 计算检出区域的最小有向外接矩形
    SmallestDirRect sdr;

    // 成员变量：hist（直方图统计）
    // 使用直方图检测经连通域标记后的面积大小，以进行分割
    Histogram hist;

    // 成员变量：morph（形态学变换）
    // 对分割后的图像进行 CLOSE 运算，提高区域的完整性
    Morphology morph;

    // 成员变量：th（阈值分割）
    // 完成图像分割，把被标记的区域单独分割出来
    Threshold th;

    // 成员变量：downimg（图像缩小）
    // 采用 DOMINANCE 方法对图像进行缩小
    DownSampleImage downimg;

    // 成员变量：freckle（FreckleFilter）
    // 使用 FreckleFilter 对缩小后的图像进行滤波
    FreckleFilter freckle;

    // 成员变量：biInterpo（双线性插值）
    // 完成双线性插值放大，基于 Texture 的硬件插值实现
    BilinearInterpolation biInterpo;

    // 成员变量：diff（图像差分）
    // 用放大后的图像作为背景图像，计算原图与背景图像的差分图像
    ImageDiff diff;

    // 成员变量：cluster（Local Clustering）
    // 对差分图像调用 Local Clustering 算法进行 clustering
    LocalCluster cluster;

    // 成员变量：biFilter（双边滤波）
    // 对差分图像进行双边滤波
    BilateralFilter biFilter;

    // 成员变量：conn（连通域标记）
    // 标记差分后的图像中的连通区域
    ConnectRegion conn;

    // 成员变量：tpl（运算模板）
    // 使用模板工厂获取 CLOSE 运算的模板
    Template *tpl;

    // 成员变量：longSide（有向外接矩形长径），shortSide（有向外接矩形短径）
    // 这两个变量用于对检出的矩形进行筛选，分别表示长径的最大值和短径的最小值
    // 默认为 1 和 648，即不过滤
    int longSide, shortSide;

    // 成员变量：closeSize（闭运算的模板大小）
    // 默认值为 1 ，参数设置必须合法有效，闭运算是区域检测算法中必不可少的一步
    int closeSize;

    // Host 成员方法：hasValidRect（检测区域长短径函数）
    // 检测图像中包含区域的外接矩形的长短径，外接矩形的四个顶点取区域中点的最值
    // ，即以横坐标和纵坐标的最小值作为左上顶点，最大值作为右下顶点
    __host__ int       // 返回值：函数是否正确执行，若函数正确
                       // 执行，返回 NO_ERROR。
    hasValidRect(
            Image *inimg,  // 待检测图像
            bool *result   // 检测结果，true 表示长短径符合要求，false 表示不符合
    );

    // Host 成员方法：cutImages（图像分割函数）
    // 对检出的连通域进行 0 - 255 LEVEL 的灰度分割，并检出其最小外接矩形
    __host__ int                   // 返回值：函数是否正确执行，若函数正确
                                   // 执行，返回 NO_ERROR。
    cutImages(                
            Image *inimg,          // 连通域标记后的图像
            Image *outimg,         // 闭运算的输出图像
            int closeSize,         // 对分割后的图像进行 CLOSE 运算的模板大小
            DirectedRect **rects,  // 检出的最小有向外接矩形
            int *count             // 检出的外接矩形的数量
    );

public:

    // 构造函数：SimpleRegionDetect
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__
    SimpleRegionDetect()
    {
        // 使用默认值为类的各个成员变量赋值
        longSide = 1;      // 长径的默认值为 1
        shortSide = 648;   // 短径的默认值为 648
        closeSize = 1;     // 闭运算模板大小默认值为 1

        // 配置连通域标记 conn
        this->setMinArea(1000);      // 最小区域面积的的默认值为 1000
        this->setMaxArea(100000);    // 最大区域面积的的默认值为 100000
        this->setRegionTh(2);        // 连通域标记的阈值默认为 2 ，测试效果好
        // 配置图像缩小类 downimg 和 双线性插值 biInterpo
        this->setScale(1);           // 缩小倍数默认值为 1
        // 配置 freckle 滤波器 freckle
        this->freckle.setRadius(0);  // FreckleFilter 中的圆周半径的默认值为 0
        // 配置 local clustering
        this->setDest(100);          // 坐标范围的默认值为 100
        this->setCount(8);           // 方向数的默认值为 8
        this->setProBlack(10);       // 黑色像素值的默认值为 10
        this->setProWhite(250);      // 白色像素值的默认值为 250
        this->setGapThred(0);        // 邻域内像素差值的阈值的默认值为 0
        this->setDiffeThred(0);      // 差分阈值的默认值为 0
        // 配置双边滤波 biFilter
        this->setRadius(0);          // 滤波半径的默认值为 0
        this->setRepeat(0);          // 重复次数的默认值为 0
        // 配置运算模板 tpl
        this->setCloseSize(1);       // 模板大小的默认值为 1
    }

    // 构造函数：SimpleRegionDetect
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__                  
    SimpleRegionDetect(
            int minarea, int maxarea,     // 区域面积的最小和最大值
            int longside, int shortside,  // 长径，短径
            int scale,                    // 缩小倍数
	    unsigned char regionth,       // 连通域标记阈值
            int dest, int count,          // 坐标范围，方向数
            unsigned char problack,       // 黑色像素值
            unsigned char prowhite,       // 白色像素值
            unsigned char gapthred,       // 中心像素与邻域像素差值的阈值
            unsigned char diffethred,     // 差分阈值
            int closesize                 // 闭运算模板大小
    ) {
        // 使用默认值为类的各个成员变量赋值
        longSide = 1;      // 长径的默认值为 1
        shortSide = 648;   // 短径的默认值为 648
        closeSize = 1;     // 闭运算模板大小默认值为 1

        // 配置连通域标记 conn
        this->setMinArea(1000);      // 最小区域面积的的默认值为 1000
        this->setMaxArea(100000);    // 最大区域面积的的默认值为 100000
        this->setRegionTh(2);        // 连通域标记的阈值默认为 2 ，测试效果好
        // 配置图像缩小类 downimg 和 双线性插值 biInterpo
        this->setScale(1);           // 缩小倍数默认值为 1
        // 配置 freckle 滤波器 freckle
        this->freckle.setRadius(0);  // FreckleFilter 中的圆周半径的默认值为 0
        // 配置 local clustering
        this->setDest(100);          // 坐标范围的默认值为 100
        this->setCount(8);           // 方向数的默认值为 8
        this->setProBlack(10);       // 黑色像素值的默认值为 10
        this->setProWhite(250);      // 白色像素值的默认值为 250
        this->setGapThred(0);        // 邻域内像素差值的阈值的默认值为 0
        this->setDiffeThred(0);      // 差分阈值的默认值为 0
        // 配置双边滤波 biFilter
        this->setRadius(0);          // 滤波半径的默认值为 0
        this->setRepeat(0);          // 重复次数的默认值为 0
        // 配置运算模板 tpl
        this->setCloseSize(1);       // 模板大小的默认值为 1

        // 根据参数列表中的值设定成员变量的初值
        this->setMinArea(minarea);
        this->setMaxArea(maxarea);
        this->setLongSide(longside);
        this->setShortSide(shortside);
        this->setScale(scale);
        this->setRegionTh(regionth);
        this->setDest(dest);
        this->setCount(count);
        this->setProBlack(problack);
        this->setProWhite(prowhite);
        this->setGapThred(gapthred);
        this->setDiffeThred(diffethred);
        this->setCloseSize(closeSize);
    }

    // 成员方法：getMinArea（读取连通域面积最小值）
    // 读取成员变量 conn 中的 minArea 属性
    __host__ __device__ int    // 返回: 当前 minArea 成员变量的值
    getMinArea() const
    {
        // 返回 minArea 成员变量的值
        return conn.getMinArea();
    }

    // 成员方法：setMinArea（设置连通域面积最小值）
    // 设置成员变量 conn 中的 minArea 属性
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setMinArea(
            int minarea      // 连通域面积大小
    ) {
        // 将 minArea 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (minarea < 0)
            return INVALID_DATA;

	// 更新成员变量 conn 中的 minArea 属性
        return conn.setMinArea(minarea);
    }
    
    // 成员方法：getMaxArea（读取连通域面积最大值）
    // 读取成员变量 conn 中的 maxArea 属性
    __host__ __device__ int    // 返回: 当前 maxArea 成员变量的值
    getMaxArea() const
    {
        // 返回 maxArea 成员变量的值
        return conn.getMaxArea();
    }

    // 成员方法：setMaxArea（设置连通域面积最大值）
    // 设置成员变量 conn 中的 maxArea 属性
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setMaxArea(
            int maxarea      // 连通域面积大小
    ) {
        // 将 maxArea 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (maxarea < this->getMinArea())
            return INVALID_DATA;

	// 更新成员变量 conn 中的 maxArea 属性
        return conn.setMaxArea(maxarea);
    }

    // 成员方法：getLongSide（读取长径）
    // 读取 longSide 成员变量的值。
    __host__ __device__ int  // 返回: 当前 longSide 成员变量的值
    getLongSide() const
    {
        // 返回 longSide 成员变量的值
        return longSide;
    }

    // 成员方法：setLongSide（设置长径）
    // 设置 longSide 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setLongSide(
            int longside     // 长径
    ) {
        // 将 longSide 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (longside < this->shortSide)
            return INVALID_DATA;

        // 将 longSide 成员变量赋成新值
        this->longSide = longside;
        return NO_ERROR;
    }

    // 成员方法：getShortSide（读取短径）
    // 读取 shortSide 成员变量的值。
    __host__ __device__ int  // 返回: 当前 shortSide 成员变量的值
    getShortSide() const
    {
        // 返回 shortSide 成员变量的值
        return shortSide;
    }

    // 成员方法：setShortSide（设置短径）
    // 设置 shortSide 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setShortSide(
            int shortside    // 短径
    ) {
        // 将 shortSide 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (shortside < 0)
            return INVALID_DATA;

        // 将 shortSide 成员变量赋成新值
        this->shortSide = shortside;
        return NO_ERROR;
    }

    // 成员方法：getScale（读取缩小倍数）
    // 读取成员变量 biInterpo 中的 scale 属性
    __host__ __device__ int  // 返回: 当前 scale 成员变量的值
    getScale() const
    {
        // 返回 scale 成员变量的值
        return biInterpo.getScale();
    }

    // 成员方法：setRadius（设置缩小倍数）
    // 设置 scale 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setScale(
            int scale        // 中值滤波圆周半径
    ) {
        // 将 scale 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (scale < 1)
            return INVALID_DATA;

        // 更新成员变量 biInterpo 中的 scale 属性
        int errcode = biInterpo.setScale(scale);
        if(errcode != NO_ERROR)
            return errcode;
        // 更新成员变量 downimg 中的 times 属性
        errcode = downimg.setTimes(scale);
        if(errcode != NO_ERROR)
            return errcode;
        return NO_ERROR;
    }
    
    // 成员方法：getRadius（读取中值滤波圆周半径）
    // 读取 radius 成员变量的值。
    __host__ __device__ int  // 返回: 当前 radius 成员变量的值
    getRadius() const
    {
        // 返回 radius 成员变量的值
        return freckle.getRadius();
    }

    // 成员方法：setRadius（设置中值滤波圆周半径）
    // 设置成员变量 freckle 中的 radius 属性
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setRadius(
            int radius       // 中值滤波圆周半径
    ) {
        // 将 radius 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (radius < 0)
            return INVALID_DATA;

        // 更新成员变量 freckle 中的 radius 属性
        return freckle.setRadius(radius);
    }

    // 成员方法：getVarThreshold（读取方差阈值）
    // 读取成员变量 freckle 中的 varThreshold 属性
    __host__ __device__ float  // 返回: 当前 varThreshold 成员变量的值
    getVarThreshold() const
    {
        // 返回 varThreshold 成员变量的值
        return freckle.getVarTh();
    }

    // 成员方法：setVarThreshold（设置方差阈值）
    // 设置成员变量 freckle 中的 varThreshold 属性
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setVarThreshold(
            float varthreshold  // 方差阈值
    ) {
        // 将 varThreshold 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (varthreshold <= 0.0f || varthreshold > 1.0f)
            return INVALID_DATA; 

        // 更新成员变量 freckle 中的 varThreshold 属性
        return freckle.setVarTh(varthreshold);
    }

    // 成员方法：getMatchErrThreshold（读取匹配差阈值）
    // 读取成员变量 freckle 中的 matchErrThreshold 属性
    __host__ __device__ float  // 返回: 当前 matchErrThreshold 成员变量的值
    getMatchErrThreshold() const
    {
        // 返回 matchErrThreshold 成员变量的值
        return freckle.getMatchErrTh();
    }

    // 成员方法：setMatchErrThreshold（设置匹配差阈值）
    // 设置成员变量 freckle 中的 matchErrThreshold 属性
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setMatchErrThreshold(
            float matcherrthreshold    // 匹配差阈值
    ) {
        // 将 matchErrThreshold 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (matcherrthreshold <= 0.0f || matcherrthreshold > 1.0f)
            return INVALID_DATA; 

        // 更新成员变量 freckle 中的 matchErrThreshold 属性
        return freckle.setMatchErrTh(matcherrthreshold);
    }

    // 成员方法：getRegionTh（读取连通域标记的阈值）
    // 读取 regionTh 成员变量的值。
    __host__ __device__ int  // 返回: 当前 regionTh 成员变量的值
    getRegionTh() const
    {
        // 返回 regionTh 成员变量的值
        return conn.getThreshold();
    }

    // 成员方法：setRegionTh（设置连通域标记的阈值）
    // 设置 regionTh 成员变量的值。
    __host__ __device__ int         // 返回值：函数是否正确执行，若函数正确
                                    // 执行，返回 NO_ERROR。
    setRegionTh(
            unsigned char regionth  // 匹配长度参数
    ) {
        // 更新成员变量 conn 中的 regionTh 属性
        return conn.setThreshold(regionth);
    }

    // 成员方法：getDest（读取 local clustering 各方向坐标范围）
    // 读取 dest 成员变量的值。
    __host__ __device__ int  // 返回: 当前 dest 成员变量的值
    getDest() const
    {
        // 返回 dest 成员变量的值
        return cluster.getPntRange();
    }

    // 成员方法：setDest（设置 local clustering 各方向坐标范围）
    // 设置 dest 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setDest(
            int dest         // 各方向坐标范围
    ) {
        // 将 dest 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (dest > 100)
            return INVALID_DATA;

        // 更新成员变量 cluster 中的 dest 属性
        return cluster.setPntRange(dest);
    }

    // 成员方法：getCount（读取 local clustering 方向数）
    // 读取 count 成员变量的值。
    __host__ __device__ int  // 返回: 当前 count 成员变量的值
    getCount() const
    {
        // 返回 count 成员变量的值
        return cluster.getPntCount();
    }

    // 成员方法：setCount（设置 local clustering 方向数）
    // 设置 count 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setCount(
            int count        // local clustering 方向数
    ) {
        // 将 count 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (count < 0 || count > 8)
            return INVALID_DATA;

        // 更新成员变量 cluster 中的 count 属性
        return cluster.setPntCount(count);
    }

    // 成员方法：getProBlack（读取黑色像素值）
    // 读取 proBlack 成员变量的值。
    __host__ __device__ unsigned char    // 返回: 当前 proBlack 成员变量
                                         // 的值
    getProBlack() const
    {
        // 返回 proBlack 成员变量的值
        return cluster.getProBlack();
    }

    // 成员方法：setProBlack（设置黑色像素值）
    // 设置 proBlack 成员变量的值。
    __host__ __device__ int         // 返回值：函数是否正确执行，若函数正
                                    // 确执行，返回 NO_ERROR。
    setProBlack(
            unsigned char problack  // 黑色像素值
    ) {
        // 更新成员变量 cluster 中的 proBlack 属性
        return cluster.setProBlack(problack);
    }

    // 成员方法：getProWhite（读取白色像素值）
    // 读取 proWhite 成员变量的值。
    __host__ __device__ unsigned char    // 返回: 当前 proWhite 成员变量
                                         // 的值
    getProWhite() const
    {
        // 返回 proWhite 成员变量的值
        return cluster.getProWhite();
    }

    // 成员方法：setProWhite（设置白色像素值）
    // 设置 proWhite 成员变量的值。
    __host__ __device__ int         // 返回值：函数是否正确执行，若函数正
                                    // 确执行，返回 NO_ERROR。
    setProWhite(
            unsigned char prowhite  // 白色像素值
    ) {
        // 更新成员变量 cluster 中的 proWhite 属性
        return cluster.setProWhite(prowhite);
    }

    // 成员方法：getGapThred（读取中心像素与邻域像素差值的阈值）
    // 读取 gapThred 成员变量的值。
    __host__ __device__ unsigned char  // 返回: 当前 gapThred 成员变量的值
    getGapThred() const
    {
        // 返回 gapThred 成员变量的值
        return cluster.getGapThred();
    }

    // 成员方法：setGapThred（设置中心像素与邻域像素差值的阈值）
    // 设置 gapThred 成员变量的值。
    __host__ __device__ int         // 返回值：函数是否正确执行，若函数正确
                                    // 执行，返回 NO_ERROR。
    setGapThred(
            unsigned char gapthred  // 中心像素与邻域像素差值的阈值
    ) {
        // 更新成员变量 cluster 中的 gapThred 属性
        return cluster.setGapThred(gapthred);
    }

    // 成员方法：getDiffeThred（读取差分阈值）
    // 读取 diffeThred 成员变量的值。
    __host__ __device__ unsigned char  // 返回: 当前  diffeThred 成员变量的值
    getDiffeThred() const
    {
        // 返回 diffeThred 成员变量的值
        return cluster.getDiffeThred();
    }

    // 成员方法：setDiffeThred（设置差分阈值）
    // 设置 diffeThred 成员变量的值。
    __host__ __device__ int           // 返回值：函数是否正确执行，若函数正确
                                      // 执行，返回 NO_ERROR。
    setDiffeThred(
            unsigned char diffethred  // 差分阈值
    ) {
        // 更新成员变量 cluster 中的 diffeThred 属性
        return cluster.setDiffeThred(diffethred);
    }

    // 成员方法：getSigmaSpace（读取空域参数）
    // 读取 sigmaSpace 成员变量的值。
    __host__ float  // 返回: 当前 sigmaSpace 成员变量的值
    getSigmaSpace() const
    {
        // 返回 sigmaSpace 成员变量的值
        return biFilter.getSigmaSpace();
    }

    // 成员方法：setSigmaSpace（设置空域参数）
    // 设置 sigmaSpace 成员变量的值。
    __host__ int   // 返回值：函数是否正确执行，若函数正确
                              // 执行，返回 NO_ERROR。
    setSigmaSpace(
            float sigmaspace  // 空域参数
    ) {
        // 更新成员变量 biFilter 中的 sigmaSpace 属性
        return biFilter.setSigmaSpace(sigmaspace);
    }

    // 成员方法：getSigmaRange（读取颜色域参数）
    // 读取 sigmaRange 成员变量的值。
    __host__ float  // 返回: 当前 sigmaRange 成员变量的值
    getSigmaRange() const
    {
        // 返回 sigmaRange 成员变量的值
        return biFilter.getSigmaRange();
    }

    // 成员方法：setSigmaRange（设置颜色域参数）
    // 设置 sigmaRange 成员变量的值。
    __host__ int   // 返回值：函数是否正确执行，若函数正确
                              // 执行，返回 NO_ERROR。
    setSigmaRange(
            float sigmarange  // 颜色域参数
    ) {
        // 更新成员变量 biFilter 中的 sigmaRange 属性
        return biFilter.setSigmaRange(sigmarange);
    }

    // 成员方法：getFilterRadius（读取滤波半径）
    // 读取 filterRadius 成员变量的值。
    __host__ __device__ int  // 返回: 当前 filterRadius 成员变量的值
    getFilterRadius() const
    {
        // 返回 filterRadius 成员变量的值
        return biFilter.getRadius();
    }

    // 成员方法：setFilterRadius（设置滤波半径）
    // 设置 filterRadius 成员变量的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确
                              // 执行，返回 NO_ERROR。
    setFilterRadius(
            int filterradius  // 颜色域参数
    ) {
        // 将 filterRadius 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (filterradius < 0 || filterradius > DEF_FILTER_RANGE)
            return INVALID_DATA;

        // 更新成员变量 biFilter 中的 filterRadius 属性
        return biFilter.setRadius(filterradius);
    }

    // 成员方法：getRepeat（读取迭代次数）
    // 读取 repeat 成员变量的值。
    __host__ __device__ int  // 返回: 当前 repeat 成员变量的值
    getRepeat() const
    {
        // 返回 repeat 成员变量的值
        return biFilter.getRepeat();
    }

    // 成员方法：setRepeat（设置迭代次数）
    // 设置 repeat 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setRepeat(
            int repeat       // 迭代次数
    ) {
        // 将 repeat 成员变量赋成新值
        // 若设置的值不在区间内则属于非法数据
        if (repeat < 1)
            return INVALID_DATA;
 
        // 更新成员变量 biFilter 中的 repeat 属性
        return biFilter.setRepeat(repeat);
    }

    // 成员方法：getCloseSize（读取闭运算模板大小）
    // 读取 closeSize 成员变量的值。
    __host__ __device__ int  // 返回: 当前 length 成员变量的值
    getCloseSize() const
    {
        // 返回 closeSize 成员变量的值
        return closeSize;
    }

    // 成员方法：setCloseSize（设置闭运算模板大小）
    // 设置 closeSize 成员变量的值。
    __host__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回 NO_ERROR。
    setCloseSize(
            int closesize    // 闭运算模板大小
    ) {
        if(closesize < 0)
            return INVALID_DATA;
        int errcode = TemplateFactory::getTemplate(&tpl, TF_SHAPE_BOX, 
                                                   closesize, NULL); 
        if (errcode != NO_ERROR) {
             TemplateFactory::putTemplate(tpl);
             return errcode;
        }
        this->closeSize = closesize;
        return NO_ERROR;
    }

    // Host 成员方法：detectRegion（区域检测）
    // 对输入图像进行预处理之后使用检测算法匹配计算，将符合要求的最小外接有向矩
    // 形加入到方法的输出参数中
    __host__ int                           // 返回值：函数是否正确执行，若函数
	                                   // 正确执行，返回 NO_ERROR。 
    detectRegion(
            Image *inimg,                  // 输入图像
            DirectedRect **regionsdetect,  // 接收算法输出的有向矩形指针
            int *regioncount               // 检测区域数目
    );
    
};

#endif
