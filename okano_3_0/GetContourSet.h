// GetContourSet.h 
// 创建人：王媛媛
//
// 有连接性的封闭轮廓的获得（GetContourSet）
// 功能说明：根据设定半径，对输入闭曲线做膨胀处理，再调用 SmoothVector 操作，将
//         环划分成内外环，最终输出内外环交接处点坐标值以及交界处内所有点坐标值。
//
// 修订历史：
// 2012年10月08日（王媛媛）
//     初始版本
// 2012年10月30日（王媛媛）
//     完成进行算法核心代码设计，包括生成初始输入图像，进行膨胀处理，
//     标记值数组的初始化。
// 2012年11月20日（王媛媛）
//     修改代码格式，添加相应的注释。
// 2012年11月20日（王媛媛）
//     实现了闭曲线的膨胀处理，外环的查找。
// 2012年12月04日（王媛媛）
//     更正了闭曲线的外环查找与更新标记值数组值操作。
// 2012年12月15日（王媛媛）
//     使用连通域类似方法实现了内外环点标记值的赋值。
// 2013年01月07日（王媛媛）
//     添加了对于二分类方法的调用。
// 2013年01月09日（王媛媛）
//     更正了代码规范。
// 2013年01月12日（王媛媛）
//     更正了算法主函数，添加参数 inOrder 指示是否按序输出
//     内外环交界处坐标点集。
// 2013年01月13日（王媛媛）
//     更正算法找到内外边界点核函数以及部分代码规范。
// 2013年04月07日（王媛媛）
//     添加二分类方法 TorusSegmentation 作为调用参数。
// 2013年04月09日（王媛媛）
//     修改对于二分类算法的调用过程，
//     添加生成与图像大小一致的 Class 数组 Host 端程序。
// 2013年04月12日（王媛媛）
//     修改内存拷贝错误，使算法调用达到预期效果。更正部分代码规范。
// 2013年04月12日（王媛媛）
//     修改原有的图像转坐标集为已封装完成的接口。
// 2013年04月15日（王媛媛）
//     修改代码注释规范。

#ifndef __GETCONTOURSET_H__
#define __GETCONTOURSET_H__

#include "Image.h"
#include "Template.h"
#include "CoordiSet.h"
#include "TemplateFactory.h"
#include "Morphology.h"
#include "FeatureVecCalc.h"
#include "FeatureVecArray.h"
#include "SmoothVector.h"
#include "Segmentation.h"
#include "TorusSegmentation.h"
#include "ImgConvert.h"
#include "ErrorCode.h"

// 类：GetContourSet
// 继承自：无
// 根据设定的半径，对输入闭曲线做膨胀处理生成圆环，
// 再调用 SmoothVector 操作，对圆环进行划分成内外环，
// 最终输出内外环交接处点坐标值以及交界处内所有点坐标值。
class GetContourSet {

protected:

    // 成员变量：radiusForCurve （膨胀半径）
    // 进行膨胀的半径。
    int radiusForCurve;

    // 成员变量：inorder （按序输出参数）
    // 表示是否按序输出坐标点集，true 表示顺序输出，false 表示乱序输出。
    bool inorder;

    // 成员变量：morForCurve （形态学方法——膨胀）
    // 进行环状曲线膨胀算法，生成圆环。
    Morphology morForCurve;

    // 成员变量：tsForCurve （二分类方法）
    // 进行二分类方法，生成内外圆环。
    TorusSegmentation tsForCurve;

    // Host 成员方法：findMinMaxCoordinates （计算最左最右最上最下点坐标值）
    // 根据输入闭曲线分别找到曲线最左，最右，最上，最下点的对应坐标值
    // 输出到一个一维数组 resultcs 中。
    __host__ int                     // 返回值：函数是否正确执行，
                                     // 若函数正确执行，返回NO_ERROR。
    findMinMaxCoordinates(
            CoordiSet *incoordiset,  // 输入坐标点集
            int *resultcs            // 输出指示最上、最下、最左、最右坐标集数组
    );

    // Host 成员方法：sortContour （按序输出坐标点集）
    // 根据输入 inArray 按顺时针方向顺序输出有序的点集，并将结果
    // 输出到一个坐标集 outcoordiset 中。
    __host__ int                      // 返回值：函数是否正确执行，若函数正确执
                                      // 行，返回NO_ERROR。
    sortContour(
            int inarray[],            // 输入 contour 值，其中 200 表示边界。
            CoordiSet *outcoordiset,  // 输出保存内外交界处坐标点集
            int width, int height     // 图像宽和图像高
    );

public:
    // 构造函数：GetContourSet
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    GetContourSet() {
        // 使用默认值为类的各个成员变量赋值。
        this->radiusForCurve = 2;             // 膨胀半径默认为 2。
        this->inorder = false;                // 默认乱序输出坐标点集。
        this->tsForCurve.setNeighborSize(1);  // 邻域宽度默认为 1。

        // 使用默认值为膨胀算法的各个成员变量赋值。
        int diameter = 2 * this->radiusForCurve + 1;
        Template *tplfordilate;
        dim3 size(diameter, diameter);
        // 设置膨胀圆形模板
        TemplateFactory::getTemplate(&tplfordilate, TF_SHAPE_CIRCLE,
                                     size , NULL);
        this->morForCurve.setTemplate(tplfordilate);
    }

    // 构造函数：GetContourSet
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    GetContourSet(
            int radiusforcurve,  // 膨胀半径
            bool inorder         // 按序输出参数
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->radiusForCurve = 2;             // 半径值默认为 2。
        this->inorder = false;                // 默认乱序输出坐标点集。
        this->tsForCurve.setNeighborSize(1);  // 邻域宽度默认为 1。
        // 使用默认值为膨胀算法的各个成员变量赋值。
        int diameter = 2 * this->radiusForCurve + 1;
        Template *tplfordilate;
        dim3 size(diameter, diameter);
        // 设置膨胀圆形模板
        TemplateFactory::getTemplate(&tplfordilate, TF_SHAPE_CIRCLE,
                                     size , NULL);
        this->morForCurve.setTemplate(tplfordilate);

        // 根据参数列表中的值设定成员变量的初值
        setRadius(radiusforcurve);
        setOrder(inorder);
    }

    // 析构函数：~GetContourSet
    // 用于释放膨胀模板元素。
    __host__
    ~GetContourSet() {
        // 如果模板元素已经申请，则需要释放掉模板元素。
        if (this->morForCurve.getTemplate() != NULL)
            TemplateFactory::putTemplate(this->morForCurve.getTemplate());
    }

    // 成员方法：getRadius（获取半径值）
    // 获取成员变量 radiusForCurve 的值。
    __host__ __device__ int  // 返回值：成员变量 radiusForCurve 的值
    getRadius() const
    {
        // 返回 radiusForCurve 成员变量的值。
        return this->radiusForCurve;
    } 

    // 成员方法：setRadius（设置半径值）
    // 设置成员变量 radiusForCurve 的值。
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执
                                // 行，返回 NO_ERROR。
    setRadius(
            int radiusforcurve  // 膨胀半径
    ) {
        // 若半径超出允许范围不赋值直接退出
        if (radiusforcurve <= 0)
            return INVALID_DATA;
        // 将 radiusForCurve 成员变量赋成新值
        this->radiusForCurve = radiusforcurve;
        return NO_ERROR;
    }

    // 成员方法：getOrder（获取是否按序变量的值）
    // 获取成员变量 inorder 的值。
    __host__ __device__ bool  // 返回值：成员变量 inorder 的值
    getOrder() const
    {
        // 返回 inorder 成员变量的值。
        return this->inorder;
    } 

    // 成员方法：setOrder（设置是否按序变量的值）
    // 设置成员变量 inOrder 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setOrder(
            bool inorder     // 按序输出参数
    ) {
        // 将 inorder 成员变量赋成新值
        this->inorder = inorder;

        return NO_ERROR;
    }

    // Host 成员方法：getContourSet（执行有连接性的封闭轮廓的获得算法）
    // 根据输入输入闭曲线坐标点集，生成内环内所有点坐标到 innerCoordiset 中，
    // 生成内外环交界处坐标点集并输出到 contourCoordiset 中。
    __host__ int                         // 返回值：函数是否正确执行，
                                         // 若函数正确执行，返回NO_ERROR。
    getContourSet(
            CoordiSet *incoordiset,      // 输入闭曲线坐标点集
            CoordiSet *innercoordiset,   // 输出内环内所有点坐标点集
            CoordiSet *contourcoordiset  // 输出内外环交界处坐标点集
    );
};

#endif

