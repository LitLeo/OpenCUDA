// ConsolidateAndIdentifyContours.h
// 创建人：于玉龙
//
// 实例检出（ConsolidateAndIdentifyContours）
// 功能说明：利用图像中图形的轮廓信息检测物体。
//
// 修订历史：
// 2014年09月26日（于玉龙）
//     初始版本
// 2014年09月28日（于玉龙）
//     实现了对图像的前处理。
// 2014年09月29日（于玉龙）
//     实现了根据标准轮廓图像和标准区域图像标记检出轮廓的处理步骤。


#ifndef __CONSOLIDATEANDIDENTIFYCONTOURS_H__
#define __CONSOLIDATEANDIDENTIFYCONTOURS_H__

#include "Image.h"
#include "ErrorCode.h"
#include "Morphology.h"
#include "CombineImage.h"
#include "RobustEdgeDetection.h"
#include "Thinning.h"
#include "Binarize.h"


// 类：ConsolidateAndIdentifyContours
// 继承自：无
// 对图像进行边缘检测，利用边缘信息，匹配已知物体的边缘轮廓信息。
class ConsolidateAndIdentifyContours
{
protected:
    // 成员变量：dilationRad（膨胀处理半径）
    // 图像提取边缘后需要通过膨胀和细化处理取得连续的轮廓信息，这个变量记录了
    // 进行膨胀操作的半径。
    unsigned dilationRad;

    // 成员变量：trackRad（搜索半径）
    // 检出的边缘需要匹配标准物体的边缘，这里是进行匹配过程中容许的误差半径。
    unsigned trackRad;

    // 成员变量：primitiveContour（标准轮廓图像）
    // 标准轮廓图像，其中不同物体的轮廓通过标记值 1、2、3 等记录于图像中。
    Image *primitiveContour;

    // 成员变量：primitiveRegion（标准区域图像）
    // 标准区域图像，其中不同物体的区域通过标记值 1、2、3 等记录于图像中。
    Image *primitiveRegion;

    // 静态成员变量：redMachine（边缘检测处理机）
    // 进行边缘检测的算法。
    static RobustEdgeDetection *redMachine;

    // 成员变量：morphMachine（膨胀处理机）
    // 进行膨胀处理的算法。
    Morphology morphMachine;

    // 成员变量：combineMachine（图像合并处理机）
    // 用于合并有多个边缘检测处理机检测出来的
    CombineImage combineMachine;

    // 成员变量：thinMachine（细化处理机）
    // 用于将边缘检测的合成输出图像进行细化处理
    Thinning thinMachine;

    // 成员变量：binMachine（二值化处理机）
    // 用于将边缘处理得到的最终图像转化为二值图像，以方便进行后续的匹配处理。
    //Binarize binMachine;

    // Host 成员方法：initRedMachine（初始化边缘检测处理机）
    // 对边缘检测处理机进行初始化。考虑到边缘检测在整个系统范围内采用了同样的
    // 参数，因此这里用静态成员，避免导致重复的资源申请与释放的操作。
    __host__ int  // 返回值：错误码，如果处理正确返回 NO_ERROR。
    initRedMachine();

    // Host 成员方法：initMorphMachine（初始化膨胀处理机）
    // 对膨胀处理机进行初始化。需要将膨胀处理参数从模版工厂中找出对应的模版，
    // 赋给膨胀处理机。
    __host__ int  // 返回值：错误码，如果处理正确返回 NO_ERROR。
    initMorphMachine();
public:
    // Host 成员方法：getCsldtContoursImg（获取轮廓图像）
    // 在进行轮廓匹配前调用该函数，从输入图片生成相对标准的轮廓图像。生成该轮
    // 廓图像的步骤包括：（1）首先利用多种不同参数进行边缘检测，并合并这些检测
    // 结果，形成比较粗糙的边缘；（2）然后通过膨胀连接边缘的断线；（3）之后，
    // 通过细化算法将较粗的边缘转换为单像素线宽的曲线线条。
    __host__ int           // 返回值：错误码，如果处理正确返回 NO_ERROR。
    getCsldtContoursImg(
            Image *inimg,  // 输入图像，通常为原始图像
            Image *outimg  // 输出图像，有单像素线宽的曲线线条构成的图像
    );

    // Host 成员方法：searchPrimitiveContour（匹配并标记轮廓）
    // 从检测中的轮廓图像中匹配标准轮廓图像中的相关轮廓。被匹配上的边缘将被标记
    // 成对应的标号信息，未匹配上的轮廓被标记为异常点。
    __host__ int
    searchPrimitiveContour(
            Image *inimg,       // 输入图像，已检测出来的边缘图像
            Image *outimg,      // 输出图像，匹配并被标记出来的轮廓图像。
            Image *abnormalimg  // 异常点图像
    );

public:
    // 构造函数：ConsolidateAndIdentifyContours
    // 无参数版本的构造函数，各种处理半径默认值为 1。
    __host__
    ConsolidateAndIdentifyContours()
    {
        // 向各个参数赋默认值
        dilationRad = 1;
        trackRad = 1;
        primitiveContour = NULL;
        primitiveRegion = NULL;

        // 初始化各个算法的处理器
        initRedMachine();
        initMorphMachine();
    }

    // 构造函数：ConsolidateAndIdentifyContours
    // 传递参数包括膨胀半径和搜索半径，默认值为 1。
    __host__ __device__
    ConsolidateAndIdentifyContours(
            unsigned dilationrad,    // 膨胀处理半径
            unsigned trackrad,       // 搜索半径
            Image *prmtcont = NULL,  // 标准轮廓图像
            Image *prmtreg = NULL    // 标准区域图像
    ) {
        // 向各个参数赋默认值
        dilationRad = 1;
        trackRad = 1;
        primitiveContour = NULL;
        primitiveRegion = NULL;

        // 初始化各个算法的处理器
        initRedMachine();
        initMorphMachine();

        // 向各个参数赋指定值
        setDilationRad(dilationRad);
        setTrackRad(trackRad);
    }

    // 成员方法：getDilationRad（获取膨胀处理半径）
    // 获取膨胀处理半径。
    __host__ __device__ unsigned  // 返回值：膨胀处理半径
    getDilationRad()
    {
        // 直接返回膨胀处理半径参数。
        return this->dilationRad;
    }

    // 成员方法：setDilationRad（设置膨胀处理半径）
    // 设置膨胀处理半径。
    __host__ __device__ int       // 返回值：错误码，如果处理正确返回 
                                  // NO_ERROR。
    setDilationRad(
            unsigned dilationrad  // 新的膨胀处理半径
    ) {
        // 取出原来的膨胀处理半径，已被模板生成失败时可以回滚到原来的参数。
        unsigned oldrad = this->dilationRad;

        // 由于膨胀半径可以为任何非负整数，因此所有的无符号数都是合法的输入。
        this->dilationRad = dilationrad;

        // 如果新设置的半径和原来的半径是一样的，则不需要申请新的模板，直接返
        // 回即可。
        if (oldrad == this->dilationRad)
            return NO_ERROR;

        // 根据新设置的半径，更新膨胀处理器中对应的模板。
        int errcode;
        errcode = initMorphMachine();

        // 如果申请新模板失败，则回滚到原来的半径参数。
        if (errcode != NO_ERROR)
            this->dilationRad = oldrad;

        // 处理完毕，返回。
        return errcode;
    }

    // 成员方法：getTrackRad（获取搜索半径）
    // 获取搜索半径。
    __host__ __device__ unsigned  // 返回值：搜索半径
    getTrackRad()
    {
        // 直接返回搜索半径参数。
        return this->trackRad;
    }

    // 成员方法：setTrackRad（设置搜索半径）
    // 设置搜索半径。
    __host__ __device__ int       // 返回值：错误码，如果处理正确返回 
                                  // NO_ERROR。
    setTrackRad(
            unsigned trackrad  // 新的搜索半径
    ) { 
        // 由于搜索半径可以为任何非负整数，因此所有的无符号数都是合法的输入。
        this->trackRad = trackrad;
        return NO_ERROR;
    }

    // 成员方法：getPrimitiveContour（获取标准轮廓图像）
    // 获取标准轮廓图像。
    __host__ __device__ Image *  // 返回值：标准轮廓图像
    getPrimitiveContour()
    {
        // 直接返回标准轮廓图像参数。
        return this->primitiveContour;
    }

    // 成员方法：setPrimitiveContour（设置标准轮廓图像）
    // 设置标准轮廓图像。
    __host__ __device__ int  // 返回值：错误码，如果处理正确返回 NO_ERROR。
    setPrimitiveContour(
            Image *prmtcont  // 新的标准轮廓图像
    ) {
        // 适用新的标准轮廓图像替换原来的图像
        this->primitiveContour = prmtcont;
        return NO_ERROR;
    }

    // 成员方法：getPrimitiveRegion（获取标准区域图像）
    // 获取标准区域图像。
    __host__ __device__ Image *  // 返回值：标准区域图像
    getPrimitiveRegion()
    {
        // 直接返回标准区域图像参数。
        return this->primitiveRegion;
    }

    // 成员方法：setPrimitiveRegion（设置标准区域图像）
    // 设置标准轮廓图像。
    __host__ __device__ int  // 返回值：错误码，如果处理正确返回 NO_ERROR。
    setPrimitiveRegion(
            Image *prmtreg   // 新的标准区域图像
    ) {
        // 适用新的标准区域图像替换原来的图像
        this->primitiveRegion = prmtreg;
        return NO_ERROR;
    }
};

#endif
