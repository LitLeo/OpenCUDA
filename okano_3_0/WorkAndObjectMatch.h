// WorkAndObjectMatch.h
// 创建人：罗劼
//
// WORK and OBJETC 匹配（WORK and OBJETC match）
// 功能说明：给定一张 TEST 图像，通过给定的一组标准的 WORK 图像，用这组标准的
//           WORK 图像对 TEST 图像进行匹配，得到 TEST 中 WORK 图像的位置，然后
//           将 WORK 图像从 TEST 图像中 clip 出来，之后的匹配都是针对 WORK 图像
//           进行匹配。再给定一组 OBJECT 图像，每个 OBJECT 分别对 WORK 图像进行
//           匹配，每个 OBJECT 有不同的 TEMPLATE 样本，找到每个 OBJECT 中与 WORK
//           最匹配的 TEMPLATE 以及所使用的旋转角
//
// 修订历史：
// 2012年12月13日（罗劼）
//     初始版本
// 2012年12月20日（李冬）
//     修正了代码中函数调用的错误。
// 2013年03月08日（侯怡婷）
//     更新缩小图像算法的名字。
// 2013年04月19日（于玉龙）
//     更新了 DownSmapleImage 的函数名。

#ifndef __WORKANDOBJTCTMATCH_H__
#define __WORKANDOBJTCTMATCH_H__

#include "ErrorCode.h"
#include "Image.h"
#include "ImageMatch.h"
#include "RotateTable.h"


#include "AffineTrans.h"
#include "DownSampleImage.h"
#include "RoiCopy.h"

// 结构体：ImagesInfo（保存一组 TEMPLATE 的信息）
// 用来保存一组在大小、方向上有差异的 TEMPLATE 的信息，包括这组 TEMPLATE 的图像
// 数据、TEMPLATE 的数量，匹配使用的旋转表、匹配的摄动范围、匹配的摄动中心
typedef struct ImagesInfo_st {
    Image **images;            // 这组 TEMPLATE 的图像数据
    int count;                 // 这组 TEMPLATE 图像的数量
    RotateTable *rotateTable;  // 这组 TEMPLATE 所使用的旋转表
    int dWidth;                // TEMPLATE 进行匹配时的摄动范围的宽
    int dHeight;               // TEMPLATE 进行匹配时的摄动范围的高
    int dX;                    // TEMPLATE 进行匹配时摄动中心的横坐标
    int dY;                    // TEMPLATE 进行匹配时摄动中心的纵坐标
} ImagesInfo;

// 类：WorkAndObjectMatch
// 继承自：无
// 给定一张 TEST 图像，通过给定的一组标准的 WORK 图像，用这组标准的WORK 图像
// 对 TEST 图像进行匹配，得到 TEST 中 WORK 图像的位置，然后将 WORK 图像从 TEST 
// 图像中 clip 出来，之后的匹配都是针对 WORK 图像进行匹配。再给定一组 OBJECT 图
// 像，每个 OBJECT 分别对 WORK 图像进行匹配，每个 OBJECT 有不同的 TEMPLATE 样
// 本，找到每个 OBJECT 中与 WORK最匹配的 TEMPLATE 以及所使用的旋转角
class WorkAndObjectMatch {

protected:

    // 成员变量：normalWork（一组标准的 WORK）
    // 一组标准的 WORK 图像信息，用来对 TEST 图像进行匹配，在 TEST 图像找到
    // WORK 图像
    ImagesInfo *normalWork;

    // 成员变量：objects（OBJECT 图像）
    // 每个 OBJECT 图像有不同的 TEMPLATE，这些 TEMPLATE 在大小、方向上存在差异，
    // 所以每个 OBJECT 是用 ImagesInfo 来存储
    ImagesInfo *objects;

    // 成员变量：objectCount（OBJECT 的数量）
    // 用来记录 OBJECT 的数量
    int objectCount;

    // 成员变量：angleCount（设置对变形的 TEST 图像生成的图像的数量）
    // 对匹配得到的变形的 TEST 图像用 AFFINE 生成 2 * angleCount 个角度的回旋
    // 图像，旋转角的单位是 0.2，范围为
    //  θ - 0.2 * angleCount ~ θ + 0.2 * angleCount
    int angleCount;

    // 成员函数：setDefParameter（设置成员变量的默认值）
    // 为所有的成员变量设置默认值
    __host__ __device__ void  // 返回值：无
    setDefParameter()
    {
        this->normalWork = NULL;  // 设置 normalWork 的默认值为空
        this->objects = NULL;     // 设置 objects 的默认值为空
        this->objectCount = 0;    // 设置 OBJECT 的默认值为 0
        this->angleCount = 10;    // 设置 angleCount 的默认值为 10
    }

    // 成员方法：getMatchWork（获取 TEST 图像中的 WORK 图像）
    // 获取 TEST 图像中的 WORK 图像，并从 TEST 图像中将 WORK 图像 clip 出来
    __host__ int          // 返回指：函数是否正确执行，若函数正确执行，返回
                          // NO_ERROR
    getMatchWork(
            Image *test,  // TEST 图像
            Image *work   // WORK 图像
    );

public:

    // 构造函数：WorkAndObjectMatch
    // 无参数版本的构造函数，所有的成员变量都使用默认值
    __host__ __device__
    WorkAndObjectMatch()
    {
        // 为所有的成员变量设置默认值
        setDefParameter();  
    }

    // 构造函数：WorkAndObjectMatch
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值可以在程序运行过程
    // 中改变
    __host__ __device__
    WorkAndObjectMatch(
            ImagesInfo *normalwork,  // 标准的 WORK 图像
            ImagesInfo *objects,     // OBJECT 图像
            int objectcount,         // OBJECT 数量
            int anglecount           // angleCount 的值
    ) {
        // 使用默认值初始化各成员变量
        setDefParameter();

        // 根据参数列表中的值设定成员变量的初值
        setNormalWork(normalwork);
        setObject(objects, objectcount);
        setAngleCount(anglecount);
    }

    // 成员方法：getNormalWork（获取标准 WORK 图像信息的指针）
    // 获取成员变量 normalWork 的值
    __host__ __device__ ImagesInfo *  // 返回值：成员变量 normalWork 的值
    getNormalWork() const
    {
        // 返回成员变量 normalWork 的值
        return this->normalWork;
    }

    // 成员方法：setNormalWork（设置标准 WORK 图像的信息）
    // 设置 normalWork 的值
    __host__ __device__ int         // 返回值：函数是否正确执行，若函数正确
                                    // 执行，返回 NO_ERROR
    setNormalWork(
            ImagesInfo *normalwork  // 新的标准 WORK 图像
    ) {
        // 设置新的 normalWork 的值
        this->normalWork = normalwork;

        // 处理完毕，返回 NO_ERROR
        return NO_ERROR;
    }

    // 成员方法：getObject（获取 OBJECT 的信息）
    // 获取 objects 的值
    __host__ __device__ ImagesInfo *  // 返回值：成员变量 objects 的值
    getObject() const
    {
        // 返回成员变量 objects 的值
        return this->objects;
    }

    // 成员方法：getObjectCount（获取 OBJECT 的数量）
    // 获取 objectCount 的值
    __host__ __device__ int  // 返回值：成员变量 objectCount 的值
    getObjectCount() const
    {
        // 返回成员变量 objectCount 的值
        return this->objectCount;
    }

    // 成员方法：setObject（设置 OBJECT 的信息）
    // 设置 objects 和 objectCount 的值
    __host__ __device__ int       // 返回值：函数是否正确执行，若函数正确执行，
                                  // 返回 NO_ERROR
    setObject(
            ImagesInfo *objects,  // 新的 OBJECT 的信息
            int objectcount       // 新的 OBJECT 的数量
    ) {
        // 检查 objectcount 是否合法，若不合法，直接返回
        if (objectcount <= 0)
            return INVALID_DATA;

        // 设置新的 objects 的值
        this->objects = objects;
        // 设置新的 objectCount 的值
        this->objectCount = objectcount;

        // 处理完毕，返回 NO_ERROR
        return NO_ERROR;
    }

    // 成员方法：getAngleCount（获取对变形 TEST 图像需要生成的角度的数量）
    // 获取 angleCount 的值
    __host__ __device__ int  // 返回值：返回 angleCount 的值
    getAngleCount() const
    {
        // 返回 angleCount 的值
        return this->angleCount;
    }

    // 成员方法：setAngleCount（设置需要对变形的 TEST 生成的不用旋转角的数量）
    // 设置 angleCount 的值
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR
    setAngleCount(
            int anglecount   // 新的 angleCount 的值
    ) {
        // 判断 anglecount 的值是否合法，若不合法，直接返回错误
        if (anglecount <= 0)
            return INVALID_DATA;

        // 更新 angleCount 的值
        this->angleCount = anglecount;

        // 处理完毕，返回
        return NO_ERROR;
    }

    // 成员方法：workAndObjectMatch（进行 WORK and OBJECT 进行匹配操作）
    // 进行 WORK and OBJECT 进行匹配操作的主方法，如果 rescount 小于 objectCount
    // 的值，则只对前 rescount 的 OBJECT 进行处理
    __host__ int            // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR
    workAndObjectMatch(
            Image *test,    // TEST 图像数据
            MatchRes *res,  // 存放每个 OBJECT 匹配后得到的结果
            int rescount    // res 数组的长度
    );
};

#endif

