// Curve.h
//
// 曲线定义（Curve）
// 功能说明：定义了曲线的数据结构和对曲线的基本操作。


#ifndef __CURVE_H__
#define __CURVE_H__

#include "ErrorCode.h"
#include "GeoPrimaryProperties.h"
                                                                                
                                                                               
// 结构体：Curve（曲线）
// 该结构体定义了曲线的数据结构，其中包含了曲线的数据和和曲线相关的逻辑参数，
// 如曲线中点的数量和曲线中的各个点的坐标。注意，曲线的锚点（Anchor）为坐标 
// (0, 0) 点。
typedef struct Curve_st {
    bool closed;                               // 标记曲线是否闭合
    int startCordiX;                           // 曲线的起点 x 坐标
    int startCordiY;                           // 曲线的起点 y 坐标
    int endCordiX;                             // 曲线的终点 x 坐标，闭合曲线时
                                               // 与起始 x 坐标相同
    int endCordiY;                             // 曲线的终点 y 坐标，闭合曲线时
                                               // 与起始 y 坐标相同
    int maxCordiX;                             // 曲线上的最右点的 x 坐标
    int minCordiX;                             // 曲线上的最左点的 x 坐标
    int maxCordiY;                             // 曲线上的最下点的 y 坐标
    int minCordiY;                             // 曲线上的最上点的 y 坐标
    int aveX;                                  // 曲线 x 坐标的平均值
    int aveY;                                  // 曲线 y 坐标的平均值
    size_t curveLength;                        // 曲线上点空间的数量  

    int *crvData;                              // 曲线上点的信息，其中相邻的两
                                               // 个下标的数组数据表示一个点。
                                               // 即，曲线中第 i 个点对应的数组
                                               // 下标为 2 * i 和 2 * i + 1，其
                                               // 中下标为 2 * i 的数据表示该点
                                               // 的横坐标，下标为 2 * i + 1 的
                                               // 数据表示该点的纵坐标。这些坐
                                               // 标是参照锚点的坐标，因此，坐
                                               // 标可以含有负值。
    int  smWindowSize;                         // the nieghors’ number for 
                                               // smoothing 曲线上的坐标
    float *smCurveCordiXY;                     // smoothed 曲线上的点的坐标信息
                                               // 设计规则同 crvData。

    float *tangent;                            // 曲线上各点处的切线斜率，x 轴
                                               // 右向为正，y 轴左下为正。
    bool geoProperty;                          // 决定下列参数是否有效。
    GeoPrimaryProperties*  primaryProperties;  // 待用变量，暂时全部设为空值。
    unsigned char eptGrayFeature1[3];          // expected gray feature
    unsigned char eptGrayFeatureMin1[3]; 
    unsigned char eptGrayFeatureMax1[3]; 

    unsigned char eptGrayFeature2[3];          // expected gray feature
    unsigned char eptGrayFeatureMin2[3]; 
    unsigned char eptGrayFeatureMax2[3]; 

    unsigned char eptGrayFeature3[3];          // expected gray feature
    unsigned char eptGrayFeatureMin3[3]; 
    unsigned char eptGrayFeatureMax3[3]; 
} Curve;                                                                       
                                                                                
// 结构体：CurveCuda（曲线的 CUDA 相关数据）
// 该结构体定义了与 CUDA 相关的曲线数据。该结构体通常在算法内部使用，上层用户通
// 常见不到也不需要了解此结构体
typedef struct CurveCuda_st {
    Curve crvMeta;       // 曲线数据，保存了对应的曲线逻辑数据。
    size_t capacity;     // 曲线内实际点的数量
    int deviceId;        // 当前数据所处的内存。如果数据在 GPU 的内存上，则
                         // deviceId 为对应设备的编号；若 deviceId < 0，则说明
                         // 数据存储在 Host 内存上。
    int smCurveIsValid;  // 标记当前曲线对象内 smCurveCordiXY 曲线数据是否有效
                         // 由于 smCurveCordiXY  曲线在基础算法中并没有使用，
                         // 所以定义此变量做标记。
    int tangentIsValid;  // 标记当前曲线对象内 tangent 数据是否有效，功能同
                         // smCurveIsValid 。

} CurveCuda;

// 宏：CURVE_CUDA
// 给定 Curve 型指针，返回对应的 CurveCuda 型指针。该宏通常在算法内部使
// 用，用来获取关于 CUDA 的曲线数据。
#define CURVE_CUDA(crv)                                                    \
    ((CurveCuda *)((unsigned char *)(crv) -                                \
                   (unsigned long)(&(((CurveCuda *)NULL)->crvMeta))))

// 类：CurveBasicOp（曲线基本操作）
// 继承自：无
// 该类包含了对于曲线的基本操作，如曲线的创建与销毁、曲线的读取、曲线在各地址空
// 间之间的拷贝等。要求所有的曲线实例，都需要通过该类创建，否则，将会导致系统运
// 行的紊乱（需要保证每一个 Curve 型的数据都有对应的 CurveCuda 数据）。
class CurveBasicOp {

public:

    // Host 静态方法：newCurve（创建曲线）
    // 创建一个新的曲线实例，并通过参数 outcrv 返回。注意，所有系统中所使用的
    // 曲线都需要使用该函数创建，否则，就会导致无法找到对应的 CurveCuda 型数据
    // 而使系统执行紊乱。
    static __host__ int     // 返回值：函数是否正确执行，若函数正确执行，返
                            // 回 NO_ERROR。
    newCurve(
            Curve **outcrv  // 返回的新创建的曲线指针。
    );

    // Host 静态方法：deleteCurve（销毁曲线）
    // 销毁一个不再被使用曲线实例。
    static __host__ int   // 返回值：函数是否正确执行，若函数正确执行，返回
                          // NO_ERROR。
    deleteCurve(
            Curve *incrv  // 输入参数，不再需要使用、需要被释放的曲线。
    );

    // Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
    // 针对空曲线，如果 crvData 为空，则直接在当前 Device 内存中为其申请一段
    // 指定的大小的空间，不赋值;如果 crvData不为空，则为曲线点数据直接赋值 
    // crvData 参数数据。如果不是空曲线，则该方法会报错。
    static __host__ int          // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    makeAtCurrentDevice(
            Curve *crv,          // 曲线，要求是空曲线。
            size_t count,        // 指定的曲线中点的数量。
            int *crvData = NULL  // 坐标数据
    );

    // Host 静态方法：makeAtHost（在 Host 内存中构建数据）
    // 针对空曲线，如果 crvData 为空，则直接在 Host 内存中为其申请一段指定
    // 的大小的空间，不赋值；如果 crvData不为空，则为曲线点数据直接赋值
    // crvData 参数数据。如果不是空曲线，则该方法会报错。
    static __host__ int          // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    makeAtHost(
            Curve *crv,          // 曲线，要求是空曲线。
            size_t count,        // 指定的曲线中点的数量。
            int *crvData = NULL  // 坐标数据
    );

    // Host 静态方法：readFromFile（从文件读取曲线）
    // 从文件中读取曲线数据，由于目前国际上没有关于曲线数据文件的标准，因此，我
    // 们设计了自己的曲线文件数据。该文件数据包含 3 个区段。第一个区段为 4 字节
    // 表示文件类型，统一为 CRVT；第二个区段为 4 * 7 个字节，为七个无符号整型数
    // 据，分别表示曲线是否闭合、曲线上最右点的 x 坐标、曲线上最左点的 x 坐标、
    // 曲线上最下点的 y 坐标、曲线上最上点的 y 坐标、曲线中点空间的个数、曲线内
    // 实际点数目；其后的数据表示曲线中各点的坐标，存储的格式同内存中存储的格式
    // 是相同的。不过上层用户不必过于关心文件的格式，因为通常这一文件格式对用户
    // 透明。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    readFromFile(
            const char *filepath,  // 曲线数据文件的路径
            Curve *outcrv          // 输出参数，从稳健读取曲线后，将曲线的内容
                                   // 存放于其中
    );
                                                                               
    // Host 静态方法：writeToFile（将曲线写入文件）
    // 将曲线中的数据写入到文件中。由于目前国际上没有关于曲线数据文件的标准，因
    // 此，我们设计了自己的曲线文件数据。该文件数据包含 3 个区段。第一个区段为
    // 4 字节，表示文件类型，统一为 CRVT；第第二个区段为 4 * 7 个字节，为七个无
    // 符号整型数据，分别表示曲线是否闭合、曲线上最右点的 x 坐标、曲线上最左点
    // 的 x 坐标、曲线上最下点的 y 坐标、曲线上最上点的 y 坐标、曲线中点空间的
    // 个数、曲线中实际点的个数；其后的数据表示曲线中各点的坐标，存储的格式同
    // 内存中存储的格式是相同的。不过上层用户不必过于关心文件的格式，因为通常
    // 这一文件格式对用户透明。此外本函数要求输入的曲线必须有数据。经过本函数，
    // 曲线的数据会自动的拷贝到 Host 内存中，因此，不要频繁的调用本函数，这样
    // 会使数据频繁的在 Host 与 Device 间拷贝，造成性能下降。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    writeToFile(
            const char *filepath,  // 曲线数据文件的路径
            Curve *incrv           // 待写入文件的曲线，写入文件后，曲线将会自
                                   // 动的存放入 Host 内存中，因此不要频繁进行
                                   // 该操作，以免曲线在 Host 和 Device 之间频
                                   // 繁传输而带来性能下降。
    );

    // Host 静态方法：copyToCurrentDevice（将曲线拷贝到当前 Device 内存上）
    // 这是一个 In-Place 形式的拷贝。如果曲线数据本来就在当前的 Device 上，则该
    // 函数不会进行任何操作，直接返回。如果曲线数据不在当前 Device 上，则会将数
    // 据拷贝到当前 Device 上，并更新 crvData 指针。原来的数据将会被释放。
    static __host__ int   // 返回值：函数是否正确执行，若函数正确执行，返回
                          // NO_ERROR。
    copyToCurrentDevice(
            Curve *crv    // 需要将数据拷贝到当前 Device 的曲线。
    );

    // Host 静态方法：copyToCurrentDevice（将曲线拷贝到当前 Device 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srccrv 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstcrv，且数据是存储于当前的 Device 上的。如果
    // dstcrv 中原来存在有数据，且原来的数据同新的数据尺寸相同，也存放在当前
    // Device 上，则覆盖原内容，不重新分配空间；否则原数据将会被释放，并重新申
    // 请空间。
    static __host__ int     // 返回值：函数是否正确执行，若函数正确执行，返
                            // 回 NO_ERROR。
    copyToCurrentDevice(
            Curve *srccrv,  // 源曲线，要求曲线中必须有数据。
            Curve *dstcrv   // 目标曲线，要求该曲线板必须经过 newCurve 申请。
    );

    // Host 静态方法：copyToHost（将曲线拷贝到 Host 内存上）
    // 这是一个 In-Place 形式的拷贝。如果曲线数据本来就在 Host 上，则该函数不会
    // 进行任何操作，直接返回。如果曲线数据不在 Host 上，则会将数据拷贝到 Host
    // 上，并更新 crvData 指针。原来的数据将会被释放。
    static __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                         // NO_ERROR。
    copyToHost(
            Curve *crv   // 需要将数据拷贝到 Host 上的曲线。
    );

    // Host 静态方法：copyToHost（将曲线拷贝到 Host 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srccrv 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstcrv，且数据是存储于 Host 上的。如果 dstcrv
    // 中原来存在有数据，如果原来的数据同新的数据尺寸相同，且也存放在 Host 上，
    // 则覆盖原内容，但不重新分配空间；否则原数据将会被释放，并重新申请空间。
    static __host__ int     // 返回值：函数是否正确执行，若函数正确执行，返
                            // 回 NO_ERROR。
    copyToHost(
            Curve *srccrv,  // 源曲线，要求曲线中必须有数据。
            Curve *dstcrv   // 目标图像，要求该曲线必须经过 newCurve 申请。
    );   

    // Host 静态方法：assignData（为曲线数据赋值）
    // 为曲线内的数据赋值，对于数据不为空的曲线则会覆盖原数据。同时为根据新数据
    // 重新计算曲线的各个属性值。此函数目前只适用于 Host 端。
    static __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                         // 回 NO_ERROR。
    assignData(
            Curve *crv,
            int *data,
            size_t count
    );

    // Host 静态方法：setSmCurveValid（设置 smoothing 曲线有效）
    // 因为 smoothing 曲线在初始化曲线时默认设置为无效，如果需要在使用 
    // smoothing 曲线数据，则需要在 newCurve 后执行此函数。
    static __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                         // 回 NO_ERROR。
    setSmCurveValid(
            Curve *crv
    );

    // Host 静态方法：setTangentValid（设置曲线斜率有效）
    // 因为曲线斜率在初始化曲线时默认设置为无效，如果需要在使用曲线斜率数据，
    // 则需要在 newCurve 后执行此函数。
    static __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                         // 回 NO_ERROR。
    setTangentValid(                                                            
            Curve *crv
    );
};

#endif

