// CurveFluctuPropers.h
// 创建人：邱孝兵
//
// 曲线波动特征属性（CurveFluctuPropers）
// 功能说明：定义了曲线的波动特征属性，主要包含了该曲线中若干个最大偏移
// 距离（相对于平滑曲线的偏移距离），出现这些最大偏移距离的点的横纵坐标
// 以及这些偏移的平均值。
// 修订历史：
// 2013年08月26日（邱孝兵）
//     初始版本


#ifndef __CURVEFLUCTUPROPERS_H__
#define __CURVEFLUCTUPROPERS_H__


// 结构体：CurveFluctuPropers（曲线波动特征属性）
// 该结构体定义了曲线的波动特征属性，主要包含了该曲线中若干个最大偏移
// 距离（相对于平滑曲线的偏移距离），出现这些最大偏移距离的点的横纵坐标
// 以及这些偏移的平均值。
typedef struct CurveFluctuPropers_st{
    int smNieghbors;   // 外部设定参数，曲线平滑邻域大小
    int maxFluctuNum;  // 外部设定参数，统计的最大偏移点的个数
    int *maxFluctu;    // 统计得出的 maxFluctuNum 个最大偏移距离
    int *maxFluctuX;   // maxFluctuNum 个最大偏移点的横坐标
    int *maxFluctuY;   // maxFluctuNum 个最大偏移点的纵坐标
    int aveFluctu;     // 偏移距离均值
    int xyAveFluctu;   // 偏移坐标均值
} CurveFluctuPropers;


// 类：CurveFluctuPropersBasicOp（曲线波动特征属性基本操作）
// 继承自：无
// 该类包含了对于曲线波动特征属性的基本操作，如其的创建与销毁，在
// 主机端和设备端的互相拷贝。要求所有的曲线波动特征属性的基本操作都要通过
// 该类创建，否则将会导致系统运行的紊乱。
class CurveFluctuPropersBasicOp {

public:

    // Host 静态方法：makeAtHost（在 Host 内存中构建数据）
    // 针对空向量组，在 Host 内存中为其申请一段指定的大小的空间，这段空间
    // 中的数据是未被赋值的混乱数据。
    static __host__ int               // 返回值：函数是否正确执行，若函数
                                      // 正确执行，返回 NO_ERROR
    makeAtHost(
            CurveFluctuPropers *cfp,  // 曲线波动特征，要求为空
            int maxFluctuNum          // 统计的最大偏移点的个数
    );

    // Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
    // 针对空向量组，在当前 Device 内存中为其申请一段指定的大小的空间，这段空间
    // 中的数据是未被赋值的混乱数据。
    static __host__ int               // 返回值：函数是否正确执行，若函数
                                      // 正确执行，返回 NO_ERROR
    makeAtCurrentDevice(
            CurveFluctuPropers *cfp,  // 曲线波动特征，要求为空
            int maxFluctuNum          // 统计的最大偏移点的个数
    );    

    // Host 静态方法：copytToHost 方法（将当前 Device 内存中数据拷贝到 Host 端）
    // 这是一个 Out-Place 形式的拷贝。会为 dstcfp 分配响应大小的空间，然后把
    // srccfp 中的数据拷贝到 dstcfp 中。这里只针对 CurveFluctuPropers 的数组字段
    // 进行拷贝，其他单值字段始终只做 Host 端的互相拷贝，并不涉及主机设备之间的
    // 传输
    static __host__ int                  // 返回值：函数是否正确执行，若函数正确
                                         // 执行，返回 NO_ERROR。
    copyToHost(
            CurveFluctuPropers *srccfp,  // 源波动特征，要求其中必须有数据。
            CurveFluctuPropers *dstcfp   // 目标波动特征。
    ); 

    // Host 静态方法：copytToCurrentDevice 方法（将 Host 中的数据拷贝到 
    // device 端）
    // 这是一个 Out-Place 形式的拷贝。会为 dstcfp 分配响应大小的空间，然后把
    // srccfp 中的数据拷贝到 dstcfp 中。这里只针对 CurveFluctuPropers 的数组字段
    // 进行拷贝，其他单值字段始终只做 Host 端的互相拷贝，并不涉及主机设备之间的
    // 传输
    static __host__ int                  // 返回值：函数是否正确执行，若函数正确
                                         // 执行，返回 NO_ERROR。
    copyToCurrentDevice(
            CurveFluctuPropers *srccfp,  // 源波动特征，要求其中必须有数据。
            CurveFluctuPropers *dstcfp   // 目标波动特征。
    ); 

    // Host 静态方法：deleteFromHost（销毁 Host 端曲线波动特征属性）
    // 在 Host 端销毁一个不再使用的曲线波动特征属性。
    static __host__ int             // 返回值：函数是否正确执行，若函数
                                    // 正确执行，返回 NO_ERROR。
    deleteFromHost(
        CurveFluctuPropers *srccfp  // 输入参数，不再需要使用的曲线波动特征。
    );

    // Host 静态方法：deleteFromCurrentDevice（销毁当前设备端曲线波动特征属性）
    // 在当前设备端销毁一个不再使用的曲线波动特征属性。
    static __host__ int             // 返回值：函数是否正确执行，若函数
                                    // 正确执行，返回 NO_ERROR。
    deleteFromCurrentDevice(
        CurveFluctuPropers *srccfp  // 输入参数，不再需要使用的曲线波动特征。
    );
};

#endif
 