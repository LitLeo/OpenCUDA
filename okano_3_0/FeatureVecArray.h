// FeatureVecArray.h
// 创建人：邱孝兵
//
// 特征向量组（FeatureVecArray）
// 功能说明：定义了 SmoothVector 操作中特征向量的数据接口，
// 以及和特征向量相关的基本操作。
//
// 修订历史：
// 2012年10月30日（邱孝兵）
//     初始版本。
// 2012年11月23日（邱孝兵）
//     将数据结构名称由 FeaturedVectors 修改为 FeatureVecArray
// 2012年12月23日（邱孝兵）
//     增加为 FeatureVecArrayBasicOp 增加 copyToHost 方法

#ifndef __FEATUREVECARRAY_H__
#define __FEATUREVECARRAY_H__



// 结构体：FeatureVecArray（特征向量组）
// 该结构体定义了特征向量组的数据结构，组成这个向量组的每个向量都是五维的，
// 这五维分别是 X 坐标、 Y 坐标， CV，SD，NC，此外还有向量组的大小 count
typedef struct FeatureVecArray_st {
    size_t count;         // 特征向量组的大小
    int *x, *y;           // x，y 坐标的数组
    float *CV, *SD, *NC;  // 三个特征值的数组
} FeatureVecArray;

// 类：FeatureVecArrayBasicOp（特征向量组基本操作）
// 继承自：无
// 该类包含了对于特征向量组的基本操作，如向量组的创建与销毁，向量组在各地址
// 空间之间的拷贝等。要求所有的特征向量组实例，都需要通过该类创建，否则将会
// 导致系统运行的紊乱。
class FeatureVecArrayBasicOp {

public:

    // Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
    // 针对空向量组，在当前 Device 内存中为其申请一段指定的大小的空间，这段空间
    // 中的数据是未被赋值的混乱数据。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确执行，
                                   // 返回 NO_ERROR
    makeAtCurrentDevice(
            FeatureVecArray *fvp,  // 特征向量组，要求为空
            size_t count           // 指定的模板中点的数量
    );

    // Host 静态方法：copytToHost 方法（将当前 Device 内存中数据拷贝到 Host 端）
    // 这是一个 Out-Place 形式的拷贝。无论 srcfvp 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstfvp，且数据是存储于 Host 上的。如果 dstfvp
    // 中原来存在有数据，如果原来的数据同新的数据尺寸相同，且也存放在 Host 上，
    // 则覆盖原内容，但不重新分配空间；否则原数据将会被释放，并重新申请空间。
    static __host__ int               // 返回值：函数是否正确执行，若函数正确
                                      // 执行，返回 NO_ERROR。
    copyToHost(
            FeatureVecArray *srcfvp,  // 源特征向量组，要求其中必须有数据。
            FeatureVecArray *dstfvp   // 目标特征向量组。
    );    

    // Host 静态方法：deleteFeatureVecArray（销毁模板）
    // 销毁一个不再被使用的特征向量组。
    static __host__ int           // 返回值：函数是否正确执行，若函数正确执行，
                                  // 返回 NO_ERROR。
    deleteFeatureVecArray(
            FeatureVecArray *fvp  // 输入参数，不再需要使用、需要被释放的
                                  // 特征向量组。
    );
};

#endif
