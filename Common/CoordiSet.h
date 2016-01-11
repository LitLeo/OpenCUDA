// CoordiSet.h
// 
// 坐标集定义（CoordiSet）
// 功能说明：定义了坐标集的数据接口，以及和坐标集相关的基本操作。实际上，坐标集
//           就是模板，所有的操作也是借用模板的操作。

#ifndef __COORDISET_H__
#define __COORDISET_H__

#include "Template.h"

// 结构体：CoordiSet（坐标集）
// 该结构体定义了坐标集的数据结构，这一数据结构直接借用 Template 的数据结构。其
// 中，坐标集的锚点用 (0,0) 点表示。
typedef Template CoordiSet;
//typedef struct CoordiSet_st {
//    size_t count;             // 坐标集中点的数量。
//    int *tplData/*cstData*/;  // 坐标集中个点的信息，其中相邻的两个下标的数组
//                              // 数据表示一个点。即，坐标集中第 i 个点对应的
//                              // 数组下标为 2 * i 和 2 * i + 1，其中下标为 2 
//                              // * i 的数据表示该点的横坐标，下标为 2 * i + 1 
//                              // 的数据表示该点的纵坐标。这些坐标是参照锚点的
//                              // 坐标，因此，坐标可以含有负值。
//} CoordiSet;

// 结构体：CoordiSet（坐标集的 CUDA 相关数据）
// 该结构体定义了与 CUDA 相关的坐标集数据。该结构体通常在算法内部使用，上层用户
// 通常见不到也不需要了解此结构体。这一结构体也直接借用了 Template 的数据结构。
typedef TemplateCuda CoordiSetCuda;
//typedef struct CoordiSetCuda_st {
//    CoordiSet tplMeta/*cstMeta*/;  // 坐标集数据，保存了对应的坐标集逻辑数据。
//    float *attachedData;           // 坐标点附带的数据，这些数据结合 tplData 
//                                   // 可形成稀疏矩阵。
//    int deviceId;                  // 当前数据所处的内存。如果数据在 GPU 的内
//                                   // 存上，则 deviceId 为对应设备的编号；若 
//                                   // deviceId < 0，则说明数据存储在 Host 内
//                                   // 存上。
//} CoordiSetCuda;

// 宏：COORDISET_CUDA
// 给定 CoordiSet 型指针，返回对应的 CoordiSetCuda 型指针。该宏通常在算法内部使
// 用，用来获取关于 CUDA 的坐标集数据。该宏也是借用 Template 的定义。
#define COORDISET_CUDA(cst) TEMPLATE_CUDA(cst)
//#define COORDISET_CUDA(tpl)                                                 \
//    ((CoordiSetCuda *)((unsigned char *)(tpl) -                             \
//                       (unsigned long)(&(((CoordiSetCuda *)NULL)->          \
//                                         tplMeta/*cstMeta*/))))

// 宏：ATTACHED_DATA
// 给定 CoordiSet 型指针，返回其中的 attachedData 指针，用于访问附加数据。注意
// 这里没有修改 attachedData 的存储位置，因此用户使用前需要调用相应的同步函数以
// 确保数据的可访问性。
// 该宏已在 Template 中定义过了，因此可以直接使用之。
//#define ATTACHED_DATA(cst)  (COORDISET_CUDA(cst)->attachedData)

// 类：CoordiSetBasicOp（坐标集基本操作）
// 继承自：无
// 该类包含了对于坐标集的基本操作，如坐标集的创建与销毁、坐标集的读取、坐标集在
// 各地址空间之间的拷贝等。要求所有的坐标集实例，都需要通过该类创建，否则，将会
// 导致系统运行的紊乱（需要保证每一个 CoordiSet 型的数据都有对应的
// CoordiSetCuda 数据）。本类中所有的函数都直接调用 TemplateBasicOp 中的方法。
class CoordiSetBasicOp {

public:

    // Host 静态方法：newCoordiSet（创建坐标集）
    // 创建一个新的坐标集实例，并通过参数 outcst 返回。注意，所有系统中所使用的
    // 坐标集都需要使用该函数创建，否则，就会导致无法找到对应的 TemplateCuda 型
    // 数据而使系统执行紊乱。
    static inline __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
    newCoordiSet(
            CoordiSet **outcst  // 返回的新创建的坐标集指针。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::newTemplate(outcst);
    }

    // Host 静态方法：deleteCoordiSet（销毁坐标集）
    // 销毁一个不再被使用的坐标集实例。
    static inline __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
    deleteCoordiSet(
            CoordiSet *incst    // 输入参数，不再需要使用、需要被释放的坐标集。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::deleteTemplate(incst);
    }

    // Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
    // 针对空坐标集，在当前 Device 内存中为其申请一段指定的大小的空间，这段空间
    // 中的数据是未被赋值的混乱数据。如果不是空坐标集，则该方法会报错。
    static inline __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
    makeAtCurrentDevice(
            CoordiSet *cst,     // 坐标集，要求是空坐标集。
            size_t count        // 指定的坐标集中点的数量。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::makeAtCurrentDevice(cst, count);
    }

    // Host 静态方法：makeAtHost（在 Host 内存中构建数据）
    // 针对空坐标集，在 Host 内存中为其申请一段指定的大小的空间，这段空间中的数
    // 据是未被赋值的混乱数据。如果不是空坐标集，则该方法会报错。
    static inline __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
    makeAtHost(
            CoordiSet *cst,     // 坐标集，要求是空坐标集。
            size_t count        // 指定的坐标集中点的数量。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::makeAtHost(cst, count);
    }

    // Host 静态方法：readFromFile（从文件读取坐标集）
    // 从文件中读取坐标集数据，由于目前国际上没有关于坐标集数据文件的标准，因
    // 此，我们设计了自己的坐标集文件数据。该文件数据包含 3 个区段。第一个区段
    // 为 4 字节，表示文件类型，统一为 TPLT；第二个区段为 4 个字节，为一个无符
    // 号整型数据，表示坐标集中点的个数；其后的数据表示坐标集中各点的坐标，存储
    // 的格式同内存中存储的格式是相同的。不过上层用户不必过于关心文件的格式，因
    // 为通常这一文件格式对用户透明。
    static inline __host__ int     // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    readFromFile(
            const char *filepath,  // 坐标集数据文件的路径
            CoordiSet *outcst      // 输出参数，从稳健读取坐标集后，将坐标集的
                                   // 内容存放于其中。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::readFromFile(filepath, outcst);
    }

    // Host 静态方法：writeToFile（将坐标集写入文件）
    // 将坐标集中的数据写入到文件中。由于目前国际上没有关于坐标集数据文件的标
    // 准，因此，我们设计了自己的坐标集文件数据。该文件数据包含 3 个区段。第一
    // 个区段为 4 字节，表示文件类型，统一为 TPLT；第二个区段为 4 个字节，为一
    // 个无符号整型数据，表示坐标集中点的个数；其后的数据表示坐标集中各点的坐
    // 标，存储的格式同内存中存储的格式是相同的。不过上层用户不必过于关心文件的
    // 格式，因为通常这一文件格式对用户透明。此外，本函数要求输入的坐标集必须有
    // 数据。经过本函数，坐标集的数据会自动的拷贝到 Host 内存中，因此，不要频繁
    // 的调用本函数，这样会使数据频繁的在 Host 与 Device 间拷贝，造成性能下降。
    static inline __host__ int     // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    writeToFile(
            const char *filepath,  // 坐标集数据文件的路径
            CoordiSet *incst       // 待写入文件的坐标集，写入文件后，坐标集将
                                   // 会自动的存放入 Host 内存中，因此不要频繁
                                   // 进行该操作，以免坐标集在 Host 和 Device 
                                   // 之间频繁传输而带来性能下降。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::writeToFile(filepath, incst);
    }

    // Host 静态方法：copyToCurrentDevice（将坐标集拷贝到当前 Device 内存上）
    // 这是一个 In-Place 形式的拷贝。如果坐标集数据本来就在当前的 Device 上，则
    // 该函数不会进行任何操作，直接返回。如果坐标集数据不在当前 Device 上，则会
    // 将数据拷贝到当前 Device 上，并更新 tplData 指针。原来的数据将会被释放。
    static inline __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
    copyToCurrentDevice(
            CoordiSet *cst      // 需要将数据拷贝到当前 Device 的坐标集。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::copyToCurrentDevice(cst);
    }

    // Host 静态方法：copyToCurrentDevice（将坐标集拷贝到当前 Device 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srccst 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstcst，且数据是存储于当前的 Device 上的。如果
    // dstcst 中原来存在有数据，且原来的数据同新的数据尺寸相同，也存放在当前
    // Device 上，则覆盖原内容，不重新分配空间；否则原数据将会被释放，并重新申
    // 请空间。
    static inline __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
    copyToCurrentDevice(
            CoordiSet *srccst,  // 源坐标集，要求坐标集中必须有数据。
            CoordiSet *dstcst   // 目标坐标集，要求该坐标集必须经过
                                // newCoordiSet 申请。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::copyToCurrentDevice(srccst, dstcst);
    }

    // Host 静态方法：copyToHost（将坐标集拷贝到 Host 内存上）
    // 这是一个 In-Place 形式的拷贝。如果坐标集数据本来就在 Host 上，则该函数不
    // 会进行任何操作，直接返回。如果坐标集数据不在 Host 上，则会将数据拷贝到 
    // Host 上，并更新 tplData 指针。原来的数据将会被释放。
    static inline __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
    copyToHost(
            CoordiSet *cst      // 需要将数据拷贝到 Host 上的坐标集。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::copyToHost(cst);
    }

    // Host 静态方法：copyToHost（将坐标集拷贝到 Host 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srccst 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstcst，且数据是存储于 Host 上的。如果 dstcst
    // 中原来存在有数据，如果原来的数据同新的数据尺寸相同，且也存放在 Host 上，
    // 则覆盖原内容，但不重新分配空间；否则原数据将会被释放，并重新申请空间。
    static inline __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返
                                // 回 NO_ERROR。
    copyToHost(
            CoordiSet *srccst,  // 源坐标集，要求坐标集中必须有数据。
            CoordiSet *dstcst   // 目标图像，要求该坐标集必须经过 newCoordiSet 
                                // 申请。
    ) {
        // 直接调用 TemplateBasicOp 中的方法。
        return TemplateBasicOp::copyToHost(srccst, dstcst);
    }
};

#endif
