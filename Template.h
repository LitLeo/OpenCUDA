// Template.h
// 创建者：于玉龙
// 
// 模板定义（Template）
// 功能说明：定义了模板的数据接口，以及和模板相关的基本操作。
//
// 修订历史：
// 2012年07月21日（于玉龙）
//     初始版本。
// 2012年07月23日（于玉龙）
//     实现了 newTemplate 和 deleteTemplate 方法。
//     实现了 makeAtHost 和 makeAtCurrentDevice 方法。
//     实现了 readFromFile 和 writeToFile 方法。
//     修改了部分注释中的错别字，以及函数参数命名。
// 2012年07月24日（于玉龙）
//     实现了 copyToCurrentDevice 和 copyToHost 的 In-place 版本方法。
// 2012年07月25日（于玉龙）
//     实现了 copyToCurrentDevice 和 copyToHost 的 Out-place 版本方法。
// 2012年09月02日（于玉龙、罗劼）
//     修正了模板内存空间分配过程中的一处严重 Bug。
// 2012年09月08日（于玉龙）
//     优化了 copyToCurrentDevice 和 copyToHost 的健壮性。
// 2012年09月12日（于玉龙）
//     为 Template 增加了附属数据，可以在更加丰富的算法中应用模版。
//     修正了部分注释中的表达错误。
// 2012年09月27日（于玉龙）
//     修正了部分注释中的表达错误。
// 2012年10月11日（于玉龙）
//     增加了 Template 数据访问 attachedData 的宏。
// 2012年11月10日（于玉龙、杨晓光）
//     修正了模版读取和写入中一处潜在的 Bug。

#ifndef __TEMPLATE_H__
#define __TEMPLATE_H__


// 结构体：Template（模板）
// 该结构体定了模板的数据结构，其中包含了和模板相关的逻辑参数，如模板中点的数量
// 和模板中的各个点的坐标。注意，模板的锚点（Anchor）为坐标 (0, 0) 点。
typedef struct Template_st {
    size_t count;  // 模板中点的数量。
    int *tplData;  // 模板中个点的信息，其中相邻的两个下标的数组数据表示一个
                   // 点。即，模板中第 i 个点对应的数组下标为 2 * i 和 2 * i +
                   // 1，其中下标为 2 * i 的数据表示该点的横坐标，下标为 2 * i
                   // + 1 的数据表示该点的纵坐标。这些坐标是参照锚点的坐标，因
                   // 此，坐标可以含有负值。
} Template;

// 结构体：TemplateCuda（模板的 CUDA 相关数据）
// 该结构体定义了与 CUDA 相关的模板数据。该结构体通常在算法内部使用，上层用户通
// 常见不到也不需要了解此结构体
typedef struct TemplateCuda_st {
    Template tplMeta;     // 模板数据，保存了对应的模板逻辑数据。
    float *attachedData;  // 坐标点附带的数据，这些数据结合 tplData 可形成稀疏
                          // 矩阵。
    int deviceId;         // 当前数据所处的内存。如果数据在 GPU 的内存上，则
                          // deviceId 为对应设备的编号；若 deviceId < 0，则说明
                          // 数据存储在 Host 内存上。
} TemplateCuda;

// 宏：TEMPLATE_CUDA
// 给定 Template 型指针，返回对应的 TemplateCuda 型指针。该宏通常在算法内部使用
// ，用来获取关于 CUDA 的模板数据。
#define TEMPLATE_CUDA(tpl)                                                    \
    ((TemplateCuda *)((unsigned char *)(tpl) -                                \
                      (unsigned long)(&(((TemplateCuda *)NULL)->tplMeta))))

// 宏：ATTACHED_DATA
// 给定 Template 型指针，返回其中的 attachedData 指针，用于访问附加数据。注意这
// 里没有修改 attachedData 的存储位置，因此用户使用前需要调用相应的同步函数以确
// 保数据的可访问性。
#define ATTACHED_DATA(tpl)  (TEMPLATE_CUDA(tpl)->attachedData)

// 类：TemplateBasicOp（模板基本操作）
// 继承自：无
// 该类包含了对于模板的基本操作，如模板的创建与销毁、模板的读取、模板在各地址空
// 间之间的拷贝等。要求所有的模板实例，都需要通过该类创建，否则，将会导致系统运
// 行的紊乱（需要保证每一个 Template 型的数据都有对应的 TemplateCuda 数据）。
class TemplateBasicOp {

public:

    // Host 静态方法：newTemplate（创建模板）
    // 创建一个新的模板实例，并通过参数 outtpl 返回。注意，所有系统中所使用的模
    // 板都需要使用该函数创建，否则，就会导致无法找到对应的 TemplateCuda 型数据
    // 而使系统执行紊乱。
    static __host__ int        // 返回值：函数是否正确执行，若函数正确执行，返
                               // 回 NO_ERROR。
    newTemplate(
            Template **outtpl  // 返回的新创建的模板指针。
    );

    // Host 静态方法：deleteTemplate（销毁模板）
    // 销毁一个不再被使用的模板实例。
    static __host__ int      // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    deleteTemplate(
            Template *intpl  // 输入参数，不再需要使用、需要被释放的模板。
    );

    // Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
    // 针对空模板，在当前 Device 内存中为其申请一段指定的大小的空间，这段空间中
    // 的数据是未被赋值的混乱数据。如果不是空模板，则该方法会报错。
    static __host__ int     // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR。
    makeAtCurrentDevice(
            Template *tpl,  // 模板，要求是空模板。
            size_t count    // 指定的模板中点的数量。
    );

    // Host 静态方法：makeAtHost（在 Host 内存中构建数据）
    // 针对空模板，在 Host 内存中为其申请一段指定的大小的空间，这段空间中的数据
    // 是未被赋值的混乱数据。如果不是空模板，则该方法会报错。
    static __host__ int     // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR。
    makeAtHost(
            Template *tpl,  // 模板，要求是空模板。
            size_t count    // 指定的模板中点的数量。
    );

    // Host 静态方法：readFromFile（从文件读取模板）
    // 从文件中读取模板数据，由于目前国际上没有关于模板数据文件的标准，因此，我
    // 们设计了自己的模板文件数据。该文件数据包含 4 个区段。第一个区段为 4 字
    // 节，表示文件类型，统一为 TPLT；第二个区段为 4 个字节，为一个无符号整型数
    // 据，表示模板中点的个数；第三个区段为 20 个字节，为保留位，无数据，其他存
    // 储数据结构可调用此函数实现文件的读操作；其后的数据表示模板中各点的坐标，
    // 存储的格式同内存中存储的格式是相同的。不过上层用户不必过于关心文件的格
    // 式，因为通常这一文件格式对用户透明。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    readFromFile(
            const char *filepath,  // 模板数据文件的路径
            Template *outtpl       // 输出参数，从稳健读取模板后，将模板的内容
                                   // 存放于其中
    );

    // Host 静态方法：writeToFile（将模板写入文件）
    // 将模板中的数据写入到文件中。由于目前国际上没有关于模板数据文件的标准，因
    // 此，我们设计了自己的模板文件数据。该文件数据包含 4 个区段。第一个区段为
    // 4 字节，表示文件类型，统一为 TPLT；第二个区段为 4 个字节，为一个无符号整
    // 型数据，表示模板中点的个数；第三个区段为 20 个字节，为保留位，无数据，
    // 其他存储数据结构可调用此函数实现文件的写操作；其后的数据表示模板中各点的
    // 坐标，存储的格式同内存中存储的格式是相同的。不过上层用户不必过于关心文件
    // 的格式，因为通常这一文件格式对用户透明。此外，本函数要求输入的模板必须有
    // 数据。经过本函数，模板的数据会自动的拷贝到 Host 内存中，因此，不要频繁的
    // 调用本函数，这样会使数据频繁的在 Host 与 Device 间拷贝，造成性能下降。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    writeToFile(
            const char *filepath,  // 模板数据文件的路径
            Template *intpl        // 待写入文件的模板，写入文件后，模板将会自
                                   // 动的存放入 Host 内存中，因此不要频繁进行
                                   // 该操作，以免模板在 Host 和 Device 之间频
                                   // 繁传输而带来性能下降。
    );

    // Host 静态方法：copyToCurrentDevice（将模板拷贝到当前 Device 内存上）
    // 这是一个 In-Place 形式的拷贝。如果模板数据本来就在当前的 Device 上，则该
    // 函数不会进行任何操作，直接返回。如果模板数据不在当前 Device 上，则会将数
    // 据拷贝到当前 Device 上，并更新 tplData 指针。原来的数据将会被释放。
    static __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    copyToCurrentDevice(
            Template *tpl  // 需要将数据拷贝到当前 Device 的模板。
    );

    // Host 静态方法：copyToCurrentDevice（将模板拷贝到当前 Device 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srctpl 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dsttpl，且数据是存储于当前的 Device 上的。如果
    // dsttpl 中原来存在有数据，且原来的数据同新的数据尺寸相同，也存放在当前
    // Device 上，则覆盖原内容，不重新分配空间；否则原数据将会被释放，并重新申
    // 请空间。
    static __host__ int        // 返回值：函数是否正确执行，若函数正确执行，返
                               // 回 NO_ERROR。
    copyToCurrentDevice(
            Template *srctpl,  // 源模板，要求模板中必须有数据。
            Template *dsttpl   // 目标模板，要求该模板板必须经过 newTemplate 申
                               // 请。
    );

    // Host 静态方法：copyToHost（将模板拷贝到 Host 内存上）
    // 这是一个 In-Place 形式的拷贝。如果模板数据本来就在 Host 上，则该函数不会
    // 进行任何操作，直接返回。如果模板数据不在 Host 上，则会将数据拷贝到 Host
    // 上，并更新 tplData 指针。原来的数据将会被释放。
    static __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    copyToHost(
            Template *tpl  // 需要将数据拷贝到 Host 上的模板。
    );

    // Host 静态方法：copyToHost（将模板拷贝到 Host 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srctpl 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dsttpl，且数据是存储于 Host 上的。如果 dsttpl
    // 中原来存在有数据，如果原来的数据同新的数据尺寸相同，且也存放在 Host 上，
    // 则覆盖原内容，但不重新分配空间；否则原数据将会被释放，并重新申请空间。
    static __host__ int        // 返回值：函数是否正确执行，若函数正确执行，返
                               // 回 NO_ERROR。
    copyToHost(
            Template *srctpl,  // 源模板，要求模板中必须有数据。
            Template *dsttpl   // 目标图像，要求该模板必须经过 newTemplate 申
                               // 请。
    );    
};

#endif

