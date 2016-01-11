// Matrix.h
//
// 矩阵定义（Matrix）
// 功能说明：定义了矩阵的数据结构和对矩阵的基本操作。


#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "ErrorCode.h"

// 结构体：Matrix（矩阵）
// 该结构体定义了矩阵的数据结构，其中包含了矩阵的数据和矩阵属性的描述。
typedef struct Matrix_st {
    size_t width;    // 矩阵的宽度（width >= 0）。
    size_t height;   // 矩阵的高度（height >= 0）。
    int roiX1;       // ROI 左上角点的横坐标，要求 0 <= roiX1 < width。ROI 意为
                     // 感兴趣区域（Region of Interest）
    int roiY1;       // ROI 左上角点的纵坐标，要求 0 <= roiY1 < height。
    int roiX2;       // ROI 右下角点的横坐标，要求 roiX1 <= roiX2 < width。
    int roiY2;       // ROI 右下角点的纵坐标，要求 roiY1 <= roiY2 < height。
    float *matData;  // 矩阵数据。由于该指针所指向的空间仅通过指针信息无法确
                     // 定，因此程序中不可直接读取该指针所指向的内存区域。需要
                     // 借助该矩阵对应的 MatrixCuda 型数据才可以以读取。
} Matrix;

// 结构体：MatrixCuda（矩阵的 CUDA 相关数据）
// 该结构体定义了与 CUDA 相关的矩阵数据。该结构体通常在算法内部使用，上层用户通
// 常见不到也不需要了解此结构体。
typedef struct MatrixCuda_st {
    Matrix matMeta;    // 矩阵数据，保存了对应的矩阵逻辑数据。
    int deviceId;      // 当前数据所处的内存，如果数据在 GPU 的内存上，则
                       // deviceId 为对应设备的编号；若 deviceId < 0，则说明数
                       // 据存储在 Host 内存上。
    size_t pitchWords; // Padding 后矩阵每行数据所占内存的字数（4 字节），要求
                       // pitchBytes >= width
} MatrixCuda;

// 宏：MATRIX_CUDA
// 给定 Matrix 型指针，返回对应的 MatrixCuda 型指针。该宏通常在算法内部使用，用
// 来获取关于 CUDA 的矩阵数据。
#define MATRIX_CUDA(mat)                                                      \
    ((MatrixCuda *)((unsigned char *)mat -                                    \
                    (unsigned long)(&(((MatrixCuda *)NULL)->matMeta))))


// 类：MatrixBasicOp（矩阵基本操作）
// 继承自：无
// 该类包含了对于矩阵的基本操作，如矩阵的创建与销毁、矩阵的读取、矩阵在各地址空
// 间之间的拷贝等。要求所有的矩阵实例，都需要通过该类创建，否则，将会导致系统运
// 行的紊乱（需要保证每一个 Matrix 型的数据都有对应的 MatrixCuda 数据）。
class MatrixBasicOp {

public:

    // Host 静态方法：newMatrix（创建矩阵）
    // 创建一个新的矩阵实例，并通过参数 outmat 返回。注意，所有系统中所使用的图
    // 像都需要使用该函数创建，否则，就会导致无法找到对应的 MatrixCuda 型数据而
    // 使系统执行紊乱。
    static __host__ int      // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    newMatrix(
            Matrix **outmat  // 返回的新创建的矩阵的指针。
    );

    // Host 静态方法：deleteMatrix（销毁矩阵）
    // 销毁一个不再被使用的矩阵实例。
    static __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    deleteMatrix(
            Matrix *inmat  // 输入参数，不再需要使用、需要被释放的矩阵。
    );

    // Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
    // 针对空矩阵，在当前 Device 内存中为其申请一段指定的大小的空间，这段空间中
    // 的数据是未被赋值的混乱数据。如果不是空矩阵，则该方法会报错。
    static __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    makeAtCurrentDevice(
            Matrix *mat,   // 矩阵。要求是空矩阵。
            size_t width,  // 指定的矩阵宽度
            size_t height  // 指定的矩阵高度
    );

    // Host 静态方法：makeAtHost（在 Host 内存中构建数据）
    // 针对空矩阵，在 Host 内存中为其申请一段指定的大小的空间，这段空间中的数据
    // 是未被赋值的混乱数据。如果不是空矩阵，则该方法会报错。
    static __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    makeAtHost(
            Matrix *mat,   // 矩阵。要求是空矩阵。
            size_t width,  // 指定的矩阵宽度
            size_t height  // 指定的矩阵高度
    );

    // Host 静态方法：readFromFile（从文件读取矩阵）
    // 从指定的文件路径中读取一个矩阵。该矩阵必须事先经过 newMatrix 创建。读取
    // 后矩阵默认放于 Host 内存中，但这通常不影响使用，因为矩阵处理算法通常会自
    // 动把矩阵传送到当前的 Device 上，这一过程并不需要上层程序员的干预。此外，
    // 仅支持 BMP 格式的灰度图像（即每像素占 8 bit 的 BMP 图像文件），读取出来
    // 的数据都会进行归一化操作，即，图像中的黑色为 0.0，白色为 1.0。此外，如果
    // 矩阵中原来有数据，若原有矩阵数据存储于 Host 上，且原有数据同新的矩阵的尺
    // 寸相同，则不重新申请空间，直接将矩阵放入原始数据的位置，不重新申请空间，
    // 且不对矩阵的 ROI 进行修改；若原有矩阵数据位于 Device 上，或原油矩阵数据
    // 通新的矩阵数据尺寸不一致，则重新在 Host 上申请矩阵数据的空间，释放原矩阵
    // 数据的空间，并根据新的尺寸调整 ROI：若 ROI 在新尺寸下合法，则不调整
    // ROI，若 ROI 的位置超出了新矩阵的范围，则调整 ROI 为最大可能的尺寸。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    readFromFile(
            const char *filepath,  // BMP 矩阵文件的路径。
            Matrix *outmat         // 输出参数，从文件读取矩阵后，将矩阵内容存
                                   // 放于其中。
    );

    // Host 静态方法：writeToFile（将矩阵写入文件）
    // 矩阵写入到指定的文件中。该矩阵中必须包含有数据，写入文件的部分仅包含 ROI
    // 部分的内容，而 ROI 以外的部分将不会被写入到文件中。进行该操作后，矩阵会
    // 自动的倍传送到 Host 上。矩阵文件的格式为 BMP 8-bit 灰度图像。0.0 为图像
    // 中的黑色，1.0 为图像中的白色；所有小于 0.0 矩阵元素在图像中都表现为黑
    // 色，所有大于 1.0 的矩阵元素在图像中都表现为白色。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    writeToFile(
            const char *filepath,  // BMP 矩阵文件的路径。
            Matrix *inmat          // 待写入文件的矩阵，写入文件后，矩阵将会自
                                   // 动的存放入 Host 内存中，因此不要频繁进行
                                   // 该操作，以免矩阵在 Host 和 Device 之间频
                                   // 繁传输而带来性能下降。
    );

    // Host 静态方法：copyToCurrentDevice（将矩阵拷贝到当前 Device 内存上）
    // 这是一个 In-Place 形式的拷贝。如果矩阵数据本来就在当前的 Device 上，则该
    // 函数不会进行任何操作，直接返回。如果矩阵数据不在当前 Device 上，则会将数
    // 据拷贝到当前 Device 上，并更新 matData 指针。原来的数据将会被释放。
    static __host__ int   // 返回值：函数是否正确执行，若函数正确执行，返回
                          // NO_ERROR。
    copyToCurrentDevice(
            Matrix *mat   // 需要将数据拷贝到当前 Device 的矩阵。
    );

    // Host 静态方法：copyToCurrentDevice（将矩阵拷贝到当前 Device 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srcmat 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstmat，且数据是存储于当前的 Device 上的。如果
    // dstmat 中原来存在有数据，且原来的数据同新的数据尺寸相同，也存放在当前
    // Device 上，则覆盖原内容，不重新分配空间；否则原数据将会被释放，并重新申
    // 请空间。
    static __host__ int      // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    copyToCurrentDevice(
            Matrix *srcmat,  // 源矩阵，要求矩阵中必须有数据。
            Matrix *dstmat   // 目标矩阵，要求该矩阵必须经过 newMatrix 申请。
    );

    // Host 静态方法：copyToHost（将矩阵拷贝到 Host 内存上）
    // 这是一个 In-Place 形式的拷贝。如果矩阵数据本来就在 Host 上，则该函数不会
    // 进行任何操作，直接返回。如果矩阵数据不在 Host 上，则会将数据拷贝到 Host
    // 上，并更新 matData 指针。原来的数据将会被释放。
    static __host__ int   // 返回值：函数是否正确执行，若函数正确执行，返回
                          // NO_ERROR。
    copyToHost(
            Matrix *mat   // 需要将数据拷贝到 Host 上的矩阵。
    );

    // Host 静态方法：copyToHost（将矩阵拷贝到 Host 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srcmat 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstmat，且数据是存储于 Host 上的。如果 dstmat
    // 中原来存在有数据，如果原来的数据同新的数据尺寸相同，且也存放在 Host 上，
    // 则覆盖原内容，但不重新分配空间；否则原数据将会被释放，并重新申请空间。
    static __host__ int      // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    copyToHost(
            Matrix *srcmat,  // 源矩阵，要求矩阵中必须有数据。
            Matrix *dstmat   // 目标矩阵，要求该矩阵必须经过 newMatrix 申请。
    );
    
    // 静态方法：roiSubMatrix（给出 ROI 指定的子矩阵）
    // 该函数不进行拷贝操作，只是根据输入矩阵的 ROI 信息给出对应的子矩阵，看上
    // 去好像是一个独立的矩阵。该函数主要是为了方便代码的编写工作，防止重复编码
    // 工作，上层用户实现更高层算法的时候不需要调用此函数。注意，由此函数得到的
    // MatrixCuda 型矩阵不需要调用 deleteMatrix 销毁。另一注意，由于 ROI 子矩阵
    // 中的 ROI 域显然是冗余数据，为了有效提高数据存储，子矩阵中的 ROI 域仍旧保
    // 留了原矩阵中的 ROI 域。这样的设计会造成一定的数据不一致性，因此，需要程
    // 序员在编写上层程序时注意区分子矩阵和原矩阵。
    static __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    roiSubMatrix(
            Matrix *inmat,           // 输入矩阵
            MatrixCuda *submatCud    // 输出的子矩阵
    ) {
        // 检查输入矩阵和存放输出子矩阵是否为 NULL，若为 NULL 则报错。
        if (inmat == NULL || submatCud == NULL)
            return NULL_POINTER;
        
        // 获取 inmat 对应的 MatrixCuda 型指针。
        MatrixCuda *inmatCud = MATRIX_CUDA(inmat);
        
        // 如果输入矩阵中不包含数据，则报错返回。
        if (inmat->matData == NULL || inmat->width == 0 ||
            inmat->height == 0 || inmatCud->pitchWords == 0)
            return INVALID_DATA;

        // 设置子矩阵的各种信息。
        // 子矩阵的长和宽分别为 ROI 的长和宽。
        submatCud->matMeta.width = inmat->roiX2 - inmat->roiX1;
        submatCud->matMeta.height = inmat->roiY2 - inmat->roiY1;
        // 子矩阵的 ROI 显然是一个冗余数据，因此我们利用它来记录子矩阵在原图中
        // 的位置。这样，子矩阵的 ROI 域保持同原矩阵相同。这种方式要求编程人员
        // 注意区分矩阵实例和这种临时的子矩阵，不要用错了。
        submatCud->matMeta.roiX1 = inmat->roiX1;
        submatCud->matMeta.roiY1 = inmat->roiY1;
        submatCud->matMeta.roiX2 = inmat->roiX2;
        submatCud->matMeta.roiY2 = inmat->roiY2;
        // 子矩阵的数据起始地址，为 ROI 区域的起始地址。显然，编程人员需要注
        // 意，这中直接弄到地址的方式使得子矩阵不能够使用 deleteMatrix 释放，否
        // 则将引起不可预知的问题。编程人员在使用完子矩阵时，直接弃用即可，不需
        // 要任何特殊的处理。
        submatCud->matMeta.matData = inmat->matData +
                                     inmat->roiY1 * inmatCud->pitchWords +
                                     inmat->roiX1;
        // 子矩阵的 Device ID 和 Pitch 保持和输入矩阵相同。
        submatCud->deviceId = inmatCud->deviceId;
        submatCud->pitchWords = inmatCud->pitchWords;
        
        // 处理完毕，退出。
        return NO_ERROR;
    }
};

#endif
