// Image.h
// 创建人：于玉龙
//
// 图像定义（Image）
// 功能说明：定义了图像的数据结构和对图像的基本操作。
//
// 修订历史：
// 2012年07月14日（于玉龙）
//     初始版本。
// 2012年07月22日（于玉龙）
//     增加了 make 函数族，包括了 makeAtCurrentDevice 和 makeAtHost 两个函数。
//     修正了一些注释中的错误和错别字。
// 2012年07月23日（于玉龙）
//     优化了部分代码的表现形式。
// 2012年07月24日（于玉龙）
//     优化了 In-place 版本的 copyToCurrerntDevice 和 copyToHost 函数的代码。
// 2012年07月25日（于玉龙）
//     优化了 Out-place 版本的 copyToCurrerntDevice 和 copyToHost 函数的代码。
// 2012年08月18日（于玉龙）
//     增加了根据 ROI 提取子图像的方法，该方法用于 Kernel 函数的封装函数中，方
//     便代码的编写工作。
// 2012年08月20日（于玉龙）
//     补充了提取 ROI 子图像的注意事项的注释说明。
// 2012年09月08日（于玉龙）
//     优化了 copyToCurrerntDevice 和 copyToHost 函数的健壮性。
// 2012年11月10日（杨晓光、于玉龙）
//     修正了图像读取和存储方法中一个潜在的错误。修改了一处代码格式错误。
// 2012年11月19日（于玉龙）
//     修改了一处由于代码格式导致的潜在错误。
// 2012年11月20日（于玉龙）
//     修改了代码中内存分配过程中存在的潜在错误。
// 2012年11月22日（于玉龙）
//     修正了代码中关于内存拷贝的一处潜在错误。
// 2012年11月23日（于玉龙）
//     修改了文件保存中的一处潜在的错误。
// 2014年09月26日（高新凯、于玉龙）
//     增加了对 Host 零拷贝映射内存的支持。
// 2014年09月27日（高新凯、于玉龙）
//     修改了对 Host 零拷贝映射内存操作的潜在错误。
//     修改了两处 Bug，一处为读取文件时判断内存空间可重用性；另一处为判断数据
//     所在设备编号。

#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "ErrorCode.h"

// 结构体：Image（图像）
// 该结构体定义了图像的数据结构，其中包含了图像的数据和图像属性的描述。
typedef struct Image_st {
    size_t width;            // 图像的宽度（width >= 0）。
    size_t height;           // 图像的高度（height >= 0）。
    int roiX1;               // ROI 左上角点的横坐标，要求 0 <= roiX1 < width。
                             // ROI 意为感兴趣区域（Region of Interest）
    int roiY1;               // ROI 左上角点的纵坐标，要求 0 <= roiY1 < 
                             // height。
    int roiX2;               // ROI 右下角点的横坐标，要求 roiX1 <= roiX2 <
                             // width。
    int roiY2;               // ROI 右下角点的纵坐标，要求 roiY1 <= roiY2 <
                             // height。
    unsigned char *imgData;  // 图像数据。由于该指针所指向的空间仅通过指针信息
                             // 无法确定，因此程序中不可直接读取该指针所指向的
                             // 内存区域。需要借助该图像对应的 ImageCuda 型数
                             // 据才可以以读取。
} Image;

// 结构体：ImageCuda（图像的 CUDA 相关数据）
// 该结构体定义了与 CUDA 相关的图像数据。该结构体通常在算法内部使用，上层用户通
// 常见不到也不需要了解此结构体。
typedef struct ImageCuda_st {
    Image imgMeta;             // 图像数据，保存了对应的图像逻辑数据。
    int deviceId;              // 当前数据所处的内存，若数据在 GPU 的内存上，则
                               // deviceId 为对应设备的编号；若 deviceId < 0，
                               // 则说明数据存储在 Host 内存上。
    unsigned char *d_imgData;
    unsigned char *mapSource;  // 内存映射源指针，如果指针不为空，则说明已经
                               // 使用了内内存映射。
    size_t pitchBytes;         // Padding 后图像每行数据所占内存的字节数，要求 
                               // pitchBytes >= width * sizeof (unsigned char)
} ImageCuda;

// 宏：IMAGE_CUDA
// 给定 Image 型指针，返回对应的 ImageCuda 型指针。该宏通常在算法内部使用，用来
// 获取关于 CUDA 的图像数据。
#define IMAGE_CUDA(img)                                                      \
    ((ImageCuda *)((unsigned char *)img -                                    \
                   (unsigned long)(&(((ImageCuda *)NULL)->imgMeta))))


// 类：ImageBasicOp（图像基本操作）
// 继承自：无
// 该类包含了对于图像的基本操作，如图像的创建与销毁、图像的读取、图像在各地址空
// 间之间的拷贝等。要求所有的图像实例，都需要通过该类创建，否则，将会导致系统运
// 行的紊乱（需要保证每一个 Image 型的数据都有对应的 ImageCuda 数据）。
class ImageBasicOp {

public:

    // Host 静态方法：newImage（创建图像）
    // 创建一个新的图像实例，并通过参数 outimg 返回。注意，所有系统中所使用的图
    // 像都需要使用该函数创建，否则，就会导致无法找到对应的 ImageCuda 型数据而
    // 使系统执行紊乱。
    static __host__ int     // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR。
    newImage(
            Image **outimg  // 返回的新创建的图像的指针。
    );
    
    // Host 静态方法：deleteImage（销毁图像）
    // 销毁一个不再被使用的图像实例。
    static __host__ int   // 返回值：函数是否正确执行，若函数正确执行，返回
                          // NO_ERROR。
    deleteImage(
            Image *inimg  // 输入参数，不再需要使用、需要被释放的图像。
    );

    // Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
    // 针对空图像，在当前 Device 内存中为其申请一段指定的大小的空间，这段空间中
    // 的数据是未被赋值的混乱数据。如果不是空图像，则该方法会报错。
    static __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    makeAtCurrentDevice(
            Image *img,    // 图像。要求是空图像。
            size_t width,  // 指定的图像宽度
            size_t height  // 指定的图像高度
    );

    // Host 静态方法：makeAtHost（在 Host 内存中构建数据）
    // 针对空图像，在 Host 内存中为其申请一段指定的大小的空间，这段空间中的数据
    // 是未被赋值的混乱数据。如果不是空图像，则该方法会报错。
    static __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    makeAtHost(
            Image *img,    // 图像。要求是空图像。
            size_t width,  // 指定的图像宽度
            size_t height  // 指定的图像高度
    );

    // Host 静态方法：readFromFile（从文件读取图像）
    // 从指定的文件路径中读取一个图像。该图像必须事先经过 newImage 创建。读取后
    // 图像默认放于 Host 内存中，但这通常不影响使用，因为图像处理算法通常会自动
    // 把图像传送到当前的 Device 上，这一过程并不需要上层程序员的干预。此外，仅
    // 支持 BMP 格式的灰度图像（即每像素占 8 bit 的 BMP 图像文件）。此外，如果
    // 图像中原来有数据，若原有图像数据存储于 Host 上，且原有数据同新的图像的尺
    // 寸相同，则不重新申请空间，直接将图像放入原始数据的位置，不重新申请空间，
    // 且不对图像的 ROI 进行修改；若原有图像数据位于 Device 上，或原油图像数据
    // 通新的图像数据尺寸不一致，则重新在 Host 上申请图像数据的空间，释放原图像
    // 数据的空间，并根据新的尺寸调整 ROI：若 ROI 在新尺寸下合法，则不调整
    // ROI，若 ROI 的位置超出了新图像的范围，则调整 ROI 为最大可能的尺寸。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    readFromFile(
            const char *filepath,  // BMP 图像文件的路径。
            Image *outimg          // 输出参数，从文件读取图像后，将图像内容存
                                   // 放于其中。
    );

    // Host 静态方法：writeToFile（将图像写入文件）
    // 图像写入到指定的文件中。该图像中必须包含有数据，写入文件的部分仅包含 ROI
    // 部分的内容，而 ROI 以外的部分将不会被写入到文件中。进行该操作后，图像会
    // 自动的倍传送到 Host 上。图像文件的格式为 BMP 8-bit 灰度图像。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR。
    writeToFile(
            const char *filepath,  // BMP 图像文件的路径。
            Image *inimg           // 待写入文件的图像，写入文件后，图像将会自
                                   // 动的存放入 Host 内存中，因此不要频繁进行
                                   // 该操作，以免图像在 Host 和 Device 之间频
                                   // 繁传输而带来性能下降。
    );

    // Host 静态方法：copyToCurrentDevice（将图像拷贝到当前 Device 内存上）
    // 这是一个 In-Place 形式的拷贝。如果图像数据本来就在当前的 Device 上，则该
    // 函数不会进行任何操作，直接返回。如果图像数据不在当前 Device 上，则会将数
    // 据拷贝到当前 Device 上，并更新 imgData 指针。原来的数据将会被释放。
    static __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                         // NO_ERROR。
    copyToCurrentDevice(
            Image *img   // 需要将数据拷贝到当前 Device 的图像。
    );
    
    // Host 静态方法：copyToCurrentDevice（将图像拷贝到当前 Device 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srcimg 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstimg，且数据是存储于当前的 Device 上的。如果
    // dstimg 中原来存在有数据，且原来的数据同新的数据尺寸相同，也存放在当前
    // Device 上，则覆盖原内容，不重新分配空间；否则原数据将会被释放，并重新申
    // 请空间。
    static __host__ int     // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR。
    copyToCurrentDevice(
            Image *srcimg,  // 源图像，要求图像中必须有数据。
            Image *dstimg   // 目标图像，要求该图像必须经过 newImage 申请。
    );

    // Host 静态方法：copyToHost（将图像拷贝到 Host 内存上）
    // 这是一个 In-Place 形式的拷贝。如果图像数据本来就在 Host 上，则该函数不会
    // 进行任何操作，直接返回。如果图像数据不在 Host 上，则会将数据拷贝到 Host
    // 上，并更新 imgData 指针。原来的数据将会被释放。如果原始图像为内存映射，
    // 则此处会做解除映射操作，使其恢复为普通 Host 图像。
    static __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                         // NO_ERROR。
    copyToHost(
            Image *img   // 需要将数据拷贝到 Host 上的图像。
    );

    // Host 静态方法：copyToHost（将图像拷贝到 Host 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srcimg 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstimg，且数据是存储于 Host 上的。如果 dstimg
    // 中原来存在有数据，如果原来的数据同新的数据尺寸相同，且也存放在 Host 上，
    // 则覆盖原内容，但不重新分配空间；否则原数据将会被释放，并重新申请空间。
    static __host__ int     // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR。
    copyToHost(
            Image *srcimg,  // 源图像，要求图像中必须有数据。
            Image *dstimg   // 目标图像，要求该图像必须经过 newImage 申请。
    );
    
        
    // Host 静态方法：mapToCurrentDevice（将图像映射到当前 Device 内存上）
    // 将已经分配在 pinned memory 中的图像映射到当前 Device 上，也就是使用 
    // zero-copy 功能，使 Device 可以在不进行内存拷贝的情况下使用位于 Host 内存
    // 上的图像数据。应该注意，此时虽然图像所在位置被标记为当前 Device ，但实际
    // 数据仍然存储于 Host 端。
    static __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                         // NO_ERROR。
    mapToCurrentDevice(
            Image *img   // 需要将数据映射到当前 Device 的原始图像。
    );
    
    // Host 静态方法：unmapToHost（解除图像映射关系）
    // 解除图像的内存映射关系，一般与 mapToCurrentDevice 成对使用。对于已经映射
    // 到设备端的图像，将其恢复为普通 Host 端图像。
    static __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                         // NO_ERROR。
    unmapToHost(
            Image *img   // 需要解除内存映射关系的图像。
    );
    
    // 静态方法：roiSubImage（给出 ROI 指定的子图像）
    // 该函数不进行拷贝操作，只是根据输入图像的 ROI 信息给出对应的子图像，看上
    // 去好像是一个独立的图像。该函数主要是为了方便代码的编写工作，防止重复编码
    // 工作，上层用户实现更高层算法的时候不需要调用此函数。注意，由此函数得到的
    // ImageCuda 型图像不需要调用 deleteImage 销毁。另一注意，由于 ROI 子图像中
    // 的 ROI 域显然是冗余数据，为了有效提高数据存储，子图像中的 ROI 域仍旧保留
    // 了原图像中的 ROI 域。这样的设计会造成一定的数据不一致性，因此，需要程序
    // 员在编写上层程序时注意区分子图像和原图像。
    static __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                                    // 行，返回 NO_ERROR。
    roiSubImage(
            Image *inimg,           // 输入图像
            ImageCuda *subimgCud    // 输出的子图像
    ) {
        // 检查输入图像和存放输出子图像是否为 NULL，若为 NULL 则报错。
        if (inimg == NULL || subimgCud == NULL)
            return NULL_POINTER;

        // 获取 inimg 对应的 ImageCuda 型指针。
        ImageCuda *inimgCud = IMAGE_CUDA(inimg);

        // 如果输入图像中不包含数据，则报错返回。
        if (inimg->imgData == NULL || inimg->width == 0 || 
            inimg->height == 0 || inimgCud->pitchBytes == 0)
            return INVALID_DATA;

        // 设置子图像的各种信息。
        // 子图像的长和宽分别为 ROI 的长和宽。
        subimgCud->imgMeta.width = inimg->roiX2 - inimg->roiX1;
        subimgCud->imgMeta.height = inimg->roiY2 - inimg->roiY1;
        // 子图像的 ROI 显然是一个冗余数据，因此我们利用它来记录子图像在原图中
        // 的位置。这样，子图像的 ROI 域保持同原图像相同。这种方式要求编程人员
        // 注意区分图像实例和这种临时的子图像，不要用错了。
        subimgCud->imgMeta.roiX1 = inimg->roiX1;
        subimgCud->imgMeta.roiY1 = inimg->roiY1;
        subimgCud->imgMeta.roiX2 = inimg->roiX2;
        subimgCud->imgMeta.roiY2 = inimg->roiY2;
        // 子图像的数据起始地址，为 ROI 区域的起始地址。显然，编程人员需要注
        // 意，这中直接弄到地址的方式使得子图像不能够使用 deleteImage 释放，否
        // 则将引起不可预知的问题。编程人员在使用完子图像时，直接弃用即可，不
        // 需要任何特殊的处理。
        subimgCud->imgMeta.imgData = inimg->imgData +
                                     inimg->roiY1 * inimgCud->pitchBytes +
                                     inimg->roiX1;
        // 子图像的 Device ID 和 Pitch 保持和输入图像相同。
        subimgCud->deviceId = inimgCud->deviceId;
        subimgCud->pitchBytes = inimgCud->pitchBytes;
        subimgCud->mapSource = inimgCud->mapSource;
        // 处理完毕，退出。
        return NO_ERROR;
    }
};

#endif

