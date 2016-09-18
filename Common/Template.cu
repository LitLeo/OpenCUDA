// Template.cu
// 模板数据结构的定义和模板的基本操作。

#include "Template.h"

#include <iostream>
#include <fstream>
using namespace std;

#include "ErrorCode.h"


// Host 静态方法：newTemplate（创建模板）
__host__ int TemplateBasicOp::newTemplate(Template **outtpl)
{
    // 检查用于盛放新模板的指针是否为 NULL。
    if (outtpl == NULL)
        return NULL_POINTER;

    // 申请一个新的 TemplateCuda 型数据，本方法最后会将其中的 tplMeta 域返回给
    // outtpl，这样 outtpl 就有了一个对应的 TemplateCuda 型伴随数据。
    TemplateCuda *tplCud = new TemplateCuda;

    // 初始化各种元数据。
    tplCud->tplMeta.count = 0;
    tplCud->tplMeta.tplData = NULL;
    tplCud->attachedData = NULL;
    tplCud->deviceId = -1;

    // 将 TemplateCuda 型数据中的 tplMeta 赋值给输出参数。
    *outtpl = &(tplCud->tplMeta);

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 静态方法：deleteTemplate（销毁模板）
__host__ int TemplateBasicOp::deleteTemplate(Template *intpl)
{
    // 检查模板的指针是否为 NULL。
    if (intpl == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Template 指针，得到对应的 TemplateCuda 型数据。
    TemplateCuda *intplCud = TEMPLATE_CUDA(intpl);

    // 检查模板所在的地址空间是否合法，如果模板所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (intplCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 释放模板数据，即坐标数据。
    if (intpl->tplData == NULL || intpl->count == 0) {
        // 如果输入模板是空的，则不进行模板数据释放操作（因为本来也没有数据可被
        // 释放）。
        // Do Nothing;
    } if (intplCud->deviceId < 0) {
        // 对于数据存储于 Host 内存，直接利用 delete 关键字释放图像数据。
        delete[] intpl->tplData;
        delete[] intplCud->attachedData;
    } else {
        // 对于数据存储于 Device 内存中，则需要首先切换设备，将该设备作为当前
        // Device 设备，然后释放之，最后还需要将设备切换回来以保证后续处理的正
        // 确性。
        cudaSetDevice(intplCud->deviceId);
        cudaFree(intpl->tplData);
        cudaFree(intplCud->attachedData);
        cudaSetDevice(curdevid);
    }

    // 最后还需要释放模板的元数据
    delete intplCud;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
__host__ int TemplateBasicOp::makeAtCurrentDevice(Template *tpl, size_t count)
{
    // 检查输入模板是否为 NULL
    if (tpl == NULL)
        return NULL_POINTER;

    // 检查给定的模板中坐标点数量
    if (count < 1)
        return INVALID_DATA;

    // 检查模板是否为空模板
    if (tpl->tplData != NULL)
        return UNMATCH_IMG;

    // 获取 tpl 对应的 TemplateCuda 型数据。
    TemplateCuda *tplCud = TEMPLATE_CUDA(tpl);

    // 在当前的 Device 上申请存储指定坐标数量的模板所需要的内存空间。
    cudaError_t cuerrcode;
    cuerrcode = cudaMalloc((void **)(&tpl->tplData), 2 * count * sizeof (int));
    if (cuerrcode != cudaSuccess) {
        tpl->tplData = NULL;
        return CUDA_ERROR;
    }

    // 为附属数据在当前的 Device 上申请内存空间。
    cuerrcode = cudaMalloc((void **)(&tplCud->attachedData),
                           count * sizeof (float));
    if (cuerrcode != cudaSuccess) {
        // 如果附属数据空间申请失败，需要释放掉先前申请的坐标数据的内存空间。
        cudaFree(tpl->tplData);
        tpl->tplData = NULL;
        tplCud->attachedData = NULL;
        return CUDA_ERROR;
    }


    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 修改模板的元数据。
    tpl->count = count;
    tplCud->deviceId = curdevid;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 静态方法：makeAtHost（在 Host 内存中构建数据）
__host__ int TemplateBasicOp::makeAtHost(Template *tpl, size_t count)
{
    // 检查输入模板是否为 NULL
    if (tpl == NULL)
        return NULL_POINTER;

    // 检查给定的模板中坐标点数量
    if (count < 1)
        return INVALID_DATA;

    // 检查模板是否为空模板
    if (tpl->tplData != NULL)
        return UNMATCH_IMG;

    // 获取 tpl 对应的 TemplateCuda 型数据。
    TemplateCuda *tplCud = TEMPLATE_CUDA(tpl);

    // 为图像数据在 Host 内存中申请空间
    tpl->tplData = new int[count * 2];
    if (tpl->tplData == NULL)
        return OUT_OF_MEM;

    // 为附属数据在 Host 内存中申请空间。
    tplCud->attachedData = new float[count];
    if (tplCud->attachedData == NULL) {
        delete[] tpl->tplData;
        tpl->tplData = NULL;
        return OUT_OF_MEM;
    }

    // 设置模板中的元数据
    tpl->count = count;
    tplCud->deviceId = -1;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：readFromFile（从文件读取模板）
__host__ int TemplateBasicOp::readFromFile(const char *filepath,
                                           Template *outtpl)
{
    // 这段代码仅支持 int 型尺寸为 2、4、8 三种情况。目前绝大部分的系统，采用了
    // sizeof (int) == 4 的情况，少数早期的 DOS 和 Windows 系统中 sizeof (int)
    // == 2。
    if (sizeof (int) != 2 && sizeof (int) != 4 && sizeof (int) != 8)
        return UNIMPLEMENT;

    // 检查文件路径和模板是否为 NULL。
    if (filepath == NULL || outtpl == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Template 型指针，得到对应的 TemplateCuda 型数据。
    TemplateCuda *outtplCud = TEMPLATE_CUDA(outtpl);

    // 检查模板所在的地址空间是否合法，如果模板所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (outtplCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 打开模板文件。
    ifstream tplfile(filepath, ios::in | ios::binary);
    if (!tplfile)
        return NO_FILE;

    // 将文件读指针挪到文件的开头处。该步骤虽然显得多余，但是却可以确保操作的正
    // 确。
    tplfile.seekg(0, ios::beg);

    // 读取文件的前四个字节，这是文件的类型头，如果类型头为 TPLT，则说明该文件
    // 是模板文件。
    char typestr[5] = { '\0' };
    tplfile.read(typestr, 4);
    if (strcmp(typestr, "TPLT") != 0)
        return WRONG_FILE;

    // 从文件中获取模板中包含的坐标点的数量。如果坐标点数量小于 1，则报错。
    size_t count = 0;
    tplfile.read(reinterpret_cast<char *>(&count), 4);
    if (count < 1)
        return WRONG_FILE;

    // 读取并丢弃掉 20 个字节的保留位。
    char disdata[21] = { '\0' };
    tplfile.read(disdata, 20);

    // 为在内存中保存模板的坐标点而申请新的数据空间。为了避免频繁的数据申请与释
    // 放，如果发现原来模板中的坐标点数量和新的数据中坐标点数量相同，且原来的数
    // 据存储于 Host 内存，则会重用这段内存空间，不去重新申请内存。
    int *newdata;
    float *newattach;
    bool reusedata;
    if (outtpl->tplData != NULL && outtpl->count == count &&
        outtplCud->deviceId == -1) {
        // 若数据可以重用，则使用原来的内存空间。
        newdata = outtpl->tplData;
        newattach = outtplCud->attachedData;
        reusedata = true;
    } else {
        // 若数据不能重用，则重新申请合适的内存空间。
        newdata = new int[count * 2];
        newattach = new float[count];
        reusedata = false;
        if (newdata == NULL || newattach == NULL) {
            delete[] newdata;
            delete[] newattach;
            return OUT_OF_MEM;
        }
    }

    // 读取坐标点数据。因为文件中存储的坐标点采用了 32 位有符号整形数，这里需要
    // 根据系统中 int 型数据的尺寸采取不同的转换策略。
    if (sizeof (int) == 2) {
        // 对于 sizeof (int) == 2 的系统通常 long 型数据为 32 位，因此需要逐个
        // 读取后转成 int 型存放到数据数组中。
        long tmp;
        for (int i = 0; i < count * 2; i++) {
            tplfile.read(reinterpret_cast<char *>(&tmp), 4);
            newdata[i] = (int)tmp;
        }
    } else if (sizeof (int) == 8) {
        // 对于 sizeof (int) == 8 的系统通常 short 型数据为 32 位，因此需要逐个
        // 读取后转成 int 型存放到数据数组中。
        short tmp;
        for (int i = 0; i < count * 2; i++) {
            tplfile.read(reinterpret_cast<char *>(&tmp), 4);
            newdata[i] = (int)tmp;
        }
    } else {
        // 对于 sizeof (int) == 4 的系统，不需要进行任何的转换，读取后的数据可
        // 读取存放到数据数组中。
        tplfile.read(reinterpret_cast<char *>(newdata), count * 2 * 4);
    }

    // 根据 IEEE 的规定 float 型数据为 4 字节，因此这里就采用了直接读取，而没有
    // 像处理坐标点数据那样做了很多的数据尺寸判断。
    tplfile.read(reinterpret_cast<char *>(newattach), count * 4);

    // 当数据已经成功的读取后，释放原来数据占用的内存空间，防止内存泄漏。
    if (outtpl->tplData != NULL && !reusedata) {
        if (outtplCud->deviceId == -1) {
            // 如果原来的数据存放在 Host 内存中，则直接通过 delete 关键字释放。
            delete[] outtpl->tplData;
            delete[] outtplCud->attachedData;
        } else {
            // 如果原来的数据存放在 Device 内存中，则切换到相应的 Device 后，使
            // 用 cudaFree 释放。
            cudaSetDevice(outtplCud->deviceId);
            cudaFree(outtpl->tplData);
            cudaFree(outtplCud->attachedData);
            cudaSetDevice(curdevid);
        }
    }

    // 使用新的数据更新模板的元数据。
    outtpl->count = count;
    outtpl->tplData = newdata;
    outtplCud->attachedData = newattach;
    outtplCud->deviceId = -1;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：writeToFile（将模板写入文件）
__host__ int TemplateBasicOp::writeToFile(const char *filepath,
                                          Template *intpl)
{
    // 这段代码仅支持 int 型尺寸为 2、4、8 三种情况。目前绝大部分的系统，采用了
    // sizeof (int) == 4 的情况，少数早期的 DOS 和 Windows 系统中 sizeof (int)
    // == 2。
    if (sizeof (int) != 2 && sizeof (int) != 4 && sizeof (int) != 8)
        return UNIMPLEMENT;

    // 检查文件路径和模板是否为 NULL。
    if (filepath == NULL || intpl == NULL)
        return NULL_POINTER;

    // 打开需要写入的文件。
    ofstream tplfile(filepath, ios::out | ios::binary);
    if (!tplfile)
        return NO_FILE;

    // 根据输入参数的 Template 型指针，得到对应的 TemplateCuda 型数据。
    TemplateCuda *intplCud = TEMPLATE_CUDA(intpl);

    // 将模板的数据拷贝回 Host 内存中，这样模板就可以被下面的代码所读取，然后将
    // 模板的数据写入到磁盘中。这里需要注意的是，安排模板的拷贝过程在文件打开之
    // 后是因为，如果一旦文件打开失败，则不会改变模板在内存中的存储状态，这可能
    // 会对后续处理更加有利。
    int errcode;
    errcode = TemplateBasicOp::copyToHost(intpl);
    if (errcode < 0)
        return errcode;

    // 向文件中写入文件类型字符串
    static char typestr[] = "TPLT";
    tplfile.write(typestr, 4);

    // 向文件中写入模板含有的坐标点数量。
    tplfile.write(reinterpret_cast<char *>(&intpl->count), 4);

    // 向文件中写入 20 个字节的保留位
    static char reserved[20] = { '\0' };
    tplfile.write(reserved, 20);

    // 向文件中写入坐标数据，因为考虑到。为了保证每个整型数据占用 4 个字节，这
    // 里对不同的情况进行了处理。不过针对目前绝大部分系统来说，sizeof (int) ==
    // 4，因此绝大部分情况下，编译器会选择 else 分支。如果委托方认为系统是运行
    // 在 sizeof (int) == 4 的系统之上，也可以删除前面的两个分支，直接使用最后
    // 的 else 分支。
    if (sizeof (int) == 2) {
        // 对于 sizeof (int) == 2 的系统来说，long 通常是 32 位的，因此，需要逐
        // 个的将数据转换成 32 位的 long 型，然后进行处理。
        long tmp;
        for (int i = 0; i < intpl->count * 2; i++) {
            tmp = (long)(intpl->tplData[i]);
            tplfile.write(reinterpret_cast<char *>(&tmp), 4);
        }
    } else if (sizeof (int) == 8) {
        // 对于 sizeof (int) == 8 的系统来说，short 通常是 32 位的，因此，需要
        // 逐个的将数据转换成 32 位的 short 型，然后进行处理。
        short tmp;
        for (int i = 0; i < intpl->count * 2; i++) {
            tmp = (short)(intpl->tplData[i]);
            tplfile.write(reinterpret_cast<char *>(&tmp), 4);
        }
    } else {
        // 如果 sizeof (int) == 4，则可以直接将数据写入磁盘，而不需要任何的转换
        // 过程。
        tplfile.write(reinterpret_cast<char *>(intpl->tplData),
                      intpl->count * 2 * 4);
    }

    // 根据 IEEE 的规定，float 型数据通常采用 4 字节的形式，因此这里没有做数据
    // 长度的判断，而是直接使用了 4 字节存储数据到磁盘。
    tplfile.write(reinterpret_cast<char *>(intplCud->attachedData),
                  intpl->count * 4);

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：copyToCurrentDevice（将模板拷贝到当前 Device 内存上）
__host__ int TemplateBasicOp::copyToCurrentDevice(Template *tpl)
{
    // 检查模板是否为 NULL。
    if (tpl == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Template 型指针，得到对应的 TemplateCuda 型数据。
    TemplateCuda *tplCud = TEMPLATE_CUDA(tpl);

    // 检查模板所在的地址空间是否合法，如果模板所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (tplCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果模板是一个不包含数据的空模板，则报错。
    if (tpl->tplData == NULL || tpl->count == 0)
        return UNMATCH_IMG;

    // 对于不同的情况，将模板数据拷贝到当前设备上。
    if (tplCud->deviceId < 0) {
        // 如果模板的数据位于 Host 内存上，则需要在当前 Device 的内存空间上申请
        // 空间，然后将 Host 内存上的数据拷贝到当前 Device 上。
        int *devptr;            // 新的坐标数据空间，在当前 Device 上。
        float *attachptr;       // 新的附属数据空间，在当前 Device 上。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 在当前设备上申请坐标数据的空间。
        cuerrcode = cudaMalloc((void **)(&devptr), 
                               tpl->count * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 当前设备商申请附属数据的空间。
        cuerrcode = cudaMalloc((void **)(&attachptr),
                               tpl->count * sizeof (float));
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }


        // 将原来存储在 Host 上坐标数据拷贝到当前 Device 上。
        cuerrcode = cudaMemcpy(devptr, tpl->tplData, 
                               tpl->count * 2 * sizeof (int),
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            cudaFree(attachptr);
            return CUDA_ERROR;
        }

        // 将原来存储在 Host 上附属数据拷贝到当前 Device 上。
        cuerrcode = cudaMemcpy(attachptr, tplCud->attachedData,
                               tpl->count * sizeof (float),
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            cudaFree(attachptr);
            return CUDA_ERROR;
        }

        // 释放掉原来存储于 Host 内存上的数据。
        delete[] tpl->tplData;
        delete[] tplCud->attachedData;

        // 更新模版数据，把新的在当前 Device 上申请的数据和相关数据写入模版元数
        // 据中。
        tpl->tplData = devptr;
        tplCud->attachedData = attachptr;
        tplCud->deviceId = curdevid;

        // 操作完毕，返回。
        return NO_ERROR;

    } else if (tplCud->deviceId != curdevid) {
        // 对于数据存在其他 Device 的情况，仍旧要在当前 Device 上申请数据空间，
        // 并从另一个 Device 上拷贝数据到新申请的当前 Device 的数据空间中。
        int *devptr;            // 新申请的当前 Device 上的坐标数据。
        float *attachptr;       // 新申请的当前 Device 上的附属数据。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 在当前 Device 上申请坐标数据空间。
        cuerrcode = cudaMalloc((void **)(&devptr), 
                               tpl->count * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 在当前 Device 上申请附属数据空间。
        cuerrcode = cudaMalloc((void **)(&attachptr),
                               tpl->count * sizeof (float));
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }

        // 将数据从模板原来的存储位置拷贝到当前的 Device 上。
        cuerrcode = cudaMemcpyPeer(devptr, curdevid,
                                   tpl->tplData, tplCud->deviceId,
                                   tpl->count * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            cudaFree(attachptr);
            return CUDA_ERROR;
        }

        // 将附属数据从模板原来的存储位置拷贝到当前的 Device 上。
        cuerrcode = cudaMemcpyPeer(attachptr, curdevid,
                                   tplCud->attachedData, tplCud->deviceId,
                                   tpl->count * sizeof (float));
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            cudaFree(attachptr);
            return CUDA_ERROR;
        }

        // 释放掉模板在原来的 Device 上的数据。
        cudaFree(tpl->tplData);
        cudaFree(tplCud->attachedData);

        // 将新的图像数据信息写入到图像元数据中。
        tpl->tplData = devptr;
        tplCud->attachedData = attachptr;
        tplCud->deviceId = curdevid;

        // 操作完成，返回。
        return NO_ERROR;
    }

    // 对于其他情况，即模板数据本来就在当前 Device 上，则直接返回，不进行任何的
    // 操作。
    return NO_ERROR;
}

// Host 静态方法：copyToCurrentDevice（将模板拷贝到当前 Device 内存上）
__host__ int TemplateBasicOp::copyToCurrentDevice(
        Template *srctpl, Template *dsttpl)
{
    // 检查输入模板是否为 NULL。
    if (srctpl == NULL || dsttpl == NULL)
        return NULL_POINTER;

    // 如果输出模板为 NULL 或者和输入模板为同一个模板，则转而调用对应的 
    // In-place 版本的函数。
    if (dsttpl == NULL || dsttpl == srctpl)
        return copyToCurrentDevice(srctpl);

    // 获取 srctpl 和 dsttpl 对应的 TemplateCuda 型指针。
    TemplateCuda *srctplCud = TEMPLATE_CUDA(srctpl);
    TemplateCuda *dsttplCud = TEMPLATE_CUDA(dsttpl);

    // 用来存放旧的 dsttpl 数据，使得在拷贝操作失败时可以恢复为原来的可用的数据
    // 信息，防止系统进入一个混乱的状态。
    TemplateCuda olddsttplCud = *dsttplCud;  // 旧的 dsttpl 数据
    bool reusedata = true;                // 记录是否重用了原来的模板数据空间。
                                          // 该值为 ture，则原来的数据空间被重
                                          // 用，不需要在之后释放数据，否则需要
                                          // 在最后释放旧的空间。

    // 如果源模板是一个空模板，则不进行任何操作，直接报错。
    if (srctpl->tplData == NULL || srctpl->count == 0)
        return INVALID_DATA;

    // 检查模板所在的地址空间是否合法，如果模板所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srctplCud->deviceId >= devcnt || dsttplCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果目标模板中存在有数据，则需要根据情况，若原来的数据不存储在当前的
    // Device 上，或者即使存储在当前的 Device 上，但数据尺寸不匹配，则需要释放
    // 掉原来申请的空间，以便重新申请合适的内存空间。此处不进行真正的释放操作，
    // 其目的在于当后续操作出现错误时，可以很快的恢复 dsttpl 中原来的信息，使得
    // 整个系统不会处于一个混乱的状态，本函数会在最后，确定 dsttpl 被成功的更换
    // 为了新的数据以后，才会真正的将原来的模板数据释放掉。
    if (dsttplCud->deviceId != curdevid) {
        // 对于数据存在 Host 与其他的 Device 上，则直接释放掉原来的数据空间。
        reusedata = false;
        dsttpl->tplData = NULL;
        dsttplCud->attachedData = NULL;
    } else if (dsttpl->count != srctpl->count) {
        // 对于数据存在于当前 Device 上，则需要检查数据的尺寸是否和源图像相匹
        // 配。如果目标模板和源模板的尺寸不匹配则仍旧需要释放目标图像原来的数据
        // 空间。
        reusedata = false;
        dsttpl->tplData = NULL;
        dsttplCud->attachedData = NULL;
    }

    // 将目标模板的尺寸更改为源模板的尺寸。
    dsttpl->count = srctpl->count;

    // 更改目标模板的数据存储位置为当前 Device。
    dsttplCud->deviceId = curdevid;

    // 如果目标模板需要重新申请空间（因为上一步将无法重用原来内存空间的情况的
    // dsttpl->tplData 都置为 NULL，因此此处通过检查 dsttpl->tplData == NULL来
    // 确定是否需要重新申请空间），则在当前的 Device 内存中申请空间。
    cudaError_t cuerrcode;
    if (dsttpl->tplData == NULL) {
        // 申请坐标数据的内存空间
        cuerrcode = cudaMalloc((void **)(&dsttpl->tplData),
                               srctpl->count * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess) {
            // 如果空间申请操作失败，则恢复原来的目标模板的数据，以防止系统进入
            // 混乱状态。
            *dsttplCud = olddsttplCud;
            return CUDA_ERROR;
        }

        // 申请附属数据的内存空间
        cuerrcode = cudaMalloc((void **)(&dsttplCud->attachedData),
                               srctpl->count * sizeof (float));
        if (cuerrcode != cudaSuccess) {
            // 如果空间申请操作失败，则恢复原来的目标模板的数据，以防止系统进入
            // 混乱状态。
            cudaFree(dsttpl->tplData);
            *dsttplCud = olddsttplCud;
            return CUDA_ERROR;
        }

    }

    // 将数据拷贝如目标模板中。
    if (srctplCud->deviceId < 0) {
        // 如果源模板存储于 Host，则通过 cudaMemcpy 将数据从 Host 拷贝到 Device
        // 上。
        // 拷贝坐标数据
        cuerrcode = cudaMemcpy(dsttpl->tplData, srctpl->tplData,
                               srctpl->count * 2 * sizeof (int),
                               cudaMemcpyHostToDevice);

        // 拷贝附属数据
        if (cuerrcode == cudaSuccess) {
            cuerrcode = cudaMemcpy(dsttplCud->attachedData,
                                   srctplCud->attachedData,
                                   srctpl->count * sizeof (float),
                                   cudaMemcpyHostToDevice);
        }
    } else {
        // 如果源模板存储于 Device，则通过 cudaMemcpyPeer 进行设备间的数据拷
        // 贝。
        // 拷贝坐标数据
        cuerrcode = cudaMemcpyPeer(dsttpl->tplData, curdevid,
                                   srctpl->tplData, srctplCud->deviceId,
                                   srctpl->count * 2 * sizeof (int));

        // 拷贝附属数据
        if (cuerrcode == cudaSuccess) {
            cuerrcode = cudaMemcpyPeer(dsttplCud->attachedData, curdevid,
                                       srctplCud->attachedData,
                                       srctplCud->deviceId,
                                       srctpl->count * sizeof (float));
        }
    }

    // 如果上述的数据拷贝过程失败，进入这个 if 分支进行报错处理。
    if (cuerrcode != cudaSuccess) {
        // 报错处理分为两个步骤：第一步，如果数据空间不是重用原来的数据空间时，
        // 则需要释放掉新申请的数据空间；第二步，恢复原来的目标模板的元数据。
        if (!reusedata) {
            cudaFree(dsttpl->tplData);
            cudaFree(dsttplCud->attachedData);
        }
        *dsttplCud = olddsttplCud;
        return CUDA_ERROR;
    }

    // 到此步骤已经说明新的模板数据空间已经成功的申请并拷贝了新的数据，因此，旧
    // 的数据空间已毫无用处。本步骤就是释放掉旧的数据空间以防止内存泄漏。这里，
    // 作为拷贝的 olddsttplCud 是局部变量，因此相应的元数据会在本函数退出后自动
    // 释放，不用理会。
    if (olddsttplCud.tplMeta.tplData != NULL) {
        if (olddsttplCud.deviceId < 0) {
            // 如果旧数据空间是 Host 内存上的，则需要无条件释放。
            delete[] olddsttplCud.tplMeta.tplData;
            delete[] olddsttplCud.attachedData;
        } else if (!reusedata) {
            // 如果旧数据空间不是当前 Device 内存上的其他 Device 内存上的数据，
            // 则也需要无条件的释放。
            cudaSetDevice(olddsttplCud.deviceId);
            cudaFree(olddsttplCud.tplMeta.tplData);
            cudaFree(olddsttplCud.attachedData);
            cudaSetDevice(curdevid);
        }
    }

    return NO_ERROR;
}

// Host 静态方法：copyToHost（将模板拷贝到 Host 内存上）
__host__ int TemplateBasicOp::copyToHost(Template *tpl)
{
    // 检查模板是否为 NULL。
    if (tpl == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Template 型指针，得到对应的 TemplateCuda 型数据。
    TemplateCuda *tplCud = TEMPLATE_CUDA(tpl);

    // 检查模板所在的地址空间是否合法，如果模板所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (tplCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果模板是一个不好含数据的空模板，则报错。
    if (tpl->tplData == NULL || tpl->count == 0)
        return UNMATCH_IMG;

    // 对于不同的情况，将模板数据拷贝到当前设备上。
    if (tplCud->deviceId < 0) {
        // 如果模板位于 Host 内存上，则不需要进行任何操作。
        return NO_ERROR;

    } else {
        // 如果模板的数据位于 Device 内存上，则需要在 Host 的内存空间上申请空
        // 间，然后将数据拷贝到 Host 上。
        int *hostptr;            // 新的数据空间，在 Host 上。
        float *attachptr;        // 新的附属数据空间，在 Host 上。
        cudaError_t cuerrcode;   // CUDA 调用返回的错误码。

        // 在 Host 上申请坐标数据空间。
        hostptr = new int[tpl->count * 2];
        if (hostptr == NULL)
            return OUT_OF_MEM;

        // 在 Host 上申请附属数据空间。
        attachptr = new float[tpl->count];
        if (attachptr == NULL) {
            delete[] hostptr;
            return OUT_OF_MEM;
        }

        // 将设备切换到数据所在的 Device 上。
        cudaSetDevice(tplCud->deviceId);

        // 拷贝坐标数据
        cuerrcode = cudaMemcpy(hostptr, tpl->tplData, 
                               tpl->count * 2 * sizeof (int),
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
            // 后报错返回。
            delete[] hostptr;
            delete[] attachptr;
            return CUDA_ERROR;
        }

        // 拷贝附属数据
        cuerrcode = cudaMemcpy(attachptr, tplCud->attachedData,
                               tpl->count * sizeof (float),
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
            // 后报错返回。
            delete[] hostptr;
            delete[] attachptr;
            return CUDA_ERROR;
        }

        // 释放掉原来存储于 Device 内存上的模板数据。
        cudaFree(tpl->tplData);
        cudaFree(tplCud->attachedData);

        // 对 Device 内存的操作完毕，将设备切换回当前 Device。
        cudaSetDevice(curdevid);

        // 更新模板数据，把新的在当前 Device 上申请的数据和相关数据写入模板元数
        // 据中。
        tpl->tplData = hostptr;
        tplCud->attachedData = attachptr;
        tplCud->deviceId = -1;

        // 操作完毕，返回。
        return NO_ERROR;
    }

    // 程序永远也不会到达这个分支，因此如果到达这个分支，则说明系统紊乱。对于多
    // 数编译器来说，会对此句报出不可达语句的 Warning，因此这里将其注释掉，以防
    // 止不必要的 Warning。
    //return UNKNOW_ERROR;
}

// Host 静态方法：copyToHost（将模板拷贝到 Host 内存上）
__host__ int TemplateBasicOp::copyToHost(
        Template *srctpl, Template *dsttpl)
{
    // 检查输入模板是否为 NULL。
    if (srctpl == NULL || dsttpl == NULL)
        return NULL_POINTER;

    // 如果输出模板为 NULL 或者和输入模板同为一个模板，则调用对应的 In-place 版
    // 本的函数。
    if (dsttpl == NULL || dsttpl == srctpl)
        return copyToHost(srctpl);

    // 获取 srctpl 和 dsttpl 对应的 TemplateCuda 型指针。
    TemplateCuda *srctplCud = TEMPLATE_CUDA(srctpl);
    TemplateCuda *dsttplCud = TEMPLATE_CUDA(dsttpl);

    // 用来存放旧的 dsttpl 数据，使得在拷贝操作失败时可以恢复为原来的可用的数据
    // 信息，防止系统进入一个混乱的状态。
    TemplateCuda olddsttplCud = *dsttplCud;  // 旧的 dsttpl 数据
    bool reusedata = true;                // 记录是否重用了原来的图像数据空间。
                                          // 该值为 true，则原来的数据空间被重
                                          // 用。不需要在之后释放数据，否则需要
                                          // 释放就的空间。

    // 如果源模板是一个空模板，则不进行任何操作，直接报错。
    if (srctpl->tplData == NULL || srctpl->count == 0)
        return INVALID_DATA;

    // 检查模板所在的地址空间是否合法，如果模板所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srctplCud->deviceId >= devcnt || dsttplCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果目标模板中存在有数据，则需要根据情况，若原来的数据不存储在 Host 上，
    // 或者即使存储在 Host 上，但数据尺寸不匹配，则需要释放掉原来申请的空间，以
    // 便重新申请合适的内存空间。此处不进行真正的释放操作，其目的在于当后续操作
    // 出现错误时，可以很快的恢复 dsttpl 中原来的信息，使得整个系统不会处于一个
    // 混乱的状态，本函数会在最后，确定 dsttpl 被成功的更换为了新的数据以后，才
    // 会真正的将原来的模板数据释放掉。
    if (dsttplCud->deviceId >= 0) {
        // 对于数据存在于 Device 上，则亦直接释放掉原来的数据空间。
        reusedata = false;
        dsttpl->tplData = NULL;
        dsttplCud->attachedData = NULL;
    } else if (srctpl->count != dsttpl->count) {
        // 对于数据存在于 Host 上，则需要检查数据的尺寸是否和源模板相匹配。检查
        // 的标准：源模板和目标模板的尺寸相同时，可重用原来的空间。
        reusedata = false;
        dsttpl->tplData = NULL;
        dsttplCud->attachedData = NULL;
    }

    // 将目标模板的尺寸修改为源模板的尺寸。
    dsttpl->count = srctpl->count;

    // 更改目标模板的数据存储位置为 Host。
    dsttplCud->deviceId = -1;

    // 如果目标模板的 tplData == NULL，说明目标模板原本要么是一个空图像，要么目
    // 标模板原本的数据空间不合适，需要重新申请。这时，需要为目标模板重新在 
    // Host 上申请一个合适的数据空间。
    if (dsttpl->tplData == NULL) {
        // 申请坐标数据
        dsttpl->tplData = new int[srctpl->count * 2];
        if (dsttpl->tplData == NULL) {
            // 如果申请内存的操作失败，则再报错返回前需要将旧的目标模板数据
            // 恢复到目标模板中，以保证系统接下的操作不至于混乱。
            *dsttplCud = olddsttplCud;
            return OUT_OF_MEM;
        }

        // 申请附属数据
        dsttplCud->attachedData = new float[srctpl->count];
        if (dsttplCud->attachedData == NULL) {
            // 如果申请内存的操作失败，则再报错返回前需要将旧的目标模板数据
            // 恢复到目标模板中，以保证系统接下的操作不至于混乱。
            delete[] dsttpl->tplData;
            *dsttplCud = olddsttplCud;
            return OUT_OF_MEM;
        }
    }

    // 将坐标数据从源模板中拷贝到目标模板中。
    if (srctplCud->deviceId < 0) {
        // 如果源模板数据存储于 Host 内存，则直接使用 C 标准支持库中的 memcpy
        // 完成拷贝。

        // 将 srctpl 内的坐标数据拷贝到 dsttpl 中。memcpy 不返回错误，因此，没
        // 有进行错误检查。
        memcpy(dsttpl->tplData, srctpl->tplData,
               srctpl->count * 2 * sizeof (int));
        memcpy(dsttplCud->attachedData, srctplCud->attachedData,
               srctpl->count * sizeof (float));

    } else {
        // 如果源模板数据存储于 Device 内存（无论是当前 Device 还是其他的 
        // Device），都是通过 CUDA 提供的函数进行拷贝。。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 首先切换到 srctpl 坐标数据所在的 Device，以方便进行内存操作。
        cudaSetDevice(srctplCud->deviceId);

        // 这里使用 cudaMemcpy 将 srctpl 中处于 Device 上的数据拷贝到 dsttpl 中
        // 位于 Host 的内存空间上面。
        // 拷贝坐标数据
        cuerrcode = cudaMemcpy(dsttpl->tplData, srctpl->tplData, 
                               srctpl->count * 2 * sizeof (int),
                               cudaMemcpyDeviceToHost);
        // 拷贝附属数据
        if (cuerrcode == cudaSuccess) {
            cuerrcode = cudaMemcpy(dsttplCud->attachedData,
                                   srctplCud->attachedData,
                                   srctpl->count * sizeof (float),
                                   cudaMemcpyDeviceToHost);
        }

        if (cuerrcode != cudaSuccess) {
            // 如果拷贝操作失败，则再报错退出前，需要将旧的目标模板数据恢复到目
            // 标模板中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
            // 防止内存泄漏。最后，还需要把 Device 切换回来，以免整个程序乱套。
            if (!reusedata) {
                delete[] dsttpl->tplData;
                delete[] dsttplCud->attachedData;
            }
            *dsttplCud = olddsttplCud;
            cudaSetDevice(curdevid);
            return CUDA_ERROR;
        }

        // 对内存操作完毕后，将设备切换回当前的 Device。
        cudaSetDevice(curdevid);
    }

    // 到此步骤已经说明新的模板数据空间已经成功的申请并拷贝了新的数据，因此，旧
    // 的数据空间已毫无用处。本步骤就是释放掉旧的数据空间以防止内存泄漏。这里，
    // 作为拷贝的 olddsttplCud 是局部变量，因此相应的元数据会在本函数退出后自动
    // 释放，不用理会。
    if (olddsttplCud.tplMeta.tplData != NULL) {
        if (olddsttplCud.deviceId > 0) {
            // 如果旧数据是存储于 Device 内存上的数据，则需要无条件的释放。
            cudaSetDevice(olddsttplCud.deviceId);
            cudaFree(olddsttplCud.tplMeta.tplData);
            cudaFree(olddsttplCud.attachedData);
            cudaSetDevice(curdevid);
        } else if (!reusedata) {
            // 如果旧数据就在 Host 内存上，则对于 reusedata 未置位的情况进行释
            // 放，因为一旦置位，旧的数据空间就被用于承载新的数据，则不能释放。
            delete[] olddsttplCud.tplMeta.tplData;
            delete[] olddsttplCud.attachedData;
        }
    }

    // 处理完毕，退出。
    return NO_ERROR;
}
