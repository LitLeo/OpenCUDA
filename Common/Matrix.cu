// Matrix.cu
// 矩阵数据结构定义和矩阵的基本操作

#include "Matrix.h"

#include <iostream>
#include <fstream>
using namespace std;

#include "ErrorCode.h"


// Host 静态方法：newMatrix（创建矩阵）
__host__ int MatrixBasicOp::newMatrix(Matrix **outmat)
{
    MatrixCuda *resmatCud;  // 对应于返回的 outmat 的 MatrixCuda 型数据。

    // 检查装载输出矩阵的指针是否为 NULL。
    if (outmat == NULL)
        return NULL_POINTER;

    // 申请矩阵元数据的空间。
    resmatCud = new MatrixCuda;

    // 初始化矩阵上的数据为空矩阵。
    resmatCud->matMeta.width = 0;
    resmatCud->matMeta.height = 0;
    resmatCud->matMeta.roiX1 = 0;
    resmatCud->matMeta.roiY1 = 0;
    resmatCud->matMeta.roiX2 = 0;
    resmatCud->matMeta.roiY2 = 0;
    resmatCud->matMeta.matData = NULL;
    resmatCud->deviceId = -1;
    resmatCud->pitchWords = 0;

    // 将 Matrix 赋值给输出参数。
    *outmat = &(resmatCud->matMeta);

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：deleteMatrix（销毁矩阵）
__host__ int MatrixBasicOp::deleteMatrix(Matrix *inmat)
{
    // 检查矩阵的指针是否为 NULL。
    if (inmat == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Matrix 型指针，得到对应的 MatrixCuda 型数据。
    MatrixCuda *inmatCud = MATRIX_CUDA(inmat);

    // 检查矩阵所在的地址空间是否合法，如果矩阵所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (inmatCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 释放矩阵数据，即像素数据。
    if (inmat->matData == NULL || inmat->width == 0 || inmat->height == 0 ||
        inmatCud->pitchWords == 0) {
        // 如果输入矩阵是空的，则不进行矩阵数据释放操作（因为本来也没有数据可被
        // 释放）。
        // Do Nothing;
    } if (inmatCud->deviceId < 0) {
        // 对于数据存储于 Host 内存，直接利用 delete 关键字释放矩阵数据。
        delete[] inmat->matData;
    } else if (inmatCud->deviceId == curdevid) {
        // 对于数据存储于当前 Device 内存中，则直接利用 cudaFree 接口释放该矩阵
        // 数据。
        cudaFree(inmat->matData);
    } else {
        // 对于数据存储于非当前 Device 内存中，则需要首先切换设备，将该设备作为
        // 当前 Device，然后释放之，最后还需要将设备切换回来以保证后续处理的正
        // 确性。
        cudaSetDevice(inmatCud->deviceId);
        cudaFree(inmat->matData);
        cudaSetDevice(curdevid);
    }

    // 释放矩阵的元数据。
    delete inmatCud;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
__host__ int MatrixBasicOp::makeAtCurrentDevice(Matrix *mat,
                                                size_t width, size_t height)
{
    // 检查输入矩阵是否为 NULL
    if (mat == NULL)
        return NULL_POINTER;

    // 检查给定的矩阵的长宽是否合法
    if (width < 1 || height < 1)
        return INVALID_DATA;

    // 检查矩阵是否为空矩阵
    if (mat->matData != NULL)
        return UNMATCH_IMG;

    // 获取 mat 对应的 MatrixCuda 型数据。
    MatrixCuda *matCud = MATRIX_CUDA(mat);

    // 在当前的 Device 上申请存储指定尺寸图片所需要的内存空间。
    cudaError_t cuerrcode;
    float *newspace;
    size_t pitchbytes;
    cuerrcode = cudaMallocPitch((void **)(&newspace), &pitchbytes,
                                width * sizeof (float), height);
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 修改矩阵的元数据。其中 ROI 被设为整幅图片。
    mat->width = width;
    mat->height = height;
    mat->roiX1 = 0;
    mat->roiY1 = 0;
    mat->roiX2 = width;
    mat->roiY2 = height;
    mat->matData = newspace;
    matCud->deviceId = curdevid;
    matCud->pitchWords = pitchbytes / sizeof (float);
    // 这里我们坚信由 cudaMallocPitch 得到的 pitch 是可以被 sizeof (float) 整除
    // 的。如果真的不能整除，上面的那行除法会带来错误（当然，这在实际 CUDA 中是
    // 不会出现的）。

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 静态方法：makeAtHost（在 Host 内存中构建数据）
__host__ int MatrixBasicOp::makeAtHost(Matrix *mat,
                                       size_t width, size_t height)
{
    // 检查输入矩阵是否为 NULL
    if (mat == NULL)
        return NULL_POINTER;

    // 检查给定的矩阵的长宽是否合法
    if (width < 1 || height < 1)
        return INVALID_DATA;

    // 检查矩阵是否为空矩阵
    if (mat->matData != NULL)
        return UNMATCH_IMG;

    // 获取 mat 对应的 MatrixCuda 型数据。
    MatrixCuda *matCud = MATRIX_CUDA(mat);

    // 为矩阵数据在 Host 内存中申请空间
    mat->matData = new float[width * height];
    if (mat->matData == NULL)
        return OUT_OF_MEM;

    // 设置矩阵中的元数据
    mat->width = width;
    mat->height = height;
    mat->roiX1 = 0;
    mat->roiY1 = 0;
    mat->roiX2 = width;
    mat->roiY2 = height;
    matCud->deviceId = -1;
    matCud->pitchWords = width;

    // 处理完毕，退出
    return NO_ERROR;
}

// Host 静态方法：readFromFile（从文件读取矩阵）
__host__ int MatrixBasicOp::readFromFile(const char *filepath, Matrix *outmat)
{
    // 检查文件路径和矩阵是否为 NULL。
    if (filepath == NULL || outmat == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Matrix 型指针，得到对应的 MatrixCuda 型数据。
    MatrixCuda *outmatCud = MATRIX_CUDA(outmat);

    // 检查矩阵所在的地址空间是否合法，如果矩阵所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetErrorString(cudaGetDeviceCount(&devcnt));
    if (outmatCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 打开矩阵文件。
    ifstream matfile(filepath, ios::in | ios::binary);
    if (!matfile)
        return NO_FILE;

    // 读取文件头部的文件类型信息，如果文件的头两个字节不是 BM，则说明该文件不
    // 是 BMP 文件，则报错。
    char headstr[2] = { '\0' };
    matfile.seekg(0x0000, ios::beg);
    matfile.read(headstr, 2);
    if (headstr[0] != 'B' || headstr[1] != 'M')
       return WRONG_FILE;

    // 读取文件中的 BPP 字段（每个像素占用的比特数量），如果 BPP 的值不为 8，说
    // 明该文件不是一个灰度 BMP 矩阵，则报错。
    unsigned short bpp = 0;
    matfile.seekg(0x001C, ios::beg);
    matfile.read(reinterpret_cast<char *>(&bpp), 2);
    if (bpp != 8)
        return WRONG_FILE;

    // 从文件中读取矩阵宽度和高度信息。
    unsigned int width = 0, height = 0;
    matfile.seekg(0x0012, ios::beg);
    matfile.read(reinterpret_cast<char *>(&width), 4);
    matfile.read(reinterpret_cast<char *>(&height), 4);
    // 如果矩阵的尺寸不合法，则报错退出。
    if (width < 1 || height < 1)
        return WRONG_FILE;

    // 从文件中读取像素数据所在的文件中的偏移位置。
    unsigned int dataoff = 0;
    matfile.seekg(0x000A, ios::beg);
    matfile.read(reinterpret_cast<char *>(&dataoff), 4);

    // 获取存放矩阵像素数据的 Host 内存空间。本着尽量重用的思想，如果原来的矩阵
    // 内存数据是存储于 Host 内存，且尺寸和新的矩阵尺寸一致时，则不重新申请
    // Host 内存空间，直接利用原来的空间存放新的矩阵数据。
    float *matdata = outmat->matData;
    bool reusedata = true;
    if (outmat->matData == NULL || outmatCud->deviceId >= 0 || 
        outmat->width != width || outmat->height != height) {
        matdata = new float[width * height];
        // 如果没有申请到新的数据，则报错。
        if (matdata == NULL)
            return OUT_OF_MEM;
        reusedata = false;
    }

    // 计算 BMP 文件中每行的 Padding 尺寸。在 BMP 文件中，每行的数据都需要保证
    // 4 字节对齐。如果某行的宽度不是 4 的整数倍（注意，灰度图中每行的像素个数
    // 同每行实际数据占用的字节数是相等的），则需要补充一些字节，使其达到 4 的
    // 整数倍。
    unsigned int dummybytes = (4 - (width & 3)) & 3;

    // 将文件指针移动到数据存储的开始位置
    matfile.seekg(dataoff, ios::beg);

    // 文件读取的数据缓冲区，每次读取一行的数据。
    unsigned char *pbufdata = new unsigned char[width + dummybytes];

    // 读取矩阵中的各行的矩阵数据。由于 BMP 采用了右手坐标，即矩阵的左下角点为
    // 原点，整个矩阵位于第一象限，而我们系统内部使用的是左手坐标，即矩阵的左上
    // 角点为原点，整个矩阵亦位于第一象限。这样，BMP 文件中的第一行矩阵数据，其
    // 时是最后一行数据，因此，这个关于 r 的 for 循环是从大到小的循环。
    for (int r = height - 1; r >= 0; r--) {
        // 读取矩阵数据（每次读取一行的数据）
        matfile.read(reinterpret_cast<char *>(pbufdata), width + dummybytes);

        // 通过这个 for 循环将读取到的一行数据逐一的转换成浮点型归一化的数据，
        // 即从 0.0 到 1.0 范围内测数据。
        for (int c = 0; c < width; c++) {
            // 将对应列的数据归一化到 [0.0, 1.0]，存放到矩阵的浮点型数组中。
            outmat->matData[r * outmatCud->pitchWords + c] =
                    ((float)pbufdata[c]) / 255.0f;
        }
    }

    // 到此为止，矩阵数据读取完毕，这是可以安全的释放掉矩阵原来的数据。一直拖到
    // 最后才释放原来的数据，正是为了防止一旦矩阵读取失败，不至于让系统进入一个
    // 混乱的状态，因为原来的数据还是处于一个可用的状态。
    if (reusedata == false || outmat->matData != NULL) {
        if (outmatCud->deviceId < 0) {
            // 如果原来的数据存放于 Host 内存中，则使用 delete 关键字释放。
            delete[] outmat->matData;
        } else {
            // 如果原来的数据存放于 Device 内存中，则首先调到对应的 Device，然
            // 后使用 cudaFree 释放掉内存。
            cudaSetDevice(outmatCud->deviceId);
            cudaFree(outmat->matData);
            cudaSetDevice(curdevid);
        }
    }

    // 为矩阵赋值新的元数据。这里 ROI 被重置为整幅矩阵。
    outmat->width = width;
    outmat->height = height;
    outmat->roiX1 = 0;
    outmat->roiY1 = 0;
    outmat->roiX2 = width;
    outmat->roiY2 = height;
    outmat->matData = matdata;
    outmatCud->deviceId = -1;
    outmatCud->pitchWords = width;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：writeToFile（将矩阵写入文件）
__host__ int MatrixBasicOp::writeToFile(const char *filepath, Matrix *inmat)
{
    // 检查文件路径和矩阵是否为 NULL。
    if (filepath == NULL || inmat == NULL)
        return NULL_POINTER;

    // 打开需要写入的文件。
    ofstream matfile(filepath, ios::out | ios::binary);
    if (!matfile) 
        return NO_FILE;

    // 根据输入参数的 Matrix 型指针，得到对应的 MatrixCuda 型数据。
    MatrixCuda *inmatCud = MATRIX_CUDA(inmat);

    // 将图片的数据拷贝回 Host 内存中，这样图片就可以被下面的代码所读取，然后将
    // 矩阵的数据写入到磁盘中。这里需要注意的是，安排图片的拷贝过程在文件打开之
    // 后是因为，如果一旦文件打开失败，则不会改变矩阵在内存中的存储状态，这可能
    // 会对后续处理更加有利。
    int errcode;
    errcode = MatrixBasicOp::copyToHost(inmat);
    if (errcode < 0)
        return errcode;

    // 计算一些和 BMP 矩阵相关的参数：
    // 计算 BMP 文件中每行的 Padding 尺寸。在 BMP 文件中，每行的数据都需要保证
    // 4 字节对齐。如果某行的宽度不是 4 的整数倍（注意，灰度图中每行的像素个数
    // 同每行实际数据占用的字节数是相等的），则需要补充一些字节，使其达到 4 的
    // 整数倍。
    unsigned int dummybytes = (4 - (inmat->width & 3)) & 3;

    // 计算在磁盘上存储图片总共需要的字节数量，这个数量包括了上面提到的 Padding
    // 的尺寸。
    unsigned int datalen = inmat->height * (inmat->width + dummybytes);

    // 在存储到磁盘中后，像素数据实际的起始位置。因为 BMP 文件存在信息头，实际
    // 的像素数据是在这些信息头的后面的。对于系统中使用到的灰度矩阵来说，信息头
    // 包含了两个部分，最前面的是矩阵的元数据（如矩阵的宽度、高度；数据的尺寸等
    // 信息），紧随其后的是颜色表，颜色表共有 256 个条目，对应了 256 级灰度，每
    // 个条目包含了 4 个字节，这四个字节分别为 RGBA 四个通道的亮度值。
    unsigned int dataoff = 4 * 256 + 54;

    // 向文件中写入 BMP 头信息
    unsigned short ustemp;  // 这三个变量用来保存头信息中的临时域的值，三个变量
    unsigned int uitemp;    // 用来处理不同的数据类型。
    int sitemp;

    // 文件类型头
    ustemp = 0x4D42;
    matfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // 文件长度
    uitemp = datalen + dataoff;
    matfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 保留区段甲
    ustemp = 0;
    matfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // 保留区段乙
    ustemp = 0;
    matfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // 像素数据在文件中开始的位置
    uitemp = dataoff;
    matfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 矩阵信息头尺寸
    uitemp = 40;
    matfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 矩阵宽度
    sitemp = inmat->width;
    matfile.write(reinterpret_cast<char *>(&sitemp), 4);
    // 矩阵高度
    sitemp = inmat->height;
    matfile.write(reinterpret_cast<char *>(&sitemp), 4);
    // 矩阵层次数量
    ustemp = 1;
    matfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // BPP（每像素的比特数量）
    ustemp = 8;
    matfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // 压缩算法
    uitemp = 0;
    matfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 矩阵尺寸
    uitemp = datalen;
    matfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 每公尺的像素数量（X-方向）
    sitemp = 0;
    matfile.write(reinterpret_cast<char *>(&sitemp), 4);
    // 每公尺的像素数量（Y-方向）
    sitemp = 0;
    matfile.write(reinterpret_cast<char *>(&sitemp), 4);
    // ClrUsed
    uitemp = 256;
    matfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // ClrImportant
    uitemp = 0;
    matfile.write(reinterpret_cast<char *>(&uitemp), 4);

    // 写入颜色表信息
    // 颜色信息共有 256 个条目，对应了 256 个灰度级；每个条目包含了 4 个颜色通
    // 道的数据。由于矩阵是灰度矩阵，因此对于灰度为 i 的对应的颜色值为 < i, i,
    // i, FF >。
    unsigned char coloritem[4] = { 0x00, 0x00, 0x00, 0xFF };
    for (int i = 0; i < 256; i++) {
        coloritem[0] = coloritem[1] = coloritem[2] = i;
        matfile.write(reinterpret_cast<char *>(coloritem), 4);
    }

    // 保存一行图像像素数据的缓冲空间。
    unsigned char *pbufdata = new unsigned char[inmat->width + dummybytes];

    // 为了防止引起不必要的麻烦与错误，这里将补白区间内的数据手动赋值为 0。
    for (int i = inmat->width; i < inmat->width + dummybytes; i++)
        pbufdata[i] = '\0';
    
    // 逐行写入矩阵的像素数据。由于 BMP 采用了右手坐标，即矩阵的左下角点为原
    // 点，整个矩阵位于第一象限，而我们系统内部使用的是左手坐标，即矩阵的左上角
    // 点为原点，整个矩阵亦位于第一象限。这样，BMP 文件中的第一行矩阵数据，其时
    // 是最后一行数据，因此该循环为从大到小的循环。
    for (int r = inmat->height - 1; r >= 0; r--) {
        // 逐一将当前行的各列数据转换成对应的 [0, 255] 灰度值。
        for (int c = 0; c < inmat->width; c++) {
            pbufdata[c] = (unsigned char)(
                    inmat->matData[r * inmatCud->pitchWords + c] * 255.0f);
        }

        // 写入当前行的像素数据。
        matfile.write(reinterpret_cast<char *>(pbufdata),
                      inmat->width + dummybytes);
    }

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：copyToCurrentDevice（将矩阵拷贝到当前 Device 内存上）
__host__ int MatrixBasicOp::copyToCurrentDevice(Matrix *mat)
{
    // 检查矩阵是否为 NULL。
    if (mat == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Matrix 型指针，得到对应的 MatrixCuda 型数据。
    MatrixCuda *matCud = MATRIX_CUDA(mat);

    // 检查矩阵所在的地址空间是否合法，如果矩阵所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (matCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果矩阵是一个不包含数据的空矩阵，则报错。
    if (mat->matData == NULL || mat->width == 0 || mat->height == 0 || 
        matCud->pitchWords == 0) 
        return UNMATCH_IMG;

    // 对于不同的情况，将矩阵数据拷贝到当前设备上。
    if (matCud->deviceId < 0) {
        // 如果矩阵的数据位于 Host 内存上，则需要在当前 Device 的内存空间上申请
        // 空间，然后将 Host 内存上的数据进行 Padding 后拷贝到当前 Device 上。
        float *devptr;  // 新的数据空间，在当前 Device 上。
        size_t pitch;           // Padding 后的每行尺寸
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 在当前设备上申请空间，使用 Pitch 版本的申请函数，用来进行 Padding。
        cuerrcode = cudaMallocPitch((void **)(&devptr), &pitch, 
                                    mat->width * sizeof (float), mat->height);
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 进行 Padding 并拷贝数据到当前 Device 上。注意，这里 mat->pitchWords
        // == mat->width。
        cuerrcode = cudaMemcpy2D(devptr, pitch, 
                                 mat->matData, matCud->pitchWords,
                                 mat->width * sizeof (float), mat->height,
                                 cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }

        // 释放掉原来存储于 Host 内存上的矩阵数据。
        delete[] mat->matData;

        // 更新矩阵数据，把新的在当前 Device 上申请的数据和相关数据写入矩阵元数
        // 据中。
        mat->matData = devptr;
        matCud->deviceId = curdevid;
        matCud->pitchWords = pitch / sizeof (float);

        // 操作完毕，返回。
        return NO_ERROR;

    } else if (matCud->deviceId != curdevid) {
        // 对于数据存在其他 Device 的情况，仍旧要在当前 Device 上申请数据空间，
        // 并从另一个 Device 上拷贝数据到新申请的当前 Device 的数据空间中。
        float *devptr;  // 新申请的当前 Device 上的数据。
        size_t datasize = matCud->pitchWords * mat->height *  // 数据尺寸。
                          sizeof (float);
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 在当前 Device 上申请空间。
        cuerrcode = cudaMalloc((void **)(&devptr), datasize);
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 将数据从矩阵原来的存储位置拷贝到当前的 Device 上。
        cuerrcode = cudaMemcpyPeer(devptr, curdevid, 
                                   mat->matData, matCud->deviceId,
                                   datasize);
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }

        // 释放掉矩阵在原来的 Device 上的数据。
        cudaFree(mat->matData);

        // 将新的矩阵数据信息写入到矩阵元数据中。
        mat->matData = devptr;
        matCud->deviceId = curdevid;

        // 操作完成，返回。
        return NO_ERROR;
    }

    // 对于其他情况，即矩阵数据本来就在当前 Device 上，则直接返回，不进行任何的
    // 操作。
    return NO_ERROR;
}

// Host 静态方法：copyToCurrentDevice（将矩阵拷贝到当前 Device 内存上）
__host__ int MatrixBasicOp::copyToCurrentDevice(Matrix *srcmat, Matrix *dstmat)
{
    // 检查输入矩阵是否为 NULL。
    if (srcmat == NULL)
        return NULL_POINTER;

    // 如果输出矩阵为 NULL，或者输出矩阵和输入矩阵为同一各矩阵，则调用 In-place
    // 版本的函数。
    if (dstmat == NULL || dstmat == srcmat)
        return copyToCurrentDevice(srcmat);

    // 获取 srcmat 和 dstmat 对应的 MatrixCuda 型指针。
    MatrixCuda *srcmatCud = MATRIX_CUDA(srcmat);
    MatrixCuda *dstmatCud = MATRIX_CUDA(dstmat);

    // 用来存放旧的 dstmat 数据，使得在拷贝操作失败时可以恢复为原来的可用的数据
    // 信息，防止系统进入一个混乱的状态。
    MatrixCuda olddstmatCud = *dstmatCud;  // 旧的 dstmat 数据
    bool reusedata = true;                // 记录是否重用了原来的矩阵数据空间。
                                          // 该值为 ture，则原来的数据空间被重
                                          // 用，不需要在之后释放数据，否则需要
                                          // 在最后释放旧的空间。

    // 如果源矩阵是一个空矩阵，则不进行任何操作，直接报错。
    if (srcmat->matData == NULL || srcmat->width == 0 || srcmat->height == 0 ||
        srcmatCud->pitchWords == 0)
        return INVALID_DATA;

    // 检查矩阵所在的地址空间是否合法，如果矩阵所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srcmatCud->deviceId >= devcnt || dstmatCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果目标矩阵中存在有数据，则需要根据情况，若原来的数据不存储在当前的
    // Device 上，或者即使存储在当前的 Device 上，但数据尺寸不匹配，则需要释放
    // 掉原来申请的空间，以便重新申请合适的内存空间。此处不进行真正的释放操作，
    // 其目的在于当后续操作出现错误时，可以很快的恢复 dstmat 中原来的信息，使得
    // 整个系统不会处于一个混乱的状态，本函数会在最后，确定 dstmat 被成功的更换
    // 为了新的数据以后，才会真正的将原来的矩阵数据释放掉。
    if (dstmatCud->deviceId != curdevid) {
        // 对于数据存在 Host 或其他的 Device 上，则直接释放掉原来的数据空间。
        reusedata = 0;
        dstmat->matData = NULL;
    } else if (!(((srcmatCud->deviceId < 0 && 
                   srcmat->width == dstmat->width) ||
                  dstmatCud->pitchWords == srcmatCud->pitchWords) &&
                 srcmat->height == dstmat->height)) {
        // 对于数据存在于当前 Device 上，则需要检查数据的尺寸是否和源矩阵相匹
        // 配。检查的标准包括：要求源矩阵的 Padding 后的行宽度和目标矩阵的相
        // 同，源矩阵和目标矩阵的高度相同；如果源矩阵是存储在 Host 内存中的，则
        // 仅要求源矩阵和目标矩阵的宽度相同即可。如果目标矩阵和源矩阵的尺寸不匹
        // 配则仍旧需要释放目标矩阵原来的数据空间。
        reusedata = 0;
        dstmat->matData = NULL;
    }

    // 将目标矩阵的尺寸更改为源矩阵的尺寸。
    dstmat->width = srcmat->width;
    dstmat->height = srcmat->height;

    // 将目标矩阵的 ROI 更改为源矩阵的 ROI。
    dstmat->roiX1 = srcmat->roiX1;
    dstmat->roiY1 = srcmat->roiY1;
    dstmat->roiX2 = srcmat->roiX2;
    dstmat->roiY2 = srcmat->roiY2;

    // 更改目标矩阵的数据存储位置为当前 Device。
    dstmatCud->deviceId = curdevid;

    // 将矩阵数据从源矩阵中拷贝到目标矩阵中。
    if (srcmatCud->deviceId < 0) {
        // 如果源矩阵数据存储于 Host 内存，则使用 cudaMemcpy2D 进行 Padding 形
        // 式的拷贝。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 如果目标矩阵的 matData == NULL，说明目标矩阵原本要么是一个空矩阵，要
        // 么目标矩阵原本的数据空间不合适，需要重新申请。这时，需要为目标矩阵重
        // 新在当前 Device 上申请一个合适的数据空间。
        if (dstmat->matData == NULL) {
            cuerrcode = cudaMallocPitch((void **)(&dstmat->matData), 
                                        &dstmatCud->pitchWords,
                                        dstmat->width * sizeof (float),
                                        dstmat->height);
            if (cuerrcode != cudaSuccess) {
                // 如果申请内存的操作失败，则再报错返回前需要将旧的目标矩阵数据
                // 恢复到目标矩阵中，以保证系统接下的操作不至于混乱。
                *dstmatCud = olddstmatCud;
                return CUDA_ERROR;
            }

            // 将通过 cudaMallocPitch 得到的以字节为单位的 pitch 值转换为以字为
            // 单位的值。
            dstmatCud->pitchWords /= sizeof (float);
        }

        // 使用 cudaMemcpy2D 进行 Padding 形式的拷贝。
        cuerrcode = cudaMemcpy2D(dstmat->matData,
                                 dstmatCud->pitchWords * sizeof (float),
                                 srcmat->matData,
                                 srcmatCud->pitchWords * sizeof (float),
                                 srcmat->width * sizeof (float),
                                 srcmat->height,
                                 cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            // 如果拷贝操作失败，则再报错退出前，需要将旧的目标矩阵数据恢复到目
            // 标矩阵中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
            // 防止内存泄漏。
            if (!reusedata)
                cudaFree(dstmat->matData);
            *dstmatCud = olddstmatCud;
            return CUDA_ERROR;
        }
    } else {
        
        // 如果源矩阵数据存储于 Device 内存（无论是当前 Device 还是其他的 
        // Device），都是用端到端的拷贝。
        cudaError_t cuerrcode;             // CUDA 调用返回的错误码。
        size_t datasize = srcmatCud->pitchWords * srcmat->height *
                          sizeof (float);  // 数据尺寸，单位：字节。

        // 如果目标矩阵需要申请数据空间，则进行申请。
        if (dstmat->matData == NULL) {
            cuerrcode = cudaMalloc((void **)(&dstmat->matData), datasize);
            if (cuerrcode != cudaSuccess) {
                // 如果发生错误，则需要首先恢复旧的矩阵数据，之后报错。恢复旧的
                // 矩阵数据以防止系统进入混乱状态。
                *dstmatCud = olddstmatCud;
                return CUDA_ERROR;
            }
        }

        // 更新目标矩阵的 Padding 尺寸与源矩阵相同。注意，因为源矩阵也存储在
        // Device 上，在 Device 上的数据都是经过 Padding 的，又因为
        // cudaMemcpyPeer 方法没有提供 Pitch 版本接口，所以，我们这里直接借用源
        // 矩阵的 Padding 尺寸。
        dstmatCud->pitchWords = srcmatCud->pitchWords;

        // 使用 cudaMemcpyPeer 实现两个 Device （可以为同一个 Device）间的数据
        // 拷贝，将源矩阵在 Device 上的数据信息复制到目标矩阵中。
        cuerrcode = cudaMemcpyPeer(dstmat->matData, curdevid,
                                   srcmat->matData, srcmatCud->deviceId,
                                   datasize);
        if (cuerrcode != cudaSuccess) {
            // 如果拷贝操作失败，则再报错退出前，需要将旧的目标矩阵数据恢复到目
            // 标矩阵中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
            // 防止内存泄漏。
            if (!reusedata)
                cudaFree(dstmat->matData);
            *dstmatCud = olddstmatCud;
            return CUDA_ERROR;
        }
    }

    // 到此步骤已经说明新的矩阵数据空间已经成功的申请并拷贝了新的数据，因此，旧
    // 的数据空间已毫无用处。本步骤就是释放掉旧的数据空间以防止内存泄漏。这里，
    // 作为拷贝的 olddstmatCud 是局部变量，因此相应的元数据会在本函数退出后自动
    // 释放，不用理会。
    if (olddstmatCud.matMeta.matData != NULL) {
        if (olddstmatCud.deviceId < 0) {
            // 如果旧数据空间是 Host 内存上的，则需要无条件释放。
            delete[] olddstmatCud.matMeta.matData;
        } else if (olddstmatCud.deviceId != curdevid) {
            // 如果旧数据空间不是当前 Device 内存上的其他 Device 内存上的数据，
            // 则也需要无条件的释放。
            cudaSetDevice(olddstmatCud.deviceId);
            cudaFree(olddstmatCud.matMeta.matData);
            cudaSetDevice(curdevid);
        } else if (!reusedata) {
            // 如果旧数据就在当前的 Device 内存上，则对于 reusedata 未置位的情
            // 况进行释放，因为一旦置位，旧的数据空间就被用于承载新的数据，则不
            // 能释放。
            cudaFree(olddstmatCud.matMeta.matData);
        }
    }

    return NO_ERROR;
}

// Host 静态方法：copyToHost（将矩阵拷贝到 Host 内存上）
__host__ int MatrixBasicOp::copyToHost(Matrix *mat)
{
    // 检查矩阵是否为 NULL。
    if (mat == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Matrix 型指针，得到对应的 MatrixCuda 型数据。
    MatrixCuda *matCud = MATRIX_CUDA(mat);

    // 检查矩阵所在的地址空间是否合法，如果矩阵所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (matCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果矩阵是一个不包含数据的空矩阵，则报错。
    if (mat->matData == NULL || mat->width == 0 || mat->height == 0 || 
        matCud->pitchWords == 0) 
        return UNMATCH_IMG;

    // 对于不同的情况，将矩阵数据拷贝到当前设备上。
    if (matCud->deviceId < 0) {
        // 如果矩阵位于 Host 内存上，则不需要进行任何操作。
        return NO_ERROR;

    } else {
        // 如果矩阵的数据位于 Device 内存上，则需要在 Host 的内存空间上申请空
        // 间，然后将数据消除 Padding 后拷贝到 Host 上。
        float *hostptr;         // 新的数据空间，在 Host 上。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 在 Host 上申请空间。
        hostptr = new float[mat->width * mat->height];
        if (hostptr == NULL)
            return OUT_OF_MEM;

        // 将设备切换到数据所在的 Device 上。
        cudaSetDevice(matCud->deviceId);

        // 消除 Padding 并拷贝数据
        cuerrcode = cudaMemcpy2D(hostptr, mat->width * sizeof (float),
                                 mat->matData,
                                 matCud->pitchWords * sizeof (float),
                                 mat->width * sizeof (float), mat->height,
                                 cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
            // 后报错返回。
            delete[] hostptr;
            return CUDA_ERROR;
        }

        // 释放掉原来存储于 Device 内存上的矩阵数据。
        cudaFree(mat->matData);

        // 对 Device 内存的操作完毕，将设备切换回当前 Device。
        cudaSetDevice(curdevid);

        // 更新矩阵数据，把新的在当前 Device 上申请的数据和相关数据写入矩阵元数
        // 据中。
        mat->matData = hostptr;
        matCud->deviceId = -1;
        matCud->pitchWords = mat->width;

        // 操作完毕，返回。
        return NO_ERROR;
    }

    // 程序永远也不会到达这个分支，因此如果到达这个分支，则说明系统紊乱。对于多
    // 数编译器来说，会对此句报出不可达语句的 Warning，因此这里将其注释掉，以防
    // 止不必要的 Warning。
    //return UNKNOW_ERROR;
}

// Host 静态方法：copyToHost（将矩阵拷贝到 Host 内存上）
__host__ int MatrixBasicOp::copyToHost(Matrix *srcmat, Matrix *dstmat)
{
    // 检查输入矩阵是否为 NULL。
    if (srcmat == NULL)
        return NULL_POINTER;

    // 如果输出矩阵为 NULL 或者和输入矩阵同为一个矩阵，则调用对应的 In-place 版
    // 本的函数。
    if (dstmat == NULL || dstmat == srcmat)
        return copyToHost(srcmat);

    // 获取 srcmat 和 dstmat 对应的 MatrixCuda 型指针。
    MatrixCuda *srcmatCud = MATRIX_CUDA(srcmat);
    MatrixCuda *dstmatCud = MATRIX_CUDA(dstmat);

    // 用来存放旧的 dstmat 数据，使得在拷贝操作失败时可以恢复为原来的可用的数据
    // 信息，防止系统进入一个混乱的状态。
    MatrixCuda olddstmatCud = *dstmatCud;  // 旧的 dstmat 数据
    bool reusedata = true;                // 记录是否重用了原来的矩阵数据空间。
                                          // 该值为 true，则原来的数据空间被重
                                          // 用，不需要在之后释放数据，否则需要
                                          // 释放旧的空间。

    // 如果源矩阵是一个空矩阵，则不进行任何操作，直接报错。
    if (srcmat->matData == NULL || srcmat->width == 0 || srcmat->height == 0 ||
        srcmatCud->pitchWords == 0)
        return INVALID_DATA;

    // 检查矩阵所在的地址空间是否合法，如果矩阵所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srcmatCud->deviceId >= devcnt || dstmatCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果目标矩阵中存在有数据，则需要根据情况，若原来的数据不存储在 Host 上，
    // 或者即使存储在 Host 上，但数据尺寸不匹配，则需要释放掉原来申请的空间，以
    // 便重新申请合适的内存空间。此处不进行真正的释放操作，其目的在于当后续操作
    // 出现错误时，可以很快的恢复 dstmat 中原来的信息，使得整个系统不会处于一个
    // 混乱的状态，本函数会在最后，确定 dstmat 被成功的更换为了新的数据以后，才
    // 会真正的将原来的矩阵数据释放掉。
    if (dstmatCud->deviceId >= 0) {
        // 对于数据存在于 Device 上，则亦直接释放掉原来的数据空间。
        reusedata = 0;
        dstmat->matData = NULL;
    } else if (!(srcmat->width == dstmat->width &&
                 srcmat->height == dstmat->height)) {
        // 对于数据存在于 Host 上，则需要检查数据的尺寸是否和源矩阵相匹配。检查
        // 的标准：源矩阵和目标矩阵的尺寸相同时，可重用原来的空间。
        reusedata = 0;
        dstmat->matData = NULL;
    }

    // 将目标矩阵的尺寸更改为源矩阵的尺寸。
    dstmat->width = srcmat->width;
    dstmat->height = srcmat->height;

    // 将目标矩阵的 ROI 更改为源矩阵的 ROI。
    dstmat->roiX1 = srcmat->roiX1;
    dstmat->roiY1 = srcmat->roiY1;
    dstmat->roiX2 = srcmat->roiX2;
    dstmat->roiY2 = srcmat->roiY2;

    // 更改目标矩阵的数据存储位置为 Host。
    dstmatCud->deviceId = -1;

    // 由于 Host 内存上的数据不使用 Padding，因此设置 Padding 尺寸为矩阵的宽
    // 度。
    dstmatCud->pitchWords = dstmat->width;

    // 如果目标矩阵的 matData == NULL，说明目标矩阵原本要么是一个空矩阵，要么目
    // 标矩阵原本的数据空间不合适，需要重新申请。这时，需要为目标矩阵重新在 
    // Host 上申请一个合适的数据空间。
    if (dstmat->matData == NULL) {
        dstmat->matData = new float[srcmat->width * srcmat->height];
        if (dstmat->matData == NULL) {
            // 如果申请内存的操作失败，则再报错返回前需要将旧的目标矩阵数据
            // 恢复到目标矩阵中，以保证系统接下的操作不至于混乱。
            *dstmatCud = olddstmatCud;
            return OUT_OF_MEM;
        }
    }

    // 将矩阵数据从源矩阵中拷贝到目标矩阵中。
    if (srcmatCud->deviceId < 0) {
        // 如果源矩阵数据存储于 Host 内存，则直接使用 C 标准支持库中的 emcpy
        // 完成拷贝。

        // 将 srcmat 内的矩阵数据拷贝到 dstmat 中。memcpy 不返回错误，因此，没
        // 有进行错误检查。
        memcpy(dstmat->matData, srcmat->matData, 
               srcmat->width * srcmat->height * sizeof (float));

    } else {
        // 如果源矩阵数据存储于 Device 内存（无论是当前 Device 还是其他的 
        // Device），都是 2D 形式的拷贝，并消除 Padding。
        cudaError_t cuerrcode;                     // CUDA 调用返回的错误码。

        // 首先切换到 srcmat 矩阵数据所在的 Device，以方便进行内存操作。
        cudaSetDevice(srcmatCud->deviceId);

        // 这里使用 cudaMemcpy2D 将 srcmat 中处于 Device 上的数据拷贝到 dstmat
        // 中位于 Host 的内存空间上面，该拷贝会同时消除 Padding。
        cuerrcode = cudaMemcpy2D(dstmat->matData,
                                 dstmatCud->pitchWords * sizeof (float),
                                 srcmat->matData,
                                 srcmatCud->pitchWords * sizeof (float),
                                 srcmat->width * sizeof (float),
                                 srcmat->height,
                                 cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 如果拷贝操作失败，则再报错退出前，需要将旧的目标矩阵数据恢复到目
            // 标矩阵中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
            // 防止内存泄漏。最后，还需要把 Device 切换回来，以免整个程序乱套。
            if (!reusedata)
                delete[] dstmat->matData;
            *dstmatCud = olddstmatCud;
            cudaSetDevice(curdevid);
            return CUDA_ERROR;
        }

        // 对内存操作完毕后，将设备切换回当前的 Device。
        cudaSetDevice(curdevid);
    }

    // 到此步骤已经说明新的矩阵数据空间已经成功的申请并拷贝了新的数据，因此，旧
    // 的数据空间已毫无用处。本步骤就是释放掉旧的数据空间以防止内存泄漏。这里，
    // 作为拷贝的 olddstmatCud 是局部变量，因此相应的元数据会在本函数退出后自动
    // 释放，不用理会。
    if (olddstmatCud.matMeta.matData != NULL) {
        if (olddstmatCud.deviceId > 0) {
            // 如果旧数据是存储于 Device 内存上的数据，则需要无条件的释放。
            cudaSetDevice(olddstmatCud.deviceId);
            cudaFree(olddstmatCud.matMeta.matData);
            cudaSetDevice(curdevid);
        } else if (!reusedata) {
            // 如果旧数据就在 Host 内存上，则对于 reusedata 未置位的情况进行释
            // 放，因为一旦置位，旧的数据空间就被用于承载新的数据，则不能释放。
            delete[] olddstmatCud.matMeta.matData;
        }
    }

    // 处理完毕，退出。
    return NO_ERROR;
}
