// Image.cu
// 图像数据结构定义和图像的基本操作

#include "Image.h"

#include <iostream>
#include <fstream>
using namespace std;

#include "ErrorCode.h"


// Host 静态方法：newImage（创建图像）
__host__ int ImageBasicOp::newImage(Image **outimg)
{
    // 静态局部变量，标记是否设置过设备属性。
    static bool flagSetted  = false;
    
    // 判断设备是否设置过使用内存映射
    if (!flagSetted) {
        // 错误码
        cudaError_t cuerrcode;

        // 获取设备属性，检查是否可以使用内存映射功能。
        cudaDeviceProp deviceProp;  
        cudaGetDeviceProperties(&deviceProp, 0);
        if (!deviceProp.canMapHostMemory)
            return CUDA_ERROR; 

        // 设置设备属性为使用内存映射模式。
        cuerrcode = cudaSetDeviceFlags(cudaDeviceMapHost);  
        if (cuerrcode != cudaSuccess) 
            return CUDA_ERROR;   
        
        // 将标记更新为已设置
        flagSetted = true;
    }

    ImageCuda *resimgCud;  // 对应于返回的 outimg 的 ImageCuda 型数据。

    // 检查装载输出图像的指针是否为 NULL。
    if (outimg == NULL)
        return NULL_POINTER;

    // 申请图像元数据的空间。
    resimgCud = new ImageCuda;

    // 初始化图像上的数据为空图像。
    resimgCud->imgMeta.width = 0;
    resimgCud->imgMeta.height = 0;
    resimgCud->imgMeta.roiX1 = 0;
    resimgCud->imgMeta.roiY1 = 0;
    resimgCud->imgMeta.roiX2 = 0;
    resimgCud->imgMeta.roiY2 = 0;
    resimgCud->imgMeta.imgData = NULL;
    resimgCud->deviceId = -1;
    resimgCud->pitchBytes = 0;
    resimgCud->mapSource = NULL;

    // 将 Image 赋值给输出参数。
    *outimg = &(resimgCud->imgMeta);
    
    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：deleteImage（销毁图像）
__host__ int ImageBasicOp::deleteImage(Image *inimg)
{
    // 检查图像的指针是否为 NULL。
    if (inimg == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Image 型指针，得到对应的 ImageCuda 型数据。
    ImageCuda *inimgCud = IMAGE_CUDA(inimg);

    // 检查图像所在的地址空间是否合法，如果图像所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (inimgCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 如果原图像为内存映射图像，应该先解除内存映射，以保证正确释放地址空间。 
    if (inimgCud->mapSource != NULL) {
        int errcode;
        errcode = ImageBasicOp::unmapToHost(inimg);
        if (errcode < NO_ERROR)
            return errcode;
    }
        
    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 释放图像数据，即像素数据。
    if (inimg->imgData == NULL || inimg->width == 0 || inimg->height == 0 ||
        inimgCud->pitchBytes == 0) {
        // 如果输入图像是空的，则不进行图像数据释放操作（因为本来也没有数据可被
        // 释放）。
        // Do Nothing;
    } 
    if (inimgCud->deviceId < 0) {
        // 对于数据存储于主机 pinned 内存，调用 cudaFreeHost 释放图像数据。
        cudaFreeHost(inimg->imgData); 
    } else if (inimgCud->deviceId == curdevid) {
        // 对于数据存储于当前 Device 内存中，对于普通图像，直接利用 cudaFree 
        // 接口释放该图像数据。
        cudaFree(inimg->imgData);
    } else {
        // 对于数据存储于非当前 Device 内存中，则需要首先切换设备，将该设备作为
        // 当前 Device，然后释放之，最后还需要将设备切换回来以保证后续处理的正
        // 确性。
        cudaSetDevice(inimgCud->deviceId);
        cudaFree(inimg->imgData);
        cudaSetDevice(curdevid);
    }

    // 释放图像的元数据。
    delete inimgCud;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
__host__ int ImageBasicOp::makeAtCurrentDevice(Image *img,
                                               size_t width, size_t height)
{
    // 检查输入图像是否为 NULL
    if (img == NULL)
        return NULL_POINTER;

    // 检查给定的图像的长宽是否合法
    if (width < 1 || height < 1)
        return INVALID_DATA;

    // 检查图像是否为空图像
    if (img->imgData != NULL)
        return UNMATCH_IMG;

    // 获取 img 对应的 ImageCuda 型数据。
    ImageCuda *imgCud = IMAGE_CUDA(img);
        
    // 在当前的 Device 上申请存储指定尺寸图片所需要的内存空间。
    cudaError_t cuerrcode;
    cuerrcode = cudaMallocPitch((void **)(&img->imgData), &imgCud->pitchBytes,
                                width * sizeof (unsigned char), height);
    if (cuerrcode != cudaSuccess) {
        img->imgData = NULL;
        return CUDA_ERROR;
    }

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 修改图像的元数据。其中 ROI 被设为整幅图片。
    img->width = width;
    img->height = height;
    img->roiX1 = 0;
    img->roiY1 = 0;
    img->roiX2 = width;
    img->roiY2 = height;
    imgCud->deviceId = curdevid;
    // 由于 pitchBytes 已经在 cudaMallocPitch 中赋值，此处则不再对其进行赋值。

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 静态方法：makeAtHost（在 Host 内存中构建数据）
__host__ int ImageBasicOp::makeAtHost(Image *img,
                                      size_t width, size_t height)
{
    // 检查输入图像是否为 NULL
    if (img == NULL)
        return NULL_POINTER;

    // 检查给定的图像的长宽是否合法
    if (width < 1 || height < 1)
        return INVALID_DATA;

    // 检查图像是否为空图像
    if (img->imgData != NULL)
        return UNMATCH_IMG;

    // 获取 img 对应的 ImageCuda 型数据。
    ImageCuda *imgCud = IMAGE_CUDA(img);

    // 为图像数据在 Host 内存中申请空间
    cudaError_t cuerrcode;
    cuerrcode = cudaHostAlloc((void**)&(img->imgData), width * height * 
                              sizeof(unsigned char),cudaHostAllocMapped);
    if (cuerrcode != cudaSuccess) {
        img->imgData = NULL;
        return CUDA_ERROR;
    }

    // 设置图像中的元数据
    img->width = width;
    img->height = height;
    img->roiX1 = 0;
    img->roiY1 = 0;
    img->roiX2 = width;
    img->roiY2 = height;
    imgCud->deviceId = -1;
    imgCud->pitchBytes = width;

    // 处理完毕，退出
    return NO_ERROR;
}

// Host 静态方法：readFromFile（从文件读取图像）
__host__ int ImageBasicOp::readFromFile(const char *filepath, Image *outimg)
{
    // 检查文件路径和图像是否为 NULL。
    if (filepath == NULL || outimg == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Image 型指针，得到对应的 ImageCuda 型数据。
    ImageCuda *outimgCud = IMAGE_CUDA(outimg);

    // 检查图像所在的地址空间是否合法，如果图像所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetErrorString(cudaGetDeviceCount(&devcnt));
    if (outimgCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 打开图像文件。
    ifstream imgfile(filepath, ios::in | ios::binary);
    if (!imgfile)
        return NO_FILE;

    // 读取文件头部的文件类型信息，如果文件的头两个字节不是 BM，则说明该文件不
    // 是 BMP 文件，则报错。
    char headstr[2] = { '\0' };
    imgfile.seekg(0x0000, ios::beg);
    imgfile.read(headstr, 2);
    if (headstr[0] != 'B' || headstr[1] != 'M')
       return WRONG_FILE;

    // 读取文件中的 BPP 字段（每个像素占用的比特数量），如果 BPP 的值不为 8，说
    // 明该文件不是一个灰度 BMP 图像，则报错。
    unsigned short bpp = 0;
    imgfile.seekg(0x001C, ios::beg);
    imgfile.read(reinterpret_cast<char *>(&bpp), 2);
    if (bpp != 8)
        return WRONG_FILE;

    // 从文件中读取图像宽度和高度信息。
    unsigned int width = 0, height = 0;
    imgfile.seekg(0x0012, ios::beg);
    imgfile.read(reinterpret_cast<char *>(&width), 4);
    imgfile.read(reinterpret_cast<char *>(&height), 4);
    // 如果图像的尺寸不合法，则报错退出。
    if (width < 1 || height < 1)
        return WRONG_FILE;

    // 从文件中读取像素数据所在的文件中的偏移位置。
    unsigned int dataoff = 0;
    imgfile.seekg(0x000A, ios::beg);
    imgfile.read(reinterpret_cast<char *>(&dataoff), 4);
    
    // 如果图像是内存映射图像，则应先解除内存映射，以保证正确操作 Host 内存。
    if (outimgCud->mapSource != NULL) {
        int errcode;
        errcode = ImageBasicOp::unmapToHost(outimg);
        if (errcode < 0)
            return errcode;
    }
    
    // 获取存放图像像素数据的 Host 内存空间。本着尽量重用的思想，如果原来的图像
    // 内存数据是存储于 Host 内存，且尺寸和新的图像尺寸一致时，则不重新申请
    // Host 内存空间，直接利用原来的空间存放新的图像数据。
    unsigned char *imgdata = outimg->imgData;
    bool reusedata = true;
    if (outimg->imgData == NULL || outimgCud->deviceId >= 0 || 
        outimg->width != width || outimg->height != height) {
        // 为图像数据在 Host 上申请 pinned memory，
        // 并设置参数为使用内存映射功能。
        cudaError_t cuerrcode;
        cuerrcode = cudaHostAlloc((void**)&imgdata, width * height * 
                                  sizeof(unsigned char),cudaHostAllocMapped);
        if (cuerrcode != cudaSuccess) 
            return CUDA_ERROR;

        reusedata = false;
    }

    // 计算 BMP 文件中每行的 Padding 尺寸。在 BMP 文件中，每行的数据都需要保证
    // 4 字节对齐。如果某行的宽度不是 4 的整数倍（注意，灰度图中每行的像素个数
    // 同每行实际数据占用的字节数是相等的），则需要补充一些字节，使其达到 4 的
    // 整数倍。
    unsigned int dummybytes = (4 - (width & 3)) & 3;

    // 将文件指针移动到数据存储的开始位置
    imgfile.seekg(dataoff, ios::beg);

    // 由于 BMP 采用了右手坐标，即图像的左下角点为原点，整个图像位于第一象限，
    // 而我们系统内部使用的是左手坐标，即图像的左上角点为原点，整个图像亦位于第
    // 一象限。这样，BMP 文件中的第一行图像数据，其时是最后一行数据，因此指针初
    // 始指向图像的最后一行。
    unsigned char *pdata = imgdata + (height - 1) * width;

    // 读取图像中的各行的图像数据。
    for (int r = 0; r < height; r ++) {
        // 读取图像数据（每次读取一行的数据）
        imgfile.read(reinterpret_cast<char *>(pdata), width);
        // 舍弃掉每行结尾的填充字节
        if (dummybytes > 0)
            imgfile.seekg(dummybytes, ios::cur);
        // 由于 BMP 图像采用右手坐标，因此指针需要向前移动。
        pdata -= width;
    }

    // 到此为止，图像数据读取完毕，这是可以安全的释放掉图像原来的数据。一直拖到
    // 最后才释放原来的数据，正是为了防止一旦图像读取失败，不至于让系统进入一个
    // 混乱的状态，因为原来的数据还是处于一个可用的状态。
 
    if (reusedata == false && outimg->imgData != NULL) {
        if (outimgCud->deviceId < 0) {
            // 原来的数据存放于 Host 内存中则调用 cudaFreeHost 释放。
            cudaFreeHost(outimg->imgData); 
        } else {
            // 如果原来的数据存放于 Device 内存中，则首先调到对应的 Device，然
            // 后使用 cudaFree 释放掉内存。
            cudaSetDevice(outimgCud->deviceId);
            cudaFree(outimg->imgData);
            cudaSetDevice(curdevid);
        }
    }

    // 为图像赋值新的元数据。这里 ROI 被重置为整幅图像。
    outimg->width = width;
    outimg->height = height;
    outimg->roiX1 = 0;
    outimg->roiY1 = 0;
    outimg->roiX2 = width;
    outimg->roiY2 = height;
    outimg->imgData = imgdata;
    outimgCud->deviceId = -1;
    outimgCud->pitchBytes = width;
    outimgCud->mapSource = NULL;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：writeToFile（将图像写入文件）
__host__ int ImageBasicOp::writeToFile(const char *filepath, Image *inimg)
{
    // 检查文件路径和图像是否为 NULL。
    if (filepath == NULL || inimg == NULL)
        return NULL_POINTER;

    // 打开需要写入的文件。
    ofstream imgfile(filepath, ios::out | ios::binary);
    if (!imgfile) 
        return NO_FILE;

    // 根据输入参数的 Image 型指针，得到对应的 ImageCuda 型数据。
    ImageCuda *inimgCud = IMAGE_CUDA(inimg);

    // 将图片的数据拷贝回 Host 内存中，这样图片就可以被下面的代码所读取，然后将
    // 图像的数据写入到磁盘中。这里需要注意的是，安排图片的拷贝过程在文件打开之
    // 后是因为，如果一旦文件打开失败，则不会改变图像在内存中的存储状态，这可能
    // 会对后续处理更加有利。
    int errcode;
    errcode = ImageBasicOp::copyToHost(inimg);
    if (errcode < 0)
        return errcode;

    // 计算一些和 BMP 图像相关的参数：
    // 计算 BMP 文件中每行的 Padding 尺寸。在 BMP 文件中，每行的数据都需要保证
    // 4 字节对齐。如果某行的宽度不是 4 的整数倍（注意，灰度图中每行的像素个数
    // 同每行实际数据占用的字节数是相等的），则需要补充一些字节，使其达到 4 的
    // 整数倍。
    unsigned int dummybytes = (4 - (inimg->width & 3)) & 3;

    // 计算在磁盘上存储图片总共需要的字节数量，这个数量包括了上面提到的 Padding
    // 的尺寸。
    unsigned int datalen = inimg->height * (inimg->width + dummybytes);

    // 在存储到磁盘中后，像素数据实际的起始位置。因为 BMP 文件存在信息头，实际
    // 的像素数据是在这些信息头的后面的。对于系统中使用到的灰度图像来说，信息头
    // 包含了两个部分，最前面的是图像的元数据（如图像的宽度、高度；数据的尺寸等
    // 信息），紧随其后的是颜色表，颜色表共有 256 个条目，对应了 256 级灰度，每
    // 个条目包含了 4 个字节，这四个字节分别为 RGBA 四个通道的亮度值。
    unsigned int dataoff = 4 * 256 + 54;

    // 向文件中写入 BMP 头信息
    unsigned short ustemp;  // 这三个变量用来保存头信息中的临时域的值，三个变量
    unsigned int uitemp;    // 用来处理不同的数据类型。
    int sitemp;

    // 文件类型头
    ustemp = 0x4D42;
    imgfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // 文件长度
    uitemp = datalen + dataoff;
    imgfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 保留区段甲
    ustemp = 0;
    imgfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // 保留区段乙
    ustemp = 0;
    imgfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // 像素数据在文件中开始的位置
    uitemp = dataoff;
    imgfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 图像信息头尺寸
    uitemp = 40;
    imgfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 图像宽度
    sitemp = inimg->width;
    imgfile.write(reinterpret_cast<char *>(&sitemp), 4);
    // 图像高度
    sitemp = inimg->height;
    imgfile.write(reinterpret_cast<char *>(&sitemp), 4);
    // 图像层次数量
    ustemp = 1;
    imgfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // BPP（每像素的比特数量）
    ustemp = 8;
    imgfile.write(reinterpret_cast<char *>(&ustemp), 2);
    // 压缩算法
    uitemp = 0;
    imgfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 图像尺寸
    uitemp = datalen;
    imgfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // 每公尺的像素数量（X-方向）
    sitemp = 0;
    imgfile.write(reinterpret_cast<char *>(&sitemp), 4);
    // 每公尺的像素数量（Y-方向）
    sitemp = 0;
    imgfile.write(reinterpret_cast<char *>(&sitemp), 4);
    // ClrUsed
    uitemp = 256;
    imgfile.write(reinterpret_cast<char *>(&uitemp), 4);
    // ClrImportant
    uitemp = 0;
    imgfile.write(reinterpret_cast<char *>(&uitemp), 4);

    // 写入颜色表信息
    // 颜色信息共有 256 个条目，对应了 256 个灰度级；每个条目包含了 4 个颜色通
    // 道的数据。由于图像是灰度图像，因此对于灰度为 i 的对应的颜色值为 < i, i,
    // i, FF >。
    unsigned char coloritem[4] = { 0x00, 0x00, 0x00, 0xFF };
    for (int i = 0; i < 256; i++) {
        coloritem[0] = coloritem[1] = coloritem[2] = i;
        imgfile.write(reinterpret_cast<char *>(coloritem), 4);
    }

    // 写入图像像素数据
    char dummybuf[4] = { '\0' };  // 每行末尾的 Padding 的补白数据。

    // 由于 BMP 采用了右手坐标，即图像的左下角点为原点，整个图像位于第一象限，
    // 而我们系统内部使用的是左手坐标，即图像的左上角点为原点，整个图像亦位于第
    // 一象限。这样，BMP 文件中的第一行图像数据，其时是最后一行数据，因此指针初
    // 始指向图像的最后一行。
    unsigned char *pdata = inimg->imgData + (inimg->height - 1) * inimg->width;

    // 逐行写入图像的像素数据。
    for (int r = 0; r < inimg->height; r++) {
        // 写入某行的像素数据。
        imgfile.write(reinterpret_cast<char *>(pdata), inimg->width);
        // 写入为了 Padding 的补白数据。
        if (dummybytes > 0)
            imgfile.write(dummybuf, dummybytes);
        // 由于 BMP 图像采用右手坐标，因此指针需要向前移动。
        pdata -= inimgCud->pitchBytes;
    }

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：copyToCurrentDevice（将图像拷贝到当前 Device 内存上）
__host__ int ImageBasicOp::copyToCurrentDevice(Image *img)
{
    // 检查图像是否为 NULL。
    if (img == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Image 型指针，得到对应的 ImageCuda 型数据。
    ImageCuda *imgCud = IMAGE_CUDA(img);
    
    // 检查图像所在的地址空间是否合法，如果图像已经映射至 Device ，
    // 则不能使用 In-Place 式的 copy ，该函数报“数据溢出”错误，表示无法处理。
    if (imgCud->mapSource != NULL)
        return OP_OVERFLOW;
        
    // 检查图像所在的地址空间是否合法，如果图像所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (imgCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果图像是一个不包含数据的空图像，则报错。
    if (img->imgData == NULL || img->width == 0 || img->height == 0 || 
        imgCud->pitchBytes == 0) 
        return UNMATCH_IMG;

    // 对于不同的情况，将图像数据拷贝到当前设备上。
    if (imgCud->deviceId < 0) {
        // 如果图像的数据位于 Host 内存上，则需要在当前 Device 的内存空间上申请
        // 空间，然后将 Host 内存上的数据进行 Padding 后拷贝到当前 Device 上。
        unsigned char *devptr;  // 新的数据空间，在当前 Device 上。
        size_t pitch;           // Padding 后的每行尺寸
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 在当前设备上申请空间，使用 Pitch 版本的申请函数，用来进行 Padding。
        cuerrcode = cudaMallocPitch((void **)(&devptr), &pitch, 
                                    img->width * sizeof (unsigned char), 
                                    img->height);
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 进行 Padding 并拷贝数据到当前 Device 上。注意，这里 img->pitchBytes
        // == img->width。
        cuerrcode = cudaMemcpy2D(devptr, pitch, 
                                 img->imgData, imgCud->pitchBytes,
                                 img->width * sizeof (unsigned char), 
                                 img->height,
                                 cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }

        // 释放掉原来存储于 Host 内存上的图像数据。
        cudaFreeHost(img->imgData); 
        
        // 更新图像数据，把新的在当前 Device 上申请的数据和相关数据写入图像元数
        // 据中。
        img->imgData = devptr;
        imgCud->deviceId = curdevid;
        imgCud->pitchBytes = pitch;

        // 操作完毕，返回。
        return NO_ERROR;

    } else if (imgCud->deviceId != curdevid) {
        // 对于数据存在其他 Device 的情况，仍旧要在当前 Device 上申请数据空间，
        // 并从另一个 Device 上拷贝数据到新申请的当前 Device 的数据空间中。
        unsigned char *devptr;  // 新申请的当前 Device 上的数据。
        size_t datasize = imgCud->pitchBytes * img->height;  // 数据尺寸。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 在当前 Device 上申请空间。
        cuerrcode = cudaMalloc((void **)(&devptr), datasize);
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 将数据从图像原来的存储位置拷贝到当前的 Device 上。
        cuerrcode = cudaMemcpyPeer(devptr, curdevid, 
                                   img->imgData, imgCud->deviceId,
                                   datasize);
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }

        // 释放掉图像在原来的 Device 上的数据。
        cudaFree(img->imgData);

        // 将新的图像数据信息写入到图像元数据中。
        img->imgData = devptr;
        imgCud->deviceId = curdevid;

        // 操作完成，返回。
        return NO_ERROR;
    }

    // 对于其他情况，即图像数据本来就在当前 Device 上，则直接返回，不进行任何的
    // 操作。
    return NO_ERROR;
}

// Host 静态方法：copyToCurrentDevice（将图像拷贝到当前 Device 内存上）
__host__ int ImageBasicOp::copyToCurrentDevice(Image *srcimg, Image *dstimg)
{
    // 检查输入图像是否为 NULL。
    if (srcimg == NULL)
        return NULL_POINTER;

    // 如果输出图像为 NULL，或者输出图像和输入图像为同一各图像，则调用 In-place
    // 版本的函数。
    if (dstimg == NULL || dstimg == srcimg)
        return copyToCurrentDevice(srcimg);

    // 获取 srcimg 和 dstimg 对应的 ImageCuda 型指针。
    ImageCuda *srcimgCud = IMAGE_CUDA(srcimg);
    ImageCuda *dstimgCud = IMAGE_CUDA(dstimg);

    // 用来存放旧的数据，使得在拷贝操作失败时可以恢复为原来的可用的数据
    // 信息，防止系统进入一个混乱的状态。
    ImageCuda olddstimgCud = *dstimgCud;  // 旧的 dstimg 数据
    bool reusedata = true;                // 记录是否重用了原来的图像数据空间。
                                          // 该值为 ture，则原来的数据空间被重
                                          // 用，不需要在之后释放数据，否则需要
                                          // 在最后释放旧的空间。

    // 如果源图像是一个空图像，则不进行任何操作，直接报错。
    if (srcimg->imgData == NULL || srcimg->width == 0 || srcimg->height == 0 ||
        srcimgCud->pitchBytes == 0)
        return INVALID_DATA;
  
    // 检查图像所在的地址空间是否合法，如果图像所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srcimgCud->deviceId >= devcnt || dstimgCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果目标图像中存在有数据，则需要根据情况，若原来的数据不存储在当前的
    // Device 上，或者即使存储在当前的 Device 上，但数据尺寸不匹配，则需要释放
    // 掉原来申请的空间，以便重新申请合适的内存空间。此处不进行真正的释放操作，
    // 其目的在于当后续操作出现错误时，可以很快的恢复 dstimg 中原来的信息，使得
    // 整个系统不会处于一个混乱的状态，本函数会在最后，确定 dstimg 被成功的更换
    // 为了新的数据以后，才会真正的将原来的图像数据释放掉。
    if (dstimgCud->deviceId != curdevid) {
        // 对于数据存在 Host 或其他的 Device 上，则直接释放掉原来的数据空间。
        reusedata = 0;
        dstimg->imgData = NULL;
    } else if (!(((srcimgCud->deviceId < 0 && 
                   srcimg->width == dstimg->width) ||
                  dstimgCud->pitchBytes == srcimgCud->pitchBytes) &&
                 srcimg->height == dstimg->height)) {
        // 对于数据存在于当前 Device 上，则需要检查数据的尺寸是否和源图像相匹
        // 配。检查的标准包括：要求源图像的 Padding 后的行宽度和目标图像的相
        // 同，源图像和目标图像的高度相同；如果源图像是存储在 Host 内存中的，则
        // 仅要求源图像和目标图像的宽度相同即可。如果目标图像和源图像的尺寸不匹
        // 配则仍旧需要释放目标图像原来的数据空间。
        reusedata = 0;
        dstimg->imgData = NULL;
    }

    // 将目标图像的尺寸更改为源图像的尺寸。
    dstimg->width = srcimg->width;
    dstimg->height = srcimg->height;

    // 将目标图像的 ROI 更改为源图像的 ROI。
    dstimg->roiX1 = srcimg->roiX1;
    dstimg->roiY1 = srcimg->roiY1;
    dstimg->roiX2 = srcimg->roiX2;
    dstimg->roiY2 = srcimg->roiY2;

    // 更改目标图像的数据存储位置为当前 Device。
    dstimgCud->deviceId = curdevid;
    
    // 将 Device 端图像的映射地址设置为空。
    dstimgCud->mapSource = NULL;
    
    // 将图像数据从源图像中拷贝到目标图像中。
    if (srcimgCud->deviceId < 0) {
        // 如果源图像数据存储于 Host 内存，则使用 cudaMemcpy2D 进行 Padding 形
        // 式的拷贝。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 如果目标图像的 imgData == NULL，说明目标图像原本要么是一个空图像，要
        // 么目标图像原本的数据空间不合适，需要重新申请。这时，需要为目标图像重
        // 新在当前 Device 上申请一个合适的数据空间。
        if (dstimg->imgData == NULL) {
            cuerrcode = cudaMallocPitch((void **)(&dstimg->imgData), 
                                        &dstimgCud->pitchBytes,
                                        dstimg->width * sizeof (unsigned char), 
                                        dstimg->height);
            if (cuerrcode != cudaSuccess) {
                // 如果申请内存的操作失败，则在报错返回前需要将旧的图像数据
                // 恢复到图像中，以保证系统接下的操作不至于混乱。
                *dstimgCud = olddstimgCud;
                return CUDA_ERROR;
            }
        }

        // 使用 cudaMemcpy2D 进行 Padding 形式的拷贝。
        cuerrcode = cudaMemcpy2D(dstimg->imgData, dstimgCud->pitchBytes,
                                 srcimg->imgData, srcimgCud->pitchBytes,
                                 srcimg->width * sizeof (unsigned char), 
                                 srcimg->height,
                                 cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {   
            // 如果拷贝操作失败，则再报错退出前，需要将旧的图像数据恢复到原始
            // 图像中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
            // 防止内存泄漏。
            if (!reusedata)
                cudaFree(dstimg->imgData);
            *dstimgCud = olddstimgCud;
            return CUDA_ERROR;
        }
    } else {
        
        // 如果源图像数据存储于 Device 内存（无论是当前 Device 还是其他的 
        // Device），都是用端到端的拷贝。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。
        size_t datasize = srcimgCud->pitchBytes * srcimg->height;

        // 如果目标图像需要申请数据空间，则进行申请。
        if (dstimg->imgData == NULL) {
            cuerrcode = cudaMalloc((void **)(&dstimg->imgData), datasize);
            if (cuerrcode != cudaSuccess) {
                // 如果发生错误，则需要首先恢复旧的图像数据，之后报错。恢复旧的
                // 图像数据以防止系统进入混乱状态。
                *dstimgCud = olddstimgCud;
                return CUDA_ERROR;
            }
        }
        
        // 更新目标图像的 Padding 尺寸与源图像相同。注意，因为源图像也存储在
        // Device 上，在 Device 上的数据都是经过 Padding 的，又因为
        // cudaMemcpyPeer 方法没有提供 Pitch 版本接口，所以，我们这里直接借用源
        // 图像的 Padding 尺寸。
        dstimgCud->pitchBytes = srcimgCud->pitchBytes;

        // 使用 cudaMemcpyPeer 实现两个 Device （可以为同一个 Device）间的数据
        // 拷贝，将源图像在 Device 上的数据信息复制到目标图像中。
        cuerrcode = cudaMemcpyPeer(dstimg->imgData, curdevid,
                                   srcimg->imgData, srcimgCud->deviceId,
                                   datasize);
        if (cuerrcode != cudaSuccess) {
            // 如果拷贝操作失败，则在报错退出前，需要将旧的图像数据恢复到原始
            // 图像中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
            // 防止内存泄漏。
            if (!reusedata)
                cudaFree(dstimg->imgData);
            *dstimgCud = olddstimgCud;
            return CUDA_ERROR;
        }
    }

    // 到此步骤已经说明新的图像数据空间已经成功的申请并拷贝了新的数据，因此，旧
    // 的数据空间已毫无用处。本步骤就是释放掉旧的数据空间以防止内存泄漏。这里，
    // 作为拷贝的 olddstimgCud 是局部变量，因此相应的元数据会在本函数退出后自动
    // 释放，不用理会。

    // 释放旧的内存空间。
    if (olddstimgCud.mapSource != NULL){ 
        // 如果原始目标图像数据是内存映射图，应该无条件释放主机端数据
        cudaFreeHost(olddstimgCud.mapSource); 
    } else if (olddstimgCud.imgMeta.imgData != NULL) {
        if (olddstimgCud.deviceId < 0) {
            // 如果旧数据空间是 Host 内存上的，则需要无条件释放。
             cudaFreeHost(olddstimgCud.imgMeta.imgData); 
        } else if (olddstimgCud.deviceId != curdevid) {
            // 如果旧数据空间不是当前 Device 内存上的其他 Device 内存上的数据，
            // 则也需要无条件的释放。
            cudaSetDevice(olddstimgCud.deviceId);
            cudaFree(olddstimgCud.imgMeta.imgData);
            cudaSetDevice(curdevid);
        } else if (!reusedata) {
            // 如果旧数据就在当前的 Device 内存上，则对于 reusedata 未置位的情
            // 况进行释放，因为一旦置位，旧的数据空间就被用于承载新的数据，则不
            // 能释放。
            cudaFree(olddstimgCud.imgMeta.imgData);
        }
    }

    return NO_ERROR;
}

// Host 静态方法：copyToHost（将图像拷贝到 Host 内存上）
__host__ int ImageBasicOp::copyToHost(Image *img)
{
    // 检查图像是否为 NULL。
    if (img == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Image 型指针，得到对应的 ImageCuda 型数据。
    ImageCuda *imgCud = IMAGE_CUDA(img);

    // 检查图像所在的地址空间是否合法，如果图像所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (imgCud->deviceId >= devcnt)
        return OP_OVERFLOW;
        
    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果图像是一个不包含数据的空图像，则报错。
    if (img->imgData == NULL || img->width == 0 || img->height == 0 || 
        imgCud->pitchBytes == 0) 
        return UNMATCH_IMG;

    // 对于不同的情况，将图像数据拷贝到当前设备上。
    if (imgCud->deviceId < 0) {
        // 如果图像位于 Host 上，则不需要进行任何操作。
        return NO_ERROR;
    } else {
        // 如果图像数据为主机端内存映射，则通过解除映射的方式使图像回到 Host 。
        if (imgCud->mapSource != NULL)
            return ImageBasicOp::unmapToHost(img);
            
        // 如果图像的数据位于 Device 内存上，则需要在 Host 的内存空间上申请空
        // 间，然后将数据消除 Padding 后拷贝到 Host 上。
        unsigned char *hostptr;  // 新的数据空间，在 Host 上。
        cudaError_t cuerrcode;   // CUDA 调用返回的错误码。

        // 在 Host 上申请空间。
        cuerrcode = cudaHostAlloc((void**)&(hostptr), img->width *
                                  img->height * sizeof(unsigned char),
                                  cudaHostAllocMapped);
        if (cuerrcode != cudaSuccess) {
                return CUDA_ERROR;
        }

        // 将设备切换到数据所在的 Device 上。
        cudaSetDevice(imgCud->deviceId);

        // 消除 Padding 并拷贝数据
        cuerrcode = cudaMemcpy2D(hostptr, img->width, 
                                 img->imgData, imgCud->pitchBytes,
                                 img->width, img->height,
                                 cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
            // 后报错返回。
            cudaFreeHost(hostptr); 
            return CUDA_ERROR;
        }

        // 释放掉原来存储于 Device 内存上的图像数据。
        cudaFree(img->imgData);

        // 对 Device 内存的操作完毕，将设备切换回当前 Device。
        cudaSetDevice(curdevid);

        // 更新图像数据，把新的在当前 Device 上申请的数据和相关数据写入图像元数
        // 据中。
        img->imgData = hostptr;
        imgCud->deviceId = -1;
        imgCud->pitchBytes = img->width;

        // 操作完毕，返回。
        return NO_ERROR;
    }

    // 程序永远也不会到达这个分支，因此如果到达这个分支，则说明系统紊乱。对于多
    // 数编译器来说，会对此句报出不可达语句的 Warning，因此这里将其注释掉，以防
    // 止不必要的 Warning。
    //return UNKNOW_ERROR;
}

// Host 静态方法：copyToHost（将图像拷贝到 Host 内存上）
__host__ int ImageBasicOp::copyToHost(Image *srcimg, Image *dstimg)
{
    // 检查输入图像是否为 NULL。
    if (srcimg == NULL)
        return NULL_POINTER;

    // 如果输出图像为 NULL 或者和输入图像同为一个图像，则调用对应的 In-place 版
    // 本的函数。
    if (dstimg == NULL || dstimg == srcimg)
        return copyToHost(srcimg);

    // 获取 srcimg 和 dstimg 对应的 ImageCuda 型指针。
    ImageCuda *srcimgCud = IMAGE_CUDA(srcimg);
    ImageCuda *dstimgCud = IMAGE_CUDA(dstimg);
    
    // 用来存放旧的图像数据，使得在操作失败时可以恢复为原来的可用的数据
    // 信息，防止系统进入一个混乱的状态。
    ImageCuda olddstimgCud = *dstimgCud;  // 旧的 dstimg 数据
    bool reusedata = true;                // 记录是否重用了原来的图像数据空间。
                                          // 该值为 true，则原来的数据空间被重
                                          // 用，不需要在之后释放数据，否则需要
                                          // 释放旧的空间。

    // 如果源图像是一个空图像，则不进行任何操作，直接报错。
    if (srcimg->imgData == NULL || srcimg->width == 0 || srcimg->height == 0 ||
        srcimgCud->pitchBytes == 0)
        return INVALID_DATA;

    // 检查图像所在的地址空间是否合法，如果图像所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srcimgCud->deviceId >= devcnt || dstimgCud->deviceId >= devcnt)
        return OP_OVERFLOW;
     
    // 如果目标图像为内存映射图像，应该先解除内存映射，以便正确判断是否重用数据。
    if (dstimgCud->mapSource != NULL) {
        int errcode;
        errcode = ImageBasicOp::unmapToHost(dstimg);
        if (errcode < 0)
            return errcode;
    }
 
    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果目标图像中存在有数据，则需要根据情况，若原来的数据不存储在 Host 上，
    // 或者即使存储在 Host 上，但数据尺寸不匹配，则需要释放掉原来申请的空间，以
    // 便重新申请合适的内存空间。此处不进行真正的释放操作，其目的在于当后续操作
    // 出现错误时，可以很快的恢复 dstimg 中原来的信息，使得整个系统不会处于一个
    // 混乱的状态，本函数会在最后，确定 dstimg 被成功的更换为了新的数据以后，才
    // 会真正的将原来的图像数据释放掉。
    if (dstimgCud->deviceId >= 0) {
        // 对于数据存在于 Device 上，则亦直接释放掉原来的数据空间。
        reusedata = false;
        dstimg->imgData = NULL;
    } else if (!(srcimg->width == dstimg->width &&
                 srcimg->height == dstimg->height)) {
        // 对于数据存在于 Host 上，则需要检查数据的尺寸是否和源图像相匹配。检查
        // 的标准：源图像和目标图像的尺寸相同时，可重用原来的空间。
        reusedata = false;
        dstimg->imgData = NULL;  
    }

    // 将目标图像的尺寸更改为源图像的尺寸。
    dstimg->width = srcimg->width;
    dstimg->height = srcimg->height;

    // 将目标图像的 ROI 更改为源图像的 ROI。
    dstimg->roiX1 = srcimg->roiX1;
    dstimg->roiY1 = srcimg->roiY1;
    dstimg->roiX2 = srcimg->roiX2;
    dstimg->roiY2 = srcimg->roiY2;

    // 更改目标图像的数据存储位置为 Host。
    dstimgCud->deviceId = -1;

    // 由于 Host 内存上的数据不使用 Padding，因此设置 Padding 尺寸为图像的宽
    // 度。
    dstimgCud->pitchBytes = dstimg->width;

    // 如果目标图像的 imgData == NULL，说明目标图像原本要么是一个空图像，要么目
    // 标图像原本的数据空间不合适，需要重新申请。这时，需要为目标图像重新在 
    // Host 上申请一个合适的数据空间。
    if (dstimg->imgData == NULL) {
        cudaError_t cuerrcode;
        cuerrcode = cudaHostAlloc((void**)&(dstimg->imgData), srcimg->width * 
                                  srcimg->height * sizeof(unsigned char),
                                  cudaHostAllocMapped);
        if (cuerrcode != cudaSuccess) {
            // 如果申请内存的操作失败，则在报错返回前需要将旧的目标图像数据
            // 恢复到目标图像中，以保证系统接下的操作不至于混乱。
            *dstimgCud = olddstimgCud;
            return OUT_OF_MEM;
        }
    }

    // 将图像数据从源图像中拷贝到目标图像中。
    if (srcimgCud->deviceId < 0) {
        // 如果源图像数据存储于 Host 内存，则直接使用 C 标准支持库中的 memcpy
        // 完成拷贝。

        // 将 srcimg 内的图像数据拷贝到 dstimg 中。memcpy 不返回错误，因此，没
        // 有进行错误检查。
        memcpy(dstimg->imgData, srcimg->imgData, 
               srcimg->width * srcimg->height * sizeof (unsigned char));
    } else {
        // 如果源图像数据存储于 Device 内存（无论是当前 Device 还是其他的 
        // Device），都是 2D 形式的拷贝，并消除 Padding。
        cudaError_t cuerrcode;                     // CUDA 调用返回的错误码。

        // 首先切换到 srcimg 图像数据所在的 Device，以方便进行内存操作。
        cudaSetDevice(srcimgCud->deviceId);

        // 这里使用 cudaMemcpy2D 将 srcimg 中处于 Device 上的数据拷贝到 dstimg
        // 中位于 Host 的内存空间上面，该拷贝会同时消除 Padding。
        cuerrcode = cudaMemcpy2D(dstimg->imgData, dstimgCud->pitchBytes,
                                 srcimg->imgData, srcimgCud->pitchBytes,
                                 srcimg->width, srcimg->height,
                                 cudaMemcpyDeviceToHost);

        if (cuerrcode != cudaSuccess) {
            // 如果拷贝操作失败，则再报错退出前，需要将旧的目标图像数据恢复到目
            // 标图像中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
            // 防止内存泄漏。最后，还需要把 Device 切换回来，以免整个程序乱套。
            if (!reusedata)
                cudaFreeHost(dstimg->imgData);
            *dstimgCud = olddstimgCud;
            cudaSetDevice(curdevid);
            return CUDA_ERROR;
        }

        // 对内存操作完毕后，将设备切换回当前的 Device。
        cudaSetDevice(curdevid);
    }

    // 到此步骤已经说明新的图像数据空间已经成功的申请并拷贝了新的数据，因此，旧
    // 的数据空间已毫无用处。本步骤就是释放掉旧的数据空间以防止内存泄漏。这里，
    // 作为拷贝的 olddstimgCud 是局部变量，因此相应的元数据会在本函数退出后自动
    // 释放，不用理会。
    
    // 如果原始目标图像数据是内存映射图，应该先将数据指针恢复为主机端指针，
    // 然后进行释放操作，以保证可以正确使用释放方法。
    if (olddstimgCud.mapSource != NULL){
        // 将图像数据指针改回原始 Host 端指针。
        olddstimgCud.imgMeta.imgData = olddstimgCud.mapSource;
        // 图像所在位置改为 Host 。
        olddstimgCud.deviceId = -1;
    }

    // 释放旧的内存空间。
    if (olddstimgCud.imgMeta.imgData != NULL) {
        if (olddstimgCud.deviceId >= 0) {
            // 如果旧数据是存储于 Device 内存上的数据，则需要无条件的释放。
            cudaSetDevice(olddstimgCud.deviceId);
            cudaFree(olddstimgCud.imgMeta.imgData);
            cudaSetDevice(curdevid);
        } else if (!reusedata) {
            // 如果旧数据就在 Host 内存上，则对于 reusedata 未置位的情况进行释
            // 放，因为一旦置位，旧的数据空间就被用于承载新的数据，则不能释放。
            cudaFreeHost(olddstimgCud.imgMeta.imgData);
        }
    }

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 静态方法：mapToCurrentDevice（将图像映射到当前 Device 内存上）
__host__ int ImageBasicOp::mapToCurrentDevice(Image *img)
{ 
    // 局部变量，错误码。
    cudaError_t cuerrcode; 
    
    // 检查图像是否为 NULL。由于内存映射功能需要同时维护 Device 端和 Host 端
    // 两个指针，因此必须要两个图像分别位于 Device 端和 Host 上
    if (img == NULL)
        return NULL_POINTER;

    // 获取 srcimg 和 dstimg 对应的 ImageCuda 型指针。
    ImageCuda *imgCud = IMAGE_CUDA(img);
    
    // 如果图像已经进行了内存映射，则不用进行处理。
    if (imgCud->mapSource != NULL)
        return NO_ERROR;

    // 检查图像所在的地址空间是否合法，如果源图像所在地址空间不属于 Host 端，
    // 则该函数报“数据溢出”错误，表示无法处理。 
    if (imgCud->deviceId >= 0)
        return OP_OVERFLOW;
        
    // 如果源图像是一个空图像，则不进行任何操作，直接报错。
    if (img->imgData == NULL || img->width == 0 || img->height == 0 ||
        imgCud->pitchBytes == 0)
        return INVALID_DATA;
        
    // 新的 Device 端指针
    unsigned char *devptr;  
 
    // 根据源图像 host 端指针获取 device 端指针
    cuerrcode = cudaHostGetDevicePointer((void **)&devptr, 
                                         (void *)img->imgData, 0);
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;
            
    // 将映射源改为原始 Host 指针。
    imgCud->mapSource = img->imgData;
    
    // 图像数据使用设备端指针。
    img->imgData = devptr;
    
    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);
    // 图像所在位置改为当前 Device 。
    imgCud->deviceId = curdevid;
    
    // 操作完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：unmapToHost（解除当前图像的内存映射）
__host__ int ImageBasicOp::unmapToHost(Image *img)
{
    // 检查图像是否为 NULL。由于内存映射功能需要同时维护 Device 端和 Host 端
    // 两个指针，因此必须要两个图像分别位于 Device 端和 Host 上
    if (img == NULL)
        return NULL_POINTER;

    // 获取 srcimg 和 dstimg 对应的 ImageCuda 型指针。
    ImageCuda *imgCud = IMAGE_CUDA(img);   
    
    // 如果源图像是一个空图像，则不进行任何操作，直接报错。
    if (img->imgData == NULL || img->width == 0 || img->height == 0 ||
        imgCud->pitchBytes == 0)
        return INVALID_DATA;
        
    // 检查图像所在的地址空间，如果源图像所在地址空间属于 Host 端，
    // 则说明未进行内存映射，直接返回。 
    if (imgCud->deviceId < 0)
        return NO_ERROR;  
        
    // 如果图像位于 Device 且未进行过内存映射，则不能使用解除映射功能。       
    if (imgCud->mapSource == NULL)
        return OP_OVERFLOW;     
        
    // 将图像数据指针改回原始 Host 端指针。
    img->imgData = imgCud->mapSource;
    
    // 将映射源改为空。
    imgCud->mapSource = NULL;
    
    // 图像所在位置改为 Host 。
    imgCud->deviceId = -1;
    
    // 操作完毕，返回。
    return NO_ERROR;
}
