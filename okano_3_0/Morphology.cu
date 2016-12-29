// Morphology.cu
// 实现形态学图像算法，包括腐蚀，膨胀，开操作，闭操作

#include "Morphology.h"

#include <iostream>
using namespace std;

#include "ErrorCode.h"

// 宏：MOR_USE_INTERMEDIA
// 开关宏，如果使能该宏，则 CLASS 内部提供开运算和闭运算的中间变量，免除返回申
// 请释放的开销，但这样做会由于中间图像尺寸较大而是的 Cache 作用被削弱，因此，
// 未必会得到好性能。关闭该宏，则每次调用开闭运算，都会临时申请中间变量。
#define MOR_USE_INTERMEDIA

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：MOR_IMI_WIDTH 和 MOR_IMI_HEIGHT
// 定义了中间图像的尺寸。
#define MOR_IMI_WIDTH   4000
#define MOR_IMI_HEIGHT  4000

// static变量：_defTpl
// 当用户未定义有效的模板时，使用此默认模板，默认为 3 x 3
static Template *_defTpl = NULL;

// Host 函数：_initDefTemplate（初始化默认的模板指针）
// 函数初始化默认模板指针 _defTpl，如果原来模板不为空，则直接返回，否则初始化
// 为 3 x 3 的默认模板
static __host__ Template*     // 返回值：返回默认模板指针 _defTpl 
_initDefTemplate();

// Kernel 函数：_erosion（实现腐蚀算法操作）
static __global__ void     // Kernel 函数无返回值
_erosion(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 输出图像
        Template tpl       // 模板
);

// Kernel 函数：_dilation（实现膨胀算法操作）
static __global__ void     // Kernel 函数无返回值
_dilation(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 输出图像
        Template tpl       // 模板
); 

// Host 函数：_preOp（在算法操作前进行预处理）
// 在进行腐蚀，膨胀操作前，先进行预处理，包括：（1）对输入和输出图像
// 进行数据准备，包括申请当前Device存储空间；（2）对模板进行处理，包
// 申请当前Device存储空间
static __host__ int      // 返回值：函数是否正确执行，若正确执行，返回
                         // NO_ERROR 
_preOp(
        Image *inimg,    // 输入图像
        Image *outimg,   // 输出图像
        Template *tp     // 模板
);

// Host 函数：_adjustRoiSize（调整 ROI 子图的大小）
// 调整 ROI 子图的大小，使输入和输出的子图大小统一
static __host__ void       // 无返回值
_adjustRoiSize(
        ImageCuda *inimg,  // 输入图像
        ImageCuda *outimg  // 输出图像
);

// Host 函数：_getBlockSize（获取 Block 和 Grid 的尺寸）
// 根据默认的 Block 尺寸，使用最普通的线程划分方法获取 Grid 的尺寸
static __host__ int      // 返回值：函数是否正确执行，若正确执行，返回
                         // NO_ERROR 
_getBlockSize(
        int width,       // 需要处理的宽度
        int height,      // 需要处理的高度
        dim3 *gridsize,  // 计算获得的 Grid 的尺寸
        dim3 *blocksize  // 计算获得的 Block 的尺寸
); 

// Host 函数：_initDefTemplate（初始化默认的模板指针）
static __host__ Template* _initDefTemplate()
{
    // 如果 _defTpl 不为空，说明已经初始化了，则直接返回
    if (_defTpl != NULL)
        return _defTpl;

    // 如果 _defTpl 为空，则初始化为 3 x 3 的模板
    TemplateBasicOp::newTemplate(&_defTpl);
    TemplateBasicOp::makeAtHost(_defTpl, 9);
    // 分别处理每一个点
    for (int i = 0; i < 9; i++) {
        // 分别计算每一个点的横坐标和纵坐标
        _defTpl->tplData[2 * i] = i % 3 - 1;
        _defTpl->tplData[2 * i + 1] = i / 3 - 1;
    }
    return _defTpl;
}

// Kernel 函数：_erosion（实现腐蚀算法操作）
static __global__ void _erosion(ImageCuda inimg, ImageCuda outimg, Template tpl)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，
    // c 表示 column， r 表示 row）。由于采用并行度缩减策略 ，令一个线程
    // 处理 4 个输出像素，这四个像素位于同一行的相邻 4 行上，因此，对于
    // dstr 需要进行乘 4 的计算
    int dstc = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y)/* * 1*/;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃
    if (dstc >= inimg.imgMeta.width || dstr >= inimg.imgMeta.height)
        return;

    // 用来保存临时像素点的坐标的 x 和 y 分量
    int dx, dy; 

    // 用来记录当前模板所在的位置的指针
    int *curtplptr = tpl.tplData;

    // 用来记录当前输入图像所在位置的指针
    unsigned char *curinptr;

    // 用来保存输出像素在模板范围内的最小像素的值，由于采用并行度缩减策略，一
    // 个像素处理 4 个输出像素，所以这里定义一个大小为 4 的数组，因为是存放
    // 最小值，所以先初始化为最大值 0xff
    union {
        unsigned char pixel[4];
        unsigned int data;
    } _min;
    _min.pixel[0] = _min.pixel[1] = _min.pixel[2] = _min.pixel[3] = 0xff;

    // 存放临时像素点的像素值
    unsigned char pixel;

    // 扫描模板范围内的每个输入图像的像素点
    for (int i = 0; i < tpl.count; i++) {
        // 计算当前模板位置所在像素的 x 和 y 分量，模板使用相邻的两个下标的
        // 数组表示一个点，所以使当前模板位置的指针作加一操作 
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);

        // 先判断当前像素的 y 分量是否越界，如果越界，则跳过，扫描下一个模板点
        // 如果没有越界，则分别处理当前行的相邻的 4 个像素
        if (dy >= 0 && dy < inimg.imgMeta.height) {
            // 根据 dx 和 dy 获取第一个像素的位置
            curinptr = inimg.imgMeta.imgData + dx + dy * inimg.pitchBytes;
            // 检测此像素的 x 分量是否越界
            if (dx >= 0 && dx < inimg.imgMeta.width) {
                // 和 min[0] 比较，如果比 min[0] 小，则将当前像素赋值给 min[0]
                pixel = *curinptr;
                pixel < _min.pixel[0] ? (_min.pixel[0] = pixel) : '\0';
            }

            // 处理当前行的剩下的 3 个像素
            for (int j = 1; j < 4; j++) {
                // 获取当前列的下一列的像素的位置
                curinptr++;
                // 使 dx 加一，得到当前要处理的像素的 x 分量
                dx++;
                // 检测 dy 是否越界
                if (dx >= 0 && dx < inimg.imgMeta.width) {
                    // 和 min[j] 比较，如果比 min[j] 小，则将当前像素赋值给 
                    // min[j]
                    pixel = *curinptr;
                    pixel < _min.pixel[j] ? (_min.pixel[j] = pixel) : '\0';
                }
            }
        }
    }
    
    // 定义输出图像位置的指针
    unsigned int *outptr;

    // 获取对应的第一个输出图像的位置
    outptr = (unsigned int *)(outimg.imgMeta.imgData + 
                              dstr * outimg.pitchBytes + dstc);
    // 将计算得到的 min[0] 赋值给输出图像
    *outptr = _min.data;
}

// 成员方法：erode
__host__ int Morphology::erode(Image *inimg, Image *outimg)
{
    int errcode;        // 局部变量，错误码
    dim3 gridsize;
    dim3 blocksize;
    
    // 检查输入图像，输出图像，以及模板是否为空
    if (inimg == NULL || outimg == NULL || tpl == NULL)
        return NULL_POINTER;

    // 对输入图像，输出图像和模板进行预处理
    errcode = _preOp(inimg, outimg, tpl);
    if (errcode != NO_ERROR)
        return errcode; 

    // 提取输入图像的 ROI 子图像
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的 ROI 子图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 调整输入和输出图像的 ROI 子图，使大小统一
    _adjustRoiSize(&insubimgCud, &outsubimgCud);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    errcode = _getBlockSize(outsubimgCud.imgMeta.width,
                           outsubimgCud.imgMeta.height,
                           &gridsize, &blocksize);
    if (errcode != NO_ERROR) 
        return errcode;

    // 调用 Kernel 函数进行腐蚀操作
    _erosion<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud, *tpl); 

    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return NO_ERROR;
}

// Kernel 函数：_dilation（实现膨胀算法操作）
static __global__ void _dilation(ImageCuda inimg, ImageCuda outimg, 
                                Template tpl)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，
    // c 表示 column， r 表示 row）。由于采用并行度缩减策略 ，令一个线程
    // 处理 4 个输出像素，这四个像素位于同一行的相邻 4 行上，因此，对于
    // dstr 需要进行乘 4 的计算
    int dstc = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y)/* * 1*/;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃
    if (dstc >= inimg.imgMeta.width || dstr >= inimg.imgMeta.height)
        return;

    // 用来保存临时像素点的坐标的 x 和 y 分量
    int dx, dy; 

    // 用来记录当前模板所在的位置的指针
    int *curtplptr = tpl.tplData;

    // 用来记录当前输入图像所在位置的指针
    unsigned char *curinptr;

    // 用来保存输出像素在模板范围内的最大像素的值，由于采用并行度缩减策略，一
    // 个像素处理 4 个输出像素，所以这里定义一个大小为 4 的数组，因为是存放
    // 最大值，所以先初始化为最小值 0x00 
    union {
        unsigned char pixel[4];
        unsigned int data;
    } _max;
    _max.pixel[0] = _max.pixel[1] = _max.pixel[2] = _max.pixel[3] = 0x00;

    // 存放临时像素点的像素值
    unsigned char pixel;

    // 扫描模板范围内的每个输入图像的像素点
    for (int i = 0; i < tpl.count; i++) {
        // 计算当前模板位置所在像素的 x 和 y 分量，模板使用相邻的两个下标的
        // 数组表示一个点，所以使当前模板位置的指针作加一操作 
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);

        // 先判断当前像素的 y 分量是否越界，如果越界，则跳过，扫描下一个模板点
        // 如果没有越界，则分别处理当前行的相邻的 4 个像素
        if (dy >= 0 && dy < inimg.imgMeta.height) {
            // 根据 dx 和 dy 获取第一个像素的位置
            curinptr = inimg.imgMeta.imgData + dx + dy * inimg.pitchBytes;
            // 检测此像素的 y 分量是否越界
            if (dx >= 0 && dx < inimg.imgMeta.width) {
                // 和 max[0] 比较，如果比 max[0] 大，则将当前像素赋值给 max[0]
                pixel = *curinptr;
                pixel > _max.pixel[0] ? (_max.pixel[0] = pixel) : '\0';
            }

            // 处理当前列的剩下的 3 个像素
            for (int j = 1; j < 4; j++) {
                // 获取当前列的下一列的像素的位置
                curinptr++;
                // 使 dx 加一，得到当前要处理的像素的 x 分量
                dx++;
                // 检测 dy 是否越界
                if (dx >= 0 && dx < inimg.imgMeta.width) {
                    // 和 max[j] 比较，如果比 max[j] 大，则将当前像素值赋值给
                    // max[j]
                    pixel = *curinptr;
                    pixel > _max.pixel[j] ? (_max.pixel[j] = pixel) : '\0';
                }
            }
        }
    }
    
    // 定义输出图像位置的指针
    unsigned int *outptr;

    // 获取对应的第一个输出图像的位置
    outptr = (unsigned int *)(outimg.imgMeta.imgData + 
                              dstr * outimg.pitchBytes + dstc);
    // 将计算得到的 max[0] 赋值给输出图像
    *outptr = _max.data;
}

// 成员方法：dilate
__host__ int Morphology::dilate(Image *inimg, Image *outimg)
{
    int errcode;        // 局部变量，错误码
    dim3 gridsize;
    dim3 blocksize;
    
    // 检查输入图像，输出图像，以及模板是否为空
    if (inimg == NULL || outimg == NULL || tpl == NULL)
        return NULL_POINTER;

    // 对输入图像，输出图像和模板进行预处理
    errcode = _preOp(inimg, outimg, tpl);
    if (errcode != NO_ERROR)
        return errcode; 

    // 提取输入图像的 ROI 子图像
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的 ROI 子图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 调整输入和输出图像的 ROI 子图，使大小统一
    _adjustRoiSize(&insubimgCud, &outsubimgCud);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    errcode = _getBlockSize(outsubimgCud.imgMeta.width,
                           outsubimgCud.imgMeta.height,
                           &gridsize, &blocksize);
    if (errcode != NO_ERROR) 
        return errcode;

    // 调用 Kernel 函数进行膨胀操作
    _dilation<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud, *tpl); 

    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return NO_ERROR;
}

// Host 函数：_preOp（在算法操作前进行预处理）
static __host__ int _preOp(Image *inimg, Image *outimg, Template *tp)
{
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝到 Device 内存中
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 计算 roi 子图的宽和高
        int roiwidth = inimg->roiX2 - inimg->roiX1; 
        int roiheight = inimg->roiY2 - inimg->roiY1;
        // 如果输出图像无数据，则会创建一个和输出图像子图像尺寸相同的图像
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, roiwidth, 
                                                   roiheight); 
        // 如果创建图像依然操作失败，则返回错误
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 将模板拷贝到 Device 内存中
    errcode = TemplateBasicOp::copyToCurrentDevice(tp);
    if (errcode != NO_ERROR)
        return errcode;   

    return NO_ERROR;
}

// Host 函数：_adjustRoiSize（调整输入和输出图像的 ROI 的大小）
inline static __host__ void _adjustRoiSize(ImageCuda *inimg, ImageCuda *outimg)
{
    if (inimg->imgMeta.width > outimg->imgMeta.width)
        inimg->imgMeta.width = outimg->imgMeta.width;
    else
        outimg->imgMeta.width = inimg->imgMeta.width;

    if (inimg->imgMeta.height > outimg->imgMeta.height)
        inimg->imgMeta.height = outimg->imgMeta.height;
    else
        outimg->imgMeta.height = inimg->imgMeta.height;
}

// Host 函数：_getBlockSize（获取 Block 和 Grid 的尺寸）
inline static __host__ int _getBlockSize(int width, int height, dim3 *gridsize,
                                        dim3 *blocksize)
{
    // 检测 girdsize 和 blocksize 是否是空指针
    if (gridsize == NULL || blocksize == NULL)
        return NULL_POINTER; 

    // blocksize 使用默认的尺寸
    blocksize->x = DEF_BLOCK_X;
    blocksize->y = DEF_BLOCK_Y;

    // 使用最普通的方法划分 Grid 
    gridsize->x = (width + blocksize->x * 4 - 1) / (blocksize->x * 4);
    gridsize->y = (height + blocksize->y * 1 - 1) / (blocksize->y * 1);

    return NO_ERROR;
}

// 构造函数：Morphology
__host__ Morphology::Morphology(Template *tp)
{
    // 设置类成员中的模板参数。
    setTemplate(tp);

#ifdef MOR_USE_INTERMEDIA
    // 初始化中间图像。如果中间图像没有申请成功，则置为 NULL。
    int errcode;
    errcode = ImageBasicOp::newImage(&intermedImg);
    if (errcode != NO_ERROR) {
        intermedImg = NULL;
        return;
    }

    // 为中间图像申请内存空间。如果中间图像没有申请成功，则置为 NULL。
    errcode = ImageBasicOp::makeAtCurrentDevice(
            intermedImg, MOR_IMI_WIDTH, MOR_IMI_HEIGHT);
    if (errcode != NO_ERROR) {
        ImageBasicOp::deleteImage(intermedImg);
        intermedImg = NULL;
        return;
    }
#else
    intermedImg = NULL;
#endif
}

// 析构函数：~Morphology
__host__ Morphology::~Morphology()
{
    // 如果中间图像已经申请，则需要释放掉中间图像。
    if (intermedImg != NULL)
        ImageBasicOp::deleteImage(intermedImg);
}

// 成员方法：getTemplate
__host__ Template* Morphology::getTemplate() const
{
    // 如果模板指针和默认模板指针相同，则返回空
    if (this->tpl == _defTpl) 
        return NULL;

    // 否则返回设置的模板指针
    return this->tpl;
}

// 成员方法：setTemplate
__host__ int Morphology::setTemplate(Template *tp)
{
    // 如果 tp 为空，则只用默认的模板指针
    if (tp == NULL) {
        this->tpl = _initDefTemplate();
    }
    // 否则将 tp 赋值给 tpl
    else {
        this->tpl = tp;
    }
    return NO_ERROR;
}

// 成员方法：open（开运算）
__host__ int Morphology::open(Image *inimg, Image *outimg)
{
    int errcode;           // 局部变量，错误码
    Image *imiimg;         // 局部变量，用来存储腐蚀操作的返回结果，再对此图像
                           // 进行膨胀操作
    ImageCuda *imiimgCud;  // 只在使用 CLASS 提供的中间图像时使用
    size_t pitchold;       // 只在使用 CLASS 提供的中间图像时使用，用于记录原始
                           // 原始参数。

    // 检查输入图像，输出图像，以及模板是否为空
    if (inimg == NULL || outimg == NULL || tpl == NULL)
        return NULL_POINTER;

    // 若无法使用 CLASS 自身提供的中间图像，则需要自行申请中间图像，这样的图像
    // 在处理完毕后需要释放掉，因此通过一个 bool 变量标识目前使用的是否是一个临
    // 时申请的中间图像。
    bool useprivimiimg = false;

    if (intermedImg == NULL) {
        // 如果 CLASS 的中间图像为 NULL，则说明其在 CLASS 构造时没有成功申请空
        // 间，因此只能使用临时申请的中间图像。
        useprivimiimg = true;
        errcode = ImageBasicOp::newImage(&imiimg);
        if (errcode != NO_ERROR)
            return errcode;
    } else {
        // 计算当前计算所需要的中间图像尺寸，即两个图像 ROI 区域取较小者。
        int roiw1 = inimg->roiX2 - inimg->roiX1;
        int roih1 = inimg->roiY2 - inimg->roiY1;
        int roiw2 = outimg->roiX2 - outimg->roiX1;
        int roih2 = outimg->roiY2 - outimg->roiY1;
        if (roiw2 == 0) roiw2 = roiw1;
        if (roih2 == 0) roih2 = roih1;
        int roiw = (roiw1 <= roiw2) ? roiw1 : roiw2;
        int roih = (roih1 <= roih2) ? roih1 : roih2;

        // 根据当前计算所需要的中间图像尺寸，来决定是否使用 CLASS 提供的中间图
        // 像。
        if (roiw <= intermedImg->width && roih <= intermedImg->height) {
            // 如果 CLASS 提供的图像图像可以满足尺寸要求，则直接使用 CLASS 提供
            // 的中间图像。
            useprivimiimg = false;

            // 这里需要调整一下 CLASS 提供图像的 ROI 尺寸，以防止上次计算对 ROI
            // 尺寸的影响，而得到不正确的图像结果。考虑到 cache 问题，由于较大的
            // 中间图像会导致 cache 局部性的问题，这里我们强行更改了 pitch 以保
            // 证具有更好的局部性。
            // 首先，将 CLASS 提供的中间图像的元数据复制出来一份。
            imiimg = intermedImg;
            imiimgCud = IMAGE_CUDA(imiimg);
            pitchold = imiimgCud->pitchBytes;

            // 然后强行修改 CLASS 的尺寸参数（由于判断了尺寸的合适性，因此，修
            // 改参数的操作是安全的。
            imiimg->width = roiw;
            imiimg->height = roih;
            imiimgCud->pitchBytes = roiw;
            imiimg->roiX1 = imiimg->roiY1 = 0;
            imiimg->roiX2 = roiw;
            imiimg->roiY2 = roih;
        } else {
            // 如果尺寸不满足要求，只能使用临时申请的中间图像。
            useprivimiimg = true;
            errcode = ImageBasicOp::newImage(&imiimg);
            if (errcode != NO_ERROR)
                return errcode;
        }
    }

    do {
        // 先对输入图像进行腐蚀操作，结果临时存在中间图像中
        errcode = erode(inimg, imiimg);
        if (errcode != NO_ERROR)
            break;

        // 再将腐蚀操作得到的中间图像进行膨胀操作，结果放在 outimg 中
        errcode = dilate(imiimg, outimg);
        if (errcode != NO_ERROR)
            break;
    } while (0);

    // 释放中间图像的资源
    if (useprivimiimg) {
        ImageBasicOp::deleteImage(imiimg);
    } else {
        // 还原 CLASS 中间图像回原来的参数。
        imiimg->width = MOR_IMI_WIDTH;
        imiimg->height = MOR_IMI_HEIGHT;
        imiimgCud->pitchBytes = pitchold;
        imiimg->roiX1 = imiimg->roiY1 = 0;
        imiimg->roiX2 = MOR_IMI_WIDTH;
        imiimg->roiY2 = MOR_IMI_HEIGHT;
    }

    return errcode;
} 

// 成员方法：close
__host__ int Morphology::close(Image *inimg, Image *outimg)
{
    int errcode;           // 局部变量，错误码
    Image *imiimg;         // 局部变量，用来存储腐蚀操作的返回结果，再对此图像
                           // 进行膨胀操作
    ImageCuda *imiimgCud;  // 只在使用 CLASS 提供的中间图像时使用
    size_t pitchold;       // 只在使用 CLASS 提供的中间图像时使用，用于记录原始
                           // 原始参数。

    // 检查输入图像，输出图像，以及模板是否为空
    if (inimg == NULL || outimg == NULL || tpl == NULL)
        return NULL_POINTER;

    // 若无法使用 CLASS 自身提供的中间图像，则需要自行申请中间图像，这样的图像
    // 在处理完毕后需要释放掉，因此通过一个 bool 变量标识目前使用的是否是一个临
    // 时申请的中间图像。
    bool useprivimiimg = false;

    if (intermedImg == NULL) {
        // 如果 CLASS 的中间图像为 NULL，则说明其在 CLASS 构造时没有成功申请空
        // 间，因此只能使用临时申请的中间图像。
        useprivimiimg = true;
        errcode = ImageBasicOp::newImage(&imiimg);
        if (errcode != NO_ERROR)
            return errcode;
    } else {
        // 计算当前计算所需要的中间图像尺寸，即两个图像 ROI 区域取较小者。
        int roiw1 = inimg->roiX2 - inimg->roiX1;
        int roih1 = inimg->roiY2 - inimg->roiY1;
        int roiw2 = outimg->roiX2 - outimg->roiX1;
        int roih2 = outimg->roiY2 - outimg->roiY1;
        if (roiw2 == 0) roiw2 = roiw1;
        if (roih2 == 0) roih2 = roih1;
        int roiw = (roiw1 <= roiw2) ? roiw1 : roiw2;
        int roih = (roih1 <= roih2) ? roih1 : roih2;

        // 根据当前计算所需要的中间图像尺寸，来决定是否使用 CLASS 提供的中间图
        // 像。
        if (roiw <= intermedImg->width && roih <= intermedImg->height) {
            // 如果 CLASS 提供的图像图像可以满足尺寸要求，则直接使用 CLASS 提供
            // 的中间图像。
            useprivimiimg = false;

            // 这里需要调整一下 CLASS 提供图像的 ROI 尺寸，以防止上次计算对 ROI
            // 尺寸的影响，而得到不正确的图像结果。考虑到 cache 问题，由于较大的
            // 中间图像会导致 cache 局部性的问题，这里我们强行更改了 pitch 以保
            // 证具有更好的局部性。
            // 首先，将 CLASS 提供的中间图像的元数据复制出来一份。
            imiimg = intermedImg;
            imiimgCud = IMAGE_CUDA(imiimg);
            pitchold = imiimgCud->pitchBytes;

            // 然后强行修改 CLASS 的尺寸参数（由于判断了尺寸的合适性，因此，修
            // 改参数的操作是安全的。
            imiimg->width = roiw;
            imiimg->height = roih;
            imiimgCud->pitchBytes = roiw;
            imiimg->roiX1 = imiimg->roiY1 = 0;
            imiimg->roiX2 = roiw;
            imiimg->roiY2 = roih;
        } else {
            // 如果尺寸不满足要求，只能使用临时申请的中间图像。
            useprivimiimg = true;
            errcode = ImageBasicOp::newImage(&imiimg);
            if (errcode != NO_ERROR)
                return errcode;
        }
    }

    do {
        // 先对输入图像进行膨胀操作，结果临时存在中间图像中
        errcode = dilate(inimg, imiimg);
        if (errcode != NO_ERROR)
            break;

        // 再将腐蚀操作得到的中间图像进行膨胀操作，结果放在 outimg 中
        errcode = erode(imiimg, outimg);
        if (errcode != NO_ERROR)
            break;
    } while (0);

    // 释放中间图像的资源
    if (useprivimiimg) {
        ImageBasicOp::deleteImage(imiimg);
    } else {
        // 还原 CLASS 中间图像回原来的参数。
        imiimg->width = MOR_IMI_WIDTH;
        imiimg->height = MOR_IMI_HEIGHT;
        imiimgCud->pitchBytes = pitchold;
        imiimg->roiX1 = imiimg->roiY1 = 0;
        imiimg->roiX2 = MOR_IMI_WIDTH;
        imiimg->roiY2 = MOR_IMI_HEIGHT;
    }

    return errcode;
}
 
