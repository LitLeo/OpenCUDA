// LinearFilter.cu
// 实现图像的线性滤波

#include <iostream>
#include "LinearFilter.h"
#include "ErrorCode.h"
#include "ImageDiff.h"
#include "Common.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8
#define DEF_TEMPLATE_COUNT 9

// static变量：_defTpl
// 当用户未定义有效的模板时，使用此默认模板，默认为3x3，默认模板值为1
static TemplateCuda *_defTpl = NULL;

__constant__ static int devptr[DEF_TEMPLATE_COUNT*2];
__constant__ static float attachptr[DEF_TEMPLATE_COUNT];

// Host 函数：_initDefTemplate（初始化默认的模板指针）
// 函数初始化默认模板指针 _defTpl，如果原来模板不为空，则直接返回，否则初始化
// 为3x3的默认模板
static __host__ TemplateCuda *  // 返回值：返回默认模板指针 _defTpl 
_initDefTemplate();

static __host__ int 
_copyDefTemplateToConstantMem(
    Template *tpl
);

// Kernel 函数：_linearFilterKer（实现线性滤波操作）
 static __global__ void    // Kernel 函数无返回值
_linearFilterKer(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 输出图像
        TemplateCuda tpl,  // 模板
        int imptype        // 滤波操作的实现方式
);

// Host 函数：_preOp（在算法操作前进行预处理）
// 在滤波操作前，先进行预处理，包括：（1）对输入和输出图像
// 进行数据准备，包括申请当前Device存储空间；（2）对模板进行处理，包
// 申请当前Device存储空间
static __host__ int     // 返回值：函数是否正确执行，若正确执行，返回
                        // NO_ERROR 
_preOp(
        Image *inimg,   // 输入图像
        Image *outimg,  // 输出图像 
        Template *tp    // 模板
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
static __host__ TemplateCuda *_initDefTemplate()
{
    // 如果 _defTpl 不为空，说明已经初始化了，则直接返回
    if (_defTpl != NULL)
        return _defTpl;

    // 如果 _defTpl 为空，则初始化为大小为3x3，模板值为1的模板
    Template *tmpdef;
    TemplateBasicOp::newTemplate(&tmpdef);
    TemplateBasicOp::makeAtHost(tmpdef, DEF_TEMPLATE_COUNT);
    printf("%s\n", );
    _defTpl = TEMPLATE_CUDA(tmpdef);
    // 分别处理每一个点
    for (int i = 0; i < DEF_TEMPLATE_COUNT; i++) {
        // 分别计算每一个点的横坐标和纵坐标
        _defTpl->tplMeta.tplData[2 * i] = i % 3 - 1;
        _defTpl->tplMeta.tplData[2 * i + 1] = i / 3 - 1;
        // 将每个点的模板值设为1
        _defTpl->attachedData[i] = 1;
    }
    return _defTpl;
}

static __host__ int _copyDefTemplateToConstantMem(Template *tpl)
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
        // 如果模板的数据位于 Host 内存上，则需要将 Host 内存上的数据拷贝到常量
        // 内存上。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 将原来存储在 Host 上坐标数据拷贝到常量内存上。
        cuerrcode = cudaMemcpyToSymbol(devptr, tpl->tplData, 
                                       tpl->count * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            cudaFree(attachptr);
            return CUDA_ERROR;
        }

        // 将原来存储在 Host 上附属数据拷贝到常量内存上。
        cuerrcode = cudaMemcpyToSymbol(attachptr, tplCud->attachedData,
                                       tpl->count * sizeof (float));
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
    }
    else
        return CUDA_ERROR;
}

// Kernel 函数：_linearFilterKer（实现滤波算法操作）     
static __global__ void _linearFilterKer(ImageCuda inimg, ImageCuda outimg, 
                                        TemplateCuda tpl, int imptype)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，
    // c 表示 column， r 表示 row）。由于采用并行度缩减策略 ，令一个线程
    // 处理 4 个输出像素，这四个像素位于统一列的相邻 4 行上，因此，对于
    // dstr 需要进行乘 4 的计算
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
                         
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃
    if (dstc >= inimg.imgMeta.width || dstr >= inimg.imgMeta.height)
        return;

    // 用来保存临时像素点的坐标的 x 和 y 分量
    int dx, dy; 

    // 用来记录当前模板所在的位置的指针
    int *curtplptr = tpl.tplMeta.tplData;

    // 用来记录当前输入图像所在位置的指针
    unsigned char *curinptr;

    // 用来存放模板中像素点的像素值加和
    unsigned int tplsum[4] = { 0, 0, 0, 0 };
    
    // 用来记录当前滤波操作的除数
    float tmpcount[4] = { 0, 0, 0, 0 };
    
    // 扫描模板范围内的每个输入图像的像素点
    for (int i = 0; i < tpl.tplMeta.count; i++) {
        // 计算当前模板位置所在像素的 x 和 y 分量，模板使用相邻的两个下标的
        // 数组表示一个点，所以使当前模板位置的指针作加一操作 
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);
        
        // 先判断当前像素的 x 分量是否越界，如果越界，则跳过，扫描下一个模板点
        // 如果没有越界，则分别处理当前列的相邻的 4 个像素
        if (dx >= 0 && dx < inimg.imgMeta.width) { 
            // 根据 dx 和 dy 获取第一个像素的位置
            curinptr = inimg.imgMeta.imgData + dx + dy * inimg.pitchBytes;
            
            // 检测此像素的 y 分量是否越界    
            if (dy >= 0 && dy < inimg.imgMeta.height) {     
                // 将第一个像素点邻域内点的像素值累加	 
                tplsum[0] += (*curinptr) * (tpl.attachedData[i]);
                
                // 针对不同的实现类型，选择不同的路径进行处理
                switch(imptype)
                {
                // 使用邻域像素总和除以像素点个数的运算方法实现线性滤波
                case LNFT_COUNT_DIV:
                    // 记录当前像素点邻域内已累加点的个数
                    tmpcount[0] += 1;
                    break;
                        
                 // 使用邻域像素总和除以像素点权重之和的运算方法实现线性滤波    
                case LNFT_WEIGHT_DIV:
                    // 记录当前像素点权重之和
                    tmpcount[0] += tpl.attachedData[i];
                    break;
                       
                // 使用邻域像素直接带权加和的运算方法实现线性滤波    
                case LNFT_NO_DIV:
                    // 设置除数为 1
                    tmpcount[0] = 1;
                    break;
                }
            }
        
            // 处理当前列的剩下的 3 个像素
            for (int j = 1; j < 4; j++) {
                // 获取当前像素点的位置
                curinptr += inimg.pitchBytes;
            
                // 使 dy 加一，得到当前要处理的像素的 y 分量
                dy++;
           
                // 检测 dy 是否越界，如果越界，则跳过，扫描下一个模板点
                // 如果 y 分量未越界，则处理当前像素点
                if (dy >= 0 && dy < inimg.imgMeta.height) {                    
                    // 将当前像素点邻域内点的像素值累加
                    tplsum[j] += (*curinptr) * (tpl.attachedData[i]);
                    
                    // 针对不同的实现类型，选择不同的路径进行处理
                    switch(imptype)
                    {
                    // 使用邻域像素总和除以像素点个数的运算方法实现线性滤波
                    case LNFT_COUNT_DIV:
                        // 记录当前像素点邻域内已累加点的个数
                        tmpcount[j] += 1;
                        break;
                        
                    // 使用邻域像素总和除以像素点权重之和的运算方法实现线性滤波    
                    case LNFT_WEIGHT_DIV:
                        // 记录当前像素点权重之和
                        tmpcount[j] += tpl.attachedData[i];
                        break;
                       
                    // 使用邻域像素直接带权加和的运算方法实现线性滤波    
                    case LNFT_NO_DIV:
                        // 设置除数为 1
                        tmpcount[j] = 1;
                        break;
                    }
                }
            } 
        }
    }
    
    // 将 4 个平均值分别赋值给对应的输出图像
    // 定义输出图像位置的指针
    unsigned char *outptr;
 
    // 获取对应的第一个输出图像的位置
    outptr = outimg.imgMeta.imgData + dstr * outimg.pitchBytes + dstc;
    
    // 计算邻域内点的像素平均值并赋值给输出图像
    *outptr = (tplsum[0] / tmpcount[0]);

    // 处理剩下的 3 个点
    for (int i = 1; i < 4; i++) {
        // 先判断 y 分量是否越界,如果越界，则可以确定后面的点也会越界，所以
        // 直接返回
        if (++dstr >= outimg.imgMeta.height) 
            return; 

        // 获取当前列的下一行的位置指针
        outptr = outptr + outimg.pitchBytes;
        
        // 计算邻域内点的像素平均值并赋值给输出图像  
        *outptr = (tplsum[i] / tmpcount[i]);
    }
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

    if(tp->count==DEF_TEMPLATE_COUNT){
        // 如果是默认大小的模板，则拷贝到常量内存中
        errcode = _copyDefTemplateToConstantMem(tp);
        if (errcode != NO_ERROR)
            return errcode; 
    }
    else{
        // 否则，将模板拷贝到 Device 内存中
        errcode = TemplateBasicOp::copyToCurrentDevice(tp);
        if (errcode != NO_ERROR)
            return errcode;   
    }
    return NO_ERROR;
}

// Host 函数：_adjustRoiSize（调整输入和输出图像的 ROI 的大小）
static __host__ void _adjustRoiSize(ImageCuda *inimg, ImageCuda *outimg)
{
    // 如果输入图像宽度大于输出图像，则将输出图像宽度值赋给输入图像，
    // 否则将输入图像宽度值赋给输出图像
    if (inimg->imgMeta.width > outimg->imgMeta.width) 
        inimg->imgMeta.width = outimg->imgMeta.width;
    else
        outimg->imgMeta.width = inimg->imgMeta.width;
        
    // 如果输入图像高度大于输出图像，则将输出图像高度值赋给输入图像，
    // 否则将输入图像高度值赋给输出图像
    if (inimg->imgMeta.height > outimg->imgMeta.height)
        inimg->imgMeta.height = outimg->imgMeta.height;
    else
        outimg->imgMeta.height = inimg->imgMeta.height;
}

// Host 函数：_getBlockSize（获取 Block 和 Grid 的尺寸）
static __host__ int _getBlockSize(int width, int height, dim3 *gridsize,
                                  dim3 *blocksize)
{
    // 检测 girdsize 和 blocksize 是否是空指针
    if (gridsize == NULL || blocksize == NULL)
        return NULL_POINTER; 

    // blocksize 使用默认的尺寸
    blocksize->x = DEF_BLOCK_X;
    blocksize->y = DEF_BLOCK_Y;

    // 使用最普通的方法划分 Grid 
    gridsize->x = (width + blocksize->x - 1) / blocksize->x;
    gridsize->y = (height + blocksize->y * 4 - 1) / (blocksize->y * 4);

    return NO_ERROR;
}

// 构造函数：LinearFilter
__host__ LinearFilter::LinearFilter(int imptype, Template *tp)
{
    // 设置滤波操作的实现方式
    setImpType(imptype);

    // 设置滤波操作所要使用的模板
    setTemplate(tp);
}

// 成员方法：getImpType
__host__ int LinearFilter::getImpType() const
{
    // 返回 impType 成员变量的值
    return this->impType;
}

// 成员方法：setImpType
__host__ int LinearFilter::setImpType(int imptype)
{
    // 检查输入参数是否合法
    if (imptype != LNFT_COUNT_DIV && imptype != LNFT_WEIGHT_DIV &&
        imptype != LNFT_NO_DIV)
        return INVALID_DATA;

    // 将 impType 成员变量赋成新值
    this->impType = imptype;
    return NO_ERROR;
}

// 成员方法：getTemplate
__host__ Template *LinearFilter::getTemplate() const
{
    // 如果模板指针和默认模板指针相同，则返回空
    if (this->tpl == &(_defTpl->tplMeta)) 
        return NULL;

    // 否则返回设置的模板指针
    return this->tpl;
}

// 成员方法：setTemplate
__host__ int LinearFilter::setTemplate(Template *tp)
{
    // 如果 tp 为空，则只用默认的模板指针，否则将 tp 赋值给 tpl
    if (tp == NULL) {
        this->tpl = &(_initDefTemplate()->tplMeta);
    } else {
        this->tpl = tp;
    }
    return NO_ERROR;
}

// 成员方法：linearFilter	        
__host__ int LinearFilter::linearFilter(Image *inimg, Image *outimg)
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
        
    // 检查滤波实现方式是否为合法值
    if (impType != LNFT_COUNT_DIV && impType != LNFT_WEIGHT_DIV &&
        impType != LNFT_NO_DIV)
        return INVALID_DATA; 
     
    // 调用 Kernel 函数进行均值滤波操作
    _linearFilterKer<<<gridsize, blocksize>>>(insubimgCud, 
                                              outsubimgCud, 
                                              *TEMPLATE_CUDA(tpl), 
                                              impType);
    // 调用 cudaGetLastError 判断程序是否出错
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        return CUDA_ERROR;
        
    // 处理完毕，退出 
    return errcode; 
}

// 成员方法：linearFilterMultiGPU           
__host__ int LinearFilter::linearFilterMultiGPU(Image *inimg, Image *outimg)
{
    int errcode;        // 局部变量，错误码
    dim3 gridsize;
    dim3 blocksize;
    
    // 检查输入图像，输出图像，以及模板是否为空
    if (inimg == NULL || outimg == NULL || tpl == NULL)
        return NULL_POINTER;

    cudaGetDeviceCount(&deviceCount);
    ImageCuda *deviceinimg, *deviceoutimg;

    deviceinimg = imageCut(inimg);
    deviceoutimg = imageCut(outimg);

    cudaStream_t stream[2];
    for(int i = 0; i < deviceCount; ++i){
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
        
        errcode = cudaMallocPitch((void **)(&deviceinimg[i].d_imgData), &deviceinimg[i].pitchBytes, 
                                  deviceinimg[i].imgMeta.width * sizeof (unsigned char), 
                                  deviceinimg[i].imgMeta.height);
        if (errcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        errcode = cudaMallocPitch((void **)(&deviceoutimg[i].d_imgData), &deviceoutimg[i].pitchBytes, 
                                   deviceoutimg[i].imgMeta.width * sizeof (unsigned char), 
                                   deviceoutimg[i].imgMeta.height);
        if (errcode != cudaSuccess) {
            return CUDA_ERROR;
        }
    }

    for(int i =0;i < deviceCount; ++i){
        cudaSetDevice(i);
        errcode = cudaMemcpy2DAsync (deviceinimg[i].d_imgData, deviceinimg[i].pitchBytes, 
                                     deviceinimg[i].imgMeta.imgData, deviceinimg[i].pitchBytes, 
                                     deviceinimg[i].imgMeta.width * sizeof (unsigned char), 
                                     deviceinimg[i].imgMeta.height,
                                     cudaMemcpyHostToDevice, stream[i]);
        if (errcode != cudaSuccess) {
            return CUDA_ERROR;
        }

       
        errcode = cudaMemcpy2DAsync (deviceoutimg[i].d_imgData,deviceoutimg[i].pitchBytes, 
                                     deviceoutimg[i].imgMeta.imgData, deviceoutimg[i].pitchBytes, 
                                     deviceoutimg[i].imgMeta.width * sizeof (unsigned char), 
                                     deviceoutimg[i].imgMeta.height,
                                     cudaMemcpyHostToDevice, stream[i]);
        if (errcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        if(tp->count==DEF_TEMPLATE_COUNT){
        // 如果是默认大小的模板，则拷贝到常量内存中
        errcode = _copyDefTemplateToConstantMem(tpl);
        if (errcode != NO_ERROR)
            return errcode; 
        }
        else{
            // 否则，将模板拷贝到 Device 内存中
            errcode = TemplateBasicOp::copyToCurrentDevice(tpl);
            if (errcode != NO_ERROR)
                return errcode;   
        }

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
        errcode = _getBlockSize(outimg[i].imgMeta.width,
                                outimg[i].imgMeta.height,
                                &gridsize, &blocksize);
        if (errcode != NO_ERROR) 
            return errcode;

        _linearFilterKer<<<gridsize, blocksize, 0, stream[i]>>>(deviceinimg[i], 
                                                                deviceoutimg[i], 
                                                                *TEMPLATE_CUDA(tpl), 
                                                                impType);
        // 调用 cudaGetLastError 判断程序是否出错
        if (cudaGetLastError() != cudaSuccess) 
            return CUDA_ERROR;
        errcode = cudaMemcpy2DAsync(deviceoutimg[i].imgMeta.imgData, deviceoutimg[i].pitchBytes, 
                                    deviceoutimg[i].d_imgData, 
                                    deviceoutimg[i].pitchBytes,
                                    deviceoutimg[i].imgMeta.width, 
                                    deviceoutimg[i].imgMeta.height,
                                    cudaMemcpyDeviceToHost, stream[i]);
        if (errcode != cudaSuccess) {
            return CUDA_ERROR;
        }
    }

    for(int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        //Wait for all operations to finish
        cudaStreamSynchronize(stream[i]);
        
        cudaStreamDestroy(stream[i]);
        cudaFree(devicefrimg[i].d_imgData);
        cudaFree(devicebaimg[i].d_imgData);
        cudaFree(deviceoutimg[i].d_imgData);
    }

    // 处理完毕，退出。
    return NO_ERROR;
}