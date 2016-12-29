// Normalization.cu
// 对图像进行正规化

#include "Normalization.h"
#include "Template.h"

// 宏：用来定义使用 online 算法求平均值和方差
//#define USE_ONLINE

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸
#define DEF_BLOCK_X 32
#define DEF_BLOCK_Y  8

#ifdef USE_ONLINE
// Kernel 函数：_nomalizeKer（实现对输入图像的每个点进行正规化）
// 对输入图像的每一个点，以该点为中心的邻域，求出该邻域内的平均值和总体方差,
// 然后将该点的像素与平均值作差，再除以总体方差，得到的值作为输出
static __global__ void    // Kernel 无返回值
_nomalizeKer(
        ImageCuda inimg,  // 输入图像
        Template tpl,     // 模版，用来指定邻域范围
        float *res,       // 输出的计算的结果
        size_t pitch      // res 的 pitch 值  
);
#endif

// Host 函数：_creatTemplate（创建模版）
// 创建指定大小的方形模版，模版必须为空模版
static __host__ int    // 返回值：函数是否正确指向，若函数正确指向，返回
                       // NO_ERROR
_creatTemplate(
        int k,         // 指定要创建的方形模版的边长
        Template *tpl  // 模版指针，模版必须为空模版
);

// Host 函数：_creatTemplate（创建模版）
static __host__ int  _creatTemplate(int k, Template *tpl)
{
    int errcode;         // 局部变量，错误码
    // 判断 tpl 是否为空
    if (tpl == NULL)
        return NULL_POINTER;

    // 判断 k 是否合法
    if (k <= 0)
        return INVALID_DATA;

    // 计算模版中点的数量
    int count = k * k;

    // 计算中心点
    int center = k / 2;

    // 为模版构建数据
    errcode = TemplateBasicOp::makeAtHost(tpl, count);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 构造方形模版中的点
    for (int i = 0; i < count; i++) {
        tpl->tplData[2 * i] = i % k - center;
        tpl->tplData[2 * i + 1] = i / k - center;
    }

    // 计算完毕，返回
    return NO_ERROR;
}

#ifdef USE_ONLINE
// Kernel 函数：_nomalizeKer（实现对输入图像的每个点进行正规化）
static __global__ void _nomalizeKer(ImageCuda inimg, Template tpl, 
                                    float *res, size_t pitch)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量
    // c 表示 column， r 表示 row）。由于采用并行度缩减策略 ，令一个线程
    // 处理 4 个输出像素，这四个像素位于统一列的相邻 4 行上，因此，对于
    // dstr 需要进行乘 4 的计算
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致系统崩溃
    if (dstc >= inimg.imgMeta.width || dstr >= inimg.imgMeta.height)
        return;

    // 用来保存临时像素点的坐标的 x 和 y 分量
    int dx, dy;

    // 用来记录当前模版所在位置的指针
    int *curtplptr = tpl.tplData;

    // 用来记录当前输入图像所在位置的指针
    unsigned char *curinptr;

    // 计数器，用来记录某点在模版范围内拥有的点的个数
    int count[4] = { 0 , 0, 0, 0 };

    // 迭代求平均值和总体方差使用的中间值
    float m[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // 计算得到的平均值
    float mean[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // 计算得到的总体方差
    float variance[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    int pix;             // 局部变量，临时存储像素值
    // 扫描模版范围内的每个输入图像的像素点
    for (int i = 0; i < tpl.count; i++) {
        // 计算当模版位置所在像素的 x 和 y 分量，模版使用相邻的两个下标的
       // 数组表示一个点，所以使用当前模版位置的指针加一操作
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);

        float temp;          // 局部变量，在进行迭代时的中间变量

        // 先判断当前像素的 x 分量是否越界，如果越界，则跳过，扫描下一个模版点,
        // 如果没有越界，则分别处理当前列的相邻的 4 个像素
        if (dx >= 0 && dx < inimg.imgMeta.width) {
            // 根据 dx 和 dy 获取第一个像素的指针
            curinptr = inimg.imgMeta.imgData + dx + dy * inimg.pitchBytes;
            // 检测此像素点的 y 分量是否越界
            if (dy >= 0 && dy < inimg.imgMeta.height) {
                // 对第一个点利用 on-line 算法进行迭代
                pix = *(curinptr);
                count[0]++;
                temp = pix - mean[0];
                mean[0] += temp / count[0];
                m[0] += temp * (pix - mean[0]);
            }

            // 分别处理剩下三个像素点
            for (int j = 1; j < 4; j++) {
                // 获取下一个像素点的指针
                curinptr = curinptr + inimg.pitchBytes;
                dy++;
                // 检测第二个像素点的 y 分量是否越界
                if (dy >= 0 && dy < inimg.imgMeta.height) {
                    // 对第二个点利用 on-line 算法进行迭代
                    pix = *(curinptr);
                    count[j]++;
                    temp = pix - mean[j];
                    mean[j] += temp / count[j];
                    m[j] += temp * (pix - mean[j]);
                }    
            }
        }
    }

    // 计算 4 个像素点中每个的正规化值

    // 计算第一个像素点的正规化
    // 定义并计算指向第一个像素在输出数组中的指针
    float *outptr =(float *)((char *)res + dstr * pitch) + dstc;
    // 第一个点的像素值
    curinptr = inimg.imgMeta.imgData + dstc + dstr * inimg.pitchBytes;
    pix = *(curinptr);
    // 判断 m 值是否为 0，如果为 0，则将对应的正规化值设置为 0
    if (m[0] <= 0.000001f && m[0] >= -0.000001f) {
        *outptr = 0.0f;
    } else {
        // 计算第一个像素点的总体方差
        variance[0] = sqrtf(m[0] / count[0]);
        // 计算第一个像素点的正规化画像值
        *outptr = (pix - mean[0]) / variance[0];
    }

    // 分别计算剩下三个点的像素值
    for (int i = 1; i < 4; i++) {
        // 判断该点的 y 分量是否越界，如果越界，则可以确定后面的点也越界，直接
        // 返回
        if (++dstr >= inimg.imgMeta.height)
            return;
        // 计算该点在输出数组中的指针
        outptr = (float *)((char *)outptr + pitch);
        // 该点的像素值
        curinptr = curinptr + inimg.pitchBytes;
        pix = *(curinptr);
        // 判断 m 值是否为 0，如果为 0，则将对应的正规化值设置为 0
        if (m[i] <= 0.000001f && m[i] >= -0.000001f) {
            *outptr = 0.0f;
        } else {
            // 计算该像素点的标准方差
            variance[i] = sqrtf(m[i] / count[i]);
            // 计算该像素点的正规化画像值
            *outptr = (pix - mean[i]) / variance[i];
        }
    }
}
#endif

// Kernel 函数：_nomalizeKer（使用常规方法求平均值和方差）
static __global__ void _nomalizenorKer(ImageCuda inimg, Template tpl, 
                                       float *res, size_t pitch)
{
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstc >= inimg.imgMeta.width || dstr >= inimg.imgMeta.height)
        return;

    int *curtplptr = tpl.tplData;
    int dx, dy;
    int count = 0;
    int sum = 0;
    float mean;
    float variance = 0.0f;

    for (int i = 0; i < tpl.count; i++) {
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);
        if (dx < 0 || dx >= inimg.imgMeta.width || 
            dy < 0 || dy >= inimg.imgMeta.height) {
            continue;
        }
        count++;
        sum += *(inimg.imgMeta.imgData + dy * inimg.pitchBytes + dx); 
    } 
    mean = (float)sum / count;
    curtplptr = tpl.tplData;
    for (int i = 0; i < tpl.count; i++) {
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);
        if (dx < 0 || dx >= inimg.imgMeta.width || 
            dy < 0 || dy >= inimg.imgMeta.height) {
            continue;
        }
        int pix = *(inimg.imgMeta.imgData + dy * inimg.pitchBytes + dx);
        variance += (mean - pix) * (mean - pix);
    } 

    float *outptr = (float *)((char *)res + dstr * pitch) + dstc;

    if (variance < 0.00001f)
        *outptr = 0.0f;
    else {
        int pix = *(inimg.imgMeta.imgData + dstr * inimg.pitchBytes + dstc);
        *outptr = (mean - pix) / sqrtf(variance);
    }
}

// Host 成员方法：normalize（对输入图像进行正规化）
__host__ int Normalization::normalize(Image *inimg, float *out, size_t pitch, 
                                      int width, int height, bool ishost)
{
    int errcode;           // 局部变量，错误码
    cudaError_t cudaerr;   // 局部变量，CUDA 调用返回的错误码
    dim3 gridsize;
    dim3 blocksize;

    // 判断 inimg 和 out 是否是为空
    if (inimg == NULL || out == NULL) 
        return NULL_POINTER;

    // 判断 width 和 height 参数的合法性
    if (width <= 0 || height <= 0)
        return INVALID_DATA;

    // 判断 pitch 的合法性
    if (pitch < width * sizeof (float))
        return INVALID_DATA;

    // 对输入图像申请 Device 存储空间
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR) 
        return errcode;

    // 提取输入图像的 ROI 子图
    ImageCuda inimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &inimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 调整输入图像和输出数组的长和宽
    if (inimgCud.imgMeta.width > width) 
        inimgCud.imgMeta.width = width;
    if (inimgCud.imgMeta.height > height)
        inimgCud.imgMeta.height = height;

    // 计算线程块的数量
    // blocksize 使用默认线程块
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    // 使用最普通的方法划分 Grid
    gridsize.x = (inimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (inimgCud.imgMeta.height + blocksize.y * 4 - 1) / 
                  blocksize.y * 4;

    // 创建模版
    Template *tpl;
    errcode = TemplateBasicOp::newTemplate(&tpl);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 设置模版形状
    errcode = _creatTemplate(k, tpl);
    if (errcode != NO_ERROR)
        return errcode;

    errcode = TemplateBasicOp::copyToCurrentDevice(tpl);
    if (errcode != NO_ERROR)
        return errcode;

    float *resCud;     // 指向 Device 内存，用来存储正规化得到的结果
    size_t respitch;   // resCud 的 pitch
    // 判断 out 是否指向 host，如果是，需要在创建 Device 中创建空间
    if (ishost) {
        // 为 resCud 申请内存空间
        cudaerr = cudaMallocPitch((void **)&resCud, &respitch, 
                                  width * sizeof (float), height);
        if (cudaerr != cudaSuccess) 
            return CUDA_ERROR;
    } else {
        resCud = out;
        respitch = pitch;
    }

    dim3 blocksize1;
    dim3 gridsize1;
    // blocksize 使用默认线程块
    blocksize1.x = DEF_BLOCK_X;
    blocksize1.y = DEF_BLOCK_Y;
    // 使用最普通的方法划分 Grid
    gridsize1.x = (inimgCud.imgMeta.width + blocksize1.x - 1) / blocksize1.x;
    gridsize1.y = (inimgCud.imgMeta.height + blocksize1.y - 1) / 
                  blocksize1.y;

    #ifdef USE_ONLINE
    _nomalizeKer<<<gridsize, blocksize>>>(inimgCud, *tpl, resCud, respitch);
    #else
    _nomalizenorKer<<<gridsize1, blocksize1>>>(inimgCud, *tpl, resCud, respitch);
    #endif
    
    

    // 调用 Kernel 函数进行正规化操作
   // _nomalizeKer<<<gridsize, blocksize>>>(inimgCud, *tpl, resCud, respitch);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 如果 out 是指向 host 内存的，则需要将 resCud 的内容拷贝到 out 中，
    // 并且释放 resCud 指向的内存空间
    if (ishost) {
        // 将正规化得到的结果从 Device 内存中拷贝到 Host 内存
        cudaerr = cudaMemcpy2D(out, width * sizeof (float), resCud, respitch, 
                               width * sizeof (float), height, 
                               cudaMemcpyDeviceToHost);
        if (cudaerr != cudaSuccess)
            errcode = CUDA_ERROR;
        else
            errcode = NO_ERROR;
        // 释放 resCud 指向的内存空间
        cudaFree(resCud);
    }

    // 释放模版空间
    TemplateBasicOp::deleteTemplate(tpl);

    // 处理完毕，返回
    return errcode;
}

