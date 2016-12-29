// ImageMacth.cu
// 实现对图像进行匹配的操作

#include "ConnectRegion.h"
#include "ErrorCode.h"
#include "ImageMatch.h"
#include "LabelIslandSortArea.h"
#include "Normalization.h"
#include "Rectangle.h"
#include "RoiCopy.h"
#include "SmallestDirRect.h"
#include "Template.h"
#include "TemplateFactory.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸
#define DEF_BLOCK_X 32
#define DEF_BLOCK_Y  8

// 宏：FAST_RUN
// 打开该开关，在 Kernel 中将不会搜索当前点的领域，可以加快运行时间，但匹配的
// 准确度将没有那么高
#define FAST_RUN

#ifdef FAST_RUN
// Device 数据，_tpl1x1Gpu
// 为了加快算法的运行时间，用此数据来替代 _tpl3x3Gpu
static __device__ int _tpl1x1Gpu[] = {0, 0};
#else
// Device 数据：_tpl3x3Gpu
// 3 * 3 的模版,为了加快算法的速度，暂时不用
static __device__ int _tpl3x3Gpu[] = { -1, -1, 0, -1, 1, -1,
                                       -1,  0, 0,  0, 1,  0,
                                       -1,  1, 0,  1, 1,  1  };
#endif

// Device 函数：_rotateXYDev（计算旋转表坐标对应的在 TEST 图像上的坐标）
// 计算 TEMPLATE 坐标对应的旋转表坐标在 TEST 图像上所对应的坐标
static __device__ int            // 返回值：函数是否正确执行，如果函数
                                 // 正确执行，返回 NO_ERROR
_rotateXYDev(
        int x, int y,            // TEMPLATE 对应的旋转表的横坐标和纵
                                 // 坐标，旋转中心为原点
        int xc, int yc,          // 现在匹配的横坐标和纵坐标，以左上角为原点
        RotateTable rotatetable, // 旋转表
        float angle,             // 旋转角
        int *rx,                 // 转换后得到对应 TEST 图像的横坐标
        int *ry                  // 转换后得到对应 TEST 图像的纵坐标 
);

// Device 函数：_getSuitValueFromNormalTplDev（在正规化结果中找到合适值）
// 在 TEMPLATE 的指定坐标的一个邻域内找到正规化后与 flag 最接近的一个值
static __device__ float         // 返回值：在正规化结果中与 flag 最接近的值
_getSuitValueFromNormalTplDev(
        int x, int y,           // 在 TEMPLATE 的坐标位置
        float *normalizedata,   // TEMPLATE 中正规化的结果
        size_t pitch,           // normalizedata 的 pitch 值
        int width, int height,  // TEMPLATE 的宽和高
        float flag              // 需要对比的值
);

// Host 函数：_getCormapMaxIndex（获取 cormapsum 中最大的值的索引）
// 在匹配得到的结果中找到最大的值
static __host__ int        // 返回值：cormapsum 中最大的值的索引
_getCormapMaxIndex(
        float *cormapcpu,  // cormapsum 的数据
        int count          // cormapsum 中数据的数量
);

// Kernel 函数：_calCorMapSumKer（计算每个点的邻域内 cormap 的和）
// 计算一每个点为中心的邻域内 cormap 的和
static __global__ void            // 返回值：Kernel 无返回值
_calCorMapSumKer(
        float *cormap,            // cormap 的数据
        int dwidth, int dheight,  // 摄动范围的宽和高
        int scope,                // 邻域的范围，以 scope * scope 的范围计算
        float *cormapsumgpu       // 求和得到的结果
);

// Kernel 函数：_matchKer（将一组 TEMPLATE 分别和 TEST 图像进行匹配）
// 将一组 TEMPLATE 图像正规化后，用不同的旋转角与 TEST 图像进行匹配，得到每个
// 点的相关系数
static __global__ void                  // 返回值：Kernel 无返回值
_matchKer(
        float **tplnormalization,       // 每个 TEMPLATE 正规化的结果
        size_t *tplpitch,               // 每个 TEMPLATE 正规化数组的 ptich 值
        int tplcount,                   // TEMPLATE 的数量 
        int tplwidth, int tplheight,    // 每个 TEMPLATE 的宽和高
        float *testnormalization,       // TEST 正规化的结果
        size_t testpitch,               // TEST 正规化结果的 pitch 值
        int testwidth, int testheight,  // TEST 图像的宽和高
        RotateTable rotatetable,        // 旋转表
        float *cormap,                  // 用来存储每个点匹配得到的相关系数
        int offsetx, int offsety,       // 摄动范围的偏移量
        int dwidth, int dheight,        // 摄动范围的宽和高
        int tploffx, int tploffy        // TEMPLATE 的偏移量
);

// Kernel 函数：_localCheckErrKer（进行局部异常检查）
// 对匹配得到的结果进行局部异常检查
static __global__ void                      // 返回值：Kernel 无返回值                  
_localCheckErrKer( 
        float *besttplnor,                  // 匹配得到的 TEMPLATE 的正规化数据
        size_t besttplpitch,                // besttplnor 的 pitch 值
        int tplwidth, int tplheight,        // TEMPLATE 的宽和高
        float *testnormalization,           // TEST 图像的正规化数据
        size_t testpitch,                   // testnormalization 的 pitch 值
        int testwidth, int testheight,      // TEST 图像的宽和高
        RotateTable rotatetable,            // 旋转表
        int *errmap,                        // 记录异常情况的数组
        size_t errmappitch,                 // errmap 的 pitch 值
        int errmapwidth, int errmapheight,  // errmap 数组的大小
        float errthreshold,                 // 异常检查的阈值
        int mx, int my,                     // 匹配得到的匹配中心
        float angle,                        // 匹配得到的旋转角
        int tploffx, int tploffy            // TEMPLATE 的偏移量
);

// Kernel 函数：_binarizeKer（对 errmap 进行二值化）
// 对 errMap 进行二值化
static __global__ void                      // 返回值：Kernel 无返回值
_binarizeKer(
        int *errmap,                        // errmap 的数据 
        size_t errmappitch,                 // errmap 的 pitch 值
        int errmapwidth, int errmapheight,  // errmap 的宽和高
        ImageCuda out                       // 二值化的输出图像
);

// Kernel 函数：_getMaxWinKer（获取每个 window 高值点的个数）
// 扫描每个 window，获取每个 window 的个数
static __global__ void        // 返回值：函数是否正确执行，若正确执行，
                              // 返回 NO_ERROR
_getMaxWinKer(
        ImageCuda errmapimg,  // errmap 的二值化图像
        Template wintpl,      // window 的模版
        int *wincountcud      // 存放每个 window 的扫描结果
);

// Host 函数：_getDirectRectForErrMap（获取 errmap 的最小有向四边形）
// 获取 errmap 中孤岛的最小有向四边形
static __host__ int            // 返回值：函数是否正确执行，若正确执行，
                               // 返回 NO_ERROR
_getDirectRectForErrMap(
        int *errmap,           // errmap 数据
        size_t errmappitch,    // errmap 的 pitch 值
        int errmapwidth,       // errmap 的宽
        int errmapheight,      // errmap 的高
        int errwinwidth,       // window 的宽
        int errwinheight,      // window 的高
        int errwinthreshold,   // window 的阈值
        DirectedRect *dirrect  // 得到的最小有向四边形
);

// Device 函数：_getSuitValueFromNormalTplDev（在正规化结果中找到合适值）
static __device__ float _getSuitValueFromNormalTplDev(int x, int y, 
                                                      float *normalizedata,
                                                      size_t pitch, int width,
                                                      int height, float flag)
{
#ifdef FAST_RUN
    int *tpl = _tpl1x1Gpu;     // 指向 1 * 1 模版的指针
#else
    int *tpl = _tpl3x3Gpu;     // 指向 3 * 3 模版的指针
#endif
    int currx = x, curry = y;  // 当前在 TEMPLATE 中的坐标
    float currvalue;           // 存放当前最接近 flag 的值
    float currdiff;            // 存放 currvalue 与 flag 的差的绝对值

    // 初始化 currvalue 为 （x, y）对应的正规化值
    currvalue = *((float *)((char *)normalizedata + curry * pitch) + currx);
    // 记录与 flag 最接近的值的差，初始化为（x, y）对应的正规化的值与 flag 的
    // 绝对值
    float mindiff = fabsf(currvalue - flag);
    // 记录与 flag 最接近的值，初始化为（x, y）对应的正规化的值
    float minvalue = currvalue;

    // 扫描模版的每一个点，找到与 flag 最接近的值
#ifdef FAST_RUN
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++) {
#else
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
#endif
            // 计算当前点的坐标
            currx = x + *(tpl++);
            curry = y + *(tpl++);
            // 判断当前点的坐标是否越界，如果越界则跳过
            if (currx >= 0 && currx < width && curry >= 0 && curry < height) {
                // 计算当前坐标的正规化的值
                currvalue = *((float *)((char *)normalizedata + curry * pitch) +
                              currx);
                // 计算 currvalue 与 flag 的绝对值
                currdiff = fabsf(currvalue - flag);
                // 如果 currdiff 比 mindiff 小，则用 currvalue 替换 minvalue
                if (currdiff < mindiff)
                    mindiff = currdiff; 
                    minvalue = currvalue;
            }
        }
    }
    // 返回与 flag 最接近的值
    return minvalue; 
}

// Device 函数：_rotateXYDev（计算旋转表坐标对应的在 TEST 图像上的坐标）
static __device__ int _rotateXYDev(int x, int y, int xc, int yc,
                                   RotateTable rotatetable, float angle,
                                   int *rx, int *ry)
{
    int errcode;   // 局部变量，错误码
    float tx, ty;  // 存放通过旋转表得到的旋转后的坐标

    // 获取旋转表旋转后的坐标
    errcode = rotatetable.getRotatePos(x, y, angle, tx, ty);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算旋转后的坐标对应在 TEST 图像上的坐标，先对 tx 和 ty 进行四舍五入
    *rx = (int)(tx + 0.5f) + xc;
    *ry = (int)(ty + 0.5f) + yc;

    // 执行完毕，返回 NO_ERROR
    return NO_ERROR;
}

// 成员方法：initNormalizeData（对设置的 TEMPLATE 进行初始化）
__host__ int ImageMatch::initNormalizeData()
{
    int cudaerr;  // 局部变量，调用 CUDA 系统 API 返回的错误码 
    
    // 删除之前的 TEMPLATE 的相关数据
    deleteNormalizeData();
    // 将临时记录 TEMPLATE 的数量的 tplTmpCount 赋值给 tplCount
    this->tplCount = this->tplTmpCount;

    // 为 tplNormalization 申请空间
    tplNormalization = new float *[this->tplCount];
    if (tplNormalization == NULL)
        return OUT_OF_MEM;
    // 为 ptich 申请空间
    pitch = new size_t[this->tplCount];
    if (pitch == NULL) 
        return OUT_OF_MEM; 

    // 依次为 tplNormalization 的每一个元素分别申请 Device 空间
    for (int i = 0; i < tplCount; i++) {
        // 申请 Device 空间
        cudaerr = cudaMallocPitch((void **)&(tplNormalization[i]), &(pitch[i]),
                                  tplWidth * sizeof (float), tplHeight);
        if (cudaerr != cudaSuccess)
            return CUDA_ERROR;
    }
    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}

// 成员方法：deleteNormalizeData（删除 TEMPLATE 正规化的数据）
__host__ int ImageMatch::deleteNormalizeData()
{
    // 先判断 tplNormalization 是否为空，若不为空，则删除每个 tplNormalization
    // 的 Device 空间，然后删除 tplNormalization 指向的空间
    if (tplNormalization != NULL) {
        // 扫描每一个 tplNormalization 成员
        for(int i = 0; i < tplCount; i++)
            // 释放每一个 tplNormalization 成员指向的 Device 内存空间
            cudaFree(tplNormalization[i]);

        // 释放 tplNormalization 指向的空间
        delete tplNormalization;
        // 将 tplNormalization 置空，防止成为野指针
        tplNormalization = NULL;
    }

    // 先判断 pitch 是否为空，若为空，则跳过
    if (pitch != NULL) {
        // 若不为空，则释放 pitch 指向的内存空间
        delete pitch;
        // 将 pitch 置空，防止成为野指针
        pitch = NULL;
    }

    // 处理完毕，返回 NO_ERROR 
    return NO_ERROR;
}

// 成员方法：normalizeForTpl（对设定的 TEMPLATE 进行正规化） 
__host__ int ImageMatch::normalizeForTpl()
{
    int errcode;  //局部变量，错误码
    
    // 先判断是否需要对 TEMPLATE 进行正规化，若不需要，则直接返回，这样可以增
    // 加效率
    if (this->needNormalization) {
        // 先初始化正规化的数据，为存放正规化的结果申请空间
        errcode = initNormalizeData();
        if (errcode != NO_ERROR)
            return errcode;

        // 创建一个用来正规化操作的对象
        Normalization normal(3);
        // 扫每一个 TEMPLATE 
        for (int i = 0; i < tplCount; i++) {
            // 分别对每一个 TEMPLATE 进行正规化操作
            errcode = normal.normalize(tplImages[i], tplNormalization[i],
                                       pitch[i], tplWidth, tplHeight, false);
            if (errcode != NO_ERROR) {
                return errcode;
            }
        }

        // 将 needNormalization 变量设置为 false，防止下次再进行正规化
        this->needNormalization = false;
    }
    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}

// Host 函数：_getCormapMaxIndex（获取 cormap 中最大的值的索引）
static __host__ int _getCormapMaxIndex(float *cormapcpu, int count)
{
    // 记录 cormap 中的最大值，初始化为 cormap 中的第 1 个数
    float max = cormapcpu[0];
    // 记录 cormap 中的最大值的索引，初始化为第 1 个数
    int maxindex = 0;

    // 扫描每个 cormap 中的数据
    for (int i = 1; i < count; i++) {
        // 若当前 cormap 比 max 大，则替换 max 的值，并记录当前的索引
        if (max < cormapcpu[i]) {
            max = cormapcpu[i];
            maxindex = i;
        }
    }

    // 处理完毕，返回 cormap 中最大值的索引
    return maxindex;
}

// 对 errMap 进行二值化
static __global__ void _binarizeKer(int *errmap, size_t errmappitch, 
                                    int errmapwidth, int errmapheight,
                                    ImageCuda out)
{
    // 计算当前像素要处理的点的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断要处理的点的坐标是否越界，若越界，则直接返回
    if (x < 0 || x >= errmapwidth || y < 0 || y >= errmapheight)
        return;

    // 获取当前坐标对应的 errmap 的指针
    int *perr = (int *)((char *)errmap + y * errmappitch) + x;
    // 判断当前坐标对应的 errmap 中的值是否大于阈值，若大于，则赋值为 255，否则
    // 赋值为 0
    *(out.imgMeta.imgData + y * out.pitchBytes + x) = (*perr > 0) ? 255 : 0; 
}

// 找 window 内点最大的 window 的中心
static __global__ void _getMaxWinKer(ImageCuda errmapimg, Template wintpl,
                                     int *wincountcud)
{
    // 计算当前像素要处理的点的坐标
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断要处理的点的坐标是否越界，若越界，则直接返回
    if (dstc < 0 || dstc >= errmapimg.imgMeta.width 
            || dstr < 0 || dstr >= errmapimg.imgMeta.height)
        return;
   
    // 用来保存临时像素点的坐标的 x 和 y 分量
    int dx, dy;
    // 用来记录当前模版所在位置的指针
    int *curtplptr = wintpl.tplData;
    // 用来统计 window 的高值点的个数
    int count = 0;
    // 用来记录图像某点的位置
    unsigned char pix;
    
    // 扫描模版范围内的每个像素点
    for (int i = 0; i < wintpl.count; i++) {
        // 计算当前模版位置所在像素的 x 和 y 分量，模版使用相邻的两个下标的
        // 数组表示一个点，所以使当前模版位置的指针作加一操作
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);
        // 判断 dx 和 dy 是否越界
        if (dx >= 0 && dx < errmapimg.imgMeta.width 
                && dy >= 0 && dy < errmapimg.imgMeta.height) {
            // 获取（dx，dy）所在图像的位置指针
            pix = *(errmapimg.imgMeta.imgData + dy * errmapimg.pitchBytes + dx);
            // 判断当前像素是否大于 0，若大于 0，则 count 加一
            (pix > 0) ? (count++) : 0;
        }
    } 
    // 将统计的个数存放到 wincountcud 中
    *(wincountcud + dstr * errmapimg.imgMeta.width + dstc) = count;
}

// 获取 errMap 中密度较大的各个孤岛的外接最小有向四边形
static __host__ int _getDirectRectForErrMap(int *errmap, size_t errmappitch, 
                                            int errmapwidth, int errmapheight,
                                            int errwinwidth, int errwinheight,
                                            int errwinthreshold,
                                            DirectedRect *dirrect)
{
    // 判断 errmap 是否为空，若为空，则返回 NULL_POINTER
    if (errmap == NULL || dirrect == NULL)
        return NULL_POINTER;

    int errcode;          // 局部变量，错误码
    cudaError_t cudaerr;  // 局部变量，CUDA 调用返回错误码
    dim3 blocksize;
    dim3 gridsize;

    // window 的模版
    Template *wintpl;
    // 从 TemplateFactory 中获取一个矩形模版
    errcode = TemplateFactory::getTemplate(&wintpl, TF_SHAPE_BOX,
                                           dim3(errwinwidth, errwinheight)); 
    // 若获取失败，则返回错误
    if (errcode != NO_ERROR)
        return errcode;
    // 将模版数据拷贝到设备端
    TemplateBasicOp::copyToCurrentDevice(wintpl);

    // 在设备端，用来存放每个 window 的高值点个数
    int *wincountcud;
    // 用来存放 wincount 的大小
    int wincountsize = sizeof (int) * errmapwidth * errmapheight;
    // 为 wincountcud 在设备端申请一段内存
    cudaerr = cudaMalloc((void **)&wincountcud, wincountsize);
    // 若申请失败，则返回错误
    if (cudaerr != cudaSuccess) {
        TemplateFactory::putTemplate(wintpl); 
        return CUDA_ERROR;
    }
 
    // 计算线程块的大小
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (errmapwidth + blocksize.x - 1) / blocksize.x; 
    gridsize.y = (errmapheight + blocksize.y - 1) / blocksize.y; 

    // errmap 通过二值化得到的图像数据
    Image *errmapimg;
    errcode = ImageBasicOp::newImage(&errmapimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 为 errmapimg 在 Device 端申请空间
    errcode = ImageBasicOp::makeAtCurrentDevice(errmapimg, 
                                                errmapwidth, errmapheight);
    if (errcode != NO_ERROR) {
        ImageBasicOp::deleteImage(errmapimg);
        return errcode;
    }

    // 将 errmapimg 转化为 ImageCuda 格式
    ImageCuda *errmapimgcud;
    errmapimgcud = IMAGE_CUDA(errmapimg);

    // 调用二值化函数对 errmap 转化为二值图像 errmapimg
    _binarizeKer<<<gridsize, blocksize>>>(errmap, errmappitch,
                                          errmapwidth, errmapheight, 
                                          *errmapimgcud);     
    // 若二值化出现错误，则释放内存空间，然后返回 CUDA_ERROR
    if (cudaGetLastError() != cudaSuccess) {
        TemplateFactory::putTemplate(wintpl);
        cudaFree(wincountcud);
        return CUDA_ERROR;
    }
 
    // 统计每个 window 的高值点的个数
    _getMaxWinKer<<<gridsize, blocksize>>>(*errmapimgcud, *wintpl, wincountcud);
    // 若 kernel 函数错误，则释放内存空间，然后返回 CUDA_ERROR
    if (cudaGetLastError() != cudaSuccess) {
        TemplateFactory::putTemplate(wintpl);
        cudaFree(wincountcud);
        return CUDA_ERROR;
    }
    // 将模版放回到模版工厂里
    TemplateFactory::putTemplate(wintpl);

    // 在 CPU 端申请一段空间，用来存放得到的每个 window 的高值点的个数
    int *wincount;
    wincount = (int *)malloc(wincountsize);
    if (wincount == NULL) {
        cudaFree(wincountcud);
        return OUT_OF_MEM; 
    } 

    // 将 wincountcud 中的数据从设备端拷贝到 CPU 端
    cudaerr = cudaMemcpy(wincount, wincountcud, wincountsize,
                         cudaMemcpyDeviceToHost);
    if (cudaerr != cudaSuccess) {
        cudaFree(wincountcud);
        free(wincount);
        return CUDA_ERROR;
    }

    // 记录高值点个数最多的 window 的中心坐标
    int maxwinx = 0, maxwiny = 0;
    // 记录 window 中高值点最大的数量，初始化为阈值
    int maxwinvalue = errwinthreshold;
    for (int i = 0; i < errmapheight; i++) {
        for (int j = 0; j < errmapwidth; j++) {
            if (maxwinvalue < *(wincount + i * errmapwidth + j)) {
                // 记录当前高值点最大的中心点
                maxwinvalue = *(wincount + i * errmapwidth + j);
                maxwinx = j;
                maxwiny = i;
            }
        }
    }
    
    // 若最大值等于阈值，则不存在局部异常
    if (maxwinvalue == errwinthreshold) {
        // 将 dirrect 的所有成员都置零
        dirrect->angle = 0.0f;
        dirrect->centerPoint[0] = 0;
        dirrect->centerPoint[1] = 0;
        dirrect->length1 = 0;
        dirrect->length2 = 0;
    } else {
        // 定义一个用来获取最小有向四边形的对象
        SmallestDirRect sdr;
        // 设置 errmapimg 的 ROI
        errmapimg->roiX1 = maxwinx - errwinwidth / 2;   
        errmapimg->roiY1 = maxwiny - errwinheight / 2;
        errmapimg->roiX2 = maxwinx + errwinwidth / 2;
        errmapimg->roiY2 = maxwiny + errwinheight / 2;
    
        RoiCopy roicopy;
        Image *timg;
        ImageBasicOp::newImage(&timg);
        roicopy.roiCopyAtHost(errmapimg, timg);
    
        // 调用最小有向四边形算法来得到 errmapimg 的最小有向四边形
        errcode = sdr.smallestDirRect(timg, dirrect); 
        // 若失败，则释放内存空间，然后返回错误
        if (errcode != NO_ERROR) {
            cudaFree(wincountcud);
            free(wincount);
            return errcode;
        }
    }

    // 释放空间
    cudaFree(wincountcud);
    free(wincount);
    // 处理完毕，返回
    return NO_ERROR;
}

// Kernel 函数：_calCorMapSumKer（计算每个点的邻域内 cormap 的和）
static __global__ void _calCorMapSumKer(float *cormap, 
                                        int dwidth, int dheight, 
                                        int scope, float *cormapsumgpu)
{
    // Kernel 采用三维来处理，其中第一、二维是摄动范围的坐标，第三维用来记录是
    // 所有模版的数量，其中 z % 模版数量，表示当前是第几个模版，z / 模版数量
    // 表示当前是第几个旋转角度
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    
    // 判断坐标是否越界，如果越界，则直接返回
    if (x < 0 || x >= dwidth || y < 0 || y >= dheight)
        return ;

    // 记录当前处理的坐标
    int currx, curry;
    // 记录（x, y）邻域内的 cormap 的和，初始化为 0.0
    float sum = 0.0f;

    // 依次扫描邻域内的每个点，然后求和
    for (int i = 0; i < scope; i++) {
        for (int j = 0; j < scope; j++) {
            // 计算当前需要处理的点的坐标
            currx = x + j - scope / 2;
            curry = y + i - scope / 2;
            // 判断当前坐标是否越界
            if (currx >= 0 && currx < dwidth && curry >= 0 && curry < dheight) {
                // 将当前坐标对应的 cormap 值加到 sum 中
                sum += *(cormap + z * dwidth * dheight + curry * dwidth + 
                         currx);
            } 
        }
    }

    // 将得到的结果存入 cormapsumgpu 中
    *(cormapsumgpu + z * dwidth * dheight + y * dwidth + x) = sum;
}

// Kernel 函数：_matchKer（将一组 TEMPLATE 分别和 TEST 图像进行匹配）
static __global__ void _matchKer(float **tplnormalization, size_t *tplpitch,
                                 int tplcount, int tplwidth, int tplheight,
                                 float *testnormalization, size_t testpitch,
                                 int testwidth, int testheight,
                                 RotateTable rotatetable, float *cormap,
                                 int offsetx, int offsety,
                                 int dwidth, int dheight, int tploffx,
                                 int tploffy)
{
    // Kernel 采用三维来处理，其中第一、二维是摄动范围的坐标，第三维用来记录是
    // 所有模版的数量，其中 z % 模版数量，表示当前是第几个模版，z / 模版数量
    // 表示当前是第几个旋转角度
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    // 判断坐标是否越界，如果越界，则直接返回
    if (x < 0 || x >= dwidth || y < 0 || y >= dheight)
        return;

    // 记录原始的（x, y）的坐标，在后面寻址的时候需要使用
    int tx = x;
    int ty = y;
    // 计算实际的（x, y）在 TEST 图像上的坐标，这里只需要加上一个偏移量就能
    // 计算出
    x += offsetx;
    y += offsety;

    int errcode;           // 局部变量，错误码
    int testx, testy;      // 局部变量，记录在 TEST 图像上的坐标
    float *testnorptr;     // 局部变量，指向 TEST 的正规化的结果的指针
    float tcormap = 0.0f;  // 局部变量，存储（x, y）的相关系数

    // 获取旋转角度
    float angle = rotatetable.getAngleVal(z / tplcount);
    // 计算当前 TEMPLATE 的缩影
    int tplindex = z % tplcount;

    // 依次扫描当前要处理的 TEMPLATE
    for (int i = 0; i < tplheight; i++) {
        for (int j = 0; j < tplwidth; j++) {
            // 计算每个 TEMPLATE 点旋转 angle 后对应在 TEST 图像上的坐标
            errcode = _rotateXYDev(j - tploffx, i - tploffy, 
                                   x, y, rotatetable, angle,
                                   &testx, &testy);
            // 若返回错误，则跳过，处理下一个点
            if (errcode != NO_ERROR) 
                continue ;
            
            // 如果 testx 和 testy 不在 TEST 图像内，则跳过，处理下一个点
            if (testx < 0 || testx >= testwidth || 
                testy < 0 || testy >= testheight)
                continue;

            // 获取旋转后得到的点在 TEST 正规化结果的指针
            testnorptr = (float *)((char *)testnormalization + 
                                   testy * testpitch) + testx;
            // 得到对应的 TEST 正规化的值
            float testnor = *testnorptr; 
            // 在对应 TEMPLATE 的正规化结果的邻域中找到与 testnor 最接近的 
            float tplnor = _getSuitValueFromNormalTplDev(
                                   j, i,
                                   tplnormalization[tplindex],
                                   tplpitch[tplindex],
                                   tplwidth, tplheight,
                                   testnor);

            // 将当前的 TEMPLATE 坐标的相关系数加到 tcormap 中
            tcormap += testnor * tplnor;
        }
    }

    // 将计算得到的相关系数写入 cormap 中
    *(cormap + z * dwidth * dheight + ty * dwidth + tx) = tcormap;
}

// Kernel 函数：_localCheckErrKer（进行局部异常检查）
static __global__ void _localCheckErrKer(float *besttplnor, size_t besttplpitch,
                                         int tplwidth, int tplheight, 
                                         float *testnormalization, 
                                         size_t testpitch,
                                         int testwidth, int testheight,
                                         RotateTable rotatetable, 
                                         int *errmap, size_t errmappitch,
                                         int errmapwidth, int errmapheight,
                                         float errthreshold, int mx, int my,
                                         float angle, int tploffx, int tploffy)
{
    // Kernel 采用二维来处理，第一维为 TEMPLATE 的横坐标，第二维为 TEMPLATE 的
    // 纵坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断坐标是否越界，若越界，直接返回
    if (x < 0 || x >= tplwidth || y < 0 || y >= tplheight)
        return;
    
    int errcode;        // 局部变量，错误码
    
int testx, testy;   // 局部变量，记录在 TEST 图像上的坐标

    // 计算坐标（x, y）旋转 angle 后在 TEST 图像上的坐标
    errcode = _rotateXYDev(x - tploffx, y - tploffy, mx, my, rotatetable,
                           angle, &testx, &testy);
    // 若计算错误，则直接返回
    if (errcode != NO_ERROR)
        return;

    // 判断 TEMPLATE 旋转后，在 TEST 图像上是否越界，若越界，直接返回
    if (testx < 0 || testx >= testwidth || testy < 0 || testy >= testheight)
        return;

    // 计算坐标（testx, testy）在 TEST 图像上的正规化值的指针
    float testnor = 
            *((float *)((char *)testnormalization + testy * testpitch) + testx);
    // 获取坐标（x, y）邻域内在 TEMPLATE 上的与 testnor 最接近的值
    float tplnor = _getSuitValueFromNormalTplDev(x, y, besttplnor, besttplpitch,
                                                 tplwidth, tplheight, testnor);
    // 计算（x, y）在 TEMPLATE 上与对应在 TEST 图像上的点的差异
    float v = (testnor - tplnor) * (testnor - tplnor);
    // 若差距大于阈值，则扩大 200 倍，然后赋值给 errmap，否则置 0
    *((int *)((char *)errmap + testy * errmappitch) + testx) = 
            (v > errthreshold) ? (int)(200.0f * v) : 0;
}

// 宏：FAIL_MEM_FREE
// 该宏用于清理临时申请的内存空间
#define FAIL_MEM_FREE do {                    \
        if (testnormalization != NULL) {      \
            cudaFree(testnormalization);      \
            testnormalization = NULL;         \
        }                                     \
        if (bigmem != NULL) {                 \
            cudaFree(bigmem);                 \
            bigmem = NULL;                    \
        }                                     \
        if (errmap != NULL) {                 \
            cudaFree(errmap);                 \
        }                                     \
        if (cormapcpu != NULL) {              \
            delete [] cormapcpu;              \
            cormapcpu = NULL;                 \
        }                                     \
    } while (0)                               \

// 成员方法：imageMatch（用给定图像及不同旋转角对待匹配的图像进行匹配）
__host__ int ImageMatch::imageMatch(Image *matchimage, MatchRes *matchres, 
                                    DirectedRect *dirrect)
{
    int errcode;          // 局部变量，错误码
    cudaError_t cudaerr;  // 局部变量，CUDA 调用返回错误码
    dim3 gridsize;
    dim3 blocksize;

    // 检查旋转表是否为空
    if (rotateTable == NULL)
        return NULL_POINTER;

    // 检查 TEMPLATE 图像数组，待匹配图像以及存放匹配结果的 matchres 是否为空
    if (matchimage == NULL || matchres == NULL)
        return NULL_POINTER;

    // 初始化旋转表，计算给定的坐标点集对应的旋转表
    if (rotateTable->getCurrentState() == NULL_RTT) {
        errcode = rotateTable->initRotateTable();
        if (errcode != NO_ERROR) {
            return errcode;
        }
    }

    // 将 TEST 图像数据拷贝到 Device 内存中
    errcode = ImageBasicOp::copyToCurrentDevice(matchimage);    
    if (errcode != NO_ERROR)
        return errcode;

    // 对设置的 TEMPLATE 图像进行正规化
    errcode = normalizeForTpl();
    if (errcode != NO_ERROR)
        return errcode;

    // 提取待匹配图像的 ROI 子图
    ImageCuda matchimageCud;
    errcode = ImageBasicOp::roiSubImage(matchimage, &matchimageCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算 TEST 图像的宽和高
    int testwidth = matchimageCud.imgMeta.width;
    int testheight = matchimageCud.imgMeta.height;

    // 局部变量，存储 TEST 图像正规化的结果
    float *testnormalization = NULL;  
    // 局部变量，testnormalization 的 ptich 值
    size_t testpitch;          
    // 申请一块足够大的空间，后面需要使用的空间可以直接在这里获取
    char *bigmem = NULL;
    // 在 Host 上申请一段内存空间，用来将 cormapsumgpu 中的数据拷贝到 Host 上
    float *cormapcpu = NULL; 
    // 申请临时变量，用来存储临时的 errmap 空间和 pitch
    int *errmap = NULL;

    // 为 testnormalization 申请 Device 空间
    cudaerr = cudaMallocPitch((void **)&testnormalization, &testpitch, 
                              testwidth * sizeof (float), testheight);
    if (cudaerr != cudaSuccess)
        return CUDA_ERROR;

    // 创建一个正规化操作的对象
    Normalization normal(3);
    // 对 TEST 图像进行正规化
    errcode = normal.normalize(&(matchimageCud.imgMeta), testnormalization, 
                               testpitch, testwidth, testheight, false);
    if (errcode != NO_ERROR) { 
        FAIL_MEM_FREE;
        return errcode;
    }

    // 计算旋转角的数量
    int anglecount = rotateTable->getAngleCount();  

    int offsetx, offsety;  // 记录摄动中心的偏移良
    // 距离摄动中心的偏移量
    offsetx = dx - dWidth / 2;
    offsety = dy - dHeight / 2;

    // bigmem 的游标，用来指定剩余内存的起始地址
    char *cursor;
    cudaerr = cudaMalloc((void **)&bigmem,
                         tplCount * sizeof (float *) + 
                         tplCount * sizeof (size_t) + 
                         2 * dWidth * dHeight * tplCount * anglecount * 
                         sizeof (float));
    // 判断是否申请成功，若失败，释放之前的空间，防止内存泄漏，然后返回错误
    if (cudaerr != cudaSuccess) {
        FAIL_MEM_FREE;
        return CUDA_ERROR;
    }
    // 游标初始指向 bigmem
    cursor = bigmem;

    // 用来存储一组 TEMPLATE 正规化的结果的指针，指向 Device 
    float **tplnormalizationCud;
    // 存储一组 TEMPLATE 正规化结果的 ptich 值，指向 Device
    size_t *tplpitchCud;

    // 从 bigmem 中获取内存空间
    tplnormalizationCud = (float **)cursor;
    // 更新游标的值
    cursor += tplCount * sizeof (float *);
    
    // 将每个 TEMPLATE 的正规化的指针拷贝到 Device 内存中
    cudaerr = cudaMemcpy(tplnormalizationCud, tplNormalization, 
                         sizeof (float *) * tplCount, cudaMemcpyHostToDevice);
    // 若拷贝失败，则释放先前申请的空间，防止内存泄漏，然后返回错误
    if (cudaerr != cudaSuccess) {
        FAIL_MEM_FREE;
        return CUDA_ERROR;
    }

    // 从 bigmem 中获取一块内存空间
    tplpitchCud = (size_t *)cursor;
    // 更新游标的值
    cursor += tplCount * sizeof (size_t);

    // 将 ptich 拷贝到 Device 内存空间
    cudaerr = cudaMemcpy(tplpitchCud, pitch, sizeof (size_t) * tplCount, 
                         cudaMemcpyHostToDevice);
    // 若拷贝失败，则释放先前申请的空间，防止内存泄漏，然后返回错误
    if (cudaerr != cudaSuccess) {
        FAIL_MEM_FREE;
        return CUDA_ERROR;
    }

    // 用来存储摄动范围每个点与每个 TEMPLATE 的不同旋转角匹配得到的相关系数
    // 这里先存储的是摄动范围的每个点对每个 TEMPLATE 的匹配结果，然后是同一个旋
    // 转角的 TEMPLATE，最后是不同的旋转角
    float *cormapgpu = NULL;
    // 在 bigmem 中获取内存空间
    cormapgpu = (float *)cursor;
    // 更新游标的值
    cursor += dWidth * dHeight * tplCount * anglecount * sizeof (float);

    // 计算 TEMPLATE 的偏移量
    int tploffx = tplWidth / 2;
    int tploffy = tplHeight / 2;

    // 计算线程块的尺寸 
    // block 使用默认的线程尺寸
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = 1;
    // 采用基本的分块方案
    gridsize.x = (dWidth + blocksize.x - 1) / blocksize.x;
    gridsize.y = (dHeight + blocksize.y - 1) / blocksize.y;
    // 第三维表示所有的 TEMPLATE，其中包括所有的旋转角度
    gridsize.z = anglecount * tplCount;

    // 调用匹配函数对每个 TEMPLATE 在摄动范围内进行匹配
    _matchKer<<<gridsize, blocksize>>>(tplnormalizationCud, tplpitchCud, 
                                       tplCount, tplWidth, tplHeight,
                                       testnormalization, testpitch,
                                       testwidth, testheight,
                                       *rotateTable, cormapgpu, 
                                       offsetx, offsety, dWidth, dHeight,
                                       tploffx, tploffy);

    // 若 Kernel 函数执行失败，则释放先前申请的内存空间，防止内存泄漏，然后返回
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_MEM_FREE;
        return CUDA_ERROR;
    }

    // 用来存储每个点在邻域内的相关系数的和
    float *cormapsumgpu = NULL;
    // 从 bigmem 中获取内存空间
    cormapsumgpu = (float *)cursor;

    // 调用 Kernel 函数对每个点求得的相关系数在 scope 邻域内求和，结果存放在
    // cormapsumgpu 指向的内存中
    _calCorMapSumKer<<<gridsize, blocksize>>>(cormapgpu, dWidth, dHeight, 
                                              scope, cormapsumgpu);

    // 若 Kernel 函数执行失败，则释放先前申请的内存空间，防止内存泄漏，然后返回
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_MEM_FREE;
        return CUDA_ERROR;
    }

    // 在 Host 上申请一段内存空间，用来将 cormapsumgpu 中的数据拷贝到 Host 上
    cormapcpu = new float[dWidth * dHeight * tplCount * anglecount];
    // 如果内存申请失败，则释放之前申请的空间，然后返回错误
    if (cormapcpu == NULL) {
        FAIL_MEM_FREE;
        return OUT_OF_MEM;
    }

    // 将 cormapsumgpu 中的数据拷贝到 cormapcpu 中 
    cudaerr = cudaMemcpy(cormapcpu, cormapsumgpu, 
                         dWidth * dHeight * tplCount * anglecount * 
                         sizeof (float), cudaMemcpyDeviceToHost);
    // 若拷贝失败，则释放先前申请的空间，防止内存泄漏，然后返回错误
    if (cudaerr != cudaSuccess) {
        FAIL_MEM_FREE;
        return CUDA_ERROR;
    }

    // 获取 cormapcpu 中最大值的索引
    int maxindex = _getCormapMaxIndex(cormapcpu, 
                                      dWidth * dHeight * tplCount * anglecount);

    // 计算最佳匹配的旋转
    int angleindex = maxindex / (dWidth * dHeight) / tplCount;
    matchres->angle = rotateTable->getAngleVal(angleindex);
    // 计算最佳匹配的 TEMPLATE 的索引
    matchres->tplIndex = (maxindex / (dWidth * dHeight)) % tplCount;
    // 计算最佳匹配的 TEST 上的横坐标
    matchres->matchX = maxindex % dWidth + offsetx;
    // 计算最佳匹配的 TEST 上的纵坐标
    matchres->matchY = (maxindex % (dWidth * dHeight)) / dWidth + offsety;
    // 计算最佳匹配时的相关系数
    matchres->coefficient = cormapcpu[maxindex] / 
                            (scope * scope);

    // 如果 errMap 不为空，则进行局部异常检查
    if (dirrect != NULL) {

        size_t errmappitch;
        // errmap 的宽和高
        int errMapWidth = testwidth, errMapHeight = testheight;
        
        // 在 Device 端创建 errMap 空间
        cudaerr = cudaMallocPitch((void **)&errmap, &errmappitch, 
                                  sizeof (int) * errMapWidth, errMapHeight);
        // 若创建失败，则释放空间，然后返回
        if (cudaerr != cudaSuccess) {
            FAIL_MEM_FREE;
            return CUDA_ERROR;    
        }
        // 将 errmap 内的数据置 0
        cudaMemset2D(errmap, errmappitch, 0, sizeof (int) * errMapWidth, 
                     errMapHeight);

        // 先计算线程块的大小
        gridsize.x = (tplWidth + blocksize.x - 1) / blocksize.x;
        gridsize.y = (tplHeight + blocksize.y - 1) / blocksize.y;
        // 进行局部异常检查
        _localCheckErrKer<<<gridsize, blocksize>>>(
                         tplNormalization[matchres->tplIndex],
                         pitch[matchres->tplIndex], tplWidth, tplHeight,
                         testnormalization, testpitch, testwidth, testheight,
                         *rotateTable, errmap, errmappitch, errMapWidth,
                         errMapHeight, errThreshold, matchres->matchX,
                         matchres->matchY, matchres->angle, tploffx, tploffy);
        // 判断 Kernel 是否发生错误，若有错误，则释放空间，然后返回错误
        if (cudaGetLastError() != cudaSuccess) {
            FAIL_MEM_FREE;
            return CUDA_ERROR;
        }
        // 获取 errmap 的最小有向四边形
        errcode = _getDirectRectForErrMap(errmap, errmappitch,
                                          errMapWidth, errMapHeight,
                                          errWinWidth, errWinHeight,
                                          errWinThreshold, dirrect);
        if (errcode != NO_ERROR) {
            FAIL_MEM_FREE;
            return errcode;
        }
    }
    // 释放之前申请的空间，防止内存泄露
    FAIL_MEM_FREE;

    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}
#undef FAIL_MEM_FREE
