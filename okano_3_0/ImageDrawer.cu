// ImageDrawer.cu
// 在图像上绘制相应的几何构建。

#include "ImageDrawer.h"


// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：DEF_BLOCK_1D
// 定义了默认的 1D Block 尺寸。
#define DEF_BLOCK_1D  512


// Kernel 函数：_brushAllImageKer（涂满整幅图像）
// 将整幅图像使用同一种颜色涂满。
static __global__ void     // Kernel 函数无返回值
_brushAllImageKer(
        ImageCuda img,     // 待涂刷的图像
        unsigned char clr  // 颜色值
);

// Kernel 函数：_drawLinesKer（绘制直线）
// 根据坐标点集中给定的直线参数，在图像上绘制直线。如果 color 小于 0，则使用坐
// 标点集中的附属数据做为亮度值（颜色值）；如果 color 大于等于 0，则直接使用
// color 所指定的颜色。
static __global__ void      // Kernel 函数无返回值
_drawLinesKer(
        ImageCuda img,      // 待绘制直线的图像
        CoordiSetCuda cst,  // 用坐标点集表示的直线参数
        int color           // 绘图使用的颜色值
);

// Kernel 函数：_drawLinesKer（绘制直线）
// 根据坐标点集中给定的椭圆参数，在图像上绘制椭圆。如果 color 小于 0，则使用坐
// 标点集中的附属数据做为亮度值（颜色值）；如果 color 大于等于 0，则直接使用
// color 所指定的颜色。
static __global__ void      // Kernel 函数无返回值
_drawEllipseKer(
        ImageCuda img,      // 待绘制直线的椭圆
        CoordiSetCuda cst,  // 用坐标点集表示的椭圆参数
        int color           // 绘图使用的颜色值
);


// Kernel 函数：_brushAllImageKer（涂满整幅图像）
static __global__ void _brushAllImageKer(ImageCuda img, unsigned char clr)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= img.imgMeta.width || r >= img.imgMeta.height)
        return;
    
    // 计算第一个像素点对应的图像数据数组下标。
    int idx = r * img.pitchBytes + c;

    // 为第一个像素点赋值。
    img.imgMeta.imgData[idx] = clr;

    // 处理剩下的三个像素点。
    for (int i = 1; i < 4; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++r >= img.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        idx += img.pitchBytes;

        // 为当前像素点赋值。
        img.imgMeta.imgData[idx] = clr;
    }
}

// Host 成员方法：brushAllImage（涂满整幅图像）
__host__ int ImageDrawer::brushAllImage(Image *img)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (img == NULL)
        return NULL_POINTER;

    // 如果算法 CLASS 的背景色为透明色，则不需要进行任何处理，直接返回即可。
    if (this->brushColor == IDRAW_TRANSPARENT)
        return NO_ERROR;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(img);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda subimgCud;
    errcode = ImageBasicOp::roiSubImage(img, &subimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (subimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (subimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 调用核函数完成计算。
    _brushAllImageKer<<<gridsize, blocksize>>>(subimgCud, this->brushColor);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
			
    // 处理完毕，退出。	
    return NO_ERROR;
}

static __global__ void _drawLinesKer(ImageCuda img, CoordiSetCuda cst,
                                     int color)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 计算当前线程所要处理的直线的编号。
    int lineidx = blockIdx.z;

    // 判断当前处理的直线是否已经越界。
    if (2 * lineidx + 1 >= cst.tplMeta.count)
        return;

    // 共享内存声明。使用共享内存预先读出来一些数据，这样可以在一定程度上加快 
    // Kernel 的执行速度。
    __shared__ int lineptShd[5];       // 直线的两个端点。
    int *curcolorShd = &lineptShd[4];  // 绘制当前直线所用的颜色。

    // 初始化直线的两个端点。使用当前 Block 的前四个 Thread 读取两个坐标，一共
    // 四个整形数据。
    if (threadIdx.x < 4 && threadIdx.y == 0) {
       lineptShd[threadIdx.x] = 
               cst.tplMeta.tplData[4 * lineidx + threadIdx.x];
    }

    // 初始化当前直线的颜色。
    if (threadIdx.x == 0 && threadIdx.y == 0) {
       // 如果颜色值小于 0，说明当前使用了动态颜色模式，此时从坐标点集的附属数
       // 据中读取颜色值。否则，直接使用参数的颜色值作为颜色值。
       if (color < 0)
           curcolorShd[0] = 
                   ((int)(fabs(cst.attachedData[2 * lineidx]) * 255)) % 256;
       else
           curcolorShd[0] = color % 256;
    }

    // 同步 Block 内的所有 Thread，使得上面的初始化过程对所有的 Thread 可见。
    __syncthreads();

    // 如果当前处理的坐标不在图像所表示的范围内，则直接退出。
    if (c >= img.imgMeta.width || r >= img.imgMeta.height)
        return;

    // 计算当前 Thread 对应的图像下标。
    int idx = r * img.pitchBytes + c;

    // 将 Shared Memory 中的数据读取到寄存器中，以使得接下来的计算速度更快。
    int pt0x = lineptShd[0];
    int pt0y = lineptShd[1];
    int pt1x = lineptShd[2];
    int pt1y = lineptShd[3];
    unsigned char curcolor = curcolorShd[0];

    // 针对不同的情况，处理直线。主要的思路就是判断当前点是否在直线上，如果在直
    // 线上，则将当前点的像素值赋值为前景色值；否则什么都不做。显然，相同的 
    // Block 处理的是相同的直线，因此对于该 if-else 语句会进入相同的分支，所
    // 以，这个分支语句不会导致分歧执行，也不会产生额外的性能下降。
    if (pt0x == pt1x) {
        // 当直线平行于 y 轴，则判断当前点是否跟直线端点具有同样的 x 坐标，如果
        // 是，并且 y 坐标在端点的范围内，则绘制点，否则什么都不做。

        // 对于当前点和端点具有不同 x 坐标的像素点，则直接退出。
        if (c != pt0x)
            return;

        // 计算两个端点之间的 y 坐标的范围。
        int miny = min(pt0y, pt1y);
        int maxy = max(pt0y, pt1y);

        // 对于 y 坐标范围已经不在图像像素范围内的情况，则直接退出。
        if (maxy < 0 || miny >= img.imgMeta.height)
            return;

        // 检查当前处理的第一个点是否在 y 坐标范围内，如果在，则在该点绘制颜色
        // 点。
        if (r >= miny && r <= maxy)
            img.imgMeta.imgData[idx] = curcolor;

        // 处理剩余的三个点。
        for (int i = 1; i < 4; i++) {
            // 检查这些点是否越界。
            if (++r >= img.imgMeta.height)
                return;

            // 调整下标值，根据上下两点之间的位置关系，可以从前一下标值推算出下
            // 一个下标值。
            idx += img.pitchBytes;

            // 检查当前处理的剩余三个点是否在 y 坐标范围内，如果在，则在该点绘
            // 制颜色点。
            if (r >= miny && r <= maxy)
                img.imgMeta.imgData[idx] = curcolor;
        }
    } else if (pt0y == pt1y) {
        // 当直线平行于 x 轴，则判断当前点是否跟直线端点具有同样的 y 坐标，如果
        // 是，并且 x 坐标在端点的范围内，则绘制点，否则什么都不做。

        // 计算两个端点之间的 x 坐标的范围。
        int minx = min(pt0x, pt1x);
        int maxx = max(pt0x, pt1x);

        // 对于当前点的 x 坐标不在两个端点的范围内的像素点，则直接退出。
        if (c < minx || c > maxx)
            return;

        // 对于当前点的 y 坐标等于端点的 y 坐标，则在该点绘制颜色值。由于在该点
        // 绘制了颜色后，可以断定其后就不会再有点可以绘制了，因此绘制后直接返
        // 回。
        if (r == pt0y) {
            img.imgMeta.imgData[idx] = curcolor;
            return;
        }

        // 处理剩余的三个点。
        for (int i = 1; i < 4; i++) {
            // 检查这些点是否越界。
            if (++r >= img.imgMeta.height)
                return;

            // 调整下标值，根据上下两点之间的位置关系，可以从前一下标值推算出下
            // 一个下标值。
            idx += img.pitchBytes;

            // 对于当前点的 y 坐标等于端点的 y 坐标，则在该点绘制颜色值。由于在
            // 该点绘制了颜色后，可以断定其后就不会再有点可以绘制了，因此绘制后
            // 直接返回。
            if (r == pt0y) {
                img.imgMeta.imgData[idx] = curcolor;
                return;
            }
        }   
    } else {
        // 对于其他情况，可以直接按照直线方程进行判断。

        // 计算两个端点之间的 x 坐标的范围。
        int minx = min(pt0x, pt1x);
        int maxx = max(pt0x, pt1x);

        // 计算两个端点之间的 y 坐标的范围。
        int miny = min(pt0y, pt1y);
        int maxy = max(pt0y, pt1y);

        // 对于当前点的 x 坐标不在不在两个端点的范围内的像素点，则直接退出。
        if (c < minx || c > maxx)
            return;

        // 计算直线关于 x 轴的斜率，预先计算斜率可以在后面的计算过程中重复利
        // 用，以减少计算。
        float dydx = (float)(pt1y - pt0y) / (pt1x - pt0x);

        // 计算直线方程。这里如果斜率的绝对值大于 1 时（坡度大于 45 度），按照
        // 关于 y 的方程进行计算，这样函数值变化和图像栅格坐标的变化是在同数量
        // 级的。
        float fx, detfx;
        if (fabs(dydx) <= 1.0f ) {
            fx = dydx * (c - pt0x) + pt0y - r;
            detfx = -1.0f;
        } else {
            dydx = 1.0f / dydx;
            fx = dydx * (r - pt0y) + pt0x - c;
            detfx = dydx;
        }

        // 如果点落在直线上（即方程等于 0）则绘制该点的颜色值。这里判断 0.5 是
        // 考虑到图像的栅格坐标。
        if (fabs(fx) <= 0.5f && r >= miny && r <= maxy)
            img.imgMeta.imgData[idx] = curcolor;

        // 处理剩余的三个点。
        for (int i = 1; i < 4; i++) {
            // 检查这些点是否越界。
            if (++r >= img.imgMeta.height)
                return;

            // 调整下标值和函数值，根据上下两点之间的位置关系，可以从前一下标值
            // 推算出下一个下标值。
            fx += detfx;
            idx += img.pitchBytes;

            // 如果点落在直线上（即方程等于 0）则绘制该点的颜色值。这里判断
            // 0.5 是考虑到图像的栅格坐标。
            if (fabs(fx) <= 0.5f && r >= miny && r <= maxy)
                img.imgMeta.imgData[idx] = curcolor;
        }
    }
}

// Host 成员方法：drawLines（绘制直线）
__host__ int ImageDrawer::drawLines(Image *img, CoordiSet *cst)
{
    // 判断参数中的图像和坐标点集是否为 NULL。
    if (img == NULL || cst == NULL)
        return NULL_POINTER;

    // 如果坐标点击中含有少于 2 个点则无法完成之间绘制，因此报错退出。
    if (cst->count < 2)
        return INVALID_DATA;

    // 计算绘制索要使用的颜色。
    int curcolor = this->lineColor;
    if (this->colorMode == IDRAW_CM_STATIC_COLOR) {
        // 如果是静态着色，且颜色值为透明色，则直接退出。
        if (curcolor == IDRAW_TRANSPARENT)
            return NO_ERROR;
    } else if (this->colorMode == IDRAW_CM_DYNAMIC_COLOR) {
        // 对于动态绘图模式，则将颜色值赋值成一个负数，这样 Kernel 函数就能知道
        // 当前使用的是动态着色模式。
        curcolor = -1;
    }

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(img);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda subimgCud;
    errcode = ImageBasicOp::roiSubImage(img, &subimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 将坐标点集拷贝到 Device 内存中。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 取出坐标点集对应的 CoordiSetCuda 型数据。
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。这里使用 Block 的 z 维
    // 度表示不同的直线。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    blocksize.z = 1;
    gridsize.x = (subimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (subimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);
    gridsize.z = cst->count / 2;

    // 调用 Kernel 完成计算。
    _drawLinesKer<<<gridsize, blocksize>>>(subimgCud, *cstCud, curcolor);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。 
    return NO_ERROR;
}

// Kernel 函数：_extendLineCstKer（扩展坐标点集以绘制连续直线）
static __global__ void _extendLineCstKer(CoordiSetCuda incst, 
                                         CoordiSetCuda outcst)
{
    // 计算输出坐标集的下标，本函数为每个输出坐标点配备一个 Thread。
    int outidx = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果输出点对应的是越界数据，则直接退出。
    if (outidx >= outcst.tplMeta.count)
        return;

    // 计算输入下标。根据输出数组中各点对应于输入数组中的各点的关系，有【0】
    // 【1】【1】【2】【2】【3】【3】等等，据此推演出输出坐标点和输入坐标点之间
    // 的下标关系。
    int inidx = (outidx + 1) / 2;

    // 如果计算出来的输入下标是越界的，则对计算出来的输入下标取模（之所以单独提
    // 出来进行取模操作，是应为，取模操作比较耗时，且多数情况下不会出现越界的情
    // 况）。
    if (inidx >= incst.tplMeta.count)
        inidx %= incst.tplMeta.count;

    // 将对应的坐标点坐标和附属数据从输入坐标点集中拷贝到输出坐标点集中。
    outcst.tplMeta.tplData[2 * outidx] = 
            incst.tplMeta.tplData[2 * inidx];
    outcst.tplMeta.tplData[2 * outidx + 1] = 
            incst.tplMeta.tplData[2 * inidx + 1];
    outcst.attachedData[outidx] = incst.attachedData[inidx];
}

// 宏：FAIL_DRAWTRACE_FREE
// 当下面的函数失败退出时，负责清理内存，防止内存泄漏。
#define FAIL_DRAWTRACE_FREE  do {                       \
        if (extcst != NULL)                             \
            CoordiSetBasicOp::deleteCoordiSet(extcst);  \
    } while (0)

// Host 成员方法：drawTrace（绘制连续直线）
__host__ int ImageDrawer::drawTrace(Image *img, CoordiSet *cst, bool circled)
{
    // 检查参数中的指针是否为 NULL。
    if (img == NULL || cst == NULL)
        return NULL_POINTER;

    // 如果 cst 的坐标点少于 3 个，则只能绘制 0 条或 1 条线，因此可直接调用绘制
    // 直线算法处理。
    if (cst->count < 3)
        return this->drawLines(img, cst);

    // 局部变量，错误码
    int errcode;

    // 申请扩展后的坐标点集，由于该函数最终要调用 drawLines 完成绘图工作，因此
    // 首先需要根据输入的坐标点集生成所需要的扩展型坐标点集。
    CoordiSet *extcst = NULL;
    errcode = CoordiSetBasicOp::newCoordiSet(&extcst);
    if (errcode != NO_ERROR) {
        FAIL_DRAWTRACE_FREE;
        return errcode;
    }

    // 计算扩展坐标点集中点的数量，并申请相应大小的内存空间。对于环形绘图相对于
    // 蛇形绘图会多一条直线。
    int extcstcnt = (circled ? 2 * cst->count : 2 * (cst->count - 1));
    errcode = CoordiSetBasicOp::makeAtCurrentDevice(extcst, extcstcnt);
    if (errcode != NO_ERROR) {
        FAIL_DRAWTRACE_FREE;
        return errcode;
    }

    // 将输入的坐标点集拷贝当当前 Device。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(cst);
    if (errcode != NO_ERROR) {
        FAIL_DRAWTRACE_FREE;
        return errcode;
    }

    // 获取两个坐标点集的 CUDA 型数据指针。
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);
    CoordiSetCuda *extcstCud = COORDISET_CUDA(extcst);

    // 计算启动 Kernel 所需要的 Block 尺寸和数量。
    size_t blocksize = DEF_BLOCK_1D;
    size_t gridsize = (extcstcnt + DEF_BLOCK_1D - 1) / DEF_BLOCK_1D;

    // 启动 Kernel 完成扩展坐标集的计算操作。
    _extendLineCstKer<<<gridsize, blocksize>>>(*cstCud, *extcstCud);

    // 检查 Kernel 执行是否正确。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_DRAWTRACE_FREE;
        return errcode;
    }

    // 使用扩展坐标集绘制连续直线。
    errcode = this->drawLines(img, extcst);
    if (errcode != NO_ERROR) {
        FAIL_DRAWTRACE_FREE;
        return errcode;
    }

    // 处理完毕，清除临时使用的扩展坐标点集。
    CoordiSetBasicOp::deleteCoordiSet(extcst);

    // 处理完毕，退出。
    return NO_ERROR;
}

static __global__ void _drawEllipseKer(ImageCuda img, CoordiSetCuda cst,
                                       int color)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 共享内存声明。使用共享内存预先读出来一些数据，这样可以在一定程度上加快 
    // Kernel 的执行速度。
    __shared__ int ellipseptShd[5];          // 椭圆的外界矩形的左上顶点和右下
                                             // 顶点的坐标
    __shared__ float ellipsecenterptShd[2];  // 椭圆的中心坐标
    __shared__ float ellipseradiuShd[2];     // 椭圆的半径 rx，ry
    int *curcolorShd = &ellipseptShd[4];     // 绘制当前椭圆所用的颜色。

    // 初始化椭圆的外界矩形的 2 个顶点。使用前 4 个 Thread 读取 2 个坐标，一共
    // 4 个整型数据。
    if (threadIdx.x < 4 && threadIdx.y == 0) {
        ellipseptShd[threadIdx.x] = 
               cst.tplMeta.tplData[threadIdx.x];
    }

    // 使用第 5 个 Thread 初始化当前椭圆的颜色。
    if (threadIdx.x == 4 && threadIdx.y == 0) {
       // 如果颜色值小于 0，说明当前使用了动态颜色模式，此时从坐标点集的附属数
       // 据中读取颜色值。否则，直接使用参数的颜色值作为颜色值。
       if (color < 0)
           curcolorShd[0] = 
                   ((int)(fabs(cst.attachedData[0]) * 255)) % 256;
       else
           curcolorShd[0] = color % 256;
    }

    // 同步 Block 内的所有 Thread，使得上面的初始化过程对所有的 Thread 可见。
    __syncthreads();

    // 使用前 2 个 Thread 初始化椭圆的中心坐标
    if (threadIdx.x < 2 && threadIdx.y == 0) {
        // 初始化椭圆的中心坐标
        ellipsecenterptShd[threadIdx.x] =
                (float)(ellipseptShd [threadIdx.x] +
                        ellipseptShd [threadIdx.x + 2]) / 2;
    }

    // 使用第 3 个和第 4 个 Thread 初始化椭圆的半径
    if (threadIdx.x >=2 && threadIdx.x <4 && threadIdx.y == 0) {
        ellipseradiuShd[threadIdx.x - 2] =
                (float)(ellipseptShd [threadIdx.x] -
                        ellipseptShd [threadIdx.x - 2]) / 2;
    }

    // 同步 Block 内的所有 Thread，使得上面的初始化过程对所有的 Thread 可见。
    __syncthreads();

    // 如果当前处理的坐标不在图像所表示的范围内，则直接退出。
    if (c >= img.imgMeta.width || r >= img.imgMeta.height)
        return;

    // 计算当前 Thread 对应的图像下标。
    int idx = r * img.pitchBytes + c;

    // 将 Shared Memory 中的数据读取到寄存器中，以使得接下来的计算速度更快。
    int pt0x = ellipseptShd[0];
    int pt0y = ellipseptShd[1];
    int pt1x = ellipseptShd[2];
    int pt1y = ellipseptShd[3];
    float rx = ellipseradiuShd[0];
    float ry = ellipseradiuShd[1];
    float xc = ellipsecenterptShd[0];
    float yc = ellipsecenterptShd[1];
    unsigned char curcolor = curcolorShd[0];

    // 计算椭圆的 x 坐标的范围。
    int minx = min(pt0x, pt1x);
    int maxx = max(pt0x, pt1x);

    // 计算椭圆的 y 坐标的范围。
    int miny = min(pt0y, pt1y);
    int maxy = max(pt0y, pt1y);

    // 针对不同的情况，处理椭圆。主要的思路就是判断当前点是否在椭圆上，如果在
    // 椭圆上，则将当前点的像素值赋值为前景色值；否则什么都不做。
    if (minx == maxx || miny == maxy) {
        // 当输入点集在一条直线上时，退出程序。
        return;
    } else {
        // 对于其他情况，可以直接按照椭圆方程进行判断。

        // 对于当前点的 x，y 坐标，若不在椭圆的左上四分之一范围内，则直接退出。
        if (c < minx || c > xc || r < miny || r > yc)
            return;

        // 计算椭圆方程。判断点（c,r）的斜率是否大于 1 ，若大于 1 则计算
        // 点（c,r + 1）的方程的值，否则计算点（c + 1,r）的方程的值。
        float fx1,fx2;
        int r1,c1;
        if (ry * ry * (c - xc) >= rx * rx * (r - yc)) {
            r1 = r + 1;
            c1 = c;
        } else {
            r1 = r;
            c1 = c + 1;
        }
        fx1 = (float)ry * ry * (c - xc)* (c - xc) +
              rx * rx * (r - yc)* (r - yc) - rx * rx * ry * ry;
        fx2 = (float)ry * ry * (c1 - xc)* (c1 - xc) +
              rx * rx * (r1 - yc)* (r1 - yc) - rx * rx * ry * ry;

        // 如果相邻两点分别落在椭圆内和椭圆外（即方程的值一正一负）则绘制距离
        // 椭圆较近的点（即方程值的绝对值较小的点）及该点关于直线 x = xc，
        // y = yc 及关于点（xc,yc）对称的 3 个点的颜色值。
        if(fx1 >= 0 && fx2 < 0) {
            if(fabs(fx1) < fabs(fx2)) {
                img.imgMeta.imgData[idx] = curcolor;
                // 关于直线 x = xc 对称的点
                img.imgMeta.imgData[r * img.pitchBytes + (int)(2 * xc) - c] =
                        curcolor;
                // 关于直线 y = yc 对称的点
                img.imgMeta.imgData[((int)(2 * yc) - r) * img.pitchBytes + c] =
                        curcolor;
                // 关于点（xc,yc）对称的点
                img.imgMeta.imgData[(int)((2 * yc - r) * img.pitchBytes +
                                          2 * xc - c)] = curcolor;
            } else {
                img.imgMeta.imgData[r1 * img.pitchBytes + c1] = curcolor;
                // 关于直线 x = xc 对称的点
                img.imgMeta.imgData[r1 * img.pitchBytes +
                                    ((int)(2 * xc) - c1)] = curcolor;
                // 关于直线 y = yc 对称的点
                img.imgMeta.imgData[(int)((2 * yc) - r1) *
                                    img.pitchBytes + c1] = curcolor;
                // 关于点（xc,yc）对称的点
                img.imgMeta.imgData[(int)((2 * yc - r1) *img.pitchBytes +
                                          2 * xc - c1)] = curcolor;
            }
        }

        // 处理剩余的三个点。
        for (int i = 1; i < 4; i++) {
            // 检查这些点是否越界。
            if (++r > yc)
                return;

            // 计算椭圆方程。判断点（c,r）的斜率是否大于 1 ，若大于 1 则计算
            // 点（c,r + 1）的方程的值，否则计算点（c + 1,r）的方程的值。
            if(ry * ry * (c - xc) >= rx * rx * (r - yc)) {
                r1 = r + 1;
                c1 = c;
            } else {
                r1 = r;
                c1 = c + 1;
            }
            fx1 = (float)ry * ry * (c - xc)* (c - xc) +
                  rx * rx * (r - yc)* (r - yc) - rx * rx * ry * ry;
            fx2 = (float)ry * ry * (c1 - xc)* (c1 - xc) +
                  rx * rx * (r1 - yc)* (r1 - yc) - rx * rx * ry * ry;

            idx += img.pitchBytes;

            // 如果相邻两点分别落在椭圆内和椭圆外（即方程的值一正一负）则绘制
            // 距离椭圆较近的点（即方程值的绝对值较小的点）及该点关于直线
            // x = xc，y = yc 及关于点（xc,yc）对称的 3 个点的颜色值。
            if (fx1 >= 0 && fx2 < 0) {
                if(fabs(fx1) < fabs(fx2)) {
                    img.imgMeta.imgData[idx] = curcolor;
                    // 关于直线 x = xc 对称的点
                    img.imgMeta.imgData[r * img.pitchBytes + (int)(2 * xc) -
                                        c] =curcolor;
                    // 关于直线 y = yc 对称的点
                    img.imgMeta.imgData[((int)(2 * yc) - r) * img.pitchBytes +
                                        c] = curcolor;
                    // 关于点（xc,yc）对称的点
                    img.imgMeta.imgData[(int)((2 * yc - r) * img.pitchBytes +
                                              2 * xc - c)] = curcolor;
                } else {
                    img.imgMeta.imgData[r1 * img.pitchBytes + c1] = curcolor;
                    // 关于直线 x = xc 对称的点
                    img.imgMeta.imgData[r1 * img.pitchBytes + (int)(2 * xc) -
                                        c1] = curcolor;
                    // 关于直线 y = yc 对称的点
                    img.imgMeta.imgData[((int)(2 * yc) - r1) * img.pitchBytes +
                                        c1] = curcolor;
                    // 关于点（xc,yc）对称的点
                    img.imgMeta.imgData[(int)((2 * yc - r1) * img.pitchBytes +
                                              2 * xc - c1)] = curcolor;
                }
            }
        }
    }
}

// Host 成员方法：drawEllipse（绘制椭圆）
__host__ int ImageDrawer::drawEllipse(Image *img, CoordiSet *cst)
{
    // 判断参数中的图像和坐标点集是否为 NULL。
    if (img == NULL || cst == NULL)
        return NULL_POINTER;

    // 如果坐标点击中含有少于 2 个点则无法完成椭圆绘制，因此报错退出。
    if (cst->count < 2)
        return INVALID_DATA;

    // 计算绘制所要使用的颜色。
    int curcolor = this->lineColor;
    if (this->colorMode == IDRAW_CM_STATIC_COLOR) {
        // 如果是静态着色，且颜色值为透明色，则直接退出。
        if (curcolor == IDRAW_TRANSPARENT)
            return NO_ERROR;
    } else if (this->colorMode == IDRAW_CM_DYNAMIC_COLOR) {
        // 对于动态绘图模式，则将颜色值赋值成一个负数，这样 Kernel 函数就能知道
        // 当前使用的是动态着色模式。
        curcolor = -1;
    }

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(img);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda subimgCud;
    errcode = ImageBasicOp::roiSubImage(img, &subimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 将坐标点集拷贝到 Device 内存中。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(cst);
    if (errcode != NO_ERROR)
        return errcode;

    // 取出坐标点集对应的 CoordiSetCuda 型数据。
    CoordiSetCuda *cstCud = COORDISET_CUDA(cst);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (subimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (subimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 调用 Kernel 完成计算。
    _drawEllipseKer<<<gridsize, blocksize>>>(subimgCud, *cstCud, curcolor);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。 
    return NO_ERROR;
}