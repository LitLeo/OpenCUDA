// SuperSmooth.cu
// 对图像进行超平滑处理。

#include "SuperSmooth.h"
#include "Template.h"
#include "TemplateFactory.h"

#include <iostream>
using namespace std;
#include <stdio.h>
// 宏：LOCALCLUSTER_DEF_BLOCK_X 和 LOCALCLUSTER_DEF_BLOCK_Y
// 以及 LOCALCLUSTER_DEF_BLOCK_Z
// 定义了第 1 个 kernel 默认的线程块的尺寸，本算法中采用了三维的线程块。
#define LOCALCLUSTER_DEF_BLOCK_X  32
#define LOCALCLUSTER_DEF_BLOCK_Y   2
#define LOCALCLUSTER_DEF_BLOCK_Z   4

// 宏：AVG_DEF_BLOCK_X 和 AVG_DEF_BLOCK_Y
// 定义了第 2 个 kernel 默认的线程块的尺寸，本算法中采用了二维的线程块。
#define AVG_DEF_BLOCK_X  32
#define AVG_DEF_BLOCK_Y   8

// 宏：SYN_DEF_BLOCK_X 和 SYN_DEF_BLOCK_Y
// 定义了第 3 个 kernel 默认的线程块的尺寸，本算法中采用了二维的线程块。
#define SYN_DEF_BLOCK_X  32
#define SYN_DEF_BLOCK_Y   8

// 宏：CNTTENPERCENT
// 定义了邻域窗口 10% 的点的数目。
#define CNTTENPERCENT  121

// Device 数组：_argsLocalDev[4][4]
// 因为不同线程处理的点的坐标是不一样的，为了最大化的并行，特意提取出来一些参数，
// 有了这些参数可以增大并行化。
const int static __device__  _argsLocalDev[4][4] = {
    { 0, -1,  0,  1}, {-1,  0,  1,  0},
    {-1,  1,  1, -1}, {-1, -1,  1,  1}
};

// Kernel 函数：_localClusterKer（在点的 8 个方向上进行平滑操作）
// 在每一个点的八个方向上遍历一定数量的点，利用这些点的累加值进行算术处理，得到
// 平滑后的图像值。
static __global__ void  // Kernel 函数无返回值
_localClusterKer(
        ImageCuda inimg,       // 输入图像。
        ImageCuda outimg,      // 输出图像。
        int diffthred,         // 当前像素点和遭遇的点的像素值差相关的阈值。
        int diffcntthred,      // 当前像素点的像素值与正遭遇点的像素值差大于等于
                               // diffthred 的连续点的个数超过 diffcntthred 时停
                               // 止该方向的遍历。
        unsigned char flag[],  // 标记数组，初始值都为 0
        int ccntthred,         // 它与当前像素点的像素值与正遭遇点的像素值差小于
                               // diffthred 的点的个数有关。
        int searchscope        // LocalCluster kernel中在每一个方向上搜索的最大
                               // 范围。按照河边老师的需求该值小于 16。
);

// Kernel 函数：_minmaxAvgKer（利用点的邻域进行平滑处理）
// 对以当前计算点为中心的 window * window 范围内的点排序，计算前后各 10% 的点的
// 平均值，并以此计算当前点的最终像素值。
static __global__ void  // Kernel 函数无返回值
_minmaxAvgKer(
        ImageCuda inimg,    // 输入图像。
        ImageCuda avgimg,   // 输出图像。
        Template atemplate  // 模板，用来定位邻域点的坐标
);

// Kernel 函数：_synthImageKer（综合前两个核函数的结果，得到超平滑的结果）
// 根据前两个和函数的计算结果，得到最终的结果图像。
static __global__ void  // Kernel 函数无返回值
_synthImageKer(
        ImageCuda avgimg,     // _minmaxAvgKer 得到的临时图像。
        ImageCuda outimg,     // 输出图像。
        unsigned char flag[]  // 通过第一个 kernel 得到的标记数组，每一个点都对
                              // 应一个标记值，0 或者 1。
);

// Device 函数：_sortDev（选择排序，升序）
static __device__ int  // 若正确执行该操作，则返回 NO_ERROR。
_sortDev(
        unsigned char pixelarr[],  // 要排序的数组
        int length                 // 数组的长度
);

// Device 函数：_sortDev（选择排序，升序）
static __device__ int _sortDev(
        unsigned char pixelarr[], int length)
{
    // 判断参数是否为空。
    if (pixelarr == NULL || length <= 0)
        return INVALID_DATA;

    // 排序中用到的临时变量，记录数值值。
    unsigned char temp = 0;
    // 临时变量，用来记录每一次循环找到的最小值的下标。
    int index = 0;
    for (int i = 0; i < length; i++) {
        index = i;
        temp = pixelarr[i];
        for (int j = i + 1; j < length; j++) {
            if (temp > pixelarr[j]) {
                index = j;
                temp = pixelarr[j];
            }
        }
        if (index != i) {
            pixelarr[index] = pixelarr[i];
            pixelarr[i] = temp;
        }
    }
    // 正确执行，返回 NO_ERROR。
    return NO_ERROR;
}

// Kernel 函数：_localClusterKer（在点的 8 个方向上进行平滑操作）
static __global__ void _localClusterKer(
        ImageCuda inimg, ImageCuda outimg,
        int diffthred, int diffcntthred, unsigned char flag[], int ccntthred,
        int searchscope)
{
    // 计算当前线程在 Grid 中的位置。其中，c 对应点的横坐标，r 对应点的纵坐标。
    // z 对应的是检索的方向。采用了一个线程处理四个点的策略。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    // 申请共享内存。
    extern __shared__ unsigned char pixeltmpShd[];
    // 当前线程在线程块内部的索引
    int idxblk = threadIdx.z * blockDim.x * blockDim.y +
                 threadIdx.y * blockDim.x + threadIdx.x;

    // 当前线程需用到的共享内存，用来存放线程计算出来的该点可能像素值（可能不是
    // 最终值，因为在这些值中最接近当前点的值才被选择为最终值），并将其赋初值 0。
    // flagblkShd 用来为每一个线程标记，当 ccount > cCntThred 时将标记为 1。
    unsigned char *curpixelShd, *flagblkShd;
    if (idxblk == 0) {
        curpixelShd = pixeltmpShd;
        flagblkShd = curpixelShd + LOCALCLUSTER_DEF_BLOCK_X *
                     LOCALCLUSTER_DEF_BLOCK_Y * LOCALCLUSTER_DEF_BLOCK_Z;
    }
    flagblkShd[idxblk] = 0;
    // 块内同步。
    __syncthreads();

    // 判断点的坐标是否出界。
    if (c < searchscope || c >= inimg.imgMeta.width - searchscope ||
        r < searchscope || r >= inimg.imgMeta.height - searchscope)
        return;

    // 当前遭遇点的坐标
    int curc = c, curr = r;

    // 当前计算的第一个点的索引,取出像素值。
    int idx = r * inimg.pitchBytes + c;
    // 声明数组用来存放当前计算的 4 个点的像素值。
    unsigned char pixel = inimg.imgMeta.imgData[idx];

    // 分配共享内存，用来存放当前线程遭遇到的满足条件的点的像素值累加和。
    int sumv = 0;

    // 当前遭遇点的像素值和索引，都是临时变量在下面的计算中会用到。
    unsigned char curpixel = 0;
    int curidx = idx;

    // 临时变量。
    int ccount = 0, dcount = 0;

    // 当前点和遭遇到的点的像素值差的绝对值。
    unsigned char diff = 0 ;
    // 取出 _argsLocalDev 中的值到寄存器中，减少频繁访存带来的开销。
    int offx = _argsLocalDev[z][0];
    int offy = _argsLocalDev[z][1];

    // 从（x，y）开始向上（左、左下、左上）方向扫描。
    for (int i = 1; i <= searchscope; i++) {
        // 当前遭遇到的点的坐标。
        curc = c + offx * i;
        curr = r + offy * i;
        // 判断点的坐 标是否出界。
        if (curc < 0 || curc >= inimg.imgMeta.width ||
            curr < 0 || curr >= inimg.imgMeta.height)
            break;

        // 当前遭遇到的点的索引和像素值。
        curidx = curc + curr * inimg.pitchBytes;
        curpixel = inimg.imgMeta.imgData[curidx];
        // 计算当前点和遭遇点的像素值差的绝对值。
        diff = abs(pixel - curpixel);
        // 若 diff 小于了 diffthred 则做以下的处理。
        // 若不满足条件则跳出循环。
        if (diff < diffthred) {
            ccount++;
            sumv += curpixel;
            dcount = 0;
        } else {
            if (++dcount > diffcntthred)
                break;
        }
    }
    dcount = 0;
    offx = _argsLocalDev[z][2];
    offy = _argsLocalDev[z][3];
    // 从（x，y）开始向下（右、右上、右下）方向扫描。
    for (int i = 1; i <= searchscope; i++) {
        curc = c + offx * i;
        curr = r + offy * i;

        // 判断点的坐 标是否出界。
        if (curc < 0 || curc >= inimg.imgMeta.width ||
            curr < 0 || curr >= inimg.imgMeta.height)
            break;

        // 当前遭遇到的点的索引和像素值。
        curidx = curc + curr * inimg.pitchBytes;
        curpixel = inimg.imgMeta.imgData[curidx];

        // 计算当前点和遭遇点的像素值差的绝对值。
        diff = abs(pixel - curpixel);
        // 若 diff 小于了 diffthred 则做以下的处理。
        // 若不满足条件则跳出循环。
        if (diff < diffthred) {
            ccount++;
            sumv += curpixel;
            dcount = 0;
        } else {
            if (++dcount > diffcntthred)
                    break;
        }
    }

    if (ccount > ccntthred) {
        flag[idx] = 1;
        flagblkShd[idxblk] = 1;
        // 计算当前点的可能值，并将其存放在共享内存中。
        unsigned char tmppixel =  0;
        tmppixel = (unsigned char)(ccount  == 0 ? (pixel + 1.0f) / 2.0f :
                                   (pixel + sumv / ccount + 1.0f) / 2.0f);
        curpixelShd[idxblk] = (tmppixel > 255 ? 255 : tmppixel);
    }       
    // 块内同步
    __syncthreads();

    // 在每一个点对应的 4 个之中找到与当前点最接近的值作为该点的最终值。此处只需
    // 用一个线程来实现。
    if (z != 0 || flag[idx] != 1)
        return;

    int offinblk = blockDim.x * blockDim.y;  // 局部变量，方便以下计算。
    // 从共享内存中将每一个点对应的 4 个值取出来存放在寄存器中。
    unsigned char temppixel[4] = { 0 };
    for (int i = idxblk, j = 0; i < idxblk + 4 * offinblk;
         i += offinblk, j++)
        temppixel[j] = (flagblkShd[i] == 1 ? curpixelShd[i] : 0);

    // 找到与当前点最接近的像素值。
    unsigned char ipixel = temppixel[0];
    for (int i = 1; i < 4; i++)
        if (abs(pixel - ipixel) > abs(pixel - temppixel[i]))
            ipixel = temppixel[i];

    // 当标记值为 0 时，则不赋值。（因此根据本算法要求，当标记值为 0，该点的
    // 最终像素值由第二个核函数来计算得到）
    outimg.imgMeta.imgData[idx] = ipixel;
}

// Kernel 函数：_minmaxAvgKer（利用点的邻域进行平滑处理）
static __global__ void _minmaxAvgKer(
        ImageCuda inimg, ImageCuda avgimg, Template atemplate)
{
   // 计算当前线程在 Grid 中的位置。其中，c 对应点的横坐标，r 对应点的纵坐标。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断是否越界。
    if (c < 0 || c >= inimg.imgMeta.width ||
        r < 0 || r >= inimg.imgMeta.height)
        return;

    // 模板内坐标的个数。
    int length = atemplate.count;

    // 分配一个数组用来存放邻域范围的点的像素值。
    unsigned char pixelarr[CNTTENPERCENT];

    // 当前像素点的和正遭遇到的点一维索引。
    int idx = r * avgimg.pitchBytes + c;
    int index = 0;

    // 遭遇到的像素点的横纵坐标。
    int curc = c, curr = r;

    // 当前遭遇点的像素值。
    unsigned char curpixel = 0;

    // 临时变量，记录当前邻域内点的个数，pntcnt 与 length 不一样，length 表示的
    // 是模板内点的数目。
    int pntcnt = 0;
    for (int i = 0; i < length; i++) {
        curc = c + atemplate.tplData[2 * i];
        curr = r + atemplate.tplData[2 * i + 1];

        // 判断目前遭遇到的点是否越界。
        if (curc < 0 || curc >= inimg.imgMeta.width ||
            curr < 0 || curr >= inimg.imgMeta.height)
            continue;

        // 计算遭遇到的点的索引。
        index = curr * inimg.pitchBytes + curc;
        curpixel = inimg.imgMeta.imgData[index];
        pixelarr[pntcnt++] = curpixel;
    }

    // 排序。
    int err = _sortDev(pixelarr, pntcnt);
    if (err != NO_ERROR)
        return;

    // 计算邻域内 10% 的点数目。
    int cnt10percent = (int)(pntcnt * 0.1f + 0.5f);

    // 计算前后 10% 的平均值
    float sumhigh = pixelarr[pntcnt - 1], sumlow = pixelarr[0];
    for (int i = 1; i < cnt10percent; i++) {
        sumlow += pixelarr[i];
        sumhigh += pixelarr[pntcnt - 1 - i];
    }
    float high = (cnt10percent == 0 ? sumhigh : sumhigh / cnt10percent);
    float low = (cnt10percent == 0 ? sumlow : sumlow / cnt10percent);

    // 计算点的最终值。
    float temp = (high + low) / 2.0f + 0.5f;
    avgimg.imgMeta.imgData[idx] = (temp > 255 ? 255 : (unsigned char)temp);
}

// Kernel 函数：_synthImageKer（综合前两个核函数的结果，得到超平滑的结果）
static __global__ void _synthImageKer(
        ImageCuda avgimg, ImageCuda outimg, unsigned char *flag)
{
    // 计算当前线程在 Grid 中的位置。其中，c 对应点的横坐标，r 对应点的纵坐标。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断是否越界。
    if (c < 0 || c >= avgimg.imgMeta.width ||
        r < 0 || r >= avgimg.imgMeta.height)
        return;

    // 计算当前线程的一维索引，同时也对应图像中点的坐标。
    int idx = r * avgimg.pitchBytes + c;

    // 更新结果图像中的像素值。
    if (flag[idx] == 0)
        outimg.imgMeta.imgData[idx] = avgimg.imgMeta.imgData[idx];
}

// 宏：FREE_SUPERSMOOTH（清理 Device 端或者 Host 端的内存）
// 该宏用于清理在 SuperSmooth 过程中申请的设备端或者主机端内存空间。
#define FREE_SUPERSMOOTH  do  {                          \
        if (avgimg != NULL)                              \
            ImageBasicOp::deleteImage(avgimg);           \
        if (flagDev != NULL)                             \
            cudaFree(flagDev);                           \
        if (atemplate != NULL)                           \
            TemplateFactory::putTemplate(atemplate);     \
        if (stream[0] != NULL)                           \
            cudaStreamDestroy(stream[0]);                \
        if (stream[1] != NULL)                           \
            cudaStreamDestroy(stream[1]);                \
    } while (0)

// Host 成员方法：superSmooth（对输入图像进行超平滑操作）
__host__ int SuperSmooth::superSmooth(Image *inimg, Image *outimg)
{
    // 检查输入图像是否为 NULL
    if (inimg == NULL || inimg->imgData == NULL)
        return NULL_POINTER;

    // 输入图像的 ROI 区域尺寸
    int imgroix = inimg->roiX2 - inimg->roiX1;
    int imgroiy = inimg->roiY2 - inimg->roiY1;

    int errcode;  // 局部变量，接受自定义函数返回的错误值。
    cudaError_t cudaerr;  // 局部变量，接受 CUDA 端返回的错误值。
    Image* avgimg = NULL;  // 声明的临时图像。
    unsigned char *flagDev = NULL;  // 声明一个 Device 端的标记数组。
    Template *atemplate = NULL;     // 声明第二个 kernel 需要用到的模板。

    // 创建两个流，一个用来执行第一个 kernel，另一个用来执行第二个 kernel。
    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++)
        cudaStreamCreate(&stream[i]);

    // 将输入图像复制到 device
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将 outimg 复制到 device
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和
    // 输入图像尺寸相同的图像             
    if (errcode != NO_ERROR) {
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, imgroix, imgroiy);
        // 如果创建图像也操作失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 声明一个临时 ImageCuda 对象，用来存放 _minmaxAvgKer 的结果。
    ImageCuda avgimgCud;
    errcode = ImageBasicOp::newImage(&avgimg);
    if (errcode != NO_ERROR) {
        FREE_SUPERSMOOTH;
        return errcode;
    }
    // 在 Device 端创建该图像。
    errcode = ImageBasicOp::makeAtCurrentDevice(avgimg, inimg->width,
                                                inimg->height);
    if (errcode != NO_ERROR) {
        FREE_SUPERSMOOTH;
        return errcode;
    }
    // 提取 ROI 子图像。
    errcode = ImageBasicOp::roiSubImage(avgimg, &avgimgCud);
    if (errcode != NO_ERROR) {
        FREE_SUPERSMOOTH;
        return errcode;
    }

    // 申请一个 GPU 端的标记数组，具有和图像一样的尺寸，初始值设置为 0。
    cudaerr = cudaMalloc((void **)&flagDev, insubimgCud.pitchBytes *
                                            sizeof(unsigned char) *
                                            insubimgCud.imgMeta.height);
    if (cudaerr != cudaSuccess) {
        FREE_SUPERSMOOTH;
        return cudaerr;
    }
    // 在 Device 端为 flagDev 赋初值为 0。
    cudaerr = cudaMemset(flagDev, 0, insubimgCud.pitchBytes *
                         insubimgCud.imgMeta.height * sizeof(unsigned char));

    if (cudaerr != cudaSuccess) {
        FREE_SUPERSMOOTH;
        return cudaerr;
    }

    // 为第一个 kernel 分配线程
    dim3 blocksize1, gridsize1;
    blocksize1.x = LOCALCLUSTER_DEF_BLOCK_X;
    blocksize1.y = LOCALCLUSTER_DEF_BLOCK_Y;
    blocksize1.z = LOCALCLUSTER_DEF_BLOCK_Z;
    gridsize1.x = (insubimgCud.imgMeta.width + blocksize1.x - 1) /
                  blocksize1.x;
    gridsize1.y = (insubimgCud.imgMeta.height + blocksize1.y - 1) /
                   blocksize1.y;
    gridsize1.z = 1;

    // 计算第一个核函数的共享内存的大小。
    int memsize = LOCALCLUSTER_DEF_BLOCK_X * LOCALCLUSTER_DEF_BLOCK_Y *
                  LOCALCLUSTER_DEF_BLOCK_Z * 2 * sizeof(unsigned char);

    // 调用第一个核函数。
    _localClusterKer<<<gridsize1, blocksize1, memsize, stream[0]>>>(
            insubimgCud, outsubimgCud, this->diffThred, this->diffCntThred,
            flagDev, this->cCntThred, this->searchScope);
    if (cudaGetLastError() != cudaSuccess) {
        FREE_SUPERSMOOTH;
        return CUDA_ERROR;
    }

    // 为第二个 kernel 分配线程
    dim3 blocksize2, gridsize2;
    blocksize2.x = AVG_DEF_BLOCK_X;
    blocksize2.y = AVG_DEF_BLOCK_Y;
    gridsize2.x = (insubimgCud.imgMeta.width + blocksize2.x - 1) /
                  blocksize2.x;
    gridsize2.y = (insubimgCud.imgMeta.height + blocksize2.y - 1) /
                  blocksize2.y;

    // 创建第二个  kernel 用到的模板。
    dim3 temsize(this->windowSize, this->windowSize, 1);
    errcode = TemplateFactory::getTemplate(&atemplate, TF_SHAPE_BOX,
                                           temsize, NULL);
//for (int i = 0; i < atemplate->count;i++)
//cout<<atemplate->tplData[2 * i] <<","<<atemplate->tplData[2 * i + 1]<<endl;
    // 将模板拷贝到 Device 内存中
    errcode = TemplateBasicOp::copyToCurrentDevice(atemplate);
    if (errcode != NO_ERROR) {
        FREE_SUPERSMOOTH;
        return errcode;
    }
    // 调用第二个核函数。
    _minmaxAvgKer<<<gridsize2, blocksize2, 0, stream[1]>>>(
            insubimgCud, avgimgCud, *atemplate);
    if (cudaGetLastError() != cudaSuccess) {
        FREE_SUPERSMOOTH;
        return CUDA_ERROR;
    }

    //数据同步
    cudaThreadSynchronize();
    // 销毁流。
    for (int i = 0; i < 2; i++)
        cudaStreamDestroy(stream[i]);

    cudaDeviceSynchronize();

    // 为第三个 kernel 分配线程
    dim3 blocksize3, gridsize3;
    blocksize3.x = SYN_DEF_BLOCK_X;
    blocksize3.y = SYN_DEF_BLOCK_Y;
    gridsize3.x = (insubimgCud.imgMeta.width + blocksize3.x - 1) /
                  blocksize3.x;
    gridsize3.y = (insubimgCud.imgMeta.height + blocksize3.y - 1) /
                blocksize3.y;

    // 调用第三个核函数。
    _synthImageKer<<<gridsize3, blocksize3>>>(avgimgCud, outsubimgCud, flagDev);
    if (cudaGetLastError() != cudaSuccess) {
        FREE_SUPERSMOOTH;
        return CUDA_ERROR;
    }

    // 处理完毕，退出。
    return NO_ERROR;
}
