// FlutterBinarize.cu
// 实现灰度图像的抖动二值化操作。

#include "FlutterBinarize.h"
#include "Histogram.h"

#include <iostream>
#include <string.h>
using namespace std;

#include "time.h"
#include "stdlib.h"
#include "curand.h"
#include "curand_kernel.h"

// 宏：GET_MIN(a, b)
// 返回两个数的最小值。
#define GET_MIN(a, b)  ((a) < (b) ? (a) : (b))

// 宏：GET_MAX(a, b)
// 返回两个数的最大值。
#define GET_MAX(a, b)  ((a) > (b) ? (a) : (b))

// 宏：GET_ABS(x)
// 返回一个数的绝对值。
#define GET_ABS(x)  ((x) >= 0 ? (x) : (-(x)))

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：FLUTTER_RADIO
// 定义了每个点向周围扩散的概率。
#define FLUTTER_RADIO  (4.0f / 7.0f)

// 宏：RANDOM_BOUND
// 定义了生成随机数的上限，生成 [0 , RANDOM_BOUND - 1 ] 范围内的随机数。
#define RANDOM_BOUND  100

// 宏：MAX_INF
// 定义正无穷的最大值。
#define MAX_INF  1 << 29

// static变量：imgTpl[MAX_PIXEL]
// 每个像素值对应的 TEMPLATE_SIZE * TEMPLATE_SIZE 大小的模板图像指针。
static Image *imgTpl[MAX_PIXEL] = {NULL};

// static变量：subimgCudTpl[MAX_PIXEL]
// 每个像素值对应的模板对应的图像逻辑数据。
static ImageCuda subimgCudTpl[MAX_PIXEL];

// static变量：map[TEMPLATE_SIZE * TEMPLATE_SIZE]
// 存储当前的图像模板。
static unsigned char tempTpl[TEMPLATE_SIZE * TEMPLATE_SIZE];

// static变量：cover[TEMPLATE_SIZE][TEMPLATE_SIZE]
// visit[i][j] 表示当前点是否被访问过。
static bool visit[TEMPLATE_SIZE][TEMPLATE_SIZE];

// static变量：weight[TEMPLATE_SIZE][TEMPLATE_SIZE]
// weight[i][j] 表示当前点的权值。
static int weight[TEMPLATE_SIZE][TEMPLATE_SIZE];

// static变量：save
// 存储满足条件的二维坐标点。
static CoordiSet *save = NULL;

// Host 函数：manhattan（计算两个点的曼哈顿（manhattan）距离）
// 计算 (x , y) 与 (ox , oy) 的 manhattan 距离。
static __host__ int  // 返回值：返回两个点的 manhattan 距离。
manhattan(
        int x,       // 第一个点的行坐标
        int y,       // 第一个点的列坐标
        int ox,      // 第二个点的行坐标
        int oy       // 第二个点的列坐标
);

// Host 函数：initalVariables（初始化变量）
// 在每次生成一个模板之前对变量进行初始化并赋初始值。
static __host__ void  // 无返回值。
initalVariables(
        int dis       // 任意两像素值为 255 的最小 manhattan 距离
);

// Host 函数：findMinWeightPoint（找到最小权值的点）
// 在当前模板中找到权值最小的点，即下一个标记为 255 的点。
static __host__ CoordiSet*  // 返回值：返回一个 CoordiSet 指针。
findMinWeightPoint();

// Host 函数：check （检查当前放置模板的合法性）
// 检查放置 pixel 个像素值为 255 的像素点且其中任两个像素点 manhattan 距离大于
// 等于 dis 方案的合法性。
static __host__ bool  // 返回值：如果能找到一个满足条件的模板，则返回 true，
                      // 否则返回 false。
checkValidity(
        int dis,      // 任意两像素值为 255 的最小 manhattan 距离
        int pixel     // 当前模板的像素值
);

// Host函数：scatter（将模板中像素点分散）
// 对一组可行模板中像素值为 255 的像素点进行发散操作。
static __host__ void  // 无返回值。
scatter(
        int dis,      // 任意两像素值为 255 的最小 manhattan 距离
        int pixel     // 当前模板的像素值
);

// Host函数：binarySearch（二分法找到并生成一个与像素值为 pixel 对应的模板）
// 二分法找到任意两像素值为 255 的最大 manhattan 距离，并生成与像素值为 pixel
// 对应的模板。
static __host__ void
binarySearch(
        int pixel
);

// Kernel 函数：_makeTemplateKer（制作模板）
// 根据每层像素点数量，把输入图像的坐标存放到模板中的相应位置，得到按层划分的
// 坐标模板。
static __global__ void            // Kernel 函数无返回值。
_makeTemplateKer(
        ImageCuda imgCud,         // 输入图像
        TemplateCuda coordinate,  // 模板
        int packrange,            // 图像每层像素值数量
        unsigned int *devcnt      // 图像每层像素点数量
);

// Kernel 函数：_genRandomKer（生成随机数）
// 在 Device 端生成与输入图片大小相同的随机数数组，用于抖动二值化像素值扩散。
static __global__ void 
_genRandomKer(
        int *randnumdev,  // 随机数矩阵
        int timehost,     // 时间参数
        int width         // 随机数矩阵的宽度
);

// Kernel 函数：_flutterBinarizeLayerKer（抖动二值化）
// 对图像进行抖动二值化处理。根据对像素点划分出的不同层定义不同的扩散能力（扩
// 散理解为把周围点像素值设为 0），分层处理，从而实现图像的抖动二值化操作。
static __global__ void               // Kernel 函数无返回值。
_flutterBinarizeLayerKer(
        ImageCuda imgCud,            // 输出图像
        TemplateCuda coordinateCud,  // 模板
        int *randnumdev,             // 随机数数组
        int from,                    // 当前层在模板中的起始位置
        int to,                      // 当前层在模板中的终止位置
        int threshold                // 当前层像素点扩散的阈值
);

// Kernel 函数：_flutterBinarizeKer（抖动二值化）
// 对图像进行抖动二值化处理。根据像素点值对应的模板处理当前像素点，从而实现图
// 像的抖动二值化操作。
static __global__ void           // Kernel 函数无返回值。
_flutterBinarizeKer(
        ImageCuda inimgCud,      // 输出图像
        ImageCuda outimgCud,     // 输出图像
        ImageCuda *subimgCudTpl  // 每个像素值对应的模板对应的图像逻辑数据指针
);

// Host 函数：manhattan（计算两个点的曼哈顿（manhattan）距离）
static __host__ int manhattan(int x, int y, int ox, int oy)
{
    // 返回 x 与 y 坐标差的绝对值。
    return GET_ABS(x - ox) + GET_ABS(y - oy);
}

// Host 函数：initalVariables（初始化变量）
static __host__ void initalVariables(int dis)
{
    // 初始化 map，visit，weight 变量。
    memset(tempTpl, 0, sizeof (tempTpl));
    memset(visit, false, sizeof (visit));
    memset(weight, 0, sizeof (weight));

    // 对 weight 数组赋初值。
    for (int i = 0; i < TEMPLATE_SIZE; i++) {
        for (int j = 0; j < TEMPLATE_SIZE; j++) {
            // 局部变量，up 表示与当前点的 manhattan 距离在 dis 范围内最上端的
            // 点的行坐标。
            int up = GET_MAX(0, i - dis);

            // 局部变量，dw 表示与当前点的 manhattan 距离在 dis 范围内最下端的
            // 点的行坐标。
            int dw = GET_MIN(TEMPLATE_SIZE - 1, i + dis);

            // 枚举满足条件的行更新 weight 数组。
            for (int k = up; k <= dw; k++) {
                // 局部变量，delta 表示第 k 行点与当前点 manhattan 距离剩余值。
                int delta = dis - GET_ABS(i - k);

                // 局部变量，le 表示在第 k 行点中与当前点 manhattan 距离在 dis
                // 范围内最左端的点的列坐标。
                int le = GET_MAX(0, j - delta);

                // 局部变量，ri 表示在第 k 行点中与当前点 manhattan 距离在 dis
                // 范围内最右端的点的列坐标。
                int ri = GET_MIN(TEMPLATE_SIZE - 1, j + delta);

                // 累加当前点的权值。
                weight[i][j] += ri - le + 1;
            }
        }
    }
    return;
}

// Host 函数：findMinWeightPoint（找到最小权值的点）
static __host__ CoordiSet* findMinWeightPoint()
{
    // 声明答案节点。
    CoordiSet *ans = NULL;

    // 初始化该节点。
    CoordiSetBasicOp::newCoordiSet(&ans);
    CoordiSetBasicOp::makeAtHost(ans, 1);

    // 局部变量，minValue 表示最小权值，初始化为 MAX_INF。
    int minValue = MAX_INF;
    for (int i = 0; i < TEMPLATE_SIZE; i++) {
        for (int j = 0; j < TEMPLATE_SIZE; j++) {
            // 如果当前点未被访问并权值小于 minValue，则进行赋值操作。
            if (visit[i][j] == false && weight[i][j] < minValue) {
                // 更新答案。
                ans->tplData[0] = i;
                ans->tplData[1] = j;
                minValue = weight[i][j];
            }
        }
    }

    // 如果找不到答案节点返回 NULL。
    if (minValue == MAX_INF)
        return NULL;

    // 返回答案节点。
    return ans;
}

// Host 函数：check （检查当前放置模板的合法性）
static __host__ bool checkValidity(int dis, int pixel)
{
    // 初始化变量。
    initalVariables(dis);

    // 局部变量：left 表示还需要放置像素值为 255 的像素点个数。
    int left = GET_MIN(pixel, MAX_PIXEL - pixel);

    while(left) {
        // 找到当前应该放置的下一个点的 CoordiSet 指针。
        CoordiSet *cur = findMinWeightPoint();

        // 如果没找到则跳出循环。
        if (cur == NULL)
            break;

        // 将当前点像素值标记为 255。
        tempTpl[cur->tplData[0] * TEMPLATE_SIZE + cur->tplData[1]] = 255;

        // 更新 left。
        left--;

        // 更新 visit 和 weight 数组。
        for (int i = 0; i < TEMPLATE_SIZE; i++) {
            for (int j = 0; j < TEMPLATE_SIZE; j++) {
                // 如果当前点未访问过且与放置点的 manhattan 小于等于 dis，则
                // 更新对应的 visit 和 weight 数组。
                if (visit[i][j] == false &&
                    manhattan(cur->tplData[0], cur->tplData[1], i, j) <= dis) {
                    // 将当前点标记为访问过。
                    visit[i][j] = true;

                    // 更新与当前点 manhattan 距离小于等于 dis 的权值。
                    for (int ii = 0; ii < TEMPLATE_SIZE; ii++) {
                        for (int jj = 0; jj < TEMPLATE_SIZE; jj++) {
                            // 如果与当前点的 manhattan 距离小于等于 dis 则
                            // 更新该权值。
                            if (manhattan(i, j, ii, jj) <= dis)
                                weight[ii][jj]--;
                        }
                    }
                }
            }
        }
    }

    // 返回剩余放置像素值为 255 的点的个数是否为 0。
    return (left == 0);
}

// Host函数：scatter（将模板中像素点分散）
static __host__ void scatter(int dis, int pixel)
{
    // 初始化 weight 数组。
    memset(weight, 0, sizeof (weight));

    // 给 weight 数组赋初值。
    for (int i = 0; i < TEMPLATE_SIZE; i++) {
        for (int j = 0; j < TEMPLATE_SIZE; j++) {
            // 如果当前点像素值为 255，则更新 weight 数组。
            if (tempTpl[i * TEMPLATE_SIZE + j] == 255) {
                // 局部变量，up 表示与当前点的 manhattan 距离在 dis * 2 范围内
                // 最上端的点的行坐标。
                int up = GET_MAX(0, i - dis * 2);

                // 局部变量，dw 表示与当前点的 manhattan 距离在 dis * 2 范围内
                // 最下端的点的行坐标。
                int dw = GET_MIN(TEMPLATE_SIZE - 1, i + dis * 2);

                // 枚举满足条件的行更新 weight 数组。
                for (int k = up; k <= dw; k++) {
                    // 局部变量，delta 表示第 k 行点与当前点 manhattan 距离
                    // 的剩余值。
                    int delta = dis * 2 - GET_ABS(i-k);

                    // 局部变量，le 表示在第 k 行点中与当前点 manhattan 距离在
                    // dis * 2 范围内最左端的点的列坐标。
                    int le = GET_MAX(0, j - delta);

                    // 局部变量，ri 表示在第 k 行点中与当前点 manhattan 距离在
                    // dis * 2 范围内最右端的点的列坐标。
                    int ri = GET_MIN(TEMPLATE_SIZE - 1, j + delta);

                    // 累加当前点的权值。每个点的权值表示当前点被像素值为 255
                    // 的点覆盖的次数。
                    weight[i][j] += ri - le + 1;
                }
            }
        }
    }

    // 如果 save 为空，则初始化开辟空间，否则说明已经初始化了。
    if (save == NULL) {
        // 如果 save 为空，则初始化为 TEMPLATE_SIZE * TEMPLATE_SIZE 的模板。
        CoordiSetBasicOp::newCoordiSet(&save);
        CoordiSetBasicOp::makeAtHost(save, TEMPLATE_SIZE * TEMPLATE_SIZE);
    }

    // 局部变量，times 表示循环的次数。
    int times = MAX_PIXEL;
    while(times--) {
        // 局部变量，(frx, fry) 表示被交换的节点坐标，(tox, toy) 表示交换到的
        // 节点坐标。
        int frx,fry,tox,toy;

        // 局部变量，记录最大权值。
        int maxValue = 0;

        // 局部变量，top 表示可交换节点的个数。
        int top = 0;

        // 找到权值最大的被交换的节点组。
        for (int i = 0; i < TEMPLATE_SIZE; i++) {
            for (int j = 0; j < TEMPLATE_SIZE; j++) {
                // 如果当前点像素值为 255，则更新。
                if (tempTpl[i * TEMPLATE_SIZE + j] == 255) {
                    // 如果小于最大权值，则更新 maxValue 并重置 top 为 0。
                    if (weight[i][j] > maxValue) {
                        maxValue = weight[i][j];
                        top = 0;
                        save->tplData[top++] = i;
                        save->tplData[top++] = j;
                        // 如果等于最大权值，则存储 save 指针中。
                    } else if (weight[i][j] == maxValue) {
                        save->tplData[top++] = i;
                        save->tplData[top++] = j;
                    }
                }
            }
        }

        // 如果不存在权值最小的点，跳出循环。
        if (top == 0) break;

        // 随机找到一个点作为被交换的节点下标。
        int id = rand() % (top/2);

        // 得到对应的 (frx, fry)。
        frx = save->tplData[id << 1];
        fry = save->tplData[id << 1 | 1];

        // 重置 top 变量。
        top = 0;

        // 找到满足 dis 限制的交换到的节点组。
        for (int i = 0; i < TEMPLATE_SIZE; i++) {
            for (int j = 0; j < TEMPLATE_SIZE; j++) {
                // 如果当前点像素值为 0，则判断该点是否满足条件。
                if (tempTpl[i * TEMPLATE_SIZE + j] == 0) {
                    // 局部变量，flag 表示当前点是否满足条件。
                    bool flag = true;

                    // 局部变量，up 表示与当前点的 manhattan 距离在 dis 范围内
                    // 最上端的点的行坐标。
                    int up = GET_MAX(0, i - dis);

                    // 局部变量，dw 表示与当前点的 manhattan 距离在 dis 范围内
                    // 最下端的点的行坐标。
                    int dw = GET_MIN(TEMPLATE_SIZE - 1, i + dis);

                    // 枚举满足条件的行判断当前点是否满足条件。
                    for (int ii = up; ii <= dw; ii++) {
                        // 局部变量，delta 表示第 ii 行点与当前点 manhattan
                        // 距离的剩余值。
                        int delta = dis - GET_ABS(i - ii);

                        // 局部变量，le 表示在第 ii 行点中与当前点 manhattan
                        // 距离在 dis 范围内最左端的点的列坐标。
                        int le = GET_MAX(0, j - delta);

                        // 局部变量，ri 表示在第 ii 行点中与当前点 manhattan
                        // 距离在 dis 范围内最右端的点的列坐标。
                        int ri = GET_MIN(TEMPLATE_SIZE - 1, j + delta);

                        // 枚举当前行中的点判断当前点是否满足条件。
                        for (int jj = le; jj <= ri; jj++) {
                            // 如果是 (frx, fry) 点则跳过改循环。
                            if (ii == frx && jj == fry)
                                continue;

                            // 如果该点在 dis 内的 manhattan 距离存在像素值为
                            // 255 的点，则该点不满足条件，跳出循环。
                            if (tempTpl[ii * TEMPLATE_SIZE + jj] == 255) {
                                flag = false;
                                break;
                            }
                        }

                        // 如果当前点不满足条件，则跳出循环。
                        if (flag == false) break;
                    }

                    // 如果该点满足条件，则存储 save 指针中。
                    if (flag) {
                        save->tplData[top++] = i;
                        save->tplData[top++] = j;
                    }
                }
            }
        }

        // 如果不存在满足条件的点，跳过本次循环。
        if (top == 0) continue;

        // 随机找到一个点作为被交换的节点下标。
        id = rand() % (top/2);

        // 得到对应的 (tox, toy)。
        tox = save->tplData[id << 1];
        toy = save->tplData[id << 1 | 1];

        // 交换 (frx, fry) 和 (tox, toy) 两个节点的像素值。
        tempTpl[frx * TEMPLATE_SIZE + fry] = 0;
        tempTpl[tox * TEMPLATE_SIZE + toy] = 255;

        // 更新 weight 数组。
        for (int i = 0; i < TEMPLATE_SIZE; i++) {
            for (int j = 0; j < TEMPLATE_SIZE; j++) {
                // 如果该点与 fr 点 manhattan 距离小于 dis * 2，则执行减操作。
                if (manhattan(i,j,frx,fry) <= dis * 2)
                    weight[i][j]--;

                // 如果该点与 to 点 manhattan 距离小于 dis * 2，则执行加操作。
                if (manhattan(i,j,tox, toy) <= dis * 2)
                    weight[i][j]++;
            }
        }
    }
}

// Host函数：binarySearch（二分法找到并生成一个与像素值为 pixel 对应的模板）
static __host__ void binarySearch(int pixel)
{
    // 初始化 l r 边界值。
    int l = 1, r = TEMPLATE_SIZE << 1;

    // 二分答案。
    while(l <= r) {
        // 局部变量，mid 为当前 l r 的中值。
        int mid = (l + r) >> 1;

        // 如果 mid 值合法，更新 l 值。
        if (checkValidity(mid, pixel)) l = mid + 1;

        // 否则更新 r 值。
        else r = mid - 1;
    }

    // 得到一组可行模板。
    checkValidity(r, pixel);

    // 调用 scatter 函数使得当前模板中像素值为 255 的点尽可能分散。
    scatter(r, pixel);

    // 如果 pixel 个数大于 MAX_PIXEL / 2 像素值取反。
    if (pixel > MAX_PIXEL / 2) {
        for (int i = 0; i < TEMPLATE_SIZE; i++) {
            for (int j = 0; j < TEMPLATE_SIZE; j++) {
                tempTpl[i * TEMPLATE_SIZE + j] =
                        (tempTpl[i * TEMPLATE_SIZE + j] == 0) ? 255 : 0;
            }
        }
    }
}

// Kernel 函数：_makeTemplateKer（制作模板）
static __global__ void _makeTemplateKer(ImageCuda imgCud,
                                        TemplateCuda coordinateCud,
                                        int packrange, unsigned int *devcnt)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标的
    // x 和 y 分量（其中，c 表示column，r 表示row）。由于采用的并行度缩减策略，
    // 令一个线程处理 4 个输出像素，这四个像素位于同一列的相邻 4 行上，因此，
    // 对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 计算第一个坐标点对应的图像数据数组下标。
    int idx = r * imgCud.pitchBytes + c;

    for (int i = 0; i < 4; i++) {
        // 检查每个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
        // 一方面防止由于段错误导致的程序崩溃。
        if (c >= imgCud.imgMeta.width || r >= imgCud.imgMeta.height)
            break;

        // 使用原子减操作获得当前像素点在模板中的下标。
        unsigned int val = imgCud.imgMeta.imgData[idx];
        int top = atomicSub(&(devcnt[val / packrange]), 1) - 1;

        // 将当前像素点的坐标和像素值写到对应的模板中。
        coordinateCud.tplMeta.tplData[top << 1] = c;
        coordinateCud.tplMeta.tplData[(top << 1) | 1] = r;
        coordinateCud.attachedData[top] = (float)val;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        idx += imgCud.pitchBytes;
        r++;
    }
}

// Host 成员方法：initializeLayerTpl（初始化模板处理）
__host__ int FlutterBinarize::initializeLayerTpl(Image *img, int groupnum,
                                                 unsigned int *&cnt,
                                                 Template *&coordinate)
{
    //  检查输入图像是否为空。
    if (img == NULL)
        return NULL_POINTER;

    // 局部变量，错误码
    int errcode;

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(img);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda subimgCud;
    errcode = ImageBasicOp::roiSubImage(img, &subimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 在 Device 端分配存储每层坐标点的在模板中数量的数组。
    unsigned int *devcnt;

    // 在 Device 上分配存储临时每层像素点数量数组的空间。
    errcode = cudaMalloc((void **)&devcnt, MAX_PIXEL *  sizeof (unsigned int));
    if (errcode != NO_ERROR) {
        cudaFree(devcnt);
        return errcode;
    }

    // 初始化 Device 上的内存空间。
    errcode = cudaMemset(devcnt, 0, MAX_PIXEL * sizeof (unsigned int));
    if (errcode != NO_ERROR) {
        cudaFree(devcnt);
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (subimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (subimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                 (blocksize.y * 4);

    // 计算输入图像的直方图。
    Histogram hist;
    errcode = hist.histogram(img, devcnt, 0);
    if (errcode != NO_ERROR) {
        cudaFree(devcnt);
        return errcode;
    }

    // 将统计每层像素点数量的结果拷回 Host 端内存中。
    errcode = cudaMemcpy(cnt, devcnt, MAX_PIXEL * sizeof (unsigned int),
                         cudaMemcpyDeviceToHost);
    if (errcode != NO_ERROR) {
        cudaFree(devcnt);
        return errcode;
    }

    // 获得每层坐标点在模板中的区域范围。
    for (int i = 0; i < groupnum; i++) {
        // 定义变量 sum 表示前 i 层像素点的坐标在模板中的下标上限。
        int sum = (i == 0) ? 0 : cnt[i-1];

        // 定义第 i 层像素点在模板中的起始位置和终止位置。
        int stpos = i * packrange;
        int endpos = GET_MIN(MAX_PIXEL, (i + 1) * packrange);
        for (int j = stpos; j < endpos; j++)
            sum += cnt[j];

        // 给 cnt[i] 赋值。
        cnt[i] = sum;
    }

    // 初始化 Device 上的内存空间。
    errcode = cudaMemset(devcnt, 0, groupnum * sizeof (unsigned int));
    if (errcode != NO_ERROR) {
        cudaFree(devcnt);
        return errcode;
    }

    // 将每层像素点在模板中的区域范围拷回 Host 端内存中。
    errcode = cudaMemcpy(devcnt, cnt, groupnum * sizeof(unsigned int),
                         cudaMemcpyHostToDevice);
    if (errcode != NO_ERROR) {
        cudaFree(devcnt);
        return errcode;
    }

    errcode = TemplateBasicOp::makeAtCurrentDevice(coordinate,
                                                   subimgCud.imgMeta.width *
                                                   subimgCud.imgMeta.height);
    if (errcode != NO_ERROR) {
        cudaFree(devcnt);
        return errcode;
    }

    // 根据 Template 指针，得到对应的 TemplateCuda 型数据。
    TemplateCuda *coordinateCud = TEMPLATE_CUDA(coordinate);

    // 调用核函数，制作模板。
    _makeTemplateKer<<<gridsize, blocksize>>>(subimgCud, *coordinateCud,
                                              packrange, devcnt);

    // 若调用 CUDA 出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(devcnt);
        return CUDA_ERROR;
    }

    // 释放 Device 端的统计数组存储空间。
    cudaFree(devcnt);

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：initializeTemplate（初始化模板处理）
__host__ int FlutterBinarize::initializeTemplate()
{
    // 局部变量，错误码
    int errcode;

    // 初始化像素值在 [0, 255] 范围内的每个像素值对应的模板。
    for (int i = 0; i < MAX_PIXEL; i++) {
        // 初始化像素值为 i 的模板。如果没有申请成功，则置为 NULL。
        errcode = ImageBasicOp::newImage(&imgTpl[i]);
        if (errcode != NO_ERROR) {
            imgTpl[i] = NULL;
            return errcode;
        }

        // 为像素值为 i 的模板申请内存空间。如果没有申请成功，则置为 NULL。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                imgTpl[i], TEMPLATE_SIZE, TEMPLATE_SIZE);
        if (errcode != NO_ERROR) {
            ImageBasicOp::deleteImage(imgTpl[i]);
            imgTpl[i] = NULL;
            return errcode;
        }
    }

    for (int i = 0; i < MAX_PIXEL; i++) {
        binarySearch(i);
        errcode = cudaMemcpy(imgTpl[i]->imgData, tempTpl, TEMPLATE_SIZE *
                             TEMPLATE_SIZE * sizeof (unsigned char),
                             cudaMemcpyHostToDevice);
        if (errcode != NO_ERROR)
            return errcode;

        // 提取当前像素值 i 对应的模板的 ROI 子图像。
        errcode = ImageBasicOp::roiSubImage(imgTpl[i], &subimgCudTpl[i]);
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 处理完毕，退出。
    return NO_ERROR;
}

// Kernel 函数：_genRandomKer（生成随机数）
static __global__ void _genRandomKer(int *randnumdev, int timehost, int width)
{
    // 计算当前线程的位置。
    int index = blockIdx.x * 4;

    // curand随机函数初始化。
    curandState state;
    curand_init(timehost, index, 0, &state);
    
    // 得到当前行在随机数矩阵中的偏移。
    int position = index * width;

    // 一个线程生成 4 行随机数。
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < width; j++) {
            // 生成 [0 , RANDOM_BOUND - 1 ] 范围内的随机数。
            randnumdev[position + j] = curand(&state) % RANDOM_BOUND;
        }

        // 获取下一行的偏移。
        position += width;
    }
}

// Kernel 函数：_flutterBinarizeLayerKer（抖动二值化）
static __global__ void _flutterBinarizeLayerKer(ImageCuda imgCud,
                                                TemplateCuda coordinateCud,
                                                int *randnumdev, int from,
                                                int to, int threshold)
{
    // 计算当前线程处理的像素点的下标。
    int tid = blockIdx.x * blockDim.x + threadIdx.x + from;

    // 判断该像素点是否在当前层。
    if (tid < to) {
        // 得到当前像素点的 (x,y) 坐标。
        int x = coordinateCud.tplMeta.tplData[tid << 1];
        int y = coordinateCud.tplMeta.tplData[(tid << 1) | 1];

        // 计算当前像素点的索引值。
        int offset = y * imgCud.pitchBytes + x;

        // 如果当前像素点已经被扩散，则不进行扩散处理。
        if (imgCud.imgMeta.imgData[offset] != 255)
            return;

        // 得到当前像素点在随机数数组中的偏移。
        int position = y * imgCud.imgMeta.width + x;

        // 得到当前 position 位置上对应的随机数。
        int radval = *(randnumdev + position);

        // 如果当前点不被选择扩散，则不进行扩散处理。
        if (radval >= (int)(threshold * FLUTTER_RADIO))
            return;

        // 将当前点像素值标记为 0。
        imgCud.imgMeta.imgData[offset] = 0;

        // 该点向周围 8 个方向进行扩散。
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                // 如果是当前点则跳过不处理。
                if (i == 0 && j == 0)
                    continue;

                // 得到当前被扩散像素点的 (x,y) 坐标。
                int tx = x + i;
                int ty = y + j;

                // 如果该点坐标非法，则跳过不进行扩散处理。
                if (tx < 0 || tx >= imgCud.imgMeta.width ||
                    ty < 0 || ty >= imgCud.imgMeta.height)
                    continue;

                // 计算当前被扩散点的索引值。
                int toffset = ty * imgCud.pitchBytes + tx;

                // 如果当前像素点已经被扩散，则不进行扩散处理。
                if (imgCud.imgMeta.imgData[toffset] != 255)
                    continue;

                // 得到当前像素点在随机数数组中的偏移。
                int tposition = ty * imgCud.imgMeta.width + tx;

                // 得到当前 position 位置上对应的随机数。
                radval = *(randnumdev + tposition);

                // 如果当前点不被选择扩散，则跳过不进行扩散处理。
                if (radval >= (int)(threshold * FLUTTER_RADIO))
                    continue;

                // 将对当前点进行扩散，像素值标记为 0。
                imgCud.imgMeta.imgData[toffset] = 0;
            }
        }
    }
}

// Kernel 函数：_flutterBinarizeKer（抖动二值化）
static __global__ void _flutterBinarizeKer(ImageCuda inimgCud,
                                           ImageCuda outimgCud,
                                           ImageCuda *subimgCudTplDev)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标的
    // x 和 y 分量（其中，c 表示column，r 表示row）。由于采用的并行度缩减策略，
    // 令一个线程处理 4 个输出像素，这四个像素位于同一列的相邻 4 行上，因此，
    // 对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 计算第一个坐标点对应的图像数据数组下标。
    int idx = r * inimgCud.pitchBytes + c;
    int tid = (r % TEMPLATE_SIZE) * TEMPLATE_SIZE + (c % TEMPLATE_SIZE);
    for (int i = 0; i < 4; i++) {
        // 检查每个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
        // 一方面防止由于段错误导致的程序崩溃。
        if (c >= inimgCud.imgMeta.width || r >= inimgCud.imgMeta.height)
            break;

        // 获得输入图像对应像素点的像素值。
        unsigned char val = inimgCud.imgMeta.imgData[idx];

        // 用对应模板中相应位置的像素值对输入图像进行抖动二值化处理。
        outimgCud.imgMeta.imgData[idx] =
                subimgCudTplDev[(int)val].imgMeta.imgData[tid];

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        idx += inimgCud.pitchBytes;
        r++;

        // 对应的 tid 只需要加上 TEMPLATE_SIZE 即可。
        tid += TEMPLATE_SIZE;
    }
}

// 宏：FAIL_FLUTTER_BINARIZE_FREE
// 如果出错，就释放之前申请的内存。
#define FAIL_FLUTTER_BINARIZE_FREE do {                   \
        if (cnt != NULL)                                  \
            delete [] cnt;                                \
        if (coordinate != NULL)                           \
            TemplateBasicOp::deleteTemplate(coordinate);  \
        if (randnumdev != NULL)                           \
            cudaFree(randnumdev);                         \
        if (subimgCudTplDev != NULL)                      \
            cudaFree(subimgCudTplDev);                    \
        for (int i = 0; i < MAX_PIXEL; i++) {             \
            if (imgTpl[i] != NULL) {                      \
                ImageBasicOp::deleteImage(imgTpl[i]);     \
                imgTpl[i] = NULL;                         \
            }                                             \
        }                                                 \
    } while (0)

// Host 成员方法：flutterBinarize（抖动二值化处理）
__host__ int FlutterBinarize::flutterBinarize(Image *inimg, Image *outimg)
{
    // 检查输入图像和输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 局部变量，根据每层像素值的范围划分出分组的数量。
    int groupnum = MAX_PIXEL / packrange;

    // 局部变量，错误码
    int errcode;

    // 声明函数内使用的所有变量，并初始化为空。

    // 每层坐标点的在模板中的区域范围，[cnt[i - 1], cnt[i] - 1 ] 表示第 i 层点
    // 在模板中的区域范围。
    unsigned int *cnt = NULL;

    // 模板指针，依次存储每层坐标点的坐标。
    Template *coordinate = NULL;

    // 在 Device 端申请随机数数组所需要的空间。
    int *randnumdev = NULL;

    // 声明 Device 端 ROI 子图像模板指针。
    ImageCuda *subimgCudTplDev = NULL;

    // 在 Host 端为 cnt 分配空间。
    cnt = new unsigned int[MAX_PIXEL];

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放空间。
        FAIL_FLUTTER_BINARIZE_FREE;
        return errcode;
    }

    // 将输出图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图像
        // 的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->roiX2 - inimg->roiX1,
                inimg->roiY2 - inimg->roiY1);
        // 如果创建图像也操作失败，则说明操作彻底失败。
        if (errcode != NO_ERROR) {
            // 遇到错误则释放空间。
            FAIL_FLUTTER_BINARIZE_FREE;
            return errcode;
        }
    }

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放空间。
        FAIL_FLUTTER_BINARIZE_FREE;
        return errcode;
    }

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放空间。
        FAIL_FLUTTER_BINARIZE_FREE;
        return errcode;
    }

    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一。
    insubimgCud.imgMeta.width = GET_MIN(insubimgCud.imgMeta.width,
                                        outsubimgCud.imgMeta.width);
    outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;

    insubimgCud.imgMeta.height = GET_MIN(insubimgCud.imgMeta.height,
                                         outsubimgCud.imgMeta.height);
    outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;

    // 初始化输出图像的 ROI 子图像，像素值标记为 255。
    errcode = cudaMemset2D(outimg->imgData, outsubimgCud.pitchBytes, 255,
                           outimg->width * sizeof (unsigned char),
                           outimg->height);
    if (errcode != NO_ERROR) {
        // 遇到错误则释放空间。
        FAIL_FLUTTER_BINARIZE_FREE;
        return errcode;
    }

    // selectmethod = 1 执行第一种算法。
    if (selectmethod == 1) {
        // 初始化空模板并在 Device 端开辟与输入图像等大的空间。
        errcode = TemplateBasicOp::newTemplate(&coordinate);
        if (errcode != NO_ERROR) {
            // 遇到错误则释放空间。
            FAIL_FLUTTER_BINARIZE_FREE;
            return errcode;
        }

        // 初始化模板。
        errcode = this->initializeLayerTpl(inimg, groupnum, cnt, coordinate);
        if (errcode != NO_ERROR) {
            // 遇到错误则释放空间。
            FAIL_FLUTTER_BINARIZE_FREE;
            return errcode;
        }

        // 计算随机数数组的大小。
        int total = outsubimgCud.imgMeta.width * outsubimgCud.imgMeta.height;

        // 在 Device 端分配随机数数组所需要的空间。
        errcode = cudaMalloc((void **)&randnumdev, total * sizeof (int));
        if (errcode != NO_ERROR) {
            // 遇到错误则释放空间。
            FAIL_FLUTTER_BINARIZE_FREE;
            return errcode;
        }

        // 在 Host 端获取时间。由于使用标准 C++ 库获得的时间是精确到秒的，这个
        // 时间精度是远远大于两次可能的调用间隔，因此，我们只在程序启动时取当
        // 前时间，之后对程序的时间直接进行自加，以使得每次的时间都是不同的，
        // 这样来保证种子在各次调用之间的不同，从而获得不同的随机数。
        static int timehost = (int)time(NULL);
        timehost++;

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 gridsize, blocksize;
        gridsize.x = (outsubimgCud.imgMeta.height + 3) / 4;
        blocksize.x = 1;

        // 随机数矩阵的宽度。
        int width = outsubimgCud.imgMeta.width;

        // 调用生成随机数的 Kernel 函数。
        _genRandomKer<<<gridsize, blocksize>>>(randnumdev, timehost, width);

        // 若调用 CUDA 出错返回错误代码。
        if (cudaGetLastError() != cudaSuccess) {
            // 遇到错误则释放空间。
            FAIL_FLUTTER_BINARIZE_FREE;
            return CUDA_ERROR;
        }

        // 根据 Template 指针，得到对应的 TemplateCuda 型数据。
        TemplateCuda *coordinateCud = TEMPLATE_CUDA(coordinate);

        // 局部变量，前一层像素点的终止下标（即当前层像素点的起始下标）。
        int prepos = 0;

        for (int i = 0; i < groupnum; i++) {
            // 如果当前层不存在则跳过不处理。
            if (prepos == cnt[i])
                continue;

            // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
            blocksize.x = DEF_BLOCK_X * DEF_BLOCK_Y;
            blocksize.y = 1;
            gridsize.x = (cnt[i] - prepos + blocksize.x - 1) / blocksize.x;
            gridsize.y = 1;

            // 调用核函数，对当前层像素点进行抖动二值化。
            _flutterBinarizeLayerKer<<<gridsize, blocksize>>>(
                    outsubimgCud, *coordinateCud, randnumdev, prepos, cnt[i],
                    (int)(RANDOM_BOUND * (groupnum - i) / groupnum));

            // 若调用 CUDA 出错返回错误代码。
            if (cudaGetLastError() != cudaSuccess) {
                // 遇到错误则释放空间。
                FAIL_FLUTTER_BINARIZE_FREE;
                return CUDA_ERROR;
            }

            // 更新 prepos。
            prepos = cnt[i];
        }

    // selectmethod = 2 执行第二种算法。
    } else {
        // 初始化模板。
        errcode = this->initializeTemplate();
        if (errcode != NO_ERROR) {
            // 遇到错误则释放空间。
            FAIL_FLUTTER_BINARIZE_FREE;
            return errcode;
        }

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 blocksize, gridsize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) /
                      blocksize.x;
        gridsize.y = (insubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                     (blocksize.y * 4);

        // 为 subimgCudTplDev 分配内存空间。
        errcode = cudaMalloc((void**)&subimgCudTplDev,
                             MAX_PIXEL * sizeof (ImageCuda));
        if (errcode != NO_ERROR) {
            // 遇到错误则释放空间。
            FAIL_FLUTTER_BINARIZE_FREE;
            cudaFree(subimgCudTplDev);
            return CUDA_ERROR;
        }

        // 将 Host 上的 subimgCudTplDev 拷贝到 Device 上。
        errcode = cudaMemcpy(subimgCudTplDev, subimgCudTpl,
                             MAX_PIXEL * sizeof (ImageCuda),
                             cudaMemcpyHostToDevice);
        if (errcode != NO_ERROR) {
            // 遇到错误则释放空间。
            FAIL_FLUTTER_BINARIZE_FREE;
            cudaFree(subimgCudTplDev);
            return CUDA_ERROR;
        }

        // 调用核函数，对当前层像素点进行抖动二值化。
        _flutterBinarizeKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud,
                                                     subimgCudTplDev);

        // 若调用 CUDA 出错返回错误代码。
        if (cudaGetLastError() != cudaSuccess) {
            // 遇到错误则释放空间。
            FAIL_FLUTTER_BINARIZE_FREE;
            cudaFree(subimgCudTplDev);
            return CUDA_ERROR;
        }
    }

    // 释放 Host 端的统计数组存储空间。
    delete [] cnt;

    // 清空模板指针。
    TemplateBasicOp::deleteTemplate(coordinate);

    // 释放 Device 内存中随机数组数据。
    cudaFree(randnumdev);

    // 释放像素值在 [0, 255] 范围内的每个像素值对应的模板。
    for (int i = 0; i < MAX_PIXEL; i++) {
        ImageBasicOp::deleteImage(imgTpl[i]);
        imgTpl[i] = NULL;
    }

    // 释放已分配的模板指针，避免内存泄露。
    cudaFree(subimgCudTplDev);

    // 处理完毕，退出。
    return NO_ERROR;
}

// 取消前面的宏定义。
#undef FAIL_FLUTTER_BINARIZE_FREE

// Host 成员方法：flutterBinarize（抖动二值化处理）
__host__ int FlutterBinarize::flutterBinarize(Image *img)
{
    // 调用 Out-Place 版本的成员方法。
    return flutterBinarize(img, img);
}

