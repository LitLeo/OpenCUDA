// RobustEdgeDetection.cu
// 实现图像的边缘检测算法。

#include "RobustEdgeDetection.h"
#include<stdio.h>

// 宏：SA_DEF_BLOCK_X 和 SA_DEF_BLOCK_Y 和 SA_DEF_BLOCK_Z
// 定义了默认单纯平均法线程快的尺寸。
#define SA_DEF_BLOCK_X    32
#define SA_DEF_BLOCK_Y     2
#define SA_DEF_BLOCK_Z     4

// 宏：FV_DEF_BLOCK_X 和 FV_DEF_BLOCK_Y 和 FV_DEF_BLOCK_Z
// 定义了默认三维特征向量法线程快的尺寸。
#define FV_DEF_BLOCK_X    32
#define FV_DEF_BLOCK_Y     2
#define FV_DEF_BLOCK_Z     4

// 宏：RED_CODE_CHOICE
// 定义一个选择宏，来决定注释掉原来算法的代码。
// #define RED_CODE_CHOICE

// 宏：RED_DIV_PERCENTAGE
// 定义特征向量法中求中值平均时去掉前后一定百分比像素点的辅助宏。
#define RED_DIV_PERCENTAGE    20

// 宏：RED_ARRAY_MAXLEN
// 特征向量法计算临时数组的最大尺寸的辅助宏即对向邻域的最大尺寸，
// 此处默认为 11。
#define RED_ARRAY_MAXLEN    11

// Device 全局常量：_argSupreDev[4][4]（最大增强法操作的参数）
// 最大增强法的参数。
const int static __device__ _argSupreDev[4][4] = {
    // [0][ ], [1][ ]
    { -1,  0,  1,  0 }, { 0, -1,  0,  1 },
    // [2][ ], [3][ ]
    { -1, -1,  1,  1 }, { 1, -1, -1,  1 }
};

// Host 函数：_initNeighbor（初始化对向邻域坐标）
// 设置 8 个对向邻域的坐标，为了方便传入 Device 端，此处把 8 个邻域的数据赋值
// 在一个数组内，通过偏移来寻找每一个线程应该用到的坐标。
static __host__ int      // 函数若正确执行返回 NO_ERROR。
_initNeighbor(
        int neighbor[],  // 欲初始化的坐标数组。
        int diffsize,    // 对向邻域的大小。
        int *sizedp01,   // dp 号为 0 和 1 的对向邻域的尺寸。
        int *sizedp23    // dp 号为 2 和 3 的对向邻域的尺寸。
);

// Device 函数：_maxEnhancementDev（最大增强法）
// 进行最大增强操作。差分正规化计算得到的临时图片中每一个位置（ndif）都对应一个
// THREAD，各个 THREAD 在其对应的 dP 号所指示的方向上检查像素点的像素值，考察在 
// ndif[][] 上蓝箭头所经过的 pixel 范围内 60% (0.6) 以上的位置上的值小于 rate *
// ndif(x,y)。如果是，则 pixel(x,y) 及其 8 邻域对应位置上置 200。每个对向临域对
// 应两种检测方向，分别为上下，左右，左上右下和右上左下，根据这四种方向以及检测
// 范围参数 searchscope 进行遍历 (3 x 3 的邻域，箭头下共有 4 个 pixel；5 x 5的邻
// 域，箭头下共有 6 个 pixel；7 x 7 的邻域，箭头下共有 6 个 pixel；9 x 9 和 11 x
// 11 邻域，箭头下则都是共有8个pixel)，最后将最终得到的像素值赋给输出图像。
static __device__ void      // 无返回值。
_maxEnhancementDev(
        ImageCuda inimg,    // 输入图像。
        ImageCuda outimg,   // 输出图像。
        ImageCuda tempimg,  // 需要用到的临时图像。
        int searchscope,    // 控制最大增强方法的搜索范围参数。
        int ndif,           // 进行最大增强法的正规化后的点的像素值。
        int dp,             // 对应搜索方向的 dp 号。
        int c,              // 进行最大增强的点的横坐标。
        int r               // 进行最大增强的点的纵坐标。
);

// Device 函数：_computeMavSgmDev（计算中值平均即 MAV，方差值 SGM）
// 对某一邻域内部的像素值排序，分别计算两个邻域块内点的像素平均值和方差值，辅助
// 求解正规化差分值。在本算法中，传入一个已经统计好的数组。
static __device__ int                 // 函数若正确执行，返回 NO_ERROR。
_computeMavSgmDev(
        unsigned char pixeltmpDev[],  // 传入的一个临时数组，记载了邻域内部不同
                                      // 的像素值。
        int pixelareacnt,             // 对向邻域内不同像素点的个数。
        float *mav,                   // 计算中值平均结果。
        float *sgm                    // 计算某一个邻域的方差值 SGM
);

// Device 函数：_computeMavMaxDev（计算各个对向邻域的3个统计量）
// 计算特征向量值。分别计算已排好序的对向邻域的像素值高端处 10% 个图像值的平均值
// 作为高灰度值 hg ;计算排序结果的低端处 10% 个图像值的平均值作为低灰度值 lg;计
// 算排序结果中部 20% 个图像值的平均值作为中央均值 ag;各对向域内的 pixel的灰度的
// 整体平均值，并以此求对向域内的灰度标准偏差 sd。并同时计算出对向邻域内的像素值
// 最大值 max。
static __device__ int                 // 函数若正确执行，返回 NO_ERROR。
_computeMavMaxDev(
        unsigned char pixeltmpDev[],  // 传入的一个临时数组，记载了邻域内部不同
                                      // 的像素值。
        int pixelareacnt,             // 对向邻域内不同像素点的个数。
        float *hg,                    // 高端处 10% 个图像值的平均值,为高灰度值
        float *lg,                    // 低端处 10% 个图像值的平均值,为低灰度值
        float *ag,                    // 中部 20% 个图像值的平均值，为中央均值
        int *max                      // 对向邻域内的像素最大值 max。
);

// Kernel 函数：_detectEdgeSAKer（单纯平均法）
// 直接进行对向差分运算。利用四种对向临域的模版，进行差分正规化，得到差分计算的
// 最大值，然后进行最大增强操作。即由最新的 Maximum Enhancement 方法代替原来的
// Non-maximum Suppression 方法，对正规化后的差分值结果进行处理。最后再进行 
// Thinning 和 FrekleFilter 处理，得到单像素宽度边缘图像。但是由于非最大抑制操作
// 的消除自身的特点，检测出的边缘有很大的可能是非连续的。
static __global__ void      // Kernel 函数无返回值
_detectEdgeSAKer(
        int searchscope,    // 控制非极大值抑制方法的搜索范围参数
        int diffsize,       // 对向临域像素点个数的一半
        int neighbor[],     // 传入的模板，可以通过计算偏移量得到 8 个对向邻域的
                            // 模板坐标值。
        ImageCuda inimg,    // 输入图像
        ImageCuda tempimg,  // 在 host 函数里申请的临时图片，然后传入
                            // Kernel 函数中
                            // 用来存储差分计算后得到的各个像素点像素值
        ImageCuda outimg,   // 输出图像
        int sizedp01,       // dp 号为 0 和 1 的对向邻域的尺寸。
        int sizedp23        // dp 号为 2 和 3 的对向邻域的尺寸。
);

// Kernel 函数：_detectEdgeFVKer（特征向量法）
// 通过公式计算，进行边缘检测。首先，运用并行计算，一个线程计算一个像素点，通过
// 四种对向临域方向计算（分别为上下，左右，左上右下，右上左下四种方向），计算出
// 每一个邻域的 MAV，MMD，SGM，以及整幅图片的 EMAV，EMMD，ESGM，再利用河边老师提
// 供的公式进行计算，可以得到四个 disp 值，从中选择出最大的值，并记下其 dp 号，
// 在 dp 号对应方向上进行非最大抑制。最后，将最终结果赋值到输出图像 outimg 上，
// 再进行 Thinning 和 FreckleFilter 处理，得到最终结果。
static __global__ void      // Kernel 函数无返回值
_detectEdgeFVKer(
        ImageCuda inimg,    // 输入图像
        ImageCuda tempimg,  // 用于中间存储的临时图像。
        ImageCuda outimg,   // 输出图像
        int diffsize,       // 对向临域像素点个数
        int searchscope,    // 在 dp 方向上进行搜索的范围。
        int neighbor[],     // 传入的模板，可以通过计算偏移量得到 8 个对向邻域的
                            // 模板坐标值。
        int sizedp01,       // dp 号为 0 和 1 的对向邻域的尺寸。
        int sizedp23        // dp 号为 2 和 3 的对向邻域的尺寸。
);

// Host 函数：_initNeighbor（初始化对向邻域坐标）
static __host__ int _initNeighbor(
        int neighbor[], int diffsize, int *sizedp01, int *sizedp23)
{
    // 判断指针参数是否合法
    if (neighbor == NULL || sizedp01 == NULL || sizedp23 == NULL)
        return NULL_POINTER;

    // dp 为 0 和 1 时的邻域大小为 diffsize * diffsize。
    // 计算 dp 为 2 和 3 时重叠区域大小。
    int overlap;
    if ((diffsize + 1) & 2 != 0) {
        overlap = (diffsize + 1) / 2;
    } else {
        overlap = (diffsize - 1) / 2;
    }

    // 临时变量，用于计算点的坐标。
    int pntlocationtmp = diffsize >> 1;
    int pntlocationofftmp = overlap >> 1;
    // 分别处理每一个点的索引。
    int idx = 0;
    // 为了减少 for 循环的次数，在计算 dp 为 0 时可以根据数学分析一并将 dp 为 2
    // 的点计算，dp 为 3 和 4 时同理。
    // dp 为 0 和 dp 为 1 时的尺寸。
    int offdp01 = diffsize * diffsize * 2;
    // dp 为 0 和 dp 为 1 时存放邻域内点所需要的内存大小。 
    *sizedp01 = offdp01;
    // dp 为 2 和 dp 为 3 时存放邻域内点所需要的内存大小。
    int offdp23 = offdp01 - overlap * (overlap + 1);
    *sizedp23 = offdp23;
    // 存放 dp 为 2 及其之后的对向邻域的点坐标，相对于数组起始位置的偏移量。
    int offdp12 = offdp01 << 2;
    // 为模板赋值。
    for (int i = -diffsize; i < 0; i++) {
        for (int j = -pntlocationtmp; j <= pntlocationtmp; j++, idx++) {
            // dp 为 0 时记录点的横纵坐标。
            neighbor[idx << 1] = i;
            neighbor[(idx << 1) + 1] = j;
            // 据分析 dp 为 0 时左块和右块点的坐标存在着某种关系：
            // 关于 y 轴对称。
            neighbor[offdp01 + (idx << 1)] = -i;
            neighbor[offdp01 + (idx << 1) + 1] = j;
            
            // 据分析dp 为 0 和 dp 为 1 的点的坐标存在着某种关系：
            // x 和 y 值交换。
            // 故可以推出 dp 为 1 时的坐标，如下：
            neighbor[(offdp01 << 1) + (idx << 1)] = j;
            neighbor[(offdp01 << 1) + (idx << 1) + 1] = i;

            // 据分析 dp 为 1 上块和下块的点的坐标存在着某种关系：
            // 关于 x 轴对称。
            neighbor[offdp01 * 3 + (idx << 1)] = j;
            neighbor[offdp01 * 3 + (idx << 1) + 1] = -i;
        }
    }        
    // 为数组赋值 dp 为 2 和 dp 为 3 时的情形。
    // 计算左边第一个点的横纵坐标。
    int firstpntx = -(diffsize - (overlap + 1) / 2);
    int firstpnty = -(diffsize - (overlap + 1) / 2);
    // 计算模板值。
    idx = 0;
    for (int i = firstpntx; i <= pntlocationofftmp; i++) {
        for (int j = firstpnty; j <= pntlocationofftmp; j++) {
            // 保证点的有效性，舍去不符合要求的点。
            // 而此时也已经找到了邻域中所有的点，所以可以返回 NO_ERROR
            if (i + j >= 0)
                continue;
            // dp 为 2 时记录点的横纵坐标。
            neighbor[offdp12 + (idx << 1)] = i;
            neighbor[offdp12 + (idx << 1) + 1] = j;

            // 根据分析发现，dp 为 2 时左上块和右下块点的坐标存在着某种
            // 关系：关于 y = -x 对称。
            neighbor[offdp12 + offdp23 + (idx << 1)] = -j;
            neighbor[offdp12 + offdp23 + (idx << 1) + 1] = -i;
            
            // 根据分析发现，dp 为 2 左上块和 dp 为 3 右上块的点的坐标存
            // 在着某种关系：x 值互为相反数。
            // 故可以推出 dp 为 3 时的模板，如下：
            neighbor[offdp12 + (offdp23 << 1) + (idx << 1)] = -i;
            neighbor[offdp12 + (offdp23 << 1) + (idx << 1) + 1] = j;

            // 根据分析发现，dp 为 3 左上块和右下块的点的坐标存在着某种
            // 关系：关于 y = x 对称。
            neighbor[offdp12 + offdp23 * 3 + (idx << 1)] = j;
            neighbor[offdp12 + offdp23 * 3 + (idx << 1) + 1] = -i;

            idx++;
        }
    }

    // 调试代码，输出所有生成的坐标点集
    //for (int i = 0; i < offdp12 + offdp23 * 4; i += 2) {
    //    cout << "(" << neighbor[i] << ", " << neighbor[i + 1] << "), ";
    //    if (i % 30 == 0) cout << endl;
    //}
    //cout << endl;

    // 处理完毕，退出。
    return NO_ERROR;
}

// Device 函数：_maxEnhancementDev（最大增强法）
static __device__ void _maxEnhancementDev(
        ImageCuda inimg, ImageCuda outimg, ImageCuda tempimg,
        int searchscope, int ndif, int dp, int c, int r)
{
    // dp 号为 0，左右方向检测
    // dp 号为 1，上下方向检测
    // dp 号为 2，左上右下方向检测
    // dp 号为 3，右上左下方向检测
    int curc = c, curr = r;  // 临时变量表示点的坐标。
    int arrsub = 0;          // 定义一个临时变量，用来定位 Device 数组的下标。
    int icounter = 0;        // 辅助计算的计数临时变量。
    float rate = 0.4f;        // 外部指定参数，要求 0 < rate <= 1，这里设为0.5。
    for (int i = 1; i <= searchscope; i++) {
        // 在左（上、左上、右上）方向上搜索，如果有符合要求的值，则计数变量自加
        arrsub = 0;
        curc = c + _argSupreDev[dp][arrsub++] * i;
        curr = r + _argSupreDev[dp][arrsub++] * i;
        if (tempimg.imgMeta.imgData[curr * inimg.pitchBytes + curc] < 
            (unsigned char)(rate * ndif) && curc >= 0 && 
            curc < inimg.imgMeta.width && curr >= 0 
            && curr < inimg.imgMeta.height) {
            icounter++;
        }

        // 在右（下、右下、左下）方向上搜索 ，如果有符合要求的值，则计数变量自加
        curc = c + _argSupreDev[dp][arrsub++] * i;
        curr = r + _argSupreDev[dp][arrsub++] * i;
        if (tempimg.imgMeta.imgData[curr * inimg.pitchBytes + curc] <
            (unsigned char)(rate * ndif) && curc >= 0 && 
            curc < inimg.imgMeta.width && curr >= 0 
            && curr < inimg.imgMeta.height) {
            icounter++;
        }
    }

    // 在所判断方向上经过的 pixel 范围内 60% 以上的位置，如果都符合要求，则
    // pixel(x,y) 及其 8邻域对应位置上置 200
    if((icounter * 1.0f / searchscope) > 0.3) {
        outimg.imgMeta.imgData[r * inimg.pitchBytes + c] = 200;
        outimg.imgMeta.imgData[r * inimg.pitchBytes + c + 1] = 200;
        outimg.imgMeta.imgData[r * inimg.pitchBytes + c - 1] = 200;
        outimg.imgMeta.imgData[(r + 1) * inimg.pitchBytes + c] = 200;
        outimg.imgMeta.imgData[(r - 1) * inimg.pitchBytes + c] = 200;
        outimg.imgMeta.imgData[(r + 1) * inimg.pitchBytes + c + 1] = 200;
        outimg.imgMeta.imgData[(r + 1) * inimg.pitchBytes + c + 1] = 200;
        outimg.imgMeta.imgData[(r - 1) * inimg.pitchBytes + c - 1] = 200;
        outimg.imgMeta.imgData[(r - 1) * inimg.pitchBytes + c - 1] = 200;
    }
    // 否则置 0。
    else
        outimg.imgMeta.imgData[r * inimg.pitchBytes + c] = 0;
}

// Kernel 函数：_detectEdgeSAKer（单纯平均法）
static __global__ void _detectEdgeSAKer(
        int searchscope, int diffsize, int neighbor[], ImageCuda inimg,
        ImageCuda tempimg, ImageCuda outimg, int sizedp01, int sizedp23)
{
    // 计算线程对应的输出点的位置，线程处理的像素点的坐标的 c 和 r 分量，z 表示
    // 对应的邻域方向，其中 0 到 3 分别表示左右、上下、左上右下、右上左下。
    // 采用的是二维的 grid，三维的 block。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    // 计算当前线程在线程块内的索引。
    int tidinblk = z * blockDim.x * blockDim.y +
                   threadIdx.y * blockDim.x + threadIdx.x;

    // 申请动态共享内存。
    extern __shared__ int shdedgesa[];

    // 为了只使用一个线程来做最大增强法，此处默认选择 z 为 0 的线程来做最大增强
    // 法，但是这个线程可能在边界处理时被 return 掉，因此需要一个标记值，当 z 
    // 为 0 的线程 return 之后由其他线程来做最大增强。
    int *shdflag = &shdedgesa[0];
    // 每一个点的 0 号线程在线程块内的索引。
    int index = threadIdx.y * blockDim.x + threadIdx.x;
    if (z == 0)
        shdflag[index] = 0; 
    // 在共享内存中申请出一段空间用来存放 4 个对向邻域差分正规化值的结果。
    int *shddiffvalue = &shdflag[blockDim.x * blockDim.y];

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算当前线程要用到的模板地址。
    int templateidx = (z < 2 ? z * sizedp01 :
                       (sizedp01 << 1) + (z - 2) * sizedp23);
    int *curtemplate = &neighbor[templateidx];

    // 计算重叠区域大小。
    int overlap;
    if ((diffsize + 1) & 2 != 0) {
        overlap = (diffsize + 1) >> 1;
    } else {
        overlap = (diffsize - 1) >> 1;
    }
    // 一个临时变量用来判断 dp 为 3 和 4 时的边界。
    int offcoord = diffsize - (overlap  + 1) / 2;

    // 判断边缘点，将其像素值置零。
    // 并不再进行下面的处理。
    // 分别对应 dp 为 0，1，2 和 3 时的边界情况。
    // 分别对应 dp 为 0，1，2 和 3 时的边界点放在数组中。
    unsigned char edgec[4] = {diffsize, diffsize >> 1, offcoord, offcoord};
    unsigned char edger[4] = {diffsize >> 1, diffsize, offcoord, offcoord};

    // 判断是否是边界点，如果是则置零并退出。
    if (c < edgec[z] || c >= inimg.imgMeta.width - edgec[z] ||
        r < edger[z] || r >= inimg.imgMeta.height - edger[z]) {
        tempimg.imgMeta.imgData[r * inimg.pitchBytes + c] = 0;
        // 为了防止某些点的 4 个线程都出界，故先将输出图像的对应点也置为 0；
        outimg.imgMeta.imgData[r * inimg.pitchBytes + c] = 0;
        // 将值写入到共享内存。
        shddiffvalue[tidinblk] = 0;
        // 如果 z 为 0 的线程由于边界判断被 return，则将重新设置标记值。
        // 此处的 255 仅仅表示非 0 的概念，没有实际意义。
        if (z == 0)
            shdflag[index] = 255;
        return;
    }

    // 当标记值非 0 时，即 z 为 0 的线程已经不复存在了，此时需要更换新的标记值。
    // 这时可能用的有 z 为 1、2、3 线程，为了减少 bank conflict，同时又因为 z
    // 为 2 和 3 必然同时存在或者同时被 return，故判断 z 是否为奇数，可以将 3
    // 路冲突改为 2 路冲突。
    if (shdflag[index] != 0 && z & 1 != 0)
        shdflag[index] = z;

    // 块内同步，保证新的标记值已经写入到共享内存中。
    __syncthreads();

    // 在计算中用来记录点的像素值的临时变量。
    int curgray = 0;

    // 申请两个中间数组，分别用来存放对向邻域两个块内点的值。
    unsigned char pixeltmpDev1[RED_ARRAY_MAXLEN * RED_ARRAY_MAXLEN] = { 0 };
    unsigned char pixeltmpDev2[RED_ARRAY_MAXLEN * RED_ARRAY_MAXLEN] = { 0 };

    // 用 for 循环，分别算出每个对向临域的各个点的索引。
    int curc = c, curr = r;
    // 点的坐标个数。
    int pntcnt = (z < 2) ? sizedp01: sizedp23;

    // 邻域内部点的数目.
    int pixelareacnt = 0;
    for (int i = 0; i < pntcnt; i = i + 2, pixelareacnt++) {
        // 统计对向邻域的第一模板内的点的坐标。
        curc = c + curtemplate[i];
        curr = r + curtemplate[i + 1];
        // 取出第一个邻域内的点的像素值并统计到对应的数组中。
        curgray = inimg.imgMeta.imgData[curr * inimg.pitchBytes + curc];
        // 利用像素个数进行判断，将两个对向邻域块内的值分别记录到两个数组中。
        if (pixelareacnt < (pntcnt >> 2))
            pixeltmpDev1[pixelareacnt] = curgray;
        else 
            pixeltmpDev2[pixelareacnt - (pntcnt >> 2)] = curgray;
    }

    // 块内同步，保证块内的差分值都写入到了共享内存中。
    __syncthreads();

    // 设置临时变量 sgm1，sgm2，来分别存储两个对向邻域块的方差值，sgm记录 sgm1 
    // 和 sgm2 中较大的值，来进行最后的正规化计算。
    float sgm1, sgm2, sgm;

    // 设置临时变量 sgm1，sgm2，来分别存储两个对向邻域块的平均值。
    float mav1, mav2;

    // 调用 device 端的函数求解对向邻域的像素平均值和方差。
    _computeMavSgmDev(pixeltmpDev1, pixelareacnt / 2, &mav1, &sgm1);

    // 调用 device 端的函数求解对向邻域的像素平均值和方差。
    _computeMavSgmDev(pixeltmpDev2, pixelareacnt / 2, &mav2, &sgm2);

    // 比较出 sgm1 和 sgm2 两者中较大的赋值给 sgm。
    sgm = (sgm1 > sgm2) ? sgm1 : sgm2;

    // 设 ndif 为两个对向域之间的正规化差分值,数组 t 和 k 为计算正规化差分值
    // ndif 的参数数组，大小都为 10。
    int ndif = 0, dp = 0;
    double t[10] = { 9, 25, 49, 81, 121, 169, 225, 289, 361, 441 } ;
    double k[10] = { 0.001, 0.005, 0.025, 0.125, 0.625, 3.125,
                     15.624, 78.125, 390.625, 1953.125 } ;

    shddiffvalue[tidinblk] = (mav1 - mav2) * (mav1 - mav2) / 
                             (t[4] + k[5] * sgm);

    // 块内同步，保证块内的正规化差分值都写入到了共享内存中。
    __syncthreads();

    // 只需要标记的线程来做以下处理。
    if (z != shdflag[index])
        return;
    // 用设定的变量 ndif 来存储四种方向正规化差分计算的最大值，dp 记录四种方向
    // 中差分值最大的方向。
    // 局部变量，方便一下计算。
    int offinblk = blockDim.x * blockDim.y;
    ndif = shddiffvalue[index];
    for (int i = index + offinblk, j = 1;
        i < index + 4 * offinblk; i += offinblk, j++) {
        if (ndif < shddiffvalue[i]) {
            ndif = shddiffvalue[i];
            dp = j;
        }
    }

    // 块内同步，保证块内的正规化差分值都写入到了共享内存中。
    __syncthreads();

    // 根据是否有宏 RED_CODE_CHOICE，来决定调用那部分检测边缘的代码。
#ifndef RED_CODE_CHOICE
    // 判断是否是边界点。如果 ndfi 大于等于 1，则证明是边界点，在输出图像的对应
    // 位置像素点置 200，否则置 0。
    if(ndif >= 1)
        outimg.imgMeta.imgData[r * inimg.pitchBytes + c] = 200;
    else
        outimg.imgMeta.imgData[r * inimg.pitchBytes + c] = 0;

    _maxEnhancementDev(inimg, tempimg, inimg, searchscope, ndif, dp, c, r); 
#else
    // 将正规化后的差分值赋给当前像素点，存储到新的临时图片上。
    tempimg.imgMeta.imgData[r * inimg.pitchBytes + c] = ndif;
    __syncthreads();

    // 设置数组 assist 辅助计算最大增强法的搜索范围参数 searchscope，其中第一个
    // 数据 1 为辅助数据，没有实际意义。
    int assist[6] = {1, 2, 3, 3, 4, 4};
    // 计算 searchscope 的值。
    searchscope = assist[(diffsize - 1) / 2];

    // 进行最大增强操作。
    _maxEnhancementDev(inimg, outimg, tempimg, searchscope, ndif, dp, c, r);  
#endif
}

// 宏：FAIL_RED_SA_FREE
// 该宏用于清理在申请的设备端或者主机端内存空间。
#define FAIL_RED_SA_FREE  do  {                   \
        if (tempimg != NULL)                      \
            ImageBasicOp::deleteImage(tempimg);   \
        if (tempimg1 != NULL)                     \
            ImageBasicOp::deleteImage(tempimg1);  \
        if (neighborDev != NULL)                  \
            cudaFree(neighborDev);                \
        if (neighbor != NULL)                     \
            delete [] (neighbor);                 \
    } while (0)

// Host 成员方法：detectEdgeSA（单纯平均法）
__host__ int RobustEdgeDetection::detectEdgeSA(
        Image *inimg, Image *outimg, CoordiSet *guidingset)
{
    // 检查输入图像和输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 检查图像是否为空
    if (inimg->imgData == NULL)
        return UNMATCH_IMG;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码
    
    // guidingset 为边缘检测的指导区域，如果 guidingset 不为空，暂未实现。
    if (guidingset != NULL) {
        return UNIMPLEMENT;
    }

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图像
        // 的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->roiX2 - inimg->roiX1,
                inimg->roiY2 - inimg->roiY1);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 临时图像和存放邻域坐标的数组已经在 Device 端的邻域坐标数组。
    Image *tempimg = NULL, * tempimg1 = NULL;
    int *neighbor = NULL, *neighborDev = NULL;
    // 创建临时图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR) {
        FAIL_RED_SA_FREE;
        return errcode;
    }

    // 将 temp 图像在 Device 内存中建立数据。
    errcode = ImageBasicOp::makeAtCurrentDevice(
            tempimg, inimg->roiX2 - inimg->roiX1,
            inimg->roiY2 - inimg->roiY1);
    // 如果创建图像操作失败，则释放内存报错退出。
    if (errcode != NO_ERROR) {
        FAIL_RED_SA_FREE;
        return errcode;
    }

    // 创建第二幅临时图像，供调用 Thinning 函数和 FreckleFilter 函数使用
    errcode = ImageBasicOp::newImage(&tempimg1);
    if (errcode != NO_ERROR) {
        FAIL_RED_SA_FREE;
        return errcode;
    }

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR) {
        FAIL_RED_SA_FREE;
        return errcode;
    }

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        FAIL_RED_SA_FREE;
        return errcode;
    }

    // 提取临时图像 tempimg 的 ROI 子图像。
    ImageCuda subimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &subimgCud);
    if (errcode != NO_ERROR) {
        FAIL_RED_SA_FREE;
        return errcode;
    }

    // 记录不同 dp 情况下邻域中点的横纵坐标个数总和。
    int sizedp01 = 0, sizedp23 = 0;
    // 申请 Host 端记录邻域坐标的空间。
    neighbor = new int[16 * diffsize * diffsize];
    // 判断空间是否申请成功。
    if (neighbor == NULL) {
        // 释放空间。
        FAIL_RED_SA_FREE;
        return OUT_OF_MEM;    
    }
    // 调用 _initNeighbor
    errcode = _initNeighbor(neighbor, diffsize, &sizedp01, &sizedp23);
    // 如果调用失败，则删除临时图像和临时申请的空间。
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_SA_FREE;
        return errcode;
    }

    // 将邻域坐标 neighbor 传入 Device 端。
    // 为 neighborDev 在 Device 端申请空间。
    int cudaerrcode = cudaMalloc((void **)&neighborDev,
                                 sizeof (int) * 16 * diffsize * diffsize);
    // 若申请不成功则释放空间。
    if (cudaerrcode != cudaSuccess) {
        // 释放空间。
        FAIL_RED_SA_FREE;
        return CUDA_ERROR;
    }
    // 将 neighbor 拷贝到 Device 端的 neighborDev 中。
    cudaerrcode = cudaMemcpy(neighborDev, neighbor,
                             sizeof (int) * 16 * diffsize * diffsize,
                             cudaMemcpyHostToDevice);
    // 如果拷贝不成功，则释放空间。
    if (cudaerrcode != cudaSuccess) {
        // 释放空间。
        FAIL_RED_SA_FREE;
        return CUDA_ERROR;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = SA_DEF_BLOCK_X;
    blocksize.y = SA_DEF_BLOCK_Y;
    blocksize.z = SA_DEF_BLOCK_Z;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) /
                 blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) /
                 blocksize.y;
    gridsize.z = 1;

    // 计算用到的共享内存空间大小。
    int memsize = SA_DEF_BLOCK_X * SA_DEF_BLOCK_Y * (SA_DEF_BLOCK_Z + 1) *
                  sizeof (int);

    // 调用核函数
    _detectEdgeSAKer<<<gridsize, blocksize, memsize>>>(
            searchScope, diffsize, neighborDev, insubimgCud,
            subimgCud, outsubimgCud, 2 * sizedp01, 2 * sizedp23);

    // 判断 CUDA 调用是否出错。
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_RED_SA_FREE;
        return CUDA_ERROR;
    }

    // 调用 Thinning
    errcode = this->thinning.thinMatlabLike(outimg, tempimg1);
    if (errcode != NO_ERROR) {
        FAIL_RED_SA_FREE;
        return errcode;
    }

    // 调用 FreckleFilter
    errcode = this->frecklefilter.freckleFilter(tempimg1, outimg);
    if (errcode != NO_ERROR) {
        FAIL_RED_SA_FREE;
        return errcode;
    }

    // 退出前删除临时图像。
    ImageBasicOp::deleteImage(tempimg);
    ImageBasicOp::deleteImage(tempimg1);
    delete [] neighbor;
    cudaFree(neighborDev);

    // 处理完毕，退出。
    return NO_ERROR;
}

// Device 函数：_computeMavSgmDev（计算中值平均即 MAV，方差值 SGM）
static __device__ int _computeMavSgmDev(
        unsigned char pixeltmpDev[], int pixelareacnt, float *mav, float *sgm)
{
    // 判断输入参数的有效性。
    if (pixeltmpDev == NULL || mav == NULL || sgm == NULL)
        return INVALID_DATA;

    // 若邻域内部的点的值都一样，则不用再进行下面的计算，直接返回。
    if (pixeltmpDev[0] == pixeltmpDev[pixelareacnt - 1]) {
        *mav = pixeltmpDev[0];
        *sgm = 0.0f;
        return NO_ERROR;
    }

    // 定义一个临时变量 sumpixel 来计算数组里下标从 up 到 down 的像素值和。
    double sumpixel = 0.0f;

    // 累加像素值和。
    for (int i = 0; i < pixelareacnt; i++)
        sumpixel += pixeltmpDev[i];

    // 计算 MAV。
    *mav = sumpixel / pixelareacnt;

    // 计算 SGM。
    double sum = 0.0;
    // 先累加每一个像素值和平均值的差的平方和。
    for (int i = 0; i < pixelareacnt; i++)
        sum += (pixeltmpDev[i] - *mav) * (pixeltmpDev[i] - *mav);
    // 计算方差值。 
    *sgm = sum / pixelareacnt;

    // 正常执行，返回 NO_ERROR。
    return NO_ERROR;
}

// Device 函数：_computeMavMaxDev（计算各个对向邻域的3个统计量）
static __device__ int _computeMavMaxDev(
        unsigned char pixeltmpDev[], int pixelareacnt, float *hg, float *lg,
        float *ag, int *max) 
{
    // 判断输入参数的有效性。
    if (pixeltmpDev == NULL || max == NULL || hg == NULL || lg == NULL ||
        ag == NULL)
        return INVALID_DATA;

    // 若邻域内部的点的值都一样，则不用再进行下面的计算，直接返回。
    if (pixeltmpDev[0] == pixeltmpDev[pixelareacnt - 1]) {
        *max = pixeltmpDev[0];
        *hg = pixeltmpDev[0];
        *lg = pixeltmpDev[0];
        *ag = pixeltmpDev[0];
        return NO_ERROR;
    }

    // 排序后数组成降序，可直接获取最大值。
    *max = pixeltmpDev[pixelareacnt - 1];

    // 计数器 icounter，jcounter，初始化均为 0。
    int icounter = 0, jcounter = 0;
    // 统计变量 sum，初始化为 0。
    float sum = 0.0f;

    // 计算低灰度值 lg
    jcounter = pixelareacnt / 10;
    for(icounter = 0; icounter < jcounter; icounter++)
        sum += pixeltmpDev[icounter];

    *lg = sum / jcounter;

    // 计算高灰度值 hg
    sum = 0.0f;
    jcounter = pixelareacnt - (pixelareacnt / 10);
    for(icounter = jcounter; icounter < pixelareacnt; icounter++)
        sum += pixeltmpDev[icounter];

    jcounter = pixelareacnt / 10;
    *hg = sum / jcounter;

    // 计算中央灰度均值 ag
    sum = 0.0f;
    jcounter = pixelareacnt / 10;
    for(icounter = (4 * jcounter); icounter < (6 * jcounter); icounter++)
        sum += pixeltmpDev[icounter];

    *ag = sum / jcounter;

    // 正常执行，返回 NO_ERROR。
    return NO_ERROR;
}

// Kernel 函数：_detectEdgeFVKer（特征向量法）
// 通过公式计算，进行边缘检测。
static __global__ void _detectEdgeFVKer(
        ImageCuda inimg, ImageCuda tempimg, ImageCuda outimg,
        int diffsize, int searchscope, int neighbor[], int sizedp01, 
        int sizedp23)
{
   // 计算线程对应的输出点的位置，线程处理的像素点的坐标的 c 和 r 分量，z 表示
    // 对应的邻域方向，其中 0 到 3 分别表示左右、上下、左上右下、右上左下。
    // 采用的是二维的 grid，三维的 block。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    // 计算当前线程在线程块内的索引。
    int tidinblk = z * blockDim.x * blockDim.y +
                   threadIdx.y * blockDim.x + threadIdx.x;

    // 申请动态共享内存。
    extern __shared__ int shdedgesa[];

    // 为了只使用一个线程来做非极大值抑制，此处默认选择 z 为 0 的线程来做非极大
    // 值抑制，但是这个线程可能在边界处理时被 return 掉，因此需要一个标记值，
    // 当 z 为 0 的线程 return 之后由其他线程来做非极大值抑制。
    int *shdflag = &shdedgesa[0];

    // 块内同步，保证新的标记值已经写入到共享内存中。
    __syncthreads();

    // 每一个点的 0 号线程在线程块内的索引。
    int index = threadIdx.y * blockDim.x + threadIdx.x;
    if (z == 0)
        shdflag[index] = 0; 

    // 块内同步，保证新的标记值已经写入到共享内存中。
    __syncthreads();

    // 在共享内存中申请出一段空间用来存放 4 个对向邻域差分正规化值的结果。
    int *shddiffvalue = &shdflag[blockDim.x * blockDim.y];

    // 块内同步，保证新的标记值已经写入到共享内存中。
    __syncthreads();

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算当前线程要用到的模板地址。
    int templateidx = (z < 2 ? z * sizedp01 :
                       (sizedp01 << 1) + (z - 2) * sizedp23);
    int *curtemplate = &neighbor[templateidx];

    // 计算重叠区域大小。
    int overlap;
    if ((diffsize + 1) & 2 != 0) {
        overlap = (diffsize + 1) >> 1;
    } else {
        overlap = (diffsize - 1) >> 1;
    }
    // 一个临时变量用来判断 dp 为 3 和 4 时的边界。
    int offcoord = diffsize - (overlap + 1) / 2;

    // 判断边缘点，将其像素值置零。
    // 并不再进行下面的处理。
    // 分别对应 dp 为 0，1，2 和 3 时的边界情况。
    // 分别对应 dp 为 0，1，2 和 3 时的边界点放在数组中。
    unsigned char edgec[4] = {diffsize, diffsize >> 1, offcoord, offcoord};
    unsigned char edger[4] = {diffsize >> 1, diffsize, offcoord, offcoord};

    // 判断是否是边界点，如果是则置零并退出。
    if (c < edgec[z] || c >= inimg.imgMeta.width - edgec[z] ||
        r < edger[z] || r >= inimg.imgMeta.height - edger[z]) {
        tempimg.imgMeta.imgData[r * inimg.pitchBytes + c] = 0;
        // 为了防止某些点的 4 个线程都出界，故先将输出图像的对应点也置为 0；
        outimg.imgMeta.imgData[r * inimg.pitchBytes + c] = 0;
        // 将值写入到共享内存。
        shddiffvalue[tidinblk] = 0;
        // 如果 z 为 0 的线程由于边界判断被 return，则将重新设置标记值。
        // 此处的 255 仅仅表示非 0 的概念，没有实际意义。
        if (z == 0)
            shdflag[index] = 255;
        return;
    }

    // 块内同步，保证新的标记值已经写入到共享内存中。
    __syncthreads();

    // 当标记值非 0 时，即 z 为 0 的线程已经不复存在了，此时需要更换新的标记值。
    // 这时可能用的有 z 为 1、2、3 线程，为了减少 bank conflict，同时又因为 z
    // 为 2 和 3 必然同时存在或者同时被 return，故判断 z 是否为奇数，可以将 3
    // 路冲突改为 2 路冲突。
    if (shdflag[index] != 0 && z & 1 != 0)
        shdflag[index] = z;

    // 块内同步，保证新的标记值已经写入到共享内存中。
    __syncthreads();

    // 在计算中用来记录点的像素值的临时变量。
    int curgray = 0;

    // 申请两个中间数组，分别用来存放对向邻域两个块内点的值。
    unsigned char pixeltmpDev1[RED_ARRAY_MAXLEN * RED_ARRAY_MAXLEN] = { 0 };
    unsigned char pixeltmpDev2[RED_ARRAY_MAXLEN * RED_ARRAY_MAXLEN] = { 0 };

    // 用 for 循环，分别算出每个对向临域的各个点的索引
    int curc = c, curr = r;
    // 点的坐标个数。
    int pntcnt = (z < 2) ? sizedp01: sizedp23;
    int middle = pntcnt >> 2;
    // 邻域内部点的数目.
    int pixelareacnt = 0;
    for (int i = 0; i < pntcnt; i = i + 2, pixelareacnt++) {
        // 统计对向邻域的第一模板内的点的坐标。
        curc = c + curtemplate[i];
        curr = r + curtemplate[i + 1];
        // 取出第一个邻域内的点的像素值并统计到对应的数组中。
        curgray = inimg.imgMeta.imgData[curr * inimg.pitchBytes + curc];
        // 利用像素个数进行判断，将两个对向邻域块内的值分别记录到两个数组中。
        if (pixelareacnt < middle)
            pixeltmpDev1[pixelareacnt] = curgray;
        else{
            pixeltmpDev2[pixelareacnt - middle] = curgray;}
    }

    // 块内同步，保证块内的差分值都写入到了共享内存中。
    __syncthreads();

    // 为两个 pixeltmpDev 排序，经过上面的处理，现在数组内部只有 pixelareacnt 
    // 个值，此时可以使用插入排序。
    for (int i = 1; i < (pixelareacnt / 2); i++) {
        int sorttmp = pixeltmpDev1[i];
        int j = i - 1;
        while (j >= 0 && sorttmp < pixeltmpDev1[j]) {
            pixeltmpDev1[j + 1] = pixeltmpDev1[j];
            j = j - 1;
        }
        pixeltmpDev1[j + 1] = sorttmp;
    }

    for (int i = 1; i < (pixelareacnt / 2); i++) {
        int sorttmp = pixeltmpDev2[i];
        int j = i - 1;
        while (j >= 0 && sorttmp < pixeltmpDev2[j]) {
            pixeltmpDev2[j + 1] = pixeltmpDev2[j];
            j = j - 1;
        }
        pixeltmpDev2[j + 1] = sorttmp;
    }

    // 计算邻域平均高灰度值 hg，平均低灰度值 lg 和中央均值ag 以
    // 及对向邻域内像素值的最大值 max。
    // 计算邻域整体的平均值 MAV，方差 SGM。
    // 调用 device 函数。
    float mav1 = 0.0f, mav2 = 0.0f;
    float sgm1 = 0.0f, sgm2 = 0.0f;
    float hg1 = 0.0f, hg2 = 0.0f;
    float lg1 = 0.0f, lg2 = 0.0f;
    float ag1 = 0.0f, ag2 = 0.0f;
    int max1 = 0, max2 = 0;

    // 调用 device 端的函数求解一个对向邻域块的 MAV 和 SGM。
    _computeMavSgmDev(pixeltmpDev1, pixelareacnt / 2, &mav1, &sgm1);
    // 调用 device 端的函数求解一个对向邻域块儿的平均高灰度值 hg，平均低灰度值
    // lg 和中央均值ag 以及对向邻域内像素值的最大值 max。
    _computeMavMaxDev(pixeltmpDev1, pixelareacnt / 2, &hg1, &lg1, &ag1, &max1);

    // 调用 device 端的函数求解另一个对向邻域块的 MAV 和 SGM。
    _computeMavSgmDev(pixeltmpDev2, pixelareacnt / 2, &mav2, &sgm2);
    // 调用 device 端的函数求解另一个对向邻域块儿的平均高灰度值 hg，平均低灰度值
    // lg 和中央均值ag 以及对向邻域内像素值的最大值 max。
    _computeMavMaxDev(pixeltmpDev2, pixelareacnt / 2, &hg2, &lg2, &ag2, &max2);

    // 求解对向邻域的三个特征向量。
    // 设置临时变量。
    int aa = 1, bb = 1, cc = 1;  // 外部参数

    float abc = aa + bb + cc;
    float A = aa / abc;
    float B = bb / abc;
    float C = cc / abc;

    float dg1 = hg1 - lg1, dg2 = hg2 - lg2;
    float s1 = 255 / (1 + max1 - mav1), s2 = 255 / (1 + max2 - mav2);
    float sd1 = s1 * sqrt(sgm1), sd2 = s2 * sqrt(sgm2);

    // 特征向量对向域间的相关系数 indexc。
    float indexc;

    // 设置几个辅助变量，辅助计算特征向量对向域间的相关系数 indexc, 无实际含义。
    float m0 = A * dg1 * dg2 + B * ag1 * ag2 + C * sd1 * sd2;
    float m1 = A * dg1 * dg1 + B * ag1 * ag1 + C * sd1 * sd1;
    float m2 = A * dg2 * dg2 + B * ag2 * ag2 + C * sd2 * sd2;

    // 用块内的偶数号线程来根据公式计算，可以得到 4 个值，
    // 并存放在共享内存中。
    // 计算特征向量对向域间的相关系数 indexc。
    indexc = (m1 * m2 == 0) ? 1.0f : m0 / (3 * sqrt(m1 * m2));

    // 将结果存入共享内存。
    shddiffvalue[tidinblk] = indexc;

    // 块内同步。
    __syncthreads();

    // 以下处理主要是找出 indexc 的最小值，并在其对应的 dp 方向上做最大增强，如
    // 果只用 1 个线程来处理该步骤可能会出现该线程被提前 return 掉的情形，所以此
    // 步骤采用偶数线程来处理，并将选出的 minc 的最小值处理，赋值赋值给临时图片
    // tempimg。
    // 只需要标记的线程来做以下处理。
    if (z != shdflag[index])
        return;
    // 记录最小 minc 值的方向。
    // 设定变量 minc 来存储四种方向差分计算的最小值，dp 记录四种方向中差分值最
    // 大的方向。
    int offinblk = blockDim.x * blockDim.y;  // 局部变量，方便一下计算。
    float minc = shddiffvalue[index];
    int dp = 0;
    for (int i = index + offinblk, j = 1;
        i < index + 4 * offinblk; i += offinblk, j++) {
        if (minc > shddiffvalue[i]) {
            minc = shddiffvalue[i];
            dp = j;
        }
    }

    // 块内同步，保证新的标记值已经写入到共享内存中。
    __syncthreads();

    // 外部指定参数
    float distRate = 0.5f;
    float maxMahaDist = 3.0f;
    float correTh = 0.5f;
    float centerFV[3] = { 1.0f, 1.0f, 1.0f };
    // 此乃给定的对称半正定矩阵
    float variMatrix[9] =
            { 1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 2.0f, 1.0f } ;

    // 将 minc 与外部传入参数做比较，进行判断赋值
    int ipixel = 0;
    if(minc < correTh) { 
        // 将最大值赋给当前像素点，存储到临时图片上。
        ipixel = 255 * (1 - minc);
        tempimg.imgMeta.imgData[r * inimg.pitchBytes + c] = ipixel;
        // edge- likelihood- score的印加方法
        if(centerFV != NULL  &&  variMatrix != NULL){ 
            // edge- likelihood- score的印加
            // 计算两个对向域特征 (dgi, agi, sdi) (i =1,2) 和 centerFV 之间的马
            // 氏距离。
            float v1[3], v2[3];
            float md1, md2, minmd;
            v1[0] = abs(dg1 - centerFV[0]) * A;
            v1[1] = abs(ag1 - centerFV[1]) * B;
            v1[2] = abs(sd1 - centerFV[2]) * C;

            v2[0] = abs(dg2 - centerFV[0]) * A;
            v2[1] = abs(ag2 - centerFV[1]) * B;
            v2[2] = abs(sd2 - centerFV[2]) * C;

            md1 = sqrt((v1[0] * (v1[0] * variMatrix[0] + v1[1] * variMatrix[1] +
                                 v1[2] * variMatrix[2])) +
                       (v1[1] * (v1[0] * variMatrix[0] + v1[1] * variMatrix[1] +
                                 v1[2] * variMatrix[2])) +
                       (v1[2] * (v1[0] * variMatrix[0] + v1[1] * variMatrix[1] +
                                 v1[2] * variMatrix[2])));

            md2 = sqrt((v2[0] * (v2[0] * variMatrix[0] + v2[1] * variMatrix[1] +
                                 v2[2] * variMatrix[2])) +
                       (v2[1] * (v2[0] * variMatrix[0] + v2[1] * variMatrix[1] +
                                 v2[2] * variMatrix[2])) +
                       (v2[2] * (v2[0] * variMatrix[0] + v2[1] * variMatrix[1] +
                                 v2[2] * variMatrix[2])));

            minmd = (md1 < md2) ? md1 : md2;

            if(minmd > (distRate * maxMahaDist)) {
                ipixel = 255 * (1 - minmd / maxMahaDist);
                tempimg.imgMeta.imgData[r * inimg.pitchBytes + c] = ipixel;
            }
        } 
    } 

    else
    // 最终为图片赋值。
    tempimg.imgMeta.imgData[r * inimg.pitchBytes + c] = 0;

    // 设置数组 assist 辅助计算最大增强法的搜索范围参数 searchscope，其中第一个
    // 数据 1 为辅助数据，没有实际意义。
    int assist[6] = {1, 2, 3, 3, 4, 4};
    // 计算 searchscope 的值。
    searchscope = assist[(diffsize - 1) / 2];

    // 进行最大增强操作。
    _maxEnhancementDev(inimg, outimg, tempimg, searchscope, ipixel, dp, c, r);
}

// 宏：FAIL_RED_FV_FREE
// 该宏用于清理在申请的设备端或者主机端内存空间。
#define FAIL_RED_FV_FREE  do  {                   \
        if (tempimg != NULL)                      \
            ImageBasicOp::deleteImage(tempimg);   \
        if (tempimg1 != NULL)                     \
            ImageBasicOp::deleteImage(tempimg1);  \
        if (neighborDev != NULL)                  \
            cudaFree(neighborDev);                \
        if (neighbor != NULL)                     \
            delete [] (neighbor);                 \
    } while (0)

// Host 成员方法：detectEdgeFV（特征向量法）
__host__ int RobustEdgeDetection::detectEdgeFV(
        Image *inimg, Image *outimg, CoordiSet *guidingset)
{
    // 检查输入图像和输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 检查图像是否为空
    if (inimg->imgData == NULL)
        return UNMATCH_IMG;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    // 局部变量，错误码
    int errcode;
    cudaError_t cudaerrcode; 

    // guidingset 为边缘检测的指导区域，如果 guidingset 不为空暂时未实现。
    if (guidingset != NULL)
        return UNIMPLEMENT;

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->roiX2 - inimg->roiX1,
                inimg->roiY2 - inimg->roiY1);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 创建临时图像
    Image *tempimg = NULL, *tempimg1 = NULL;
    // 创建记录邻域坐标的指针和对应在 Device 端的指针。
    int *neighbor = NULL, *neighborDev = NULL;
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }
    // 将 tempimg 图像在 Device 内存中建立数据。
    errcode = ImageBasicOp::makeAtCurrentDevice(
            tempimg, inimg->roiX2 - inimg->roiX1,
            inimg->roiY2 - inimg->roiY1);

    // 如果创建图像操作失败，则释放内存报错退出。
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }

    // 创建第二幅临时图像，供调用 Thinning 函数和 FreckleFilter 函数使用
    errcode = ImageBasicOp::newImage(&tempimg1);
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }

    // 提取临时图像的 ROI 子图像。
    ImageCuda subimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &subimgCud);
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }

    // 如果调用失败，则删除临时图像。
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }

    // 调用 _initNeighbor
    // 记录不同 dp 情况下邻域中点的横纵坐标个数总和。
    int sizedp01 = 0, sizedp23 = 0;
    neighbor = new int[16 * diffsize * diffsize];
    // 判断空间是否申请成功。
    if (neighbor == NULL) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return NULL_POINTER;    
    }
    errcode = _initNeighbor(neighbor, diffsize, &sizedp01, &sizedp23);
    // 如果调用失败，则删除临时图像和临时申请的空间。
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }

    // 将邻域坐标 neighbor 传入 Device 端。
    // 为 neighborDev 在 Device 端申请空间。
    cudaerrcode = cudaMalloc((void **)&neighborDev,
                             sizeof (int) * 16 * diffsize * diffsize);
    // 若申请不成功则释放空间。
    if (cudaerrcode != cudaSuccess) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return CUDA_ERROR;
    }
    // 将 neighbor 拷贝到 Device 端的 neighborDev 中。
    cudaerrcode = cudaMemcpy(neighborDev, neighbor,
                             sizeof (int) * 16 * diffsize * diffsize,
                             cudaMemcpyHostToDevice);
    // 如果拷贝不成功，则释放空间。
    if (cudaerrcode != cudaSuccess) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return CUDA_ERROR;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = FV_DEF_BLOCK_X;
    blocksize.y = FV_DEF_BLOCK_Y;
    blocksize.z = FV_DEF_BLOCK_Z;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) /
                 blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) /
                 blocksize.y;
    gridsize.z = 1;
 
    // 计算共享内存的空间。
    int memsize = FV_DEF_BLOCK_Y * FV_DEF_BLOCK_X * FV_DEF_BLOCK_Z *
                  (3 * sizeof (float) + sizeof (int));
    // 调用核函数
    _detectEdgeFVKer<<<gridsize, blocksize, memsize>>>( 
            insubimgCud, subimgCud, outsubimgCud, diffsize, searchScope, 
            neighborDev, sizedp01 * 2, sizedp23 * 2);

    // 检查 kernel 是否调用出错，若出错则释放空间。
    if (cudaGetLastError() != cudaSuccess) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return CUDA_ERROR;
    }

    // 调用 Thinning，如果调用出错则释放空间。
    errcode = this->thinning.thinMatlabLike(outimg, tempimg1);
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }

    // 调用 FreckleFilter，如果调用出错则释放空间。
    errcode = this->frecklefilter.freckleFilter(tempimg1, outimg);
    if (errcode != NO_ERROR) {
        // 释放空间。
        FAIL_RED_FV_FREE;
        return errcode;
    }

    // 退出前删除临时图像。
    ImageBasicOp::deleteImage(tempimg);
    ImageBasicOp::deleteImage(tempimg1);
    delete [] neighbor;
    cudaFree(neighborDev);

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员方法：CPU 端的 sobel 算子边缘检测。
__host__ int RobustEdgeDetection::sobelHost(Image *src, Image *out)
{
    int x, y, s, s1, s2;
    for (x = 1; x < src->width - 1; x++)
    {
        for (y = 1; y < src->height - 1; y++)
        {
            //横向梯度[-1 0 1; -2 0 2; -1 0 1]
            s1 = src->imgData[x + 1 + (y - 1) * src->width] + 
                    2 * src->imgData[x + 1 + y * src->width] + 
                    src->imgData[x + 1 + (y + 1) * src->width];
            s1 = s1 - src->imgData[x - 1 + (y - 1) * src->width] - 
                    2 * src->imgData[x - 1 + y * src->width] - 
                    src->imgData[x - 1 + (y + 1) * src->width];

            //纵向梯度[-1 -2 -1; 0 0 0; 1 2 1]
            s2 = src->imgData[x + 1 + (y + 1) * src->width] + 
                    2 * src->imgData[x + (y + 1) * src->width] + 
                    src->imgData[x - 1 + (y + 1) * src->width];
            s2 = s2 - src->imgData[x + 1 + (y - 1) * src->width] - 
                    2 * src->imgData[x + (y - 1) * src->width] - 
                    src->imgData[x - 1 + (y - 1) * src->width];

            //给图像赋值
            s = s1 * s1 + s2 * s2;
            s = sqrt(s);
            out->imgData[y * src->width + x] = s;
        }
    }
    return  NO_ERROR;
}

// 核函数：GPU 端的 sobel 算子边缘检测。
static __global__ void      // Kernel 函数无返回值
_sobelKer(ImageCuda inimg, ImageCuda outimg)
{
    // 并行策略：一个线程处理一个像素点
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (x >= inimg.imgMeta.width || y >= inimg.imgMeta.height)
        return;

    int s, s1, s2;

    //横向梯度[-1 0 1; -2 0 2; -1 0 1]
    s1 = inimg.imgMeta.imgData[x + 1 + (y - 1) * inimg.pitchBytes] + 
            2 * inimg.imgMeta.imgData[x + 1 + y * inimg.pitchBytes] + 
                    inimg.imgMeta.imgData[x + 1 + (y + 1) * inimg.pitchBytes];
    s1 = s1 - inimg.imgMeta.imgData[x - 1 + (y - 1) * inimg.pitchBytes] - 
            2 * inimg.imgMeta.imgData[x - 1 + y * inimg.pitchBytes] - 
            inimg.imgMeta.imgData[x - 1 + (y + 1) * inimg.pitchBytes];
    //纵向梯度[-1 -2 -1; 0 0 0; 1 2 1]
    s2 = inimg.imgMeta.imgData[x + 1 + (y + 1) * inimg.pitchBytes] + 
            2 * inimg.imgMeta.imgData[x + (y + 1) * inimg.pitchBytes] + 
            inimg.imgMeta.imgData[x - 1 + (y + 1) * inimg.pitchBytes];
    s2 = s2 - inimg.imgMeta.imgData[x + 1 + (y - 1) * inimg.pitchBytes] - 
            2 * inimg.imgMeta.imgData[x + (y - 1) * inimg.pitchBytes] - 
            inimg.imgMeta.imgData[x - 1 + (y - 1) * inimg.pitchBytes];

    //给图像赋值
    s = s1 * s1 + s2 * s2;
    s = sqrtf(s);
    outimg.imgMeta.imgData[y * inimg.pitchBytes + x] = s;
}
// Host 成员方法：GPU 端的 sobel 算子边缘检测。
__host__ int RobustEdgeDetection::sobel(Image *inimg, Image *outimg)
{
    // 检查输入图像和输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 检查图像是否为空
    if (inimg->imgData == NULL)
        return UNMATCH_IMG;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    // 局部变量，错误码
    int errcode;

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(
                outimg, inimg->roiX2 - inimg->roiX1,
                inimg->roiY2 - inimg->roiY1);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    }
    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = SA_DEF_BLOCK_X;
    blocksize.y = SA_DEF_BLOCK_Y;
    blocksize.z = SA_DEF_BLOCK_Z;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) /
                 blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) /
                 blocksize.y;
    gridsize.z = 1;

    // 调用核函数
    _sobelKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud);

    return NO_ERROR;
}
