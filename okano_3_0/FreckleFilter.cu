// FreckleFilter
// 实现广义的中值滤波

#include "FreckleFilter.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;


// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8


// Device 函数：_getMaxMatchValueDev（得到两个直方图的 length 长度
// 相似度匹配最大值）
// 对圆周和园内两直方图进行长度为 length 的相似度匹配，返回最大值
static __device__ void             // 返回值：无返回值
_getMaxMatchValueDev(
        unsigned int *histogram1,  // 圆周上的直方图
        unsigned int *histogram2,  // 圆周上的直方图
        float &maxmatchvalue,      // 像素点对应的最大匹配值指针
        int length,                // 相似度匹配的长度参数
        int hisnum = 256           // 直方图的数组大小，本方法大小为 256
); 

// Kernel 函数：_freckleFilterByVarSumCountKer（获得输出图像的每点像素平均值总
// 和与累加次数算法操作）
// 根据方差阈值大小，得到输出图像的每点像素平均值总和与累加次数算法操作
static __global__ void     // Kernel 函数无返回值
_freckleFilterByVarSumCountKer(
        ImageCuda inimg,   // 输入图像
        Template radtpl,   // 圆形模板，用于指定圆内领域
        Template archtpl,  // 环形模板，用于指定圆周的邻域
        float varTh,       // 外部指定的方差阈值
        float *sum,        // 像素平均值累加总和
        int *count         // 像素平均值累加次数
);

// Kernel 函数：_freckleFilterPixelKer（实现给输出图像设定像素值算法操作）
// 根据每点像素累加总和与累加次数，给输出图像设定像素平均值
static __global__ void     // Kernel 函数无返回值
_freckleFilterSetPixelKer(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 输出图像
        float *sum,        // 像素平均值累加总和
        int *count,        // 像素平均值累加次数
        int select         // 最后赋值时的选择参数
);

// Kernel 函数：_freckleFilterByStrMscKer（获得输出图像的每点像素平均值总
// 和与累加次数算法操作）
// 通过相似度匹配，根据匹配差阈值，得到输出图像的每点像素平均值总和与
// 累加次数算法操作
static __global__ void     // Kernel 函数无返回值
_freckleFilterByStrMscKer(
        ImageCuda inimg,   // 输入图像
        Template radtpl,   // 圆形模板，用于指定圆内领域
        Template archtpl,  // 环形模板，用于指定圆周的邻域
        float matchErrTh,  // 外部指定的匹配差阈值
        int length,        // 相似度匹配的长度参数
        int radius,        // 圆领域的半径
        float *sum,        // 像素平均值累加总和
        int *count         // 像素平均值累加次数
);


// Kernel 函数：_freckleFilterByVarSumCountKer（实现给输出图像设定像素值算法
// 操作）
static __global__ void _freckleFilterByVarSumCountKer(
        ImageCuda inimg, Template radtpl, Template archtpl, float varTh,
        float *sum, int *count)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量
    // dstc 表示 column， dstr 表示 row）。由于采用并行度缩减策略 ，令一个线程
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
    int *curtplptr;

    // 用来记录当前输入图像所在位置的指针
    unsigned char *curinptr;

    // 计数器，用来记录某点在模版范围内拥有的点的个数
    int statistic[4] = { 0 , 0, 0, 0 };

    // 迭代求平均值和方差使用的中间值
    float m[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // 计算得到的平均值
    float mean[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // 计算得到的拱圆模板领域方差
    float variance[4] = { 0.0f, 0.0f, 0.0f, 0.0f };   

    int pix;  // 局部变量，临时存储像素值
  
    // 指定当前环形模版所在位置
    curtplptr = archtpl.tplData;

    // 扫描环形模版范围内的每个输入图像的像素点
    for (int i = 0; i < archtpl.count; i++) {
        // 计算当模版位置所在像素的 x 和 y 分量，模版使用相邻的两个下标的
        // 数组表示一个点，所以使用当前模版位置的指针加一操作
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);
        
        float temp;  // 局部变量，在进行迭代时的中间变量

        // 先判断当前像素的 x 分量是否越界，如果越界，则跳过，扫描下一个模版点,
        // 如果没有越界，则分别处理当前列的相邻的 4 个像素
        if (dx >= 0 && dx < inimg.imgMeta.width) {
            // 根据 dx 和 dy 获取第一个像素的指针
            curinptr = inimg.imgMeta.imgData + dx + dy * inimg.pitchBytes;
            // 检测此像素点的 y 分量是否越界
            if (dy >= 0 && dy < inimg.imgMeta.height) {
                // 对第一个点进行迭代
                pix = *(curinptr);
                statistic[0]++;
                temp = pix - mean[0];
                mean[0] += temp / statistic[0];
                m[0] += temp * (pix - mean[0]);
            }

            // 获取第二个像素点的指针
            curinptr = curinptr + inimg.pitchBytes;
            dy++;
            // 检测第二个像素点的 y 分量是否越界
            if (dy >= 0 && dy < inimg.imgMeta.height) {
                // 对第二个点进行迭代
                pix = *(curinptr);
                statistic[1]++;
                temp = pix - mean[1];
                mean[1] += temp / statistic[1];
                m[1] += temp * (pix - mean[1]);
            }
            
            // 获取第三个像素点的指针
            curinptr = curinptr + inimg.pitchBytes;
            dy++;
            // 检测第三个像素点的 y 分量是否越界
            if (dy >= 0 && dy < inimg.imgMeta.height) {
                // 对第三个点进行迭代
                pix = *(curinptr);
                statistic[2]++;
                temp = pix - mean[2];
                mean[2] += temp / statistic[2];
                m[2] += temp * (pix - mean[2]);
            }
            
            // 获取第四个像素点的指针
            curinptr = curinptr + inimg.pitchBytes;
            dy++;
            // 检测第四个像素点的 y 分量是否越界
            if (dy >= 0 && dy < inimg.imgMeta.height) {
                // 对第四个点进行迭代
                pix = *(curinptr);
                statistic[3]++;
                temp = pix - mean[3];
                mean[3] += temp / statistic[3];
                m[3] += temp * (pix - mean[3]);
            }
        }
    }
    
    // 计算输出坐标点对应的图像数据数组下标。
    int index;
     
    // 对每个像素点求圆周上点的方差大小，根据方差与阈值大小给输出点累加和
    for(int i = 0; i < 4; i++) {
        // 如果圆周领域内的的点个数为 0，则判断下一个像素点 
        if(statistic[i] == 0)
            continue;
        // 计算环形模板领域的方差
        variance[i] = m[i] / statistic[i];

        // 如果方差小于给定阈值，则对圆形模板里的所有点赋平均值
        if (variance[i] < varTh) {
            // 指定当前圆形模版所在位置
            curtplptr = radtpl.tplData;

            // 扫描圆形模版范围内的每个输入图像的像素点
            for (int j = 0; j < radtpl.count; j++) {
                // 计算当模版位置所在像素的 x 和 y 分量，模版使用相邻的两个
                // 下标的数组表示一个点，所以使用当前模版位置的指针加一操作
                dx = dstc + *(curtplptr++);
                dy = dstr + *(curtplptr++);

                // 根据 dx 和 dy 获取像素下标
                dy = dy + i;
                index = dx + dy * inimg.imgMeta.width;

                // 如果没有越界，则分别处理当前列的相邻的符合条件的像素
                // 给累加和累加平均值，累加次数相应加 1
                if (dx >= 0 && dx < inimg.imgMeta.width &&
                    dy >= 0 && dy < inimg.imgMeta.height) {
                    atomicAdd(&sum[index], mean[i]);
                    atomicAdd(&count[index], 1);
                }
            }
        }
    }
}

// Kernel 函数：_freckleFilterSetPixelKer（实现给输出图像设定像素值算法操作）
static __global__ void _freckleFilterSetPixelKer(
        ImageCuda inimg, ImageCuda outimg, float *sum, int *count, int select)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，
    // c 表示 column， r 表示 row）。由于采用并行度缩减策略 ，令一个线程
    // 处理 4 个输出像素，这四个像素位于统一列的相邻 4 行上，因此，对于
    // dstr 需要进行乘 4 的计算
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
  
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃
    if (dstc >= outimg.imgMeta.width || dstr >= outimg.imgMeta.height)
        return;

    // 计算第一个输入坐标点对应的图像数据数组下标。
    int outidx = dstr * outimg.imgMeta.width + dstc;
    int out = dstr * outimg.pitchBytes + dstc;
 
    int temp;  // 临时变量用于 float 型数据转 int 型，需要四舍五入

    // 计算每一个点的像素平均值，并且四舍五入 float 转 int 型
    if (count[outidx] == 0) {
        // 如果该点没有被累加和，如果为 FRECKLE_OPEN 则应该赋值为
        // 原图像对应灰度值，如果为 FRECKLE_CLOSE，则赋值为 0
        if (select == FRECKLE_OPEN)
            temp = inimg.imgMeta.imgData[out];
        else if (select == FRECKLE_CLOSE)
            temp = 0;
    } else {
        // 如果被累加和，则按以下方式求像素平均值并按要求处理
        temp = (int)(sum[outidx] / count[outidx] + 0.5f);
    }

    // 对图像每点像素值赋上对应值
    outimg.imgMeta.imgData[out] = (unsigned char)temp;
    
    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++dstr >= outimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        outidx += outimg.imgMeta.width;
        out += outimg.pitchBytes;

        // 计算每一个点的像素平均值，并且四舍五入 float 转 int 型
        if (count[outidx] == 0) {
            // 如果该点没有被累加和，如果为 FRECKLE_OPEN 则应该赋值为
            // 原图像对应灰度值，如果为 FRECKLE_CLOSE，则赋值为 0
            if (select == FRECKLE_OPEN)
                temp = inimg.imgMeta.imgData[out];
            else if (select == FRECKLE_CLOSE)
                temp = 0;
        } else {
            // 如果被累加和，则按以下方式求像素平均值并按要求处理
            temp = (int)(sum[outidx] / count[outidx] + 0.5f);
        }

        // 对图像每点像素值赋上对应值
        outimg.imgMeta.imgData[out] = (unsigned char)temp;
    }
}

// Device 函数：_getMaxMatchValueDev（得到两个直方图的 length 长度
// 相似度匹配最大值）
static __device__ void _getMaxMatchValueDev(
        unsigned int *histogram1, unsigned int *histogram2,
        float &maxmatchvalue, int length, int hisnum)
{
    // 临时变量 matchvalue，存储匹配的结果值
    float matchvalue = 0.0f;

    // 从左端开始匹配
    // 临时变量 location，用于定位匹配最右位置
    int location = hisnum - length; 
    for (int j = 0; j <= location; j++) {

        // 临时变量，存储计算相关系数的和
        unsigned int sum1 = { 0 }; 
        unsigned int sum2 = { 0 }; 
        unsigned int sum3 = { 0 };
        unsigned int sum4 = { 0 };
        unsigned int sum5 = { 0 };
        // 临时变量，存储获得数组对应值
        unsigned int tmp1, tmp2;
        // 临时变量，存储计算相关系数算法的分母
        float m1, m2;

        // 计算相似度需要用到的临时变量
        for (int k = 0; k < length; k++) {
            // 取得对应直方图值
            tmp1 = *(histogram1 + j + k);
            tmp2 = *(histogram2 + j + k);

            // 计算相似度要用到的累加和
            sum1 += tmp1;
            sum2 += tmp2;
            sum3 += tmp1 * tmp2;
            sum4 += tmp1 * tmp1;
            sum5 += tmp2 * tmp2;
        }
        // 计算相似度的分母临时变量
        m1 = sqrtf((float)(length * sum4 - sum1 * sum1));
        m2 = sqrtf((float)(length * sum5 - sum2 * sum2));
        // 计算匹配的相似度
        if (m1 <= 0.000001f || m2 <= 0.000001f)
            matchvalue = 0.0f;
        else
            matchvalue = ((int)(length * sum3 - sum1 * sum2)) /
                         (m1 * m2);

        // 取相似度最大值
        if (matchvalue > maxmatchvalue) {
            maxmatchvalue = matchvalue;
        }
    }
}

// Kernel 函数：_freckleFilterByStrMscKer（实现
// 给输出图像设定像素值算法操作）
static __global__ void _freckleFilterByStrMscKer(
        ImageCuda inimg, Template radtpl, Template archtpl, float matchErrTh,
        int length, int radius, float *sum, int *count)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量
    // dstc 表示 column， dstr 表示 row）。由于采用并行度缩减策略 ，令一个线程
    // 处理 4 个输出像素，这四个像素位于统一列的相邻 4 行上，因此，对于
    // dstr 需要进行乘 4 的计算
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否符合作为圆心的条件，若不符，则不进行处理
    if (dstc % radius != 0 || dstr % radius != 0 || dstc <= 0 || dstr <= 0 ||
        dstc >= inimg.imgMeta.width - 1 || dstr >= inimg.imgMeta.height - 1)
        return;
   
    // 用来保存临时像素点的坐标的 x 和 y 分量
    int dx, dy;

    // 用来记录当前模版所在位置的指针
    int *curtplptr;

    // 用来记录当前输入图像所在位置的指针
    unsigned char *curinptr;

    // 圆周上的图像直方图 histogram1
    unsigned int histogram1[256] = { 0 };

    // 圆内的图像直方图 histogram2
    unsigned int histogram2[256] = { 0 };

    // 计数器，用来记录某点在圆周上和园内拥有的点的个数
    int statistic = 0;

    unsigned int pix;  // 局部变量，临时存储像素值
  
    // 指定当前环形模版所在位置
    curtplptr = archtpl.tplData;

    // 扫描环形模版范围内的每个输入图像的像素点
    for (int i = 0; i < archtpl.count; i++) {
        // 计算当模版位置所在像素的 x 和 y 分量，模版使用相邻的两个下标的
        // 数组表示一个点，所以使用当前模版位置的指针加一操作
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);

        // 先判断当前像素的 x 分量，y 分量是否越界，如果越界，则跳过，扫描
        // 下一个模版点，如果没有越界，则分别处理当前列的相邻的 4 个像素
        if (dx >= 0 && dx < inimg.imgMeta.width &&
            dy >= 0 && dy < inimg.imgMeta.height) {
            // 根据 dx 和 dy 获取像素的指针
            curinptr = inimg.imgMeta.imgData + dx + dy * inimg.pitchBytes;
            pix = *(curinptr);
            histogram1[pix]++;
            statistic++;
        }
    }
    
    // 如果圆周领域内的的点个数为 0 这直接返回 
    if(statistic == 0)
        return;
               
    // 指定当前圆形模版所在位置
    curtplptr = radtpl.tplData;

    // 扫描环形模版范围内的每个输入图像的像素点
    for (int i = 0; i < radtpl.count; i++) {
        // 计算当模版位置所在像素的 x 和 y 分量，模版使用相邻的两个下标的
        // 数组表示一个点，所以使用当前模版位置的指针加一操作
        dx = dstc + *(curtplptr++);
        dy = dstr + *(curtplptr++);

        // 先判断当前像素的 x 分量，y 分量是否越界，如果越界，则跳过，扫描
        // 下一个模版点，如果没有越界，则分别处理当前列的相邻的 4 个像素
        if (dx >= 0 && dx < inimg.imgMeta.width) {
            // 根据 dx 和 dy 获取第一个像素的指针
            curinptr = inimg.imgMeta.imgData + dx + dy * inimg.pitchBytes;
            pix = *(curinptr);
            histogram2[pix]++;
        }
    }

    // 存储以四个像素圆心得到两直方图的匹配最大值
    float maxmatchvalue = 0.0f;

    // 得到四个像素的两直方图的匹配最大值
    _getMaxMatchValueDev(histogram1, histogram2, maxmatchvalue, length, 256);

    // 计算输出坐标点对应的图像数据数组下标。
    int index;
 
    // 根据匹配差与阈值大小对符合条件像素点对其圆周上点进行排序，
    // 取中间 50% 灰度平均，给输出点累加和累加赋值

    // 如果匹配差大于给定阈值，则对圆形模板里的所有点赋平均值
    if (1 - maxmatchvalue >  matchErrTh) {
        // 存储圆周上的图像值的中值平均（取排序后中间 50% 平均）
        float mean;

        // 去掉排序结果中前端的数量
        int lownum = (int)(statistic * 0.25f + 0.5f);  
        // 去掉排序结果中末端端的数量
        int highnum = (int)(statistic * 0.25f + 0.5f);

        // 对直方图前后端个数统计
        int lowcount = 0, highcount = 0;
        // 在前后端统计时，中间段少加的值
        int lowvalue = 0, highvalue = 0;
        // 前后端统计时的开关
        bool lowmask = false, highmask = false;
        // 直方图中间段的两端索引
        int lowindex = 0, highindex = 0;

        for (int k = 0; k < 256; k++) {
            // 计算直方图前端的个数
            lowcount += histogram1[k];
            if (!lowmask && lowcount >= lownum) {
                lowindex = k + 1;
                lowvalue = (lowcount - lownum) * k;
                lowmask = true;
            }
            // 直方图后端的循环索引
            int high = 255 - k;
            // 计算直方图后端的个数
            highcount += histogram1[high];
            if (!highmask && highcount >= highnum) {
                highindex = high - 1;
                highvalue = (highcount - highnum) * high;
                highmask = true;
            }
            // 如果前后端开关都打开，表示都找到了对应位置，就退出循环
            if (lowmask && highmask)
                break;
        }

        // 如果 lowindex 大于 highindex，表示没有要处理的元素，则返回
        if (lowindex > highindex)
            return;

        // 计算领域内的像素值总和
        float tmpsum = (float)(lowvalue + highvalue);
        for (int k = lowindex; k <= highindex; k++) 
            tmpsum += k * histogram1[k];

        // 计算平均值
        mean = tmpsum / (statistic - lownum - highnum);

        // 指定当前圆形模版所在位置
        curtplptr = radtpl.tplData;

        // 扫描圆形模版范围内的每个输入图像的像素点
        for (int j = 0; j < radtpl.count; j++) {
            // 计算当模版位置所在像素的 x 和 y 分量，模版使用相邻的两个
            // 下标的数组表示一个点，所以使用当前模版位置的指针加一操作
            dx = dstc + *(curtplptr++);
            dy = dstr + *(curtplptr++);

            // 根据 dx 和 dy 获取像素下标
            dy++;
            index = dx + dy * inimg.imgMeta.width;

            // 如果没有越界，则分别处理当前列的相邻的符合条件的像素
            // 给累加和累加平均值，累加次数相应加 1
            if (dx >= 0 && dx < inimg.imgMeta.width &&
                dy >= 0 && dy < inimg.imgMeta.height) {
                atomicAdd(&sum[index], mean);
                atomicAdd(&count[index], 1);
            }
        }
    }
}

// Host 成员方法：freckleFilter（广义的中值滤波）
__host__ int FreckleFilter::freckleFilter(Image *inimg, Image *outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 开关错误检查，如果既不是开选择也不是闭选择，则返回错误
    if (select != FRECKLE_OPEN && select != FRECKLE_CLOSE)
        return INVALID_DATA;

    int errcode;  // 局部变量，错误码
    
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
    if (errcode != NO_ERROR)
        return errcode;
    
    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 根据子图像的大小对长，宽进行调整，选择长度小的长，宽进行子图像的统一
    if (insubimgCud.imgMeta.width > outsubimgCud.imgMeta.width)
        insubimgCud.imgMeta.width = outsubimgCud.imgMeta.width;
    else
        outsubimgCud.imgMeta.width = insubimgCud.imgMeta.width;
	
    if (insubimgCud.imgMeta.height > outsubimgCud.imgMeta.height)
        insubimgCud.imgMeta.height = outsubimgCud.imgMeta.height;
    else
        outsubimgCud.imgMeta.height = insubimgCud.imgMeta.height;

    // 定义模板 radtpl 用于获取圆形领域模板
    Template *radtpl;

    // 定义圆形模板的尺寸
    dim3 radsize(this->radius * 2 + 1, this->radius * 2 + 1, 1);

    // 通过模板工厂得到圆形领域模板
    errcode = TemplateFactory::getTemplate(&radtpl, TF_SHAPE_CIRCLE, 
                                           radsize, NULL);

    // 检查圆形模板是否为 NULL，如果为 NULL 直接报错返回。
    if (errcode != NO_ERROR)
        return errcode;    

    // 将模板拷贝到 Device 内存中
    errcode = TemplateBasicOp::copyToCurrentDevice(radtpl);
    if (errcode != NO_ERROR) {
        // 放回 radtpl 模板
        TemplateFactory::putTemplate(radtpl);
        return errcode;
    }

    // 定义模板 archtpl 用于获取环形领域模板
    Template *archtpl;    

    // 定义环形模板的尺寸
    dim3 arcsize(this->radius * 2 + 1, (this->radius + 4) * 2 + 1, 1);

    // 得到环形领域模板
    errcode = TemplateFactory::getTemplate(&archtpl, TF_SHAPE_ARC,
                                           arcsize, NULL);

    // 检查环形模板是否为 NULL，如果为 NULL 报错返回。
    if (errcode != NO_ERROR) {
        // 放回 radtpl 模板
        TemplateFactory::putTemplate(radtpl);
        return errcode;
    } 

    // 将模板拷贝到 Device 内存中
    errcode = TemplateBasicOp::copyToCurrentDevice(archtpl);
    if (errcode != NO_ERROR) {
        // 放回模板
        TemplateFactory::putTemplate(radtpl);
        TemplateFactory::putTemplate(archtpl);
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize1, gridsize2;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize1.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize1.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
                  (blocksize.y * 4);
    gridsize2.x = gridsize1.x;
    gridsize2.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 得到要处理的像素总个数
    size_t datasize = outsubimgCud.imgMeta.width * outsubimgCud.imgMeta.height;
 
    cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

    // 定义 sum 全局变量指针，申请一个 outsubimgCud.imgMeta.width *
    // outsubimgCud.imgMeta.height 的 float 型数组，用于存储每点像素平均值累加
    // 总和。
    float *sum;

    // 定义 count 全局变量指针，申请一个 outsubimgCud.imgMeta.width * 
    // outsubimgCud.imgMeta.height 的 int 型数组，用于存储每点像素平均值累加
    // 次数。
    int *count;

    // 定义局部变量，用于多份数据的一份申请
    void *temp_dev;

    // 在设备端申请内存，然后分配给各个变量
    cuerrcode = cudaMalloc(
            (void **)&temp_dev,
            datasize * sizeof (float) + datasize * sizeof (int));
    if (cuerrcode != cudaSuccess) {
        // 放回模板
        TemplateFactory::putTemplate(radtpl);
        TemplateFactory::putTemplate(archtpl);
        return CUDA_ERROR;
    }

    // 为变量分配内存
    sum = (float *)temp_dev;
    count = (int *)(sum + datasize);

    // 初始化累加和的所有值为 0
    cuerrcode = cudaMemset(sum, 0, datasize * sizeof (float)); 
    if (cuerrcode != cudaSuccess) {
        // 放回模板
        TemplateFactory::putTemplate(radtpl);
        TemplateFactory::putTemplate(archtpl);
        // 释放累加和与累加次数的总空间
        cudaFree(temp_dev);
        return CUDA_ERROR;
    }

    // 初始化累加次数的所有值为 0
    cuerrcode = cudaMemset(count, 0, datasize * sizeof (int));  
    if (cuerrcode != cudaSuccess) {
        // 放回模板
        TemplateFactory::putTemplate(radtpl);
        TemplateFactory::putTemplate(archtpl);
        // 释放累加和与累加次数的总空间
        cudaFree(temp_dev);
        return CUDA_ERROR;
    }
    
    if (method == FRECKLE_VAR_TH) {
        // 若方法为方差阈值法，则调用相应方差阈值法的 Kernel 获得
        // 输出图像的每点像素平均值累加总和与累加次数。
        _freckleFilterByVarSumCountKer<<<gridsize1, blocksize>>>(
                insubimgCud, *radtpl, *archtpl, this->varTh, sum, count);
    } else if (method == FRECKLE_MATCH_ERRTH) {

        // 若方法为相似度匹配法，则调用相应相似度匹配法的 Kernel 获得
        // 输出图像的每点像素平均值累加总和与累加次数。
        _freckleFilterByStrMscKer<<<gridsize2, blocksize>>>(
                insubimgCud, *radtpl, *archtpl, this->matchErrTh, this->length,
                this->radius, sum, count);
    } else {
        // method 错误检查，进入这条分支表示没有外部方法设置有误
        return INVALID_DATA;  
    }
 
    // 检查核函数运行是否出错
    if (cudaGetLastError() != cudaSuccess) {
        // 放回模板
        TemplateFactory::putTemplate(radtpl);
        TemplateFactory::putTemplate(archtpl);
        // 释放累加和与累加次数的总空间
        cudaFree(temp_dev);
        return CUDA_ERROR;
    }

    // 放回模板
    TemplateFactory::putTemplate(radtpl);
    TemplateFactory::putTemplate(archtpl);

    // 调用 Kernel 函数实现给输出图像设定像素值。
    _freckleFilterSetPixelKer<<<gridsize1, blocksize>>>(
            insubimgCud, outsubimgCud, sum, count, this->select);

    // 检查核函数运行是否出错
    if (cudaGetLastError() != cudaSuccess) {
        // 释放累加和与累加次数的总空间
        cudaFree(temp_dev);
        return CUDA_ERROR;
    }

    // 释放累加和与累加次数的总空间
    cudaFree(temp_dev);

    // 处理完毕，退出。	
    return NO_ERROR;
}

