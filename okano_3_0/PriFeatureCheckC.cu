// PriFeatureCheckC.cu
// 任意形状曲线的基础特征检查

#include "PriFeatureCheckC.h"

// 宏：DEF_BLOCK_X
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32


// Kernel 函数：_ComputeSquareKer（计算闭合曲线面积）
// 计算闭合曲线的面积，方便后续计算。
static __global__ void  // Kernel 函数无返回值
_computeSquareKer(
        int *coordset,  // 输入曲线的坐标。
        int *area,      // 闭合曲线的面积。
        int curlen      // 曲线点的个数
);

// Host 函数：comCurSquare（计算闭合曲线的面积）
static __host__ int     // 函数若正确执行返回 NO_ERROR
comCurSquare(
        int *coordset,  // 输入曲线的坐标集
        int *area,      // 闭合曲线的面积
        int curlen      // 闭合曲线点数目
);
/*
// Kernel 函数：_ComputeCornerCntKer（计算角点个数）
// 计算曲线角点的个数，原理是道格拉斯算法。
static __global__ void  // Kernel 函数无返回值
_computeCornerCntKer(
        CurveCuda curve,  // 输入曲线
        int *count        // 角点的个数
);

// Host 函数：ComputeCornerCnt（计算角点个数）
// 计算曲线角点的个数，原理是道格拉斯算法。
static __host__ int  // 函数若正确执行返回 NO_ERROR
computeCornerCnt(
        Curve *curve,  // 输入曲线
        int *count     // 角点的个数
);
*/
// Kernel 函数：_computeSquareKer（计算闭合曲线面积）
static __global__ void _computeSquareKer(
        int *coordset, int *area, int curlen)
{
    // 判断参数是否合法。
    if (coordset == NULL || area == NULL)
        return;

    // 线程索引。
    int idx = threadIdx.x;

    // 判断是否越界。
    if (idx < 0 || idx >= curlen)
        return;

    // 当前处理的点及其相邻的下一个点的横坐标的索引。
    int curxidx = 2 * idx;
    int nextxidx = curxidx + 2;

    // 记录当前处理点和相邻点的坐标。
    int curx = coordset[curxidx];
    int nextx = coordset[nextxidx];
    int cury = coordset[curxidx + 1];
    int nexty = coordset[nextxidx + 1];
    // 一个中间变量，它是对曲线上相邻两个点的坐标进行运算。
    int coordtemp = abs(curx * nexty - nextx * cury);

    // 使用原子加求面积和中心坐标的中间值。
    atomicAdd(area, coordtemp);

    // 块内同步。
    __syncthreads();

    // 计算闭合曲线最终面积和中心坐标。
    if (threadIdx.x != 0)
        return;
    *area = *area >> 1;
}

// Host 函数：comCurSquare（计算闭合曲线的面积）
static __host__ int comCurSquare(
        int *coordset, int *area, int curlen)
{
    // 判断参数合法性。
    if (coordset == NULL || area == NULL)
        return INVALID_DATA;
    // 在 Device 端申请 area 的空间
    int *areaCud;
    int cudaerrcode = cudaMalloc((void**)&areaCud, sizeof (int));
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;
    cudaerrcode = cudaMemcpy(areaCud, area, sizeof (int),
                             cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 计算线程块的大小。
    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    gridsize.x = (curlen + blocksize.x - 1) / blocksize.x;
    _computeSquareKer<<<gridsize, blocksize>>>(coordset, areaCud, curlen);
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 将 area 拷贝至 Host 端
    cudaerrcode = cudaMemcpy(area, areaCud, sizeof (int),
                             cudaMemcpyDeviceToHost);
    if (cudaerrcode != cudaSuccess)
        return cudaerrcode;

    // 释放 Device 端空间。
    cudaFree(areaCud);

    // 处理完毕，退出。
    return NO_ERROR;
}
/*
// Kernel 函数：_ComputeCornerCntKer（计算角点个数）
static __global__ void _computeCornerCntKer(CurveCuda curve, int *count)
{
    // UN_IMPLEMENT
    return;
}

// Host 函数：ComputeCornerCnt（计算角点个数）
static __host__ int computeCornerCnt(Curve *curve, int *count)
{
    // UN_IMPLEMENT
    return NO_ERROR;
}
*/
// Host 函数：priFeatureCheckC（任意曲线的基本特征检查）
__host__ int PriFeatureCheckC::priFeatureCheckC(
        Curve *curve, float *result, bool *errorJudge)
{
    // 判断参数是否合理
    if (curve == NULL || curve->crvData == NULL ||
        result == NULL || errorJudge == NULL)
        return INVALID_DATA;
    // 取出曲线的点数
    int pntcnt = curve->curveLength;

    // 标记该曲线是否是闭合曲线，如果不是闭合曲线则其面积默认为 0。
    // 若闭合标记为 1。
    int flag = 0;
    if (curve->closed == true)
        flag = 1;

    // 申请一张临时图片，此图片由 curve 转换来，利用该图像调用 Moments 和 MDR
    // 算法。
    Image *inimg;
    int errcode = ImageBasicOp::newImage(&inimg);
    if (errcode != cudaSuccess)
        return errcode;

    errcode = ImageBasicOp::makeAtHost(inimg, curve->maxCordiX + 1,
                                       curve->maxCordiY + 1);
    if (errcode != cudaSuccess) {
        ImageBasicOp::deleteImage(inimg);
        return errcode;
    }

    errcode = imgconvert.curConvertToImg(curve, inimg);
    if (errcode != cudaSuccess) {
        ImageBasicOp::deleteImage(inimg);
        return errcode;
    }

    // 将结果图像拷贝到 Host 端
    errcode = ImageBasicOp::copyToHost(inimg);
    if (errcode != cudaSuccess) {
        ImageBasicOp::deleteImage(inimg);
        return errcode;
    }

    // （1）计算曲线点数和 MDR 周长的比值
    DirectedRect *rect = new DirectedRect[1];
    errcode = sdr.smallestDirRect(inimg, rect);  // 调用最小有向外接矩形
    if (errcode != cudaSuccess) {
        ImageBasicOp::deleteImage(inimg);
        delete rect;
        return errcode;
    }
    
    // 计算最小外接矩形的长和宽。
    int length = rect->length1;
    int width = rect->length2;

    // 最小外接矩形的周长和面积
    float mdrperimeter = 2 * (length + width);
    float square = length * width;

    // 计算 LengthRatio
    result[0] = pntcnt / mdrperimeter;

    // （1）判断 LengthRatio 是否在指标范围内。
    if (result[0] >= minLengthRatio && result[0] <= maxLengthRatio)
        errorJudge[0] = false;
    else
        errorJudge[0] = true;

    // （2）计算 MDRSideRatio 并判断是否在指标范围内。
    result[1] = width * 1.0f / length;
    if (result[1] >= minMDRSideRatio && result[1] <= maxMDRSideRatio)
        errorJudge[1] = false;
    else
        errorJudge[1] = true;


    // （3）检查 AMIS
    MomentSet *momentset = new MomentSet[1];
    // 调用算法，求解 AMIS
    errcode = this->ami.affineMoments(inimg, momentset);
    if (errcode != cudaSuccess) {
        ImageBasicOp::deleteImage(inimg);
        delete rect;
        delete momentset;
        return errcode;
    }
    // 存储 9 个 AMIS 值
    double amis[9] = {momentset->ami1, momentset->ami2, momentset->ami3,
                      momentset->ami4, momentset->ami6, momentset->ami7,
                      momentset->ami8, momentset->ami9, momentset->ami19};
    // 若 amis 中某一项不满足条件则在 errorjudge 中赋为 false
    for (int i = 0; i < 9; i++) {
        if (abs(amis[i] - this->avgAMIs[i]) > maxAMIsError[i])
            errorJudge[2] = true;
        else
            errorJudge[2] = false;
    }

    // （4）闭合曲线的面积与 MDR 的面积比
    // 首先判断是否闭合曲线，若不是闭合曲线则跳过此步骤
    if (flag == 0) {
        int area = 0;
        errcode = comCurSquare(curve->crvData, &area, pntcnt);
        if (errcode != cudaSuccess) {
            ImageBasicOp::deleteImage(inimg);
            delete rect;
            delete momentset;
            return errcode;
        }

        // 计算闭合曲线的面积与 MDR 的面积比
        result[3] = area * 1.0f / square;
        if (result[3] >= minContourAreaRatio || result[3] <= maxContourAreaRatio)
            errorJudge[3] = false;
        else
            errorJudge[3] = true;
    }

    // （5）曲线个数与角点比值还未实现
    return NO_ERROR;

}


