// FeatureVecCalc.cu
// 实现计算起始特征向量

#include "FeatureVecCalc.h" 
 
 
// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  256 
#define DEF_BLOCK_Y    1 


// 结构体 FeatureVectorProcessorParam（特征向量处理参数）
// 该结构体中定义了初始特征向量进一步处理的参数，有
// cv、sd、nc 的上限，下限，均值。
typedef struct ProcessorParam_st
{  
    float mincv, maxcv, avgcv, rangecv;  // cv 的上限，下限，均值，极限值的差。
    float minsd, maxsd, avgsd, rangesd;  // sd 的上限，下限，均值，极限值的差。
    float minnc, maxnc, avgnc, rangenc;  // nc 的上限，下限，均值，极限值的差。
} FeatureVectorProcessorParam;


// 核函数 _calcFeatureVectorKer（生成初始特征向量）
// 该方法用于计算指定坐标集所规定的图像范围内的各 PIXEL 的初始特征向量。
// 利用需求文档中给出的公式，计算每个像素的三个特征值：灰度中心値 float CV 、
// 灰度値标准差 float SD 、最大灰度非共起系数 float NC。
static __global__ void                       // Kernel 函数无返回值
_calcFeatureVectorKer(
        ImageCuda inimgcud,                  // 输入图像
        CoordiSet incoordiset,               // 输入坐标集
        FeatureVecArray outfeaturevecarray,  // 输出特征向量
        FeatureVecCalc featureveccalc,       // 平滑向量处理类
        unsigned char *neighbortmp           // 暂存空间，暂存每个像素邻域像素
                                             // 每个像素的邻域存两份，一个为排
                                             // 序前，一个为排序后
);

// 核函数 _processFeatureVectorKer（处理初始特征向量）
// 该方法用于进一步处理在 _calcFeatureVectorKer 中计算出来的初始特征向量，
// 利用的参数是在初始特征向量的上下限、均值等
static __global__ void                         // Kernel 函数无返回值
_processFeatureVectorKer(
        FeatureVecArray inoutfeaturevecarray,  // 输入输出特征向量
        FeatureVecCalc featureveccalc,         // 平滑向量处理类
        FeatureVectorProcessorParam param      // 处理参数
);

// 核函数：_calcFeatureVectorKer（计算初始特征向量）
static __global__ void _calcFeatureVectorKer(
        ImageCuda inimgcud, CoordiSet incoordiset, FeatureVecArray 
        outfeaturevecarray, FeatureVecCalc featureveccalc,  
        unsigned char *neighbortmppointer)
{
    // 计算当前 Thread 所对应的坐标集中的点的位置
    int index  = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果当前索引超过了坐标集中的点的个数，直接返回
    if(index >= incoordiset.count)
        return;

    // 计算该点在原图像中的位置
    int xcrd = incoordiset.tplData[2 * index];
    int ycrd = incoordiset.tplData[2 * index + 1];

    // 将输出特征向量的 X，Y 坐标写入到特征向量组中
    outfeaturevecarray.x[index] = xcrd;
    outfeaturevecarray.y[index] = ycrd;
    
    // 计算邻域正方形的边长及线性数组的长度。
    int n = featureveccalc.getNeighborWidth();
    int neighborwidth = n * 2 + 1;
    int length = neighborwidth * neighborwidth;

    // 计算当前像素使用的邻域暂存空间的偏移
    int offset = index * length * 2;
    unsigned char *neighbortmp = neighbortmppointer + offset;
    
    // 复制邻域像素到邻域暂存中
    for (int i = 0; i < neighborwidth; i++) {
        for (int j = 0; j < neighborwidth; j++) {
            // 计算对应图像中的坐标
            int xcrdi = xcrd - n + i;
            int ycrdj = ycrd - n + j;

            // 如果当前坐标超过图像边缘，设定为边缘值 
            if (xcrdi >= inimgcud.imgMeta.width)
                xcrdi = inimgcud.imgMeta.width - 1;
            if (xcrdi < 0)
                xcrdi = 0;
            if (ycrdj >= inimgcud.imgMeta.height)
                ycrdj = inimgcud.imgMeta.width - 1;
            if (ycrdj < 0)
                ycrdj = 0;

            // 将图像中像素值拷贝到邻域暂存空间
            neighbortmp[i * neighborwidth + j] = 
                    inimgcud.imgMeta.imgData[ycrdj * inimgcud.pitchBytes + 
                                             xcrdi]; 
        }
    }

    // 统计邻域内每个像素值的点的个数
    int pixelcount[PIXELRANGE];

    // 排序后的数组
    unsigned char *neighbortmpsorted = neighbortmp + length;
    
    // 对neighbortmp进行排序
    featureveccalc.sortNeighbor(neighbortmp, neighbortmpsorted, pixelcount,
                                length);
 
    // 计算平均灰度值
    float cv = featureveccalc.calAvgPixel(neighbortmpsorted, length * 1 / 3,
                                          length * 2 / 3);
    // 将平均灰度值写入到输出特征向量组中
    outfeaturevecarray.CV[index] = cv;

    // 计算灰度标准差
    float sd = featureveccalc.calPixelSd(neighbortmpsorted, length, cv);

    // 将灰度标准差写入到输出特征向量组中
    outfeaturevecarray.SD[index] = sd;

    // 计算最大非共起系数，首先求八个方向的灰度平均值
    // 从0-8依次为从方向右逆时针开始到方向右下结束
    float eightdirectionavg[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int j = 1; j <= n; j++) {
        // 当前像素右边的点的灰度求和
        eightdirectionavg[0] += neighbortmp[n * neighborwidth + n + j];  

        // 当前像素右上的点的灰度求和
        eightdirectionavg[1] += neighbortmp[(n - j) * neighborwidth + n + j];  
        
        // 当前像素上边的点的灰度求和
        eightdirectionavg[2] += neighbortmp[(n - j) * neighborwidth + n];

        // 当前像素左上的点的灰度求和
        eightdirectionavg[3] += neighbortmp[(n - j) * neighborwidth + n - j];

        // 当前像素左边的点的灰度求和
        eightdirectionavg[4] += neighbortmp[n * neighborwidth + n - j];

        // 当前像素左下的点的灰度求和
        eightdirectionavg[5] += neighbortmp[(n + j) * neighborwidth + n - j];

        // 当前像素下边的点的灰度求和
        eightdirectionavg[6] += neighbortmp[(n + j) * neighborwidth + n];

        // 当前像素右下的点的灰度求和
        eightdirectionavg[7] += neighbortmp[(n + j) * neighborwidth + n + j];
    }

    // 求八个方向上平均灰度值
    for (int i = 0; i < 8; i++) {
        eightdirectionavg[i] /= n;
    }
 
    // 根据给定的公式计算八个方向的共起系数并找出最大共起系数
    int t = (featureveccalc.getPitch() + 1) / 2;
    int m = (n - 2 * t + 1) * powf((2 * n + 1), 4);
    float eightncs[8]= { 0, 0, 0, 0, 0, 0, 0, 0 };  // 八个方向的共起系数
    float nc = 0;
    unsigned char pixeltmp1 = 0;                    // 暂存对角线上的像素
    unsigned char pixeltmp2 = 0;                    // 暂存对角线上的像素
 
    for (int j = t; j <= n - t; j++) {
        if (featureveccalc.getPitch() % 2 == 0) {

            // 当前像素右边的点共起系数求和
            pixeltmp1 = neighbortmp[n * neighborwidth + n + j - t];
            pixeltmp2 = neighbortmp[n * neighborwidth + n + j + t - 1];
            eightncs[0] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[0]) * 
                               (pixeltmp2 - eightdirectionavg[0])); 

            // 当前像素右上的点共起系数求和
            pixeltmp1 = neighbortmp[(n + j - t) * neighborwidth + n + j - t];
            pixeltmp2 = 
                    neighbortmp[(n + j + t - 1) * neighborwidth + 
                                n + j + t - 1];
            eightncs[1] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[1]) * 
                               (pixeltmp2 - eightdirectionavg[1])); 

            // 当前像素上边的点共起系数求和
            pixeltmp1 = neighbortmp[(n + j - t) * neighborwidth + n];
            pixeltmp2 = neighbortmp[(n + j + t - 1) * neighborwidth + n];
            eightncs[2] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[2]) * 
                               (pixeltmp2 - eightdirectionavg[2])); 

            // 当前像素左上的点共起系数求和
            pixeltmp1 = neighbortmp[(n + j - t) * neighborwidth + n - (j - t)];
            pixeltmp2 = 
                    neighbortmp[(n + j + t - 1) * neighborwidth + 
                                n - (j + t - 1)];
            eightncs[3] +=pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[3]) * 
                               (pixeltmp2 - eightdirectionavg[3])); 

            // 当前像素左边的点共起系数求和
            pixeltmp1 = neighbortmp[n * neighborwidth + n - (j - t)];
            pixeltmp2 = neighbortmp[n * neighborwidth + n - (j + t - 1)];
            eightncs[4] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[4]) * 
                               (pixeltmp2 - eightdirectionavg[4])); 

            // 当前像素左下的点共起系数求和
            pixeltmp1 = 
                    neighbortmp[(n - (j - t)) * neighborwidth + n - (j - t)];
            pixeltmp2 = 
                    neighbortmp[(n - (j + t - 1)) * neighborwidth + 
                                n - (j + t - 1)];
            eightncs[5] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[5]) * 
                               (pixeltmp2 - eightdirectionavg[5])); 

            // 当前像素下边的点共起系数求和
            pixeltmp1 = neighbortmp[(n - (j - t)) * neighborwidth + n];
            pixeltmp2 = neighbortmp[(n - (j + t - 1)) * neighborwidth + n];
            eightncs[6] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[6]) * 
                               (pixeltmp2 - eightdirectionavg[6])); 

            // 当前像素右下的点共起系数求和
            pixeltmp1 = neighbortmp[(n - (j - t)) * neighborwidth + n + j - t];
            pixeltmp2 = 
                    neighbortmp[(n - (j + t - 1)) * neighborwidth + 
                                n + j + t - 1];
            eightncs[7] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[7]) * 
                               (pixeltmp2 - eightdirectionavg[7])); 
        } else {
            // 当前像素右边的点共起系数求和
            pixeltmp1 = neighbortmp[n * neighborwidth + n + j - t];
            pixeltmp2 = neighbortmp[n * neighborwidth + n + j + t];
            eightncs[0] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[0]) * 
                               (pixeltmp2 - eightdirectionavg[0])); 

            // 当前像素右上的点共起系数求和
            pixeltmp1 = neighbortmp[(n + j - t) * neighborwidth + n + j - t];
            pixeltmp2 = 
                    neighbortmp[(n + j + t) * neighborwidth + n + j + t];
            eightncs[1] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[1]) * 
                               (pixeltmp2 - eightdirectionavg[1])); 

            // 当前像素上边的点共起系数求和
            pixeltmp1 = neighbortmp[(n + j - t) * neighborwidth + n];
            pixeltmp2 = neighbortmp[(n + j + t) * neighborwidth + n];
            eightncs[2] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[2]) * 
                               (pixeltmp2 - eightdirectionavg[2])); 

            // 当前像素左上的点共起系数求和
            pixeltmp1 = neighbortmp[(n + j - t) * neighborwidth + n - (j - t)];
            pixeltmp2 = 
                    neighbortmp[(n + j + t) * neighborwidth + n - (j + t)];
            eightncs[3] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[3]) * 
                               (pixeltmp2 - eightdirectionavg[3])); 

            // 当前像素左边的点共起系数求和
            pixeltmp1 = neighbortmp[n * neighborwidth + n - (j - t)];
            pixeltmp2 = neighbortmp[n * neighborwidth + n - (j + t)];
            eightncs[4] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[4]) * 
                               (pixeltmp2 - eightdirectionavg[4])); 

            // 当前像素左下的点共起系数求和
            pixeltmp1 = 
                    neighbortmp[(n - (j - t)) * neighborwidth + n - (j - t)];
            pixeltmp2 = 
                    neighbortmp[(n - (j + t)) * neighborwidth + n - (j + t)];
            eightncs[5] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[5]) * 
                               (pixeltmp2 - eightdirectionavg[5])); 

            // 当前像素下边的点共起系数求和
            pixeltmp1 = neighbortmp[(n - (j - t)) * neighborwidth + n];
            pixeltmp2 = neighbortmp[(n - (j + t)) * neighborwidth + n];
            eightncs[6] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[6]) * 
                               (pixeltmp2 - eightdirectionavg[6])); 

            // 当前像素右下的点共起系数求和
            pixeltmp1 = neighbortmp[(n - (j - t)) * neighborwidth + n + j - t];
            pixeltmp2 = 
                    neighbortmp[(n - (j + t)) * neighborwidth + n + j + t];
            eightncs[7] += pixelcount[pixeltmp1] * pixelcount[pixeltmp2] * 
                           abs((pixeltmp1 - eightdirectionavg[7]) * 
                               (pixeltmp2 - eightdirectionavg[7])); 
        }
    }

    // 计算并找出最大非共起系数
    for (int i = 0; i < 8; i++) {
        eightncs[i] /= m;
         
        // 记录最大非共起系数
        if (nc < eightncs[i])
            nc = eightncs[i];
    }

    // 将最大非共起系数写入输出特征向量组中
    outfeaturevecarray.NC[index] = nc;
}

// 核函数 _processFeatureVectorKer（处理初始特征向量）
static __global__ void                        
_processFeatureVectorKer(FeatureVecArray inoutfeaturevecarray,
        FeatureVecCalc featureveccalc, FeatureVectorProcessorParam param)
{
    // 计算当前 Thread 所对应的坐标集中的点的位置
    int index  = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果 index 超过了处理的点的个数不做处理直接返回
    if(index >= inoutfeaturevecarray.count)
        return;

    // 取出上一步计算得到的特征值
    float cv = inoutfeaturevecarray.CV[index];
    float sd = inoutfeaturevecarray.SD[index];
    float nc = inoutfeaturevecarray.NC[index];

    // 如果特征值低于下界或者高于上界，则置为下界或上界
    if (cv < param.mincv)
        cv = param.mincv;
    if (cv > param.maxcv)
        cv = param.maxcv;
    if (sd < param.minsd)
        sd = param.minsd;
    if (sd > param.maxsd)
        sd = param.maxsd;
    if (nc < param.minnc)
        nc = param.minnc;
    if (nc > param.maxnc)
        nc = param.maxnc;

    // 计算外部参数
    float a = 1 / (1 + featureveccalc.getAlpha() + featureveccalc.getBeta());

    // 处理特征值
    if (param.rangecv != 0)
        cv = a * (cv - param.avgcv) / param.rangecv;
    if (param.rangesd != 0)
        sd = a * featureveccalc.getAlpha() * (sd - param.avgsd) / 
             param.rangesd;
    if (param.rangenc != 0)
        nc = a * featureveccalc.getBeta() * (nc - param.avgnc) / 
             param.rangenc;
    
    // 将特征值赋回去
    inoutfeaturevecarray.CV[index] = cv;
    inoutfeaturevecarray.SD[index] = sd;
    inoutfeaturevecarray.NC[index] = nc;
}

// 全局函数：cmp （两个 float 变量的比较函数）
// 使用于针对特征值快速排序中的比较函数指针
int cmp(const void * a, const void * b)
{
    return((*(float *)a - *(float *)b > 0) ? 1 : -1);
}


// 宏 DELETE_THREE_FEATURE_ARRAY_HOST （删除 Host 端三个特征向量数组）
// 在出错或者函数运行结束后，清理三个特征向量数组的值
#define DELETE_THREE_FEATURE_ARRAY_HOST do {  \
        if ((cvshost) != NULL)                \
            delete  [] (cvshost);             \
        if ((sdshost) != NULL)                \
            delete  [] (sdshost);             \
        if ((ncshost) != NULL)                \
            delete  [] (ncshost);             \
    } while(0)


// Host 成员方法：calFeatureVector（计算初始特征向量）
__host__ int FeatureVecCalc::calFeatureVector(Image *inimg,
        CoordiSet *incoordiset, FeatureVecArray *outfeaturevecarray)
{
    // 检查输入参数是否为 NULL， 如果为 NULL 直接报错返回
    if (inimg == NULL || incoordiset == NULL || outfeaturevecarray == NULL)
        return NULL_POINTER;

    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR) 
        return errcode;

    // 将坐标集拷贝到 Device 内存中
    errcode = CoordiSetBasicOp::copyToCurrentDevice(incoordiset);
    if (errcode != NO_ERROR)
        return errcode;

    // 申请邻域存储空间
    unsigned char *neighbortmp = NULL ;
    int neighborsize = (neighborWidth * 2 + 1) * (neighborWidth * 2 + 1);
    errcode = cudaMalloc(
            (void **)(&neighbortmp), 
            2 * incoordiset->count * neighborsize * sizeof (unsigned char));
    if (errcode != cudaSuccess) {
        return CUDA_ERROR;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = incoordiset->count / blocksize.x + 1;
    gridsize.y = 1;

    // 计算初始特征向量
    _calcFeatureVectorKer<<<gridsize,blocksize>>>(insubimgCud, *incoordiset, 
                                                  *outfeaturevecarray, *this, 
                                                  neighbortmp);
   
    // 及时释放申请的 neighbortmp
    cudaFree(neighbortmp);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // Host 端三个特征向量数组声明
    float *cvshost = NULL;
    float *sdshost = NULL;
    float *ncshost = NULL;
 
    // 申请 Host 端 cv 特征值数组空间
    cvshost = new float[incoordiset->count];
    if (cvshost == NULL) {
        DELETE_THREE_FEATURE_ARRAY_HOST;
        return NULL_POINTER;
    }
    
    // 申请 Host 端 sd 特征值数组空间    
    sdshost = new float[incoordiset->count];
    if (sdshost == NULL) {
        DELETE_THREE_FEATURE_ARRAY_HOST;
        return NULL_POINTER;
    }        
    
    // 申请 Host 端 nc 特征值数组空间    
    ncshost = new float[incoordiset->count];   
    if (ncshost == NULL) {
        DELETE_THREE_FEATURE_ARRAY_HOST;
        return NULL_POINTER;
    }

    // 将 CV 拷贝到 Host 端
    errcode = cudaMemcpy(cvshost, outfeaturevecarray->CV, 
                         incoordiset->count * sizeof (float),
                         cudaMemcpyDeviceToHost);

    // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
    // 后报错返回。
    if (errcode != cudaSuccess) {
        // 清除之前申请的内存
        DELETE_THREE_FEATURE_ARRAY_HOST;
        return CUDA_ERROR;
    }

    // 将 SD 拷贝到 Host 端
    errcode = cudaMemcpy(sdshost, outfeaturevecarray->SD, 
                         incoordiset->count * sizeof (float),
                         cudaMemcpyDeviceToHost);

    // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
    // 后报错返回。
    if (errcode != cudaSuccess) {
        // 清除之前申请的内存
        DELETE_THREE_FEATURE_ARRAY_HOST;
        return CUDA_ERROR;
    }

    // 将 NC 拷贝到 Host 端
    errcode = cudaMemcpy(ncshost, outfeaturevecarray->NC,
                         incoordiset->count * sizeof (float),
                         cudaMemcpyDeviceToHost);
 
    // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
    // 后报错返回。
    if (errcode != cudaSuccess) {
        // 清除之前申请的内存
        DELETE_THREE_FEATURE_ARRAY_HOST;
        return CUDA_ERROR;
    } 

    // 使用快速排序分别对三组特征值排序
    qsort(cvshost, incoordiset->count, sizeof(float), cmp);
    qsort(sdshost, incoordiset->count, sizeof(float), cmp);
    qsort(ncshost, incoordiset->count, sizeof(float), cmp);

    // 处理参数定义
    FeatureVectorProcessorParam param;

    // 计算上下限的下标
    int bordermin = (int)(0.05 * incoordiset->count);
    int bordermax = (int)(0.95 * incoordiset->count);
    
    // 给各个处理参数赋值
    param.mincv = cvshost[bordermin];
    param.maxcv = cvshost[bordermax];
    param.avgcv = calAvgFeatureValue(cvshost, bordermin, bordermax);
    param.rangecv = param.maxcv - param.mincv;

    param.minsd = sdshost[bordermin];
    param.maxsd = sdshost[bordermax];
    param.avgsd = calAvgFeatureValue(sdshost, bordermin, bordermax);
    param.rangesd = param.maxsd - param.minsd;

    param.minnc = ncshost[bordermin];
    param.maxnc = ncshost[bordermax];
    param.avgnc = calAvgFeatureValue(ncshost, bordermin, bordermax);
    param.rangenc = param.maxnc - param.minnc;

    // 对初始特征向量进行简单处理
    _processFeatureVectorKer<<<gridsize,blocksize>>>(*outfeaturevecarray, 
                                                     *this, param);
    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;     
               
    // 内存清理    
    DELETE_THREE_FEATURE_ARRAY_HOST;
 
    return NO_ERROR;
}

