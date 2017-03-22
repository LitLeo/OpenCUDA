// InscribedCircle
// 实现的曲线内接圆

#include "InscribedCircle.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace std;



// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X    32
#define DEF_BLOCK_Y     8

// 宏：IN_LABEL 和 OUT_LABEL
// 定义了曲线内的点和曲线外的点标记值
#define IN_LABEL  255
#define OUT_LABEL   0

// Kernel 函数：_setCloseAreaKer（将封闭曲线包围的内部区域的值变为白色）
// 该核函数使用著名的射线法确定点和一个封闭曲线的位置关系，即如果由当前点引射线，
// 与曲线有奇数个交点则在内部，如果有偶数个交点，则在曲线外部（ 0 属于偶数），
// 引用该算法实现将封闭曲线包围的内部区域的值变为白色，并且需要
// 得到闭合曲线包围的点的个数，用于后续处理
static __global__ void      // Kernel 函数无返回值
_setCloseAreaKer(
        CurveCuda curve,    // 输入曲线
        ImageCuda maskimg,  // 输出标记结果
        int *count          // 闭合曲线包围点的个数
);


// 全局函数：compare （两个 int 变量的比较函数）
// 使用于针对特征值快速排序中的比较函数指针
int compare(const void * a, const void * b)
{
    return *(int *)b - *(int *)a;
    // return *(float *)b > *(float *)a ? 1:-1;
}

// Kernel 函数：_setCloseAreaKer（将封闭曲线包围的内部区域的值变为白色）
static __global__ void _setCloseAreaKer(CurveCuda curve, ImageCuda maskimg,
                                        int *count)
{
    // 计算当前线程的索引
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    // 判断当前线程是否越过输入图像尺寸
    if (xidx >= maskimg.imgMeta.width || yidx >= maskimg.imgMeta.height)
        return;

    // 定义部分寄存器变量   
    int downcount = 0;                       // 向下引射线和曲线的交点个数
    int length = curve.crvMeta.curveLength;  // 曲线上的点的个数 
    int outpitch = maskimg.pitchBytes;       // 输出标记图像的 pitch

    // 首先将所有点标记为曲线外的点
    maskimg.imgMeta.imgData[yidx * outpitch + xidx] = OUT_LABEL;

    int flag = 0; // 判断是否进入切线区域

    // 遍历曲线，统计上述各个寄存器变量的值
    for (int i = 0; i < length; i++) {
        int x = curve.crvMeta.crvData[2 * i];
        int y = curve.crvMeta.crvData[2 * i + 1];

        // 曲线中的下一个点的位置
        int j = (i + 1) % length;
        int x2 = curve.crvMeta.crvData[2 * j];

        // 曲线中上一个点的位置
        int k = (i - 1 + length) % length;
        int x3 = curve.crvMeta.crvData[2 * k];

        // 曲线上的第 i 个点与当前点在同一列上
        if (x == xidx) {
             if (y == yidx) {
                ////当前点在曲线上，此处把曲线上的点也作为曲线内部的点
                // maskimg.imgMeta.imgData[yidx *  outpitch+ xidx] = IN_LABEL;
                 return;
             }

            // 交点在当前点的下方
            if (y > yidx) {
                // 曲线上下一个也在射线上时，避免重复统计，同时设置 flag 
                // 标记交点行开始。如果下一个点不在射线上，通过 flag 判断到
                // 底是交点行结束还是单点相交，如果是单点相交判断是否为突出点
                // 如果是交点行结束判断是否曲线在交点行同侧，以上都不是统计值
                // 加一.
                if (x2 == xidx) {
                    if (flag == 0) 
                        flag = x3 - x;   
                } else {
                    if (flag == 0) {
                        if ((x3 - x) * (x2 - x) <= 0) 
                            downcount++;                        
                    } else {
                        if (flag * (x2 - x) < 0) 
                            downcount++;
                        flag = 0;
                    }
                }
            }
        }
    }

    // 交点数均为奇数则判定在曲线内部
    if (downcount % 2 == 1) {
        maskimg.imgMeta.imgData[yidx * outpitch + xidx] = IN_LABEL;
        atomicAdd(count, 1);
    }
}

// Kernel 函数：_calInsCirRadiusKer（）
static __global__ void _calInsCirRadiusKer(CurveCuda curve, ImageCuda maskimg,
        int *dev_inscirDist, int *dev_inscirX, int *dev_inscirY, int *dev_num)
{
    // 计算当前线程的索引
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 判断当前线程是否越过输入图像尺寸
    if (xidx >= maskimg.imgMeta.width || yidx >= maskimg.imgMeta.height)
        return;
    
    // 得到标记图像的 pitch
    int pitch = maskimg.pitchBytes;
    int width = maskimg.imgMeta.width;
    int index = yidx * width + xidx;
    
    // 曲线上的点的个数 
    int length = curve.crvMeta.curveLength;  

    if (maskimg.imgMeta.imgData[yidx * pitch+ xidx] == OUT_LABEL) {
         dev_inscirDist[index] = 0;
    } else {
        int min = 65535;
        int dist;
        for (int i = 0; i < length; i++) {
            int x = curve.crvMeta.crvData[2 * i];
            int y = curve.crvMeta.crvData[2 * i + 1];
            dist = (xidx - x) * (xidx - x) + (yidx - y) * (yidx - y);
            if (dist < min) 
                min = dist;
        }
        dev_inscirDist[index] = (int)sqrtf(min);
    }
}

__host__ void getH_inscirDist(int *tmp_inscirDist, int width, int height, 
                              int *h_inscirDist, int *h_inscirX, int *h_inscirY)
{
    int size = 0;
    int tmp;
    int i, j;
    for (i = 0; i < height; i++) {
        for(j = 0; j < width; j++) {
            tmp = *(tmp_inscirDist + i * width + j);
            if (tmp) {
                *(h_inscirDist + size) = tmp;
                *(h_inscirX + size) = j;
                *(h_inscirY + size) = i;
                size++;
            }
        }
    }
}

__host__ void swap(int *h_inscirDist, int *h_inscirX, int *h_inscirY,
                   int index, int max_index)
{
    *(h_inscirDist + index) = *(h_inscirDist + index) ^ *(h_inscirDist + max_index);
    *(h_inscirDist + max_index) = *(h_inscirDist + index) ^ *(h_inscirDist + max_index);
    *(h_inscirDist + index) = *(h_inscirDist + index) ^ *(h_inscirDist + max_index);
    
    *(h_inscirX + index) = *(h_inscirX + index) ^ *(h_inscirX + max_index);
    *(h_inscirX + max_index) = *(h_inscirX + index) ^ *(h_inscirX + max_index);
    *(h_inscirX + index) = *(h_inscirX + index) ^ *(h_inscirX + max_index);
    
    *(h_inscirY + index) = *(h_inscirY + index) ^ *(h_inscirY + max_index);
    *(h_inscirY + max_index) = *(h_inscirY + index) ^ *(h_inscirY + max_index);
    *(h_inscirY + index) = *(h_inscirY + index) ^ *(h_inscirY + max_index);
    
}

__host__ void setFlag(bool *flag, int *inscirX, int *inscirY, 
                      int cnum, int index, int disTh)
{
    int x = inscirX[index];
    int y = inscirY[index];
    //int max = 0;
    int length = disTh * disTh;
    //int flagnum = 0;
    for(int i = index + 1; i < cnum; i++) {
        if (flag[i]) {
            int dis = (inscirX[i] - x) * (inscirX[i] - x) + 
                      (inscirY[i] - y) * (inscirY[i] - y);
            //if(max<dis) max = dis;
            if (dis < length) {
                flag[i] = false;
                //flagnum++;
            }
        }
    }
}

__host__ void getInscirDist(int *inscirDist, int *inscirX, int *inscirY,
                            int num, int disTh, int &count,int *h_inscirDist, 
                            int *h_inscirX, int *h_inscirY, int cnum)
{
    int max = 0;
    int max_index = 0;
    int tmp;
    int i, j;
    
    bool *flag = new bool[cnum];
    memset(flag, true, cnum * sizeof (bool));
    //cout<<"getInscirDist num disTh: "<<num<<" "<<disTh<<endl;
    for(i = 0; i < num; i++) {
        max = 0;
        max_index = 0;
        bool in = false;
        for (j = i; j < cnum; j++) {
            tmp = *(h_inscirDist + j);
            if (!flag[j]) continue;
            if (tmp > max) {
                max = tmp;
                max_index = j;
                in = true;
            }
        }
        
        if (!in) {
            count = i;
            break;
        }
        
        swap(h_inscirDist, h_inscirX, h_inscirY, i, max_index);
        *(inscirDist + i) = *(h_inscirDist + i);
        *(inscirX + i) = *(h_inscirX + i);
        *(inscirY + i) = *(h_inscirY + i);
        flag[i] = false;
        setFlag(flag, h_inscirX, h_inscirY, cnum, i, disTh);
    }
    
    if(i == num) {
        count = num;
    }
    delete [] flag;
}

// Host 成员方法：inscribedCircle（曲线最大内接圆）
__host__ int InscribedCircle::inscribedCircle(Curve *curve, int width, 
        int height, int &count, int *inscirDist, int *inscirX, int *inscirY)
{
    // 判断输入曲线,输入半径数组，输入圆心坐标是否为空
    if (curve == NULL || inscirDist == NULL ||
        inscirX == NULL || inscirY == NULL)
        return NULL_POINTER;
        
    // 检查输入参数是否有数据
    if (curve->curveLength <= 0 || width <= 0 || height <= 0)
        return INVALID_DATA;
    
    // 检查输入曲线是否为封闭曲线，如果不是封闭曲线返回错误
    //if (!curve->closed)
    //     return INVALID_DATA;
        
    // 局部变量，错误码。
    int errcode;
    cudaError_t cuerrcode;
    
    // 将曲线拷贝到 Device 内存中
    errcode = CurveBasicOp::copyToCurrentDevice(curve);
    if (errcode != NO_ERROR)
        return errcode;

    // 获取 CurveCuda 指针
    CurveCuda *curvecud = CURVE_CUDA(curve);

    // 定义设备端局部变量，用于多份数据的一份申请
    void *temp_dev = NULL;
    
    // 定义临时标记图像指针
    Image *maskimg = NULL;
    
    // 给临时标记图像在设备申请空间
    ImageBasicOp::newImage(&maskimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(maskimg, width, height);
    if (errcode != NO_ERROR) {
        //
        return errcode;
    }

    // 获取 ImageCuda 指针
    ImageCuda *maskimgcud = IMAGE_CUDA(maskimg);
    size_t datasize = width * height;

    // 给 temp_dev 在设备申请空间
    cuerrcode = cudaMalloc((void**)&temp_dev, (datasize * 3 + 1) * sizeof (int));
    if (cuerrcode != cudaSuccess) {
        //
        return CUDA_ERROR;
    }

    // 定义设备指针
    int *dev_inscirDist = (int *)temp_dev;
    int *dev_inscirX = (int *)(dev_inscirDist + datasize);
    int *dev_inscirY = (int *)(dev_inscirX + datasize);
    int *dev_count = (int *)(dev_inscirY + datasize);
    cudaMemset(dev_count, 0, sizeof (int));

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y  - 1) / blocksize.y;

    // 调用核函数，将封闭曲线包围的内部区域的值变为白色，并且得到包围点的个数
    _setCloseAreaKer<<<gridsize, blocksize>>>( 
            *curvecud, *maskimgcud, dev_count); 

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        //FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }

    // 调用核函数，将
    _calInsCirRadiusKer<<<gridsize, blocksize>>>(*curvecud, *maskimgcud, 
            dev_inscirDist, dev_inscirX, dev_inscirY, dev_count); 

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess) {
        //FREE_CURVE_TOPOLOGY;
        return CUDA_ERROR;
    }
    
    int cnum; 
    cudaMemcpy(&cnum, dev_count, sizeof (int), cudaMemcpyDeviceToHost);

    int *h_inscirDist = new int [cnum];
    
    int *h_inscirX = new int [cnum];
    int *h_inscirY = new int [cnum];

    int *tmp_inscirDist = new int [datasize];
    
    cudaMemcpy(tmp_inscirDist, dev_inscirDist, sizeof (int) * datasize, 
               cudaMemcpyDeviceToHost);
    
    getH_inscirDist(tmp_inscirDist, width, height,
                    h_inscirDist, h_inscirX, h_inscirY);
    
    getInscirDist(inscirDist, inscirX, inscirY, this->num, this->disTh, count,
                  h_inscirDist, h_inscirX, h_inscirY, cnum);
    

    // for(int i = 0; i < count; i++) {
        // cout<<"["<<h_inscirDist[i]<<",("<<h_inscirX[i]<<","<<h_inscirY[i]<<")]";
    // }
    // cout<<endl;
     for(int i = 0; i < count; i++) {
         cout<<"["<<inscirDist[i]<<",("<<inscirX[i]<<","<<inscirY[i]<<")]";
     }
    //sort(tmp,tmp+datasize,greater<int>());
    //for(int i=0;i<m;i++) {
    //  if(i%50==0) cout<<endl;
    //  cout<<inscirDist[i]<< " ";
    //}
    //memcpy(t1, inscirDist, cnum * sizeof (int));
    // memset(t1+cnum,0,(n - cnum) * sizeof (int));
    // int *t2 = new int [n];
    // SortArray sort(1024,n/1024,1,true);
    // sort.shearSort(t1,t2);
    
    cout<<endl<<"cnum count: "<<cnum<<" "<<count<<endl;
    //cout<<width<<" "<<height<<endl;
    cudaFree(temp_dev);
    delete [] h_inscirDist;
    delete [] h_inscirX;
    delete [] h_inscirY;
    delete [] tmp_inscirDist;
    
    return NO_ERROR;
}
