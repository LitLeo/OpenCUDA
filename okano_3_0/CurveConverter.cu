// CurveConverter.h
// 创建人：曹建立
//
// 结构体 curve 和 Image 之间相互转换（CurveConverter）

#include "CurveConverter.h"
#include <iostream>

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X    32
#define DEF_BLOCK_Y     8

// 宏：DEF_BLOCK_1D
// 定义一维线程块尺寸。
#define DEF_BLOCK_1D  512

// --------------------------内核方法实现-------------------------------------
// Kernel 函数：_initiateImgKer（实现将图像初始化为 lowpixel 算法） 
static __global__ void _initiateImgKer(ImageCuda inimg, unsigned char lowpixel)
{
    // 计算线程对应的输出点的位置，其中 dstc 和 dstr 分别表示线程处理的像素点
    // 的坐标的 x 和 y 分量（其中，dstc 表示 column；dstr 表示 row）。由于我们
    // 采用了并行度缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一
    // 列的相邻 4 行上，因此，对于 dstr 需要进行乘 4 计算。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (dstc >= inimg.imgMeta.width || dstr >= inimg.imgMeta.height)
        return;

    // 计算第一个输入坐标点对应的图像数据数组下标。
    int index = dstr * inimg.pitchBytes + dstc;

    // 将目标点赋值为 lowpixel。
    inimg.imgMeta.imgData[index] = lowpixel;

    // 处理剩下的三个像素点。
    for (int i = 0; i < 3; i++) {
        // 这三个像素点，每个像素点都在前一个的下一行，而 x 分量保持不变。因
        // 此，需要检查这个像素点是否越界。检查只针对 y 分量即可，x 分量在各
        // 点之间没有变化，故不用检查。
        if (++dstr >= inimg.imgMeta.height)
            return;

        // 根据上一个像素点，计算当前像素点的对应的输出图像的下标。由于只有 y
        // 分量增加 1，所以下标只需要加上一个 pitch 即可，不需要在进行乘法计
        // 算。
        index += inimg.pitchBytes;

        // 将目标点赋值为 lowpixel。
        inimg.imgMeta.imgData[index] = lowpixel;
    }
}


// Kernel 函数：_Cur2ImgKer（实现将坐标集转化为图像算法）
static __global__ void _curve2ImgKer(int length,
                                     int * curveDatadev,
                                     ImageCuda inimg,
                                     unsigned char higpixel)
{
    // index 表示线程处理的 Curve 中一个像素点的坐标。
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查坐标点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (index >= length)
        return;

    // 获得目标点在图像中的对应位置。y*w+x
    int imgpos = curveDatadev[index*2+1] * inimg.pitchBytes 
                +curveDatadev[index*2];

    // 将坐标集中坐标在图像中对应的像素点的像素值置为 higpixel。
    inimg.imgMeta.imgData[imgpos] = higpixel;
}
// --------------------------成员方法实现-------------------------------------

#define FREE_MEMORY     if (curveDatadev != NULL) cudaFree(curveDatadev);

// 成员方法：Curve2Img(把 Curve 中的点集合绘制到指定图像上)
__host__ int                     // 返回值：函数是否正确执行，若函数正确执
                                 // 行，返回 NO_ERROR。
    CurveConverter::curve2Img(
    Curve *curv,                   // 输入的结构体
    Image *outimg                   // 输出的图像,需要事先分配好空间
    ){
        // 局部变量，错误码。
        int errcode;
        
        // 检查输入坐标集，输出图像是否为空。
        if (curv == NULL || outimg == NULL)
            return NULL_POINTER;

        // 将输出图像拷贝到 device 端。
        errcode = ImageBasicOp::copyToCurrentDevice(outimg);
        if (errcode != NO_ERROR)
            return errcode;

        // 提取输出图像。
        ImageCuda outsubimgCud;
        errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
        if (errcode != NO_ERROR) {
            return errcode;
        }

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 gridsize, blocksize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
        gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) /
            (blocksize.y * 4);

        // 初始化输入图像内所有的像素点的的像素值为 lowpixel，为转化做准备。
        _initiateImgKer<<<gridsize, blocksize>>>(outsubimgCud, bkcolor);
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。调用一维核函数，
        // 在这里设置线程块内的线程数为 512，用 DEF_BLOCK_1D 表示。
        // 每个线程处理 curv 中一个点
        size_t blocksize1, gridsize1;
        blocksize1 = DEF_BLOCK_1D;
        gridsize1 = (curv->curveLength + blocksize1 - 1) / blocksize1;

        // 把需要使用的 curv 中的部分数据拷贝到 device 内存中
         
        int *curveDatadev;

        cudaError_t cuerrcode;


        // 计算字节数
        int datasize=(curv->curveLength*2)*sizeof (int);
       
        // 申请空间。
        cuerrcode = cudaMalloc((void **)&curveDatadev, datasize);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        cuerrcode = cudaMemcpy(curveDatadev, curv->crvData,
                               datasize, 
                                cudaMemcpyHostToDevice);
        if (cudaGetLastError() != cudaSuccess) {
            FREE_MEMORY
            return CUDA_ERROR;
        }
 
        // 将输入坐标集转化为输入图像图像，即将坐标集内点映射在图像上点的
        // 像素值置为 highpixel。
        _curve2ImgKer<<<gridsize1, blocksize1>>>(curv->curveLength,
                                                   curveDatadev,
                                                   outsubimgCud,
                                                   bordercolor);
        if (cudaGetLastError() != cudaSuccess){
            FREE_MEMORY
            return CUDA_ERROR;
        }
        // 处理完毕，退出。
        FREE_MEMORY
        return NO_ERROR;
}

#undef FREE_MEMORY 

// 成员方法：Img2Curve（把图像上的点保存到 Curve 结构体中）
// 只对curve的minX，minY，maxX，maxY，curveCoordiX，curveCoordiY，length域赋值。
__host__ int                     // 返回值：函数是否正确执行，若函数正确执
    // 行，返回 NO_ERROR。
    CurveConverter::img2Curve(
    Image *img,                    // 输入的图像
    Curve *cur                     // 输出的结构体
    ){
        ImageBasicOp::copyToHost(img);
        if(img==NULL || cur==NULL)
            return 1;
        int w,h;
        w=img->width;
        h=img->height;
        int imgsize=w*h;
        // 每个点（x，y）占用两个整数存放,按最大容量分配空间
        int *curData=new int[2*imgsize];

        // 把图像前景点放入临时数组中
        int count=0;
        // 最值初始化，最小值初始化为最大，最大值初始化为最小。
        cur->minCordiX=w; cur->maxCordiX=0;
        cur->minCordiY=h; cur->maxCordiY=0;
        for(int j=0;j<h;j++)
            for(int i=0;i<w;i++)
            {
                // 图像中的点（i，j）
                int curpix=img->imgData[j*w+i];
                if(curpix==bordercolor ){
                    curData[count]=i;
                    count++;
                    curData[count]=j;
                    count++;
                    // 记录最值
                    if(i<cur->minCordiX)cur->minCordiX=i;
                    if(i>cur->maxCordiX)cur->maxCordiX=i;
                    if(j<cur->minCordiY)cur->minCordiY=j;
                    if(j>cur->maxCordiY)cur->maxCordiY=j;
                }
            }

        // 给curve的长度变量、数据数组分配空间并赋值
        cur->curveLength=count/2;//count表示整数的个数，长度应为count的一半
        cur->crvData=new int[count];
        memcpy(cur->crvData,curData,count*sizeof(int));

        delete[] curData;

        #ifdef DEBUG_TIME
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&runTime, start, stop);
                    cout << "[coor] img to coor " << runTime << " ms" << endl;
        #endif

        return NO_ERROR;
}



