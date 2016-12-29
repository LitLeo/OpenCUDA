// FillCurve.cu
// 创建人：曹建立

#include "FillCurve.h"
#include <iostream>
using namespace std;
// 宏：DEBUG_IMG
// 定义是否输出中间图像调试信息
// #define DEBUG_IMG
// 宏：DEBUG_TIME
// 定义是否输出时间调试信息
// #define DEBUG_TIME

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X 32 
#define DEF_BLOCK_Y 8


// 宏：CUDA_PIXEL_GLO(x,y) 
// 获取全局内存中内核函数中图像中（x，y）像素的位置
#define CUDA_PIXEL_GLO(x,y) imgcud.imgMeta.imgData[(y)*imgcud.pitchBytes+(x)]
// 宏：CUDA_VALID_GLO(x,y) 
// 判断全局内存中内核函数中（x，y）像素的位置是否合法
#define CUDA_VALID_GLO(x,y) (x>=0 && x<imgcud.imgMeta.width && y>=0 && y<imgcud.imgMeta.height)
// 宏：CUDA_PIXEL_SHR(x,y) 
// 获取共享内存中内核函数中图像中（x，y）像素的位置
//#define CUDA_PIXEL_SHR(x,y) imgcud.imgMeta.imgData[(y)*imgcud.pitchBytes+(x)]
#define CUDA_PIXEL_SHR(x,y) shareImg[(y)*w+(x)]
// 宏：CUDA_VALID_SHR(x,y) 
// 判断共享内存内核函数中（x，y）像素的位置是否合法
#define CUDA_VALID_SHR(x,y) (x>=0 && x<w && y>=0 && y<h)

// 宏：CUDA_STACK_SIZE 
// 自定义的cuda栈最大容量,根据测试，不太复杂的图像，最大深度为4，因此最大值定义12足够
#define CUDA_STACK_SIZE 12



//--------------------------内核方法声明------------------------------------
// Kernel 函数：_seedScanLineOutConGlobalKer（并行的种子扫描线算法，种子在轮廓外部）
// 全局内存版
static __global__ void _seedScanLineOutConGlobalKer(
    ImageCuda imgcud,              // 要填充的轮廓图像
    int arrayxlen
);
// Kernel 函数：_seedScanLineOutConShareKer（并行的种子扫描线算法，种子在轮廓外部）
// 共享内存版
static __global__ void _seedScanLineOutConShareKer(
    ImageCuda imgcud,              // 要填充的轮廓图像
    int threadsize
    );
// Kernel 函数：_intersectionKer（求两幅图像交,结果放入outbordercud中）
static __global__ void _intersectionKer(
    ImageCuda outborderCud,        // 外轮廓被填充过后的图像
    ImageCuda inborderCud          // 内轮廓被填充过后的图像
);

//--------------------------内核方法实现------------------------------------
// Kernel 函数：_seedScanLineOutConGlobalKer（并行的种子扫描线算法，种子在轮廓外部）
static __global__ void _seedScanLineOutConGlobalKer(
    ImageCuda imgcud,               // 要填充的轮廓图像
    int len
 ){
    // 计算线程对应的输出点的位置的 x 和 y 分量
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int seedx,seedy;
    // 线程号转换成顺时针种子点编号，根据编号，计算该点坐标
    int w=imgcud.imgMeta.width;
    int h=imgcud.imgMeta.height;
    // 超过线程范围的线程退出
    if(x>=len) return;
    if (x<w) {
        seedy=0;
        seedx=x;
    }
    else if(x< w+h-1){
        seedx=w-1;
        seedy=x-(w-1);
    }
    else if(x< w*2+h-2){
        seedx=x-(w+h-2);
        seedy=h-1;
    }
    else {
        seedx=0;
        seedy=x-(2*w+h-2);
    }

    // 如果得到的种子超过图像范围，或者不是背景点(可能是轮廓点或者已经被其他线程
    // 填充，则直接退出)
    if(seedx>=imgcud.imgMeta.width || 
       seedy >= imgcud.imgMeta.height || 
       CUDA_PIXEL_GLO(seedx,seedy) != BK_COLOR)
        return;

    // 填充工作
    // 输入：轮廓线workimg,种子seed；
    // 输出：填充过的workimg
    int cudastack[CUDA_STACK_SIZE];
    int stackptr=0;
    int xtemp,xright,xleft;
    int spanfill;
    // 种子入栈
    cudastack[stackptr++]=seedx;
    cudastack[stackptr++]=seedy;
    // stackptr==0表示栈为空，>0说明栈不空，每个像素点占用2个位置
    while(stackptr>0){
        int curx,cury;
        // 统计堆栈最大深度
        //if(stackptr>stackmaxsize[0])
            //stackmaxsize[0]=stackptr;
        // 入栈顺序x、y，出栈顺序应y、x。

        cury=cudastack[--stackptr];
        curx=cudastack[--stackptr];
        // 填充当前点
        CUDA_PIXEL_GLO(curx,cury)=BORDER_COLOR;
        // 向右填充,填充过程中检测当前点坐标
        for(xtemp=curx+1;CUDA_VALID_GLO(xtemp,cury)&&CUDA_PIXEL_GLO(xtemp,cury)!=BORDER_COLOR;xtemp++){
            CUDA_PIXEL_GLO(xtemp,cury)=BORDER_COLOR;}
        //纪录当前线段最右位置
        xright=xtemp-1;
        // 向左填充
        for(xtemp=curx-1;CUDA_VALID_GLO(xtemp,cury)&&CUDA_PIXEL_GLO(xtemp,cury)!=BORDER_COLOR;xtemp--){
            CUDA_PIXEL_GLO(xtemp,cury)=BORDER_COLOR;
            }
        // 纪录当前线段最左位置
        xleft=xtemp+1;

        //cout<<"hang:"<<cury<<"["<<xleft<<","<<xright<<"]"<<endl;

        // 下方相邻扫描线,从左向右扫描
        xtemp=xleft; cury++;
        // 每次循环把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright && cury>=0 && cury<imgcud.imgMeta.height){
            spanfill=0;
            // 找到一个线段的最右点
            while(CUDA_PIXEL_GLO(xtemp,cury)!=BORDER_COLOR &&
                  xtemp<=xright){
                spanfill=1;
                xtemp++;
            }
            // 最右点(xtemp-1,cury)入栈
            if(spanfill==1){
                cudastack[stackptr++]=xtemp-1;
                cudastack[stackptr++]=cury;
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while(
                xtemp<=xright && 
                CUDA_PIXEL_GLO(xtemp,cury)==BORDER_COLOR)
                 xtemp++;
        } // 下方扫描线结束

        //上方相邻扫描线
        xtemp=xleft; cury-=2;
        // 循环一次，把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright && cury>=0 && cury<imgcud.imgMeta.height){
            spanfill=0;
            // 找到一个线段的最右点
            while(
                xtemp<=xright && 
                CUDA_PIXEL_GLO(xtemp,cury)!=BORDER_COLOR 
            ){
            spanfill=1;
            xtemp++;
            }
            // 最右点入栈
            if(spanfill==1){
                cudastack[stackptr++]=xtemp-1;
                cudastack[stackptr++]=cury;
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while(CUDA_PIXEL_GLO(xtemp,cury)==BORDER_COLOR
                   && xtemp<=xright)
                 xtemp++;
        } // 上方扫描线结束

    }// 填充结束
    return ;
}
// Kernel 函数：_seedScanLineOutConShareKer（并行的种子扫描线算法，种子在轮廓外部）
static __global__ void _seedScanLineOutConShareKer(
    ImageCuda imgcud,               // 要填充的轮廓图像
    int threadsize
    ){
        // 计算线程对应的输出点的位置的 x 和 y 分量
        int x = threadIdx.x;
        int seedx,seedy;
        // 图像放入共享内存，动态申请，大小由参数决定,和在全局内存中的大小一致
        // 超过线程范围的线程退出
        // if(x>=threadsize) return;
        extern __shared__ unsigned char shareImg[];
        int w=imgcud.imgMeta.width;
        int h=imgcud.imgMeta.height;
        // 图像拷贝到共享内存
        int imgarraysize=w*h;
        // 计算每个线程需要负责多少个像素点的复制
        register int stride=imgarraysize/threadsize+1;
        // 本线程负责的像素点在数组中的开始位置下标
        register int beginIdx=x*stride;

        register int ny,nx;
        if (beginIdx<imgarraysize){
            // 本线程负责的像素点开始坐标（nx,ny）
            ny=beginIdx / w;
            nx=beginIdx % w;
            // 从（nx,ny）开始的 stride 个像素点从全局内存中复制到共享内存中。
            // 注意：全局内存总的pitchBytes
            for(int i=0;i<stride;i++){
                // 末尾可能越界的像素点跳过
                if(ny*w+nx+i>=imgarraysize) break;
                shareImg[ny*w+nx+i]=imgcud.imgMeta.imgData[ny*imgcud.pitchBytes+nx+i];
            }
        }

        __syncthreads(); 

        // 线程号转换成顺时针种子点编号，根据编号，计算该点坐标
        if (x<w) {
            seedy=0;
            seedx=x;
        }
        else if(x< w+h-1){
            seedx=w-1;
            seedy=x-(w-1);
        }
        else if(x< w*2+h-2){
            seedx=x-(w+h-2);
            seedy=h-1;
        }
        else {
            seedx=0;
            seedy=x-(2*w+h-2);
        }

        // 如果得到的种子超过图像范围，或者不是背景点(可能是轮廓点或者已经被其他线程
        // 填充，则直接退出)
        if(seedx>=imgcud.imgMeta.width || 
            seedy >= imgcud.imgMeta.height || 
            CUDA_PIXEL_SHR(seedx,seedy) != BK_COLOR)
            return;

        // 填充工作
        // 输入：轮廓线workimg,种子seed；
        // 输出：填充过的workimg
        int cudastack[CUDA_STACK_SIZE];
        int stackptr=0;
        int xtemp,xright,xleft;
        int spanfill;
        // 种子入栈
        cudastack[stackptr++]=seedx;
        cudastack[stackptr++]=seedy;
        // stackptr==0表示栈为空，>0说明栈不空，每个像素点占用2个位置
        while(stackptr>0){
            int curx,cury;
            // 统计堆栈最大深度
            //if(stackptr>stackmaxsize[0])
            //stackmaxsize[0]=stackptr;
            // 入栈顺序x、y，出栈顺序应y、x。

            cury=cudastack[--stackptr];
            curx=cudastack[--stackptr];
            // 填充当前点
            CUDA_PIXEL_SHR(curx,cury)=BORDER_COLOR;
            // 向右填充,填充过程中检测当前点坐标
            for(xtemp=curx+1;CUDA_VALID_SHR(xtemp,cury)&&CUDA_PIXEL_SHR(xtemp,cury)!=BORDER_COLOR;xtemp++){
                CUDA_PIXEL_SHR(xtemp,cury)=BORDER_COLOR;}
            //纪录当前线段最右位置
            xright=xtemp-1;
            // 向左填充
            for(xtemp=curx-1;CUDA_VALID_SHR(xtemp,cury)&&CUDA_PIXEL_SHR(xtemp,cury)!=BORDER_COLOR;xtemp--){
                CUDA_PIXEL_SHR(xtemp,cury)=BORDER_COLOR;
            }
            // 纪录当前线段最左位置
            xleft=xtemp+1;

            //cout<<"hang:"<<cury<<"["<<xleft<<","<<xright<<"]"<<endl;

            // 下方相邻扫描线,从左向右扫描
            xtemp=xleft; cury++;
            // 每次循环把一个线段种子放入堆栈（一条扫描线中可能多个线段）
            while(xtemp<=xright && cury>=0 && cury<imgcud.imgMeta.height){
                spanfill=0;
                // 找到一个线段的最右点
                while(CUDA_PIXEL_SHR(xtemp,cury)!=BORDER_COLOR &&
                    xtemp<=xright){
                        spanfill=1;
                        xtemp++;
                }
                // 最右点(xtemp-1,cury)入栈
                if(spanfill==1){
                    cudastack[stackptr++]=xtemp-1;
                    cudastack[stackptr++]=cury;
                }
                // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
                while(
                    xtemp<=xright && 
                    CUDA_PIXEL_SHR(xtemp,cury)==BORDER_COLOR)
                    xtemp++;
            } // 下方扫描线结束

            //上方相邻扫描线
            xtemp=xleft; cury-=2;
            // 循环一次，把一个线段种子放入堆栈（一条扫描线中可能多个线段）
            while(xtemp<=xright && cury>=0 && cury<imgcud.imgMeta.height){
                spanfill=0;
                // 找到一个线段的最右点
                while(
                    xtemp<=xright && 
                    CUDA_PIXEL_SHR(xtemp,cury)!=BORDER_COLOR 
                    ){
                        spanfill=1;
                        xtemp++;
                }
                // 最右点入栈
                if(spanfill==1){
                    cudastack[stackptr++]=xtemp-1;
                    cudastack[stackptr++]=cury;
                }
                // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
                while(CUDA_PIXEL_SHR(xtemp,cury)==BORDER_COLOR
                    && xtemp<=xright)
                    xtemp++;
            } // 上方扫描线结束

        }// 填充结束

        // 全部线程填充结束后，图像拷贝回全局内存。
        __syncthreads(); 
        // 计算填充主题时间间隔

        // 从（nx,ny）开始的 stride 个像素点从全局内存中复制到共享内存中。
        // 注意：全局内存总的pitchBytes
        if (beginIdx<imgarraysize)
            for(int i=0;i<stride;i++){
                // 末尾可能越界的像素点跳过
                if(ny*w+nx+i>=imgarraysize) break;
                imgcud.imgMeta.imgData[ny*imgcud.pitchBytes+nx+i]=shareImg[ny*w+nx+i];
            }
            return ;
}

// Kernel 函数：_intersectionKer（求两幅图像交,结果放入outbordercud中）
static __global__ void _intersectionKer(
    ImageCuda outborderCud,        // 外轮廓被填充过后的图像
    ImageCuda inborderCud          // 内轮廓被填充过后的图像
){
    // 此版本中，填充色就是轮廓色，因此，逻辑判定简单
    // 计算线程对应的输出点的位置的 x 和 y 分量
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index=y*outborderCud.pitchBytes+x;
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if(x>=outborderCud.imgMeta.width || y >= outborderCud.imgMeta.height )
        return;


    // 内边界填充图填充部分 且 外边界填充图未填充部分 是要求的结果，其余部分
    // 认为是背景
    if(outborderCud.imgMeta.imgData[index] != BORDER_COLOR && 
       inborderCud.imgMeta.imgData[index] == BORDER_COLOR)
       outborderCud.imgMeta.imgData[index]=BORDER_COLOR;
    else
       outborderCud.imgMeta.imgData[index]=BK_COLOR;

    return;
}
// Kernel 函数：_negateKer（对输入图像求反 BORDER_COLOR<-->BK_COLOR）
static __global__ void _negateKer(
    ImageCuda outborderCud        // 外轮廓被填充过的图形，反转后是轮廓内部填充
){
    // 计算线程对应的输出点的位置的 x 和 y 分量
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index=y*outborderCud.pitchBytes+x;
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if(x>=outborderCud.imgMeta.width || y >= outborderCud.imgMeta.height )
        return;

    // BORDER_COLOR 变成 BK_COLOR ，或者相反
    if(outborderCud.imgMeta.imgData[index] == BORDER_COLOR)
       outborderCud.imgMeta.imgData[index]=BK_COLOR;
    else
       outborderCud.imgMeta.imgData[index]=BORDER_COLOR;

    return;
}
//--------------------------全局方法声明------------------------------------
// 函数：_findMinMaxCoordinates(根据输入点集的坐标，找到最上、最下、最左、最右
// 的点，从而确定图像的宽和高)
static __host__ int _findMinMaxCoordinates(CoordiSet *guidingset, 
                                           int *xmin, int *ymin,
                                           int *xmax, int *ymax);

//--------------------------全局方法实现------------------------------------
// 函数：_findMinMaxCoordinates(根据输入点集的坐标，找到最上、最下、最左、最右
// 的点，从而确定图像的宽和高)
static __host__ int _findMinMaxCoordinates(CoordiSet *guidingset, 
                                           int *xmin, int *ymin,
                                           int *xmax, int *ymax)
{
    // 声明局部变量。
    int errcode;

    // 在 host 端申请一个新的 CoordiSet 变量。
    CoordiSet *tmpcoordiset;
    errcode = CoordiSetBasicOp::newCoordiSet(&tmpcoordiset);
    if (errcode != NO_ERROR) 
        return errcode;
    
    errcode = CoordiSetBasicOp::makeAtHost(tmpcoordiset, guidingset->count);
    if (errcode != NO_ERROR) 
        return errcode;
    
    // 将坐标集拷贝到 Host 端。
    errcode = CoordiSetBasicOp::copyToHost(guidingset, tmpcoordiset);
    if (errcode != NO_ERROR) 
        return errcode;

    // 初始化 x 和 y 方向上的最小最大值。
    xmin[0] = xmax[0] = tmpcoordiset->tplData[0];
    ymin[0] = ymax[0] = tmpcoordiset->tplData[1]; 
    // 循环寻找坐标集最左、最右、最上、最下的坐标。   
    for (int i = 1;i < tmpcoordiset->count;i++) {
        //　寻找 x 方向上的最小值。
        if (xmin[0] > tmpcoordiset->tplData[2 * i])
            xmin[0] = tmpcoordiset->tplData[2 * i];
        //　寻找 x 方向上的最大值    
        if (xmax[0] < tmpcoordiset->tplData[2 * i])
            xmax[0] = tmpcoordiset->tplData[2 * i];
            
        //　寻找 y 方向上的最小值。
        if (ymin[0] > tmpcoordiset->tplData[2 * i + 1])
            ymin[0] = tmpcoordiset->tplData[2 * i + 1];
        //　寻找 y 方向上的最大值
        if (ymax[0] < tmpcoordiset->tplData[2 * i + 1])
            ymax[0] = tmpcoordiset->tplData[2 * i + 1];
    }
    
    // 释放临时坐标集变量。
    CoordiSetBasicOp::deleteCoordiSet(tmpcoordiset);
    
    return errcode;
}

//--------------------------成员方法实现------------------------------------


    // 成员方法：seedScanLineCoorGlo（并行种子扫描线算法填充 coordiset 集合围起的区域）
__host__ int                // 返回值：函数是否正确执行，若函数正确执
    // 行，返回 NO_ERROR。
    FillCurve::seedScanLineCoorGlo(
    CoordiSet *outbordercoor,          // 输入的 coordiset ，内容为封闭区域
                                       // 外轮廓闭合曲线
    CoordiSet *inbordercoor,           // 输入的 coordiset ，内容为封闭区域
                                       // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
    CoordiSet *fillcoor                // 输出填充过的的 coordiset 
    ){
    // 获取坐标集中点的分布范围，即包围盒坐标
    int minx,maxx,miny,maxy;
    // ----------------------输入coor参数转化成img----------------------------
    Image *outborderimg=NULL;
    ImageBasicOp::newImage(&outborderimg);
    Image *inborderimg=NULL;
    ImageBasicOp::newImage(&inborderimg);

    ImgConvert imgcvt(BORDER_COLOR,BK_COLOR);

    #ifdef DEBUG_TIME
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float runTime;
        cudaEventRecord(start, 0);
    #endif
    // --------------------------轮廓坐标集转换成img-------------------------------
    if(outbordercoor!=NULL){
        // 预处理，得到外轮廓大小
        int errorcode=_findMinMaxCoordinates(outbordercoor,&minx,&miny,&maxx,&maxy);
        if(errorcode!=NO_ERROR)
            return 0;
        // 处理外轮廓
        // 创建工作图像
    
        //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1，保证轮廓外连通
        ImageBasicOp::makeAtHost(outborderimg,maxx+2 ,maxy+2);
        // 把坐标集绘制到图像上,前景255，背景0
        imgcvt.cstConvertToImg(outbordercoor,outborderimg);
        if(inbordercoor!=NULL){
            //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1，保证轮廓外连通
            ImageBasicOp::makeAtHost(inborderimg,maxx+2 ,maxy+2);
            // 把坐标集绘制到图像上,前景255，背景0
            imgcvt.cstConvertToImg(inbordercoor,inborderimg);
        }
        #ifdef DEBUG_IMG
            // 把填充前的图像保存到文件
            ImageBasicOp::copyToHost(outborderimg);
            ImageBasicOp::writeToFile("outborder_notFilled.bmp",outborderimg);
            ImageBasicOp::copyToHost(inborderimg);
            ImageBasicOp::writeToFile("inborder_notFilled.bmp",inborderimg);
         #endif
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[coor] coor->img time" << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
         // --------------------------调用图像填充算法-------------------------------
        seedScanLineImgGlo(outborderimg,inborderimg);


    }// end of out border
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[coor] fill time" << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif



    //------------------------串行图像转化成coor，返回-------------------------
    ImageBasicOp::copyToHost(outborderimg);
    #ifdef DEBUG_IMG
        // 最终图像输出到文件
        ImageBasicOp::writeToFile("[coor]intersection.bmp",outborderimg);
    #endif
    // 此时imgcvt的设置是前景255,背景0，灰色部分会忽略,故自定义串行转化方法
    //imgcvt.imgConvertToCst(outborderimg,fillcoor);
    int w,h;
    w=outborderimg->width;
    h=outborderimg->height;
    int imgsize=w*h;
    // 每个点（x，y）占用两个整数存放
    int *coorarray=(int *)malloc(2*imgsize*sizeof(int));
    int coorcount=0;
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++){
            // 图像中的点（i，j）
            int curpix=outborderimg->imgData[j*w+i];
            if(curpix==BORDER_COLOR ){
                coorarray[coorcount*2]=i;
                coorarray[coorcount*2+1]=j;
                coorcount++;
            }
        }

    // 创建coor，给count、和数据数组赋值
    CoordiSetBasicOp::makeAtHost(fillcoor,coorcount);
    memcpy(fillcoor->tplData,coorarray,coorcount*2*sizeof(int));
    free(coorarray);

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[coor] img to coor " << runTime << " ms" << endl;
    #endif

    /*
    //------------------------并行图像转化成coor，返回-------------------------
    // 经过测试，效率不如串行，故不采用
    imgcvt.imgConvertToCst(outborderimg,fillcoor);
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "con img to coor " << runTime << " ms" << endl;
    #endif
    */
//------------------------内存回收-------------------------------------
    ImageBasicOp::deleteImage(outborderimg);
    ImageBasicOp::deleteImage(inborderimg);
    return NO_ERROR;
}

    // 成员方法：seedScanLineImgGlo（并行种子扫描线算法填充 coordiset 集合围起的区域）
__host__ int                // 返回值：函数是否正确执行，若函数正确执
    // 行，返回 NO_ERROR。
    FillCurve::seedScanLineImgGlo(
    Image *outborderimg,          // 外轮廓闭合曲线图像,同时也是输出结果
    Image *inborderimg            // 内轮廓闭合曲线图像，没有内轮廓设为空
    ){
     ImageCuda outborderCud;
     ImageCuda inborderCud;

     #ifdef DEBUG_TIME
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float runTime;
        cudaEventRecord(start, 0);
     #endif
    // --------------------------处理外轮廓-------------------------------
    if(outborderimg!=NULL){
        int errcode;
        // 将输入图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(outborderimg);
        if (errcode != NO_ERROR) {
            return errcode;
        }

        // 提取输入图像的 ROI 子图像。
        errcode = ImageBasicOp::roiSubImage(outborderimg, &outborderCud);
        if (errcode != NO_ERROR) {
            return errcode;
        }

        // 计算边缘点个数，即最大线程个数
        int outmaxthreadsize=(outborderimg->width+outborderimg->height-2)<<1;
        dim3 grid,block;
        block.x=DEF_BLOCK_X;
        block.y=1;
        block.z=1;
        grid.x=(outmaxthreadsize+DEF_BLOCK_X-1)/DEF_BLOCK_X;
        grid.y=1;
        grid.z=1;
        //---------------------------------

        _seedScanLineOutConGlobalKer<<<grid,block>>>
            (outborderCud,outmaxthreadsize); 
        //---------------------------------------------

        #ifdef DEBUG_IMG
          ImageBasicOp::copyToHost(outborderimg);
          ImageBasicOp::writeToFile("outborder_Filled.bmp",outborderimg);
          // 交操作还要在 device 端使用图像,将输入图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(outborderimg);
        // 经过一次主存显存传输，ROI子图像需要重新提取
        errcode = ImageBasicOp::roiSubImage(outborderimg, &outborderCud);

        #endif
    }// end of out border
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[img] out border fill time" << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
    // --------------------------处理内轮廓-------------------------------
    if(outborderimg!=NULL && inborderimg!=NULL){

        int errcode;
        // 将输入图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(inborderimg);
        if (errcode != NO_ERROR) {
            return errcode;
        }
        // 提取输入图像的 ROI 子图像。
        errcode = ImageBasicOp::roiSubImage(inborderimg, &inborderCud);
        if (errcode != NO_ERROR) {
            return errcode;
        }
        // 计算边缘点个数，即最大线程个数
        int inmaxthreadsize=(inborderimg->width+inborderimg->height-2)<<1;
        dim3 grid,block;
        block.x=DEF_BLOCK_X;
        block.y=1;
        block.z=1;
        grid.x=(inmaxthreadsize+DEF_BLOCK_X-1)/DEF_BLOCK_X;
        grid.y=1;
        grid.z=1;
        _seedScanLineOutConGlobalKer<<<grid,block>>>(inborderCud,inmaxthreadsize);

        #ifdef DEBUG_IMG
            ImageBasicOp::copyToHost(inborderimg);
            ImageBasicOp::writeToFile("inborderFilled.bmp",inborderimg);
          // 交操作还要在 device 端使用图像,将输入图像拷贝入 Device 内存。
            errcode = ImageBasicOp::copyToCurrentDevice(inborderimg);
         // 经过一次主存显存传输，ROI子图像需要重新提取
            errcode = ImageBasicOp::roiSubImage(inborderimg, &inborderCud);
        #endif
    }// end of in border & process

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[img] in border fill time " << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
        //--------------如果有内轮廓，则内外轮廓填充图像求交----------------------
    if(outborderimg!=NULL && inborderimg!=NULL){
        dim3 gridsize,blocksize;
        // 计算调用计算局部最大值的 kernel 函数的线程块的尺寸和线程块的数量。  
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (outborderimg->width + blocksize.x - 1) / blocksize.x;
        gridsize.y = (outborderimg->height + blocksize.y - 1) / blocksize.y;

        // 调用 kernel 函数求交,结果放入outbordercud中，此时outborderCud和
        // inborderCud都在divice中，不用再次copytodevice
        _intersectionKer<<<gridsize, blocksize>>>(
            outborderCud,
            inborderCud
        );
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;  
     }
     //--------------如果没有内轮廓，则仅仅对外轮廓填充结果求反---------
     else{
        dim3 gridsize,blocksize;
        // 计算调用计算局部最大值的 kernel 函数的线程块的尺寸和线程块的数量。  
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (outborderimg->width + blocksize.x - 1) / blocksize.x;
        gridsize.y = (outborderimg->height + blocksize.y - 1) / blocksize.y;

        // 调用 kernel 函数求反
        _negateKer<<<gridsize, blocksize>>>(
            outborderCud
        );
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;  
    }

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[img ]inter or negate " << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
    #ifdef DEBUG_IMG
        ImageBasicOp::copyToHost(outborderimg);
        ImageBasicOp::writeToFile("[img]fill_result.bmp",outborderimg);
    #endif
    ImageBasicOp::copyToHost(outborderimg);
    return NO_ERROR;
}



// 成员方法：seedScanLineCurveGlo（并行种子扫描线算法填充 Curve 集合围起的区域）
// 使用本并行算法时，内外轮廓要放入不同的 Curve 中。
__host__ int                    // 返回值：函数是否正确执行，若函数正确执
    // 行，返回 NO_ERROR。
    FillCurve::seedScanLineCurveGlo(
    Curve *outbordercurve,          // 输入的 Curve ，内容为封闭区域
                                           // 外轮廓闭合曲线
    Curve *inbordercurve,           // 输入的 Curve ，内容为封闭区域
                                           // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
    Curve *fillcurve                // 输出填充过的的 Curve 
    ){
 
    // ----------------------输入Curve参数转化成img----------------------------
    Image *outborderimg=NULL;    
    Image *inborderimg=NULL;

    CurveConverter curcvt(BORDER_COLOR,BK_COLOR);

    #ifdef DEBUG_TIME
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float runTime;
        cudaEventRecord(start, 0);
    #endif
    // --------------------------轮廓坐标集转换成img-------------------------------
    if(outbordercurve==NULL)
        return INVALID_DATA;

        ImageBasicOp::newImage(&outborderimg);
        // 创建工作图像    
        //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1，保证轮廓外连通
        ImageBasicOp::makeAtHost(outborderimg,outbordercurve->maxCordiX+2 ,outbordercurve->maxCordiY+2);
        // 把坐标集绘制到图像上,前景255，背景0
        curcvt.curve2Img(outbordercurve,outborderimg);
        if(inbordercurve!=NULL){
            ImageBasicOp::newImage(&inborderimg);
            //给工作图像分配空间,按照外轮廓大小分配
            ImageBasicOp::makeAtHost(inborderimg,outbordercurve->maxCordiX+2 ,outbordercurve->maxCordiY+2);
            // 把坐标集绘制到图像上,前景255，背景0
            curcvt.curve2Img(inbordercurve,inborderimg);
        }
        #ifdef DEBUG_IMG
            // 把填充前的图像保存到文件
            ImageBasicOp::copyToHost(outborderimg);
            ImageBasicOp::writeToFile("outborder_notFilled.bmp",outborderimg);
            ImageBasicOp::copyToHost(inborderimg);
            ImageBasicOp::writeToFile("inborder_notFilled.bmp",inborderimg);
         #endif
        #ifdef DEBUG_TIME
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&runTime, start, stop);
            cout << "[Curve] coor->img time" << runTime << " ms" << endl;
            cudaEventRecord(start, 0);
        #endif
         // --------------------------调用图像填充算法-------------------------------
        seedScanLineImgGlo(outborderimg,inborderimg);

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[Curve] fill time" << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif

    //------------------------图像转化成curve，返回-------------------------


        cudaThreadSynchronize();
    curcvt.img2Curve(outborderimg,fillcurve);

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[Curve] img to Curve " << runTime << " ms" << endl;
    #endif

//------------------------内存回收-------------------------------------
    ImageBasicOp::deleteImage(outborderimg);
    ImageBasicOp::deleteImage(inborderimg);

    return NO_ERROR;
}



    // 成员方法：seedScanLineCoorShr（并行种子扫描线算法填充 coordiset 集合围起的区域）
__host__ int                // 返回值：函数是否正确执行，若函数正确执
    // 行，返回 NO_ERROR。
    FillCurve::seedScanLineCoorShr(
    CoordiSet *outbordercoor,          // 输入的 coordiset ，内容为封闭区域
                                       // 外轮廓闭合曲线
    CoordiSet *inbordercoor,           // 输入的 coordiset ，内容为封闭区域
                                       // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
    CoordiSet *fillcoor                // 输出填充过的的 coordiset 
    ){
    // 获取坐标集中点的分布范围，即包围盒坐标
    int minx,maxx,miny,maxy;
    // ----------------------输入coor参数转化成img----------------------------
    Image *outborderimg=NULL;
    ImageBasicOp::newImage(&outborderimg);
    Image *inborderimg=NULL;
    ImageBasicOp::newImage(&inborderimg);

    ImgConvert imgcvt(BORDER_COLOR,BK_COLOR);

    #ifdef DEBUG_TIME
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float runTime;
        cudaEventRecord(start, 0);
    #endif
    // --------------------------轮廓坐标集转换成img-------------------------------
    if(outbordercoor!=NULL){
        // 预处理，得到外轮廓大小
        int errorcode=_findMinMaxCoordinates(outbordercoor,&minx,&miny,&maxx,&maxy);
        if(errorcode!=NO_ERROR)
            return 0;
        // 处理外轮廓
        // 创建工作图像
    
        //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1，保证轮廓外连通
        ImageBasicOp::makeAtHost(outborderimg,maxx+2 ,maxy+2);
        // 把坐标集绘制到图像上,前景255，背景0
        imgcvt.cstConvertToImg(outbordercoor,outborderimg);
        if(inbordercoor!=NULL){
            //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1，保证轮廓外连通
            ImageBasicOp::makeAtHost(inborderimg,maxx+2 ,maxy+2);
            // 把坐标集绘制到图像上,前景255，背景0
            imgcvt.cstConvertToImg(inbordercoor,inborderimg);
        }
        #ifdef DEBUG_IMG
            // 把填充前的图像保存到文件
            ImageBasicOp::copyToHost(outborderimg);
            ImageBasicOp::writeToFile("outborder_notFilled.bmp",outborderimg);
            ImageBasicOp::copyToHost(inborderimg);
            ImageBasicOp::writeToFile("inborder_notFilled.bmp",inborderimg);
         #endif
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[coor] coor->img time" << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
         // --------------------------调用图像填充算法-------------------------------
        seedScanLineImgShr(outborderimg,inborderimg);


    }// end of out border
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[coor] fill time" << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif

    //------------------------串行图像转化成coor，返回-------------------------
    ImageBasicOp::copyToHost(outborderimg);
    #ifdef DEBUG_IMG
        // 最终图像输出到文件
        ImageBasicOp::writeToFile("[coor]intersection.bmp",outborderimg);
    #endif
    // 此时imgcvt的设置是前景255,背景0，灰色部分会忽略,故自定义串行转化方法
    //imgcvt.imgConvertToCst(outborderimg,fillcoor);
    int w,h;
    w=outborderimg->width;
    h=outborderimg->height;
    int imgsize=w*h;
    // 每个点（x，y）占用两个整数存放
    int *coorarray=(int *)malloc(2*imgsize*sizeof(int));
    int coorcount=0;
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++){
            // 图像中的点（i，j）
            int curpix=outborderimg->imgData[j*w+i];
            if(curpix==BORDER_COLOR ){
                coorarray[coorcount*2]=i;
                coorarray[coorcount*2+1]=j;
                coorcount++;
            }
        }

    // 创建coor，给count、和数据数组赋值
    CoordiSetBasicOp::makeAtHost(fillcoor,coorcount);
    memcpy(fillcoor->tplData,coorarray,coorcount*2*sizeof(int));
    free(coorarray);

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[coor] img to coor " << runTime << " ms" << endl;
    #endif

    /*
    //------------------------并行图像转化成coor，返回-------------------------
    // 经过测试，效率不如串行，故不采用
    imgcvt.imgConvertToCst(outborderimg,fillcoor);
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "con img to coor " << runTime << " ms" << endl;
    #endif
    */
//------------------------内存回收-------------------------------------
    ImageBasicOp::deleteImage(outborderimg);
    ImageBasicOp::deleteImage(inborderimg);
    return NO_ERROR;
}

    // 成员方法：seedScanLineImgShr（并行种子扫描线算法填充 coordiset 集合围起的区域）
__host__ int                // 返回值：函数是否正确执行，若函数正确执
    // 行，返回 NO_ERROR。
    FillCurve::seedScanLineImgShr(
    Image *outborderimg,          // 外轮廓闭合曲线图像,同时也是输出结果
    Image *inborderimg            // 内轮廓闭合曲线图像，没有内轮廓设为空
    ){
     ImageCuda outborderCud;
     ImageCuda inborderCud;

     #ifdef DEBUG_TIME
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float runTime;
        cudaEventRecord(start, 0);
     #endif
    // --------------------------处理外轮廓-------------------------------
    if(outborderimg!=NULL){
        int errcode;
        // 将输入图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(outborderimg);
        if (errcode != NO_ERROR) {
            return errcode;
        }

        // 提取输入图像的 ROI 子图像。
        errcode = ImageBasicOp::roiSubImage(outborderimg, &outborderCud);
        if (errcode != NO_ERROR) {
            return errcode;
        }

        // 计算边缘点个数，即最大线程个数
        int outmaxthreadsize=(outborderimg->width+outborderimg->height-2)<<1;
        if(outmaxthreadsize>maxThreadsPerBlock) 
            outmaxthreadsize=maxThreadsPerBlock;
        dim3 grid,block;
        block.x=outmaxthreadsize;
        block.y=1;
        block.z=1;
        grid.x=1;
        grid.y=1;
        grid.z=1;
        //---------------------------------
        int sharedmemsize=outborderCud.imgMeta.height*
                          outborderCud.imgMeta.width*
                          sizeof (unsigned char);

        _seedScanLineOutConShareKer<<<grid,block,sharedmemsize>>>
            (outborderCud,
            outmaxthreadsize
        );
        //---------------------------------------------

        #ifdef DEBUG_IMG
          ImageBasicOp::copyToHost(outborderimg);
          ImageBasicOp::writeToFile("outborder_Filled.bmp",outborderimg);
          // 交操作还要在 device 端使用图像,将输入图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(outborderimg);
        // 经过一次主存显存传输，ROI子图像需要重新提取
        errcode = ImageBasicOp::roiSubImage(outborderimg, &outborderCud);

        #endif
    }// end of out border
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[img] out border fill time" << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
    // --------------------------处理内轮廓-------------------------------
    if(outborderimg!=NULL && inborderimg!=NULL){

        int errcode;
        // 将输入图像拷贝入 Device 内存。
        errcode = ImageBasicOp::copyToCurrentDevice(inborderimg);
        if (errcode != NO_ERROR) {
            return errcode;
        }
        // 提取输入图像的 ROI 子图像。
        errcode = ImageBasicOp::roiSubImage(inborderimg, &inborderCud);
        if (errcode != NO_ERROR) {
            return errcode;
        }
        // 计算边缘点个数，即最大线程个数
        int inmaxthreadsize=(inborderimg->width+inborderimg->height-2)<<1;
        if(inmaxthreadsize>maxThreadsPerBlock) 
            inmaxthreadsize=maxThreadsPerBlock;
        dim3 grid,block;
        block.x=inmaxthreadsize;
        block.y=1;
        block.z=1;
        grid.x=1;
        grid.y=1;
        grid.z=1;

        int insharedmemsize=inborderCud.imgMeta.width*
                            inborderCud.imgMeta.height*
                            sizeof (unsigned char);

        _seedScanLineOutConShareKer<<<grid,block,insharedmemsize>>>(inborderCud,inmaxthreadsize);

        #ifdef DEBUG_IMG
            ImageBasicOp::copyToHost(inborderimg);
            ImageBasicOp::writeToFile("inborderFilled.bmp",inborderimg);
          // 交操作还要在 device 端使用图像,将输入图像拷贝入 Device 内存。
            errcode = ImageBasicOp::copyToCurrentDevice(inborderimg);
         // 经过一次主存显存传输，ROI子图像需要重新提取
            errcode = ImageBasicOp::roiSubImage(inborderimg, &inborderCud);
        #endif
    }// end of in border & process

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[img] in border fill time " << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
        //--------------如果有内轮廓，则内外轮廓填充图像求交----------------------
    if(outborderimg!=NULL && inborderimg!=NULL){
        dim3 gridsize,blocksize;
        // 计算调用计算局部最大值的 kernel 函数的线程块的尺寸和线程块的数量。  
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (outborderimg->width + blocksize.x - 1) / blocksize.x;
        gridsize.y = (outborderimg->height + blocksize.y - 1) / blocksize.y;

        // 调用 kernel 函数求交,结果放入outbordercud中，此时outborderCud和
        // inborderCud都在divice中，不用再次copytodevice
        _intersectionKer<<<gridsize, blocksize>>>(
            outborderCud,
            inborderCud
        );
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;  
     }
     //--------------如果没有内轮廓，则仅仅对外轮廓填充结果求反---------
     else{
        dim3 gridsize,blocksize;
        // 计算调用计算局部最大值的 kernel 函数的线程块的尺寸和线程块的数量。  
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (outborderimg->width + blocksize.x - 1) / blocksize.x;
        gridsize.y = (outborderimg->height + blocksize.y - 1) / blocksize.y;

        // 调用 kernel 函数求反
        _negateKer<<<gridsize, blocksize>>>(
            outborderCud
        );
        if (cudaGetLastError() != cudaSuccess)
            return CUDA_ERROR;  
    }

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[img ]inter or negate " << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
    #ifdef DEBUG_IMG
        ImageBasicOp::copyToHost(outborderimg);
        ImageBasicOp::writeToFile("[img]fill_result.bmp",outborderimg);
    #endif
    ImageBasicOp::copyToHost(outborderimg);
    return NO_ERROR;
}



// 成员方法：seedScanLineCurveShr（并行种子扫描线算法填充 Curve 集合围起的区域）
// 使用本并行算法时，内外轮廓要放入不同的 Curve 中。
__host__ int                    // 返回值：函数是否正确执行，若函数正确执
    // 行，返回 NO_ERROR。
    FillCurve::seedScanLineCurveShr(
    Curve *outbordercurve,          // 输入的 Curve ，内容为封闭区域
                                           // 外轮廓闭合曲线
    Curve *inbordercurve,           // 输入的 Curve ，内容为封闭区域
                                           // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
    Curve *fillcurve                // 输出填充过的的 Curve 
    ){

        // ----------------------输入Curve参数转化成img----------------------------
        Image *outborderimg=NULL;    
        Image *inborderimg=NULL;

        CurveConverter curcvt(BORDER_COLOR,BK_COLOR);

#ifdef DEBUG_TIME
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float runTime;
        cudaEventRecord(start, 0);
#endif
        // --------------------------轮廓坐标集转换成img-------------------------------
        if(outbordercurve==NULL)
            return INVALID_DATA;

        ImageBasicOp::newImage(&outborderimg);
        // 创建工作图像    
        //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1，保证轮廓外连通
        ImageBasicOp::makeAtHost(outborderimg,outbordercurve->maxCordiX+2 ,outbordercurve->maxCordiY+2);
        // 把坐标集绘制到图像上,前景255，背景0
        curcvt.curve2Img(outbordercurve,outborderimg);
        if(inbordercurve!=NULL){
            ImageBasicOp::newImage(&inborderimg);
            //给工作图像分配空间,按照外轮廓大小分配
            ImageBasicOp::makeAtHost(inborderimg,outbordercurve->maxCordiX+2 ,outbordercurve->maxCordiY+2);
            // 把坐标集绘制到图像上,前景255，背景0
            curcvt.curve2Img(inbordercurve,inborderimg);
        }
#ifdef DEBUG_IMG
        // 把填充前的图像保存到文件
        ImageBasicOp::copyToHost(outborderimg);
        ImageBasicOp::writeToFile("outborder_notFilled.bmp",outborderimg);
        ImageBasicOp::copyToHost(inborderimg);
        ImageBasicOp::writeToFile("inborder_notFilled.bmp",inborderimg);
#endif
#ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[Curve] coor->img time" << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
#endif
        // --------------------------调用图像填充算法-------------------------------
        seedScanLineImgGlo(outborderimg,inborderimg);

#ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[Curve] fill time" << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
#endif

        //------------------------图像转化成curve，返回-------------------------


        cudaThreadSynchronize();
        curcvt.img2Curve(outborderimg,fillcurve);

#ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "[Curve] img to Curve " << runTime << " ms" << endl;
#endif

        //------------------------内存回收-------------------------------------
        ImageBasicOp::deleteImage(outborderimg);
        ImageBasicOp::deleteImage(inborderimg);

        return NO_ERROR;
}


