#include "FillCoor.h"
#include "ImageDrawer.h"
#include "ImgConvert.h" 
#include <iostream>
#include <stack>
using namespace std;
// 宏：DEBUG
// 定义是否输出调试信息
//#define DEBUG_IMG
//#define DEBUG_TIME

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X 32 
#define DEF_BLOCK_Y 8
// 宏：PIXEL(x,y) 
// 获取图像中（x，y）像素的位置
#define PIXEL(x,y) workimg->imgData[(y)*workimg->width+(x)]
// 宏：VALID(x,y) 
// 判断（x，y）像素的位置是否合法
#define VALID(x,y) (x>=0 && x<workimg->width && y>=0 && y<workimg->height)

// 宏：CUDA_PIXEL(x,y) 
// 获取内核函数中图像中（x，y）像素的位置
#define CUDA_PIXEL(x,y) imgcud.imgMeta.imgData[(y)*imgcud.pitchBytes+(x)]
// 宏：CUDA_VALID(x,y) 
// 判断内核函数中（x，y）像素的位置是否合法
#define CUDA_VALID(x,y) (x>=0 && x<imgcud.imgMeta.width && y>=0 && y<imgcud.imgMeta.height)

// 宏：CUDA_STACK_SIZE 
// 自定义的cuda栈最大容量,根据测试，不太复杂的图像，最大深度为4，因此最大值定义64足够
#define CUDA_STACK_SIZE 64

// 结构体：mypoint
// 记录像素点的位置
typedef struct mypoint{
    int x;
    int y;
}point;
//--------------------------内核方法声明------------------------------------
/*
// Kernel 函数：_seedScanLineConKer（并行的种子扫描线算法,种子在轮廓内部）
static __global__ void _seedScanLineInConKer(
    ImageCuda imgcud,               // 要填充的轮廓图像
    int * stackmaxsize              // 返回自定义堆栈最大使用深度
);*/
// Kernel 函数：_seedScanLineOutConKer（并行的种子扫描线算法，种子在轮廓外部）
static __global__ void _seedScanLineOutConKer(
    ImageCuda imgcud              // 要填充的轮廓图像
    //int * stackmaxsiz              // 返回自定义堆栈最大使用深度
);
// Kernel 函数：_intersectionKer（求两幅图像交,结果放入outbordercud中）
static __global__ void _intersectionKer(
    ImageCuda outborderCud,        // 外轮廓被填充过后的图像
    ImageCuda inborderCud          // 内轮廓被填充过后的图像
);
//--------------------------内核方法实现------------------------------------
/*
// Kernel 函数：_seedScanLineInConKer（并行的种子扫描线算法,种子在轮廓内）
static __global__ void _seedScanLineInConKer(
    ImageCuda imgcud,               // 要填充的轮廓图像
    int * stackmaxsize              // 返回自定义堆栈最大使用深度
){
    // 计算线程对应的输出点的位置的 x 和 y 分量
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 检查像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if(x>=imgcud.imgMeta.width || y >= imgcud.imgMeta.height )
        return;
    int cudastack[CUDA_STACK_SIZE];
    // 填充工作，
    // 输入：轮廓线workimg,种子seed；
    // 输出：填充过的workimg

    int stackptr=0;
    point seed;
    seed.x=40+x*10;
    seed.y=40+y*10;

    if(seed.x>=imgcud.imgMeta.width || seed.y >= imgcud.imgMeta.height )
        return;


    int xtemp,xright,xleft;
    int spanfill;
    // 种子入栈
    cudastack[stackptr++]=seed.x;
    cudastack[stackptr++]=seed.y;
    // stackptr==0表示栈为空，>0说明栈不空，每个像素点占用2个位置
    while(stackptr>0){
        point cur;
        // 统计堆栈最大深度
        if(stackptr>stackmaxsize[0])
            stackmaxsize[0]=stackptr;
        // 入栈顺序x、y，出栈顺序应y、x。

        cur.y=cudastack[--stackptr];
        cur.x=cudastack[--stackptr];
        // 填充当前点
        CUDA_PIXEL(cur.x,cur.y)=BORDER_COLOR;
        // 向右填充,填充过程中检测当前点坐标，如果越界，说明种子在图形外
        for(xtemp=cur.x+1;CUDA_PIXEL(xtemp,cur.y)!=BORDER_COLOR;xtemp++){
            if(CUDA_VALID(xtemp,cur.y)==false) return ;
            CUDA_PIXEL(xtemp,cur.y)=BORDER_COLOR;}
        //纪录当前线段最右位置
        xright=xtemp-1;
        // 向左填充
        for(xtemp=cur.x-1;CUDA_PIXEL(xtemp,cur.y)!=BORDER_COLOR;xtemp--){
            if(CUDA_VALID(xtemp,cur.y)==false) return ;
            CUDA_PIXEL(xtemp,cur.y)=BORDER_COLOR;
            }
        // 纪录当前线段最左位置
        xleft=xtemp+1;

        //cout<<"hang:"<<cur.y<<"["<<xleft<<","<<xright<<"]"<<endl;
        // 下方相邻扫描线
        xtemp=xleft; cur.y++;
        // 循环一次，把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright){
            spanfill=0;
            // 找到一个线段的最右点
            while(CUDA_PIXEL(xtemp,cur.y)!=BORDER_COLOR && 
                  xtemp<=xright){
            spanfill=1;
            xtemp++;
            }
            // 最右点(xtemp-1,cur.y)入栈
            if(spanfill==1){
                cudastack[stackptr++]=xtemp-1;
                cudastack[stackptr++]=cur.y;
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while((CUDA_PIXEL(xtemp,cur.y)==BORDER_COLOR 
                   && xtemp<=xright)
                 xtemp++;
        } // 下方扫描线结束

        //上方相邻扫描线
        xtemp=xleft; cur.y-=2;
        // 循环一次，把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright){
            spanfill=0;
            // 找到一个线段的最右点
            while(CUDA_PIXEL(xtemp,cur.y)!=BORDER_COLOR && 
                  xtemp<=xright){
            spanfill=1;
            xtemp++;
            }
            // 最右点入栈
            if(spanfill==1){
                cudastack[stackptr++]=xtemp-1;
                cudastack[stackptr++]=cur.y;
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while(CUDA_PIXEL(xtemp,cur.y)==BORDER_COLOR || 
                   && xtemp<=xright)
                 xtemp++;
        } // 上方扫描线结束
    }// 填充结束
    return ;
}*/

// Kernel 函数：_seedScanLineOutConKer（并行的种子扫描线算法，种子在轮廓外部）
static __global__ void _seedScanLineOutConKer(
    ImageCuda imgcud               // 要填充的轮廓图像
    //int * stackmaxsize              // 返回自定义堆栈最大使用深度
 ){
    // 计算线程对应的输出点的位置的 x 和 y 分量
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    point seed;
    switch(y){
        // 第一行当种子
        case 0:
            seed.x=x;
            seed.y=0;
            break;
        // 最后一行当种子
        case 1:
            seed.x=x;
            seed.y=imgcud.imgMeta.height-1;
            break;
        // 第一列当种子
        case 2:
            seed.x=0;
            seed.y=x;
            break;
        // 最后一列当种子
        case 3:
            seed.x=imgcud.imgMeta.width-1;
            seed.y=x;
            break;
    }
    // 如果得到的种子超过图像范围，或者不是背景点(可能是轮廓点或者已经被其他线程
    // 填充，则直接退出)
    if(seed.x>=imgcud.imgMeta.width || 
       seed.y >= imgcud.imgMeta.height || 
       CUDA_PIXEL(seed.x,seed.y) != BK_COLOR)
        return;

    // 填充工作
    // 输入：轮廓线workimg,种子seed；
    // 输出：填充过的workimg
    int cudastack[CUDA_STACK_SIZE];
    int stackptr=0;
    int xtemp,xright,xleft;
    int spanfill;
    // 种子入栈
    cudastack[stackptr++]=seed.x;
    cudastack[stackptr++]=seed.y;
    // stackptr==0表示栈为空，>0说明栈不空，每个像素点占用2个位置
    while(stackptr>0){
        point cur;
        // 统计堆栈最大深度
        //if(stackptr>stackmaxsize[0])
            //stackmaxsize[0]=stackptr;
        // 入栈顺序x、y，出栈顺序应y、x。

        cur.y=cudastack[--stackptr];
        cur.x=cudastack[--stackptr];
        // 填充当前点
        CUDA_PIXEL(cur.x,cur.y)=BORDER_COLOR;
        // 向右填充,填充过程中检测当前点坐标
        for(xtemp=cur.x+1;CUDA_VALID(xtemp,cur.y)&&CUDA_PIXEL(xtemp,cur.y)!=BORDER_COLOR;xtemp++){
            CUDA_PIXEL(xtemp,cur.y)=BORDER_COLOR;}
        //纪录当前线段最右位置
        xright=xtemp-1;
        // 向左填充
        for(xtemp=cur.x-1;CUDA_VALID(xtemp,cur.y)&&CUDA_PIXEL(xtemp,cur.y)!=BORDER_COLOR;xtemp--){
            CUDA_PIXEL(xtemp,cur.y)=BORDER_COLOR;
            }
        // 纪录当前线段最左位置
        xleft=xtemp+1;

        //cout<<"hang:"<<cur.y<<"["<<xleft<<","<<xright<<"]"<<endl;

        // 下方相邻扫描线
        xtemp=xleft; cur.y++;
        // 每次循环把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright && cur.y>=0 && cur.y<imgcud.imgMeta.height){
            spanfill=0;
            // 找到一个线段的最右点
            while(
                CUDA_PIXEL(xtemp,cur.y)!=BORDER_COLOR &&
                xtemp<=xright){
            spanfill=1;
            xtemp++;
            }
            // 最右点(xtemp-1,cur.y)入栈
            if(spanfill==1){
                cudastack[stackptr++]=xtemp-1;
                cudastack[stackptr++]=cur.y;
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while(
                xtemp<=xright && 
                CUDA_PIXEL(xtemp,cur.y)==BORDER_COLOR)
                 xtemp++;
        } // 下方扫描线结束

        //上方相邻扫描线
        xtemp=xleft; cur.y-=2;
        // 循环一次，把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright && cur.y>=0 && cur.y<imgcud.imgMeta.height){
            spanfill=0;
            // 找到一个线段的最右点
            while(
                xtemp<=xright && 
                CUDA_PIXEL(xtemp,cur.y)!=BORDER_COLOR 
            ){
            spanfill=1;
            xtemp++;
            }
            // 最右点入栈
            if(spanfill==1){
                cudastack[stackptr++]=xtemp-1;
                cudastack[stackptr++]=cur.y;
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while(CUDA_PIXEL(xtemp,cur.y)==BORDER_COLOR
                   && xtemp<=xright)
                 xtemp++;
        } // 上方扫描线结束

    }// 填充结束

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
// 成员方法：fillCoordiSetSeri（串行方法，填充 coordiset 集合围起的区域）
__host__ int                    // 返回值：函数是否正确执行，若函数正确执
                                // 行，返回 NO_ERROR。
FillCoor::seedScanLineSeri(
CoordiSet *incoor,                // 输入的coordiset，内容为一条闭合曲线
CoordiSet *outcoor,               // 输出填充过的的coordiset，内容一个填充区域
int x,                            // 种子x坐标
int y                             // 种子y坐标
){

    // 获取坐标集中点的分布范围，即包围盒坐标
    int minx,maxx,miny,maxy;
    int errorcode=_findMinMaxCoordinates(incoor,&minx,&miny,&maxx,&maxy);
    if(errorcode!=NO_ERROR)
        return 0;
    // 创建工作图像
    Image *workimg;
    ImageBasicOp::newImage(&workimg);
    //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数
    ImageBasicOp::makeAtHost(workimg,maxx+1 ,maxy+1);
    // 把坐标集绘制到图像上
    ImgConvert imgcvt(BORDER_COLOR,BK_COLOR);
    imgcvt.cstConvertToImg(incoor,workimg);
    // 把填充前的图像保存到文件
    ImageBasicOp::copyToHost(workimg);
    ImageBasicOp::writeToFile("biforeFill.bmp",workimg);

//----------------------------------------------------
    // 填充工作，
    // 输入：轮廓线workimg,种子seed；
    // 输出：填充过的workimg
    int deepestnum=0;
    point seed;
    seed.x=x;
    seed.y=y;


    int xtemp,xright,xleft;
    int spanfill;
    stack <point>st;

    st.push(seed);
    int loopnum=0;
    while(!st.empty()){
        point cur;
        loopnum++;
        // 统计堆栈最大深度

        if(st.size()>deepestnum){
            deepestnum=st.size();
        }
        cur=st.top();
        st.pop();
        PIXEL(cur.x,cur.y)=BORDER_COLOR;
        // 向右填充,填充过程中检测当前点坐标，如果越界，说明种子在图形外
        for(xtemp=cur.x+1;PIXEL(xtemp,cur.y)!=BORDER_COLOR;xtemp++){
            if(VALID(xtemp,cur.y)==false) return INVALID_DATA;
            PIXEL(xtemp,cur.y)=BORDER_COLOR;}
        //纪录当前线段最右位置
        xright=xtemp-1;
        // 向左填充
        for(xtemp=cur.x-1;PIXEL(xtemp,cur.y)!=BORDER_COLOR;xtemp--){
            if(VALID(xtemp,cur.y)==false) return INVALID_DATA;
            PIXEL(xtemp,cur.y)=BORDER_COLOR;
            }
        // 纪录当前线段最左位置
        xleft=xtemp+1;

        //cout<<"hang:"<<cur.y<<"["<<xleft<<","<<xright<<"]"<<endl;
        // 下方相邻扫描线
        xtemp=xleft; cur.y++;
        // 循环一次，把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright){
            spanfill=0;
            // 找到一个线段的最右点
            while(PIXEL(xtemp,cur.y)!=BORDER_COLOR && 
                  xtemp<=xright){
            spanfill=1;
            xtemp++;
            }
            // 最右点入栈
            if(spanfill==1){
                point t;
                t.x=xtemp-1;
                t.y=cur.y;
                st.push(t);
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while(PIXEL(xtemp,cur.y)==BORDER_COLOR && 
                   xtemp<=xright)
                 xtemp++;
        } // 下方扫描线结束

        //上方相邻扫描线
        xtemp=xleft; cur.y-=2;
        // 循环一次，把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright){
            spanfill=0;
            // 找到一个线段的最右点
            while(PIXEL(xtemp,cur.y)!=BORDER_COLOR && 
                  xtemp<=xright){
            spanfill=1;
            xtemp++;
            }
            // 最右点入栈
            if(spanfill==1){
                point t;
                t.x=xtemp-1;
                t.y=cur.y;
                st.push(t);
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while(PIXEL(xtemp,cur.y)==BORDER_COLOR
                   && xtemp<=xright)
                 xtemp++;
        } // 上方扫描线结束

    }// 填充结束

    ImageBasicOp::copyToHost(workimg);
    #ifdef DEBUG_IMG
        printf("loopnum= %3d, deepestnum=%2d \n  ",loopnum,deepestnum);
        // 填充后的图像保存到文件
        ImageBasicOp::writeToFile("afterFill.bmp",workimg);
    #endif

    imgcvt.imgConvertToCst(workimg,outcoor);
    ImageBasicOp::deleteImage(workimg);
    return NO_ERROR;
}

// 成员方法：isInCoordiSetSeri（串行方法，判断当前点是否在 coordiset 集合围起的区域中）
    __host__ bool                // 返回值：在内部返回真，否则返回假
        FillCoor::isInCoordiSetSeri(
        CoordiSet *incoor,          // 输入的coordiset，内容为一条闭合曲线
        int x,                      // 坐标点x坐标
        int y                       // 坐标点y坐标
        ){
    // 获取坐标集中点的分布范围，即包围盒坐标
    int minx,maxx,miny,maxy;
    int errorcode=_findMinMaxCoordinates(incoor,&minx,&miny,&maxx,&maxy);
    if(errorcode!=NO_ERROR)
        return 0;
    // 创建工作图像
    Image *workimg;
    ImageBasicOp::newImage(&workimg);
    //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数
    ImageBasicOp::makeAtHost(workimg,maxx+1 ,maxy+1);
    // 把坐标集绘制到图像上
    ImgConvert imgcvt(BORDER_COLOR,BK_COLOR);
    imgcvt.cstConvertToImg(incoor,workimg);
    // 把填充前的图像保存到文件
    //ImageBasicOp::copyToHost(workimg);
    //ImageBasicOp::writeToFile("biforeFill.bmp",workimg);
    //----------------------------------------------------
    // 填充工作，
    // 输入：轮廓线workimg,种子seed；
    // 输出：填充过的workimg
    point seed;
    seed.x=x;
    seed.y=y;
    //ImageBasicOp::readFromFile("bordertest.bmp",workimg);
    ImageBasicOp::copyToHost(workimg);
    int xtemp,xright,xleft;
    int spanfill;
    stack <point>st;

    st.push(seed);
    while(!st.empty()){
        point cur;
        cur=st.top();
        st.pop();
        PIXEL(cur.x,cur.y)=BORDER_COLOR;
        // 向右填充,填充过程中检测当前点坐标，如果越界，说明种子在图形外
        for(xtemp=cur.x+1;PIXEL(xtemp,cur.y)!=BORDER_COLOR;xtemp++){
            if(VALID(xtemp,cur.y)==false) return false;
            PIXEL(xtemp,cur.y)=BORDER_COLOR;}
        //纪录当前线段最右位置
        xright=xtemp-1;
        // 向左填充
        for(xtemp=cur.x-1;PIXEL(xtemp,cur.y)!=BORDER_COLOR;xtemp--){
            if(VALID(xtemp,cur.y)==false) return false;
            PIXEL(xtemp,cur.y)=BORDER_COLOR;
            }
        //纪录当前线段最左位置
        xleft=xtemp+1;

        //cout<<"hang:"<<cur.y<<"["<<xleft<<","<<xright<<"]"<<endl;
        //下方相邻扫描线
        xtemp=xleft; cur.y++;
        // 循环一次，把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright){
            spanfill=0;
            // 找到一个线段的最右点
            while(PIXEL(xtemp,cur.y)!=BORDER_COLOR && 
                  xtemp<=xright){
            spanfill=1;
            xtemp++;
            }
            // 最右点入栈
            if(spanfill==1){
                point t;
                t.x=xtemp-1;
                t.y=cur.y;
                st.push(t);
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while(PIXEL(xtemp,cur.y)==BORDER_COLOR 
                   && xtemp<=xright)
                 xtemp++;
        } // 下方扫描线结束

        //上方相邻扫描线
        xtemp=xleft; cur.y-=2;
        // 循环一次，把一个线段种子放入堆栈（一条扫描线中可能多个线段）
        while(xtemp<=xright){
            spanfill=0;
            // 找到一个线段的最右点
            while(PIXEL(xtemp,cur.y)!=BORDER_COLOR && 
                  xtemp<=xright){
            spanfill=1;
            xtemp++;
            }
            // 最右点入栈
            if(spanfill==1){
                point t;
                t.x=xtemp-1;
                t.y=cur.y;
                st.push(t);
            }
            // 继续向右走，跳过边界和已经填充部分，找到下一段未填充线段
            while(PIXEL(xtemp,cur.y)==BORDER_COLOR 
                   && xtemp<=xright)
                 xtemp++;
        } // 上方扫描线结束

    }// 填充结束
    //-------------回收工作---------------------------
    //ImageBasicOp::copyToHost(workimg);
    //ImageBasicOp::writeToFile(outfile.c_str(),workimg);
    ImageBasicOp::deleteImage(workimg);
    return true;
    }


    // 成员方法：seedScanLineCon（并行种子扫描线算法填充 coordiset 集合围起的区域）
__host__ int                // 返回值：函数是否正确执行，若函数正确执
    // 行，返回 NO_ERROR。
    FillCoor::seedScanLineCon(
    CoordiSet *outbordercoor,          // 输入的 coordiset ，内容为封闭区域
                                       // 外轮廓闭合曲线
    CoordiSet *inbordercoor,           // 输入的 coordiset ，内容为封闭区域
                                       // 内轮廓闭合曲线。如果没有内轮廓，设为NULL
    CoordiSet *fillcoor                // 输出填充过的的 coordiset 
    ){
    // 获取坐标集中点的分布范围，即包围盒坐标
    int minx,maxx,miny,maxy;
    // ----------------------输入coor参数转化成img----------------------------
    Image *outborderimg;
    ImageBasicOp::newImage(&outborderimg);
    Image *inborderimg;
    ImageBasicOp::newImage(&inborderimg);
    ImageCuda outborderCud;
    ImageCuda inborderCud;
    ImgConvert imgcvt(BORDER_COLOR,BK_COLOR);

    #ifdef DEBUG_TIME
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float runTime;
        cudaEventRecord(start, 0);
    #endif
    // --------------------------处理外轮廓-------------------------------
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
        #ifdef DEBUG_IMG
            // 把填充前的图像保存到文件
            ImageBasicOp::copyToHost(outborderimg);
            ImageBasicOp::writeToFile("outborder_notFilled.bmp",outborderimg);
         #endif
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
        // 找长宽的最大值当内核函数的参数
        int outmaxsize=outborderimg->height>outborderimg->width?
                    outborderimg->height:outborderimg->width;
        dim3 grid,block;
        block.x=DEF_BLOCK_X;
        block.y=1;
        block.z=1;
        grid.x=(outmaxsize+DEF_BLOCK_X-1)/DEF_BLOCK_X;
        grid.y=4;
        grid.z=1;

        _seedScanLineOutConKer<<<grid,block>>>(outborderCud);      

        #ifdef DEBUG_IMG
          ImageBasicOp::copyToHost(outborderimg);
          ImageBasicOp::writeToFile("outborder_Filled.bmp",outborderimg);
          // 交操作还要在 device 端使用图像
          ImageBasicOp::copyToCurrentDevice(outborderimg);
        #endif
    }// end of out border
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "out border fill " << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
    // --------------------------处理内轮廓-------------------------------
    if(outbordercoor!=NULL && inbordercoor!=NULL){
        // 注意，内边界图像将来要和外边界图像求交，大小按外边界分配
        // 给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1，保证轮廓外连通
        ImageBasicOp::makeAtHost(inborderimg,maxx+2 ,maxy+2);
        // 把坐标集绘制到图像上,前景255，背景0
        imgcvt.cstConvertToImg(inbordercoor,inborderimg);
        #ifdef DEBUG_IMG
            // 把填充前的图像保存到文件
            ImageBasicOp::copyToHost(inborderimg);
            ImageBasicOp::writeToFile("inborder_notFilled.bmp",inborderimg);
        #endif
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
        // 找长宽的最大值当内核函数的参数
        int inmaxsize=inborderimg->height>inborderimg->width?
                    inborderimg->height:inborderimg->width;
        dim3 grid,block;
        block.x=DEF_BLOCK_X;
        block.y=1;
        block.z=1;
        grid.x=(inmaxsize+DEF_BLOCK_X-1)/DEF_BLOCK_X;
        grid.y=4;
        grid.z=1;
        _seedScanLineOutConKer<<<grid,block>>>(inborderCud);

        #ifdef DEBUG_IMG
            ImageBasicOp::copyToHost(inborderimg);
            ImageBasicOp::writeToFile("inborderFilled.bmp",inborderimg);
            // 交操作还要在 device 端使用图像
            ImageBasicOp::copyToCurrentDevice(inborderimg);
        #endif
    }// end of in border & process

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "in border fill " << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif
        //--------------如果有内轮廓，则内外轮廓填充图像求交----------------------
    if(outbordercoor!=NULL && inbordercoor!=NULL){
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
        cout << "inter or negate " << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif


    //------------------------串行图像转化成coor，返回-------------------------
    ImageBasicOp::copyToHost(outborderimg);
    #ifdef DEBUG_IMG
        // 最终图像输出到文件
        ImageBasicOp::writeToFile("intersection.bmp",outborderimg);
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
            int curpix=outborderimg->imgData[j*w+i];
            if(curpix==BORDER_COLOR ){
                coorarray[coorcount*2]=i;
                coorarray[coorcount*2+1]=j;
                coorcount++;
            }
        }
    CoordiSetBasicOp::makeAtHost(fillcoor,coorcount);
    memcpy(fillcoor->tplData,coorarray,coorcount*2*sizeof(int));
    free(coorarray);

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "seri img to coor " << runTime << " ms" << endl;
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

    // 成员方法：seedScanLineCon（并行种子扫描线算法填充 coordiset 集合围起的区域）
__host__ int                // 返回值：函数是否正确执行，若函数正确执
    // 行，返回 NO_ERROR。
    FillCoor::seedScanLineCon(
    Image *outborderimg,          // 外轮廓闭合曲线图像,同时也是输出结果
    Image *inborderimg            // 内轮廓闭合曲线图像，没有内轮廓设为空
    ){
     ImageCuda outborderCud;
     ImageCuda inborderCud;
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
        // 找长宽的最大值当内核函数的参数
        int outmaxsize=outborderimg->height>outborderimg->width?
                    outborderimg->height:outborderimg->width;
        dim3 grid,block;
        block.x=DEF_BLOCK_X;
        block.y=1;
        block.z=1;
        grid.x=(outmaxsize+DEF_BLOCK_X-1)/DEF_BLOCK_X;
        grid.y=4;
        grid.z=1;

        _seedScanLineOutConKer<<<grid,block>>>(outborderCud);

        #ifdef DEBUG_IMG
          ImageBasicOp::copyToHost(outborderimg);
          ImageBasicOp::writeToFile("outborder_Filled.bmp",outborderimg);
          // 交操作还要在 device 端使用图像
          ImageBasicOp::copyToCurrentDevice(outborderimg);
        #endif
    }// end of out border
    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "out border fill " << runTime << " ms" << endl;
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
        // 找长宽的最大值当内核函数的参数
        int inmaxsize=inborderimg->height>inborderimg->width?
                    inborderimg->height:inborderimg->width;
        dim3 grid,block;
        block.x=DEF_BLOCK_X;
        block.y=1;
        block.z=1;
        grid.x=(inmaxsize+DEF_BLOCK_X-1)/DEF_BLOCK_X;
        grid.y=4;
        grid.z=1;
        _seedScanLineOutConKer<<<grid,block>>>(inborderCud);

        #ifdef DEBUG_IMG
            ImageBasicOp::copyToHost(inborderimg);
            ImageBasicOp::writeToFile("inborderFilled.bmp",inborderimg);
            // 交操作还要在 device 端使用图像
            ImageBasicOp::copyToCurrentDevice(inborderimg);
        #endif
    }// end of in border & process

    #ifdef DEBUG_TIME
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        cout << "in border fill " << runTime << " ms" << endl;
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
        cout << "inter or negate " << runTime << " ms" << endl;
        cudaEventRecord(start, 0);
    #endif


    return NO_ERROR;
}
