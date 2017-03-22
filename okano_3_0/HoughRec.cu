// HoughRec.cu
// 实现 Hough 变换检测矩形
#include "HoughRec.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "ErrorCode.h"
#include "CoordiSet.h"
using namespace std;

// 宏：BORDER_COLOR
// 定义边界颜色
#define BORDER_COLOR 255
// 宏：BK_COLOR
// 定义背景颜色
#define BK_COLOR 0

// 宏：DEBUG
// 定义是否输出调试信息
//#define DEBUG

// 宏：M_PI
// π 值。对于某些操作系统，M_PI 可能没有定义，这里补充定义 M_PI。
#ifndef M_PI
#define M_PI 3.14159265359
#endif

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

//--------------------------类声明实现------------------------------------
// 类：Pair（存放一组平行线组（超过两条）中的一对）
// 继承自：无
// 从一组平行线中提取所有平行线对时，用该结构当返回值
class Pair{
public:
    int rho1;// 直线1的距离值
    int rho2;// 直线2的距离值
    int vote1;// 直线1的投票值
    int vote2;// 直线2的投票值
    Pair(int a,int b,int v1,int v2)
    {
        rho1=a;
        rho2=b;
        vote1=v1;
        vote2=v2;
    }
};
// 类：ThetaCluster（存放一组角度相同直线角度、距离rho、票数，即记录一组平行线）
// 继承自：无
// 检测平行四边形时，用此类存放可能的对边。
class ThetaCluster{
public:
    float theta;            // 该簇平行线的角度值，弧度制
    vector<int> rhoList;    // 存放多条直线的距离值
    vector<int> voteList;    // 存放多条直线的投票值
    // 构造方法
    ThetaCluster(float ang,int rho,int vote){
        theta=ang;
        rhoList.push_back(rho);
        voteList.push_back(vote);
    }
    // 加入一条直线的距离值
    void addRho(int rho){
        rhoList.push_back(rho);
    }
    // 加入一条直线的投票值
    void addVote(int vote){
        voteList.push_back(vote);
    }
    // 提取该组中所有的平行线对，结果用pairList向量返回
    void getPair(vector<Pair> &pairList)
    {
        if(rhoList.size()>=2)
            for(int i=0;i<rhoList.size();i++)
                for(int j=i+1;j<rhoList.size();j++){
                    Pair p(rhoList[i],rhoList[j],voteList[i],voteList[j]);
                    pairList.push_back(p);
                }
    }
};
// 类：PossiRectSet（记录可能形成平行四边形的两组平行线组在向量中的位置）
// 继承自：无
// 平行线组向量中，如果两个平行线组满足指定的角度，则产生一个对象，把这两个平行
// 线组的下标记录到该对象中，一个对象记录的两组平行线组可能产生多个平行四边形
class PossiRectSet
{    
public:
    int indexA;
    int indexB;
    PossiRectSet(int a,int b)
    {indexA=a;indexB=b;}
};

//--------------------------成员方法实现------------------------------------
// Host 成员方法：detectParallelogram（Hough 变换检测平行四边形）
// 把参数给出的多条直线按角度（可以指定误差范围）聚类，每个角度形成一个平行线组
// （每组中要两条以上直线，否则无法构成四边形），然后检测哪些两个组的角度满足参
// 数要求，则每组中任两条直线可以和另一组中任两条直线构成平行四边形，放入参数数
// 组中返回。
__host__ int HoughRec::detectParallelogram(
    LineParam *lineparam,
    int linemax, 
    int *recsmax,
    RecPolarParam *recpolarparm,
    float anglelist[],
    int anglenum
){
    // 检查输入直线坐标集是否为 NULL，如果为 NULL 直接报错返回。
    if (lineparam == NULL)
        return NULL_POINTER;
    // 用角度来聚类，放入平行线簇向量
    vector<ThetaCluster> thetaIndexParam;
    // 遍历参数给出的每一条直线
    for(int i=0;i<linemax;i++)
    {
        bool inserted=false;
        float temp_angle=lineparam[i].angle;
         // 如果当前直线倾角和平行线簇向量中任一倾角相似，则追加到相似倾角中，
         // 簇向量长度不用增加。
         for(int j=0;j<thetaIndexParam.size();j++)
             // 角度相似情况是共线
             if(fabs(temp_angle-thetaIndexParam[j].theta)<toloranceAngle)
                 {                    
                    thetaIndexParam[j].addRho(lineparam[i].distance);
                    thetaIndexParam[j].addVote(lineparam[i].votes);
                    inserted=true;
                    break;
                    }
             // 角度相差180情况左右也是共线（1度和179度），不过此段用途不大，直
             // 线处理时已经做过类似的处理了。
             else if(fabs( M_PI-temp_angle-thetaIndexParam[j].theta)
                     <toloranceAngle )
                 {// 如果两个角度相差约180，则他们的cos值大小相等，符号相反
                  // 故他们的distance值符号会相反，需要调整过来，求反
                    thetaIndexParam[j].addRho(-lineparam[i].distance);
                    thetaIndexParam[j].addVote(lineparam[i].votes);
                    inserted=true;
                    break;
                    }
         // 如果和列表中任一倾角都不相似，则单独作为一项加入向量，向量长度加一
         if(inserted == false){
            ThetaCluster temp(lineparam[i].angle,lineparam[i].distance,
                              lineparam[i].votes);
            thetaIndexParam.push_back(temp);
         }
    }// end of for 
    #ifdef DEBUG
    cout<<"parallel lines groups num="<<thetaIndexParam.size()<<endl;
    #endif
    // 遍历直线簇向量，选出相差指定角度的簇，他们可能形成四边形，放入列表
    int size=thetaIndexParam.size();
    vector<PossiRectSet> PossiRectSetList;
    for(int i=0;i<size;i++)
        // 平行线条数起码两条以上，才有可能构成平行四边形
        if(thetaIndexParam[i].rhoList.size()>=2 )
        for(int j=i+1;j<size;j++)
        {   // 平行线条数起码两条以上，才有可能构成平行四边形
            if(thetaIndexParam[j].rhoList.size()>=2)
            {
                float diff_angle=fabs(thetaIndexParam[i].theta
                                       -thetaIndexParam[j].theta);
                // 满足指定角度列表中的任一角度(或其补角)，可能是需要的平行四边
                //形，放入列表
                bool satify=false;
                for(int a=0;a<anglenum;a++)                                 
                    if( fabs(diff_angle-anglelist[a])<toloranceAngle 
                         || fabs((M_PI-diff_angle)-anglelist[a])<toloranceAngle)
                    {satify=true;    break;}
                 
                if(satify)
                {    PossiRectSet prs(i,j);
                    PossiRectSetList.push_back(prs);
                }
            }
        }
    #ifdef DEBUG
    cout<<"rect groups num="<<PossiRectSetList.size()<<endl;
    #endif
    // 产生可能的平行四边形的参数列表
    vector<RecPolarParam> RectParamList;
    for(int i=0;i<PossiRectSetList.size();i++)
    {    // 对满足夹角条件的每两个平行线簇，检测所有可能的平行四边形
        int sideA=PossiRectSetList[i].indexA;
        int sideB=PossiRectSetList[i].indexB;
        ThetaCluster clusterA=thetaIndexParam[sideA];
        ThetaCluster clusterB=thetaIndexParam[sideB];
        // 从向量中取出满足指定夹角的平行线对
        vector<Pair> pairAList;
        vector<Pair> pairBList;
        clusterA.getPair(pairAList);
        clusterB.getPair(pairBList);
        // 任两个线对都能构成一个平行四边形
        for(int a=0;a<pairAList.size();a++)
            for(int b=0;b<pairBList.size();b++)
            {
                RecPolarParam rect;
                rect.theta1=clusterA.theta;
                // 把对边的两条直线票数合并，作为票数
                rect.votes1=pairAList[a].vote1+pairAList[a].vote2;
                rect.rho1a=pairAList[a].rho1;
                rect.rho1b=pairAList[a].rho2;
                rect.theta2=clusterB.theta;
                // 把对边的两条直线票数合并，作为票数
                rect.votes2=pairBList[b].vote1+pairBList[b].vote2;
                rect.rho2a=pairBList[b].rho1;
                rect.rho2b=pairBList[b].rho2;
                
                // 得到的四边形放入列表中
                RectParamList.push_back(rect);
            }
    }// end of outer for
    #ifdef DEBUG
    cout<<" rects num="<<RectParamList.size()<<endl;
    #endif
    // 输出可能平行四边形列表
    #ifdef DEBUG
    if(RectParamList.size()>0)
    {    cout<<"all possible rectangle para(no only returned)  \n";
        cout<<"theat1   rho1a  rho1b   vote1 theat2  rho2a  rho2b vote2 \n ";
        for(int i=0;i<RectParamList.size();i++)
        cout<<"["<<RectParamList[i].theta1<<"],"
        <<RectParamList[i].rho1a<<","
        <<RectParamList[i].rho1b<<", { "
        <<RectParamList[i].votes1<<"},  "
        <<"["<<RectParamList[i].theta2<<"],"
        <<RectParamList[i].rho2a<<","
        <<RectParamList[i].rho2b<<" {"
        <<RectParamList[i].votes2<<"},  \n";
    }
    #endif
    // 根据参数给出的数量，从列表中复制到返回数组中
    // 如果列表中的四边形数量少于最大值，则全部返回
     if(RectParamList.size()<*recsmax)
         *recsmax=RectParamList.size();
     // 如果如果列表中的四边形数量大于等于最大值，则返回列表中的前面若干个
     for(int i=0;i<*recsmax;i++)
         recpolarparm[i]=RectParamList[i];

     return NO_ERROR;
}

// Host 成员方法：detectRectangle（Hough 变换检测矩形）
__host__ int HoughRec::detectRectangle(
    LineParam *lineparam, 
    int linemax, 
    int *recsmax,
    RecPolarParam *recpolarparm
){
            float angleList[1];
            // 直角 90度（弧度）
            angleList[0]=M_PI/2.0;
            return HoughRec::detectParallelogram(lineparam,linemax,recsmax,
                                                 recpolarparm,angleList,1);
             
}
// Host 成员方法：polar2XYparam(角度距离坐标转换成直角坐标)
// 注意，此方法中
 __host__ int HoughRec::polar2XYparam (
    RecPolarParam *recpolarparam,
    RecXYParam *recxyparam,
    int recnum, 
    float derho
){
    // 检查输入直线坐标集是否为 NULL，如果为 NULL 直接报错返回。
    if (recpolarparam == NULL)
        return NULL_POINTER;
    
    // 检查输入直线坐标集是否为 NULL，如果为 NULL 直接报错返回。
    if (recxyparam == NULL)
        return NULL_POINTER;
        
    // 临时变量，四个角点和中心点的坐标。
    int x[4];
    int y[4];
    // 两条相邻边与横轴的角度与正弦余弦值。
    float theta1, theta2;
    float sintheta1, sintheta2, costheta1, costheta2;
    // 两条相邻边与原点的距离。
    int rho1[4];

    int rho2[4];
        
    // 依次处理每个矩形。
    int idx=0;
    for (idx=0; idx<recnum; idx++) {
        // 获得两条相邻边的参数。
        // 角度转换为弧度
        theta1=recpolarparam[idx].theta1;
        theta2=recpolarparam[idx].theta2;

        // 求正弦余弦值，重复使用。
        sintheta1=sin(theta1);
        costheta1=cos(theta1);
        sintheta2=sin(theta2);
        costheta2=cos(theta2);

        // 顺时针排列各直线与原点的距离。
        rho1[0]=recpolarparam[idx].rho1a;
        rho1[1]=recpolarparam[idx].rho1a;
        rho1[2]=recpolarparam[idx].rho1b;
        rho1[3]=recpolarparam[idx].rho1b;

        rho2[0]=recpolarparam[idx].rho2a;
        rho2[1]=recpolarparam[idx].rho2b;
        rho2[2]=recpolarparam[idx].rho2b;
        rho2[3]=recpolarparam[idx].rho2a;
        
        
        // 根据直线与横轴的夹角决定代数方程
        // 防止出现 cos(90),sin(0)，sin(180)，即分母不能为 0
        for (int i=0; i<4; i++) {
            if (fabs(theta1 * 180.0f / M_PI)<45.0f || 
                fabs(fabs(theta1 * 180.0f / M_PI)-180.0f)<45.0f) 
            {
                y[i]=(costheta1 * rho2[i] * derho -
                        costheta2 * rho1[i] * derho) / 
                       (costheta1 * sintheta2-costheta2 * sintheta1);
                x[i]=(rho1[i] * derho-y[i] * sintheta1) / costheta1;
            } else {
                x[i]=(sintheta1 * rho2[i] * derho -
                        sintheta2 * rho1[i] * derho) / 
                       (costheta2 * sintheta1-costheta1 * sintheta2);
                y[i]=(rho1[i] * derho-x[i] * costheta1) / sintheta1;
            }
        }

        // 将坐标参数写入矩形结构体。
        x[0]=(x[0]>0 ? x[0] :-x[0]);
        x[1]=(x[1]>0 ? x[1] :-x[1]);
        x[2]=(x[2]>0 ? x[2] :-x[2]);
        x[3]=(x[3]>0 ? x[3] :-x[3]);
        recxyparam[idx].x1=x[0];
        recxyparam[idx].x2=x[1];
        recxyparam[idx].x3=x[2]; 
        recxyparam[idx].x4=x[3];
        recxyparam[idx].xc=(x[0]+x[1]+x[2]+x[3]) / 4;
        
        x[0]=(x[0]>0 ? x[0] :-x[0]);
        x[1]=(x[1]>0 ? x[1] :-x[1]);
        x[2]=(x[2]>0 ? x[2] :-x[2]);
        x[3]=(x[3]>0 ? x[3] :-x[3]);
        recxyparam[idx].y1=y[0];
        recxyparam[idx].y2=y[1];
        recxyparam[idx].y3=y[2];
        recxyparam[idx].y4=y[3];
        recxyparam[idx].yc=(y[0]+y[1]+y[2]+y[3]) / 4;
        
        recxyparam[idx].votes=
                2 * (recpolarparam[idx].votes1+recpolarparam[idx].votes2);
    }
      
    // 处理完毕，退出。 
    return NO_ERROR;
}

// Host 成员方法：detectRectangle(检测inimg图像中的矩形，放入数组返回)
__host__ int 
HoughRec:: detectRectangle(
    Image *inimg,               // 输入图像
    int linenum,                // 最大直线数量
    int linethres,              // 直线票数阈值
    float lineangthres,         // 相似直线角度
    int linedisthres,           // 相似直线距离
    int *rectnum,               // 返回矩形数量
    RecXYParam *rectxypara      // 返回矩形xy坐标参数
){
    HoughLine houghline;
    // 直线检测角度和距离的步长
    houghline.setDeTheta(M_PI / 180.0);
    houghline.setDeRho(1);
    // 票数阈值，根据图像分片大小和图像中直线的粗细设定
    houghline.setThreshold(linethres);
    // 合并相似直线采用的参数，倾角相差6度内且dis值相差15以内，可以认为是同一条直线
    houghline.setThresAng(lineangthres);
    houghline.setThresDis(linedisthres);


    int linesMax =linenum;
    LineParam *lineparam= new LineParam[linesMax];

    // 直线检测
    #ifdef DEBUG
       Image *outimg;
       ImageBasicOp::newImage(&outimg);
       ImageBasicOp::makeAtHost(outimg, inimg->width, inimg->height);
     
       houghline.houghLineImg(inimg, NULL,outimg, &linesMax, lineparam);
       
       ImageBasicOp::copyToHost(outimg);
       ImageBasicOp::writeToFile("line_out.bmp", outimg); 
       ImageBasicOp::deleteImage(outimg);
     
       cout << "linesMax = " << linesMax << endl;
       printf("序号  angle  distance  vote \n");
       for (int i = 0; i < linesMax; i++)
           printf("%4d %12f(%12f) %5d %5d\n",i,lineparam[i].angle,lineparam[i].angle/M_PI*180,
           lineparam[i].distance,lineparam[i].votes);
    #else    
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float runTime;
    cudaEventRecord(start, 0);*/
       houghline.houghLine(inimg, NULL, &linesMax, lineparam);

    /*
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runTime, start, stop);
    printf(" %f ",runTime);*/
    #endif




    // 根据直线结果处理矩形
    cudaThreadSynchronize();// 不加同步语句，则下面的代码结果不正确

    // 申请倾角-距离参数数组
    RecPolarParam *rectpp=new RecPolarParam[*rectnum];
    // 检测矩形，放入倾角-距离参数数组，数量放入*rectnum
    detectRectangle(lineparam,linesMax,rectnum,rectpp);
    // 倾角-距离参数数组转换成矩形XY坐标参数结构体
    polar2XYparam (rectpp, rectxypara, *rectnum, 1);
    // 把矩形xy坐标由分片局部坐标加上分片远点坐标，转换成全局坐标
    for(int i=0;i<*rectnum;i++){
        rectxypara[i].x1 += inimg->roiX1;
        rectxypara[i].x2 += inimg->roiX1;
        rectxypara[i].x3 += inimg->roiX1;
        rectxypara[i].x4 += inimg->roiX1;
        rectxypara[i].y1 += inimg->roiY1;
        rectxypara[i].y2 += inimg->roiY1;
        rectxypara[i].y3 += inimg->roiY1;
        rectxypara[i].y4 += inimg->roiY1;
    }
    // 局部动态内存回收
    if(lineparam!=NULL)
        {delete[] lineparam;lineparam=NULL;}
    if(rectpp!=NULL)
        {delete[] rectpp;rectpp=NULL;}

    return NO_ERROR;
  }

// Host 成员方法：detectRectangle(检测CoordiSet中的矩形，放入数组返回)
__host__ int 
HoughRec:: detectRectangle(
    CoordiSet *coor,               // 输入坐标集
    int linenum,                // 最大直线数量
    int linethres,              // 直线票数阈值
    float lineangthres,         // 相似直线角度
    int linedisthres,           // 相似直线距离
    int *rectnum,               // 返回矩形数量
    RecXYParam *rectxypara      // 返回矩形xy坐标参数
    ){
    if(coor==NULL)
        return INVALID_DATA;
   // 获取坐标集中点的分布范围，即包围盒坐标
    int minx,maxx,miny,maxy;    
    Image *inimg;
    ImageBasicOp::newImage(&inimg);
    ImgConvert imgcvt(BORDER_COLOR,BK_COLOR);
    // ----------------------输入coor参数转化成img----------------------------
    // 预处理，得到外轮廓大小
    int errorcode=_findMinMaxCoordinates(coor,&minx,&miny,&maxx,&maxy);
    if(errorcode!=NO_ERROR)
        return 0;
    // 创建工作图像
    //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1,保证点在图像内部
    ImageBasicOp::makeAtHost(inimg,maxx+2 ,maxy+2);
    // 把坐标集绘制到图像上,前景255，背景0
    imgcvt.cstConvertToImg(coor,inimg);

    // 调用图像接口完成剩下操作。
    detectRectangle(
        inimg,                  // 输入坐标集
        linenum,                // 最大直线数量
        linethres,              // 直线票数阈值
        lineangthres,           // 相似直线角度
        linedisthres,           // 相似直线距离
        rectnum,               // 返回矩形数量
        rectxypara             // 返回矩形xy坐标参数
        );

    ImageBasicOp::deleteImage(inimg);
    return NO_ERROR;
    }

// Host 成员方法：detectRealRectangle(检测矩形数组中真实矩形数量，
// 放入数组返回，参照为坐标集)
__host__ int 
HoughRec:: detectRealRectangle(
    CoordiSet *coor,               // 输入坐标集
    int rectnum,                // 可能矩形数量
    RecXYParam *rectxypara,     //可能矩形参数数组
    int distance,               // 真实直线判定距离
    float percent,              // 真实直线判定阈值
    int *realrectnum,           //真实矩形数量
    RecXYParam *realrectxypara  //真实矩形xy坐标参数
    ){
    if(coor==NULL)
        return INVALID_DATA;
   // 获取坐标集中点的分布范围，即包围盒坐标
    int minx,maxx,miny,maxy;    
    Image *inimg;
    ImageBasicOp::newImage(&inimg);
    ImgConvert imgcvt(BORDER_COLOR,BK_COLOR);
    // ----------------------输入coor参数转化成img----------------------------
    // 预处理，得到外轮廓大小
    int errorcode=_findMinMaxCoordinates(coor,&minx,&miny,&maxx,&maxy);
    if(errorcode!=NO_ERROR)
        return 0;
    // 创建工作图像
    //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1,保证点在图像内部
    ImageBasicOp::makeAtHost(inimg,maxx+2 ,maxy+2);
    // 把坐标集绘制到图像上,前景255，背景0
    imgcvt.cstConvertToImg(coor,inimg);

    // 调用图像接口完成剩余操作。
    detectRealRectangle(
        inimg,                 // 输入图像
        rectnum,                // 可能矩形数量
        rectxypara,             //可能矩形参数数组
        distance,               // 真实直线判定距离
        percent,              // 真实直线判定阈值
        realrectnum,           //真实矩形数量
        realrectxypara        //真实矩形xy坐标参数
    );

    ImageBasicOp::deleteImage(inimg);
    return NO_ERROR;
}

// Host 成员方法：detectRealRectangle(检测矩形数组中真实矩形数量，放入数组返回)
__host__ int 
HoughRec:: detectRealRectangle(
    Image *inimg,               // 输入图像
    int rectnum,                // 可能矩形数量
    RecXYParam *rectxyparam,     //可能矩形参数数组
    int distance,               // 真实直线判定距离
    float percent,              // 真实直线判定阈值
    int *realrectnum,           //真实矩形数量
    RecXYParam *realrectxypara  //真实矩形xy坐标参数
){
    int pointer=0;
    HoughLine houghline;
    for(int i=0; i< rectnum; i++) {
       // 对矩形四个边进行真实性判定
       bool b1,b2,b3,b4;
       inimg->roiX1=0;
       inimg->roiX2=inimg->width;
       inimg->roiY1=0;
       inimg->roiY2=inimg->height; 

       b1=houghline.realLine(inimg,rectxyparam[i].x1,rectxyparam[i].y1,
                         rectxyparam[i].x2,rectxyparam[i].y2, distance,percent);
       b2=houghline.realLine(inimg,rectxyparam[i].x2,rectxyparam[i].y2,
                         rectxyparam[i].x3,rectxyparam[i].y3, distance,percent);
       b3=houghline.realLine(inimg,rectxyparam[i].x3,rectxyparam[i].y3,
                         rectxyparam[i].x4,rectxyparam[i].y4, distance,percent);
       b4=houghline.realLine(inimg,rectxyparam[i].x4,rectxyparam[i].y4,
                         rectxyparam[i].x1,rectxyparam[i].y1, distance,percent);
       // 判断四个边是否是真实直线，全都是的话，放入结果数组中。
        #ifdef DEBUG
            cout<<"b1="<<b1<<"  b2="<<b2<<"  b3="<<b3<<"  b4="<<b4<< endl;
        #endif
        if( b1 && b2 && b3 && b4 && pointer<*realrectnum){
            realrectxypara[pointer]=rectxyparam[i];
            pointer++;
         }
    }

    // 返回真实矩形的个数
    *realrectnum=pointer;
    return NO_ERROR;
  }

// Host 成员方法：drawRect(把直角坐标四边形绘制到指定图像文件中)
__host__ int 
HoughRec:: drawRect(
    string filename,
    size_t w,
    size_t h,
    RecXYParam recxyparam[],
    int rectmax 
){   // 创建坐标集
     CoordiSet *cst;
     CoordiSetBasicOp::newCoordiSet(&cst);
     // 只有4个点的一个坐标集
     CoordiSetBasicOp::makeAtHost(cst, 4);     
     // 创建输出图像
     Image *recimg;
     ImageBasicOp::newImage(&recimg);
     // 输出图像和
     ImageBasicOp::makeAtHost(recimg,w,h);
     ImageDrawer id;
     id.setBrushColor(0);
     // 刷背景色
     id.brushAllImage(recimg);

     for (int i=0; i< rectmax; i++) {
         CoordiSetBasicOp::copyToHost(cst);
         cst->tplData[0]=recxyparam[i].x1;
         cst->tplData[1]=recxyparam[i].y1;
         cst->tplData[2]=recxyparam[i].x2;
         cst->tplData[3]=recxyparam[i].y2;
         cst->tplData[4]=recxyparam[i].x3;
         cst->tplData[5]=recxyparam[i].y3;
         cst->tplData[6]=recxyparam[i].x4;
         cst->tplData[7]=recxyparam[i].y4;
         id.drawTrace(recimg, cst);// 把cst中点顺序连接成一个闭合图形
     }

     // 图像写入硬盘文件
     ImageBasicOp::copyToHost(recimg);
     ImageBasicOp::writeToFile(filename.c_str(), recimg);
     ImageBasicOp::deleteImage(recimg);

     return NO_ERROR;
}

// Host 成员方法：pieceRealRectImg（分片检测img图像中的矩形写入图像文件）
__host__ int
HoughRec:: pieceRealRectImg(
    Image *inimg,
    string lineoutfile1,
    string lineoutfile2,
    string rectoutfile,
    int piecenum,
    int linenum,
    int linethres,
    float lineangthres,
    int linedisthres,
    int rectnum,
    int distance,
    float percent
){

    // 两个不同分块直线检测的结果放入两个图像中
    Image *outimg;
    ImageBasicOp::newImage(&outimg);
    ImageBasicOp::makeAtHost(outimg, inimg->width, inimg->height);
    Image *outimg2;
    ImageBasicOp::newImage(&outimg2);
    ImageBasicOp::makeAtHost(outimg2, inimg->width, inimg->height);
 
     // 创建矩形输出图像            
    Image *recoutimg;
    ImageBasicOp::newImage(&recoutimg);
    ImageBasicOp::makeAtHost(recoutimg, inimg->width, inimg->height);

    // 用imageDrawer给矩形输出图像刷背景色
    ImageDrawer id;
    id.setBrushColor(0);    
    id.brushAllImage(recoutimg);

    // 计算分片的大小
    int cell_x=inimg->width/piecenum;
    int cell_y=inimg->height/piecenum;
    #ifdef DEBUG
        printf("cell_x=%d cell_y=%d\n",cell_x,cell_y);
    #endif
    HoughLine hough;
    // 直线检测的角度步长=1度，距离步长=1像素
    hough.setDeTheta(M_PI / 180.0);
    hough.setDeRho(1);
    // 票数阈值 ，根据图像分片大小和图像中直线的粗细设定
    hough.setThreshold(linethres);
    // 合并相似直线采用的参数，倾角相差6度内且dis值相差15以内，可以认为是同一条直线
    hough.setThresAng(lineangthres);
    hough.setThresDis(linedisthres);

     // 开始分块处理
    for(int y=0;y<piecenum;y++)
         for(int x=0;x<piecenum;x++)
         {// .....................分块第一阶段.........................
             #ifdef DEBUG
                 printf(" \n.................y=[%d] x=[%d]\n",y,x);
             #endif
             int linesMax =linenum;
             LineParam *lineparam= new LineParam[linesMax];
             for(int i=0;i<linesMax;i++){
                 lineparam[i].angle=-1;
                 lineparam[i].distance=-1;
                 lineparam[i].votes=-1;
             }
            inimg->roiX1=x*cell_x;
            inimg->roiX2=x*cell_x+cell_x-1;
            inimg->roiY1=y*cell_y;
            inimg->roiY2=y*cell_y+cell_y-1;
            outimg->roiX1= inimg->roiX1;
            outimg->roiX2= inimg->roiX2;
            outimg->roiY1= inimg->roiY1;
            outimg->roiY2= inimg->roiY2;
            #ifdef DEBUG
            printf("x1=%d x2=%d y1=%d y2=%d \n"
                ,inimg->roiX1,inimg->roiX2
                ,inimg->roiY1,inimg->roiY2);
            #endif
            // 注意，此时得到的直线参数是分片内的局部坐标，非全局坐标，要转换
            hough.houghLineImg(inimg, NULL, outimg, &linesMax, lineparam);

            // 根据直线结果处理矩形
            // 不加同步语句，则下面的代码结果不正确
            cudaThreadSynchronize();

            RecPolarParam *rectpp=new RecPolarParam[rectnum];
            // 初始化
            for(int i=0;i<rectnum;i++)
                rectpp[i].theta1=-10000;

            // 用于返回每个分片中真正的矩形个数
            int rectMax=rectnum;
            detectRectangle(lineparam,linesMax,&rectMax,rectpp);

            RecXYParam *recxyparam;
            recxyparam=new RecXYParam[rectMax];
            // 初始化
            for(int i=0;i<rectMax;i++)
                recxyparam[i].x1=-10000;

            // 输入矩形极坐标参数结构体，输出矩形XY坐标参数结构体
            polar2XYparam (rectpp, recxyparam, rectMax, 1);
            // 把矩形xy坐标由分片局部坐标加上分片原点坐标，转换成全局坐标
            for(int i=0;i<rectMax;i++){
                recxyparam[i].x1 += inimg->roiX1;
                recxyparam[i].x2 += inimg->roiX1;
                recxyparam[i].x3 += inimg->roiX1;
                recxyparam[i].x4 += inimg->roiX1;
                recxyparam[i].y1 += inimg->roiY1;
                recxyparam[i].y2 += inimg->roiY1;
                recxyparam[i].y3 += inimg->roiY1;
                recxyparam[i].y4 += inimg->roiY1;
            }
 
            // 绘制矩形：输入 recxyparam 、rectMax 绘制到recoutimg图像中
             // 创建只有4个点的一个坐标集
            CoordiSet *cst;
            CoordiSetBasicOp::newCoordiSet(&cst);            
            CoordiSetBasicOp::makeAtHost(cst, 4);
            // 每个矩形四个顶点放入坐标集
            for (int i=0; i< rectMax; i++) {
                CoordiSetBasicOp::copyToHost(cst);
                cst->tplData[0]=recxyparam[i].x1;
                cst->tplData[1]=recxyparam[i].y1;
                cst->tplData[2]=recxyparam[i].x2;
                cst->tplData[3]=recxyparam[i].y2;
                cst->tplData[4]=recxyparam[i].x3;
                cst->tplData[5]=recxyparam[i].y3;
                cst->tplData[6]=recxyparam[i].x4;
                cst->tplData[7]=recxyparam[i].y4;
                // 对矩形四个边进行真实性判定
                bool b1,b2,b3,b4;
                inimg->roiX1=0;
                inimg->roiX2=inimg->width;
                inimg->roiY1=0;
                inimg->roiY2=inimg->height; 
                b1=hough.realLine(inimg,recxyparam[i].x1,recxyparam[i].y1,
                                  recxyparam[i].x2,recxyparam[i].y2, distance,percent);
                b2=hough.realLine(inimg,recxyparam[i].x2,recxyparam[i].y2,
                                  recxyparam[i].x3,recxyparam[i].y3, distance,percent);
                b3=hough.realLine(inimg,recxyparam[i].x3,recxyparam[i].y3,
                                  recxyparam[i].x4,recxyparam[i].y4, distance,percent);
                b4=hough.realLine(inimg,recxyparam[i].x4,recxyparam[i].y4,
                                  recxyparam[i].x1,recxyparam[i].y1, distance,percent);
                // 判断四个边是否是真实直线，全都是的话，就把cst中点顺序连接
                // 成一个闭合图形,绘制到recoutimg中。
                #ifdef DEBUG
                    cout<<"b1="<<b1<<"  b2="<<b2<<"  b3="<<b3<<"  b4="<<b4<< endl;
                #endif
                if( b1 && b2 && b3 && b4)
                    id.drawTrace(recoutimg,cst);
            }
            // 循环内声明的局部动态内存，循环内回收
            if(lineparam != NULL)
                {delete[] lineparam;lineparam=NULL;}
            if(recxyparam != NULL)
                {delete[] recxyparam;recxyparam=NULL;}
            if(rectpp!=NULL)
                {delete[] rectpp;rectpp=NULL;}
        // ...................分块第二阶段 ........................
        if(x<piecenum-1 && y<piecenum-1){
            #ifdef DEBUG
                printf(" \n---------- step2 of[%d][%d]--------\n",y,x);
            #endif
            int linesMax =linenum;
            LineParam *lineparam=new LineParam[linesMax];
            for(int i=0;i<linesMax;i++){
                lineparam[i].angle=-1;
                lineparam[i].distance=-1;
                lineparam[i].votes=-1;
            }
            // 每个分片向下、向右移动半个单位
            inimg->roiX1=x*cell_x+cell_x/2;
            inimg->roiX2=x*cell_x+cell_x/2+cell_x-1;
            inimg->roiY1=y*cell_y+cell_y/2;
            inimg->roiY2=y*cell_y+cell_y/2+cell_y-1; 
            outimg2->roiX1=inimg->roiX1;
            outimg2->roiX2=inimg->roiX2;
            outimg2->roiY1=inimg->roiY1;
            outimg2->roiY2=inimg->roiY2;
            #ifdef DEBUG
                printf("x1=%d x2=%d y1=%d y2=%d \n",
                        inimg->roiX1,inimg->roiX2,
                        inimg->roiY1,inimg->roiY2);
            #endif
            // 注意，此时得到的直线参数是局部坐标，非全局坐标，要转换
            hough.houghLineImg(inimg, NULL, outimg2, &linesMax, lineparam);
            // 根据直线结果处理矩形
            cudaThreadSynchronize();// 不加同步语句，则下面的代码结果不正确

            RecPolarParam *rectpp;
            rectpp=new RecPolarParam[rectnum];
            // 初始化
            for(int i=0;i<rectnum;i++)
                rectpp[i].theta1=-10000;

             // 用于返回每个分片中真正的矩形个数
            int rectMax=rectnum;
            detectRectangle(lineparam,linesMax,&rectMax,rectpp);

            RecXYParam *recxyparam;
            recxyparam=new RecXYParam[rectMax];
            // 初始化
            for(int i=0;i<rectMax;i++)
                recxyparam[i].x1=-10000;        
            // 输入矩形极坐标参数结构体，输出矩形XY坐标参数结构体
            polar2XYparam (rectpp, recxyparam, rectMax, 1);
             // 把矩形xy坐标由分片局部坐标转换成全局坐标
            for(int i=0;i<rectMax;i++){
                recxyparam[i].x1 += inimg->roiX1;
                recxyparam[i].x2 += inimg->roiX1;
                recxyparam[i].x3 += inimg->roiX1;
                recxyparam[i].x4 += inimg->roiX1;
                recxyparam[i].y1 += inimg->roiY1;
                recxyparam[i].y2 += inimg->roiY1;
                recxyparam[i].y3 += inimg->roiY1;
                recxyparam[i].y4 += inimg->roiY1;
            }
            // 绘制矩形：输入 recxyparam 、rectMax 绘制到recoutimg图像中
            // 创建坐标集
            CoordiSet *cst;
            CoordiSetBasicOp::newCoordiSet(&cst);
            // 只有4个点的一个坐标集
            CoordiSetBasicOp::makeAtHost(cst,4);
            for (int i=0; i< rectMax; i++){
                CoordiSetBasicOp::copyToHost(cst);
                cst->tplData[0]=recxyparam[i].x1;
                cst->tplData[1]=recxyparam[i].y1;
                cst->tplData[2]=recxyparam[i].x2;
                cst->tplData[3]=recxyparam[i].y2;
                cst->tplData[4]=recxyparam[i].x3;
                cst->tplData[5]=recxyparam[i].y3;
                cst->tplData[6]=recxyparam[i].x4;
                cst->tplData[7]=recxyparam[i].y4;
                bool b1,b2,b3,b4;
                inimg->roiX1=0;
                inimg->roiX2=inimg->width;
                inimg->roiY1=0;
                inimg->roiY2=inimg->height; 
                b1=hough.realLine(inimg,recxyparam[i].x1,recxyparam[i].y1,
                                  recxyparam[i].x2,recxyparam[i].y2, distance,percent);
                b2=hough.realLine(inimg,recxyparam[i].x2,recxyparam[i].y2,
                                  recxyparam[i].x3,recxyparam[i].y3, distance,percent);
                b3=hough.realLine(inimg,recxyparam[i].x3,recxyparam[i].y3,
                                  recxyparam[i].x4,recxyparam[i].y4, distance,percent);
                b4=hough.realLine(inimg,recxyparam[i].x4,recxyparam[i].y4,
                                  recxyparam[i].x1,recxyparam[i].y1, distance,percent);
                // 判断四个边是否是真实直线，全都是的话，就把cst中点顺序连接
                // 成一个闭合图形,绘制到recoutimg中。
                #ifdef DEBUG
                    cout<<"b1="<<b1<<"  b2="<<b2<<"  b3="<<b3<<"  b4="<<b4<< endl;
                #endif
                if( b1 && b2 && b3 && b4)
                    id.drawTrace(recoutimg, cst);
            }
            // 循环内声明的局部动态内存，循环内回收
            if(lineparam != NULL)
                {delete[] lineparam;lineparam=NULL;}
            if(recxyparam != NULL)
                {delete[] recxyparam;recxyparam=NULL;}
            if(rectpp!=NULL)
                {delete[] rectpp;rectpp=NULL;}    
            }// end of step2 if
        }// end of for x, y 

     // 图像写入文件中
     ImageBasicOp::copyToHost(outimg);
     ImageBasicOp::writeToFile( lineoutfile1.c_str(), outimg);
     ImageBasicOp::copyToHost(outimg2);
     ImageBasicOp::writeToFile( lineoutfile2.c_str(), outimg2);
     ImageBasicOp::copyToHost(recoutimg);
     ImageBasicOp::writeToFile(rectoutfile.c_str(), recoutimg);
     // 回收资源
     ImageBasicOp::deleteImage(outimg);
     ImageBasicOp::deleteImage(outimg2);
     ImageBasicOp::deleteImage(recoutimg);

    return NO_ERROR;
  }
// Host 成员方法：重载pieceRealRectImg（分片检测坐标集合中的矩形写入图像文件）
__host__ int
HoughRec:: pieceRealRectImg(
    CoordiSet* coor,
    string lineoutfile1,
    string lineoutfile2,
    string rectoutfile,
    int piecenum,
    int linenum,
    int linethres,
    float lineangthres,
    int linedisthres,
    int rectnum,
    int distance,
    float percent
){
    if(coor==NULL)
        return INVALID_DATA;
   // 获取坐标集中点的分布范围，即包围盒坐标
    int minx,maxx,miny,maxy;    
    Image *inimg;
    ImageBasicOp::newImage(&inimg);
    ImgConvert imgcvt(BORDER_COLOR,BK_COLOR);
    // ----------------------输入coor参数转化成img----------------------------
    // 预处理，得到外轮廓大小
    int errorcode=_findMinMaxCoordinates(coor,&minx,&miny,&maxx,&maxy);
    if(errorcode!=NO_ERROR)
        return 0;
    // 创建工作图像
    //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1,保证点在图像内部
    ImageBasicOp::makeAtHost(inimg,maxx+2 ,maxy+2);
    // 把坐标集绘制到图像上,前景255，背景0
    imgcvt.cstConvertToImg(coor,inimg);
    #ifdef DEBUG_IMG
        // 把填充前的图像保存到文件
        ImageBasicOp::copyToHost(inimg);
        ImageBasicOp::writeToFile("coorimg.bmp",inimg);
     #endif
    // --------------调用图像接口的pieceRealRectImg--------------------
    pieceRealRectImg(
        inimg,
        lineoutfile1,
        lineoutfile2,
        rectoutfile,
        piecenum,
        linenum,
        linethres,
        lineangthres,
        linedisthres,
        rectnum,
        distance,
        percent
        );
    // 回收资源
    ImageBasicOp::deleteImage(inimg);

    return NO_ERROR;
  }

// Host 成员方法：pieceRealRect(分片检测inimg图像中的矩形，放入数组返回)
__host__ int 
HoughRec:: pieceRealRect(
    Image *inimg,               // 输入图像
    int piecenum,
    int linenum,
    int linethres,
    float lineangthres,
    int linedisthres,
    int rectnum,
    int distance,
    float percent,
    int *realrectnum,
    RecXYParam *realxypara
){
    // 矩形计数器清零
    int pointer=0;
    // 计算分片的大小
    int cell_x=inimg->width/piecenum;
    int cell_y=inimg->height/piecenum;
    #ifdef DEBUG
        printf("cell_x=%d cell_y=%d\n",cell_x,cell_y);
    #endif
    HoughLine hough;
    // 直线检测角度和距离的步长
    hough.setDeTheta(M_PI / 180.0);
    hough.setDeRho(1);
    // 票数阈值，根据图像分片大小和图像中直线的粗细设定
    hough.setThreshold(linethres);
    // 合并相似直线采用的参数，倾角相差6度内且dis值相差15以内，可以认为是同一条直线
    hough.setThresAng(lineangthres);
    hough.setThresDis(linedisthres);

    // 开始分块处理
    for(int y=0;y<piecenum;y++)
        for(int x=0;x<piecenum;x++)
        {//.......................分块第一阶段..........................
         #ifdef DEBUG
             printf(" \n----------------- y=[%d] x=[%d]\n",y,x);
         #endif
         int linesMax =linenum;
             LineParam *lineparam= new LineParam[linesMax];
             for(int i=0;i<linesMax;i++){
                  lineparam[i].angle=-1;
                  lineparam[i].distance=-1;
                  lineparam[i].votes=-1;
             }
             inimg->roiX1=x*cell_x;
             inimg->roiX2=x*cell_x+cell_x-1;
             inimg->roiY1=y*cell_y;
             inimg->roiY2=y*cell_y+cell_y-1; 

             #ifdef DEBUG
             printf("x1=%d x2=%d y1=%d y2=%d \n",
                 inimg->roiX1,inimg->roiX2,
                 inimg->roiY1,inimg->roiY2);
             #endif

             hough.houghLine(inimg, NULL, &linesMax, lineparam);
             // 根据直线结果处理矩形
             cudaThreadSynchronize();// 不加同步语句，则下面的代码结果不正确

             RecPolarParam *rectpp=new RecPolarParam[rectnum];
             // 初始化
             for(int i=0;i<rectnum;i++)
                 rectpp[i].theta1=-10000;
             // 用于返回每个分片中真正的矩形个数
             int rectMax=rectnum;
             detectRectangle(lineparam,linesMax,&rectMax,rectpp);

             RecXYParam *recxyparam;
             recxyparam=new RecXYParam[rectMax];
             // 初始化
             for(int i=0;i<rectMax;i++)
                 recxyparam[i].x1=-10000;
        
             // 输入矩形极坐标参数结构体，输出矩形XY坐标参数结构体
             polar2XYparam (rectpp, recxyparam, rectMax, 1);
             // 把矩形xy坐标由分片局部坐标加上分片远点坐标，转换成全局坐标
             for(int i=0;i<rectMax;i++){
                 recxyparam[i].x1 += inimg->roiX1;
                 recxyparam[i].x2 += inimg->roiX1;
                 recxyparam[i].x3 += inimg->roiX1;
                 recxyparam[i].x4 += inimg->roiX1;
                 recxyparam[i].y1 += inimg->roiY1;
                 recxyparam[i].y2 += inimg->roiY1;
                 recxyparam[i].y3 += inimg->roiY1;
                 recxyparam[i].y4 += inimg->roiY1;
             }
             for(int i=0; i< rectMax; i++) {
                // 对矩形四个边进行真实性判定
                bool b1,b2,b3,b4;
                inimg->roiX1=0;
                inimg->roiX2=inimg->width;
                inimg->roiY1=0;
                inimg->roiY2=inimg->height; 
                b1=hough.realLine(inimg,recxyparam[i].x1,recxyparam[i].y1,
                                  recxyparam[i].x2,recxyparam[i].y2, distance,percent);
                b2=hough.realLine(inimg,recxyparam[i].x2,recxyparam[i].y2,
                                  recxyparam[i].x3,recxyparam[i].y3, distance,percent);
                b3=hough.realLine(inimg,recxyparam[i].x3,recxyparam[i].y3,
                                  recxyparam[i].x4,recxyparam[i].y4, distance,percent);
                b4=hough.realLine(inimg,recxyparam[i].x4,recxyparam[i].y4,
                                  recxyparam[i].x1,recxyparam[i].y1, distance,percent);
                // 判断四个边是否是真实直线，全都是的话，放入结果数组中。
                #ifdef DEBUG
                    cout<<"b1="<<b1<<"  b2="<<b2<<"  b3="<<b3<<"  b4="<<b4<< endl;
                #endif
                if( b1 && b2 && b3 && b4 && pointer<*realrectnum){
                    realxypara[pointer]=recxyparam[i];
                    pointer++;
                 }
            }
            // 循环内声明的局部动态内存，循环内回收
            if(lineparam!=NULL)
                {delete[] lineparam;lineparam=NULL;}
            if(recxyparam!=NULL)
                {delete[] recxyparam;recxyparam=NULL;}
            if(rectpp!=NULL)
                {delete[] rectpp;rectpp=NULL;}
        //.........................分块第二阶段........................
        if(x<piecenum-1 && y<piecenum-1){
            #ifdef DEBUG
                printf(" \n-----------step2 of[%d][%d]-----------\n",y,x);
            #endif
            int linesMax =linenum;
            LineParam *lineparam=new LineParam[linesMax];
            for(int i=0;i<linesMax;i++){
                lineparam[i].angle=-1;
                lineparam[i].distance=-1;
                lineparam[i].votes=-1;
            }
            // 每个分片向下、向右移动半个单位
            inimg->roiX1=x*cell_x+cell_x/2;
            inimg->roiX2=x*cell_x+cell_x/2+cell_x-1;
            inimg->roiY1=y*cell_y+cell_y/2;
            inimg->roiY2=y*cell_y+cell_y/2+cell_y-1;

            #ifdef DEBUG
                printf("x1=%d x2=%d y1=%d y2=%d \n",
                       inimg->roiX1,inimg->roiX2,
                       inimg->roiY1,inimg->roiY2);
            #endif
            // 注意，此时得到的直线参数是局部坐标，非全局坐标，要转换
            hough.houghLine(inimg, NULL, &linesMax, lineparam);

            // 根据直线结果处理矩形
            // 不加同步语句，则下面的代码结果不正确
            cudaThreadSynchronize();

            RecPolarParam *rectpp;
            rectpp=new RecPolarParam[rectnum];
            // 初始化
            for(int i=0;i<rectnum;i++)
                rectpp[i].theta1=-10000;

            // 用于返回每个分片中真正的矩形个数
            int rectMax=rectnum;
            detectRectangle(lineparam,linesMax,&rectMax,rectpp);

            RecXYParam *recxyparam;
            recxyparam=new RecXYParam[rectMax];
            // 初始化
            for(int i=0;i<rectMax;i++)
                recxyparam[i].x1=-10000;        
            // 输入矩形极坐标参数结构体，输出矩形XY坐标参数结构体
            polar2XYparam (rectpp, recxyparam, rectMax, 1);
             // 把矩形xy坐标由分片局部坐标转换成全局坐标
            for(int i=0;i<rectMax;i++){
                recxyparam[i].x1 += inimg->roiX1;
                recxyparam[i].x2 += inimg->roiX1;
                recxyparam[i].x3 += inimg->roiX1;
                recxyparam[i].x4 += inimg->roiX1;
                recxyparam[i].y1 += inimg->roiY1;
                recxyparam[i].y2 += inimg->roiY1;
                recxyparam[i].y3 += inimg->roiY1;
                recxyparam[i].y4 += inimg->roiY1;
            }
        
            for (int i=0; i< rectMax; i++){
                bool b1,b2,b3,b4;
                inimg->roiX1=0;
                inimg->roiX2=inimg->width;
                inimg->roiY1=0;
                inimg->roiY2=inimg->height; 
                b1=hough.realLine(inimg,recxyparam[i].x1,recxyparam[i].y1,
                                      recxyparam[i].x2,recxyparam[i].y2, distance,percent);
                b2=hough.realLine(inimg,recxyparam[i].x2,recxyparam[i].y2,
                                      recxyparam[i].x3,recxyparam[i].y3, distance,percent);
                b3=hough.realLine(inimg,recxyparam[i].x3,recxyparam[i].y3,
                                      recxyparam[i].x4,recxyparam[i].y4, distance,percent);
                b4=hough.realLine(inimg,recxyparam[i].x4,recxyparam[i].y4,
                                      recxyparam[i].x1,recxyparam[i].y1, distance,percent);
                // 判断四个边是否是真实直线，全都是的话，放入结果数组中。
                #ifdef DEBUG
                    cout<<"b1="<<b1<<"  b2="<<b2<<"  b3="<<b3<<"  b4="<<b4<< endl;
                #endif
                if( b1 && b2 && b3 && b4 && pointer<*realrectnum)                
                  {
                    realxypara[pointer]=recxyparam[i];
                    pointer++;
                    }
                }
            // 循环内声明的局部动态内存，循环内回收
            if(lineparam!=NULL)
                {delete[] lineparam;lineparam=NULL;}
            if(recxyparam!=NULL)
                {delete[] recxyparam;recxyparam=NULL;}
            if(rectpp!=NULL)
                {delete[] rectpp;rectpp=NULL;}
            }// end of step2 if
        }// end of for x,for y

     // 返回真实矩形的个数
     *realrectnum=pointer;

    return NO_ERROR;
  }

  // Host 成员方法：重载pieceRealRect(分片检测coor坐标集中的矩形，放入数组返回)
__host__ int 
HoughRec:: pieceRealRect(
    CoordiSet* coor,               // 输入坐标集
    int piecenum,
    int linenum,
    int linethres,
    float lineangthres,
    int linedisthres,
    int rectnum,
    int distance,
    float percent,
    int *realrectnum,
    RecXYParam *realxypara
){
    if(coor!=NULL){
       // 获取坐标集中点的分布范围，即包围盒坐标
        int minx,maxx,miny,maxy;    
        Image *inimg;
        ImageBasicOp::newImage(&inimg);
        ImgConvert imgcvt(BORDER_COLOR,BK_COLOR);
        // ----------------------输入coor参数转化成img----------------------------
        // 预处理，得到外轮廓大小
        int errorcode=_findMinMaxCoordinates(coor,&minx,&miny,&maxx,&maxy);
        if(errorcode!=NO_ERROR)
            return 0;
        // 创建工作图像    
        //给工作图像分配空间,宽度是最大坐标值+1，因为坐标从0开始计数,再+1,保证点在图像内部
        ImageBasicOp::makeAtHost(inimg,maxx+2 ,maxy+2);
        // 把坐标集绘制到图像上,前景255，背景0
        imgcvt.cstConvertToImg(coor,inimg);
        #ifdef DEBUG_IMG
            // 把填充前的图像保存到文件
            ImageBasicOp::copyToHost(inimg);
            ImageBasicOp::writeToFile("coorimg.bmp",inimg);
         #endif
        // ------------------调用图像接口的pieceRealRect（）获得结果---------------------
        pieceRealRect(
            inimg,
            piecenum,
            linenum,
            linethres,
            lineangthres,
            linedisthres,
            rectnum,
            distance,
            percent,
            realrectnum,
            realxypara
        );
        // 回收内存
        ImageBasicOp::deleteImage(inimg);
        return NO_ERROR;
    }
    else
        return INVALID_DATA;
  }
