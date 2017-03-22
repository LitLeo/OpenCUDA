// HoughRec.h 
// 创建人：侯怡婷, 冯振, 曹建立
// 
// Hough 变换检测矩形（HoughRec）
// 功能说明：实现 Hough 变换检测矩形。输入参数为坐标集 guidingset 或者输
//           入图像 inimg，若 guidingset 不为空，则只处理该坐标集;若为
//           空，则对图像 inimg 进行矩形检测。
// 
// 修订历史：
// 2013年01月07日（冯振）
//    初始版本
// 2013年03月21日（冯振）
//    引用 ImageDrawer 类绘制矩形
// 2013年07月21日（曹建立）
//    增加基于平行线组的平行四边形检测方法
// 2013年08月19日（曹建立）
//    增加对图像进行先分片再检测矩形的算法

#ifndef __HOUGHREC_H__
#define __HOUGHREC_H__

#include "Image.h"
#include "CoordiSet.h"
#include "ErrorCode.h"
#include "HoughLine.h"
#include "cuda_runtime.h"
#include "ImageDrawer.h"
#include "ImgConvert.h" 

#include<string>
using namespace std;
// 结构体：recPolarParam（矩形极坐标返回参数）
// 包括两组平行线的参数，两组平行线互相垂直。
typedef struct RecPolarParam_st {
    float theta1;  // 第一组平行线与横轴的夹角。（弧度制）
    float theta2;  // 第二组平行线与横轴的夹角。（弧度制）
    int rho1a;     // 第一组平行线中直线段 a 与坐标原点的距离。
    int rho1b;     // 第一组平行线中直线段 b 与坐标原点的距离。
    int rho2a;     // 第二组平行线中直线段 a 与坐标原点的距离。
    int rho2b;     // 第二组平行线中直线段 b 与坐标原点的距离。
    int votes1;    // 第一组平行线中每条直线的得票数，即 BufHough 矩形中的数据。
    int votes2;    // 第二组平行线中每条直线的得票数，即 BufHough 矩形中的数据。
} RecPolarParam;

// 结构体：recXYParam（矩形 XY 坐标返回参数）
// 包括四个角点及中心点的坐标。
typedef struct RecXYParam_st {
    int x1;      // 第一个角点的列号。
    int y1;      // 第一个角点的行号。
    int x2;      // 第二个角点的列号。
    int y2;      // 第二个角点的行号。
    int x3;      // 第三个角点的列号。
    int y3;      // 第三个角点的行号。
    int x4;      // 第四个角点的列号。
    int y4;      // 第四个角点的行号。
    int xc;      // 中心点的列号。
    int yc;      // 中心点的行号。
    long votes;  // 矩形得票数。
} RecXYParam;

// 类：HoughRec
// 继承自：无
// 实现 Hough 变换检测矩形。输入参数为采用极坐标的直线参数坐标集。
class HoughRec {
private:
    // 识别平行线组时的误差容许范围，邻边夹角误差容许范围，弧度制
    float toloranceAngle;

public:
    // 构造函数：HoughRec
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    HoughRec(){
        toloranceAngle=M_PI/180*3;
    }

    // 构造函数：HoughRec
    // 有参数版本的构造函数，根据需要给定参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    HoughRec(float tolorance){
        toloranceAngle=tolorance;
    }

    // 成员方法：getToloranceAngle（获取角度差阈值）
    // 获取成员变量 toloranceAngle 的值。
    __host__ __device__ float  // 返回值：成员变量 toloranceAngle 的值
    getToloranceAngle() const{
        return this->toloranceAngle;
    } 

    // 成员方法：setToloranceAngle（设置最小角度单位）
    // 设置成员变量 toloranceAngle 的值。
    __host__ __device__ int       // 返回值：函数是否正确执行，
                                  // 若函数正确执行，返回 NO_ERROR。
    setToloranceAngle( float Theta ) // 设定新的角度差阈值。
    {
        toloranceAngle=Theta;
        return NO_ERROR;
    }

    // Host 成员方法：detectParallelogram（Hough 变换检测平行四边形）
    // 对输入的直线集 lineparam 进行平行四边形检测，返回可能的四边形参数
    // 结果以及检测到的四边形数量。
    __host__ int                     // 返回值：函数是否正确执行，若函数
                                     // 正确执行，返回 NO_ERROR。
    detectParallelogram(
        LineParam *lineparam,        // 输入的直线参数集结构体。
        int linenum,                 // 输入的直线的数目。
        int *rectmax,                // 返回检测四边形的最大数量。
        RecPolarParam *recpolarparm, // 返回四边形返回参数结构体。
        float anglelist[],           // 希望检测出的四边形夹角列表
        int anglenum                 // 夹角列表中元素个数
    );

    // Host 成员方法：detectRectangle（Hough 变换检测矩形）
    // 调用detectParallelogram方法，指定夹角列表中只有一个直角。返回倾角-距离参数坐标。
    __host__ int                        // 返回值：函数是否正确执行，若函数
                                        // 正确执行，返回 NO_ERROR。
    detectRectangle(
        LineParam *lineparam,                  // 输入的直线参数集结构体。
        int linemax,                           // 输入的直线的数目。
        int *recsmax,                          // 返回检测四边形的最大数量。
        RecPolarParam *recpolarparm            // 返回四边形返回参数结构体。
    );
    // Host 成员方法：detectRectangle(检测inimg图像中的矩形，放入数组返回)
    __host__ int 
    detectRectangle(
        Image *inimg,               // 输入图像
        int linenum,                // 最大直线数量
        int linethres,              // 直线票数阈值
        float lineangthres,         // 相似直线角度
        int linedisthres,           // 相似直线距离
        int *rectnum,               // 返回矩形数量
        RecXYParam *rectxypara      // 返回矩形xy坐标参数
        );

    // Host 成员方法：detectRectangle(检测CoordiSet中的矩形，放入数组返回)

    __host__ int 
        detectRectangle(
        CoordiSet *coor,               // 输入坐标集
        int linenum,                // 最大直线数量
        int linethres,              // 直线票数阈值
        float lineangthres,         // 相似直线角度
        int linedisthres,           // 相似直线距离
        int *rectnum,               // 返回矩形数量
        RecXYParam *rectxypara      // 返回矩形xy坐标参数
        );

    // Host 成员方法：detectRealRectangle(检测矩形数组中真实矩形数量，
    // 放入数组返回，参照为坐标集)
    __host__ int 
    detectRealRectangle(
        CoordiSet *coor,               // 输入坐标集
        int rectnum,                // 可能矩形数量
        RecXYParam *rectxypara,     //可能矩形参数数组
        int distance,               // 真实直线判定距离
        float percent,              // 真实直线判定阈值
        int *realrectnum,           //真实矩形数量
        RecXYParam *realrectxypara  //真实矩形xy坐标参数
        );

    // Host 成员方法：detectRealRectangle(检测矩形数组中真实矩形数量，
    // 放入数组返回，参照为图像)
    __host__ int 
        detectRealRectangle(
        Image *inimg,               // 输入图像
        int rectnum,                // 可能矩形数量
        RecXYParam *rectxypara,     //可能矩形参数数组
        int distance,               // 真实直线判定距离
        float percent,              // 真实直线判定阈值
        int *realrectnum,           //真实矩形数量
        RecXYParam *realrectxypara  //真实矩形xy坐标参数
        );
    // Host 成员方法：polar2XYparam(角度距离坐标转换成直角坐标)
    __host__ int                            // 返回值： 函数是否正确执行，若函数
                                            // 正确执行，返回 NO_ERROR。
    polar2XYparam (
        RecPolarParam *recpolarparam,        // 输入的矩形极坐标参数。
        RecXYParam *recxyparam,                // 返回的矩形 XY 坐标参数。
        int recnum,                            // 输入的矩形的个数。
        float derho                            // 输入的直线距离步长参数。
    );


    // Host 成员方法：drawRect(把直角坐标四边形绘制到指定图像文件中)
    __host__ int                    // 返回值：函数是否正确执行，若函数
                                    // 正确执行，返回 NO_ERROR。 
    drawRect(
        string filename,            // 要写入的图像文件名
        size_t w,                   // 图像宽度
        size_t h,                   // 图像高度
        RecXYParam recxyparam[],    // 矩形直角坐标数组
        int rectmax                 // 矩形个数
    );

    // Host 成员方法：pieceRealRectImg（分片检测inimg图像中的矩形写入图像文件）
    // 对其边的真实性进行判定，中间结果和最终结果都放入指定的图像文件中
    __host__ int                    // 返回值：函数是否正确执行，若函数
                                    // 正确执行，返回 NO_ERROR。 
    pieceRealRectImg(
        Image *inimg,               // 输入图像
        string lineoutfile1,        // 直线检测中间结果1
        string lineoutfile2,        // 直线检测中间结果2
        string rectoutfile,         // 矩形输出文件名
        int piecenum,               // 分片个数
        int linenum,                // 每个分片中直线最大数量
        int linethres,               // 每个分片中直线票数阈值
        float lineangthres,            // 每个分片中相似直线角度判定阈值
        int linedisthres,            // 每个分片中相似直线距离判定阈值
        int rectnum,                // 每个分片中矩形最大数量
        int distance=3,             // 真实线段判定参数，距离线段多远的点有效
        float percent=0.7           // 真实线段判定参数，多大比例的真实点作为判定阈值
    );
    // Host 成员方法：重载pieceRealRectImg（分片检测coor坐标集合中的矩形写入图像文件）
    // 对其边的真实性进行判定，中间结果和最终结果都放入指定的图像文件中
    __host__ int                    // 返回值：函数是否正确执行，若函数
    // 正确执行，返回 NO_ERROR。 
    pieceRealRectImg(
    CoordiSet* coor,               // 输入坐标集
    string lineoutfile1,        // 直线检测中间结果1
    string lineoutfile2,        // 直线检测中间结果2
    string rectoutfile,         // 矩形输出文件名
    int piecenum,               // 分片个数
    int linenum,                // 每个分片中直线最大数量
    int linethres,               // 每个分片中直线票数阈值
    float lineangthres,            // 每个分片中相似直线角度判定阈值
    int linedisthres,            // 每个分片中相似直线距离判定阈值
    int rectnum,                // 每个分片中矩形最大数量
    int distance=3,             // 真实线段判定参数，距离线段多远的点有效
    float percent=0.7           // 真实线段判定参数，多大比例的真实点作为判定阈值
    );
    // Host 成员方法：pieceRealRect(分片检测inimg图像中的矩形，放入数组返回)
    // 分片检测inFile中的矩形并对其边的真实性进行判定,结果放入矩形直角坐标参数数组中返回
    __host__ int 
    pieceRealRect(
        Image *inimg,               // 输入图像
        int piecenum,                // 分片个数
        int linenum,                 // 每个分片中直线最大数量
        int linethres,               // 每个分片中直线票数阈值
        float lineangthres,            // 每个分片中相似直线角度判定阈值
        int linedisthres,            // 每个分片中相似直线距离判定阈值
        int rectnum,                 // 每个分片中矩形最大数量
        int distance,                // 真实线段判定参数，距离线段多远的点有效
        float percent,               // 真实线段判定参数，多大比例的真实点作为判定阈值
        int *realrectnum,            // 用于返回最终检测出的矩形数量
        RecXYParam *realxyparam       // 用于返回最终检测出的矩形参数数组
    );

    // Host 成员方法：重载pieceRealRect(分片检测coor坐标集中的矩形，放入数组返回)
    // 分片检测inFile中的矩形并对其边的真实性进行判定,结果放入矩形直角坐标参数数组中返回
    __host__ int 
        pieceRealRect(
        CoordiSet* coor,             // 输入坐标集
        int piecenum,                // 分片个数
        int linenum,                 // 每个分片中直线最大数量
        int linethres,               // 每个分片中直线票数阈值
        float lineangthres,            // 每个分片中相似直线角度判定阈值
        int linedisthres,            // 每个分片中相似直线距离判定阈值
        int rectnum,                 // 每个分片中矩形最大数量
        int distance,                // 真实线段判定参数，距离线段多远的点有效
        float percent,               // 真实线段判定参数，多大比例的真实点作为判定阈值
        int *realrectnum,            // 用于返回最终检测出的矩形数量
        RecXYParam *realxyparam       // 用于返回最终检测出的矩形参数数组
        );
}; 

#endif
