// HoughLine.h 
// 创建人：侯怡婷
//
// Hough变换检测直线（HoughLine）
// 功能说明：实现 Hough 变换检测直线。输入参数为坐标集 guidingset 或者输
//         入图像 inimg，若 guidingset 不为空，则优先处理该坐标集;若为
//         空，则对图像 inimg 进行直线检测。直线采用角度-距离形式表示，检
//         测的角度范围为 0 - π。
//         需要注意，如果的是图像，且该图像roi区域不是整个图像，则得到直线
//         参数是以roi区域左上角为原点的，如果需要全局坐标下参数，则需要调用
//         getGlobalParam（）函数转换一下。如果处理的是coordiset坐标集则因为
//         coordiset不存在ROI区域，得到的直线参数永远是全局坐标。
// 修订历史：
// 2012年09月15日（侯怡婷）
//     初始版本
// 2012年10月15日（侯怡婷）
//     完成计算得票数的 kernel 函数。
// 2012年10月24日（侯怡婷）
//     对头文件隐藏的错误进行修改。
// 2012年10月28日（侯怡婷）
//     完成直线检测的剩余操作。
// 2012年10月31日（侯怡婷）
//     更改注释规范。
// 2012年11月05日（侯怡婷）
//     完善了代码执行错误时的内存释放，合并可以统一申请的内存空间。
// 2012年11月15日（侯怡婷）
//     加入对坐标集的 Hough 变换处理。
// 2012年11月19日（侯怡婷）
//     对坐标集的 Hough 变换进行测试并修改注释规范。
// 2012年12月02日（侯怡婷）
//     对 Hough 变换中计算直线距离参数的公式做修改，以便检测
//     出更多的直线。
// 2012年12月04日（侯怡婷）
//     增加没有输出图像的接口，算法只返回检测出的直线的参数。
// 2012年12月05日（侯怡婷）
//     增加只处理点集的接口，算法只返回检测出的直线的参数。
// 2012年12月28日（侯怡婷）
//     删除申请的多余的 Device 端空间。
// 2013年01月06日（侯怡婷）
//     修改部分注释。
// 2013年01月13日（侯怡婷）
//     修改计算局部最大值时的错误。
// 2013年01月15日（侯怡婷）
//     增加根据距离判断检测出的两条直线是否是一条。
// 2013年08月15日（侯怡婷）
//     增加根据距离判断检测出的两条直线是否是一条。
// 2013.07.02 (曹建立)修复相似直线合并未时未考虑两直线相差180度的bug这也解释
//     了为什么之前结果图中其他方向直线顺利合并，而垂直直线经常是多条未合并现象
// 2013年08月19日（曹建立）
//     增加线段真实性判断算法
// 2013年10月13日（曹建立）
//     增加参数为图像时，ROI区域不是全部图像时，局部坐标参数转化全局坐标参数函数

#ifndef __HOUGHLINE_H__
#define __HOUGHLINE_H__

#include "Image.h"
#include "CoordiSet.h"
#include "ErrorCode.h"

// 宏：M_PI
// π 值。对于某些操作系统，M_PI 可能没有定义，这里补充定义 M_PI。
#ifndef M_PI
#define M_PI 3.14159265359
#endif 
    
// 结构体：lineParam（直线返回参数）
// 将直线的参数：与横轴的夹角，与坐标轴原点的距离等定义为
// 结构体，作为函数最终的输出结果。
typedef struct LineParam_st {
    float angle;   // 与横轴的夹角，角度采用弧度制。
    int distance;  // 与坐标原点的距离。
    int votes;     // 得票数，即 BufHough 矩阵中的数据。
} LineParam;

// 类：HoughLine
// 继承自：无
// 实现 Hough 变换检测直线。输入参数为坐标集 guidingset 或
// 者输入图像 inimg，若 guidingset 不为空，则只处理该坐标集；
// 若为空，则对图像 inimg 进行直线检测。直线采用极坐标
// 形式表示，检测的角度范围为 0 —— π。
class HoughLine {

protected:
    
    // 成员变量： detheta（最小角度单位，弧度制）
    // 描述在检测直线时，每一次的角度增量，角度采用弧度制。
    double detheta;
    
    // 成员变量： derho（最小距离单位）
    // 描述在检测直线时，每一次的距离增量。
    double derho;

    // 成员变量：threshold（直线阈值）
    // 若累加器中相应的累计值大于该参数，则认为是
    // 一条直线，则函数返回这条线段。
    int threshold;

    // 成员变量：thresang（区别两条直线的最小角度，弧度制）
    // 若两条检测出来的直线的参数 angle，
    // 即与坐标轴横轴的夹角小于该参数，则认为
    // 这两条直线实质上是一条。
    float thresang;

    // 成员变量：thresdis（区别两条直线的最小距离）
    // 若两条检测出来的直线的参数 distance，
    // 即与坐标原点的距离小于该参数，则认为
    // 这两条直线实质上是一条。
    int thresdis;
	


public:
	// 构造函数：HoughLine
	// 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    HoughLine()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->detheta = M_PI / 180;  // 最小角度单位设置为 π / 180。
        this->derho = 1.0;           // 最小距离单位设置为 1.0。
        this->threshold = 90;        // 直线阈值默认为 90。
        this->thresang = M_PI/180*5.0f;// 区别两条直线的最小角度设置为 5.0(弧度制)
        this->thresdis = 50;         // 区别两条直线的最小距离设置为 50。
    }

    // 构造函数：HoughLine
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    HoughLine(
            double detheta,  // 最小角度单位
            double derho,    // 最小距离单位
            int threshold,   // 直线阈值
            float thresang,  // 区别两条直线的最小角度(弧度制)。
            int thresdis     // 区别两条直线的最小距离。
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中
        // 给了非法的初始值而使系统进入一个未知的状态。
        this->detheta = M_PI / 180;  // 最小角度单位设置为 π / 180。
        this->derho = 1.0;           // 最小距离单位设置为 1.0。 
        this->threshold = 90;        // 直线阈值默认为 90。
        this->thresang =M_PI/180*5.0;// 区别两条直线的最小角度设置为 5.0(弧度制)。
        this->thresdis = 50;         // 区别两条直线的最小距离设置为 50。

        // 根据参数列表中的值设定成员变量的初值
        setDeTheta(detheta);
        setDeRho(derho);
        setThreshold(threshold);
        setThresAng(thresang);
        setThresDis(thresdis);
    }
   
    // 成员方法：getDeTheta（获取最小角度单位）
    // 获取成员变量 detheta 的值。
    __host__ __device__ double  // 返回值：成员变量 detheta 的值
    getDeTheta() const
    {
        // 返回 detheta 成员变量的值。
        return this->detheta;
    } 

    // 成员方法：setDeTheta（设置最小角度单位）
    // 设置成员变量 detheta 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setDeTheta(
            double detheta   // 设定新的角度增量
    ) { 
        // 将 detheta 成员变量赋成新值
        this->detheta = detheta;

        return NO_ERROR;
    }

    // 成员方法：getDeRho（获取最小距离单位）
    // 获取成员变量 derho 的值。
    __host__ __device__ double  // 返回值：成员变量 derho 的值
    getDeRho() const
    {
        // 返回 detheta 成员变量的值。
        return this->derho;
    } 

    // 成员方法：setDeRho（设置最小距离单位）
    // 设置成员变量 derho 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setDeRho(
            double derho     // 设定新的距离增量
    ) { 
        // 将 derho 成员变量赋成新值
        this->derho = derho;

        return NO_ERROR;
    }

    // 成员方法：getThreshold（获取直线阈值）
    // 获取成员变量 threshold 的值。
    __host__ __device__ int  // 返回值：成员变量 threshold 的值
    getThreshold() const
    {
        // 返回 threshold 成员变量的值。
        return this->threshold;
    }  

    // 成员方法：setThreshold（设置直线阈值）
    // 设置成员变量 threshold 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setThreshold(
            int threshold    // 设定新的直线阈值
    ) {
        // 将 threshold 成员变量赋成新值
        this->threshold = threshold;

        return NO_ERROR;
    }

    // 成员方法：getThresAng（获取区别两条直线的最小角度）
    // 获取成员变量 thresang 的值。
    __host__ __device__ float  // 返回值：成员变量 thresang 的值
    getThresAng() const
    {
        // 返回 thresang 成员变量的值。
        return this->thresang;
    }  

    // 成员方法：setThresAng（设置区别两条直线的最小角度）
    // 设置成员变量 thresang 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setThresAng(
            float thresang   // 设定新的区别两条直线的最小角度
    ) {
        // 将 thresang 成员变量赋成新值
        this->thresang = thresang;

        return NO_ERROR;
    }
 
    // 成员方法：getThresDis（获取区别两条直线的最小距离）
    // 获取成员变量 thresdis 的值。
    __host__ __device__ int  // 返回值：成员变量 thresdis 的值
    getThresDis() const
    {
        // 返回 thresdis 成员变量的值。
        return this->thresdis;
    }  

    // 成员方法：setThresDis（设置区别两条直线的最小距离）
    // 设置成员变量 thresdis 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setThresDis(
            int thresdis    // 设定区别两条直线的最小距离
    ) {
        // 将 thresdis 成员变量赋成新值
        this->thresdis = thresdis;

        return NO_ERROR;
    }
    
    // Host 成员方法：houghlineCor（Hough 变换检测直线）
    // 对输入点集 guidingset 进行直线检测，返回最终的直线检测结果
    // 以及检测到的直线数量。
    __host__ int                    // 返回值：函数是否正确执行，若函数
                                    // 正确执行，返回NO_ERROR。
    houghLineCor(
            CoordiSet *guidingset,  // 输入坐标集
            int *linesmax,          // 检测直线的最大数量
            LineParam *lineparam    // 直线返回参数结构体
    );
    
    // Host 成员方法：houghline（Hough 变换检测直线）
    // 若输入的坐标集 guidingset 不为空，则只处理该坐标集；若该坐标集为空，则 
    // Hough 变换就处理输入图像的 ROI 区域，并把最终检测直线的结果返回
    // 到定义的参数结构体中，并且返回检测到的直线的数量。输入图像可以为空。
    // 如果图像roi区域不是整个图像，则检测得到直线参数是以roi区域左上角为原点的，
    // 如果需要全局坐标下参数，则需要调用getGlobalParam（）函数转换一下。
    __host__ int                    // 返回值：函数是否正确执行，若函数
                                    // 正确执行，返回 NO_ERROR。
    houghLine(
            Image *inimg,           // 输入图像
            CoordiSet *guidingset,  // 输入坐标集
            int *linesmax,          // 检测直线的最大数量
            LineParam *lineparam    // 直线返回参数结构体
    );

    // Host 成员方法：houghlineimg（Hough 变换检测直线）
    // 输入图像不能为空，若输入的坐标集 guidingset 不为空，则只处理该坐标集；
    // 若该坐标集为空，则 Hough 变换就处理输入图像的 ROI 区域，并把最终检测
    // 直线的结果返回到定义的参数结构体中，返回检测到的直线的数量,以及输出图像。
    // 如果图像roi区域不是整个图像，则检测得到直线参数是以roi区域左上角为原点的，
    // 如果需要全局坐标下参数，则需要调用getGlobalParam（）函数转换一下。
    __host__ int                    // 返回值：函数是否正确执行，若函数
                                    // 正确执行，返回 NO_ERROR。
    houghLineImg(
            Image *inimg,           // 输入图像
            CoordiSet *guidingset,  // 输入坐标集
            Image *outimg,          // 输出图像
            int *linesmax,          // 检测直线的最大数量
            LineParam *lineparam    // 直线返回参数结构体
    );
    // Host 成员方法：realLine（判断给出线段的真实性）
    // 统计给出图像中两端点间有效点个数，大于参数给出的比例，认为线段真实返回真。
    __host__ bool                    // 返回值：若两点间存在真实直线，返回真，
                                     // 否则返回假
        realLine(
        Image *inimg,             // 输入图像
        int x1,                   // 要判断的线段两端点坐标
        int y1,                   // 要判断的线段两端点坐标
        int x2,                   // 要判断的线段两端点坐标
        int y2,                   // 要判断的线段两端点坐标
        float threshold,          // 距离线段多远的点认为是线段有效点，单位像素
        float thresperc           // 线段真实性判定阈值，线段上有效点和线段理论上应该有的
        // 点的比值超过此阈值，认为线段真实存在
        );

    // Host 成员方法：getGlobalParam（ROI局部直线参数转换成为全局坐标下的参数）
    // 把 hough 变换检测到的直线参数转换成为全局坐标下的参数
    __host__ int                     // 返回值：函数是否正确执行，若函数
                                     // 否则返回假
        getGlobalParam(
        Image *inimg,             // 输入图像
        int *linesmax,          // 检测直线的最大数量
        LineParam *lineparam    // 直线返回参数结构体
        );

};
#endif

