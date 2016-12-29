// CurveTracing
// 实现的曲线跟踪

#include "CurveTracing.h"

#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#include "Template.h"
#include "TemplateFactory.h"

// 宏：CURVE_VALUE(曲线最大数目）
// 设置图像能获得的曲线最大数目
#define CURVE_VALUE 1000

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X    32
#define DEF_BLOCK_Y     8


// Kernel 函数：_traverseKer（并行遍历图像得到端点数组和交点数组，并且得到去掉
// 交点后的输出图像）
// 遍历图像，得到曲线的所有端点坐标和交点坐标，并且得到去掉交点后的输出图像，
// 对每个像素点取其周围八领域像素点，如果八领域像素点的个数为 1，则这个为端点，
// 若八领域像素点的个数为大于等于 3，则认为这个点作为伪交点，存储起来，这些伪
// 交点中有部分是真正的交点，后面计算需要从一堆伪交点中得到真正的交点。
static __global__ void     // Kernel 函数无返回值
_traverseKer(
        ImageCuda inimg,   // 输入图像
        ImageCuda outimg,  // 去掉交点后的输出图像
        int *array1_dev,   // 存储端点的数组
        int *array2_dev,   // 存储交点的数组
        Template boxtpl    // 3 * 3 领域模板
);

// Kernel 函数：_traverseKerNew（遍历图像，得到图像上所有的像素点）
// 遍历图像，保存图像上所有灰度值不为 0 的像素点，主要用于 CPU 串行代码中第二次
// 遍历的实现 
static __global__ void    // Kernel 函数无返回值
_traverseKerNew(
        ImageCuda inimg,  // 输入图像
        int *array1_dev   // 存储端点的数组
);

// Host 函数：traverse（遍历图像，得到端点坐标、交点坐标及去掉交点后的图像）
// 遍历图像，得到曲线的所有端点坐标和交点坐标，并且得到去掉交点后的输出图像，
// 对每个像素点取其周围八领域像素点，如果八领域像素点的个数为 1，则这个为端点，
// 若八领域像素点的个数为大于等于 3，则认为这个点作为伪交点，存储起来，这些伪
// 交点中有部分是真正的交点，后面计算需要从一堆伪交点中得到真正的交点。主要是
// 用于 CPU 串行代码的实现中处理
static __host__ void               // 无返回值
traverse(
        DynamicArrays &Vertex,     // 存储端点的动态数组
        DynamicArrays &Intersect,  // 存储伪交点的动态数组 
        Image *inimg,              // 输入图像
        Image *outimg,             // 输出图像
        int *tpl                   // 八领域模板
);

// Host 函数：traverseNew（遍历图像，得到图像上所有的像素点）
// 遍历图像，保存图像上所有灰度值不为 0 的像素点，主要用于 CPU 串行代码中第二次
// 遍历的实现 
static __host__ void               // 无返回值
traverseNew(
        DynamicArrays &array,      //存储点的坐标
        Image *inimg               // 输入图像
);

// Host 函数：getCurve（得到去掉交点后的所有曲线段）
// 递归调用函数，得到去掉交点后的所有曲线段，并且这些曲线段都是非闭合曲线
static __host__ void              // 无返回值
getCurve(
        DynamicArrays *pcurve,    // 存储所有提取到的非闭合曲线
        int &test,                // 检测某端点开始的曲线是否已提取过
        int count,                // 曲线条数
        Image *img,               // 输入图像，待提取曲线的图像
        int *mark,                // 标志数组，大小为图像大小，表示像素点是否
                                  // 访问，初始都为 0，如果访问则对应位置设为 1
        int *tpl,                 // 八领域模板
        int Vx,                   // 提取的曲线起点 x 坐标
        int Vy                    // 提取的曲线起点 y 坐标
);

// Host 函数：insectClassify（从得到的一堆交点中，进行分类，确定交点个数） 
// 递归调用函数，实现原先得到的一堆交点进行分类，每一类是一部分点集，并且同一类
// 的点集是连通的，这些点集中可以找到一个合适的交点，同时根据分类的结果可以得到
// 交点的个数，有多少类就有多少交点
static __host__ void                // 无返回值
insectClassify(
        int x,                      // 点的 x 坐标
        int y,                      // 点的 y 坐标
        DynamicArrays &Intersect,   // 存储交点的动态数组
        DynamicArrays *insect,      // 存储分类的结果
        int sectnum,                // 交点个数，即分类的类数
        int *tpl                    // 八领域模板
);

// Host 函数：makeCur（根据两点坐标得到一条曲线） 
// 根据两点坐标得到一条曲线，两个点的连线方式为从第一个点开始，先从对角线往
// 第二个点移动，如果第二个点的 x 或者 y 坐标的值与对角线 45° 移动的对应坐标值
// 一样，则沿着 x 或者 y 坐标移动直到重合，从而得到一条简短曲线
static __host__ void             // 无返回值
makeCur(
        DynamicArrays &cur,  // 存储得到的曲线
        int dx1,             // 曲线第一个点的 x 坐标
        int dy1,             // 曲线第一个点的 y 坐标
        int dx2,             // 曲线第一个点的 x 坐标
        int dy2              // 曲线第一个点的 y 坐标
);

// Host 函数：interAssemble（交点曲线与原先得到的曲线进行重组，得到重组后曲线）
// 根据得到的交点扩散出的曲线和原先得到的非闭合曲线进行重组，得到重组后曲线，
// 以便之后的曲线还原
static __host__ void             // 无返回值 
interAssemble(
        DynamicArrays *pcurve,   // 非闭合曲线集
        int count,               // 曲线条数
        DynamicArrays *insect,   // 交点分类的结果
        int sectnum,             // 交点曲线条数
        DynamicArrays realsect,  // 真正的交点数组
        int *tpl                 // 八领域模板
);

// Host 函数：bpConnect（根据用户输入的半径得到近域点集）
// 根据用户输入的半径得到近域点集，并且更新端点动态数组
static __host__ void
bpConnect(
        DynamicArrays *pcurve,  // 输入的曲线集
        int count,              // 输入的曲线集条数
        int radius,             // 半径大小参数
        DynamicArrays *psect,   // 得到新增加的近域点集
        int *pcount,            // 得到新增加的交点个数
        DynamicArrays &Vertex   // 存储端点的动态数组 
);

// Host 函数：AidNorm（判断两个端点之间距离是否在领域大小内，若在则添加到点集中）
// 判断两个端点之间距离是否在领域大小内，如果在领域半径大小内，则把找到的端点加
// 入到新增加的近域点集
static __host__ bool            // 返回值为 bool 型，如果表示相同就返回 true，
                                // 否则返回 false
AidNorm(
        DynamicArrays *pcurve,  // 输入的曲线集
        int i,                  // 从编号为 i 的曲线往后搜索
        int count,              // 输入的曲线集条数
        DynamicArrays *psect,   // 新增加的近域点集
        int pcount,             // 新增加的交点个数
        int radius,             // 半径大小参数
        DynamicArrays &Vertex,  // 存储端点的动态数组 
        int x, int y            // 曲线的端点坐标
);

// Host 函数：pcurveAcord（根据坐标得到曲线的编号）
// 根据曲线的端点坐标查找曲线的编号，遍历所有曲线的端点，查找是否存在和给定的坐
// 标相等的点，则得到相应的返回结果。
static __host__ int              // 返回值，如果找到的是曲线首部返回 0，
                                 // 如果找到的是曲线尾部则返回 1，否则返回 -1。
pcurveAcord(
        DynamicArrays *pcurve,   // 输入的曲线集
        int count,               // 输入的曲线集条数
        int &location,           // 得到曲线的编号
        int x, int y             // 端点坐标 
);

// Host 函数：verAssemble（断点重组，根据近域点集重组曲线集）
// 根据近域点集重组曲线集，根据近域的每个集合里的那些点，进行计算，得到其中
// 最合适的点作为中心点，这个中心点也即是一个新产生的交点，然后发散出去多条曲线，
// 把这些曲线更新到原来的曲线集中。更抽象成层的含义，断点的重组，根据用户输入的
// 半径进行曲线端点组合，如果两个端点离得太近就变成一段连续的曲线，
// 达到的端点的连接性。
static __host__ void 
verAssemble(
        DynamicArrays *pcurve,   // 曲线集
        int count,               // 曲线集的条数
        DynamicArrays *psect,    // 近域点集
        int pcount,              // 近域交点个数
        DynamicArrays &realsect  // 更新交点集合
);

// Host 函数：IsFindPoint（判断坐标是不是坐标集动态数组里的点）
static __host__ bool           // 返回值为 bool 型，如果表示相同就返回 true，
                               // 否则返回 false
IsFindPoint(
        DynamicArrays &array,  // 判断该坐标是不是动态数组里的点集
        int x, int y           // 坐标
);
                                 
// Host 函数：makeNode（根据曲线的起点和终点，以及边的情况，得到曲线的编号）
// 根据曲线的起点和终点，以及边的情况，从而得到曲线的编号，并且编号的起点和终点
// 是唯一的，也不会和边的编号重复，为之后构图提供方便
static __host__ void            // 无返回值
makeNode(
        DynamicArrays *pcurve,  // 输入的曲线集
        int count,              // 曲线的条数
        DynamicArrays *pcurno   // 存储曲线的编号
);

// Host 函数：openCurvePath（得到非闭合曲线编号序列）
// 根据图的搜索得到非闭合曲线对应的编号序列，用于得到从 start 到 end 的所有路径
static __host__ void                 // 无返回值
openCurvePath(
        DynamicArrays *opencurnode,  // 存储非闭合曲线编号集
        int *openNum,                // 得到非闭合曲线的条数
        Graph *G,                    // 曲线构建的图
        int start,                   // 搜索的起点
        int end                      // 搜索的终点
);

// Host 函数：closeCurvePath（得到闭合曲线编号序列）
// 根据图的搜索得到闭合曲线对应的编号序列，用于得到从 start 到 end 的所有路径
static __host__ void                  // 无返回值
closeCurvePath(
        DynamicArrays *closecurnode,  // 存储闭合曲线编号集
        int *closeNum,                // 得到闭合曲线的条数
        Graph *G,                     // 曲线构建的图
        int insect                    // 搜索的起点，闭合曲线起点和终点一样
);

// Host 函数：IsArrayEqual（判断两个动态数组表示的曲线是否表示同一条曲线）
// 判断两个动态数组表示的曲线是否表示同一条曲线，首先得到的是曲线编号，且不会
// 出现编号顺序一致的数组，可能会出现数量和编号一样但是顺序不一样的数组，排序后
// 比较结果，主要用于闭合曲线的提取，由于闭合曲线头尾编号一样，排序比较的时候
// 不算最后编号数
static __host__ bool            // 返回值为 bool 型，如果表示相同就返回 true，
                                // 否则返回 false
IsArrayEqual(
        DynamicArrays object1,  // 动态数组1
        DynamicArrays object2   // 动态数组2
);

// Host 函数：getPointNo（根据坐标对得到数组内对应编号）
// 通过得到的曲线序列，及首尾编号，得到点坐标对的数组对应编号
static __host__ void            // 无返回值
getPointNo(
        DynamicArrays *pcurve,  // 提取的曲线序列
        int count,              // 曲线数目
        DynamicArrays *pcurno,  // 与曲线序列相对应的首尾编号
        DynamicArrays &array,   // 点坐标对数组
        DynamicArrays &arrayno  // 存储得到的对应编号
);

// Host 函数：getCurveNonFormat（得到非格式化输出曲线数据有序序列）
// 通过曲线编号集合和首尾编号集得到非格式化输出曲线数据有序序列
static __host__ void             // 无返回值
getCurveNonFormat(
        DynamicArrays *curnode,  // 曲线编号集
        DynamicArrays *pcurve,   // 提取的曲线序列
        int count,               // 提取的曲线序列的数量
        DynamicArrays *pcurno,   // 与曲线序列相对应的首尾编号
        DynamicArrays *cur,      // 最终得到的曲线非格式输出数据
        int num,                 // 曲线的数量
        bool close = false       // 标志闭合还是非闭合曲线，默认为非闭合     
);


// Host 函数：traverse（遍历图像，得到端点坐标、交点坐标及去掉交点后的图像）
static __host__ void traverse(DynamicArrays &Vertex, DynamicArrays &Intersect, 
                              Image *inimg, Image *outimg,int *tpl)
{
    // 定义临时变量，用于循环
    int i, j, k;
    // 定义临时变量，存储八领域的值
    int dx, dy;
 
    // 对每一个像素值不为 0 的像素点进行八领域处理
    for (i = 0; i < inimg->height; i++) {
        for(j = 0; j < inimg->width; j++) {
            // 如果该像素点为 0 则扫描下一个像素点
            if (inimg->imgData[i * inimg->width + j] == 0) {
                outimg->imgData[i * inimg->width + j] = 0;
                continue;
            }
            // 定义变量并且初始化为 0，用于取八领域下标
            int m = 0;
            // 定义变量并且初始化为 0，用于得到八领域内有多少个像素值不为 0 的点
            int flag = 0; 
            for(k = 0; k < 8; k++) {
                dx = j + tpl[m++];
                dy = i + tpl[m++];
                // 符合要求的八领域内的点的像素值如果不为 0，就累加到 flag 中
                if (dx >= 0 && dx < inimg->width &&
                    dy >= 0 && dy < inimg->height) {
                    if (inimg->imgData[dy * inimg->width + dx] != 0) {
                        flag++;
                    } 
                }
            }
            // 如果 flag 为 0，表示该像素八领域没有不为 0 的像素点，则该点是
            // 孤立点，则给对应输出图像在该处赋值为 0
            if (flag == 0) {       
                outimg->imgData[i * inimg->width + j] = 0;
            // 如果 flag 为 1，表示该像素八领域有一个不为 0 的像素点，则该点是
            // 曲线端点，并给对应输出图像在该处赋值原图像对应点像素值
            } else if (flag == 1) {
                Vertex.addElem(j);
                Vertex.addElem(i);
                outimg->imgData[i * inimg->width + j] =
                        inimg->imgData[i * inimg->width + j];
            // 如果 flag 大于等于 3，表示该像素点作为曲线交点，则给对应输出图像
            // 在该处赋值为 0
            } else if (flag >= 3) {
                Intersect.addElem(j);
                Intersect.addElem(i);
                outimg->imgData[i * inimg->width + j] = 0;
            // 否则flag则为 2，表示该像素点作为曲线上的点,并给对应输出图像在该处
            // 赋值原图像对应点像素值
            } else {
                outimg->imgData[i * inimg->width + j] =
                        inimg->imgData[i * inimg->width + j];
            }
        }
    }
}

// Host 函数：traverseNew（遍历图像，得到图像上所有的像素点）
static __host__ void traverseNew(DynamicArrays &array, Image *inimg)
{

    // 定义临时变量，用于循环
    int i, j;
 
    // 对每一个像素值不为 0 的像素点进行八领域处理
    for (i = 0; i < inimg->height; i++) {
        for(j = 0; j < inimg->width; j++) {
            // 如果该像素点不为 0 则保存
            if (inimg->imgData[i * inimg->width + j] != 0) {
                // 得到所有灰度值不为 0 的像素点
                array.addElem(j);
                array.addElem(i);
            }
        }
    }
}

// Host 函数：getCurve（得到去掉交点后的所有曲线段）
static __host__ void getCurve(DynamicArrays *pcurve, int &test, int count,
                              Image *img, int *mark, int *tpl, int Vx, int Vy)
{
    // 标志点是否已经访问过，如果访问过，test 加 1，并且退出，主要是判断该端点
    // 是否和另一个端点是同一条曲线，如果是就不需要再重复提取
    if (mark[Vy * img->width + Vx] == 1) {
        test++;
        return;
    }
    // 定义临时变量，存储八领域的值
    int dx, dy;
    int j = 0;  // 定义变量，用于循环
    // 定义标志，表示八领域是否还有像素值不为 0 的点
    int flag = 0;
    
    // 把该点的坐标值加入第 count 条曲线中，并且设置标志该点已经访问过
    pcurve[count].addElem(Vx);
    pcurve[count].addElem(Vy);
    mark[Vy * img->width + Vx] = 1;
    
    // 按顺时针访问八领域的像素点
    for(int i = 0; i < 8; i++) {
        dx = Vx + tpl[j++];
        dy = Vy + tpl[j++];
        // 得到第一个不为 0 并且没有访问过的像素点就退出循环，并且标志 flag 为 1
        if (img->imgData[dy * img->width + dx] != 0 &&
            mark[dy * img->width + dx] != 1) {
            flag = 1;
            break;
        }
    }
    // 如果 flag 为 1，说明找到了一个曲线上的点，以该点递归调用函数
    if (flag == 1) {
        getCurve(pcurve, test, count, img, mark, tpl, dx, dy);
    }
    // 如果找不到了，说明已经全部搜索完，退出
    return;  
}

// Host 函数：insectClassify（从得到的一堆交点中，进行分类，确定交点个数）
static __host__ void insectClassify(int x, int y, DynamicArrays &Intersect, 
                                    DynamicArrays *insect, int sectnum, 
                                    int *tpl)
{
    // 把 x，y 坐标加入交点曲线中
    insect[sectnum - 1].addElem(x);  
    insect[sectnum - 1].addElem(y);
    // 加入完后就删除交点数组中的 x，y 坐标
    Intersect.delElem(x, y);
    //
    if (Intersect.getSize() == 0)
        return;
    // 定义临时变量，存储八领域的坐标点
    int dx, dy;
    for(int i = 0; i < 16; i += 2) {
        dx = x + tpl[i];
        dy = y + tpl[i + 1];
        // 寻找到交点中是否有和八领域一样的坐标点，若有，则递归调用函数
        for(int j = 0; j < Intersect.getSize(); j += 2) {
            if (dx == Intersect[j] && dy == Intersect[j + 1]) {
                insectClassify(dx, dy, Intersect, insect, sectnum, tpl);
            }
        }
    }
    // 返回
    return;
}

// Host 函数：makeCur（根据两点坐标得到一条曲线）
static __host__ void makeCur(DynamicArrays &cur,
                             int dx1, int dy1, int dx2, int dy2)
{
    // 定义临时变量，存储坐标值
    int x, y;
    // 首先把起始点加入临时曲线中
    cur.addElem(dx1);
    cur.addElem(dy1);
    
    // 如果两坐标值一样，则返回，无须后续步骤
    if (dx1 == dx2 && dy1 == dy2)
        return;
        
    // 分别计算两坐标值的差
    int m = dx1 - dx2, n = dy1 - dy2;

    // 设置起始点
    x = dx1;
    y = dy1;

    // 通过差值开始给交点曲线赋值，首先通过差值相对可以分成四个象限，第一、
    // 第二、第三、第四象限，并且以第一个点为中心开始。
    // 如果 m >= 0 并且 n >= 0，则表示第二个点相对第一个点在第一象限或者坐标轴
    if (m >= 0 && n >= 0) {
        // 计算坐标差值的差
        int d = m - n;
        // 根据差值的差给交点曲线赋值
        if (d >= 0) {
            for (int c = 0; c < n; c++) {
                x--;
                y--;
                cur.addElem(x);
                cur.addElem(y);
            }
            for (int c = 0; c < d; c++) {
                x--;
                cur.addElem(x);
                cur.addElem(y);
            }
        } else {
            for (int c = 0; c < m; c++) {
                x--;
                y--;
                cur.addElem(x);
                cur.addElem(y);
            }
            for (int c = 0; c < -d; c++) {
                y--;
                cur.addElem(x);
                cur.addElem(y);
            }
                     
        }
    // 如果 m >= 0 并且 n < 0，则表示第二个点相对第一个点在第四象限或者坐标轴
    } else if (m >= 0 && n < 0) {
        n = -n;
        int d = m - n;
        if (d >= 0) {
            for (int c = 0; c < n; c++) {
                x--;
                y++;
                cur.addElem(x);
                cur.addElem(y);
            }
            for (int c = 0; c < d; c++) {
                x--;
                cur.addElem(x);
                cur.addElem(y);
            }
        } else {
            for (int c = 0; c < m; c++) {
                x--;
                y++;
                cur.addElem(x);
                cur.addElem(y);
            }
            for (int c = 0; c < -d; c++) {
                y++;
                cur.addElem(x);
                cur.addElem(y);
            }  
        }
    // 如果 m < 0 并且 n >= 0，则表示第二个点相对第一个点在第二象限或者坐标轴
    } else if (m < 0 && n >= 0) {
        m = -m;
        int d = m - n;
        if (d >= 0) {
            for (int c = 0; c < n; c++) {
                x++;
                y--;
                cur.addElem(x);
                cur.addElem(y);
            }
            for (int c = 0; c < d; c++) {
                x++;
                cur.addElem(x);
                cur.addElem(y);
            }
        } else {
            for (int c = 0; c < m; c++) {
                x++;
                y--;
                cur.addElem(x);
                cur.addElem(y);
            }
            for (int c = 0; c < -d; c++) {
                y--;
                cur.addElem(x);
                cur.addElem(y);
            }
                     
        }
    // 否则 m < 0 并且 n < 0，则表示第二个点相对第一个点在第三象限
    } else {
        m = -m; n = -n;
        int d = m - n;
        if (d >= 0) {
            for (int c = 0; c < n; c++) {
                x++;
                y++;
                cur.addElem(x);
                cur.addElem(y);
            }
            for (int c = 0; c < d; c++) {
                x++;
                cur.addElem(x);
                cur.addElem(y);
            }
        } else {
            for (int c = 0; c < m; c++) {
                x++;
                y++;
                cur.addElem(x);
                cur.addElem(y);
            }
            for (int c = 0; c < -d; c++) {
                y++;
                cur.addElem(x);
                cur.addElem(y);
            }
                     
        }
        
    }
}

// Host 函数：interAssemble（交点曲线与原先得到的曲线进行重组，得到重组后曲线）
static __host__ void interAssemble(DynamicArrays *pcurve, int count,
                                   DynamicArrays *insect, int sectnum,
                                   DynamicArrays realsect, int *tpl)
{
    // 如果没有交点则直接返回
    if (realsect.getSize() == 0)
        return;
        
    // 定义临时变量
    int i, j, k, x1, y1, x2, y2, dx1, dy1, dx2, dy2, num, num2;
    int mark1, mark2, flag1, flag2;
    // 对每一条得到的曲线，先得到其首尾端点，进行八领域寻找交点曲线的尾端点，
    // 如果找到就把交点曲线添加到原曲线中，实现交点曲线与原先得到的曲线重组
    for(i = 0; i < count; i++) {
        // 初始化首尾都没有找到交点曲线的尾端点
        flag1 = 0; flag2 = 0;
        // 初始化找到的交点曲线的曲线下标为 -1
        mark1 = -1; mark2 = -1;
        // 得到原曲线动态数组的大小
        num = pcurve[i].getSize();
        // 得到原曲线的首尾端点坐标
        x1 = pcurve[i][0];
        y1 = pcurve[i][1];
        x2 = pcurve[i][num - 2];
        y2 = pcurve[i][num - 1];
        // 首先对原曲线的首端点开始进行查找
        for (j = 0; j < 16; j += 2) {
            // 得到八领域的坐标
            dx1 = x1 + tpl[j];
            dy1 = y1 + tpl[j + 1];
            // 进行查找，找到退出循环
            for (k = 0; k < sectnum; k++) {
                // 得到交点曲线的动态数组的大小
                num2 = insect[k].getSize();
                // 找到就相应赋值，并且退出循环
                for (int m = 0; m < num2; m += 2) {
                    if (dx1 == insect[k][m] && dy1 == insect[k][m + 1]) {
                        mark1 = k; flag1 += 1; break;
                    }
                }
                // 找到退出循环
                if (flag1) {
                    break;   
                }
            }
            // 找到退出循环
            if (flag1) {
                break;   
            }
        }

        // 对原曲线的尾端点开始进行查找     
        for (j = 0; j < 16; j += 2) {
            // 得到八领域的坐标
            dx2 = x2 + tpl[j];
            dy2 = y2 + tpl[j + 1];
            // 进行查找，找到退出循环
            for (k = 0; k < sectnum; k++) {
                // 得到交点曲线的动态数组的大小
                num2 = insect[k].getSize();
                // 找到就相应赋值，并且退出循环
                for (int m = 0; m < num2; m += 2) {
                    if (dx2 == insect[k][m] && dy2 == insect[k][m + 1]) {
                        mark2 = k; flag2 += 1; break;
                    }
                }
                // 找到退出循环
                if (flag2) {
                    break;   
                }
            }
            // 找到退出循环
            if (flag2) {
                break;  
            }
        }

        // 如果没有找到可以组合的交点曲线，则进行下一个循环
        if (mark1 < 0 && mark2 < 0) {
            continue;
        }

        // 如果首部找到了可以组合的交点曲线，尾部没有，则原曲线反转，然后把交点
        // 曲线添加到反转后的曲线后边
        if (mark1 >= 0 && mark2 < 0) {
            // 曲线反转
            pcurve[i].reverse();
            // 构造曲线加入到当前曲线中
            DynamicArrays temp;
            makeCur(temp, dx1, dy1,
                    realsect[2 * mark1], realsect[2 * mark1 + 1]);
            pcurve[i].addArray(temp);

        // 如果尾部找到了可以组合的交点曲线，首部没有，直接把交点曲线添加到原来
        // 曲线后边
        } else if (mark1 < 0 && mark2 >= 0) {
            // 构造曲线加入到当前曲线中
            DynamicArrays temp;
            makeCur(temp, dx2, dy2,
                    realsect[2 * mark2], realsect[2 * mark2 + 1]);
            pcurve[i].addArray(temp);

        // 如果首部和尾部都找到了可以组合的交点曲线，先把尾部找到的交点曲线添加
        // 到原来曲线后边，然后反转曲线，然后把首部找到的交点曲线添加到反转后的
        // 曲线后边
        } else {
            // 构造曲线加入到当前曲线中
            DynamicArrays temp;
            makeCur(temp, dx2, dy2,
                    realsect[2 * mark2], realsect[2 * mark2 + 1]);
            pcurve[i].addArray(temp);
            
            // 清空得到的曲线
            temp.clear();

            // 曲线反转            
            pcurve[i].reverse();
            // 构造曲线加入到当前曲线中
            makeCur(temp, dx1, dy1,
                    realsect[2 * mark1], realsect[2 * mark1 + 1]);
            pcurve[i].addArray(temp);
        }

    }
}

// Host 函数：pcurveAcord（根据坐标得到曲线的编号）
static __host__ int pcurveAcord(DynamicArrays *pcurve, int count, int &location,
                                int x, int y)
{
    // 定义临时变量
    int i, dx1, dy1, dx2, dy2;
    // 根据输入坐标查找曲线集中对应的曲线编号 location
    for (i = 0; i < count; i++) {
        // 得到曲线的两个端点
        dx1 = pcurve[i][0];
        dy1 = pcurve[i][1];
        dx2 = pcurve[i][pcurve[i].getSize() - 2];
        dy2 = pcurve[i][pcurve[i].getSize() - 1];
        // 根据端点查找对应的曲线，如果找到则返回首尾情况，表示端点是曲线的首部
        // 还是尾部， 0 表示曲线首部，1 表示尾部
        if ((dx1 == x) && (dy1 == y)) {
            location = i;
            return 0;
        }
        if ((dx2 == x) && (dy2 == y)) {
            location = i;
            return 1;
        }
    }
    // 如果没有找到则返回 -1
    return -1;
}

// Host 函数：verAssemble（根据近域点集重组曲线集）
static __host__ void verAssemble(DynamicArrays *pcurve, int count,
                                 DynamicArrays *psect, int pcount, 
                                 DynamicArrays &realsect)
{
    // 定义临时变量
    int i, j, dx, dy, mark, location;
    int cen_x, cen_y;

    // 计算得到每个点集中的最中心点，加入到交点集合中
    for (i = 0; i < pcount; i++) {
        cen_x = 0;
        cen_y = 0;
        for (j = 0; j < psect[i].getSize();) {
            cen_x += psect[i][j++];
            cen_y += psect[i][j++];
        }
        // 得到最中心点
        cen_x = cen_x * 2 / j;
        cen_y = cen_y * 2 / j;
        realsect.addTail(cen_x, cen_y);
        // 组合曲线，更新曲线集合和交点动态数组
        for (j = 0; j < psect[i].getSize();) {
            dx = psect[i][j++];
            dy = psect[i][j++];
            if ((mark = pcurveAcord(pcurve, count, location, dx, dy)) != -1) {
                if(!mark) {
                    pcurve[location].reverse();
                }
                DynamicArrays temp;
                makeCur(temp, dx, dy, cen_x, cen_y);
                temp.delElemXY(dx, dy);
                pcurve[location].addArray(temp);
            }
        }
    }
}

// Host 函数：IsFindPoint（判断坐标是不是坐标集动态数组里的点）
static __host__ bool IsFindPoint(DynamicArrays &array, int x, int y)
{
    // 遍历动态数组里的点
    for (int i = 0; i < array.getSize(); i += 2) {
        // 找到就返回 true
        if (array[i] == x && array[i + 1] == y)
            return true;
    }
    // 没有找到则返回 false
    return false;
}

// Host 函数：AidNorm（判断两个端点之间距离是否在领域大小内，若在则添加到点集中）
static __host__ bool AidNorm(DynamicArrays *pcurve, int i, int count, 
                    DynamicArrays *psect, int pcount, int radius,
                    DynamicArrays &Vertex, int x, int y)
{
    // 定义临时变量
    int j, dx1, dy1, dx2, dy2;
    int dis1, dis2;
    bool mark1, mark2;
    bool find = false;
    // 查找编号 i 之后的曲线端点是否存在距离小于半径的端点
    for (j = i + 1; j < count; j++) {
        // 得到曲线的两个端点坐标
        dx1 = pcurve[j][0];
        dy1 = pcurve[j][1];
        dx2 = pcurve[j][pcurve[j].getSize() - 2];
        dy2 = pcurve[j][pcurve[j].getSize() - 1];
        mark1 = false;
        mark2 = false;
        // 查找第一个端点到曲线 i 端点的距离是否小于 radius
        if (IsFindPoint(Vertex, dx1, dy1)) {
            // 得到两点之间的距离并且向上取整
            dis1 = (int)floor(sqrt((dx1 - x) * (dx1 - x) + 
                                  (dy1 - y) * (dy1 - y)));

            if (dis1 <= radius) {
                mark1 = true;
            }
        }
        // 查找第二个端点到曲线 i 端点的距离是否小于 radius
        if(IsFindPoint(Vertex, dx2, dy2)) {
            // 得到两点之间的距离并且向上取整
            dis2 = (int)floor(sqrt((dx2 - x) * (dx2 - x) + 
                                  (dy2 - y) * (dy2 - y)));
            if (dis2 <= radius) {
                mark2 = true;
            }
        }
        // 找到两个端点中到到曲线 i 端点的距离最小的端点进行处理
        if (mark1 && mark2) {
            if (dis1 <= dis2) {
                psect[pcount].addTail(dx1, dy1);
                Vertex.delElem(dx1, dy1);
            } else {
                psect[pcount].addTail(dx2, dy2);
                Vertex.delElem(dx2, dy2);
            }
            find = true;
        } else if (mark1 && !mark2) {
            psect[pcount].addTail(dx1, dy1);
            Vertex.delElem(dx1, dy1);
            find = true;
        } else if (!mark1 && mark2) {
            psect[pcount].addTail(dx2, dy2);
            Vertex.delElem(dx2, dy2);
            find = true;
        }
    }
    // 返回值 find
    return find;
}

// Host 函数：bpConnect（断点的重组，根据用户输入的半径进行曲线端点组合）
static __host__ void bpConnect(DynamicArrays *pcurve, int count, int radius,
                               DynamicArrays *psect, int *pcount,
                               DynamicArrays &Vertex)
{
    // 定义临时变量
    int i, num; 
    int x1, y1, x2, y2;
    bool find;
    // 初始化为新增加的交点数为 0
    *pcount = 0;
    // 循环遍历每条曲线的两个端点
    for (i = 0; i < count - 1; i++) {
        num = pcurve[i].getSize();
        // 得到曲线的端点坐标
        x1 = pcurve[i][0];
        y1 = pcurve[i][1];
        x2 = pcurve[i][num - 2];
        y2 = pcurve[i][num - 1];
        find = false;
        // 判断原先是不是从端点点集得到的端点
        if (IsFindPoint(Vertex, x1, y1)) {
            // 从编号往后的曲线中找到符合条件的端点
            find = AidNorm(pcurve, i, count, psect, *pcount, radius, Vertex,
                           x1, y1);
            // 如果找到，从端点数组中删除这个端点，增加到编号为 *pcount 的
            // 近域点集中
            if (find) {
                Vertex.delElem(x1, y1);
                psect[*pcount].addTail(x1, y1);
                *pcount += 1;
            }
        }
        find = false;
        // 判断原先是不是从端点点集得到的端点
        if (IsFindPoint(Vertex, x2, y2)) {
            // 从编号往后的曲线中找到符合条件的端点
            find = AidNorm(pcurve, i, count, psect, *pcount, radius, Vertex,
                           x2, y2);
            // 如果找到，从端点数组中删除这个端点，增加到编号为 *pcount 的
            // 近域点集中
            if (find) {
                Vertex.delElem(x2, y2);
                psect[*pcount].addTail(x2, y2);
                *pcount += 1;
            }
        }
    }
}

// Host 函数：makeNode（根据曲线的起点和终点，以及边的情况，得到曲线的编号）
static __host__ void makeNode(DynamicArrays *pcurve, int count,
                              DynamicArrays *pcurno)
{
    // 定义临时变量
    int num1 = 0, num2 = 1, num = 0;
    int i, j, size1, size2;
    int x1, y1, x2, y2;
    // 定义 bool 型变量，表示查找是否之前相同的端点出现过
    bool find1, find2;
    // 给第一条曲线，添加首尾端点编号为 0 1
    pcurno[0].addTail(0, 1);
    // 接下来的端点编号从 2 开始
    num = 2;
    // 循环给剩下的曲线端点编号，并且编号不能重复
    for (i = 1; i < count; i++) {
        // 初始化没有找到
        find1 = find2 = false;
        // 得到当前曲线的动态数组长度
        size2 = pcurve[i].getSize();
        // 查找之前的曲线端点
        for (j = i - 1; j >= 0; j--) {
            // 得到当前曲线的动态数组长度
            size1 = pcurve[j].getSize();
            // 得到当前曲线的首尾端点坐标
            x1 = pcurve[j][0];
            y1 = pcurve[j][1];
            x2 = pcurve[j][size1 - 2];
            y2 = pcurve[j][size1 - 1];
            // 如果找到了首端点编号，得到当前编号值
            if (pcurve[i][0] == x1 && pcurve[i][1] == y1) {
                num1 = pcurno[j][0];
                find1 = true;
            } else if (pcurve[i][0] == x2 && pcurve[i][1] == y2) {
                num1 = pcurno[j][1];
                find1 = true;
            }
            // 如果找到了尾端点编号，得到当前编号值
            if (pcurve[i][size2 - 2] == x1 && pcurve[i][size2 - 1] == y1) {
                num2 = pcurno[j][0];
                find2 = true;
            } else if (pcurve[i][size2 - 2] == x2 && pcurve[i][size2 - 1] == y2) {
                num2 = pcurno[j][1];
                find2 = true;
           }
        }
        // 如果首尾端点都找到了，则把之前得到的编号赋给当前曲线
        if (find1 && find2) {
            pcurno[i].addTail(num1, num2);
        // 如果仅仅首端点找到了，则把之前得到的编号赋给当前曲线         
        } else if (find1 && !find2) {
            pcurno[i].addTail(num1, num);
            num++;
        // 如果仅仅尾端点找到了，则把之前得到的编号赋给当前曲线 
        } else if (!find1 && find2) {
            pcurno[i].addTail(num, num2);
            num++;
        // 如果首尾端点都没有找到，则重新累加赋值
        } else {
            pcurno[i].addTail(num, num + 1);
            num += 2;
        }
    }
    // 曲线端点编号结束后，给曲线的边赋值，也不会重复
    for (i = 0; i < count; i++) {
        pcurno[i].addElem(num++);
    } 
}

// Host 函数：openCurvePath（得到非闭合曲线编号序列）
static __host__ void openCurvePath(DynamicArrays *opencurnode, int *openNum,
                                   Graph *G, int start, int end)
{
    // 定义动态数组变量，表示边栈和点栈
    DynamicArrays edgestack, vexstack;
    // 定义点栈顶和边栈顶数，并且初始化
    int vtop = -1, etop = -1;
    // 定义点栈和边栈的大小
    int vstacksize, estacksize;
    // 定义临时变量
    int curnode;
    // 定义临时边指针
    Edge *cur;
    // 首端点入栈
    vexstack.addElem(start);
    // 复位所有当前要访问的边
    G->resetCurrent();

    // 循环，用于得到从起点到终点的所有路径
    while (vexstack.getSize() != 0) {
        // 得到当前栈的大小
        vstacksize = vexstack.getSize();
        estacksize = edgestack.getSize();
        // 如果栈顶的值为终点
        if (vexstack[vstacksize - 1] == end) {
            // 得到一条从起点到终点的路径并且保存。即添加端点编号和边编号
            for (int i = 0; i < estacksize; i++) {
                opencurnode[*openNum].addTail(vexstack[i], edgestack[i]);
            }
            // 添加终点编号
            opencurnode[*openNum].addElem(end);
            // 曲线条数增加 1
            *openNum += 1;
            // 删除点栈顶和边栈顶的端点，搜索下一条可能的路径
            vexstack.delTail(vtop);
            edgestack.delTail(etop);
        // 如果栈顶的值不是终点，则继续搜索可能的路径
        } else {
            // 得到当前栈顶的值
            curnode = vexstack[vstacksize - 1];
            // 得到图的当前点要访问的边
            cur = G->vertexlist[curnode].current;
            // 如果当前要访问的边不为空
            if (cur != NULL) {
                // 得到当前边的另一个顶点，如果该顶点不在点栈中，当前边也不在边
                // 栈中，则把当前点和边分别入栈，把当前要访问的边指针指向下一条
                // 边。判断是为了确保路径的点和边不能重复
                if (!edgestack.findElem(cur->eno) &&
                    !vexstack.findElem(cur->jvex)) { 
                    vexstack.addElem(cur->jvex);
                    edgestack.addElem(cur->eno);
                }
                G->vertexlist[curnode].current = cur->link;
            // 如果当前要访问的边为空，则当前点连接的边都访问过，删除点栈顶和
            // 边栈顶的端点，重新设置当前栈顶端点的当前要访问的边
            } else {
                vexstack.delTail(vtop);
                edgestack.delTail(etop);
                // 如果点栈顶的值等于起始点，则退出循环
                if (vtop == start)
                    break;
                // 设置当前栈顶端点的当前要访问的边为第一条边
                G->vertexlist[vtop].current = G->vertexlist[vtop].firstedge;
            }
        }        
    }
}

// Host 函数：closeCurvePath（得到闭合曲线编号序列）
static __host__ void closeCurvePath(DynamicArrays *closecurnode, int *closeNum,
                                    Graph *G, int insect)
{
    // 定义动态数组变量，表示边栈和点栈
    DynamicArrays edgestack, vexstack;
    // 定义点栈顶和边栈顶数，并且初始化
    int vtop = -1, etop = -1;
    // 定义点栈和边栈的大小
    int vstacksize, estacksize;
    // 定义临时变量
    int curnode;
    // 是否找到一样的路径，尽管顺序不一样
    bool isFind;
    // 定义临时边指针
    Edge *cur;
    // 路径起始端点入栈 
    vexstack.addElem(insect);
    // 闭合曲线数量
    int num = *closeNum;
    // 复位所有当前要访问的边
    G->resetCurrent();
    while (vexstack.getSize() != 0) {
        // 得到当前栈的大小 
        vstacksize = vexstack.getSize();
        estacksize = edgestack.getSize();
        // 初始化 isFind 为 false
        isFind = false;
        // 当边栈不为空，且点栈栈顶元素值为起点，则保存一条得到的闭合路径
        if (estacksize != 0 && vexstack[vstacksize - 1] == insect) {
            for (int i = 0; i < estacksize; i++) {
                closecurnode[num].addTail(vexstack[i], edgestack[i]);
            }
            closecurnode[num].addElem(insect);
            // 查找是否和之前得到的路径表示是同一条路径
            for (int j = 0; j < num; j++) {
                if (IsArrayEqual(closecurnode[j], closecurnode[num])) {
                    isFind = true;
                    break; 
                }
            }
            // 如果找到了一样的路径，就清空当前得到的闭合路径
            if (isFind) {
                closecurnode[num].clear();
            // 如果没有找到，则保存当前得到的闭合路径，并且路径数量加 1
            } else {
                num++;
            }

            // 删除点栈顶和边栈顶的端点，搜索下一条可能的路径
            vexstack.delTail(vtop);
            edgestack.delTail(etop);
        // 栈顶不是起点，则继续搜索可能的路径
        } else {
            // 得到当前栈顶的值
            curnode = vexstack[vstacksize - 1];

            // 得到图的当前点要访问的边
            cur = G->vertexlist[curnode].current;
            // 如果当前要访问的边不为空
            if (cur != NULL) {
                // 得到当前边的另一个顶点，如果当前边不在边栈中，则把当前点和边
                // 分别入栈，把当前要访问的边指针指向下一条边。
                if (!edgestack.findElem(cur->eno)) { 
                    if ((cur->jvex == insect)|| !vexstack.findElem(cur->jvex)) {
                        vexstack.addElem(cur->jvex);
                        edgestack.addElem(cur->eno);
                    }
                }
                G->vertexlist[curnode].current = cur->link;

            // 如果当前要访问的边为空，则当前点连接的边都访问过，删除点栈顶和
            // 边栈顶的端点，重新设置当前栈顶端点的当前要访问的边
            } else {
                vexstack.delTail(vtop);
                edgestack.delTail(etop);
                // 如果点栈顶的值等于起始点，则退出循环
                if (vtop == insect)
                    break;
                // 设置设置当前栈顶端点的当前要访问的边为第一条边
                G->vertexlist[vtop].current = G->vertexlist[vtop].firstedge;
            }
        }        
    }
    // 得到闭合曲线的数量
    *closeNum = num;
}

// Host 函数：IsArrayEqual（判断两个动态数组表示的曲线是否表示同一条曲线）
static __host__ bool IsArrayEqual(DynamicArrays object1, DynamicArrays object2)
{
    // 两个动态数组的大小不一致，则直接返回 false
    if (object1.getSize() != object2.getSize()) {
        return false;
    // 否则看排序后结果是否一样，如果一样，则返回 true，否则返回 false
    // 由于处理的是闭合曲线编号，头尾是一致的，则排序比较不包括最后一个编号
    } else {
        // 得到曲线编号动态数组大小
        int size = object1.getSize();
        // 定义临时指针变量，得到第一个动态数组的整型指针
        int *p = object1.getCrvDatap();
        // 定义临时变量，用于交换数据
        int temp;
        // 临时变量
        int min;
        // 排序第一个动态数组的数据
        for (int i = 0; i < size - 2; i++) {
            min = i;
            for (int j = i + 1; j < size - 1; j++) {
                if (p[j] < p[min]) {
                    min = j;
                }
            }
            // 如果找到其他最小的则交换
            if (min != i) {
                temp = p[i];
                p[i] = p[min];
                p[min] = temp;
            }
        }
        // 定义临时指针变量，得到第二个动态数组的整型指针
        int *q = object2.getCrvDatap();
        // 排序第二个动态数组的数据
        for (int i = 0; i < size - 2; i++) {
            min = i;
            for (int j = i + 1; j < size - 1; j++) {
                if (q[j] < q[min]) {
                    min = j;
                }
            }
            // 如果找到其他最小的则交换
            if (min != i) {
                temp = q[i];
                q[i] = q[min];
                q[min] = temp;
            }
        }

        // 排序结果如果不一样，则返回 false
        for (int i = 0; i < size - 1; i++) {
            if (p[i] != q[i]) {
                return false;
            }
        }

        // 表示同一条路径，返回 true
        return true;
    } 
}

// Host 函数：getPointNo（根据坐标对得到数组内对应编号）
static __host__ void getPointNo(DynamicArrays *pcurve, int count,
                                DynamicArrays *pcurno, DynamicArrays &array,
                                DynamicArrays &arrayno)
 {
    // 临时变量，用于循环计数
    int i, j;
    // 定义临时变量，存储坐标
    int dx, dy;
    
    // 循环得到数组内坐标对编号
    for (i = 0; i < array.getSize();) {
        // 得到数组的 x，y 坐标
        dx = array[i++];
        dy = array[i++];
        
        // 根据得到的曲线头尾坐标得到相应编号
        for (j = 0; j < count; j++) {
            // 如果为曲线首部
            if (dx == pcurve[j][0] && dy == pcurve[j][1]) {
                arrayno.addElem(pcurno[j][0]);
                break;
            // 如果为曲线尾部
            } else if (dx == pcurve[j][pcurve[j].getSize() - 2] &&
                       dy == pcurve[j][pcurve[j].getSize() - 1]) {
                arrayno.addElem(pcurno[j][1]);
                break;
            }
        }
    }
 }

// Host 函数：getCurveNonFormat（得到非格式化输出曲线数据有序序列）
static __host__ void getCurveNonFormat(DynamicArrays *curnode,
                                       DynamicArrays *pcurve, int count,
                                       DynamicArrays *pcurno,
                                       DynamicArrays *cur, int num, bool close)
{
    // 临时变量，存储曲线编号数组的大小
    int nodesize;
    
    // 临时变量，存储得到的端点编号值
    int inode;
    
    // 临时变量，得到点的数目
    int vnum = pcurno[count - 1][2] - count + 1;
    
    // 临时变量，存储得到的曲线下标
    int icur;
    
    // 定义循环计数变量
    int i, j;
    
    // 临时变量，作为函数参数得到曲线的末尾坐标
    int xtop, ytop;
    
    // 根据得到的曲线编号集获得对应曲线
    for (i = 0; i < num; i++) {
        // 得到曲线编号数组的大小
        nodesize = curnode[i].getSize();
        // 循环得到曲线端点和边编号并且得到组合曲线
        for (j = 0; j < nodesize;) {
            // 得到点编号
            inode  = curnode[i][j++];
            // 如果超过大小，则推出循环
            if (j >= nodesize) break;
            // 根据边编号得到曲线下标
            icur = curnode[i][j++] - vnum;
            
            // 点编号和曲线下标，得到组合曲线
            if (inode == pcurno[icur][0]) {
                cur[i].addArray(pcurve[icur]);
                if (j != nodesize - 1) {
                    cur[i].delTail(ytop);
                    cur[i].delTail(xtop);
                }
                
            } else if (inode == pcurno[icur][1]) {
                pcurve[icur].reverse();
                cur[i].addArray(pcurve[icur]);
                if (j != nodesize - 1) {
                    cur[i].delTail(ytop);
                    cur[i].delTail(xtop);
                }
                pcurve[icur].reverse();
            }
        }

        // 如果为闭合曲线就删除末尾坐标
        if (close) {
            // 由于末尾坐标和起始一样，删除末尾坐标
            cur[i].delTail(ytop);
            cur[i].delTail(xtop);
        }
    }
}

// Host 函数：freeCurve（释放曲线申请的空间）
void freeCurve(Curve ***curveList, int count)
{
    if (curveList == NULL)
        return;
    // 循环释放空间
    for (int i = 0; i < count; i++) {
        CurveBasicOp::deleteCurve((*curveList)[i]);
    }
    delete [](*curveList);
}

// Kernel 函数：_traverseKer（并行遍历图像得到端点数组和交点数组，并且得到去掉
// 交点后的输出图像）
static __global__ void _traverseKer(ImageCuda inimg, ImageCuda outimg,
                                    int *array1_dev, int *array2_dev,
                                    Template boxtpl)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;
    
    // 计算输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算输出坐标点对应的图像数据数组下标。
    int outidx = r * inimg.imgMeta.width + c;

    // 如果当前像素点为 0，则输出图像对应位置零，并且返回。
    if (inimg.imgMeta.imgData[inidx] == 0) {
        outimg.imgMeta.imgData[inidx] = 0;
        return;
    }
    
    int tmpidx;               // 临时变量，存储模板其他点的图像数据数组下标
    int count = 0;            // 临时变量，存储灰度不为 0 的个数
    int dx, dy;               // 临时变量，存储模板坐标
    int *p = boxtpl.tplData;  // 临时变量，得到模板指针

    // 扫描该点模版范围内有多少个灰度值不为 0 的点
    for (int i = 0; i < boxtpl.count; i++) {
        // 计算当模版位置所在像素的 x 和 y 分量，模版使用相邻的两个下标的
        // 数组表示一个点，所以使用当前模版位置的指针加一操作
        dx = c + *(p++);
        dy = r + *(p++);
        // 如果是当前点则理下一个点
        if (dx == c && dy == r)
            continue;

        // 计算坐标点对应的图像数据数组下标。
        tmpidx = dy * inimg.pitchBytes + dx;
        // 得到当前点 8 领域内的非零像素点个数
        if (inimg.imgMeta.imgData[tmpidx] != 0) {
            count++;
        }
    }
    // 如果 count 为 0，表示该像素八领域没有不为 0 的像素点，则该点是
    // 孤立点，则给对应输出图像在该处赋值为 0
    if (count == 0) {
        outimg.imgMeta.imgData[inidx] = 0;
        return;
    // 如果 flag 大于等于 3，表示该像素点作为曲线交点，则给对应输出图像
    // 在该处赋值为 0
    } else if (count >= 3) {
        array2_dev[2 * outidx] = c;
        array2_dev[2 * outidx + 1] = r;
        outimg.imgMeta.imgData[inidx] = 0;
    // 如果 count 为 1，表示该像素八领域有一个不为 0 的像素点，则该点是
    // 曲线端点，并给对应输出图像在该处赋值原图像对应点像素值
    } else if (count == 1) {
        array1_dev[2 * outidx] = c;
        array1_dev[2 * outidx + 1] = r;
        outimg.imgMeta.imgData[inidx] = inimg.imgMeta.imgData[inidx];
    // 否则flag则为 2，表示该像素点作为曲线上的点,并给对应输出图像在该处
    // 赋值原图像对应点像素值
    } else {
        outimg.imgMeta.imgData[inidx] = inimg.imgMeta.imgData[inidx];
    }

}

// Kernel 函数：_traverseKerNew（遍历图像，得到图像上所有的像素点）
static __global__ void _traverseKerNew(ImageCuda inimg, int *array1_dev)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算输入坐标点对应的图像数据数组下标。
    int inidx = r * inimg.pitchBytes + c;
    // 计算输出坐标点对应的图像数据数组下标。
    int outidx = r * inimg.imgMeta.width + c;

    // 如果当前像素点不为 0，则得到该坐标
    if (inimg.imgMeta.imgData[inidx] != 0) {
        array1_dev[2 * outidx] = c;
        array1_dev[2 * outidx + 1] = r;
    }
}

// 宏：FAIL_CURVETRACING_FREE
// 当下面函数运行出错时，使用该宏清除内存，防止内存泄漏。
#define FAIL_CURVETRACING_FREE  do {               \
        if (outimg1 != NULL) {                     \
            ImageBasicOp::deleteImage(outimg1);    \
            outimg1 = NULL;                        \
        }                                          \
        if (outimg2 != NULL) {                     \
            ImageBasicOp::deleteImage(outimg2);    \
            outimg2 = NULL;                        \
        }                                          \
        if (tmpdev != NULL) {                      \
            cudaFree(tmpdev);                      \
            tmpdev = NULL;                         \
        }                                          \
        if (array1 != NULL) {                      \
            delete []array1;                       \
            array1 = NULL;                         \
        }                                          \
        if (array2 != NULL) {                      \
            delete []array2;                       \
            array2 = NULL;                         \
        }                                          \
        if (boxtpl != NULL) {                      \
            TemplateFactory::putTemplate(boxtpl);  \
            boxtpl = NULL;                         \
        }                                          \
        if (mark != NULL) {                        \
            delete []mark;                         \
            mark = NULL;                           \
        }                                          \
        if (pcurve != NULL) {                      \
            delete []pcurve;                       \
            pcurve = NULL;                         \
        }                                          \
        if (insect != NULL) {                      \
            delete []insect;                       \
            insect = NULL;                         \
        }                                          \
        if (psect != NULL) {                       \
            delete []psect;                        \
            psect = NULL;                          \
        }                                          \
        if (pcurno != NULL) {                      \
            delete []pcurno;                       \
            pcurno = NULL;                         \
        }                                          \
        if (opencur != NULL) {                     \
            delete []opencur;                      \
            opencur = NULL;                        \
        }                                          \
        if (closecur != NULL) {                    \
            delete []closecur;                     \
            closecur = NULL;                       \
        }                                          \
        if (G != NULL) {                           \
            delete G;                              \
            G = NULL;                              \
        }                                          \
    } while (0)

// Host 成员方法：curveTracing（曲线跟踪）
// 对图像进行曲线跟踪，得到非闭合曲线和闭合曲线的有序序列
__host__ int CurveTracing::curveTracing(Image *inimg, Curve ***curveList,
                                        int *openNum, int *closeNum)
{   
    // 如果输入图像指针为空或者输出的曲线集指针为空，错误返回
    if (inimg == NULL || curveList == NULL)
        return NULL_POINTER;
    
    // 定义错误码变量
    int errcode;
    cudaError_t cuerrcode;
    
    // 定义输出图像 1 和 2
    Image *outimg1 = NULL;
    Image *outimg2 = NULL;
    
    // 定义指针 tmpdev 给设备端端点数组和交点数组创建存储空间
    int *tmpdev = NULL;
    
    // 定义 CPU 端端点数组和交点数组
    int *array1 = NULL;
    int *array2 = NULL;
    
    // 定义模板 boxtpl 用于获取模板
    Template *boxtpl = NULL;

    // 定义标志数组，标志图像上非零点的访问情况
    int *mark = NULL;
    // 定义曲线数组，存储得到的曲线
    DynamicArrays *pcurve = NULL;
    // 定义交点分类的动态数组，存储分类的结果
    DynamicArrays *insect = NULL;
    // 定义近域点集动态数组，用于断点连续的处理
    DynamicArrays *psect = NULL;
    // 定义变量，存储曲线的编号;
    DynamicArrays *pcurno = NULL;
    // 定义非闭合曲线
    DynamicArrays *opencur = NULL;
    // 定义闭合曲线
    DynamicArrays *closecur = NULL;
    
    // 定义图类的指针变量
    Graph *G = NULL;
    
    // 给输出图像构建空间
    ImageBasicOp::newImage(&outimg1);
    ImageBasicOp::makeAtHost(outimg1, inimg->width, inimg->height);

    ImageBasicOp::newImage(&outimg2);
    ImageBasicOp::makeAtHost(outimg2, inimg->width, inimg->height);

    // 将图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR) {
        FAIL_CURVETRACING_FREE;
        return errcode;
    }

    errcode = ImageBasicOp::copyToCurrentDevice(outimg1);
    if (errcode != NO_ERROR) {
        FAIL_CURVETRACING_FREE;
        return errcode;
    }

    errcode = ImageBasicOp::copyToCurrentDevice(outimg2);
    if (errcode != NO_ERROR) {
        FAIL_CURVETRACING_FREE;
        return errcode;
    }

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR) {
        FAIL_CURVETRACING_FREE;
        return errcode;
    }
    
    // 提取输出图像 1 的 ROI 子图像。
    ImageCuda outsubimgCud1;
    errcode = ImageBasicOp::roiSubImage(outimg1, &outsubimgCud1);
    if (errcode != NO_ERROR) {
        FAIL_CURVETRACING_FREE;
        return errcode;
    }  
    // 提取输出图像 2的 ROI 子图像。
    ImageCuda outsubimgCud2;
    errcode = ImageBasicOp::roiSubImage(outimg2, &outsubimgCud2);
    if (errcode != NO_ERROR) {
        FAIL_CURVETRACING_FREE;
        return errcode;
    }
    // 定义八领域模板
    int tpl[16] = { -1, -1, 0, -1, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0 };
    // 定义变量，用于循环
    int i, j, k;
    // 定义临时变量，得到第一次遍历得到的端点和交点动态数组大小
    int num1 = 0, num2 = 0;
    // 定义临时变量存储坐标值
    int dx, dy;
    // 计算数据尺寸。
    int arraysize = inimg->width * inimg->height * 2;
    int datasize = arraysize * 2 * sizeof(int);
    
    // 在当前设备上申请坐标数据的空间。
    cuerrcode = cudaMalloc((void **)(&tmpdev), datasize);
    if (cuerrcode != cudaSuccess) {
        FAIL_CURVETRACING_FREE;
        return CUDA_ERROR;
    }
    // 给该空间内容全部赋值为 -1 
    cuerrcode = cudaMemset(tmpdev, -1, datasize);
    if (cuerrcode != cudaSuccess) {
        FAIL_CURVETRACING_FREE;
        return CUDA_ERROR;
    }
    // 定义设备端端点数组和交点数组
    int *array1_dev  = tmpdev;
    int *array2_dev = tmpdev + arraysize;

    // 定义模板的尺寸
    dim3 boxsize(3, 3, 1);

    // 通过模板工厂得到圆形领域模板
    errcode = TemplateFactory::getTemplate(&boxtpl, TF_SHAPE_BOX, 
                                           boxsize, NULL);

    // 检查模板是否为 NULL，如果为 NULL 直接报错返回。
    if (errcode != NO_ERROR) {
        FAIL_CURVETRACING_FREE;
        return errcode;
    }

    // 将模板拷贝到 Device 内存中
    errcode = TemplateBasicOp::copyToCurrentDevice(boxtpl);
    if (errcode != NO_ERROR) {
        FAIL_CURVETRACING_FREE;
        return errcode;
    }
    
    // 计算调用第一个 Kernel 所需要的线程块尺寸。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y  - 1) / blocksize.y;

    // 调用第一个 Kernel 生成图像标志位数组。
    _traverseKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud1,
                                          array1_dev, array2_dev, *boxtpl);
    if (cudaGetLastError() != cudaSuccess) {
        FAIL_CURVETRACING_FREE;
        return CUDA_ERROR;
    }

    // 给 CPU 端端点数组和交点数组申请空间
    array1 = new int[arraysize];
    array2 = new int[arraysize];
    
    // 把两个数组拷贝到 Host 端
    cuerrcode = cudaMemcpy(array1, array1_dev, arraysize * sizeof (int),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        FAIL_CURVETRACING_FREE;
        return CUDA_ERROR;
    }
    cuerrcode = cudaMemcpy(array2, array2_dev, arraysize * sizeof (int),
                           cudaMemcpyDeviceToHost);
    if (cuerrcode != cudaSuccess) {
        FAIL_CURVETRACING_FREE;
        return CUDA_ERROR;
    }
    
    // 定义端点动态数组和交点动态数组
    DynamicArrays Vertex, Intersect;
    // 把得到的端点和交点数组的非 -1 值赋值给端点动态数组和交点动态数组
    for (i = 0; i < arraysize; i++) {
        if (array1[i] != -1) {
            Vertex.addElem(array1[i]);
        }
    }
    for (i = 0; i < arraysize; i++) {
        if (array2[i] != -1) {
            Intersect.addElem(array2[i]);
        }
    }
    
    // 得到第一次遍历得到的端点和交点动态数组大小
    num1 = Vertex.getSize();
    num2 = Intersect.getSize();

    // 如果图像上曲线有端点和交点时，说明有曲线相交，可能有闭合和非闭合曲线，
    // 如果图像上曲线有端点没有交点时，但是经过断续连接有可能产生闭合和
    // 非闭合曲线
    if ((num1 && num2) || (num1 && !num2)) {
        // 重新给该空间内容全部赋值为 -1，用于第二次遍历
        cuerrcode = cudaMemset(tmpdev, -1, datasize);
        if (cuerrcode != cudaSuccess) {
            FAIL_CURVETRACING_FREE;
            return CUDA_ERROR;
        }
        // 第二次并行遍历
        _traverseKer<<<gridsize, blocksize>>>(outsubimgCud1, outsubimgCud2,
                                              array1_dev, array2_dev, *boxtpl);
        if (cudaGetLastError() != cudaSuccess) {
            FAIL_CURVETRACING_FREE;
            return CUDA_ERROR;
        }
        // 把端点数组拷贝到 Host 端
        cuerrcode = cudaMemcpy(array1, array1_dev, arraysize * sizeof (int),
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            FAIL_CURVETRACING_FREE;
            return CUDA_ERROR;
        }
        // 定义第二次遍历要得到的端点动态数组
        DynamicArrays Vertex1;
        for (i = 0; i < arraysize; i++) {
            if (array1[i] != -1) {
                Vertex1.addElem(array1[i]);
            }
        }

        // 将图像拷贝到 Device 内存中。
        errcode = ImageBasicOp::copyToHost(outimg1);
        if (errcode != NO_ERROR) {
            FAIL_CURVETRACING_FREE;
            return errcode;
        }

        // 申请标志数组的空间，大小和图像一样
        mark = new int[arraysize / 2];
        // 初始化标志数组的值为 0
        memset(mark, 0, sizeof(int) * arraysize / 2);
        // 定义变量 count 表示得到的曲线输量
        int count = 0;
        // 标志曲线跟踪的端点是否已经在曲线中，用于 getCurve 函数调用
        int test  = 0;
        // 申请曲线数组空间，曲线最多数目是端点的个数
        pcurve = new DynamicArrays [Vertex1.getSize() / 2];

        // 循环调用 getCurve 函数得到非闭合曲线的有序序列
        for(i = 0; i < Vertex1.getSize(); i += 2) {
            getCurve(pcurve, test, count, outimg1, mark, tpl,
                     Vertex1[i], Vertex1[i + 1]);
            // 如果 test 不为 0，则 count 不加 1，继续循环，否则曲线数目加 1
            if (test) {
                test = 0;
                continue;
            }
            count++;
        }

        // 定义临时变量存储坐标值
        int x, y;
        // 定义变量，存储交点的个数
        int sectnum = 0; 
        // 申请交点分类的动态数组空间
        insect = new DynamicArrays [num2 / 2];

        // 循环得到交点分类动态数组值
        while (Intersect.getSize()) {
            x = Intersect[0];
            y = Intersect[1];
            sectnum++;
            insectClassify(x, y, Intersect, insect, sectnum, tpl);
        }

        // 定义真正的交点数组，得到的是唯一确定的交点，与交点曲线方向动态数组集
        // 相对应，大小其实为交点个数。从分类的交点集中取领域数最大的点作为交点
        DynamicArrays realsect;

        // 循环得到交点曲线方向动态数组集和真正的交点数组
        for (i = 0; i < sectnum; i++) {
            // 定义变量，存储领域数最大的点标记值，并初始化为 0
            int maxvalue = 0;
            // 定义变量，存储坐标值，初始化为第一条曲线第一个点的坐标值
            int insect_x = insect[i][0], insect_y = insect[i][1];
            // 根据之前的分类结果，循环得到交点曲线方向动态数组
            for (j = 0; j < insect[i].getSize(); j += 2) {
               x = insect[i][j];
               y = insect[i][j + 1];
               // 定义临时变量，存储分类集合中的点的八领域内有多少个点
               int value = 0;
               for (k = 0; k < 16; k += 2) {
                   dx = x + tpl[k];
                   dy = y + tpl[k + 1];
                   // 遍历点周围有多少个点
                   for (int s = 0; s < insect[i].getSize(); s += 2) {
                       if (dx == insect[i][s] && dy == insect[i][s + 1]) {
                           value++;
                       }
                   }
               }
               // 找到最中心的交点
               if (value > maxvalue) {
                   maxvalue = value;
                   insect_x = x;
                   insect_y = y;
               }
            }
            // 得到交点坐标值
            realsect.addElem(insect_x);
            realsect.addElem(insect_y);
        }

        // 调用函数得到重组后的曲线，还是存储于 pcurve 中
        interAssemble(pcurve, count, insect, sectnum, realsect, tpl);
        
        // 定义近域点集的个数，得到新产生的交点个数
        int pcount = 0;
        // 给近域点集申请最大空间
        psect = new DynamicArrays[Vertex.getSize() / 2];
        
        // 根据用户输入的半径得到近域点集，并且更新端点动态数组
        bpConnect(pcurve, count, radius, psect, &pcount, Vertex);

        // 断点重组，根据用户输入的半径进行曲线断点组合，更新交点动态数组
        verAssemble(pcurve, count, psect, pcount, realsect);

        // 存储曲线的编号，空间大小和之前提取的曲线一样
        pcurno = new DynamicArrays[count];
        
        // 调用函数得到曲线编号集合
        makeNode(pcurve, count, pcurno);

        // 定义变量，存储图的边数，并且赋值
        int edgenum = count;
        // 定义变量，存储图的点数，并且赋值
        int vexnum = pcurno[count - 1][2] - edgenum + 1;
        // 给图申请空间，根据边数和点数，初始化图
        G = new Graph(vexnum, edgenum);
        
        // 根据曲线编号集，给图设置相应的边
        for (i = 0; i < count; i++) {
            G->setEdge(pcurno[i][0], pcurno[i][1], pcurno[i][2]);
        }

        // 定义曲线编号集数组，分为非闭合曲线和闭合曲线
        DynamicArrays opencurnode[CURVE_VALUE], closecurnode[CURVE_VALUE];
        
        // 定义端点编号数组和交点编号数组，分别得到端点和交点的坐标对应的编号数
        DynamicArrays vertexno;
        DynamicArrays intersectno;
        
        // 调用函数得到数组端点的编号
        getPointNo(pcurve, count, pcurno, Vertex, vertexno); 
        
        // 调用函数得到数组交点的编号
        if (realsect.getSize() > 0)
            getPointNo(pcurve, count, pcurno, realsect, intersectno); 

        // 起始闭合和非闭合曲线的数目都设置为 0
        *openNum = 0;
        *closeNum = 0;
        
        // 循环得到非闭合曲线的路径编号
        for (i = 0; i < vertexno.getSize(); i++) {
            // 定义起始点
            int start, end;
            start = vertexno[i];
            for (j = i + 1; j < vertexno.getSize(); j++) {
                end = vertexno[j];
                // 调用函数，得到非闭合曲线编号序列集
                openCurvePath(opencurnode, openNum, G, start, end);
            }
        }

        // 循环得到闭合曲线的路径编号
        for (i = 0; i < intersectno.getSize(); i++) {
            // 调用函数，得到闭合曲线编号序列集
            closeCurvePath(closecurnode, closeNum, G, intersectno[i]);
        }

        // 申请非闭合曲线空间
        opencur = new DynamicArrays[*openNum];
        
        // 申请闭合曲线空间
        closecur = new DynamicArrays[*closeNum];

        // 调用函数得到非格式输出的非闭合曲线
        getCurveNonFormat(opencurnode, pcurve, count, pcurno, opencur,
                          *openNum, false);
        
        // 调用函数得到非格式输出的闭合曲线
        getCurveNonFormat(closecurnode, pcurve, count, pcurno, closecur,
                          *closeNum, true);
        
        // 定义曲线总数
        int total = *openNum + *closeNum;

        // 给输出结果赋值，首先申请空间大小
        *curveList = new Curve *[total];
        
        // 定义变量，表示曲线长度
        size_t curveLength;
        
        // 定义变量，表示动态数组里的整型指针
        int *crvData;

        // 循环得到输出非闭合曲线
        for (i = 0; i < *openNum; i++) {
            // 申请曲线空间
            errcode = CurveBasicOp::newCurve(&((*curveList)[i]));
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i);
                FAIL_CURVETRACING_FREE;
                return OUT_OF_MEM;
            }
            // 得到曲线长度
            curveLength = (size_t)(opencur[i].getSize() / 2);
            // 得到动态数组里的整型指针
            crvData = opencur[i].getCrvDatap();
            if (crvData == NULL) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE;
                return NULL_POINTER;
            }
            // 在 CPU 端构建曲线值
            errcode = CurveBasicOp::makeAtHost((*curveList)[i], curveLength,
                                               crvData);
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE;
                return errcode;
            }
        }
        
        // 循环得到输出闭合曲线
        for (; i < total; i++) {
            // 申请曲线空间
            errcode = CurveBasicOp::newCurve(&((*curveList)[i]));
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i);
                FAIL_CURVETRACING_FREE;
                return OUT_OF_MEM;;
            }
            // 得到曲线长度
            curveLength = (size_t)(closecur[i - *openNum].getSize() / 2);
            // 得到动态数组里的整型指针
            crvData = closecur[i - *openNum].getCrvDatap();
            if (crvData == NULL) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE;
                return NULL_POINTER;
            }
            // 在 CPU 端构建曲线值
            errcode = CurveBasicOp::makeAtHost((*curveList)[i], curveLength,
                                               crvData);
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE;
                return errcode;
            }
        }
    } 

    // 如果图像上没有端点只有交点时候，说明是闭合曲线相交
    else if (!num1 && num2) 
    {
        // 重新给该空间内容全部赋值为 -1，用于第二次遍历
        cuerrcode = cudaMemset(tmpdev, -1, datasize);
        if (cuerrcode != cudaSuccess) {
            FAIL_CURVETRACING_FREE;
            return CUDA_ERROR;
        }
        // 第二次并行遍历
        _traverseKer<<<gridsize, blocksize>>>(outsubimgCud1, outsubimgCud2,
                                              array1_dev, array2_dev, *boxtpl);
        if (cudaGetLastError() != cudaSuccess) {
            FAIL_CURVETRACING_FREE;
            return CUDA_ERROR;
        }
        // 把端点数组拷贝到 Host 端
        cuerrcode = cudaMemcpy(array1, array1_dev, arraysize * sizeof (int),
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            FAIL_CURVETRACING_FREE;
            return CUDA_ERROR;
        }
        // 定义第二次遍历要得到的端点动态数组
        DynamicArrays Vertex1;
        for (i = 0; i < arraysize; i++) {
            if (array1[i] != -1) {
                Vertex1.addElem(array1[i]);
            }
        }

        // 将图像拷贝到 Device 内存中。
        errcode = ImageBasicOp::copyToHost(outimg1);
        if (errcode != NO_ERROR) {
            FAIL_CURVETRACING_FREE;
            return errcode;
        }

        // 申请标志数组的空间，大小和图像一样
        mark = new int[arraysize / 2];
        // 初始化标志数组的值为 0
        memset(mark, 0, sizeof(int) * arraysize / 2);
        // 定义变量 count 表示得到的曲线输量
        int count = 0;
        // 标志曲线跟踪的端点是否已经在曲线中，用于 getCurve 函数调用
        int test  = 0;
        // 申请曲线数组空间，曲线最多数目是端点的个数
        pcurve = new DynamicArrays [Vertex1.getSize() / 2];

        // 循环调用 getCurve 函数得到非闭合曲线的有序序列
        for(i = 0; i < Vertex1.getSize(); i += 2) {
            getCurve(pcurve, test, count, outimg1, mark, tpl,
                     Vertex1[i], Vertex1[i + 1]);
            // 如果 test 不为 0，则 count 不加 1，继续循环，否则曲线数目加 1
            if (test) {
                test = 0;
                continue;
            }
            count++;
        }

        // 定义临时变量存储坐标值
        int x, y;
        // 定义变量，存储交点的个数
        int sectnum = 0; 
        // 申请交点分类的动态数组空间
        insect = new DynamicArrays [num2 / 2];

        // 循环得到交点分类动态数组值
        while (Intersect.getSize()) {
            x = Intersect[0];
            y = Intersect[1];
            sectnum++;
            insectClassify(x, y, Intersect, insect, sectnum, tpl);
        }

        // 定义真正的交点数组，得到的是唯一确定的交点，与交点曲线方向动态数组集
        // 相对应，大小其实为交点个数。从分类的交点集中取领域数最大的点作为交点
        DynamicArrays realsect;

        // 循环得到交点曲线方向动态数组集和真正的交点数组
        for (i = 0; i < sectnum; i++) {
            // 定义变量，存储领域数最大的点标记值，并初始化为 0
            int maxvalue = 0;
            // 定义变量，存储坐标值，初始化为第一条曲线第一个点的坐标值
            int insect_x = insect[i][0], insect_y = insect[i][1];
            // 根据之前的分类结果，循环得到交点曲线方向动态数组
            for (j = 0; j < insect[i].getSize(); j += 2) {
               x = insect[i][j];
               y = insect[i][j + 1];
               // 定义临时变量，存储分类集合中的点的八领域内有多少个点
               int value = 0;
               for (k = 0; k < 16; k += 2) {
                   dx = x + tpl[k];
                   dy = y + tpl[k + 1];
                   // 遍历点周围有多少个点
                   for (int s = 0; s < insect[i].getSize(); s += 2) {
                       if (dx == insect[i][s] && dy == insect[i][s + 1]) {
                           value++;
                       }
                   }
               }
               // 找到最中心的交点
               if (value > maxvalue) {
                   maxvalue = value;
                   insect_x = x;
                   insect_y = y;
               }
            }
            // 得到交点坐标值
            realsect.addElem(insect_x);
            realsect.addElem(insect_y);
        }

        // 调用函数得到重组后的曲线，还是存储于 pcurve 中
        interAssemble(pcurve, count, insect, sectnum, realsect, tpl);
        
        // 申请曲线编号大小，空间大小和之前提取的曲线一样
        pcurno = new DynamicArrays[count];
        
        // 调用函数得到曲线编号集合
        makeNode(pcurve, count, pcurno);
        
        // 定义变量，存储图的边数，并且赋值
        int edgenum = count;
        // 定义变量，存储图的点数，并且赋值
        int vexnum = pcurno[count - 1][2] - edgenum + 1;
        // 根据边数和点数，初始化图
        G = new Graph(vexnum, edgenum);
        
        // 根据曲线编号集，给图设置相应的边
        for (i = 0; i < count; i++) {
            G->setEdge(pcurno[i][0], pcurno[i][1], pcurno[i][2]);
        }

        // 定义曲线编号集数组，只有闭合曲线
        DynamicArrays closecurnode[CURVE_VALUE];
        
        // 定义交点编号数组，得到端点坐标对应的编号数
        DynamicArrays intersectno;
        
        // 调用函数得到数组交点的编号
        getPointNo(pcurve, count, pcurno, realsect, intersectno); 

        // 起始闭合和非闭合曲线的数目都设置为 0
        *openNum = 0;
        *closeNum = 0;
        
        // 循环得到闭合曲线的路径编号
        for (i = 0; i < intersectno.getSize(); i++) {
            // 调用函数，得到闭合曲线编号序列集
            closeCurvePath(closecurnode, closeNum, G, intersectno[i]);
        }

        // 申请闭合曲线空间
        closecur = new DynamicArrays[*closeNum];
        
        // 调用函数得到非格式输出的闭合曲线
        getCurveNonFormat(closecurnode, pcurve, count, pcurno, closecur,
                          *closeNum, true);
        
        // 定义曲线总数
        int total = *openNum + *closeNum;

        // 给输出结果赋值，首先申请空间大小
        *curveList = new Curve *[total];
        
        // 定义变量，表示曲线长度
        size_t curveLength;
        
        // 定义变量，表示动态数组里的整型指针
        int *crvData;
        
        // 循环得到输出闭合曲线
        for (i = 0; i < total; i++) {
            // 申请曲线空间
            errcode = CurveBasicOp::newCurve(&((*curveList)[i]));
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i);
                FAIL_CURVETRACING_FREE;
                return OUT_OF_MEM;;
            }
            // 得到曲线长度
            curveLength = (size_t)(closecur[i - *openNum].getSize() / 2);
            // 得到动态数组里的整型指针
            crvData = closecur[i - *openNum].getCrvDatap();
            if (crvData == NULL) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE;
                return NULL_POINTER;
            }
            // 在 CPU 端构建曲线值
            errcode = CurveBasicOp::makeAtHost((*curveList)[i], curveLength,
                                               crvData);
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE;
                return errcode;
            }
        }
    } 
    // 否则只有闭合曲线，且闭合曲线之间没有相交
    else 
    {
        // 重新给该空间内容全部赋值为 -1，用于第二次遍历
        cuerrcode = cudaMemset(array1_dev, -1, arraysize * sizeof (int));
        if (cuerrcode != cudaSuccess) {
            FAIL_CURVETRACING_FREE;
            return CUDA_ERROR;
        }
        // 第二次并行遍历，得到曲线上所有点集
        _traverseKerNew<<<gridsize, blocksize>>>(outsubimgCud1, array1_dev);
        if (cudaGetLastError() != cudaSuccess) {
            FAIL_CURVETRACING_FREE;
            return CUDA_ERROR;
        }
        // 把端点数组拷贝到 Host 端
        cuerrcode = cudaMemcpy(array1, array1_dev, arraysize * sizeof (int),
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            FAIL_CURVETRACING_FREE;
            return CUDA_ERROR;
        }
        // 定义第二次遍历要得到的点集
        DynamicArrays point;

        // 把得到的端点和交点数组的非 -1 值赋值给端点动态数组和交点动态数组
        for (i = 0; i < arraysize; i++) {
            if (array1[i] != -1) {
                point.addElem(array1[i]);
            }
        }

         // 将图像拷贝到 Device 内存中。
        errcode = ImageBasicOp::copyToHost(outimg1);
        if (errcode != NO_ERROR) {
            FAIL_CURVETRACING_FREE;
            return errcode;
        }

        // 申请标志数组的空间，大小和图像一样
        mark = new int[arraysize / 2];
        // 初始化标志数组的值为 0
        memset(mark, 0, sizeof(int) * arraysize / 2);
        // 定义变量 count 表示得到的曲线输量
        int count = 0;
        // 标志曲线跟踪的端点是否已经在曲线中，用于 getCurve 函数调用
        int test  = 0;
        // 申请曲线数组空间，曲线最多数目是端点的个数
        pcurve = new DynamicArrays [point.getSize() / 2];

        // 循环调用 getCurve 函数得到非闭合曲线的有序序列
        for(i = 0; i < point.getSize(); i += 2) {
            getCurve(pcurve, test, count, outimg1, mark, tpl,
                     point[i], point[i + 1]);
            // 如果 test 不为 0，则 count 不加 1，继续循环，否则曲线数目加 1
            if (test) {
                test = 0;
                continue;
            }
            count++;
        }
   
        // 起始闭合和非闭合曲线的数目都设置为 0
        *openNum = 0;
        *closeNum = 0;
        
        *closeNum = count;
        // 定义曲线总数
        int total = count;

        // 给输出结果赋值，首先申请空间大小
        *curveList = new Curve *[total];
        
        // 定义变量，表示曲线长度
        size_t curveLength;
        
        // 定义变量，表示动态数组里的整型指针
        int *crvData;
     
        // 循环得到输出闭合曲线
        for (i = 0; i < total; i++) {
            // 申请曲线空间
            errcode = CurveBasicOp::newCurve(&((*curveList)[i]));
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i);
                FAIL_CURVETRACING_FREE;
                return OUT_OF_MEM;;
            }
            // 得到曲线长度
            curveLength = (size_t)(pcurve[i].getSize() / 2);
            // 得到动态数组里的整型指针
            crvData = pcurve[i].getCrvDatap();
            if (crvData == NULL) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE;
                return NULL_POINTER;
            }
            // 在 CPU 端构建曲线值
            errcode = CurveBasicOp::makeAtHost((*curveList)[i], curveLength,
                                               crvData);
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE;
                return errcode;
            }
        } 
    }

    // 释放动态申请的空间
    FAIL_CURVETRACING_FREE;
    // 函数执行完毕，返回
    return NO_ERROR;
}

// 宏：FAIL_CURVETRACING_FREE_CPU
// 当下面函数运行出错时，使用该宏清除内存，防止内存泄漏。
#define FAIL_CURVETRACING_FREE_CPU  do {           \
        if (outimg1 != NULL) {                     \
            ImageBasicOp::deleteImage(outimg1);    \
            outimg1 = NULL;                        \
        }                                          \
        if (outimg2 != NULL) {                     \
            ImageBasicOp::deleteImage(outimg2);    \
            outimg2 = NULL;                        \
        }                                          \
        if (mark != NULL) {                        \
            delete []mark;                         \
            mark = NULL;                           \
        }                                          \
        if (pcurve != NULL) {                      \
            delete []pcurve;                       \
            pcurve = NULL;                         \
        }                                          \
        if (insect != NULL) {                      \
            delete []insect;                       \
            insect = NULL;                         \
        }                                          \
        if (pcurno != NULL) {                      \
            delete []pcurno;                       \
            pcurno = NULL;                         \
        }                                          \
        if (opencur != NULL) {                     \
            delete []opencur;                      \
            opencur = NULL;                        \
        }                                          \
        if (closecur != NULL) {                    \
            delete []closecur;                     \
            closecur = NULL;                       \
        }                                          \
        if (G != NULL) {                           \
            delete G;                              \
            G = NULL;                              \
        }                                          \
    } while (0)

// Host 成员方法：curveTracingCPU（曲线跟踪）
// 对图像进行曲线跟踪，得到非闭合曲线和闭合曲线的有序序列
__host__ int CurveTracing::curveTracingCPU(Image *inimg, Curve ***curveList,
                                           int *openNum, int *closeNum)
{
    // 如果输入图像指针为空或者输出的曲线集指针为空，错误返回
    if (inimg == NULL || curveList == NULL)
        return NULL_POINTER;

    // 定义错误码变量
    int errcode;

    // 定义输出图像 1 和 2
    Image *outimg1 = NULL;
    Image *outimg2 = NULL;

    // 定义标志数组，标志图像上非零点的访问情况
    int *mark = NULL;
    // 定义曲线数组，存储得到的曲线
    DynamicArrays *pcurve = NULL;
    // 定义交点分类的动态数组，存储分类的结果
    DynamicArrays *insect = NULL;
    // 定义变量，存储曲线的编号;
    DynamicArrays *pcurno = NULL;
    // 定义非闭合曲线
    DynamicArrays *opencur = NULL;
    // 定义闭合曲线
    DynamicArrays *closecur = NULL;
    
    // 定义图类的指针变量
    Graph *G = NULL;

    // 构建输出图像 1 和 2
    ImageBasicOp::newImage(&outimg1);
    ImageBasicOp::makeAtHost(outimg1, inimg->width, inimg->height);

    ImageBasicOp::newImage(&outimg2);
    ImageBasicOp::makeAtHost(outimg2, inimg->width, inimg->height);

    // 定义八领域模板
    int tpl[16] = { -1, -1, 0, -1, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0 };
    
    // 定义临时变量，得到第一次遍历得到的端点和交点动态数组大小
    int num1 = 0, num2 = 0;

    // 定义第一次遍历要得到的端点动态数组和交点动态数组
    DynamicArrays Vertex;
    DynamicArrays Intersect;

    // 定义变量，用于循环
    int i, j, k;
    
    // 定义临时变量存储坐标值
    int dx, dy;
    
    // 遍历图像，得到端点和交点的动态数组
    traverse(Vertex, Intersect, inimg, outimg1, tpl);
    
    // 得到第一次遍历得到的端点和交点动态数组大小
    num1 = Vertex.getSize();
    num2 = Intersect.getSize();
    
    // 如果图像上曲线有端点和交点时，说明有曲线相交，可能有闭合和非闭合曲线
    if (num1 && num2) {
        // 定义第二次遍历要得到的端点动态数组和交点动态数组
        DynamicArrays Vertex1, Intersect1;

        // 第二次遍历图像，得到端点和交点的动态数组
        traverse(Vertex1, Intersect1, outimg1, outimg2, tpl);

        // 定义变量得到输入图像的像素点数目
        int maxnum = inimg->width * inimg->height;
        // 申请标志数组的空间
        mark = new int[maxnum];
        // 初始化标志数组的值为 0
        memset(mark, 0, sizeof(int) * maxnum);
        // 定义变量 count 表示得到的曲线输量
        int count = 0;
        // 标志曲线跟踪的端点是否已经在曲线中，用于 getCurve 函数调用
        int test  = 0;
        // 定义曲线数组，并且申请空间，曲线最多数目是端点的个数
        DynamicArrays *pcurve = new DynamicArrays [Vertex1.getSize() / 2];

        // 循环调用 getCurve 函数得到非闭合曲线的有序序列
        for(i = 0; i < Vertex1.getSize(); i += 2) {
            getCurve(pcurve, test, count, outimg1, mark, tpl,
                     Vertex1[i], Vertex1[i + 1]);
            // 如果 test 不为 0，则 count 不加 1，继续循环，否则曲线数目加 1
            if (test) {
                test = 0;
                continue;
            }
            count++;
        }

        // 定义临时变量存储坐标值
        int x, y;
        // 定义变量，存储交点的个数
        int sectnum = 0; 
        // 定义交点分类的动态数组，存储分类的结果，并且申请空间
        insect = new DynamicArrays [num2 / 2];

        // 循环得到交点分类动态数组值
        while (Intersect.getSize()) {
            x = Intersect[0];
            y = Intersect[1];
            sectnum++;
            insectClassify(x, y, Intersect, insect, sectnum, tpl);

        }

        // 定义真正的交点数组，得到的是唯一确定的交点，与交点曲线方向动态数组集
        // 相对应，大小其实为交点个数。从分类的交点集中取领域数最大的点作为交点
        DynamicArrays realsect;

        // 循环得到交点曲线方向动态数组集和真正的交点数组
        for (i = 0; i < sectnum; i++) {
            // 定义变量，存储领域数最大的点标记值，并初始化为 0
            int maxvalue = 0;
            // 定义变量，存储坐标值，初始化为第一条曲线第一个点的坐标值
            int insect_x = insect[i][0], insect_y = insect[i][1];
            // 根据之前的分类结果，循环得到交点曲线方向动态数组
            for (j = 0; j < insect[i].getSize(); j += 2) {
               x = insect[i][j];
               y = insect[i][j + 1];
               // 定义临时变量，存储分类集合中的点的八领域内有多少个点
               int value = 0;
               for (k = 0; k < 16; k += 2) {
                   dx = x + tpl[k];
                   dy = y + tpl[k + 1];
                   // 遍历点周围有多少个点
                   for (int s = 0; s < insect[i].getSize(); s += 2) {
                       if (dx == insect[i][s] && dy == insect[i][s + 1]) {
                           value++;
                       }
                   }
               }
               // 找到最中心的交点
               if (value > maxvalue) {
                   maxvalue = value;
                   insect_x = x;
                   insect_y = y;
               }
            }
            // 得到交点坐标值
            realsect.addElem(insect_x);
            realsect.addElem(insect_y);
        }

        // 调用函数得到重组后的曲线，还是存储于 pcurve 中
        interAssemble(pcurve, count, insect, sectnum, realsect, tpl);
        
        // 定义变量，存储曲线的编号，空间大小和之前提取的曲线一样
        pcurno = new DynamicArrays[count];
        
        // 调用函数得到曲线编号集合
        makeNode(pcurve, count, pcurno);
        
        // 定义变量，存储图的边数，并且赋值
        int edgenum = count;
        // 定义变量，存储图的点数，并且赋值
        int vexnum = pcurno[count - 1][2] - edgenum + 1;
        // 定义图的指针变量，根据边数和点数，初始化图
        G = new Graph(vexnum, edgenum);
        
        // 根据曲线编号集，给图设置相应的边
        for (i = 0; i < count; i++) {
            G->setEdge(pcurno[i][0], pcurno[i][1], pcurno[i][2]);
        }

        // 定义曲线编号集数组，分为非闭合曲线和闭合曲线
        DynamicArrays opencurnode[CURVE_VALUE], closecurnode[CURVE_VALUE];
        
        // 定义端点编号数组和交点编号数组，分别得到顶点和端点的坐标对应的编号数
        DynamicArrays vertexno;
        DynamicArrays intersectno;
        
        // 调用函数得到数组端点的编号
        getPointNo(pcurve, count, pcurno, Vertex, vertexno); 
        
        // 调用函数得到数组交点的编号
        getPointNo(pcurve, count, pcurno, realsect, intersectno); 

        
        // 起始闭合和非闭合曲线的数目都设置为 0
        *openNum = 0;
        *closeNum = 0;
        
        // 循环得到非闭合曲线的路径编号
        for (i = 0; i < vertexno.getSize(); i++) {
            // 定义起始点
            int start, end;
            start = vertexno[i];
            for (j = i + 1; j < vertexno.getSize(); j++) {
                end = vertexno[j];
                // 调用函数，得到非闭合曲线编号序列集
                openCurvePath(opencurnode, openNum, G, start, end);
            }
        }

        // 循环得到闭合曲线的路径编号
        for (i = 0; i < intersectno.getSize(); i++) {
            // 调用函数，得到闭合曲线编号序列集
            closeCurvePath(closecurnode, closeNum, G, intersectno[i]);
        }

        // 定义非闭合曲线，并且申请空间
        opencur = new DynamicArrays[*openNum];
        
        // 定义闭合曲线，并且申请大小空间
        closecur = new DynamicArrays[*closeNum];

        // 调用函数得到非格式输出的非闭合曲线
        getCurveNonFormat(opencurnode, pcurve, count, pcurno, opencur,
                          *openNum, false);
        
        // 调用函数得到非格式输出的闭合曲线
        getCurveNonFormat(closecurnode, pcurve, count, pcurno, closecur,
                          *closeNum, true);
        
        // 定义曲线总数
        int total = *openNum + *closeNum;

        // 给输出结果赋值，首先申请空间大小
        *curveList = new Curve *[total];
        
        // 定义变量，表示曲线长度
        size_t curveLength;
        
        // 定义变量，表示动态数组里的整型指针
        int *crvData;

        // 循环得到输出非闭合曲线
        for (i = 0; i < *openNum; i++) {
            // 申请曲线空间
            errcode = CurveBasicOp::newCurve(&((*curveList)[i]));
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i);
                FAIL_CURVETRACING_FREE_CPU;
                return OUT_OF_MEM;
            }
            // 得到曲线长度
            curveLength = (size_t)(opencur[i].getSize() / 2);
            // 得到动态数组里的整型指针
            crvData = opencur[i].getCrvDatap();
            if (crvData == NULL) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return NULL_POINTER;
            }
            // 在 CPU 端构建曲线值
            errcode = CurveBasicOp::makeAtHost((*curveList)[i], curveLength,
                                               crvData);
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return errcode;
            }
        }
        
        // 循环得到输出闭合曲线
        for (; i < total; i++) {
            // 申请曲线空间
            errcode = CurveBasicOp::newCurve(&((*curveList)[i]));
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i);
                FAIL_CURVETRACING_FREE_CPU;
                return OUT_OF_MEM;;
            }
            // 得到曲线长度
            curveLength = (size_t)(closecur[i - *openNum].getSize() / 2);
            // 得到动态数组里的整型指针
            crvData = closecur[i - *openNum].getCrvDatap();
            if (crvData == NULL) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return NULL_POINTER;
            }
            // 在 CPU 端构建曲线值
            errcode = CurveBasicOp::makeAtHost((*curveList)[i], curveLength,
                                               crvData);
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return errcode;
            }
        }
    
    } 
    
    // 如果图像上没有端点只有交点时候，说明是闭合曲线相交
    else if (num1 && !num2) {
        // 定义变量得到输入图像的像素点数目
        int maxnum = inimg->width * inimg->height;
        // 定义标志数组，并且申请和图像大小的空间
        mark = new int[maxnum];
        // 初始化标志数组的值为 0
        memset(mark, 0, sizeof(int) * maxnum);
        // 定义变量 count 表示得到的曲线输量
        int count = 0;
        // 标志曲线跟踪的端点是否已经在曲线中，用于 getCurve 函数调用
        int test  = 0;
        // 定义曲线数组，并且申请空间，曲线最多数目是端点的个数
        DynamicArrays *pcurve = new DynamicArrays [Vertex.getSize() / 2];

        // 循环调用 getCurve 函数得到非闭合曲线的有序序列
        for(i = 0; i < Vertex.getSize(); i += 2) {
            getCurve(pcurve, test, count, outimg1, mark, tpl,
                     Vertex[i], Vertex[i + 1]);
            // 如果 test 不为 0，则 count 不加 1，继续循环，否则曲线数目加 1
            if (test) {
                test = 0;
                continue;
            }
            count++;
        }

        // 定义变量，存储曲线的编号，空间大小和之前提取的曲线一样
        pcurno = new DynamicArrays[count];
        
        // 调用函数得到曲线编号集合
        makeNode(pcurve, count, pcurno);
        
        // 定义变量，存储图的边数，并且赋值
        int edgenum = count;
        // 定义变量，存储图的点数，并且赋值
        int vexnum = pcurno[count - 1][2] - edgenum + 1;
        // 定义图的指针变量，根据边数和点数，初始化图
        G = new Graph(vexnum, edgenum);
        
        // 根据曲线编号集，给图设置相应的边
        for (i = 0; i < count; i++) {
            G->setEdge(pcurno[i][0], pcurno[i][1], pcurno[i][2]);
        }

        // 定义曲线编号集数组，只有非闭合曲线
        DynamicArrays opencurnode[CURVE_VALUE];
        
        // 定义端点编号数组和交点编号数组，分别得到顶点和端点的坐标对应的编号数
        DynamicArrays vertexno;

        // 调用函数得到数组端点的编号
        getPointNo(pcurve, count, pcurno, Vertex, vertexno); 

        // 起始闭合和非闭合曲线的数目都设置为 0
        *openNum = 0;
        *closeNum = 0;
        
        // 循环得到非闭合曲线的路径编号
        for (i = 0; i < vertexno.getSize(); i++) {
            // 定义起始点
            int start, end;
            start = vertexno[i];
            for (j = i + 1; j < vertexno.getSize(); j++) {
                end = vertexno[j];
                // 调用函数，得到非闭合曲线编号序列集
                openCurvePath(opencurnode, openNum, G, start, end);
            }
        }

        // 定义非闭合曲线，并且申请空间
        opencur = new DynamicArrays[*openNum];
        
        // 调用函数得到非格式输出的非闭合曲线
        getCurveNonFormat(opencurnode, pcurve, count, pcurno, opencur,
                          *openNum, false);
        
        // 定义曲线总数
        int total = *openNum;

        // 给输出结果赋值，首先申请空间大小
        *curveList = new Curve *[total];
        
        // 定义变量，表示曲线长度
        size_t curveLength;
        
        // 定义变量，表示动态数组里的整型指针
        int *crvData;
        
        // 循环得到输出非闭合曲线
        for (i = 0; i < *openNum; i++) {
            // 申请曲线空间
            errcode = CurveBasicOp::newCurve(&((*curveList)[i]));
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i);
                FAIL_CURVETRACING_FREE_CPU;
                return OUT_OF_MEM;
            }
            // 得到曲线长度
            curveLength = (size_t)(opencur[i].getSize() / 2);
            // 得到动态数组里的整型指针
            crvData = opencur[i].getCrvDatap();
            if (crvData == NULL) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return NULL_POINTER;
            }
            // 在 CPU 端构建曲线值
            errcode = CurveBasicOp::makeAtHost((*curveList)[i], curveLength,
                                               crvData);
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return errcode;
            }
        }
    }
    
    // 如果图像上没有端点只有交点时候，说明是闭合曲线相交
    else if (!num1 && num2) 
    {
        // 定义第二次遍历要得到的端点动态数组和交点动态数组
        DynamicArrays Vertex1, Intersect1;

        // 第二次遍历图像，得到端点和交点的动态数组
        traverse(Vertex1, Intersect1, outimg1, outimg2, tpl);

        // 定义变量得到输入图像的像素点数目
        int maxnum = inimg->width * inimg->height;
        // 定义标志数组，并且申请和图像大小的空间
        mark = new int[maxnum];
        // 初始化标志数组的值为 0
        memset(mark, 0, sizeof(int) * maxnum);
        // 定义变量 count 表示得到的曲线输量
        int count = 0;
        // 标志曲线跟踪的端点是否已经在曲线中，用于 getCurve 函数调用
        int test  = 0;
        // 定义曲线数组，并且申请空间，曲线最多数目是端点的个数
        DynamicArrays *pcurve = new DynamicArrays [Vertex1.getSize() / 2];

        // 循环调用 getCurve 函数得到非闭合曲线的有序序列
        for(i = 0; i < Vertex1.getSize(); i += 2) {
            getCurve(pcurve, test, count, outimg1, mark, tpl,
                     Vertex1[i], Vertex1[i + 1]);
            // 如果 test 不为 0，则 count 不加 1，继续循环，否则曲线数目加 1
            if (test) {
                test = 0;
                continue;
            }
            count++;
        }

        // 定义临时变量存储坐标值
        int x, y;
        // 定义变量，存储交点的个数
        int sectnum = 0; 
        // 定义交点分类的动态数组，存储分类的结果，并且申请空间
        insect = new DynamicArrays [num2 / 2];

        // 循环得到交点分类动态数组值
        while (Intersect.getSize()) {
            x = Intersect[0];
            y = Intersect[1];
            sectnum++;
            insectClassify(x, y, Intersect, insect, sectnum, tpl);
        }

        // 定义真正的交点数组，得到的是唯一确定的交点，与交点曲线方向动态数组集
        // 相对应，大小其实为交点个数。从分类的交点集中取领域数最大的点作为交点
        DynamicArrays realsect;

        // 循环得到交点曲线方向动态数组集和真正的交点数组
        for (i = 0; i < sectnum; i++) {
            // 定义变量，存储领域数最大的点标记值，并初始化为 0
            int maxvalue = 0;
            // 定义变量，存储坐标值，初始化为第一条曲线第一个点的坐标值
            int insect_x = insect[i][0], insect_y = insect[i][1];
            // 根据之前的分类结果，循环得到交点曲线方向动态数组
            for (j = 0; j < insect[i].getSize(); j += 2) {
               x = insect[i][j];
               y = insect[i][j + 1];
               // 定义临时变量，存储分类集合中的点的八领域内有多少个点
               int value = 0;
               for (k = 0; k < 16; k += 2) {
                   dx = x + tpl[k];
                   dy = y + tpl[k + 1];
                   // 遍历点周围有多少个点
                   for (int s = 0; s < insect[i].getSize(); s += 2) {
                       if (dx == insect[i][s] && dy == insect[i][s + 1]) {
                           value++;
                       }
                   }
               }
               // 找到最中心的交点
               if (value > maxvalue) {
                   maxvalue = value;
                   insect_x = x;
                   insect_y = y;
               }
            }
            // 得到交点坐标值
            realsect.addElem(insect_x);
            realsect.addElem(insect_y);
        }

        // 调用函数得到重组后的曲线，还是存储于 pcurve 中
        interAssemble(pcurve, count, insect, sectnum, realsect, tpl);
        
        // 定义变量，存储曲线的编号，空间大小和之前提取的曲线一样
        pcurno = new DynamicArrays[count];
        
        // 调用函数得到曲线编号集合
        makeNode(pcurve, count, pcurno);
        
        // 定义变量，存储图的边数，并且赋值
        int edgenum = count;
        // 定义变量，存储图的点数，并且赋值
        int vexnum = pcurno[count - 1][2] - edgenum + 1;
        // 定义图的指针变量，根据边数和点数，初始化图
        G = new Graph(vexnum, edgenum);
        
        // 根据曲线编号集，给图设置相应的边
        for (i = 0; i < count; i++) {
            G->setEdge(pcurno[i][0], pcurno[i][1], pcurno[i][2]);
        }

        // 定义曲线编号集数组，只有闭合曲线
        DynamicArrays closecurnode[CURVE_VALUE];
        
        // 定义交点编号数组，得到端点坐标对应的编号数
        DynamicArrays intersectno;
        
        // 调用函数得到数组交点的编号
        getPointNo(pcurve, count, pcurno, realsect, intersectno); 

        // 起始闭合和非闭合曲线的数目都设置为 0
        *openNum = 0;
        *closeNum = 0;
        
        // 循环得到闭合曲线的路径编号
        for (i = 0; i < intersectno.getSize(); i++) {
            // 调用函数，得到闭合曲线编号序列集
            closeCurvePath(closecurnode, closeNum, G, intersectno[i]);
        }
  
        // 定义闭合曲线，并且申请大小空间
        closecur = new DynamicArrays[*closeNum];
        
        // 调用函数得到非格式输出的闭合曲线
        getCurveNonFormat(closecurnode, pcurve, count, pcurno, closecur,
                          *closeNum, true);
        
        // 定义曲线总数
        int total = *openNum + *closeNum;

        // 给输出结果赋值，首先申请空间大小
        *curveList = new Curve *[total];
        
        // 定义变量，表示曲线长度
        size_t curveLength;
        
        // 定义变量，表示动态数组里的整型指针
        int *crvData;
        
        // 循环得到输出闭合曲线
        for (i = 0; i < total; i++) {
            // 申请曲线空间
            errcode = CurveBasicOp::newCurve(&((*curveList)[i]));
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i);
                FAIL_CURVETRACING_FREE_CPU;
                return OUT_OF_MEM;;
            }
            // 得到曲线长度
            curveLength = (size_t)(closecur[i - *openNum].getSize() / 2);
            // 得到动态数组里的整型指针
            crvData = closecur[i - *openNum].getCrvDatap();
            if (crvData == NULL) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return NULL_POINTER;
            }
            // 在 CPU 端构建曲线值
            errcode = CurveBasicOp::makeAtHost((*curveList)[i], curveLength,
                                               crvData);
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return errcode;
            }
        }
    } 

    // 否则只有闭合曲线，且闭合曲线之间没有相交
    else 
    {
        // 定义第二次遍历要得到的点集
        DynamicArrays point;

        // 第二次遍历图像，得到端点和交点的动态数组
        traverseNew(point, outimg1);

        // 定义变量得到输入图像的像素点数目
        int maxnum = inimg->width * inimg->height;
        // 定义标志数组，并且申请和图像大小的空间
        mark = new int[maxnum];
        // 初始化标志数组的值为 0
        memset(mark, 0, sizeof(int) * maxnum);
        // 定义变量 count 表示得到的曲线输量
        int count = 0;
        // 标志曲线跟踪的端点是否已经在曲线中，用于 getCurve 函数调用
        int test  = 0;
        // 定义曲线数组，并且申请空间，曲线最多数目是端点的个数
        pcurve = new DynamicArrays [point.getSize()];

        // 循环调用 getCurve 函数得到非闭合曲线的有序序列
        for(i = 0; i < point.getSize(); i += 2) {
            getCurve(pcurve, test, count, outimg1, mark, tpl,
                     point[i], point[i + 1]);
            // 如果 test 不为 0，则 count 不加 1，继续循环，否则曲线数目加 1
            if (test) {
                test = 0;
                continue;
            }
            count++;
        }

        // 起始闭合和非闭合曲线的数目都设置为 0
        *openNum = 0;
        *closeNum = 0;
        
        *closeNum = count;
        // 定义曲线总数
        int total = *openNum + *closeNum;

        // 给输出结果赋值，首先申请空间大小
        *curveList = new Curve *[total];
        
        // 定义变量，表示曲线长度
        size_t curveLength;
        
        // 定义变量，表示动态数组里的整型指针
        int *crvData;
        
        // 循环得到输出闭合曲线
        for (i = 0; i < total; i++) {
            // 申请曲线空间
            errcode = CurveBasicOp::newCurve(&((*curveList)[i]));
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i);
                FAIL_CURVETRACING_FREE_CPU;
                return OUT_OF_MEM;;
            }
            // 得到曲线长度
            curveLength = (size_t)(pcurve[i].getSize() / 2);
            // 得到动态数组里的整型指针
            crvData = pcurve[i].getCrvDatap();
            if (crvData == NULL) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return NULL_POINTER;
            }
            // 在 CPU 端构建曲线值
            errcode = CurveBasicOp::makeAtHost((*curveList)[i], curveLength,
                                               crvData);
            if (errcode != NO_ERROR) {
                // 释放动态申请的空间
                freeCurve(curveList, i + 1);
                FAIL_CURVETRACING_FREE_CPU;
                return errcode;
            }
        } 
    }

    // 释放动态申请的空间
    FAIL_CURVETRACING_FREE_CPU;
    // 函数执行完毕，返回
    return NO_ERROR;
}
