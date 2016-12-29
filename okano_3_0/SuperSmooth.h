// SuperSmooth.h
// 创建人：刘婷

// 超平滑（SuperSmooth）
// 功能说明：对于一幅图像，分别先做两种平滑处理，一种是在每一个点的八个方向上分
//           别做平滑处理；另一种是利用点的邻域范围内的点做平滑（例如，对于 3*3
//           邻域内的像素值排序，利用前 10% 的平均值和后 10% 的平均值进行处理），
//           最后再结合两种平滑的结果做最终的处理，得到最终的结果图像。
//
// 修订历史：
// 2013年09月05日（刘婷）
//     编写头文件。
// 2013年09月09日（刘婷）
//     实现第 1 个核函数
// 2013年09月10日（刘婷）
//     实现第 2 个核函数
// 2013年09月11日（刘婷）
//     实现第 3 个核函数
// 2013年09月12日（刘婷）
//     调试代码，更改一些重要的 bug。
// 2013年09月13日（刘婷）
//     更改源文件中的一些 bug。
// 2013年09月14日（刘婷）
//     继续调试代码，修改文件规范。
// 2013年12月05日（刘婷）
//     根据2013年11月5日河边老师新发来的文档做了更改，增加了属性searchScope，该
//     属性在 LocalCluster 中表示每一个方向上搜索的最大范围。

#ifndef __SUPERSMOOTH_H__
#define __SUPERSMOOTH_H__

#include "ErrorCode.h"
#include "Image.h"

// 按照河边老师的文档内容，该宏为 LocalCluster 在每一个方向上的搜索范围。
#define MAX_SEARCHSCOPE  16

// 类：SuperSmooth（超平滑）
// 继承自：无
// 对于一幅图像，分别先做两种平滑处理，一种是在每一个点的八个方向上分别做平滑处
// 理；另一种是利用点的邻域范围内的点做平滑（例如，对于 3*3 邻域内的像素值排序，
// 利用前 10% 的平均值和后 10% 的平均值进行处理），最后再结合两种平滑的结果做最
// 终的处理，得到最终的结果图像。
class SuperSmooth {

protected:

    // 成员变量：diffThred（与像素值差相关的阈值）
    // 与当前像素点和某方向上遭遇的点的像素值差相关的阈值。
    int diffThred;

    // 成员变量：diffCntThred（与点的个数相关的阈值）
    // 当正遭遇点的像素值与当前计算点的像素值差大于等于 difftThred 的多个连续点
    // 的个数超过 diffCntThred 时停止该方向的遍历。
    int diffCntThred;

    // 成员变量：cCntThred（与点的个数相关的阈值，但不同于 diffCntThred）
    // 它与当前像素点的像素值与正遭遇点的像素值差小于 diffThred 的点的个数有关。
    int cCntThred;

    // 成员变量：searchScope（LocalCluster kernel中每一个方向上搜索的最大范围）
    // LocalCluster kernel中在每一个方向上搜索的最大范围。按照河边老师的需求该
    // 值小于 16。
    int searchScope;

    // 成员变量：windowSize（邻域尺寸）
    // 利用邻域进行平滑操作的时候用来限定邻域大小。
    int windowSize;

public:

    // 构造函数：SuperSmooth
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    SuperSmooth()
    {
        // 使用默认值为成员变量赋值。
        this->diffThred = 10;    // 当前点与遭遇到的点的像素值差的阈值，设置初始
                                 // 值为 60。
        this->diffCntThred = 5;  // 不满足 diffThred 的点个数的上限，若超过则停
                                 // 止该方向上的遍历，初始值设为 5。
        this->cCntThred = 15;    // 在四个方向（左右，上下，两个对角线方向）满
                                 // 足 diffThred 要求的点的个数的下限，初始值
                                 // 设置为 10。
        this->windowSize = 5;    // 邻域尺寸初始值设置为 5，即默认 3 * 3 的邻域
                                 // 范围。
        this->searchScope = 8;   // LocalCluster kernel中在每一个方向上搜索的最
                                 // 大范围。
    }

    // 构造函数：SuperSmooth
    // 有参数版本的构造函数。
    __host__ __device__
    SuperSmooth(
            int diffthred,     // 变量含义请见上
            int diffcntthred,  // 变量含义请见上
            int ccntthred,     // 变量含义请见上
            int windowsize,    // 变量含义请见上
            int searchscope    // 变量含义请见上
    ){
        // 使用默认值为成员变量赋值，
        this->diffThred = 10;    // 当前点与遭遇到的点的像素值差的阈值，设置初
                                 // 始值为 60。
        this->diffCntThred = 5;  // 不满足 diffThred 的点个数的上限，若超过则
                                 // 停止该方向上的遍历，初始值设为 5。
        this->cCntThred = 15;    // 在四个方向（左右，上下，两个对角线方向）满
                                 // 足 diffThred 要求的点的个数的下限，初始值
                                 // 设置为 10。
        this->windowSize = 5;    // 邻域尺寸初始值设置为 5，即默认 3 * 3 的邻域
                                 // 范围。
        this->searchScope = 8;   // LocalCluster kernel中在每一个方向上搜索的最
                                 // 大范围。
        // 使用给定的参数初始化。
        setDiffThred(diffthred);
        setDiffCntThred(diffcntthred);
        setCCntThred(ccntthred);
        setWindowSize(windowsize);
        setWindowSize(searchscope);
    }

    // 成员方法：getDiffThred（获取成员变量 diffThred 的值）
    // 获取成员变量 diffThred 的值。
    __host__ __device__ int  // 返回值：成员变量 diffThred 的值。
    getDiffThred() const
    {
        // 返回成员变量 diffThred 的值。
        return this->diffThred;    
    } 

    // 成员方法：setDiffThred（设置成员变量 diffThred 的值）
    // 设置成员变量 diffThred 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，
                             // 返回 NO_ERROR。
    setDiffThred(
            int diffthred  // 新的 diffThred 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (diffthred <= 0)
            return INVALID_DATA;

        // 为成员变量 diffThred 赋新值。
        this->diffThred = diffthred;

        return NO_ERROR;
    }

    // 成员方法：getDiffCntThred（获取成员变量 diffCntThred 的值）
    // 获取成员变量 diffCntThred 的值。
    __host__ __device__ int  // 返回值：成员变量 diffCntThred 的值。
    getDiffCntThred() const
    {
        // 返回成员变量 diffCntThred 的值。
        return this->diffCntThred;    
    } 

    // 成员方法：setDiffCntThred（设置成员变量 diffCntThred 的值）
    // 设置成员变量 diffCntThred 的值。
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR。
    setDiffCntThred(
            int diffcntthred  // 新的 diffCntThred 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (diffcntthred <= 0)
            return INVALID_DATA;

        // 为成员变量 diffCntThred 赋新值。
        this->diffCntThred = diffcntthred;

        return NO_ERROR;
    }

    // 成员方法：getCCntThred（获取成员变量 cCntThred 的值）
    // 获取成员变量 cCntThred 的值。
    __host__ __device__ int  // 返回值：成员变量 cCntThred 的值。
    getCCntThred() const
    {
        // 返回成员变量 cCntThred 的值。
        return this->cCntThred;    
    } 

    // 成员方法：setCCntThred（设置成员变量 cCntThred 的值）
    // 设置成员变量 cCntThred 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，
                             // 返回 NO_ERROR。
    setCCntThred(
            int ccntthred    // 新的 cCntThred 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (ccntthred <= 0)
            return INVALID_DATA;

        // 为成员变量 cCntThred 赋新值。
        this->cCntThred = ccntthred;

        return NO_ERROR;
    }

    // 成员方法：getSearchScope（获取成员变量 SearchScope 的值）
    // 获取成员变量 cCntThred 的值。
    __host__ __device__ int  // 返回值：成员变量 SearchScope 的值。
    getSearchScope() const
    {
        // 返回成员变量 searchScope 的值。
        return this->searchScope;    
    } 

    // 成员方法：setSearchScope（设置成员变量 searchScope 的值）
    // 设置成员变量 searchScope 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，
                             // 返回 NO_ERROR。
    setSearchScope(
            int searchscope    // 新的 searchScope 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (searchscope <= 0 || searchscope >= MAX_SEARCHSCOPE)
            return INVALID_DATA;

        // 为成员变量 searchScope 赋新值。
        this->searchScope = searchscope;

        return NO_ERROR;
    }

    // 成员方法：getWindowSize（获取成员变量 windowSize的值）
    // 获取成员变量 windowSize 的值。
    __host__ __device__ int  // 返回值：成员变量 windowSize 的值。
    getWindowSize() const
    {
        // 返回成员变量 windowSize 的值。
        return this->windowSize;    
    } 

    // 成员方法：setWindow（设置成员变量 windowSize 的值）
    // 设置成员变量 windowSize 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，
                             // 返回 NO_ERROR。
    setWindowSize(
            int windowsize   // 新的 windowSize 值
    ) {
        // 判断参数是否合法，若不合法则报错。
        if (windowsize <= 0)
            return INVALID_DATA;

        // 为成员变量 windowSize 赋新值。
        this->windowSize = windowsize;

        return NO_ERROR;
    }

    // 成员方法：superSmooth（超平滑）
    // 给定图像 ，对图像进行平滑处理，得到超级平滑后的图像。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，
                           // 返回 NO_ERROR。
    superSmooth(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    ); 

};

#endif

