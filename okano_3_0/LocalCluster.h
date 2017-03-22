// LocalCluster.h
// 创建人：刘婷
//
// 局部聚类（Local Cluster）
// 功能说明：给定一张图像略去图像的边缘部分，在每点的八个方向上各
//           求出 pntRange 个点的像素平均值(根据河边老师发来的
//           串行实现代码，pntRange 不能超过 100），利用这些
//           平均值与当前点的图像值做差存放在临时变量(temp）中，
//           再在这个数组中选取最接近 0 的 pntCount 个值求出其
//           平均值(根据河边老师发来的串行实现代码，pntCount 
//           不能超过 8），将该平均值与当前像素值相加最终得出点的
//           新像素值。
//
// 修订历史：
// 2012年10月15日（刘婷）
//     进行类设计
// 2012年10月16日（刘婷）
//     函数的初步实现
// 2012年10月28日 （于玉龙，刘瑶，刘婷）
//     修改了代码中的一些错误
// 2012年10月29日（刘婷）
//     修改了代码中的一些错误
// 2012年11月03日（于玉龙，刘婷）
//     修改代码格式
// 2012年11月04日（于玉龙，刘婷）
//     修改代码格式
// 2012年11月14日（刘婷）
//     修改了代码并合并了 kernel
// 2012年12月02日（于玉龙，刘婷）
//     调整代码格式
// 2012年12月04日（刘婷）
//     更新了 kernel，减少了一些重复计算
// 2012年12月12日（刘婷）
//     更改了对 temp 数组的操作
// 2012年12月24日（刘婷）
//     更改了处理 temp 的一处 bug
// 2012年12月25日（刘婷）
//     更改了处理 temp 的一处 bug

#ifndef __LOCALCLUSTER__H
#define __LOCALCLUSTER__H

#include "Image.h"
#include "ErrorCode.h"


// 类:LocalCluster
// 继承自：无
// 给定一张图像略去图像的边缘部分，在每一个点的八个方向上各求出 
// pntRange 个点的像素平均值(根据河边老师发来的串行实现代码，pntRange 
// 不能超过 100），利用这些平均值与当前点的图像值做差存放在一个临时变量
// temp 中，再在这个数组中选取最接近 0 的 pntCount 个值求出其平均值 
// (根据河边老师发来的串行实现代码，pntCount 不能超过 8），将该平均值
// 与当前像素值相加，最终得出点的新像素值。
class LocalCluster {

protected:

    // 成员变量：gapThred
    // 当前像素点和相邻点的灰度差的阈值
    unsigned char gapThred;
    
    // 成员变量：diffeThred
    // 正在计算点的两侧，每侧取两个紧邻当前点的像素点相加
    // 得到 side1 和 side2， diffThred 是其差的阈值。
    unsigned char diffeThred;
    
    // 成员变量:proBlack， proWhite
    // 具有显著性的点的像素值划分上界和下界
    // 像素值，其中 proWhite 默认为 250， proBlack 默认为 10。
    unsigned char proWhite, proBlack;
    
    // 成员变量:pntRange
    // 在当前像素点的八个方向上，每个方向上取得点的个数，
    // 根据河边老师发来的串行实现代码，不超过 100。
    int pntRange;
    
    // 成员变量:pntCount
    // 利用循环在 temp 中寻找最接近 0 的值时，循环次数的上界，
    // 根据河边老师发来的串行实现代码，不超过 8。
    int pntCount;
    
public:

    // 构造函数:LocalCluster
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    LocalCluster()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->diffeThred = 40;  // 正在计算点的两侧差的阈值 
        this->gapThred = 40;    // 当前像素点和相邻点的灰度差的阈值
        this->pntCount = 7;     // 循环次数的上界，
                                // 不超过 8。
        this->pntRange = 16;    // 每个方向上取得点的个数，
                                // 不超过 100。
        this->proBlack = 10;    // 具有显著性的点的像素值划分上界
        this->proWhite = 250;   // 具有显著性的点的像素值划分下界
    }
    
    // 构造函数:LocalCluster
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 可以改变的。
    // proBlack 和 proWhite 采用默认为 10 和 250，用户可以不输入
    // 这两个值而调用默认值。
    __host__ __device__
    LocalCluster(
            unsigned char gapthred,       // 解释见成员变量
            unsigned char diffethred,     // 解释见成员变量
            int pntrange,                 // 解释见成员变量
            int pntcount,                 // 解释见成员变量
            unsigned char problack = 10,  // 像素值，默认为 10
            unsigned char prowhite = 250  // 像素值，默认为 250
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给
        // 了非法的初始值而使系统进入一个未知的状态。
        this->diffeThred = 10;  // 正在计算点的两侧差的阈值 
        this->gapThred = 10;    // 当前像素点和相邻点的灰度差的阈值
        this->pntCount = 5;     // 循环次数的上界
        this->pntRange = 8;     // 每个方向上取得点的个数
        this->proBlack = 10;    // 具有显著性的点的像素值划分上界
        this->proWhite = 250;   // 具有显著性的点的像素值划分下界
        
        // 根据参数列表中的值设定成员变量的初值
        this->setGapThred(gapthred);
        this->setDiffeThred(diffethred);
        this->setPntRange(pntrange);
        this->setPntCount(pntcount);
        this->setProBlack(problack);
        this->setProWhite(prowhite);
    }
    
    // 成员方法：getGapThred（读取当前像素点和相邻点的灰度差的阈值）
    // 读取 gapThred 成员变量的值
    __host__ __device__ unsigned char  // 返回值：
                                       // gapThred 成员变量的值。
    getGapThred() const
    {
        // 返回 gapThred 成员变量的值。
        return this->gapThred;
    }
    
    // 成员方法：setGapThred（设置当前像素点和相邻点的灰度差的阈值）
    // 设置 gapThred 成员变量的值
    __host__ __device__ int         // 返回值：函数是否正确执行，若
                                    // 函数正确执行，返回 NO_ERROR。 
    setGapThred(
            unsigned char gapthred  // 新 gapThred 值
    ) {
        // 将 gapThred 成员变量赋成新值
        this->gapThred = gapthred;

        return NO_ERROR;
    }
    
    // 成员方法：getDiffeThred（读取正在计算点两侧差的阈值）
    // 读取 gapThred 成员变量的值
    __host__ __device__ unsigned char  // 返回值：当前 diffeThred 
                                       // 成员变量的值。
    getDiffeThred() const
    {
        // 返回 diffeTred 成员变量的值。
        return this->diffeThred;
    }
    
    // 成员方法：setDiffeThred（设置正在计算点两侧差的阈值）
    // 设置 gapThred 成员变量的值
    __host__ __device__ int           // 返回值：函数是否正确执行，
                                      // 若函数正确执行，返回 NO_ERROR。 
    setDiffeThred(
            unsigned char diffethred  // 新 diffeThred 值
    ) {
        // 将 diffeThred 成员变量赋成新值
        this->diffeThred = diffethred;

        return NO_ERROR;
    }
    
    // 成员方法：getProBlack（读取具有显著性的点的像素值划分上界）
    // 读取 proBlack 成员变量的值
    __host__ __device__ unsigned char  // 返回值：当前 proBlack 
                                       // 成员变量的值。
    getProBlack() const
    {
        // 返回 proBlack 成员变量的值。
        return this->proBlack;
    }
    
    // 成员方法：setProBlack（设置具有显著性的点的像素值划分上界）
    // 设置 proBlack 成员变量的值
    __host__ __device__ int         // 返回值：函数是否正确执行，若
                                    // 函数正确执行，返回 NO_ERROR。 
    setProBlack(
            unsigned char problack  // 新 proBlack 值
    ) {    
        // 将 proBlack 成员变量赋成新值
        this->proBlack = problack;

        return NO_ERROR;
    }
    
    // 成员方法：getProWhite（读取具有显著性的点的像素值划分下界）
    // 读取 proWhite 成员变量的值
    __host__ __device__ unsigned char  // 返回值：当前 proWhite 
                                       // 成员变量的值。
    getProWhite() const
    {
        // 返回 proWhite 成员变量的值。
        return this->proWhite;
    }
    
    // 成员方法：setProWhite（设置具有显著性的点的像素值划分界下）
    // 设置 proWhite 成员变量的值
    __host__ __device__ int         // 返回值：函数是否正确执行，若
                                    // 函数正确执行，返回 NO_ERROR。 
    setProWhite(
            unsigned char prowhite  // 新 proWhite 值
    ) {
            
        // 将 proWhite 成员变量赋成新值
        this->proWhite = prowhite;

        return NO_ERROR;
    }
    
    // 成员方法：getPntRange（读取每个方向上取得点的个数）
    // 读取 pntRange 成员变量的值
    __host__ __device__ unsigned char  // 返回值：当前 pntRange 
                                       // 成员变量的值。
    getPntRange() const
    {
        // 返回 pntRange 成员变量的值。
        return this->pntRange;
    }
    
    // 成员方法：setPntRange（设置每个方向上取得点的个数）
    // 设置 pntRange 成员变量的值
    __host__ __device__ int         // 返回值：函数是否正确执行，若
                                    // 函数正确执行，返回 NO_ERROR 
    setPntRange(
            unsigned char pntrange  // 新 pntRange 值
                                    // 根据河边老师发来的串行实现代码，
                                    // pntRange 不超过 100。
    ) {
        // 判断参数是否合法
        if (pntrange > 100)
            return INVALID_DATA;
        
        // 将 pntRange 成员变量赋成新值
        this->pntRange = pntrange;

        return NO_ERROR;
    }
    
    // 成员方法：getPntCount（读取循环次数的上界）
    // 读取 pntCount 成员变量的值
    __host__ __device__ unsigned char  // 返回值：当前 pntCount 
                                       // 成员变量的值。
    getPntCount() const
    {
        // 返回 pntCount 成员变量的值。
        return this->pntCount;
    }
    
    // 成员方法：setPntCount（设置循环次数的上界）
    // 设置 pntCount 成员变量的值
    __host__ __device__ int         // 返回值：函数是否正确执行，若
                                    // 函数正确执行，返回 NO_ERROR 
    setPntCount(
            unsigned char pntcount  // 新 pntCount 值
                                    // 根据河边老师发来的串行实现代码，
                                    // pntCount 不超过 100。            
    ) {
        // 判断参数是否合法
        if (pntcount > 8)
            return INVALID_DATA;
            
        // 将 pntCount 成员变量赋成新值
        this->pntCount = pntcount;

        return NO_ERROR;
    }
    
    // 成员方法：localCluster（局部聚类）
    // 算法的主函数
    // 略去图像的边缘部分，对输入图像的每一个点并行处理，最终得到输出图像
    // 在每一个点的八个方向上各求出 pntRange 个点的像素平均值，利用这些
    // 平均值与当前点的图像值做运算,最终得出点的新像素值.
    __host__ int            // 返回值：函数是否正确执行，若函数正确执行，
                            // 返回 NO_ERROR
    localCluster(  
             Image *inimg,  // 输入图像
             Image *outimg  // 输出图像
    );
};

#endif

