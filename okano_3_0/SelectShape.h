// SelectShape.h
// 创建人：刘宇
// 
// 形状选择（SelectShape）
// 功能说明：根据输入参数不同选择满足条件的形状区域。该算法一共包括三种不同的
//           实现：（1）根据根据特征数组的下标索引；（2）根据形状特征值进行查
//           找；（3）根据特征值的最大到最小的范围进行查找。
//
// 修订历史：
// 2012年08月10日 （刘宇）
//     初始版本。
// 2012年08月28日 （刘宇）
//     完善注释规范。
// 2012年09月06日 （刘宇）
//     修改 ROI 处理。
// 2012年10月25日 （刘宇）
//     修正 __device__ 方法的定义位置和 blocksize 和 gridsize 中一处潜在的错误
// 2012年11月13日（刘宇）
//     在核函数执行后添加 cudaGetLastError 判断语句
// 2012年11月23日（刘宇）
//     添加输入输出参数的空指针判断

#ifndef __SELECTSHAPE_H__
#define __SELECTSHAPE_H__

#include "Image.h"
#include "ErrorCode.h"


// 类：SelectShape（选择形状算法类）
// 继承自：无
// 该类包含了对图像各个区域的形状进行选取的操作。包括根据特征数组的下标索引；
// 包括根据形状特征值进行查找；以及包括根据特征值的最大到最小的范围进行查找；
// 一个图像上可以有很多形状区域，区域上的所有点具有相同的标记 label ，区域的
// 特征值有多种，例如面积等。
class SelectShape {

protected:

    // 成员变量：rank（形状特征值数组）
    // 形状特征值数组，以（特征值，标记）的键值对形式存储的形状特征值数组。
    int *rank;              

    // 成员变量：pairsnum（特征数组中键值对的个数）
    // 特征值数组中键值对的个数，即形状区域的个数。
    int pairsnum;           

    // 成员变量：index（下标索引值）
    // 特征值数组中标记 label 的下标索引值。
    int index;              

    // 成员变量：value（查找的特征值大小）
    // 查找特征值数组时的特征值大小。
    int value;     

    // 成员变量：minvalue 和 maxvalue（查找的最小和最大特征值）
    // 查找特征值数组时的特征值最大最小的范围。
    int minvalue, maxvalue;  

public:

    // 构造函数：SelectShape
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__  __device__     
    SelectShape()
    {
        // 使用默认值为类的各个成员变量赋值。
        rank = NULL;   // 形状特征值数组默认为空
        pairsnum = 0;  // 键值对的个数默认为 0
        index = 0;     // 下标索引值默认为 0
        value = 0;     // 查找的特征值大小默认为 0
        minvalue = 0;  // 查找的最小特征值默认为 0
        maxvalue = 0;  // 查找的最大特征值默认为 0
    }

    // 构造函数：SelectShape
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。 
    __host__  __device__  
    SelectShape(
            int *rank,                  // 形状特征值数组
            int pairsnum,               // 特征数组中键值对的个数
            int index,                  // 下标索引值
            int value,                  // 查找的特征值大小
            int minvalue, int maxvalue  // 查找的最小和最大特征值
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法的初
        // 始值而使系统进入一个未知的状态。
        this->rank = NULL;   // 形状特征值数组默认为空
        this->pairsnum = 0;  // 键值对的个数默认为 0
        this->index = 0;     // 下标索引值默认为 0
        this->value = 0;     // 查找的特征值大小默认为 0
        this->minvalue = 0;  // 查找的最小特征值默认为 0
        this->maxvalue = 0;  // 查找的最大特征值默认为 0

        // 根据参数列表中的值设定成员变量的初值
        setRank(rank);
        setPairsnum(pairsnum);
        setIndex(index);
        setValue(value);
        setMinvalue(minvalue);
        setMaxvalue(maxvalue);
    }

    // 成员方法：getRank（读取特征值数组）
    // 读取 rank 成员变量的值。
    __host__ __device__ int *  // 返回值：当前 rank 成员变量的值。
    getRank() const 
    {
        // 返回 rank 成员变量的值。
        return this->rank;
    }

    // 成员方法：setRank（设置特征值数组）
    // 设置 rank 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setRank(                
            int *rank        // 指定的特征值数组。
    ) {
        // 检查输入参数是否合法
        if (rank == NULL)
            return INVALID_DATA;

        // 将 rank 成员变量赋成新值
        this->rank = rank;

        return NO_ERROR;
    }
                                                                                               
    // 成员方法：getpairsnum（读取特征值数组中键值对的个数）
    // 读取 pairsnum 成员变量的值。
    __host__ __device__ int  // 返回值：当前 pairsnum 成员变量的值。
    getPairsnum() const
    {
        // 返回 parisNum 成员变量的值。
        return this->pairsnum;
    }

    // 成员方法：setpairsnum（设置特征值数组中键值对的个数）
    // 设置 pairsnum 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setPairsnum(            
            int pairsnum     // 指定的特征值数组中键值对的个数。
    ) {
        // 检查输入参数是否合法
        if (pairsnum == 0)
            return INVALID_DATA;

        // 将 pairsnum 成员变量赋成新值
        this->pairsnum = pairsnum;

        return NO_ERROR;
    }

    // 成员方法：getIndex（读取特征值数组中下标索引值）
    // 读取 index 成员变量的值。
    __host__ __device__ int  // 返回值：当前 index 成员变量的值。
    getIndex() const
    {
        // 返回 index 成员变量的值。
        return this->index;
    }

    // 成员方法：setIndex（设置特征值数组中下标索引值）
    // 设置 index 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setIndex(               
            int index        // 指定的特征值数组中下标索引值。
    ) {
        // 将 index 成员变量赋成新值
        this->index = index;

        return NO_ERROR;
    }

    // 成员方法：getValue（读取查找特征值数组时的特征值大小）
    // 读取 value 成员变量的值。
    __host__ __device__ int       // 返回值：当前 value 成员变量的值。
    getValue() const
    {
        // 返回 value 成员变量的值。
        return this->value;
    }

    // 成员方法：setValue（设置查找特征值数组时的特征值大小）
    // 设置 value 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setValue(               
            int value        // 指定的查找特征值数组时的特征值大小。
    ) {
        // 将 value 成员变量赋成新值
        this->value = value;

        return NO_ERROR;
    }

    // 成员方法：getminvalue（读取查找特征值数组时的最小特征值）
    // 读取 minvalue 成员变量的值。
    __host__ __device__ int  // 返回值：当前 minvalue 成员变量的值。
    getMinvalue() const
    {
        // 返回 minvalue 成员变量的值。
        return this->minvalue;
    }

    // 成员方法：setminvalue（设置查找特征值数组时的最小特征值）
    // 设置 minvalue 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMinvalue(            
            int minvalue     // 指定的查找特征值数组时的最小特征值。
    ) {
        // 将 minvalue 成员变量赋成新值
        this->minvalue = minvalue;

        return NO_ERROR;
    }

    // 成员方法：getmaxvalue（读取查找特征值数组时的最大特征值）
    // 读取 maxvalue 成员变量的值。
    __host__ __device__ int  // 返回值：当前 maxvalue 成员变量的值。
    getMaxvalue() const
    {
        // 返回 maxvalue 成员变量的值。
        return this->maxvalue;
    }

    // 成员方法：setmaxvalue（设置查找特征值数组时的最大特征值）
    // 设置 maxvalue 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMaxvalue(             
            int maxvalue     // 指定的查找特征值数组时的最大特征值。
    ) {
        // 将 maxvalue 成员变量赋成新值
        this->maxvalue = maxvalue;

        return NO_ERROR;
    }

    // Host 成员方法：selectShapeByIndex（根据标记索引形状）
    // 根据参数 index ,索引特征值数组 rank 中的标记值，将满足条件的形状区域拷贝
    // 到输出图像中。如果输出图像 outimg 为空或者等于输入图像 inimg，则将输出结
    // 果覆盖到输入图像 inimg上。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    selectShapeByIndex(
            Image *inimg,  // 输入图像。
            Image *outimg  // 输出图像。
    );

    // Host 成员方法：selectShapeByValue（根据特征值查找形状）
    // 根据参数 value ,查找特征值数组 rank 中的特征值，将满足条件的形状区域拷贝
    // 到输出图像中，可能有多个区域。如果输出图像 outimg 为空或者等于输入图像 
    // inimg，则将输出结果覆盖到输入图像 inimg上。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    selectShapeByValue(
            Image *inimg,  // 输入图像。
            Image *outimg  // 输出图像。
    );

    // Host 成员方法：selectShapeByMinMax（根据特征值最大最小范围查找形状）
    // 根据参数 minvalue 和 maxvalue ,查找特征值数组 rank 中的特征值，将满足
    // 条件的形状区域拷贝到输出图像中，可能有多个区域。如果输出图像 outimg 为空 
    // 或者等于输入图像inimg，则将输出结果覆盖到输入图像 inimg上。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    selectShapeByMinMax(
            Image *inimg,  // 输入图像。
            Image *outimg  // 输出图像。
    );
};


#endif
