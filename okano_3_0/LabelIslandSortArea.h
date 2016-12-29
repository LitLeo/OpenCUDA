// LabelIslandSortArea.h 
// 创建人：刘宇
//
// 区域排序（LabelIslandSortArea）
// 功能说明：对标记后的所有区域按照面积进行排序。
//
// 修订历史：
// 2012年8月17日 （刘宇）
//     初始版本。
// 2012年8月29日 （刘宇）
//     完善注释规范。
// 2012年9月5日 （刘宇）
//     添加 ishost 成员变量。
// 2012年10月25日 （刘宇）
//     修正了 __device__ 方法的定义位置，防止了跨文件访问出现未定义的错误。
// 2012年11月13日（刘宇）
//     在核函数执行后添加 cudaGetLastError 判断语句
// 2012年11月23日（刘宇）
//     添加输入输出参数的空指针判断

#ifndef __LABELISLANDSORTAREA_H__
#define __LABELISLANDSORTAREA_H__

#include "Image.h"
#include "ErrorCode.h"


// 类：LabelIslandSortArea（区域排序类）
// 继承自：无
// 该类包括区域排序的基本操作。输入的图像是经过区域分割后的标记图像；
// 利用直方图方法计算每类标记的数量，即区域的面积。之后，调用并行双调
// 排序算法对各个区域进行排序，然后按照（面积-标记）的键值对的形式进行
// 输出。
class LabelIslandSortArea {

protected:

    // 成员变量：length（不同标记的数量）
    // 该变量作为保留变量，调用函数前不需要指定。
    int length;

    // 成员变量：minarea（最小面积）
    int minarea;

    // 成员变量：maxarea（最大面积）
    int maxarea;

    // 成员变量：sortflag（排序标记）
    // sortflag 等于 1，降序排序； sortflag 等于 2，升序排序。
    int sortflag;

    // 成员变量：ishost（判断输出数组位置）
    // 当 ishost 等于 1 时，表示输出数组在 Host 端，否则 ishost 等于 0，  
    // 表示在 Device 端。
    int ishost;

public:

    // 构造函数：LabelIslandSortArea
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    LabelIslandSortArea()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->length = 0;       // 区域标记的数量默认为 0。
        this->minarea = 0;      // 最小面积默认为 0。
        this->maxarea = 10000;  // 最大面积默认为 10000.
        this->sortflag = 1;     // 排序标识，默认为 1，降序排序。
        this->ishost = 1;       // 判断输出数组位置标记值，默认为 1。
    }

    // 构造函数：LabelIslandSortArea
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    LabelIslandSortArea (
            int length,                 // 标记的数量（具体解释见成员变量）
            int minarea, int maxarea,   // 最小最大面积（具体解释见成员变量）
            int sortflag,               // 排序标记（具体解释见成员变量）
            int ishost                  // 输出数组位置（具体解释见成员变量）
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非
        // 法的初始值而使系统进入一个未知的状态。
        this->length = 0;       // 区域标记的数量默认为 0。
        this->minarea = 0;      // 最小面积默认为 0。
        this->maxarea = 10000;  // 最大面积默认为 10000.
        this->sortflag = 1;     // 排序标识，默认为 1，降序排序。
        this->ishost = 1;       // 判断输出数组位置标记值，默认为 1。

        // 根据参数列表中的值设定成员变量的初值
        setLength(length);
        setMinarea(minarea);
        setMaxarea(maxarea);
        setSortflag(sortflag);
        setIshost(ishost);
    }

    // 成员方法：getLength（读取不同标记的个数）
    // 读取 lentgh 成员变量。
    __host__ __device__ int  // 返回值：当前 length 成员变量的值。
    getLength() const
    {
        // 返回 length 成员变量的值。
        return this->length;
    }

    // 成员方法：setLength（设置不同标记的个数）
    // 设置 lentgh 成员变量。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setLength(               
            int length       // 指定的不同标记的个数。
    ) {
        // 将 length 成员变量赋成新值
        this->length = length;

        return NO_ERROR;
    }

    // 成员方法：getminarea（读取指定的区域最小面积值）
    // 读取 minarea 成员变量。
    __host__ __device__ int  // 返回值：当前 minarea 成员变量的值。
    getMinarea() const
    {
        // 返回 minarea 成员变量的值。
        return this->minarea;
    }

    // 成员方法：setminarea（设置指定的区域最小面积值）
    // 设置 minarea 成员变量。。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMinarea(     
            int minarea      // 指定的区域最小面积值。
    ) {
        // 将 minarea 成员变量赋成新值
        this->minarea = minarea;

        return NO_ERROR;
    }

    // 成员方法：getmaxarea（读取指定的区域最大面积值）
    // 读取 maxarea 成员变量。
    __host__ __device__ int  // 返回值：当前 maxarea 成员变量的值。
    getMaxarea() const
    {
        // 返回 maxarea 成员变量的值。
        return this->maxarea;
    }

    // 成员方法：setmaxarea（设置指定的区域最大面积值）
    // 设置 maxarea 成员变量。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMaxarea(              
            int maxarea      // 指定的区域最大面积值。
    ) {
        // 将 minarea 成员变量赋成新值
        this->maxarea = maxarea;

        return NO_ERROR;
    }

    // 成员方法：getsortflag（读取区域排序标记）
    // 读取 sortflag 成员变量。
    __host__ __device__ int  // 返回值：当前 sortflag 成员变量的值。
    getSortflag() const
    {
        // 返回 sortflag 成员变量的值。
        return this->sortflag;
    }

    // 成员方法：setsortflag（设置区域排序标记）
    // 设置 sortflag 成员变量。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setSortflag(            
            int sortflag     // 指定的区域排序标记。
    ) {
        // 将 sortflag 成员变量赋成新值
        this->sortflag = sortflag;

        return NO_ERROR;
    }

    // 成员方法：getishost（读取判断输出数组位置标记值）
    // 读取 ishost 成员变量。
    __host__ __device__ int  // 返回值：当前 ishost 成员变量的值。
    getIshost() const
    {
        // 返回 ishost 成员变量的值。
        return this->ishost;
    }

    // 成员方法：setishost（设置判断输出数组位置标记值）
    // 设置 ishost 成员变量。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setIshost(        
            int ishost       // 指定的判断输入和输出数组位置标记值。
    ) {
        // 将 ishost 成员变量赋成新值
        this->ishost = ishost;

        return NO_ERROR;
    }

    // Host 成员方法：bitonicSortPairs（对区域面积进行排序）
    // 输入 inarray 满足最大最小面积要求的区域面积数组，通过实现并行双调排序
    // 对其进行排序，最后按照（面积值-标记值）键值对的形式进行输出。输入输出
    // 参数都位于 Device 端。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确执
                                    // 行，返回NO_ERROR。
    bitonicSortPairs(
            unsigned int *inarray,  // 面积数组（位于 Device 端）。
            unsigned int *areaRank  // 输出排序后的键值对（位于 Device 端）。
    );

    // Host 成员方法：labelIslandSortArea（对标记后的所有区域按照面积进行排序）
    // 输入图像是经过区域分割后的标记图像，根据直方图计算每个标记的数量，即
    // 不同区域的面积大小。之后实现并行双调排序算法对所有面积进行排序，筛选
    // 出面积在最大最小范围之间的面积区域，按照（面积值-标记值）的键值对的形式
    // 保存到输出数组 areaRank 中。当 ishost 等于 1 时，表示areaRank在 Host 端，
    // 否则 ishost 等于 0，表示areaRank在 Device 端。
    __host__ int                    // 返回值：函数是否正确执行，若函数正确执
                                    // 行，返回NO_ERROR。
    labelIslandSortArea(
            Image *inimg,           // 输入图像。
            unsigned int *areaRank  // 输出（面积-标记）键值对数组。
    );

};
#endif
