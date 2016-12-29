// SortArray.h
// 创建人：刘宇
//
// 并行排序（SortArray）
// 功能说明：实现并行排序算法，包括：双调排序，Batcher's 奇偶合并排序，
//           以及 shear 排序。其中，双调排序和 Batcher's 奇偶合并排序
//           的数组长度不能超过一个块内的 shared 内存最大限制（一般为 1024）。
//           当数组个数大于 1024 时，可以调用 shear 排序，其最大限制为 
//           1024×1024 的矩阵。
//
// 修订历史：
// 2012年08月18日（刘宇）
//     初始版本。
// 2012年08月31日（刘宇，杨伟光）
//     完善注释规范。
// 2012年09月08日（于玉龙）
//     修改了一些格式不规范的地方。
// 2012年10月25日 （刘宇）
//     修正了 __device__ 方法的定义位置，防止了跨文件访问出现未定义的错误。
// 2012年10月27日 （侯怡婷，刘宇）
//     纠正 shear 排序的实现方式。
// 2012年11月13日（刘宇）
//     在核函数执行后添加 cudaGetLastError 判断语句
// 2012年11月23日（刘宇）
//     添加输入输出参数的空指针判断
// 2013年01月17日（杨伟光）
//     添加模版类操作，重载多种排序的数据类型
// 2013年04月12日（杨伟光）
//     排序添加了 char 和 double 两种数据类型
// 2013年04月14日（杨伟光，刘宇）
//     修改了一些格式不规范的地方
// 2013年04月17日（杨伟光）
//     修改了代码中隐含的几处错误

#ifndef __SORTARRAY_H__
#define __SORTARRAY_H__

#include "Image.h"
#include "ErrorCode.h"

// 宏：SORT_ARRAY_TYPE_ASC
// 排序标识，升序排序。
#define SORT_ARRAY_TYPE_ASC  2

// 宏：SORT_ARRAY_TYPE_DESC
// 排序标识，降序排序。
#define SORT_ARRAY_TYPE_DESC  1

// 类：SortArray（排序类）
// 继承自：无
// 实现并行排序算法，包括：双调排序，Batcher's 奇偶合并排序，以及 shear 排序。
// 注意输入的数组长度必须是 2 的幂次方（排序算法本身要求）。
class SortArray {

protected:

    // 成员变量：length（排序数组长度）
    // 要求必须是 2 的幂次方。
    int length;

    // 成员变量：lensec（排序矩阵的宽度）
    // 默认是一维数组，所以 lensec 等于 1。
    // 要求必须是 2 的幂次方。
    int lensec;

    // 成员变量：sortflag（排序标记）
    // sortflag 等于 1，降序排序；sortflag 等于 2，升序排序。
    int sortflag;
 
    // 成员变量：ishost（判断输入和输出数组位置）
    // 当 ishost 等于 true 时，表示输入输出数组在 Host 端，否则 ishost 等于
    // false，表示在 Device 端。
    bool ishost;

public:
    // 构造函数：SortArray
    // 无参数构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    SortArray()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->length = 0;     // 排序数组长度默认为 0。
        this->lensec = 1;     // 排序矩阵的宽度默认为 1，表示一维数组。
        this->sortflag = 1;   // 排序标识，默认为 1，降序排序。
        this->ishost = true;  // 判断输入和输出数组位置标记值，默认为 true。
    }

    // 构造函数：SortArray
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    SortArray (
            int length,    // 排序数组长度（具体解释见成员变量）
            int lensec,    // 排序矩阵的宽度（具体解释见成员变量）
            int sortflag,  // 排序标记（具体解释见成员变量）
            bool ishost    // 输入和输出数组位置标记（具体解释见成员变量）
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非
        // 法的初始值而使系统进入一个未知的状态。
        this->length = 0;     // 排序数组长度默认为 0。
        this->lensec = 1;     // 排序矩阵的宽度默认为 1，表示一维数组。
        this->sortflag = 1;   // 排序标识，默认为 1，降序排序。
        this->ishost = true;  // 判断输入和输出数组位置标记值，默认为 true。

        // 根据参数列表中的值设定成员变量的初值
        setLength(length);
        setLensec(lensec);
        setSortflag(sortflag);
        setIshost(ishost);
    }

    // 成员方法：getLength（读取排序数组长度）
    // 读取 lentgh 成员变量。
    __host__ __device__ int  // 返回值：当前 length 成员变量的值。
    getLength() const
    {
        // 返回 length 成员变量的值。
        return this->length;
    }

    // 成员方法：setLength（设置排序数组长度）
    // 设置 lentgh 成员变量。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setLength(               
            int length       // 指定的排序数组长度 length。
    ) {
        // 如果长度不是 2 的幂次方，则表示非法数据。
        if (length & (length - 1) != 0)
            return INVALID_DATA;

        // 将 length 成员变量赋成新值
        this->length = length;

        return NO_ERROR;
    }

    // 成员方法：getLensec（读取排序矩阵的宽度）
    // 读取 lensec 成员变量。
    __host__ __device__ int        // 返回值：当前 lensec 成员变量的值。
    getLensec() const
    {
        // 返回 lensec 成员变量的值。
        return this->lensec;
    }

    // 成员方法：setLensec（设置排序矩阵的宽度）
    // 设置 lensec 成员变量。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setLensec(
            int lensec       // 指定的排序矩阵的宽度 lensec。
    ) {
        // 如果行数小于 0， 则表示非法数据。
        if (lensec < 0)
            return INVALID_DATA;

        // 将 lensec 成员变量赋成新值
        this->lensec = lensec;

        return NO_ERROR;
    }

    // 成员方法：getSortflag（读取排序标记）
    // 读取 sortflag 成员变量。
    __host__ __device__ int  // 返回值：当前 sortflag 成员变量的值。
    getSortflag() const
    {
        // 返回 sortflag 成员变量的值。
        return this->sortflag;
    }

    // 成员方法：setSortflag（设置排序标记）
    // 设置 sortflag 成员变量。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setSortflag(        
            int sortflag     // 指定的排序标记。
    ) {
        // 排序标识只能为 1 和 2。
        if (sortflag != SORT_ARRAY_TYPE_DESC && 
            sortflag != SORT_ARRAY_TYPE_ASC)
            return INVALID_DATA;

        // 将 sortflag 成员变量赋成新值
        this->sortflag = sortflag;

        return NO_ERROR;
    }

    // 成员方法：getIshost（读取判断输入和输出数组位置标记值）
    // 读取 ishost 成员变量。
    __host__ __device__ bool  // 返回值：当前 ishost 成员变量的值。
    getIshost() const
    {
        // 返回 ishost 成员变量的值。
        return this->ishost;
    }

    // 成员方法：setIshost（设置判断输入和输出数组位置标记值）
    // 设置 ishost 成员变量。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setIshost(        
            bool ishost      // 指定的判断输入和输出数组位置标记值。
    ) {
        // 将 ishost 成员变量赋成新值
        this->ishost = ishost;

        return NO_ERROR;
    }

    // Host 成员方法：bitonicSort（并行双调排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 int 整型。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    bitonicSort(
            int *inarray,  // 输入数组。
            int *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：bitonicSort（并行双调排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 float 浮点型。
    __host__ int             // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    bitonicSort(
            float *inarray,  // 输入数组。
            float *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：bitonicSort（并行双调排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 unsigned char 类型。
    __host__ int                     // 返回值：函数是否正确执行，若函数正确
                                     // 执行，返回 NO_ERROR。
    bitonicSort(
            unsigned char *inarray,  // 输入数组。
            unsigned char *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：bitonicSort（并行双调排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 char 类型。
    __host__ int            // 返回值：函数是否正确执行，若函数正确
                            // 执行，返回 NO_ERROR。
    bitonicSort(
            char *inarray,  // 输入数组。
            char *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：bitonicSort（并行双调排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 double 类型。
    __host__ int              // 返回值：函数是否正确执行，若函数正确
                              // 执行，返回 NO_ERROR。
    bitonicSort(
            double *inarray,  // 输入数组。
            double *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：oddEvenMergeSort（并行Batcher's 奇偶合并排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 int 整型。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    oddEvenMergeSort(
            int *inarray,  // 输入数组。
            int *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：oddEvenMergeSort（并行Batcher's 奇偶合并排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 float 浮点型。
    __host__ int             // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    oddEvenMergeSort(
            float *inarray,  // 输入数组。
            float *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：oddEvenMergeSort（并行Batcher's 奇偶合并排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 unsigned char 类型。
    __host__ int                     // 返回值：函数是否正确执行，若函数正确
                                     // 执行，返回 NO_ERROR。
    oddEvenMergeSort(
            unsigned char *inarray,  // 输入数组。
            unsigned char *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：oddEvenMergeSort（并行Batcher's 奇偶合并排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 char 类型。
    __host__ int            // 返回值：函数是否正确执行，若函数正确
                            // 执行，返回 NO_ERROR。
    oddEvenMergeSort(
            char *inarray,  // 输入数组。
            char *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：oddEvenMergeSort（并行Batcher's 奇偶合并排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。
    // 排序后的数组输出到 outarray 中。数据类型是 double 类型。
    __host__ int              // 返回值：函数是否正确执行，若函数正确
                              // 执行，返回 NO_ERROR。
    oddEvenMergeSort(
            double *inarray,  // 输入数组。
            double *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：shearSort（并行 shear 排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。shear 先进行
    // 列排序，在进行行排序，其中内部调用并行双调排序。排序矩阵最大为
    // 1024×1024 排序后的数组输出到 outarray 中。数据类型是 int 整型。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    shearSort(
            int *inarray,  // 输入数组。
            int *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：shearSort（并行 shear 排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。shear 先进行
    // 列排序，在进行行排序，其中内部调用并行双调排序。排序矩阵最大为
    // 1024×1024 排序后的数组输出到 outarray 中。数据类型是 float 浮点型。
    __host__ int             // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    shearSort(
            float *inarray,  // 输入数组。
            float *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：shearSort（并行 shear 排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。shear 先进行
    // 列排序，在进行行排序，其中内部调用并行双调排序。排序矩阵最大为
    // 1024×1024 排序后的数组输出到 outarray 中。数据类型是 unsigned char 类型。
    __host__ int                     // 返回值：函数是否正确执行，若函数正确
                                     // 执行，返回 NO_ERROR。
    shearSort(
            unsigned char *inarray,  // 输入数组。
            unsigned char *outarray  // 排序后的输出数组。
    ); 

    // Host 成员方法：shearSort（并行 shear 排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。shear 先进行
    // 列排序，在进行行排序，其中内部调用并行双调排序。排序矩阵最大为
    // 1024×1024 排序后的数组输出到 outarray 中。数据类型是 char 类型。
    __host__ int            // 返回值：函数是否正确执行，若函数正确
                            // 执行，返回 NO_ERROR。
    shearSort(
            char *inarray,  // 输入数组。
            char *outarray  // 排序后的输出数组。
    );

    // Host 成员方法：shearSort（并行 shear 排序）
    // 输入 inarray 的长度必须是 2 的幂次方（这是由算法本身决定）。shear 先进行
    // 列排序，在进行行排序，其中内部调用并行双调排序。排序矩阵最大为
    // 1024×1024 排序后的数组输出到 outarray 中。数据类型是 double 类型。
    __host__ int              // 返回值：函数是否正确执行，若函数正确
                              // 执行，返回 NO_ERROR。
    shearSort(
            double *inarray,  // 输入数组。
            double *outarray  // 排序后的输出数组。
    );   
};

#endif

