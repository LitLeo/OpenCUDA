// TemplateFactory.cu
// 模板工厂产生各种形状的模板

#include "TemplateFactory.h"

#include <iostream>
#include <ctime>
#include <cstring>
using namespace std;

#include "ErrorCode.h"

// 宏：TF_ENABLE_KICK（开启替换规则）
// 开关宏，使能该宏，则在模板池存放满了模板以后，新的模板会替换不常用的模板，但
// 这种做法可能会拉慢单次调用的性能；如果关闭该宏，则单次处理性能会提高，但是从
// 总体来看，后续的模板频繁的申请与释放会带来内存处理的压力，总体性能可能会下
// 降。将以在模板种类使用较少时关闭该宏；如果使用模板种类较多，建议使能该宏。
#define TF_ENABLE_KICK

// 结构体：TFTemplateVendor（生产模板的机器）
// 该结构体给出了模板制造的机制，提供了产生新模板、判断参数合法性等方法的接口，
// 各种模板都实现一个这样的结构提的实例，并实现这其中所有的函数指针。并将这个结
// 构体实例添加到模板机器池中，这样通过外接的接口就可以调用到创造模板的这些函数
// 接口的实现方法。
typedef struct TFTemplateVendor_st {
    bool (*isLegalParam)(dim3 size,               // 不同的模板对于参数有着不同
                         void *privated);         // 的要求，该函数接口用于判断
                                                  // 给定的参数对应当下的模板类
                                                  // 型是否是合法的。例如，对于
                                                  // 圆形模板我们要求直径要大于
                                                  // 3 且为奇数。该函数返回真时
                                                  // 说明参数是合法的。
    int (*getHashIndex)(dim3 size,                // 由于不同类型的模板的参数结
                        void *privated);          // 构不尽相同，因此其 Hash 算
                                                  // 法也会不同。该接口用于返回
                                                  // 给定参数的 Hash 值，如果给
                                                  // 定的参数不合法，或运算过程
                                                  // 中发生错误，则返回错误码。
                                                  // 返回的 Hash 值范围为 0 至
                                                  // TF_VOL_SHAPE - 1 的整型
                                                  // 数。
    Template *(*createTemplate)(dim3 size,        // 该函数接口根据给定的参数返
                                void *privated);  // 回一个该尺寸的 Template。
                                                  // 如果给定的参数不合法，或者
                                                  // 计算过程中出现错误，则返回
                                                  // NULL。
    bool (*isEqualSize)(dim3 size1,               // 该函数用于检查给定的两组参
                        void *privated1,          // 数是否相等。由于不同的模板
                        dim3 size2,               // 对参数的要求不同，因此其各
                        void *privated2);         // 自都有着不同的判断方法，因
                                                  // 此，我们将这个问题抛给具体
                                                  // 模板，如果两个参数中有一个
                                                  // 不合法则恒返回假。
    void *(*copyPrivated)(void *privated);        // 由于专属参数 privated 满足
                                                  // 普适性，只能以指针的形式给
                                                  // 出，但如果在模板池中以不稳
                                                  // 定的指针形式存储会给系统带
                                                  // 来风险，因此此函数接口用来
                                                  // 拷贝出一个仅在模板池内部使
                                                  // 用的 privated，将这个参数
                                                  // 存入模板池，是的系统的稳定
                                                  // 性得到保证。
    bool (*deletePrivated)(void *privated);       // 释放 privated 的内存。由于
                                                  // 各种不同种类的模板具有不同
                                                  // 的专属参数，因此需要不同的
                                                  // 的释放函数，这个用于从模板
                                                  // 池中踢出模板时使用。
} TFTemplateVendor;


// 定义不同的模板（CLASS 的实现在文件末尾）：

// 标准模板生成器示例代码和一些辅助的函数：

// Host 函数：_stretchDigit（抻拉二进制数）
// 抻拉一个二进制数，再各个二进制位之间补充若干个 0。该函数作为混悬数据的一个基
// 础操作存在。二进制数被抻拉后，只保留能存储的低位数据，高位数据被舍去。
static __host__ unsigned int  // 数字被抻拉后的结果
_stretchDigit(
        unsigned int num,     // 原始数据
        int extent            // 抻拉的程度，即在各位间添加的 0 的个数
);

// Host 函数：_combineDigit（混悬一个三维向量）
// 混悬一个三维向量。所谓混悬，即将三个数据从低位到高位排列，混悬后得到的结果的
// 第 0 至 2 位为输入参数中 x、y、z 分量的第 0 位，结果的第 3 至 5 位位输入参数
// 中 x、y、z 分量的第 1 位，以此类推。这里可以选择混悬的数量，可选值包括 1、
// 2、3，分别表示混悬 x 分量（即不混悬），混悬 x、y 分量，以及混悬 x、y、z 分
// 量。
static __host__ unsigned int  // 混悬的结果
_combineDigit(
        dim3 num,             // 输入的数据，三维向量
        int count             // 需要混悬的数量，可选值为 1、2、3，对于小于 1 
                              // 的数，函数直接返回 num 中的 x 分量，对于大于 3
                              // 的数，则按照 3 处理。
);

// Host 函数：_defIsLegalParam（标准参数判断函数）
// 给出一种一般情况下判断参数是否为合法的函数，对于没有特殊要求的模板类型，可以
// 直接使用该函数而不需要自己再重新写一个函数。
static __host__ bool    // 返回值：由于没有给定具体的模板类型，该函数会恒返回真
_defIsLegalParam(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_defGetHashIndex（标准 Hash 算法函数）
// 该函数根据 size 的三维数据进行混合，然后通过取模运算得到合理的 Hash 值，该函
// 数并不将 private 的值考虑进 Hash 值的计算过程中。
static __host__ int     // 返回值：Hash 值，如果出现错误则该函数返回负数。
_defGetHashIndex(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_defCreateTemplate（标准模板生成函数）
// 该函数只是用来作为演示这一类函数的书写格式，并不会真正的返回一个模板，该函数
// 只会返回 NULL。
static __host__ Template *  // 返回值：生成的模板，但该函数只会返回 NULL。
_defCreateTemplate(
        dim3 size,          // 尺寸参数
        void *privated      // 专属参数
);

// Host 函数：_defIsEqualSize（标准的尺寸相等判断函数）
// 该函数是标准的判断两个尺寸是否相等的函数。该函数通过比较两个尺寸参数的各个维
// 度上是否相等，以及两个专属参数是否地址相同来判断两个尺寸是否相同，这是一种最
// 通常的判断方式。
static __host__ bool      // 返回值：给定的两个尺寸是否是一样的。
_defIsEqualSize(
        dim3 size1,       // 第一个尺寸的尺寸参数
        void *privated1,  // 第一个尺寸的专属参数
        dim3 size2,       // 第二个尺寸的尺寸参数
        void *privated2   // 第二个尺寸的专属参数
);

// Host 函数：_defCopyPrivated（标准的专属参数拷贝函数）
// 该函数只是用来作为演示这一类函数的书写格式，并不会真正的进行拷贝工作，返回的
// 指针恒为 NULL。
static __host__ void *  // 返回值：拷贝后和输入参数内容完全一致的新的地址空间的
                        // 指针，但本函数只会返回 NULL。
_defCopyPrivated(
        void *privated  // 待拷贝的专属参数。
);

// Host 函数：_defDeletePrivated（标准的专属参数释放函数）
// 该函数只是用来作为演示这一类函数的书写格式。对于 NULL 参数，该函数不进行任何
// 操作，对于非 NULL 参数，该函数也不会进行任何处理，因为使用 delete 释放
// void * 型指针会产生 Warning。
static __host__ bool    // 返回值：是否释放成功，该函数如果参数是 NULL 则返回
                        // false。
_defDeletePrivated(
        void *privated  // 待释放的专属函数
);

// Host 函数：_stretchDigit（抻拉二进制数）
static __host__ unsigned int _stretchDigit(unsigned int num, int extent)
{
    // 由于需要在结果中舍去高位数据，因此这里需要计算在给定的 extent 的情况下，
    // 究竟旳多少位为有效位。
    int maxbitlim = 8 * sizeof (unsigned int) / (extent + 1);

    // BIT 游标，用来掩出指定位的 BIT 数据。
    unsigned int bitvernier = 0x1;
    // 存放结果的数据。
    unsigned int resnum = 0x0;

    // 从低位开始循环，依次计算每一位的情况，然后逐位赋值给结果。
    for (int i = 0; i < maxbitlim; i++) {
        // 通过 BIT 游标掩出当前位的 BIT 数据，然后左移，使得在结果中它和相邻位
        // 之间出现了 extent 个 0。
        resnum |= ((num & bitvernier) << (extent * i));

        // 左移一位游标，以便下次循环时计算的是更高一位的数据。
        bitvernier <<= 1;
    }
    // 计算完毕，返回结果数据。
    return resnum;
}

// Host 函数：_combineDigit（混悬一个三维向量）
static __host__ unsigned int _combineDigit(dim3 num, int count)
{
    // 存放输出结果的累加变量。
    unsigned int res = 0u;

    // 如果 count <= 1，则不需要任何处理，直接将 num.x 返回。
    if (count <= 1)
        return num.x;

    // 如果 count >= 3，则将其归一化到 3，并先行处理 num.z（抻拉后通过移位放入
    // 相应的位置）
    if (count >= 3) {
        count = 3;
        res = (_stretchDigit(num.z, count) << 2);
    }

    // 此时可以确定 count >= 2，因此这里对 num.x 和 num.y 进行抻拉，并组合到一
    // 起。
    res |= _stretchDigit(num.x, count);
    res |= (_stretchDigit(num.y, count) << 1);

    // 计算完毕，将结果返回。
    return res;
}

// Host 全局变量：_defTemplateVendor（标准模板生成器）
// 归纳定义标准模板生成所需要的函数。
static TFTemplateVendor _defTemplateVendor = {
    _defIsLegalParam,
    _defGetHashIndex,
    _defCreateTemplate,
    _defIsEqualSize,
    _defCopyPrivated,
    _defDeletePrivated
};

// Host 函数：_defIsLegalParam（标准参数判断函数）
static __host__ bool _defIsLegalParam(dim3/* size*/, void * /*privated*/)
{
    // 该函数直接返回
    return true;
}

// Host 函数：_defGetHashIndex（标准 Hash 算法函数）
static __host__ int _defGetHashIndex(dim3 size, void * /*privated*/)
{
    // 直接将三维的 size 混悬后的数据返回。
    return _combineDigit(size, 3) % TF_VOL_SHAPE;
}

// Host 函数：_defCreateTemplate（标准模板生成函数）
static __host__ Template *_defCreateTemplate(dim3/* size*/,
                                             void * /*privated*/)
{
    // 该函数只是用来作为演示这一类函数的书写格式，并不会真正的返回一个模板，该
    // 函数只会返回 NULL。
    return NULL;
}

// Host 函数：_defIsEqualSize（标准的尺寸相等判断函数）
static __host__ bool _defIsEqualSize(dim3 size1, void *privated1,
                                     dim3 size2, void *privated2)
{
    // 本函数采用了一种最为通用的尺寸判断方式，即尺寸参数中各个维度要想等，并且
    // 专属参数的地址值要相等，才能酸味两个尺寸的相等。在实际中具体对于某个类型
    // 的模板来说，这个条件可能会进行一定程度的放宽。
    if (size1.x == size2.x && size1.y == size2.y && size1.z == size2.z &&
        privated1 == privated2)
        return true;
    else
        return false;
}

// Host 函数：_defCopyPrivated（标准的专属参数拷贝函数）
static __host__ void *_defCopyPrivated(void * /*privated*/)
{
    // 本演示函数只会返回 NULL。
    return NULL;
}

// Host 函数：_defDeletePrivated（标准的专属参数释放函数）
static __host__ bool _defDeletePrivated(void *privated)
{
    // 如果输入的参数是 NULL，则直接返回。
    if (privated == NULL)
        return false;

    // 使用 delete 关键字释放 privated，然后返回。
    //delete privated;
    return true;
}


// 方形模板的定义：

// Host 函数：_boxIsLegalParam（方形模板参数判断函数）
// 检查方形模板的参数是否合格，合格的模板要求尺寸参数的 z 分量为 1，专属参数为
// NULL（因为方形模板没有专属参数）
static __host__ bool    // 返回值：如果模板合法，则返回 true，否则返回 false
_boxIsLegalParam(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_boxGetHashIndex（方形模板 Hash 算法函数）
// 方形模板的 Hash 算法。该算法混悬尺寸参数的 x 和 y 分量，由于方形模板没有专属
// 参数，所以在计算 Hash 的时候没有考虑专属参数。
static __host__ int     // 返回值：Hash 值，如果出现错误则该函数返回负数。
_boxGetHashIndex(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_boxCreateTemplate（方形模板生成函数）
// 生成方形模板的函数。
static __host__ Template *  // 返回值：生成的模板，若无法生成模板会返回 NULL。
_boxCreateTemplate(
        dim3 size,          // 尺寸参数
        void *privated      // 专属参数
);

// Host 函数：_boxIsEqualSize（方形模板的尺寸相等判断函数）
// 方形模板使用了尺寸中的两个维度，因此该函数只会检查尺寸参数的 x 和 y 两个维度
// 是否相等。
static __host__ bool      // 返回值：给定的两个尺寸是否是一样的。
_boxIsEqualSize(
        dim3 size1,       // 第一个尺寸的尺寸参数
        void *privated1,  // 第一个尺寸的专属参数
        dim3 size2,       // 第二个尺寸的尺寸参数
        void *privated2   // 第二个尺寸的专属参数
);

// Host 函数：_boxCopyPrivated（方形模板的专属参数拷贝函数）
// 由于方形模板没有专属参数，并不会真正的进行拷贝工作，会直接返回 NULL。
static __host__ void *  // 返回值：直接返回 NULL。
_boxCopyPrivated(
        void *privated  // 待拷贝的专属参数。
);

// Host 函数：_boxDeletePrivated（方形模板的专属参数释放函数）
// 由于方形模板没有专属参数，所以该函数不会释放任何的内存空间。如果给定的
// privated 不是 NULL，则该函数会直接返回 NULL。
static __host__ bool    // 返回值：如果参数为 NULL 返回 true，否则返回 false。
_boxDeletePrivated(
        void *privated  // 待释放的专属函数
);

// Host 全局变量：_boxTemplateVendor（方形模板生成器）
// 归纳定义方形模板生成所需要的函数。
static TFTemplateVendor _boxTemplateVendor = {
    _boxIsLegalParam,
    _boxGetHashIndex,
    _boxCreateTemplate,
    _boxIsEqualSize,
    _boxCopyPrivated,
    _boxDeletePrivated
};

// Host 函数：_boxIsLegalParam（方形模板参数判断函数）
static __host__ bool _boxIsLegalParam(dim3 size, void *privated)
{
    // 如果尺寸参数的 z 分量不等于 1，或者专属变量不为 NULL 则该判定该参数是非
    // 法参数。
    if (size.z != 1 || privated != NULL)
        return false;
    // 如果 x 和 y 分量的尺寸小于 1，该参数也会被判定为非法。
    else if (size.x < 1 || size.y < 1)
        return false;
    else
        return true;
}

// Host 函数：_boxGetHashIndex（方形模板 Hash 算法函数）
static __host__ int _boxGetHashIndex(dim3 size, void * /*privated*/)
{
    // 混悬尺寸参数的 x 和 y 分量，由于方形模板没有专属参数，所以在计算 Hash 的
    // 时候没有考虑专属参数。
    return _combineDigit(size, 2) % TF_VOL_SHAPE;
}

// Host 函数：_boxCreateTemplate（方形模板生成函数）
static __host__ Template *_boxCreateTemplate(dim3 size, void * /*privated*/)
{
    // 定义临时变指针量 boxtpl，用于模板返回值
    Template *boxtpl;

    // 申请一个新的模板
    int errcode;
    errcode = TemplateBasicOp::newTemplate(&boxtpl);
    if (errcode != NO_ERROR)
        return NULL;
    
    // 计算模版中点的数量，并在 Host 上获得存储空间
    int count = size.x * size.y;
    errcode = TemplateBasicOp::makeAtHost(boxtpl, count);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(boxtpl);
        return NULL;
    }
    
    // 将坐标的指针和附加数据的指针取出，然后通过指针游标的方式写入数据
    int *ptsdata = boxtpl->tplData;
    float *attdata = ATTACHED_DATA(boxtpl);

    // 计算方形模板中点集的范围。这里设定方形的中心点为原点。
    int startc = -((size.x - 1) / 2), endc = size.x / 2;
    int startr = -((size.y - 1) / 2), endr = size.y / 2;

    // 为了使所有坐标点的附加数据加和值为 1，这里取坐标点数量的倒数为每个点的附
    // 加数据。
    float attdataconst = 1.0f / count;

    // 迭代赋值模板中的点集坐标数据和附加数据 
    for (int r = startr; r < endr; r++) {
        for (int c = startc; c < endc; c++) {
            *(ptsdata++) = c;
            *(ptsdata++) = r;
            *(attdata++) = attdataconst;
        }
    }

    // 返回方形模板指针
    return boxtpl;
}

// Host 函数：_boxIsEqualSize（方形模板的尺寸相等判断函数）
static __host__ bool _boxIsEqualSize(dim3 size1, void * /*privated1*/,
                                     dim3 size2, void * /*privated2*/)
{
    // 由于方形只有两维的尺寸，因此这里只考虑 x 和 y 分量
    if (size1.x == size2.x && size1.y == size2.y)
        return true;
    else
        return false;
}

// Host 函数：_boxCopyPrivated（方形模板的专属参数拷贝函数）
static __host__ void * _boxCopyPrivated(void * /*privated*/)
{
    // 由于方形模板没有专属参数，因此不进行任何的拷贝工作，直接返回。
    return NULL;
}

// Host 函数：_boxDeletePrivated（方形模板的专属参数释放函数）
static __host__ bool _boxDeletePrivated(void *privated)
{
    // 由于方形模板没有专属参数，因此不进行任何的内存释放，只是象征性的进行一下
    // 判断和返回结果。
    if (privated == NULL)
        return true;
    else
        return false;
}


// 圆形模板的定义：

// Host 函数：_circleIsLegalParam（圆形模板参数判断函数）
// 检查圆形模板的参数是否合格，合格的模板要求尺寸参数的 x 和 y 分量必须相等，且
// 大于等于 3，z 分量为 1，专属参数为 NULL（因为圆形模板没有专属参数）
static __host__ bool    // 返回值：如果模板合法，则返回 true，否则返回 false
_circleIsLegalParam(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_circleGetHashIndex（圆形模板 Hash 算法函数）
// 圆形模板的 Hash 算法。该函数只是将尺寸参数的 x 分量取模。
static __host__ int     // 返回值：Hash 值，如果出现错误则该函数返回负数。
_circleGetHashIndex(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_circleCreateTemplate（圆形模板生成函数）
// 生成圆形模板的函数。
static __host__ Template *  // 返回值：生成的模板，若无法生成模板会返回 NULL。
_circleCreateTemplate(
        dim3 size,          // 尺寸参数
        void *privated      // 专属参数
);

// Host 函数：_circleIsEqualSize（圆形模板的尺寸相等判断函数）
// 圆形模板使用了尺寸中的两个维度，因此该函数只会检查尺寸参数的 x 维度是否相
// 等。
static __host__ bool      // 返回值：给定的两个尺寸是否是一样的。
_circleIsEqualSize(
        dim3 size1,       // 第一个尺寸的尺寸参数
        void *privated1,  // 第一个尺寸的专属参数
        dim3 size2,       // 第二个尺寸的尺寸参数
        void *privated2   // 第二个尺寸的专属参数
);

// Host 函数：_circleCopyPrivated（圆形模板的专属参数拷贝函数）
// 由于圆形模板没有专属参数，并不会真正的进行拷贝工作，会直接返回 NULL。
static __host__ void *  // 返回值：直接返回 NULL。
_circleCopyPrivated(
        void *privated  // 待拷贝的专属参数。
);

// Host 函数：_circleDeletePrivated（圆形模板的专属参数释放函数）
// 由于圆形模板没有专属参数，所以该函数不会释放任何的内存空间。如果给定的
// privated 不是 NULL，则该函数会直接返回 NULL。
static __host__ bool    // 返回值：如果参数为 NULL 返回 true，否则返回 false。
_circleDeletePrivated(
        void *privated  // 待释放的专属函数
);

// Host 全局变量：_circleTemplateVendor（圆形模板生成器）
// 归纳定义圆形模板生成所需要的函数。
static TFTemplateVendor _circleTemplateVendor = {
    _circleIsLegalParam,
    _circleGetHashIndex,
    _circleCreateTemplate,
    _circleIsEqualSize,
    _circleCopyPrivated,
    _circleDeletePrivated
};

// Host 函数：_circleIsLegalParam（圆形模板参数判断函数）
static __host__ bool _circleIsLegalParam(dim3 size, void *privated)
{
    // 如果尺寸参数的 z 分量不等于 1，或者专属变量不为 NULL 则该判定该参数是非
    // 法参数。
    if (size.z != 1 || privated != NULL)
        return false;
    // 如果 x 和 y 分量的尺寸小于 3，或者 x 和 y 分量不想等，该参数也会被判定为
    // 非法。
    else if (size.x < 3 || /*size.y < 3 || */size.x != size.y)
        return false;
    // 如果尺寸参数为偶数，该参数也会被判定为非法。
    else if (size.x % 2 == 0/* || size.y % 2 == 0*/)
        return false;
    else
        return true;
}

// Host 函数：_circleGetHashIndex（圆形模板 Hash 算法函数）
static __host__ int _circleGetHashIndex(dim3 size, void * /*privated*/)
{
    // 这里只是用了 x 分量，由于只可能是 x 大于 2 的奇数，因此这里将其除以 2，
    // 可以更高效的利用空间。
    return ((size.x - 1) / 2) % TF_VOL_SHAPE;
}

// Host 函数：_circleCreateTemplate（圆形模板生成函数）
static __host__ Template *_circleCreateTemplate(dim3 size, void * /*privated*/)
{
    // 得到圆形模板半径
    int radius = (size.x - 1) / 2;
    radius = ((radius < 1) ? 1 : radius);
    int radius2 = radius * radius;
    
    // 声明一些局部变量，用来保存模板的临时值。这些变量包括 tmptpldata，用来保
    // 存临时的坐标点信息，这段内存空间申请大小为所求圆形的外接矩形的大小；
    // count 为游标，记录下一个 tmptpldata 的存储下标，当整个求解完成后，该值存
    // 储信息为整个圆形模板所占用的内存字数；x 和 y 为当前迭代的坐标，其起始位
    // 置为 (0, radius)。
    size_t maxdatasize = 2 * (2 * radius + 1) * (2 * radius + 1);
    int *tmptpldata = new int[maxdatasize];
    int count = 0;
    int x = 0, y = radius;

    // 如果临时数据空间申请失败，则无法进行后续的操作，则只能报错返回。
    if (tmptpldata == NULL)
        return NULL;

    // 整个迭代过程采用经典的八分圆迭代法，即只迭代推导出圆的右上 1/8 部分（依
    // 右手坐标来说），即从 (0, raidus) 至 (sqrt(radius), sqrt(radius)) 段的
    // 点，其余的点都通过圆自身的对称性映射得到。如果迭代得到 (x, y) 为圆上一
    // 点，那么 (-x, y)、(x, -y)、(-x, -y)、(y, x)、(-y, x)、(y, -x)、(-y, -x)
    // 也都将是圆上的点。由于本算法要得到一个实心圆体，所以，每得到一对关于 x
    // 轴的对称点后，则填充之间的所有点。

    // 这是算法的第一步，将 (0, radius) 和 (0, -radius) 计入其中。之所以要特殊
    // 对待是因为没有 0 和 -0 之区别，下段代码的 for 循环也是处理特殊的 0 点。
    tmptpldata[count++] = 0;
    tmptpldata[count++] = radius;
    tmptpldata[count++] = 0;
    tmptpldata[count++] = -radius;

    // 计入 (radius, 0) 和 (-radius, 0) 并填充该行内这两点间的所有点。
    for (int ix = -radius; ix <= radius; ix++) {
        tmptpldata[count++] = ix;
        tmptpldata[count++] = 0;
    }

    // 当 x < y 时，(x, y) 处于圆的右上方 1/8 的部分，我们只计算此 1/8 的部分，
    // 其他的部分通过圆的对称性计算出来。 
    while (x < y) {
        // 计算下一个点。这里对于下一个点只有两种可能，一种是 (x + 1, y)，另一
        // 种是 (x + 1, y - 1)。具体选择这两种中的哪一个，要看它们谁更接近真实
        // 的圆周曲线。这段代码就是计算这两种情况距离圆周曲线的距离平方（开平方
        // 计算太过复杂，却不影响这里的结果，因此我们没有进行开平方计算，而使用
        // 距离的平方值作为判断的条件）。
        x++;
        int d1 = x * x + y * y - radius2;
        int d2 = x * x + (y - 1) * (y - 1) - radius2;
        d1 = ((d1 < 0) ? -d1 : d1);
        d2 = ((d2 < 0) ? -d2 : d2);

        // 比较两个候选点相距圆周曲线的距离
        if (d1 < d2) {
            // 如果点 (x + 1, y) 更加接近圆周曲线，则将首先将 90 度对称点的四个
            // 点写入到坐标集中，这里没有进行内部的填充，是水平方向上的内部点已
            // 经在前些步骤时被填充了
            tmptpldata[count++] = x;
            tmptpldata[count++] = y;
            tmptpldata[count++] = -x;
            tmptpldata[count++] = y;
            tmptpldata[count++] = x;
            tmptpldata[count++] = -y;
            tmptpldata[count++] = -x;
            tmptpldata[count++] = -y;
        } else {
            // 如果点 (x + 1, y - 1) 更加接近圆周曲线，则需要将 (-x - 1, y - 1)
            // 和 (x + 1, y - 1)，以及 (-x - 1, 1 - y) 和 (x + 1, 1 - y) 之间
            // （含）的所有点都添加到坐标集中。
            y--;

            // 由于此前进行了 x++ 操作，所以 y-- 操作会导致 x > y 的情况产生，
            // 显然 x > y 的情况都已经被其他的 45 度对称点所处理，因此，这里需
            // 惊醒该检查，一旦发现 x > y 则直接跳出循环。
            if (x > y)
                break;

            // 将对应的 y > 0 和 y < 0 所在的横向内部坐标点计入到坐标集中。
            for (int ix = -x; ix <= x; ix++) {
                tmptpldata[count++] = ix;
                tmptpldata[count++] = y;
                tmptpldata[count++] = ix;
                tmptpldata[count++] = -y;
            }
        }

        // 处理 45 度的各个对称点的情况，因为每走一步都有 x + 1 的操作，所以处
        // 理 45 度对称点的时候都需要将对应的两点之间的横向内部点进行填充。但这
        // 里有一个例外的情况，就是当 x >= y 时，该行的点已经在其他计算的 90 度
        // 对称点中进行了处理，所有这些时候，就不需要在对 45 度对称点进行处理
        // 了。
        if (x < y) {
            for (int iy = -y; iy <= y; iy++) {
                tmptpldata[count++] = iy;
                tmptpldata[count++] = x;
                tmptpldata[count++] = iy;
                tmptpldata[count++] = -x;
            }
        }
    }

    // 申请一个新的 Template 用来承装这些圆形模板中的坐标点集。
    Template *restpl;
    int errcode;

    // 申请新的模板。
    errcode = TemplateBasicOp::newTemplate(&restpl);
    if (errcode != NO_ERROR) {
        // 如果出现错误需要释放掉申请的临时空间以防止内存泄漏。
        delete[] tmptpldata;
        return NULL;
    }

    // 在 Device 内存上申请合适大小的空间，用来存放坐标数据。
    int ptscnt = count / 2;
    errcode = TemplateBasicOp::makeAtHost(restpl, ptscnt);
    if (errcode != NO_ERROR) {
        // 如果操作失败，需要释放掉之前申请的临时坐标集数据空间和模板数据结构，
        // 以防止内存泄漏。
        TemplateBasicOp::deleteTemplate(restpl);
        delete[] tmptpldata;
        return NULL;
    }

    // 将临时坐标集中的坐标数据拷贝到模板的坐标数据中。
    memcpy(restpl->tplData, tmptpldata, count * sizeof (int));

    // 取出模板的附加数据，然后为附加数据赋值为坐标点个数的倒数。
    float attdataconst = 1.0f / ptscnt;
    float *attdata = ATTACHED_DATA(restpl);

    // 用迭代的方式将数据写入到附加数据中。
    for (int i = 0; i < ptscnt; i++) {
        *(attdata++) = attdataconst;
    }

    // 坐标数据已经拷贝到了需要返回给用户的模板中，因此，这个临时坐标集数据空间
    // 已经不再需要了，因此需要将其释放掉。
    delete[] tmptpldata;

    // 处理完毕，返回已经装配好的模板。
    return restpl;
}

// Host 函数：_circleIsEqualSize（圆形模板的尺寸相等判断函数）
static __host__ bool _circleIsEqualSize(dim3 size1, void * /*privated1*/,
                                        dim3 size2, void * /*privated2*/)
{
    // 由于圆形只有一维的尺寸，因此这里只考虑 x 分量
    if (size1.x == size2.x/* && size1.y == size2.y*/)
        return true;
    else
        return false;
}

// Host 函数：_circleCopyPrivated（圆形模板的专属参数拷贝函数）
static __host__ void * _circleCopyPrivated(void * /*privated*/)
{
    // 由于圆形模板没有专属参数，因此不进行任何的拷贝工作，直接返回。
    return NULL;
}

// Host 函数：_circleDeletePrivated（圆形模板的专属参数释放函数）
static __host__ bool _circleDeletePrivated(void *privated)
{
    // 由于圆形模板没有专属参数，因此不进行任何的内存释放，只是象征性的进行一下
    // 判断和返回结果。
    if (privated == NULL)
        return true;
    else
        return false;
}


// 环形模板的定义：

// Host 函数：_arcIsLegalParam（环形模板参数判断函数）
// 检查环形模板的参数是否合格，合格的模板要求尺寸参数的 z 分量为 1，专属参数为
// NULL（因为环形模板没有专属参数）；此外环形模板要求 x 和 y 分量尺寸必须大于等
// 于 1，且 y 分量应该大于 x 分量（用 y 分量来定义外环，x 分量来定义内环）。
static __host__ bool    // 返回值：如果模板合法，则返回 true，否则返回 false
_arcIsLegalParam(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_arcGetHashIndex（环形模板 Hash 算法函数）
// 环形模板的 Hash 算法。该算法混悬尺寸参数的 x 和 y 分量，由于环形模板没有专属
// 参数，所以在计算 Hash 的时候没有考虑专属参数。
static __host__ int     // 返回值：Hash 值，如果出现错误则该函数返回负数。
_arcGetHashIndex(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_arcCreateTemplate（环形模板生成函数）
// 生成环形模板的函数。
static __host__ Template *  // 返回值：生成的模板，若无法生成模板会返回 NULL。
_arcCreateTemplate(
        dim3 size,          // 尺寸参数
        void *privated      // 专属参数
);

// Host 函数：_arcIsEqualSize（环形模板的尺寸相等判断函数）
// 环形模板使用了尺寸中的两个维度，因此该函数只会检查尺寸参数的 x 和 y 两个维度
// 是否相等。
static __host__ bool      // 返回值：给定的两个尺寸是否是一样的。
_arcIsEqualSize(
        dim3 size1,       // 第一个尺寸的尺寸参数
        void *privated1,  // 第一个尺寸的专属参数
        dim3 size2,       // 第二个尺寸的尺寸参数
        void *privated2   // 第二个尺寸的专属参数
);

// Host 函数：_arcCopyPrivated（环形模板的专属参数拷贝函数）
// 由于环形模板没有专属参数，并不会真正的进行拷贝工作，会直接返回 NULL。
static __host__ void *  // 返回值：直接返回 NULL。
_arcCopyPrivated(
        void *privated  // 待拷贝的专属参数。
);

// Host 函数：_arcDeletePrivated（环形模板的专属参数释放函数）
// 由于环形模板没有专属参数，所以该函数不会释放任何的内存空间。如果给定的
// privated 不是 NULL，则该函数会直接返回 NULL。
static __host__ bool    // 返回值：如果参数为 NULL 返回 true，否则返回 false。
_arcDeletePrivated(
        void *privated  // 待释放的专属函数
);

// Host 全局变量：_arcTemplateVendor（环形模板生成器）
// 归纳定义环形模板生成所需要的函数。
static TFTemplateVendor _arcTemplateVendor = {
    _arcIsLegalParam,
    _arcGetHashIndex,
    _arcCreateTemplate,
    _arcIsEqualSize,
    _arcCopyPrivated,
    _arcDeletePrivated
};

// Host 函数：_arcIsLegalParam（环形模板参数判断函数）
static __host__ bool _arcIsLegalParam(dim3 size, void *privated)
{
    // 如果尺寸参数的 z 分量不等于 1，或者专属变量不为 NULL 则该判定该参数是非
    // 法参数。
    if (size.z != 1 || privated != NULL)
        return false;
    // 如果 x 和 y 分量的尺寸小于 1，或者 x 分量大于 y 分量，该参数也会被判定为
    // 非法。
    else if (size.x < 1 || size.y <= size.x)
        return false;
    // 由于 size 表示直径，因此，这里要求两个同心圆的直径必须皆为奇数。
    else if (size.x % 2 == 0 || size.y % 2 == 0)
        return false;
    else
        return true;
}

// Host 函数：_arcGetHashIndex（环形模板 Hash 算法函数）
static __host__ int _arcGetHashIndex(dim3 size, void * /*privated*/)
{
    // 混悬尺寸参数的 x 和 y 分量，由于方形模板没有专属参数，所以在计算 Hash 的
    // 时候没有考虑专属参数。由于 x 和 y 分量肯定为奇数，为了保证 Hash 算法的满
    // 满映射，这里分别将 x 和 y 尺寸分量右移一位，抛弃其最低位。
    size.x >>= 1;
    size.y >>= 1;
    return _combineDigit(size, 2) % TF_VOL_SHAPE;
}

// Host 函数：_arcCreateTemplate（环形模板生成函数）
static __host__ Template *_arcCreateTemplate(dim3 size, void * /*privated*/)
{
    // 计算得到环形模板内侧圆的半径，以及半径的平方值。
    int sr = (size.x - 1) / 2;
    sr = ((sr < 1) ? 1 : sr);
    int sr2 = sr * sr;

    // 计算得到环形模板外侧圆的半径，以及半径的平方值。
    int lr = (size.y - 1) / 2;
    lr = ((lr <= sr) ? (sr + 1) : lr);
    int lr2 = lr * lr;

    // 申请用于存放坐标点的临时空间，为了防止内存越界访问，我们申请了最大可能的
    // 空间，即外侧圆的外接矩形的尺寸。
    int maxsize = 2 * (2 * lr + 1) * (2 * lr + 1);
    int *tmptpldata = new int[maxsize];

    // 初始化迭代求点所必须的一些局部变量。
    int count = 0;  // 当前下标。随着各个坐标点的逐渐求出，该值不断递增，用来记
                    // 录下一个存储位置的下标值
    int x = 0, sy = sr, ly = lr;  // 算法依 x 为主迭代变量，自增 x 后求的合适的
                                  // y 值，由于存在内外两侧圆形，因此，这里设定
                                  // 两个 y 的变量，sy 表示内侧圆的 y 值，ly 表
                                  // 示外侧圆的 y 值。

    // 整个迭代过程与求解圆形模板的方式相同，采用八分圆方法，通过迭代求解 1 / 8
    // 个圆形，然后通过圆的对称性得到圆的另外部分，在每求解一个坐标后，填充两个
    // 圆之间的部分坐标。
    // 由于内侧圆会比外侧圆更快的达到结束点，因此在达到结束点后，则内测圆取直线
    // x - y = 0 上的点。这样在填充内部点的时候才不会重复处理，将多余的重复点加
    // 入到坐标点集中。
    // _|____    /
    //  |求解\  / <-- 直线 x - y = 0
    //  |区域 \/
    // _|___  /\
    //  |   \/  \
    //  |   /\   \ <-- 外侧圆
    //  |  /  \ <-- 内侧圆
    //  | /    |  |
    // _|/_____|__|__

    // 需要实现处理坐标轴上的点，由于 0 不分正负，所以不能通过下面迭代 while 循
    // 环的通用方法来实现。将两个半径之间的点加入坐标点集。
    for (int y = sr; y < lr; y++) {
        tmptpldata[count++] = 0;
        tmptpldata[count++] = y;
        tmptpldata[count++] = 0;
        tmptpldata[count++] = -y;
        tmptpldata[count++] = y;
        tmptpldata[count++] = 0;
        tmptpldata[count++] = -y;
        tmptpldata[count++] = 0;
    }

    // 迭代直到外侧圆的 1 / 8 区域求解完毕。
    while (x < ly) {
        // 自加 x，然后分别求解内侧圆和外侧圆的 y 坐标。
        x++;

        // 如果内侧圆还没有求解完成，则需要在两个可能的 y 坐标中选择 y 坐标。
        // 注意，由于上面的自加过程，x 已经更新为下一点的 x 坐标，但该判断语句
        // 需要使用原来的 x 坐标，因此，这里使用 x - 1 做为判断变量。
        if (x - 1 < sy) {
            // 从两个可能的下一点坐标 (x + 1, y) 和 (x + 1, y - 1) 中选择一个更
            // 加接近圆形方程的坐标点做为下一点的坐标。
            int sd1 = abs(x * x + sy * sy - sr2);
            int sd2 = abs(x * x + (sy - 1) * (sy - 1) - sr2);
            sy = (sd1 <= sd2) ? sy : (sy - 1);
        }

        // 如果 x >= sy 说明内侧圆已经求解完毕，因此这时应该按照直线 x - y = 0
        // 来计算，这才不会造成重复点。
        if (x >= sy)
            sy = x;

        // 求解外侧圆的下一个点坐标，从两个可能的下一点坐标 (x + 1, y) 和 
        // (x + 1, y - 1) 中选择一个更加接近圆形方程的坐标点做为下一点的坐标。
        int ld1 = abs(x * x + ly * ly - lr2);
        int ld2 = abs(x * x + (ly - 1) * (ly - 1) - lr2);
        ly = (ld1 <= ld2) ? ly : (ly - 1);

        // 如果 x > ly 说明外侧圆已经求解完毕，因此跳出迭代。
        if (x > ly)
            break;

        // 将两个圆（或者外侧圆与直线 x - y = 0）当前点之间的坐标写入到坐标点集
        // 中。考虑到圆的对称性关系，这里将 8 个对称坐标点也同时写入了坐标点
        // 集。
        for (int y = sy; y < ly; y++) {
            tmptpldata[count++] = x;
            tmptpldata[count++] = y;
            tmptpldata[count++] = -x;
            tmptpldata[count++] = y;
            tmptpldata[count++] = x;
            tmptpldata[count++] = -y;
            tmptpldata[count++] = -x;
            tmptpldata[count++] = -y;

            // 如果当前点的 x 和 y 相等，那么其关于直线 x - y = 0 或 x + y = 0
            // 的对称点就是其自身，因此没有必要在次加入到坐标点集中。
            if (x == y)
                continue;
            tmptpldata[count++] = y;
            tmptpldata[count++] = x;
            tmptpldata[count++] = -y;
            tmptpldata[count++] = x;
            tmptpldata[count++] = y;
            tmptpldata[count++] = -x;
            tmptpldata[count++] = -y;
            tmptpldata[count++] = -x;
        }
    }

    // 申请一个新的 Template 用来承装这些圆形模板中的坐标点集。
    Template *restpl;
    int errcode;

    // 申请新的模板。
    errcode = TemplateBasicOp::newTemplate(&restpl);
    if (errcode != NO_ERROR) {
        // 如果出现错误需要释放掉申请的临时空间以防止内存泄漏。
        delete[] tmptpldata;
        return NULL;
    }

    // 在 Device 内存上申请合适大小的空间，用来存放坐标数据。
    int ptscnt = count / 2;
    errcode = TemplateBasicOp::makeAtHost(restpl, ptscnt);
    if (errcode != NO_ERROR) {
        // 如果操作失败，需要释放掉之前申请的临时坐标集数据空间和模板数据结构，
        // 以防止内存泄漏。
        TemplateBasicOp::deleteTemplate(restpl);
        delete[] tmptpldata;
        return NULL;
    }

    // 将临时坐标集中的坐标数据拷贝到模板的坐标数据中。
    memcpy(restpl->tplData, tmptpldata, count * sizeof (int));

    // 取出模板的附加数据，然后为附加数据赋值为坐标点个数的倒数。
    float attdataconst = 1.0f / ptscnt;
    float *attdata = ATTACHED_DATA(restpl);

    // 用迭代的方式将数据写入到附加数据中。
    for (int i = 0; i < ptscnt; i++) {
        *(attdata++) = attdataconst;
    }

    // 坐标数据已经拷贝到了需要返回给用户的模板中，因此，这个临时坐标集数据空间
    // 已经不再需要了，因此需要将其释放掉。
    delete[] tmptpldata;

    // 处理完毕，返回已经装配好的模板。
    return restpl;
}

// Host 函数：_arcIsEqualSize（环形模板的尺寸相等判断函数）
static __host__ bool _arcIsEqualSize(dim3 size1, void * /*privated1*/,
                                     dim3 size2, void * /*privated2*/)
{
    // 由于方形只有两维的尺寸，因此这里只考虑 x 和 y 分量
    if (size1.x == size2.x && size1.y == size2.y)
        return true;
    else
        return false;
}

// Host 函数：_arcCopyPrivated（环形模板的专属参数拷贝函数）
static __host__ void *_arcCopyPrivated(void * /*privated*/)
{
    // 由于环形模板没有专属参数，因此不进行任何的拷贝工作，直接返回。
    return NULL;
}

// Host 函数：_arcDeletePrivated（环形模板的专属参数释放函数）
static __host__ bool _arcDeletePrivated(void *privated)
{
    // 由于环形模板没有专属参数，因此不进行任何的内存释放，只是象征性的进行一下
    // 判断和返回结果。
    if (privated == NULL)
        return true;
    else
        return false;
}

// 高斯模板的定义：

// Host 函数：_gaussIsLegalParam（高斯模板参数判断函数）
// 检查高斯模板的参数是否合格，合格的模板要求尺寸参数的 z 分量为 1，专属参数不
// 能为NULL；此外高斯模板还要求 x 和 y 分量尺寸必须大于等于 1，且 y 分量必须等
// 于 x 分量；另外高斯模板要求尺寸必须为奇数。
static __host__ bool    // 返回值：如果模板合法，则返回 true，否则返回 false
_gaussIsLegalParam(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_gaussGetHashIndex（高斯模板 Hash 算法函数）
// 高斯模板的 Hash 算法。该算法之使用尺寸参数的 x 分量，由于高斯模板使用专属
// 参数，所以在计算 Hash 的时候也考虑专属参数。
static __host__ int     // 返回值：Hash 值，如果出现错误则该函数返回负数。
_gaussGetHashIndex(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_gaussCreateTemplate（高斯模板生成函数）
// 生成高斯模板的函数。
static __host__ Template *  // 返回值：生成的模板，若无法生成模板会返回 NULL。
_gaussCreateTemplate(
        dim3 size,          // 尺寸参数
        void *privated      // 专属参数
);

// Host 函数：_gaussIsEqualSize（高斯模板的尺寸相等判断函数）
// 高斯模板使用了尺寸中的一个维度，因此该函数只会检查尺寸参数的 x 维度是否相
// 等。同时还检查了 privated 在 float 型数据的条件下是否相等（即两个数的差的绝
// 对值小于某一个很小的正数）
static __host__ bool      // 返回值：给定的两个尺寸是否是一样的。
_gaussIsEqualSize(
        dim3 size1,       // 第一个尺寸的尺寸参数
        void *privated1,  // 第一个尺寸的专属参数
        dim3 size2,       // 第二个尺寸的尺寸参数
        void *privated2   // 第二个尺寸的专属参数
);

// Host 函数：_gaussCopyPrivated（高斯模板的专属参数拷贝函数）
// 按照 float 型数据的方式，申请一个新的 float 型数据空间，并将 privated 所指向
// 的 float 型数据拷贝到新的空间中。
static __host__ void *  // 返回值：如果 privated 为 NULL，则返回 NULL；否则返回
                        // 新申请的数据空间。
_gaussCopyPrivated(
        void *privated  // 待拷贝的专属参数。
);

// Host 函数：_gaussDeletePrivated（高斯模板的专属参数释放函数）
// 按照释放一个 float 型数据的地址空间的方法释放给定的空间。如果 privated 是
// NULL 则不进行任何操作。
static __host__ bool    // 返回值：如果释放成功返回 true，对于 NULL 参数，返回
                        // false。
_gaussDeletePrivated(
        void *privated  // 待释放的专属函数
);

// Host 全局变量：_gaussTemplateVendor（高斯模板生成器）
// 归纳定义高斯模板生成所需要的函数。
static TFTemplateVendor _gaussTemplateVendor = {
    _gaussIsLegalParam,
    _gaussGetHashIndex,
    _gaussCreateTemplate,
    _gaussIsEqualSize,
    _gaussCopyPrivated,
    _gaussDeletePrivated
};

// Host 函数：_gaussIsLegalParam（高斯模板参数判断函数）
static __host__ bool _gaussIsLegalParam(dim3 size, void *privated)
{
    // 由于高斯模板使用了 float 型的专属参数，因此需要保证 privated 不为 NULL。
    if (privated == NULL)
        return false;
    // 由于只是用了一个维度的尺寸变量，这里要求 z 维度必须为 1。
    else if (size.z != 1)
        return false;
    // 这里了要求尺寸必须大于等于 3 且 x 和 y 分量必须相等
    else if (size.x < 3 || size.y != size.x)
        return false;
    // 这里还要求尺寸必须为奇数。
    else if (size.x % 2 == 0)
        return false;
    // 如果附属数据的值等于 0，则判断为非法。
    else if (fabs(*((float *)privated)) < 1.0e-8f)
        return false;
    else
        return true;
}

// Host 函数：_gaussGetHashIndex（高斯模板 Hash 算法函数）
static __host__ int _gaussGetHashIndex(dim3 size, void *privated)
{
    // 如果高斯模板的专属参数为 NULL，则返回 -1 报错。
    if (privated == NULL)
        return -1;

    // 这里将尺寸参数的 x 分量和专属参数进行异或拼合。考虑到 size.x 为大于等于
    // 3 的奇数，为了更加有效的利用存储空间，这里将 size.x / 2 - 1 以消除码距。
    return ((size.x / 2 - 1) ^ (int)(fabs(*(float *)privated) * 10.0f)) %
           TF_VOL_SHAPE;
}

// Host 函数：_gaussCreateTemplate（高斯模板生成函数）
static __host__ Template *_gaussCreateTemplate(dim3 size, void *privated)
{
    // 如果专属参数为 NULL 则报错返回。
    if (privated == NULL)
        return NULL;

    // 取出专属参数的值，该值在为 float 型，在生成高斯模板的过程中，称为 sigma
    float sigma = *((float *)privated);
    // 如果 sigma 值等于 0 则无法完成后续计算，报错退出。
    if (fabs(sigma) < 1.0e-8f)
        return NULL;
    // 计算出 2 * sigma ^ 2，方便后续计算使用。
    float sigma22 = 2 * sigma * sigma;

    // 计算出半径尺寸。
    int radius = size.x / 2;
    if (radius < 1)
        radius = 1;
    // 根据半径推算出边长和模板中点的总数量。
    int edgelen = 2 * radius + 1;
    int maxptscnt = edgelen * edgelen;

    // 申请新的模板
    Template *restpl;
    int errcode;
    errcode = TemplateBasicOp::newTemplate(&restpl);
    if (errcode != NO_ERROR)
        return NULL;

    // 为新模板申请内存空间。
    errcode = TemplateBasicOp::makeAtHost(restpl, maxptscnt);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(restpl);
        return NULL;
    }

    // 取出存放坐标点和附属数据的内存空间指针，这样可以通过游标指针方便对数据空
    // 间进行赋值操作。
    int *tpldata = restpl->tplData;
    float *attdata = ATTACHED_DATA(restpl);

    // 首先将原点信息添加到模板中。
    *(tpldata++) = 0;
    *(tpldata++) = 0;
    *(attdata++) = 1.0f;

    // 之后依次从内向外逐渐的添加模板数据，之所以从内向外添加模板数据，是因为这
    // 样做可以实现很好的复用性，即，模板的前 i * i 个元素就是边长为 i 的高斯模
    // 板。
    for (int i = 1; i <= radius; i++) {
        // 计算指定半径下的各个坐标点信息。这里，利用对称性，计算处一个边的坐标
        // 点然后利用对称性，得到其他的坐标点。为了防止拐角点重复计算，这里计算
        // 的范围为 -i - 1 到 i。
        for (int j = -i + 1; j <= i; j++) {
            // 由于四个对称点的二阶模相等，因此预先计算出附属数据，以减少计算
            // 量。
            float curatt = exp(-(i * i + j * j) / sigma22);

            // 将对称的四个坐标点添加到模板中，同时添加附属数据。
            *(tpldata++) = i;
            *(tpldata++) = j;
            *(attdata++) = curatt;
            *(tpldata++) = -i;
            *(tpldata++) = j;
            *(attdata++) = curatt;
            *(tpldata++) = j;
            *(tpldata++) = i;
            *(attdata++) = curatt;
            *(tpldata++) = j;
            *(tpldata++) = -i;
            *(attdata++) = curatt;
        }
    }

    // 计算完毕，返回新的模板。
    return restpl;
}

// Host 函数：_gaussIsEqualSize（高斯模板的尺寸相等判断函数）
static __host__ bool _gaussIsEqualSize(dim3 size1, void *privated1,
                                     dim3 size2, void *privated2)
{
    // 如果专属参数为 NULL，则恒返回 false 报错。
    if (privated1 == NULL || privated2 == NULL)
        return false;

    // 如果按照 float 类型判断，两个专属参数不相等（即绝对值差大于某个小正
    // 数），则判定为不相等。
    if (fabs(*((float *)privated1) - *((float *)privated2)) >= 1.0e-6f)
        return false;

    // 如果尺寸参数不相等，则会判定为两个参数不相等
    if (size1.x != size2.x/* || size1.y !=  size2.y*/)
        return false;

    return true;
}

// Host 函数：_gaussCopyPrivated（高斯模板的专属参数拷贝函数）
static __host__ void *_gaussCopyPrivated(void *privated)
{
    // 如果专属参数为 NULL，则直接返回 NULL。
    if (privated == NULL)
        return NULL;

    // 申请一个 float 型的空间
    float *resptr = new float;
    if (resptr == NULL)
        return NULL;

    // 然后将数据拷贝如这个空间内。
    *resptr = *((float *)privated);

    // 返回这个已拷贝了数据的新申请的空间地址。
    return resptr;
}

// Host 函数：_gaussDeletePrivated（高斯模板的专属参数释放函数）
static __host__ bool _gaussDeletePrivated(void *privated)
{
    // 如果专属参数为 NULL，则直接返回 NULL。
    if (privated == NULL)
        return false;

    // 释放掉 privated 的空间。
    delete (float *)privated;
    return true;
}


// 欧式模板的定义：

// Host 函数：_euclideIsLegalParam（欧式模板参数判断函数）
// 检查欧式模板的参数是否合格，合格的模板要求尺寸参数的 z 分量为 1，专属参数不
// 能为NULL；此外欧式模板还要求 x 和 y 分量尺寸必须大于等于 1，且 y 分量必须等
// 于 x 分量或者 y 分量等于 1。
static __host__ bool    // 返回值：如果模板合法，则返回 true，否则返回 false
_euclideIsLegalParam(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_euclideGetHashIndex（欧式模板 Hash 算法函数）
// 欧式模板的 Hash 算法。该算法之使用尺寸参数的 x 分量，由于欧式模板使用专属
// 参数，所以在计算 Hash 的时候也考虑专属参数。
static __host__ int     // 返回值：Hash 值，如果出现错误则该函数返回负数。
_euclideGetHashIndex(
        dim3 size,      // 尺寸参数
        void *privated  // 专属参数
);

// Host 函数：_euclideCreateTemplate（欧式模板生成函数）
// 生成欧式模板的函数。
static __host__ Template *  // 返回值：生成的模板，若无法生成模板会返回 NULL。
_euclideCreateTemplate(
        dim3 size,          // 尺寸参数
        void *privated      // 专属参数
);

// Host 函数：_euclideIsEqualSize（欧式模板的尺寸相等判断函数）
// 欧式模板使用了尺寸中的一个维度，因此该函数只会检查尺寸参数的 x 维度是否相
// 等。同时还检查了 privated 在 float 型数据的条件下是否相等（即两个数的差的绝
// 对值小于某一个很小的正数）
static __host__ bool      // 返回值：给定的两个尺寸是否是一样的。
_euclideIsEqualSize(
        dim3 size1,       // 第一个尺寸的尺寸参数
        void *privated1,  // 第一个尺寸的专属参数
        dim3 size2,       // 第二个尺寸的尺寸参数
        void *privated2   // 第二个尺寸的专属参数
);

// Host 函数：_euclideCopyPrivated（欧式模板的专属参数拷贝函数）
// 按照 float 型数据的方式，申请一个新的 float 型数据空间，并将 privated 所指向
// 的 float 型数据拷贝到新的空间中。
static __host__ void *  // 返回值：如果 privated 为 NULL，则返回 NULL；否则返回
                        // 新申请的数据空间。
_euclideCopyPrivated(
        void *privated  // 待拷贝的专属参数。
);

// Host 函数：_euclideDeletePrivated（欧式模板的专属参数释放函数）
// 按照释放一个 float 型数据的地址空间的方法释放给定的空间。如果 privated 是
// NULL 则不进行任何操作。
static __host__ bool    // 返回值：如果释放成功返回 true，对于 NULL 参数，返回
                        // false。
_euclideDeletePrivated(
        void *privated  // 待释放的专属函数
);

// Host 全局变量：_euclideTemplateVendor（欧式模板生成器）
// 归纳定义欧式模板生成所需要的函数。
static TFTemplateVendor _euclideTemplateVendor = {
    _euclideIsLegalParam,
    _euclideGetHashIndex,
    _euclideCreateTemplate,
    _euclideIsEqualSize,
    _euclideCopyPrivated,
    _euclideDeletePrivated
};

// Host 函数：_euclideIsLegalParam（欧式模板参数判断函数）
static __host__ bool _euclideIsLegalParam(dim3 size, void *privated)
{
    // 由于高斯模板使用了 float 型的专属参数，因此需要保证 privated 不为 NULL。
    if (privated == NULL)
        return false;
    // 由于只是用了一个维度的尺寸变量，这里要求 z 维度必须为 1。
    else if (size.z != 1)
        return false;
    // 这里了要求尺寸必须大于等于 1 且 x 和 y 分量必须相等或者 y 分量等于 1。
    else if (size.x < 1 || (size.y != size.x && size.y != 1))
        return false;
    // 如果附属数据的值等于 0，则判断为非法。
    else if (fabs(*((float *)privated)) < 1.0e-8f)
        return false;
    else
        return true;
}

// Host 函数：_euclideGetHashIndex（欧式模板 Hash 算法函数）
static __host__ int _euclideGetHashIndex(dim3 size, void *privated)
{
    // 如果高斯模板的专属参数为 NULL，则返回 -1 报错。
    if (privated == NULL)
        return -1;

    // 这里将尺寸参数的 x 分量和专属参数进行异或拼合。
    return (size.x ^ (int)(fabs(*(float *)privated) * 10.0f)) % TF_VOL_SHAPE;
}

// Host 函数：_euclideCreateTemplate（欧式模板生成函数）
static __host__ Template *_euclideCreateTemplate(dim3 size, void *privated)
{
    // 如果专属参数为 NULL 则报错返回。
    if (privated == NULL)
        return NULL;

    // 如果尺寸小于 1，则无法完成有效的计算，报错返回。
    if (size.x < 1)
        return NULL;

    // 取出专属参数的值，该值在为 float 型，在生成欧式模板的过程中，称为 sigma
    float sigma = *((float *)privated);
    // 如果 sigma 值等于 0 则无法完成后续计算，报错退出。
    if (fabs(sigma) < 1.0e-8f)
        return NULL;
    // 计算出 2 * sigma ^ 2，方便后续计算使用。
    float sigma22 = 2 * sigma * sigma;

    // 申请新的模板
    Template *restpl;
    int errcode;
    errcode = TemplateBasicOp::newTemplate(&restpl);
    if (errcode != NO_ERROR)
        return NULL;

    // 为新模板申请内存空间。
    errcode = TemplateBasicOp::makeAtHost(restpl, size.x);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(restpl);
        return NULL;
    }

    // 取出存放坐标点和附属数据的内存空间指针，这样可以通过游标指针方便对数据空
    // 间进行赋值操作。
    int *tpldata = restpl->tplData;
    float *attdata = ATTACHED_DATA(restpl);

    // 依次添加模板数据。
    for (int i = 0; i < size.x; i++) {
        // 计算当前点对应的附属数据值。
        float curatt = exp(-(i * i) / sigma22);

        // 将坐标点添加到模板中，同时添加附属数据。
        *(tpldata++) = i;
        *(tpldata++) = 0;
        *(attdata++) = curatt;
    }

    // 计算完毕，返回新的模板。
    return restpl;
}

// Host 函数：_euclideEqualSize（欧式模板的尺寸相等判断函数）
static __host__ bool _euclideIsEqualSize(dim3 size1, void *privated1,
                                         dim3 size2, void *privated2)
{
    // 如果专属参数为 NULL，则恒返回 false 报错。
    if (privated1 == NULL || privated2 == NULL)
        return false;

    // 如果按照 float 类型判断，两个专属参数不相等（即绝对值差大于某个小正
    // 数），则判定为不相等。
    if (fabs(*((float *)privated1) - *((float *)privated2)) >= 1.0e-6f)
        return false;

    // 如果尺寸参数不相等，则会判定为两个参数不相等
    if (size1.x != size2.x/* || size1.y !=  size2.y*/)
        return false;

    return true;
}

// Host 函数：_euclideCopyPrivated（欧式模板的专属参数拷贝函数）
static __host__ void *_euclideCopyPrivated(void *privated)
{
    // 如果专属参数为 NULL，则直接返回 NULL。
    if (privated == NULL)
        return NULL;

    // 申请一个 float 型的空间
    float *resptr = new float;
    if (resptr == NULL)
        return NULL;

    // 然后将数据拷贝如这个空间内。
    *resptr = *((float *)privated);

    // 返回这个已拷贝了数据的新申请的空间地址。
    return resptr;
}

// Host 函数：_euclideDeletePrivated（欧式模板的专属参数释放函数）
static __host__ bool _euclideDeletePrivated(void *privated)
{
    // 如果专属参数为 NULL，则直接返回 NULL。
    if (privated == NULL)
        return false;

    // 释放掉 privated 的空间。
    delete (float *)privated;
    return true;
}


// 综合归纳定义各个模板：

// Host 全局变量：_templateVendorArray（模板生成器集合）
// 通过这个数组管理系统中所有的模板生成器，通过索引可以灵活的访问到系统中所有的
// 模板生成器的函数。
static TFTemplateVendor *_templateVendorArray[] = {
    &_boxTemplateVendor,      // 矩形模板生成器
    &_circleTemplateVendor,   // 圆形模板生成器
    &_arcTemplateVendor,      // 环形模板生成器
    &_gaussTemplateVendor,    // 高斯模板生成器
    &_euclideTemplateVendor,  // 欧式模板生成器
    &_defTemplateVendor       // 标准模板生成器（用于演示的代码示例）
};


// TemplateFactory CLASS 的实现方法

// 静态成员变量：tplPool（模板池）
// 初始化 tplPool 的值为 NULL。
Template *
TemplateFactory::tplPool[TF_CNT_SHAPE][TF_VOL_SHAPE * TF_SET_SHAPE] = { NULL };

// 静态成员变量：sizePool（模板池尺寸参数）
// 初始化 sizePool 的值为 (0, 0, 0)。
dim3
TemplateFactory::sizePool[TF_CNT_SHAPE][TF_VOL_SHAPE * TF_SET_SHAPE] = {
    dim3(0, 0, 0)
};

// 静态成员变量：privatePool（模板池专属参数）
// 初始化 privatePool 的值为 NULL。
void *
TemplateFactory::privatePool[TF_CNT_SHAPE][TF_VOL_SHAPE * TF_SET_SHAPE] = {
    NULL
};

// 静态成员变量：countPool（模板池使用计数器）
// 初始化 countPool 的值为 0。
int
TemplateFactory::countPool[TF_CNT_SHAPE][TF_VOL_SHAPE * TF_SET_SHAPE] = { 0 };

// Host 静态方法：boostTemplateEntry（提升指定的模板条目）
__host__ bool TemplateFactory::boostTemplateEntry(int shape, int idx)
{
    // 检查形状参数和下标参数的合法性（由于这一内部函数在掉用前可以保证正确性，
    // 因此我们注释了下面的代码）
    //if (shape < 0 || shape >= TF_CNT_SHAPE ||
    //    idx < 0 || idx >= TF_VOL_SHAPE * TF_SET_SHAPE)
    //    return false;

    // 如果给定的下标可以被组块尺寸整除则说明该下标为某个组块的开头一个下标，这
    // 个下标下的条目已经不能够在继续提升了，因为再提升就会进入其他的组块，从而
    // 造成后续处理的错误。
    if (idx % TF_SET_SHAPE == 0)
        return false;

    // 进行两个模板条目的交换。
    // 首先将 idx 位置的条目保存到一个临时内存区域内。
    Template *tmptpl = tplPool[shape][idx];
    dim3 tmpsize = sizePool[shape][idx];
    void *tmppriv = privatePool[shape][idx];
    int tmpcount = countPool[shape][idx];
    // 再将 idx - 1 位置的标目保存到 idx 位置处。
    tplPool[shape][idx] = tplPool[shape][idx - 1];
    sizePool[shape][idx] = sizePool[shape][idx - 1];
    privatePool[shape][idx] = privatePool[shape][idx - 1];
    countPool[shape][idx] = countPool[shape][idx - 1];
    // 最后将存在来临时内存区域内的原 idx 位置的数据放入 idx - 1 处。
    tplPool[shape][idx - 1] = tmptpl;
    sizePool[shape][idx - 1] = tmpsize;
    privatePool[shape][idx - 1] = tmppriv;
    countPool[shape][idx - 1] = tmpcount;

    // 处理完毕，退出返回
    return true;
}

// Host 静态方法：getTemplate（根据参数得到需要的模板）
__host__ int TemplateFactory::getTemplate(Template **tpl, int shape, 
                                          dim3 size, void *privated)
{
    // 判断给定的形状是否合法
    if (shape < 0 || shape >= TF_CNT_SHAPE)
        return INVALID_DATA;
    
    // 检查用于输出的模板指针是否为空。
    if (tpl == NULL)
        return NULL_POINTER;

    // 判断尺寸是否合法
    if (!_templateVendorArray[shape]->isLegalParam(size, privated))
        return INVALID_DATA;

    int hashidx;      // Hash 索引值
    int posidx;       // 模板池的下标游标值，该值通过 Hash 索引值推算而得。
    int startposidx;  // 模板池中存放所要查找模板的起始下标。
    int endposidx;    // 模板池中存放所要查找模板的最后一个位置的下一个位置的下
                      // 标。

    // 调用模板生成器中的 Hash 函数生成索引
    hashidx = _templateVendorArray[shape]->getHashIndex(size, privated);
    // 如果模板生成器返回的 Hash 值无法使用，则报错退出。
    if (hashidx < 0 || hashidx > TF_VOL_SHAPE)
        return INVALID_DATA;

    // 由 Hash 索引定位模板的对应首位置与结束位置
    startposidx = hashidx * TF_SET_SHAPE;
    endposidx = startposidx + TF_SET_SHAPE;

    // 循环查找模板池是否有对应模板
    for (posidx = startposidx; posidx < endposidx; posidx++) {

        // 如果发现当前位置的模板为 NULL，所有所要查找的模板尚未出现在模板池
        // 中，需要创建创建这个模板。
        if (tplPool[shape][posidx] == NULL) {
            break;

        } else if (_templateVendorArray[shape]->isEqualSize(
                           sizePool[shape][posidx], privatePool[shape][posidx],
                           size, privated)) {

            // 在模板池中找到了需要的模板
            *tpl = tplPool[shape][posidx];
            countPool[shape][posidx]++;

#ifdef TF_ENABLE_KICK
            // 如果当前被选中的模板不处于模板池的首位置，则跟它前面的模板进行交
            // 换位置，使得这个模板不容易被替换出去。该段代码只在启动了替换机制
            // 后才会被执行。
            boostTemplateEntry(shape, posidx);
#endif

            // 找到模板后直接退出
            return NO_ERROR;
        }
    }

    // 如果在模板池中找到了对应的模板，则带么已经通过上面的 else if 分支返回到
    // 上层函数，因此下面的代码将处理没有找到对应模板的情况。

    // 首先，创建对应的模板。这里先创建模板是因为如果一旦创建失败，就没有必要再
    // 去尝试这将这个模板放入到模板池中。
    *tpl = _templateVendorArray[shape]->createTemplate(size, privated);
    // 检查得到的方形模板是否为空
    if (*tpl == NULL)
        return OUT_OF_MEM;

    // 然后，如果模板池已经满了，那么必须从模板池中踢出一个模板，然后在放入新的
    // 模板。（当然，如果我们吧替换功能关闭了，那么该部分代码直接以没有找到替换
    // 位置的姿态离开该段代码）
    if (posidx >= endposidx) {
#ifdef TF_ENABLE_KICK
        // 一些要使用到的临时变量
        bool replace = false;    // 标志位，表示是否找到和替换位置
        posidx = endposidx - 1;  // 因为落在后面的模板是更为不常用的模板，因此
                                 // 我们设定起始下标为最末一个下标。
        int stopposidx =                        // 查找停止下标。为了防止不合理
                (startposidx + endposidx) / 2;  // 的替换，我们令在前面的模板不
                                                // 会被替换掉。
        // 从后向前查找，找到一个暂时没有被使用的模板，然后替换掉他。这里使用
        // do-while 循环的形式，就是保证至少查找了最后一个条目。用来防止，如果
        // 组块设置得太小的时候会导致不进行查找直接认为没有找到合适的条目的情
        // 况。
        do {
            // 如果当前的条目下的模板没有被使用。这里之所以要寻找没有使用的模
            // 板，是因为一旦将正在使用的模板踢出模板池，我们将无法控制这个模板
            // 何时应该释放：如果当场释放，那么虽然不会带来内存泄漏的问题，但是
            // 正在使用模板的程序可能会因为读不到数据而崩溃；如果过后释放，则仍
            // 就可能会导致其他使用者的崩溃。因此，这里我们目前只是寻找目前没有
            // 在使用中的模板，这样我们就能够安全的释放它，而不用担心内存泄漏或
            // 者使用者崩溃的问题。
            if (countPool[shape][posidx] == 0) {
                // 设定标志位，并且删除该模板，腾出内存空间。
                replace = true;

                // 删除模板，防止内存泄漏。由于后续步骤马上就要改写模板池中的内
                // 容，因此这里注视掉了将模板池重置为 NULL 的代码，以减少一些不
                // 必要的代码执行。下面注释掉的专属变量重置为 NULL 的代码被注释
                // 掉的原因亦然。
                TemplateBasicOp::deleteTemplate(tplPool[shape][posidx]);
                //tplPool[shape][posidx] = NULL;
                
                // 这里还需要释放掉模板池中的专属参数，因为这个空间都是在将模板
                // 加入到模板池中的时候才申请的，因此需要在内部将其释放掉。
                _templateVendorArray[shape]->deletePrivated(
                        privatePool[shape][posidx]);
                //privatePool[shape][posidx] = NULL;
                
                // 找到合适的位置后直接从循环中跳出。
                break;
            }
        } while (posidx-- >= stopposidx);

        // 如果没有找到提供换位置，则将下标赋值为一个哑值。
        if (!replace)
            posidx = -1;
#else
        // 对于关闭了替换功能的代码，这里直接将下标置为一个哑值。
        posidx = -1;
#endif
    }

    // 给模板池和模板对应参数赋值，这里首先要检查 posidx 是不是哑值，因为即便是
    // 在模板池内没有找到合适的位置来放置新的模板，我们也会创造一个模板来给用户
    // 使用，这样的模板会在调用 putTemplate 时被销毁（该函数会发现该模板没有在
    // 模板池中存在）。
    if (posidx >= startposidx && posidx < endposidx) {
        tplPool[shape][posidx] = *tpl;
        sizePool[shape][posidx] = size;
        // 这里我们调用 copyPrivated 函数，因为用户输入的 privated 指针指向的数
        // 据是不稳定的，它可能会被改写，会被释放，这样我们的系统也会变得不稳
        // 定。为了防止这种不稳定性，我们将它拷贝一份出来，供模板池内部专用。
        privatePool[shape][posidx] =
                _templateVendorArray[shape]->copyPrivated(privated);
        countPool[shape][posidx] = 1;
    }

    // 处理完毕，退出
    return NO_ERROR;
}

// host 静态方法：putTemplate（放回通过 getTemplate 得到的模板）
__host__ int TemplateFactory::putTemplate(Template *tpl)
{
    // 如果输入的模板是 NULL，则直接返回。
    if (tpl == NULL)
        return NULL_POINTER;

    // 查找模板池中有没有对应的模板，如果模板池中存在对应的模板，则将模板的使用
    // 计数器进行调整。
    for (int shape = 0; shape < TF_CNT_SHAPE; shape++) {
        for (int i = 0; i < TF_VOL_SHAPE * TF_SET_SHAPE; i++) {
            // 模板池中找到了对应模板，让模板计数器减 1。
            if (tplPool[shape][i] == tpl) {
                if (countPool[shape][i] > 0)
                    countPool[shape][i]--;

                // 处理完毕，退出
                return NO_ERROR;
            }
        }
    }

    // 模板此没有找到对应的模板，则直接释放该模板
    TemplateBasicOp::deleteTemplate(tpl);

    // 处理完毕，退出
    return NO_ERROR; 
}
