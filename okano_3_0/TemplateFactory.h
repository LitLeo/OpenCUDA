// TemplateFactory.h
// 创建者：于玉龙
//
// 模板工厂（TemplateFactory）
// 功能说明：创建和得到各种形状的模板，暂时先实现了矩形模板（包括长方形和
//           正方形）、 圆形模板、环形模板、高斯模板、欧式模板
//
// 修订历史：
// 2012年11月17日（欧阳翔，于玉龙）
//     初始版本。
// 2012年11月22日（欧阳翔）
//     在设计上做了修改，使用了函数指针数组，减少了重复代码。
// 2012年11月23日（欧阳翔）
//     模板替换策略上做了修改，修正了一些格式错误。
// 2012年11月24日（欧阳翔，于玉龙）
//     在模板特殊参数比较方法上做了修正
// 2012年11月25日（欧阳翔）
//     修改了一些格式错误
// 2012年11月28日（于玉龙）
//     修正了圆形模版生成算法，计算圆形模版更加准确。
//     修正了代码中的部分格式错误。
// 2012年12月05日（于玉龙）
//     聚合生成一类模板所需要的各种函数，组成一个有函数指针构成的结构体，方便不
//     同类型模板的管理。
// 2012年12月16日（于玉龙）
//     全面翻新内部代码，使用结构体中的函数指针来管理各个模板生成算法，修正了原
//     代码中大量潜在的 Bug。
//     修改“拱形模板”的中文名称为“环形模板”。
// 2012年12月18日（于玉龙）
//     重新实现了环形模板，其计算量更小，求解更加准确合理。此外，修改了参数的含
//     义，原尺寸参数为半径，现在指直径。
// 2012年12月23日（于玉龙）
//     重新实现了高斯模板。
//     修正了环形模版生成中一处潜在的计算错误。
// 2012年12月24日（于玉龙）
//     修正了高斯模板计算过程中的一个错误。
//     增加了欧式模板的实现。

#ifndef __TEMPLATEFACTORY_H__
#define __TEMPLATEFACTORY_H__

#include "ErrorCode.h"
#include "Template.h"

// 宏：TF_CNT_SHAPE（系统所支持的模版数量）
// 定义了模版工厂所支持的模版形状的数量。
#define TF_CNT_SHAPE       5

// 宏：TF_VOL_SHAPE 和 TF_SET_SHAPE（模版工厂容量）
// 定义了模版工厂中的资源池所能容纳的模版数量。由于存储于资源池的模版采用哈希表
// 组映射的方式存储，因此这里采用两个宏，分别来记录组的数量和每组内的模版数量。
#define TF_VOL_SHAPE     128
#define TF_SET_SHAPE       4

// 宏：模版工厂所支持的形状
// 定义了所有模版工厂所支持的形状。目前的版本中我们包含了五种形状，分别为矩形、
// 圆形、环形、高斯模版、欧式模板。
#define TF_SHAPE_BOX       0
#define TF_SHAPE_CIRCLE    1
#define TF_SHAPE_ARC       2
#define TF_SHAPE_GAUSS     3
#define TF_SHAPE_EUCLIDE   4


// 类：TemplateFactory（模板工厂）
// 继承自：无
// 该类包含了对于产生不同形状的模板操作，包括矩形模板、圆形模板、环形模板、高
// 斯模板等的创建，以及在模板池中查找以上模板操作。超过模板池容量后从对应模板
// 适当位置开始替换需要新建的模板。
class TemplateFactory {

protected:

    // 静态成员变量：tplPool（模板池）
    // 存储不同类型的模板，初始化成 NULL。做成二维数组的形式，Template 以 Hash
    // 的形式存入资源池，可以方便写入，也方便查找。
    static Template *tplPool[TF_CNT_SHAPE][TF_VOL_SHAPE * TF_SET_SHAPE];

    // 静态成员变量：sizePool（模板池尺寸参数）
    // 这是模板对应的参数，查找时需要对比的。
    // 尺寸池，静态全局变量，对应于模板池，dim3类型，初始化为 (0, 0, 0)
    static dim3 sizePool[TF_CNT_SHAPE][TF_VOL_SHAPE * TF_SET_SHAPE];

    // 静态成员变量：privatePool（模板池专属参数）
    // 模板对应的专属参数，查找时需要对比的。
    // 模板其他属性池，静态全局变量，对应于模板池，void * 类型，初始化为 NULL。
    static void *privatePool[TF_CNT_SHAPE][TF_VOL_SHAPE * TF_SET_SHAPE];

    // 静态成员变量：countPool（模板池使用计数器）
    // 模板计数，统计每一个模板的 getTemplate 的次数
    // 初始化为 0，当对应大小的池满，执行替换策略，若对应模板的 count 为 0，
    // 则表示外部已经放回了得到的模板，可以执行替换，否则不能执行
    static int countPool[TF_CNT_SHAPE][TF_VOL_SHAPE * TF_SET_SHAPE];

    // Host 静态方法：boostTemplateEntry（提升指定的模板条目）
    // 将指定的模板池中模板的条目与其前面的条目进行交换，这可以使该模板条目处于
    // 更加有优势的地位，避免其被替换操作释放掉。
    static __host__ bool  // 返回值：是否交还成功，如果模板处于某个组块的第一个
                          // 模板的位置，则该模板就不会在进行提升操作，会返回
                          // false
    boostTemplateEntry(
            int shape,    // 形状
            int idx       // 模板池中的下标
    );

public:

    // Host 静态方法：getTemplate（得到需要的模板）
    // privated 是形状的特有参数，通常来说不使用。
    // 这是一个显示的内联函数，根据需要得到对应的模板，需要调用下面重载的
    static inline __host__ int     // 返回值：函数是否正确执行，若函数正确执行
                                   // 返回 NO_ERROR。
    getTemplate(
            Template **tpl,        // 模板指针
            int shape,             // 形状参数，表示想得到的形状
            size_t size,           // 需要模板的尺寸大小
            void *privated = NULL  // 模板区分形状的特有参数
    ) {
        // 根据用户尺寸转化为 dim3 类型的尺寸大小
        dim3 tmpsize(size, size);
        // 调用得到模板函数
        return getTemplate(tpl, shape, tmpsize, privated); 
    }

    // Host 静态方法：getTemplate（得到需要的模板）
    // privated 是形状的特有参数，通常来说不使用。
    // 这个函数首先判断所需要的模板是否已经在 tplPool 中，若在，则直接返回；
    // 若不在，调用 create 生成，返回这个刚生成的模板，并且将这个模板存放在
    // tplPool 对应的位置中。并且 countPool 对应位置需要加 1。
    static __host__ int            // 返回值：函数是否正确执行，若函数正确
                                   // 执行，返回 NO_ERROR。
    getTemplate(
            Template **tpl,        // 模板指针
            int shape,             // 形状参数，表示想得到的形状
            dim3 size,             // 需要模板的尺寸大小
            void *privated = NULL  // 模板区分形状的特有参数
    );

    // Host 静态方法：putTemplate（放回通过 getTemplate 得到的模板）
    // 主要是为了平衡一般性思维，由于前面有得到模板，这个逻辑上觉得需要放回
    // 如果查找模板不在 tplPool 池中，则直接释放模板空间。否则对应位置的
    // countPool中的值减 1。
    static __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    putTemplate(
            Template *tpl  // 放回模板指针
    );
};

#endif

