// ImageMatch.h
// 创建人：罗劼
//
// 由坐标映射的图像匹配（Image match by coordinate mapping）
// 功能说明：用坐标映射的图像的相关计算用来对图像进行匹配。提供一组TEMPLATE图像
//           以及一张需要匹配的TEST图像，分别用不同旋转角的TEMPLATE图像对TEST图
//           像进行匹配，得到最大匹配度相对应的旋转角以及在TEST图像上匹配的相应
//           位置。
//
// 修订历史：
// 2012年11月21日（罗劼）
//     初始版本
// 2012年11月22日（罗劼，于玉龙）
//     对代码的一些格式不规则的地方进行了修改
// 2012年11月23日（罗劼，于玉龙）
//     修改了对 Device 函数的命名以及在分配 Device 内存时，采用先申请一块足够大
//     的内存，然后从这快大内存中获取需要的内存
// 2012年11月29日（罗劼）
//     修改了计算最佳匹配的旋转角的错误
// 2012年12月04日（罗劼）
//     在 MatchRes 结构体中增加了一个变量，用来记录匹配得到的最大相关系数
// 2012年12月05日（罗劼）
//     增加了在初始化旋转表之前先判断旋转表是否为空的判断
// 2012年12月28日（罗劼）
//     增加了局部异常检查
// 2013年01月14日（罗劼）
//     修改了设置局部异常检查的错误以及实现了部分获取 errmap 的最小有向四边形
// 2013年04月17日（罗劼）
//     完成了局部异常检查模块，修改了调用图像匹配的接口
// 2013年04月17日（罗劼）
//     将释放内存部分改写成宏
// 2013年05月07日（罗劼）
//     修改了一处计算相关系数的错误

#ifndef __IMAGEMATCH_H__
#define __IMAGEMATCH_H__

#include "Image.h"
#include "RotateTable.h"
#include "ConnectRegion.h"
#include "ErrorCode.h"
#include "LabelIslandSortArea.h"
#include "Normalization.h"
#include "Rectangle.h"
#include "RoiCopy.h"
#include "SmallestDirRect.h"
#include "Template.h"
#include "TemplateFactory.h"

// 结构体：MatchRes（匹配后得到的结果）
// 该结构体定义了匹配后得到的结果，包括最佳匹配位置，最佳匹配对应的旋转角以及最
// 佳匹配对应的 TEMPLATE 的索引
typedef struct MatchRes_st {
    int matchX;         // 匹配后得到的最佳匹配位置的横坐标
    int matchY;         // 匹配后得到的最佳匹配位置的纵坐标
    float angle;        // 最佳匹配对应的旋转角
    int tplIndex;       // 最佳匹配对应的 TEMPLATE 的索引 
    float coefficient;  // 匹配得到的最大的相关系数
} MatchRes;

// 类：ImageMatch
// 继承自：无
// 用坐标映射的图像的相关计算用来对图像进行匹配。提供一组 TEMPLATE 图像以及一张
// 需要匹配的 TEST 图像，分别用不同旋转角的 TEMPLATE 图像对 TEST 图像进行匹配，
// 得到最大匹配度相对应的旋转角以及在 TEST 图像上匹配的相应位置。
class ImageMatch {

protected:

    // 成员变量：rotateTable（旋转表）
    // 记录了 TEMPLATE 的各个旋转角度对应的坐标，该指针指向 Device 内存空间
    RotateTable *rotateTable;

    // 成员变量：dWidth 和 dHeight（摄动范围的宽和高）
    // 记录摄动范围的宽和高
    int dWidth, dHeight;

    // 成员变量：dx 和 dy（摄动中心）
    // 记录摄动范围的中心
    int dx, dy;

    // 成员变量：scope（标记在每个点为中心对相关系数求和的邻域的大小）
    // 标记在每个点为中心对相关系数求和的邻域的大小
    int scope;

    // 成员变量：errThreshold（局部异常检查的阈值）
    // 在进行局部异常检查时，需要用到此阈值
    float errThreshold;

    // 成员变量：errWinWidth（errmap 窗口的宽）
    // 进行局部异常检查时 errmap 窗口的宽
    int errWinWidth;

    // 成员变量：errWinHeight（errmap 窗口的高）
    // 进行局部异常检查时 errmap 窗口的高
    int errWinHeight;

    // 成员变量：errWinThreshold（errmap 窗口的阈值）
    // 进行局部异常检查是 errmap 窗口高值点的阈值
    int errWinThreshold;

    // 成员变量：needNormalization（标记是否需要对 TEMPLATE 进行正规化）
    // 标记是否需要对 TEMPLATE 进行正规化
    bool needNormalization;

    // 成员变量：tplImages（一组 TEMPLATE）
    // 存储指向 TEMPLATE 的指针
    Image **tplImages;

    // 成员变量：tplCount（TEMPLATE 数量）
    // 记录 TEMPLATE 的数量
    int tplCount;

    // 成员变量：tplTmpCount（TEMPLATE 的临时数量）
    // 临时记录 TEMPLATE 的数量，只有在调用 initNormalizeData 函数时，才把
    // tplTmpCount 赋值给 tplCount
    int tplTmpCount;

    // 成员变量：tplNormalization（TEMPLATE 正规化后得到的结果）
    // 用来存储每个 TEMPLATE 正规化后得到的结果，每个 TEMPLATE 正规化的结果都
    // 存放在 Device 内存中
    float **tplNormalization;

    // 成员变量：pitch（TEMPLATE 正规化结果的 pitch 值）
    // 用来记录每个 TEMPLATE 的 pitch 值 
    size_t *pitch;

    // 成员变量：tplWidth 和 tplHeight（TEMPLATE 的宽和高）
    // 记录每个 TEMPLATE 的宽和高，这里要求每个 TEMPLATE 的宽和高必须相等
    int tplWidth, tplHeight;

    // 成员方法：setDefParameter（设置默认的参数）
    // 为所有的成员变量设置默认的参数
    __host__  __device__ void           // 无返回值
    setDefParameter()
    {
        this->rotateTable = NULL;       // 旋转表默认为空
        this->errWinWidth = 3;          // window 的宽默认为 3
        this->errWinHeight = 3;         // window 的高默认为 3
        this->errWinThreshold = 1;      // window 的阈值默认为 1
        this->errThreshold = 0;         // 局部异常检查的阈值默认为 0
        this->scope = 3;                // 设置默认的 scope 值为 3
        this->dWidth = 32;              // 设置默认的摄动范围的宽为 32
        this->dHeight = 32;             // 设置默认的摄动范围的高为 32
        this->dx = 32;                  // 设置默认的摄动中心的横坐标为 32
        this->dy = 32;                  // 设置默认的摄动中心的纵坐标为 32
        this->tplImages = NULL;         // 设置默认的 TEMPLATE 为 NULL
        this->tplNormalization = NULL;  // 设置默认的 TEMPLATE 正规化结果为 NULL
        this->pitch = NULL;             // 设置默认的 ptich 为 NULL
        this->tplCount = 0;             // 设置默认的 TEMPLATE 数量为 0
    }

    // 成员方法：normalizeForTpl（对设定的 TEMPLATE 进行正规化） 
    // 对所有的设置的 TEMPLATE 进行正规化 
    __host__ int        // 函数是否正确，若函数执行正确，返回 NO_ERROR
    normalizeForTpl();

    // 成员方法：initNormalizeData（对设置的 TEMPLATE 进行初始化）
    // 对 TEMPLATE 进行初始化，为 tplNormalization 分配空间 
    __host__ int          // 函数是否正确，若函数执行正确，返回 NO_ERROR  
    initNormalizeData();

    // 成员方法：deleteNormalizeData（删除 TEMPLATE 正规化的数据）
    // 删除 TEMPLATE 正规化的数据，即 tplNormalization 分配的空间 
    __host__ int            // 函数是否正确，若函数执行正确，返回 NO_ERROR   
    deleteNormalizeData();  

    // 成员方法：copyNormalizeDataToDevice（拷贝正规化结构的指针到 Device）
    // 拷贝 tplNormalization 指向的每个 TEMPLATE 正规化的结果的指针到 Device
    __host__ int                  // 函数是否正确，若函数执行正确，
    copyNormalizeDataToDevice();  // 返回 NO_ERROR

public:

    // 构造函数：ImageMatch
    // 无参数版本的构造函数，所有的成员变量都初始化为默认值
    __host__ __device__
    ImageMatch() {

        // 为成员变量设置默认的参数
        setDefParameter();
    }

    // 构造函数：ImageMatch
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数还可以通过其他成员函
    // 数改变
    __host__ __device__
    ImageMatch(
            RotateTable *rotatetable,           // 旋转表
            int dwidth, int dheight,            // 摄动范围的宽和高
            int dx, int dy,                     // 摄动中心的坐标
            int scope,                          // 对相关系数求和的邻域的大小
            int errwinwidth, int errwinheight,  // window 的窗口大小
            int errwinthreshold,                // window 高值点的阈值
            float errthreshold                  // 异常检查的阈值
    ) {

        // 为成员变量设置默认参数，防止用户在构造函数的参数中传递非法的初
        // 始值而使系统进入一个未知的状态
        setDefParameter();

        // 根据参数列表中的值设定成员变量的初值
        setRotateTable(rotatetable);
        setDWidth(dwidth);
        setDHeight(dheight);
        setDX(dx);
        setDY(dy);
        setScope(scope);
        setErrWinWidth(errwinwidth);
        setErrWinHeight(errwinheight);
        setErrWinThreshold(errwinthreshold);
        setErrThreshold(errthreshold);
    }

    // 成员方法：getRotateTable（获取旋转表的指针）
    // 获取 rotatetable 的值
    __host__ __device__ RotateTable *  // 返回值：成员变量 rotateTable 的值
    getRotateTable() const
    {
        // 返回 rotateTable 的值
        return this->rotateTable;
    } 

    // 成员方法：setRotateTable（设置旋转表）
    // 设置 rotatetable 的值
    __host__ __device__ int           // 函数是否正确执行，若函数正确执行，返
                                      // 回 NO_ERROR
    setRotateTable(
            RotateTable *rotatetable  // 旋转表的指针 
    ) {
        // 更新旋转表的值
        this->rotateTable = rotatetable; 

        // 处理完毕，返回
        return NO_ERROR;
    }

    // 成员方法：getScope（获取对相关系数求和时的邻域的大小）
    // 获取 scope 的值
    __host__ __device__ int  // 返回值：scope 的值
    getScope() const
    {
        // 返回 scope 的值
        return this->scope;
    }

    // 成员方法：setScope（设置对相关系数求和时的邻域的大小）
    // 设置 scope 的值
    __host__ __device__ int  // 函数是否正确执行，若函数正确执行，返回 NO_ERROR
    setScope(
            int scope        // 新的 scope 的值
    ) {
        // 判断 scope 的值是否合法，若不合法，直接返回
        if (scope <= 0)
            return INVALID_DATA;

        // 更新 scope 的值
        this->scope = scope;

        // 执行完毕，返回
        return NO_ERROR;
    }

    // 成员方法：getDWidth（获取摄动范围的宽）
    // 获取 dWidth 的值
    __host__ __device__ int  // 返回 dWidth 的值
    getDWidth() const
    {
        // 返回 dWidth 的值
        return this->dWidth;
    }

    // 成员方法：setDWidth（设置摄动范围的宽）
    // 设置 dWidth 的值
    __host__ __device__ int  // 函数是否正确执行，若函数正确执行，返回 NO_ERROR
    setDWidth(
            int dwidth       // 新的 dWidth 的值
    ) {
        // 判断 dwidth 的值是否合法，如果不合法，直接返回错误
        if (dwidth <= 0) 
            return INVALID_DATA;
 
        // 更新 dWidth 的值
        this->dWidth = dwidth;
 
        // 执行完毕，返回
        return NO_ERROR;
    }

    // 成员方法：getDHeight（获取摄动范围的高）
    // 获取 dHeight 的值
    __host__ __device__ int  // 返回 dHeight 的值 
    getDHeight() const
    {
        // 返回 dHeight 的值
        return this->dHeight;
    }

    // 成员方法：setDHeight（设置摄动范围的高）
    // 设置 dHeight 的值
    __host__ __device__ int  // 函数是否正确执行，若函数正确执行，返回 NO_ERROR 
    setDHeight(
            int dheight      // 新的摄动范围的高
    ) {
        // 判断 dheight 的值是否合法，如果不合法，直接返回
       if (dheight <= 0)
            return INVALID_DATA;

        // 更新 dHeight 的值
        this->dHeight = dheight;

        // 执行完毕，返回
        return NO_ERROR;
    }

    // 成员方法：getDX（获取摄动中心的横坐标）
    // 获取 dx 的值
    __host__ __device__ int  // 返回 dx 的值
    getDX() const
    {
        // 返回 dx 的值
        return this->dx;
    }

    // 成员方法：setDX（设置摄动中心的纵坐标）
    // 设置 dx 的值
    __host__ __device__ int  // 函数是否正确执行，若函数正确执行，返回 NO_ERROR
    setDX(
            int dx           // 新的 dx 的值
    ) {
        // 判断 dx 是否合法，若不合法，直接返回
        if (dx < 0)
            return INVALID_DATA;

        // 更新 dx 的值
        this->dx = dx;

        // 执行完毕，返回
        return NO_ERROR;
    }

    // 成员方法：getDY（获取摄动中心的纵坐标）
    // 获取 dy 的值
    __host__ __device__ int  // 返回 dy 的值
    getDY() const
    {
        // 返回 dy 的值
        return dy;
    }

    // 成员方法：setDY（设置摄动中心的纵坐标）
    // 设置 dy 的值
    __host__ __device__ int  // 函数是否正确执行，若函数正确执行，返回 NO_ERROR 
    setDY(
            int dy           // 新的 dy 的值
    ) {
        // 判断 dy 是否合法，若不合法，直接返回错误
        if (dy < 0) 
            return INVALID_DATA;

        // 更新 dy 的值
        this->dy = dy;

        // 执行完毕，返回
        return NO_ERROR;
    }

    // 成员方法：setErrWinWidth（设置 errmap 窗口的宽）
    // 设置 errWinWidth 的值
    __host__ __device__ int  // 函数是否正确执行，若函数正确执行，
                             // 则返回 NO_ERROR
    setErrWinWidth(
            int errwinwidth  // 新的 errwinwidth 的值
    ) {
        // 判断 errwinwidth 是否合法，若不合法，直接返回错误
        if (errwinwidth <= 0)
            return INVALID_DATA;
        
        // 更新 errWinWidth 的值
        this->errWinWidth = errwinwidth;
   
        // 执行完毕，返回
        return NO_ERROR;
    }

    // 成员方法：getErrWinWidth（获取 errmap 窗口的宽）
    // 获取成员变量 errWinWidth 的值
    __host__ __device__ int  // 返回 errmap 窗口的宽
    getErrWinWidth() const
    {
        // 返回 errWinWidth 的值
        return errWinWidth;
    }

    // 成员方法：setErrWinHeight（设置 errmap 窗口的高）
    // 设置 errWinHeight 的值
    __host__ __device__ int   // 函数是否正确执行，若正确执行，返回 NO_ERROR
    setErrWinHeight(
            int errwinheight  // 新的 errWinHeight 的值
    ) {
        // 判断 errwinheight 是否合法，若不合法，直接返回错误
        if (errwinheight <= 0)
            return INVALID_DATA;
        
        // 更新 errWinHeight 的值
        this->errWinHeight = errwinheight;
   
        // 处理完毕，返回
        return NO_ERROR;
    }

    // 成员方法：getErrWinHeight（获取 errmap 窗口的高）
    // 获取成员变量 errWinHeight 的值
    __host__ __device__ int  // 返回 errmap 窗口的高
    getErrWinHeight() const
    {
        // 返回 errWinHeight 的值
        return errWinHeight;
    }
    
    // 成员方法：setErrWinThreshold（设置 errmap 窗口的阈值）
    // 设置 errWinThreshold 的值
    __host__ __device__ int      // 函数是否正确执行，若正确执行，返回 NO_ERROR
    setErrWinThreshold(
            int errwinthreshold  // 新的 errWinThreshold 的值 
    ) {
        // 判断 errwinthreshold 是否合法，若不合法，直接返回错误
        if (errwinthreshold < 0) 
            return INVALID_DATA;
  
        // 更新 errWinThreshold 的值
        this->errWinThreshold = errwinthreshold;
  
        // 处理完毕，返回
        return NO_ERROR;
    }

    // 成员方法：getErrWinThreshold（获取 errmap 窗口的阈值）
    // 获取成员变量 errWinThreshold 的值
    __host__ __device__ int     // 返回 errmap 窗口的阈值
    getErrWinThreshold() const  
    {
        // 返回 errWinThreshold 的值
        return errWinThreshold;
    }

    // 成员方法：getErrThreshold（获取局部异常检查的阈值）
    // 获取 errThreshold 的值
    __host__ __device__ int  // 返回值：局部异常检查的阈值
    getErrThreshold() const
    {
        // 获取成员变量 errThreshold 的值
        return errThreshold;
    }       

    // 成员方法：setErrThreshold（设置局部异常检查的阈值）
    // 设置局部异常检查时使用的阈值
    __host__ __device__ int   // 返回值：函数是否正确执行，若函数正确执行，
                              // 返回 NO_ERROR
    setErrThreshold(
            float errthreshold  // 新的局部异常检查的阈值
    ) {

        // 如果参数合法，则进行赋值，否则，直接返回
        if (errthreshold > 0) {
            this->errThreshold = errthreshold;
        }

        // 处理完毕，返回
        return NO_ERROR;
    }

    // 成员方法：setTemplateImage（设置 TEMPLATE 的数据）
    // 设置所有的 TEMPLATE 的数据
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执行，
                                // 返回 NO_ERROR
    setTemplateImage(
            Image **tplimages,  // 要设置的一组 TEMPLATE 的图像
            int count           // TEMPLATE 的数量
    ) {
        // 判断 tplimages 是否为空，若为空，直接返回错误
        if (tplimages == NULL)
            return NULL_POINTER;

        // 判断 count 是否合法，若不合法，直接返回错误
        if (count <= 0)
            return INVALID_DATA;

        // 扫描 tplimages 的每一个成员，检查每个成员是否为空
        for (int i = 0; i < count; i++) {
            // 若某个成员为空，则直接返回错误
            if (tplimages[i] == NULL) 
                return NULL_POINTER;
        }

        // 更新 TEMPLATE 图像数据
        this->tplImages = tplimages;
        // 更新 tplTmpCount 数据，这里不能将 count 赋值给 tplCount 变量，因
        // 为需要使用 tplCount 变量来删除原来的 tplNormalization 数据
        this->tplTmpCount = count;
        // 更新模版的宽
        this->tplWidth = tplimages[0]->width;
        // 更新模版的高
        this->tplHeight = tplimages[0]->height;

        // 将 needNormalization 复制为 ture，标记需要对新的 TEMPLATE 进行初始化
        this->needNormalization = true;

        // 处理完毕，返回 NO_ERROR
        return NO_ERROR;
    }

    // 成员函数：imageMatch（用给定图像及不同旋转角度对待匹配的图像进行匹配）
    // 函数用给定的一组 TEMPLATE 在一个旋转角范围内对一张给定的图像进行匹配，得
    // 到最佳匹配位置以及对应的旋转角和 TEMPLATE 的索引。其中 tplimages 里的每
    // 一张图片的 ROI 子图的长和宽必须相等 
    __host__ int                   // 返回值：函数是否正确执行，若函数正确执
                                   // 行，返回 NO_ERROR
    imageMatch(
            Image *matchimage,     // 待匹配的图像
            MatchRes *matchres,    // 匹配后得到的结果，包括最佳匹配位置，最佳
                                   // 匹配对应的旋转角以及对应的 TEMPLATE 的索引
            DirectedRect *dirrect  // 局部异常检查得到的最小有向四边形，若传入
                                   // 时为空，则不进行局部异常检查
    );
};

#endif

