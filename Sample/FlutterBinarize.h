// FlutterBinarze.h
//
// 抖动二值化（FlutterBinarze）
// 功能说明：用像素点疏密表示像素值大小，对灰度图像进行二值抖动，得到二值图像。

#ifndef __FLUTTERBINARIZE_H__
#define __FLUTTERBINARIZE_H__

#include "Image.h"
#include "Template.h"
#include "CoordiSet.h"
#include "ErrorCode.h"

// 宏：MAX_PIXEL
// 定义了像素值的范围大小。
#define MAX_PIXEL  256

// 宏：TEMPLATE_SIZE
// 定义了模板的范围大小。
#define TEMPLATE_SIZE  16

// 宏：METHOD_ONE
// 定义了第一种算法的执行代码。
#define METHOD_ONE  1

// 宏：METHOD_TWO
// 定义了第二种算法的执行代码。
#define METHOD_TWO  2

// 类：FlutterBinarize
// 继承自：无
// 用像素点疏密表示像素值大小，对灰度图像进行二值抖动，得到二值图像。如果像素值
// 越小黑色像素点越密集（白色像素点越稀疏），如果像素值越大黑色像素点越稀疏（白
// 色像素点越密集）。
class FlutterBinarize {

protected:

    // 成员变量：selectmethod（抖动二值化的算法选择变量）
    // selectmethod = METHOD_ONE 代表执行第一种分层处理像素点算法；
    // selectmethod = METHOD_TWO 代表执行第二种根据生成模板处理像素点算法。
    int selectmethod;

    // 成员变量：packrange（每层像素值的范围）
    // 像素值分组的标准，范围是 [1, 256]。
    int packrange;

    // Host 成员方法：initializeLayerTpl（初始化模板处理）
    // 对图像进行初始化操作。包括计算图像的压缩直方图操作和制作模板操作。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确执行
                                   // 返回 NO_ERROR。
    initializeLayerTpl(
            Image *inimg,          // 输入图像
            int groupnum,          // 分组的数量
            unsigned int *&cnt,    // 每层坐标点的在模板中的区域范围
            Template *&coordinate  // 输入图像
    );

    // Host 成员方法：initializeTemplate（初始化模板处理）
    // 对图像进行初始化操作。生成 [0, 255] 范围内的每个像素值对应的模板。
    __host__ int                   // 返回值：函数是否正确执行，若函数正确执行
                                   // 返回 NO_ERROR。
    initializeTemplate();

public:

    // 构造函数：FlutterBinarize
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    FlutterBinarize()
    {
        // 使用默认值为类成员 packrange 赋值。
        this->packrange = 8;              // 每层像素值的范围为 8。

        // 使用默认值为类成员 selectmethod 赋值。
        this->selectmethod = METHOD_TWO;  // 抖动二值化的默认算法执行代码为
                                          // METHOD_TWO。
    }

    // 构造函数：FlutterBinarize
    // 有参数版本的构造函数，根据需要设定每层像素值的范围。
    __host__ __device__
    FlutterBinarize(
            int packrange,
            int selectmethod
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->packrange = 8;              // 每层像素值的范围为 8。

        // 根据参数列表中的值设定成员变量 packrange 的初值。
        setPackrange(packrange);

        this->selectmethod = METHOD_TWO;  // 抖动二值化的默认算法执行代码为
                                          // METHOD_TWO。

        // 根据参数列表中的值设定成员变量 selectmethod 的初值。
        setSelectmethod(selectmethod);
    }

    // 成员方法：getPackrange（获取每层像素值的数量）
    // 获取成员变量 packrange 的值。
    __host__ __device__ int  // 返回值：成员变量 packrange 的值
    getPackrange() const
    {
        // 返回 packrange 成员变量的值。
        return this->packrange;
    }

    // 成员方法：setPackrange（设置每层像素值的范围）
    // 设置成员变量 packrange 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行
                             // 返回 NO_ERROR。
    setPackrange(
            int packrange    // 设定新的每层像素值的范围值。
    ) {
        // 判断输入 packrange 是否合法。
        if (packrange < 1 || packrange > 256)
            return INVALID_DATA;

        // 将 packrange 成员变量赋成新值。
        this->packrange = packrange;

        return NO_ERROR;
    }

    // 成员方法：getSelectmethod（获取每层像素值的数量）
    // 获取成员变量 selectmethod 的值。
    __host__ __device__ int  // 返回值：成员变量 selectmethod 的值
    getSelectmethod() const
    {
        // 返回 selectmethod 成员变量的值。
        return this->selectmethod;
    }

    // 成员方法：setSelectmethod（设置每层像素值的范围）
    // 设置成员变量 selectmethod 的值。
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执行
                                // 返回 NO_ERROR。
    setSelectmethod(
            int selectmethod    // 设定新的每层像素值的范围值。
    ) {
        // 判断输入 selectmethod 是否合法。
        if (selectmethod != METHOD_ONE && selectmethod != METHOD_TWO)
            return INVALID_DATA;

        // 将 selectmethod 成员变量赋成新值。
        this->selectmethod = selectmethod;

        return NO_ERROR;
    }

    // Host 成员方法：flutterBinarize（抖动二值化处理）
    // 对图像进行抖动二值化处理。这是一个 Out-Place 形式的。算法一：根据对像素
    // 点划分出的不同层定义不同的扩散能力（扩散理解为把周围点像素值设为 0），
    // 分层处理，从而实现图像的抖动二值化操作。算法二：根据像素点值对应的模板
    // 处理当前像素点，从而实现图像的抖动二值化操作。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回 
                           // NO_ERROR。
    flutterBinarize(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );

    // Host 成员方法：flutterBinarize（抖动二值化处理）
    // 对图像进行抖动二值化处理。这是一个 In-Place 形式的。算法一：根据对像素
    // 点划分出的不同层定义不同的扩散能力（扩散理解为把周围点像素值设为 0），
    // 分层处理，从而实现图像的抖动二值化操作。算法二：根据像素点值对应的模板
    // 处理当前像素点，从而实现图像的抖动二值化操作。
    __host__ int        // 返回值：函数是否正确执行，若函数正确执行返回
                        // NO_ERROR。
    flutterBinarize(
            Image *img  // 输入图像
    );
};

#endif

