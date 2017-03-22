// ImageDrawer.h
// 创建人：于玉龙
//
// 绘制几何构建（Image Drawer）
// 功能说明：在给定的图像上绘制相应的标记，如直线、圆等。
//
// 修订历史：
// 2013年03月18日（于玉龙）
//     初始版本。
// 2013年03月19日（于玉龙）
//     修正了设定着色模式时的一处 Bug。实现了绘制直线的算法。
//     补全了目前代码所需要的注释。修正了代码中一处潜在的 Bug。
// 2013年03月20日（于玉龙）
//     完成了连续直线绘制的函数。
// 2013年03月21日（于玉龙）
//     修正了直线绘制过程中的一个 Bug。
// 2013年05月26日（张鹤）
//     完成了椭圆绘制的函数。


#ifndef __IMAGEDRAWER_H__
#define __IMAGEDRAWER_H__

#include "ErrorCode.h"
#include "Image.h"
#include "CoordiSet.h"


// 宏：IDRAW_CM_STATIC_COLOR
// 这是一种着色模式，用来表示当前绘图时，前景色采用算法 CLASS 成员变量中所指定
// 的颜色（所谓颜色，就是指亮度值）
#define IDRAW_CM_STATIC_COLOR   0

// 宏：IDRAW_CM_DYNAMIC_COLOR
// 这是一种着色模式，用来表示当前绘图时，前景色采用参数中某些特定元素代表的颜
// 色（所谓颜色，就是指亮度值）
#define IDRAW_CM_DYNAMIC_COLOR  1

// 宏：IDRAW_TRANSPARENT
// 透明色
#define IDRAW_TRANSPARENT      -1

// 类：ImageDrawer（构建绘图器）
// 继承自：无
// 在给定的图像上绘制相应的标记，如直线、圆等。
class ImageDrawer {

protected:

    // 成员变量：colorMode（着色模式）
    // 设定着色模式，在绘制某些构建时需要从不同的颜色源来选择前景色，该成员规定
    // 了绘图时如何选择颜色源。
    int colorMode;

    // 成员变量：lineColor（前景色）
    // 规定了使用静态着色模式时的前景色，可以定义为透明色。
    int lineColor;

    // 成员变量：brushColor（背景色）
    // 规定了绘图时使用的背景色，可以为透明色。
    int brushColor;

public:

    // 构造函数：ImageDrawer
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    ImageDrawer()
    {
        // 使用默认值为类的各个成员变量赋值。
        colorMode = IDRAW_CM_STATIC_COLOR;  // 着色类型默认设为静态着色。
        lineColor = 255;                    // 直线前景色默认设为白色。
        brushColor = IDRAW_TRANSPARENT;     // 背景色默认设为透明色。
    }

    // 构造函数：ImageDrawer
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    ImageDrawer(
            int clrmode,     // 着色模式
            int lineclr,     // 直线前景色
            int brushclr     // 背景色
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        colorMode = IDRAW_CM_STATIC_COLOR;  // 着色类型默认设为静态着色。
        lineColor = 255;
        brushColor = IDRAW_TRANSPARENT;     // 背景色默认设为透明色。

        // 根据参数列表中的值设定成员变量的初值
        setColorMode(clrmode);
        setLineColor(lineclr);
        setBrushColor(brushclr);
    }

    // 成员方法：getColorMode（获取着色模式）
    // 获取成员变量 colorMode 的值。
    __host__ __device__ int  // 返回值：成员变量 colorMode 的值。
    getColorMode() const
    {
        // 返回成员变量 colorMode 的值。
        return this->colorMode;
    }

    // 成员方法：setColorMode（设置着色模式）
    // 设置成员变量 colorMode 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回 
                             // NO_ERROR。
    setColorMode(
            int clrmode      // 新的着色模式
    ) {
        // 检查参数的合法性。
        if (clrmode != IDRAW_CM_STATIC_COLOR &&
            clrmode != IDRAW_CM_DYNAMIC_COLOR)
            return INVALID_DATA;

        // 设置新的着色模式。
        this->colorMode = clrmode;

        // 处理完毕，返回。
        return NO_ERROR;
    }

    // 成员方法：getLineColor（获取直线前景色）
    // 获取成员变量 lineColor 的值。
    __host__ __device__ int  // 返回值：成员变量 lineColor 的值。
    getLineColor() const
    {
        // 返回成员变量 lineColor 的值。
        return this->lineColor;
    }

    // 成员方法：setLineColor（设置直线前景色）
    // 设置成员变量 lineColor 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回 
                             // NO_ERROR。
    setLineColor(
            int lineclr      // 新的直线前景色
    ) {
        // 检查参数的合法性。
        if (lineclr != IDRAW_TRANSPARENT && (lineclr < 0 || lineclr > 255))
            return INVALID_DATA;

        // 设置新的直线前景色。
        this->lineColor = lineclr;

        // 处理完毕，返回。
        return NO_ERROR;
    }

    // 成员方法：getBrushColor（获取背景色）
    // 获取成员变量 brushColor 的值。
    __host__ __device__ int  // 返回值：成员变量 brushColor 的值。
    getBrushColor() const
    {
        // 返回成员变量 brushColor 的值。
        return this->brushColor;
    }

    // 成员方法：setBrushColor（设置背景色）
    // 设置成员变量 brushColor 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回 
                             // NO_ERROR。
    setBrushColor(
            int brushclr      // 新的背景色
    ) {
        // 检查参数的合法性。
        if (brushclr != IDRAW_TRANSPARENT && (brushclr < 0 || brushclr > 255))
            return INVALID_DATA;

        // 设置新的背景色。
        this->brushColor = brushclr;

        // 处理完毕，返回。
        return NO_ERROR;
    }

    // Host 成员方法：brushAllImage（涂满整幅图像）
    // 将图像中所有的像素都涂成背景色。
    __host__ int        // 返回值：函数是否正确执行，若函数正确执行，返回 
                        // NO_ERROR。
    brushAllImage(
            Image *img  // 待背涂色的图像。
    );

    // Host 成员方法：drawLines（绘制直线）
    // 在图像上绘制直线。如果当前使用的着色模式是静态模式，则直线使用 CLASS 内
    // 定义的前景色；如果是动态模式，则使用点集中左侧点的附属数据所表示的颜色
    // 值。背景色在该函数中不起作用。
    __host__ int            // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR。
    drawLines(
            Image *img,     // 待绘制直线的图像
            CoordiSet *cst  // 坐标点集，每两个相邻点绘制一条直线。
    );

    // Host 成员方法：drawTrace（绘制连续直线）
    // 在图像上绘制连续直线最后形成一个折线。可以选择是绘制成收尾相接的环形，还
    // 是收尾不相接的蛇形。
    __host__ int                 // 返回值：函数是否正确执行，若函数正确执行，
                                 // 返回 NO_ERROR。
    drawTrace(
            Image *img,          // 待绘制直线的图像
            CoordiSet *cst,      // 坐标点击，每两点绘制出一条连续的折线
            bool circled = true  // 是否将坐标集的最后一个点和第一个点连起来。
                                 // 若该值为 true，则最后绘制出一个多边形，否则
                                 // 绘制出一条蛇形折线。
    );

    // Host 成员方法：drawEllipse（绘制椭圆）
    // 在图像上绘制椭圆。如果当前使用的着色模式是静态模式，则直线使用 CLASS 内
    // 定义的前景色；如果是动态模式，则使用点集中左侧点的附属数据所表示的颜色
    // 值。背景色在该函数中不起作用。
    __host__ int            // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR。
    drawEllipse(
            Image *img,     // 待绘制椭圆的图像
            CoordiSet *cst  // 坐标点集（椭圆的外界矩形的左上角和右下角的坐标）
    );
};

#endif

