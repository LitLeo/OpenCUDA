// LinearEnhancement.h
// 创建人：邱孝兵
//
// 图像增强（Image Enhancement）
// 功能说明：对于输入图像的每个像素根据传入的函数指针进行变化，实现
//           线性增强，像素的变换是从 unsigned char 到 unsigned char 。
//
// 修订历史：
// 2012年09月11日（邱孝兵）
//    初始版本。
// 2012年09月14日（邱孝兵）
//    调整格式（缩进，空格等），增加 outplace 函数，将
//    函数指针类型转换为类类型，添加图像增强类
// 2012年09月15日（邱孝兵）
//     经证实设备端不能支持函数指针，函数模板以及虚类继承
//     所以简化设计放弃 Enhancement，重新定义线性增强类：LinearEnhancement
// 2012年09月21日（邱孝兵）
//     修改格式问题，在增强类中添加像素转置表
// 2012年10月24日（邱孝兵） 
//     将设备端函数修改为内联模式，即声明实现放在一起

#ifndef __LINEARENHANCEMENT_H__ 
#define __LINEARENHANCEMENT_H__ 
 
#include "Image.h" 
#include "ErrorCode.h" 

// 宏：PIXEL_TABLE_SIZE
// 定义了像素表的默认大小
#define PIXEL_TABLE_SIZE 256
 
 
// 类：LinearEnhancement（图像线性增强）
// 继承自：无
// 根据提供的四个参数，确定一个线性增强函数,
// 提供多种分段线性图像增强的操作。
class LinearEnhancement { 
 
protected: 
 
    // 成员变量：left
    // 增强函数的左折点的横坐标
    unsigned char left;
 
    // 成员变量：right
    // 增强函数的右折点的横坐标
    unsigned char right;
 
    // 成员变量：bottom
    // 增强函数的左折点的纵坐标
    unsigned char bottom;
 
    // 成员变量：top
    // 增强函数的右折点的纵坐标
    unsigned char top;

    // 成员变量：pixelTable
    // 该增强类的像素转置表，通过查这个表可以得到每个像素
    // 在新图像中的新的像素值。该变量不提供 seter。
    unsigned char pixelTable[PIXEL_TABLE_SIZE];
    
    // 成员函数：init （初始化各成员变量）
    // 将所有成员变量初始化为默认值。
    __host__ __device__ void 
    init()
    {
        left = 0;
        right = 255;
        bottom = 0;
        top = 255;
    
        // 计算像素转置表
        calPixelTable();
    }
    
    // 成员函数：calPixelTable （计算像素转置表）
    // 根据四个成员变量，计算像素转置表
    __host__ __device__ void
    calPixelTable()
    {
        // 计算中间斜线部分的斜率。
        float slope = (top - bottom) * 1.0 / (right - left);
    
        // 根据当前 i 值来确定转置表中的值，
        // 如果 i < left ，则 pixelTable[i] = bottom
        // 如果 i > right ，则 pixelTable[i] = top
        // 否则，(i, pixelTable[i]) 为根据 (left, bottom) 
        // 和 (right, top) 确定的直线上的点   
        for ( int i = 0; i < PIXEL_TABLE_SIZE; i++){
            if (i < left)
                pixelTable[i] = bottom;
            else if (i > right)
                pixelTable[i] = top;
            else
            pixelTable[i] = top - (int)(slope * (right - i));    
        }
    }
 
public:    
 
    // 构造函数：EnhancementOpt
    // 无参数版本的构造函数，成员变量初始化为默认值。
    __host__ __device__ 
    LinearEnhancement()
    {
        init();
    }

    // 构造函数：EnhancementOpt
    // 有参数版本的构造函数，根据需要给定各个参数
    __host__ __device__ 
    LinearEnhancement( 
            unsigned char left,    // 增强函数的左折点的横坐标
            unsigned char right,   // 增强函数的右折点的横坐标
            unsigned char bottom,  // 增强函数的左折点的纵坐标
            unsigned char top      // 增强函数的右折点的纵坐标
    ) {          
        // 首先初始化为默认值
        init();                                    
    
        // 初始化各个参数
        setLeft(left);
        setRight(right);
        setBottom(bottom);
        setTop(top);
    }          
 
    // 成员方法：getLeft （获取 left 的值）
    // 获取成员变量 left 的值
    __host__ __device__  unsigned char  // 返回值：成员变量 left 的值
    getLeft() const
    {
        // 返回 left 成员变量的值
        return this->left;
    }
   
    // 成员方法：setLeft （设置 left 的值）
    // 设置成员变量 left 的值。
    __host__ __device__ int     // 返回值：函数是否正确执行，若函数正确执
                                // 行，返回 NO_ERROR。
    setLeft(
            unsigned char left  // 设定新 left 值
    ) {
        if (left < right)
        {
            // 将 left 成员变量赋成新值
            this->left = left;

            // 重新计算像素转置表
            calPixelTable();
        }
        return NO_ERROR;
    }
    
    // 成员方法：getRight （获取 right 的值）
    // 获取成员变量 right 的值
    __host__ __device__  unsigned char  // 返回值：成员变量 right 的值
    getRight() const
    {
        // 返回 right 成员变量的值
        return this->right;
    }
    
    // 成员方法：setRight （设置 right 的值）
    // 设置成员变量 right 的值。
    __host__ __device__ int      // 返回值：函数是否正确执行，若函数正确执
                                 // 行，返回 NO_ERROR。
    setRight(
            unsigned char right  // 设定新的 right 值
    ) {
        // 合法性检查
        if (right > left)
        {
            // 将 right 成员变量赋成新值
            this->right = right;
      
            // 重新计算像素转置表
            calPixelTable();
        }

        return NO_ERROR;
    }
    
    // 成员方法：setLeftRight （同时设置 left 和 right 值）
    // 设置成员变量 left 和 right 的值
    __host__ __device__ int      // 返回值：函数是否正确执行，若函数正确执
                                 // 行，返回 NO_ERROR。
    setLeftRight(
            unsigned char left,  // 新的 left 值
            unsigned char right  // 新的 right 值
    ) {
        // 合法性检查
        if (right > left)
        {
            // 将 left 和 right 成员变量赋成新值
            this->left = left;
            this->right = right;
      
            // 重新计算像素转置表
            calPixelTable();
        }

        return NO_ERROR;
    }
	
    // 成员方法：getBottom （获取 bottom 的值）
    // 获取成员变量 bottom 的值
    __host__ __device__  unsigned char  // 返回值：成员变量 bottom 的值
    getBottom() const
    {
        // 返回 bottom 成员变量的值
        return this->bottom;
    }
    
    // 成员方法：setBottom （设置 bottom 的值）
    // 设置成员变量 bottom 的值。
    __host__ __device__ int       // 返回值：函数是否正确执行，若函数正确执
                                  // 行，返回 NO_ERROR。
    setBottom(
            unsigned char bottom  // 设定新的 bottom 值
    ) {
        // 将 bottom 成员变量赋成新值
        this->bottom = bottom;

        // 重新计算像素转置表
        calPixelTable();

        return NO_ERROR;
    }

    // 成员方法：getTop （获取 top 的值）
    // 获取成员变量 top 的值
    __host__ __device__  unsigned char  // 返回值：成员变量 top 的值
    getTop() const
    {
        // 返回 top 成员变量的值
        return this->top;
    }
    
    // 成员方法：setTop （设置 top 的值）
    // 设置成员变量 top 的值。
    __host__ __device__ int    // 返回值：函数是否正确执行，若函数正确执
                               // 行，返回 NO_ERROR。
    setTop(
            unsigned char top  // 设定新的 top 值
    ) {
        // 将 top 成员变量赋成新值
        this->top = top;

        // 重新计算像素转置表
        calPixelTable();

        return NO_ERROR;
    }

    // 成员方法：getPixelTable （获取 pixelTable 的值）
    // 获取成员变量 pixelTable 的值
    __host__ __device__  unsigned char *  // 返回值：成员变量 pixelTable 的值
    getPixelTable()
    {
        // 返回 pixelTable 成员变量的值
        return this->pixelTable;
    }
    
    // Host成员方法：enhance （图像增强）
    // 根据指定的参数对图像进行增强（ In-place 版本），
    // 在函数中根据四个成员变量的值构造增强函数，对于
    // 输入图像的每个像素进行增强，并保存到输入图像中
    __host__ int             // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    linearEnhance( 
            Image *inoutimg  // 输入输出图像
    );
 
    // Host成员方法：enhance（图像增强）
    // 根据指定的参数对图像进行增强（Out-place 版本）
    // 在函数中根据四个成员变量的值构造增强函数，对于
    // 输入图像的每个像素进行增强，并保存到输出图像中
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    linearEnhance(
            Image *inImg,  // 输入图像
            Image *outImg  // 输出图像
    );
};

#endif
 
