// FillUp.h
//
// 像素处理（FillUp）
// 功能说明：检查一个像素的邻域，若其邻域同时存在 l 像素和 v 像素，
//           当 v 像素的个数大于等于某一值时，将所有的 l 像素置为
//           v 像素。

#ifndef __FILLUP_H__
#define __FILLUP_H__

#include "Image.h"
#include "Template.h"
#include "ErrorCode.h"

// 类：FillUp（像素处理类）
// 继承自：无
// 检查一个像素的邻域，若其邻域同时存在 l 像素和 v 像素， 当 v 像素的个数
// 大于等于某一值时，将所有的 l 像素置为 v 像素。
class FillUp {

protected:

    // 成员变量：tpl (模板指针)
    // 通过它来找到图像的像素
    Template *tpl;

    // 成员变量：r (数量比例系数)
    // 领域内 v 的数量比例系数，大于等于 0.1 且小于等于 0.8。
    float r;

    // 成员变量：maxw (最大模板的尺寸)
    // 尺寸要大于等于 3。
    int maxw;

    // 成员变量：l (主要像素)
    // l 的取值大于等于 0 且小于等于 255。
    unsigned char l; 

    // 成员变量：v (替换像素)
    // v 的取值大于等于 0 且小于等于 255。
    unsigned char v;  

public:

    // 构造函数：FillUp
    // 传递模板指针，如果不传，则默认为空。
    __host__ 
    FillUp(
            Template *tp = NULL  // 处理像素操作需要使用到模板，默认为空。
    );

    // 构造函数：FillUp
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__      
    FillUp(
            Template *tp,     // 处理像素操作需要使用到模板，默认为空。
	    unsigned char l,  // 主要像素。
            unsigned char v,  // 替换像素。 
	    int maxw,         // 最大模板的尺寸。
	    float r           // 数量比例系数。
    );
    
    // 成员方法：getTemplate
    // 获取模板指针，如果模板指针和默认模板指针相同，则返回空。
    __host__ Template*   // 返回值：如果模板和默认模板指针相同，则返
                         // 回空，否则返回模板指针。
    getTemplate() const; 
  
    // 成员方法：setTemplate
    // 设置模板指针，如果参数 tp 为空，这使用默认的模板。
    __host__  int          // 返回值：若函数正确执行，返回 NO_ERROR。
    setTemplate(
            Template *tp  // 处理像素操作需要使用的模板。
    );

    // 成员方法：getL（获取 l 像素的值）
    // 获取成员变量 l 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 l 的值。
    getL() const
    {
        // 返回 l 成员变量的值。
        return this->l;
    }  

    // 成员方法：setL（设置 l 像素的值）
    // 设置成员变量 l 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setL(
            unsigned char l  // 设定新的 l 的值。
    ) {
        // 将 l 成员变量赋成新值
        this->l = l;

        return NO_ERROR;
    }

    // 成员方法：getV（获取 v 像素的值）
    // 获取成员变量 v 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 v 的值。
    getV() const 
    {
        // 返回 v 成员变量的值。
        return this->v;
    } 

    // 成员方法：setV（设置 v 像素的值）
    // 设置成员变量 v 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setV(
            unsigned char v  // 设定新的 v 的值。
    ) {
        // 将 v 成员变量赋成新值
        this->v = v;

        return NO_ERROR;
    }

    // 成员方法：getR（获取比例系数 r 的值）
    // 获取成员变量 r 的值。
    __host__ __device__ float  // 返回值：成员变量 r 的值。
    getR() const 
    {
        // 返回 r 成员变量的值。
        return this->r;
    }  

    // 成员方法：setR（设置比例系数 r 的值）
    // 设置成员变量 r 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setR(
            float r          // 设定新的 r 的值。
    ) {
        // 对参数进行检测。
        if (r < 0.1f && r > 0.8f) 
            return INVALID_DATA;
    
        // 将 r 成员变量赋成新值
        this->r = r;
   
        return NO_ERROR;
    }

    // 成员方法：getMaxw（获取 maxw 的值）
    // 获取成员变量 maxw 的值。
    __host__ __device__ int  // 返回值：成员变量 maxw 的值。
    getMaxw() const 
    {
        // 返回 maxw 成员变量的值。
        return this->maxw;
    }  

    // 成员方法：setMaxw（设置 maxw 的值）
    // 设置成员变量 maxw 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setMaxw(
            int maxw         // 设定新的 maxw 的值。
    ) {
        // 对参数进行检测。
        if (maxw < 3) 
	    return INVALID_DATA;
	 
        // 将 maxw 成员变量赋成新值
        this->maxw = maxw;
    
        return NO_ERROR;
    }
	
    // Host 成员方法：fillUp
    // 检查一个像素的邻域，若其邻域同时存在 l 像素和 v 像素，当 v 像素的
    // 个数大于等于某一值时，将所有的 l 像素置为 v 像素。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    fillUp(
            Image *inimg,  // 输入图像。
	    Image *outimg  // 输出图像。
    );
	
    // Host 成员方法：fillUpAdv
    // 用最大模板的尺寸 maxw 来初始化图像，如果满足 maxw 的值大于 3，就不断
    // 的调用 fillUp 函数,然后将 maxw 的值减半，不断进行处理，直到 maxw 的值
    // 减为 3。最后，对所有 l 像素的 8 个领域进行检查，如果它的 8 个领
    // 域当中有 5 个或 5 个以上的 v 的像素值，就将 v 的像素值赋给 l。 
    __host__ int            // 返回值：函数是否正确执行，若函数正确执行，返回
                            // NO_ERROR。
    fillUpAdv(
            Image *inimg,   // 输入图像。
	    Image *outimg,  // 输出图像。
	    int *stateflag  // 迭代次数。
    );
};

#endif

