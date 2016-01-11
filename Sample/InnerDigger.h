// InnerDigger.h
// 创建者：雷健龙
//
// 区域抠心（InnerDigger）
// 功能说明：输入灰度图，图像中有一个 ROI 区域，对区域中的每个点作如下判断：
//           如果该点的八领域内或者四领域内所有的点都是白色，则置为 0；否则保留
//           原值。

#ifndef __INNERDIGGER_H__
#define __INNERDIGGER_H__

#include"Image.h"
#include"ErrorCode.h"


// 宏：CR_FOUR_AREAS
// 宏定义，用于识别四领域区域抠心
#define CR_FOUR_AREAS    1

// 宏：CR_EIGHT_AREAS
// 宏定义，用于识别八领域区域抠心
#define CR_EIGHT_AREAS    2


// 类：InnerDigger
// 继承自：无
// 该类包含了对于图像的区域抠心操作，如果该点的八领域内或者四领域内所有的点都是
// 白色，则置为 0；否则保留原值其中含有两种操作，分别为四邻域的区域抠心和八领域
// 的区域抠心
class InnerDigger {

protected:

    // 成员变量：mode（区域抠心模式）
    // 在进行区域扣心前需设置它来设置当前区域抠心的模式
    int mode;  

public:

    // 构造函数：InnerDigger
    // 无参数构造函数
    __host__ __device__
    InnerDigger()
    {
        // 成员变量 mode 默认 CR_EIGHT_AREAS
        this->mode = CR_EIGHT_AREAS;
    }

    // 构造函数：InnerDigger
    // 有参数构造函数
    __host__ __device__
    InnerDigger(
            int m  // 设置成员变量 mode
    ) {
        // 成员变量 mode 默认 CR_EIGHT_AREAS
        this->mode = CR_EIGHT_AREAS;

        // 根据参数设定 mode 的值
        setMode(m);
    }

    // 成员方法：getMode（获取成员变量 mode）
    // 获取成员变量 mode 的值
    __host__ __device__ int  // 返回值：当前设置的区域抠心模式
    getMode() const
    {
        // 返回成员变量 mode 的值
        return this->mode;
    }

    // 成员方法：setMode（设置成员变量 mode）
    // 设置区域抠心模式，识别两个输入：CR_FOUR_AREAS 表示四领域抠心，
    // CR_EIGHT_AREAS 表示八领域扣心
    __host__ __device__ int  // 返回值：若函数正确执行，返回 NO_ERROR，否则返回
                             // 相应的错误码
    setMode(
            int m            // 设置成员变量 mode 的值
    ) {
         // 非 CR_FOUR_AREAS 和 CR_EIGHT_AREAS 的输入值返回 INVALID_DATA
        if (m != CR_FOUR_AREAS && m != CR_EIGHT_AREAS) 
            return INVALID_DATA;
        
        // 根据参数设定 mode 的值
        this->mode = m;
        // 返回 NO_ERROR
        return NO_ERROR;
    }

    // Host 成员方法：innerDigger（区域抠心）
    // 针对一个灰度图，对其 ROI 区域实行区域抠心操作，并由参数 outimg 返回，参数
    // model 识别操作模式，输入参数 CR_FOUR_AREAS 表示四领域抠心，输入参数 
    // CR_EIGHT_AREAS 表示八领域抠心
    __host__ int           // 返回值：若函数正确执行，返回 NO_ERROR，否则返回
                           // 相应的错误码
    innerDigger(    
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
};

#endif

