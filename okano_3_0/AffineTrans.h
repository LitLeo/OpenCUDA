// AffineTrans.h
// 创建人：于玉龙
//
// 旋转仿射变换（Affine Transform）
// 功能说明：实现图像的旋转变换，旋转变换不改变原有图像的尺寸，旋转后，超出输出
//           图片的部分结果会被裁剪调。该变换具有三种不同的实现：（1）基于纹理
//           内存的硬件插值功能实现的旋转变换；（2）使用软件实现的 Fanczos 插值
//           实现的旋转变换；（3）封装 NPP 支持库实现的旋转变换。
//
// 修订历史：

#ifndef __AFFINETRANS_H__
#define __AFFINETRANS_H__

#include "ErrorCode.h"
#include "Image.h"
#include "FanczosIpl.h"


// 宏：AFFINE_SOFT_IPL
// 用于设置 AffineTrans 类中的 impType 成员变量，告知类的实例选用软件实现的 
// Fanczos 插值实现旋转变换。
#define AFFINE_SOFT_IPL    1

// 宏：AFFINE_HARD_IPL
// 用于设置 AffineTrans 类中的 impType 成员变量，告知类的实例选用基于纹理内存的
// 硬件插值功能实现的旋转变换。
#define AFFINE_HARD_IPL    2

// 宏：AFFINE_NVIDIA_LIB
// 用于设置 AffineTrans 类中的 impType 成员变量，告知类的实例选用封装 NPP 支持
// 库实现的旋转变换。
#define AFFINE_NVIDIA_LIB  3


// 类：AffineTrans（旋转仿射变换）
// 继承自：无
// 实现图像的旋转变换，旋转变换不改变原有图像的尺寸，旋转后，超出输出图片的部分
// 结果会被裁剪调。该变换具有三种不同的实现：（1）基于纹理内存的硬件插值功能实
// 现的旋转变换；（2）使用软件实现的 Fanczos 插值实现的旋转变换；（3）封装 NPP
// 支持库实现的旋转变换。
class AffineTrans {

protected:
   
    // 成员变量：impType（实现类型）
    // 设定三种实现类型中的一种，在调用仿射变换函数的时候，使用对应的实现方式。
    int impType;
    
    // 成员变量：x 和 y（基准坐标）
    // 用于设定基准坐标，在 rotateCenter 成员方法中，它作为旋转变换的中心点；在
    // rotateShift 成员方法中，它作为旋转前的平移量。
    int x, y;
    
    // 成员变量：alpha（旋转角度）
    // 指定旋转变换的角度，单位是“度”（°）。
    float alpha;
    
    // 成员变量：stateFlag（调用状态)
    // 输出参数，其值为上一次调用 rotateCenter 或 rotateShift 方法的返回的错误
    // 码，如果实例还没有调用过这两个成员方法，则默认值为 NO_ERROR。
    int stateFlag;
    
public:

    // 构造函数：AffineTrans
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    AffineTrans()
    {
        // 使用默认值为类的各个成员变量赋值。
        impType = AFFINE_HARD_IPL;  // 实现类型的默认值为“硬件插值”
        x = y = 0;                  // 基准坐标的默认值为 (0, 0)
        alpha = 0.0f;               // 旋转角度的默认值为 0°
        stateFlag = NO_ERROR;       // 调用状态的默认值为“无错误”
    }
    
    // 构造函数：AffineTrans
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    AffineTrans (
            int imptype,   // 实现类型（具体解释见成员变量）
            int x, int y,  // 基准坐标（具体解释见成员变量）
            float alpha    // 旋转角度（具体解释见成员变量）
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->impType = AFFINE_HARD_IPL;  // 实现类型的默认值为“硬件插值”
        this->x = this->y = 0;            // 基准坐标的默认值为 (0, 0)
        this->alpha = 0.0f;               // 旋转角度的默认值为 0°
        this->stateFlag = NO_ERROR;       // 调用状态的默认值为“无错误”

        // 根据参数列表中的值设定成员变量的初值
        this->setImpType(imptype);
        this->setX(x);
        this->setY(y);
        this->setAlpha(alpha);

        // 将调用状态赋值为默认的“无错误”
        this->stateFlag = NO_ERROR;
    }

    // 成员方法：getImpType（读取实现类型）
    // 读取 impType 成员变量的值。
    __host__ __device__ int  // 返回值：当前 impType 成员变量的值。
    getImpType() const
    {
        // 返回 impType 成员变量的值。
        return this->impType;
    }
    
    // 成员方法：setImpType（设置实现类型）
    // 设置 impType 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setImpType(
            int imptype      // 新的实现类型
    ) {
        // 检查输入参数是否合法
        if (imptype != AFFINE_HARD_IPL && imptype != AFFINE_SOFT_IPL &&
            imptype != AFFINE_NVIDIA_LIB)
            return INVALID_DATA;

        // 将 impType 成员变量赋成新值
        this->impType = imptype;
        return NO_ERROR;
    }
    
    // 成员方法：getX（读取基准坐标 X 分量）
    // 读取 x 成员变量的值。
    __host__ __device__ int  // 返回值：当前 x 成员变量的值。
    getX() const
    {
        // 返回 x 成员变量的值。
        return this->x;
    }
    
    // 成员方法：setX（设置基准坐标 X 分量）
    // 设置 x 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setX(
            int x            // 新的基准坐标 X 分量
    ) {
        // 将 x 成员变量赋成新值
        this->x = x;
        return NO_ERROR;
    }
    
    // 成员方法：getY（读取基准坐标 Y 分量）
    // 读取 y 成员变量的值。
    __host__ __device__ int  // 返回值：当前 y 成员变量的值。
    getY() const
    {
        // 返回 y 成员变量的值。
        return this->y;
    }
    
    // 成员方法：setY（设置基准坐标 Y 分量）
    // 设置 y 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setY(
            int y            // 新的基准坐标 Y 分量
    ) {
        // 将 y 成员变量赋成新值
        this->y = y;
        return NO_ERROR;
    }
    
    // 成员方法：getAlpha（读取旋转角度）
    // 读取 alpha 成员变量的值。
    __host__ __device__ float  // 返回值：当前 alpha 成员变量的值。
    getAlpha() const
    {
        // 返回 alpha 成员变量的值。
        return this->alpha;
    }
    
    // 成员方法：setAlpha（设置旋转角度）
    // 设置 alpha 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setAlpha(
            float alpha      // 新的旋转角度
    ) {
        // 为了方便计算，首先将符号和数字剥离，这样 alpha 为正数。
        bool insign = (alpha >= 0);
        if (!insign)
            alpha = -alpha;

        // 角度是以 360°为一个轮回，因此，去除多余的 360°，使角度归一化到
        // ±360°之间。
        float purealpha = alpha - floor(alpha / 360.0f) * 360.0f;
        if (!insign)
            purealpha = -purealpha;

        // 由于 ±360°之间有 720°，因此进一步归一化，使角度归一化到 ±360°之
        // 间。这样旋转后角的边落入一、二象限时，角度为正；边落入三、四象限时，
        // 角度为负。
        if (purealpha > 180.0f) {
            purealpha -= 360.0f;
        } else if (purealpha <= -180.0f) {
            purealpha += 360.0f;
        }

        // 将 alpha 成员变量赋值成新值
        this->alpha = purealpha;
        return NO_ERROR;
    }
    
    // 成员方法：getStateFlag（读取调用状态）
    // 读取 stateFlag 成员变量的值。
    __host__ __device__ int  // 返回值：当前 stateFlag 成员变量的值。
    getStateFlag() const
    {
        // 返回 stateFlag 成员变量的值。
        return this->stateFlag;
    }

    // 成员方法：resetStateFlag（恢复调用状态）
    // 清除前次调用记录在 stateFlag 中的错误码，使其恢复到 NO_ERROR。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    resetStateFlag()
    {
        // 将 stateFlag 恢复为默认值 NO_ERROR。
        this->stateFlag = NO_ERROR;
        return NO_ERROR;
    }
    
    // Host 成员方法：rotateCenter（基于中心的旋转）
    // 将基准坐标作为旋转中心的旋转仿射变换。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    rotateCenter(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );

    // Host 成员方法：rotateShift（基于平移的旋转）
    // 将基准坐标作为旋转前的平移向量的旋转仿射变换，在该成员方法中，旋转的中心
    // 点为输入图像的中心点。
    __host__ int           // 返回值：函数是否正确执行，若函数正确执行，返回
                           // NO_ERROR。
    rotateShift(
            Image *inimg,  // 输入图像
            Image *outimg  // 输出图像
    );
};


#endif
