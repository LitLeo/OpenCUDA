// SalientImg.h
// 创建人：冯振
//
// 计算显著图（SalientImg）
// 功能说明：实现计算图像的显著图。输出的结果保存在Image型的指针。
//
// 修订历史：
// 2013年10月15日（冯振）
//     初始版本。
// 2013年12月5日（冯振）
//     规范了编码格式

#ifndef __SALIENTIMG_H__
#define __SALIENTIMG_H__

#include "Image.h"


// 类：SalientImg（显著图）
// 继承自：无
// 功能说明：实现计算图像的显著图。
class SalientImg {

protected:
    // int wsizea;        // 计算advanSalientImg时使用的窗口邻域大小。
    // int wsizeg;        // 计算gaussSmoothImg时使用的窗口邻域大小。
    unsigned char gapth;  // 计算advanSalientImg时使用的阈值。
   
public:

    // 构造函数：SalientImg
    // 无参数版本的构造函数，设置默认参数。
    __host__ __device__
    SalientImg()
    {
        this->gapth = 55;
    }

    // 构造函数：SalientImg
    // 有参数版本。
    __host__ __device__
    SalientImg(unsigned char gapth)
    {
        this->gapth = gapth;
    }

    // 成员方法：getGapTh（获取计算advanSalientImg时使用的阈值）
    // 获取成员变量 gapth 的值。
    __host__ __device__ unsigned char  // 返回值：成员变量 gapth 的值
    getGapTh() const
    {
        // 返回 gapth 成员变量的值。
        return this->gapth;
    }

    // 成员方法：setGapTh（设置计算advanSalientImg时使用的阈值）
    // 设置成员变量 wsizeg 的值。
    __host__ __device__ unsigned char  // 返回值：函数是否正确执行，若函数正确执
                                       // 行，返回 NO_ERROR。
    setGapTh(
            unsigned char gapth        // 设定计算advanSalientImg时使用的阈值。
    ) { 
        // 将 gapth 成员变量赋成新值
        this->gapth = gapth;

        return NO_ERROR;
    }

    // Host 成员方法：advanSalient（根据输入图像originalImg计算advanSalientImg）
    __host__ int                // 返回值：函数正确执行，则返回 NO_ERROR。
    advanSalient(
        Image* originalimg,     // 输入的原始图像。
        Image* advansalientimg  // 输出的advanSalientImg图像。
    );
    //Host 成员方法：advanSalient（根据输入图像originalImg计算advanSalientImg）
	__host__ int				// 返回值：函数正确执行，则返回 NO_ERROR。
	advanSalientMultiGPU(
		Image* originalimg,		// 输入的原始图像。
		Image* sdvansalientimg	// 输出的advanSalientImg图像。
	);	
	
    // Host 成员函数：makeSalientImg（计算 SalientImg）
    __host__ int                 // 返回值：函数正确执行，则返回 NO_ERROR。
    makeSalientImg(
        Image* advansalientimg,  // 输入的预显著图像。
        Image* gausssmoothimg,   // 输入的高斯平滑图像。
        Image* salientimg        // 输出的显著图想。
    );
	//Host 成员函数：makeSalientImg（计算 SalientImg）
	__host__ int				 // 返回值：函数正确执行，则返回 NO_ERROR。
	makeSalientImgMultiGPU(
		Image* advansalientimg,	 // 输入的预显著图像。
		Image* gausssmoothimg,	 // 输入的高斯平滑图像。
		Image* salientimg		 // 输出的显著图想。
	)
};

#endif

