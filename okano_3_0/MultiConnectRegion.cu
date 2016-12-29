#include "MultiConnectRegion.h"
#include "ImageFilter.h"
#include "ConnectRegion.h"
#include <iostream>
using namespace std;

// 成员方法：multiConnectRegion (多图连通区域)
int MultiConnectRegion::multiConnectRegion(Image *inimg, Image ***outimg)
{
    // 检查输入图像是否为 NULL，如果为 NULL 直接报错返回。
    if(inimg == NULL)
        return NULL_POINTER;

    // 局部变量，错误码
    int errcode;

    // 局部变量，多图中间结果
    Image **midoutimg;

    // 定义 ImageFilter 对象，用于图像过滤
    ImageFilter IF(thresholdArr, thresholdNum);

    // 定义ConnectRegion 对象，用于求连通区域
    ConnectRegion CR(threshold, maxArea, minArea);

    // 对输入图像进行阈值过滤
    errcode = IF.calImageFilterOpt(inimg, &midoutimg);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 为输出图像开辟空间
    *outimg = new Image *[thresholdNum];
    for(int i = 0; i < thresholdNum; i++) {
        errcode = ImageBasicOp::newImage(&((*outimg)[i]));
        if (errcode != NO_ERROR) {
            return errcode;
        }

        errcode = ImageBasicOp::makeAtCurrentDevice((*outimg)[i], inimg->width,
                                                     inimg->height);
        if (errcode != NO_ERROR) {
            return errcode;
        }
    }

    // 对每一幅图像做连通区域
    for(int i = 0; i < thresholdNum; i++) {
        errcode = CR.connectRegion(midoutimg[i], (*outimg)[i]);
        if (errcode != NO_ERROR) {
            return errcode;
        }
    }

    // 释放中间图像
    for(int i = 0;i < thresholdNum; i++)
    {
        ImageBasicOp::deleteImage(midoutimg[i]);
    }

    return NO_ERROR;
}
                                                                               