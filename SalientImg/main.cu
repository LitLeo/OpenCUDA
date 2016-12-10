#include <iostream>
#include <stdio.h>
using namespace std;

#include "common.h"
#include "Image.h"
#include "SalientImg.h"
#include "timer.h"

int main(int argc, char *argv[])
{

    char *inimgfilename = "2048_2048.bmp";

    if(argc != 1) {
        inimgfilename = argv[1];
    }
    
    cout << "inimg : " << inimgfilename << endl;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout<<"deviceCount = "<<deviceCount<<endl;

    Image *frimg;
    ImageBasicOp::newImage(&frimg);
    ImageBasicOp::readFromFile(inimgfilename, frimg);

    Image *baimg;
    ImageBasicOp::newImage(&baimg);
    ImageBasicOp::readFromFile(inimgfilename, baimg);

    Image *outimg;
    ImageBasicOp::newImage(&outimg);
    ImageBasicOp::readFromFile(inimgfilename,outimg);
    SalientImg T;

    // 预热 http://blog.csdn.net/litdaguang/article/details/50520549这里要用循环
    for(unsigned i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        warmup();
    }
    if(argc > 2) {
        StartTimer();
        T.advanSalientMultiGPU(frimg,outimg);
        printf(" Multi GPU Processing time: %f (ms)\n\n", GetTimer());
    } else {
        StartTimer();
        T.advanSalient(frimg,outimg);
        printf("  GPU Processing time: %f (ms)\n\n", GetTimer());
    }
	if(argc > 2) {
        StartTimer();
        T.makeSalientImgMultiGPU(frimg,baimg,outimg);
        printf(" Multi GPU Processing time: %f (ms)\n\n", GetTimer());
    } else {
        StartTimer();
        T.makeSalientImg(frimg,baimg,outimg);
        printf("  GPU Processing time: %f (ms)\n\n", GetTimer());
    }
    ImageBasicOp::deleteImage(frimg);
    ImageBasicOp::deleteImage(baimg);
    ImageBasicOp::deleteImage(outimg);

    return 0;
}

