#include <stdio.h>
#include "LinearFilter.h"
#include "Image.h"
#include "Template.h"
#include "timer.h"

int main(int argc, char *argv[]){
	Image *inImg,*outImg1,*outImg2;
	ImageBasicOp::newImage(&inImg);
    if(argc!=1)
        ImageBasicOp::readFromFile(argv[1],inImg);
    else
        ImageBasicOp::readFromFile("src.bmp",inImg);
        
    ImageBasicOp::newImage(&outImg1);
    ImageBasicOp::makeAtHost(outImg1,inImg->width,inImg->height);
    ImageBasicOp::newImage(&outImg2);
    ImageBasicOp::makeAtHost(outImg2,inImg->width,inImg->height);
    LinearFilter LF;

    warmup();
    StartTimer();
    LF.linearFilter(inImg,outImg1);
    printf("  GPU Processing time: %f (ms)\n\n", GetTimer());

    StartTimer();
    LF.linearFilterMultiGPU(inImg,outImg2);
    printf("  Multi GPU Processing time: %f (ms)\n\n", GetTimer());

    ImageBasicOp::writeToFile("dest1.bmp",outImg1);
    ImageBasicOp::writeToFile("dest2.bmp",outImg2);
    ImageBasicOp::deleteImage(outImg1);
    ImageBasicOp::deleteImage(outImg2);
    return 0;
}