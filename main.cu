#include <stdio.h>
#include "LinearFilter.h"
#include "Image.h"
#include "Template.h"
#include "timer.h"

int main(int argc, char *argv[]){
	Image *inImg,*outImg;
	ImageBasicOp::newImage(&inImg);
    if(argc!=1)
        ImageBasicOp::readFromFile(argv[1],inImg);
    else
        ImageBasicOp::readFromFile("src.bmp",inImg);
        
    ImageBasicOp::newImage(&outImg);
    ImageBasicOp::makeAtHost(outImg,inImg->width,inImg->height);
    LinearFilter LF;

    cudaSetDevice(0);
    warmup();
    cudaSetDevice(1);
    warmup();
    cudaSetDevice(0);

    int choise=0;
    while(choise!=1&&choise!=2){
        printf("Single or multi GPU?\n1.Single GPU\n2.Multi GPU\n");
        scanf("%d",&choise);
    }

    if(choise == 1){
        StartTimer();
        LF.linearFilter(inImg,outImg);
        printf("  GPU Processing time: %f (ms)\n\n", GetTimer());
    }
    else{
        StartTimer();
        LF.linearFilterMultiGPU(inImg,outImg);
        printf("  Multi GPU Processing time: %f (ms)\n\n", GetTimer());
    }

    ImageBasicOp::writeToFile("dest.bmp",outImg);
    ImageBasicOp::deleteImage(outImg);
    return 0;
}