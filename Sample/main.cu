#include <iostream>
#include "stdio.h"
//#include "HistogramSpec.h"//xiugai
//#include "Histogram.h"
#include "GaussianSmoothImage.h"
using namespace std;
int main()
{
    Image *inimg, *outimg , *outimg1;  

    ImageBasicOp::newImage(&inimg);
    ImageBasicOp::readFromFile("temp.bmp",inimg);
    
    ImageBasicOp::newImage(&outimg);
    ImageBasicOp::makeAtHost(outimg, inimg->width, inimg->height);
    ImageBasicOp::newImage(&outimg1);
    ImageBasicOp::makeAtHost(outimg1, inimg->width, inimg->height);
   // ImageBasicOp::readFromFile("lighting.bmp",outimg);
    GaussSmoothImage his;
    
    printf( "%d",his.gaussSmoothImageMultiGPU(inimg,600,680,0,10,7,outimg));
//   printf( "%d",his.gaussSmoothImage(inimg,500,500,150,0,7,outimg1));
    ImageBasicOp::writeToFile("out.bmp", outimg);
    ImageBasicOp::writeToFile("out1.bmp", outimg1);
    ImageBasicOp::deleteImage(outimg);
    return 0;

}

