#include <iostream>
using namespace std;

#include "LinearFilter.h"
#include "Image.h"
#include "Template.h"

int main(){
	Image *inImg,*outImg;
	ImageBasicOp::newImage(&inImg);
    ImageBasicOp::readFromFile("test.bmp",inImg);
    ImageBasicOp::newImage(&outImg);
    ImageBasicOp::makeAtHost(outImg,inImg->width,inImg->height);
    LinearFilter LF;
    LF.linearFilter(inImg,outImg);
    ImageBasicOp::writeToFile("dest.bmp",outImg);
    ImageBasicOp::deleteImage(outImg);
    return 0;
}