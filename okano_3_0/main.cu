#include <iostream>
using namespace std;

#include "Image.h"
#include "Template.h"
#include "CombineImage.h"

#define ROUND_NUM 1
#define INIMG_CNT 3
char *infilename[] = { "okano01.bmp", "okano02.bmp", "hist_in.bmp" };

int main()
{
    CombineImage ci;

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    
    Image *inimg[INIMG_CNT];
    for (int i = 0; i < INIMG_CNT; i++) {
        ImageBasicOp::newImage(&inimg[i]);
        ImageBasicOp::readFromFile(infilename[i], inimg[i]);
    }

    Image *outimg;
    ImageBasicOp::newImage(&outimg);
    //ImageBasicOp::makeAtCurrentDevice(outimg, 648, 482);
	
    cout << "AA" << endl;
    for (int i = 0; i <= ROUND_NUM; i++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        //cout << "Test start!" << endl;
    
        cudaEventRecord(start, 0);

        ci.combineImageMax(inimg, INIMG_CNT, outimg);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        //cout << cudaGetErrorString(cudaGetLastError()) << endl;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        cout << elapsedTime << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

    }

    ImageBasicOp::copyToHost(outimg);
    ImageBasicOp::writeToFile("out.bmp", outimg);

    for (int i = 0; i < INIMG_CNT; i++) 
        ImageBasicOp::deleteImage(inimg[i]);
    ImageBasicOp::deleteImage(outimg);


    return 0;
}

