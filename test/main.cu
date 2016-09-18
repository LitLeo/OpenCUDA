#include <iostream>
using namespace std;

#include "Mosaic.h"

#define ROUND_NUM 1

int main()
{
    Mosaic ms;
    /*ms.setMossize(512);*/

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    
    Image *inimg;
    ImageBasicOp::newImage(&inimg);
    
    ImageBasicOp::readFromFile("hist_in.bmp", inimg);
    inimg->roiX1 = 400;
    inimg->roiY1 = 400;
    inimg->roiX2 = 600;
    inimg->roiY2 = 600;


    Image *outimg;
    ImageBasicOp::newImage(&outimg);
	
    cout << "Mosaic inimg:" << endl;
    for (int i = 0; i <= ROUND_NUM; i++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    
        cudaEventRecord(start, 0);

        ms.mosaic(inimg, outimg);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        cout << elapsedTime << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

    }

    ImageBasicOp::copyToHost(outimg);
    ImageBasicOp::writeToFile("out.bmp", outimg);

    ImageBasicOp::deleteImage(inimg);
    ImageBasicOp::deleteImage(outimg);

    return 0;
}

