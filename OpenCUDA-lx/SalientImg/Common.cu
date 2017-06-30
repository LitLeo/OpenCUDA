 #include "Image.h"
 #include "Common.h"
 __host__ ImageCuda* cudaCommon::imageCut(
        Image *img
    )
 {
    int deviceCount=1;
    cudaGetDeviceCount(&deviceCount);

    //有两个输入一个输出，所以需要3个ImgaeCuda数组
    ImageCuda* deviceimg;
    deviceimg = new ImageCuda[deviceCount];
    ImageCuda* imgCud = IMAGE_CUDA(img); 
    size_t pitch = imgCud->pitchBytes; 
    for(int i=0;i<deviceCount;++i){
        // cudaSetDevice(i);

        //为成员变量赋值
        deviceimg[i].imgMeta.width = img->width;
        deviceimg[i].imgMeta.height = (img->height)/deviceCount;
        deviceimg[i].pitchBytes = pitch;
        deviceimg[i].imgMeta.imgData = img->imgData + i*deviceimg[i].imgMeta.width*
                                 deviceimg[i].imgMeta.height;
    }
    return deviceimg;
}

// warmup 函数，用于计时时 warmup GPU，实际是一个 vector 相加
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
        C[i] = A[i] + B[i];
}
void warmup()
{
    int numElements = 1024;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);

    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);

    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

}