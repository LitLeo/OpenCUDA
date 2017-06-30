// SalientImg.cu
// 实现图像显著图算法

#include <iostream>
#include "SalientImg.h"
#include "ErrorCode.h"

using namespace std;



// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32 
#define DEF_BLOCK_Y  8

// 定义窗口的大小
#define HSIZE 3
#define WSIZE HSIZE * 2 + 1

// Kernel 函数：_advantSalientKer（计算 advanSalientImg）
static __global__ void                  // Kernel 函数无返回值。
_advanSalientKer(
            ImageCuda originalimg,      // 输入的原始图像。
            ImageCuda advansalientimg,  // 输出的 advansalient 图像。
            unsigned char gapth         // 阈值
){
    // 处理当前线程对应的图像点(c,r)。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (c >= originalimg.imgMeta.width || r >= originalimg.imgMeta.height)
        return;

    int inidx = r * originalimg.pitchBytes + c;    

    unsigned char sortedwindow[WSIZE * WSIZE];

    if (c >= HSIZE + 1 && r >= HSIZE + 1 && 
        c <= originalimg.imgMeta.width - HSIZE &&
        r <= originalimg.imgMeta.height - HSIZE)
    {    
        int i, j;
        int ind = 0;
        for (i = -HSIZE; i < HSIZE; i++) {
            for (j = -HSIZE; j < HSIZE; j++) {
                sortedwindow[ind] = 
                    originalimg.imgMeta.imgData[inidx + 
                    j * originalimg.imgMeta.width + i];
                ind ++;
            }
        }
        
        // 插入排序
        unsigned char temp;
        for (i = 1; i < WSIZE * WSIZE; i++)
        { 
            temp = sortedwindow[i];     //temp为要插入的元素
            j = i - 1; 

            while (j >= 0 && temp < sortedwindow[j])
            {   //从a[i-1]开始找比a[i]小的数，同时把数组元素向后移
                sortedwindow[j + 1] = sortedwindow[j]; 
                j --;
            }
            sortedwindow[j + 1] = temp; //插入
        }

        
        // 计算 high vale mean 和 low vlaue meam。
        unsigned char hvm = 0, lvm = 0;
        int num = WSIZE * WSIZE * 0.1; 
        for(i = 0; i < num; i++)
        {
            hvm += sortedwindow[WSIZE * WSIZE - 1 - i];
            lvm += sortedwindow[i];
        }
        hvm /= num;
        lvm /= num;

        unsigned char v0 = originalimg.imgMeta.imgData[inidx];
        unsigned char dv = hvm - lvm;
        unsigned char dv2 = (unsigned char)(dv/2.0f + 0.5f);
        if ( dv < gapth ) {
            advansalientimg.imgMeta.imgData[inidx] = dv2;
        }

        else {
            if (abs (v0- hvm) < abs (v0- lvm)) {
                advansalientimg.imgMeta.imgData[inidx] = (abs (v0 - hvm) < 
                    abs (v0 - dv2) ? hvm : dv2);
            }
            else {
                advansalientimg.imgMeta.imgData[inidx] = (abs (v0 - lvm) < 
                    abs (v0 - dv2) ? lvm : dv2);
            }
        }

    } else {
        advansalientimg.imgMeta.imgData[inidx] = 
            originalimg.imgMeta.imgData[inidx];
    }

    
}


// Host 成员方法：advanSalient（根据输入图像originalImg计算advanSalientImg）
__host__ int SalientImg::advanSalientMultiGPU(Image* originalimg, 
                                      Image* advansalientimg)
{
    if (originalimg == NULL)
        return NULL_POINTER;
    
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
	// 局部变量，错误码
    //int errcode;  
	 // CUDA 调用返回的错误码。
	cudaError_t cuerrcode; 
    int deviceCount = 1;
	cudaGetDeviceCount(&deviceCount);
    // 在host 端进行切割
    ImageCuda *h_originalimgCud = new ImageCuda[deviceCount];
    ImageCuda *h_advansalientimgCud = new ImageCuda[deviceCount];
    ImageBasicOp::cutImage(originalimg, h_originalimgCud, deviceCount);
    ImageBasicOp::cutImage(advansalientimg, h_advansalientimgCud, deviceCount);

    // device 端定义，申请内存
    Image **deviceoriginalimg = new Image*[deviceCount];
    Image **deviceadvansalientimg = new Image*[deviceCount];

    ImageCuda **deviceoriginalimgCud = new ImageCuda*[deviceCount];
    ImageCuda **deviceadvansalientimgCud = new ImageCuda*[deviceCount];

    cudaStream_t *stream = new cudaStream_t[deviceCount];
    // cout<<devicefrimg[0].imgMeta.width<<'@'<<devicefrimg[0].imgMeta.height<<endl;
    // cout<<devicebaimg[0].imgMeta.width<<'@'<<devicebaimg[0].imgMeta.height<<endl;
    // cout<<deviceoutimg[0].imgMeta.width<<'@'<<deviceoutimg[0].imgMeta.height<<endl;

    // ImageBasicOp::writeToFile("fr11.bmp", &devicefrimg[0].imgMeta);
    // ImageBasicOp::writeToFile("fr22.bmp", &devicefrimg[1].imgMeta);

    // ImageBasicOp::writeToFile("ba11.bmp", &devicebaimg[0].imgMeta);
    // ImageBasicOp::writeToFile("ba22.bmp", &devicebaimg[1].imgMeta);
    // cout<<"pitch = "<<devicefrimg[0].pitchBytes<<endl;
    // cout<<"pitch = "<<devicefrimg[1].pitchBytes<<endl;
	for(unsigned i = 0; i < deviceCount; ++i) {
        ImageBasicOp::newImage(&deviceoriginalimg[i]);
        ImageBasicOp::newImage(&deviceadvansalientimg[i]);

        cudaSetDevice(i);
        ImageBasicOp::makeAtCurrentDevice(deviceoriginalimg[i],
                                          h_originalimgCud[i].imgMeta.width,
                                          h_originalimgCud[i].imgMeta.height);
        deviceoriginalimgCud[i] = IMAGE_CUDA(deviceoriginalimg[i]);
        ImageBasicOp::makeAtCurrentDevice(deviceadvansalientimg[i],
                                          h_advansalientimgCud[i].imgMeta.width,
                                          h_advansalientimgCud[i].imgMeta.height);

        deviceadvansalientimgCud[i] = IMAGE_CUDA(deviceadvansalientimg[i]);
        cudaStreamCreate(&stream[i]);
    }

    /*for(int i = 0; i < deviceCount; ++i){
        cudaSetDevice(i);
		
	    ///申请空间和数据拷贝
        cuerrcode = cudaMallocPitch((void **)(&deviceoriginalimg[i].d_imgData), &deviceoriginalimg[i].pitchBytes, 
                                    deviceoriginalimg[i].imgMeta.width * sizeof (unsigned char), 
                                    deviceoriginalimg[i].imgMeta.height);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }
        cuerrcode = cudaMallocPitch((void **)(&deviceadvansalientimg[i].d_imgData), &deviceadvansalientimg[i].pitchBytes, 
                                    deviceadvansalientimg[i].imgMeta.width * sizeof (unsigned char), 
                                    deviceadvansalientimg[i].imgMeta.height);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }
    }*/
    for(int i = 0; i < deviceCount; ++i){
        cudaSetDevice(i);
		
        cuerrcode = cudaMemcpy2DAsync (deviceoriginalimgCud[i]->imgMeta.imgData, deviceoriginalimgCud[i]->pitchBytes,
                                h_originalimgCud[i].imgMeta.imgData, h_originalimgCud[i].imgMeta.width,
                                h_originalimgCud[i].imgMeta.width * sizeof (unsigned char),
                                h_originalimgCud[i].imgMeta.height,
                                cudaMemcpyHostToDevice, stream[i]);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        cuerrcode = cudaMemcpy2DAsync (deviceadvansalientimgCud[i]->imgMeta.imgData, deviceadvansalientimgCud[i]->pitchBytes,
                                h_advansalientimgCud[i].imgMeta.imgData, h_advansalientimgCud[i].imgMeta.width,
                                h_advansalientimgCud[i].imgMeta.width * sizeof (unsigned char),
                                h_advansalientimgCud[i].imgMeta.height,
                                cudaMemcpyHostToDevice, stream[i]);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 gridsize, blocksize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (deviceoriginalimgCud[i]->imgMeta.width + blocksize.x - 1) / blocksize.x;
        gridsize.y = (deviceoriginalimgCud[i]->imgMeta.height + blocksize.y * 4 - 1) / 
                    (blocksize.y * 4);
        // 调用核函数
        _advanSalientKer<<<gridsize, blocksize, 0, stream[i]>>>(*deviceoriginalimgCud[i],
                                            *deviceadvansalientimgCud[i], gapth);
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

        // cudaFree(deviceoutimg[i].d_imgData);
		
    }
	for(int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        //Wait for all operations to finish
        cudaStreamSynchronize(stream[i]);
        
        cudaStreamDestroy(stream[i]);
        cudaFree(deviceoriginalimg[i]->imgData);
        cudaFree(deviceadvansalientimg[i]->imgData);
    }


    // 处理完毕，退出。
    return NO_ERROR;
}


// Host 成员方法：advanSalient（根据输入图像originalImg计算advanSalientImg）
__host__ int SalientImg::advanSalient(Image* originalimg, 
                                      Image* advansalientimg)
{
    if (originalimg == NULL)
        return NULL_POINTER;
    
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(originalimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(originalimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(advansalientimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(advansalientimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 调用核函数。
    _advanSalientKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud, gapth);
    if (cudaGetLastError() != cudaSuccess) {
        return CUDA_ERROR;
    }

    return NO_ERROR;
}

// Kernel 函数：_makeSalientImgKer（计算 SalientImg）
static __global__ void                  // Kernel 函数无返回值。
_makeSalientImgKer(
            ImageCuda advansalientimg,  // 输入的预显著图像。
            ImageCuda gausssmoothimg,   // 输入的高斯平滑图像。
            ImageCuda salientimg        // 输出的显著图像。
){
    // 处理当前线程对应的图像点(c,r)。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int inidx = r * advansalientimg.pitchBytes + c;    

    int v = (advansalientimg.imgMeta.imgData[inidx] - 
            gausssmoothimg.imgMeta.imgData[inidx]) * 0.55 + 255 / 2.0f;

    if (v < 0) {
        salientimg.imgMeta.imgData[inidx] = 0;
    } else if (v > 250) {
        salientimg.imgMeta.imgData[inidx] = 250;
    } else {
        salientimg.imgMeta.imgData[inidx] = v;
    }
    
}

// Host 成员函数：makeSalientImg（计算 SalientImg）
__host__ int SalientImg::makeSalientImgMultiGPU(
            Image* advansalientimg,  // 输入的预显著图像。
            Image* gausssmoothimg,   // 输入的高斯平滑图像。
            Image* salientimg        // 输出的显著图像。
) {
	 /// 检查输入、输出图像是否为 NULL，如果为 NULL 直接报错返回。
    if (advansalientimg == NULL || gausssmoothimg == NULL || salientimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 图像准备内存空间，以便盛放数据。
    // int errcode;  // 局部变量，错误码
	
	 // CUDA 调用返回的错误码。
    cudaError_t cuerrcode; 
	
    int deviceCount = 1;
    cudaGetDeviceCount(&deviceCount);
    // 在host 端进行切割
    ImageCuda *h_advansalientimgCud = new ImageCuda[deviceCount];
    ImageCuda *h_gausssmoothimgCud = new ImageCuda[deviceCount];
    ImageCuda *h_salientimgCud = new ImageCuda[deviceCount];
    ImageBasicOp::cutImage(advansalientimg, h_advansalientimgCud, deviceCount);
    ImageBasicOp::cutImage(gausssmoothimg, h_gausssmoothimgCud, deviceCount);
    ImageBasicOp::cutImage(salientimg, h_salientimgCud, deviceCount);

    // device 端定义，申请内存
    Image **deviceadvansalientimg = new Image*[deviceCount];
    Image **devicegausssmoothimg = new Image*[deviceCount];
    Image **devicesalientimg = new Image*[deviceCount];

    ImageCuda **deviceadvansalientimgCud= new ImageCuda*[deviceCount];
    ImageCuda **devicegausssmoothimgCud = new ImageCuda*[deviceCount];
    ImageCuda **devicesalientimgCud = new ImageCuda*[deviceCount];
	
	//开启流
	cudaStream_t *stream = new cudaStream_t[deviceCount];
    // cout<<devicefrimg[0].imgMeta.width<<'@'<<devicefrimg[0].imgMeta.height<<endl;
    // cout<<devicebaimg[0].imgMeta.width<<'@'<<devicebaimg[0].imgMeta.height<<endl;
    // cout<<deviceoutimg[0].imgMeta.width<<'@'<<deviceoutimg[0].imgMeta.height<<endl;

    // ImageBasicOp::writeToFile("fr11.bmp", &devicefrimg[0].imgMeta);
    // ImageBasicOp::writeToFile("fr22.bmp", &devicefrimg[1].imgMeta);

    // ImageBasicOp::writeToFile("ba11.bmp", &devicebaimg[0].imgMeta);
    // ImageBasicOp::writeToFile("ba22.bmp", &devicebaimg[1].imgMeta);
    // cout<<"pitch = "<<devicefrimg[0].pitchBytes<<endl;
    // cout<<"pitch = "<<devicefrimg[1].pitchBytes<<endl;
	for(unsigned i = 0; i < deviceCount; ++i) {
        ImageBasicOp::newImage(&deviceadvansalientimg[i]);
        ImageBasicOp::newImage(&devicegausssmoothimg[i]);
        ImageBasicOp::newImage(&devicesalientimg[i]);

        cudaSetDevice(i);
        ImageBasicOp::makeAtCurrentDevice(deviceadvansalientimg[i],
                                          h_advansalientimgCud[i].imgMeta.width,
                                          h_advansalientimgCud[i].imgMeta.height);
        deviceadvansalientimgCud[i] = IMAGE_CUDA(deviceadvansalientimg[i]);
        ImageBasicOp::makeAtCurrentDevice(devicegausssmoothimg[i],
                                          h_gausssmoothimgCud[i].imgMeta.width,
                                          h_gausssmoothimgCud[i].imgMeta.height);
        devicegausssmoothimgCud[i] = IMAGE_CUDA(devicegausssmoothimg[i]);
        ImageBasicOp::makeAtCurrentDevice(devicesalientimg[i],
                                          h_salientimgCud[i].imgMeta.width,
                                          h_salientimgCud[i].imgMeta.height);

        devicesalientimgCud[i] = IMAGE_CUDA(devicesalientimg[i]);
        cudaStreamCreate(&stream[i]);
    }

  /*  for( unsigned i = 0; i < deviceCount; ++i){
        cudaSetDevice(i);

        ///申请空间和数据拷贝
        cuerrcode = cudaMallocPitch((void **)(&deviceadvansalientimg[i].d_imgData), &deviceadvansalientimg[i].pitchBytes, 
                                    deviceadvansalientimg[i].imgMeta.width * sizeof (unsigned char), 
                                    deviceadvansalientimg[i].imgMeta.height);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }
        cuerrcode = cudaMallocPitch((void **)(&devicegausssmoothimg[i].d_imgData), &devicegausssmoothimg[i].pitchBytes, 
                                    devicegausssmoothimg[i].imgMeta.width * sizeof (unsigned char), 
                                    devicegausssmoothimg[i].imgMeta.height);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        cuerrcode = cudaMallocPitch((void **)(&devicesalientimg[i].d_imgData), &devicesalientimg[i].pitchBytes, 
                                    devicesalientimg[i].imgMeta.width * sizeof (unsigned char), 
                                    devicesalientimg[i].imgMeta.height);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }
    }
*/
    for(int i = 0; i < deviceCount; ++i){
        cudaSetDevice(i);
        cuerrcode = cudaMemcpy2DAsync (deviceadvansalientimgCud[i]->imgMeta.imgData, deviceadvansalientimgCud[i]->pitchBytes,
                                h_advansalientimgCud[i].imgMeta.imgData, h_advansalientimgCud[i].imgMeta.width,
                                h_advansalientimgCud[i].imgMeta.width * sizeof (unsigned char),
                                h_advansalientimgCud[i].imgMeta.height,
                                cudaMemcpyHostToDevice, stream[i]);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        cuerrcode = cudaMemcpy2DAsync (devicegausssmoothimgCud[i]->imgMeta.imgData, devicegausssmoothimgCud[i]->pitchBytes,
                                h_gausssmoothimgCud[i].imgMeta.imgData, h_gausssmoothimgCud[i].imgMeta.width,
                                h_gausssmoothimgCud[i].imgMeta.width * sizeof (unsigned char),
                                h_gausssmoothimgCud[i].imgMeta.height,
                                cudaMemcpyHostToDevice, stream[i]);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

       
        cuerrcode = cudaMemcpy2DAsync (devicesalientimgCud[i]->imgMeta.imgData, devicesalientimgCud[i]->pitchBytes,
                                h_salientimgCud[i].imgMeta.imgData, h_salientimgCud[i].imgMeta.width,
                                h_salientimgCud[i].imgMeta.width * sizeof (unsigned char),
                                h_salientimgCud[i].imgMeta.height,
                                cudaMemcpyHostToDevice, stream[i]);
        if (cuerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
        dim3 gridsize, blocksize;
        blocksize.x = DEF_BLOCK_X;
        blocksize.y = DEF_BLOCK_Y;
        gridsize.x = (deviceadvansalientimgCud[i]->imgMeta.width + blocksize.x - 1) / blocksize.x;
        gridsize.y = (deviceadvansalientimgCud[i]->imgMeta.height + blocksize.y * 4 - 1) / 
                    (blocksize.y * 4);
        // 调用核函数，根据阈值进行贴图处理。
        _makeSalientImgKer<<<gridsize, blocksize, 0, stream[i]>>>(*deviceadvansalientimgCud[i], *devicegausssmoothimgCud[i], 
                                            *devicesalientimgCud[i]);
        // 若调用 CUDA 出错返回错误代码
        if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

        
        // cudaFree(deviceoutimg[i].d_imgData);

    }
	
    for(int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        //Wait for all operations to finish
        cudaStreamSynchronize(stream[i]);
        
        cudaStreamDestroy(stream[i]);
        cudaFree(deviceadvansalientimg[i]->imgData);
        cudaFree(devicegausssmoothimg[i]->imgData);
        cudaFree(devicesalientimg[i]->imgData);
    }


    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 成员函数：makeSalientImg（计算 SalientImg）
__host__ int SalientImg::makeSalientImg(
            Image* advansalientimg,  // 输入的预显著图像。
            Image* gausssmoothimg,   // 输入的高斯平滑图像。
            Image* salientimg        // 输出的显著图想。
) {
    if (advansalientimg == NULL)
    return NULL_POINTER;

    if (gausssmoothimg == NULL)
    return NULL_POINTER;
    
    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为输
    // 入和输出图像准备内存空间，以便盛放数据。
    int errcode;  // 局部变量，错误码

    // 将输入图像1拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(advansalientimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像1的 ROI 子图像。
    ImageCuda insubimgCud1;
    errcode = ImageBasicOp::roiSubImage(advansalientimg, &insubimgCud1);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输入图像2拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(gausssmoothimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输入图像2的 ROI 子图像。
    ImageCuda insubimgCud2;
    errcode = ImageBasicOp::roiSubImage(gausssmoothimg, &insubimgCud2);
    if (errcode != NO_ERROR)
        return errcode;


    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(salientimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(salientimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud1.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud1.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 调用核函数。
    _makeSalientImgKer<<<gridsize, blocksize>>>(insubimgCud1, insubimgCud2, 
                                                outsubimgCud);
    if (cudaGetLastError() != cudaSuccess) {
        return CUDA_ERROR;
    }

    return NO_ERROR;
}

