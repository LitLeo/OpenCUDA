// TorusSegmentation.cu
// 实现圆环二分类

#include "TorusSegmentation.h"

// 宏：DEF_BLOCK_1D
// 定义了默认的 1D 线程块的尺寸。
#define DEF_BLOCK_1D    256 

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了二维结构的并行线程块默认的尺寸。
#define DEF_BLOCK_X    32
#define DEF_BLOCK_Y     8

// 宏：DEF_BLACK 和 DEF_WHITE
// 定义了黑色和白色的像素值。
#define DEF_BLACK      0
#define DEF_WHITE    255


// 核函数：_initLblMatToZeroKer（初始化标记矩阵为全 0）
// 在设备端初始化标记矩阵为全零。
static __global__ void          // Kernel 函数无返回值。
_initLblMatToZeroKer(
        ImageCuda outlblimgcud  // 坐标集位置标记矩阵 
);

// 核函数：_initLblMatKer（根据坐标集初始化标记矩阵）
// 在设备端初始化标记矩阵，坐标集内部的点初始化标记为 1。
static __global__ void          // Kernel 函数无返回值。
_initLblMatKer(        
        CoordiSet incoordiset,  // 输入坐标集
        ImageCuda outlblimgcud  // 坐标集位置标记矩阵 
);

// 核函数：_torusSegLblKer（标记坐标集实现二分类）
// 在设备端通过考查当前像素邻域是否都在坐标集内，标记分类。
static __global__ void          // Kernel 函数无返回值。
_torusSegLblKer(
        ImageCuda inlblimgcud,  // 输入坐标集位置标记矩阵
        CoordiSet incoordiset,  // 输入坐标集
        TorusSegmentation ts,   // 分割操作类 
        unsigned char *outlbl   // 输出标记数组
);

// 核函数：_labelToImgKer（将分割结果反映到图像上）
// 该核函数，根据之前分割得到的标记数组，将分割结果映射到图像上去，在 
// CoordiSet 中记录了在图像中该点的位置，将对应位置的像素二值化为标记值。
static __global__ void 
_labelToImgKer(                  // Kernel 函数没有返回值。
        CoordiSet incoordiset,   // 输入坐标集
        unsigned char *inlabel,  // 输入的分类结果数组
        ImageCuda outimgcud      // 用于标记的图像
);

// 核函数：_initLblMatToZeroKer（初始化标记矩阵为全 0）
static __global__ void _initLblMatToZeroKer(        
        ImageCuda outlblimgcud)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。由于我们采用了并行度
    // 缩减的策略，令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 行
    // 上，因此，对于 r 需要进行乘 4 计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
 
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= outlblimgcud.imgMeta.width || r >= outlblimgcud.imgMeta.height) 
        return;
   
    for (int i = 0; i < 4; i++) {
        // 给当前标记矩阵标记为零。
        outlblimgcud.imgMeta.imgData[r * outlblimgcud.pitchBytes + c] = 0; 

        // 继续处理该线程中下一行同一列的点。
        r++;

        // 检查是否越界
        if (r >= outlblimgcud.imgMeta.height)
            return;
    }
}

// 核函数：_initLblMatKer（根据坐标集初始化标记矩阵）
static __global__ void _initLblMatKer(        
        CoordiSet incoordiset, ImageCuda outlblimgcud)
{
    // 计算当前 Thread 所对应的坐标集中的点的位置。
    int index  = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果当前索引超过了坐标集中的点的个数，直接返回。
    if(index >= incoordiset.count)
        return;

    // 计算该点在原图像中的位置。
    int xcrd = incoordiset.tplData[2 * index];
    int ycrd = incoordiset.tplData[2 * index + 1];

    // 将标记矩阵中对应位置的点标记为 1。
    outlblimgcud.imgMeta.imgData[ycrd * outlblimgcud.pitchBytes + xcrd] = 1;
}

// 核函数：_torusSegLblKer（标记坐标集实现二分类）
static __global__ void _torusSegLblKer(
        ImageCuda inlblimgcud, CoordiSet incoordiset, 
        TorusSegmentation ts, unsigned char *outlbl)
{
    // 计算当前 Thread 所对应的坐标集中的点的位置。
    int index  = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果当前索引超过了坐标集中的点的个数，直接返回。
    if (index >= incoordiset.count)
        return;

    // 计算该点在原图像中的位置。
    int xcrd = incoordiset.tplData[2 * index];
    int ycrd = incoordiset.tplData[2 * index + 1];

    // 获取邻域宽度。
    int neighborsize = ts.getNeighborSize();

    // 获取标记图像尺寸。
    int width = inlblimgcud.imgMeta.width;
    int height = inlblimgcud.imgMeta.height;
    int pitch = inlblimgcud.pitchBytes;

    // 如果当前像素邻域宽度超过了物理范围，直接标记为 1 类别，核函数返回。
    if (xcrd + neighborsize >= width || xcrd - neighborsize < 0 ||
        ycrd + neighborsize >= height || ycrd - neighborsize < 0) {
        outlbl[index] = 1;
        return;
    }

    // 遍历当前点的 neighborsize 邻域，发现坐标集外的点即将当前点标记为 1 类别。
    for (int i = ycrd - neighborsize; i <= ycrd + neighborsize; i++) {
        for (int j = xcrd - neighborsize; j <= xcrd + neighborsize; j++) {
            if (inlblimgcud.imgMeta.imgData[i * pitch + j] == 0) {
                outlbl[index] = 1;
                return;
            }
        }
    }

    // 其余的情况标记为 2 类别。
    outlbl[index] = 2;
}

// 核函数：_labelToImgKer（将分割结果反映到图像上）
static __global__ void _labelToImgKer(
        CoordiSet incoordiset, unsigned char *inlabel, ImageCuda outimgcud)
{
    // 计算当前 Thread 所对应的坐标集中的点的位置。
    int index  = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果当前索引超过了坐标集中的点的个数，直接返回。
    if (index >= incoordiset.count)
        return;

    // 计算该点在原图像中的位置。
    int xcrd = incoordiset.tplData[2 * index];
    int ycrd = incoordiset.tplData[2 * index + 1];

    // 获取标记图像 pitch。
    int pitch = outimgcud.pitchBytes;

    // 根据标记值设置对应图像的像素值。
    if (inlabel[index] == 1) 
        outimgcud.imgMeta.imgData[ycrd * pitch + xcrd] = DEF_WHITE;
    else
        outimgcud.imgMeta.imgData[ycrd * pitch + xcrd] = DEF_BLACK;
}


// 宏：FREE_LOCAL_MEMORY_TORUS_SEGREGATE（清理局部申请的设备端或者主机端内存）
// 该宏用于清理在 torusSegregate 过程中申请的设备端或者主机端内存空间。
#define FREE_LOCAL_MEMORY_TORUS_SEGREGATE do {    \
        if ((lblimg) != NULL)                     \
            ImageBasicOp::deleteImage((lblimg));  \
        if ((outlbldev) != NULL)                  \
            cudaFree((outlbldev));                \
    } while (0)


// Host 成员函数：torusSegregate（对圆环进行二分割）
__host__ int TorusSegmentation::torusSegregate(
        int width, int height, CoordiSet *incoordiset, unsigned char *outlbl)
{
    // 检查指针是否为空。
    if (incoordiset == NULL || outlbl == NULL) 
        return NULL_POINTER;

    // 检查参数是否合法。
    if (width <= 0 || height <= 0 || incoordiset->count <= 0)
        return INVALID_DATA;

    // 局部变量 count， 简化代码书写。
    int count = incoordiset->count;

    // 申明局部变量。
    unsigned char *outlbldev;  // 设备端标记数组
    Image *lblimg;             // 坐标集范围二维标记矩阵
    int errcode;               // 错误码

    // 将坐标集拷贝到 Device 内存中。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(incoordiset);
    if (errcode != NO_ERROR)
        return errcode;

    // 创建坐标集范围二维标记矩阵指针。
    errcode = ImageBasicOp:: newImage(&lblimg);
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE;
        return CUDA_ERROR;
    }

    // 在设备端坐标集范围二维标记矩阵指针。
    errcode = ImageBasicOp::makeAtCurrentDevice(lblimg, width, height);
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE;
        return CUDA_ERROR;
    }
    
    // 获取设备端标记矩阵。
    ImageCuda lblimgcud;  // 坐标集范围设备端标记矩阵
    errcode = ImageBasicOp::roiSubImage(lblimg, &lblimgcud);
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE;
        return CUDA_ERROR;
    }

    // 计算调用初始化矩阵的核函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y * 4 - 1) / (blocksize.y * 4);

    // 调用核函数，初始化标记矩阵为零。
    _initLblMatToZeroKer<<<gridsize, blocksize>>>(lblimgcud);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。    
    blocksize.x = DEF_BLOCK_1D;
    blocksize.y = 1;
    gridsize.x = count / blocksize.x + 1;
    gridsize.y = 1;

    // 调用核函数，使用坐标集初始化标记矩阵对应位置为 1。
    _initLblMatKer<<<gridsize, blocksize>>>(*incoordiset, lblimgcud);

    // 在设备端申请标记数组。
    errcode = cudaMalloc((void **)&outlbldev, count * sizeof(unsigned char));
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE;
        return CUDA_ERROR;
    }

    // 调用核函数，进行圆环区域二分类。
    _torusSegLblKer<<<gridsize, blocksize>>>(lblimgcud, *incoordiset, 
                                             *this, outlbldev);

    // 将标记值拷贝到主机端。
    errcode = cudaMemcpy(outlbl, outlbldev, count * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost);
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE;
        return CUDA_ERROR;
    }

    // 内存清理。   
    FREE_LOCAL_MEMORY_TORUS_SEGREGATE;
    
    return NO_ERROR;
}


// 宏：FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG（清理申请的设备端或主机端内存）
// 该宏用于清理在 torusSegregateToImg 过程中申请的设备端或者主机端内存空间。
#define FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG do {  \
        if ((outlabel) != NULL)                        \
            delete [] (outlabel);                      \
        if ((outlabeldev) != NULL)                     \
            cudaFree((outlabeldev));                   \
    }while (0)


// Host 成员函数：torusSegregateToImg（对圆环进行二分割，结果体现到图像上）
__host__ int TorusSegmentation::torusSegregateToImg(
        int width, int height, CoordiSet *incoordiset, Image *outimg)
{
    // 检查指针是否为空。
    if (incoordiset == NULL || outimg == NULL) 
        return NULL_POINTER;

    // 检查参数是否合法。
    if (width <= 0 || height <= 0 || incoordiset->count <= 0)
        return INVALID_DATA;

    // 局部变量 count， 简化代码书写。
    int count = incoordiset->count;

    // 定义局部变量。
    int errcode;                        // 错误码
    cudaError_t cuerrcode;              // CUDA 错误码
    unsigned char *outlabel = NULL;     // 主机端标记数组
    unsigned char *outlabeldev = NULL;  // 设备端标记数组
    ImageCuda insubimgCud;              // ImgCuda 对象

    // 将坐标集拷贝到 Device 内存中。
    errcode = CoordiSetBasicOp::copyToCurrentDevice(incoordiset);
    if (errcode != NO_ERROR)
        return errcode;

    // 申请主机端标记数组空间。
    outlabel = new unsigned char[count];
    if (outlabel == NULL) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG;
        return OUT_OF_MEM;
    }
    
    // 调用圆环分割 host 函数。
    errcode = torusSegregate(width, height, incoordiset, outlabel);
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG;
        return errcode;
    }

    // 申请设备端标记数组空间。
    cuerrcode = cudaMalloc((void **)&outlabeldev, 
                           sizeof(unsigned char) * count);
    if (cuerrcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG;
        return cuerrcode;
    }

    // 将标记数组拷贝到设备端。
    cuerrcode = cudaMemcpy(outlabeldev, outlabel, sizeof(unsigned char) * count,
                           cudaMemcpyHostToDevice);
    if (cuerrcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG;
        return cuerrcode;
    }

    // 将输出图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG;
        return errcode;
    }

    // 提取输入图像的 ROI 子图像。    
    errcode = ImageBasicOp::roiSubImage(outimg, &insubimgCud);
    if (errcode != NO_ERROR) {
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG;
        return errcode;
    }

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (count + blocksize.x - 1) / blocksize.x;
    gridsize.y = 1;

    // 调用核函数，将标记数组映射到图像上。
    _labelToImgKer<<<gridsize, blocksize>>>(*incoordiset, outlabeldev,
                                            insubimgCud);

    // 若调用 CUDA 出错返回错误代码。
    if (cudaGetLastError() != cudaSuccess) {
        // 释放申请的内存，防止内存泄漏。
        FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG;
        return CUDA_ERROR;
    }

    // 释放内存。
    FREE_LOCAL_MEMORY_TORUS_SEGREGATE_TO_IMG;

    return NO_ERROR;
}
  
