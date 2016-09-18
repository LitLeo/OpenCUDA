// ImageToText.h 
//
// 图像转文本
// 功能说明：将输入的灰度图像转成制定大小的文本，文本中用一个字符代表特定的灰度
//           级。首先将原图缩放到和文本同样大小，然后按照灰度级对应找到文本，
//           写入字符串中

#ifndef __IMAGETOTEXT_H__ 
#define __IMAGETOTEXT_H__ 

#include "Image.h"
#include "ErrorCode.h"
#include "ImageStretch.h"

// 类：ImageToText（图像转文本）
// 继承自：无
// 根据输入的灰度图像，对其每个像素点用特定字符表示，得到字符串 outstr。
class ImageToText {

protected:
    // 成员变量：ascii（ASCII 码字符）
    // 图像各个像素点转换成字符的标准
    char *ascii;

    // 成员变量：size（ASCII 码长度）
    // ascii 字符数组的长度
    unsigned int size;

    // 成员变量：level（转化等级）
    // 一个 ASCII 码代表的灰度值的个数
    unsigned int level;

    // 成员变量：imageStretch（图像拉伸）
    // 对输入图像进行拉伸
    ImageStretch imageStretch;

public:  
    // Host 成员方法：newText（创建文本）
    // 创建一个新的文本实例，相当于类的无参构造函数。
    __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                  // NO_ERROR。
    newText()
    {
        // 使用默认值为类的各个成员变量赋值。
        size = 32;  // ascii 字符数组的默认长度为 32
        level = 8;  // 一个 ASCII 码默认代表 8 个灰度值

        // 局部变量，错误码
        cudaError_t cuerrcode;
        // 为 ascii 开辟空间
        cuerrcode = cudaMalloc((void **)&ascii, size * sizeof (char));
        if (cuerrcode != cudaSuccess) {
            cudaFree(ascii);
            return CUDA_ERROR;
        }

        // ascii 字符数组默认值为 “ `.^,:~\"<!ct+{i7?u30pw4A8DX%#HWM\0”
        cuerrcode = cudaMemcpy(ascii, " `.^,:~\"<!ct+{i7?u30pw4A8DX%#HWM",
                               size * sizeof (char), cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            cudaFree(ascii);
            return CUDA_ERROR;
        }

        return NO_ERROR;
    }

    // Host 成员方法：newText（创建文本）
    // 创建一个新的文本实例，相当于类的有参构造函数。
    __host__ int                     // 返回值：函数是否正确执行，若函数正确
                                     // 执行，返回 NO_ERROR。
    newText(
            char *ascii,             // ascii 字符数组
            unsigned int size,       // ascii 字符数组的长度
            bool onhostarray = true  // 判断 ascii 是否是 Host 内存的指针，
                                     // 默认为“是”。
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        size = 32;  // ascii 字符数组的默认长度为 32
        level = 8;  // 一个 ASCII 码默认代表 8 个灰度值

        // 局部变量，错误码
        cudaError_t cuerrcode;
        // 为 ascii 开辟空间
        cuerrcode = cudaMalloc((void **)&ascii, size * sizeof (char));
        if (cuerrcode != cudaSuccess) {
            cudaFree(ascii);
            return CUDA_ERROR;
        }

        // ascii 字符数组默认值为 “ `.^,:~\"<!ct+{i7?u30pw4A8DX%#HWM\0”
        cuerrcode = cudaMemcpy(ascii, " `.^,:~\"<!ct+{i7?u30pw4A8DX%#HWM",
                               size * sizeof (char), cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            cudaFree(ascii);
            return CUDA_ERROR;
        }

        // 调用 setAscii 函数根据参数列表中的值设定成员变量的初值
        this->setAscii(ascii, size, onhostarray);

        return NO_ERROR;
    }

    // 析构函数：~ImageToText
    // 类的析构函数，在类销毁时被调用，在类销毁时将字符数组数据空间释放。
    __host__ __device__
    ~ImageToText()
    {
        // 若 ascii 不为空，释放内存空间
        if (ascii != NULL) {
            cudaFree(ascii);
            ascii = NULL;
        }
    }

    // 成员方法：getAscii（获取 ascii 字符数组）
    // 获取成员变量 ascii 的值。
    __host__ __device__ char *  // 返回值：成员变量 ascii 的值
    getAscii() const
    {
        // 返回 ascii 成员变量的值。
        return this->ascii;
    } 

    // Host 成员方法：setAscii（设置 ascii 字符数组）
    // 设置成员变量 ascii 的值。
    __host__ int                     // 返回值：函数是否正确执行，若函数正确执
                                     // 行，返回 NO_ERROR。
    setAscii(
            char *ascii,             // 设定新的 ascii 字符数组
            unsigned int size,       // 新的 ascii 字符数组的长度
            bool onhostarray = true  // 判断 ascii 是否是 Host 内存的指针，
                                     // 默认为“是”。
    ) {
        // 将 size 成员变量赋成新值
        this->size = size;

        // 将 level 成员变量赋成新值
        this->level = (256 + this->size - 1) / this->size;

        // 若 this->ascii 不为空，将其空间释放
        if (this->ascii != NULL) {
            cudaFree(this->ascii);
            this->ascii = NULL;
        }

        // 局部变量，错误码
        cudaError_t cuerrcode;

        // 为 this->ascii 分配空间
        cuerrcode = cudaMalloc((void **)&this->ascii, size * sizeof (char));
        if (cuerrcode != cudaSuccess) {
            cudaFree(this->ascii);
            return CUDA_ERROR;
        }

        // 判断当前 ascii 数组是否存储在 Host 端，并将 ascii 数组拷贝至
        // this->ascii
        if(onhostarray) {
            // 将 Host 端的 ascii 拷贝至 Device 端的 this->ascii
            cuerrcode = cudaMemcpy(this->ascii, ascii, size * sizeof (char),
                                   cudaMemcpyHostToDevice);
            if (cuerrcode != cudaSuccess) {
                cudaFree(this->ascii);
                return CUDA_ERROR;
            }
        } else {
            // 将 Device 端的 ascii 拷贝至 Device 端的 this->ascii
            cuerrcode = cudaMemcpy(this->ascii, ascii, size * sizeof (char),
                                   cudaMemcpyDeviceToDevice);
            if (cuerrcode != cudaSuccess) {
                cudaFree(this->ascii);
                return CUDA_ERROR;
            }
        }

        return NO_ERROR;
    }

    // 成员方法：getSize（获取 ascii 字符数组的长度）
    // 获取成员变量 size 的值。
    __host__ __device__ unsigned int  // 返回值：成员变量 size 的值
    getSize() const
    {
        // 返回 size 成员变量的值。
        return this->size;
    } 

    // 成员方法：getLevel（获取一个 ASCII 码代表的灰度值的个数）
    // 获取成员变量 level 的值。
    __host__ __device__ unsigned int  // 返回值：成员变量 level 的值
    getLevel() const
    {
        // 返回 level 成员变量的值。
        return this->level;
    } 

    // Host 成员方法：imageToText（图像转文本）
    // 对输入的 inimg 的每个像素点用特定字符表示，得到结果字符串 outstr。
    __host__ int                      // 返回值：函数是否正确执行，若函数正确
                                      // 执行，返回NO_ERROR。
    imageToText(
            Image *inimg,             // 输入图像
            char *outstr,             // 输出参数，转化后生成的字符串
            size_t width,             // 文本的宽度（width >= 0）
            size_t height,            // 文本的高度（height >= 0）
            bool onhostarray = true   // 判断 outstr 是否是 Host 内存的指针，
                                      // 默认为“是”。
    );
};

#endif