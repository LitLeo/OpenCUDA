// ErrorCode.h
// 创建者：于玉龙
//
// 错误码（Error Codes）
// 功能说明：定义了系统中通用的错误码和状态码。按照委托方河边老师的要求，错误码
//           使用负数；正确执行使用 0；其余的非错误状态使用正数。用户和各算法可
//           以根据自己的需要定义额外的错误码和状态码。
//
// 修订历史：
// 2012年07月15日（于玉龙）
//     初始版本。

#ifndef __ERRORCODE_H__
#define __ERRORCODE_H__

// 宏：NO_ERROR（无错误）
// 它表示函数正确执行，没有错误发生。
#define NO_ERROR       0

// 宏：INVALID_DATA（无效数据）
// 它表示参数中包含了无效的数据。
#define INVALID_DATA  -1

// 宏：NULL_POINTER（空指针）
// 它表示不可为 NULL 的变量或参数，意外的出现了 NULL 值。
#define NULL_POINTER  -2

// 宏：OVERFLOW（计算溢出）
// 它表示函数中某些计算产生了溢出，其中包括了除零错误。
#define OP_OVERFLOW   -3

// 宏：NO_FILE（文件未找到）
// 它表示函数未找到指定的文件。
#define NO_FILE       -4

// 宏：WRONG_FILE（文件错误）
// 它表示给定的文件的格式是错误的。
#define WRONG_FILE    -5

// 宏：OUT_OF_MEM（内存耗尽）
// 它表示当前已没有额外的内存支持所要进行的操作了。
#define OUT_OF_MEM    -6

// 宏：CUDA_ERROR（CUDA 错误）
// 它表示由于 CUDA 调用报错，无法继续完成相应的操作。
#define CUDA_ERROR    -7

// 宏：UNMATCH_IMG（图像尺寸不匹配）
// 它表示给定的图像尺寸和操作所要求的图像尺寸不匹配，无法进一步完成操作。
#define UNMATCH_IMG   -8

// 宏：UMIMPLEMENT（未实现）
// 它表示所调用的操作尚未实现，该错误不会出现在提交给委托方的代码中。
#define UNIMPLEMENT   -998

// 宏：UNKNOW_ERROR（未知错误）
// 它表示系统可断定是一个错误，但并不清楚错误的原因。
#define UNKNOW_ERROR  -999


#endif

