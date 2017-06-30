// Curve.cu
// 曲线数据结构的定义和曲线的基本操作。
                                                                               
#include "Curve.h"

#include <iostream>
#include <fstream>
using namespace std;

#include "ErrorCode.h"

// 宏：SMCURVE_VALID 和 SMCURVE_NOT_VALID
// 定义了 smCurveCordiXY 是否有效的范围值。
#define SMCURVE_VALID      1
#define SMCURVE_NOT_VALID  0

// 宏：TANGENT_VALID 和 TANGENT_NOT_VALID
// 定义了 tangent 是否有效的范围值。
#define TANGENT_VALID      1 
#define TANGENT_NOT_VALID  0 
                                                                                                                                                           
// Host 静态方法：_calCrvXY（得到某曲线数据对应的属性坐标）
// 计算得到输入数据的起点坐标、终点坐标、最右点的 x 坐标、最左点的 x 坐标、
// 最下点的 y 坐标、最上点的 y 坐标、曲线坐标的平均值，并将得到的属性值赋
// 给输入曲线内相对应的属性值。
static __host__ int 
_calCrvXY(
        Curve *incrv,  // 输入曲线
        int *data,     // 输入坐标数据
        int size       // 输入坐标数据的大小
);

// Host 静态方法：_calCrvXY（得到某曲线数据对应的属性坐标）
__host__ int _calCrvXY(Curve *incrv, int *data, int size)
{
    // 检查输入曲线和输入坐标数据是否为 NULL。
    if (incrv == NULL || data == NULL) 
        return NULL_POINTER;

    // 检查输入数据的大小是否小于 1。
    if (size < 1)
        return INVALID_DATA;

    // 定义一个一维数组用于保存计算得到的四个属性坐标,初始化为坐标数据的
    // 第一个值。
    int tempXY[4] = { data[0], data[0], data[0], data[0] };

    // 定义曲线坐标的和。
    int sumX = 0;
    int sumY = 0;

    // 遍历输入数据，计算属性坐标。
    for (int i = 0; i < size; i++) {
        if (data[2 * i] > tempXY[0]) {
            // 计算曲线上的最右点的 x 坐标
            tempXY[0] = data[2 * i];
        } else if(data[2 * i] < tempXY[1]) {
            // 计算曲线上的最左点的 x 坐标
            tempXY[1] = data[2 * i];
        }

        if (data[2 * i + 1] > tempXY[2]) {
            // 计算曲线上的最下点的 y 坐标
            tempXY[2] = data[2 * i + 1];
        } else if(data[2 * i + 1] < tempXY[3]) {
            // 计算曲线上的最上点的 y 坐标
            tempXY[3] = data[2 * i + 1];
        }

        // 分别计算 x、y坐标的和。
        sumX += data[2 * i];
        sumY += data[2 * i + 1];
    }

    // 将计算得到的属性值赋给曲线 
    incrv->maxCordiX = tempXY[0];
    incrv->minCordiX = tempXY[1];
    incrv->maxCordiY = tempXY[2];
    incrv->minCordiY = tempXY[3];
    incrv->aveX = sumX / size;
    incrv->aveY = sumY / size;

    // 根据曲线内点首尾两点是否互为 8 邻域来判断曲线是否闭合。
    int diffX = data[0] - data[(size - 1) * 2];
    int diffY = data[1] - data[(size - 1) * 2 + 1];
    if (diffX >= -1 && diffX <= 1 && diffY >= -1 && diffY <= 1) {
        incrv->closed = true;
        incrv->startCordiX = incrv->endCordiX = data[0];
        incrv->startCordiY = incrv->endCordiY = data[1];
    } else {
        incrv->closed = false;
        incrv->startCordiX = data[0];
        incrv->startCordiY = data[1];
        incrv->endCordiX = data[(size - 1) * 2];
        incrv->endCordiY = data[(size - 1) * 2 + 1];
    }

    return NO_ERROR;
}

// Host 静态方法：newCurve（创建曲线）
__host__ int CurveBasicOp::newCurve(Curve **outcrv)
{
    // 检查用于盛放新曲线的指针是否为 NULL。
    if (outcrv == NULL)
        return NULL_POINTER;

    // 申请一个新的 CurveCuda 型数据，本方法最后会将其中的 crvMeta 域返回给
    // outcrv，这样 outcrv 就有了一个对应的 CurveCuda 型伴随数据。
    CurveCuda *crvCud = new CurveCuda;

    // 初始化各种元数据。
    crvCud->crvMeta.closed = 0;
    crvCud->crvMeta.startCordiX = 0;
    crvCud->crvMeta.startCordiY = 0;
    crvCud->crvMeta.endCordiX = 0;
    crvCud->crvMeta.endCordiY = 0;
    crvCud->crvMeta.maxCordiX = 0;
    crvCud->crvMeta.minCordiX = 0;
    crvCud->crvMeta.maxCordiY = 0;
    crvCud->crvMeta.minCordiY = 0;
    crvCud->crvMeta.aveX = 0;
    crvCud->crvMeta.aveY = 0;
    crvCud->crvMeta.curveLength = 0;
    crvCud->crvMeta.crvData = NULL;
    crvCud->crvMeta.smWindowSize = 0;
    crvCud->crvMeta.smCurveCordiXY = NULL;
    crvCud->crvMeta.tangent = NULL;
    crvCud->crvMeta.geoProperty = false;
    crvCud->crvMeta.primaryProperties = NULL;
    crvCud->capacity = 0;
    crvCud->deviceId = -1;
    crvCud->smCurveIsValid = SMCURVE_NOT_VALID;
    crvCud->tangentIsValid = TANGENT_NOT_VALID;

    // 将 CurveCuda 型数据中的 crvMeta 赋值给输出参数。
    *outcrv = &(crvCud->crvMeta);

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 静态方法：deleteCurve（销毁曲线）
__host__ int CurveBasicOp::deleteCurve(Curve *incrv)
{
    // 检查曲线的指针是否为 NULL。
    if (incrv == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Curve 指针，得到对应的 CurveCuda 型数据。
    CurveCuda *incrvCud = CURVE_CUDA(incrv);

    // 检查曲线所在的地址空间是否合法，如果曲线所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (incrvCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 释放曲线数据。
    if (incrv->crvData == NULL || incrv->curveLength == 0) {
        // 如果输入曲线是空的，则不进行曲线数据释放操作（因为本来也没有数据可被
        // 释放）。
        // Do Nothing;
    } if (incrvCud->deviceId < 0) {
        // 对于数据存储于 Host 内存，直接利用 delete 关键字释放曲线数据。
        delete[] incrv->crvData;
        // 如果 smoothed 曲线有效，则释放数据。
        if (incrvCud->smCurveIsValid == SMCURVE_VALID)
            delete[] incrv->smCurveCordiXY;
        
        // 如果曲线斜率数据有效，则释放数据。
        if (incrvCud->tangentIsValid == TANGENT_VALID)
            delete[] incrv->tangent;
    } else {
        // 对于数据存储于 Device 内存中，则需要首先切换设备，将该设备作为当前
        // Device 设备，然后释放之，最后还需要将设备切换回来以保证后续处理的正
        // 确性。
        cudaSetDevice(incrvCud->deviceId);
        cudaFree(incrv->crvData);

        // 如果 smoothed 曲线有效，则释放数据。
        if (incrvCud->smCurveIsValid == SMCURVE_VALID)
            cudaFree(incrv->smCurveCordiXY);
        
        // 如果曲线斜率数据有效，则释放数据。
        if (incrvCud->tangentIsValid == TANGENT_VALID)
            cudaFree(incrv->tangent);
        cudaSetDevice(curdevid);
    }

    // 最后还需要释放曲线的元数据
    delete incrvCud;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
__host__ int CurveBasicOp::makeAtCurrentDevice(Curve *crv, size_t curveLength, 
                                               int *crvData)
{
    // 检查输入曲线是否为 NULL
    if (crv == NULL)
        return NULL_POINTER;

    // 检查给定的曲线中坐标点数量
    if (curveLength < 1)
        return INVALID_DATA;

    // 检查曲线是否为空曲线
    if (crv->crvData != NULL)
        return UNMATCH_IMG;

    // 获取 crv 对应的 CurveCuda 型数据。
    CurveCuda *crvCud = CURVE_CUDA(crv);

    // 定义错误变量
    cudaError_t cuerrcode;

    // 如果初始化曲线长度不为 0，为曲线内数据开空间。
    if (curveLength != 0) {
        // 在当前的 Device 上申请存储指定坐标数量的曲线所需要的内存空间。
        cuerrcode = cudaMalloc((void **)(&crv->crvData), 
                               2 * curveLength * sizeof (int));
        if (cuerrcode != cudaSuccess) {
            crv->crvData = NULL;
            return CUDA_ERROR;
        }

        // 判断 smoothed 曲线是否有效，若有效则开空间。
        if (crvCud->smCurveIsValid == SMCURVE_VALID) {
            // 为 smoothed 曲线数据开空间。
            cuerrcode = cudaMalloc((void **)(&crv->smCurveCordiXY), 
                                   2 * curveLength * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(crv->crvData);
                crv->smCurveCordiXY = NULL;
                return CUDA_ERROR;
            }
        }
        
        // 判断切线斜率是否有效，若有效则开空间。
        if (crvCud->tangentIsValid == TANGENT_VALID) {
            // 为曲线上各点处的切线斜率数据开空间。
            cuerrcode = cudaMalloc((void **)(&crv->tangent), 
                                   curveLength * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(crv->crvData);
                if (crvCud->smCurveIsValid == SMCURVE_VALID)
                    cudaFree(crv->smCurveCordiXY);
                crv->tangent = NULL;
                return CUDA_ERROR;
            }
        }
    }
    
    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 修改曲线的元数据。
    crv->curveLength = curveLength;
    crvCud->deviceId = curdevid;

    if (crvData == NULL) {
        // 当输入坐标数据为空时，设置曲线内实际点数量为 0.
        crvCud->capacity = 0;
    } else {
        // 当输入坐标数据不为空时，将输入坐标数据拷贝到曲线内的坐标数据中。
        cuerrcode = cudaMemcpy(crv->crvData, crvData,
                               crv->curveLength * 2 * sizeof (int),
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            // 拷贝错误，释放申请的内存。
            cudaFree(crv->crvData);
            if (crvCud->smCurveIsValid == SMCURVE_VALID)
                cudaFree(crv->smCurveCordiXY);
            if (crvCud->tangentIsValid == TANGENT_VALID)
                cudaFree(crv->tangent);
            return CUDA_ERROR;
        }

        // 曲线内实际点数量等于曲线内点空间的数量。
        crvCud->capacity = curveLength;

        // 局部变量，错误码。
        int errcode;

        // 计算曲线坐标内的各属性值
        errcode = _calCrvXY(crv, crvData, curveLength);
        if (errcode != NO_ERROR) {
            cudaFree(crv->crvData);
            if (crvCud->smCurveIsValid == SMCURVE_VALID)
                cudaFree(crv->smCurveCordiXY);
            if (crvCud->tangentIsValid == TANGENT_VALID)
                cudaFree(crv->tangent);
            crvCud->capacity = 0;
            crv->curveLength = 0;
            return errcode;
        }
    }

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 静态方法：makeAtHost（在 Host 内存中构建数据）
__host__ int CurveBasicOp::makeAtHost(Curve *crv, size_t curveLength,
                                      int *crvData)
{
    // 检查输入曲线是否为 NULL
    if (crv == NULL)
        return NULL_POINTER;

    // 检查给定的曲线中坐标点数量
    if (curveLength < 1)
        return INVALID_DATA;

    // 检查曲线是否为空曲线
    if (crv->crvData != NULL)
        return UNMATCH_IMG;

    // 获取 crv 对应的 CurveCuda 型数据。
    CurveCuda *crvCud = CURVE_CUDA(crv);

    // 为曲线数据在 Host 内存中申请空间,不赋值。
    crv->crvData = new int[curveLength * 2];
    if (crv->crvData == NULL)
        return OUT_OF_MEM;

    // 判断 smoothed 曲线是否有效，若有效则开空间。
    if (crvCud->smCurveIsValid == SMCURVE_VALID) {
        // 为 smoothed 曲线数据开空间。
        crv->smCurveCordiXY = new float[curveLength * 2];
        if (crv->smCurveCordiXY == NULL) {
            delete [] crv->crvData;
            return OUT_OF_MEM;
        }
    }
    
    // 判断切线斜率是否有效，若有效则开空间。
    if (crvCud->tangentIsValid == TANGENT_VALID) {
        // 为曲线上各点处的切线斜率数据开空间。
        crv->tangent = new float[curveLength];
        if (crv->tangent == NULL) {
            delete [] crv->crvData;
            if (crvCud->smCurveIsValid == SMCURVE_VALID)
                delete [] crv->smCurveCordiXY;
            return OUT_OF_MEM;
        }
    }
    
    if (crvData == NULL) {
        // 当输入数据为空时，设置曲线内实际点的数量为 0。
        // 不为曲线赋值。
        crvCud->capacity = 0;
    } else {
        // 当输入坐标数据不为空时，直接将输入坐标数据赋值给
        // 曲线内的坐标数据。
        // 将 crvData 内的坐标数据拷贝到 crv->crvData 中。memcpy 不返回错误，
        // 因此，没有进行错误检查。
        memcpy(crv->crvData, crvData, curveLength * 2 * sizeof (int));
        
        // 曲线内实际点数量等于曲线内点空间的数量。
        crvCud->capacity = curveLength;

        // 局部变量，错误码。
        int errcode;

        // 计算曲线坐标内的各属性值
        errcode = _calCrvXY(crv, crvData, curveLength);
        if (errcode != NO_ERROR) {
            cudaFree(crv->crvData);
            if (crvCud->smCurveIsValid == SMCURVE_VALID)
                cudaFree(crv->smCurveCordiXY);
            if (crvCud->tangentIsValid == TANGENT_VALID)
                cudaFree(crv->tangent);
            crvCud->capacity = 0;
            crv->curveLength = 0;
            return errcode;
        }
    }

    // 设置曲线中的元数据
    crv->curveLength = curveLength;
    crvCud->deviceId = -1;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：readFromFile（从文件读取曲线）
__host__ int CurveBasicOp::readFromFile(const char *filepath,
                                        Curve *outcrv)
{
    // 这段代码仅支持 int 型尺寸为 2、4、8 三种情况。目前绝大部分的系统，采用了
    // sizeof (int) == 4 的情况，少数早期的 DOS 和 Windows 系统中 sizeof (int)
    // == 2。
    if (sizeof (int) != 2 && sizeof (int) != 4 && sizeof (int) != 8)
        return UNIMPLEMENT;

    // 检查文件路径和曲线是否为 NULL。
    if (filepath == NULL || outcrv == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Curve 型指针，得到对应的 CurveCuda 型数据。
    CurveCuda *outcrvCud = CURVE_CUDA(outcrv);

    // 检查曲线所在的地址空间是否合法，如果曲线所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (outcrvCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 打开曲线文件。
    ifstream crvfile(filepath, ios::in | ios::binary);
    if (!crvfile)
        return NO_FILE;

    // 将文件读指针挪到文件的开头处。该步骤虽然显得多余，但是却可以确保操作的正
    // 确。
    crvfile.seekg(0, ios::beg);

    // 读取文件的前四个字节，这是文件的类型头，如果类型头为 CRVT，则说明该文件
    // 是曲线文件。
    char typestr[5] = { '\0' };
    crvfile.read(typestr, 4);
    if (strcmp(typestr, "CRVT") != 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的是否闭合的标记。如果标记不为 0 且不为 1，则报错
    int closed = 0;
    crvfile.read(reinterpret_cast<char *>(&closed), 4);
    if (closed != 1 && closed != 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的起点的 x 坐标。
    int startCordiX = 0;
    crvfile.read(reinterpret_cast<char *>(&startCordiX), 4);
    if (startCordiX < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的起点的 y 坐标。
    int startCordiY = 0;
    crvfile.read(reinterpret_cast<char *>(&startCordiY), 4);
    if (startCordiY < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的终点的 x 坐标。
    int endCordiX = 0;
    crvfile.read(reinterpret_cast<char *>(&endCordiX), 4);
    if (endCordiX < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的终点的 y 坐标。
    int endCordiY = 0;
    crvfile.read(reinterpret_cast<char *>(&endCordiY), 4);
    if (endCordiY < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的最右点的 x 坐标。
    int maxCordiX = 0;
    crvfile.read(reinterpret_cast<char *>(&maxCordiX), 4);
    if (maxCordiX < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的最左点的 x 坐标
    int minCordiX = 0;
    crvfile.read(reinterpret_cast<char *>(&minCordiX), 4);
    if (minCordiX < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的最下点的 y 坐标
    int maxCordiY = 0;
    crvfile.read(reinterpret_cast<char *>(&maxCordiY), 4);
    if (maxCordiY < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的最上点的 y 坐标
    int minCordiY = 0;
    crvfile.read(reinterpret_cast<char *>(&minCordiY), 4);
    if (minCordiY < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的 x 坐标的平均值
    int aveX = 0;
    crvfile.read(reinterpret_cast<char *>(&aveX), 4);
    if (aveX < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的 y 坐标的平均值
    int aveY = 0;
    crvfile.read(reinterpret_cast<char *>(&aveY), 4);
    if (aveY < 0)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的点空间的数量。如果坐标点数量小于 1，则报错。
    size_t curveLength = 0;
    crvfile.read(reinterpret_cast<char *>(&curveLength), 4);
    if (curveLength < 1)
        return WRONG_FILE;

    // 从文件中获取曲线中包含的实际点的数量。
    size_t capacity = 0;
    crvfile.read(reinterpret_cast<char *>(&capacity), 4);
    if (capacity < 1)
        return WRONG_FILE;

    // 为在内存中保存曲线的坐标点而申请新的数据空间。为了避免频繁的数据申请与释
    // 放，如果发现原来曲线中的坐标点数量和新的数据中坐标点数量相同，且原来的数
    // 据存储于 Host 内存，则会重用这段内存空间，不去重新申请内存。
    int *newdata;
    bool reusedata;
    if (outcrv->crvData != NULL && outcrv->curveLength == curveLength &&
        outcrvCud->deviceId == -1) {
        // 若数据可以重用，则使用原来的内存空间。
        newdata = outcrv->crvData;
        reusedata = true;
    } else {
        // 若数据不能重用，则重新申请合适的内存空间。
        newdata = new int[curveLength * 2];
        reusedata = false;
        if (newdata == NULL) {
            return OUT_OF_MEM;
        }
    }

    // 读取坐标点数据。因为文件中存储的坐标点采用了 32 位有符号整形数，这里需要
    // 根据系统中 int 型数据的尺寸采取不同的转换策略。
    if (sizeof (int) == 2) {
        // 对于 sizeof (int) == 2 的系统通常 long 型数据为 32 位，因此需要逐个
        // 读取后转成 int 型存放到数据数组中。
        long tmp;
        for (int i = 0; i < curveLength * 2; i++) {
            crvfile.read(reinterpret_cast<char *>(&tmp), 4);
            newdata[i] = (int)tmp;
        }
    } else if (sizeof (int) == 8) {
        // 对于 sizeof (int) == 8 的系统通常 short 型数据为 32 位，因此需要逐个
        // 读取后转成 int 型存放到数据数组中。
        short tmp;
        for (int i = 0; i < curveLength * 2; i++) {
            crvfile.read(reinterpret_cast<char *>(&tmp), 4);
            newdata[i] = (int)tmp;
        }
    } else {
        // 对于 sizeof (int) == 4 的系统，不需要进行任何的转换，读取后的数据可
        // 读取存放到数据数组中。
        crvfile.read(reinterpret_cast<char *>(newdata), curveLength * 2 * 4);
    }

    // 当数据已经成功的读取后，释放原来数据占用的内存空间，防止内存泄漏。
    if (outcrv->crvData != NULL && !reusedata) {
        if (outcrvCud->deviceId == -1) {
            // 如果原来的数据存放在 Host 内存中，则直接通过 delete 关键字释放。
            delete[] outcrv->crvData;
        } else {
            // 如果原来的数据存放在 Device 内存中，则切换到相应的 Device 后，使
            // 用 cudaFree 释放。
            cudaSetDevice(outcrvCud->deviceId);
            cudaFree(outcrv->crvData);
            cudaSetDevice(curdevid);
        }
    }

    // 从文件中获取曲线中 smooth 曲线标记。
    int smCurveIsValid = 0;
    crvfile.read(reinterpret_cast<char *>(&smCurveIsValid), 4);
    if (smCurveIsValid != SMCURVE_VALID && smCurveIsValid != SMCURVE_NOT_VALID)
        return WRONG_FILE;

    // 新的 smooth 曲线数据，如果标记有效，则为其开空间，否则程序结束后自动舍弃。
    float *newsmdata;

    // 曲线的 smWindowSize 数据，如果标记有效，则读取数据。
    int smWindowSize = 0;

    // 是否重用原曲线smooth 曲线数据标记。
    bool reusesmdata = true;

    // 如果 smooth 曲线标记有效，则读取 smooth 曲线数据。
    if (smCurveIsValid == SMCURVE_VALID) {
        crvfile.read(reinterpret_cast<char *>(&smWindowSize), 4);
                                                                               
        // 为在内存中保存曲线的 smooth 曲线数据而申请新的数据空间。为了避免频繁
        // 的数据申请与释放，如果发现原来曲线中的坐标点数量和新的数据中坐标点数
        // 量相同，且原来的数据存储于 Host 内存，则会重用这段内存空间，不去重新
        // 申请内存。
        if (outcrv->smCurveCordiXY != NULL && 
            outcrv->curveLength == curveLength &&
            outcrvCud->deviceId == -1) {
            // 若数据可以重用，则使用原来的内存空间。
            newsmdata = outcrv->smCurveCordiXY;
            reusesmdata = true;
        } else {
            // 若数据不能重用，则重新申请合适的内存空间。
            newsmdata = new float[curveLength * 2];
            reusesmdata = false;
            if (newsmdata == NULL) {
                if (reusedata == false)
                    delete[] newdata;
                return OUT_OF_MEM;
            }
        }

        // 读取 smooth 曲线数据。
        crvfile.read(reinterpret_cast<char *>(newsmdata), curveLength * 2 * 4);

        // 当数据已经成功的读取后，释放原来数据占用的内存空间，防止内存泄漏。
        if (outcrv->smCurveCordiXY != NULL && !reusesmdata) {
            if (outcrvCud->deviceId == -1) {
                // 如果原来的数据存放在 Host 内存中，则直接通过 delete 关键字释
                // 放。
                delete[] outcrv->smCurveCordiXY;
            } else {
                // 如果原来的数据存放在 Device 内存中，则切换到相应的 Device 
                // 后，使用 cudaFree 释放。
                cudaSetDevice(outcrvCud->deviceId);
                cudaFree(outcrv->smCurveCordiXY);
                cudaSetDevice(curdevid);
            }
        }
    }

    // 从文件中获取曲线中曲线斜率标记标记。
    int tangentIsValid = 0;
    crvfile.read(reinterpret_cast<char *>(&tangentIsValid), 4);
    if (tangentIsValid != TANGENT_VALID && tangentIsValid != TANGENT_NOT_VALID)
        return WRONG_FILE;

     // 新的曲线斜率数据，如果标记有效，则为其开空间否则自动舍弃。
    float *newtandata;

    // 如果曲线斜率标记有效，则读取曲线斜率数据。
    if (tangentIsValid == TANGENT_VALID) {
        // 为在内存中保存曲线的斜率数据而申请新的数据空间。为了避免频繁的数据申
        // 请与释放，如果发现原来曲线中的坐标点数量和新的数据中坐标点数量相同，
        // 且原来的数据存储于 Host 内存，则会重用这段内存空间，不去重新申请内
        // 存。
        bool reusetandata;
        if (outcrv->tangent != NULL && outcrv->curveLength == curveLength &&
            outcrvCud->deviceId == -1) {
            // 若数据可以重用，则使用原来的内存空间。
            newtandata = outcrv->tangent;
            reusetandata = true;
        } else {
            // 若数据不能重用，则重新申请合适的内存空间。
            newtandata = new float[curveLength];
            reusetandata = false;
            if (newtandata == NULL) {
                if (reusedata == false)
                    delete[] newdata;
                if (smCurveIsValid == SMCURVE_VALID && reusesmdata == false)
                    delete[] newsmdata;
                return OUT_OF_MEM;
            }
        }

        // 读取 smooth 曲线数据。
        crvfile.read(reinterpret_cast<char *>(newtandata), curveLength * 4);

        // 当数据已经成功的读取后，释放原来数据占用的内存空间，防止内存泄漏。
        if (outcrv->tangent != NULL && !reusetandata) {
            if (outcrvCud->deviceId == -1) {
                // 如果原来的数据存放在 Host 内存中，则直接通过 delete 关键字释放。
                delete[] outcrv->tangent;
            } else {
                // 如果原来的数据存放在 Device 内存中，则切换到相应的 Device 后，使
                // 用 cudaFree 释放。
                cudaSetDevice(outcrvCud->deviceId);
                cudaFree(outcrv->tangent);
                cudaSetDevice(curdevid);
            }
        }
    }

    // 使用新的数据更新曲线的元数据。
    outcrv->closed = (closed == 1 ? true : false);
    outcrv->startCordiX = startCordiX;
    outcrv->startCordiY = startCordiY;
    outcrv->endCordiX = endCordiX;
    outcrv->endCordiY = endCordiY;
    outcrv->maxCordiX = maxCordiX;
    outcrv->minCordiX = minCordiX;
    outcrv->maxCordiY = maxCordiY;
    outcrv->minCordiY = minCordiY;
    outcrv->aveX = aveX;
    outcrv->aveY = aveY;
    outcrv->curveLength = curveLength;
    outcrv->crvData = newdata;
    outcrv->smWindowSize = smWindowSize;
    outcrv->smCurveCordiXY = newsmdata;
    outcrv->tangent = newtandata;
    outcrvCud->capacity = capacity;
    outcrvCud->deviceId = -1;
    outcrvCud->smCurveIsValid = smCurveIsValid;
    outcrvCud->tangentIsValid = tangentIsValid;

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：writeToFile（将曲线写入文件）
__host__ int CurveBasicOp::writeToFile(const char *filepath, Curve *incrv)
{
    // 这段代码仅支持 int 型尺寸为 2、4、8 三种情况。目前绝大部分的系统，采用了
    // sizeof (int) == 4 的情况，少数早期的 DOS 和 Windows 系统中 sizeof (int)
    // == 2。
    if (sizeof (int) != 2 && sizeof (int) != 4 && sizeof (int) != 8)
        return UNIMPLEMENT;

    // 检查文件路径和曲线是否为 NULL。
    if (filepath == NULL || incrv == NULL)
        return NULL_POINTER;

    // 打开需要写入的文件。
    ofstream crvfile(filepath, ios::out | ios::binary);
    if (!crvfile)
        return NO_FILE;

    // 将曲线的数据拷贝回 Host 内存中，这样曲线就可以被下面的代码所读取，然后将
    // 曲线的数据写入到磁盘中。这里需要注意的是，安排曲线的拷贝过程在文件打开之
    // 后是因为，如果一旦文件打开失败，则不会改变曲线在内存中的存储状态，这可能
    // 会对后续处理更加有利。
    int errcode;
    errcode = CurveBasicOp::copyToHost(incrv);
    if (errcode < 0)
        return errcode;

    // 向文件中写入文件类型字符串
    static char typestr[] = "CRVT";
    crvfile.write(typestr, 4);

    // 向文件中写入曲线含有的是否闭合标记，写入之前将 bool 型转化为 int 型。
    int closed = incrv->closed == true ? 1 : 0;
    crvfile.write(reinterpret_cast<char *>(&closed), 4);

    // 向文件中写入曲线含有的起点的 x 坐标。
    crvfile.write(reinterpret_cast<char *>(&incrv->startCordiX), 4);

    // 向文件中写入曲线含有的起点的 y 坐标。
    crvfile.write(reinterpret_cast<char *>(&incrv->startCordiY), 4);

    // 向文件中写入曲线含有的终点的 x 坐标。
    crvfile.write(reinterpret_cast<char *>(&incrv->endCordiX), 4);

    // 向文件中写入曲线含有的终点的 y 坐标。
    crvfile.write(reinterpret_cast<char *>(&incrv->endCordiY), 4);

    // 向文件中写入曲线含有的最右点的 x 坐标。
    crvfile.write(reinterpret_cast<char *>(&incrv->maxCordiX), 4);

    // 向文件中写入曲线含有的最左点的 x 坐标。
    crvfile.write(reinterpret_cast<char *>(&incrv->minCordiX), 4);

    // 向文件中写入曲线含有的最下点的 y 坐标。
    crvfile.write(reinterpret_cast<char *>(&incrv->maxCordiY), 4);

    // 向文件中写入曲线含有的最上点的 y 坐标。
    crvfile.write(reinterpret_cast<char *>(&incrv->minCordiY), 4);

    // 向文件中写入曲线含有的 x 坐标的平均值。
    crvfile.write(reinterpret_cast<char *>(&incrv->aveX), 4);

    // 向文件中写入曲线含有的 y 坐标的平均值。
    crvfile.write(reinterpret_cast<char *>(&incrv->aveY), 4);

    // 向文件中写入曲线含有的点空间的数量。
    crvfile.write(reinterpret_cast<char *>(&incrv->curveLength), 4);

    // 获取 crv 对应的 CurveCuda 型数据。
    CurveCuda *incrvCud = CURVE_CUDA(incrv);

    // 向文件中写入曲线含有的实际点的数量。
    crvfile.write(reinterpret_cast<char *>(&incrvCud->capacity), 4);

    // 向文件中写入坐标数据，因为考虑到。为了保证每个整型数据占用 4 个字节，这
    // 里对不同的情况进行了处理。不过针对目前绝大部分系统来说，sizeof (int) ==
    // 4，因此绝大部分情况下，编译器会选择 else 分支。如果委托方认为系统是运行
    // 在 sizeof (int) == 4 的系统之上，也可以删除前面的两个分支，直接使用最后
    // 的 else 分支。
    if (sizeof (int) == 2) {
        // 对于 sizeof (int) == 2 的系统来说，long 通常是 32 位的，因此，需要逐
        // 个的将数据转换成 32 位的 long 型，然后进行处理。
        long tmp;
        for (int i = 0; i < incrv->curveLength * 2; i++) {
            tmp = (long)(incrv->crvData[i]);
            crvfile.write(reinterpret_cast<char *>(&tmp), 4);
        }
    } else if (sizeof (int) == 8) {
        // 对于 sizeof (int) == 8 的系统来说，short 通常是 32 位的，因此，需要
        // 逐个的将数据转换成 32 位的 short 型，然后进行处理。
        short tmp;
        for (int i = 0; i < incrv->curveLength * 2; i++) {
            tmp = (short)(incrv->crvData[i]);
            crvfile.write(reinterpret_cast<char *>(&tmp), 4);
        }
    } else {
        // 如果 sizeof (int) == 4，则可以直接将数据写入磁盘，而不需要任何的转换
        // 过程。
        crvfile.write(reinterpret_cast<char *>(incrv->crvData),
                      incrv->curveLength * 2 * 4);
    }

    // 向文件中写入 smooth 曲线标记。
    crvfile.write(reinterpret_cast<char *>(&incrvCud->smCurveIsValid), 4);

    // 如果 smooth 曲线标记有效，则将数据直接写入磁盘。
    if (incrvCud->smCurveIsValid == SMCURVE_VALID) {
        // 向文件中写入曲线的 smWindowSize 数据。
        crvfile.write(reinterpret_cast<char *>(&incrv->smWindowSize), 4);

        // 向文件中写入 smooth 曲线数据。
        crvfile.write(reinterpret_cast<char *>(incrv->smCurveCordiXY),
                      incrv->curveLength * 2 * 4);
    }

    // 向文件中写入曲线斜率数据标记。
    crvfile.write(reinterpret_cast<char *>(&incrvCud->tangentIsValid), 4);

    // 如果曲线斜率数据标记有效，则将数据直接写入磁盘。
    if (incrvCud->tangentIsValid == TANGENT_VALID) {
        // 向文件中写入曲线斜率数据。
        crvfile.write(reinterpret_cast<char *>(incrv->tangent),
                      incrv->curveLength * 4);
    }

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 静态方法：copyToCurrentDevice（将曲线拷贝到当前 Device 内存上）
__host__ int CurveBasicOp::copyToCurrentDevice(Curve *crv)
{
    // 检查曲线是否为 NULL。
    if (crv == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Curve 型指针，得到对应的 CurveCuda 型数据。
    CurveCuda *crvCud = CURVE_CUDA(crv);

    // 检查曲线所在的地址空间是否合法，如果曲线所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (crvCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果曲线是一个不包含数据的空曲线，则报错。
    if (crv->crvData == NULL || crv->curveLength == 0)
        return UNMATCH_IMG;
        
    // 对于不同的情况，将曲线数据拷贝到当前设备上。
    if (crvCud->deviceId < 0) {
        // 如果曲线的数据位于 Host 内存上，则需要在当前 Device 的内存空间上申请
        // 空间，然后将 Host 内存上的数据拷贝到当前 Device 上。
        int *devptr;            // 新的坐标数据空间，在当前 Device 上。
        float *devsm;           // 新的 smoothed 曲线数据空间。
        float *devtangent;      // 新的斜率数据空间。

        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 在当前设备上申请坐标数据的空间。
        cuerrcode = cudaMalloc((void **)(&devptr), 
                               crv->curveLength * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 将原来存储在 Host 上坐标数据拷贝到当前 Device 上。
        cuerrcode = cudaMemcpy(devptr, crv->crvData, 
                               crv->curveLength * 2 * sizeof (int),
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }

        // 释放掉原来存储于 Host 内存上的数据。
        delete[] crv->crvData;

        // 更新模版数据，把新的在当前 Device 上申请的数据和相关数据写入模版元数
        // 据中。
        crv->crvData = devptr;

        // 判断 smoothed 曲线是否有效；如果有效，操作同曲线坐标数据。
        if (crvCud->smCurveIsValid == SMCURVE_VALID) {
            // 申请空间。     
            cuerrcode = cudaMalloc((void **)(&devsm), 
                                   crv->curveLength * 2 * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(devptr);
                return CUDA_ERROR;
            }

            // 拷贝数据。
            cuerrcode = cudaMemcpy(devsm, crv->smCurveCordiXY, 
                                   crv->curveLength * 2 * sizeof (float),
                                   cudaMemcpyHostToDevice);
            if (cuerrcode != cudaSuccess) {
                cudaFree(devptr);
                cudaFree(devsm);
                return CUDA_ERROR;
            }

            // 释放位于 host 端的原数据并将位于 device 端的数据重新赋值给曲线。
            delete[] crv->smCurveCordiXY;
            crv->smCurveCordiXY = devsm;
        }
        
        // 判断曲线是斜率否有效；如果有效，操作同曲线坐标数据。
        if (crvCud->tangentIsValid == TANGENT_VALID) {
            // 申请空间。   
            cuerrcode = cudaMalloc((void **)(&devtangent), 
                                   crv->curveLength * 2 * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(devptr);
                if (crvCud->smCurveIsValid == SMCURVE_VALID)
                    cudaFree(devsm);
                return CUDA_ERROR;
            }

            // 拷贝数据。
            cuerrcode = cudaMemcpy(devtangent, crv->tangent, 
                                   crv->curveLength * sizeof (float),
                                   cudaMemcpyHostToDevice);
            if (cuerrcode != cudaSuccess) {
                cudaFree(devptr);
                if (crvCud->smCurveIsValid == SMCURVE_VALID)
                    cudaFree(devsm);
                cudaFree(devtangent);
                return CUDA_ERROR;
            }
                                                                               
            // 释放位于 host 端的原数据并将位于 device 端的数据重新赋值给曲线。
            delete[] crv->tangent;
            crv->tangent = devtangent;
        }

        // 更新模版数据。
        crvCud->deviceId = curdevid;

        // 操作完毕，返回。
        return NO_ERROR;

    } else if (crvCud->deviceId != curdevid) {
        // 对于数据存在其他 Device 的情况，仍旧要在当前 Device 上申请数据空间，
        // 并从另一个 Device 上拷贝数据到新申请的当前 Device 的数据空间中。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 新申请的当前 Device 上的坐标数据。
        int *devptr;    

        // 新的 smoothed 曲线数据空间。
        float *devsm;   

        // 新的斜率数据空间。
        float *devtangent;         
        
        // 在当前 Device 上申请坐标数据空间。
        cuerrcode = cudaMalloc((void **)(&devptr), 
                               crv->curveLength * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // 将数据从曲线原来的存储位置拷贝到当前的 Device 上。
        cuerrcode = cudaMemcpyPeer(devptr, curdevid,
                                   crv->crvData, crvCud->deviceId,
                                   crv->curveLength * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }

        // 释放掉曲线在原来的 Device 上的数据。
        cudaFree(crv->crvData);

        // 将新的曲线数据信息写入到曲线元数据中。
        crv->crvData = devptr;

        // 判断 smoothed 曲线是否有效；如果有效，操作同曲线坐标数据。
        if (crvCud->smCurveIsValid == SMCURVE_VALID) {
            // 申请空间。     
            cuerrcode = cudaMalloc((void **)(&devsm), 
                                   crv->curveLength * 2 * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(devptr);
                return CUDA_ERROR;
            }

            // 拷贝数据。
            cuerrcode = cudaMemcpyPeer(devsm, curdevid,
                                       crv->smCurveCordiXY, crvCud->deviceId,
                                       crv->curveLength * 2 * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(devptr);
                cudaFree(devsm);
                return CUDA_ERROR;
            }

            // 释放位于 其他 Device 端的原数据并将位于 device 端的数据重新赋
            // 值给曲线。
            cudaFree(crv->smCurveCordiXY);
            crv->smCurveCordiXY = devsm;
        }

        // 判断曲线是斜率否有效；如果有效，操作同曲线坐标数据。
        if (crvCud->tangentIsValid == TANGENT_VALID) {
            // 申请空间。   
            cuerrcode = cudaMalloc((void **)(&devtangent), 
                                   crv->curveLength * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(devptr);
                if (crvCud->smCurveIsValid == SMCURVE_VALID)
                    cudaFree(devsm);
                return CUDA_ERROR;
            }

            // 拷贝数据。
            cuerrcode = cudaMemcpyPeer(devptr, curdevid,
                                       crv->crvData, crvCud->deviceId,
                                       crv->curveLength * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                cudaFree(devptr);
                if (crvCud->smCurveIsValid == SMCURVE_VALID)
                    cudaFree(devsm);
                cudaFree(devtangent);
                return CUDA_ERROR;
            }
                                                                               
            // 释放位于 其他 Device 端的原数据并将位于 device 端的数据重新赋
            // 值给曲线。
            cudaFree(crv->tangent);
            crv->tangent = devtangent;
        }   

        crvCud->deviceId = curdevid;

        // 操作完成，返回。
        return NO_ERROR;
    }

    // 对于其他情况，即曲线数据本来就在当前 Device 上，则直接返回，不进行任何的
    // 操作。
    return NO_ERROR;
}

// Host 静态方法：copyToCurrentDevice（将曲线拷贝到当前 Device 内存上）
__host__ int CurveBasicOp::copyToCurrentDevice(Curve *srccrv, Curve *dstcrv)
{
    // 检查输入曲线是否为 NULL。
    if (srccrv == NULL || dstcrv == NULL)
        return NULL_POINTER;

    // 如果输出曲线为 NULL 或者和输入曲线为同一个曲线，则转而调用对应的 
    // In-place 版本的函数。
    if (dstcrv == NULL || dstcrv == srccrv)
        return copyToCurrentDevice(srccrv);

    // 获取 srccrv 和 dstcrv 对应的 CurveCuda 型指针。
    CurveCuda *srccrvCud = CURVE_CUDA(srccrv);
    CurveCuda *dstcrvCud = CURVE_CUDA(dstcrv);

    // 用来存放旧的 dstcrv 数据，使得在拷贝操作失败时可以恢复为原来的可用的数据
    // 信息，防止系统进入一个混乱的状态。
    CurveCuda olddstcrvCud = *dstcrvCud;  // 旧的 dstcrv 数据
    bool reusedata = true;                // 记录是否重用了原来的曲线数据空间。
                                          // 该值为 ture，则原来的数据空间被重
                                          // 用，不需要在之后释放数据，否则
                                          // 需要在最后释放旧的空间。

    // 如果源曲线是一个空曲线，则不进行任何操作，直接报错。
    if (srccrv->crvData == NULL || srccrv->curveLength == 0)
        return INVALID_DATA;

    // 检查曲线所在的地址空间是否合法，如果曲线所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srccrvCud->deviceId >= devcnt || dstcrvCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果目标曲线中存在有数据，则需要根据情况，若原来的数据不存储在当前的
    // Device 上，或者即使存储在当前的 Device 上，但数据尺寸不匹配，则需要释放
    // 掉原来申请的空间，以便重新申请合适的内存空间。此处不进行真正的释放操作，
    // 其目的在于当后续操作出现错误时，可以很快的恢复 dstcrv 中原来的信息，使得
    // 整个系统不会处于一个混乱的状态，本函数会在最后，确定 dstcrv 被成功的更换
    // 为了新的数据以后，才会真正的将原来的曲线数据释放掉。
    if (dstcrvCud->deviceId != curdevid) {
        // 对于数据存在 Host 与其他的 Device 上，则直接释放掉原来的数据空间。
        reusedata = false;
        dstcrv->crvData = NULL;
    } else if (dstcrv->curveLength != srccrv->curveLength) {
        // 对于数据存在于当前 Device 上，则需要检查数据的尺寸是否和源曲线相匹
        // 配。如果目标曲线和源曲线的尺寸不匹配则仍旧需要释放目标曲线原来的数据
        // 空间。
        reusedata = false;
        dstcrv->crvData = NULL;
    }

    // 将目标曲线的属性更改为源曲线的属性。
    dstcrv->closed = srccrv->closed;
    dstcrv->startCordiX = srccrv->startCordiX;
    dstcrv->startCordiY = srccrv->startCordiY;
    dstcrv->endCordiX = srccrv->endCordiX;
    dstcrv->endCordiY = srccrv->endCordiY;
    dstcrv->maxCordiX = srccrv->maxCordiX;
    dstcrv->minCordiX = srccrv->minCordiX;
    dstcrv->maxCordiY = srccrv->maxCordiY;
    dstcrv->minCordiY = srccrv->minCordiY;
    dstcrv->aveX = srccrv->aveX;
    dstcrv->aveY = srccrv->aveY;
    dstcrv->curveLength = srccrv->curveLength;

    // 将目标曲线的实际点数量更改为源曲线的实际点数量。
    dstcrvCud->capacity = srccrvCud->capacity;

    // 更改目标曲线的数据存储位置为当前 Device。
    dstcrvCud->deviceId = curdevid;

    // 更新目标曲线的标记数据。
    dstcrvCud->smCurveIsValid = srccrvCud->smCurveIsValid;
    dstcrvCud->tangentIsValid = srccrvCud->tangentIsValid;

    // 如果目标曲线需要重新申请空间（因为上一步将无法重用原来内存空间的情况的
    // dstcrv->crvData 都置为 NULL，因此此处通过检查 dstcrv->crvData == NULL来
    // 确定是否需要重新申请空间），则在当前的 Device 内存中申请空间。
    cudaError_t cuerrcode;
    if (dstcrv->crvData == NULL) {
        // 申请坐标数据的内存空间
        cuerrcode = cudaMalloc((void **)(&dstcrv->crvData),
                               srccrv->curveLength * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess) {
            // 如果空间申请操作失败，则恢复原来的目标曲线的数据，以防止系统进入
            // 混乱状态。
            *dstcrvCud = olddstcrvCud;
            return CUDA_ERROR;
        }

        // 判断 smoothed 曲线是否有效；如果有效，则申请空间。
        if (dstcrvCud->smCurveIsValid == SMCURVE_VALID) {
            cuerrcode = cudaMalloc((void **)(&dstcrv->smCurveCordiXY),
                                   srccrv->curveLength * 2 * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                // 如果空间申请操作失败，则恢复原来的目标曲线的数据，以防止系统进入
                // 混乱状态。
                cudaFree(dstcrv->crvData);
                *dstcrvCud = olddstcrvCud;
                return CUDA_ERROR;
            }
        }
        
        // 判断曲线斜率数据是否有效；如果有效，则申请空间。
        if (dstcrvCud->tangentIsValid == TANGENT_VALID) {
            cuerrcode = cudaMalloc((void **)(&dstcrv->tangent),
                                   srccrv->curveLength * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                // 如果空间申请操作失败，则恢复原来的目标曲线的数据，以防止系统进入
                // 混乱状态。
                cudaFree(dstcrv->crvData);
                if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                    cudaFree(dstcrv->smCurveCordiXY);
                *dstcrvCud = olddstcrvCud;
                return CUDA_ERROR;
            }
        }
    }

    // 将数据拷贝到目标曲线内。
    if (srccrvCud->deviceId < 0) {
        // 如果源曲线存储于 Host，则通过 cudaMemcpy 将数据从 Host 拷贝到 Device
        // 上。
        // 拷贝数据
        cuerrcode = cudaMemcpy(dstcrv->crvData, srccrv->crvData,
                               srccrv->curveLength * 2 * sizeof (int),
                               cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            // 报错处理分为两个步骤：第一步，如果数据空间不是重用原来的数据空间时，
            // 则需要释放掉新申请的数据空间；第二步，恢复原来的目标曲线的元数据。
            if (!reusedata) {
                cudaFree(dstcrv->crvData);
                if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                    cudaFree(dstcrv->smCurveCordiXY);
                if (dstcrvCud->tangentIsValid == TANGENT_VALID)
                    cudaFree(dstcrv->tangent);
            }

            *dstcrvCud = olddstcrvCud;
            return CUDA_ERROR;
        }

        // 判断 smoothed 曲线是否有效；如果有效，则拷贝数据。
        if (dstcrvCud->smCurveIsValid == SMCURVE_VALID) {
            cuerrcode = cudaMemcpy(dstcrv->smCurveCordiXY, srccrv->smCurveCordiXY,
                                   srccrv->curveLength * 2 * sizeof (float),
                                   cudaMemcpyHostToDevice);
            if (cuerrcode != cudaSuccess) {
                // 报错处理分为两个步骤：第一步，如果数据空间不是重用原来的数据空间时，
                // 则需要释放掉新申请的数据空间；第二步，恢复原来的目标曲线的元数据。
                if (!reusedata) {
                    cudaFree(dstcrv->crvData);
                    cudaFree(dstcrv->smCurveCordiXY);
                    if (dstcrvCud->tangentIsValid == TANGENT_VALID)
                        cudaFree(dstcrv->tangent);
                }

                *dstcrvCud = olddstcrvCud;
                return CUDA_ERROR;
            }
        }
        
        // 判断曲线斜率数据是否有效；如果有效，则拷贝数据。
        if (dstcrvCud->tangentIsValid == TANGENT_VALID) {
            cuerrcode = cudaMemcpy(dstcrv->tangent, srccrv->tangent,
                                   srccrv->curveLength * sizeof (float),
                                   cudaMemcpyHostToDevice);
            if (cuerrcode != cudaSuccess) {
                // 报错处理分为两个步骤：第一步，如果数据空间不是重用原来的数据空间时，
                // 则需要释放掉新申请的数据空间；第二步，恢复原来的目标曲线的元数据。
                if (!reusedata) {
                    cudaFree(dstcrv->crvData);
                    if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                        cudaFree(dstcrv->smCurveCordiXY);
                    cudaFree(dstcrv->tangent);
                }

                *dstcrvCud = olddstcrvCud;
                return CUDA_ERROR;
            }
        }
    } else {
        // 如果源曲线存储于 Device，则通过 cudaMemcpyPeer 进行设备间的数据拷
        // 贝。
        // 拷贝曲线数据
        cuerrcode = cudaMemcpyPeer(dstcrv->crvData, curdevid,
                                   srccrv->crvData, srccrvCud->deviceId,
                                   srccrv->curveLength * 2 * sizeof (int));
        if (cuerrcode != cudaSuccess) {
            // 报错处理分为两个步骤：第一步，如果数据空间不是重用原来的数据空间时，
            // 则需要释放掉新申请的数据空间；第二步，恢复原来的目标曲线的元数据。
            if (!reusedata) {
                cudaFree(dstcrv->crvData);
                if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                    cudaFree(dstcrv->smCurveCordiXY);
                if (dstcrvCud->tangentIsValid == TANGENT_VALID)
                    cudaFree(dstcrv->tangent);
            }

            *dstcrvCud = olddstcrvCud;
            return CUDA_ERROR;
        }

        // 判断 smoothed 曲线是否有效；如果有效，则拷贝数据。
        if (dstcrvCud->smCurveIsValid == SMCURVE_VALID) {
            cuerrcode = cudaMemcpyPeer(dstcrv->smCurveCordiXY, curdevid,
                                       srccrv->smCurveCordiXY, srccrvCud->deviceId,
                                       srccrv->curveLength * 2 * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                // 报错处理分为两个步骤：第一步，如果数据空间不是重用原来的数据空间时，
                // 则需要释放掉新申请的数据空间；第二步，恢复原来的目标曲线的元数据。
                if (!reusedata) {
                    cudaFree(dstcrv->crvData);
                    cudaFree(dstcrv->smCurveCordiXY);
                    if (dstcrvCud->tangentIsValid == TANGENT_VALID)
                        cudaFree(dstcrv->tangent);
                }

                *dstcrvCud = olddstcrvCud;
                return CUDA_ERROR;
            }
        }

        // 判断曲线斜率数据是否有效；如果有效，则拷贝数据。
        if (dstcrvCud->tangentIsValid == TANGENT_VALID) {
            cuerrcode = cudaMemcpyPeer(dstcrv->tangent, curdevid,
                                       srccrv->tangent, srccrvCud->deviceId,
                                       srccrv->curveLength * sizeof (float));
            if (cuerrcode != cudaSuccess) {
                // 报错处理分为两个步骤：第一步，如果数据空间不是重用原来的数据空间时，
                // 则需要释放掉新申请的数据空间；第二步，恢复原来的目标曲线的元数据。
                if (!reusedata) {
                    cudaFree(dstcrv->crvData);
                    if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                        cudaFree(dstcrv->smCurveCordiXY);
                    cudaFree(dstcrv->tangent);
                }

                *dstcrvCud = olddstcrvCud;
                return CUDA_ERROR;
            }
        }
    }

    // 到此步骤已经说明新的曲线数据空间已经成功的申请并拷贝了新的数据，因此，旧
    // 的数据空间已毫无用处。本步骤就是释放掉旧的数据空间以防止内存泄漏。这里，
    // 作为拷贝的 olddstcrvCud 是局部变量，因此相应的元数据会在本函数退出后自动
    // 释放，不用理会。
    if (olddstcrvCud.crvMeta.crvData != NULL) {
        if (olddstcrvCud.deviceId < 0) {
            // 如果旧数据空间是 Host 内存上的，则需要无条件释放。
            delete [] olddstcrvCud.crvMeta.crvData;
            if(olddstcrvCud.smCurveIsValid == SMCURVE_VALID)
                delete [] olddstcrvCud.crvMeta.smCurveCordiXY;
            if (olddstcrvCud.tangentIsValid == TANGENT_VALID)
                delete [] olddstcrvCud.crvMeta.tangent;
        } else if (!reusedata) {
            // 如果旧数据空间不是当前 Device 内存上的其他 Device 内存上的数据，
            // 则也需要无条件的释放。
            cudaSetDevice(olddstcrvCud.deviceId);
            cudaFree(olddstcrvCud.crvMeta.crvData);
            if(olddstcrvCud.smCurveIsValid == SMCURVE_VALID)
                cudaFree(olddstcrvCud.crvMeta.smCurveCordiXY);
            if (olddstcrvCud.tangentIsValid == TANGENT_VALID)
                cudaFree(olddstcrvCud.crvMeta.tangent);
            cudaSetDevice(curdevid);
        }
    }

    return NO_ERROR;
}

// Host 静态方法：copyToHost（将曲线拷贝到 Host 内存上）
__host__ int CurveBasicOp::copyToHost(Curve *crv)
{
    // 检查曲线是否为 NULL。
    if (crv == NULL)
        return NULL_POINTER;

    // 根据输入参数的 Curve 型指针，得到对应的 CurveCuda 型数据。
    CurveCuda *crvCud = CURVE_CUDA(crv);

    // 检查曲线所在的地址空间是否合法，如果曲线所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (crvCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果曲线是一个不好含数据的空曲线，则报错。
    if (crv->crvData == NULL || crv->curveLength == 0)
        return UNMATCH_IMG;

    // 对于不同的情况，将曲线数据拷贝到当前设备上。
    if (crvCud->deviceId < 0) {
        // 如果曲线位于 Host 内存上，则不需要进行任何操作。
        return NO_ERROR;
    } else {
        // 如果曲线的数据位于 Device 内存上，则需要在 Host 的内存空间上申请空
        // 间，然后将数据拷贝到 Host 上。
        cudaError_t cuerrcode;   // CUDA 调用返回的错误码。

        // 新的数据空间，在 Host 上。
        int *hostptr;    
        
        // 新的 smooth 曲线数据空间。
        float *hostsm; 

        // 新的切线斜率数据空间。
        float *hosttangent;            

        // 在 Host 上申请坐标数据空间。
        hostptr = new int[crv->curveLength * 2];
        if (hostptr == NULL)
            return OUT_OF_MEM;

        // 判断 smooth 曲线标记是否有效；如果有效则申请空间。
        if (crvCud->smCurveIsValid == SMCURVE_VALID) {
            // 在 Host 上申请新的 smooth 曲线数据空间。
            hostsm = new float[crv->curveLength * 2];
            if (hostsm == NULL) {
                delete [] hostptr;
                return OUT_OF_MEM;
            }
        }
        
        // 判断曲线斜率标记是否有效；如果有效则申请空间。
        if (crvCud->tangentIsValid ==TANGENT_VALID) {
            // 在 Host 上申请新的切线斜率数据空间。
            hosttangent = new float[crv->curveLength];
            if (hosttangent == NULL) {
                delete [] hostptr;
                if (crvCud->smCurveIsValid == SMCURVE_VALID) 
                    delete [] hostsm;
                return OUT_OF_MEM;
            }
        }
        

        // 将设备切换到数据所在的 Device 上。
        cudaSetDevice(crvCud->deviceId);

        // 拷贝曲线数据
        cuerrcode = cudaMemcpy(hostptr, crv->crvData, 
                               crv->curveLength * 2 * sizeof (int),
                               cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
            // 后报错返回。
            delete [] hostptr;
            if (crvCud->smCurveIsValid == SMCURVE_VALID)
                delete [] hostsm;
            if (crvCud->tangentIsValid == TANGENT_VALID)
                delete [] hosttangent;
            return CUDA_ERROR;
        }

        // 释放掉原来存储于 Device 内存上的曲线数据。
        cudaFree(crv->crvData);

        // 判断 smooth 曲线标记是否有效；如果有效则拷贝数据。
        if (crvCud->smCurveIsValid == SMCURVE_VALID)
        {
            cuerrcode = cudaMemcpy(hostsm, crv->smCurveCordiXY, 
                                   crv->curveLength * 2 * sizeof (float),
                                   cudaMemcpyDeviceToHost);
            if (cuerrcode != cudaSuccess) {
                // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
                // 后报错返回。
                delete [] hostptr;
                delete [] hostsm;
                if (crvCud->tangentIsValid == TANGENT_VALID)
                    delete [] hosttangent;
                return CUDA_ERROR;
            }

            // 释放掉原来存储于 Device 内存上的 smooth 曲线数据。
            cudaFree(crv->smCurveCordiXY);
        }
        
        // 判断曲线斜率标记是否有效；如果有效则拷贝数据。
        if (crvCud->tangentIsValid ==TANGENT_VALID) {
            cuerrcode = cudaMemcpy(hosttangent, crv->tangent, 
                                   crv->curveLength * sizeof (float),
                                   cudaMemcpyDeviceToHost);
            if (cuerrcode != cudaSuccess) {
                // 如果拷贝失败，则需要释放掉刚刚申请的内存空间，以防止内存泄漏。之
                // 后报错返回。
                delete [] hostptr;
                if (crvCud->smCurveIsValid == SMCURVE_VALID)
                    delete [] hostsm;
                delete [] hosttangent;
                return CUDA_ERROR;
            }
            // 释放掉原来存储于 Device 内存上的曲线斜率数据。
            cudaFree(crv->tangent);
        }

        // 对 Device 内存的操作完毕，将设备切换回当前 Device。
        cudaSetDevice(curdevid);

        // 更新曲线数据，把新的在当前 Device 上申请的数据和相关数据写入曲线元数
        // 据中。
        crv->crvData = hostptr;
        if (crvCud->smCurveIsValid == SMCURVE_VALID)
            crv->smCurveCordiXY = hostsm;
        if (crvCud->tangentIsValid == TANGENT_VALID)
            crv->tangent = hosttangent;
        crvCud->deviceId = -1;

        // 操作完毕，返回。
        return NO_ERROR;
    }

    // 程序永远也不会到达这个分支，因此如果到达这个分支，则说明系统紊乱。对于多
    // 数编译器来说，会对此句报出不可达语句的 Warning，因此这里将其注释掉，以防
    // 止不必要的 Warning。
    //return UNKNOW_ERROR;
}

// Host 静态方法：copyToHost（将曲线拷贝到 Host 内存上）
__host__ int CurveBasicOp::copyToHost(Curve *srccrv, Curve *dstcrv)
{
    // 检查输入曲线是否为 NULL。
    if (srccrv == NULL || dstcrv == NULL)
        return NULL_POINTER;

    // 如果输出曲线为 NULL 或者和输入曲线同为一个曲线，则调用对应的 In-place 版
    // 本的函数。
    if (dstcrv == NULL || dstcrv == srccrv)
        return copyToHost(srccrv);

    // 获取 srccrv 和 dstcrv 对应的 CurveCuda 型指针。
    CurveCuda *srccrvCud = CURVE_CUDA(srccrv);
    CurveCuda *dstcrvCud = CURVE_CUDA(dstcrv);

    // 用来存放旧的 dstcrv 数据，使得在拷贝操作失败时可以恢复为原来的可用的数据
    // 信息，防止系统进入一个混乱的状态。
    CurveCuda olddstcrvCud = *dstcrvCud;  // 旧的 dstcrv 数据
    bool reusedata = true;                // 记录是否重用了原来的曲线数据空间。
                                          // 该值为 true，则原来的数据空间被重
                                          // 用。不需要在之后释放数据，否则需要
                                          // 释放就的空间。

    // 如果源曲线是一个空曲线，则不进行任何操作，直接报错。
    if (srccrv->crvData == NULL || srccrv->curveLength == 0)
        return INVALID_DATA;

    // 检查曲线所在的地址空间是否合法，如果曲线所在地址空间不属于 Host 或任何一
    // 个 Device，则该函数报“数据溢出”错误，表示无法处理。
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srccrvCud->deviceId >= devcnt || dstcrvCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // 获取当前 Device ID。
    int curdevid;
    cudaGetDevice(&curdevid);

    // 如果目标曲线中存在有数据，则需要根据情况，若原来的数据不存储在 Host 上，
    // 或者即使存储在 Host 上，但数据尺寸不匹配，则需要释放掉原来申请的空间，以
    // 便重新申请合适的内存空间。此处不进行真正的释放操作，其目的在于当后续操作
    // 出现错误时，可以很快的恢复 dstcrv 中原来的信息，使得整个系统不会处于一个
    // 混乱的状态，本函数会在最后，确定 dstcrv 被成功的更换为了新的数据以后，才
    // 会真正的将原来的曲线数据释放掉。
    if (dstcrvCud->deviceId >= 0) {
        // 对于数据存在于 Device 上，则亦直接释放掉原来的数据空间。
        reusedata = false;
        dstcrv->crvData = NULL;
    } else if (srccrv->curveLength != dstcrv->curveLength) {
        // 对于数据存在于 Host 上，则需要检查数据的尺寸是否和源曲线相匹配。检查
        // 的标准：源曲线和目标曲线的尺寸相同时，可重用原来的空间。
        reusedata = false;
        dstcrv->crvData = NULL;
    }

    // 将目标曲线的属性更改为源曲线的属性。
    dstcrv->closed = srccrv->closed;
    dstcrv->startCordiX = srccrv->startCordiX;
    dstcrv->startCordiY = srccrv->startCordiY;
    dstcrv->endCordiX = srccrv->endCordiX;
    dstcrv->endCordiY = srccrv->endCordiY;
    dstcrv->maxCordiX = srccrv->maxCordiX;
    dstcrv->minCordiX = srccrv->minCordiX;
    dstcrv->maxCordiY = srccrv->maxCordiY;
    dstcrv->minCordiY = srccrv->minCordiY;
    dstcrv->aveX = srccrv->aveX;
    dstcrv->aveY = srccrv->aveY;
    dstcrv->curveLength = srccrv->curveLength;

    // 将目标曲线的实际点数量更改为源曲线的实际点数量。
    dstcrvCud->capacity = srccrvCud->capacity;

    // 更改目标曲线的数据存储位置为 Host。
    dstcrvCud->deviceId = -1;

    // 更新目标曲线的标记数据。
    dstcrvCud->smCurveIsValid = srccrvCud->smCurveIsValid;
    dstcrvCud->tangentIsValid = srccrvCud->tangentIsValid;

    // 如果目标曲线的 crvData == NULL，说明目标曲线原本要么是一个空曲线，要么目
    // 标曲线原本的数据空间不合适，需要重新申请。这时，需要为目标曲线重新在 
    // Host 上申请一个合适的数据空间。
    if (dstcrv->crvData == NULL) {
        // 申请曲线数据空间
        dstcrv->crvData = new int[srccrv->curveLength * 2];
        if (dstcrv->crvData == NULL) {
            // 如果申请内存的操作失败，则再报错返回前需要将旧的目标曲线数据
            // 恢复到目标曲线中，以保证系统接下的操作不至于混乱。
            *dstcrvCud = olddstcrvCud;
            return OUT_OF_MEM;
        }

        // 如果 smooth 曲线标记有效，则申请数据空间。
        if (dstcrvCud->smCurveIsValid == SMCURVE_VALID) {
            dstcrv->smCurveCordiXY = new float[srccrv->curveLength * 2];
            if (dstcrv->crvData == NULL) {
                // 如果申请内存的操作失败，则再报错返回前需要将旧的目标曲线数据
                // 恢复到目标曲线中，以保证系统接下的操作不至于混乱。
                delete [] dstcrv->crvData;
                *dstcrvCud = olddstcrvCud;
                return OUT_OF_MEM;
            }
        }
        
        // 如果曲线斜率标记有效，则申请数据空间。
        if (dstcrvCud->tangentIsValid == TANGENT_VALID) {
            dstcrv->tangent = new float[srccrv->curveLength];
            if (dstcrv->crvData == NULL) {
                // 如果申请内存的操作失败，则再报错返回前需要将旧的目标曲线数据
                // 恢复到目标曲线中，以保证系统接下的操作不至于混乱。
                delete [] dstcrv->crvData;
                if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                    delete [] dstcrv->smCurveCordiXY;
                *dstcrvCud = olddstcrvCud;
                return OUT_OF_MEM;
            }
        }
            
    }

    // 将坐标数据从源曲线中拷贝到目标曲线中。
    if (srccrvCud->deviceId < 0) {
        // 如果源曲线数据存储于 Host 内存，则直接使用 C 标准支持库中的 memcpy
        // 完成拷贝。

        // 将 srccrv 内的坐标数据拷贝到 dstcrv 中。memcpy 不返回错误，因此，没
        // 有进行错误检查。
        memcpy(dstcrv->crvData, srccrv->crvData,
               srccrv->curveLength * 2 * sizeof (int));
        if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
            memcpy(dstcrv->smCurveCordiXY, srccrv->smCurveCordiXY,
                   srccrv->curveLength * 2 * sizeof (float));
        if (dstcrvCud->tangentIsValid == TANGENT_VALID)
            memcpy(dstcrv->tangent, srccrv->tangent,
                   srccrv->curveLength * sizeof (float));

    } else {
        // 如果源曲线数据存储于 Device 内存（无论是当前 Device 还是其他的 
        // Device），都是通过 CUDA 提供的函数进行拷贝。
        cudaError_t cuerrcode;  // CUDA 调用返回的错误码。

        // 首先切换到 srccrv 坐标数据所在的 Device，以方便进行内存操作。
        cudaSetDevice(srccrvCud->deviceId);

        // 这里使用 cudaMemcpy 将 srccrv 中处于 Device 上的数据拷贝到 dstcrv 中
        // 位于 Host 的内存空间上面。
        // 拷贝坐标数据
        cuerrcode = cudaMemcpy(dstcrv->crvData, srccrv->crvData, 
                               srccrv->curveLength * 2 * sizeof (int),
                               cudaMemcpyDeviceToHost);

        if (cuerrcode != cudaSuccess) {
            // 如果拷贝操作失败，则再报错退出前，需要将旧的目标曲线数据恢复到目
            // 标曲线中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
            // 防止内存泄漏。最后，还需要把 Device 切换回来，以免整个程序乱套。
            if (!reusedata) {
                delete [] dstcrv->crvData;
                if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                    delete [] dstcrv->smCurveCordiXY;
                if (dstcrvCud->tangentIsValid == TANGENT_VALID)
                    delete [] dstcrv->tangent;
            }
            *dstcrvCud = olddstcrvCud;
            cudaSetDevice(curdevid);
            return CUDA_ERROR;
        }

        // 如果 smooth 曲线标记有效，则拷贝数据。
        if (dstcrvCud->smCurveIsValid == SMCURVE_VALID){
            cuerrcode = cudaMemcpy(dstcrv->smCurveCordiXY, srccrv->smCurveCordiXY, 
                                   srccrv->curveLength * 2 * sizeof (float),
                                   cudaMemcpyDeviceToHost);

            if (cuerrcode != cudaSuccess) {
                // 如果拷贝操作失败，则再报错退出前，需要将旧的目标曲线数据恢复到目
                // 标曲线中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
                // 防止内存泄漏。最后，还需要把 Device 切换回来，以免整个程序乱套。
                if (!reusedata) {
                    delete [] dstcrv->crvData;
                    delete [] dstcrv->smCurveCordiXY;
                    if (dstcrvCud->tangentIsValid == TANGENT_VALID)
                        delete [] dstcrv->tangent;
                }
                *dstcrvCud = olddstcrvCud;
                cudaSetDevice(curdevid);
                return CUDA_ERROR;
            }
        }

        // 如果曲线斜率标记有效，则拷贝数据。
        if (dstcrvCud->tangentIsValid == TANGENT_VALID) {
            cuerrcode = cudaMemcpy(dstcrv->tangent, srccrv->tangent, 
                                   srccrv->curveLength * sizeof (float),
                                   cudaMemcpyDeviceToHost);

            if (cuerrcode != cudaSuccess) {
                // 如果拷贝操作失败，则再报错退出前，需要将旧的目标曲线数据恢复到目
                // 标曲线中。此外，如果数据不是重用的，则需要释放新申请的数据空间，
                // 防止内存泄漏。最后，还需要把 Device 切换回来，以免整个程序乱套。
                if (!reusedata) {
                    delete [] dstcrv->crvData;
                    if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                        delete [] dstcrv->smCurveCordiXY;
                    delete [] dstcrv->tangent;
                }
                *dstcrvCud = olddstcrvCud;
                cudaSetDevice(curdevid);
                return CUDA_ERROR;
            }
        }

        // 对内存操作完毕后，将设备切换回当前的 Device。
        cudaSetDevice(curdevid);
    }

    // 到此步骤已经说明新的曲线数据空间已经成功的申请并拷贝了新的数据，因此，旧
    // 的数据空间已毫无用处。本步骤就是释放掉旧的数据空间以防止内存泄漏。这里，
    // 作为拷贝的 olddstcrvCud 是局部变量，因此相应的元数据会在本函数退出后自动
    // 释放，不用理会。
    if (olddstcrvCud.crvMeta.crvData != NULL) {
        if (olddstcrvCud.deviceId > 0) {
            // 如果旧数据是存储于 Device 内存上的数据，则需要无条件的释放。
            cudaSetDevice(olddstcrvCud.deviceId);
            cudaFree(olddstcrvCud.crvMeta.crvData);
            if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                cudaFree(olddstcrvCud.crvMeta.smCurveCordiXY);
            if (dstcrvCud->tangentIsValid == TANGENT_VALID)
                cudaFree(olddstcrvCud.crvMeta.tangent);
            cudaSetDevice(curdevid);
        } else if (!reusedata) {
            // 如果旧数据就在 Host 内存上，则对于 reusedata 未置位的情况进行释
            // 放，因为一旦置位，旧的数据空间就被用于承载新的数据，则不能释放。
            delete [] olddstcrvCud.crvMeta.crvData;
            if (dstcrvCud->smCurveIsValid == SMCURVE_VALID)
                delete [] olddstcrvCud.crvMeta.smCurveCordiXY;
            if (dstcrvCud->tangentIsValid == TANGENT_VALID)
                delete [] olddstcrvCud.crvMeta.tangent;
        }
    }

    // 处理完毕，退出。
    return NO_ERROR;
}

// Host 静态方法：assignData（为曲线数据赋值）
__host__ int CurveBasicOp::assignData(Curve *crv, int *data, size_t count)
{
    // 检查输入曲线是否为 NULL
    if (crv == NULL)
        return NULL_POINTER;

    // 检查给定的曲线中坐标点数量
    if (count < 1)
        return INVALID_DATA;

    // 局部变量，错误码。
    int errcode;

    // 计算曲线坐标内的各属性值
    errcode = _calCrvXY(crv, data, count);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    // 获取 crv 对应的 CurveCuda 型数据。
    CurveCuda *crvCud = CURVE_CUDA(crv);

    // 如果曲线内存在数据，则将释放原数据。
    if (crv->crvData != NULL) {
        delete [] crv->crvData;
    }

    // 为新的曲线数据开空间。
    crv->crvData = new int[count * 2];

    // 拷贝数据。
    memcpy(crv->crvData, data, count * 2 * sizeof (int));

    // 将新的曲线数据信息写入到曲线元数据中。
    crv->curveLength = count;
    crvCud->capacity = count;

    return NO_ERROR;
}

// Host 静态方法：setSmCurveValid（设置 smoothing 曲线有效）
__host__ int CurveBasicOp::setSmCurveValid(Curve *crv) 
{
    // 检查输入曲线是否为 NULL
    if (crv == NULL)
        return NULL_POINTER;

    // 判断曲线是否已经有数据，有则返回错误；
    // 此函数必须执行在为曲线数据申请空间之前。
    if (crv->crvData != NULL)
        return INVALID_DATA;

    // 获取 crv 对应的 CurveCuda 型指针。
    CurveCuda *crvCud = CURVE_CUDA(crv);    

    // 设置 smoothing 曲线有效。
    crvCud->smCurveIsValid = SMCURVE_VALID;

    return NO_ERROR;
}

// Host 静态方法：setTangentValid（设置曲线斜率有效）
__host__ int CurveBasicOp::setTangentValid(Curve *crv)
{
    // 检查输入曲线是否为 NULL
    if (crv == NULL)
        return NULL_POINTER;

    // 判断曲线是否已经有数据，有则返回错误；
    // 此函数必须执行在为曲线数据申请空间之前。
    if (crv->crvData != NULL)
        return INVALID_DATA;

    // 获取 crv 对应的 CurveCuda 型指针。
    CurveCuda *crvCud = CURVE_CUDA(crv);    

    // 设置曲线斜率有效。
    crvCud->tangentIsValid = TANGENT_VALID;

    return NO_ERROR;
}
