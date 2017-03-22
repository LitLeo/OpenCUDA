// ConsolidateAndIdentifyContours.cu
// 利用图像中图形的轮廓信息检测物体。

#include "ConsolidateAndIdentifyContours.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;



// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 宏：RED_MAC_COUNT
// 定义了重复进行边缘检测的次数。
#define RED_MAC_COUNT  4

// 宏：DILATE_TPL_SHAPE 和 SEARCH_TPL_SHAPE
// 定义了膨胀操作和搜索标准轮廓时的临域模板形状
#define DILATE_TPL_SHAPE  TF_SHAPE_CIRCLE
#define SEARCH_TPL_SHAPE  TF_SHAPE_BOX

// 宏：OBJ_IDX_OFFSET
// 定义了输出到标记轮廓中的标号偏移量。
#define OBJ_IDX_OFFSET  100


// 全局变量：_redMacDiffSize
// 不同的边缘检测器的检测半径。
static unsigned _redMacDiffSize[RED_MAC_COUNT] = { 3, 5, 7, 9 };

// 静态成员变量：redMachine（边缘检测处理机）
RobustEdgeDetection *ConsolidateAndIdentifyContours::redMachine = NULL;


// Kernel 函数：_searchPrimitiveContourKer（匹配并标记轮廓）
// 从检测中的轮廓图像中匹配标准轮廓图像中的相关轮廓。被匹配上的边缘将被标记成对
// 应的标号信息，未匹配上的轮廓被标记为异常点。
static __global__ void
_searchPrimitiveContourKer(
        ImageCuda inimg,     // 轮廓输入图像
        ImageCuda outimg,    // 标记后的输出图像
        ImageCuda abnorimg,  // 异常点图像
        ImageCuda prmtcont,  // 标准轮廓图像
        ImageCuda prmtreg,   // 物体区域图像
        unsigned trackrad    // 搜索半径
);


// Host 成员方法：initRedMachine（初始化边缘检测处理机）
__host__ int ConsolidateAndIdentifyContours::initRedMachine()
{
    // 如果 redMachine 不为 NULL，则说明已经初始化过了。
    if (redMachine != NULL)
        return NO_ERROR;

    // 申请指定个数的边缘检测器。
    redMachine = new RobustEdgeDetection[RED_MAC_COUNT];
    if (redMachine == NULL)
        return OUT_OF_MEM;

    // 迭代设定各个边缘检测器的检测半径。
    int errcode = NO_ERROR;
    for (int i = 0; i < RED_MAC_COUNT; i++) {
        int curerrcode = redMachine[i].setDiffsize(_redMacDiffSize[i]);

        // 最终返回的错误码应该是更加严重的错误。
        if (curerrcode < errcode)
            errcode = curerrcode;
    }

    // 初始化完毕，返回赋值过程中累积下来的错误码。
    return errcode;
}

// Host 成员方法：initMorphMachine（初始化膨胀处理机）
__host__ int ConsolidateAndIdentifyContours::initMorphMachine()
{
    // 取出膨胀处理机中原来的模板
    Template *oldtpl = morphMachine.getTemplate();

    // 通过模版工厂生成新的模板，这里暂时适用方形模板。
    int errcode;
    Template *curtpl = NULL;
    size_t boxsize = this->dilationRad * 2 + 1;
    errcode = TemplateFactory::getTemplate(&curtpl, DILATE_TPL_SHAPE, boxsize);
    if (errcode != NO_ERROR)
        return errcode;

    // 将新生成的模板放入膨胀处理机中
    errcode = morphMachine.setTemplate(curtpl);
    if (errcode != NO_ERROR)
        return errcode;

    // 如果原始的模板不为 NULL，则需要释放对该模板的占用。
    if (oldtpl != NULL)
        TemplateFactory::putTemplate(oldtpl);

    // 处理完毕，返回。
    return NO_ERROR;
}

// Host 成员方法：getCsldtContoursImg（获取轮廓图像）
__host__ int ConsolidateAndIdentifyContours::getCsldtContoursImg(
        Image *inimg, Image *outimg)
{
    // 检查输入输出图像是否为 NULL。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 由于后续的失败处理要清除所申请的临时图像，因此设计一个宏来简化代码，方便
    // 代码维护。
#define CAIC_GETCONT_ERRFREE(errcode)  do {                           \
            for (int _i_cge = 0; _i_cge < RED_MAC_COUNT; _i_cge++) {  \
                if (edgetmpimg[_i_cge] != NULL)                       \
                    ImageBasicOp::deleteImage(edgetmpimg[_i_cge]);    \
            }                                                         \
            return (errcode);                                         \
        } while (0)

    // 该迭代完成两件事情：第一是完成边缘检测输出图像的创建；另一件则是调用边缘
    // 检测算法完成边缘检测。
    int errcode = NO_ERROR;
    Image *edgetmpimg[RED_MAC_COUNT] = { NULL };

    for (int i = 0; i < RED_MAC_COUNT; i++) {
        // 创建边缘检测的输出图像。
        errcode = ImageBasicOp::newImage(edgetmpimg + i);
        if (errcode != NO_ERROR)
            CAIC_GETCONT_ERRFREE(errcode);
        //cout << "AA" << i << endl;

        // 调用边缘检测方法，获得边缘图像。
        errcode = redMachine[i].detectEdgeSA(inimg, edgetmpimg[i], NULL);
        if (errcode != NO_ERROR)
            CAIC_GETCONT_ERRFREE(errcode);
        //cout << "BB" << i << endl;
    }

    // 合并在不同参数下边缘检测的结果。
    errcode = combineMachine.combineImageMax(edgetmpimg, RED_MAC_COUNT, outimg);
    if (errcode != NO_ERROR)
        CAIC_GETCONT_ERRFREE(errcode);

    // 对边缘进行膨胀操作，以连接一些断线的点
    errcode = morphMachine.dilate(outimg, edgetmpimg[0]);
    if (errcode != NO_ERROR)
        CAIC_GETCONT_ERRFREE(errcode);

    // 对膨胀后的边缘进行细化操作，在此恢复其单一线宽。
    errcode = thinMachine.thinMatlabLike(edgetmpimg[0], outimg);
    if (errcode != NO_ERROR)
        CAIC_GETCONT_ERRFREE(errcode);

    // 由于边缘检测算法输出的图像为二值图像，因此不需要再进行二值化处理了
    //errcode = binMachine.binarize(outimg);
    //if (errcode != NO_ERROR)
    //    CAIC_GETCONT_ERRFREE(errcode);

    // 处理完毕返回。
    CAIC_GETCONT_ERRFREE(NO_ERROR);
#undef CAIC_GETCONT_ERRFREE
}

// Kernel 函数：_searchPrimitiveContourKer（匹配并标记轮廓）
__global__ void _searchPrimitiveContourKer(
        ImageCuda inimg, ImageCuda outimg, ImageCuda abnorimg,
        ImageCuda prmtcont, ImageCuda prmtreg, unsigned trackrad)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示线程处理的像素点的坐标
    // 的 x 和 y 分量（其中，c 表示 column；r 表示 row）。
    unsigned c =  blockIdx.x * blockDim.x + threadIdx.x;
    unsigned r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，一
    // 方面防止由于段错误导致的程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 计算输入输出图像的访存下标。
    unsigned inidx = r * inimg.pitchBytes + c;
    unsigned outidx = r * outimg.pitchBytes + c;
    unsigned abnoridx = r * abnorimg.pitchBytes + c;

    // 读取输入图像中对应的像素值。
    unsigned char inpixel = inimg.imgMeta.imgData[inidx];

    // 如果该点像素值为0，即当前点不在检出的轮廓上，则在输出图像中直接赋 0 值，
    // 不进行后续的搜索处理。
    if (inpixel == 0) {
        outimg.imgMeta.imgData[outidx] = 0;
        abnorimg.imgMeta.imgData[abnoridx] = 0;
        return;
    }

    // 按照由中心向周围的方式搜索当前点对应的物体标记值。最先判断当前点位置在标
    // 准轮廓图像中是否有对应的轮廓标记。如果存在对应的轮廓标记，通过后续的循环
    // 条件中的 prmtcontpxl == 0 则会略过整个后续搜索。
    unsigned prmtcontidx = r * prmtcont.pitchBytes + c;
    unsigned char prmtcontpxl = prmtcont.imgMeta.imgData[prmtcontidx];

    // 由近及远搜索当前位置的临近位置，查看是否可以命中标准轮廓上的点。
    int curr, curc;
    // 外层循环处理半径
    for (int currad = 1; currad <= trackrad && prmtcontpxl == 0; currad++) {
        // 迭代各个半径下的点，由中点向对角点检测。当发现某一标准轮廓点时，退出
        // 循环，不再进一步搜索。
        for (int i = 0; i < trackrad && prmtcontpxl == 0; i++) {
            // 检测上方右侧点
            curc = c + i;
            curr = r - currad;
            prmtcontidx = curr * prmtcont.pitchBytes + curc;
            if (curc < prmtcont.imgMeta.width || curr >= 0)
                prmtcontpxl = prmtcont.imgMeta.imgData[prmtcontidx];
            if (prmtcontpxl != 0)
                break;

            // 检测下方右侧点
            curc = c + i;
            curr = r + currad;
            prmtcontidx = curr * prmtcont.pitchBytes + curc;
            if (curc < prmtcont.imgMeta.width || curr < prmtcont.imgMeta.height)
                prmtcontpxl = prmtcont.imgMeta.imgData[prmtcontidx];
            if (prmtcontpxl != 0)
                break;

            // 检测左方下侧点
            curc = c - currad;
            curr = r + i;
            prmtcontidx = curr * prmtcont.pitchBytes + curc;
            if (curc >= 0 || curr < prmtcont.imgMeta.height)
                prmtcontpxl = prmtcont.imgMeta.imgData[prmtcontidx];
            if (prmtcontpxl != 0)
                break;

            // 检测右方下侧点
            curc = c + currad;
            curr = r + i;
            prmtcontidx = curr * prmtcont.pitchBytes + curc;
            if (curc < prmtcont.imgMeta.width || curr < prmtcont.imgMeta.height)
                prmtcontpxl = prmtcont.imgMeta.imgData[prmtcontidx];
            if (prmtcontpxl != 0)
                break;

            // 根据计算公式，左侧点（上侧点）要比右侧点（下侧点）更加外围一些，
            // 故而统一在稍候检测左侧系列点。
            // 检测上方左侧点
            curc = c - i - 1;
            curr = r - currad;
            prmtcontidx = curr * prmtcont.pitchBytes + curc;
            if (curc >= 0 || curr >= 0)
                prmtcontpxl = prmtcont.imgMeta.imgData[prmtcontidx];
            if (prmtcontpxl != 0)
                break;

            // 检测下方左侧点
            curc = c - i - 1;
            curr = r + currad;
            prmtcontidx = curr * prmtcont.pitchBytes + curc;
            if (curc >= 0 || curr < prmtcont.imgMeta.height)
                prmtcontpxl = prmtcont.imgMeta.imgData[prmtcontidx];
            if (prmtcontpxl != 0)
                break;

            // 检测左方上侧点
            curc = c - currad;
            curr = r - i - 1;
            prmtcontidx = curr * prmtcont.pitchBytes + curc;
            if (curc >= 0 || curr >= 0)
                prmtcontpxl = prmtcont.imgMeta.imgData[prmtcontidx];
            if (prmtcontpxl != 0)
                break;

            // 检测右方上侧点
            curc = c + currad;
            curr = r - i - 1;
            prmtcontidx = curr * prmtcont.pitchBytes + curc;
            if (curc < prmtcont.imgMeta.width || curr >= 0)
                prmtcontpxl = prmtcont.imgMeta.imgData[prmtcontidx];
            if (prmtcontpxl != 0)
                break;
        }
    }

    // 根据是否找到标准轮廓点作出输出动作
    if (prmtcontpxl != 0) {
        // 当匹配到标准轮廓点时，标记输出图像，但不标记异常点图像。
        outimg.imgMeta.imgData[outidx] = prmtcontpxl + OBJ_IDX_OFFSET;
        abnorimg.imgMeta.imgData[abnoridx] = 0;
    } else {
        // 当匹配标准轮廓点失败时，标记该点为异常点，写入异常点图像。
        outimg.imgMeta.imgData[outidx] = 0;
        abnorimg.imgMeta.imgData[abnoridx] = 
                prmtreg.imgMeta.imgData[r * prmtreg.pitchBytes + c];
    }
}

// Host 成员方法：searchPrimitiveContour（匹配并标记轮廓）
__host__ int ConsolidateAndIdentifyContours::searchPrimitiveContour(
        Image *inimg, Image *outimg, Image *abnormalimg)
{
    // 当标准轮廓点和标准区域点未设置时，报错返回。
    if (this->primitiveContour == NULL || this->primitiveRegion == NULL)
        return OP_OVERFLOW;

    // 当输入的参数含有 NULL 指针时，报错返回。
    if (inimg == NULL || outimg == NULL || abnormalimg == NULL)
        return NULL_POINTER;

    // 这一段代码进行图像的预处理工作。图像的预处理主要完成在 Device 内存上为
    // 输入和输出图像准备内存空间，以便盛放数据。
    int errcode;    // 局部变量，错误码
    // 局部变量，本次操作的图像尺寸
    size_t imgw = inimg->roiX2 - inimg->roiX1;
    size_t imgh = inimg->roiY2 - inimg->roiY1;

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将标准轮廓图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(this->primitiveContour);
    if (errcode != NO_ERROR)
        return errcode;

    // 根据标准轮廓图像的 ROI 区域尺寸调整计算尺寸。
    if (imgw > this->primitiveContour->roiX2 - this->primitiveContour->roiX1)
        imgw = this->primitiveContour->roiX2 - this->primitiveContour->roiX1;
    if (imgh > this->primitiveContour->roiY2 - this->primitiveContour->roiY1)
        imgh = this->primitiveContour->roiY2 - this->primitiveContour->roiY1;

    // 将标准区域图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(this->primitiveRegion);
    if (errcode != NO_ERROR)
        return errcode;

    // 根据标准区域图像的 ROI 区域尺寸调整计算尺寸。
    if (imgw > this->primitiveRegion->roiX2 - this->primitiveRegion->roiX1)
        imgw = this->primitiveRegion->roiX2 - this->primitiveRegion->roiX1;
    if (imgh > this->primitiveRegion->roiY2 - this->primitiveRegion->roiY1)
        imgh = this->primitiveRegion->roiY2 - this->primitiveRegion->roiY1;

    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, imgw, imgh);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    } else {
        // 如果输出图片已经含有数据，则用这个数据更新最终参与计算的尺寸
        if (imgw > outimg->roiX2 - outimg->roiX1)
            imgw = outimg->roiX2 - outimg->roiX1;
        if (imgh > outimg->roiY2 - outimg->roiY1)
            imgh = outimg->roiY2 - outimg->roiY1;
    }

    // 将输出图像拷贝入 Device 内存。
    errcode = ImageBasicOp::copyToCurrentDevice(abnormalimg);
    if (errcode != NO_ERROR) {
        // 如果输出图像无数据（故上面的拷贝函数会失败），则会创建一个和输入图
        // 像的 ROI 子图像尺寸相同的图像。
        errcode = ImageBasicOp::makeAtCurrentDevice(abnormalimg, imgw, imgh);
        // 如果创建图像也操作失败，则说明操作彻底失败，报错退出。
        if (errcode != NO_ERROR)
            return errcode;
    } else {
        // 如果输出图片已经含有数据，则用这个数据更新最终参与计算的尺寸
        if (imgw > abnormalimg->roiX2 - abnormalimg->roiX1)
            imgw = abnormalimg->roiX2 - abnormalimg->roiX1;
        if (imgh > abnormalimg->roiY2 - abnormalimg->roiY1)
            imgh = abnormalimg->roiY2 - abnormalimg->roiY1;
    }

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取标准轮廓图像的 ROI 子图像。
    ImageCuda prmtcontsubimgCud;
    errcode = ImageBasicOp::roiSubImage(this->primitiveContour, &prmtcontsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取标准区域图像的 ROI 子图像。
    ImageCuda prmtregsubimgCud;
    errcode = ImageBasicOp::roiSubImage(this->primitiveRegion, &prmtregsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的 ROI 子图像。
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取异常点图像的 ROI 子图像。
    ImageCuda abnorsubimgCud;
    errcode = ImageBasicOp::roiSubImage(abnormalimg, &abnorsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 根据之前得到的计算区域尺寸调整子图像的尺寸。
    insubimgCud.imgMeta.width = prmtcontsubimgCud.imgMeta.width = 
                                prmtregsubimgCud.imgMeta.width = 
                                outsubimgCud.imgMeta.width = 
                                abnorsubimgCud.imgMeta.width = imgw;
    insubimgCud.imgMeta.height = prmtcontsubimgCud.imgMeta.height = 
                                 prmtregsubimgCud.imgMeta.height =
                                 outsubimgCud.imgMeta.height = 
                                 abnorsubimgCud.imgMeta.height = imgh;

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    // 调用核函数，根据阈值 threshold 进行二值化处理。
    _searchPrimitiveContourKer<<<gridsize, blocksize>>>(
            insubimgCud, outsubimgCud, abnorsubimgCud,
            prmtcontsubimgCud, prmtregsubimgCud, this->trackRad);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    // 处理完毕，退出。 
    return NO_ERROR;
}

