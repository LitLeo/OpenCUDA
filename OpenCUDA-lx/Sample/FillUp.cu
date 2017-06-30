// FillUp.cu
// 实现对输入图像像素的处理

#include <iostream> 
using namespace std;

#include "FillUp.h"
#include "ErrorCode.h"

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// static变量：_defTpl
// 当用户未定义有效的模板时，使用此默认模板，默认为 3 x 3。
static Template *_defTpl = NULL;

// Kernel 函数：_fillupFirstKer（实现检查像素操作）
// 检查一个像素的邻域，若其邻域同时存在 l 像素和 v 像素，
// 当 v 像素的个数大于等于某一值时，就将所设标记置为 1，
// 否则置为0。
static __global__ void     // Kernel 函数无返回值。
_fillupFirstKer(
       ImageCuda inimg,    // 输入图像。
       ImageCuda flagimg,  // 标记图像。
       Template tpl,       // 模板。
       unsigned char l,    // 主要像素。
       unsigned char v,    // 替换像素。
       int percentage      // 定值，即 r * w * w。
);

// Kernel 函数：_fillupFinalKer（将修改过的像素输出到图像）
// 当 v 的像素的个数大于等于某一值时，将标记置为 1 后，把标记
// 为 1 的像素值 v 输出到图像上。
static __global__ void     // Kernel 函数无返回值。
_fillupFinalKer(
       ImageCuda inimg,    // 输入图像。
       ImageCuda flagimg,  // 标记图像。
       ImageCuda outimg,   // 输出图像。
       Template tpl,       // 模板。
       unsigned char v     // 替换像素。
);

// Kernel 函数：_fillupLastKer（对 l 像素的领域检查，并修改像素）
// 对所有 l 像素的 8 个领域进行检查，如果它的 8 个领
// 域当中有 5 个或 5 个以上的 v 的像素值，就将 v 的像素值赋给 l。 
static __global__ void     // Kernel 函数无返回值。
_fillupLastKer(
       ImageCuda inimg,    // 输入图像。
       ImageCuda outimg,   // 输出图像。  
       unsigned char l,    // 主要像素。
       unsigned char v,    // 替换像素。
       int lastpercentage  // 定值，值为 5。
);

// Host 函数：_initDefTemplate（初始化默认的模板指针）
// 函数初始化默认模板指针 _defTpl，如果原来模板不为空，则直接返回，否则初始化
// 为 3 x 3 的默认模板。
static __host__ Template*  // 返回值：返回默认模板指针 _defTpl。 
_initDefTemplate();

// Host 函数：_preOp（在算法操作前进行预处理）
// 在进行处理像素操作前，先进行预处理，包括：（1）对输入和输出图像
// 进行数据准备，包括申请当前Device存储空间；（2）对模板进行处理，包
// 申请当前Device存储空间。
static __host__ int     // 返回值：函数是否正确执行，若正确执行，返回
                        // NO_ERROR 。
_preOp(
        Image *inimg,   // 输入图像。
        Image *outimg,  // 输出图像。
        Template *tp    // 模板。
);

// Host 函数：_adjustRoiSize（调整 ROI 子图的大小）
// 调整 ROI 子图的大小，使输入和输出的子图大小统一
static __host__ void       // 无返回值。
_adjustRoiSize(
        ImageCuda *inimg,  // 输入图像。
        ImageCuda *outimg  // 输出图像。
);

// Host 函数：_getBlockSize（获取 Block 和 Grid 的尺寸）
// 根据默认的 Block 尺寸，使用最普通的线程划分方法获取 Grid 的尺寸
static __host__ int      // 返回值：函数是否正确执行，若正确执行，返回
                         // NO_ERROR 。
_getBlockSize(
        int width,       // 需要处理的宽度。
        int height,      // 需要处理的高度。
        dim3 *gridsize,  // 计算获得的 Grid 的尺寸。
        dim3 *blocksize  // 计算获得的 Block 的尺寸。
);
 
// 构造函数：FillUp
__host__ FillUp::FillUp(Template *tp)
{
    setTemplate(tp);
}

// 构造函数：FillUp
__host__ FillUp::FillUp(Template *tp, unsigned char l, unsigned char v ,          
			int maxw, float r)
{
    // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法的
    // 初始值而使系统进入一个未知的状态。
    setTemplate(tp);
    this->l = 255;   // l 值默认为 255。
    this->v = 0;     // v 值默认为 0。
    this->maxw = 5;  // maxw 值默认为 5。
    this->r = 0.2;   // r 值默认为 0.2。
	
    // 根据参数列表中的值设定成员变量的初值
    setL(l);
    setV(v);
    setMaxw(maxw);
    setR(r);
}

// 成员方法：getTemplate
__host__ Template* FillUp::getTemplate() const
{
    // 如果模板指针和默认模板指针相同，则返回空
    if (this->tpl == _defTpl) 
        return NULL;

    // 否则返回设置的模板指针
    return this->tpl;
}

// 成员方法：setTemplate
__host__ int FillUp::setTemplate(Template *tp)
{
    if (tp == NULL) {
        // 如果 tp 为空，则只用默认的模板指针
        this->tpl = _initDefTemplate();
    } else {
        // 否则将 tp 赋值给 tpl
        this->tpl = tp;
    }
    return NO_ERROR;
}

// Kernel 函数：_fillupFirstKer（实现检查像素操作）
static __global__ void _fillupFirstKer(
        ImageCuda inimg, ImageCuda flagimg, Template tpl,
	unsigned char l, unsigned char v, int percentage)
{   
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，
    // c 表示 column， r 表示 row）。由于采用并行度缩减策略 ，
    // 令一个线程处理 4 个输出像素，这四个像素位于统一列的相邻 4 
    // 行上，因此，对于 r 需要进行乘 4 的计算。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 用来保存临时像素点的坐标的 x 和 y 分量
    int dx, dy; 

    // 用来记录当前模板所在的位置的指针
    int *curtplptr = tpl.tplData;

    // 用来记录当前输入图像所在位置的指针
    unsigned char *curinptr;
    
    // 用来保存 4 个像素中 v 的个数
    int sum[4] = { 0 };

    // 存放临时像素点的像素值
    unsigned char pixel;
    
    // 用来记录输出图像所在位置的指针 
    unsigned char *outptr;
	
    // 存放输出像素的值
    unsigned char outvalue[4] = { 1, 1, 1, 1 };
    
    // 扫描模板范围内的每个输入图像的像素点
    for (int i = 0; i < tpl.count; i++) {
        // 计算当前模板位置所在像素的 x 和 y 分量，模板使用相邻的两个下标的
        // 数组表示一个点，所以使当前模板位置的指针作加一操作 
        dx = c + *(curtplptr++);
        dy = r + *(curtplptr++);

        // 先判断当前像素的 x 分量是否越界，如果越界，则跳过，扫描下一个模板
        // 点,如果没有越界，则分别处理当前列的相邻的 4 个像素
        if (dx < 0 || dx >= inimg.imgMeta.width)
            continue;
    
        if (dx != c || dy != r) {
            // 得到当前位置的像素值，并且判断该像素值是否等于 v,如果相等，
	    // 就计数它。如果该像素值既不等于 v，也不等于 l,就将标记置为 0。 
                
            // 得到第一个点
            curinptr = inimg.imgMeta.imgData + dx + dy * inimg.pitchBytes;
            if (dy >= 0 && dy < inimg.imgMeta.height) {
                pixel = *curinptr;
                (pixel == v) ? (sum[0]++) : 0;
                (pixel != v && pixel != l) ? (outvalue[0] = 0) : 0;
            }

	    // 处理当前列的剩下的 3 个像素
	    for (int j = 1; j < 4; j++) {
                // 获取当前列的下一行的像素的位置
		curinptr = curinptr + inimg.pitchBytes;
		    
		// 使 dy 加一，得到当前要处理的像素的 y 分量
	        dy++;
	            
		// 检测 dy 是否越界
	        if (dy >= 0 && dy < inimg.imgMeta.height) {
	            // 判断该像素值是否等于 v,如果相等，就计数它。如果该像素值
	            // 既不等于 v，也不等于 l,就将标记置为 0。
	            pixel = *curinptr;
	            (pixel == v) ? (sum[j]++) : 0;
	            (pixel != v && pixel != l) ? (outvalue[j] = 0) : 0;
		}
	    }
        }
    }
    
    // 如果 v 的像素的数量大于 percentage 这个定值，就将标记置为 1，
    // 如果不是，将标记置为 0。
    // 获取对应的第一个输出图像的位置
    outptr = flagimg.imgMeta.imgData + r * flagimg.pitchBytes + c;
    *outptr = ((sum[0] >= percentage) && outvalue[0]);

    // 检测 y 分量是否越界,如果越界,直接返回。
    if (++r >= flagimg.imgMeta.height)
        return;
    outptr = outptr + flagimg.pitchBytes;
    *outptr = ((sum[1] >= percentage) && outvalue[1]);

    // 检测 y 分量是否越界,如果越界,直接返回。
    if (++r >= flagimg.imgMeta.height)
        return;
    outptr = outptr + flagimg.pitchBytes;
    *outptr = ((sum[2] >= percentage) && outvalue[2]);

    // 检测 y 分量是否越界,如果越界,直接返回。
    if (++r >= flagimg.imgMeta.height)
        return;
    outptr = outptr + flagimg.pitchBytes;
    *outptr = ((sum[3] >= percentage) && outvalue[3]);
}

// Kernel 函数:_fillupFinalKer（将修改过的像素输出到图像）
static __global__ void _fillupFinalKer(
        ImageCuda inimg, ImageCuda flagimg, ImageCuda outimg, 
	Template tpl, unsigned char v)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示 
    // column， r 表示 row）。由于采用并行度缩减策略 ，令一个线程处理 4 
    // 个输出像素，这四个像素位于统一列的相邻 4 行上，因此，对于 r 需要进行
    // 乘 4 的计算
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
  
    // 定义变量
    int j;
    unsigned char *rowinptr[4];
    unsigned char outvalue[4];
    unsigned char curvalue;
    unsigned char *outptr;
    unsigned char *curflagptr;
	
    // 用来记录当前模板所在的位置的指针
    int *curtplptr = tpl.tplData;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃
    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    // 用来保存临时像素点的坐标的 x 和 y 分量
    int dx, dy; 
	
    // 一个线程处理四个像素点。
    // 计算当前像素点的第一个像素值，然后计算同一列的剩下的三个像素值。
    rowinptr[0] = inimg.imgMeta.imgData + c + r * inimg.pitchBytes;
    outvalue[0] = *rowinptr[0];
    rowinptr[1] = rowinptr[0] + inimg.pitchBytes;
    outvalue[1] = *rowinptr[1];
    rowinptr[2] = rowinptr[1] + inimg.pitchBytes;
    outvalue[2] = *rowinptr[2];
    rowinptr[3] = rowinptr[2] + inimg.pitchBytes;
    outvalue[3] = *rowinptr[3];
	
    for (j = 0; j < tpl.count; j++) {
        // 计算当前模板位置所在像素的 x 和 y 分量，模板使用相邻的两个下标的
        // 数组表示一个点，所以使当前模板位置的指针作加一操作 
        dx = c + *(curtplptr++);
        dy = r + *(curtplptr++);
            
        // 如果找到标志为 1 的像素，就把像素 v 输出到图像
        if (dx >= 0 && dx < flagimg.imgMeta.width) {
            curflagptr = flagimg.imgMeta.imgData + dx + dy * flagimg.pitchBytes;
            
            if (dy >= 0 && dy < flagimg.imgMeta.height) {
                curvalue = *curflagptr;
                (curvalue != 0) ? (outvalue[0] = v) : 0;
            }
            
            // 使 dy 加一，得到当前要处理的像素的 y 分量
            dy++;
            // 获取当前列的下一行的像素的位置
	    curflagptr = curflagptr + flagimg.pitchBytes;

            // 第二个像素点。
            if (dy >= 0 && dy < flagimg.imgMeta.height) {
                curvalue = *curflagptr;
                (curvalue != 0) ? (outvalue[1] = v) : 0;
            }
	    
	    // 使 dy 加一，得到当前要处理的像素的 y 分量
	    dy++;
	    // 获取当前列的下一行的像素的位置
	    curflagptr = curflagptr + flagimg.pitchBytes;
              
            // 第三个像素点。
            if (dy >= 0 && dy < flagimg.imgMeta.height) {
                curvalue = *curflagptr;
                (curvalue != 0) ? (outvalue[2] = v) : 0;
            }
	    
	    // 使 dy 加一，得到当前要处理的像素的 y 分量
	    dy++;
	    // 获取当前列的下一行的像素的位置
	    curflagptr = curflagptr + flagimg.pitchBytes;

            // 第四个像素点。
            if (dy >= 0 && dy < flagimg.imgMeta.height) {
                curvalue = *curflagptr;
                (curvalue != 0) ? (outvalue[3] = v) : 0;
            }
        } 
    }
  
    // 将像素值赋给输出图像。
    outptr = outimg.imgMeta.imgData + r * outimg.pitchBytes + c;
    *outptr = outvalue[0];
	
    // 检测 y 分量是否越界,如果越界,直接返回。
    if (++r >= outimg.imgMeta.height)
        return;
    outptr = outptr + outimg.pitchBytes;
    *outptr = outvalue[1];
	
    // 检测 y 分量是否越界,如果越界,直接返回。
    if (++r >= outimg.imgMeta.height)
        return;
    outptr = outptr + outimg.pitchBytes;
    *outptr = outvalue[2];
	
    // 检测 y 分量是否越界,如果越界,直接返回。
    if (++r >= outimg.imgMeta.height)
        return;
    outptr = outptr + outimg.pitchBytes;
    *outptr = outvalue[3];
}

// Kernel 函数：_fillupLastKer（对 l 像素的领域检查，并修改像素）
static __global__ void _fillupLastKer(
        ImageCuda inimg, ImageCuda outimg, unsigned char l, 
        unsigned char v, int lastpercentage)
{
    // 计算当前线程的位置。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 定义变量
    int i, j;
    int sum = 0;
    unsigned char *inptr, *outptr;
    unsigned char orivalue, curvalue;
	
    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算
    // 资源，另一方面防止由于段错误导致程序崩溃。
    if (r >= inimg.imgMeta.height || c >= inimg.imgMeta.width)
        return;
	   
    // 得到输入图像和输出图像当前像素的位置。
    inptr = inimg.imgMeta.imgData + c + r * inimg.pitchBytes;
    outptr = outimg.imgMeta.imgData + c + r * outimg.pitchBytes;
    orivalue = *inptr;

    // 如果当前像素值不等于 l,就直接将此值赋给输出图像。
    if (orivalue != l){
        *outptr = orivalue;
        return;
    }
    
    // 对所有 l 像素的 8 个领域进行检查，如果它的 8 个领域当中有 5 个
    // 或 5 个以上的 v 的像素值，就将 v 的像素值赋给 l。 
    for (j = r - 1; j <= r + 1; j++) {
        for (i = c - 1; i <= c + 1; i++) {
            // 判断当前像素是否越界。
            if (j >= 0 && j < inimg.imgMeta.height && 
                i >= 0 && i < inimg.imgMeta.width) {
                // 得到当前位置的像素值。
                curvalue = *(inimg.imgMeta.imgData + i + j * inimg.pitchBytes);
                // 如果其值等于 v,就计数它。
                sum += ((curvalue == v) ? 1 : 0);
            }
        }
    }
    
    // 如果v的像素的数量大于 5 个以上，就将原来的值置成 v,
    // 如果不大于，就输出原来的值。
    *outptr = (sum > lastpercentage) ? v : orivalue;
}

// Host 函数：_initDefTemplate（初始化默认的模板指针）
static __host__ Template* _initDefTemplate()
{
    // 如果 _defTpl 不为空，说明已经初始化了，则直接返回
    if (_defTpl != NULL)
        return _defTpl;

    // 如果 _defTpl 为空，则初始化为 3 x 3 的模板
    TemplateBasicOp::newTemplate(&_defTpl);
    TemplateBasicOp::makeAtHost(_defTpl, 9);
    
    // 分别处理每一个点
    for (int i = 0; i < 9; i++) {
        // 分别计算每一个点的横坐标和纵坐标
        _defTpl->tplData[2 * i] = i % 3 - 1;
        _defTpl->tplData[2 * i + 1] = i / 3 - 1;
    }
    return _defTpl;
}

// Host 函数：_preOp（在算法操作前进行预处理）
static __host__ int _preOp(Image *inimg, Image *outimg, Template *tp)
{
    // 局部变量，错误码
    int errcode;  

    // 将输入图像拷贝到 Device 内存中
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;

    // 将输出图像拷贝到 Device 内存中
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 计算 roi 子图的宽和高
        int roiwidth = inimg->roiX2 - inimg->roiX1; 
        int roiheight = inimg->roiY2 - inimg->roiY1;
        // 如果输出图像无数据，则会创建一个和输出图像子图像尺寸相同的图像
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, roiwidth, 
                                                    roiheight); 
        // 如果创建图像依然操作失败，则返回错误
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 将模板拷贝到 Device 内存中
    errcode = TemplateBasicOp::copyToCurrentDevice(tp);
    if (errcode != NO_ERROR)
        return errcode;   

    return NO_ERROR;
}

// Host 函数：_adjustRoiSize（调整输入和输出图像的 ROI 的大小）
inline static __host__ void _adjustRoiSize(ImageCuda *inimg, ImageCuda *outimg)
{
    if (inimg->imgMeta.width > outimg->imgMeta.width)
        inimg->imgMeta.width = outimg->imgMeta.width;
    else
        outimg->imgMeta.width = inimg->imgMeta.width;

    if (inimg->imgMeta.height > outimg->imgMeta.height)
        inimg->imgMeta.height = outimg->imgMeta.height;
    else
        outimg->imgMeta.height = inimg->imgMeta.height;
}

// Host 函数：_getBlockSize（获取 Block 和 Grid 的尺寸）
inline static __host__ int _getBlockSize(int width, int height, dim3 *gridsize,
                                         dim3 *blocksize)
{
    // 检测 girdsize 和 blocksize 是否是空指针
    if (gridsize == NULL || blocksize == NULL)
        return NULL_POINTER; 

    // blocksize 使用默认的尺寸
    blocksize->x = DEF_BLOCK_X;
    blocksize->y = DEF_BLOCK_Y;

    // 使用最普通的方法划分 Grid 
    gridsize->x = (width + blocksize->x - 1) / blocksize->x;
    gridsize->y = (height + blocksize->y * 4 - 1) / (blocksize->y * 4);

    return NO_ERROR;
}

// 成员方法：fillUp 
__host__ int FillUp::fillUp(Image *inimg, Image *outimg)
{
    // 局部变量，错误码。
    int errcode;  
    dim3 gridsize;
    dim3 blocksize;
    Image *flagimg;
    
    // 检查输入图像，输出图像，以及模板是否为空
    if (inimg == NULL || outimg == NULL || tpl == NULL)
        return NULL_POINTER;
    
    // 新建一个中间标记图像。
    errcode = ImageBasicOp::newImage(&flagimg);
    if (errcode != NO_ERROR) 
        return errcode;

    // 为新建的图像在设备端分配空间。
    errcode = ImageBasicOp::makeAtCurrentDevice(flagimg, inimg->width,
	                                        inimg->height);
    if (errcode != NO_ERROR) {
        // 计算 roi 子图的宽和高
        int roiwidth = inimg->roiX2 - inimg->roiX1; 
        int roiheight = inimg->roiY2 - inimg->roiY1;
        
        // 如果输出图像无数据，则会创建一个和输出图像子图像尺寸相同的图像
        errcode = ImageBasicOp::makeAtCurrentDevice(flagimg, roiwidth, 
                                                    roiheight); 
        // 如果创建图像依然操作失败，则返回错误
        if (errcode != NO_ERROR)
            return errcode;
    }
	
    // 用 r 和 maxw 得到 percentage 的值。
    int percentage = r * maxw * maxw;
	
    // 对输入图像，输出图像和模板进行预处理
    errcode = _preOp(inimg, outimg, tpl);
    if (errcode != NO_ERROR)
        return errcode; 

    // 提取输入图像的 ROI 子图像
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
		
    // 提取标记图像的 ROI 子图像
    ImageCuda flagsubimgCud;
    errcode = ImageBasicOp::roiSubImage(flagimg, &flagsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的 ROI 子图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 调整输入和输出图像的 ROI 子图，使大小统一
    _adjustRoiSize(&insubimgCud, &outsubimgCud);
    _adjustRoiSize(&insubimgCud, &flagsubimgCud);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    errcode = _getBlockSize(outsubimgCud.imgMeta.width,
                            outsubimgCud.imgMeta.height,
                            &gridsize, &blocksize);
    if (errcode != NO_ERROR) 
        return errcode;

    // 首先调用 _fillupFirstKer 这个 Kernel 函数进行处理像素操作
    _fillupFirstKer<<<gridsize, blocksize>>>(insubimgCud, flagsubimgCud, 
                                             *tpl, l, v, percentage); 

    // 再调用这个 Kernel 函数进行输出
    _fillupFinalKer<<<gridsize, blocksize>>>(insubimgCud, flagsubimgCud, 
                                             outsubimgCud,  *tpl, v); 

    // 将输出图像拷贝到 Host 上
    errcode = ImageBasicOp::copyToHost(outimg);

    // 释放标记图像
    errcode = ImageBasicOp::deleteImage(flagimg);
    if (errcode != NO_ERROR)
        return errcode;

    return errcode;
}

// 成员方法：fillUpAdv
__host__ int FillUp::fillUpAdv(Image *inimg, Image *outimg, int *stateflag)
{
    // 局部变量，错误码。
    int errcode;        
    dim3 gridsize;
    dim3 blocksize; 
	
    // 计算迭代次数。
    int step = 0;  
    Image *midimg1, *midimg2;
    int curw, nextw;
    Image *curin, *curout, *tempimg;
    Template *tl;
    
    // 检查输入图像，输出图像，以及模板是否为空
    if (inimg == NULL || outimg == NULL || tpl == NULL)
        return NULL_POINTER;
    
    // 申请需要用到的中间图片。  
    ImageBasicOp::newImage(&midimg1);
    ImageBasicOp::makeAtHost(midimg1, inimg->width, inimg->height);
   
    ImageBasicOp::newImage(&midimg2);
    ImageBasicOp::makeAtHost(midimg2, inimg->width, inimg->height);
    
    ImageBasicOp::newImage(&curout);
    ImageBasicOp::makeAtHost(curout, inimg->width, inimg->height);
    
    ImageBasicOp::newImage(&curin);
    ImageBasicOp::makeAtHost(curin, inimg->width, inimg->height);

    // 设置最后一步处理时的比例值。
    int lastpercentage = 4; 
    
    // 将当前图像的指针指向输入图像，对输入图像进行计算。
    curin = inimg;
    curout = midimg1;
	
    // 设置开始时模板的大小赋给当前模板。
    curw = maxw;
    nextw = maxw >> 1;
  
    // 如果开始时模板的尺寸小于 3，则不做处理。
    if (curw <= 3)
        curout = inimg;

    // 如果模板的尺寸大于 3，做以下处理。
    while (curw > 3) {
        // 申请模板
        errcode = TemplateBasicOp::newTemplate(&tl);
		
        // 在主机内存中初始化模板。
        errcode = TemplateBasicOp::makeAtHost(tl, curw * curw);
        if (errcode != NO_ERROR) 
            return errcode;        
		
        // 为模板赋值。 
        for (int i = 0; i < curw * curw; i++) {
            tl->tplData[2 * i] = i % curw - curw / 2;
            tl->tplData[2 * i + 1] = i / curw - curw / 2;
        }
		
        // 调用set函数为成员变量赋值。
        setTemplate(tl);
        
        // 调用FillUp算法。
        errcode = fillUp(curin, curout);
        if (errcode != NO_ERROR) 
            return errcode;
        
        // 如果模板下一次的大小小于 3，就跳出此循环。
        if (nextw <= 3){
            step += 1;
            break;
        }

        // 交换当前输入图像和当前输出图像。
        tempimg = curin;
        curin = curout;
        curout = tempimg;

        // 如果有更多 fillUp 操作，将 midimg2 赋给 curout。
        if (step == 0 && nextw > 3)
            curout = midimg2;

        // 释放模板。
        TemplateBasicOp::deleteTemplate(tl);

        // 对当前模板尺寸赋予新的值。
        curw = nextw;
		
        // 将当前模板的尺寸减为它的一半。
        nextw = nextw / 2;
		
        // 迭代次数增加 1.
        step += 1;
    } 
    
    if (maxw == 3) {
        // 将输入图像拷贝到 Device 内存中
        errcode = ImageBasicOp::copyToCurrentDevice(inimg);
        if (errcode != NO_ERROR)
            return errcode;
    }

    // 将输出图像拷贝到 Device 内存中
    errcode = ImageBasicOp::copyToCurrentDevice(outimg);
    if (errcode != NO_ERROR) {
        // 计算 roi 子图的宽和高
        int roiwidth = inimg->roiX2 - inimg->roiX1; 
        int roiheight = inimg->roiY2 - inimg->roiY1;
        // 如果输出图像无数据，则会创建一个和输出图像子图像尺寸相同的图像
        errcode = ImageBasicOp::makeAtCurrentDevice(outimg, roiwidth, 
                                                    roiheight); 
        // 如果创建图像依然操作失败，则返回错误
        if (errcode != NO_ERROR)
            return errcode;
    }
    
    // 将下一步输入图像curout拷贝至当前设备。
    ImageBasicOp::copyToCurrentDevice(curout);
    
    // 提取输入图像的 ROI 子图像
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(curout, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 提取输出图像的 ROI 子图像
    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR)
        return errcode;
        
    // 调整输入和输出图像的 ROI 子图，使大小统一
    _adjustRoiSize(&insubimgCud, &outsubimgCud);
	
    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (curout->width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (curout->height + blocksize.y - 1) / blocksize.y;
    if (errcode != NO_ERROR) 
        return errcode;  

    // 调用 Kernel 函数进行操作
    _fillupLastKer<<<gridsize, blocksize>>>(insubimgCud, outsubimgCud,
                                            l, v, lastpercentage); 
                                            
    // 将输出图像拷贝到 Host 上
    errcode = ImageBasicOp::copyToHost(outimg);
    if (errcode != NO_ERROR) 
        return errcode;
    
    // 计算最终的迭代次数。
    if (stateflag != NULL)
        *stateflag = step;
        
    // 删除申请的中间图片。
    ImageBasicOp::deleteImage(midimg1);
    ImageBasicOp::deleteImage(midimg2);	
    
    return errcode;
}

