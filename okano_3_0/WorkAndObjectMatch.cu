// WorkAndObjectMatch.cu
// WORK and OBJECT 的匹配操作

#include "WorkAndObjectMatch.h"

#include "AffineTrans.h"
#include "DownSampleImage.h"
#include "RoiCopy.h"
#include "RotateTable.h"

// Host 全局常量：_scalModulus（扩缩系数）
// 对 TEST 图像进行扩缩时使用的扩缩系数
// 暂时未实现扩缩操作，先注释掉
//static const float _scalModulus[] = { 0.80f, 0.85f, 0.90f, 0.95f, 1.0f,
//                                      1.05f, 1.10f, 1.15f, 1.20f };

// Host 全局常量：_scalModulusCount（扩缩系数的数量）
// 对 TEST 图像进行扩缩的扩缩系数的数量
static const int _scalModulusCount = 9;

// Host 函数：_shrink（对一组图像分别进行 1 / N 缩小）
// 对给定的一组图像进行 1 / N 缩小
static __host__ int      // 返回值：函数是否正确执行，若函数正确执行，返回
                         // NO_ERROR
_shrink(
        Image **inimg,   // 输入的一组图像
        Image **outimg,  // 输出的一组经过 1 / N 缩小的图像
        int imgcount,    // 输入图像的数量
        int tiems        // 需要缩小的图像的倍数
);

// Host 函数：_deleteBigImageArray（删除一个大图片数组）
// 删除一个存放图片的大数组
static __host__ int   // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR
_deleteBigImageArray(
        Image **img,  // 存放图片的大数组
        int imgcount  // 数组的大小
);

// Host 函数：_createBigImageArray（创建一个大图片的数组）
// 创建一个存放图片的大的数组
static __host__ int    // 返回值：函数是否正确执行，若函数正确执行，返回 
                       // NO_ERROR
_createBigImageArray(
        Image ***img,  // 存放图片的数组的指针
        int imgcount   // 需要创建的数组的大小
);

// Host 函数：_getBestMatchTestIndex（获取相关系数最大的结果）
// 获取相关系数最大的结果
static __host__ int     // 返回值：最大相关系数的索引
_getBestMatchTestIndex(
        MatchRes *res,  // 匹配得到的一组结果
        int rescount    // 结果的数量
);

// Host 函数：_scalAndProjective（对图像进行扩缩和射影变换）
// 对图像进行不同的扩缩和射影变换，扩缩系数由 _scalModulus 指定
static __host__ int      // 返回值：函数是否正确执行，若函数正确执行，返回
                         // NO_ERROR
_scalAndProjective(
        Image *inimg,    // 输入图像
        Image **outimg,  // 经过不同扩缩和射影变换得到的一组输出图像
        int imgcount     // 输出图像的个数
);

// Host 函数：_createAffineImages（生成 2 * anglecount 个回转图像）
// 对 TEST 图像生成 2 * angleCount 个角度的回转图像，回转角的范围是
// angle - 0.2 * anglecount ~ angle + 0.2 * anglecount
static __host__ int               // 返回值：函数是否正确执行，若函数正确执行，
                                  // 返回 NO_ERROR
_createAffineImages(
        Image *test,              // 输入图像
        int twx, int twy,         // 回转中心的横坐标和纵坐标
        float angle,              // 基准回转角
        int rwidth, int rheight,  // 回转后的图像的宽和高
        int anglecount,           // 需要回转的角度的数量
        Image **rotatetest        // 输出图像，回转后的图像
);

// Host 函数：_shrink（对一组图像分别进行 1 / N 缩小）
static __host__ int _shrink(Image **inimg, Image **outimg, int imgcount,
                            int times)
{
    // 判断 inimg 和 outimg 是否为空，若为空，则返回错误
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    int errcode;  // 局部变量，错误码

    // 判断参数是否合法，若不合法，直接返回错误
    if (imgcount <= 0 || times <= 0)
        return INVALID_DATA; 

    // 定义一个用来进行 1 / N 缩小操作的对象
    DownSampleImage shrink(times);
    
    // 依次对每一输入图像进行 1 / N 缩小操作
    for (int i = 0; i < imgcount; i++) {
        // 使用概率法进行 1 / N 缩小
        errcode = shrink.probabilityDownSImg(inimg[i], outimg[i]);
        // 若 1 / N 缩小操作失败，则直接返回
        if (errcode != NO_ERROR)
            return errcode; 
    }

    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}

// Host 函数：_deleteBigImageArray（删除一个大图片数组）
static __host__ int _deleteBigImageArray(Image **img, int imgcount)
{
    // 判断 img 是否为空，若为空，则返回错误
    if (img == NULL)
        return NULL_POINTER;

    // 依次删除数组里的每一张图像
    for (int i = 0; i < imgcount; i++) 
        ImageBasicOp::deleteImage(img[i]);
    // 删除存放图片的数组
    delete [] img;

    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}

// Host 函数：_createBigImageArray（创建一个大图片的数组）
static __host__ int _createBigImageArray(Image ***img, int imgcount)
{
    // 判断 img 是否为空，若为空，则返回错误
    if (img == NULL) 
        return NULL_POINTER;

    int errcode;  // 局部变量，错误码
    // 为数组申请空间
    *img = new Image *[imgcount];
    // 若申请空间失败，返回错误
    if (*img == NULL)
        return OUT_OF_MEM;

    // 依次创建指定数量的图片
    for (int i = 0; i < imgcount; i++) {
        errcode = ImageBasicOp::newImage(&((*img)[i]));
        // 若创建图片失败，删除先前创建的图片，然后返回
        if (errcode != NO_ERROR) {
            _deleteBigImageArray(*img, imgcount);
            return errcode;
        }
    }

    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}

// Host 函数：_getBestMatchTestIndex（获取相关系数最大的结果）
static __host__ int _getBestMatchTestIndex(MatchRes *res, int rescount)
{
    // 判断 res 是否为空，若为空，则返回错误
    if (res == NULL)
        return NULL_POINTER;

    // 默认相关系数最大的位置为第 0 个结果
    int maxindex = 0; 
    // 记录最大相关系数的值，默认为第 0 个结果的相关系数
    float max = res[maxindex].coefficient;
    // 依次和所有其他的结果比较
    for (int i = 1; i < rescount; i++) {
        // 若发现当前结果的相关系数比记录的最大值大，则将当前的相关系数赋值
        // 给 max，同时记录当前结果的索引
        if (max < res[i].coefficient) {
            max = res[i].coefficient;
            maxindex = i;
        }
    }

    // 返回具有最大相关系数的结果的索引
    return maxindex;
}

// Host 函数：_scalAndProjective（对图像进行扩缩和射影变换）
static __host__ int _scalAndProjective(Image *inimg, Image **outimg,
                                       int imgcount)
{
    // 判断 inimg 和 outimg 是否为空，若为空，则返回错误
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    int errcode;  // 局部变量，错误码
    // 此处暂时未实现扩缩和射影变换，只是单纯的拷贝图片
    for (int i = 0; i < imgcount; i++) {
        errcode = ImageBasicOp::copyToHost(inimg, outimg[i]);
        if (errcode != NO_ERROR) { 
            return errcode;
        }
    }

    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}

// Host 函数：_createAffineImages（生成 2 * anglecount 个回转图像）
static __host__ int _createAffineImages(Image *test, int twx, int twy,  
                                        float angle, int rwidth, int rheight, 
                                        int anglecount, Image **rotatetest)
{
    // 判断 test 和 rotatetest 是否为空，若为空，则返回错误
    if (test == NULL || rotatetest == NULL)
        return NULL_POINTER;

    int errcode;  // 局部变量，错误码

    // 检查参数是否合法，若不合法，直接返回错误
    if (test == NULL || rotatetest == NULL)
        return NULL_POINTER;

    // 定义一个进行回转操作的对象
    AffineTrans affine;
    // 设置旋转前的平移向量
    affine.setX(test->width / 2 - twx);
    affine.setY(test->height / 2 - twy);

    // 生成 2 * anglecount 个回转图像，角度分别为
    // angle - 0.2 * anglecount ~ angle + 0.2 * anglecount 
    for (int i = 0; i < 2 * anglecount; i++) {
        // 创建一个临时图片
        Image *t;
        errcode = ImageBasicOp::newImage(&t);
        // 如果申请空间失败，则直接返回错误
        if (errcode != NO_ERROR)
            return errcode;
        // 设置回旋的角度为
        errcode = affine.setAlpha(angle + 0.2 * (i - anglecount));
        // 若设置失败，则直接返回错误
        if (errcode != NO_ERROR)
            return errcode;
        // 对输入图像进行回转 
        errcode = affine.rotateShift(test, t);
        // 若回转失败，则直接返回错误
        if (errcode != NO_ERROR)
            return errcode;
        // 设置回转后的图像的子图的大小
        t->roiX1 = t->width / 2 - rwidth / 2;
        t->roiY1 = t->height / 2 - rheight / 2;
        t->roiX2 = t->roiX1 + rwidth;
        t->roiY2 = t->roiY1 + rheight;
        // 将子图从临时图像中 clip 出来
        RoiCopy copy;
        errcode = copy.roiCopyAtDevice(t, rotatetest[i]);
        // 删除临时图像
        ImageBasicOp::deleteImage(t);
        // 若拷贝子图失败，则直接返回
        if (errcode != NO_ERROR) 
            return errcode;
    }

    // 处理完毕，直接返回
    return NO_ERROR;
}

// 成员方法： 获取 TEST 图像中的 WORK 图像
__host__ int WorkAndObjectMatch::getMatchWork(Image *test, Image *work)
{
    // 检查 test 和 res 是否为空，若为空，则返回错误
    if (test == NULL || work == NULL)
        return NULL_POINTER;

    int errcode;  // 局部变量，错误码

    // 定义一个大的图像数组，为后面的操作提供所需要的图像空间
    Image **bigimagesarray;
    // 标记 bigimagesarray 数组的大小
    int bigimagessize;
    // 图像数组的游标
    Image **cursor;

    // 计算图像数组的大小
    bigimagessize = normalWork->count + 2 * _scalModulusCount + 2 * angleCount;
    // 创建指定大小的图像数组，为后面的操作提供图像空间
    errcode = _createBigImageArray(&bigimagesarray, bigimagessize);
    // 若图像数组创建失败，则直接返回
    if (errcode != NO_ERROR)
        return errcode;

    // 初始化游标为当前图像地址
    cursor = bigimagesarray;

    // 对标准的 WORK 图像进行 1 / 8 缩小

    // 存储 1 / 8 缩小的标准 WORK 图像
    Image **shrinknormalwork;
    // 从图像数组中获取空间
    shrinknormalwork = cursor;
    // 更新游标的位置
    cursor += normalWork->count;

    // 对一组标准的 WORK 图像进行 1 / 8 图像缩小
    errcode = _shrink(normalWork->images, shrinknormalwork, 
                      normalWork->count, 8);
    // 若 1 / 8 图像缩小操作失败，则删除图像数组，然后返回错误
    if (errcode != NO_ERROR) {
        _deleteBigImageArray(bigimagesarray, bigimagessize);
        return errcode;
    }

    // 对 TEST 图像进行扩缩和射影变换

    // 存储 TEST 图像进行不同扩缩系数的扩缩和射影变换得到的图像
    Image **scalprojecttest;
    // 从图像数组中获取空间
    scalprojecttest = cursor;
    // 更新游标的位置
    cursor += _scalModulusCount;

    // 对 TEST 图像进行不同的扩缩和射影变换
    errcode = _scalAndProjective(test, scalprojecttest, _scalModulusCount);
    // 若扩缩和射影变换操作失败，则删除图像数组的空间，然后返回错误
    if (errcode != NO_ERROR) {
        _deleteBigImageArray(bigimagesarray, bigimagessize);
        return errcode;
    }

    // 对 TEST 的各个变形的图像进行 1 / 8 缩小

    // 存储对 TEST 图像的各个变形的图像进行 1 / 8 缩小后的图像
    Image **shrinkscalprojecttest;
    // 从图像数组中获取空间
    shrinkscalprojecttest = cursor;
    // 更新游标的位置
    cursor += _scalModulusCount;
    
    // 对变形的各个 TEST 图像进行 1 / 8 缩小操作
    errcode = _shrink(scalprojecttest, shrinkscalprojecttest,
                      _scalModulusCount, 8);
    // 如果 1 / 8 操作失败，则删除图像数组，然后返回错误
    if (errcode != NO_ERROR) {
        _deleteBigImageArray(bigimagesarray, bigimagessize);
        return errcode;
    }

    // 用 1 / 8 缩小的，具有不同回旋角的 WORK 图像分别在 1 / 8 缩小的 TEST 图像
    // 进行匹配

    // 申请一段内存空间，用来存储匹配得到的结果
    MatchRes *workmatchres = new MatchRes[_scalModulusCount];
    // 若申请内存失败，则删除图像数组，然后返回错误
    if (workmatchres == NULL) {
        _deleteBigImageArray(bigimagesarray, bigimagessize);
        return OUT_OF_MEM;
    }

    // 定义一个用来进行图像匹配的对象
    ImageMatch match;
    // 设置旋转表
    match.setRotateTable(normalWork->rotateTable);
    // 设置摄动范围的宽
    match.setDWidth(normalWork->dWidth);
    // 设置摄动范围的高
    match.setDHeight(normalWork->dHeight);
    // 设置摄动中心的横坐标
    match.setDX(normalWork->dX);
    // 设置摄动中心的纵坐标
    match.setDY(normalWork->dY);
    // 设置匹配需要的 TEMPLATE 图像
    errcode = match.setTemplateImage(shrinknormalwork, normalWork->count);
    // 如果设置 TEMPLATE 图像失败，则释放先前申请的内存空间，然后返回错误
    if (errcode != NO_ERROR) {
        _deleteBigImageArray(bigimagesarray, bigimagessize);
        delete [] workmatchres;
        return errcode;
    }

    // 用 1 / 8 缩小的，具有不同回旋角的 WORK 图像分别在 1 / 8 缩小的经过扩缩和
    // 摄影变换 TEST 图像进行匹配
    for (int i = 0; i < _scalModulusCount; i++) {
        errcode = match.imageMatch(shrinkscalprojecttest[i], &workmatchres[i],
                                   NULL);
        // 若匹配操作失败，则返回释放先前申请的空间，然后返回错误
        if (errcode != NO_ERROR) {
            _deleteBigImageArray(bigimagesarray, bigimagessize);
            delete [] workmatchres;
            return errcode;
        }
    }

    // 找到最佳匹配的 TEST 图像的索引
    int besttestindex = _getBestMatchTestIndex(workmatchres, _scalModulusCount);
    // 获取扩缩和摄影变换后， 1 / 8 缩小前的最佳匹配的 TEST 图像
    Image *matchtest = scalprojecttest[besttestindex];
     
    // 对扩缩和摄影变换后，缩小前的最佳匹配的 TEST 图像生成 2 * angleCount 个
    // 角度的回转图像

    // 用来存储得到的 2 * angleCount 个回转角的回转图像
    Image **rotatetest;
    // 从图像数组中获取空间
    rotatetest = cursor;
    // 更新游标
    cursor += 2 * angleCount;
    // 计算 1 / 8 缩小前在 TEST 图像的匹配中心，由 1 / 8 缩小的 TEST 图像匹配
    // 得到的最佳匹配中心乘 8 得到
    int twx = workmatchres[besttestindex].matchX * 8;
    int twy = workmatchres[besttestindex].matchY * 8;
    // 获取 1 / 8 缩小的 TEST 图像匹配得到旋转角
    float angle = workmatchres[besttestindex].angle;
    // 设置图像回转后的大小，这里设置为标准 WORK 图像的 1.5 倍
    int rwidth = normalWork->images[0]->width * 3 / 2;
    int rheight = normalWork->images[0]->height * 3 / 2;

    // 删除先前申请的用来记录匹配得到的结果的空间
    delete [] workmatchres;
    // 创建 2 * angleCount 个角度的回转图像
    errcode = _createAffineImages(matchtest, twx, twy, angle, rwidth, rheight,
                                  angleCount, rotatetest);
    // 如果创建失败，则释放图像数组的空间，然后返回错误
    if (errcode != NO_ERROR) {
        _deleteBigImageArray(bigimagesarray, bigimagessize);
        return NO_ERROR;
    }

    // 分别用标准 WORK 图像对 2 * angleCount 个角度的回转图像进行匹配

    // 设置旋转表，由于之前已经对图像进行了回转，所以这里旋转角设置为 0
    RotateTable table(0.0f, 0.0f, 0.2f, rwidth * 2, rheight * 2);
    // 创建用来进行图像匹配操作的对象
    ImageMatch rmatch(&table, rwidth, rheight, rwidth / 2, rheight / 2, 3,
                      0, 0, 0, 0);
    // 设置匹配的 TEMPLATE 图像
    errcode = rmatch.setTemplateImage(normalWork->images, normalWork->count);
    // 如果设置失败，释放图像数组空间，然后返回错误
    if (errcode != NO_ERROR) {
        _deleteBigImageArray(bigimagesarray, bigimagessize);
        return errcode;
    }
    // 申请用来存储匹配得到的结果的空间
    workmatchres = new MatchRes[2 * angleCount];
    // 分别用标准 WORK 图像对 2 * angleCount 个角度的回转图像进行匹配
    for (int i = 0; i < 2 * angleCount; i++) {
        errcode = rmatch.imageMatch(rotatetest[i], &workmatchres[i], NULL);
        // 如果匹配失败，则释放之前申请的空间，然后返回错误
        if (errcode != NO_ERROR) {
            _deleteBigImageArray(bigimagesarray, bigimagessize);
            delete [] workmatchres;
            return errcode;
        }
    }

    // 找出 rotatetest 中的最大匹配的图像的索引 
    int bestrotatetestindex = _getBestMatchTestIndex(workmatchres, 
                                                     2 * angleCount);
    // 获取最佳匹配的图像
    Image *matchrotatetest = rotatetest[bestrotatetestindex];
    // 计算最佳匹配图像的回转角
    float bestangle = angle + 0.2 * (bestrotatetestindex - angleCount);
    // 获取最佳匹配的中心
    int cx = workmatchres[bestrotatetestindex].matchX;
    int cy = workmatchres[bestrotatetestindex].matchY;
    // 删除之前申请的存放匹配结果的内存空间
    delete [] workmatchres;

    // 对得到的最佳匹配的 rotatetest 图像进行方向矫正处理

    // 计算回转的移动向量
    int tx = rwidth / 2 - cx;
    int ty = rheight / 2 - cy;
    // 定义用来进行回转操作的对象
    AffineTrans aff(AFFINE_SOFT_IPL, tx, ty,  -bestangle);
    // 对 rotatetest 图像进行方向矫正处理
    errcode = aff.rotateShift(matchrotatetest, work);
    // 如果回转操作失败，则删除图像数组，然后返回错误
    if (errcode != NO_ERROR) {
        _deleteBigImageArray(bigimagesarray, bigimagessize);
        return errcode;
    }

    // 删除图像数组
    _deleteBigImageArray(bigimagesarray, bigimagessize);
    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}

// 成员方法：workAndObjectMatch（进行 WORK and OBJECT 进行匹配操作）
__host__ int WorkAndObjectMatch::workAndObjectMatch(Image *test, 
                                                    MatchRes *res, int rescount)
{
    // 检查 test 和 res 是否为空，若为空，则返回错误
    if (test == NULL || res == NULL)
        return NULL_POINTER;

    // 检查 rescount 是否合法，若不合法，则返回错误
    if (rescount <= 0)
        return INVALID_DATA;

    // 检查 normalWork 和 objects 是否为空，若为空，则返回错误
    if (normalWork == NULL || objects == NULL)
        return NULL_POINTER;

    // 检查标准 WORK 图像的数量是否合法，若不合法，直接返回错误
    if (normalWork->count <= 0)
        return INVALID_DATA;

    int errcode;  // 局部变量，错误码
    Image *work;  // 局部变量，TEST 图像中的 WORK 图像
    // 为 work 申请空间
    errcode = ImageBasicOp::newImage(&work);
    // 若失败，则直接返回错误
    if (errcode != NO_ERROR) 
        return errcode;   

    // 获取 TEST 图片中的 WORK 图片
    errcode = getMatchWork(test, work);
    // 若获取失败，则释放 work 图像，然后返回错误
    if (errcode != NO_ERROR) {
        ImageBasicOp::deleteImage(work);
        return errcode;
    }

    // 计算 objectCount 和 rescount 的最小值，用来指定匹配的次数
    int objectcount = objectCount < rescount ? objectCount : rescount;
    // 分别用各个 OBJECT 图像对 work 图像进行匹配 
    for (int i = 0; i < objectcount; i++) {
        // 定义一个用来匹配的对象
        ImageMatch match;
        // 设置匹配的摄动范围的宽
        match.setDWidth(objects[i].dWidth);
        // 设置匹配的摄动范围的高
        match.setDHeight(objects[i].dHeight);
        // 设置摄动中心的横坐标
        match.setDX(objects[i].dX);
        // 设置摄动中心的纵坐标
        match.setDY(objects[i].dY);
        // 设置旋转表
        match.setRotateTable(objects[i].rotateTable);
        // 设置 TEMPLATE 图像
        errcode = match.setTemplateImage(objects[i].images, objects[i].count);
        // 若设置失败，则释放 work 图像的空间，然后返回错误
        if (errcode != NO_ERROR) {
            ImageBasicOp::deleteImage(work);
            return errcode;
        }
        // 用 OBJECT 图像对 work 图像进行匹配
        errcode = match.imageMatch(work, &res[i], NULL);
        // 若匹配发生错误，则释放 work 图像的空间，然后返回错误
        if (errcode != NO_ERROR) {
            ImageBasicOp::deleteImage(work);
            return errcode;
        }
    }
    
    // 释放 work 图像的空间
    ImageBasicOp::deleteImage(work);
    // 处理完毕，返回 NO_ERROR
    return NO_ERROR;
}

