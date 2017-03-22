#include <iostream> 
#include "CovarianceMatrix.h" 
using namespace std;
#define DIMENSION 3 
                                                                               
// 函数：_calAveMatrix(求输入样本矩阵的每一维的平均值)
// 求协方差矩阵的辅助函数，根据输入的样本矩阵（inputsample），求出每一维
// （即每一列）的平均值并将其存储在 avematrix 一维数组中，长度为输入矩阵宽度。
// 注意 inputsample 是以一维方式存储，函数中会根据用户输入的矩阵宽度和高度
// 将其转换为二维，不需用户考虑。
// avematrix 需要用户先开辟空间再传入该函数，长度为输入矩阵宽度。
static __host__ int           //返回值：函数是否正确运行，
                              //若函数正确运行，返回NO_ERROR。 
_calAveMatrix( 
        MyType *inputsample,  // 输入样本矩阵(以一维方式存储）
        float *avematrix,     // 输出每一维的平均值数组
        int dimension,        // 样本维度（矩阵宽度）
        int samplenum         // 样本数量（矩阵高度）
);

// 函数：_cofactor(求输入矩阵的指定位置的元素的余子式)
// 求逆矩阵的辅助函数，在输入矩阵（inputmatrix）中删除目标位置所在的行和列，所
// 得矩阵即为该输入矩阵目标位置的余子式，余子式存储在 cofactor 中，余子式宽度
// 和高度各为输入矩阵的宽度和高度减一。注意 inputmatrix 和 cofactor 以一维方式
// 存储，函数中会根据用户输入的矩阵宽度和高度，将其转换为二维，不需用户考虑。
// cofactor 需要用户先开辟空间再传入该函数，宽度和高度各为输入矩阵的
// 宽度和高度减一。
static __host__ int           //返回值：函数是否正确运行，
                              //若函数正确运行，返回NO_ERROR。 
_cofactor( 
        MyType *inputmatrix,  // 输入矩阵(以一维方式存储)
        MyType *cofactor,     // 输出指定元素的余子式
        int width,            // 矩阵宽度
        int height,           // 矩阵高度
        int target_x,         // 目标位置横坐标
        int target_y          // 目标位置纵坐标
);

// 函数：_determinant(求得输入矩阵的行列式)
// 求逆矩阵的辅助函数，将输入矩阵（inputmatrix）以递归方式按第一行展开，直到将
// 矩阵降为2阶，从而算出矩阵行列式。注意 inputmatrix 以一维方式存储，函数中会
// 根据用户输入的矩阵宽度和高度，将其转换为二维，不需用户考虑。
static __host__ float         //行列式的值
_determinant( 
        MyType *inputmatrix,  // 输入矩阵(以一维方式存储)
        int width,            // 矩阵宽度
        int height            // 矩阵高度
);

// 函数：_transpose(将给定矩阵的转置,只针对float类型的矩阵)
// 求逆矩阵的辅助函数，将输入矩阵（inputmatrix）沿主对角线交换，所得矩阵即为
// 输入矩阵的转置矩阵，转置矩阵保存在输入矩阵空间中。注意 inputmatrix 以一维
// 方式存储，函数中会根据用户输入的矩阵宽度和高度，将其转换为二维，不需用户
// 考虑。
static __host__ int          //返回值：函数是否正确运行，
                             //若函数正确运行，返回NO_ERROR。
_transpose( 
        float *inputmatrix,  // 输入矩阵(以一维方式存储)
        int width,           // 矩阵宽度
        int height           // 矩阵高度
);

// 函数：_calAdjointMatrix(求给定矩阵的伴随矩阵)
// 求逆矩阵的辅助函数，求出输入矩阵（inputmatrix）中每一个元素的代数余子式，
// 代数余子式为该元素对应的余子式的行列式和对应符号位的乘积，这些代数余子式
// 组成的矩阵即为伴随矩阵，伴随矩阵保存在 adjointmatrix 中，伴随矩阵宽度和高度
// 和输入矩阵一致。注意 inputmatrix 和 adjointmatrix 以一维方式存储，函数中
// 会根据用户输入的矩阵宽度和高度，将其转换为二维，不需用户考虑。
// adjointmatrix 需要用户先开辟空间再传入该函数，宽度和高度和输入矩阵一致。
static __host__ int            //返回值：函数是否正确运行，
                               //若函数正确运行，返回NO_ERROR。
_calAdjointMatrix( 
        MyType *inputmatrix,   // 输入矩阵(以一维方式存储)
        float *adjointmatrix,  // 输出伴随矩阵
        int width,             // 矩阵宽度
        int height             // 矩阵高度
);

// 函数：_calAveMatrix(求输入样本矩阵的每一维的平均值)
static __host__ int _calAveMatrix(MyType *inputsample, float *avematrix, 
                                  int dimension, int samplenum) 
{
    //判断输入指针是否有效
    if (inputsample == NULL || avematrix == NULL) 
        return NULL_POINTER;
    //判断输入是否有效
    if (dimension <= 0 || samplenum <= 0) 
        return INVALID_DATA;
    //局部变量 i，j，用于循环
    int i, j;
    //循环算出每一维的平均值
    for (i = 0; i < dimension; i++) { 
        //局部变量sum，用于保存第i维中样本之和
        float sum = 0.0;
        //循环对第i维样本求和
        for(j = 0; j < samplenum; j++) 
            sum += inputsample[j * dimension + i];
        //求第i维的平均值
        avematrix[i] = sum / samplenum;
    } 
    return NO_ERROR;
} 

// 函数：calCovMatrix(求给定矩阵的协方差矩阵)
__host__ int calCovMatrix(MyType *inputsample, float *covmatrix, 
                          int dimension, int samplenum) 
{ 
    //判断输入指针是否有效
    if (inputsample == NULL || covmatrix == NULL) 
        return NULL_POINTER;
    //判断输入是否有效
    if (dimension <= 0 || samplenum <= 0) 
        return INVALID_DATA;
    //局部变量，用于保存错误码
    int errcode;
    //局部变量 i，j，k，用于循环
    int i, j, k;
    //为每一维的平均值申请空间，保存在以维度为大小的数组中
    float *avematrix = new float[dimension];
    //调用 _calAveMatrix 求得 avematrix
    errcode = _calAveMatrix(inputsample, avematrix, 
                            dimension, samplenum);
    //如果没错，则继续执行
    if (errcode == NO_ERROR) { 
        //循环算出协方差矩阵
        for (i = 0; i < dimension; i++) { 
            for (j = 0; j < dimension; j++) { 
                //局部变量sum
                float sum = 0.0;
                for (k = 0; k < samplenum; k++) { 
                    sum += (inputsample[k * dimension + j] - avematrix[j]) * 
                           (inputsample[k * dimension + i] - avematrix[i]);
                } 
                //根据公式算出协方差
                covmatrix[j * dimension + i] = sum / (samplenum - 1);
            } 
        } 
    } 
    //释放 avematrix 
    delete avematrix;
    return errcode;
} 

// 函数：_cofactor(求输入矩阵的指定位置的元素的余子式)
static __host__ int _cofactor(MyType *inputmatrix, MyType *cofactor, int width, 
                              int height, int target_x, int target_y) 
{ 
    //判断输入指针是否有效
    if (inputmatrix == NULL || cofactor == NULL) 
        return NULL_POINTER;
    //判断输入是否有效
    if (width <= 0 || height <= 0) 
        return INVALID_DATA;
    //判断输入位置是否有效
    if (target_x >= width || target_y >= height) 
        return INVALID_DATA;
    //局部变量 p_cofactor，用于遍历cofactor数组
    MyType *p_cofactor=cofactor;
    //局部变量 i，j，用于循环
    int i, j;
    //循环求得余子式
    for (j = 0; j < height; j++) { 
        if (j != target_y) { 
            for (i = 0; i < width; i++) { 
                if (i != target_x) { 
                    *p_cofactor = inputmatrix[j * width + i];
                    p_cofactor++;
                } 
            } 
        } 
    } 
    return NO_ERROR;
} 

// 函数：_determinant(求得输入矩阵的行列式)
static __host__ float _determinant(MyType *inputmatrix, int width, int height) 
{ 
    //判断是否为方阵
    if (width != height) 
        return 0.0;        //处理不太好
    //判断矩阵大小是否合法
    if (width <= 0) 
        return 0.0;        //处理不太好
    //如果矩阵大小为1，则矩阵行列式为矩阵元素，直接返回矩阵元素
    else if (width == 1) 
        return *inputmatrix;
    //如果矩阵大小为2，则矩阵行列式为主对角线乘积减去副对角线乘积
    else if (width == 2) 
        return (inputmatrix[0] * inputmatrix[3] - 
                inputmatrix[1] * inputmatrix[2]);
    //如果矩阵大小大于2，则将矩阵按第一行展开，递归实现
    else { 
        //为余子式申请空间
        MyType *cofactor = new MyType[(width - 1) * (height - 1)];
        //局部变量，
        float sum = 0.0;
        //局部变量 mark，符号标志位
        int mark = 1;
        //循环将第一行的每一个元素和它的代数余子式的乘积累加
        for (int i = 0; i < width; i++) { 
            _cofactor(inputmatrix, cofactor, width, height, i, 0);
            sum += mark * inputmatrix[0 * width+i] * 
                         _determinant(cofactor, width - 1, height - 1);
            //符号位反转
            mark *= -1;
        } 
        //释放 cofactor 
        delete cofactor;
        return sum;
    } 
} 

// 函数：_transpose(将给定矩阵的转置,只针对float类型的矩阵)
static __host__ int _transpose(float *inputmatrix, int width, int height) 
{
    //判断输入指针是否有效
    if (inputmatrix == NULL) 
        return NULL_POINTER;
    //判断矩阵是否为方阵和判断输入是否有效
    if (width != height || width <= 0 || height <= 0) 
        return INVALID_DATA;
    //局部变量 temp，用于数据交换时保存临时数据
    float temp;
    //局部变量 i,j，用于循环
    int i, j;
    //矩阵转置本质将元素按对角线交换
    for (j = 0; j < height; j++) { 
        for (i = 0; i < width; i++) { 
            //只对右上角矩阵（不含对角线）处理
            if (i > j) { 
                temp = inputmatrix[j * width + i];
                inputmatrix[j * width + i] = inputmatrix[i * width + j];
                inputmatrix[i * width + j] = temp;
            } 
        } 
    } 
    return NO_ERROR;
} 

// 函数：_calAdjointMatrix(求给定矩阵的伴随矩阵)
static __host__ int _calAdjointMatrix(MyType *inputmatrix, float *adjointmatrix, 
                                      int width, int height) 
{
    //判断输入指针是否有效
    if (inputmatrix == NULL || adjointmatrix == NULL) 
        return NULL_POINTER;
    //判断是否为方阵和判断输入是否有效
    if (width != height || width <= 0 || height <= 0) 
        return INVALID_DATA;
    //局部变量 i，j，用于循环
    int i, j;
    //局部变量 mask，用于符号标志位
    int mask = 1;
    //为余子式申请空间
    MyType *cofactor = new MyType[(width - 1) * (height - 1)];
    //计算矩阵每一个元素的代数余子式
    for (j = 0; j < height; j++) { 
        for (i = 0; i < width; i++) { 
            _cofactor(inputmatrix, cofactor, width, height, i, j);
            //如果（i+j）是偶数，mask 置1，否则置-1
            mask = ((i + j) % 2 == 0) ? 1 : -1;
            adjointmatrix[j * width + i] = mask * _determinant(cofactor, 
                                                  width - 1, height - 1);
        } 
    } 
    //释放 cofactor 
    delete cofactor;
    //将矩阵转置
    _transpose(adjointmatrix, width, height);
    return NO_ERROR;
} 

// 函数：calInverseMatrix(求给定矩阵的逆矩阵)
__host__ int calInverseMatrix(MyType *inputmatrix, float *inversematrix, 
                              int width, int height) 
{
    //判断输入指针是否有效
    if (inputmatrix == NULL || inversematrix == NULL) 
        return NULL_POINTER;
    //判断是否为方阵和判断输入是否有效
    if (width != height || width <= 0 || height <= 0) 
        return INVALID_DATA;
    //局部变量，用于保存错误码
    int errcode;
    //为伴随矩阵申请空间
    float* adjointmatrix = new float[width * height];
    //局部变量i，j，用于循环
    int i, j;
    //求矩阵行列式
    float det = _determinant(inputmatrix, width, height);
    //如果行列式为0，则没有逆矩阵
    if (det == 0) 
        return INVALID_DATA;
    //求输入矩阵对应的伴随矩阵
    errcode = _calAdjointMatrix(inputmatrix, adjointmatrix, width, height);
    if (errcode == NO_ERROR) { 
        //将伴随矩阵中的每一个元素除以行列式
        for (j = 0; j < height; j++) { 
            for (i = 0; i < width; i++) { 
                inversematrix[j * width + i] = adjointmatrix[j * width + i] / 
                                               det;
            } 
        } 
    } 
    //释放adjintmatrix
    delete adjointmatrix;
    return errcode;
}

