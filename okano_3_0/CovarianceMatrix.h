// CovarianceMatrix.h
// 创建人：丁燎原
//
// 协方差矩阵和逆矩阵
// 功能：定义了两个函数接口，分别可求矩阵的协方差矩阵和逆矩阵
//      （矩阵以一维方式输入）
//
// 修订历史：
// 2014年9月5日（丁燎原）
//     初始版本
     
#ifndef __COVARIANCEMATRIX_H__
#define __COVARIANCEMATRIX_H__

// 自定义类型，方便更换类型
typedef float MyType;

// 函数：calCovMatrix(求给定样本矩阵的协方差矩阵)
// 根据输入样本矩阵（inputsample），通过协方差公式算出协方差矩阵，协方差矩阵保
// 存在 covmatrix 中。注意 inputmatrix 和 covmatrix 以一维方式存储，函数中
// 会根据用户输入的矩阵宽度和高度，将其转换为二维，不需用户考虑。
// covmatrix 需要用户先开辟空间再传入该函数，宽度和高度都为样本维数。
__host__ int                  //返回值：函数是否正确运行，
                              //若函数正确运行，返回NO_ERROR。
calCovMatrix( 
        MyType *inputsample,  //输入样本矩阵(以一维方式存储)
        float *covmatrix,     //输出协方差矩阵(以一维方式存储)
        int dimension,        //样本维数（矩阵宽度）
        int samplenum         //样本数量（矩阵高度）
);

// 函数：calInverseMatrix(求给定矩阵的逆矩阵)
// 根据输入矩阵（inputmatrix），通过伴随矩阵法算出矩阵逆矩阵，逆矩阵保存在
// inversematrix 中。注意 inputmatrix 和 inversematrix 以一维方式存储，函数中会
// 根据用户输入的矩阵宽度和高度，将其转换为二维，不需用户考虑。
// inversematrix 需要用户先开辟空间再传入该函数，宽度和高度和输入矩阵一致。
__host__ int                  //返回值：函数是否正确运行，
                              //若函数正确运行，返回NO_ERROR。
calInverseMatrix( 
        MyType *inputmatrix,   //输入矩阵（以一维方式存储)
        float *inversematrix,  //输出逆矩阵(以一维方式存储)
        int width,             //矩阵宽度
        int height);           //矩阵高度

#endif
                                                                               