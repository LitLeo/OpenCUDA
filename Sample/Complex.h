// Complex.h
// 创建人：李苓
//
// 复数类
// 功能说明：实现复数之间的加法、乘法、求模、赋值功能。


#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#include<stdio.h>
#include<cmath>

#include "ErrorCode.h"

// 类：Complex（复数类）
// 继承自：无
// 实现复数之间的加减乘除、求模、赋值、与实数做商、输出功能。
class Complex {

protected:
    // 成员变量：real（实数部分）
    // 设定复数的实数部分
    float real;

    // 成员变量：imaginary（虚数部分）
    // 设定复数的虚数部分
    float imaginary;

public:
    // 构造函数：Complex
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Complex()
    {
        // 使用默认值为类的各个成员变量赋值。
        this->real = 0.0f;       // 实数部分为 0
        this->imaginary = 0.0f;  // 虚数部分为 0
    }

    // 构造函数：Complex
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Complex(
            float real,          // 实数的虚数部分
            float imaginary      // 复数的虚数部分
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->real = 0.0f;       // 实数部分赋值为 0
        this->imaginary = 0.0f;  // 虚数部分赋值为 0

        // 根据参数列表中的值设定成员变量的初值
        setReal(real);
        setImaginary(imaginary);
    }

    // 成员方法：getReal（获取实数部分）
    // 获取成员变量 real 的值。
    __host__ __device__ float  // 返回值：返回 real 成员变量的值
    getReal() const
    {
        // 返回 real 成员变量的值。
        return this->real;
    }

    // 成员方法：setReal（设定实数部分）
    // 设置成员变量 real 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执
                             // 行，返回 NO_ERROR。
    setReal(
            float real       // 复数的实数部分
    ) {
        // 将 real 成员变量赋成新值。
        this->real = real;

        // 若函数正确执行，返回 NO_ERROR。
        return NO_ERROR;
    }

    // 成员方法：getImaginary（获取虚数部分）
    // 获取成员变量 imaginary 的值。
    __host__ __device__ float  // 返回值：返回 imaginary 成员变量的值
    getImaginary() const
    {
        // 返回 imaginary 成员变量的值
        return this->imaginary;
    }

    // 成员方法：setImaginary（设置虚数部分）
    // 设置成员变量 imaginary 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回NO_ERROR。
    setImaginary(
            float imaginary  //复数的虚数部分
    ) {
        // 将 imaginary 成员变量赋成新值
        this->imaginary = imaginary;

        // 若函数正确执行，返回NO_ERROR。
        return NO_ERROR;
    }

    // 成员方法： set（设置复数）
    // 同时设置成员变量 real 和 imaginary 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确
                             // 执行，返回NO_ERROR。
    set(
            float real,      // 复数的实数部分
            float imaginary  // 复数的虚数部分
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了非法
        // 的初始值而使系统进入一个未知的状态。
        this->real = 0.0f;       // 实数部分赋值为 0
        this->imaginary = 0.0f;  // 虚数部分赋值为 0

        // 根据参数列表中的值设定成员变量的初值
        setReal(real);
        setImaginary(imaginary);

        // 若函数正确执行，返回NO_ERROR。
        return NO_ERROR;
    }

    // 成员方法：addition（复数的加法运算函数）
    // 给定两个复数，返回这两个复数的和
    __host__ __device__ Complex  // 返回值：返回两个复数相加的结果
    addition( 
            Complex c            // 一个复数类的对象
    ) {
        // 一个复数类的对象，用于存储两个复数相加的结果
        Complex result;

        // 两个复数的实数部分进行加和操作
        result.setReal(this->getReal() + c.getReal());
        // 两个复数的虚数部分进行加和操作
        result.setImaginary(this->getImaginary() + c.getImaginary());

        // 返回两个复数相加的结果
        return result;
    }

    // 成员方法：multiplication（复数的乘法运算函数）
    // 给定两个复数，返回这两个复数的乘积
    __host__ __device__ Complex  // 返回值：返回两个复数相乘的结果
    multiplication(
            Complex c            // 一个复数类的对象
    ) {
        // 一个复数类的对象，用于存储两个复数相乘的结果
        Complex result;

        // 获取两个复数的实部和虚部
        float re1 = this->getReal();
        float im1 = this->getImaginary();
        float re2 = c.getReal();
        float im2 = c.getImaginary();

        // 计算两个复数相乘所得结果的实数部分
        result.setReal(re1 * re2 - im1 * im2);
        // 计算两个复数相乘所得结果的虚数部分
        result.setImaginary(re1 * im2 + im1 * re2);

        // 返回两个复数相乘的结果
        return result;
    }

    // 成员方法：sub（复数的减法运算函数）
    // 给定两个复数，返回这两个复数的差
    __host__ __device__ Complex  // 返回值：返回两个复数相加的结果
    sub( 
            Complex c            // 一个复数类的对象
    ) {
        // 一个复数类的对象，用于存储两个复数相加的结果
        Complex result;

        // 两个复数的实数部分进行减操作
        result.setReal(this->getReal() - c.getReal());
        // 两个复数的虚数部分进行减操作
        result.setImaginary(this->getImaginary() - c.getImaginary());

        // 返回两个复数相减的结果
        return result;
    }

    // 成员方法：div（复数的除法运算函数）
    // 给定两个复数，返回这两个复数的差
    __host__ __device__ Complex  // 返回值：返回两个复数相加的结果
    div( 
            Complex c            // 一个复数类的对象
    ) {
        // 一个复数类的对象，用于存储两个复数相加的结果
        Complex result;

        // 两个复数的实数部分进行除操作
        result.setReal((this->real * c.getReal() + 
                        this->imaginary * c.getImaginary()) /
                       (c.getReal() * c.getReal() + 
                        c.getImaginary() * c.getImaginary()));

        // 两个复数的虚数部分进行除操作
        result.setImaginary((c.getReal() * this->imaginary - 
                             this->real * c.getImaginary()) / 
                            (c.getReal() * c.getReal() + 
                             c.getImaginary() * c.getImaginary()));

        // 返回两个复数相除的结果
        return result;
    }

    // 成员方法：div（复数除以一个实数的运算）
    // 给定一个复数，一个实数，求复数与实数相除的结果
    __host__ __device__ Complex  // 返回值：复数与实数相除的结果
    div( 
            float f              // 除数
    ) {
        // 一个复数类的对象，用于存储两个复数相加的结果
        Complex result;

        // 计算结果复数的实数部分
        result.setReal(this->real / f);
        // 计算结果复数的虚数部分
        result.setImaginary(this->imaginary / f);

        // 返回计算出的结果
        return result;
    }

    // 成员方法：modulus2（复数求模函数）
    // 求复数的模的平方
    __host__ __device__ float  // 返回值：返回复数的模的平方
    modulus2()
    {
        // 复数的求模的平方操作
        float result = this->getReal() * this->getReal() + 
                       this->getImaginary() * this->getImaginary();

        // 返回复数的模的平方
        return result;
    }

    // 成员方法：operator + （加号重载函数）
    // 给定两个复数，返回这两个复数的和
    __host__ __device__ Complex  // 返回值：返回两个复数相加的结果
    operator + (
            Complex c            // 一个复数对象
    ){
        // 调用 addition 函数，实现两个复数的相加，返回相加的结果
        return addition(c);
    }

     // 成员方法：operator *（乘号重载函数）
     // 给定两个复数，返回这两个复数的乘积
    __host__ __device__ Complex  // 返回值：返回两个复数相乘的结果
    operator * (
            Complex c            // 一个复数对象
    ){
        // 调用 multiplication 函数，实现两个复数的相乘，返回相乘的结果
        return multiplication(c);
    }

    // 成员方法：operator - （减号重载函数）
    // 给定两个复数，返回这两个复数的差
    __host__ __device__ Complex  // 返回值：返回两个复数相减的结果
    operator - (
            Complex c            // 一个复数对象
    ){
        // 调用 sub 函数，实现两个复数的做差，返回相减的结果
        return sub(c);
    }

    // 成员方法：operator / （除号重载函数）
    // 给定两个复数，返回这两个复数的商
    __host__ __device__ Complex  // 返回值：返回两个复数相除的结果
    operator / (
            Complex c            // 一个复数对象
    ){
        // 调用 div 函数，实现两个复数的相除，返回相除的结果
        return div(c);
    }

    // 成员方法：operator / （除号重载函数）
    // 给定复数与实数，求复数除以实数的商
    __host__ __device__ Complex  // 返回值：返回相除的结果
    operator / (
            float f              // 实数除数
    ){
        // 调用 div 函数，实现两个复数的相除，返回相除的结果
        return div(f);
    }

    // 成员方法：operator = （赋值运算符重载函数）
    // 将给定复数的值赋给当前复数
    __host__ __device__ void  // 无返回值
    operator = (
            Complex c         // 一个复数对象
    ) {
        // 为当前复数实数部分赋值
        this->setReal(c.getReal());
        // 为当前复数虚数部分赋值
        this->setImaginary(c.getImaginary());
    }
    
    // 成员方法：print （打印当前复数）
    // 把当前复数以一定的格式打印到屏幕上
    __host__ __device__ void  // 无返回值。
    print() {                 // 无参数。
        // 复数实数部分与复数部分，小数部分均输出4位。
        printf("%.4f%+.4fi", this->real, this->imaginary);
    }
};

#endif