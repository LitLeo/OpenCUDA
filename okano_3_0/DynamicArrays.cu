// DynamicArrays
// 实现动态数组

#include "DynamicArrays.h"


// 成员方法：addElem（往动态数组增加元素）
// 往动态数组末尾增加一个元素
__host__ int DynamicArrays::addElem(int elem)
{
    // 如果指针为空，报错返回
    if (array == NULL)
        return NULL_POINTER;

    // 如果当前大小大于等于最大容量，重新修改最大容量值，把需要的值添加进数组里
    if (size >= maxsize) {
        int sz = maxsize;
        maxsize = maxsize * 2;
        int *tmp = new int[maxsize];
        int i;
        for (i = 0; i < sz; i++) {
            tmp[i] = array[i];
        }
        tmp[i] = elem;
        delete []array;
        array = tmp;
        size++;
        return NO_ERROR;
    }
    // 直接把 elem 添加进数组里
    array[size++] = elem;
    return NO_ERROR;
}

// 成员方法：addTail（往动态数组末尾增加两个元素）
// 往动态数组末尾同时增加两个元素，赋值曲线坐标的 x 轴和 y 轴坐标
__host__ int DynamicArrays::addTail(int x, int y)
{
    // 如果指针为空，报错返回
    if (array == NULL)
        return NULL_POINTER;

    // 如果当前大小大于等于最大容量，重新修改最大容量值，把需要的值从尾部添加
    // 进数组里
    if (size >= maxsize) {
        int sz = maxsize;
        maxsize = maxsize * 2;
        int *tmp = new int[maxsize];
        int i;
        for (i = 0; i < sz; i++)
            tmp[i] = array[i];
        tmp[i++] = x;
        tmp[i++] = y;
        delete []array;
        array = tmp;
        size += 2;
        return NO_ERROR;
    }

    // 直接把需要添加的两个数添加进数组，因为最大容量是偶数，当前数组大小也为
    // 偶数，所以可以直接添加，不会越界
    array[size++] = x;
    array[size++] = y;
    return NO_ERROR;
}

// 成员方法：addHead（往动态数组头部增加两个元素）
// 往动态数组首部同时增加两个元素，赋值曲线坐标的 x 轴和 y 轴坐标
__host__ int DynamicArrays::addHead(int x, int y)
{
    // 如果指针为空，报错返回
    if (array == NULL)
        return NULL_POINTER;

    // 如果当前大小大于等于最大容量，重新修改最大容量值，把需要的值从首部添加
    // 进数组里
    if (size >= maxsize) {
        int sz = maxsize;
        maxsize = maxsize * 2;
        int *tmp = new int[maxsize];
        int i;
        for (i = 0; i < sz; i++)
            tmp[i + 2] = array[i];
        tmp[0] = x;
        tmp[1] = y;
        delete []array;
        array = tmp;
        size += 2;
        return NO_ERROR;
    }

    // 直接把需要添加的两个数添加进数组
    int *temp = new int[maxsize];
    for (int j = 0; j < size; j++) {
        temp[j + 2] = array[j];  
    }
    temp[0] = x;
    temp[1] = y;
    delete []array;
    array = temp;
    size += 2;
    return NO_ERROR;
}

// 成员方法：delElem（删除数组中相邻值为 x，y 的点）
// 曲线跟踪辅助函数，删除曲线中值为 x，y 的点，前提是数组里有这个点的坐标，这个
// 需要编写代码的人自己把握，这个删除其实就是两个元素和最后的两个元素交换，动态
// 数组大小减少 2
__host__ int DynamicArrays::delElem(int x, int y)
{
    // 如果数组大小小于 2，则报错返回
    if (size < 2)
        return UNIMPLEMENT;
    
    // 如果指针为空，报错返回
    if (array == NULL)
        return NULL_POINTER;
    
    // 定义局部变量
    int j;
    // 找到数组中值为 x，y 的点，并且删除，这里的删除不是真正意义的删除，为了
    // 加快速度是直接和末尾两个数交换，并且使当前数组大小减 2
    for (int i = 0; i < size;) {
        if (array[i++] == x && array[i++] == y) {
            j = i - 2;
            array[j] = array[size - 2];
            array[j + 1] = array[size - 1];
            size = size - 2;
            return NO_ERROR;
        }
    }
    // 没有找到的话，就报错返回
    return UNIMPLEMENT;
}

// 成员方法：delElemXY（删除数组中相邻值为 x，y 的点）
// 曲线跟踪辅助函数，删除曲线中值为 x，y 的点，前提是数组里有这个点的坐标，这个
// 需要编写代码的人自己把握，删除后，后续元素往前移动两个位置，数组大小减少 2
__host__ int DynamicArrays::delElemXY(int x, int y)
{
    // 如果数组大小小于 2，则报错返回
    if (size < 2)
        return UNIMPLEMENT;
    
    // 如果指针为空，报错返回
    if (array == NULL)
        return NULL_POINTER;
    
    // 定义局部变量
    int j;
    // 找到数组中值为 x，y 的点，并且删除，后续元素往前移动两个位置，并且使当前
    // 数组大小减 2
    for (int i = 0; i < size;) {
        if (array[i++] == x && array[i++] == y) {
            j = i - 2;
            for (; j < size - 2; j++) {
                array[j] = array[j + 2];
            }
            size = size - 2;
            return NO_ERROR;
        }
    }
    // 没有找到的话，就报错返回
    return UNIMPLEMENT;
}

// 成员方法：delTail（删除末尾最后一个数）
// 实现动态数组实现栈的 pop 方式，并且得到栈顶元素
__host__ int DynamicArrays::delTail(int &elem)
{
    // 如果数组大小为 0 报错返回
    if (size == 0)
        return UNIMPLEMENT;

    // 如果指针为空，报错返回
    if (array == NULL)
        return NULL_POINTER;

    // 得到最末尾的数值，即栈顶数
    elem = array[size - 1];

    // 容量大小减 1
    size--;

    // 正确执行返回
    return NO_ERROR;
}

// 成员方法：reverse（动态数组以成对坐标反转）
// 实现动态数组得到的曲线坐标进行点坐标的反转
__host__ int DynamicArrays::reverse()
{
    // 如果指针为空，报错返回
    if (array == NULL)
        return NULL_POINTER;

    // 定义临时局部变量
    int temp, j = 0;
    // 定义循环次数
    int count = size / 4;
    // 以坐标对为基数进行点坐标的反转
    for (int i = 0; i < count; i++) {
        temp = array[j];
        array[j] = array[size - j - 2];
        array[size - j - 2] = temp;
        j++;
        temp = array[j];
        array[j] = array[size - j];
        array[size - j] = temp;
        j++;
    }
    // 函数正确执行，返回
    return NO_ERROR;
}

// 成员方法：findElem（查找动态数组里是否有元素 elem）
// 查找动态数组里是否有外界给定的元素 elem
__host__ bool DynamicArrays::findElem(int elem)
{
    // 找到返回 true
    for (int i = 0; i < size; i++) {
        if (elem == array[i]) 
            return true;
    }
    
    // 没找到返回 false
    return false;
}

// 成员方法：findElem（查找动态数组里是否有要查找的坐标对）
__host__ bool DynamicArrays::findElemXY(int x, int y)
{
    // 找到返回 true
    for (int i = 0; i < size / 2; i++) {
        if ((x == array[2 * i]) && (y == array[2 * i + 1])) 
            return true;
    }
   
    // 没找到返回 false
    return false; 
}

// 成员方法：addArray（动态数组的连接）
// 连接两个动态数组，曲线跟踪辅助函数，实现两个曲线的连接
__host__ int DynamicArrays::addArray(DynamicArrays &object)
{
    // 如果指针为空，报错返回
    if (object.getCrvDatap() == NULL)
        return NULL_POINTER;

    // 重新申请一个大小为两个数组最大容量和的数组 ，并且拷贝当前数组的值 
    int sz = this->size;
    this->size = this->size + object.getSize();
    this->maxsize = this->maxsize + object.maxsize;
    int *temp = new int[this->maxsize];
    memcpy(temp, array, sz * sizeof (int));
    delete []array;
    array = temp;
    
    // 定义临时变量
    int i, j;
    // 开始衔接数组，如果要连接的曲线坐标的首部和当前曲线尾部是一样的，
    // 则从第二个点坐标开始连接，否则直接连接
    if ((array[sz - 2] == object[0]) && (array[sz - 1] == object[1])) {
        // 跳过当前曲线坐标的尾部
        this->size = this->size - 2;
        for (i = sz - 2, j = 0; i < this->size; i++, j++) {
            array[i] = object[j];
        }
    } else {
        for (i = sz, j = 0; i < this->size; i++, j++) {
            array[i] = object[j];
        }
    }
    // 函数正确执行返回
    return NO_ERROR;
}
