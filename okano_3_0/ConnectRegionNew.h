// ConnectRegionNew.h
// 创建人：孙慧琪
//
// 连通区域新方法（ConnectRegionNew）
// 功能说明：根据指定阈值将输入图像分割成若干区域。
// 
// 修订历史：
// 2013年03月28日（王媛媛）
//     初始版本
// 2013年06月10日（孙慧琪）
//     每个线程块处理一定行图像数据，对每个线程块一共处理三次，分别进行第一次扫
//     描，标记统一和最终标记。将三次处理分别在三个核函数中实现。
// 2013年05月28日（孙慧琪）
//     增加新的处理操作，解决对于 U 型图像的处理不准确问题。
// 2013年06月20日（王媛媛、孙慧琪）
//     合并三个核函数，加入 _spin 操作实现线程块同步。对小型测试数据能出现正确结
//     果，大图像由于有序标记问题还不能出现正确预期结果。
// 2013年06月24日（孙慧琪）
//     将旧的方法和新方法和并，在主函数中由用户选择所要使用的什么方法。
// 2013年06月26日（王媛媛、孙慧琪）
//     删除参数 threshold，添加新参数 indGray 以及 indGrayNum，并添加结构体
//     FilteredRegions，LabelMaps。
// 2013年07月01日（孙慧琪）
//     增加对 LabelMaps 的相关处理。在 device 端开辟空间。
// 2013年07月05日（孙慧琪）
//     修改原来的核函数的相关细节，将原来根据指定灰度值得到连通区域的算法改为根
//     据当前灰度值是否在需要处理的灰度范围内得到连通区域结果。
// 2013年07月10日（孙慧琪）
//     增加面积的相关处理，并根据面积是否在规定的范围内得到每个区域集中的连通区
//     域的个数。
// 2013年07月14日（王媛媛）
//     优化相关面积算法，增加面积数组，存储每个连通区域的面积信息，使用三维 
//     block 优化对面积的计算和比较。
// 2013年07月17日（孙慧琪）
//     增加区域结构体的处理函数，通过迭代比较得到各个区域结构体的具体信息，增加
//     对区域结构体的成员变量的初始化赋值，并通过对各个位置的遍历，最终得到每个
//     区域的左上角坐标和右下角坐标。
// 2013年07月22日(孙慧琪）
//     修改程序中的一些小错误，将原来的算法继续优化。
// 2013年07月30日（孙慧琪）
//     将区域的结果从 device 端拷回到 host 端。
// 2013年08月15日（孙慧琪）
//     解决了区域集结构体中的区域结构体指针的多层拷贝问题，将最终结果全部拷回到
//     host 端。

#ifndef __CONNECTREGIONNEW_H__
#define __CONNECTREGIONNEW_H__

#include "Image.h"
#include "ErrorCode.h"

// 结构体：FilteredRegions（区域）
// 该结构体定义了区域的数据结构，其中包含了区域属性的描述。
typedef struct FilteredRegions_st {
    int regionX1;               // 区域左上角点的横坐标，要求 0 <= regionX1 <
                                // regionWidth
    int regionY1;               // 区域左上角点的纵坐标，要求 0 <= regionY1 <
                                // regionHeight
    int regionX2;               // 区域右下角点的横坐标，要求 regionX1 <=
                                // regionX2 < regionWidth
    int regionY2;               // 区域右下角点的纵坐标，要求 regionY1 <=
                                // regionY2 < regionHeight
    int labelMapNum;            // 在区域集合中的标记值
    int index;                  // 索引值
} FilteredRegions;

// 结构体：LabelMaps（区域集）
// 该结构体定义了区域集的数据结构，其中包含了区域集的描述，如区域个数，区域属性。
typedef struct LabelMaps_st
{
    int regionCount;       // 区域集中所含区域个数
    FilteredRegions *fr;   // 各区域的信息
    int *gLabel;           // 区域集对应的标记值数组
    int *area;             // 各标记值对应的连通区域对应的面积值   
}LabelMaps;

// 类：ConnectRegionNew
// 继承自：无
// 根据参数 threshold，将满足条件的形状区域进行按序标记后拷贝到输出图像中。
class ConnectRegionNew {

protected:

    // 成员变量：maxArea和 minArea（区域面积的最小和最大值）
    // 进行区域面积判断时的面积值最大最小的范围。
    int maxArea, minArea;

public:
    // 构造函数：ConnectRegionNew
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值
    __host__ __device__
    ConnectRegionNew() {
        // 使用默认值为类的各个成员变量赋值
        this->maxArea = 100000; // 区域最大面积默认为100000
        this->minArea = 100;    // 区域最小面积默认为100
    }

    // 构造函数：ConnectRegionNew
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    ConnectRegionNew(
            int maxArea, int minArea  // 区域面积的最大值和最小值
    ) {
        // 使用默认值为类的各个成员变量赋值，防止用户在构造函数的参数中给了
        // 非法的初始值而使系统进入一个未知的装填
        this->maxArea = 100000;
        this->minArea = 100;
    
        // 根据参数列表中的值设定成员变量的初值
        setMaxArea(maxArea);
        setMinArea(minArea);
    }

    // 成员方法：getMinArea（读取进行区域面积判断时的最小面积值）
    // 读取 minArea 成员变量的值。
    __host__ __device__ int  // 返回值：当前 minArea 成员变量的值。
    getMinArea() const
    {
        // 返回 minArea 成员变量的值。
        return this->minArea;
    }

    // 成员方法：setMinArea（设置进行区域面积判断时的最小面积值）
    // 设置 minArea 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMinArea( 
            int minArea      // 指定的进行区域面积判断时的最小面积值。
    ) {
        // 将 minArea 成员变量赋成新值
        this->minArea = minArea;
    
        // 处理完毕，返回。
        return NO_ERROR;
    }

    // 成员方法：getMaxArea（读取进行区域面积判断时的最小面积值）
    // 读取 maxArea 成员变量的值。
    __host__ __device__ int  // 返回值：当前 maxArea 成员变量的值。
    getMaxArea() const
    {
        // 返回 maxArea 成员变量的值。
        return this->maxArea;
    }

    // 成员方法：setMaxArea（设置进行区域面积判断时的最大面积值）
    // 设置 maxArea 成员变量的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMaxArea( 
            int maxArea      // 指定的进行区域面积判断时的最大面积值。
    ) {
        // 将 maxArea 成员变量赋成新值
        this->maxArea = maxArea;
    
        // 处理完毕，返回。
        return NO_ERROR;
    }

    // Host 成员方法：connectRegionNew（连通区域的标记的新方法）
    // 根据参数 indGray 和 indGrayNum，将满足条件的形状区域进行按序标记后拷贝到
    // 输出区域集中。
    // 如果输出区域集 labelM 为空，则返回错误。
    __host__ int              // 返回值：函数是否正确执行，若函数正确执行，返回
                              // NO_ERROR。
    connectRegionNew(
            Image *inimg,     // 输入图像。
            int * indGray,    // 需要处理的灰度范围数组
            int indGrayNum,   // 所处理的灰度范围的组数
            LabelMaps *labelM // 输出区域集信息，共有 indGrayNum 个
    );
};

#endif

