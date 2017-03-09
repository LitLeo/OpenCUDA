// Graph.h
// 
// 图（Graph）
// 功能说明：无向图数据结构，为了实现特定功能而设计，在曲线跟踪算法中，最后需要转化
//           图，进行图的搜索遍历实现路径连通，这个图的结构是为了辅助实现曲线
//           跟踪算法的，便于算法更简单的实现。由于两个点之间存在多条边又是
//           无向图，所以采用邻接表实现


#ifndef __GRAPH_H__
#define __GRAPH_H__


// 类：Edge（边）
// 继承自：无
// 边数据结构，为了实现特定功能而设计，在曲线跟踪算法中，最后需要转化图，图的
// 数据结构中需要包含边的信息，便于图的实现
class Edge {

public:
    // 成员变量：ivex（边的起点）
    // 表示边的前端
    int ivex;

    // 成员变量：jvex（边的终点）
    // 表示边的后端
    int jvex;
    
    // 成员变量：eno（边的编号）
    // 表示边的编号，编号是唯一的，也不会和点的编号重复
    int eno;
    
    // 成员变量：link（ivex 为起点的边链表）
    // 表示 ivex 为起点的边链表，指向下一条边
    Edge *link;
    
    // 构造函数：Edge
    // 无参版本的构造函数，初始化成员变量为默认值
    __host__ __device__
    Edge()
    {
        ivex = -1;     // 初始化 ivex 为 -1
        jvex = -1;     // 初始化 jvex 为 -1
        eno = -1;      // 初始化边的编号 eno 为 -1
        link = NULL;  // 初始化 ilink 为空
    }
};

// 类：Vex（点）
// 继承自：无
// 点数据结构，为了实现特定功能而设计，在曲线跟踪算法中，最后需要转化图，图的
// 数据结构中需要包含点的信息，便于图的实现
class Vex {

public:
    // 成员变量：vno（点的编号）
    // 表示点的编号，编号是唯一的，也不会和边的编号重复
    int vno;
    
    // 成员变量：firstedge（对于当前点的边链表的头指针）
    // 表示当前点连接的边链表的头指针，指向第一条边
    Edge *firstedge;

    // 成员变量：current（对于当前点的正在访问的边链表指针）
    // 表示当前点的外界正在访问的边链表指针，便于了解边的访问情况
    Edge *current;
    
    // 构造函数：Vex
    // 无参版本的构造函数，初始化成员变量为默认值
    __host__ __device__
    Vex()
    {
        vno = -1;          // 初始化点的编号 vno 为 -1
        firstedge = NULL;  // 初始化 firstedge 为空
        current = NULL;    // 初始化 current 为空
    }
};

// 类：Graph（图）
// 继承自：无
// 无向图数据结构，为了实现特定功能而设计，在曲线跟踪算法中，最后需要转化图，
// 进行图的搜索遍历实现路径连通，这个图的结构是为了辅助实现曲线跟踪算法的，
// 便于算法更简单的实现。由于两个点之间存在多条边又是无向图，所以采用邻接表实现
class Graph {

public:
    // 成员变量：vertexlist（点的数组，用于构建邻接表图）
    // 点的数组，每个点里包含了连接的边信息，用于构建邻接表图
    Vex *vertexlist;
    
    // 成员变量：vexnum（图中点的个数参数）
    // 表示图中点的数目
    int vexnum;
    
    // 成员变量：edgenum（图中边的个数参数）
    // 表示图中边的数目
    int edgenum;
    
    // 构造函数：Graph
    // 无参版本的构造函数，初始化成员变量为默认值
    __host__ __device__
    Graph()
    {
        vertexlist = NULL;  // 初始化 vertexlist 为空
        vexnum = 0;         // 初始化图中点的数目 vexnum 为 0
        edgenum = 0;        // 初始化图中边的数目 edgenum为 0
    }
    
    // 构造函数：Graph
    // 有参版本的构造函数，初始化成员变量为默认值
    __host__ __device__
    Graph (
            int vexnum,  // 图中点的个数参数（具体解释见成员变量）
            int edgenum  // 图中边的个数参数（具体解释见成员变量）
    ) {
        this->vexnum = vexnum;          // 设置点的数目
        this->edgenum = edgenum;        // 设置边的数目
        vertexlist = new Vex [vexnum];  // 根据点的个数申请数组空间
    }
    
    // 成员方法：setEdge（设置边信息）
    // 边中包含了点的信息，这个函数也达到了建图的功能
    __host__ int   // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR
    setEdge(
        int ivex,  // 边的起点
        int jvex,  // 边的终点
        int eno    // 边的编号
    );

    // 成员方法：resetCurrent（重置所有将要访问的边）
    // 重置所有将要访问的边，设置为 firstedge
    __host__ __device__ void  // 无返回值
    resetCurrent()
    {
        // 循环重置将要访问的边
        for (int i = 0; i < vexnum; i++) {
            vertexlist[i].current = vertexlist[i].firstedge;
        }
    }

    // 析构函数：~Graph
    // 释放内存空间
     __host__ __device__   
    ~Graph() 
    {
        // 定义指针变量 p, q
        Edge *p, *q;
        // 循环释放内存空间
        for (int i = 0; i < vexnum; i++) {
            // 得到对应结点的链表头
            p = vertexlist[i].firstedge;
            // 给当前需要访问的边置空
            vertexlist[i].current = NULL;
            // 释放链表所占内存空间
            while (p != NULL) {
                q = p;
                p = p->link;
                // 释放空间后置空
                delete q;
                q = NULL;
            }
        }
        // 释放 vertexlist 申请的空间
        delete [] vertexlist;  
    }
};

#endif

