// Graph
// 实现图的数据结构

#include "Graph.h"
#include "ErrorCode.h"


// 成员方法：setEdge（设置边信息）
// 边中包含了点的信息，这个函数也达到了建图的功能
int Graph::setEdge(int ivex, int jvex, int eno)
{
    // 如果起点和终点号大于等于点的个数或者起点和终点一样编号，返回
    if (ivex >= vexnum || jvex >= vexnum || ivex == jvex)
        return UNIMPLEMENT;

    // 定义边的指针变量
    Edge *newedge1, *newedge2, *p;
    
    // 根据输入的信息产生一条新的边
    newedge1 = new Edge;
    newedge1->ivex = ivex;
    newedge1->jvex = jvex;
    newedge1->eno = eno;
    
    // 由于是无向图，同时也得到反方向的一条新边
    newedge2 = new Edge;
    newedge2->ivex = jvex;
    newedge2->jvex = ivex;
    newedge2->eno = eno;

    // 得到 ivex 为起始点的第一条边
    p = vertexlist[ivex].firstedge;
    // 如果该点没有第一条边则设置当前边为第一条边
    if (p == NULL) {
        vertexlist[ivex].firstedge = newedge1;
        // 设置点对应的边链表的正在访问边为第一条边
        vertexlist[ivex].current = vertexlist[ivex].firstedge;
    } else {
        // 找到合适的位置放置新边，连接到边链表中
        while (p->link != NULL) {
            p = p->link;
        }
        p->link = newedge1;              
    }

    // 得到 jvex 为起始点的第一条边
    p = vertexlist[jvex].firstedge;
    // 如果该点没有第一条边则设置当前边为第一条边
    if (p == NULL) {
        vertexlist[jvex].firstedge = newedge2;
        // 设置点对应的边链表的正在访问边为第一条边
        vertexlist[jvex].current = vertexlist[jvex].firstedge;
    } else {
        // 找到合适的位置放置新边，连接到边链表中
        while (p->link != NULL) {
            p = p->link;
        }
        p->link = newedge2;          
    }
    
    // 无错返回
    return NO_ERROR;
}