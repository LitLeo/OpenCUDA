// CoordiSet.cu
// 坐标集数据结构的定义和坐标集的基本操作。

#include "CoordiSet.h"

// 由于所有的函数都是 inline 类型，已在 CoordiSet.h 中定义，故本文件无任何代码
// 内容。

//// Host 静态方法：newCoordiSet（创建坐标集）
//__host__ int CoordiSetBasicOp::newCoordiSet(CoordiSet **outcst)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::newTemplate(outcst);
//}
//
//// Host 静态方法：deleteCoordiSet（销毁坐标集）
//__host__ int CoordiSetBasicOp::deleteCoordiSet(CoordiSet *incst)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::deleteTemplate(incst);
//}
//
//// Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
//__host__ int CoordiSetBasicOp::makeAtCurrentDevice(CoordiSet *cst,
//                                                   size_t count)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::makeAtCurrentDevice(cst, count);
//}
//
//// Host 静态方法：makeAtHost（在 Host 内存中构建数据）
//__host__ int CoordiSetBasicOp::makeAtHost(CoordiSet *cst, size_t count)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::makeAtHost(cst, count);
//}
//
//// Host 静态方法：readFromFile（从文件读取坐标集）
//__host__ int CoordiSetBasicOp::readFromFile(const char *filepath,
//                                            CoordiSet *outcst)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::readFromFile(filepath, outcst);
//}
//
//// Host 静态方法：writeToFile（将坐标集写入文件）
//__host__ int CoordiSetBasicOp::writeToFile(const char *filepath,
//                                           CoordiSet *incst)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::writeToFile(filepath, incst);
//}
//
//// Host 静态方法：copyToCurrentDevice（将坐标集拷贝到当前 Device 内存上）
//__host__ int CoordiSetBasicOp::copyToCurrentDevice(CoordiSet *cst)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::copyToCurrentDevice(cst);
//}
//
//// Host 静态方法：copyToCurrentDevice（将坐标集拷贝到当前 Device 内存上）
//__host__ int CoordiSetBasicOp::copyToCurrentDevice(CoordiSet *srccst,
//                                                   CoordiSet *dstcst)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::copyToCurrentDevice(srccst, dstcst);
//}
//
//// Host 静态方法：copyToHost（将坐标集拷贝到 Host 内存上）
//__host__ int CoordiSetBasicOp::copyToHost(CoordiSet *cst)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::copyToHost(cst);
//}
//
//// Host 静态方法：copyToHost（将坐标集拷贝到 Host 内存上）
//__host__ int CoordiSetBasicOp::copyToHost(CoordiSet *srccst, CoordiSet *dstcst)
//{
//    // 直接调用 TemplateBasicOp 中的方法。
//    return TemplateBasicOp::copyToHost(srccst, dstcst);
//}
