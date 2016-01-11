# OpenCUDA
CUDA(Compute Unified Device Architecture)，是显卡厂商NVIDIA推出的运算平台。
随着GPU的发展，CUDA使用人数也越来越多。但关于CUDA的库基本都是不开源的，目前手里有上一个项目遗留下的一套图像处理代码，准备整理整理，一步一步的开源出来。想分享出来和大家一起学习。也希望各位CUDA大牛能够奉献自己的源码。

**1. 因为是项目代码，所以前期不能完全开源，源码也只能发一些简单学习型算法的源码。修改后将考虑全部开源和加一些复杂算法**

**2. 目前还是求学阶段，有学业和课程压力，能不能一直做下去还是个未知数**

2015-12.01 更新 版本v0.1

1. 初始版本，添加马赛克（Mosaic）算法，内有详细注释；
2. 目前支持的平台为linux；
3. Image类只提供.o文件，编译方式见Makefile

2015-12.03 更新 版本v0.2

1. 版本v0.2，添加二值化（Binarize）算法，内有详细注释；
2. 目前支持的平台为linux；
3. Image类只提供.o文件，编译方式见Makefile

2015-12.04 更新 版本v0.3

1. 版本v0.3，添加并行排序（SortArray）,内有详细注释
功能说明：实现并行排序算法，包括：双调排序，Batcher's 奇偶合并排序，
          以及 shear 排序。其中，双调排序和 Batcher's 奇偶合并排序
          的数组长度不能超过一个块内的 shared 内存最大限制（一般为 1024）。
          当数组个数大于 1024 时，可以调用 shear 排序，其最大限制为 
          1024×1024 的矩阵。

2015-12.04 更新 版本v1.0
1. 提供Image类源码
2.  

The Tesla K40 used for this research was donated by the NVIDIA Corporation.

> 我的CSDN博客：http://blog.csdn.net/litdaguang

