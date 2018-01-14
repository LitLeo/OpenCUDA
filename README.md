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
1. 提供CoordiSet Curve ErrorCode Graph Image Matrix Template基础结构类源码
2. 新增等基础练手算法（CombineImage Complex DownSampleImage EdgeDetection FillUp Flip FlutterBinarze GaussianElimination ICcircleRadii ImageDiff ImageFilter ImageHide ImageOverlay ImageScaling DownSampleImage ImageToText InnerDigger Julia LearningFilter Mandelbrot Mosaic Tattoo Zoom），算法列表见下表，sample文件夹下的算法不再维护

|   文件名     | 算法名    |  功能说明  |
| --------   | -----:  | :----:  |
| CombineImage.h/cu      | 融合图像（Combine Image） |  将若干幅图像融合成一幅图像。要求这些图像的 ROI 子区域的尺寸完全相     |
| Complex. h        |   复数类（Complex）   |   实现复数之间的加法、乘法、求模、赋值功能。   |
| DownSampleImage. h        |    缩小图像（DownSampleImage）    |  根据给定的缩小倍数 N，将输入图像缩小，将其尺寸从width * height 变成 (width / N) * (height / N)   |
| EdgeDetection.h/cu        |    边缘检测（EdgeDetection）    |  实现画出两种颜色的边界  |
| FillUp.h/cu        |    像素处理（FillUp）    |  检查一个像素的邻域，若其邻域同时存在 l 像素和 v 像素，当 v 像素的个数大于等于某一值时，将所有的 l 像素置为v 像素。  |
| Flip.h/cu        |    图像翻转（Flip）    |  实现图像的水平和竖直翻转。  |
| FlutterBinarze.h/cu        |    抖动二值化（FlutterBinarze）    |  用像素点疏密表示像素值大小，对灰度图像进行二值抖动，得到二值图像。  |
| GaussianElimination.h/cu        |    高斯消元法（GaussianElimination）    |  通过高斯消元法，求出输入行数与列数相等的方阵的上三角方阵  |
| ICcircleRadii.h/cu        |    最远距离最小的点与最近距离最大的点（ICcircleRadii）    |  对一个给定轮廓的坐标集，求其所包围领域内的一点最远距离最小的点-----外接圆的半径最近距离最大的点-----内接圆的半径  |
| ImageDiff.h/cu        |    图像做差（Difference of two Image）    |  根据输入的两幅灰度图像，对其相应为位置的像素值做差得到差值图像。  |
| ImageFilter.h/cu        |    多阈值图像过滤（image filter）    |  给定一幅图像，根据用户输入的阈值数组，取出每个阈值区间对应的图像，如阈值数组为如[0,50,100,150,200],则该算法输出5幅图像，第一幅保原图像的（0,50）灰度值，其余灰度值设为0，第二幅保留原图像的（50,100）灰度值，以此类推，最后一幅保留原图像的（200，255）灰度值。用户输入阈值时应注意，若您选定的阈值为50,100,150,200,则您输入的阈值数组应为[0,50,100,150,200]，前面加0，后面不加255。之所以这么处理，使阈值数组元素个数和输出图像函数个数一致。  |
| ImageHide.h/cu        |    图像隐藏（ImageHide）    |  实现二值图隐藏于正常图片中（以数据最低位实现）。  |
| ImageOverlay.h/cu        |    图像叠加（ImageOverlay）    |  将 n 幅输入图像叠加到一起，输出图像的灰度值等于所有输入图像对应点灰度值乘以相应权重值并求和。  |
| ImageScaling.h/cu        |    图像扩缩（ImageScaling）    |  根据给定扩缩中心和扩缩系数，实现图像的扩大或缩小  |
| DownSampleImage.h/cu        |    拉伸图像（ImageStretch）    |  根据给定的长宽拉伸倍数 timesWidth 和 timesHeight，将输入图像拉伸，将其尺寸从 width * height 变成(width * timesWidth) * (height * timesHeight)。  |
| ImageToText.h/cu        |    图像转文本（ImageToText）    |  将输入的灰度图像转成制定大小的文本，文本中用一个字符代表特定的灰度级。首先将原图缩放到和文本同样大小，然后按照灰度级对应找到文本，写入字符串中  |
| InnerDigger.h/cu        |    区域抠心（InnerDigger）    |  输入灰度图，图像中有一个 ROI 区域，对区域中的每个点作如下判断：如果该点的八领域内或者四领域内所有的点都是白色，则置为 0；否则保留原值。  |
| Julia.h/cu        |    生成 Julia 集（Julia）    |  以显示区域 W 内所有的点 (mc, mr) 作为初始迭代值 z = z * z + p 进行times 次迭代，z 的初始值为 mc + mr。迭代完成后比较 z 的模和逃逸半径 radius 的大小，根据比较结果进行着色。  |
| LearningFilter.h/cu        |    学习型滤波（LearningFilter）    |  根据设定的阈值，对灰度图像进行二值化处理，得到二值图像。  |
| Mandelbrot.h/cu        |    生成Mandelbrot集（Mandelbrot）    |  以显示区域W内所有的点(mc, mr)作为初始迭代值 p = mc + mri 对z = z * z + p 进行 times 次迭代，z 的初始值为 0。迭代完成后比较 z的模和逃逸半径 radius 的大小，根据比较结果进行着色。  |
| Mosaic.h/cu        |    马赛克（Mosaic）    |  在给定的图像范围内，将图像编程马赛克的样子（电视上常见的遮挡人脸的效果）。  |
| Tattoo.h/cu        |    贴图（Tattoo）    |  输入两幅图像，一幅为前景图，一幅为背景图，输出一幅图像：其中输出图像满足当前景图灰度值与指定透明像素相同时，则输出背景图对应的灰度值否则输出前景图灰度值。  |
| Zoom.h/cu        |    放大镜定义（Zoom）    |  定义了放大镜类，将图像进行局部放大处理，根据用户要求的对图像放大的中心点、放大尺寸局部放大图片。  |



The Tesla K40 used for this research was donated by the NVIDIA Corporation.

----------

我的CSDN博客：http://blog.csdn.net/litdaguang

![联系方式](https://raw.githubusercontent.com/LitLeo/blog_pics/master/weixin.jpg)
