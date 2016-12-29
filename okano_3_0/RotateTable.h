// RotateTable.h
// 创建人：于玉龙
//
// 旋转表生成（Rotation Table）
// 功能说明：根据设定的旋转变换范围和变换步长，求出给定坐标点集在所有旋转范围内
//           各旋转角度的下得到的新坐标。
//
// 修订历史：
// 2012年08月20日（于玉龙）
//     初始版本。
// 2012年08月21日（于玉龙）
//     实现了并行的旋转表计算。
// 2012年09月09日（于玉龙）
//     增加了旋转表的输出形式，增加了分别由两个数组表示 x 和 y 分量的重载函数。
//     修正了旋转表计算 GPU 代码中一个 Bug。
// 2012年09月10日（于玉龙）
//     增加了旋转表输入参数的形式，增加了分别由两个数组表示 x 和 y 分量的重载函
//     数。
// 2012年10月19日（于玉龙）
//     按照新的需求，修改了代码，将旋转表存入 CLASS 实例内部，并提供在 Device
//     上访问的旋转表内容的函数。
// 2012年10月20日（于玉龙）
//     增加了 dispose 函数，简化析构逻辑。
// 2012年10月25日（于玉龙）
//     修改了 Kernel 函数中计算角度时的 Bug。
// 2012年11月30日（于玉龙）
//     修改了角度和角度索引值与总角度数量之间的计算错误，调整了不合理的角度分
//     配策略。
// 2012年12月05日（于玉龙）
//     优化了 detAngle 设定时的限定条件，使其更加合理。

#ifndef __ROTATETABLE_H__
#define __ROTATETABLE_H__

#include "ErrorCode.h"

// 宏：NULL_RTT（旋转表为空）
// 旋转表的状态，当旋转表 CLASS 实例刚刚初始化完毕后，旋转表处于该状态。在该状
// 态下旋转表不可用，但允许修改旋转表中的各种参数。通过调用方法 initRotateTable
// 跳出该状态。
#define NULL_RTT    0

// 宏：READY_RTT（旋转表就绪）
// 旋转标的状态，当旋转表 CLASS 实例中旋转表处于可用状态时处于此状态。在该状态
// 中，旋转表可用，但是不允许修改旋转表中的各项参数。
#define READY_RTT   1

// 类：RotateTable
// 继承自：无
// 根据设定的旋转变换范围和变换步长，求出给定坐标点集在所有旋转范围内各旋转角度
// 的下得到的新坐标。旋转表得到一个矩阵，该矩阵的各行表示不同的旋转角度，各旋转
// 角度从小到大依次存放在各行中；矩阵各列对应于输入点集的各点，在不同的旋转角度
// 下的新坐标。
class RotateTable {

protected:

    // 成员变量：minAngle（旋转范围下限）
    // 生成旋转表中的最小的角度，单位是“度”（°）。
    float minAngle;

    // 成员变量：maxAngle（旋转范围上限）
    // 生成的旋转表中的角度上限，如果 maxAngle - minAngle 可以被 detAngle 整除
    // (通常情况下都会是这样的），则旋转表中的最后一列为 maxAngle 角度对应的旋
    // 转结果。单位是“度”（°）。
    float maxAngle;

    // 成员变量：detAngle（旋转步长）
    // 旋转表中各行之间的角度差距。单位是“度”（°）。
    float detAngle;

    // 成员变量：sizex 和 sizey（旋转表的尺寸）
    // 规定了旋转表中包含的坐标范围。
    int sizex, sizey;

    // 成员变量：offsetx, offsety（坐标偏移量）
    // 这个值根据旋转表的尺寸计算得到，外界无法访问到这个值，它随着设定旋转表的
    // 尺寸而设定，用来方便计算，并保证未来的可扩展性。该变量的实际含义是旋转表
    // 中行或者列的起始处所对应的坐标。
    int offsetx, offsety;

    // 成员变量：rttx 和 rtty（旋转表）
    // 旋转表。这是一个逻辑上三维的表，各个维度表示的含义是：列、行、角度。这里
    // 为了更快的访存，采用了 x 与 y 坐标分立的形式。
    float *rttx, *rtty;

    // 成员变量：curState（当前状态）
    // 用来表示实例当前的状态。目前可选的值为 NULL_RTT 和 READY_RTT。该值对于外
    // 界是只读的。
    int curState;

public:

    // 构造函数：RotateTable
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    RotateTable()
    {
        // 使用默认值初始化各个成员变量。
        this->minAngle = -30.0f;    // 旋转范围下线的初始值为 -30。
        this->maxAngle =  30.0f;    // 旋转范围上限的初始值为 30。
        this->detAngle =   0.2f;    // 旋转步长的初始值为 0.2。
        this->sizex = 11;           // 旋转表尺寸 x 分量初始值。
        this->sizey = 11;           // 旋转标尺寸 y 分量初始值。
        this->offsetx = -5;         // 坐标偏移量 x 分量初始值。
        this->offsety = -5;         // 坐标偏移量 y 分量初始值。
        this->rttx = NULL;          // 旋转表 x 分量表初始化为 NULL。
        this->rtty = NULL;          // 旋转表 y 分量表初始化为 NULL。
        this->curState = NULL_RTT;  // 状态的初始值为 NULL_RTT，即旋转表尚未被
                                    // 计算。
    }

    // 构造函数：RotateTable
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中还
    // 是可以改变的。
    __host__ __device__
    RotateTable(
            float minangle,       // 旋转范围角度下限
            float maxangle,       // 旋转范围角度上限
            float detangle,       // 旋转步长
            int sizex, int sizey  // 旋转表的尺寸
    ) {
        // 使用默认值初始化各个成员变量。
        this->minAngle = -30.0f;    // 旋转范围下线的初始值为 -30。
        this->maxAngle =  30.0f;    // 旋转范围上限的初始值为 30。
        this->detAngle =   0.2f;    // 旋转步长的初始值为 0.2。
        this->sizex = 11;           // 旋转表尺寸 x 分量初始值。
        this->sizey = 11;           // 旋转标尺寸 y 分量初始值。
        this->offsetx = -5;         // 坐标偏移量 x 分量初始值。
        this->offsety = -5;         // 坐标偏移量 y 分量初始值。
        this->rttx = NULL;          // 旋转表 x 分量表初始化为 NULL。
        this->rtty = NULL;          // 旋转表 y 分量表初始化为 NULL。
        this->curState = NULL_RTT;  // 状态的初始值为 NULL_RTT，即旋转表尚未被
                                    // 计算。

        // 根据参数列表中的值设定成员变量的初值
        this->setMinAngle(minangle);
        this->setMaxAngle(maxangle);
        this->setDetAngle(detangle);
        this->setSizeX(sizex);
        this->setSizeY(sizey);
    }

    // 成员方法： getMinAngle（获取旋转范围下限）
    // 获取成员变量 minAngle 的值。
    __host__ __device__ float  // 返回值：成员变量 minAngle 的值
    getMinAngle() const
    {
        // 返回成员变量 minAngle 的值。
        return this->minAngle;
    }

    // 成员方法：setMinAngle（设置旋转范围下限）
    // 设置成员变量 minAngle 的值。如果新的 minAngle 的值大于或等于 maxAngle 的
    // 值，则会适当的调整 maxAngle 的值。采用这中容忍错误的做法，是因为程序如果
    // 在随后也设置了 maxAngle，不会导致程序执行中存在陷阱。如，当前的旋转范围
    // 为 [-10, 10]，而新的范围是 [20, 40]，如果不使用容忍方式，则程序设置会，
    // 会变为 [-10, 40]（如果出错则不处理，直接返回）或 [9.8, 40]（如果出错，则
    // 调整最小值到最合适后返回）。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMinAngle(
            float minangle   // 新的旋转范围角度下限。
    ) {
        // 如果实例当前不是处于旋转表还未计算的状态，则直接报错，因为这样会导致
        // 系统状态的混乱。
        if (this->curState != NULL_RTT)
            return INVALID_DATA;
        // 将新的旋转范围下限赋值给 minAngle。
        this->minAngle = minangle;

        // 如果新的旋转范围下限大于当前旋转角度上限，则调整角度上限为新的下限
        if (minangle > this->maxAngle)
            this->maxAngle = minangle;

        // 处理完毕，返回。
        return NO_ERROR;
    }

    // 成员方法： getMaxAngle（获取旋转范围上限）
    // 获取成员变量 maxAngle 的值。
    __host__ __device__ float  // 返回值：成员变量 maxAngle 的值
    getMaxAngle() const
    {
        // 返回成员变量 maxAngle 的值。
        return this->maxAngle;
    }

    // 成员方法：setMaxAngle（设置旋转范围上限）
    // 设置成员变量 maxAngle 的值。如果新的 maxAngle 的值小于或等于 minAngle 的
    // 值，则会适当的调整 minAngle 的值。采用这中容忍错误的做法，理由同
    // setMinAngle 相同。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setMaxAngle(
            float maxangle   // 新的旋转范围角度上限。
    ) {
        // 如果实例当前不是处于旋转表还未计算的状态，则直接报错，因为这样会导致
        // 系统状态的混乱。
        if (this->curState != NULL_RTT)
            return INVALID_DATA;
        // 将新的旋转范围上限赋值给 maxAngle。
        this->maxAngle = maxangle;

        // 如果新的旋转范围上限小于当前旋转角度下限，则调整角度下限为新的上限值
        if (maxangle < this->minAngle)
            this->minAngle = maxangle;

        // 处理完毕，返回。
        return NO_ERROR;
    }

    // 成员方法：getDetAngle（获取旋转步长）
    // 获取成员变量 detAngle 的值。
    __host__ __device__ float  // 返回值：成员变量 detAngle的值
    getDetAngle() const
    {
        // 返回成员变量 detAngle 的值。
        return this->detAngle;
    }

    // 成员方法：setDetAngle（设置旋转步长）
    // 设置成员变量 detAngle 的值。该值要求必须足够大，以保证能够不发生数值误
    // 差。如果指定的值太小，设置函数将会报错，并不会设置新值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setDetAngle(
            float detangle   // 新的旋转步长。
    ) {
        // 如果实例当前不是处于旋转表还未计算的状态，则直接报错，因为这样会导致
        // 系统状态的混乱。
        if (this->curState != NULL_RTT)
            return INVALID_DATA;
        // 需要保证 detangle 是一个正数，因此如果是负数，则翻转之。
        if (detangle < 0.0f)
            detangle = -detangle;

        // 如果 detangle 相对于旋转范围来说太小，则会引起数值错误，此外，也会使
        // 求得出的旋转表太大，因此如果 detangle 太小，则直接报错退出，不进行任
        // 何操作。
        if (detangle < 1.0e-6 * (this->maxAngle - this->minAngle) ||
            detangle < 1.0e-8)
            return INVALID_DATA;

        // 将新的旋转步长赋值给 detAngle。
        this->detAngle = detangle;
        return NO_ERROR;
    }

    // 成员方法：getSizeX（获取旋转表尺寸的 x 分量）
    // 获取成员变量 sizex 的值。
    __host__ __device__ int  // 返回值：成员变量 sizex 的值。
    getSizeX() const
    {
        // 返回成员变量 sizex 的值。
        return this->sizex;
    }

    // 成员方法：setSizeX（设置旋转表尺寸的 x 分量）
    // 设置成员变量 sizex 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setSizeX(
            int sizex        // 新的旋转表尺寸的 x 分量。
    ) {
        // 如果实例当前不是处于旋转表还未计算的状态，则直接报错，因为这样会导致
        // 系统状态的混乱。
        if (this->curState != NULL_RTT)
            return INVALID_DATA;
        // 如果新的 sizex 小于 1，则报错，因为尺寸是不可能小于 1 的。
        if (sizex < 1)
            return INVALID_DATA;

        // 将新的 sizex 设置到 sizex 成员变量中。
        this->sizex = sizex;

        // 计算新的 offsetx，整理要求原点在整个旋转表的中心，因此对于一个奇数
        // 2 * k + 1，则 offsetx 为 -k，这样整个旋转表的范围为 [-k, k]；对于一
        // 个偶数 2k，则 offsetx 为 -(k - 1)，这样整个旋转表的范围为 [-(k -1),
        // k]。
        this->offsetx = -((sizex - 1) / 2);

        // 处理完毕，返回。
        return NO_ERROR;
    }

    // 成员方法：getSizeY（获取旋转表尺寸的 y 分量）
    // 获取成员变量 sizey 的值。
    __host__ __device__ int  // 返回值：成员变量 sizey 的值。
    getSizeY() const
    {
        // 返回成员变量 sizey 的值。
        return this->sizey;
    }

    // 成员方法：setSizeY（设置旋转表尺寸的 y 分量）
    // 设置成员变量 sizey 的值。
    __host__ __device__ int  // 返回值：函数是否正确执行，若函数正确执行，返回
                             // NO_ERROR。
    setSizeY(
            int sizey        // 新的旋转表尺寸的 y 分量。
    ) {
        // 如果实例当前不是处于旋转表还未计算的状态，则直接报错，因为这样会导致
        // 系统状态的混乱。
        if (this->curState != NULL_RTT)
            return INVALID_DATA;
        // 如果参数中新的尺寸小于 1，则直接报错。因为不可能存在尺寸小于 1 的情
        // 况。
        if (sizey < 1)
            return INVALID_DATA;

        // 设置新的旋转表尺寸，然后返回。
        this->sizey = sizey;

        // 计算新的 offsety，整理要求原点在整个旋转表的中心，因此对于一个奇数
        // 2 * k + 1，则 offsety 为 -k，这样整个旋转表的范围为 [-k, k]；对于一
        // 个偶数 2k，则 offsety 为 -(k - 1)，这样整个旋转表的范围为 [-(k -1),
        // k]。
        this->offsety = -((sizey - 1) / 2);

        return NO_ERROR;
    }

    // 成员方法：getOffsetX（获取坐标偏移量 x 分量）
    // 获取成员变量 offsetx 的值。
    __host__ __device__ int  // 返回值：成员变量 offsetx 的值。
    getOffsetX() const
    {
        // 直接返回成员变量 offset 的值。
        return this->offsetx;
    }

    // 成员方法：getOffsetY（获取坐标偏移量 y 分量）
    // 获取成员变量 offsety 的值。
    __host__ __device__ int  // 返回值：成员变量 offsety 的值。
    getOffsetY() const
    {
        // 直接返回成员变量 offsety 的值。
        return this->offsety;
    }

    // 成员方法：getCurrentState（获取当前实例所处状态）
    // 获取成员变量 curState 的值。
    __host__ __device__ int  // 返回值：成员变量 curState 的值。
    getCurrentState() const
    {
        // 直接返回成员变量 curState 的值。
        return this->curState;
    }

    // Device 成员方法：getRotateTableX（获取 x 分量旋转表）
    // 获取成员变量 rttx 的值
    __device__ float *  // 返回值：成员变量 rttx 的值。
    getRotateTableX() const
    {
        // 直接返回成员变量 rttx 的值。
        return this->rttx;
    }

    // Device 成员方法：getRotateTableY（获取 y 分量旋转表）
    // 获取成员变量 rtty 的值
    __device__ float *  // 返回值：成员变量 rtty 的值。
    getRotateTableY() const
    {
        // 直接返回成员变量 rtty 的值。
        return this->rtty;
    }

    // 成员方法：getAngleCount（获取旋转表中不同角度的数量）
    // 获取旋转表中不同角度的数量。
    __host__ __device__ int  // 返回值：旋转角度的数量。
    getAngleCount() const
    {
        // 利用区间长度和角度步长的商求出旋转表的总行数（即不同角度的数量）。该
        // 表达式最后的 1.9f 用于校正由于不能整除导致的错误，其中，1.0f 用于校
        // 正 maxAngle - minAngle 所带来的个数缺失，而 0.9f 用于弥补由于不能整
        // 除所带来的额外的一张旋转表。
        return (int)((maxAngle - minAngle) / detAngle + 1.9f);
    }

    // 成员方法：getAngleIdx（获取角度对应的旋转表行数）
    // 给定一个角度，返回该角度对应的旋转表的行数。对于不是正好在表中的角度，则
    // 按照四舍五入的原则返回最近的一个行号。如果给定的角度不在角度范围之内，则
    // 返回错误码所代表的哑值。
    __host__ __device__ int  // 返回值：输入角度对应的行数
    getAngleIdx(
            float angle      // 输入参数角度
    ) const {
        // 如果给定的角度不在角度范围内，则报错退出。
        if (angle < minAngle - detAngle / 2.0f || 
            angle > maxAngle + detAngle / 2.0f)
            return INVALID_DATA;

        // 根据当前相对角度于角度范围下限之差，再除以角度步长，得到角度对应的旋
        // 转表行数。该表达式最后的 0.5 用于四舍五入。
        return (int)((angle - minAngle) / detAngle + 0.5f);
    }

    // 成员方法：getAngleVal（获取旋转表指定行的角度值）
    // 给定一个旋转表的行号，求出该行号对应的角度值。如果给定的行号小于 0 或者
    // 大于总行数，则返回一个角度范围外的角度值。
    __host__ __device__ float  // 返回值：旋转表指定行对应的角度值
    getAngleVal(
            int idx            // 旋转表中的行号
    ) const {
        // 如果给定的行数小于 0，则返回一个比 minAngle 更小的角度。
        if (idx < 0)
            return minAngle - detAngle;
        // 如果给定的行数大于或等于总行数，则返回一个比 maxAngle 更大的角度。
        if (idx >= getAngleCount())
            return maxAngle + detAngle;

        // 根据行数求出角度。注意，这个角度还不能直接返回，因为对于最后一个角度
        // 值很可能是超过 maxAngle 的值，因此，需要把它削减到 maxAngle。
        float resangle = minAngle + detAngle * idx;
        if (resangle > maxAngle)
            resangle = maxAngle;

        // 返回求出的角度值。
        return resangle;
    }

    // Host 成员方法：initRotateTable（计算旋转表）
    // 计算给定的坐标点集对应的旋转表。计算后的旋转表被存储在 CLASS 实例中的
    // rttx 和 rtty 中，这两个数组的空间申请在该函数中完成。该函数运行完毕后，
    // CLASS 实例进入 READY_RTT 状态，这个状态下可以使用 getRotatePos 方法获取
    // 旋转的信息，但是其他的各种 set 函数都不能够在被使用了。
    __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR。
    initRotateTable();

    // Host 成员方法：disposeRotateTable（销毁旋转表）
    // 释放旋转表所占用的内存空间，使 CLASS 实例从 READY_RTT 状态返回 NULL_RTT
    // 状态。正式析构 CLASS 实例前，必须调用这个函数以保证不会产生内存泄漏。这
    // 里没有将这个功能设计到析构函数中，是为了简化内存管理的代码。用户使用的时
    // 候通常要保证 initRotateTable 和 disposeRotateTable 的成对出现，这样就能
    // 够保证避免内存泄漏。
    __host__ int  // 返回值：函数是否正确执行，若函数正确执行，返回 NO_ERROR。
    disposeRotateTable();

    // Device 成员方法：getRotatePos（访问旋转表获得坐标点旋转后的坐标）
    // 当 CLASS 实例处于 READY_RTT 状态的时候可以调用这个函数，该函数根据输入的
    // 坐标，返回经过给定角度旋转后，这一点所处的坐标。如果给定的角度，不在旋转
    // 表所记录的角度上，则会按照给定的插值规则插值得到对应的坐标值，目前采用的
    // 插值方法是临近插值，即选择最近的旋转表中记录的角度返回相应的值。如果给定
    // 的坐标值或者角度超出了旋转表所支持的范围，则输出参数 rx 和 ry 不会被赋
    // 值，整个方法会报错退出。
    __device__ int 
    getRotatePos(
            int x, int y, 
            float angle,          // 旋转角度
            float &rx, float &ry  // 旋转后的坐标，这是输出参数，因此必须给顶一
                                  // 个左值
    ) const {
        // 检查旋转表所处的状态，如果不处于可用状态，则直接报错退出。该检查可能
        // 会导致性能下降，因此如果用户可以保证系统的正确性，建议注释掉该语句。
        if (this->curState != READY_RTT)
            return NULL_POINTER;

        // 如果输入的参数不在旋转表所支持的范围内，则会报错退出。当然该语句也可
        // 能会导致性能下降，因此如果用户可以保证系统的正确性，建议注释掉该语
        // 句。
        if (x < this->offsetx || x >= this->sizex + this->offsetx ||
            y < this->offsety || y >= this->sizey + this->offsety ||
            angle < this->minAngle || angle > this->maxAngle)
            return INVALID_DATA;

        // 获取用户给定的角度对应的的旋转表的序号。
        int rttidx = this->getAngleIdx(angle);

        // 计算用户所要计算的点经过给定角度旋转后的坐标值在旋转表中的下标。
        int arridx = (rttidx * this->sizey + 
                      y - this->offsety) * this->sizex + 
                     x - this->offsetx;

        // 从旋转表中读取坐标旋转后的值。
        rx = this->rttx[arridx];
        ry = this->rtty[arridx];

        // 处理完毕，退出。
        return NO_ERROR;
    }
};

#endif

