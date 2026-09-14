# al.math_ops

`al` 指 `triton.language.extra.cann.extension`。本文档介绍由
`third_party/ascend/language/cann/extension/math_ops.py` 提供的数学算子：`atan2`、`isfinited` 和 `finitef`。

这些算子基于 `triton.language.extra.cann.libdevice` 中的原语（`atan`、`isnan`、`isinf`）以纯 Triton
语言实现，走标准 SIMD 向量流水线下沉，因此在 Ascend A3 和 Ascend A5 上均可用。其实现中没有任何
SOC 相关的分支，因此在 A3 与 A5 上支持的数据类型完全一致。

本文示例使用的导入方式：

```python
import triton.language.extra.cann.extension as al
```

---

## 1. al.atan2

### 1.1 功能描述

逐元素计算 `y / x` 的反正切值，并根据两个参数的符号确定结果所在象限。结果为弧度值，范围
[-pi, pi]，语义与 `torch.atan2` 一致（包括 `x == 0` 的特殊情况：`y > 0` 时为 `+pi/2`，`y < 0` 时为
`-pi/2`，`y == 0` 时为 `0`）。

内部实现中，两个输入先转换为 `fp32`，除法和 `atan` 均在 `fp32` 上计算，最后将结果转回 `x` 的
dtype。因此低精度输入可以获得 `fp32` 的中间计算精度。

### 1.2 接口说明

<table>
  <tr>
    <td>Python<br>def atan2(<br>    y: tl.tensor,<br>    x: tl.tensor<br>) -&gt; tl.tensor :</td>
  </tr>
</table>

### 1.3 参数说明

<table>
  <tr>
    <td>参数</td>
    <td>类型</td>
    <td>必填</td>
    <td>描述</td>
  </tr>
  <tr>
    <td>y</td>
    <td>tl.tensor</td>
    <td>是</td>
    <td>Y 坐标张量，浮点类型（支持的数据类型见下表）</td>
  </tr>
  <tr>
    <td>x</td>
    <td>tl.tensor</td>
    <td>是</td>
    <td>X 坐标张量，浮点类型（支持的数据类型见下表），dtype 必须与 `y` 相同</td>
  </tr>
</table>

### 1.4 返回值

`tl.tensor` —— `y / x` 的逐元素反正切值，范围 [-pi, pi]，shape 和 dtype 与 `x` 相同。

### 1.5 数据类型支持

参数 `y` 与 `x` 支持相同的数据类型：

| 平台      | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
|-----------|-------|------|--------|-------|--------|-------|--------|-------|------|------|------|------|------------|-------------|------|
| Ascend A3 |   ×   |  ×   |   ×    |   ×   |   ×    |   ×   |   ×    |   ×   |  √   |  √   |  ×   |  √   |     ×      |      ×      |  ×   |
| Ascend A5 |   ×   |  ×   |   ×    |   ×   |   ×    |   ×   |   ×    |   ×   |  √   |  √   |  ×   |  √   |     ×      |      ×      |  ×   |

### 1.6 约束说明

- `y` 和 `x` 必须是浮点型张量。整型（`int8`/`int32`/...）、`bool`/`int1` 等非浮点类型在编译期报错：
  `Expected dtype fp16/fp32/bf16, but got ...`。
- `fp64` 能通过前端 dtype 检查，但会被后端拒绝（Ascend 不支持 `fp64` 的 cast/运算），因此在 A3 和
  A5 上 `fp64` 实际均不可用。
- 计算在 `fp32` 上进行，结果转回 `x` 的 dtype。若 `y` 和 `x` dtype 不同，结果 dtype 跟随 `x`
  （`y`、`x` 同 dtype 是经过验证的用法）。
- `y` 和 `x` 的 shape 必须相同（可广播）。

---

## 2. al.isfinited

### 2.1 功能描述

逐元素判断输入是否为有限值：元素既不是 NaN 也不是 ±无穷时返回 `True`，否则返回 `False`。语义等价
于 `torch.isfinite`。

实现方式为 `(~isnan(x)) & (~isinf(x))`，其中 `isnan`/`isinf` 来自 `cann.libdevice`，结果转换为
`int1`。

尽管函数名带 `d` 后缀（沿用了 libdevice 双精度版本的命名习惯），但 **不支持** `fp64`：底层
`isnan`/`isinf` 的 dispatch 表中没有 `fp64` 表项，`fp64` 输入会在编译期报错。

### 2.2 接口说明

<table>
  <tr>
    <td>Python<br>def isfinited(<br>    x: tl.tensor<br>) -&gt; tl.tensor :</td>
  </tr>
</table>

### 2.3 参数说明

<table>
  <tr>
    <td>参数</td>
    <td>类型</td>
    <td>必填</td>
    <td>描述</td>
  </tr>
  <tr>
    <td>x</td>
    <td>tl.tensor</td>
    <td>是</td>
    <td>输入张量，浮点类型（支持的数据类型见下表）</td>
  </tr>
</table>

### 2.4 返回值

`int1`（bool）类型的 `tl.tensor` —— `x` 为有限值（既非 NaN 也非 ±Inf）的位置为 `True`，否则为
`False`。

### 2.5 数据类型支持

| 平台      | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
|-----------|-------|------|--------|-------|--------|-------|--------|-------|------|------|------|------|------------|-------------|------|
| Ascend A3 |   ×   |  ×   |   ×    |   ×   |   ×    |   ×   |   ×    |   ×   |  √   |  √   |  ×   |  √   |     ×      |      ×      |  ×   |
| Ascend A5 |   ×   |  ×   |   ×    |   ×   |   ×    |   ×   |   ×    |   ×   |  √   |  √   |  ×   |  √   |     ×      |      ×      |  ×   |

### 2.6 约束说明

- `x` 必须是浮点型张量。整型（`int8`/`int32`/...）、`bool`/`int1` 等非浮点类型在编译期报错：
  `Expected dtype fp16/fp32/bf16, but got ...`。
- `fp64` 在编译期被拒绝：底层 `isnan`/`isinf` 的 extern dispatch 在 A3 和 A5 上均只定义了
  `fp32`/`fp16`/`bf16` 表项。
- 返回值为 `int1`；传回主机侧时需存入 `bool`（`torch.bool`）张量。

---

## 3. al.finitef

### 3.1 功能描述

逐元素判断输入是否为有限值，行为与 `al.isfinited` 完全一致，但输入限定为 `fp32` —— `f` 后缀沿用
libdevice/CUDA `finitef`（float 版本）的命名习惯。

实现方式与 `al.isfinited` 相同（通过 `cann.libdevice.isnan` / `cann.libdevice.isinf` 计算
`(~isnan(x)) & (~isinf(x))`），并增加了一个编译期检查，要求输入必须严格为 `float32`。

### 3.2 接口说明

<table>
  <tr>
    <td>Python<br>def finitef(<br>    x: tl.tensor<br>) -&gt; tl.tensor :</td>
  </tr>
</table>

### 3.3 参数说明

<table>
  <tr>
    <td>参数</td>
    <td>类型</td>
    <td>必填</td>
    <td>描述</td>
  </tr>
  <tr>
    <td>x</td>
    <td>tl.tensor</td>
    <td>是</td>
    <td>输入张量，仅支持 `fp32`</td>
  </tr>
</table>

### 3.4 返回值

`int1`（bool）类型的 `tl.tensor` —— `x` 为有限值（既非 NaN 也非 ±Inf）的位置为 `True`，否则为
`False`。

### 3.5 数据类型支持

| 平台      | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
|-----------|-------|------|--------|-------|--------|-------|--------|-------|------|------|------|------|------------|-------------|------|
| Ascend A3 |   ×   |  ×   |   ×    |   ×   |   ×    |   ×   |   ×    |   ×   |  ×   |  √   |  ×   |  ×   |     ×      |      ×      |  ×   |
| Ascend A5 |   ×   |  ×   |   ×    |   ×   |   ×    |   ×   |   ×    |   ×   |  ×   |  √   |  ×   |  ×   |     ×      |      ×      |  ×   |

### 3.6 约束说明

- 仅支持 `fp32` 输入，其他任何 dtype 在编译期报错：
  `finitef only supports float32, but got ...`。如需 `fp16`/`bf16` 输入，请使用 `al.isfinited`。

---

## 4. 用例示例

```python
import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
import torch_npu

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def math_ops_kernel(x_ptr, y_ptr, out_ptr, mask_ptr, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(x_ptr + idx)
    y = tl.load(y_ptr + idx)

    angle = al.atan2(y, x)                            # fp16 / fp32 / bf16 -> dtype 与 x 相同
    finite_all = al.isfinited(angle)                  # fp16 / fp32 / bf16 -> int1
    finite_f32 = al.finitef(angle.to(tl.float32))     # 仅 fp32 -> int1

    tl.store(out_ptr + idx, angle)
    tl.store(mask_ptr + idx, finite_all & finite_f32)


def main():
    N = 64
    dtype = torch.float32  # atan2 和 isfinited 也支持 fp16 / bfloat16

    x = torch.randn(N, dtype=dtype).npu()
    y = torch.randn(N, dtype=dtype).npu()
    x[1] = 0.0  # 覆盖 atan2 的 x == 0 特殊场景
    y[1] = -1.0

    out = torch.zeros(N, dtype=dtype).npu()
    mask = torch.zeros(N, dtype=torch.bool).npu()
    math_ops_kernel[(1, 1, 1)](x, y, out, mask, N=N)

    print("atan2 :", torch.allclose(out.cpu(), torch.atan2(y.cpu(), x.cpu()), atol=1e-5, rtol=1e-5))
    print("finite:", torch.equal(mask.cpu(), torch.isfinite(out.cpu())))


if __name__ == "__main__":
    main()
```

相关单元测试：`third_party/ascend/unittest/pytest_ut/test_atan2.py` 和
`third_party/ascend/unittest/pytest_ut/test_isfinited.py`（覆盖 `isfinited` 的
`fp32`/`fp16`/`bf16` 及 NaN/±Inf 取值，以及 `finitef` 的 `fp32` 用例）。

---

## 5. 数据类型表的说明

- 上表反映的是代码中的实际支持情况：
  - `atan2`（`extension/math_ops.py`）：前端接受任意浮点类型，但内部基于 `fp32` 的计算方式以及后端
    不支持 `fp64`，实际限制为 `fp16`/`fp32`/`bf16`。
  - `isfinited`（`extension/math_ops.py`）：委托给 `cann.libdevice.isnan`/`isinf`，其 SIMD dispatch
    表中恰好只列出 `fp32`/`fp16`/`bf16`。
  - `finitef`（`extension/math_ops.py`）：额外断言 `x.dtype == float32`。
- A5 列已在 Ascend 950PR（A5）硬件上通过编译运行验证：`fp16`/`fp32`/`bf16` 可正常编译且结果与
  `torch` 参考一致，整型/bool 输入以及 `fp64` 在编译期报错。A3 列的依据是：这些包装器及底层
  dispatch 表均与 SOC 无关（唯一的 SOC 相关分支 `TRITON_ENABLE_LIBDEVICE_SIMT` 仅存在于 A5，且只
  提供供 SIMT 模式 kernel 使用的 `fp32` extern；本文档中的 `al` 包装器始终使用 SIMD dispatch）。
