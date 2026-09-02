# al.sort

## 1. 函数概述

对输入张量 `ptr` 按维度进行升序或者降序的排序。

```python
al.sort(ptr, dim=-1, descending=False, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数名       | 类型     | 说明                       |
| ------------ | -------- | -------------------------- |
| `ptr`        | `tensor` | 张量数据                   |
| `dim`        | `int`    | 排序维度                   |
| `descending` | `bool`   | 是否降序                   |
| `_semantic`  | -        | 保留参数，暂不支持外部调用 |

返回值：

`x`：输出张量的 shape 与输入 x 的 shape 相同

### 2.2 特殊限制说明

- **只支持沿最后一维排序**：源码（`vec_ops.py::sort`）会把 `dim` 规整为 `rank - 1`，只要归一化后的 `dim` 不等于最后一维，就直接 `raise ValueError("ascend.sort only supports sorting along the last dimension (dim=... or -1) for shape ..., but got dim=...")`。也就是说 `dim` 实际上只能传 `-1`，或者显式传入等价于最后一维的正索引（例如 2 维张量只能传 `-1` 或 `1`），传其他维度会直接报错，而不是自动按该维度排序。
- 要求 `ptr` 的 rank ≥ 1，否则 `raise ValueError("ascend.sort requires tensor rank >= 1")`。
- `sort` 没有用 `@_tensor_member_fn` 装饰，不支持 `x.sort(...)` 这种成员函数调用写法，只能用 `al.sort(x, ...)`。

#### 2.2.1 DataType 支持

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|--- | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| A5 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | ×    | √    | ×    |
| Ascend A2/A3 | √    | √     | ×     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

另外 `float8e4nv`、`float8e5` 这两个类型也在 `allowed_types` 里。不在 `allowed_types` 中的 dtype 会直接 `raise TypeError`。

> A2/A3 一行为本仓库在真机（Ascend 910B4）实测数据：`int32`、`int64` 虽在源码 `allowed_types` 集合里（Python 侧不报错），但 NPUIR 侧不支持，编译直接失败（`MLIRCompilationError`）；`bool` 不支持（`tl.int1` 编译报错），与仓库自带的 `docs/zh/python-api/_ascend_constraints.py`（"Ascend does not support bool, fp64, int32, int64, uint8"）结论一致。

## 3. 使用方法

以下示例实现了对输入张量 `x` 做排序：

```python
@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    pid = tl.program_id(0)
    offx = tl.arange(0, M)
    offy = pid * M
    off2d = offx + offy
    x = tl.load(X + off2d)
    x = al.sort(x, dim=0, descending=descending)
    tl.store(Z + off2d, x)
```
