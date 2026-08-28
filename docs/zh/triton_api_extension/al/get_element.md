# al.get_element

## 1. 函数概述

根据给定的索引，从输入张量中读取单个元素。

```python
al.get_element(
    src,
    indice,
    _semantic=None,
    _generator=None
) -> scalar
```

可以作为 tensor 的成员函数调用，如 `x.get_element(...)`，与 `get_element(x, ...)` 等效。

## 2. 规格

### 2.1 参数说明

| 参数名       | 类型                                  | 说明                       |
| ------------ | ------------------------------------- | -------------------------- |
| `src`        | `tensor`                              | 要被访问的源张量           |
| `indice`     | `tuple of ints` 或 `tuple of tensors` | 用于指定元素位置的索引     |
| `_semantic`  | -                                     | 保留参数，暂不支持外部调用 |
| `_generator` | -                                     | 保留参数，暂不支持外部调用 |

返回值：

`scalar`：与 `src` 张量元素类型相同的标量值

### 2.2 说明

1. `indice` 的长度必须与 `src` 张量的维度数（rank）相同，否则会抛出 `ValueError("Indice's rank must be equal to src tensor's rank")`（这是一个普通异常，不是 assert，可以被正常捕获）。

## 3. 使用方法

以下示例实现了 `get_element` 的调用：

```python
@triton.jit
def index_select_manual_kernel(in_ptr, indices_ptr, out_ptr, dim,
                                g_stride: tl.constexpr, indice_length: tl.constexpr,
                                g_block: tl.constexpr, g_block_sub: tl.constexpr,
                                other_block: tl.constexpr):
    """
    Manual implementation using tl.get_element and tl.insert_slice.
    """
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)
        for other_offset in range(0, g_stride, other_block):
            tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride
            # Manual gather: iterate over each index
            for i in range(0, g_block_sub):
                gather_offset = al.get_element(indices, (i,)) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                tmp_buf = al.insert_slice(tmp_buf, val[None, :],
                                          offsets=(i, 0), sizes=(1, other_block), strides=(1, 1))
            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :],
                     tmp_buf, g_mask[:, None] & other_mask[None, :])
```
