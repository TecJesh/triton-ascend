"""
全量算子 DataType 支持情况测试

覆盖所有算子的最小测试用例，统计对以下15种数据类型的支持情况：
| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |

用法:
    pytest test_all_dtype_support.py -v

输出: dtype_support_result.md (Markdown 表格)
"""

import math
import os
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import test_common

# ── 配置 ──────────────────────────────────────────────────────────
ALL_DTYPES = [
    "uint8", "int8",
    "uint16", "int16",
    "uint32", "int32",
    "uint64", "int64",
    "fp16", "fp32", "fp64", "bf16",
    "fp8e4m3", "fp8e5m2",
    "bool",
]

DTYPE_DISPLAY = {
    "uint8": "uint8", "int8": "int8",
    "uint16": "uint16", "int16": "int16",
    "uint32": "uint32", "int32": "int32",
    "uint64": "uint64", "int64": "int64",
    "fp16": "fp16", "fp32": "fp32", "fp64": "fp64", "bf16": "bf16",
    "fp8e4m3": "fp8e(e4m3)", "fp8e5m2": "fp8e5(e5m2)",
    "bool": "bool",
}

FLOAT_SET = {"fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"}
INT_SET = {"uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"}
BOOL_SET = {"bool"}

# ── 全局结果记录 ──────────────────────────────────────────────────
_results: dict = {}  # {op_name: {dtype: "√"/"×"}}


def _record(op: str, dtype: str, ok: bool):
    _results.setdefault(op, {})[dtype] = "√" if ok else "×"


# ── 辅助：dtype → torch dtype ─────────────────────────────────────

def _td(name: str):
    return {
        "uint8": torch.uint8, "int8": torch.int8,
        "uint16": torch.uint16, "int16": torch.int16,
        "uint32": torch.uint32, "int32": torch.int32,
        "uint64": torch.uint64, "int64": torch.int64,
        "fp16": torch.float16, "fp32": torch.float32,
        "fp64": torch.float64, "bf16": torch.bfloat16,
        "fp8e4m3": torch.float8_e4m3fn, "fp8e5m2": torch.float8_e5m2,
        "bool": torch.bool,
    }[name]


def _make_input(dtype: str, shape=(128,), positive: bool = False):
    """生成指定 dtype 的随机输入张量（NPU 设备）。
    参照 test_dtype_support.py：直接用 device='npu' 创建，避免 CPU→NPU 搬运时 dtype 被悄悄改变。
    """
    if dtype in ("fp8e4m3", "fp8e5m2"):
        t = torch.randint(0, 255, shape, dtype=torch.int8, device='npu')
        if dtype == "fp8e4m3":
            t[(t & 0b01111100) == 0b01111100] = 0
        return t.view(_td(dtype))
    if dtype in ("uint16", "uint32", "uint64"):
        bw = int(dtype.replace("uint", ""))
        t = torch.randint(0, min(2**(bw-1), 10000), shape,
                          dtype=getattr(torch, f"int{bw}"), device='npu')
        return t.view(_td(dtype))
    if dtype.startswith(("uint", "int")):
        lo = -64 if dtype.startswith("int") else 0
        hi = 64 if dtype.startswith("int") else 128
        return torch.randint(lo, hi, shape, dtype=_td(dtype), device='npu')
    if dtype == "bool":
        return torch.randint(0, 2, shape, dtype=torch.bool, device='npu')
    # 所有 float 类型（fp16/fp32/fp64/bf16）：直接在 NPU 上创建
    # fp64 在 NPU 上会有 warning 但 dtype 保留，传给 triton kernel 会触发编译错误 → 正确标记为 ×
    t = torch.randn(shape, dtype=_td(dtype), device='npu')
    if positive:
        t = t.abs() + 0.01
    return t


def _make_output(dtype: str, shape=(128,)):
    """创建输出张量，直接在 NPU 上分配"""
    if dtype in ("fp8e4m3", "fp8e5m2"):
        return torch.empty(shape, dtype=torch.int8, device='npu')
    return torch.empty(shape, dtype=_td("bool" if dtype == "cmp" else dtype), device='npu')


# ══════════════════════════════════════════════════════════════════
#  一、元素级一元算子 (Elementwise Unary)
# ══════════════════════════════════════════════════════════════════

UNARY_OPS = {
    "abs":            "tl.abs(x)",
    "ceil":           "tl.math.ceil(x)",
    "cos":            "tl.cos(x)",
    "erf":            "tl.math.erf(x)",
    "exp":            "tl.exp(x)",
    "exp2":           "tl.math.exp2(x)",
    "floor":          "tl.math.floor(x)",
    "log":            "tl.log(x)",
    "log2":           "tl.math.log2(x)",
    "neg":            "-x",
    "invert":         "~x",                       # bitwise invert (~)
    "not_":           "tl.math.bitnot(x)",        # bitwise NOT (tl.math.bitnot)
    "sigmoid":        "tl.sigmoid(x)",
    "sin":            "tl.sin(x)",
    "sqrt":           "tl.sqrt(x)",
    "sqrt_rn":        "tl.math.sqrt_rn(x)",       # sqrt round-to-nearest
    "rsqrt":          "tl.math.rsqrt(x)",
    "softmax":        "tl.softmax(x)",
}


@triton.jit
def _unary_kernel(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx, mask=idx < N)
    z = GENERATE_TEST_HERE
    tl.store(out_ptr0 + idx, z, mask=idx < N)


def _test_unary(op: str, expr: str, dtype: str, positive: bool = False):
    """测试一元算子"""
    if op in ("log", "log2", "sqrt", "rsqrt", "exp", "exp2") and dtype in INT_SET | BOOL_SET:
        return _record(op, dtype, False)  # 数学上无意义

    if op in ("not_", "invert") and dtype in FLOAT_SET:
        return _record(op, dtype, False)

    k = triton.JITFunction(_unary_kernel.fn)
    k._unsafe_update_src(k.src.replace("GENERATE_TEST_HERE", expr))
    try:
        x = _make_input(dtype, positive=positive)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=x.numel(), NUMEL=128)
        _record(op, dtype, True)
    except Exception:
        _record(op, dtype, False)


# ══════════════════════════════════════════════════════════════════
#  二、元素级二元算子 (Elementwise Binary)
# ══════════════════════════════════════════════════════════════════

BINARY_OPS = {
    "add":            "x + y",
    "sub":            "x - y",
    "mul":            "x * y",
    "div":            "x / y",
    "floordiv":       "tl.math.floordiv(x, y)",
    "mod":            "x % y",
    "and_":           "x & y",                   # bitwise AND
    "or_":            "x | y",                   # bitwise OR
    "xor":            "x ^ y",                   # bitwise XOR
    "lshift":         "x << y",
    "rshift":         "x >> y",
    "maximum":        "tl.maximum(x, y)",
    "minimum":        "tl.minimum(x, y)",
    "cdiv":           "tl.math.cdiv(x, y)",
    "fdiv":           "tl.math.fdiv(x, y)",
    "div_rn":         "tl.math.div_rn(x, y)",
    "umulhi":         "tl.math.umulhi(x, y)",
}


@triton.jit
def _binary_kernel(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx, mask=idx < N)
    y = tl.load(in_ptr1 + idx, mask=idx < N)
    z = GENERATE_TEST_HERE
    tl.store(out_ptr0 + idx, z, mask=idx < N)


def _test_binary(op: str, expr: str, dtype: str):
    if op in ("and_", "or_", "xor", "lshift", "rshift", "mod", "cdiv", "umulhi"):
        if dtype in FLOAT_SET | BOOL_SET:
            return _record(op, dtype, False)

    if op in ("div_rn", "fdiv", "div"):
        if dtype in BOOL_SET:
            return _record(op, dtype, False)

    if op in ("maximum", "minimum"):
        if dtype in BOOL_SET:
            return _record(op, dtype, False)

    k = triton.JITFunction(_binary_kernel.fn)
    k._unsafe_update_src(k.src.replace("GENERATE_TEST_HERE", expr))
    try:
        x = _make_input(dtype)
        y = _make_input(dtype, positive=(op in ("div", "fdiv", "div_rn", "floordiv", "mod", "cdiv")))
        z = _make_output(dtype)
        k[(1, 1, 1)](x, y, z, N=x.numel(), NUMEL=128)
        _record(op, dtype, True)
    except Exception:
        _record(op, dtype, False)


# ══════════════════════════════════════════════════════════════════
#  三、比较运算 (Comparison) → 输出 bool
# ══════════════════════════════════════════════════════════════════

COMPARE_OPS = {
    "eq":  "x == y",
    "ne":  "x != y",
    "lt":  "x < y",
    "le":  "x <= y",
    "gt":  "x > y",
    "ge":  "x >= y",
}


@triton.jit
def _cmp_kernel(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx, mask=idx < N)
    y = tl.load(in_ptr1 + idx, mask=idx < N)
    z = GENERATE_TEST_HERE
    tl.store(out_ptr0 + idx, z, mask=idx < N)


def _test_compare(op: str, expr: str, dtype: str):
    k = triton.JITFunction(_cmp_kernel.fn)
    k._unsafe_update_src(k.src.replace("GENERATE_TEST_HERE", expr))
    try:
        x = _make_input(dtype)
        y = _make_input(dtype)
        z = _make_output("cmp")
        k[(1, 1, 1)](x, y, z, N=x.numel(), NUMEL=128)
        _record(op, dtype, True)
    except Exception:
        _record(op, dtype, False)


# ══════════════════════════════════════════════════════════════════
#  四、逻辑运算 (Logical) → 返回 bool
# ══════════════════════════════════════════════════════════════════

LOGICAL_OPS = {
    "logical_and": "x.logical_and(y)",
    "logical_or":  "x.logical_or(y)",
}


@triton.jit
def _logical_kernel(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx, mask=idx < N)
    y = tl.load(in_ptr1 + idx, mask=idx < N)
    z = GENERATE_TEST_HERE
    tl.store(out_ptr0 + idx, z, mask=idx < N)


def _test_logical(op: str, expr: str, dtype: str):
    k = triton.JITFunction(_logical_kernel.fn)
    k._unsafe_update_src(k.src.replace("GENERATE_TEST_HERE", expr))
    try:
        x = _make_input(dtype)
        y = _make_input(dtype)
        z = _make_output("cmp")
        k[(1, 1, 1)](x, y, z, N=x.numel(), NUMEL=128)
        _record(op, dtype, True)
    except Exception:
        _record(op, dtype, False)


# ══════════════════════════════════════════════════════════════════
#  五、三元运算 (Ternary)
# ══════════════════════════════════════════════════════════════════

TERNARY_OPS = {
    "clamp": "tl.clamp(x, y, w)",
    "fma":   "tl.math.fma(x, y, w)",
}


@triton.jit
def _ternary_kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx, mask=idx < N)
    y = tl.load(in_ptr1 + idx, mask=idx < N)
    w = tl.load(in_ptr2 + idx, mask=idx < N)
    z = GENERATE_TEST_HERE
    tl.store(out_ptr0 + idx, z, mask=idx < N)


def _test_ternary(op: str, expr: str, dtype: str):
    k = triton.JITFunction(_ternary_kernel.fn)
    k._unsafe_update_src(k.src.replace("GENERATE_TEST_HERE", expr))
    try:
        x = _make_input(dtype)
        y = _make_input(dtype, positive=True)
        w = _make_input(dtype, positive=True)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, y, w, z, N=x.numel(), NUMEL=128)
        _record(op, dtype, True)
    except Exception:
        _record(op, dtype, False)


# ══════════════════════════════════════════════════════════════════
#  六、创建类算子 (Creation Ops)
# ══════════════════════════════════════════════════════════════════

def _test_arange(dtype: str):
    """arange: 生成 [0, 128) 的整数序列 → 只有 int32 通常支持"""
    if dtype not in ("int32", "int64", "uint32", "uint64"):
        return _record("arange", dtype, False)

    @triton.jit
    def k(out_ptr0, START: tl.constexpr, END: tl.constexpr):
        off = tl.arange(0, 128)
        val = tl.arange(START, END)
        tl.store(out_ptr0 + off, val)

    try:
        z = _make_output(dtype, (128,))
        k[(1, 1, 1)](z, START=0, END=128)
        _record("arange", dtype, True)
    except Exception:
        _record("arange", dtype, False)


def _test_full(dtype: str):
    """full: 填充常量"""

    @triton.jit
    def k(out_ptr0, val: tl.constexpr, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        z = tl.full((N,), val, tl.constexpr(tl.int32))
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        z = _make_output(dtype)
        k[(1, 1, 1)](z, val=42, N=128, NUMEL=128)
        _record("full", dtype, True)
    except Exception:
        _record("full", dtype, False)


def _test_zeros(dtype: str):
    """zeros: 全零"""

    @triton.jit
    def k(out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        z = tl.zeros((N,), tl.float32)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        z = _make_output(dtype)
        k[(1, 1, 1)](z, N=128, NUMEL=128)
        _record("zeros", dtype, True)
    except Exception:
        _record("zeros", dtype, False)


def _test_zeros_like(dtype: str):
    """zeros_like"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        z = tl.zeros_like(x)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("zeros_like", dtype, True)
    except Exception:
        _record("zeros_like", dtype, False)


def _test_cat(dtype: str):
    """cat: 拼接两个张量"""

    @triton.jit
    def k(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N // 2)
        y = tl.load(in_ptr1 + idx, mask=idx < N // 2)
        z = tl.cat(x, y, can_reorder=True)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype, (64,))
        y = _make_input(dtype, (64,))
        z = _make_output(dtype, (128,))
        k[(1, 1, 1)](x, y, z, N=128, NUMEL=128)
        _record("cat", dtype, True)
    except Exception:
        _record("cat", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  七、形状变换算子 (Shape Manipulation)
# ══════════════════════════════════════════════════════════════════

def _test_broadcast(dtype: str):
    """broadcast"""

    @triton.jit
    def k(in_ptr0, in_ptr1, out_ptr0, M: tl.constexpr, N: tl.constexpr):
        off1 = tl.arange(0, M)
        off2 = tl.arange(0, N)
        x = tl.load(in_ptr0 + N * off1[:, None] + off2[None, :])
        y = tl.load(in_ptr1 + off2)
        _, yb = tl.broadcast(x, y)
        tl.store(out_ptr0 + N * off1[:, None] + off2[None, :], yb)

    try:
        x = _make_input(dtype, (32, 64))
        y = _make_input(dtype, (64,))
        z = _make_output(dtype, (32, 64))
        k[(1, 1, 1)](x, y, z, M=32, N=64)
        _record("broadcast", dtype, True)
    except Exception:
        _record("broadcast", dtype, False)


def _test_broadcast_to(dtype: str):
    """broadcast_to"""

    @triton.jit
    def k(in_ptr0, out_ptr0, M: tl.constexpr, N: tl.constexpr):
        off2 = tl.arange(0, N)
        x = tl.load(in_ptr0 + off2)
        y = tl.broadcast_to(x, (M, N))
        tl.store(out_ptr0 + M * tl.arange(0, M)[:, None] + off2[None, :], y)

    try:
        x = _make_input(dtype, (64,))
        z = _make_output(dtype, (32, 64))
        k[(1, 1, 1)](x, z, M=32, N=64)
        _record("broadcast_to", dtype, True)
    except Exception:
        _record("broadcast_to", dtype, False)


def _test_reshape(dtype: str):
    """reshape"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        z = tl.reshape(x, (N // 4, 4))
        tl.store(out_ptr0 + idx, tl.reshape(z, (N,)), mask=idx < N)

    try:
        x = _make_input(dtype, (128,))
        z = _make_output(dtype, (128,))
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("reshape", dtype, True)
    except Exception:
        _record("reshape", dtype, False)


def _test_trans(dtype: str):
    """trans (2D 转置)"""

    @triton.jit
    def k(in_ptr0, out_ptr0, M: tl.constexpr, N: tl.constexpr):
        off1 = tl.arange(0, M)
        off2 = tl.arange(0, N)
        x = tl.load(in_ptr0 + N * off1[:, None] + off2[None, :])
        z = tl.trans(x)
        tl.store(out_ptr0 + M * off2[:, None] + off1[None, :], z)

    try:
        x = _make_input(dtype, (32, 64))
        z = _make_output(dtype, (64, 32))
        k[(1, 1, 1)](x, z, M=32, N=64)
        _record("trans", dtype, True)
    except Exception:
        _record("trans", dtype, False)


def _test_view(dtype: str):
    """view"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        z = tl.view(x, (N // 2, 2))
        tl.store(out_ptr0 + idx, tl.view(z, (N,)), mask=idx < N)

    try:
        x = _make_input(dtype, (128,))
        z = _make_output(dtype, (128,))
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("view", dtype, True)
    except Exception:
        _record("view", dtype, False)


def _test_permute(dtype: str):
    """permute"""

    @triton.jit
    def k(in_ptr0, out_ptr0, M: tl.constexpr, N: tl.constexpr):
        off1 = tl.arange(0, M)
        off2 = tl.arange(0, N)
        x = tl.load(in_ptr0 + N * off1[:, None] + off2[None, :])
        z = tl.permute(x, (1, 0))
        tl.store(out_ptr0 + M * off2[:, None] + off1[None, :], z)

    try:
        x = _make_input(dtype, (32, 64))
        z = _make_output(dtype, (64, 32))
        k[(1, 1, 1)](x, z, M=32, N=64)
        _record("permute", dtype, True)
    except Exception:
        _record("permute", dtype, False)


def _test_expand_dims(dtype: str):
    """expand_dims"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        z = tl.expand_dims(x, 0)
        tl.store(out_ptr0 + idx, tl.reshape(z, (N,)), mask=idx < N)

    try:
        x = _make_input(dtype, (128,))
        z = _make_output(dtype, (128,))
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("expand_dims", dtype, True)
    except Exception:
        _record("expand_dims", dtype, False)


def _test_ravel(dtype: str):
    """ravel"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        off1 = tl.arange(0, 32)
        off2 = tl.arange(0, 4)
        x = tl.load(in_ptr0 + 4 * off1[:, None] + off2[None, :])
        z = tl.ravel(x)
        tl.store(out_ptr0 + tl.arange(0, NUMEL), z, mask=tl.arange(0, NUMEL) < N)

    try:
        x = _make_input(dtype, (32, 4))
        z = _make_output(dtype, (128,))
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("ravel", dtype, True)
    except Exception:
        _record("ravel", dtype, False)


def _test_split(dtype: str):
    """split"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        a, b = tl.split(x)  # default: split evenly
        tl.store(out_ptr0 + idx, tl.cat(a, b, can_reorder=True), mask=idx < N)

    try:
        x = _make_input(dtype, (128,))
        z = _make_output(dtype, (128,))
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("split", dtype, True)
    except Exception:
        _record("split", dtype, False)


def _test_join(dtype: str):
    """join"""

    @triton.jit
    def k(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N // 2)
        y = tl.load(in_ptr1 + idx, mask=idx < N // 2)
        z = tl.join(x, y)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype, (64,))
        y = _make_input(dtype, (64,))
        z = _make_output(dtype, (128,))
        k[(1, 1, 1)](x, y, z, N=128, NUMEL=128)
        _record("join", dtype, True)
    except Exception:
        _record("join", dtype, False)


def _test_interleave(dtype: str):
    """interleave"""

    @triton.jit
    def k(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        y = tl.load(in_ptr1 + idx, mask=idx < N)
        z = tl.interleave(x, y)
        tl.store(out_ptr0 + idx, z, mask=idx < 2 * N)

    try:
        x = _make_input(dtype, (64,))
        y = _make_input(dtype, (64,))
        z = _make_output(dtype, (128,))
        k[(1, 1, 1)](x, y, z, N=64, NUMEL=128)
        _record("interleave", dtype, True)
    except Exception:
        _record("interleave", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  八、内存指针算子 (Memory Pointer Ops)
# ══════════════════════════════════════════════════════════════════

def _test_load(dtype: str):
    """tl.load"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        tl.store(out_ptr0 + idx, x, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("load", dtype, True)
    except Exception:
        _record("load", dtype, False)


def _test_store(dtype: str):
    """tl.store"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        tl.store(out_ptr0 + idx, x, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("store", dtype, True)
    except Exception:
        _record("store", dtype, False)


def _test_make_block_ptr(dtype: str):
    """tl.make_block_ptr"""

    @triton.jit
    def k(a_ptr, b_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, N), strides=(N, 1),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        a = tl.load(a_block_ptr)
        tl.store(b_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], a)

    try:
        a = _make_input(dtype, (64, 64))
        b = _make_output(dtype, (16, 16))
        k[(4, 4, 1)](a, b, M=64, N=64, BLOCK_M=16, BLOCK_N=16)
        _record("make_block_ptr", dtype, True)
    except Exception:
        _record("make_block_ptr", dtype, False)


def _test_advance(dtype: str):
    """tl.advance"""

    @triton.jit
    def k(a_ptr, b_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, N), strides=(N, 1),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        a_block_ptr = tl.advance(a_block_ptr, (0, 0))
        a = tl.load(a_block_ptr)
        tl.store(b_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], a)

    try:
        a = _make_input(dtype, (64, 64))
        b = _make_output(dtype, (16, 64))
        k[(4, 1, 1)](a, b, M=64, N=64, BLOCK_M=16, BLOCK_N=64)
        _record("advance", dtype, True)
    except Exception:
        _record("advance", dtype, False)


def _test_make_tensor_descriptor(dtype: str):
    """tl.make_tensor_descriptor + load/store"""

    @triton.jit
    def k(a_ptr, b_ptr, BLOCK_SIZE: tl.constexpr):
        tidx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        desc = tl.make_tensor_descriptor(a_ptr, shape=(128,), strides=(1,), block_shape=(BLOCK_SIZE,))
        val = tl.load_tensor_descriptor(desc, (tidx,), (BLOCK_SIZE,), dtype=tl.float32)
        tl.store_tensor_descriptor(b_ptr, val, (tidx,), (128,))

    try:
        a = _make_input(dtype)
        b = _make_output(dtype)
        k[(1, 1, 1)](a, b, BLOCK_SIZE=128)
        _record("make_tensor_descriptor", dtype, True)
    except Exception:
        _record("make_tensor_descriptor", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  九、索引算子 (Indexing Ops)
# ══════════════════════════════════════════════════════════════════

def _test_where(dtype: str):
    """where"""

    @triton.jit
    def k(cond_ptr, x_ptr, y_ptr, out_ptr, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        cond = tl.load(cond_ptr + idx, mask=idx < N)
        x = tl.load(x_ptr + idx, mask=idx < N)
        y = tl.load(y_ptr + idx, mask=idx < N)
        z = tl.where(cond, x, y)
        tl.store(out_ptr + idx, z, mask=idx < N)

    try:
        cond = _make_input("bool")
        x = _make_input(dtype)
        y = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](cond, x, y, z, N=128, NUMEL=128)
        _record("where", dtype, True)
    except Exception:
        _record("where", dtype, False)


def _test_flip(dtype: str):
    """flip"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        z = tl.flip(x)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("flip", dtype, True)
    except Exception:
        _record("flip", dtype, False)


def _test_swizzle2d(dtype: str):
    """swizzle2d"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        z = tl.swizzle2d(x, 0, 1, 4)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("swizzle2d", dtype, True)
    except Exception:
        _record("swizzle2d", dtype, False)


def _test_gather(dtype: str):
    """gather"""

    @triton.jit
    def k(in_ptr0, idx_ptr, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        gi = tl.load(idx_ptr + idx, mask=idx < N)
        z = tl.gather(x, gi)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        src = _make_input(dtype)
        idx = torch.randint(0, 128, (128,), dtype=torch.int32, device='npu')
        z = _make_output(dtype)
        k[(1, 1, 1)](src, idx, z, N=128, NUMEL=128)
        _record("gather", dtype, True)
    except Exception:
        _record("gather", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  十、归约算子 (Reduction Ops)
# ══════════════════════════════════════════════════════════════════

def _test_sum(dtype: str):
    """sum reduction"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.sum(x)
        tl.store(out_ptr0 + tl.program_id(0), z)

    try:
        x = _make_input(dtype, (256,))
        z = _make_output(dtype, (2,))
        k[(2, 1, 1)](x, z, N=256, BLOCK=128)
        _record("sum", dtype, True)
    except Exception:
        _record("sum", dtype, False)


def _test_max_reduce(dtype: str):
    """max reduction"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.max(x)
        tl.store(out_ptr0 + tl.program_id(0), z)

    try:
        x = _make_input(dtype, (256,))
        z = _make_output(dtype, (2,))
        k[(2, 1, 1)](x, z, N=256, BLOCK=128)
        _record("max", dtype, True)
    except Exception:
        _record("max", dtype, False)


def _test_min_reduce(dtype: str):
    """min reduction"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.min(x)
        tl.store(out_ptr0 + tl.program_id(0), z)

    try:
        x = _make_input(dtype, (256,))
        z = _make_output(dtype, (2,))
        k[(2, 1, 1)](x, z, N=256, BLOCK=128)
        _record("min", dtype, True)
    except Exception:
        _record("min", dtype, False)


def _test_argmax(dtype: str):
    """argmax reduction (returns index → always int32/64 output, test input dtype)"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.argmax(x, 0)
        tl.store(out_ptr0 + tl.program_id(0), z)

    try:
        x = _make_input(dtype, (256,))
        z = torch.empty((2,), dtype=torch.int32, device='npu')
        k[(2, 1, 1)](x, z, N=256, BLOCK=128)
        _record("argmax", dtype, True)
    except Exception:
        _record("argmax", dtype, False)


def _test_argmin(dtype: str):
    """argmin reduction"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.argmin(x, 0)
        tl.store(out_ptr0 + tl.program_id(0), z)

    try:
        x = _make_input(dtype, (256,))
        z = torch.empty((2,), dtype=torch.int32, device='npu')
        k[(2, 1, 1)](x, z, N=256, BLOCK=128)
        _record("argmin", dtype, True)
    except Exception:
        _record("argmin", dtype, False)


def _test_xor_sum(dtype: str):
    """xor_sum"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.xor_sum(x)
        tl.store(out_ptr0 + tl.program_id(0), z)

    try:
        x = _make_input(dtype, (256,))
        z = _make_output(dtype, (2,))
        k[(2, 1, 1)](x, z, N=256, BLOCK=128)
        _record("xor_sum", dtype, True)
    except Exception:
        _record("xor_sum", dtype, False)


def _test_reduce(dtype: str):
    """tl.reduce (generic)"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.reduce(x, 0, tl.reduce_kind.SUM)
        tl.store(out_ptr0 + tl.program_id(0), z)

    try:
        x = _make_input(dtype, (256,))
        z = _make_output(dtype, (2,))
        k[(2, 1, 1)](x, z, N=256, BLOCK=128)
        _record("reduce", dtype, True)
    except Exception:
        _record("reduce", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  十一、原子操作 (Atomic Ops)
# ══════════════════════════════════════════════════════════════════

ATOMIC_OPS = {
    "atomic_add": "tl.atomic_add",
    "atomic_max": "tl.atomic_max",
    "atomic_min": "tl.atomic_min",
    "atomic_and": "tl.atomic_and",
    "atomic_or":  "tl.atomic_or",
    "atomic_xor": "tl.atomic_xor",
    "atomic_xchg": "tl.atomic_xchg",
    "atomic_cas": "tl.atomic_cas",
}


def _test_atomic(op: str, func: str, dtype: str):
    """原子操作 → 只需要 int32/float32 等，但我们也尝试所有类型"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        xoffset = tl.program_id(0) * BLOCK_SIZE
        xindex = xoffset + tl.arange(0, BLOCK_SIZE)
        xmask = xindex < N
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + x0, xmask)
        tmp1 = FUNC(out_ptr0 + x0, tmp0, xmask)
        tl.store(out_ptr0 + x0, tmp1, xmask)

    # atomic_cas needs 3 args: ptr, cmp, val
    if op == "atomic_cas":
        @triton.jit
        def k_cas(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            xoffset = tl.program_id(0) * BLOCK_SIZE
            xindex = xoffset + tl.arange(0, BLOCK_SIZE)
            xmask = xindex < N
            x0 = xindex
            tmp0 = tl.load(in_ptr0 + x0, xmask)
            tmp_cmp = tl.load(in_ptr1 + x0, xmask)
            tmp1 = tl.atomic_cas(out_ptr0 + x0, tmp_cmp, tmp0, xmask)
            tl.store(out_ptr0 + x0, tmp1, xmask)
        k_src = k_cas
    else:
        k_src = k

    kj = triton.JITFunction(k_src.fn)
    kj._unsafe_update_src(kj.src.replace("FUNC", func))
    try:
        x = _make_input(dtype, (128,))
        z = _make_output(dtype, (128,))
        if op == "atomic_cas":
            cmp_v = _make_input(dtype, (128,))
            kj[(2, 1, 1)](x, cmp_v, z, N=128, BLOCK_SIZE=64)
        else:
            kj[(2, 1, 1)](x, z, N=128, BLOCK_SIZE=64)
        _record(op, dtype, True)
    except Exception:
        _record(op, dtype, False)


# ══════════════════════════════════════════════════════════════════
#  十二、扫描/排序 (Scan/Sort Ops)
# ══════════════════════════════════════════════════════════════════

def _test_cumsum(dtype: str):
    """cumsum"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.cumsum(x)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, BLOCK=128)
        _record("cumsum", dtype, True)
    except Exception:
        _record("cumsum", dtype, False)


def _test_sort(dtype: str):
    """sort"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.sort(x)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, BLOCK=128)
        _record("sort", dtype, True)
    except Exception:
        _record("sort", dtype, False)


def _test_histogram(dtype: str):
    """histogram"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUM_BINS: tl.constexpr):
        idx = tl.arange(0, N)
        x = tl.load(in_ptr0 + idx, mask=idx < 128, other=0)
        z = tl.histogram(x, NUM_BINS)
        bin_idx = tl.arange(0, NUM_BINS)
        tl.store(out_ptr0 + bin_idx, z)

    try:
        x = _make_input(dtype)
        z = torch.empty((10,), dtype=torch.int32, device='npu')
        k[(1, 1, 1)](x, z, N=128, NUM_BINS=10)
        _record("histogram", dtype, True)
    except Exception:
        _record("histogram", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  十三、随机数生成 (Random) → 只测试 seed 的 dtype 支持
# ══════════════════════════════════════════════════════════════════

def _test_rand(dtype: str):
    """rand → seed dtype"""

    @triton.jit
    def k(out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr):
        bo = tl.program_id(0) * XBLOCK
        bs = XBLOCK if bo + XBLOCK <= N else N - bo
        for i in range(bs):
            go = bo + i
            rv = tl.rand(5, go)
            tl.store(out_ptr0 + go, rv)

    try:
        z = _make_output(dtype)
        k[(1, 1, 1)](z, N=128, XBLOCK=128)
        _record("rand", dtype, True)
    except Exception:
        _record("rand", dtype, False)


def _test_randint(dtype: str):
    """randint → seed dtype"""

    @triton.jit
    def k(out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr):
        bo = tl.program_id(0) * XBLOCK
        bs = XBLOCK if bo + XBLOCK <= N else N - bo
        for i in range(bs):
            go = bo + i
            rv = tl.randint(5, go + 10)
            tl.store(out_ptr0 + go, rv)

    try:
        z = _make_output(dtype)
        k[(1, 1, 1)](z, N=128, XBLOCK=128)
        _record("randint", dtype, True)
    except Exception:
        _record("randint", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  十四、其他算子
# ══════════════════════════════════════════════════════════════════

def _test_cast(dtype: str):
    """cast to float32"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        z = x.to(tl.float32)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = torch.empty((128,), dtype=torch.float32, device='npu')
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("cast", dtype, True)
    except Exception:
        _record("cast", dtype, False)


def _test_range(dtype: str):
    """tl.range → iterator, 只对 int 有意义"""

    if dtype not in INT_SET:
        return _record("range", dtype, False)

    @triton.jit
    def k(out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        s = tl.zeros((NUMEL,), tl.float32)
        for i in tl.range(0, 4):
            s = s + 1.0
        tl.store(out_ptr0 + idx, s, mask=idx < N)

    try:
        z = _make_output("fp32")
        k[(1, 1, 1)](z, N=128, NUMEL=128)
        _record("range", dtype, True)
    except Exception:
        _record("range", dtype, False)


def _test_static_range(dtype: str):
    """tl.static_range → 编译时常量，不依赖数据类型"""

    @triton.jit
    def k(out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        s = tl.zeros((NUMEL,), tl.float32)
        for i in tl.static_range(0, 4):
            s = s + 1.0
        tl.store(out_ptr0 + idx, s, mask=idx < N)

    try:
        z = _make_output("fp32")
        k[(1, 1, 1)](z, N=128, NUMEL=128)
        _record("static_range", dtype, True)
    except Exception:
        _record("static_range", dtype, False)


def _test_assume(dtype: str):
    """tl.assume"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        tl.assume(x > 0)
        tl.store(out_ptr0 + idx, x, mask=idx < N)

    try:
        x = _make_input(dtype, positive=True)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("assume", dtype, True)
    except Exception:
        _record("assume", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  十五、点积 (Dot Ops)
# ══════════════════════════════════════════════════════════════════

def _test_dot(dtype: str):
    """tl.dot → 输入 dtype"""

    @triton.jit
    def k(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        c = tl.dot(a, b)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)

    try:
        a = _make_input(dtype, (16, 16))
        b = _make_input(dtype, (16, 16))
        c = _make_output(dtype, (16, 16))
        k[(1, 1, 1)](a, b, c, M=16, N=16, K=16, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
        _record("dot", dtype, True)
    except Exception:
        _record("dot", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  十六、补充算子 (dot_scaled, cumprod, randn, randint4x, etc.)
# ══════════════════════════════════════════════════════════════════

def _test_dot_scaled(dtype: str):
    """tl.dot_scaled → 需要 fp8/bf16/fp16 输入"""

    @triton.jit
    def k(a_ptr, b_ptr, scale_a_ptr, scale_b_ptr, c_ptr,
          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        scale_a = tl.load(scale_a_ptr + rk)
        scale_b = tl.load(scale_b_ptr + rk)
        c = tl.dot_scaled(a, b, scale_a, scale_b, tl.float32)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)

    try:
        a = _make_input(dtype, (16, 16))
        b = _make_input(dtype, (16, 16))
        scale_a = _make_input("fp32", (16,))
        scale_b = _make_input("fp32", (16,))
        c = _make_output("fp32", (16, 16))
        k[(1, 1, 1)](a, b, scale_a, scale_b, c, M=16, N=16, K=16, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
        _record("dot_scaled", dtype, True)
    except Exception:
        _record("dot_scaled", dtype, False)


def _test_cumprod(dtype: str):
    """cumprod"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=1)
        z = tl.cumprod(x)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, BLOCK=128)
        _record("cumprod", dtype, True)
    except Exception:
        _record("cumprod", dtype, False)


def _test_associative_scan(dtype: str):
    """associative_scan"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_ptr0 + idx, mask=idx < N, other=0)
        z = tl.associative_scan(x, 0, lambda a, b: a + b)
        tl.store(out_ptr0 + idx, z, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, BLOCK=128)
        _record("associative_scan", dtype, True)
    except Exception:
        _record("associative_scan", dtype, False)


def _test_randn(dtype: str):
    """randn → seed dtype 支持"""

    @triton.jit
    def k(out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr):
        bo = tl.program_id(0) * XBLOCK
        bs = XBLOCK if bo + XBLOCK <= N else N - bo
        for i in range(bs):
            go = bo + i
            rv = tl.randn(5, go)
            tl.store(out_ptr0 + go, rv)

    try:
        z = _make_output("fp32")
        k[(1, 1, 1)](z, N=128, XBLOCK=128)
        _record("randn", dtype, True)
    except Exception:
        _record("randn", dtype, False)


def _test_randint4x(dtype: str):
    """randint4x → seed dtype 支持"""

    @triton.jit
    def k(out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr):
        bo = tl.program_id(0) * XBLOCK
        bs = XBLOCK if bo + XBLOCK <= N else N - bo
        for i in range(0, bs, step=4):
            go = bo + i
            rv, _, _, _ = tl.randint4x(5, go + 10)
            mask = (go + tl.arange(0, 4)) < N
            tl.store(out_ptr0 + go + tl.arange(0, 4), rv, mask)

    try:
        z = _make_output("fp32")
        k[(1, 1, 1)](z, N=128, XBLOCK=128)
        _record("randint4x", dtype, True)
    except Exception:
        _record("randint4x", dtype, False)


def _test_load_tensor_descriptor(dtype: str):
    """tl.load_tensor_descriptor → 读 TMA 描述符"""

    @triton.jit
    def k(a_ptr, b_ptr, BLOCK_SIZE: tl.constexpr):
        tidx = tl.arange(0, BLOCK_SIZE)
        desc = tl.make_tensor_descriptor(a_ptr, shape=(128,), strides=(1,), block_shape=(BLOCK_SIZE,))
        val = tl.load_tensor_descriptor(desc, (tidx,), (BLOCK_SIZE,), dtype=tl.float32)
        tl.store(b_ptr + tidx, val)

    try:
        a = _make_input(dtype)
        b = _make_output("fp32")
        k[(1, 1, 1)](a, b, BLOCK_SIZE=128)
        _record("load_tensor_descriptor", dtype, True)
    except Exception:
        _record("load_tensor_descriptor", dtype, False)


def _test_store_tensor_descriptor(dtype: str):
    """tl.store_tensor_descriptor → 写 TMA 描述符"""

    @triton.jit
    def k(a_ptr, b_ptr, BLOCK_SIZE: tl.constexpr):
        tidx = tl.arange(0, BLOCK_SIZE)
        val = tl.load(a_ptr + tidx)
        tl.store_tensor_descriptor(b_ptr, val, (tidx,), (128,))

    try:
        a = _make_input(dtype)
        b = _make_output(dtype)
        k[(1, 1, 1)](a, b, BLOCK_SIZE=128)
        _record("store_tensor_descriptor", dtype, True)
    except Exception:
        _record("store_tensor_descriptor", dtype, False)


def _test_debug_barrier(dtype: str):
    """tl.debug_barrier"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        tl.debug_barrier()
        tl.store(out_ptr0 + idx, x, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("debug_barrier", dtype, True)
    except Exception:
        _record("debug_barrier", dtype, False)


def _test_max_constancy(dtype: str):
    """tl.max_constancy"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        tl.max_constancy(x, 1)
        tl.store(out_ptr0 + idx, x, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("max_constancy", dtype, True)
    except Exception:
        _record("max_constancy", dtype, False)


def _test_max_contiguous(dtype: str):
    """tl.max_contiguous"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        tl.max_contiguous(x, 1)
        tl.store(out_ptr0 + idx, x, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("max_contiguous", dtype, True)
    except Exception:
        _record("max_contiguous", dtype, False)


def _test_multiple_of(dtype: str):
    """tl.multiple_of"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        tl.multiple_of(x, 16)
        tl.store(out_ptr0 + idx, x, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("multiple_of", dtype, True)
    except Exception:
        _record("multiple_of", dtype, False)


def _test_static_assert(dtype: str):
    """tl.static_assert → 编译期检查，不依赖 dtype，但检查是否能编译通过"""

    @triton.jit
    def k(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
        idx = tl.arange(0, NUMEL)
        x = tl.load(in_ptr0 + idx, mask=idx < N)
        tl.static_assert(NUMEL == 128)
        tl.store(out_ptr0 + idx, x, mask=idx < N)

    try:
        x = _make_input(dtype)
        z = _make_output(dtype)
        k[(1, 1, 1)](x, z, N=128, NUMEL=128)
        _record("static_assert", dtype, True)
    except Exception:
        _record("static_assert", dtype, False)


# ══════════════════════════════════════════════════════════════════
#  ── 测试执行与汇总 ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

# 将所有测试函数收集到一起，方便 pytest 参数化
ALL_TEST_SPECS = []

# 一元算子
for op, expr in UNARY_OPS.items():
    ALL_TEST_SPECS.append(("unary", op, expr, False))
# 二元算子
for op, expr in BINARY_OPS.items():
    ALL_TEST_SPECS.append(("binary", op, expr, False))
# 比较
for op, expr in COMPARE_OPS.items():
    ALL_TEST_SPECS.append(("compare", op, expr, False))
# 逻辑
for op, expr in LOGICAL_OPS.items():
    ALL_TEST_SPECS.append(("logical", op, expr, False))
# 三元
for op, expr in TERNARY_OPS.items():
    ALL_TEST_SPECS.append(("ternary", op, expr, False))
# 原子
for op, func in ATOMIC_OPS.items():
    ALL_TEST_SPECS.append(("atomic", op, func, False))


class TestAllDtypeSupport:

    @pytest.mark.parametrize("kind,op,expr,_", ALL_TEST_SPECS)
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_ops(self, kind, op, expr, _, dtype):
        if kind == "unary":
            positive = op in ("log", "log2", "sqrt", "rsqrt", "exp", "exp2")
            _test_unary(op, expr, dtype, positive)
        elif kind == "binary":
            _test_binary(op, expr, dtype)
        elif kind in ("compare",):
            _test_compare(op, expr, dtype)
        elif kind in ("logical",):
            _test_logical(op, expr, dtype)
        elif kind == "ternary":
            _test_ternary(op, expr, dtype)
        elif kind == "atomic":
            _test_atomic(op, expr, dtype)

    # ── 创建类 ──
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_arange(self, dtype): _test_arange(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_full(self, dtype): _test_full(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_zeros(self, dtype): _test_zeros(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_zeros_like(self, dtype): _test_zeros_like(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_cat(self, dtype): _test_cat(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_cast(self, dtype): _test_cast(dtype)

    # ── 形状变换 ──
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_broadcast(self, dtype): _test_broadcast(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_broadcast_to(self, dtype): _test_broadcast_to(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_reshape(self, dtype): _test_reshape(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_trans(self, dtype): _test_trans(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_view(self, dtype): _test_view(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_permute(self, dtype): _test_permute(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_expand_dims(self, dtype): _test_expand_dims(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_ravel(self, dtype): _test_ravel(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_split(self, dtype): _test_split(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_join(self, dtype): _test_join(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_interleave(self, dtype): _test_interleave(dtype)

    # ── 内存指针 ──
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_load(self, dtype): _test_load(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_store(self, dtype): _test_store(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_make_block_ptr(self, dtype): _test_make_block_ptr(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_advance(self, dtype): _test_advance(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_make_tensor_descriptor(self, dtype): _test_make_tensor_descriptor(dtype)

    # ── 索引 ──
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_where(self, dtype): _test_where(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_flip(self, dtype): _test_flip(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_swizzle2d(self, dtype): _test_swizzle2d(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_gather(self, dtype): _test_gather(dtype)

    # ── 归约 ──
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_sum(self, dtype): _test_sum(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_max_reduce(self, dtype): _test_max_reduce(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_min_reduce(self, dtype): _test_min_reduce(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_argmax(self, dtype): _test_argmax(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_argmin(self, dtype): _test_argmin(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_xor_sum(self, dtype): _test_xor_sum(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_reduce(self, dtype): _test_reduce(dtype)

    # ── 扫描/排序 ──
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_cumsum(self, dtype): _test_cumsum(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_sort(self, dtype): _test_sort(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_histogram(self, dtype): _test_histogram(dtype)

    # ── 随机数 ──
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_rand(self, dtype): _test_rand(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_randint(self, dtype): _test_randint(dtype)

    # ── 其他 ──
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_range(self, dtype): _test_range(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_static_range(self, dtype): _test_static_range(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_assume(self, dtype): _test_assume(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_dot(self, dtype): _test_dot(dtype)

    # ── 补充算子 ──
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_dot_scaled(self, dtype): _test_dot_scaled(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_cumprod(self, dtype): _test_cumprod(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_associative_scan(self, dtype): _test_associative_scan(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_randn(self, dtype): _test_randn(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_randint4x(self, dtype): _test_randint4x(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_load_tensor_descriptor(self, dtype): _test_load_tensor_descriptor(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_store_tensor_descriptor(self, dtype): _test_store_tensor_descriptor(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_debug_barrier(self, dtype): _test_debug_barrier(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_max_constancy(self, dtype): _test_max_constancy(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_max_contiguous(self, dtype): _test_max_contiguous(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_multiple_of(self, dtype): _test_multiple_of(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_static_assert(self, dtype): _test_static_assert(dtype)


# ── 会话结束输出表格 ─────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _print_table(request):
    yield
    if not _results:
        return

    header = ["Operator"] + [DTYPE_DISPLAY[d] for d in ALL_DTYPES]
    sep = [":---:"] + [":---:"] * len(ALL_DTYPES)
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for op in sorted(_results.keys()):
        row = [op]
        for d in ALL_DTYPES:
            row.append(_results[op].get(d, "×"))
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    print("\n")
    print("=" * 100)
    print("  DataType 支持情况汇总")
    print("=" * 100)
    print()
    print(table)
    print()

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dtype_support_result.md")
    with open(p, "w") as f:
        f.write("# DataType 支持测试结果\n\n")
        f.write("> √ 支持, × 不支持\n\n")
        f.write(table + "\n")
    print(f"[结果已写入 {p}]")
