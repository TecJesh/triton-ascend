"""
DataType 支持情况测试脚本

测试 ge, lt, le, eq, ne, logical_and, logical_or, abs, cdiv, ceil, clamp, cos,
div_rn, erf, exp, exp2, fdiv, floor, fma, log, log2 对所有数据类型的支持情况。

用法:
    python test_dtype_support.py [device]
    device: 默认 'npu'（Ascend），也可指定 'cuda'。

输出: DataType 支持情况汇总表格（√ 支持, × 不支持），同时写入 dtype_support_result.md。
"""

import os
import sys

import torch
import triton
import triton.language as tl

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "npu"
SIZE = 64

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
BOOL_SET = {"bool"}

print(f"[Device: {DEVICE}]")


# ── 张量创建工具 ──────────────────────────────────────────────────

def _dt(name: str):
    """dtype 名 → torch dtype"""
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


def _make_x(name: str, positive: bool = False):
    """创建 SIZE 个随机输入值"""
    if name == "bool":
        return torch.randint(0, 2, (SIZE,), device=DEVICE, dtype=torch.bool)
    if name in ("fp8e4m3", "fp8e5m2"):
        t = torch.randint(0, 255, (SIZE,), dtype=torch.int8, device=DEVICE)
        if name == "fp8e4m3":
            t[(t & 0b01111100) == 0b01111100] = 0
        return t.view(_dt(name))
    if name in ("uint16", "uint32", "uint64"):
        bw = int(name.replace("uint", ""))
        t = torch.randint(0, min(2 ** (bw - 1), 10000), (SIZE,),
                          dtype=getattr(torch, f"int{bw}"), device=DEVICE)
        return t.view(_dt(name))
    if name.startswith(("uint", "int")):
        lo = -64 if name.startswith("int") else 0
        hi = 64 if name.startswith("int") else 128
        return torch.randint(lo, hi, (SIZE,), dtype=_dt(name), device=DEVICE)
    # float
    t = torch.randn((SIZE,), dtype=_dt(name), device=DEVICE)
    return t.abs() + 0.01 if positive else t


def _make_z(name: str):
    """创建输出张量"""
    if name in ("fp8e4m3", "fp8e5m2"):
        return torch.empty((SIZE,), dtype=torch.int8, device=DEVICE)
    return torch.empty((SIZE,), dtype=_dt("bool" if name == "cmp" else name),
                       device=DEVICE)


# ── 测试执行（参照 test_core.py 的 _test_unary/_test_binary）──────

def _try_unary(op: str, expr: str, dtype: str, positive_input: bool = False):
    """测试一元算子对某个 dtype 的支持。True=√, False=×"""
    if op in ("cdiv", "logical_and", "logical_or"):
        if dtype in FLOAT_SET or dtype in BOOL_SET:
            return False

    # 模板定义在函数内部（参考 test_core.py 的 _test_unary）
    @triton.jit
    def _kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    # 创建一个新的 JITFunction 副本并替换源码中的占位符
    k = triton.JITFunction(_kernel.fn)
    src = k.src
    src = src.replace("GENERATE_TEST_HERE", expr)
    k._unsafe_update_src(src)

    try:
        x = _make_x(dtype, positive=positive_input)
        z = _make_z(dtype)
        k[(1,)](z, x, SIZE=SIZE, num_warps=4)
        return True
    except Exception:
        return False


def _try_binary(op: str, expr: str, dtype: str):
    """测试二元算子对某个 dtype 的支持"""
    if op in ("cdiv",):
        if dtype in FLOAT_SET or dtype in BOOL_SET:
            return False
    if op in ("logical_and", "logical_or"):
        if dtype in FLOAT_SET:
            return False

    @triton.jit
    def _kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    k = triton.JITFunction(_kernel.fn)
    src = k.src
    src = src.replace("GENERATE_TEST_HERE", expr)
    k._unsafe_update_src(src)

    try:
        x = _make_x(dtype)
        y = _make_x(dtype)
        is_cmp = op in ("eq", "ne", "lt", "le", "ge")
        z = _make_z("cmp" if is_cmp else dtype)
        k[(1,)](z, x, y, SIZE=SIZE, num_warps=4)
        return True
    except Exception:
        return False


def _try_ternary(op: str, expr: str, dtype: str):
    """测试三元算子对某个 dtype 的支持"""

    @triton.jit
    def _kernel(Z, X, Y, W, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        w = tl.load(W + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    k = triton.JITFunction(_kernel.fn)
    src = k.src
    src = src.replace("GENERATE_TEST_HERE", expr)
    k._unsafe_update_src(src)

    try:
        x = _make_x(dtype)
        y = _make_x(dtype, positive=True)
        w = _make_x(dtype, positive=True)
        z = _make_z(dtype)
        k[(1,)](z, x, y, w, SIZE=SIZE, num_warps=4)
        return True
    except Exception:
        return False


# ── 算子清单 ──────────────────────────────────────────────────────
# (op_name, kind, expr, positive_input)
OPS = [
    # ── 比较运算 ──
    ("eq",          "binary",  "x == y",               False),
    ("ne",          "binary",  "x != y",               False),
    ("lt",          "binary",  "x < y",                False),
    ("le",          "binary",  "x <= y",               False),
    ("ge",          "binary",  "x >= y",               False),
    # ── 逻辑运算 ──
    ("logical_and", "binary",  "x & y",                False),
    ("logical_or",  "binary",  "x | y",                False),
    # ── 一元数学 ──
    ("abs",         "unary",   "tl.abs(x)",            False),
    ("ceil",        "unary",   "tl.math.ceil(x)",      False),
    ("cos",         "unary",   "tl.cos(x)",            False),
    ("erf",         "unary",   "tl.math.erf(x)",       False),
    ("exp",         "unary",   "tl.exp(x)",            True),
    ("exp2",        "unary",   "tl.math.exp2(x)",      True),
    ("floor",       "unary",   "tl.math.floor(x)",     False),
    ("log",         "unary",   "tl.log(x)",            True),
    ("log2",        "unary",   "tl.math.log2(x)",      True),
    # ── 二元数学 ──
    ("cdiv",        "binary",  "tl.math.cdiv(x, y)",   False),
    ("div_rn",      "binary",  "tl.math.div_rn(x, y)", False),
    ("fdiv",        "binary",  "tl.math.fdiv(x, y)",   False),
    # ── 三元 ──
    ("clamp",       "ternary", "tl.clamp(x, y, w)",    False),
    ("fma",         "ternary", "tl.math.fma(x, y, w)", False),
]


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    results = {}
    total = len(OPS) * len(ALL_DTYPES)
    count = 0

    for op_name, kind, expr, positive in OPS:
        results[op_name] = {}
        for dtype in ALL_DTYPES:
            count += 1
            if kind == "unary":
                ok = _try_unary(op_name, expr, dtype, positive_input=positive)
            elif kind == "binary":
                ok = _try_binary(op_name, expr, dtype)
            else:
                ok = _try_ternary(op_name, expr, dtype)

            results[op_name][dtype] = "√" if ok else "×"
            print(f"\r[{count}/{total}] {op_name}({dtype}): {'√' if ok else '×'}",
                  end="", flush=True)

    print("\n")

    # ── 生成 Markdown 表格 ────────────────────────────────────────
    header = ["Operator"] + [DTYPE_DISPLAY[d] for d in ALL_DTYPES]
    sep = [":---:"] + [":---:"] * len(ALL_DTYPES)

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for op_name, _, _, _ in OPS:
        row = [op_name]
        for d in ALL_DTYPES:
            row.append(results[op_name].get(d, "×"))
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)

    print()
    print("=" * 100)
    print(f"  DataType 支持情况汇总  (Device: {DEVICE})")
    print("=" * 100)
    print()
    print(table)
    print()

    # ── 写文件 ────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "dtype_support_result.md")
    with open(out_path, "w") as f:
        f.write("# DataType 支持测试结果\n\n")
        f.write(f"> Device: {DEVICE}\n")
        f.write("> 以下表格由 `test_dtype_support.py` 自动生成。\n")
        f.write("> √ 表示支持（内核编译并执行成功），× 表示不支持（编译或执行失败）。\n\n")
        f.write(table)
        f.write("\n")
    print(f"[结果已写入 {out_path}]")


if __name__ == "__main__":
    main()
