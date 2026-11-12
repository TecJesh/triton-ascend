# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import glob
import os
import pytest
import re
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np

import triton

kernel_utils_src = """
import triton

@triton.jit
def mul(x, y):
    return x * y
"""

kernel_src = """
import triton
import triton.language as tl
import kernel_utils

@triton.jit
def kernel(C, A, B, M, N, K,
          stride_cm, stride_cn,
          stride_am, stride_ak,
          stride_bk, stride_bn,
          BLOCK_M: tl.constexpr,
          BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
  offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
  offs_k = tl.arange(0, BLOCK_K)
  a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

  accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_K)):
      # Load the next block of A and B, generate a mask by checking the K dimension.
      # If it is out of bounds, set it to 0.
      a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
      b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
      # We accumulate along the K dimension.
      accumulator += tl.dot(a, b)
      # Advance the ptrs to the next K block.
      a_ptrs += BLOCK_K * stride_ak
      b_ptrs += BLOCK_K * stride_bk

  c = kernel_utils.mul(accumulator, accumulator)
  # Write back the block of the output matrix C with masks.
  offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  tl.store(c_ptrs, c)
"""

vec_kernel_src = """
import triton
import triton.language as tl

@triton.jit
def vec_kernel(Y, X, n_elements, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(Y + offs, x * SCALE, mask=mask)
"""

multi_kernel_src = """
import triton
import triton.language as tl
import kernel_utils

@triton.jit
def caller_kernel(Y, X, n_elements, FACTOR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(Y + offs, kernel_utils.mul(x, FACTOR), mask=mask)

@triton.jit
def plain_kernel(Y, X, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(Y + offs, x + 1.0, mask=mask)
"""

multi_kernel_same_file_src = """
import triton
import triton.language as tl

@triton.jit
def scale(x, factor):
    return x * factor

@triton.jit
def caller_kernel(Y, X, n_elements, FACTOR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(Y + offs, scale(x, FACTOR), mask=mask)

@triton.jit
def plain_kernel(Y, X, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(Y + offs, x + 1.0, mask=mask)
"""

test_utils_src = """
#include <acl/acl.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "kernel.h"

static void write_buffer_to_csv(char *filename, int32_t *buffer, int size) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d", buffer[i]);
        if (i < size - 1) {
            fprintf(file, ",");
        }
    }
    fclose(file);
}

static void read_csv_to_buffer(char *filename, int16_t *buffer, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    int index = 0;
    while (fscanf(file, "%hd,", &buffer[index]) != EOF && index < size) {
        index++;
    }
    fclose(file);
}

static void read_csv_to_buffer32(char *filename, int32_t *buffer, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    int index = 0;
    while (fscanf(file, "%d,", &buffer[index]) != EOF && index < size) {
        index++;
    }
    fclose(file);
}"""


def is_npu():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "npu"


def _torch_npu_library_dir():
    import torch_npu

    return os.path.join(os.path.dirname(torch_npu.__file__), "lib")


def _npu_toolchain():
    """Reuse the ascend backend's own NPU extension compile/link flags.

    These are the same flags used to build npu_utils and the launcher
    runtime, so the AOT C harness always matches the installed toolchain
    (CANN include/lib layout, torch_npu paths and ABI defines) without
    hard-coding any path.
    """
    from triton.backends.ascend import utils as ascend_utils

    asc_path = str(ascend_utils._get_ascend_path())
    has_runtime_header = os.path.exists(os.path.join(asc_path, "include/experiment/runtime/runtime/rt.h"))
    flags = list(
        ascend_utils._npu_ext_build_options("npu_utils", "torch", asc_path, has_runtime_header,
                                            ascend_utils.backend_policy))
    include_flags = [flag for flag in flags if flag.startswith(("-I", "-D", "-w"))]
    link_flags = [flag for flag in flags if flag.startswith(("-L", "-l"))]
    lib_dirs = [flag[2:] for flag in link_flags if flag.startswith("-L")]
    # The launcher chain (npu_utils/launcher runtime .so) links torch
    # extension libraries (libtorch_python.so etc.); the plain C test
    # process must be able to load them, so extend the search path with
    # torch's library dirs.
    from torch.utils.cpp_extension import library_paths

    lib_dirs += [lib_dir for lib_dir in [*library_paths(), _torch_npu_library_dir()] if lib_dir not in lib_dirs]
    python_libdir = sysconfig.get_config_var("LIBDIR")
    python_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    ld_library_path = [*lib_dirs] + ([python_libdir] if python_libdir else [])
    return include_flags, link_flags, lib_dirs, python_libdir, python_lib, ld_library_path


def gen_kernel_library_npu(dir, libname):
    include_flags, link_flags, lib_dirs, _, _, _ = _npu_toolchain()
    c_files = glob.glob(os.path.join(dir, "*.c"))
    # Compile each generated launcher source, then link the shared library
    # against the CANN/torch_npu runtime libs its rt*/aclrt* calls need
    # (they are referenced, so --as-needed keeps them).
    subprocess.run(
        ["gcc"] + c_files + include_flags + ["-c", "-fPIC"],
        check=True,
        cwd=dir,
    )
    o_files = glob.glob(os.path.join(dir, "*.o"))
    command = ["gcc", *o_files, "-shared", "-o", libname]
    for lib_dir in lib_dirs:
        command += [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"]
    command += link_flags + ["-ldl"]
    subprocess.run(command, check=True, cwd=dir)


def gen_test_bin_npu(dir, M, N, K, exe="test", algo_id=0, dtype="fp16", kernel_name="matmul_fp16"):
    elem_size = 4 if dtype == "fp32" else 2
    elem_type = "int32_t" if dtype == "fp32" else "int16_t"
    read_fn = "read_csv_to_buffer32" if dtype == "fp32" else "read_csv_to_buffer"
    test_src = f"""
int main(int argc, char **argv) {{
  int M = {M}, N = {N}, K = {K};

  // initialize ACL handles
  aclrtStream stream = NULL;
  void *A = NULL, *B = NULL, *C = NULL;
  aclError err = 0;
  aclInit(NULL);
  int32_t device_id = 0;
  aclrtSetDevice(device_id);
  aclrtMalloc(&A, M * K * {elem_size}, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&B, K * N * {elem_size}, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&C, M * N * 4, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtCreateStream(&stream);
  load_{kernel_name}();

  // initialize input data
  {elem_type} hA[M*K];
  {elem_type} hB[K*N];
  memset(hA, 0, M*K*{elem_size});
  memset(hB, 0, K*N*{elem_size});
  {read_fn}(argv[1], hA, M*K);
  {read_fn}(argv[2], hB, K*N);
  aclrtMemcpyAsync(A, M*K*{elem_size}, hA, M*K*{elem_size}, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtMemcpyAsync(B, K*N*{elem_size}, hB, K*N*{elem_size}, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtSynchronizeStream(stream);

  // launch kernel
  int ret;
  int algo_id = {algo_id};
  if (algo_id == 0) {{
    ret = {kernel_name}_default(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1);
  }} else {{
    ret = {kernel_name}(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1, {algo_id});
  }}
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);
  aclrtSynchronizeStream(stream);

  // read data
  int32_t hC[M*N];
  memset(hC, 0, M*N*4);
  aclrtMemcpyAsync(hC, M*N*4, C, M*N*4, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  aclrtSynchronizeStream(stream);
  write_buffer_to_csv(argv[3], hC, M*N);

  // free acl handles
  unload_{kernel_name}();
  aclrtFree(A);
  aclrtFree(B);
  aclrtFree(C);
  aclrtDestroyStream(stream);
  aclFinalize();
}}
"""
    src = test_utils_src + test_src
    with open(os.path.join(dir, "test.c"), "w") as file:
        file.write(src)

    include_flags, link_flags, lib_dirs, python_libdir, python_lib, _ = _npu_toolchain()
    command = ["gcc", "test.c"]
    command.extend(include_flags)
    # The backend's link flags already carry the right -L/-l order. Keep
    # libpython in DT_NEEDED even though test.c never references it
    # directly: --as-needed would otherwise drop it, and the launcher
    # artifacts need Python data symbols in the process when dlopened.
    command.extend(link_flags)
    if python_libdir:
        command += [f"-L{python_libdir}", f"-Wl,-rpath,{python_libdir}"]
    for lib_dir in lib_dirs:
        command += [f"-Wl,-rpath,{lib_dir}"]
    command += [
        "-Wl,--no-as-needed", f"-l{python_lib}", "-Wl,--as-needed", "-ldl", "-L", dir, "-l", "kernel", "-o", exe
    ]
    subprocess.run(command, check=True, cwd=dir)


def _npu_run_env(tmp_dir):
    _, _, _, _, _, ld_library_path = _npu_toolchain()
    env = os.environ.copy()
    paths = [tmp_dir] + ld_library_path + [env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(path for path in paths if path)
    return env


def write_triton_kernels(dir, src, util_src):
    kernel_path = os.path.join(dir, "kernel.py")
    with open(kernel_path, "w") as file:
        file.write(src)

    kernel_utils_path = os.path.join(dir, "kernel_utils.py")
    with open(kernel_utils_path, "w") as file:
        file.write(util_src)

    return kernel_path


def _compile_kernel(dir, signature, kernel_name, out_name, out_path, num_warps, grid, kernel_path):
    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch_npu, runpy, sys; sys.argv = [sys.argv[1]] + sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')",
            compiler_path,
            "-n",
            kernel_name,
            "--signature",
            signature,
            "--out-name",
            out_name,
            "-o",
            out_path,
            "-w",
            str(num_warps),
            "-g",
            grid,
            kernel_path,
        ],
        env=_npu_run_env(dir),
        check=True,
        cwd=dir,
    )


# Edge case kernel with no specialization
def compile_aot_kernel_no_specialization(dir, kernel_path, dtype, BM, BN, BK):
    # compile all desired configs
    sig = f"*fp32, *{dtype}, *{dtype}, i32, i32, i32, i32, i32, i32, i32, i32, i32, {BM}, {BN}, {BK}"
    name = f"matmul_{dtype}"
    grid = f"M/{BM}, N/{BN}, 1"
    _compile_kernel(
        dir=dir,
        signature=sig,
        kernel_name="kernel",
        out_name=name,
        out_path=name,
        num_warps=1,
        grid=grid,
        kernel_path=kernel_path,
    )


def compile_aot_kernels(dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints):
    # compile all desired configs
    for ha, hb in ha_hb_hints:
        sig = f"*fp32:16, *{dtype}:16, *{dtype}:16, i32, i32, i32, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}"
        name = f"matmul_{dtype}"
        grid = f"M/{BM}, N/{BN}, 1"
        _compile_kernel(
            dir=dir,
            signature=sig,
            kernel_name="kernel",
            out_name=name,
            out_path=name,
            num_warps=1,
            grid=grid,
            kernel_path=kernel_path,
        )


def link_aot_kernels(dir):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(dir, "*.h"))
    cmd = [sys.executable, linker_path] + h_files + ["-o", "kernel"] + ["--backend", "npu"]
    subprocess.run(cmd, check=True, cwd=dir)


def compile_aot_vec_kernel(dir, kernel_path, scale, block):
    sig = f"*fp32, *fp32, i32, {scale}, {block}"
    name = "vec_kernel"
    grid = f"(n_elements + {block} - 1) / {block}, 1, 1"
    _compile_kernel(
        dir=dir,
        signature=sig,
        kernel_name="vec_kernel",
        out_name=name,
        out_path=name,
        num_warps=1,
        grid=grid,
        kernel_path=kernel_path,
    )


def gen_test_bin_npu_vec(dir, n, kernel_name="vec_kernel"):
    test_src = f"""
int main(int argc, char **argv) {{
  int n = {n};

  // initialize ACL handles
  aclrtStream stream = NULL;
  void *X = NULL, *Y = NULL;
  aclInit(NULL);
  int32_t device_id = 0;
  aclrtSetDevice(device_id);
  aclrtMalloc(&X, n * 4, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&Y, n * 4, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtCreateStream(&stream);
  load_{kernel_name}();

  // initialize input data
  int32_t hX[n];
  memset(hX, 0, n * 4);
  read_csv_to_buffer32(argv[1], hX, n);
  aclrtMemcpyAsync(X, n * 4, hX, n * 4, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtSynchronizeStream(stream);

  // launch kernel
  int ret = {kernel_name}_default(stream, Y, X, n);
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);
  aclrtSynchronizeStream(stream);

  // read data
  int32_t hY[n];
  memset(hY, 0, n * 4);
  aclrtMemcpyAsync(hY, n * 4, Y, n * 4, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  aclrtSynchronizeStream(stream);
  write_buffer_to_csv(argv[2], hY, n);

  // free acl handles
  unload_{kernel_name}();
  aclrtFree(X);
  aclrtFree(Y);
  aclrtDestroyStream(stream);
  aclFinalize();
}}
"""
    src = test_utils_src + test_src
    with open(os.path.join(dir, "test.c"), "w") as file:
        file.write(src)

    include_flags, link_flags, lib_dirs, python_libdir, python_lib, _ = _npu_toolchain()
    command = ["gcc", "test.c"]
    command.extend(include_flags)
    command.extend(link_flags)
    if python_libdir:
        command += [f"-L{python_libdir}", f"-Wl,-rpath,{python_libdir}"]
    for lib_dir in lib_dirs:
        command += [f"-Wl,-rpath,{lib_dir}"]
    command += [
        "-Wl,--no-as-needed", f"-l{python_lib}", "-Wl,--as-needed", "-ldl", "-L", dir, "-l", "kernel", "-o", "test"
    ]
    subprocess.run(command, check=True, cwd=dir)


def compile_multi_kernels(dir, kernel_path, block):
    sigs = [
        ("caller_kernel", f"*fp32, *fp32, i32, 2, {block}"),
        ("plain_kernel", f"*fp32, *fp32, i32, {block}"),
    ]
    grid = f"(n_elements + {block} - 1) / {block}, 1, 1"
    for kernel_name, sig in sigs:
        _compile_kernel(
            dir=dir,
            signature=sig,
            kernel_name=kernel_name,
            out_name=kernel_name,
            out_path=kernel_name,
            num_warps=1,
            grid=grid,
            kernel_path=kernel_path,
        )


def gen_test_bin_npu_multi(dir, n):
    test_src = f"""
int main(int argc, char **argv) {{
  int n = {n};

  // initialize ACL handles
  aclrtStream stream = NULL;
  void *X = NULL, *Y1 = NULL, *Y2 = NULL;
  aclInit(NULL);
  int32_t device_id = 0;
  aclrtSetDevice(device_id);
  aclrtMalloc(&X, n * 4, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&Y1, n * 4, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&Y2, n * 4, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtCreateStream(&stream);
  load_caller_kernel();
  load_plain_kernel();

  // initialize input data
  int32_t hX[n];
  memset(hX, 0, n * 4);
  read_csv_to_buffer32(argv[1], hX, n);
  aclrtMemcpyAsync(X, n * 4, hX, n * 4, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtSynchronizeStream(stream);

  // launch kernels
  int ret = caller_kernel_default(stream, Y1, X, n);
  if (ret != 0) fprintf(stderr, "caller kernel launch failed\\n");
  assert(ret == 0);
  ret = plain_kernel_default(stream, Y2, X, n);
  if (ret != 0) fprintf(stderr, "plain kernel launch failed\\n");
  assert(ret == 0);
  aclrtSynchronizeStream(stream);

  // read data
  int32_t hY[n];
  memset(hY, 0, n * 4);
  aclrtMemcpyAsync(hY, n * 4, Y1, n * 4, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  aclrtSynchronizeStream(stream);
  write_buffer_to_csv(argv[2], hY, n);
  memset(hY, 0, n * 4);
  aclrtMemcpyAsync(hY, n * 4, Y2, n * 4, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  aclrtSynchronizeStream(stream);
  write_buffer_to_csv(argv[3], hY, n);

  // free acl handles
  unload_caller_kernel();
  unload_plain_kernel();
  aclrtFree(X);
  aclrtFree(Y1);
  aclrtFree(Y2);
  aclrtDestroyStream(stream);
  aclFinalize();
}}
"""
    utils_src = test_utils_src.replace('#include "kernel.h"', "")
    includes = '#include "kernel_caller_kernel.h"\n#include "kernel_plain_kernel.h"\n'
    src = includes + utils_src + test_src
    with open(os.path.join(dir, "test.c"), "w") as file:
        file.write(src)

    include_flags, link_flags, lib_dirs, python_libdir, python_lib, _ = _npu_toolchain()
    command = ["gcc", "test.c"]
    command.extend(include_flags)
    command.extend(link_flags)
    if python_libdir:
        command += [f"-L{python_libdir}", f"-Wl,-rpath,{python_libdir}"]
    for lib_dir in lib_dirs:
        command += [f"-Wl,-rpath,{lib_dir}"]
    command += [
        "-Wl,--no-as-needed", f"-l{python_lib}", "-Wl,--as-needed", "-ldl", "-L", dir, "-l", "kernel", "-o", "test"
    ]
    subprocess.run(command, check=True, cwd=dir)


def link_aot_kernels_multi(dir, kernel_names):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    for kernel_name in kernel_names:
        h_files = glob.glob(os.path.join(dir, f"{kernel_name}.*.h"))
        cmd = [sys.executable, linker_path] + h_files + ["-o", f"kernel_{kernel_name}"] + ["--backend", "npu"]
        subprocess.run(cmd, check=True, cwd=dir)


def generate_matmul_test_data(dir, M, N, K, dtype="fp16"):
    if dtype == "fp32":
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        a_bits = a.view(np.int32)
        b_bits = b.view(np.int32)
    elif dtype == "bf16":
        import torch

        a_bits = torch.randn(M, K).to(torch.bfloat16).view(torch.int16).numpy()
        b_bits = torch.randn(K, N).to(torch.bfloat16).view(torch.int16).numpy()
        a = (a_bits.astype(np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)
        b = (b_bits.astype(np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)
    else:
        a = np.random.randn(M, K).astype(np.float16)
        b = np.random.randn(K, N).astype(np.float16)
        a_bits = a.view(np.int16)
        b_bits = b.view(np.int16)
    a_path = os.path.join(dir, "a.csv")
    b_path = os.path.join(dir, "b.csv")
    c_path = os.path.join(dir, "c.csv")
    for bits, path in [(a_bits, a_path), (b_bits, b_path)]:
        bits.ravel().tofile(path, sep=",")
    return a, b, a_path, b_path, c_path


def test_compile_link_matmul_no_specialization():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernel_no_specialization(tmp_dir, kernel_path, dtype, BM, BN, BK)

        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        subprocess.run(["./test", a_path, b_path, c_path], env=_npu_run_env(tmp_dir), check=True, cwd=tmp_dir)

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


def test_compile_link_matmul():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=[(":16", ":16")])
        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        subprocess.run(["./test", a_path, b_path, c_path], env=_npu_run_env(tmp_dir), check=True, cwd=tmp_dir)

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("dtype, atol, rtol", [("fp32", 1e-3, 1e-3), ("fp16", 1e-3, 1e-3), ("bf16", 1e-3, 1e-3)])
@pytest.mark.parametrize("shape", [(16, 16, 16), (32, 64, 24), (128, 64, 32), (64, 16, 64)])
def test_compile_link_matmul_shapes(dtype, atol, rtol, shape):
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        BM, BN, BK = 16, 16, 16
        M, N, K = shape

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernel_no_specialization(tmp_dir, kernel_path, dtype, BM, BN, BK)
        link_aot_kernels(tmp_dir)

        # compile test case
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu(tmp_dir, M, N, K, dtype=dtype, kernel_name=f"matmul_{dtype}")

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K, dtype=dtype)

        # run test case
        subprocess.run(["./test", a_path, b_path, c_path], env=_npu_run_env(tmp_dir), check=True, cwd=tmp_dir)

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(4096, ), (100, ), (64, 64), (16, 16, 16), (8, 8, 8, 8), (4, 4, 4, 4, 4),
                                   (4, 64, 4, 4, 4, 32)])
def test_compile_link_vec_shapes(shape):
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        n = int(np.prod(shape))
        scale, block = 2, 1024

        kernel_path = write_triton_kernels(tmp_dir, vec_kernel_src, "")
        compile_aot_vec_kernel(tmp_dir, kernel_path, scale, block)
        link_aot_kernels(tmp_dir)

        # compile test case
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu_vec(tmp_dir, n)

        # initialize test data
        x = np.random.randn(n).astype(np.float32)
        x_path = os.path.join(tmp_dir, "x.csv")
        y_path = os.path.join(tmp_dir, "y.csv")
        x.view(np.int32).ravel().tofile(x_path, sep=",")

        # run test case
        subprocess.run(["./test", x_path, y_path], env=_npu_run_env(tmp_dir), check=True, cwd=tmp_dir)

        # read data and compare against reference
        y = np.genfromtxt(y_path, delimiter=",", dtype=np.int32).reshape(shape).view(np.float32)
        np.testing.assert_allclose(y, x.reshape(shape) * scale, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kernel_src_variant", ["cross_file", "same_file"])
def test_compile_link_multi_kernels(kernel_src_variant):
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        n, block = 1536, 1024
        if kernel_src_variant == "cross_file":
            kernel_path = write_triton_kernels(tmp_dir, multi_kernel_src, kernel_utils_src)
        else:
            kernel_path = write_triton_kernels(tmp_dir, multi_kernel_same_file_src, "")

        compile_multi_kernels(tmp_dir, kernel_path, block)
        link_aot_kernels_multi(tmp_dir, ["caller_kernel", "plain_kernel"])

        # compile test case
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu_multi(tmp_dir, n)

        # initialize test data
        x = np.random.randn(n).astype(np.float32)
        x_path = os.path.join(tmp_dir, "x.csv")
        y1_path = os.path.join(tmp_dir, "y1.csv")
        y2_path = os.path.join(tmp_dir, "y2.csv")
        x.view(np.int32).ravel().tofile(x_path, sep=",")

        # run test case
        subprocess.run(["./test", x_path, y1_path, y2_path], env=_npu_run_env(tmp_dir), check=True, cwd=tmp_dir)

        # read data and compare against reference
        y1 = np.genfromtxt(y1_path, delimiter=",", dtype=np.int32).view(np.float32)
        y2 = np.genfromtxt(y2_path, delimiter=",", dtype=np.int32).view(np.float32)
        np.testing.assert_allclose(y1, x * 2, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(y2, x + 1.0, atol=1e-4, rtol=1e-4)


def test_launcher_has_no_available_kernel():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=[(":1", ":1")])

        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        result = subprocess.run(
            ["./test", a_path, b_path, c_path],
            env=_npu_run_env(tmp_dir),
            cwd=tmp_dir,
            capture_output=True,
            text=True,
        )

        # It should fail since the launcher requires all the strides be 1 while they are not.
        assert result.returncode == -6
        assert "kernel launch failed" in result.stderr
