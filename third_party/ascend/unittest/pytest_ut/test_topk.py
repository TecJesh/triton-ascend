# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

# import pytest

# import triton
# import triton.language as tl

# import torch
# import test_common

# @pytest.mark.interpreter
# @pytest.mark.parametrize("shape", [
#     (16, 64),
#     (16, 8, 8),
#     (2, 8, 8, 8),
# ])
# @pytest.mark.parametrize("k", [4])
# @pytest.mark.parametrize("dtype", ["float32"])
# def test_topk_nd(shape, k, dtype):

#     print(f"[topk] shape={shape}, k={k}, dtype={dtype}", flush=True)

#     numel = 1
#     for d in shape:
#         numel *= d
#     base = torch.arange(numel, dtype=torch.float32).view(shape) / 10.0
#     if dtype == "float16":
#         base = base.half()
#     elif dtype == "int32":
#         base = torch.arange(numel, dtype=torch.int32).view(shape)
#     print(f"[topk] base tensor ready, device={base.device}", flush=True)

#     x = base.npu()
#     print(f"[topk] .npu() done", flush=True)

#     y = torch.topk(base, k=k, dim=-1).values
#     print(f"[topk] torch.topk (CPU ref) done", flush=True)

#     M = int(torch.tensor(shape[:-1]).prod().item()) if len(shape) > 1 else 1
#     N = shape[-1]
#     print(f"[topk] M={M}, N={N}, k={k}", flush=True)

#     x_2d = x.view(M, N)
#     z = torch.empty((M, k), dtype=x_2d.dtype, device=x_2d.device)
#     print(f"[topk] x_2d={x_2d.shape}, z={z.shape}, device={z.device}", flush=True)

#     @triton.jit
#     def topk_kernel_nd(X, stride_xm, Z, stride_zm, M: tl.constexpr, N: tl.constexpr, k: tl.constexpr):
#         tl.device_print("=== kernel entry ===")
#         offs_m = tl.arange(0, M)
#         offs_x_n = tl.arange(0, N)
#         offs_z_n = tl.arange(0, k)
#         offs_x = offs_m[:, None] * stride_xm + offs_x_n[None, :]
#         tl.device_print("offs_x computed, shape=(%d,%d)", M, N)
#         x_val = tl.load(X + offs_x)
#         z_val = tl.topk(x_val, k)
#         offs_z = offs_m[:, None] * stride_zm + offs_z_n[None, :]
#         tl.store(Z + offs_z, z_val)
#         tl.device_print("=== kernel exit ===")

#     print(f"[topk] launching kernel...", flush=True)
#     topk_kernel_nd[(1, )](x_2d, x_2d.stride(0), z, z.stride(0), M, N, k)
#     print(f"[topk] kernel done", flush=True)

#     z_view = z.view(*shape[:-1], k) if len(shape) > 1 else z.view(k)
#     print(f"[topk] validating...", flush=True)
#     test_common.validate_cmp(dtype, z_view.cpu(), y)
#     print(f"[topk] PASS", flush=True)
