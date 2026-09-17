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

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import test_common


@triton.jit
def load_effective_token(
    committed_ptr,
    committed_stride,
    pending_ptr,
    request_start,
    request_idx,
    committed_len,
    pos,
):
    if pos < committed_len:
        return tl.load(committed_ptr + request_idx * committed_stride + pos)
    pending_pos = request_start + pos - committed_len + 1
    return tl.load(pending_ptr + pending_pos)


@triton.jit
def reproduce_kernel(
    committed_ptr,
    pending_ptr,
    lengths_ptr,
    request_mapping_ptr,
    output_ptr,
    committed_stride,
):
    token_idx = tl.program_id(0).to(tl.int64)
    request_idx = tl.load(request_mapping_ptr + token_idx)
    committed_len = tl.load(lengths_ptr + request_idx)
    effective_len = committed_len + 1
    total = tl.zeros((), dtype=tl.int32)
    for pos in tl.range(committed_len - 1, effective_len):
        token = load_effective_token(
            committed_ptr,
            committed_stride,
            pending_ptr,
            token_idx,
            request_idx,
            committed_len,
            pos,
        )
        total += token
    tl.store(output_ptr + token_idx, total)


@triton.jit
def helper_top(x_ptr, idx):
    v = tl.load(x_ptr + idx)
    if v > 0:
        return v + 1
    return v + 2


@triton.jit
def helper_add_const(x_ptr, idx, c: tl.constexpr):
    return tl.load(x_ptr + idx) + c


@triton.jit
def helper_loop_sum_early(x_ptr, idx, N: tl.constexpr):
    s = tl.zeros((), dtype=tl.int32)
    for j in tl.range(0, N):
        s += helper_add_const(x_ptr, idx * N + j, 1)
    if s >= 0:
        return s
    return 0


@triton.jit
def helper_early_select(x_ptr, idx):
    v = tl.load(x_ptr + idx)
    if v > 0:
        return v + 1
    return v + 2


@triton.jit
def helper_add_const_inline(x_ptr, idx, c: tl.constexpr):
    return tl.load(x_ptr + idx) + c


@triton.jit
def kernel_top_call(x_ptr, out_ptr):
    v = helper_top(x_ptr, 0)
    tl.store(out_ptr, v)


@triton.jit
def kernel_nested_loop_call(x_ptr, out_ptr, N: tl.constexpr):
    total = tl.zeros((), dtype=tl.int32)
    for i in tl.range(0, N):
        total += helper_loop_sum_early(x_ptr, i, N)
    tl.store(out_ptr, total)


@triton.jit
def kernel_if_call(x_ptr, out_ptr, mode_ptr):
    mode = tl.load(mode_ptr)
    v = tl.zeros((), dtype=tl.int32)
    if mode > 0:
        if mode > 1:
            v = helper_early_select(x_ptr, 0)
        else:
            v = helper_early_select(x_ptr, 1)
    else:
        v = helper_early_select(x_ptr, 2)
    tl.store(out_ptr, v)


@triton.jit
def kernel_loop_call_inline(x_ptr, out_ptr, N: tl.constexpr):
    total = tl.zeros((), dtype=tl.int32)
    for i in tl.range(0, N):
        total += helper_add_const_inline(x_ptr, i, 1)
    tl.store(out_ptr, total)


def _early_select_ref(v):
    return int(v) + 1 if int(v) > 0 else int(v) + 2


def _effective_token_reference(committed, pending, lengths, mapping):
    totals = []
    for t, r in enumerate(mapping.tolist()):
        committed_len = int(lengths[r].item())
        total = 0
        for pos in range(committed_len - 1, committed_len + 1):
            if pos < committed_len:
                total += int(committed[r, pos].item())
            else:
                pending_pos = t + pos - committed_len + 1
                total += int(pending[pending_pos].item())
        totals.append(total)
    return torch.tensor(totals, dtype=torch.int32)


def _run_effective_token_case(num_requests, max_committed_len, committed_stride):
    device = "npu"
    torch.manual_seed(0)
    committed = torch.randint(-100, 100, (num_requests, max_committed_len), dtype=torch.int32, device=device)
    pending_len = num_requests + max_committed_len
    pending = torch.randint(-100, 100, (pending_len, ), dtype=torch.int32, device=device)
    lengths = torch.randint(1, max_committed_len + 1, (num_requests, ), dtype=torch.int32, device=device)
    mapping = torch.randperm(num_requests, dtype=torch.int32, device=device)
    output = torch.empty(num_requests, dtype=torch.int32, device=device)
    reproduce_kernel[(num_requests, )](
        committed,
        pending,
        lengths,
        mapping,
        output,
        committed.stride(0),
    )
    torch.npu.synchronize()
    expected = _effective_token_reference(committed.cpu(), pending.cpu(), lengths.cpu(), mapping.cpu())
    test_common.validate_cmp("int32", output.cpu(), expected)


@pytest.mark.parametrize("N", [4, 32, 127])
def test_jit_call_at_top_level(N):
    x = test_common.generate_tensor(shape=(N, ), dtype="int32")
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_top_call[(1, )](x.npu(), out)
    expected = torch.tensor([_early_select_ref(x[0])], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


@pytest.mark.parametrize(
    "num_requests, max_committed_len, committed_stride",
    [
        (1, 2, 2),
        (3, 5, 4),
        (2, 1, 3),
    ],
)
def test_jit_call_in_for_loop_effective_token(num_requests, max_committed_len, committed_stride):
    _run_effective_token_case(num_requests, max_committed_len, committed_stride)


@pytest.mark.parametrize("N", [4, 16, 33])
def test_jit_call_nested_loop(N):
    x = torch.randint(0, 100, (N * N, ), dtype=torch.int32)
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_nested_loop_call[(1, )](x.npu(), out, N)
    expected = torch.tensor([int(x.sum()) + N * N], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


@pytest.mark.parametrize("mode, x_idx", [(0, 2), (1, 1), (2, 0)])
def test_jit_call_in_nested_if(mode, x_idx):
    x = test_common.generate_tensor(shape=(4, ), dtype="int32")
    mode_tensor = torch.tensor([mode], dtype=torch.int32, device="npu")
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_if_call[(1, )](x.npu(), out, mode_tensor)
    expected = torch.tensor([_early_select_ref(x[x_idx])], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


@pytest.mark.parametrize("N", [4, 32, 127])
def test_jit_call_in_for_loop_inlined(N):
    x = test_common.generate_tensor(shape=(N, ), dtype="int32")
    out = torch.empty((1, ), dtype=torch.int32, device="npu")
    kernel_loop_call_inline[(1, )](x.npu(), out, N)
    expected = torch.tensor([int(x.sum()) + N], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)
