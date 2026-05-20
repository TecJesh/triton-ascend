# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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


from __future__ import annotations

from triton.runtime.jit import jit
from triton.language import core
from triton.language.standard import _elementwise_max, _elementwise_min, _sum_combine, _argmax_combine_tie_break_left, \
    _argmax_combine_tie_break_fast, _argmin_combine_tie_break_left, _argmin_combine_tie_break_fast

from . import math as cann_math

# constexpr utilities


@core._tensor_member_fn
@jit
def cdiv(x, div):
    return cann_math.cdiv(x, div)


@jit
def _elementwise_max_propagate_nan(a, b):
    return core.maximum(a, b, propagate_nan = core.PropagateNan.ALL)

@core._tensor_member_fn
@jit
@core._add_reduction_docstr("maximum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False, propagate_nan = False):
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < core.constexpr(32):
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                # FIXME: Skip int8/int16 -> int32 promotion on Ascend.
                # Converting small integer types (e.g., int8) to int32 consumes excessive UB (Unified Buffer) memory,
                # which can lead to "UB overflow" errors during kernel execution.
                # Therefore, we keep the original narrow integer type and rely on backend support.
                pass  # Do not promote to int32
        if not propagate_nan:
            return core.reduce(input, axis, _elementwise_max, keep_dims=keep_dims)
        else:
            return core.reduce(input, axis, _elementwise_max_propagate_nan, keep_dims=keep_dims)


# min and argmin


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("minimum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < 32:
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                # FIXME: Skip int8/int16 -> int32 promotion on Ascend.
                # Converting small integer types (e.g., int8) to int32 consumes excessive UB (Unified Buffer) memory,
                # which can lead to "UB overflow" errors during kernel execution.
                # Therefore, we keep the original narrow integer type and rely on backend support.
                pass  # Do not promote to int32
        return core.reduce(input, axis, _elementwise_min, keep_dims=keep_dims)


# sum


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("sum", dtype_arg="dtype")
def sum(input, axis=None, keep_dims=False, dtype: core.constexpr = None):
    # Pick a default dtype for the reduction if one was not specified.
    # out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)

    # if out_dtype is not None:
    #     input = input.to(out_dtype)
    # Triton Ascend not need the type promotion logic of community as commented above, perform the operation normally 
    return core.reduce(input, axis, _sum_combine, keep_dims=keep_dims)
