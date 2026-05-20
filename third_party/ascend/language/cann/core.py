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

from warnings import warn
from typing import Optional
import builtins
from triton import knobs

from triton.language.core import builtin, _tensor_member_fn, _unwrap_iterable, _unwrap_if_constexpr, \
    _shape_check_impl, _wrap_axis, _insertion_guard, tensor, float32, expand_dims, dtype, tensor, add, sub, mul


# class ascend_tensor(tensor):
    
@builtin
def __add__(self, other, _semantic=None):
    return add(self, other, sanitize_overflow=False, _semantic=_semantic)

@builtin
def __radd__(self, other, _semantic=None):
    return add(other, self, sanitize_overflow=False, _semantic=_semantic)

@builtin
def __sub__(self, other, _semantic=None):
    return sub(self, other, sanitize_overflow=False, _semantic=_semantic)

@builtin
def __rsub__(self, other, _semantic=None):
    return sub(other, self, sanitize_overflow=False, _semantic=_semantic)

@builtin
def __mul__(self, other, _semantic=None):
    return mul(self, other, sanitize_overflow=False, _semantic=_semantic)

@builtin
def __rmul__(self, other, _semantic=None):
    return mul(other, self, sanitize_overflow=False, _semantic=_semantic)

@builtin
def to(self, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, overflow_mode: Optional[str] = None, _semantic=None):
    """
    Alias for :py:func:`tensor.cast`.
    """
    return cast(self, dtype, fp_downcast_rounding, bitcast, overflow_mode, _semantic=_semantic)

def cast(self, dtype, fp_downcast_rounding=None, bitcast=False, overflow_mode=None) -> tensor:
    ...


@_tensor_member_fn
@builtin
def view(input, *shape, _semantic=None):
    """
    Returns a tensor with the same elements as `input` but a different shape.
    The order of the elements may not be preserved.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        view(x, (32, 32))
        view(x, 32, 32)
    """
    warn("view is deprecated, please use reshape with can_reorder being true.")
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return _semantic.reshape(input, shape, can_reorder=False)


@_tensor_member_fn
@builtin
def cast(input, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, overflow_mode: Optional[str] = None, _semantic=None):
    """
    Casts a tensor to the given :code:`dtype`.

    :param dtype: The target data type.
    :type dtype: tl.dtype
    :param fp_downcast_rounding: The rounding mode for downcasting
        floating-point values. This parameter is only used when self is a
        floating-point tensor and dtype is a floating-point type with a
        smaller bitwidth. Supported values are :code:`"rtne"` (round to
        nearest, ties to even) and :code:`"rtz"` (round towards zero).
    :type fp_downcast_rounding: str, optional
    :param bitcast: If true, the tensor is bitcasted to the given
        :code:`dtype`, instead of being numerically casted.
    :type bitcast: bool, optional
    """
    input = _semantic.to_tensor(input)
    dtype = _unwrap_if_constexpr(dtype)
    fp_downcast_rounding = _unwrap_if_constexpr(fp_downcast_rounding)
    bitcast = _unwrap_if_constexpr(bitcast)
    if bitcast:
        return _semantic.bitcast(input, dtype)
    return _semantic.cast(input, dtype, fp_downcast_rounding)


# -----------------------
# Linear Algebra
# -----------------------


@builtin
def dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=float32,
        _semantic=None):
    """
    Returns the matrix product of two blocks.

    The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions.
    For three-dimensional blocks, `tl.dot` performs the batched matrix product,
    where the first dimension of each block represents the batch dimension.

    :param input: The first tensor to be multiplied.
    :type input: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param acc: The accumulator tensor. If not None, the result is added to this tensor.
    :type acc: 2D or 3D tensor of scalar-type in {:code:`float16`, :code:`float32`, :code:`int32`}
    :param input_precision: How to exercise the Tensor Cores for f32 x f32. If
      the device does not have Tensor Cores or the inputs are not of dtype f32,
      this option is ignored. For devices that do have tensor cores, the
      default precision is tf32.
    :type input_precision: string. Available options for nvidia: :code:`"tf32"`, :code:`"tf32x3"`, :code:`"ieee"`. Default: :code:`"tf32"`. Available options for amd: :code:`"ieee"`, (CDNA3 only) :code:`"tf32"`.
    :param allow_tf32: *Deprecated.* If true, input_precision is set to "tf32".
      Only one of :code:`input_precision` and :code:`allow_tf32` can be
      specified (i.e. at least one must be :code:`None`).
    """
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    assert not allow_tf32, "allow_tf32 is not supported as 'True', please use fp32 on Ascend instead."
    if input_precision is None:
        supports_tf32 = "tf32" in _semantic.builder.options.allowed_dot_input_precisions
        input_precision = knobs.language.fp32_default or ("tf32" if (supports_tf32 and
                                                                     (allow_tf32 or allow_tf32 is None)) else "ieee")
        # when setting allow_tf32, use input_precision='hf32' on Ascend instead.
        if allow_tf32:
            input_precision = "hf32"
    else:
        if input_precision == "tf32":
            input_precision = "hf32"

    input_precision = _unwrap_if_constexpr(input_precision)
    out_dtype = _unwrap_if_constexpr(out_dtype)
    max_num_imprecise_acc = _unwrap_if_constexpr(max_num_imprecise_acc)
    acc = _unwrap_if_constexpr(acc)
    return _semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype)


# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, care_padding = True, _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for
        cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1),
        and ".cv" means don’t cache and fetch again. see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    :param care_padding: represents whether user cares about padding value or not, default is True, works as below:
        1. if 'other' is not None, 'care_padding' takes no effect.
        2. if 'other' is None and 'care_padding' = True, loaded tensor will fill zeroes on masked places.
        3. if 'other' is None and 'care_padding' = False, masked places on loaded tensor will be random values, and tl.load may have a better performence.
    :type care_padding: bool, optional
    """
    # `mask` and `other` can be constexpr
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    if other is not None:
        other = _semantic.to_tensor(other)
    padding_option = _unwrap_if_constexpr(padding_option)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    care_padding = _unwrap_if_constexpr(care_padding)
    return _semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                          volatile, care_padding)


@_tensor_member_fn
@builtin
def reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None):
    """Applies the combine_fn to all elements in :code:`input` tensors along the provided :code:`axis`

    :param input: the input tensor, or tuple of tensors
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done. If None, reduce all dimensions
    :type axis: int | None
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)
    :type combine_fn: Callable
    :param keep_dims: if true, keep the reduced dimensions with length 1
    :type keep_dims: bool

    """
    if isinstance(input, tensor):
        return reduce((input, ), axis, combine_fn, keep_dims=keep_dims, _semantic=_semantic, _generator=_generator)[0]

    def make_combine_region(reduce_op):
        param_types = [t.type.scalar for t in input] * 2
        region = reduce_op.get_region(0)
        builder = _semantic.builder
        with _insertion_guard(builder):
            to_ir = lambda T: T.to_ir(builder)
            block = builder.create_block_with_parent(region, list(map(to_ir, param_types)))
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            builder.create_reduce_ret(*handles)

    def expand_ndims(t, ndims):
        for _ in builtins.range(ndims):
            t = expand_dims(t, 0, _semantic=_semantic)
        return t

    axis = _unwrap_if_constexpr(axis)
    keep_dims = _unwrap_if_constexpr(keep_dims)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    ret = _semantic.reduction(input, axis, make_combine_region)
    if keep_dims:
        if axis is not None:
            ret = builtins.tuple(expand_dims(t, axis, _semantic=_semantic) for t in ret)
        else:
            ret = builtins.tuple(expand_ndims(t, len(input[0].shape)) for t in ret)
    return ret


@builtin
def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=True, lhs_k_pack=True,
               rhs_k_pack=True, out_dtype=float32, _semantic=None):
    """
    Returns the matrix product of two blocks in microscaling format.

    lhs and rhs use microscaling formats described here:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

    Software emulation enables targeting hardware architectures without native microscaling
    operation support. Right now for such case, microscaled lhs/rhs are upcasted to
    :code:`bf16` element type beforehand for dot computation, with one exception:
    for AMD CDNA3 specifically, if one of the inputs is of :code:`fp16` element type,
    the other input is also upcasted to :code:`fp16` element type instead.
    This behavior is experimental and may be subject to change in the future.

    :param lhs: The first tensor to be multiplied.
    :type lhs: 2D tensor representing fp4, fp8 or bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8 or the corresponding fp8 type.
    :param lhs_scale: Scale factor for lhs tensor.
    :type lhs_scale: e8m0 type represented as an uint8 tensor.
    :param lhs_format: format of the lhs tensor. Available formats: {:code:`e2m1`, :code:`e4m3`, :code:`e5m2`, :code:`bf16`, :code:`fp16`}.
    :type lhs_format: str
    :param rhs: The second tensor to be multiplied.
    :type rhs: 2D tensor representing fp4, fp8 or bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8 or the corresponding fp8 type.
    :param rhs_scale: Scale factor for rhs tensor.
    :type rhs_scale: e8m0 type represented as an uint8 tensor.
    :param rhs_format: format of the rhs tensor. Available formats: {:code:`e2m1`, :code:`e4m3`, :code:`e5m2`, :code:`bf16`, :code:`fp16`}.
    :type rhs_format: str
    :param acc: The accumulator tensor. If not None, the result is added to this tensor.
    :param lhs_k_pack: If false, the lhs tensor is packed into uint8 along M dimension.
    :type lhs_k_pack: bool, optional
    :param rhs_k_pack: If false, the rhs tensor is packed into uint8 along N dimension.
    :type rhs_k_pack: bool, optional
    """
    out_dtype = _unwrap_if_constexpr(out_dtype)
    assert out_dtype == float32, "Only float32 is supported for out_dtype at the moment"
    return _semantic.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math, lhs_k_pack,
                                rhs_k_pack, out_dtype)