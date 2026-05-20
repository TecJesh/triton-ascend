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

import warnings

from typing import Optional, Tuple
import numbers

from triton._C.libtriton import ir
from triton.language import core as tl
from triton.language.semantic import TritonSemantic as BaseTritonSemantic
from triton.tools.get_ascend_devices import is_compile_on_910_95
from triton.language.semantic import TensorTy


class AscendTritonSemantic(BaseTritonSemantic):

    def mod(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, False, False, True, True)
        scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        # float % float
        if scalar_ty.is_floating():
            # input - input.div(other, rounding_mode="floor") * other
            return self.tensor(self.builder.create_frem(input.handle, other.handle), input.type)
        # % int
        elif scalar_ty.is_int():
            if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
                raise TypeError("Cannot mod " + scalar_ty.__repr__() + " by " + other_scalar_ty.__repr__() + " "
                                "because they have different signedness;"
                                "this is unlikely to result in a useful answer. Cast them to the same signedness.")
            if hasattr(input, 'was_bool_to_int8'):
                false_val = self.builder.get_int1(False)
                return self.tensor(false_val, tl.int1)
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_srem(input.handle, other.handle), input.type)
            else:
                return self.tensor(self.builder.create_urem(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {scalar_ty}")

    def logical_and(self, input: TensorTy, other: TensorTy) -> TensorTy:
        dst_sca_ty = tl.dtype("int1")
        dst_bits = dst_sca_ty.primitive_bitwidth
        if hasattr(input, 'was_bool_to_int8'):
            assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
            input = self.cast(input, tl.int1)
        if not input.type.is_int1():
            src_sca_ty = input.type.scalar
            src_bits = src_sca_ty.primitive_bitwidth
            if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
                input = self.bitcast(input, tl.int1)
            else:
                input = self.not_equal(input, 0)
        if hasattr(other, 'was_bool_to_int8'):
            assert other.type.scalar.is_int8(), "Other input wat bool to int8. However, other input.type is not int8."
            other = self.cast(other, tl.int1)
        if not other.type.is_int1():
            src_sca_ty = other.type.scalar
            src_bits = src_sca_ty.primitive_bitwidth
            if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
                other = self.bitcast(other, tl.int1)
            else:
                other = self.not_equal(other, 0)
        return self.and_(input, other)

    def logical_or(self, input: TensorTy, other: TensorTy) -> TensorTy:
        dst_sca_ty = tl.dtype("int1")
        dst_bits = dst_sca_ty.primitive_bitwidth
        if hasattr(input, 'was_bool_to_int8'):
            assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
            input = self.cast(input, tl.int1)
        if not input.type.is_int1():
            src_sca_ty = input.type.scalar
            src_bits = src_sca_ty.primitive_bitwidth
            if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
                input = self.bitcast(input, tl.int1)
            else:
                input = self.not_equal(input, 0)
        if hasattr(other, 'was_bool_to_int8'):
            assert other.type.scalar.is_int8(), "Other wat bool to int8. However, other.type is not int8."
            other = self.cast(other, tl.int1)
        if not other.type.is_int1():
            src_sca_ty = other.type.scalar
            src_bits = src_sca_ty.primitive_bitwidth
            if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
                other = self.bitcast(other, tl.int1)
            else:
                other = self.not_equal(other, 0)
        return self.or_(input, other)

    def not_(self, input: TensorTy):
        if hasattr(input, 'was_bool_to_int8'):
            assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
            input = self.cast(input, tl.int1)
        if input.type.scalar.is_floating():
            raise TypeError(f"unexpected type {input.type.scalar}")
        return self.invert(input)

    def minus(self, input: TensorTy) -> TensorTy:
        input_sca_ty = input.type.scalar
        if hasattr(input, 'was_bool_to_int8'):
            if input.type.scalar.is_int8():
                raise TypeError(f"unexpected type bool")
        if input_sca_ty.is_ptr():
            raise ValueError("wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")")
        _0 = self.tensor(self.builder.get_null_value(input_sca_ty.to_ir(self.builder)), input_sca_ty)
        return self.sub(_0, input, True)

    def invert(self, input: TensorTy) -> TensorTy:
        if hasattr(input, 'was_bool_to_int8'):
            assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
            input = self.cast(input, tl.int1)
        input_sca_ty = input.type.scalar
        if input_sca_ty.is_floating():
            raise TypeError(f"unexpected type {input_sca_ty}")
        if input_sca_ty.is_ptr():
            raise ValueError("wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")")
        _1 = self.tensor(self.builder.get_all_ones_value(input_sca_ty.to_ir(self.builder)), input_sca_ty)
        return self.xor_(input, _1)

    def arange(self, start: int, end: int, *, ret_ty: tl.block_type = None) -> TensorTy:
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("arange's arguments must be of type tl.constexpr")
        is_start_int64 = bool(start >> 32)
        is_end_int64 = bool(end >> 32)
        if is_start_int64 or is_end_int64:
            raise ValueError("arange must fit in int32")
        if end <= start:
            raise ValueError("arange's end argument must be greater than the start argument")
        range = end - start
        # Check if compile_mode is simt, then range must be a power of 2
        if self.builder.is_simt_mode():
            # Check if range is a power of 2
            if (range & (range - 1)) != 0:
                    raise ValueError("arange's range must be a power of 2")
        shape = [range]
        if ret_ty is None:
            ret_ty = tl.block_type(tl.int32, shape)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        return self.tensor(self.builder.create_make_range(ret_ty_ir, start, end), ret_ty)

    def cast(self, input: TensorTy, dst_ty: tl.dtype, fp_downcast_rounding: Optional[str] = None, overflow_mode: Optional[str] = None) -> TensorTy:
        src_ty = input.type
        src_sca_ty = src_ty.scalar
        dst_sca_ty = dst_ty.scalar
        if src_sca_ty == dst_sca_ty:
            return input
        if src_ty.is_block():
            dst_ty = src_ty.with_element_ty(dst_sca_ty)

        # For fp downcasting default rounding mode should be RTNE, for all other conversions it should
        # not be set
        fp_downcast_rounding = self._str_to_rounding_mode(fp_downcast_rounding)
        use_custom_rounding = False
        if dst_sca_ty.is_floating() and src_sca_ty.is_floating(
        ) and dst_sca_ty.primitive_bitwidth < src_sca_ty.primitive_bitwidth:
            if fp_downcast_rounding is None: fp_downcast_rounding = ir.ROUNDING_MODE.RTNE
            elif fp_downcast_rounding != ir.ROUNDING_MODE.RTNE: use_custom_rounding = True
        else:
            if fp_downcast_rounding is not None:
                raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions. "
                                 "Source scalar type is " + str(src_sca_ty) + " and destination type is " +
                                 str(dst_sca_ty))

        if (src_sca_ty.is_fp8e4b15() or dst_sca_ty.is_fp8e4b15()):
            assert self.builder.codegen_fns.get(
                "convert_custom_types") is not None, "target doesn't provide conversion for this type."
            return self.builder.codegen_fns["convert_custom_types"](input, dst_ty, fp_downcast_rounding, _semantic=self)
        # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
        # and non-default rounding modes for downcasting
        if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
           (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()) or \
           use_custom_rounding:
            return self.tensor(
                self.builder.create_fp_to_fp(input.handle, dst_ty.to_ir(self.builder), fp_downcast_rounding), dst_ty)

        # bf16 <=> (not fp32)
        if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
           (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
            return self.cast(self.cast(input, tl.float32), dst_sca_ty)

        # Standard floating types' casting: truncation
        #   fp64 => fp32, fp16, bf16
        #   fp32 => fp16, bf16
        truncate_fp = src_sca_ty.is_floating() and \
            dst_sca_ty.is_floating() and \
            src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
        if truncate_fp:
            return self.tensor(self.builder.create_fp_trunc(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Standard floating types' casting: extension
        #   fp32 => fp64
        #   fp16 => fp32, fp64
        #   bf16 => fp32, fp64
        ext_fp = src_sca_ty.is_floating() and \
            dst_sca_ty.is_floating() and \
            src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
        if ext_fp:
            return self.tensor(self.builder.create_fp_ext(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting between integer types
        if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
           (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
            sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
            if dst_sca_ty.is_bool():
                ty = input.dtype.to_ir(self.builder)
                _0 = self.tensor(self.builder.get_null_value(ty), input.dtype)
                return self.not_equal(input, _0)
            elif overflow_mode == "saturate" and \
                (src_sca_ty.is_int_unsigned() or dst_sca_ty.is_int_unsigned()) and \
                src_sca_ty.int_bitwidth >= dst_sca_ty.int_bitwidth:
                return self.cast(self.cast(input, tl.float32), dst_sca_ty)
            else:
                return self.tensor(self.builder.create_int_cast(input.handle, dst_ty.to_ir(self.builder), sign_extend),
                                   dst_ty)

        # Casting standard floating types to integer types
        if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
            if dst_sca_ty.is_bool():
                ty = input.dtype.to_ir(self.builder)
                _0 = self.tensor(self.builder.get_null_value(ty), input.dtype)
                return self.not_equal(input, _0)
            elif dst_sca_ty.is_int_signed():
                return self.tensor(self.builder.create_fp_to_si(input.handle, dst_ty.to_ir(self.builder)), dst_ty)
            else:
                return self.tensor(self.builder.create_fp_to_ui(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting integer types to standard floating types
        if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
            if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
                return self.tensor(self.builder.create_ui_to_fp(input.handle, dst_ty.to_ir(self.builder)), dst_ty)
            else:
                return self.tensor(self.builder.create_si_to_fp(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting pointer types to integer types
        if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
            bitwidth = dst_sca_ty.int_bitwidth
            if bitwidth == 64:
                return self.tensor(self.builder.create_ptr_to_int(input.handle, dst_ty.to_ir(self.builder)), dst_ty)
            if bitwidth == 1:
                return self.not_equal(self.cast(input, tl.int64), self.tensor(self.builder.get_int64(0), tl.int64))

        # Casting integer types to pointer types
        if src_sca_ty.is_int() and dst_sca_ty.is_ptr():
            return self.tensor(self.builder.create_int_to_ptr(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting pointer types to pointer types
        if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
            return self.tensor(self.builder.create_bitcast(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        assert False, f'cannot cast {input} to {dst_ty}'

    def _load_block_pointer(self, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile):
        # Load by a block pointer: `pointer_type<block_type<>>`
        # Block pointer can not have `mask` and `other` arguments
        if mask is not None or other is not None:
            raise ValueError("`mask` and `other` arguments cannot be specified for loading block pointers")

        elt_ty = ptr.type.element_ty.element_ty
        assert elt_ty != tl.int1, "`tl.int1` should be rewritten in `tl.make_block_ptr`"
        if elt_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
            raise ValueError("Padding option `nan` is not supported for integer block pointers")

        # `dst_ty` is de-referenced type of the pointer type
        dst_ty = ptr.type.element_ty

        # Check `boundary_check` argument
        boundary_check = self._canonicalize_boundary_check(boundary_check, dst_ty.get_block_shapes())

        if boundary_check and padding is None:
            padding = ir.PADDING_OPTION.PAD_ZERO

    # Build IR
        return self.tensor(
            self.builder.create_tensor_pointer_load(ptr.handle, boundary_check, padding, cache, eviction, is_volatile),
            dst_ty)

    def _load_legacy(self, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, care_padding):
        # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        if not ptr.type.scalar.is_ptr():
            raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

        # Check `mask`, `other`, `boundary_check`, and `padding` arguments
        if mask is None and other is not None:
            raise ValueError("`other` cannot be provided without `mask`")
        if padding or boundary_check:
            raise ValueError("`padding_option` or `boundary_check` argument is not supported for loading a tensor of"
                             "pointers or loading a scalar. Because the compiler does not know the boundary; please "
                             "use block pointers (defined by `make_block_ptr`) instead")

        if mask is not None and other is None and care_padding == True:
            # Get element type to determine default padding value
            elt_ty = ptr.type.scalar.element_ty
            # Use 0.0 for floating point types, 0 for integer types
            default_value = 0.0 if elt_ty.is_floating() else 0
            other = self.to_tensor(default_value)
        # For a pointer of scalar, check the type of `mask` and `other`
        if not ptr.type.is_block():
            if mask and mask.type.is_block():
                raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
            if other and other.type.is_block():
                raise ValueError("Other argument cannot be block type if pointer argument is not a block")

        # Make `mask` and `other` into the same shape as `ptr`
        if ptr.type.is_block():
            if mask is not None:
                ptr, mask = self.broadcast_impl_value(ptr, mask)
            if other is not None:
                ptr, other = self.broadcast_impl_value(ptr, other)

        # Get `pointer_type<elt_ty>` and `elt_ty`
        ptr_ty = ptr.type.scalar
        elt_ty = ptr_ty.element_ty

        # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
        is_bool = elt_ty == tl.int1
        if is_bool:
            elt_ty = tl.int8
            ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
            ptr = self.cast(ptr, ptr_ty)

        # Cast `other` into `elt_ty` type
        if other is not None:
            other = self.cast(other, elt_ty)

        # Create loaded result type `dst_ty`
        if ptr.type.is_block():
            dst_ty = ptr.type.with_element_ty(elt_ty)
        else:
            # Load by de-referencing the pointer of scalar
            dst_ty = elt_ty

        # Build IR
        if mask is None:
            load_handle = self.builder.create_load(ptr.handle, cache, eviction, is_volatile)
        else:
            load_handle = self.builder.create_masked_load(
                ptr.handle, mask.handle, other.handle if other else None, cache, eviction, is_volatile
            )

        if is_bool:
            load_handle.set_attr("was_bool_to_int8", self.builder.get_bool_attr(True))

        ret = self.tensor(load_handle, dst_ty)
        # Do not cast back to int1 when is_bool=true. We directly use the int8 tensor given by tl.load
        if is_bool:
            ret.was_bool_to_int8 = True

        return ret

    def load(self, ptr: TensorTy, mask: Optional[TensorTy], other: Optional[TensorTy], boundary_check: Tuple,
             padding_option: str, cache_modifier: str, eviction_policy: str, is_volatile: bool, care_padding: bool) -> TensorTy:
        # Cache, eviction and padding options
        cache = self._str_to_load_cache_modifier(cache_modifier)
        eviction = self._str_to_eviction_policy(eviction_policy)
        padding = self._str_to_padding_option(padding_option)

        if ptr.type.is_ptr() and ptr.type.element_ty.is_block():
            # Load by a block pointer: `pointer_type<block_type<>>`
            return self._load_block_pointer(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile)
        else:
            # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
            return self._load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, care_padding)

    def atomic_cas(self, ptr: TensorTy, cmp: TensorTy, val: TensorTy, sem: str, scope: str) -> TensorTy:
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        element_ty = ptr.type.scalar.element_ty
        if not is_compile_on_910_95:
            supported_types = [tl.int8, tl.uint8, tl.int16, tl.int32, tl.int64, tl.float16, tl.bfloat16, tl.float32]
            if element_ty not in supported_types:
                raise ValueError(f"atomic_cas does not support {str(element_ty)}. "
                                "All support dtypes are int8, uint8, int16, int32, int64, float16, bfloat16, float32.")
        else:
            unsupported_types = [tl.int1]
            if element_ty in unsupported_types:
                raise ValueError(f"atomic_cas does not support {str(element_ty)}. "
                                "All support dtypes are int8, uint8, int16, uint16, int32, uint32, int64, uint64, fp8e4m3, fp8e5m2, float16, bfloat16, float32.")
        return self.tensor(self.builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle, sem, scope), val.type)

    def atom_red_typechecking_impl(self, ptr: TensorTy, val: TensorTy, mask: TensorTy,
                                   op: str) -> Tuple[TensorTy, TensorTy, TensorTy]:
        if not ptr.type.scalar.is_ptr():
            raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())
        if ptr.type.is_const() or ptr.type.element_ty.is_const():
            raise ValueError("Cannot store to a constant pointer")
        if ptr.type.is_block():
            if mask is not None:
                mask = self.broadcast_impl_shape(mask, ptr.type.get_block_shapes())
            if val is not None:
                val = self.broadcast_impl_shape(val, ptr.type.get_block_shapes())
        val = self.cast(val, ptr.type.scalar.element_ty)
        if mask is None:
            mask_ir = self.builder.get_int1(True)
            mask_ty = tl.int1
            if ptr.type.is_block():
                mask_ty = ptr.type.with_element_ty(tl.int1)
                mask_ir = self.builder.create_splat(mask_ty.to_ir(self.builder), mask_ir)
            mask = self.tensor(mask_ir, mask_ty)
        return ptr, val, mask

    def atomic_max(self, ptr: TensorTy, val: TensorTy, mask: TensorTy, sem: str, scope: str) -> TensorTy:
        ptr, val, mask = self.atom_red_typechecking_impl(ptr, val, mask, 'max')
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        sca_ty = val.type.scalar
        # direct call to atomic_max for integers
        if sca_ty.is_int():
            if sca_ty.is_int_signed():
                return self.tensor(
                    self.builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope),
                    val.type)
            else:
                return self.tensor(
                    self.builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX, ptr.handle, val.handle, mask.handle, sem, scope),
                    val.type)
        # Design for NPU
        return self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

    def atomic_min(self, ptr: TensorTy, val: TensorTy, mask: TensorTy, sem: str, scope: str) -> TensorTy:
        ptr, val, mask = self.atom_red_typechecking_impl(ptr, val, mask, 'min')
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        sca_ty = val.type.scalar
        # direct call to atomic_min for integers
        if sca_ty.is_int():
            if sca_ty.is_int_signed():
                return self.tensor(
                    self.builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope),
                    val.type)
            else:
                return self.tensor(
                    self.builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, ptr.handle, val.handle, mask.handle, sem, scope),
                    val.type)
        # Design for NPU
        return self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

    def dot(self, lhs: TensorTy, rhs: TensorTy, acc: TensorTy, input_precision: Optional[str],
            max_num_imprecise_acc: int, out_dtype: tl.dtype) -> TensorTy:
        assert lhs.type.is_block() and rhs.type.is_block()

        if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
            # All combinations of supported fp8 x fp8 are permitted
            pass
        else:
            assert lhs.dtype in (tl.int1, tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                                 tl.float64), f"Unsupported lhs dtype {lhs.dtype}"
            assert rhs.dtype in (tl.int1, tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                                 tl.float64), f"Unsupported rhs dtype {rhs.dtype}"
            assert lhs.dtype == rhs.dtype, f"Both operands must be same dtype. Got {lhs.dtype} and {rhs.dtype}"

        if lhs.dtype.is_fp8e4b15() or rhs.dtype.is_fp8e4b15():
            if "fp8e4b15" in self.builder.options.deprecated_fp8_dot_operand_dtypes:
                warnings.warn(
                    "the use of fp8e4b15 is deprecated on Hopper and later architectures and can cause significant slow down. It will be removed in a future triton release"
                )
            # We upcast because there's no fp8e4b15 type in MLIR
            lhs = self.cast(lhs, tl.float16)
            rhs = self.cast(rhs, tl.float16)

        uses_fp8e4b8 = lhs.dtype.is_fp8e4b8() or rhs.dtype.is_fp8e4b8()
        uses_fp8e5b16 = lhs.dtype.is_fp8e5b16() or rhs.dtype.is_fp8e5b16()
        if uses_fp8e4b8 or uses_fp8e5b16:
            type_name = "fp8e4b8" if uses_fp8e4b8 else "fp8e5b16"
            if type_name in self.builder.options.deprecated_fp8_dot_operand_dtypes:
                arch = self.builder.options.arch
                warnings.warn(
                    f"{type_name} is AMD gfx942 specific and not supported on {arch} so it's upcasted to fp16 and can cause significant slow down. "
                    f"Please use OCP fp8 variants on {arch} for performance")
                lhs = self.cast(lhs, tl.float16)
                rhs = self.cast(rhs, tl.float16)

        if input_precision is None:
            input_precision = self.builder.options.default_dot_input_precision

        input_precision = self._str_to_dot_input_precision(input_precision)

        lhs_rank = len(lhs.shape)
        rhs_rank = len(rhs.shape)
        assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
        assert lhs.shape[-1].value == rhs.shape[
            -2].value, f"First input shape ({lhs.shape}) and second input shape {rhs.shape} are not compatible for matmul (second index of first shape ({lhs.shape[-1].value}) must be equal to first index of second shape ({rhs.shape[-2].value})"
        assert self.builder.codegen_fns.get(
            "min_dot_size") is not None, "target doesn't provide lower shape bounds for dot."
        min_dot_size = self.builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
        assert lhs.shape[-2].value >= min_dot_size[0] and lhs.shape[-1].value >= min_dot_size[2] \
            and rhs.shape[-1].value >= min_dot_size[1], \
                f"Input shapes should have M >= {min_dot_size[0]}, N >= {min_dot_size[1]} and K >= {min_dot_size[2]}"
        if lhs.type.scalar.is_int():
            assert lhs.type.scalar == tl.int8, "only int8 supported!"
            _0 = self.builder.get_int32(0)
            ret_scalar_ty = tl.int32
        elif out_dtype.is_bf16():
            raise ValueError(
                "out_dtype=bfloat16 is unsupported. Please use out_dtype=float32/float16 and cast with `.to(tl.bfloat16)`"
            )
        elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
            _0 = self.builder.get_fp32(0)
            ret_scalar_ty = tl.float32
        elif lhs.type.scalar.is_fp64():
            _0 = self.builder.get_fp64(0)
            ret_scalar_ty = tl.float64
        else:
            _0 = self.builder.get_fp16(0) if out_dtype.is_fp16() else self.builder.get_fp32(0)
            ret_scalar_ty = out_dtype

        M = lhs.type.shape[-2]
        N = rhs.type.shape[-1]
        K = lhs.type.shape[-1]
        B = lhs.type.shape[0] if lhs_rank == 3 else None
        ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])
        if acc is None:
            acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)
        else:
            acc_handle = acc.handle
            assert acc.type.shape == ret_ty.shape and acc.type.element_ty == out_dtype

        if (input_precision == getattr(ir.INPUT_PRECISION, "HF32")):
            if (not lhs.dtype.is_fp32() or not rhs.dtype.is_fp32() or not ret_scalar_ty.is_fp32()):
                # when input and result is not fp32, ignore input_precision (default is ieee)
                input_precision = self._str_to_dot_input_precision(self.builder.options.default_dot_input_precision)

        if max_num_imprecise_acc is not None:
            print("max_num_imprecise_acc in tl.dot is not supported on Ascend yet. Thus it is ignored.")
        max_num_imprecise_acc = 0
        return self.tensor(
            self.builder.create_dot(lhs.handle, rhs.handle, acc_handle, input_precision, max_num_imprecise_acc), ret_ty)

    def dot_scaled(self, lhs: TensorTy, lhs_scale: TensorTy, lhs_format: str, rhs: TensorTy,
                   rhs_scale: Optional[TensorTy], rhs_format: str, acc: TensorTy | None, fast_math: bool,
                   lhs_k_pack: bool, rhs_k_pack: bool, out_dtype: tl.dtype) -> TensorTy:
        assert lhs.type.is_block() and rhs.type.is_block()
        if is_compile_on_910_95:
            assert lhs.dtype in [tl.float16, tl.bfloat16, tl.uint8, tl.float8e5, tl.float8e4nv], f"lhs matrix dtype must be in [bf16, fp16, uint8, e5m2, e4m3]"
            assert rhs.dtype in [tl.float16, tl.bfloat16, tl.uint8, tl.float8e5, tl.float8e4nv], f"rhs matrix dtype must be in [bf16, fp16, uint8, e5m2, e4m3]"
        else:
            assert lhs.dtype == tl.bfloat16 or lhs.dtype == tl.float16, f"lhs matrix dtype must be bf16 or fp16"
            assert rhs.dtype == tl.bfloat16 or lhs.dtype == tl.float16, f"rhs matrix dtype must be bf16 or fp16"
        assert lhs.dtype == rhs.dtype, f"lhs rhs matrix must get same dtype"
        #TODO: validate types.
        lhs_rank = len(lhs.shape)
        rhs_rank = len(rhs.shape)
        assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
        lhs_format: str = lhs_format.value
        rhs_format: str = rhs_format.value
        lhs_format_enum = self._str_to_fp_type(lhs_format)
        rhs_format_enum = self._str_to_fp_type(rhs_format)
        if is_compile_on_910_95:
            allowed_formats = {"bf16", "fp16", "e4m3", "e5m2", "e2m1"}
        else:
            allowed_formats = {"bf16", "fp16"}  # unsupported fp8/4 dtype: "e2m1", "e4m3", "e5m2"
        assert lhs_format in allowed_formats, f"NYI: lhs_format {lhs_format}"
        assert rhs_format in allowed_formats, f"NYI: rhs_format {rhs_format}"
        rhs_scale_is_none = rhs_scale is None or (isinstance(rhs_scale, tl.constexpr) and rhs_scale.value is None)
        lhs_scale_is_none = lhs_scale is None or (isinstance(lhs_scale, tl.constexpr) and lhs_scale.value is None)
        assert isinstance(lhs_scale, tl.tensor) and (lhs_scale.dtype == tl.int8 or lhs_scale.dtype == tl.uint8), f"lhs_scale must be int8 or uint8 tensor"
        if not rhs_scale_is_none:
            assert isinstance(rhs_scale, tl.tensor) and (rhs_scale.dtype == tl.int8 or rhs_scale.dtype == tl.uint8), f"rhs_scale must be int8 or uint8 tensor"
        lhs = self._bitcast_to_fp_type(lhs, lhs_format)
        rhs = self._bitcast_to_fp_type(rhs, rhs_format)
        assert lhs_k_pack or lhs_format == "e2m1", "only mxfp4 inputs can be packed along a dimension different than K"
        assert rhs_k_pack or rhs_format == "e2m1", "only mxfp4 inputs can be packed along a dimension different than K"

        lhs_k_pack_v = lhs_k_pack.value if isinstance(lhs_k_pack, tl.constexpr) else lhs_k_pack
        rhs_k_pack_v = rhs_k_pack.value if isinstance(rhs_k_pack, tl.constexpr) else rhs_k_pack

        if lhs_k_pack_v is False:
            dims = (1, 0)
            tmp_lhs = self.permute(lhs, dims)
            lhs = self.reshape(tmp_lhs, (lhs.shape[0], lhs.shape[1]), True)

        if rhs_k_pack_v is False:
            dims = (1, 0)
            tmp_rhs = self.permute(rhs, dims)
            rhs = self.reshape(tmp_rhs, (rhs.shape[0], rhs.shape[1]), True)

        assert lhs.type.shape[-1] == rhs.type.shape[-2], (
            f"lhs last dimension (columns) {lhs.shape[-1]} "
            f"must equal rhs penultimate dimension (rows) {rhs.shape[-2]}"
        )

        assert lhs_k_pack or lhs_format == "e2m1", "only mxfp4 inputs can be packed along a dimension different than K"
        assert rhs_k_pack or rhs_format == "e2m1", "only mxfp4 inputs can be packed along a dimension different than K"
        M, K_LHS = lhs.type.shape[-2:]
        K_RHS, N = rhs.type.shape[-2:]
        PACKED_A = 2 if lhs_format == "e2m1" else 1
        PACKED_B = 2 if rhs_format == "e2m1" else 1
        PACKED_A_DIM = PACKED_A * K_LHS if lhs_k_pack else K_LHS
        PACKED_B_DIM = PACKED_B * K_RHS if rhs_k_pack else K_RHS
        assert PACKED_B_DIM == PACKED_A_DIM, f"Reduction dimension should pack the same number of elements; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
        #assert K * PACKED_B >= 64, f"scaled_dot NYI for K < 64. Got {K=}"
        B = lhs.type.shape[0] if lhs_rank == 3 else None
        if not lhs_k_pack:
            M = M * PACKED_A
        if not rhs_k_pack:
            N = N * PACKED_B
        ret_ty = tl.block_type(out_dtype, [B, M, N] if B else [M, N])
        _0 = self.builder.get_fp32(0)
        if acc is None:
            acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)
        else:
            acc_handle = acc.handle
            assert acc.type.shape == ret_ty.shape and acc.type.element_ty == out_dtype
        rhs_scale_handle = None if rhs_scale_is_none else rhs_scale.handle
        lhs_scale_handle = None if lhs_scale_is_none else lhs_scale.handle
        return self.tensor(
            self.builder.create_dot_scaled(lhs.handle, lhs_scale_handle, lhs_format_enum, rhs.handle, rhs_scale_handle,
                                           rhs_format_enum, fast_math, lhs_k_pack, rhs_k_pack, acc_handle), ret_ty)

    def gather(self, src: TensorTy, index: TensorTy, axis: int) -> TensorTy:
        assert index.dtype.is_int(), "index must be an integer tensor"
        if not (src.dtype.is_floating() or src.dtype.is_int8()):
            raise ValueError(f"Expected dtype fp16/fp32/bf16/f8E5M2/f8E4M3FN/int8, but got {src.dtype}")

        rank = len(src.type.shape)
        assert len(index.type.shape) == rank, "source and index tensors must have the same rank"

        assert -rank <= axis < rank, f"gather axis {axis} must be < source rank ({rank})"
        if axis < 0:
            axis += rank

        for d in range(rank):
            if d == axis:
                continue
            assert index.type.shape[d] == src.type.shape[d], f"index dim {axis} must match the corresponding source dim"

        gather = self.builder.create_gather(src.handle, index.handle, axis)
        return self.wrap_tensor(gather, src.type.scalar, index.type.shape)