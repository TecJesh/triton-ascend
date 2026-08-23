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

import logging
from triton._C.libtriton.ascend import ir as ascend_ir

from .testing import do_bench_npu


def _apply_ascend_patch():
    from triton.compiler.code_generator import CodeGenerator

    if not getattr(CodeGenerator, "_ascend_patch_applied", False):
        _original_cg_init = CodeGenerator.__init__

        def _patched_cg_init(self, *args, **kwargs):
            """
            Monkey Patch for Ascend:
            Injects 'hacc.target' attribute into the module after initilization of module.
            """
            _original_cg_init(self, *args, **kwargs)
            options = self.builder.options
            context = self.context
            if hasattr(options, "arch") and options.arch:
                try:
                    builder = ascend_ir.ascendnpu_ir_builder(context, options.arch)

                    target_attr_str = f'#hacc.target<"{options.arch}">'
                    self.module.set_attr("hacc.target", builder.parse_attr(target_attr_str))
                except Exception as e:
                    logging.warning(f"[Ascend Patch] Failed to set hacc.target: {e}")

        CodeGenerator.__init__ = _patched_cg_init
        CodeGenerator._ascend_patch_applied = True

    # Patch compiler.parse to support Ascend-specific IR extensions
    # for ir_override and TRITON_KERNEL_OVERRIDE features.
    from triton.compiler import compiler as _compiler_module

    if not getattr(_compiler_module, "_ascend_parse_patch_applied", False):
        from pathlib import Path as _Path
        from triton._C.libtriton import ir as _ir

        # Keep the original community logic intact, adding Ascend extensions
        # to the existing text/binary branches.
        def _patched_parse(full_name, ext, context):
            if ext == "ttir" or ext == "ttgir":
                module = _ir.parse_mlir_module(full_name, context)
                module.context = context
                return module
            if ext in ("llir", "ptx", "amdgcn", "ttadapter", "bcmlir"):
                return _Path(full_name).read_text()
            if ext in ("cubin", "hsaco", "mlirbc", "npubin"):
                return _Path(full_name).read_bytes()

        _compiler_module.parse = _patched_parse
        _compiler_module._ascend_parse_patch_applied = True

    # Patch TritonSemantic.dot for Ascend-specific HF32 guard and
    # max_num_imprecise_acc warning.
    from triton.language.semantic import TritonSemantic

    if not getattr(TritonSemantic, "_ascend_dot_patch_applied", False):
        _original_dot = TritonSemantic.dot

        def _patched_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype):
            """
            Monkey Patch for Ascend:
            - HF32 precision only works for fp32 x fp32.
              When either input is not fp32, silently fall back
              to default precision (ieee).
            - Warn when max_num_imprecise_acc is explicitly set, since
              Ascend NPU does not support imprecise accumulation.
            """
            # Ascend does not support tf32, tf32 is converted to hf32. Avoid tf32 verification interception
            if input_precision == "tf32":
                input_precision = "hf32"
            # Ascend NPU does not support imprecise accumulation.
            # Force max_num_imprecise_acc to None so the upstream None
            # branch handles it (via max_num_imprecise_acc_default = 0),
            # avoiding the fp8 ValueError path which is NVIDIA-only.
            if max_num_imprecise_acc is not None:
                print("max_num_imprecise_acc in tl.dot is not supported on Ascend yet. "
                      "Thus it is ignored.")
                max_num_imprecise_acc = None

            return _original_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype)

        TritonSemantic.dot = _patched_dot
        TritonSemantic._ascend_dot_patch_applied = True

    # Patch TritonSemantic.histogram: upstream now enforces a 32-bit integer
    # input contract (triton-lang/triton#10891), but the Ascend backend also
    # supports 64-bit integer histogram inputs (bin indices are narrowed to
    # i32 during lowering).  Restore the pre-#10891 "any integer dtype"
    # behavior for Ascend so int64/uint64 histograms keep working.
    if not getattr(TritonSemantic, "_ascend_histogram_patch_applied", False):
        _original_histogram = TritonSemantic.histogram

        def _patched_histogram(self, input, num_bins, mask):
            if input.dtype.is_int() and input.dtype.int_bitwidth != 32:
                # Pre-#10891 body, kept for Ascend-only wide integer inputs.
                import triton.language as tl

                assert len(input.shape) == 1, "histogram only supports 1D input"
                if mask is not None:
                    mask = self.broadcast_impl_shape(mask, input.shape)
                    if not mask.type.scalar.is_bool():
                        raise ValueError("Mask must have boolean scalar type")
                    mask = mask.handle
                return self.tensor(self.builder.create_histogram(input.handle, num_bins, mask),
                                   tl.block_type(tl.int32, [num_bins]))
            return _original_histogram(self, input, num_bins, mask)

        TritonSemantic.histogram = _patched_histogram
        TritonSemantic._ascend_histogram_patch_applied = True


__all__ = ["do_bench_npu"]
