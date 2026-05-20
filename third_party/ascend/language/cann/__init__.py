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

import os

from triton.tools.get_ascend_devices import is_compile_on_910_95
import triton.language.core as tl_core
import triton.language.math as tl_math
import triton.language.standard as tl_standard
from triton.backends.ascend.utils import triton_enable_libdevice_simt

from . import libdevice
from . import extension
from .language_patch import patch_language_modules
from triton.runtime.ascend_interpreter import patch_ascend_interpreter

# def _patch_ascend_interpreter():
#     if not os.getenv("TRITON_INTERPRET"):
#         return

#     try:
#         from triton.runtime import ascend_interpreter
#     except ImportError:
#         return


patch_ascend_interpreter()
patch_language_modules(tl_core, tl_math, tl_standard)

extension.parallel = extension.aux_ops.parallel
if not triton_enable_libdevice_simt():
    libdevice.atan2 = extension.math_ops.atan2
libdevice.isfinited = extension.math_ops.isfinited
libdevice.finitef = extension.math_ops.finitef
libdevice.flip = extension.flip

libdevice.umulhi = tl_math.umulhi
libdevice.exp = tl_math.exp
libdevice.exp2 = tl_math.exp2
libdevice.log = tl_math.log
libdevice.log2 = tl_math.log2
libdevice.cos = tl_math.cos
libdevice.sin = tl_math.sin
libdevice.sqrt = tl_math.sqrt
libdevice.sqrt_rn = tl_math.sqrt_rn
libdevice.rsqrt = tl_math.rsqrt
libdevice.div_rn = tl_math.div_rn
libdevice.erf = tl_math.erf
libdevice.floor = tl_math.floor
libdevice.ceil = tl_math.ceil
libdevice.fdiv = tl_math.fdiv
libdevice.fma = tl_math.fma
libdevice.abs = tl_math.abs
tl_math.tanh = libdevice.tanh

__all__ = ["libdevice", "extension"]
