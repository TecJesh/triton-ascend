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


from . import core as cann_core
from . import math as cann_math
from . import standard as cann_standard


def patch_language_modules(core_module, math_module, standard_module):
    core_module.tensor.__add__ = cann_core.__add__
    core_module.tensor.__radd__ = cann_core.__radd__
    core_module.tensor.__sub__ = cann_core.__sub__
    core_module.tensor.__rsub__ = cann_core.__rsub__
    core_module.tensor.__mul__ = cann_core.__mul__
    core_module.tensor.__rmul__ = cann_core.__rmul__
    core_module.tensor.to = cann_core.to
    core_module.tensor.cast = cann_core.cast
    core_module.view = cann_core.view
    core_module.cast = cann_core.cast
    core_module.dot = cann_core.dot
    core_module.load = cann_core.load
    core_module.reduce = cann_core.reduce
    core_module.dot_scaled = cann_core.dot_scaled

    math_module.exp = cann_math.exp
    math_module.exp2 = cann_math.exp2
    math_module.log = cann_math.log
    math_module.log2 = cann_math.log2
    math_module.cos = cann_math.cos
    math_module.sin = cann_math.sin
    math_module.sqrt = cann_math.sqrt
    math_module.sqrt_rn = cann_math.sqrt_rn
    math_module.rsqrt = cann_math.rsqrt
    math_module.div_rn = cann_math.div_rn
    math_module.erf = cann_math.erf
    math_module.floor = cann_math.floor
    math_module.ceil = cann_math.ceil

    standard_module.cdiv = cann_standard.cdiv
    standard_module.max = cann_standard.max
    standard_module.min = cann_standard.min
    standard_module.sum = cann_standard.sum