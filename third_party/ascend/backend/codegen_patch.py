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


def patch_ast_to_ttir(code_generator_module):
    original_ast_to_ttir = code_generator_module.ast_to_ttir
    if getattr(original_ast_to_ttir, "_ascend_patch_applied", False):
        return

    from triton.compiler.ascend_code_generator import AscendCodeGenerator as _ascend_code_generator_cls

    original_ast_to_ttir.__globals__["AscendCodeGenerator"] = _ascend_code_generator_cls

    def _ascend_ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
        arg_types = [None] * len(fn.arg_names)
        const_iter = iter(src.constants.items())
        kc, vc = next(const_iter, (None, None))

        for i, (ks, v) in enumerate(src.signature.items()):
            idx = fn.arg_names.index(ks)
            cexpr = None
            if kc is not None and kc[0] == i:
                cexpr = vc
                kc, vc = next(const_iter, (None, None))
            arg_types[idx] = str_to_ty(v, cexpr)
        prototype = ASTFunction([], arg_types, src.constants, src.attrs)
        file_name, begin_line = get_jit_fn_file_line(fn)
        # query function representation
        from collections import namedtuple
        leaves = filter(lambda v: len(v) == 1, src.constants)
        constants = {fn.arg_names[i[0]]: src.constants[i] for i in leaves}
        signature = src.signature
        proxy = namedtuple("SpecializationProxy", ["constants", "signature"])(constants, signature)
        generator = AscendCodeGenerator(context, prototype, gscope=fn.get_capture_scope(), function_name=fn.repr(proxy),
                              jit_fn=fn, is_kernel=True, file_name=file_name, begin_line=begin_line, options=options,
                              codegen_fns=codegen_fns, module_map=module_map, module=module, is_gluon=fn.is_gluon())
        generator.visit(fn.parse())
        module = generator.module
        # module takes ownership of the context
        module.context = context
        if not module.verify_with_diagnostics():
            if not fn.is_gluon():
                print(module)
            raise RuntimeError("error encountered during parsing")
        return module

    original_ast_to_ttir.__code__ = _ascend_ast_to_ttir.__code__
    original_ast_to_ttir.__defaults__ = _ascend_ast_to_ttir.__defaults__
    original_ast_to_ttir.__kwdefaults__ = _ascend_ast_to_ttir.__kwdefaults__
    original_ast_to_ttir._ascend_patch_applied = True
