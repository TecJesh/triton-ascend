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

import ast
import builtins
import inspect
from dataclasses import dataclass
from types import ModuleType

from typing import Any, Callable, Dict, Optional

# Import and register Ascend extension dispatch handlers
import triton.language.extra.cann.extension as extension
from triton.language.extra.cann.extension.dispatch import ASCEND_WITH_DISPATCH
from triton.language.extra.cann.extension.builder import setup_unified_builder

from triton.language.extra.extension.buffer.language.builder import setup_unified_builder_with_buffer_builder

from triton import knobs, language
from triton._C.libtriton import ir, gluon_ir, buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.language import constexpr, str_to_ty, tensor
from triton.language.core import _unwrap_if_constexpr
# ideally we wouldn't need any runtime component
from triton.runtime.jit import get_jit_fn_file_line, get_full_name, BoundConstexprFunction, ConstexprFunction, JITFunction

from triton.compiler.errors import (CompilationError)
from triton._utils import find_paths_if, get_iterable_path, set_iterable_path
from triton.compiler.code_generator import CodeGenerator, ContainsReturnChecker, enter_sub_region, ASTFunction, BoundJITMethod, \
    check_identifier_legality, mangle_fn, _check_fn_args, _is_triton_value, _is_constexpr, _apply_to_tuple_values, \
    flatten_values_to_ir, unflatten_ir_values, _is_non_scalar_tensor

# Central registry for all 'with' statement handlers
WITH_DISPATCH = {}
WITH_DISPATCH.update(ASCEND_WITH_DISPATCH)


class AscendCodeGenerator(CodeGenerator):

    def __init__(self, context, prototype, gscope, function_name, jit_fn: JITFunction, *, options, codegen_fns,
                 module_map, is_gluon, module=None, is_kernel=False, function_types: Optional[Dict] = None,
                 noinline=False, caller_context=None, file_name: Optional[str] = None, begin_line=0):
        self.context = context
        self.is_gluon = is_gluon
        if is_gluon:
            from triton.experimental.gluon.language._semantic import GluonSemantic
            self.builder = gluon_ir.GluonOpBuilder(context)
            self.semantic = GluonSemantic(self.builder)
        else:
            from triton.language.semantic import TritonSemantic
            # Only NPUOptions has force_simt_only attribute, so check for NPU backend
            compile_mode = "simt" if (hasattr(options, "force_simt_only") and options.force_simt_only) else "simd"
            self.builder = ir.builder(context)
            self.semantic = TritonSemantic(self.builder)

        self.name_loc_as_prefix = None
        self.file_name = file_name
        # node.lineno starts from 1, so we need to subtract 1
        self.begin_line = begin_line - 1
        self.builder.set_loc(file_name, begin_line, 0)
        self.builder.options = options

        # Set up unified builder interface with methods from specialized builders
        self.ascend_builder = ascend_ir.ascendnpu_ir_builder(context, getattr(options, "arch", ""),
                                                             compile_mode=compile_mode)
        self.ascend_builder.set_loc(file_name, begin_line, 0)
        setup_unified_builder(self.builder, self.ascend_builder)
        self.buffer_builder = buffer_ir.buffer_builder(context)
        self.buffer_builder.set_loc(file_name, begin_line, 0)
        setup_unified_builder_with_buffer_builder(self.builder, self.buffer_builder)

        # dict of functions provided by the backend. Below are the list of possible functions:
        # Convert custom types not natively supported on HW.
        # convert_custom_types(input_tensor, dtype, fp_downcast_rounding=None, _builder=None)
        self.builder.codegen_fns = codegen_fns
        self.builder.module_map = {} if module_map is None else module_map
        self.module = self.builder.create_module() if module is None else module
        self.function_ret_types = {} if function_types is None else function_types
        self.prototype = prototype

        self.gscope = {}
        for k, v in gscope.items():
            if isinstance(v, ModuleType):
                self.gscope[k] = module_map.get(v.__name__, v)
                continue

            module_name = getattr(v, "__module__", "")
            if module_name in module_map:
                self.gscope[k] = getattr(module_map[module_name], v.__name__)
            else:
                self.gscope[k] = v

        self.lscope = {}
        self.jit_fn = jit_fn
        # TODO: we currently generate illegal names for non-kernel functions involving constexprs!
        if is_kernel:
            function_name = function_name[function_name.rfind('.') + 1:]
            function_name = check_identifier_legality(function_name, "function")
        self.function_name = function_name
        self.is_kernel = is_kernel
        self.cur_node = None
        self.noinline = noinline
        self.caller_context = caller_context
        self.scf_stack = []
        self.ret_type = None
        # SSA-construction
        # name => language.tensor
        self.local_defs: Dict[str, tensor] = {}
        self.dereference_name: Callable[[str], Any] = self._define_name_lookup()
        self.fn = None
        # Are we currently visiting an ast.arg's default value?  These have some
        # special handling.
        self.visiting_arg_default_value = False

    def _get_insertion_point_and_loc(self, builder=None):
        # XXX: this is a hack to get the location of the insertion point.
        # The insertion point's location could be invalid sometimes,
        # so we need to explicitly set the location
        _builder = self.builder if not builder else builder
        loc = _builder.get_loc()
        ip = _builder.get_insertion_point()
        return ip, loc

    def _set_insertion_point_and_loc(self, ip, loc, builder=None):
        _builder = self.builder if not builder else builder
        _builder.restore_insertion_point(ip)
        _builder.set_loc(loc)

    def visit_With(self, node):
        """
        Handle 'with' statements with dispatch pattern for Ascend extensions,
        falling back to standard context manager protocol for general cases.

        This implementation:
        1. First tries dispatch mechanism for Ascend-specific context managers (e.g., scope)
        2. Falls back to standard Python context manager protocol for general cases
        """
        # Try dispatch mechanism for Ascend-specific context managers
        # Only attempt dispatch for single context manager with Call expression
        if len(node.items) == 1:
            context = node.items[0].context_expr
            if isinstance(context, ast.Call):
                withitemClass = self.visit(context.func)
                handler = WITH_DISPATCH.get(withitemClass)
                if handler:
                    # Dispatch to registered handler (e.g., handle_scope_with)
                    return handler(self, node)

        # Fall back to standard context manager protocol (community logic)
        # Lower `with` statements by constructing context managers and calling their enter/exit hooks
        # Instantiate each context manager with builder injection
        cm_list = []
        for item in node.items:
            call = item.context_expr
            fn = self.visit(call.func)
            args = [self.visit(arg) for arg in call.args]
            kws = dict(self.visit(kw) for kw in call.keywords)
            cm = fn(*args, _semantic=self.semantic, **kws)
            cm_list.append(cm)
        for cm, item in zip(cm_list, node.items):
            res = cm.__enter__()
            if item.optional_vars is not None:
                var_name = self.visit(item.optional_vars)
                self.set_value(var_name, res)
        if ContainsReturnChecker(self.gscope).visit(node):
            raise self._unsupported(node, "Cannot have `return` statements inside `with` statements in triton ")
        self.visit_compound_statement(node.body)
        for cm in reversed(cm_list):
            cm.__exit__(None, None, None)

    def visit_For(self, node):
        IteratorClass = self.visit(node.iter.func)
        iter_args = [self.visit(arg) for arg in node.iter.args]
        iter_kwargs = dict(self.visit(keyword) for keyword in node.iter.keywords)
        if IteratorClass == language.static_range:
            iterator = IteratorClass(*iter_args, **iter_kwargs)
            static_range = range(iterator.start.value, iterator.end.value, iterator.step.value)
            for i in static_range:
                self.lscope[node.target.id] = constexpr(i)
                self.visit_compound_statement(node.body)
                for stmt in node.orelse:
                    ast.NodeVisitor.generic_visit(self, stmt)
            return
        num_stages = None
        loop_unroll_factor = None
        disallow_acc_multi_buffer = False
        flatten = False
        warp_specialize = False
        disable_licm = False
        if IteratorClass in [language.range, extension.parallel]:
            iterator = IteratorClass(*iter_args, **iter_kwargs)
            # visit iterator arguments
            # note: only `range` iterator is supported now
            # collect lower bound (lb), upper bound (ub), and step
            lb = iterator.start
            ub = iterator.end
            step = iterator.step
            num_stages = iterator.num_stages
            loop_unroll_factor = iterator.loop_unroll_factor
            disallow_acc_multi_buffer = iterator.disallow_acc_multi_buffer
            flatten = iterator.flatten
            warp_specialize = iterator.warp_specialize
            disable_licm = iterator.disable_licm
        elif IteratorClass is range:
            # visit iterator arguments
            # note: only `range` iterator is supported now
            # collect lower bound (lb), upper bound (ub), and step
            lb = iter_args[0] if len(iter_args) > 1 else self.visit(ast.Constant(0))
            ub = iter_args[1] if len(iter_args) > 1 else self.visit(node.iter.args[0])
            step = iter_args[2] if len(iter_args) > 2 else self.visit(ast.Constant(1))
        else:
            raise RuntimeError('Only `range` and `static_range` iterators are currently supported')
        # handle negative constant step (not supported by scf.for in MLIR)
        negative_step = False
        if _is_constexpr(step) and step.value < 0:
            step = constexpr(-step.value)
            negative_step = True
            lb, ub = ub, lb
        lb = self.semantic.to_tensor(lb)
        ub = self.semantic.to_tensor(ub)
        step = self.semantic.to_tensor(step)
        # induction variable type
        if not lb.dtype.is_int() or not ub.dtype.is_int() or not step.dtype.is_int():
            raise TypeError(f"For loop bounds and step must all be ints, are ({lb.dtype}, {ub.dtype}, {step.dtype})")
        if _is_non_scalar_tensor(lb):
            raise TypeError(f"For lower bound must be a scalar, got {lb.type}")
        if _is_non_scalar_tensor(ub):
            raise TypeError(f"For upper bound must be a scalar, got {ub.type}")
        if _is_non_scalar_tensor(step):
            raise TypeError(f"For step must be a scalar, got {step.type}")
        iv_type = self.semantic.integer_promote_impl(lb.dtype, ub.dtype)
        iv_type = self.semantic.integer_promote_impl(iv_type, step.dtype)
        iv_ir_type = iv_type.to_ir(self.builder)
        iv_is_signed = iv_type.int_signedness == language.core.dtype.SIGNEDNESS.SIGNED
        # lb/ub/step might be constexpr, we need to cast them to tensor
        lb = lb.handle
        ub = ub.handle
        step = step.handle
        # ForOp can only accept IndexType as lb/ub/step. Cast integer to Index
        lb = self.builder.create_int_cast(lb, iv_ir_type, iv_is_signed)
        ub = self.builder.create_int_cast(ub, iv_ir_type, iv_is_signed)
        step = self.builder.create_int_cast(step, iv_ir_type, iv_is_signed)
        # Create placeholder for the loop induction variable
        iv_placeholder = self.builder.create_poison(iv_ir_type)
        self.set_value(node.target.id, language.core.tensor(iv_placeholder, iv_type))

        with enter_sub_region(self) as sr:
            liveins, insert_block = sr
            ip, last_loc = self._get_insertion_point_and_loc()

            names, init_handles, init_tys = self._find_carries(node, liveins, ignore={node.target.id})

            # create ForOp
            self._set_insertion_point_and_loc(ip, last_loc)
            for_op = self.builder.create_for_op(lb, ub, step, init_handles)
            if _unwrap_if_constexpr(num_stages) is not None:
                for_op.set_attr("tt.num_stages", self.builder.get_int32_attr(num_stages))
            if _unwrap_if_constexpr(loop_unroll_factor) is not None:
                for_op.set_attr("tt.loop_unroll_factor", self.builder.get_int32_attr(loop_unroll_factor))
            if disallow_acc_multi_buffer:
                for_op.set_attr("tt.disallow_acc_multi_buffer", self.builder.get_unit_attr())
            if flatten:
                for_op.set_attr("tt.flatten", self.builder.get_unit_attr())
            if warp_specialize:
                for_op.set_attr("tt.warp_specialize", self.builder.get_unit_attr())
            if disable_licm:
                for_op.set_attr("tt.disable_licm", self.builder.get_unit_attr())
            if (IteratorClass is extension.parallel):
                for_op.set_attr("hivm.parallel_loop", self.builder.get_unit_attr())

            self.scf_stack.append(node)
            for_op_body = for_op.get_body(0)
            self.builder.set_insertion_point_to_start(for_op_body)
            block_handles = [for_op_body.arg(i + 1) for i in range(len(init_handles))]
            block_args = unflatten_ir_values(block_handles, init_tys)
            for name, val in zip(names, block_args):
                self._maybe_set_loc_to_name(val, name)
                self.set_value(name, val)
            self.visit_compound_statement(node.body)
            self.scf_stack.pop()
            yield_handles = flatten_values_to_ir(self.lscope[name] for name in names)

            # create YieldOp
            if len(yield_handles) > 0:
                self.builder.create_yield_op(yield_handles)
            for_op_region = for_op_body.get_parent()
            assert for_op_region.size() == 1, "We use SCF, so the loop body should only have one block"

            # update induction variable with actual value, and replace all uses
            self.builder.set_insertion_point_to_start(for_op_body)
            iv = for_op.get_induction_var()
            if negative_step:
                iv = self.builder.create_sub(ub, iv)
                iv = self.builder.create_add(iv, lb)
            iv_placeholder.replace_all_uses_with(iv)
            self.set_value(node.target.id, language.core.tensor(iv, iv_type))
            self._maybe_set_loc_to_name(iv, node.target.id)

        # update lscope & local_defs (ForOp defines new values)
        result_handles = [for_op.get_result(i) for i in range(len(init_handles))]
        result_values = unflatten_ir_values(result_handles, init_tys)
        for name, val in zip(names, result_values):
            self.set_value(name, val)
            self._maybe_set_loc_to_name(val, name)

        for stmt in node.orelse:
            assert False, "Don't know what to do with else after for"
            ast.NodeVisitor.generic_visit(self, stmt)

    def call_JitFunction(self, fn: JITFunction, args, kwargs, caller_context=None):
        args = inspect.getcallargs(fn.fn, *args, **kwargs)
        args = [args[name] for name in fn.arg_names]
        for i, arg in enumerate(args):
            if isinstance(arg, (language.dtype, float, int, bool, JITFunction, language.PropagateNan)):
                args[i] = language.core.constexpr(arg)
        args_cst = find_paths_if(args, lambda _, x: _is_constexpr(x))
        args_cst = {path: get_iterable_path(args, path) for path in args_cst}
        args_path = find_paths_if(args, lambda _, x: not _is_constexpr(x))
        args_val = [get_iterable_path(args, path) for path in args_path]
        # mangle
        caller_context = caller_context or self.caller_context
        fn_name = mangle_fn(get_full_name(fn), [arg.type for arg in args_val], args_cst, caller_context)
        # generate function def if necessary
        if not self.module.has_function(fn_name):
            # If the callee is not set, we use the same debug setting as the caller
            file_name, begin_line = get_jit_fn_file_line(fn)
            arg_types = [
                language.core.constexpr if arg is None or isinstance(arg,
                                                                     (bool, int, language.core.dtype)) else arg.type
                for arg in args
            ]
            prototype = ASTFunction([], arg_types, args_cst, dict())
            generator = AscendCodeGenerator(self.context, prototype, fn.get_capture_scope(), module=self.module,
                                            jit_fn=fn, function_name=fn_name, function_types=self.function_ret_types,
                                            noinline=fn.noinline, file_name=file_name, begin_line=begin_line,
                                            options=self.builder.options, codegen_fns=self.builder.codegen_fns,
                                            module_map=self.builder.module_map, caller_context=caller_context,
                                            is_gluon=self.is_gluon)
            try:
                generator.visit(fn.parse())
            except Exception as e:
                # Wrap the error in the callee with the location of the call.
                if knobs.compilation.front_end_debugging:
                    raise
                raise CompilationError(self.jit_fn.src, self.cur_node, None) from e

            callee_ret_type = generator.ret_type
            self.function_ret_types[fn_name] = callee_ret_type
        else:
            callee_ret_type = self.function_ret_types[fn_name]
        symbol = self.module.get_function(fn_name)
        args_val = flatten_values_to_ir(args_val)
        call_op = self.builder.call(symbol, args_val)
        if callee_ret_type == language.void:
            return None
        handles = [call_op.get_result(i) for i in range(call_op.get_num_results())]
        return next(unflatten_ir_values(handles, [callee_ret_type]))

    def call_Function(self, node, fn, args, kws):
        if isinstance(fn, (BoundJITMethod, BoundConstexprFunction)):
            args.insert(0, fn.__self__)
            fn = fn.__func__
        if isinstance(fn, JITFunction):
            _check_fn_args(node, fn, args)
            return self.call_JitFunction(fn, args, kws)
        if (hasattr(fn, '__self__') and _is_triton_value(fn.__self__)) or language.core.is_builtin(fn) or isinstance(
                fn, ConstexprFunction):
            # Copy builder's location and insertion point.
            ip, last_loc = self._get_insertion_point_and_loc()
            # Use ascend_builder if this function is a builtin extension operation.
            _builder = self.ascend_builder if extension.is_builtin(fn) else self.builder
            self._set_insertion_point_and_loc(ip, last_loc, _builder)
            extra_kwargs = dict()

            if isinstance(fn, ConstexprFunction):
                sig = inspect.signature(fn.__call__)
            else:
                sig = inspect.signature(fn)
            if '_semantic' in sig.parameters:
                extra_kwargs["_semantic"] = self.semantic
            if '_generator' in sig.parameters:
                extra_kwargs['_generator'] = self
            try:
                ret = fn(*args, **extra_kwargs, **kws)
                # builtin functions return plain tuples for readability
                if isinstance(ret, tuple):
                    ret = language.tuple(ret)
                # Sync the builder's location before return.
                ip, last_loc = self._get_insertion_point_and_loc(_builder)
                self._set_insertion_point_and_loc(ip, last_loc)
                return ret
            except Exception as e:
                if knobs.compilation.front_end_debugging:
                    raise
                # Normally when we raise a CompilationError, we raise it as
                # `from None`, because the original fileline from the exception
                # is not relevant (and often points into code_generator.py
                # itself).  But when calling a function, we raise as `from e` to
                # preserve the traceback of the original error, which may e.g.
                # be in core.py.
                raise CompilationError(self.jit_fn.src, node, str(e)) from e

        if fn in self.builtin_namespace.values() or (hasattr(fn, '__self__') and not _is_triton_value(fn.__self__)):
            args = map(_unwrap_if_constexpr, args)
        ret = fn(*args, **kws)

        def wrap_constexpr(x):
            if _is_triton_value(x):
                return x
            return constexpr(x)

        if isinstance(ret, (builtins.tuple, language.tuple)):
            return _apply_to_tuple_values(ret, wrap_constexpr)
        return wrap_constexpr(ret)

    CodeGenerator.statically_implemented_functions[extension.int64] = CodeGenerator.static_executor(extension.int64)


def patch_ast_to_ttir(code_generator_module):
    original_ast_to_ttir = code_generator_module.ast_to_ttir
    if getattr(original_ast_to_ttir, "_ascend_patch_applied", False):
        return

    original_ast_to_ttir.__globals__["AscendCodeGenerator"] = AscendCodeGenerator

    def _ascend_ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
        arg_types = [None] * len(fn.arg_names)

        for k, v in src.signature.items():
            idx = fn.arg_names.index(k)
            arg_types[idx] = str_to_ty(v, None)

        def apply_constexpr_types(argument, indices, value):
            index = indices.pop()
            if len(indices) == 0:
                if isinstance(argument, list):
                    argument[index] = constexpr(value).type
                else:
                    argument.types[index] = constexpr(value).type
            else:
                apply_constexpr_types(argument[index], indices, value)

        for path, value in src.constants.items():
            apply_constexpr_types(arg_types, list(path)[::-1], value)

        prototype = ASTFunction([], arg_types, src.constants, src.attrs)
        file_name, begin_line = get_jit_fn_file_line(fn)
        # query function representation
        from collections import namedtuple
        leaves = filter(lambda v: len(v) == 1, src.constants)
        constants = {fn.arg_names[i[0]]: src.constants[i] for i in leaves}
        signature = src.signature
        proxy = namedtuple("SpecializationProxy", ["constants", "signature"])(constants, signature)
        generator = AscendCodeGenerator(context, prototype, gscope=fn.get_capture_scope(), function_name=fn.repr(proxy),
                                        jit_fn=fn, is_kernel=True, file_name=file_name, begin_line=begin_line,
                                        options=options, codegen_fns=codegen_fns, module_map=module_map, module=module,
                                        is_gluon=fn.is_gluon())
        generator.visit(fn.parse())
        module = generator.module
        # module takes ownership of the context
        module.context = context
        if not module.verify():
            if not fn.is_gluon():
                print(module)
            raise RuntimeError("error encountered during parsing")
        return module

    original_ast_to_ttir.__code__ = _ascend_ast_to_ttir.__code__
    original_ast_to_ttir.__defaults__ = _ascend_ast_to_ttir.__defaults__
    original_ast_to_ttir.__kwdefaults__ = _ascend_ast_to_ttir.__kwdefaults__
    original_ast_to_ttir._ascend_patch_applied = True
