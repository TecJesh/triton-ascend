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


def _ascend_compile(src, target=None, options=None, _env_vars=None):
    from triton.compiler.errors import MLIRCompilationError
    compilation_listener = knobs.compilation.listener
    if compilation_listener:
        timer = CompileTimer()

    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    # create backend
    if ir_source:
        assert isinstance(src, str), "source must be either AST or a filepath"
        context = ir.context()
        src = IRSource(src, context, backend)

    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))
    # create cache manager
    env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
    key = get_cache_key(src, backend, options, env_vars=env_vars)
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = knobs.compilation.override
    enable_ir_dump = knobs.compilation.dump_ir
    store_only_binary = knobs.compilation.store_binary_only
    fn_override_manager = get_override_manager(src.hash()) if enable_override else None
    fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = src.name[:150]
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = knobs.compilation.always_compile
    if not always_compile and metadata_path is not None:
        # cache hit!
        res = CompiledKernel(src, metadata_group, hash)
        if compilation_listener:
            compilation_listener(
                src=src,
                metadata=res.metadata._asdict(),
                metadata_group=metadata_group,
                times=timer.end(),
                cache_hit=True,
            )
        return res

    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = __version__
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(stages, options, src.language)
    first_stage = list(stages.keys()).index(src.ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1

    # For IRSource, we have already grabbed the context + called both
    # ir.load_dialects and backend.load_dialects.
    if not isinstance(src, IRSource):
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    try:
        module = src.make_ir(target, options, codegen_fns, module_map, context)
    except Exception as e:
        filter_traceback(e)
        raise

    if ir_source:
        ir_filename = f"{file_name}.{src.ext}"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)
    else:
        ir_filename = f"{file_name}.source"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

    use_ir_loc = knobs.compilation.use_ir_loc
    if ir_source and use_ir_loc:
        module.create_location_snapshot(src.path)
        print(f"Creating new locations for {src.path}")

    if compilation_listener:
        timer.finished_ir_initialization()
    for ext, compile_ir in list(stages.items())[first_stage:]:
        try:
            next_module = compile_ir(module, metadata)
        except Exception as e:
            if (ext == "ttadapter"):
                stage_name = "ConvertTritonIRToLinalgIR"
            elif (ext == "npubin"):
                stage_name = "ConvertLinalgIRToBinary"
            elif (ext == "bcmlir"):
                stage_name = "BytecodeToLinalgIRByBishengirOpt"
            elif (ext == "mlirbc"):
                stage_name = "LinalgIRToBytecodeByTritonMLIROpt"
            else:
                stage_name = "MLIRCompile"
            if hasattr(e, 'stderr') and e.stderr:
                error_detail = e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else e.stderr
            else:
                error_detail = str(e)
            from ..runtime.cache import FileCacheManager
            if isinstance(fn_cache_manager, FileCacheManager):
                error_detail += f"\n\n[INFO]: The compiled kernel cache is in {fn_cache_manager.cache_dir}\n\n"
            else:
                error_detail += f"\n\n[INFO]: The compiled kernel cache is {file_name}.{ext}\n\n"
            raise MLIRCompilationError(stage_name, error_detail) from e
        ir_filename = f"{file_name}.{ext}"
        if fn_override_manager is None:
            # Users can override kernels at scale by setting `ir_override` in autotune config
            # without TRITON_KERNEL_OVERRIDE
            if (ir_override := metadata.get("ir_override", None)) and ir_override.endswith(f".{ext}"):
                next_module = parse(ir_override, ext, context)
        elif full_name := fn_override_manager.get_file(ir_filename):
            print(f"\nOverriding kernel with file {full_name}")
            next_module = parse(full_name, ext, context)
        # If TRITON_STORE_BINARY_ONLY is 1, only store cubin/hsaco/json
        if (not store_only_binary) or (ext in ("cubin", "hsaco", "json")):
            metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
            if ext == "cubin":
                sass = get_sass(next_module)
                fn_dump_manager.put(sass, file_name + ".sass")
        # use an env variable to parse ir from file
        if use_ir_loc == ext:
            ir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ir_full_name)
            print(f"Creating new locations for {ir_full_name}")
        module = next_module
        if compilation_listener:
            timer.stage_finished(ext)
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)
    # Compilation completed, disabling multithreading in context.
    # This is needed to safely finalize threads pool inside context: if current process forks before
    # python GC deletes context object, thread pool in child process will be invalid, which could
    # lead to child crash or hang.
    #
    # However disabling multithreading causes the code to hang if the ASAN pass is enabled
    # this is likely due to the llvm-symbolizer forking a process
    # TODO: Reconcile the difference here between the ASAN and non-ASAN path with enabling
    # multithreading in the MLIR context
    if not knobs.compilation.enable_asan:
        context.disable_multithreading()

    # notify any listener
    if compilation_listener:
        compilation_listener(src=src, metadata=metadata, metadata_group=metadata_group, times=timer.end(),
                             cache_hit=False)
    # return handle to compiled kernel
    return CompiledKernel(src, metadata_group, hash)

def _raise_error(err, *args, **kwargs):
    raise copy.deepcopy(err)

def patch_compiler_runtime(compiler_module):
    compiled_kernel_cls = compiler_module.CompiledKernel
    if getattr(compiled_kernel_cls, "_ascend_patch_applied", False):
        return

    original_compile = compiler_module.compile
    original_compile.__globals__["_ascend_compile"] = _ascend_compile
    original_compile._raise_error = _raise_error

    def _patched_init(self, src, metadata_group, hash):
        from collections import namedtuple

        metadata_path = next((compiler_module.Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        metadata = compiler_module.json.loads(metadata_path.read_text())
        metadata['cluster_dims'] = tuple(metadata['cluster_dims'])
        # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
        target = metadata['target']
        metadata['target'] = compiler_module.GPUTarget(target['backend'], target['arch'], target['warp_size'])
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)
        backend = compiler_module.make_backend(self.metadata.target)
        self.packed_metadata = backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name
        # stores the text of each level of IR that was generated during compilation
        asm_files = [compiler_module.Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = backend.binary_ext
        binary_extensions = getattr(backend, 'binary_extensions', {binary_ext})
        self.asm = compiler_module.AsmDict({
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] in binary_extensions else file.read_text()
            for file in asm_files
        })
        self.metadata_group = metadata_group
        self.kernel = self.asm[binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None
        self._run = None

    def _patched_init_handles(self):
        if self.module is not None:
            return

        def raise_(err):
            # clone the exception object so that the one saved in the closure
            # of the partial function below doesn't get assigned a stack trace
            # after the subsequent raise. otherwise, the CompiledKernel instance
            # saved in the (global) kernel cache will keep references to all the
            # locals in the traceback via the exception instance in the closure.
            cloned_err = copy.deepcopy(err)
            self._run = compiler_module.functools.partial(compiler_module._raise_error, cloned_err)
            raise err

        device = compiler_module.driver.active.get_current_device()
        # create launcher
        self._run = compiler_module.driver.active.launcher_cls(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = compiler_module.max_shared_mem(device)
        if self.metadata.shared > max_shared:
            raise_(compiler_module.OutOfResources(self.metadata.shared, max_shared, "shared memory"))
        if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
            # Use blackwell max tmem size for now, this should be moved in device properties
            max_tmem_size = 512  # tmem size in number of columns
            if self.metadata.tmem_size > max_tmem_size:
                raise_(compiler_module.OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory"))
        if compiler_module.knobs.runtime.kernel_load_start_hook is not None:
            compiler_module.knobs.runtime.kernel_load_start_hook(self.module, self.function, self.name, self.metadata_group, self.hash)
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = compiler_module.driver.active.utils.load_binary(
            self.metadata.kernel_name, self.kernel, self.metadata.shared, device, self.metadata.mix_mode)
        warp_size = compiler_module.driver.active.get_current_target().warp_size
        if self.metadata.num_warps * warp_size > self.n_max_threads:
            raise_(compiler_module.OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads"))
        if compiler_module.knobs.runtime.kernel_load_end_hook is not None:
            compiler_module.knobs.runtime.kernel_load_end_hook(self.module, self.function, self.name, self.metadata_group, self.hash)

    compiled_kernel_cls.__init__ = _patched_init
    compiled_kernel_cls._init_handles = _patched_init_handles
    compiled_kernel_cls._ascend_patch_applied = True

    original_compile.__code__ = _ascend_compile.__code__
    original_compile.__defaults__ = _ascend_compile.__defaults__
    original_compile.__kwdefaults__ = _ascend_compile.__kwdefaults__
    original_compile._ascend_patch_applied = True