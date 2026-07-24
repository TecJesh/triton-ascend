# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Patches to ``triton.compiler.compiler.CompiledKernel`` for Ascend NPU.

Applied by ``patch_compiler_runtime()``.  Each patch is small and
targeted — no full-class duplication.
"""

import copy
import functools
import json
from collections import namedtuple
from pathlib import Path

from triton import knobs
from triton.compiler.compiler import (GPUTarget, AsmDict, LazyDict, make_backend, _raise_error, max_shared_mem)
from triton.runtime.autotuner import OutOfResources
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver


def apply(compiler_module):
    """Apply all Ascend compiler patches (idempotent)."""
    CompiledKernel = compiler_module.CompiledKernel
    if getattr(CompiledKernel, "_ascend_patch_applied", False):
        return

    # --- Patch 1: CompiledKernel.__init__ ---
    _original_init = CompiledKernel.__init__

    def _patched_init(self, src, metadata_group, hash):
        metadata_path = next((Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        metadata = json.loads(metadata_path.read_text())
        # --- Ascend: tuple-ify cluster_dims ---
        metadata['cluster_dims'] = tuple(metadata['cluster_dims'])
        target = metadata['target']
        metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)
        backend = make_backend(self.metadata.target)
        self.packed_metadata = backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name
        asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = backend.binary_ext
        # --- Ascend: binary_extensions for multi-format binary support ---
        binary_extensions = getattr(backend, 'binary_extensions', {binary_ext})
        self.asm = AsmDict({
            file.suffix[1:]:
            file.read_bytes() if file.suffix[1:] in binary_extensions else file.read_text()
            for file in asm_files
        })
        self.metadata_group = metadata_group
        self.kernel = self.asm[binary_ext]
        self.module = None
        self.function = None
        self._run = None

    # --- Patch 2: CompiledKernel._init_handles ---
    _original_init_handles = CompiledKernel._init_handles

    def _patched_init_handles(self):
        if self.module is not None:
            return

        def raise_(err):
            cloned_err = copy.deepcopy(err)
            self._run = functools.partial(_raise_error, cloned_err)
            raise err

        device = driver.active.get_current_device()
        self._run = driver.active.launcher_cls(self.src, self.metadata)
        max_shared = max_shared_mem(device)
        if self.metadata.shared > max_shared:
            raise_(OutOfResources(self.metadata.shared, max_shared, "shared memory"))
        # --- Ascend: tmem_size guard (Blackwell) ---
        if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
            max_tmem_size = 512
            if self.metadata.tmem_size > max_tmem_size:
                raise_(OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory"))
        if knobs.runtime.kernel_load_start_hook is not None:
            knobs.runtime.kernel_load_start_hook(self.module, self.function, self.name, self.metadata_group, self.hash)
        # --- Ascend: load_binary with kernel_name + mix_mode ---
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = \
            driver.active.utils.load_binary(
                self.metadata.kernel_name, self.kernel,
                self.metadata.shared, device, self.metadata.mix_mode)
        warp_size = driver.active.get_current_target().warp_size
        if self.metadata.num_warps * warp_size > self.n_max_threads:
            raise_(OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads"))
        if knobs.runtime.kernel_load_end_hook is not None:
            knobs.runtime.kernel_load_end_hook(self.module, self.function, self.name, self.metadata_group, self.hash)

    CompiledKernel.__init__ = _patched_init
    CompiledKernel._init_handles = _patched_init_handles
    CompiledKernel._ascend_patch_applied = True
