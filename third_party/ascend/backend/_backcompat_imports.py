# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Lazy import hooks so old import paths redirect to new locations
without modifying existing code.

  import triton.extension.buffer.language as bl
      -> triton.language.extra.extension.buffer.language

  from triton.runtime.libentry import libentry
      -> triton.backends.ascend.runtime.libentry
"""

import os
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

# (old_name, new_name, is_package)
_REDIRECTS = [
    ("triton.extension.buffer.language", "triton.language.extra.extension.buffer.language", True),
    ("triton.runtime.libentry", "triton.backends.ascend.runtime.libentry", False),
]

# Synthetic parent packages (deleted from python/triton/extension/).
_SYNTHETIC = {"triton.extension", "triton.extension.buffer"}

# Packages that were auto-created as lightweight stubs to bypass __init__.py
# side effects during backcompat redirects. If the real package is later
# imported directly, the stub is removed so the real __init__.py executes.
_SYNTH_BYPASS = set()


def _make_synth(name):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


def _ensure_synth_runtime():
    """Pre-register a lightweight parent package so importing
    triton.backends.ascend.runtime.libentry does NOT trigger
    runtime/__init__.py (and its module-level _patch_autotune call).

    Only the minimal attributes needed for submodule resolution
    (__path__, __package__) are set.  The stub is tracked in
    _SYNTH_BYPASS so it can be transparently replaced when code
    later imports the real runtime package.
    """
    name = "triton.backends.ascend.runtime"
    if name in sys.modules:
        return  # already loaded (real or synthetic)
    runtime_dir = os.path.join(os.path.dirname(__file__), "runtime")
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = [runtime_dir]
    mod.__file__ = os.path.join(runtime_dir, "__init__.py")
    sys.modules[name] = mod
    _SYNTH_BYPASS.add(name)


class _Loader(Loader):

    def __init__(self, target):
        self._target = target

    def create_module(self, spec):
        import importlib
        return importlib.import_module(self._target)

    def exec_module(self, module):
        pass


class _Finder(MetaPathFinder):
    _map = {old: (new, pkg) for old, new, pkg in _REDIRECTS}

    def find_spec(self, fullname, path, target=None):
        if fullname in _SYNTHETIC:
            _make_synth(fullname)
            return ModuleSpec(fullname, None, is_package=True)

        # If code imports the real runtime package directly, remove any
        # synthetic stub we placed earlier so the real __init__.py runs.
        if fullname in _SYNTH_BYPASS:
            del sys.modules[fullname]
            _SYNTH_BYPASS.discard(fullname)
            return None  # let Python's default import machinery load __init__.py

        pair = self._map.get(fullname)
        if pair is None:
            return None

        # Before redirecting triton.runtime.libentry → triton.backends.ascend.runtime.libentry,
        # pre-register a synthetic parent package so runtime/__init__.py is skipped.
        _ensure_synth_runtime()

        return ModuleSpec(fullname, _Loader(pair[0]), is_package=pair[1])


def install():
    """Register the backcompat import hook (idempotent)."""
    for f in sys.meta_path:
        if isinstance(f, _Finder):
            return
    for pkg in _SYNTHETIC:
        _make_synth(pkg)
    sys.meta_path.insert(0, _Finder())
