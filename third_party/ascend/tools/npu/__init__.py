"""NPU AOT tooling installed as ``triton.tools.extra.npu``.

Holds the C templates for ``triton.tools.compile`` and the helper module
``aot`` that keeps the NPU-specific AOT logic out of the patched upstream
tools (``python/triton/tools/compile.py`` / ``link.py`` carry only minimal
function calls into this package).
"""
