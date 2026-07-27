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
import sys as _sys
import types
import importlib.util as _ilu

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _load_module(module_name, file_path):
    spec = _ilu.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name!r} from {file_path!r}")
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# General information about the project.

project = 'Triton Ascend'
copyright = '2026, Huawei'
author = 'Huawei'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = ''
# The full version, including alpha/beta/rc tags.
release = ''

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'myst_parser',
]

autosummary_generate = True

_sys.path.insert(0, os.path.join(_REPO, "python"))
_force_mock = (os.environ.get("TRITON_DOCS_FORCE_MOCK", "").lower() in ("1", "true", "yes")
               or os.environ.get("READTHEDOCS") == "True")
if not _force_mock:
    try:
        import triton  # noqa: F401,E402
    except Exception as _exc:
        print(f"import triton failed ({_exc!r}); building docs with mock stubs")
        _force_mock = True

if _force_mock:
    _load_module(
        "docs.zh._mock._triton_mock",
        os.path.join(_HERE, "_mock", "_triton_mock.py"),
    ).install()

import triton  # noqa: E402

if _force_mock and "triton.language" in _sys.modules:
    _mock_lang = _sys.modules["triton.language"]
    if not getattr(_mock_lang, "__file__", None):
        for _name in list(_sys.modules):
            if _name == "triton.language" or _name.startswith("triton.language."):
                del _sys.modules[_name]
        import triton.language  # noqa: F401, E402
        import triton.language.extra as _tl_extra  # noqa: E402
    else:
        import triton.language.extra as _tl_extra  # noqa: E402
else:
    import triton.language.extra as _tl_extra  # noqa: E402

_cann_lang_path = os.path.join(_REPO, "third_party", "ascend", "language")
if _cann_lang_path not in _tl_extra.__path__:
    _tl_extra.__path__.append(_cann_lang_path)

if _force_mock:
    for _name, _path in [
        ("triton.language.extra.cann", _cann_lang_path),
        ("triton.language.extra.cann.extension", os.path.join(_cann_lang_path, "cann", "extension")),
        ("triton.language.extra.cann.libdevice", os.path.join(_cann_lang_path, "cann", "libdevice")),
        ("triton.language.extra.extension.buffer.language", os.path.join(_cann_lang_path, "extension", "buffer", "language")),
    ]:
        _stub = _sys.modules.get(_name)
        if _stub is None or not getattr(_stub, "__file__", None):
            _stub = types.ModuleType(_name)
            _stub.__package__ = _name
            _stub.__path__ = [_path]
            _sys.modules[_name] = _stub
        _parent_name, _, _child = _name.rpartition(".")
        _parent = _sys.modules.get(_parent_name)
        if _parent is not None:
            setattr(_parent, _child, _stub)

# -- I18n: detect language and root doc ---------------------------------------
_readthedocs_lang = os.environ.get('READTHEDOCS_LANGUAGE')
_is_build_by_readthedocs = _readthedocs_lang is not None

if _readthedocs_lang:
    _build_lang = _readthedocs_lang.strip().lower().replace('_', '-')
else:
    _build_lang = (os.environ.get('LANGUAGE') or 'en').strip().lower().replace('_', '-')

_is_zh = _build_lang in ('zh-cn', 'zh') or _build_lang.startswith('zh-')
language = 'zh_CN' if _is_zh else 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
if _is_zh:
    exclude_patterns.extend(['source/en'])
else:
    exclude_patterns.extend(['source/zh_cn'])

# -- General configuration ---------------------------------------------------
templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
pygments_style = "sphinx"
html_last_updated_fmt = "%b %d, %Y"

def setup(app):
    """Register Pygments lexer aliases and Ascend notes extension."""
    from sphinx.highlighting import lexers
    from pygments.lexers import get_lexer_by_name

    lexers['mlir'] = get_lexer_by_name('text')
    lexers['plaintext'] = get_lexer_by_name('text')

    app.add_css_file('custom.css')
    if not _is_build_by_readthedocs:
        app.add_js_file('lang-switcher.js')
        app.add_css_file('lang-switcher.css')

    _load_module(
        "docs.zh.python_api._inject_ascend_notes",
        os.path.join(_HERE, "python-api", "_inject_ascend_notes.py"),
    ).setup(app)

    return {'version': '0.1', 'parallel_read_safe': True}

readthedocs_version = os.environ.get('READTHEDOCS_VERSION', 'latest')
version = readthedocs_version.split('.')[0] + '.' + readthedocs_version.split('.')[1] if '.' in readthedocs_version else ''
release = readthedocs_version
