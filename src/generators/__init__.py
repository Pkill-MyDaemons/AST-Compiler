"""Language generator registry."""
from __future__ import annotations
from ..unified_ast.nodes import Module


def generate(module: Module, language: str) -> str:
    lang = language.lower().strip()
    if lang in ("python", "py"):
        from .python_gen import generate as py_gen
        return py_gen(module)
    if lang in ("rust", "rs"):
        from .rust_gen import generate as rs_gen
        return rs_gen(module)
    if lang in ("typescript", "ts", "tsx"):
        from .ts_gen import generate as ts_gen
        return ts_gen(module)
    raise ValueError(f"Unsupported language: {language!r}. Supported: python, rust, typescript")
