"""Language parser registry."""
from __future__ import annotations
from ..unified_ast.nodes import Module


def parse(source: str, language: str, filename: str = "<string>") -> Module:
    lang = language.lower().strip()
    if lang in ("python", "py"):
        from .python_parser import parse as py_parse
        return py_parse(source, filename)
    if lang in ("rust", "rs"):
        from .rust_parser import parse as rs_parse
        return rs_parse(source, filename)
    if lang in ("typescript", "ts"):
        from .ts_parser import parse as ts_parse
        return ts_parse(source, filename, tsx=False)
    if lang in ("tsx",):
        from .ts_parser import parse as ts_parse
        return ts_parse(source, filename, tsx=True)
    raise ValueError(f"Unsupported language: {language!r}. Supported: python, rust, typescript, tsx")


def detect_language(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mapping = {
        "py": "python",
        "rs": "rust",
        "ts": "typescript",
        "tsx": "tsx",
        "js": "typescript",   # treat JS as TS (TS parser handles plain JS)
        "mts": "typescript",
        "cts": "typescript",
    }
    lang = mapping.get(ext)
    if lang is None:
        raise ValueError(f"Cannot detect language from extension .{ext!r}. Pass --lang explicitly.")
    return lang
