"""Language parser registry."""
from __future__ import annotations
from ..unified_ast.nodes import Module


def parse(source: str, language: str, filename: str = "<string>") -> Module:
    lang = language.lower().strip()
    if lang in ("python", "py"):
        from .python_parser import parse as _p; return _p(source, filename)
    if lang in ("rust", "rs"):
        from .rust_parser import parse as _p; return _p(source, filename)
    if lang in ("typescript", "ts"):
        from .ts_parser import parse as _p; return _p(source, filename, tsx=False)
    if lang in ("tsx",):
        from .ts_parser import parse as _p; return _p(source, filename, tsx=True)
    if lang in ("javascript", "js"):
        from .javascript_parser import parse as _p; return _p(source, filename)
    if lang in ("go",):
        from .go_parser import parse as _p; return _p(source, filename)
    if lang in ("java",):
        from .java_parser import parse as _p; return _p(source, filename)
    if lang in ("kotlin", "kt", "kts"):
        from .kotlin_parser import parse as _p; return _p(source, filename)
    if lang in ("swift",):
        from .swift_parser import parse as _p; return _p(source, filename)
    if lang in ("zig",):
        from .zig_parser import parse as _p; return _p(source, filename)
    if lang in ("c",):
        from .c_parser import parse as _p; return _p(source, filename)
    if lang in ("cpp", "c++", "cxx"):
        from .c_parser import parse as _p; return _p(source, filename)
    raise ValueError(
        f"Unsupported language: {language!r}. "
        "Supported: python, rust, typescript, tsx, javascript, go, java, kotlin, swift, zig, c, cpp"
    )


def detect_language(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mapping = {
        "py":   "python",
        "rs":   "rust",
        "ts":   "typescript",
        "tsx":  "tsx",
        "mts":  "typescript",
        "cts":  "typescript",
        "js":   "javascript",
        "mjs":  "javascript",
        "cjs":  "javascript",
        "go":   "go",
        "java": "java",
        "kt":   "kotlin",
        "kts":  "kotlin",
        "swift": "swift",
        "zig":  "zig",
        "c":    "c",
        "h":    "c",
        "cpp":  "cpp",
        "cc":   "cpp",
        "cxx":  "cpp",
        "hpp":  "cpp",
        "hxx":  "cpp",
    }
    lang = mapping.get(ext)
    if lang is None:
        raise ValueError(f"Cannot detect language from extension .{ext!r}. Pass --lang explicitly.")
    return lang
