"""Language generator registry."""
from __future__ import annotations
from ..unified_ast.nodes import Module


def generate(module: Module, language: str) -> str:
    lang = language.lower().strip()
    if lang in ("python", "py"):
        from .python_gen import generate as _g; return _g(module)
    if lang in ("rust", "rs"):
        from .rust_gen import generate as _g; return _g(module)
    if lang in ("typescript", "ts", "tsx"):
        from .ts_gen import generate as _g; return _g(module)
    if lang in ("javascript", "js"):
        from .javascript_gen import generate as _g; return _g(module)
    if lang in ("go",):
        from .go_gen import generate as _g; return _g(module)
    if lang in ("java",):
        from .java_gen import generate as _g; return _g(module)
    if lang in ("kotlin", "kt", "kts"):
        from .kotlin_gen import generate as _g; return _g(module)
    if lang in ("swift",):
        from .swift_gen import generate as _g; return _g(module)
    if lang in ("zig",):
        from .zig_gen import generate as _g; return _g(module)
    if lang in ("c",):
        from .c_gen import generate as _g; return _g(module)
    if lang in ("cpp", "c++", "cxx"):
        from .c_gen import generate as _g; return _g(module)
    raise ValueError(
        f"Unsupported language: {language!r}. "
        "Supported: python, rust, typescript, javascript, go, java, kotlin, swift, zig, c, cpp"
    )
