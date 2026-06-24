"""Minified JSON serializer — compresses AST dict keys and kind values."""
from __future__ import annotations
import json
from typing import Any

_KEY_MAP: dict[str, str] = {
    "kind": "k",
    "id": "i",
    "module": "m",
    "body": "b",
    "stmts": "s",
    "expr": "e",
    "func": "f",
    "args": "a",
}

# text/name/value all compress to "v"; first one found in each dict wins.
_VALUE_KEYS = ("text", "name", "value")

_KIND_MAP: dict[str, str] = {
    "import": "imp",
    "function": "fn",
    "expr_stmt": "es",
    "raw_expr": "re",
    "var_decl": "vd",
    "field_access": "fa",
}


def _compress(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_compress(x) for x in obj]
    if not isinstance(obj, dict):
        return obj

    out: dict[str, Any] = {}
    v_used = False

    for key, val in obj.items():
        compressed_val = _compress(val)

        if key == "kind" and isinstance(val, str) and val in _KIND_MAP:
            out["k"] = _KIND_MAP[val]
            continue

        if key in _KEY_MAP:
            out[_KEY_MAP[key]] = compressed_val
            continue

        if key in _VALUE_KEYS and not v_used:
            out["v"] = compressed_val
            v_used = True
            continue

        out[key] = compressed_val

    return out


def minify(ast_dict: dict) -> str:
    """Compress an AST dict and return a compact JSON string."""
    root = {
        "lang": ast_dict.get("source_language", ""),
        "file": ast_dict.get("source_file", ""),
        "nodes": [_compress(n) for n in ast_dict.get("nodes", [])],
    }
    return json.dumps(root, separators=(",", ":"), ensure_ascii=False)
