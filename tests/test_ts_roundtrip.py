"""TypeScript parse → unified AST → emit tests."""
import json
import textwrap
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _has_ts():
    try:
        import tree_sitter_typescript
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_ts(), reason="tree-sitter-typescript not installed")


def _parse_ts(source: str, filename: str = "test.ts"):
    from src.parsers.ts_parser import parse
    return parse(source, filename)


def _gen_ts(module):
    from src.generators.ts_gen import generate
    return generate(module)


def _gen_py(module):
    from src.generators.python_gen import generate
    return generate(module)


def _gen_rs(module):
    from src.generators.rust_gen import generate
    return generate(module)


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------

def test_parse_sample_ts():
    source = (FIXTURES / "sample.ts").read_text()
    module = _parse_ts(source, "sample.ts")
    assert module.source_language == "typescript"
    names = {n.name for n in module.nodes if hasattr(n, "name")}
    assert "Animal" in names
    assert "Dog" in names
    assert "fibonacci" in names
    assert "Direction" in names


def test_parse_interface():
    source = (FIXTURES / "sample.ts").read_text()
    module = _parse_ts(source, "sample.ts")
    iface = next(n for n in module.nodes if hasattr(n, "name") and n.name == "Speakable")
    from src.unified_ast.nodes import TypeDefCategory
    assert iface.category == TypeDefCategory.INTERFACE
    field_names = {f.name for f in iface.fields}
    assert "name" in field_names
    assert "sound" in field_names


def test_parse_class_fields():
    source = (FIXTURES / "sample.ts").read_text()
    module = _parse_ts(source, "sample.ts")
    animal = next(n for n in module.nodes if hasattr(n, "name") and n.name == "Animal")
    field_names = {f.name for f in animal.fields}
    assert "name" in field_names
    assert "sound" in field_names


def test_parse_type_annotations():
    source = textwrap.dedent("""\
        export function add(a: number, b: number): number {
            return a + b;
        }
    """)
    module = _parse_ts(source)
    fn = module.nodes[0]
    from src.unified_ast.types import TypeKind
    assert fn.params[0].type.kind == TypeKind.NUMBER
    assert fn.params[1].type.kind == TypeKind.NUMBER
    assert fn.return_type.kind == TypeKind.NUMBER


def test_parse_optional_type():
    source = "export function maybe(x: number): number | null { return null; }"
    module = _parse_ts(source)
    fn = module.nodes[0]
    from src.unified_ast.types import TypeKind
    assert fn.return_type.kind == TypeKind.OPTIONAL
    assert fn.return_type.element.kind == TypeKind.NUMBER


def test_parse_array_type():
    source = "export function sum(xs: number[]): number { return 0; }"
    module = _parse_ts(source)
    fn = module.nodes[0]
    from src.unified_ast.types import TypeKind
    assert fn.params[0].type.kind == TypeKind.LIST
    assert fn.params[0].type.element.kind == TypeKind.NUMBER


def test_parse_generic_type():
    source = "export function wrap(xs: Array<string>): Map<string, number> { return new Map(); }"
    module = _parse_ts(source)
    fn = module.nodes[0]
    from src.unified_ast.types import TypeKind
    assert fn.params[0].type.kind == TypeKind.LIST
    assert fn.return_type.kind == TypeKind.MAP


def test_parse_enum():
    source = textwrap.dedent("""\
        export enum Color { Red, Green, Blue }
    """)
    module = _parse_ts(source)
    td = module.nodes[0]
    from src.unified_ast.nodes import TypeDefCategory
    assert td.category == TypeDefCategory.ENUM
    variant_names = {f.name for f in td.fields}
    assert "Red" in variant_names


def test_parse_imports():
    source = textwrap.dedent("""\
        import { readFileSync } from "fs";
        import * as path from "path";
        import React from "react";
    """)
    module = _parse_ts(source)
    assert len(module.nodes) == 3
    mods = {n.module for n in module.nodes}
    assert "fs" in mods
    assert "path" in mods
    assert "react" in mods


def test_parse_for_of():
    source = textwrap.dedent("""\
        export function total(items: number[]): number {
            let sum: number = 0;
            for (const x of items) {
                sum = sum + x;
            }
            return sum;
        }
    """)
    module = _parse_ts(source)
    fn = module.nodes[0]
    from src.unified_ast.expr import ForEach
    body = fn.body.stmts
    for_stmt = next(s for s in body if isinstance(s, ForEach))
    assert for_stmt.var == "x"


def test_parse_if_else():
    source = textwrap.dedent("""\
        export function sign(x: number): number {
            if (x > 0) {
                return 1;
            } else if (x < 0) {
                return -1;
            } else {
                return 0;
            }
        }
    """)
    module = _parse_ts(source)
    fn = module.nodes[0]
    from src.unified_ast.expr import If
    if_stmt = fn.body.stmts[0]
    assert isinstance(if_stmt, If)
    assert len(if_stmt.elif_branches) == 1
    assert if_stmt.else_block is not None


# ---------------------------------------------------------------------------
# Emit tests
# ---------------------------------------------------------------------------

def test_emit_function():
    source = "export function add(a: number, b: number): number { return a + b; }"
    module = _parse_ts(source)
    out = _gen_ts(module)
    assert "function add" in out
    assert "number" in out
    assert "return" in out


def test_emit_class():
    source = textwrap.dedent("""\
        export class Counter {
            private value: number = 0;
            public increment(): void {
                this.value = this.value + 1;
            }
            public get(): number {
                return this.value;
            }
        }
    """)
    module = _parse_ts(source)
    out = _gen_ts(module)
    assert "class Counter" in out
    assert "increment" in out
    assert "get" in out


def test_emit_interface():
    source = textwrap.dedent("""\
        export interface Shape {
            area(): number;
            perimeter(): number;
        }
    """)
    module = _parse_ts(source)
    out = _gen_ts(module)
    assert "interface Shape" in out
    assert "area" in out
    assert "perimeter" in out


def test_emit_enum():
    source = "export enum Direction { North, South, East, West }"
    module = _parse_ts(source)
    out = _gen_ts(module)
    assert "enum Direction" in out
    assert "North" in out


def test_emit_for_of():
    source = textwrap.dedent("""\
        export function total(items: number[]): number {
            let sum: number = 0;
            for (const x of items) {
                sum = sum + x;
            }
            return sum;
        }
    """)
    module = _parse_ts(source)
    out = _gen_ts(module)
    assert "for" in out
    assert "of" in out


def test_emit_imports():
    source = textwrap.dedent("""\
        import { readFileSync } from "fs";
        import * as path from "path";
    """)
    module = _parse_ts(source)
    out = _gen_ts(module)
    assert "from" in out
    assert "fs" in out
    assert "path" in out


def test_roundtrip_json():
    source = (FIXTURES / "sample.ts").read_text()
    module = _parse_ts(source, "sample.ts")
    d = module.to_dict()
    from src.unified_ast.nodes import Module
    module2 = Module.from_dict(json.loads(json.dumps(d)))
    out = _gen_ts(module2)
    assert "fibonacci" in out
    assert "greet" in out


def test_skeleton():
    from src.harness.skeleton import build_skeleton
    source = (FIXTURES / "sample.ts").read_text()
    module = _parse_ts(source, "sample.ts")
    skel = build_skeleton(module)
    skel_str = json.dumps(skel)
    assert "fibonacci" in skel_str
    assert '"kind": "return"' not in skel_str


def test_get_node():
    from src.harness.editor import get_node
    source = (FIXTURES / "sample.ts").read_text()
    module = _parse_ts(source, "sample.ts")
    node = get_node(module, "fn:fibonacci")
    assert node["kind"] == "function"
    assert len(node["body"]["stmts"]) > 0


# ---------------------------------------------------------------------------
# Cross-language: TypeScript ↔ Python / Rust
# ---------------------------------------------------------------------------

def test_ts_to_python():
    import ast
    source = textwrap.dedent("""\
        export function add(a: number, b: number): number {
            return a + b;
        }
    """)
    module = _parse_ts(source)
    py = _gen_py(module)
    ast.parse(py)
    assert "def add" in py


def test_ts_to_rust():
    source = textwrap.dedent("""\
        export function add(a: number, b: number): number {
            return a + b;
        }
    """)
    module = _parse_ts(source)
    rs = _gen_rs(module)
    assert "fn add" in rs
    assert "return" in rs


def test_ts_class_to_python():
    import ast
    source = textwrap.dedent("""\
        export class Counter {
            private value: number = 0;
            public increment(): void {
                this.value = this.value + 1;
            }
            public get(): number {
                return this.value;
            }
        }
    """)
    module = _parse_ts(source)
    py = _gen_py(module)
    ast.parse(py)
    assert "class Counter" in py
    assert "def increment" in py


def test_python_to_ts():
    import ast
    source = textwrap.dedent("""\
        def add(a: int, b: int) -> int:
            return a + b
    """)
    from src.parsers.python_parser import parse as py_parse
    module = py_parse(source)
    ts = _gen_ts(module)
    assert "function add" in ts
    assert "number" in ts


def test_rust_to_ts():
    from src.parsers.rust_parser import parse as rs_parse
    source = textwrap.dedent("""\
        pub fn fibonacci(n: i64) -> i64 {
            if n <= 1 {
                return n;
            }
            let mut a: i64 = 0;
            let mut b: i64 = 1;
            let mut i: i64 = 2;
            while i <= n {
                let temp: i64 = a + b;
                a = b;
                b = temp;
                i = i + 1;
            }
            b
        }
    """)
    module = rs_parse(source)
    ts = _gen_ts(module)
    assert "function fibonacci" in ts
    assert "number" in ts
    assert "while" in ts
