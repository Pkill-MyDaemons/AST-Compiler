"""Rust parse → unified AST → emit Rust tests."""
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from src.harness.skeleton import build_skeleton
from src.harness.editor import get_node


FIXTURES = Path(__file__).parent / "fixtures"


def _has_tree_sitter_rust():
    try:
        import tree_sitter_rust
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_tree_sitter_rust(),
    reason="tree-sitter-rust not installed",
)


def _parse_rust(source: str, filename: str = "test.rs"):
    from src.parsers.rust_parser import parse
    return parse(source, filename)


def _gen_rust(module):
    from src.generators.rust_gen import generate
    return generate(module)


def test_parse_sample_rs():
    source = (FIXTURES / "sample.rs").read_text()
    module = _parse_rust(source, "sample.rs")
    assert module.source_language == "rust"
    names = {n.name for n in module.nodes if hasattr(n, "name")}
    assert "Animal" in names
    assert "Dog" in names
    assert "fibonacci" in names


def test_emit_rust_struct():
    source = textwrap.dedent("""\
        pub struct Point {
            pub x: f64,
            pub y: f64,
        }
        impl Point {
            pub fn new(x: f64, y: f64) -> Self {
                Point { x, y }
            }
            pub fn distance(&self) -> f64 {
                (self.x * self.x + self.y * self.y)
            }
        }
    """)
    module = _parse_rust(source, "test.rs")
    emitted = _gen_rust(module)
    assert "struct Point" in emitted
    assert "fn new" in emitted
    assert "fn distance" in emitted


def test_roundtrip_json_rust():
    source = (FIXTURES / "sample.rs").read_text()
    module = _parse_rust(source, "sample.rs")
    d = module.to_dict()
    json_str = json.dumps(d)
    from src.unified_ast.nodes import Module
    module2 = Module.from_dict(json.loads(json_str))
    emitted = _gen_rust(module2)
    assert "fn fibonacci" in emitted
    assert "fn greet" in emitted


def test_skeleton_rust():
    source = (FIXTURES / "sample.rs").read_text()
    module = _parse_rust(source, "sample.rs")
    skel = build_skeleton(module)
    skel_str = json.dumps(skel)
    assert "fibonacci" in skel_str
    # No body statements should appear in skeleton
    assert '"kind": "return"' not in skel_str


def test_get_node_rust():
    source = (FIXTURES / "sample.rs").read_text()
    module = _parse_rust(source, "sample.rs")
    node = get_node(module, "fn:fibonacci")
    assert node["kind"] == "function"
    assert "body" in node


def test_const_item():
    source = "pub const MAX: i32 = 100;\n"
    module = _parse_rust(source, "test.rs")
    emitted = _gen_rust(module)
    assert "MAX" in emitted
    assert "100" in emitted


def test_use_declaration():
    source = "use std::collections::HashMap;\n"
    module = _parse_rust(source, "test.rs")
    emitted = _gen_rust(module)
    assert "use" in emitted
    assert "HashMap" in emitted


def test_enum_item():
    source = textwrap.dedent("""\
        pub enum Color {
            Red,
            Green,
            Blue,
        }
    """)
    module = _parse_rust(source, "test.rs")
    emitted = _gen_rust(module)
    assert "enum Color" in emitted


def test_if_expression():
    source = textwrap.dedent("""\
        pub fn sign(x: i64) -> i64 {
            if x > 0 {
                1
            } else if x < 0 {
                -1
            } else {
                0
            }
        }
    """)
    module = _parse_rust(source, "test.rs")
    emitted = _gen_rust(module)
    assert "fn sign" in emitted


def test_while_loop_rust():
    source = textwrap.dedent("""\
        pub fn countdown(mut n: i64) -> i64 {
            let mut result: i64 = 0;
            while n > 0 {
                result = result + n;
                n = n - 1;
            }
            result
        }
    """)
    module = _parse_rust(source, "test.rs")
    emitted = _gen_rust(module)
    assert "while" in emitted or "loop" in emitted
