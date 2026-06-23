"""Python parse → emit → re-parse round-trip tests."""
import ast
import json
import textwrap
from pathlib import Path

import pytest

from src.parsers.python_parser import parse
from src.generators.python_gen import generate
from src.harness.skeleton import build_skeleton
from src.harness.editor import get_node, str_replace_body, rename_node


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_sample():
    source = (FIXTURES / "sample.py").read_text()
    module = parse(source, "sample.py")
    assert module.source_language == "python"
    assert any(n.name == "Animal" for n in module.nodes if hasattr(n, "name"))
    assert any(n.name == "Dog" for n in module.nodes if hasattr(n, "name"))
    assert any(n.name == "fibonacci" for n in module.nodes if hasattr(n, "name"))


def test_emit_is_valid_python():
    source = (FIXTURES / "sample.py").read_text()
    module = parse(source, "sample.py")
    emitted = generate(module)
    # Must parse without error
    ast.parse(emitted)


def test_roundtrip_json():
    source = (FIXTURES / "sample.py").read_text()
    module = parse(source, "sample.py")
    d = module.to_dict()
    json_str = json.dumps(d)
    # Deserialise and re-emit — must still be valid Python
    from src.unified_ast.nodes import Module
    module2 = Module.from_dict(json.loads(json_str))
    emitted = generate(module2)
    ast.parse(emitted)


def test_skeleton_has_no_bodies():
    source = (FIXTURES / "sample.py").read_text()
    module = parse(source, "sample.py")
    skel = build_skeleton(module)
    skel_str = json.dumps(skel)
    # Skeleton should not contain "body" key contents
    assert "body_stmts" in skel_str
    # No actual statement dicts should be in skeleton
    assert '"kind": "return"' not in skel_str
    assert '"kind": "assign"' not in skel_str


def test_get_node_returns_body():
    source = (FIXTURES / "sample.py").read_text()
    module = parse(source, "sample.py")
    node = get_node(module, "fn:fibonacci")
    assert node["kind"] == "function"
    assert "stmts" in node["body"]
    assert len(node["body"]["stmts"]) > 0


def test_str_replace_body():
    source = "def add(a: int, b: int) -> int:\n    return a + b\n"
    module = parse(source, "test.py")
    fn_node = get_node(module, "fn:add")
    body_stmts = fn_node["body"]["stmts"]
    assert len(body_stmts) == 1
    old_stmt = json.dumps(body_stmts[0])
    # Replace return a+b with return a-b
    new_stmt = json.dumps({
        "kind": "return",
        "value": {
            "kind": "binary_op",
            "left": {"kind": "identifier", "name": "a"},
            "op": "-",
            "right": {"kind": "identifier", "name": "b"},
        }
    })
    str_replace_body(module, "fn:add", old_stmt, new_stmt)
    emitted = generate(module)
    assert "a - b" in emitted
    ast.parse(emitted)


def test_rename_function():
    source = "def hello() -> None:\n    pass\n"
    module = parse(source, "test.py")
    rename_node(module, "fn:hello", "greet")
    emitted = generate(module)
    assert "def greet" in emitted
    assert "def hello" not in emitted
    ast.parse(emitted)


def test_imports():
    source = "import os\nfrom typing import List, Optional\n"
    module = parse(source, "test.py")
    emitted = generate(module)
    assert "import os" in emitted
    assert "from typing import" in emitted
    ast.parse(emitted)


def test_class_with_methods():
    source = textwrap.dedent("""\
        class Counter:
            value: int = 0
            def increment(self) -> None:
                self.value = self.value + 1
            def get(self) -> int:
                return self.value
    """)
    module = parse(source, "test.py")
    emitted = generate(module)
    ast.parse(emitted)
    assert "class Counter" in emitted
    assert "def increment" in emitted
    assert "def get" in emitted


def test_if_else():
    source = textwrap.dedent("""\
        def classify(x: int) -> str:
            if x < 0:
                return "negative"
            elif x == 0:
                return "zero"
            else:
                return "positive"
    """)
    module = parse(source, "test.py")
    emitted = generate(module)
    ast.parse(emitted)
    assert "elif" in emitted or "else" in emitted


def test_while_loop():
    source = textwrap.dedent("""\
        def countdown(n: int) -> int:
            result: int = 0
            while n > 0:
                result = result + n
                n = n - 1
            return result
    """)
    module = parse(source, "test.py")
    emitted = generate(module)
    ast.parse(emitted)
    assert "while" in emitted


def test_for_loop():
    source = textwrap.dedent("""\
        def total(items: list[int]) -> int:
            s: int = 0
            for x in items:
                s = s + x
            return s
    """)
    module = parse(source, "test.py")
    emitted = generate(module)
    ast.parse(emitted)
    assert "for" in emitted
