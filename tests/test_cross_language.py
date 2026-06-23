"""Cross-language tests: parse Python → emit Rust and vice versa."""
import ast
import textwrap

import pytest


def _parse_python(source, filename="test.py"):
    from src.parsers.python_parser import parse
    return parse(source, filename)


def _gen_rust(module):
    from src.generators.rust_gen import generate
    return generate(module)


def _gen_python(module):
    from src.generators.python_gen import generate
    return generate(module)


def test_python_function_to_rust():
    source = textwrap.dedent("""\
        def add(a: int, b: int) -> int:
            return a + b
    """)
    module = _parse_python(source)
    rust = _gen_rust(module)
    assert "fn add" in rust
    assert "i64" in rust or "_" in rust
    assert "return" in rust or "a + b" in rust


def test_python_class_to_rust_struct():
    source = textwrap.dedent("""\
        class Point:
            x: float
            y: float
            def magnitude(self) -> float:
                return self.x * self.x + self.y * self.y
    """)
    module = _parse_python(source)
    rust = _gen_rust(module)
    assert "struct Point" in rust
    assert "fn magnitude" in rust


def test_python_if_to_rust():
    source = textwrap.dedent("""\
        def classify(n: int) -> int:
            if n > 0:
                return 1
            elif n < 0:
                return -1
            else:
                return 0
    """)
    module = _parse_python(source)
    rust = _gen_rust(module)
    assert "if" in rust
    assert "else" in rust
    assert "return" in rust


def test_python_for_to_rust():
    source = textwrap.dedent("""\
        def sum_list(items: list[int]) -> int:
            total: int = 0
            for x in items:
                total = total + x
            return total
    """)
    module = _parse_python(source)
    rust = _gen_rust(module)
    assert "for" in rust


def test_python_while_to_rust():
    source = textwrap.dedent("""\
        def repeat(n: int) -> int:
            result: int = 1
            count: int = 0
            while count < n:
                result = result * 2
                count = count + 1
            return result
    """)
    module = _parse_python(source)
    rust = _gen_rust(module)
    assert "while" in rust or "loop" in rust


def _has_tree_sitter_rust():
    try:
        import tree_sitter_rust
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_tree_sitter_rust(), reason="tree-sitter-rust not installed")
def test_rust_function_to_python():
    source = textwrap.dedent("""\
        pub fn add(a: i64, b: i64) -> i64 {
            a + b
        }
    """)
    from src.parsers.rust_parser import parse
    module = parse(source, "test.rs")
    python = _gen_python(module)
    # Must be valid Python
    ast.parse(python)
    assert "def add" in python


@pytest.mark.skipif(not _has_tree_sitter_rust(), reason="tree-sitter-rust not installed")
def test_rust_struct_to_python_class():
    source = textwrap.dedent("""\
        pub struct Counter {
            pub value: i64,
        }
        impl Counter {
            pub fn new() -> Self {
                Counter { value: 0 }
            }
            pub fn increment(&mut self) {
                self.value = self.value + 1;
            }
            pub fn get(&self) -> i64 {
                self.value
            }
        }
    """)
    from src.parsers.rust_parser import parse
    module = parse(source, "test.rs")
    python = _gen_python(module)
    ast.parse(python)
    assert "class Counter" in python
    assert "def increment" in python
    assert "def get" in python
