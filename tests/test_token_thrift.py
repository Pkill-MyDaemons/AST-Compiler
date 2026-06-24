"""Round-trip validation tests for min-json and sexpr output formats."""
from __future__ import annotations
import json
import textwrap

import pytest

from src.parsers.python_parser import parse as py_parse
from src.unified_ast.minify import minify
from src.unified_ast.sexpr import sexpr


COMPLEX_SOURCE = textwrap.dedent("""\
    import os
    from typing import List, Optional

    MAX_VAL: int = 100

    class Node:
        value: int
        children: List

        def __init__(self, value: int) -> None:
            self.value = value
            self.children = []

        def add_child(self, child) -> None:
            self.children.append(child)

        def depth(self) -> int:
            if not self.children:
                return 0
            best: int = 0
            for c in self.children:
                d = c.depth()
                if d > best:
                    best = d
            return best + 1

    def fibonacci(n: int) -> int:
        if n <= 1:
            return n
        a: int = 0
        b: int = 1
        i: int = 2
        while i <= n:
            temp: int = a + b
            a = b
            b = temp
            i = i + 1
        return b
""")


@pytest.fixture
def ast_dict():
    module = py_parse(COMPLEX_SOURCE, "test.py")
    return module.to_dict()


# ---------------------------------------------------------------------------
# min-json
# ---------------------------------------------------------------------------

class TestMinJson:
    def test_no_crash(self, ast_dict):
        result = minify(ast_dict)
        assert isinstance(result, str) and len(result) > 0

    def test_valid_json(self, ast_dict):
        parsed = json.loads(minify(ast_dict))
        assert isinstance(parsed, dict)

    def test_root_metadata(self, ast_dict):
        root = json.loads(minify(ast_dict))
        assert root["lang"] == "python"
        assert root["file"] == "test.py"
        assert "nodes" in root

    def test_no_whitespace_separators(self, ast_dict):
        result = minify(ast_dict)
        assert ", " not in result
        assert ": " not in result

    def test_node_count_preserved(self, ast_dict):
        root = json.loads(minify(ast_dict))
        assert len(root["nodes"]) == len(ast_dict["nodes"])

    def test_verbose_keys_absent(self, ast_dict):
        result = minify(ast_dict)
        assert '"kind"' not in result
        assert '"body"' not in result
        assert '"stmts"' not in result

    def test_kind_values_compressed(self, ast_dict):
        result = minify(ast_dict)
        assert '"import"' not in result
        assert '"function"' not in result
        assert '"imp"' in result
        assert '"fn"' in result

    def test_expr_stmt_compressed(self, ast_dict):
        result = minify(ast_dict)
        assert '"expr_stmt"' not in result
        assert '"es"' in result

    def test_smaller_than_verbose(self, ast_dict):
        verbose = json.dumps(ast_dict, separators=(",", ":"))
        assert len(minify(ast_dict)) < len(verbose)

    def test_function_nodes_have_body_key(self, ast_dict):
        root = json.loads(minify(ast_dict))
        fn_nodes = [n for n in root["nodes"] if n.get("k") == "fn"]
        assert len(fn_nodes) > 0
        for fn in fn_nodes:
            assert "b" in fn, "function node must have compressed body key 'b'"

    def test_import_nodes_have_module_key(self, ast_dict):
        root = json.loads(minify(ast_dict))
        imp_nodes = [n for n in root["nodes"] if n.get("k") == "imp"]
        assert len(imp_nodes) > 0
        for imp in imp_nodes:
            assert "m" in imp, "import node must have compressed module key 'm'"

    def test_empty_module(self):
        module = py_parse("", "empty.py")
        root = json.loads(minify(module.to_dict()))
        assert root["lang"] == "python"
        assert root["nodes"] == []

    def test_nested_call_args_compressed(self, ast_dict):
        result = minify(ast_dict)
        # args key should be compressed to "a"
        assert '"args"' not in result
        # func key should be compressed to "f"
        assert '"func"' not in result


# ---------------------------------------------------------------------------
# sexpr
# ---------------------------------------------------------------------------

class TestSexpr:
    def test_no_crash(self, ast_dict):
        result = sexpr(ast_dict)
        assert isinstance(result, str) and len(result) > 0

    def test_metadata_first_line(self, ast_dict):
        first_line = sexpr(ast_dict).splitlines()[0]
        assert first_line.startswith("(meta")
        assert 'lang:"python"' in first_line
        assert 'file:"test.py"' in first_line

    def test_import_node_present(self, ast_dict):
        assert "(imp " in sexpr(ast_dict)

    def test_function_node_present(self, ast_dict):
        assert "(fn " in sexpr(ast_dict)

    def test_node_lines_match_count(self, ast_dict):
        lines = [l for l in sexpr(ast_dict).splitlines() if l.strip()]
        # meta line + one S-expression per top-level node
        assert len(lines) == 1 + len(ast_dict["nodes"])

    def test_all_top_level_lines_are_sexpr(self, ast_dict):
        lines = sexpr(ast_dict).splitlines()
        for line in lines[1:]:
            assert line.startswith("("), f"expected s-expr line, got: {line!r}"

    def test_smaller_than_verbose(self, ast_dict):
        assert len(sexpr(ast_dict)) < len(json.dumps(ast_dict))

    def test_quote_escaping(self):
        source = 'def f():\n    return "hello world"\n'
        module = py_parse(source, "test.py")
        result = sexpr(module.to_dict())
        assert isinstance(result, str)
        # Ensure no literal unescaped bare double-quote breaks line structure
        for line in result.splitlines():
            assert line.count('"') % 2 == 0, f"unbalanced quotes on line: {line!r}"

    def test_class_node_in_output(self, ast_dict):
        result = sexpr(ast_dict)
        assert "class" in result or "Node" in result

    def test_empty_module(self):
        module = py_parse("", "empty.py")
        result = sexpr(module.to_dict())
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0].startswith("(meta")

    def test_function_body_present(self, ast_dict):
        result = sexpr(ast_dict)
        assert "(body" in result

    def test_fibonacci_name_in_output(self, ast_dict):
        assert "fibonacci" in sexpr(ast_dict)
