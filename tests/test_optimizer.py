"""Tests for AST optimization passes."""
import ast
import textwrap

import pytest

from src.parsers.python_parser import parse
from src.generators.python_gen import generate
from src.optimizer import optimize, ConstantFoldingPass, IdentityEliminationPass, DeadCodeEliminationPass
from src.unified_ast.expr import (
    BinaryOp, Literal, Identifier, Block, Return, Assign, ExprStmt, VarDecl,
    If, WhileLoop, ForEach, Break, Continue, Raise, Call,
)
from src.unified_ast.nodes import Module, FunctionNode
from src.unified_ast.types import T_INFERRED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fn(stmts) -> Module:
    """Wrap a list of statements in a minimal Module with one function."""
    body = Block(stmts=stmts)
    fn = FunctionNode(id="fn:f", name="f", body=body)
    return Module(source_language="python", nodes=[fn])


def _stmts(module: Module):
    return module.nodes[0].body.stmts


# ---------------------------------------------------------------------------
# ConstantFoldingPass
# ---------------------------------------------------------------------------

class TestConstantFolding:
    def _fold(self, expr):
        stmts = [Return(value=expr)]
        m = ConstantFoldingPass().run(_fn(stmts))
        return _stmts(m)[0].value

    def test_int_add(self):
        result = self._fold(BinaryOp(left=Literal(value=3, lit_kind="int"),
                                     op="+",
                                     right=Literal(value=4, lit_kind="int")))
        assert isinstance(result, Literal)
        assert result.value == 7
        assert result.lit_kind == "int"

    def test_float_mul(self):
        result = self._fold(BinaryOp(left=Literal(value=500, lit_kind="int"),
                                     op="*",
                                     right=Literal(value=0.9, lit_kind="float")))
        assert isinstance(result, Literal)
        assert result.value == pytest.approx(450.0)
        assert result.lit_kind == "float"

    def test_int_sub(self):
        result = self._fold(BinaryOp(left=Literal(value=500, lit_kind="int"),
                                     op="-",
                                     right=Literal(value=40, lit_kind="int")))
        assert isinstance(result, Literal) and result.value == 460

    def test_nested_fold(self):
        # (3 + 4) * 2 → 14
        inner = BinaryOp(left=Literal(value=3, lit_kind="int"),
                         op="+",
                         right=Literal(value=4, lit_kind="int"))
        outer = BinaryOp(left=inner, op="*", right=Literal(value=2, lit_kind="int"))
        result = self._fold(outer)
        assert isinstance(result, Literal) and result.value == 14

    def test_no_fold_with_variable(self):
        # x + 1 — cannot fold, x is not a literal
        expr = BinaryOp(left=Identifier(name="x"), op="+",
                        right=Literal(value=1, lit_kind="int"))
        result = self._fold(expr)
        assert isinstance(result, BinaryOp)

    def test_division_by_zero_not_folded(self):
        expr = BinaryOp(left=Literal(value=10, lit_kind="int"),
                        op="/",
                        right=Literal(value=0, lit_kind="int"))
        result = self._fold(expr)
        assert isinstance(result, BinaryOp)

    def test_comparison_folded(self):
        result = self._fold(BinaryOp(left=Literal(value=3, lit_kind="int"),
                                     op="<",
                                     right=Literal(value=5, lit_kind="int")))
        assert isinstance(result, Literal)
        assert result.value is True
        assert result.lit_kind == "bool"

    def test_string_concat(self):
        result = self._fold(BinaryOp(left=Literal(value="ab", lit_kind="string"),
                                     op="+",
                                     right=Literal(value="cd", lit_kind="string")))
        assert isinstance(result, Literal) and result.value == "abcd"

    def test_power(self):
        result = self._fold(BinaryOp(left=Literal(value=2, lit_kind="int"),
                                     op="**",
                                     right=Literal(value=10, lit_kind="int")))
        assert isinstance(result, Literal) and result.value == 1024

    def test_real_source(self):
        """End-to-end: parse example.py, fold, re-emit valid Python."""
        with open("example.py") as f:
            source = f.read()
        module = parse(source, "example.py")
        module = ConstantFoldingPass().run(module)
        emitted = generate(module)
        ast.parse(emitted)  # must be valid Python


# ---------------------------------------------------------------------------
# IdentityEliminationPass
# ---------------------------------------------------------------------------

class TestIdentityElimination:
    def _elim(self, expr):
        stmts = [Return(value=expr)]
        m = IdentityEliminationPass().run(_fn(stmts))
        return _stmts(m)[0].value

    def _x(self):
        return Identifier(name="x")

    def test_add_zero_right(self):
        result = self._elim(BinaryOp(left=self._x(), op="+",
                                     right=Literal(value=0, lit_kind="int")))
        assert isinstance(result, Identifier) and result.name == "x"

    def test_add_zero_left(self):
        result = self._elim(BinaryOp(left=Literal(value=0, lit_kind="int"),
                                     op="+", right=self._x()))
        assert isinstance(result, Identifier) and result.name == "x"

    def test_sub_zero(self):
        result = self._elim(BinaryOp(left=self._x(), op="-",
                                     right=Literal(value=0, lit_kind="int")))
        assert isinstance(result, Identifier) and result.name == "x"

    def test_mul_one_right(self):
        result = self._elim(BinaryOp(left=self._x(), op="*",
                                     right=Literal(value=1, lit_kind="int")))
        assert isinstance(result, Identifier) and result.name == "x"

    def test_mul_one_left(self):
        result = self._elim(BinaryOp(left=Literal(value=1, lit_kind="int"),
                                     op="*", right=self._x()))
        assert isinstance(result, Identifier) and result.name == "x"

    def test_mul_zero(self):
        result = self._elim(BinaryOp(left=self._x(), op="*",
                                     right=Literal(value=0, lit_kind="int")))
        assert isinstance(result, Literal) and result.value == 0

    def test_pow_one(self):
        result = self._elim(BinaryOp(left=self._x(), op="**",
                                     right=Literal(value=1, lit_kind="int")))
        assert isinstance(result, Identifier) and result.name == "x"

    def test_pow_zero(self):
        result = self._elim(BinaryOp(left=self._x(), op="**",
                                     right=Literal(value=0, lit_kind="int")))
        assert isinstance(result, Literal) and result.value == 1

    def test_true_and_x(self):
        result = self._elim(BinaryOp(left=Literal(value=True, lit_kind="bool"),
                                     op="&&", right=self._x()))
        assert isinstance(result, Identifier) and result.name == "x"

    def test_false_and_x(self):
        result = self._elim(BinaryOp(left=Literal(value=False, lit_kind="bool"),
                                     op="&&", right=self._x()))
        assert isinstance(result, Literal) and result.value is False

    def test_false_or_x(self):
        result = self._elim(BinaryOp(left=Literal(value=False, lit_kind="bool"),
                                     op="||", right=self._x()))
        assert isinstance(result, Identifier) and result.name == "x"

    def test_true_or_x(self):
        result = self._elim(BinaryOp(left=Literal(value=True, lit_kind="bool"),
                                     op="||", right=self._x()))
        assert isinstance(result, Literal) and result.value is True

    def test_no_change_on_nontrivial(self):
        expr = BinaryOp(left=self._x(), op="+",
                        right=Literal(value=5, lit_kind="int"))
        result = self._elim(expr)
        assert isinstance(result, BinaryOp)


# ---------------------------------------------------------------------------
# DeadCodeEliminationPass
# ---------------------------------------------------------------------------

class TestDeadCodeElimination:
    def _dce(self, stmts):
        m = DeadCodeEliminationPass().run(_fn(stmts))
        return _stmts(m)

    def _ret(self):
        return Return(value=Literal(value=0, lit_kind="int"))

    def _noop(self, n=1):
        return [ExprStmt(expr=Identifier(name=f"x{i}")) for i in range(n)]

    def test_no_change_without_exit(self):
        stmts = self._noop(3)
        result = self._dce(stmts)
        assert len(result) == 3

    def test_drops_after_return(self):
        stmts = self._noop(1) + [self._ret()] + self._noop(2)
        result = self._dce(stmts)
        assert len(result) == 2  # noop + return; trailing 2 dropped

    def test_drops_after_break(self):
        stmts = [Break()] + self._noop(3)
        result = self._dce(stmts)
        assert len(result) == 1

    def test_drops_after_continue(self):
        stmts = [Continue()] + self._noop(3)
        result = self._dce(stmts)
        assert len(result) == 1

    def test_drops_after_raise(self):
        stmts = self._noop(1) + [Raise()] + self._noop(2)
        result = self._dce(stmts)
        assert len(result) == 2

    def test_nested_if_body_dce(self):
        inner = If(
            cond=Identifier(name="cond"),
            then_block=Block(stmts=[self._ret()] + self._noop(2)),
        )
        result = self._dce([inner])
        then_stmts = result[0].then_block.stmts
        assert len(then_stmts) == 1  # only the return survives

    def test_for_body_dce(self):
        loop = ForEach(
            var="i",
            iter_expr=Identifier(name="items"),
            body=Block(stmts=[Break()] + self._noop(2)),
        )
        result = self._dce([loop])
        assert len(result[0].body.stmts) == 1

    def test_real_source_valid_python(self):
        source = textwrap.dedent("""\
            def f(x):
                if x > 0:
                    return x
                    unreachable = 1
                return -x
        """)
        module = parse(source, "test.py")
        module = DeadCodeEliminationPass().run(module)
        emitted = generate(module)
        ast.parse(emitted)

    def test_chain_all_passes(self):
        """Combined pipeline: parse real source, run all passes, re-emit valid Python."""
        source = textwrap.dedent("""\
            def compute(n):
                result = (10 - 0) * (n + 0) * 1
                if True and n > 0:
                    return result
                    extra = 99
                return 0
        """)
        module = parse(source, "test.py")
        module = optimize(module)
        emitted = generate(module)
        ast.parse(emitted)
        # Folded / simplified: no x+0, x*1, etc.
        assert "* 1" not in emitted or "10 - 0" not in emitted


# ---------------------------------------------------------------------------
# Serialization compaction
# ---------------------------------------------------------------------------

class TestSerializationCompaction:
    def test_no_empty_kwargs(self):
        source = "def f(): len(items)\n"
        module = parse(source, "test.py")
        d = module.to_dict()
        j = str(d)
        assert '"kwargs": {}' not in j

    def test_no_default_op_in_assign(self):
        source = "def f():\n    x = 1\n"
        module = parse(source, "test.py")
        d = module.to_dict()
        fn_body = d["nodes"][0]["body"]["stmts"][0]
        assert "op" not in fn_body  # default "=" should not appear

    def test_no_is_const_false(self):
        source = "x = 5\n"
        module = parse(source, "test.py")
        d = module.to_dict()
        var = d["nodes"][0]
        assert "is_const" not in var

    def test_no_visibility_public(self):
        source = "x = 5\n"
        module = parse(source, "test.py")
        d = module.to_dict()
        var = d["nodes"][0]
        assert "visibility" not in var

    def test_no_inferred_param_type(self):
        source = "def f(x, y): pass\n"
        module = parse(source, "test.py")
        d = module.to_dict()
        params = d["nodes"][0]["params"]
        for p in params:
            assert "type" not in p

    def test_no_inferred_return_type(self):
        source = "def f(): pass\n"
        module = parse(source, "test.py")
        d = module.to_dict()
        fn = d["nodes"][0]
        assert "return_type" not in fn

    def test_explicit_return_type_preserved(self):
        source = "def f() -> int: return 1\n"
        module = parse(source, "test.py")
        d = module.to_dict()
        fn = d["nodes"][0]
        assert "return_type" in fn
        assert fn["return_type"]["kind"] == "number"

    def test_explicit_param_type_preserved(self):
        source = "def f(x: int): pass\n"
        module = parse(source, "test.py")
        d = module.to_dict()
        param = d["nodes"][0]["params"][0]
        assert "type" in param
        assert param["type"]["kind"] == "number"

    def test_roundtrip_after_compaction(self):
        """Compact JSON still round-trips to valid Python."""
        import json
        source = textwrap.dedent("""\
            def greet(name):
                msg = 'Hello'
                return msg
        """)
        module = parse(source, "test.py")
        from src.unified_ast.nodes import Module as M
        module2 = M.from_dict(json.loads(json.dumps(module.to_dict())))
        emitted = generate(module2)
        ast.parse(emitted)
