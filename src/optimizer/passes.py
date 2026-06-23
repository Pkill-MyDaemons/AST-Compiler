"""Optimization passes for the unified AST.

Each pass is a class with a single `run(module) -> module` method.
Passes are pure — they return new/modified nodes, never mutate in place.

Available passes
----------------
ConstantFoldingPass   — evaluate binary ops whose both operands are literals
IdentityEliminationPass — remove algebraic no-ops (x+0, x*1, True and x, …)
DeadCodeEliminationPass — drop statements that follow an unconditional exit
"""
from __future__ import annotations

import math
import operator
from typing import Callable, List, Optional

from ..unified_ast.expr import (
    Expr, Stmt, Block,
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess, Index,
    ListLiteral, DictLiteral, TupleLiteral, Lambda, Conditional, Await, Cast, RawExpr,
    VarDecl, Assign, Return, If, WhileLoop, ForEach, Match, MatchArm,
    Break, Continue, Raise, ExprStmt, Raw,
)
from ..unified_ast.nodes import Module, FunctionNode, TypeDefNode


ExprXform = Callable[[Expr], Expr]


# ---------------------------------------------------------------------------
# Generic tree-walk helpers
# ---------------------------------------------------------------------------

def _walk_expr(expr: Expr, xform: ExprXform) -> Expr:
    """Recursively apply xform bottom-up: children first, then the node itself."""
    if isinstance(expr, BinaryOp):
        expr = BinaryOp(left=_walk_expr(expr.left, xform), op=expr.op,
                        right=_walk_expr(expr.right, xform))
    elif isinstance(expr, UnaryOp):
        expr = UnaryOp(op=expr.op, operand=_walk_expr(expr.operand, xform))
    elif isinstance(expr, Call):
        expr = Call(
            func=_walk_expr(expr.func, xform),
            args=[_walk_expr(a, xform) for a in expr.args],
            kwargs={k: _walk_expr(v, xform) for k, v in expr.kwargs.items()},
        )
    elif isinstance(expr, FieldAccess):
        expr = FieldAccess(object=_walk_expr(expr.object, xform), field_name=expr.field_name)
    elif isinstance(expr, Index):
        expr = Index(object=_walk_expr(expr.object, xform),
                     index=_walk_expr(expr.index, xform))
    elif isinstance(expr, ListLiteral):
        expr = ListLiteral(elements=[_walk_expr(e, xform) for e in expr.elements])
    elif isinstance(expr, DictLiteral):
        expr = DictLiteral(pairs=[((_walk_expr(k, xform), _walk_expr(v, xform)))
                                  for k, v in expr.pairs])
    elif isinstance(expr, TupleLiteral):
        expr = TupleLiteral(elements=[_walk_expr(e, xform) for e in expr.elements])
    elif isinstance(expr, Lambda):
        expr = Lambda(params=expr.params, body=_walk_expr(expr.body, xform))
    elif isinstance(expr, Conditional):
        expr = Conditional(
            cond=_walk_expr(expr.cond, xform),
            then_expr=_walk_expr(expr.then_expr, xform),
            else_expr=_walk_expr(expr.else_expr, xform),
        )
    elif isinstance(expr, Await):
        expr = Await(expr=_walk_expr(expr.expr, xform))
    elif isinstance(expr, Cast):
        expr = Cast(expr=_walk_expr(expr.expr, xform), target_type=expr.target_type)
    return xform(expr)


def _walk_stmt(stmt: Stmt, xform_expr: ExprXform) -> Stmt:
    """Apply xform_expr to every expression inside a statement (recursive)."""
    if isinstance(stmt, Block):
        return Block(stmts=[_walk_stmt(s, xform_expr) for s in stmt.stmts])
    if isinstance(stmt, VarDecl):
        return VarDecl(
            name=stmt.name, type=stmt.type,
            value=_walk_expr(stmt.value, xform_expr) if stmt.value is not None else None,
            is_mutable=stmt.is_mutable,
        )
    if isinstance(stmt, Assign):
        return Assign(
            target=_walk_expr(stmt.target, xform_expr),
            op=stmt.op,
            value=_walk_expr(stmt.value, xform_expr),
        )
    if isinstance(stmt, Return):
        return Return(value=_walk_expr(stmt.value, xform_expr) if stmt.value is not None else None)
    if isinstance(stmt, If):
        return If(
            cond=_walk_expr(stmt.cond, xform_expr),
            then_block=_walk_stmt(stmt.then_block, xform_expr),
            elif_branches=[(_walk_expr(c, xform_expr), _walk_stmt(b, xform_expr))
                           for c, b in stmt.elif_branches],
            else_block=_walk_stmt(stmt.else_block, xform_expr) if stmt.else_block is not None else None,
        )
    if isinstance(stmt, WhileLoop):
        return WhileLoop(cond=_walk_expr(stmt.cond, xform_expr),
                         body=_walk_stmt(stmt.body, xform_expr))
    if isinstance(stmt, ForEach):
        return ForEach(var=stmt.var, iter_expr=_walk_expr(stmt.iter_expr, xform_expr),
                       body=_walk_stmt(stmt.body, xform_expr))
    if isinstance(stmt, Match):
        return Match(
            subject=_walk_expr(stmt.subject, xform_expr),
            arms=[MatchArm(
                pattern=a.pattern,
                guard=_walk_expr(a.guard, xform_expr) if a.guard else None,
                body=_walk_stmt(a.body, xform_expr),
            ) for a in stmt.arms],
        )
    if isinstance(stmt, Raise):
        return Raise(expr=_walk_expr(stmt.expr, xform_expr) if stmt.expr is not None else None)
    if isinstance(stmt, ExprStmt):
        return ExprStmt(expr=_walk_expr(stmt.expr, xform_expr),
                        is_implicit_return=stmt.is_implicit_return)
    return stmt  # Break, Continue, Raw — no expressions


def _apply_expr_xform(module: Module, xform: ExprXform) -> Module:
    """Run an expression transform across every function body in the module."""
    new_nodes = []
    for node in module.nodes:
        if isinstance(node, FunctionNode):
            new_nodes.append(_transform_fn(node, xform))
        elif isinstance(node, TypeDefNode):
            new_nodes.append(_transform_typedef(node, xform))
        else:
            new_nodes.append(node)
    return Module(version=module.version, source_language=module.source_language,
                  source_file=module.source_file, nodes=new_nodes)


def _transform_fn(fn: FunctionNode, xform: ExprXform) -> FunctionNode:
    return FunctionNode(
        id=fn.id, name=fn.name, params=fn.params, return_type=fn.return_type,
        body=_walk_stmt(fn.body, xform),
        visibility=fn.visibility, is_async=fn.is_async, is_static=fn.is_static,
        is_constructor=fn.is_constructor, is_abstract=fn.is_abstract,
        decorators=fn.decorators, type_params=fn.type_params,
        docstring=fn.docstring, attributes=fn.attributes,
    )


def _transform_typedef(td: TypeDefNode, xform: ExprXform) -> TypeDefNode:
    return TypeDefNode(
        id=td.id, name=td.name, category=td.category, visibility=td.visibility,
        bases=td.bases, interfaces=td.interfaces, type_params=td.type_params,
        fields=td.fields,
        methods=[_transform_fn(m, xform) for m in td.methods],
        inner_types=[_transform_typedef(it, xform) for it in td.inner_types],
        docstring=td.docstring, attributes=td.attributes,
    )


# ---------------------------------------------------------------------------
# Pass 1 — Constant Folding
# ---------------------------------------------------------------------------

_ARITH_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "<<": operator.lshift,
    ">>": operator.rshift,
}

_CMP_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _fold_one(expr: Expr) -> Expr:
    if not isinstance(expr, BinaryOp):
        return expr
    left, right = expr.left, expr.right
    if not (isinstance(left, Literal) and isinstance(right, Literal)):
        return expr
    lv, rv, op = left.value, right.value, expr.op
    if lv is None or rv is None:
        return expr

    try:
        if op in _ARITH_OPS:
            if op in ("/", "//", "%") and rv == 0:
                return expr  # don't fold division by zero
            if op == "**" and (isinstance(rv, (int, float)) and rv > 512):
                return expr  # avoid huge constants
            result = _ARITH_OPS[op](lv, rv)
            if isinstance(result, bool):
                return Literal(value=result, lit_kind="bool")
            if isinstance(result, int):
                return Literal(value=result, lit_kind="int")
            if isinstance(result, float):
                if math.isnan(result) or math.isinf(result):
                    return expr
                return Literal(value=result, lit_kind="float")
            if isinstance(result, str):
                return Literal(value=result, lit_kind="string")
        elif op in _CMP_OPS:
            result = _CMP_OPS[op](lv, rv)
            return Literal(value=bool(result), lit_kind="bool")
        elif op == "&&":
            result = lv and rv
            if isinstance(result, bool):
                return Literal(value=result, lit_kind="bool")
        elif op == "||":
            result = lv or rv
            if isinstance(result, bool):
                return Literal(value=result, lit_kind="bool")
    except Exception:
        pass
    return expr


class ConstantFoldingPass:
    """Evaluate binary operations whose both operands are compile-time literals.

    Examples:
        500 * 0.9        → 450.0
        500 - 40         → 460
        "ab" + "cd"      → "abcd"
        2 ** 10          → 1024
        3 < 5            → True
    """

    def run(self, module: Module) -> Module:
        return _apply_expr_xform(module, _fold_one)


# ---------------------------------------------------------------------------
# Pass 2 — Algebraic Identity Elimination
# ---------------------------------------------------------------------------

def _is_literal(expr: Expr, value) -> bool:
    return isinstance(expr, Literal) and expr.value == value


def _identity_one(expr: Expr) -> Expr:
    if not isinstance(expr, BinaryOp):
        return expr
    l, op, r = expr.left, expr.op, expr.right

    if op == "+":
        if _is_literal(r, 0): return l
        if _is_literal(l, 0): return r
    elif op == "-":
        if _is_literal(r, 0): return l
    elif op == "*":
        if _is_literal(r, 1): return l
        if _is_literal(l, 1): return r
        if _is_literal(r, 0): return Literal(value=0, lit_kind="int")
        if _is_literal(l, 0): return Literal(value=0, lit_kind="int")
    elif op == "/":
        if _is_literal(r, 1): return l
    elif op == "//":
        if _is_literal(r, 1): return l
    elif op == "**":
        if _is_literal(r, 1): return l
        if _is_literal(r, 0): return Literal(value=1, lit_kind="int")
        if _is_literal(l, 0): return Literal(value=0, lit_kind="int")
        if _is_literal(l, 1): return Literal(value=1, lit_kind="int")
    elif op == "|":
        if _is_literal(r, 0): return l
        if _is_literal(l, 0): return r
    elif op == "^":
        if _is_literal(r, 0): return l
        if _is_literal(l, 0): return r
    elif op == "&&":
        # True and x → x,  False and x → False
        if isinstance(l, Literal) and l.lit_kind == "bool":
            return r if l.value else Literal(value=False, lit_kind="bool")
        if isinstance(r, Literal) and r.lit_kind == "bool":
            return l if r.value else Literal(value=False, lit_kind="bool")
    elif op == "||":
        # False or x → x,  True or x → True
        if isinstance(l, Literal) and l.lit_kind == "bool":
            return r if not l.value else Literal(value=True, lit_kind="bool")
        if isinstance(r, Literal) and r.lit_kind == "bool":
            return l if not r.value else Literal(value=True, lit_kind="bool")

    return expr


class IdentityEliminationPass:
    """Remove algebraic no-ops: x+0, x*1, x**1, x*0, True and x, etc."""

    def run(self, module: Module) -> Module:
        return _apply_expr_xform(module, _identity_one)


# ---------------------------------------------------------------------------
# Pass 3 — Dead Code Elimination
# ---------------------------------------------------------------------------

def _is_exit(stmt: Stmt) -> bool:
    return isinstance(stmt, (Return, Break, Continue, Raise))


def _dce_block(block: Block) -> Block:
    new_stmts: List[Stmt] = []
    for stmt in block.stmts:
        stmt = _dce_stmt(stmt)
        new_stmts.append(stmt)
        if _is_exit(stmt):
            break  # everything after is unreachable
    return Block(stmts=new_stmts)


def _dce_stmt(stmt: Stmt) -> Stmt:
    if isinstance(stmt, Block):
        return _dce_block(stmt)
    if isinstance(stmt, If):
        return If(
            cond=stmt.cond,
            then_block=_dce_block(stmt.then_block),
            elif_branches=[(c, _dce_block(b)) for c, b in stmt.elif_branches],
            else_block=_dce_block(stmt.else_block) if stmt.else_block is not None else None,
        )
    if isinstance(stmt, WhileLoop):
        return WhileLoop(cond=stmt.cond, body=_dce_block(stmt.body))
    if isinstance(stmt, ForEach):
        return ForEach(var=stmt.var, iter_expr=stmt.iter_expr, body=_dce_block(stmt.body))
    if isinstance(stmt, Match):
        return Match(
            subject=stmt.subject,
            arms=[MatchArm(pattern=a.pattern, guard=a.guard, body=_dce_block(a.body))
                  for a in stmt.arms],
        )
    return stmt


def _dce_module(module: Module) -> Module:
    new_nodes = []
    for node in module.nodes:
        if isinstance(node, FunctionNode):
            fn = node
            new_nodes.append(FunctionNode(
                id=fn.id, name=fn.name, params=fn.params, return_type=fn.return_type,
                body=_dce_block(fn.body),
                visibility=fn.visibility, is_async=fn.is_async, is_static=fn.is_static,
                is_constructor=fn.is_constructor, is_abstract=fn.is_abstract,
                decorators=fn.decorators, type_params=fn.type_params,
                docstring=fn.docstring, attributes=fn.attributes,
            ))
        elif isinstance(node, TypeDefNode):
            td = node
            new_methods = []
            for m in td.methods:
                new_methods.append(FunctionNode(
                    id=m.id, name=m.name, params=m.params, return_type=m.return_type,
                    body=_dce_block(m.body),
                    visibility=m.visibility, is_async=m.is_async, is_static=m.is_static,
                    is_constructor=m.is_constructor, is_abstract=m.is_abstract,
                    decorators=m.decorators, type_params=m.type_params,
                    docstring=m.docstring, attributes=m.attributes,
                ))
            new_nodes.append(TypeDefNode(
                id=td.id, name=td.name, category=td.category, visibility=td.visibility,
                bases=td.bases, interfaces=td.interfaces, type_params=td.type_params,
                fields=td.fields, methods=new_methods, inner_types=td.inner_types,
                docstring=td.docstring, attributes=td.attributes,
            ))
        else:
            new_nodes.append(node)
    return Module(version=module.version, source_language=module.source_language,
                  source_file=module.source_file, nodes=new_nodes)


class DeadCodeEliminationPass:
    """Remove statements that follow an unconditional exit (return/break/continue/raise)."""

    def run(self, module: Module) -> Module:
        return _dce_module(module)
