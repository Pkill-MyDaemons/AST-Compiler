"""Unified expression and statement nodes for function bodies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .types import UnifiedType, T_INFERRED


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class Literal:
    kind: str = "literal"
    value: Any = None
    lit_kind: str = "int"  # int | float | string | bool | none

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind, "value": self.value}
        if self.lit_kind != "int":
            d["lit_kind"] = self.lit_kind
        return d


@dataclass
class Identifier:
    kind: str = "identifier"
    name: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name}


@dataclass
class BinaryOp:
    kind: str = "binary_op"
    left: Expr = None
    op: str = "+"
    right: Expr = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "left": self.left.to_dict(), "op": self.op, "right": self.right.to_dict()}


@dataclass
class UnaryOp:
    kind: str = "unary_op"
    op: str = "-"
    operand: Expr = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "op": self.op, "operand": self.operand.to_dict()}


@dataclass
class Call:
    kind: str = "call"
    func: Expr = None
    args: List[Expr] = field(default_factory=list)
    kwargs: Dict[str, Expr] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind, "func": self.func.to_dict()}
        if self.args:
            d["args"] = [a.to_dict() for a in self.args]
        if self.kwargs:
            d["kwargs"] = {k: v.to_dict() for k, v in self.kwargs.items()}
        return d


@dataclass
class FieldAccess:
    kind: str = "field_access"
    object: Expr = None
    field_name: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "object": self.object.to_dict(), "field": self.field_name}


@dataclass
class Index:
    kind: str = "index"
    object: Expr = None
    index: Expr = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "object": self.object.to_dict(), "index": self.index.to_dict()}


@dataclass
class ListLiteral:
    kind: str = "list_literal"
    elements: List[Expr] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "elements": [e.to_dict() for e in self.elements]}


@dataclass
class DictLiteral:
    kind: str = "dict_literal"
    pairs: List[tuple] = field(default_factory=list)  # List[(Expr, Expr)]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "pairs": [[k.to_dict(), v.to_dict()] for k, v in self.pairs]}


@dataclass
class TupleLiteral:
    kind: str = "tuple_literal"
    elements: List[Expr] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "elements": [e.to_dict() for e in self.elements]}


@dataclass
class Lambda:
    kind: str = "lambda"
    params: List[str] = field(default_factory=list)
    body: Expr = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": self.params, "body": self.body.to_dict()}


@dataclass
class Conditional:
    kind: str = "conditional"
    cond: Expr = None
    then_expr: Expr = None
    else_expr: Expr = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "cond": self.cond.to_dict(),
            "then": self.then_expr.to_dict(),
            "else": self.else_expr.to_dict(),
        }


@dataclass
class Await:
    kind: str = "await"
    expr: Expr = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "expr": self.expr.to_dict()}


@dataclass
class Cast:
    kind: str = "cast"
    expr: Expr = None
    target_type: UnifiedType = field(default_factory=lambda: T_INFERRED)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "expr": self.expr.to_dict(), "type": self.target_type.to_dict()}


@dataclass
class RawExpr:
    """Fallback for expressions we can't fully normalize."""
    kind: str = "raw_expr"
    text: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text}


Expr = Union[
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess, Index,
    ListLiteral, DictLiteral, TupleLiteral, Lambda, Conditional,
    Await, Cast, RawExpr,
]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class VarDecl:
    kind: str = "var_decl"
    name: str = ""
    type: UnifiedType = field(default_factory=lambda: T_INFERRED)
    value: Optional[Expr] = None
    is_mutable: bool = True

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind, "name": self.name}
        if self.type.kind.value != "inferred":
            d["type"] = self.type.to_dict()
        if self.value is not None:
            d["value"] = self.value.to_dict()
        if not self.is_mutable:
            d["is_mutable"] = False
        return d


@dataclass
class Assign:
    kind: str = "assign"
    target: Expr = None
    op: str = "="  # =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=
    value: Expr = None

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind, "target": self.target.to_dict(), "value": self.value.to_dict()}
        if self.op != "=":
            d["op"] = self.op
        return d


@dataclass
class Return:
    kind: str = "return"
    value: Optional[Expr] = None

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind}
        if self.value is not None:
            d["value"] = self.value.to_dict()
        return d


@dataclass
class If:
    kind: str = "if"
    cond: Expr = None
    then_block: Block = None
    elif_branches: List[tuple] = field(default_factory=list)  # List[(Expr, Block)]
    else_block: Optional[Block] = None

    def to_dict(self) -> dict:
        d: dict = {
            "kind": self.kind,
            "cond": self.cond.to_dict(),
            "then": self.then_block.to_dict(),
        }
        if self.elif_branches:
            d["elif"] = [[c.to_dict(), b.to_dict()] for c, b in self.elif_branches]
        if self.else_block is not None:
            d["else"] = self.else_block.to_dict()
        return d


@dataclass
class WhileLoop:
    kind: str = "while"
    cond: Expr = None
    body: Block = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "cond": self.cond.to_dict(), "body": self.body.to_dict()}


@dataclass
class ForEach:
    kind: str = "for_each"
    var: str = ""
    iter_expr: Expr = None
    body: Block = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "var": self.var, "iter": self.iter_expr.to_dict(), "body": self.body.to_dict()}


@dataclass
class MatchArm:
    pattern: str = ""  # stored as text — patterns are highly language-specific
    guard: Optional[Expr] = None
    body: Block = None

    def to_dict(self) -> dict:
        d: dict = {"pattern": self.pattern, "body": self.body.to_dict()}
        if self.guard is not None:
            d["guard"] = self.guard.to_dict()
        return d


@dataclass
class Match:
    kind: str = "match"
    subject: Expr = None
    arms: List[MatchArm] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "subject": self.subject.to_dict(), "arms": [a.to_dict() for a in self.arms]}


@dataclass
class Break:
    kind: str = "break"

    def to_dict(self) -> dict:
        return {"kind": self.kind}


@dataclass
class Continue:
    kind: str = "continue"

    def to_dict(self) -> dict:
        return {"kind": self.kind}


@dataclass
class Raise:
    kind: str = "raise"
    expr: Optional[Expr] = None

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind}
        if self.expr is not None:
            d["expr"] = self.expr.to_dict()
        return d


@dataclass
class ExprStmt:
    kind: str = "expr_stmt"
    expr: Expr = None
    # Rust implicit returns: last expr in block without semicolon
    is_implicit_return: bool = False

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind, "expr": self.expr.to_dict()}
        if self.is_implicit_return:
            d["is_implicit_return"] = True
        return d


@dataclass
class Raw:
    """Fallback statement — verbatim source text."""
    kind: str = "raw"
    text: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text}


Stmt = Union[VarDecl, Assign, Return, If, WhileLoop, ForEach, Match, Break, Continue, Raise, ExprStmt, Raw, "Block"]


@dataclass
class Block:
    kind: str = "block"
    stmts: List[Stmt] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"stmts": [s.to_dict() for s in self.stmts]}


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------

def expr_from_dict(d: dict) -> Expr:
    k = d.get("kind")
    # Compact forms: kind omitted when unambiguous from other keys
    if k is None:
        if "name" in d:
            return Identifier(name=d["name"])
        import json
        return RawExpr(text=json.dumps(d))
    if k == "literal":
        return Literal(value=d["value"], lit_kind=d.get("lit_kind", "int"))
    if k == "identifier":
        return Identifier(name=d["name"])
    if k == "binary_op":
        return BinaryOp(left=expr_from_dict(d["left"]), op=d["op"], right=expr_from_dict(d["right"]))
    if k == "unary_op":
        return UnaryOp(op=d["op"], operand=expr_from_dict(d["operand"]))
    if k == "call":
        return Call(
            func=expr_from_dict(d["func"]),
            args=[expr_from_dict(a) for a in d.get("args", [])],
            kwargs={kw: expr_from_dict(v) for kw, v in d.get("kwargs", {}).items()},
        )
    if k == "field_access":
        return FieldAccess(object=expr_from_dict(d["object"]), field_name=d["field"])
    if k == "index":
        return Index(object=expr_from_dict(d["object"]), index=expr_from_dict(d["index"]))
    if k == "list_literal":
        return ListLiteral(elements=[expr_from_dict(e) for e in d.get("elements", [])])
    if k == "dict_literal":
        return DictLiteral(pairs=[(expr_from_dict(p[0]), expr_from_dict(p[1])) for p in d.get("pairs", [])])
    if k == "tuple_literal":
        return TupleLiteral(elements=[expr_from_dict(e) for e in d.get("elements", [])])
    if k == "lambda":
        return Lambda(params=d.get("params", []), body=expr_from_dict(d["body"]))
    if k == "conditional":
        return Conditional(cond=expr_from_dict(d["cond"]), then_expr=expr_from_dict(d["then"]), else_expr=expr_from_dict(d["else"]))
    if k == "await":
        return Await(expr=expr_from_dict(d["expr"]))
    if k == "cast":
        from .types import UnifiedType
        return Cast(expr=expr_from_dict(d["expr"]), target_type=UnifiedType.from_dict(d["type"]))
    if k == "raw_expr":
        return RawExpr(text=d["text"])
    # Unknown — treat as raw
    import json
    return RawExpr(text=json.dumps(d))


def stmt_from_dict(d: dict) -> Stmt:
    from .types import UnifiedType
    k = d.get("kind")
    # Compact form: block with kind omitted
    if k is None or k == "block":
        if k is None and "stmts" not in d:
            import json
            return Raw(text=json.dumps(d))
        return Block(stmts=[stmt_from_dict(s) for s in d.get("stmts", [])])
    if k == "var_decl":
        return VarDecl(
            name=d["name"],
            type=UnifiedType.from_dict(d["type"]) if "type" in d else T_INFERRED,
            value=expr_from_dict(d["value"]) if "value" in d else None,
            is_mutable=d.get("is_mutable", True),
        )
    if k == "assign":
        return Assign(target=expr_from_dict(d["target"]), op=d.get("op", "="), value=expr_from_dict(d["value"]))
    if k == "return":
        return Return(value=expr_from_dict(d["value"]) if "value" in d else None)
    if k == "if":
        then_key = "then" if "then" in d else "then_block"
        return If(
            cond=expr_from_dict(d["cond"]),
            then_block=stmt_from_dict(d[then_key]),
            elif_branches=[(expr_from_dict(c), stmt_from_dict(b)) for c, b in d.get("elif", [])],
            else_block=stmt_from_dict(d["else"]) if "else" in d else None,
        )
    if k == "while":
        return WhileLoop(cond=expr_from_dict(d["cond"]), body=stmt_from_dict(d["body"]))
    if k == "for_each":
        return ForEach(var=d["var"], iter_expr=expr_from_dict(d["iter"]), body=stmt_from_dict(d["body"]))
    if k == "match":
        arms = [MatchArm(
            pattern=a["pattern"],
            guard=expr_from_dict(a["guard"]) if "guard" in a else None,
            body=stmt_from_dict(a["body"]),
        ) for a in d.get("arms", [])]
        return Match(subject=expr_from_dict(d["subject"]), arms=arms)
    if k == "break":
        return Break()
    if k == "continue":
        return Continue()
    if k == "raise":
        return Raise(expr=expr_from_dict(d["expr"]) if "expr" in d else None)
    if k == "expr_stmt":
        return ExprStmt(expr=expr_from_dict(d["expr"]), is_implicit_return=d.get("is_implicit_return", False))
    if k == "raw":
        return Raw(text=d["text"])
    import json
    return Raw(text=json.dumps(d))
