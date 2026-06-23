"""Generate Python source from a unified AST Module."""
from __future__ import annotations
from typing import List, Optional

from ..unified_ast.types import UnifiedType, TypeKind
from ..unified_ast.expr import (
    Block, Expr, Stmt,
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess, Index,
    ListLiteral, DictLiteral, TupleLiteral, Lambda, Conditional, Await, Cast, RawExpr,
    VarDecl, Assign, Return, If, WhileLoop, ForEach, Match,
    Break, Continue, Raise, ExprStmt, Raw,
)
from ..unified_ast.nodes import (
    Visibility, TypeDefCategory,
    Param, ImportNode, VariableNode, FieldNode, FunctionNode, TypeDefNode,
    Module, ASTNode,
)

INDENT = "    "


def generate(module: Module) -> str:
    parts: List[str] = []
    for node in module.nodes:
        parts.append(_gen_node(node, indent=0))
    return "\n\n".join(p for p in parts if p) + "\n"


def _gen_node(node: ASTNode, indent: int) -> str:
    if isinstance(node, ImportNode):
        return _gen_import(node)
    if isinstance(node, VariableNode):
        return _gen_variable(node, indent)
    if isinstance(node, FunctionNode):
        return _gen_function(node, indent)
    if isinstance(node, TypeDefNode):
        return _gen_typedef(node, indent)
    return ""


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def _gen_import(node: ImportNode) -> str:
    if node.items is None:
        alias = f" as {node.alias}" if node.alias else ""
        return f"import {node.module}{alias}"
    else:
        items = ", ".join(node.items)
        alias = f" as {node.alias}" if node.alias else ""
        return f"from {node.module} import {items}{alias}"


# ---------------------------------------------------------------------------
# Module-level variables
# ---------------------------------------------------------------------------

def _gen_variable(node: VariableNode, indent: int) -> str:
    pad = INDENT * indent
    ty_str = _type_annotation(node.type)
    if ty_str and ty_str != "_":
        decl = f"{node.name}: {ty_str}"
    else:
        decl = node.name
    if node.value is not None:
        return f"{pad}{decl} = {node.value}"
    return f"{pad}{decl}"


def _type_annotation(ty: UnifiedType) -> str:
    """Convert UnifiedType to Python annotation string."""
    k = ty.kind
    if k == TypeKind.NUMBER:
        if ty.float:
            return "float"
        return "int"
    if k == TypeKind.STRING:
        return "str"
    if k == TypeKind.BOOLEAN:
        return "bool"
    if k == TypeKind.BYTES:
        return "bytes"
    if k == TypeKind.LIST:
        inner = _type_annotation(ty.element) if ty.element else "Any"
        return f"list[{inner}]"
    if k == TypeKind.MAP:
        k_str = _type_annotation(ty.key) if ty.key else "Any"
        v_str = _type_annotation(ty.value) if ty.value else "Any"
        return f"dict[{k_str}, {v_str}]"
    if k == TypeKind.SET:
        inner = _type_annotation(ty.element) if ty.element else "Any"
        return f"set[{inner}]"
    if k == TypeKind.OPTIONAL:
        inner = _type_annotation(ty.element) if ty.element else "Any"
        return f"{inner} | None"
    if k == TypeKind.TUPLE:
        parts = [_type_annotation(e) for e in ty.elements] if ty.elements else []
        return f"tuple[{', '.join(parts)}]"
    if k == TypeKind.VOID:
        return "None"
    if k == TypeKind.ANY:
        return "Any"
    if k == TypeKind.SELF:
        return "Self"
    if k == TypeKind.INFERRED:
        return ""
    if k == TypeKind.NAMED:
        return ty.name or "Any"
    if k == TypeKind.FUNCTION:
        params = [_type_annotation(p) for p in ty.params] if ty.params else []
        ret = _type_annotation(ty.ret) if ty.ret else "None"
        return f"Callable[[{', '.join(params)}], {ret}]"
    if k == TypeKind.GENERIC:
        return ty.name or "T"
    return "Any"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _gen_function(node: FunctionNode, indent: int) -> str:
    pad = INDENT * indent
    lines: List[str] = []

    for dec in node.decorators:
        lines.append(f"{pad}@{dec}")

    params_str = _gen_params(node.params)
    ret = _type_annotation(node.return_type)
    ret_str = f" -> {ret}" if ret and ret != "" else ""

    prefix = "async def" if node.is_async else "def"
    lines.append(f"{pad}{prefix} {node.name}({params_str}){ret_str}:")

    if node.docstring:
        doc = node.docstring.replace('"""', r'\"\"\"')
        lines.append(f'{pad}{INDENT}"""{doc}"""')

    body_lines = _gen_block(node.body, indent + 1)
    if not body_lines:
        lines.append(f"{pad}{INDENT}pass")
    else:
        lines.extend(body_lines)

    return "\n".join(lines)


def _gen_params(params: List[Param]) -> str:
    parts = []
    for p in params:
        if p.is_self:
            parts.append(p.name)
            continue
        ty = _type_annotation(p.type)
        name = f"*{p.name}" if p.is_variadic else p.name
        if ty:
            part = f"{name}: {ty}"
        else:
            part = name
        if p.default is not None:
            part += f" = {p.default}"
        parts.append(part)
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

def _gen_typedef(node: TypeDefNode, indent: int) -> str:
    pad = INDENT * indent
    lines: List[str] = []

    bases = list(node.bases)
    if node.interfaces:
        bases.extend(node.interfaces)
    if node.category == TypeDefCategory.INTERFACE and "ABC" not in bases:
        bases.append("ABC")

    tp = f"[{', '.join(node.type_params)}]" if node.type_params else ""
    bases_str = f"({', '.join(bases)})" if bases else ""
    lines.append(f"{pad}class {node.name}{tp}{bases_str}:")

    body_lines: List[str] = []

    if node.docstring:
        doc = node.docstring.replace('"""', r'\"\"\"')
        body_lines.append(f'{pad}{INDENT}"""{doc}"""')
        body_lines.append("")

    for f in node.fields:
        body_lines.append(_gen_field(f, indent + 1))

    if node.fields and node.methods:
        body_lines.append("")

    for m in node.methods:
        body_lines.append(_gen_function(m, indent + 1))
        body_lines.append("")

    for inner in node.inner_types:
        body_lines.append(_gen_typedef(inner, indent + 1))
        body_lines.append("")

    if not body_lines:
        body_lines.append(f"{pad}{INDENT}pass")

    # Remove trailing blank line
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    lines.extend(body_lines)
    return "\n".join(lines)


def _gen_field(node: FieldNode, indent: int) -> str:
    pad = INDENT * indent
    ty = _type_annotation(node.type)
    if ty:
        decl = f"{node.name}: {ty}"
    else:
        decl = node.name
    if node.default is not None:
        return f"{pad}{decl} = {node.default}"
    return f"{pad}{decl}"


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

def _gen_block(block: Block, indent: int) -> List[str]:
    lines: List[str] = []
    for stmt in block.stmts:
        lines.extend(_gen_stmt(stmt, indent))
    return lines


def _gen_stmt(stmt: Stmt, indent: int) -> List[str]:
    pad = INDENT * indent

    if isinstance(stmt, Raw):
        # Re-indent raw text to current indent level
        text = stmt.text
        dedented = _smart_dedent(text)
        return [(pad + line if line.strip() else line) for line in dedented.splitlines()]

    if isinstance(stmt, VarDecl):
        ty = _type_annotation(stmt.type)
        name = stmt.name
        if ty:
            lhs = f"{name}: {ty}"
        else:
            lhs = name
        if stmt.value is not None:
            return [f"{pad}{lhs} = {_gen_expr(stmt.value)}"]
        return [f"{pad}{lhs}"]

    if isinstance(stmt, Assign):
        return [f"{pad}{_gen_expr(stmt.target)} {stmt.op} {_gen_expr(stmt.value)}"]

    if isinstance(stmt, Return):
        if stmt.value is None:
            return [f"{pad}return"]
        return [f"{pad}return {_gen_expr(stmt.value)}"]

    if isinstance(stmt, If):
        lines = [f"{pad}if {_gen_expr(stmt.cond)}:"]
        lines.extend(_gen_block(stmt.then_block, indent + 1) or [f"{pad}{INDENT}pass"])
        for elif_cond, elif_block in stmt.elif_branches:
            lines.append(f"{pad}elif {_gen_expr(elif_cond)}:")
            lines.extend(_gen_block(elif_block, indent + 1) or [f"{pad}{INDENT}pass"])
        if stmt.else_block is not None:
            lines.append(f"{pad}else:")
            lines.extend(_gen_block(stmt.else_block, indent + 1) or [f"{pad}{INDENT}pass"])
        return lines

    if isinstance(stmt, WhileLoop):
        lines = [f"{pad}while {_gen_expr(stmt.cond)}:"]
        lines.extend(_gen_block(stmt.body, indent + 1) or [f"{pad}{INDENT}pass"])
        return lines

    if isinstance(stmt, ForEach):
        lines = [f"{pad}for {stmt.var} in {_gen_expr(stmt.iter_expr)}:"]
        lines.extend(_gen_block(stmt.body, indent + 1) or [f"{pad}{INDENT}pass"])
        return lines

    if isinstance(stmt, Match):
        lines = [f"{pad}match {_gen_expr(stmt.subject)}:"]
        for arm in stmt.arms:
            guard = f" if {_gen_expr(arm.guard)}" if arm.guard else ""
            lines.append(f"{pad}{INDENT}case {arm.pattern}{guard}:")
            lines.extend(_gen_block(arm.body, indent + 2) or [f"{pad}{INDENT}{INDENT}pass"])
        return lines

    if isinstance(stmt, Break):
        return [f"{pad}break"]

    if isinstance(stmt, Continue):
        return [f"{pad}continue"]

    if isinstance(stmt, Raise):
        if stmt.expr is None:
            return [f"{pad}raise"]
        return [f"{pad}raise {_gen_expr(stmt.expr)}"]

    if isinstance(stmt, ExprStmt):
        return [f"{pad}{_gen_expr(stmt.expr)}"]

    if isinstance(stmt, Block):
        return _gen_block(stmt, indent)

    return [f"{pad}# <unknown stmt>"]


def _smart_dedent(text: str) -> str:
    """Dedent raw text, stripping common leading whitespace."""
    import textwrap
    return textwrap.dedent(text).strip()


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

_PY_BINOP_REMAP = {
    "&&": "and",
    "||": "or",
}

_PY_UNOP_REMAP = {
    "not": "not ",
    "~": "~",
    "-": "-",
    "+": "+",
    "!": "not ",
}


def _gen_expr(expr: Expr) -> str:
    if isinstance(expr, Literal):
        if expr.lit_kind == "none":
            return "None"
        if expr.lit_kind == "bool":
            return "True" if expr.value else "False"
        if expr.lit_kind == "string":
            return repr(expr.value)
        return repr(expr.value) if not isinstance(expr.value, str) else expr.value

    if isinstance(expr, Identifier):
        return expr.name

    if isinstance(expr, BinaryOp):
        op = _PY_BINOP_REMAP.get(expr.op, expr.op)
        return f"({_gen_expr(expr.left)} {op} {_gen_expr(expr.right)})"

    if isinstance(expr, UnaryOp):
        op = _PY_UNOP_REMAP.get(expr.op, expr.op)
        return f"({op}{_gen_expr(expr.operand)})"

    if isinstance(expr, Call):
        func = _gen_expr(expr.func)
        args = [_gen_expr(a) for a in expr.args]
        kwargs = [f"{k}={_gen_expr(v)}" for k, v in expr.kwargs.items()]
        all_args = ", ".join(args + kwargs)
        return f"{func}({all_args})"

    if isinstance(expr, FieldAccess):
        return f"{_gen_expr(expr.object)}.{expr.field_name}"

    if isinstance(expr, Index):
        return f"{_gen_expr(expr.object)}[{_gen_expr(expr.index)}]"

    if isinstance(expr, ListLiteral):
        elems = ", ".join(_gen_expr(e) for e in expr.elements)
        return f"[{elems}]"

    if isinstance(expr, DictLiteral):
        pairs = ", ".join(f"{_gen_expr(k)}: {_gen_expr(v)}" for k, v in expr.pairs)
        return "{" + pairs + "}"

    if isinstance(expr, TupleLiteral):
        elems = ", ".join(_gen_expr(e) for e in expr.elements)
        if len(expr.elements) == 1:
            return f"({elems},)"
        return f"({elems})"

    if isinstance(expr, Lambda):
        params = ", ".join(expr.params)
        return f"lambda {params}: {_gen_expr(expr.body)}"

    if isinstance(expr, Conditional):
        return f"({_gen_expr(expr.then_expr)} if {_gen_expr(expr.cond)} else {_gen_expr(expr.else_expr)})"

    if isinstance(expr, Await):
        return f"await {_gen_expr(expr.expr)}"

    if isinstance(expr, Cast):
        ty = _type_annotation(expr.target_type)
        return f"{ty}({_gen_expr(expr.expr)})"

    if isinstance(expr, RawExpr):
        return expr.text

    return "None"
