"""Generate Rust source from a unified AST Module."""
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
    module = node.module.replace(".", "::")
    if node.alias:
        return f"use {module} as {node.alias};"
    if node.items is None:
        return f"use {module};"
    if node.items == ["*"]:
        return f"use {module}::*;"
    if len(node.items) == 1:
        return f"use {module}::{node.items[0]};"
    items = ", ".join(node.items)
    return f"use {module}::{{{items}}};"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def _type_str(ty: UnifiedType, mutable: bool = False) -> str:
    k = ty.kind
    mut = "mut " if mutable else ""

    if k == TypeKind.NUMBER:
        if ty.float:
            return f"f{ty.bits or 64}"
        if ty.signed is False:
            return f"u{ty.bits or 64}"
        return f"i{ty.bits or 64}"
    if k == TypeKind.STRING:
        return "String"
    if k == TypeKind.BOOLEAN:
        return "bool"
    if k == TypeKind.BYTES:
        return "Vec<u8>"
    if k == TypeKind.LIST:
        inner = _type_str(ty.element) if ty.element else "_"
        return f"Vec<{inner}>"
    if k == TypeKind.MAP:
        ks = _type_str(ty.key) if ty.key else "_"
        vs = _type_str(ty.value) if ty.value else "_"
        return f"HashMap<{ks}, {vs}>"
    if k == TypeKind.SET:
        inner = _type_str(ty.element) if ty.element else "_"
        return f"HashSet<{inner}>"
    if k == TypeKind.OPTIONAL:
        inner = _type_str(ty.element) if ty.element else "_"
        return f"Option<{inner}>"
    if k == TypeKind.TUPLE:
        parts = [_type_str(e) for e in ty.elements] if ty.elements else []
        return f"({', '.join(parts)})"
    if k == TypeKind.VOID:
        return "()"
    if k == TypeKind.ANY:
        return "Box<dyn Any>"
    if k == TypeKind.SELF:
        return "Self"
    if k == TypeKind.INFERRED:
        return "_"
    if k == TypeKind.NAMED:
        return ty.name or "_"
    if k == TypeKind.FUNCTION:
        params = [_type_str(p) for p in ty.params] if ty.params else []
        ret = _type_str(ty.ret) if ty.ret else "()"
        return f"fn({', '.join(params)}) -> {ret}"
    if k == TypeKind.GENERIC:
        return ty.name or "T"
    return "_"


# ---------------------------------------------------------------------------
# Variables (module-level consts/statics)
# ---------------------------------------------------------------------------

def _gen_variable(node: VariableNode, indent: int) -> str:
    pad = INDENT * indent
    ty = _type_str(node.type)
    vis = "pub " if node.visibility == Visibility.PUBLIC else ""
    kw = "const" if node.is_const else "static"
    mut = " mut" if not node.is_const and node.attributes.get("is_mutable", False) else ""
    val = node.value or "unimplemented!()"
    if ty == "_":
        return f"{pad}{vis}{kw}{mut} {node.name} = {val};"
    return f"{pad}{vis}{kw}{mut} {node.name}: {ty} = {val};"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _gen_function(node: FunctionNode, indent: int) -> str:
    pad = INDENT * indent
    lines: List[str] = []

    vis = "pub " if node.visibility == Visibility.PUBLIC else ""
    async_ = "async " if node.is_async else ""
    tp = f"<{', '.join(node.type_params)}>" if node.type_params else ""
    params_str = _gen_params(node.params)
    ret = _type_str(node.return_type)
    ret_str = f" -> {ret}" if ret not in ("()", "_") else ""

    if node.is_abstract:
        # Trait method signature only
        lines.append(f"{pad}{vis}fn {node.name}{tp}({params_str}){ret_str};")
        return "\n".join(lines)

    if node.docstring:
        for line in node.docstring.splitlines():
            lines.append(f"{pad}/// {line}")

    lines.append(f"{pad}{vis}{async_}fn {node.name}{tp}({params_str}){ret_str} {{")

    body_lines = _gen_block(node.body, indent + 1)
    if not body_lines:
        lines.append(f"{pad}}}")
    else:
        lines.extend(body_lines)
        lines.append(f"{pad}}}")

    return "\n".join(lines)


def _gen_params(params: List[Param]) -> str:
    parts = []
    for p in params:
        if p.is_self:
            # Check if mutable receiver needed — default to &mut self for methods
            parts.append("&mut self")
            continue
        ty = _type_str(p.type)
        name = f"...{p.name}" if p.is_variadic else p.name
        if ty == "_":
            parts.append(name)
        else:
            parts.append(f"{name}: {ty}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

def _gen_typedef(node: TypeDefNode, indent: int) -> str:
    pad = INDENT * indent
    vis = "pub " if node.visibility == Visibility.PUBLIC else ""
    tp = f"<{', '.join(node.type_params)}>" if node.type_params else ""

    sections: List[str] = []

    if node.docstring:
        for line in node.docstring.splitlines():
            sections.append(f"{pad}/// {line}")

    cat = node.category

    if cat == TypeDefCategory.STRUCT:
        sections.extend(_gen_struct(node, pad, vis, tp))
    elif cat == TypeDefCategory.ENUM:
        sections.extend(_gen_enum(node, pad, vis, tp))
    elif cat == TypeDefCategory.TRAIT:
        sections.extend(_gen_trait(node, pad, vis, tp))
    elif cat in (TypeDefCategory.CLASS, TypeDefCategory.INTERFACE):
        # Map Python class → Rust struct + impl
        sections.extend(_gen_struct(node, pad, vis, tp))

    # impl block for methods
    if node.methods:
        sections.extend(_gen_impl(node, pad, tp))

    return "\n".join(sections)


def _gen_struct(node: TypeDefNode, pad: str, vis: str, tp: str) -> List[str]:
    lines = [f"{pad}{vis}struct {node.name}{tp} {{"]
    for f in node.fields:
        fvis = "pub " if f.visibility == Visibility.PUBLIC else ""
        fty = _type_str(f.type)
        if fty == "_":
            lines.append(f"{pad}{INDENT}{fvis}{f.name},")
        else:
            lines.append(f"{pad}{INDENT}{fvis}{f.name}: {fty},")
    lines.append(f"{pad}}}")
    return lines


def _gen_enum(node: TypeDefNode, pad: str, vis: str, tp: str) -> List[str]:
    lines = [f"{pad}{vis}enum {node.name}{tp} {{"]
    for f in node.fields:
        # Enum variants stored in fields with raw default text
        if f.default:
            variant_text = f.default
            # Strip the full "EnumName::Variant" prefix if present
            if "::" in variant_text:
                variant_text = variant_text.split("::")[-1]
            lines.append(f"{pad}{INDENT}{variant_text},")
        else:
            lines.append(f"{pad}{INDENT}{f.name},")
    lines.append(f"{pad}}}")
    return lines


def _gen_trait(node: TypeDefNode, pad: str, vis: str, tp: str) -> List[str]:
    bases = node.bases + node.interfaces
    sup = f": {' + '.join(bases)}" if bases else ""
    lines = [f"{pad}{vis}trait {node.name}{tp}{sup} {{"]
    for m in node.methods:
        lines.append(_gen_function(m, indent=len(pad) // 4 + 1))
    lines.append(f"{pad}}}")
    return lines


def _gen_impl(node: TypeDefNode, pad: str, tp: str) -> List[str]:
    lines = [f""]
    lines.append(f"{pad}impl{tp} {node.name}{tp} {{")
    for m in node.methods:
        if not m.attributes.get("trait_impl", False):
            fn_lines = _gen_function(m, indent=len(pad) // 4 + 1).splitlines()
            lines.extend(fn_lines)
            lines.append("")
    # Remove trailing blank line before }
    while lines and not lines[-1].strip():
        lines.pop()
    lines.append(f"{pad}}}")
    return lines


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

def _gen_block(block: Block, indent: int) -> List[str]:
    lines: List[str] = []
    stmts = block.stmts
    for i, stmt in enumerate(stmts):
        is_last = i == len(stmts) - 1
        lines.extend(_gen_stmt(stmt, indent, is_last=is_last))
    return lines


def _gen_stmt(stmt: Stmt, indent: int, is_last: bool = False) -> List[str]:
    pad = INDENT * indent

    if isinstance(stmt, Raw):
        text = stmt.text
        import textwrap
        dedented = textwrap.dedent(text).strip()
        return [(pad + line if line.strip() else line) for line in dedented.splitlines()]

    if isinstance(stmt, VarDecl):
        mut = "mut " if stmt.is_mutable else ""
        ty = _type_str(stmt.type)
        ty_str = f": {ty}" if ty != "_" else ""
        val = f" = {_gen_expr(stmt.value)}" if stmt.value is not None else ""
        return [f"{pad}let {mut}{stmt.name}{ty_str}{val};"]

    if isinstance(stmt, Assign):
        op = stmt.op if stmt.op != "=" else "="
        return [f"{pad}{_gen_expr(stmt.target)} {op} {_gen_expr(stmt.value)};"]

    if isinstance(stmt, Return):
        if stmt.value is None:
            return [f"{pad}return;"]
        return [f"{pad}return {_gen_expr(stmt.value)};"]

    if isinstance(stmt, If):
        return _gen_if(stmt, indent)

    if isinstance(stmt, WhileLoop):
        cond_text = _gen_expr(stmt.cond)
        if cond_text == "true":
            lines = [f"{pad}loop {{"]
        else:
            lines = [f"{pad}while {cond_text} {{"]
        lines.extend(_gen_block(stmt.body, indent + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, ForEach):
        lines = [f"{pad}for {stmt.var} in {_gen_expr(stmt.iter_expr)} {{"]
        lines.extend(_gen_block(stmt.body, indent + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, Match):
        return _gen_match(stmt, indent)

    if isinstance(stmt, Break):
        return [f"{pad}break;"]

    if isinstance(stmt, Continue):
        return [f"{pad}continue;"]

    if isinstance(stmt, Raise):
        if stmt.expr is None:
            return [f"{pad}panic!();"]
        return [f"{pad}panic!(\"{{}}\", {_gen_expr(stmt.expr)});"]

    if isinstance(stmt, ExprStmt):
        expr_str = _gen_expr(stmt.expr)
        # Implicit return: last stmt in block, no semicolon
        if stmt.is_implicit_return or is_last:
            return [f"{pad}{expr_str}"]
        return [f"{pad}{expr_str};"]

    if isinstance(stmt, Block):
        lines = [f"{pad}{{"]
        lines.extend(_gen_block(stmt, indent + 1))
        lines.append(f"{pad}}}")
        return lines

    return [f"{pad}// <unknown stmt>"]


def _gen_if(stmt: If, indent: int) -> List[str]:
    pad = INDENT * indent
    lines = [f"{pad}if {_gen_expr(stmt.cond)} {{"]
    lines.extend(_gen_block(stmt.then_block, indent + 1))
    for elif_cond, elif_block in stmt.elif_branches:
        lines.append(f"{pad}}} else if {_gen_expr(elif_cond)} {{")
        lines.extend(_gen_block(elif_block, indent + 1))
    if stmt.else_block is not None:
        lines.append(f"{pad}}} else {{")
        lines.extend(_gen_block(stmt.else_block, indent + 1))
    lines.append(f"{pad}}}")
    return lines


def _gen_match(stmt: Match, indent: int) -> List[str]:
    pad = INDENT * indent
    lines = [f"{pad}match {_gen_expr(stmt.subject)} {{"]
    for arm in stmt.arms:
        guard = f" if {_gen_expr(arm.guard)}" if arm.guard else ""
        arm_body = _gen_block(arm.body, indent + 2)
        lines.append(f"{pad}{INDENT}{arm.pattern}{guard} => {{")
        lines.extend(arm_body)
        lines.append(f"{pad}{INDENT}}},")
    lines.append(f"{pad}}}")
    return lines


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

_RUST_BINOP_REMAP = {
    "and": "&&",
    "or": "||",
    "not": "!",
    "**": "/* ** not in Rust */",
    "//": "/",
    "is": "==",
    "is not": "!=",
    "in": "/* in */",
    "not in": "/* not in */",
}

_RUST_UNOP_REMAP = {
    "not": "!",
    "~": "!",
    "-": "-",
    "+": "",
    "not ": "!",
}


def _gen_expr(expr: Expr) -> str:
    if isinstance(expr, Literal):
        if expr.lit_kind == "none":
            return "None"
        if expr.lit_kind == "bool":
            return "true" if expr.value else "false"
        if expr.lit_kind == "string":
            val = str(expr.value).replace('"', '\\"')
            return f'"{val}"'
        return str(expr.value)

    if isinstance(expr, Identifier):
        name = expr.name
        # Python None/True/False → Rust
        if name == "None":
            return "None"
        if name == "True":
            return "true"
        if name == "False":
            return "false"
        return name

    if isinstance(expr, BinaryOp):
        op = _RUST_BINOP_REMAP.get(expr.op, expr.op)
        return f"({_gen_expr(expr.left)} {op} {_gen_expr(expr.right)})"

    if isinstance(expr, UnaryOp):
        op = _RUST_UNOP_REMAP.get(expr.op, expr.op)
        return f"({op}{_gen_expr(expr.operand)})"

    if isinstance(expr, Call):
        func = _gen_expr(expr.func)
        args = [_gen_expr(a) for a in expr.args]
        # If call has no positional args and has kwargs, it may be a struct literal
        if not args and expr.kwargs and isinstance(expr.func, Identifier) and expr.func.name[0].isupper():
            fields = ", ".join(f"{k}: {_gen_expr(v)}" for k, v in expr.kwargs.items())
            return f"{func} {{ {fields} }}"
        kwargs = [f"{k}: {_gen_expr(v)}" for k, v in expr.kwargs.items()]
        all_args = ", ".join(args + kwargs)
        return f"{func}({all_args})"

    if isinstance(expr, FieldAccess):
        return f"{_gen_expr(expr.object)}.{expr.field_name}"

    if isinstance(expr, Index):
        return f"{_gen_expr(expr.object)}[{_gen_expr(expr.index)}]"

    if isinstance(expr, ListLiteral):
        elems = ", ".join(_gen_expr(e) for e in expr.elements)
        return f"vec![{elems}]"

    if isinstance(expr, DictLiteral):
        if not expr.pairs:
            return "HashMap::new()"
        entries = ", ".join(f"({_gen_expr(k)}, {_gen_expr(v)})" for k, v in expr.pairs)
        return f"[{entries}].into_iter().collect::<HashMap<_, _>>()"

    if isinstance(expr, TupleLiteral):
        elems = ", ".join(_gen_expr(e) for e in expr.elements)
        return f"({elems})"

    if isinstance(expr, Lambda):
        params = ", ".join(expr.params)
        return f"|{params}| {_gen_expr(expr.body)}"

    if isinstance(expr, Conditional):
        return f"(if {_gen_expr(expr.cond)} {{ {_gen_expr(expr.then_expr)} }} else {{ {_gen_expr(expr.else_expr)} }})"

    if isinstance(expr, Await):
        return f"{_gen_expr(expr.expr)}.await"

    if isinstance(expr, Cast):
        ty = _type_str(expr.target_type)
        return f"({_gen_expr(expr.expr)} as {ty})"

    if isinstance(expr, RawExpr):
        return expr.text

    return "unimplemented!()"
