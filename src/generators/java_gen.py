"""Generate Java source from a unified AST Module."""
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

# ---------------------------------------------------------------------------
# Language-specific syntax tokens — edit these in each copy
# ---------------------------------------------------------------------------

_INDENT          = "    "      # indentation unit

# Import syntax: set in _emit_import
# Function syntax: set in _emit_function
# Variable syntax: set in _emit_variable

# Type keyword overrides (TypeKind → language type string)
_TYPE_MAP = {
    # TODO: fill in for your language, e.g.:
    # TypeKind.NUMBER: "int",
    # TypeKind.STRING: "string",
    # TypeKind.BOOLEAN: "bool",
    # TypeKind.VOID: "void",
    # TypeKind.ANY: "any",
}

_VISIBILITY_MAP = {
    Visibility.PUBLIC: "",      # TODO: e.g. "public " for Java
    Visibility.PRIVATE: "",     # TODO: e.g. "private " for Java
    Visibility.PROTECTED: "",
    Visibility.INTERNAL: "",
}


# ===========================================================================
# Entry point
# ===========================================================================

def generate(module: Module) -> str:
    parts: List[str] = []
    for node in module.nodes:
        s = _gen_node(node, indent=0)
        if s:
            parts.append(s)
    return "\n\n".join(parts) + "\n"


def _gen_node(node: ASTNode, indent: int) -> str:
    if isinstance(node, ImportNode):
        return _emit_import(node)
    if isinstance(node, VariableNode):
        return _emit_variable(node, indent)
    if isinstance(node, FunctionNode):
        return _emit_function(node, indent)
    if isinstance(node, TypeDefNode):
        return _emit_typedef(node, indent)
    return ""


# ===========================================================================
# Imports
# ===========================================================================

def _emit_import(node: ImportNode) -> str:
    # TODO: implement for your language
    # Examples:
    #   Go:     f'import "{node.module}"'
    #   Java:   f'import {node.module};'
    #   Swift:  f'import {node.module}'
    module = node.module
    if node.items:
        items = ", ".join(node.items)
        return f"import {{ {items} }} from \"{module}\";"   # JS-style placeholder
    if node.alias:
        return f"import {module} as {node.alias};"
    return f"import \"{module}\";"


# ===========================================================================
# Variables
# ===========================================================================

def _emit_variable(node: VariableNode, indent: int) -> str:
    # TODO: implement for your language
    pad = _INDENT * indent
    kw = "const" if node.is_const else "var"
    ty = f": {_type(node.type)}" if node.type.kind != TypeKind.INFERRED else ""
    val = f" = {node.value}" if node.value else ""
    return f"{pad}{kw} {node.name}{ty}{val}"


# ===========================================================================
# Functions
# ===========================================================================

def _emit_function(fn: FunctionNode, indent: int) -> str:
    # TODO: implement for your language
    pad = _INDENT * indent
    vis = _VISIBILITY_MAP.get(fn.visibility, "")
    params = ", ".join(_emit_param(p) for p in fn.params)
    ret = f" -> {_type(fn.return_type)}" if fn.return_type.kind not in (TypeKind.INFERRED, TypeKind.VOID) else ""
    async_kw = "async " if fn.is_async else ""
    header = f"{pad}{vis}{async_kw}func {fn.name}({params}){ret}"
    body = _emit_block(fn.body, indent + 1)
    return f"{header} {{\n{body}{pad}}}"


def _emit_param(p: Param) -> str:
    # TODO: implement for your language
    if p.type.kind == TypeKind.INFERRED:
        return p.name
    return f"{p.name}: {_type(p.type)}"


# ===========================================================================
# Type definitions
# ===========================================================================

def _emit_typedef(td: TypeDefNode, indent: int) -> str:
    # TODO: implement for your language
    pad = _INDENT * indent
    inner_pad = _INDENT * (indent + 1)
    kw = _category_keyword(td.category)
    vis = _VISIBILITY_MAP.get(td.visibility, "")
    bases = f"({', '.join(td.bases)})" if td.bases else ""

    lines: List[str] = [f"{pad}{vis}{kw} {td.name}{bases} {{"]
    for f in td.fields:
        ty = f": {_type(f.type)}" if f.type.kind != TypeKind.INFERRED else ""
        lines.append(f"{inner_pad}{f.name}{ty};")
    for m in td.methods:
        lines.append("")
        lines.append(_emit_function(m, indent + 1))
    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _category_keyword(cat: TypeDefCategory) -> str:
    # TODO: adjust per language
    return {
        TypeDefCategory.CLASS: "class",
        TypeDefCategory.STRUCT: "struct",
        TypeDefCategory.INTERFACE: "interface",
        TypeDefCategory.TRAIT: "trait",
        TypeDefCategory.ENUM: "enum",
    }.get(cat, "class")


# ===========================================================================
# Block and statement emitters
# ===========================================================================

def _emit_block(block: Block, indent: int) -> str:
    lines = []
    for s in block.stmts:
        lines.append(_stmt(s, indent))
    return "\n".join(lines) + ("\n" if lines else "")


def _stmt(s: Stmt, indent: int) -> str:
    pad = _INDENT * indent

    if isinstance(s, Return):
        val = f" {_expr(s.value)}" if s.value is not None else ""
        return f"{pad}return{val};"

    if isinstance(s, VarDecl):
        ty = f": {_type(s.type)}" if s.type.kind != TypeKind.INFERRED else ""
        val = f" = {_expr(s.value)}" if s.value is not None else ""
        kw = "var" if s.is_mutable else "let"
        return f"{pad}{kw} {s.name}{ty}{val};"

    if isinstance(s, Assign):
        return f"{pad}{_expr(s.target)} {s.op} {_expr(s.value)};"

    if isinstance(s, If):
        cond = _expr(s.cond)
        then_body = _emit_block(s.then_block, indent + 1)
        result = f"{pad}if ({cond}) {{\n{then_body}{pad}}}"
        for elif_cond, elif_block in s.elif_branches:
            eb = _emit_block(elif_block, indent + 1)
            result += f" else if ({_expr(elif_cond)}) {{\n{eb}{pad}}}"
        if s.else_block:
            eb = _emit_block(s.else_block, indent + 1)
            result += f" else {{\n{eb}{pad}}}"
        return result

    if isinstance(s, WhileLoop):
        body = _emit_block(s.body, indent + 1)
        return f"{pad}while ({_expr(s.cond)}) {{\n{body}{pad}}}"

    if isinstance(s, ForEach):
        body = _emit_block(s.body, indent + 1)
        # TODO: adjust syntax per language (for/of, for/in, for-each, range)
        return f"{pad}for ({s.var} in {_expr(s.iter_expr)}) {{\n{body}{pad}}}"

    if isinstance(s, Break):
        return f"{pad}break;"

    if isinstance(s, Continue):
        return f"{pad}continue;"

    if isinstance(s, Raise):
        val = f" {_expr(s.expr)}" if s.expr else ""
        return f"{pad}throw{val};"    # TODO: language-specific keyword

    if isinstance(s, ExprStmt):
        return f"{pad}{_expr(s.expr)};"

    if isinstance(s, Raw):
        return f"{pad}{s.text}"

    if isinstance(s, Block):
        return _emit_block(s, indent)

    return f"{pad}/* unsupported stmt: {type(s).__name__} */"


# ===========================================================================
# Expression emitter
# ===========================================================================

def _expr(e: Expr) -> str:
    if isinstance(e, Literal):
        v = e.value
        if e.lit_kind == "string":
            escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if e.lit_kind == "bool":
            return "true" if v else "false"   # TODO: per language
        if v is None:
            return "null"                      # TODO: per language (nil, None, nullptr)
        return str(v)

    if isinstance(e, Identifier):
        return e.name

    if isinstance(e, BinaryOp):
        return f"({_expr(e.left)} {e.op} {_expr(e.right)})"

    if isinstance(e, UnaryOp):
        return f"({e.op}{_expr(e.operand)})"

    if isinstance(e, Call):
        func = _expr(e.func)
        args = ", ".join(_expr(a) for a in e.args)
        kwargs = ", ".join(f"{k}: {_expr(v)}" for k, v in e.kwargs.items())
        all_args = ", ".join(filter(None, [args, kwargs]))
        return f"{func}({all_args})"

    if isinstance(e, FieldAccess):
        return f"{_expr(e.object)}.{e.field_name}"

    if isinstance(e, Index):
        return f"{_expr(e.object)}[{_expr(e.index)}]"

    if isinstance(e, ListLiteral):
        elems = ", ".join(_expr(el) for el in e.elements)
        return f"[{elems}]"

    if isinstance(e, DictLiteral):
        pairs = ", ".join(f"{_expr(k)}: {_expr(v)}" for k, v in e.pairs)
        return f"{{{pairs}}}"

    if isinstance(e, TupleLiteral):
        elems = ", ".join(_expr(el) for el in e.elements)
        return f"({elems})"

    if isinstance(e, Conditional):
        return f"({_expr(e.cond)} ? {_expr(e.then_expr)} : {_expr(e.else_expr)})"

    if isinstance(e, Await):
        return f"await {_expr(e.expr)}"    # TODO: per language

    if isinstance(e, Cast):
        return f"({_type(e.target_type)})({_expr(e.expr)})"   # C-style; TODO per language

    if isinstance(e, Lambda):
        params = ", ".join(e.params)
        return f"({params}) => {_expr(e.body)}"   # JS-style; TODO per language

    if isinstance(e, RawExpr):
        return e.text

    return f"/* unsupported expr: {type(e).__name__} */"


# ===========================================================================
# Type emitter
# ===========================================================================

def _type(t: UnifiedType) -> str:
    override = _TYPE_MAP.get(t.kind)
    if override:
        return override

    k = t.kind
    if k == TypeKind.NUMBER:
        # TODO: pick the right int/float type for the language
        if getattr(t, "float", False):
            return "float64"
        bits = getattr(t, "bits", 0)
        signed = getattr(t, "signed", True)
        if bits == 64:
            return "int64" if signed else "uint64"
        if bits == 32:
            return "int32" if signed else "uint32"
        return "int"
    if k == TypeKind.STRING:
        return "string"
    if k == TypeKind.BOOLEAN:
        return "bool"
    if k == TypeKind.VOID:
        return "void"
    if k == TypeKind.ANY:
        return "any"
    if k == TypeKind.BYTES:
        return "[]byte"
    if k == TypeKind.LIST:
        inner = _type(t.params[0]) if t.params else "any"
        return f"[]{inner}"
    if k == TypeKind.MAP:
        kt = _type(t.params[0]) if t.params else "any"
        vt = _type(t.params[1]) if len(t.params) > 1 else "any"
        return f"map[{kt}]{vt}"
    if k == TypeKind.OPTIONAL:
        inner = _type(t.params[0]) if t.params else "any"
        return f"*{inner}"
    if k == TypeKind.NAMED:
        return getattr(t, "name", "unknown")
    if k == TypeKind.GENERIC:
        return getattr(t, "name", "T")
    if k == TypeKind.INFERRED:
        return ""
    return str(k.value)
