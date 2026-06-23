"""Generate TypeScript source from a unified AST Module."""
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
        out = _gen_node(node, indent=0)
        if out:
            parts.append(out)
    return "\n\n".join(parts) + "\n"


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
    module = node.module
    # Wrap bare module names in quotes; preserve existing quotes
    mod_str = f'"{module}"'

    if node.items is None:
        # Default import
        alias = node.alias or module.split("/")[-1].split(".")[-1]
        return f"import {alias} from {mod_str};"

    if node.items == [] :
        # Side-effect import
        return f"import {mod_str};"

    if node.items == ["*"]:
        alias = node.alias or "mod"
        return f"import * as {alias} from {mod_str};"

    items = ", ".join(node.items)
    return f'import {{ {items} }} from {mod_str};'


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def _type_str(ty: UnifiedType) -> str:
    k = ty.kind
    if k == TypeKind.NUMBER:
        return "number"  # TS has one number type
    if k == TypeKind.STRING:
        return "string"
    if k == TypeKind.BOOLEAN:
        return "boolean"
    if k == TypeKind.BYTES:
        return "Uint8Array"
    if k == TypeKind.LIST:
        inner = _type_str(ty.element) if ty.element else "unknown"
        return f"{inner}[]"
    if k == TypeKind.MAP:
        ks = _type_str(ty.key) if ty.key else "string"
        vs = _type_str(ty.value) if ty.value else "unknown"
        return f"Map<{ks}, {vs}>"
    if k == TypeKind.SET:
        inner = _type_str(ty.element) if ty.element else "unknown"
        return f"Set<{inner}>"
    if k == TypeKind.OPTIONAL:
        inner = _type_str(ty.element) if ty.element else "unknown"
        return f"{inner} | null"
    if k == TypeKind.TUPLE:
        parts = [_type_str(e) for e in ty.elements] if ty.elements else []
        return f"[{', '.join(parts)}]"
    if k == TypeKind.VOID:
        return "void"
    if k == TypeKind.ANY:
        return "any"
    if k == TypeKind.SELF:
        return "this"
    if k == TypeKind.INFERRED:
        return ""
    if k == TypeKind.NAMED:
        return ty.name or "unknown"
    if k == TypeKind.FUNCTION:
        params = [_type_str(p) for p in ty.params] if ty.params else []
        ret = _type_str(ty.ret) if ty.ret else "void"
        param_strs = [f"p{i}: {t}" for i, t in enumerate(params)]
        return f"({', '.join(param_strs)}) => {ret}"
    if k == TypeKind.GENERIC:
        return ty.name or "T"
    return "any"


def _type_ann(ty: UnifiedType) -> str:
    s = _type_str(ty)
    return f": {s}" if s else ""


# ---------------------------------------------------------------------------
# Module-level variables
# ---------------------------------------------------------------------------

def _gen_variable(node: VariableNode, indent: int) -> str:
    pad = INDENT * indent
    kw = "const" if node.is_const else "let"
    export = "export " if node.visibility == Visibility.PUBLIC else ""
    ty = _type_ann(node.type)
    val = f" = {node.value}" if node.value is not None else ""
    return f"{pad}{export}{kw} {node.name}{ty}{val};"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _gen_function(node: FunctionNode, indent: int, in_class: bool = False) -> str:
    pad = INDENT * indent
    lines: List[str] = []

    if node.docstring:
        lines.append(f"{pad}/**")
        for line in node.docstring.splitlines():
            lines.append(f"{pad} * {line}")
        lines.append(f"{pad} */")

    vis_prefix = ""
    if in_class:
        if node.visibility == Visibility.PRIVATE:
            vis_prefix = "private "
        elif node.visibility == Visibility.PROTECTED:
            vis_prefix = "protected "
        else:
            vis_prefix = "public "
        if node.is_static:
            vis_prefix += "static "
        if node.is_abstract:
            vis_prefix += "abstract "
    else:
        vis_prefix = "export " if node.visibility == Visibility.PUBLIC else ""

    async_prefix = "async " if node.is_async else ""
    tp = f"<{', '.join(node.type_params)}>" if node.type_params else ""
    params_str = _gen_params(node.params)
    ret = _type_str(node.return_type)
    ret_str = f": {ret}" if ret else ""

    if in_class and node.is_constructor:
        sig = f"{pad}{vis_prefix}constructor({params_str})"
    elif in_class:
        sig = f"{pad}{vis_prefix}{async_prefix}{node.name}{tp}({params_str}){ret_str}"
    else:
        sig = f"{pad}{vis_prefix}{async_prefix}function {node.name}{tp}({params_str}){ret_str}"

    if node.is_abstract:
        lines.append(f"{sig};")
        return "\n".join(lines)

    lines.append(f"{sig} {{")
    body_lines = _gen_block(node.body, indent + 1)
    lines.extend(body_lines)
    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _gen_params(params: List[Param]) -> str:
    parts = []
    for p in params:
        if p.is_self:
            continue  # TypeScript uses `this` only for explicit this typing
        name = p.name
        ty = _type_str(p.type)
        if p.is_variadic:
            name = name.lstrip(".")
            part = f"...{name}: {ty}[]" if ty else f"...{name}"
        elif ty:
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
    cat = node.category

    if cat == TypeDefCategory.INTERFACE:
        return _gen_interface(node, pad, indent)
    if cat == TypeDefCategory.ENUM:
        return _gen_enum(node, pad)
    # class, struct → TypeScript class
    return _gen_class(node, pad, indent)


def _gen_class(node: TypeDefNode, pad: str, indent: int) -> List[str] | str:
    export = "export " if node.visibility == Visibility.PUBLIC else ""
    abstract = "abstract " if node.category == TypeDefCategory.INTERFACE else ""
    tp = f"<{', '.join(node.type_params)}>" if node.type_params else ""
    extends = f" extends {node.bases[0]}" if node.bases else ""
    implements = f" implements {', '.join(node.interfaces)}" if node.interfaces else ""

    lines: List[str] = []
    if node.docstring:
        lines.append(f"{pad}/**")
        for line in node.docstring.splitlines():
            lines.append(f"{pad} * {line}")
        lines.append(f"{pad} */")

    lines.append(f"{pad}{export}{abstract}class {node.name}{tp}{extends}{implements} {{")

    for f in node.fields:
        lines.append(_gen_field(f, indent + 1))

    if node.fields and node.methods:
        lines.append("")

    for m in node.methods:
        lines.append(_gen_function(m, indent + 1, in_class=True))
        lines.append("")

    # Remove trailing blank line
    while lines and not lines[-1].strip():
        lines.pop()

    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _gen_interface(node: TypeDefNode, pad: str, indent: int) -> str:
    export = "export " if node.visibility == Visibility.PUBLIC else ""
    tp = f"<{', '.join(node.type_params)}>" if node.type_params else ""
    bases = node.bases + node.interfaces
    extends = f" extends {', '.join(bases)}" if bases else ""

    lines: List[str] = []
    if node.docstring:
        lines.append(f"{pad}/** {node.docstring} */")

    lines.append(f"{pad}{export}interface {node.name}{tp}{extends} {{")

    for f in node.fields:
        ty = _type_str(f.type)
        optional = "?" if f.type.kind == TypeKind.OPTIONAL else ""
        ty_str = f": {_type_str(f.type.element) if f.type.element else 'unknown'}" if f.type.kind == TypeKind.OPTIONAL else (f": {ty}" if ty else "")
        lines.append(f"{pad}{INDENT}{f.name}{optional}{ty_str};")

    for m in node.methods:
        params_str = _gen_params(m.params)
        ret = _type_str(m.return_type)
        ret_str = f": {ret}" if ret else ""
        async_prefix = "async " if m.is_async else ""
        lines.append(f"{pad}{INDENT}{async_prefix}{m.name}({params_str}){ret_str};")

    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _gen_enum(node: TypeDefNode, pad: str) -> str:
    export = "export " if node.visibility == Visibility.PUBLIC else ""
    const_kw = "const " if node.attributes.get("is_const_enum") else ""
    lines = [f"{pad}{export}{const_kw}enum {node.name} {{"]
    for f in node.fields:
        if f.default is not None:
            lines.append(f"{pad}{INDENT}{f.name} = {f.default},")
        else:
            lines.append(f"{pad}{INDENT}{f.name},")
    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _gen_field(node: FieldNode, indent: int) -> str:
    pad = INDENT * indent
    vis_map = {
        Visibility.PRIVATE: "private ",
        Visibility.PROTECTED: "protected ",
        Visibility.PUBLIC: "public ",
        Visibility.INTERNAL: "",
    }
    vis = vis_map.get(node.visibility, "")
    readonly = "" if node.is_mutable else "readonly "
    ty = _type_str(node.type)
    ty_str = f": {ty}" if ty else ""
    if node.default is not None:
        return f"{pad}{vis}{readonly}{node.name}{ty_str} = {node.default};"
    return f"{pad}{vis}{readonly}{node.name}{ty_str};"


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
        import textwrap
        text = textwrap.dedent(stmt.text).strip()
        return [(pad + line if line.strip() else line) for line in text.splitlines()]

    if isinstance(stmt, VarDecl):
        kw = "const" if not stmt.is_mutable else "let"
        ty = _type_ann(stmt.type)
        val = f" = {_gen_expr(stmt.value)}" if stmt.value is not None else ""
        return [f"{pad}{kw} {stmt.name}{ty}{val};"]

    if isinstance(stmt, Assign):
        return [f"{pad}{_gen_expr(stmt.target)} {stmt.op} {_gen_expr(stmt.value)};"]

    if isinstance(stmt, Return):
        if stmt.value is None:
            return [f"{pad}return;"]
        return [f"{pad}return {_gen_expr(stmt.value)};"]

    if isinstance(stmt, If):
        return _gen_if(stmt, indent)

    if isinstance(stmt, WhileLoop):
        lines = [f"{pad}while ({_gen_expr(stmt.cond)}) {{"]
        lines.extend(_gen_block(stmt.body, indent + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, ForEach):
        lines = [f"{pad}for (const {stmt.var} of {_gen_expr(stmt.iter_expr)}) {{"]
        lines.extend(_gen_block(stmt.body, indent + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, Match):
        return _gen_switch(stmt, indent)

    if isinstance(stmt, Break):
        return [f"{pad}break;"]

    if isinstance(stmt, Continue):
        return [f"{pad}continue;"]

    if isinstance(stmt, Raise):
        if stmt.expr is None:
            return [f"{pad}throw new Error();"]
        expr_str = _gen_expr(stmt.expr)
        # If already a Call (new Error), emit directly; otherwise wrap
        if isinstance(stmt.expr, Call):
            return [f"{pad}throw {expr_str};"]
        return [f"{pad}throw new Error(String({expr_str}));"]

    if isinstance(stmt, ExprStmt):
        return [f"{pad}{_gen_expr(stmt.expr)};"]

    if isinstance(stmt, Block):
        lines = [f"{pad}{{"]
        lines.extend(_gen_block(stmt, indent + 1))
        lines.append(f"{pad}}}")
        return lines

    return [f"{pad}// <unknown stmt>"]


def _gen_if(stmt: If, indent: int) -> List[str]:
    pad = INDENT * indent
    lines = [f"{pad}if ({_gen_expr(stmt.cond)}) {{"]
    lines.extend(_gen_block(stmt.then_block, indent + 1))
    for elif_cond, elif_block in stmt.elif_branches:
        lines.append(f"{pad}}} else if ({_gen_expr(elif_cond)}) {{")
        lines.extend(_gen_block(elif_block, indent + 1))
    if stmt.else_block is not None:
        lines.append(f"{pad}}} else {{")
        lines.extend(_gen_block(stmt.else_block, indent + 1))
    lines.append(f"{pad}}}")
    return lines


def _gen_switch(stmt: Match, indent: int) -> List[str]:
    pad = INDENT * indent
    lines = [f"{pad}switch ({_gen_expr(stmt.subject)}) {{"]
    for arm in stmt.arms:
        if arm.pattern == "_":
            lines.append(f"{pad}{INDENT}default: {{")
        else:
            lines.append(f"{pad}{INDENT}case {arm.pattern}: {{")
        lines.extend(_gen_block(arm.body, indent + 2))
        lines.append(f"{pad}{INDENT}{INDENT}break;")
        lines.append(f"{pad}{INDENT}}}")
    lines.append(f"{pad}}}")
    return lines


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

_TS_BINOP_REMAP = {
    "and": "&&",
    "or": "||",
    "not": "!",
    "**": "**",
    "//": "Math.floor",  # handled specially below
    "is": "===",
    "is not": "!==",
    "in": "in",
    "not in": "/* not in */",
}

_TS_UNOP_REMAP = {
    "not": "!",
    "~": "~",
    "-": "-",
    "+": "+",
    "not ": "!",
}


def _gen_expr(expr: Expr) -> str:
    if isinstance(expr, Literal):
        if expr.lit_kind == "none":
            return "null"
        if expr.lit_kind == "bool":
            return "true" if expr.value else "false"
        if expr.lit_kind == "string":
            val = str(expr.value).replace('"', '\\"')
            return f'"{val}"'
        return repr(expr.value) if not isinstance(expr.value, str) else expr.value

    if isinstance(expr, Identifier):
        name = expr.name
        if name == "None":
            return "null"
        if name == "True":
            return "true"
        if name == "False":
            return "false"
        if name == "self":
            return "this"
        return name

    if isinstance(expr, BinaryOp):
        op = expr.op
        if op == "//":
            return f"Math.floor({_gen_expr(expr.left)} / {_gen_expr(expr.right)})"
        op = _TS_BINOP_REMAP.get(op, op)
        # Use === / !== for equality by default
        if op == "==":
            op = "==="
        if op == "!=":
            op = "!=="
        return f"({_gen_expr(expr.left)} {op} {_gen_expr(expr.right)})"

    if isinstance(expr, UnaryOp):
        op = _TS_UNOP_REMAP.get(expr.op, expr.op)
        return f"({op}{_gen_expr(expr.operand)})"

    if isinstance(expr, Call):
        func = _gen_expr(expr.func)
        args = [_gen_expr(a) for a in expr.args]
        kwargs = [f"{k}: {_gen_expr(v)}" for k, v in expr.kwargs.items()]
        # kwargs in TS are object args — emit as object literal if no positional args
        if kwargs and not args:
            return f"{func}({{{', '.join(kwargs)}}})"
        all_args = ", ".join(args)
        return f"{func}({all_args})"

    if isinstance(expr, FieldAccess):
        obj = _gen_expr(expr.object)
        if obj == "self":
            obj = "this"
        return f"{obj}.{expr.field_name}"

    if isinstance(expr, Index):
        return f"{_gen_expr(expr.object)}[{_gen_expr(expr.index)}]"

    if isinstance(expr, ListLiteral):
        elems = ", ".join(_gen_expr(e) for e in expr.elements)
        return f"[{elems}]"

    if isinstance(expr, DictLiteral):
        if not expr.pairs:
            return "{}"
        parts = []
        for k, v in expr.pairs:
            k_str = _gen_expr(k)
            # If key is a string literal, use unquoted shorthand if valid identifier
            if isinstance(k, Literal) and k.lit_kind == "string":
                k_str = str(k.value)
            parts.append(f"{k_str}: {_gen_expr(v)}")
        return "{" + ", ".join(parts) + "}"

    if isinstance(expr, TupleLiteral):
        elems = ", ".join(_gen_expr(e) for e in expr.elements)
        return f"[{elems}]"  # TS tuples are arrays at runtime

    if isinstance(expr, Lambda):
        params = ", ".join(expr.params)
        return f"({params}) => {_gen_expr(expr.body)}"

    if isinstance(expr, Conditional):
        return f"({_gen_expr(expr.cond)} ? {_gen_expr(expr.then_expr)} : {_gen_expr(expr.else_expr)})"

    if isinstance(expr, Await):
        return f"await {_gen_expr(expr.expr)}"

    if isinstance(expr, Cast):
        ty = _type_str(expr.target_type)
        return f"({_gen_expr(expr.expr)} as {ty})" if ty else _gen_expr(expr.expr)

    if isinstance(expr, RawExpr):
        # Translate `self` → `this` in raw text
        return expr.text.replace("self.", "this.")

    return "undefined"
