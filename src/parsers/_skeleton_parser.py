"""SKELETON — copy this file to <lang>_parser.py and fill in the TODO sections.

Steps to adapt:
  1. Replace LANG_NAME (string passed to Module) e.g. "go"
  2. Replace TS_PKG import line with the real tree-sitter package
  3. Fill in the _NODE_* constants below with the actual tree-sitter node type strings
  4. Fill in _PRIM_TYPES with the language's primitive type names
  5. Adjust _parse_top() to handle the language's top-level statement kinds
"""
from __future__ import annotations
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Tree-sitter import — replace with the real package
# ---------------------------------------------------------------------------
try:
    import tree_sitter_REPLACE_ME as _ts_lang      # TODO: replace package name
    from tree_sitter import Language, Parser, Node
    _LANGUAGE = Language(_ts_lang.language())
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

from ..unified_ast.types import (
    UnifiedType, TypeKind,
    T_NUMBER, T_STRING, T_BOOLEAN, T_BYTES, T_VOID, T_ANY, T_SELF, T_INFERRED,
    T_LIST, T_MAP, T_SET, T_OPTIONAL, T_TUPLE, T_NAMED, T_GENERIC,
)
from ..unified_ast.expr import (
    Block, Expr, Stmt,
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess, Index,
    ListLiteral, DictLiteral, TupleLiteral, Lambda, Conditional, Await, Cast, RawExpr,
    VarDecl, Assign, Return, If, WhileLoop, ForEach, Match, MatchArm,
    Break, Continue, Raise, ExprStmt, Raw,
)
from ..unified_ast.nodes import (
    Visibility, TypeDefCategory,
    Param, ImportNode, VariableNode, FieldNode, FunctionNode, TypeDefNode,
    Module, ASTNode,
)

# ---------------------------------------------------------------------------
# Tree-sitter node type constants — set these for your language
# ---------------------------------------------------------------------------

# Top-level declaration kinds
_NODE_IMPORT       = "TODO_import"           # e.g. "import_declaration"
_NODE_FUNCTION     = "TODO_function"         # e.g. "function_declaration"
_NODE_METHOD       = "TODO_method"           # e.g. "method_declaration" (or same as function)
_NODE_CLASS        = "TODO_class"            # e.g. "class_declaration"
_NODE_STRUCT       = "TODO_struct"           # e.g. "struct_declaration" (or same as class)
_NODE_INTERFACE    = "TODO_interface"        # e.g. "interface_declaration"
_NODE_VAR          = "TODO_var"              # e.g. "variable_declaration"
_NODE_CONST        = "TODO_const"            # e.g. "const_declaration"

# Statement kinds (body-level)
_STMT_RETURN       = "TODO_return"           # e.g. "return_statement"
_STMT_IF           = "TODO_if"               # e.g. "if_statement"
_STMT_WHILE        = "TODO_while"            # e.g. "while_statement"
_STMT_FOR          = "TODO_for"              # e.g. "for_statement"
_STMT_FOR_EACH     = "TODO_for_each"         # e.g. "for_in_statement" / "enhanced_for"
_STMT_ASSIGN       = "TODO_assign"           # e.g. "assignment_expression"
_STMT_VAR_DECL     = "TODO_var_decl"         # e.g. "local_variable_declaration"
_STMT_EXPR         = "TODO_expr_stmt"        # e.g. "expression_statement"
_STMT_BREAK        = "TODO_break"            # e.g. "break_statement"
_STMT_CONTINUE     = "TODO_continue"         # e.g. "continue_statement"

# Expression kinds
_EXPR_CALL         = "TODO_call"             # e.g. "call_expression"
_EXPR_BINARY       = "TODO_binary"           # e.g. "binary_expression"
_EXPR_UNARY        = "TODO_unary"            # e.g. "unary_expression"
_EXPR_FIELD        = "TODO_field"            # e.g. "selector_expression"
_EXPR_INDEX        = "TODO_index"            # e.g. "index_expression"
_EXPR_IDENT        = "TODO_ident"            # e.g. "identifier"
_EXPR_INT_LIT      = "TODO_int_lit"          # e.g. "int_literal"
_EXPR_FLOAT_LIT    = "TODO_float_lit"        # e.g. "float_literal"
_EXPR_STRING_LIT   = "TODO_string_lit"       # e.g. "string_literal"
_EXPR_BOOL_TRUE    = "TODO_true"             # e.g. "true"
_EXPR_BOOL_FALSE   = "TODO_false"            # e.g. "false"
_EXPR_NULL         = "TODO_null"             # e.g. "null_literal" / "nil"

# Name field used inside identifier nodes (usually "name" or the node text itself)
_IDENT_FIELD       = "name"                  # tree-sitter field name, or None to use node text

# ---------------------------------------------------------------------------
# Primitive type map: language type string → UnifiedType factory
# ---------------------------------------------------------------------------
_PRIM_TYPES: Dict[str, object] = {
    # TODO: fill in for your language, e.g.:
    # "int":    lambda: T_NUMBER(32, signed=True),
    # "int64":  lambda: T_NUMBER(64, signed=True),
    # "string": lambda: T_STRING,
    # "bool":   lambda: T_BOOLEAN,
    # "void":   lambda: T_VOID,
    # "any":    lambda: T_ANY,
}

# Language identifier (lowercase) — set this in each copy
_LANG = "unknown"

# ---------------------------------------------------------------------------
# Module-level import counter (mutable singleton avoids global state)
# ---------------------------------------------------------------------------
_IMPORT_CTR: List[int] = [0]


# ===========================================================================
# Public entry point
# ===========================================================================

def parse(source: str, filename: str = "<string>") -> Module:
    if not _TS_AVAILABLE:
        pkg = f"tree-sitter-{_LANG}"
        raise RuntimeError(f"{pkg} is not installed. Run: pip install {pkg}")

    _IMPORT_CTR[0] = 0
    parser = Parser(_LANGUAGE)
    tree = parser.parse(bytes(source, "utf8"))
    root = tree.root_node

    nodes: List[ASTNode] = []
    for child in root.children:
        if not child.is_named:
            continue
        converted = _parse_top(child, source)
        if converted is None:
            continue
        if isinstance(converted, list):
            nodes.extend(converted)
        else:
            nodes.append(converted)

    return Module(source_language=_LANG, source_file=filename, nodes=nodes)


# ---------------------------------------------------------------------------
# Top-level dispatcher — customise per language
# ---------------------------------------------------------------------------

def _parse_top(node: "Node", source: str) -> "Optional[ASTNode | List[ASTNode]]":
    t = node.type

    if t == _NODE_IMPORT:
        return _parse_import(node, source)
    if t in (_NODE_FUNCTION, _NODE_METHOD):
        return _parse_function(node, source, scope="")
    if t in (_NODE_CLASS, _NODE_STRUCT):
        return _parse_typedef(node, source, TypeDefCategory.CLASS)
    if t == _NODE_INTERFACE:
        return _parse_typedef(node, source, TypeDefCategory.INTERFACE)
    if t in (_NODE_VAR, _NODE_CONST):
        return _parse_variable(node, source, is_const=(t == _NODE_CONST))
    # Unknown / unsupported top-level node — skip silently
    return None


# ===========================================================================
# Shared tree-sitter helpers
# ===========================================================================

def _text(node: "Node", source: str) -> str:
    return source[node.start_byte:node.end_byte]


def _child(node: "Node", *types: str) -> "Optional[Node]":
    for c in node.children:
        if c.type in types:
            return c
    return None


def _children(node: "Node", *types: str) -> "List[Node]":
    return [c for c in node.children if c.type in types]


def _named(node: "Node") -> "List[Node]":
    return [c for c in node.children if c.is_named]


def _field(node: "Node", field_name: str) -> "Optional[Node]":
    """Return the first child captured under a named field."""
    for c in node.children:
        if hasattr(c, "field_name_for_child"):
            pass
    return node.child_by_field_name(field_name)


def _vis(node: "Node", public_keyword: str = "public") -> Visibility:
    """Detect visibility from modifier keywords."""
    for c in node.children:
        txt = c.type
        if txt == public_keyword or txt == "pub":
            return Visibility.PUBLIC
        if txt in ("private", "protected"):
            return Visibility.PRIVATE
    return Visibility.PUBLIC   # default to public — change per language


# ===========================================================================
# Imports
# ===========================================================================

def _parse_import(node: "Node", source: str) -> "List[ImportNode]":
    """Parse an import/use declaration.  Override per language."""
    idx = _IMPORT_CTR[0]
    _IMPORT_CTR[0] += 1
    # Fallback: store the entire import text as the module path
    raw = _text(node, source)
    return [ImportNode(id=f"import:{idx}:{raw}", module=raw)]


# ===========================================================================
# Variables / constants
# ===========================================================================

def _parse_variable(node: "Node", source: str, is_const: bool = False) -> "Optional[VariableNode]":
    """Parse a module-level variable or constant declaration."""
    name_node = _child(node, _EXPR_IDENT, "identifier", "name")
    if name_node is None:
        name_node = node.child_by_field_name("name")
    name = _text(name_node, source) if name_node else _text(node, source)

    val_node = node.child_by_field_name("value")
    value = _text(val_node, source) if val_node else None

    type_node = node.child_by_field_name("type")
    ty = _parse_type_node(type_node, source) if type_node else T_INFERRED

    vis = _vis(node)
    return VariableNode(
        id=f"var:{name}",
        name=name,
        visibility=vis,
        is_const=is_const,
        type=ty,
        value=value,
    )


# ===========================================================================
# Functions
# ===========================================================================

def _parse_function(node: "Node", source: str, scope: str) -> FunctionNode:
    name_node = (
        node.child_by_field_name("name")
        or _child(node, _EXPR_IDENT, "identifier")
    )
    name = _text(name_node, source) if name_node else "unknown"
    fn_id = f"fn:{scope}{name}"

    params = _parse_params(node, source)
    return_type = _parse_return_type(node, source)
    is_async = any(c.type in ("async", "async_keyword") for c in node.children)

    body_node = node.child_by_field_name("body") or _child(node, "block", "body")
    body = _parse_block(body_node, source) if body_node else Block()

    return FunctionNode(
        id=fn_id,
        name=name,
        params=params,
        return_type=return_type,
        body=body,
        visibility=_vis(node),
        is_async=is_async,
    )


def _parse_params(node: "Node", source: str) -> "List[Param]":
    """Parse a parameter list.  Override per language if needed."""
    params: List[Param] = []
    params_node = (
        node.child_by_field_name("parameters")
        or _child(node, "parameter_list", "formal_parameters",
                  "parameters", "function_parameters")
    )
    if not params_node:
        return params

    for c in params_node.children:
        if not c.is_named:
            continue
        if c.type in (",", "(", ")"):
            continue
        pname_node = c.child_by_field_name("name") or _child(c, _EXPR_IDENT, "identifier")
        if pname_node is None:
            # Self / this parameter
            raw = _text(c, source).strip()
            if raw in ("self", "this", "&self", "&mut self"):
                params.append(Param(name=raw.lstrip("&mut "), type=T_SELF, is_self=True))
            continue
        pname = _text(pname_node, source)
        ptype_node = c.child_by_field_name("type")
        pty = _parse_type_node(ptype_node, source) if ptype_node else T_INFERRED
        params.append(Param(name=pname, type=pty))

    return params


def _parse_return_type(node: "Node", source: str) -> UnifiedType:
    """Parse a function return type annotation.  Override per language."""
    rt_node = node.child_by_field_name("return_type") or node.child_by_field_name("result")
    if rt_node:
        return _parse_type_node(rt_node, source)
    return T_VOID


# ===========================================================================
# Type definitions (class / struct / interface)
# ===========================================================================

def _parse_typedef(node: "Node", source: str, category: TypeDefCategory) -> TypeDefNode:
    name_node = node.child_by_field_name("name") or _child(node, _EXPR_IDENT, "type_identifier")
    name = _text(name_node, source) if name_node else "Unknown"

    fields: List[FieldNode] = []
    methods: List[FunctionNode] = []

    body = (
        node.child_by_field_name("body")
        or _child(node, "class_body", "struct_body", "declaration_list",
                  "field_declaration_list", "block")
    )
    if body:
        for c in body.children:
            if not c.is_named:
                continue
            if c.type in (_NODE_FUNCTION, _NODE_METHOD, "method_declaration"):
                methods.append(_parse_function(c, source, scope=f"{name}."))
            elif c.type in ("field_declaration", "property_declaration", "var_declaration"):
                f = _parse_field(c, source, parent=name)
                if f:
                    fields.append(f)

    return TypeDefNode(
        id=f"type:{name}",
        name=name,
        category=category,
        visibility=_vis(node),
        fields=fields,
        methods=methods,
    )


def _parse_field(node: "Node", source: str, parent: str) -> "Optional[FieldNode]":
    name_node = node.child_by_field_name("name") or _child(node, _EXPR_IDENT, "identifier")
    if not name_node:
        return None
    name = _text(name_node, source)
    type_node = node.child_by_field_name("type")
    ty = _parse_type_node(type_node, source) if type_node else T_INFERRED
    return FieldNode(id=f"field:{parent}.{name}", name=name, type=ty)


# ===========================================================================
# Type parsing
# ===========================================================================

def _parse_type_node(node: "Optional[Node]", source: str) -> UnifiedType:
    if node is None:
        return T_INFERRED
    text = _text(node, source).strip()
    factory = _PRIM_TYPES.get(text)
    if factory:
        return factory()  # type: ignore[operator]
    # Generic/parameterised type — return as named
    return T_NAMED(text)


# ===========================================================================
# Blocks and statements
# ===========================================================================

def _parse_block(node: "Node", source: str) -> Block:
    stmts: List[Stmt] = []
    for c in node.children:
        if not c.is_named or c.type in ("{", "}"):
            continue
        s = _parse_stmt(c, source)
        if s is not None:
            stmts.append(s)
    return Block(stmts=stmts)


def _parse_stmt(node: "Node", source: str) -> "Optional[Stmt]":
    t = node.type

    if t == _STMT_RETURN:
        val_node = _named(node)[0] if _named(node) else None
        return Return(value=_parse_expr(val_node, source) if val_node else None)

    if t == _STMT_IF:
        return _parse_if(node, source)

    if t == _STMT_WHILE:
        cond_node = node.child_by_field_name("condition") or (_named(node)[0] if _named(node) else None)
        body_node = node.child_by_field_name("body") or _child(node, "block")
        cond = _parse_expr(cond_node, source) if cond_node else Literal(value=True, lit_kind="bool")
        body = _parse_block(body_node, source) if body_node else Block()
        return WhileLoop(cond=cond, body=body)

    if t in (_STMT_FOR, _STMT_FOR_EACH):
        return _parse_for(node, source)

    if t in (_STMT_VAR_DECL, _NODE_VAR, _NODE_CONST):
        return _parse_local_var(node, source)

    if t == _STMT_ASSIGN:
        return _parse_assign(node, source)

    if t == _STMT_EXPR:
        inner = _named(node)[0] if _named(node) else None
        if inner is None:
            return None
        # Recurse — some languages wrap statements in expression_statement
        maybe = _parse_stmt(inner, source)
        return maybe if maybe else ExprStmt(expr=_parse_expr(inner, source))

    if t == _STMT_BREAK:
        return Break()

    if t == _STMT_CONTINUE:
        return Continue()

    # Delegate expression nodes directly
    if t in (_EXPR_CALL, _EXPR_BINARY, _EXPR_ASSIGN_EXPR := "assignment_expression"):
        return ExprStmt(expr=_parse_expr(node, source))

    if node.is_named:
        expr = _parse_expr(node, source)
        if not isinstance(expr, RawExpr) or expr.text:
            return ExprStmt(expr=expr)

    return Raw(text=_text(node, source))


def _parse_if(node: "Node", source: str) -> If:
    cond_node = node.child_by_field_name("condition") or node.child_by_field_name("cond")
    then_node = node.child_by_field_name("consequence") or node.child_by_field_name("then")
    else_node = node.child_by_field_name("alternative") or node.child_by_field_name("else")

    if cond_node is None:
        nmd = _named(node)
        cond_node = nmd[0] if nmd else None
        then_node = then_node or (nmd[1] if len(nmd) > 1 else None)

    cond = _parse_expr(cond_node, source) if cond_node else Literal(value=True, lit_kind="bool")
    then_block = _parse_block(then_node, source) if then_node else Block()

    else_block = None
    elif_branches = []
    if else_node:
        if else_node.type == _STMT_IF:
            sub = _parse_if(else_node, source)
            elif_branches.append((sub.cond, sub.then_block))
            elif_branches.extend(sub.elif_branches)
            else_block = sub.else_block
        else:
            else_block = _parse_block(else_node, source)

    return If(cond=cond, then_block=then_block, elif_branches=elif_branches, else_block=else_block)


def _parse_for(node: "Node", source: str) -> "Stmt":
    # For-each style
    var_node = node.child_by_field_name("left") or node.child_by_field_name("var")
    iter_node = node.child_by_field_name("right") or node.child_by_field_name("iter")
    body_node = node.child_by_field_name("body") or _child(node, "block")

    if var_node and iter_node:
        var = _text(var_node, source)
        iter_expr = _parse_expr(iter_node, source)
        body = _parse_block(body_node, source) if body_node else Block()
        return ForEach(var=var, iter_expr=iter_expr, body=body)

    # C-style for(init; cond; post) — fall back
    return Raw(text=_text(node, source))


def _parse_local_var(node: "Node", source: str) -> Stmt:
    name_node = node.child_by_field_name("name") or _child(node, _EXPR_IDENT, "identifier")
    name = _text(name_node, source) if name_node else _text(node, source)
    val_node = node.child_by_field_name("value")
    ty_node = node.child_by_field_name("type")
    ty = _parse_type_node(ty_node, source) if ty_node else T_INFERRED
    val = _parse_expr(val_node, source) if val_node else None
    return VarDecl(name=name, type=ty, value=val)


def _parse_assign(node: "Node", source: str) -> Stmt:
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")
    op_node = node.child_by_field_name("operator")
    op = _text(op_node, source) if op_node else "="
    if left_node and right_node:
        return Assign(
            target=_parse_expr(left_node, source),
            op=op,
            value=_parse_expr(right_node, source),
        )
    return Raw(text=_text(node, source))


# ===========================================================================
# Expression parsing
# ===========================================================================

def _parse_expr(node: "Optional[Node]", source: str) -> Expr:
    if node is None:
        return RawExpr(text="")
    t = node.type

    if t == _EXPR_IDENT or t == "identifier":
        return Identifier(name=_text(node, source))

    if t == _EXPR_INT_LIT or t == "integer_literal":
        raw = _text(node, source).replace("_", "")
        try:
            return Literal(value=int(raw, 0), lit_kind="int")
        except ValueError:
            return Literal(value=raw, lit_kind="int")

    if t == _EXPR_FLOAT_LIT or t == "float_literal":
        raw = _text(node, source).replace("_", "")
        try:
            return Literal(value=float(raw), lit_kind="float")
        except ValueError:
            return Literal(value=raw, lit_kind="float")

    if t == _EXPR_STRING_LIT or t == "string_literal":
        raw = _text(node, source)
        inner = raw.strip('"\'`')
        return Literal(value=inner, lit_kind="string")

    if t == _EXPR_BOOL_TRUE or t == "true":
        return Literal(value=True, lit_kind="bool")

    if t == _EXPR_BOOL_FALSE or t == "false":
        return Literal(value=False, lit_kind="bool")

    if t in (_EXPR_NULL, "null", "nil", "nullptr", "None"):
        return Literal(value=None, lit_kind="none")

    if t == _EXPR_BINARY or t == "binary_expression":
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")
        op_node = node.child_by_field_name("operator")
        if left_node and right_node:
            op = _text(op_node, source) if op_node else "?"
            return BinaryOp(
                left=_parse_expr(left_node, source),
                op=op,
                right=_parse_expr(right_node, source),
            )

    if t == _EXPR_UNARY or t == "unary_expression":
        op_node = node.child_by_field_name("operator")
        operand_node = node.child_by_field_name("operand") or (_named(node)[0] if _named(node) else None)
        op = _text(op_node, source) if op_node else "-"
        return UnaryOp(op=op, operand=_parse_expr(operand_node, source))

    if t == _EXPR_CALL or t == "call_expression":
        func_node = node.child_by_field_name("function") or node.child_by_field_name("callee")
        args_node = node.child_by_field_name("arguments")
        func = _parse_expr(func_node, source) if func_node else RawExpr(text=_text(node, source))
        args: List[Expr] = []
        if args_node:
            args = [_parse_expr(c, source) for c in args_node.children
                    if c.is_named and c.type not in ("(", ")", ",")]
        return Call(func=func, args=args)

    if t == _EXPR_FIELD or t == "selector_expression":
        obj_node = node.child_by_field_name("operand") or node.child_by_field_name("object")
        field_node = node.child_by_field_name("field") or node.child_by_field_name("member")
        obj = _parse_expr(obj_node, source) if obj_node else RawExpr(text="")
        field = _text(field_node, source) if field_node else "?"
        return FieldAccess(object=obj, field_name=field)

    if t == _EXPR_INDEX or t == "index_expression":
        obj_node = node.child_by_field_name("operand") or node.child_by_field_name("object")
        idx_node = node.child_by_field_name("index")
        return Index(
            object=_parse_expr(obj_node, source) if obj_node else RawExpr(text=""),
            index=_parse_expr(idx_node, source) if idx_node else RawExpr(text=""),
        )

    # Parenthesised expression
    if t in ("parenthesized_expression", "paren_expression"):
        inner = _named(node)[0] if _named(node) else None
        return _parse_expr(inner, source)

    # Fallback
    return RawExpr(text=_text(node, source))
