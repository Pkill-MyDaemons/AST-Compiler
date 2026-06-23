"""Parse Python source → unified AST using stdlib ast module."""
from __future__ import annotations
import ast
import textwrap
from typing import List, Optional, Tuple

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

_IMPORT_COUNTER: List[int] = [0]


def parse(source: str, filename: str = "<string>") -> Module:
    _IMPORT_COUNTER[0] = 0
    tree = ast.parse(source, filename=filename)
    nodes: List[ASTNode] = []
    for node in tree.body:
        converted = _convert_top(node, source)
        if converted is not None:
            if isinstance(converted, list):
                nodes.extend(converted)
            else:
                nodes.append(converted)
    return Module(source_language="python", source_file=filename, nodes=nodes)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def _convert_top(node: ast.stmt, source: str):
    if isinstance(node, ast.Import):
        return _import(node)
    if isinstance(node, ast.ImportFrom):
        return _import_from(node)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _function(node, source, scope="")
    if isinstance(node, ast.ClassDef):
        return _classdef(node, source)
    if isinstance(node, ast.Assign):
        return _module_assign(node)
    if isinstance(node, ast.AnnAssign):
        return _module_ann_assign(node)
    # Skip module-level expressions (docstrings, __all__, etc.)
    return None


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def _import(node: ast.Import) -> List[ImportNode]:
    results = []
    for alias in node.names:
        idx = _IMPORT_COUNTER[0]
        _IMPORT_COUNTER[0] += 1
        results.append(ImportNode(
            id=f"import:{idx}:{alias.name}",
            module=alias.name,
            alias=alias.asname,
        ))
    return results


def _import_from(node: ast.ImportFrom) -> ImportNode:
    idx = _IMPORT_COUNTER[0]
    _IMPORT_COUNTER[0] += 1
    module = node.module or ""
    items = [alias.name for alias in node.names] if node.names else None
    return ImportNode(id=f"import:{idx}:{module}", module=module, items=items)


# ---------------------------------------------------------------------------
# Module-level variables
# ---------------------------------------------------------------------------

def _module_assign(node: ast.Assign) -> Optional[VariableNode]:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    name = node.targets[0].id
    value_text = ast.unparse(node.value)
    ty = _infer_literal_type(node.value)
    vis = Visibility.PRIVATE if name.startswith("_") else Visibility.PUBLIC
    return VariableNode(id=f"var:{name}", name=name, visibility=vis, type=ty, value=value_text)


def _module_ann_assign(node: ast.AnnAssign) -> Optional[VariableNode]:
    if not isinstance(node.target, ast.Name):
        return None
    name = node.target.id
    ty = _annotation_to_type(node.annotation)
    value_text = ast.unparse(node.value) if node.value else None
    vis = Visibility.PRIVATE if name.startswith("_") else Visibility.PUBLIC
    is_const = name.isupper()
    return VariableNode(id=f"var:{name}", name=name, visibility=vis, is_const=is_const, type=ty, value=value_text)


def _infer_literal_type(node: ast.expr) -> UnifiedType:
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return T_BOOLEAN
        if isinstance(v, int):
            return T_NUMBER()
        if isinstance(v, float):
            return T_NUMBER(float=True)
        if isinstance(v, str):
            return T_STRING
        if isinstance(v, bytes):
            return T_BYTES
        if v is None:
            return T_VOID
    return T_INFERRED


# ---------------------------------------------------------------------------
# Type annotation → UnifiedType
# ---------------------------------------------------------------------------

def _annotation_to_type(node: Optional[ast.expr]) -> UnifiedType:
    if node is None:
        return T_INFERRED
    if isinstance(node, ast.Constant) and node.value is None:
        return T_VOID
    if isinstance(node, ast.Name):
        return _name_to_type(node.id)
    if isinstance(node, ast.Attribute):
        # e.g. typing.Optional
        return _name_to_type(f"{ast.unparse(node.value)}.{node.attr}")
    if isinstance(node, ast.Subscript):
        return _subscript_to_type(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # X | Y — union, treat as Optional if one side is None
        left = _annotation_to_type(node.left)
        right = _annotation_to_type(node.right)
        if right.kind == TypeKind.VOID:
            return T_OPTIONAL(left)
        if left.kind == TypeKind.VOID:
            return T_OPTIONAL(right)
        return T_ANY
    return T_INFERRED


_PY_PRIMITIVE_MAP = {
    "int": lambda: T_NUMBER(),
    "float": lambda: T_NUMBER(float=True),
    "str": lambda: T_STRING,
    "bool": lambda: T_BOOLEAN,
    "bytes": lambda: T_BYTES,
    "bytearray": lambda: T_BYTES,
    "None": lambda: T_VOID,
    "Any": lambda: T_ANY,
    "typing.Any": lambda: T_ANY,
    "object": lambda: T_ANY,
}


def _name_to_type(name: str) -> UnifiedType:
    factory = _PY_PRIMITIVE_MAP.get(name)
    if factory:
        return factory()
    if name in ("T", "K", "V", "S", "U"):
        return T_GENERIC(name)
    return T_NAMED(name)


def _subscript_to_type(node: ast.Subscript) -> UnifiedType:
    origin = ast.unparse(node.value).split(".")[-1]  # strip typing. prefix
    slice_node = node.slice

    def single() -> UnifiedType:
        return _annotation_to_type(slice_node)

    def pair() -> Tuple[UnifiedType, UnifiedType]:
        if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
            return _annotation_to_type(slice_node.elts[0]), _annotation_to_type(slice_node.elts[1])
        return T_INFERRED, T_INFERRED

    def elems() -> List[UnifiedType]:
        if isinstance(slice_node, ast.Tuple):
            return [_annotation_to_type(e) for e in slice_node.elts]
        return [single()]

    if origin in ("List", "list", "Sequence", "MutableSequence"):
        return T_LIST(single())
    if origin in ("Dict", "dict", "Mapping", "MutableMapping"):
        k, v = pair()
        return T_MAP(k, v)
    if origin in ("Set", "set", "FrozenSet", "frozenset"):
        return T_SET(single())
    if origin in ("Optional",):
        return T_OPTIONAL(single())
    if origin in ("Tuple", "tuple"):
        parts = elems()
        if len(parts) == 2 and isinstance(slice_node, ast.Tuple) and isinstance(slice_node.elts[-1], ast.Constant) and slice_node.elts[-1].value is ...:
            return T_LIST(parts[0])
        return T_TUPLE(*parts)
    if origin in ("Callable",):
        # Callable[[A, B], R]
        if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
            params_node = slice_node.elts[0]
            ret_node = slice_node.elts[1]
            params = [_annotation_to_type(e) for e in params_node.elts] if isinstance(params_node, ast.List) else []
            ret = _annotation_to_type(ret_node)
            return UnifiedType(TypeKind.FUNCTION, params=params, ret=ret)
        return T_ANY
    if origin == "Union":
        parts = elems()
        void_parts = [p for p in parts if p.kind == TypeKind.VOID]
        non_void = [p for p in parts if p.kind != TypeKind.VOID]
        if void_parts and len(non_void) == 1:
            return T_OPTIONAL(non_void[0])
        return T_ANY
    # Unknown generic — treat as named with type params
    return T_NAMED(ast.unparse(node))


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _function(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str, scope: str) -> FunctionNode:
    name = node.name
    fn_id = f"fn:{scope + name}" if scope else f"fn:{name}"
    vis = Visibility.PRIVATE if name.startswith("_") and name != "__init__" else Visibility.PUBLIC

    params = _build_params(node.args)
    return_type = _annotation_to_type(node.returns)

    # Extract docstring
    docstring = None
    body_stmts = node.body
    if body_stmts and isinstance(body_stmts[0], ast.Expr) and isinstance(body_stmts[0].value, ast.Constant) and isinstance(body_stmts[0].value.value, str):
        docstring = body_stmts[0].value.value
        body_stmts = body_stmts[1:]

    body = Block(stmts=[_convert_stmt(s, source) for s in body_stmts])

    decorators = [ast.unparse(d) for d in node.decorator_list]
    type_params: List[str] = []
    # Python 3.12+ type_params
    if hasattr(node, "type_params"):
        type_params = [ast.unparse(tp) for tp in node.type_params]

    return FunctionNode(
        id=fn_id,
        name=name,
        params=params,
        return_type=return_type,
        body=body,
        visibility=vis,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        is_constructor=name == "__init__",
        decorators=decorators,
        type_params=type_params,
        docstring=docstring,
    )


def _build_params(args: ast.arguments) -> List[Param]:
    params = []

    # positional-only (Python 3.8+)
    for i, arg in enumerate(args.posonlyargs):
        default = _get_default(args, i - len(args.posonlyargs))
        params.append(Param(
            name=arg.arg,
            type=_annotation_to_type(arg.annotation),
            default=default,
            is_self=arg.arg == "self" or arg.arg == "cls",
        ))

    # regular args
    offset = len(args.posonlyargs)
    for i, arg in enumerate(args.args):
        default_idx = i - (len(args.args) - len(args.defaults))
        default = ast.unparse(args.defaults[default_idx]) if default_idx >= 0 else None
        params.append(Param(
            name=arg.arg,
            type=_annotation_to_type(arg.annotation),
            default=default,
            is_self=arg.arg == "self" or arg.arg == "cls",
        ))

    # *args
    if args.vararg:
        params.append(Param(
            name=args.vararg.arg,
            type=_annotation_to_type(args.vararg.annotation),
            is_variadic=True,
        ))

    # keyword-only
    for i, arg in enumerate(args.kwonlyargs):
        default = ast.unparse(args.kw_defaults[i]) if args.kw_defaults[i] is not None else None
        params.append(Param(
            name=arg.arg,
            type=_annotation_to_type(arg.annotation),
            default=default,
            is_keyword_only=True,
        ))

    return params


def _get_default(args: ast.arguments, idx: int) -> Optional[str]:
    if idx >= 0 and idx < len(args.defaults):
        return ast.unparse(args.defaults[idx])
    return None


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

def _classdef(node: ast.ClassDef, source: str) -> TypeDefNode:
    name = node.name
    cls_id = f"type:{name}"
    vis = Visibility.PRIVATE if name.startswith("_") else Visibility.PUBLIC

    bases = [ast.unparse(b) for b in node.bases if not isinstance(b, ast.keyword)]
    type_params: List[str] = []
    if hasattr(node, "type_params"):
        type_params = [ast.unparse(tp) for tp in node.type_params]

    docstring = None
    body_nodes = node.body
    if body_nodes and isinstance(body_nodes[0], ast.Expr) and isinstance(body_nodes[0].value, ast.Constant) and isinstance(body_nodes[0].value.value, str):
        docstring = body_nodes[0].value.value
        body_nodes = body_nodes[1:]

    fields: List[FieldNode] = []
    methods: List[FunctionNode] = []
    inner_types: List[TypeDefNode] = []

    for item in body_nodes:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            field_name = item.target.id
            fvis = Visibility.PRIVATE if field_name.startswith("_") else Visibility.PUBLIC
            fields.append(FieldNode(
                id=f"field:{name}.{field_name}",
                name=field_name,
                type=_annotation_to_type(item.annotation),
                visibility=fvis,
                default=ast.unparse(item.value) if item.value else None,
            ))
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    fname = target.id
                    fvis = Visibility.PRIVATE if fname.startswith("_") else Visibility.PUBLIC
                    fields.append(FieldNode(
                        id=f"field:{name}.{fname}",
                        name=fname,
                        type=_infer_literal_type(item.value),
                        visibility=fvis,
                        default=ast.unparse(item.value),
                    ))
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_function(item, source, scope=f"{name}."))
        elif isinstance(item, ast.ClassDef):
            inner_types.append(_classdef(item, source))

    # Determine category
    base_names = [b.split(".")[-1] for b in bases]
    category = TypeDefCategory.CLASS
    if any(b in ("ABC", "ABCMeta") for b in base_names):
        category = TypeDefCategory.INTERFACE

    return TypeDefNode(
        id=cls_id,
        name=name,
        category=category,
        visibility=vis,
        bases=[b for b in bases if b not in ("ABC", "object")],
        type_params=type_params,
        fields=fields,
        methods=methods,
        inner_types=inner_types,
        docstring=docstring,
    )


# ---------------------------------------------------------------------------
# Statement conversion
# ---------------------------------------------------------------------------

def _convert_stmt(node: ast.stmt, source: str) -> Stmt:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Nested function — wrap in Raw for now
        return Raw(text=ast.unparse(node))

    if isinstance(node, ast.ClassDef):
        return Raw(text=ast.unparse(node))

    if isinstance(node, ast.Return):
        val = _convert_expr(node.value) if node.value else None
        return Return(value=val)

    if isinstance(node, ast.Assign):
        if len(node.targets) == 1:
            target = _convert_expr(node.targets[0])
            value = _convert_expr(node.value)
            return Assign(target=target, op="=", value=value)
        return Raw(text=ast.unparse(node))

    if isinstance(node, ast.AugAssign):
        target = _convert_expr(node.target)
        value = _convert_expr(node.value)
        op = _aug_op(node.op) + "="
        return Assign(target=target, op=op, value=value)

    if isinstance(node, ast.AnnAssign):
        name_str = ast.unparse(node.target)
        ty = _annotation_to_type(node.annotation)
        val = _convert_expr(node.value) if node.value else None
        return VarDecl(name=name_str, type=ty, value=val, is_mutable=True)

    if isinstance(node, ast.If):
        cond = _convert_expr(node.test)
        then_block = Block(stmts=[_convert_stmt(s, source) for s in node.body])
        elif_branches = []
        else_block = None
        # Flatten elif chains
        orelse = node.orelse
        while orelse:
            if len(orelse) == 1 and isinstance(orelse[0], ast.If):
                sub = orelse[0]
                elif_branches.append((
                    _convert_expr(sub.test),
                    Block(stmts=[_convert_stmt(s, source) for s in sub.body]),
                ))
                orelse = sub.orelse
            else:
                else_block = Block(stmts=[_convert_stmt(s, source) for s in orelse])
                break
        return If(cond=cond, then_block=then_block, elif_branches=elif_branches, else_block=else_block)

    if isinstance(node, ast.While):
        cond = _convert_expr(node.test)
        body = Block(stmts=[_convert_stmt(s, source) for s in node.body])
        return WhileLoop(cond=cond, body=body)

    if isinstance(node, ast.For):
        if isinstance(node.target, ast.Name):
            var = node.target.id
        else:
            var = ast.unparse(node.target)
        iter_expr = _convert_expr(node.iter)
        body = Block(stmts=[_convert_stmt(s, source) for s in node.body])
        return ForEach(var=var, iter_expr=iter_expr, body=body)

    if isinstance(node, ast.Break):
        return Break()

    if isinstance(node, ast.Continue):
        return Continue()

    if isinstance(node, ast.Raise):
        expr = _convert_expr(node.exc) if node.exc else None
        return Raise(expr=expr)

    if isinstance(node, ast.Expr):
        return ExprStmt(expr=_convert_expr(node.value))

    if isinstance(node, ast.Pass):
        return Raw(text="pass")

    if isinstance(node, ast.Match):
        arms = []
        for case in node.cases:
            guard = _convert_expr(case.guard) if case.guard else None
            body = Block(stmts=[_convert_stmt(s, source) for s in case.body])
            arms.append(MatchArm(pattern=ast.unparse(case.pattern), guard=guard, body=body))
        return Match(subject=_convert_expr(node.subject), arms=arms)

    # Fallback
    return Raw(text=ast.unparse(node))


# ---------------------------------------------------------------------------
# Expression conversion
# ---------------------------------------------------------------------------

_BINOP_MAP = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.Mod: "%", ast.Pow: "**", ast.FloorDiv: "//",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.LShift: "<<", ast.RShift: ">>",
    ast.And: "&&", ast.Or: "||",
}

_CMPOP_MAP = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
    ast.In: "in", ast.NotIn: "not in",
}

_UNOP_MAP = {
    ast.USub: "-", ast.UAdd: "+", ast.Not: "not", ast.Invert: "~",
}

_AUG_MAP = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.Mod: "%", ast.Pow: "**", ast.FloorDiv: "//",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.LShift: "<<", ast.RShift: ">>",
}


def _aug_op(op: ast.operator) -> str:
    return _AUG_MAP.get(type(op), "+")


def _convert_expr(node: Optional[ast.expr]) -> Expr:
    if node is None:
        return Literal(value=None, lit_kind="none")

    if isinstance(node, ast.Constant):
        v = node.value
        if v is None:
            return Literal(value=None, lit_kind="none")
        if isinstance(v, bool):
            return Literal(value=v, lit_kind="bool")
        if isinstance(v, int):
            return Literal(value=v, lit_kind="int")
        if isinstance(v, float):
            return Literal(value=v, lit_kind="float")
        if isinstance(v, str):
            return Literal(value=v, lit_kind="string")
        if isinstance(v, bytes):
            return RawExpr(text=repr(v))
        return Literal(value=repr(v), lit_kind="string")

    if isinstance(node, ast.Name):
        if node.id in ("True", "False"):
            return Literal(value=(node.id == "True"), lit_kind="bool")
        if node.id == "None":
            return Literal(value=None, lit_kind="none")
        return Identifier(name=node.id)

    if isinstance(node, ast.BinOp):
        op = _BINOP_MAP.get(type(node.op), "+")
        return BinaryOp(left=_convert_expr(node.left), op=op, right=_convert_expr(node.right))

    if isinstance(node, ast.BoolOp):
        op = "&&" if isinstance(node.op, ast.And) else "||"
        result = _convert_expr(node.values[0])
        for v in node.values[1:]:
            result = BinaryOp(left=result, op=op, right=_convert_expr(v))
        return result

    if isinstance(node, ast.UnaryOp):
        op = _UNOP_MAP.get(type(node.op), "-")
        return UnaryOp(op=op, operand=_convert_expr(node.operand))

    if isinstance(node, ast.Compare):
        # a < b < c → (a < b) && (b < c)
        result = _convert_expr(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMPOP_MAP.get(type(op_node), "==")
            right = _convert_expr(comparator)
            result = BinaryOp(left=result, op=op, right=right)
        return result

    if isinstance(node, ast.Call):
        func = _convert_expr(node.func)
        args = [_convert_expr(a) for a in node.args if not isinstance(a, ast.Starred)]
        starred = [a for a in node.args if isinstance(a, ast.Starred)]
        kwargs = {kw.arg: _convert_expr(kw.value) for kw in node.keywords if kw.arg}
        result = Call(func=func, args=args, kwargs=kwargs)
        if starred:
            return RawExpr(text=ast.unparse(node))
        return result

    if isinstance(node, ast.Attribute):
        return FieldAccess(object=_convert_expr(node.value), field_name=node.attr)

    if isinstance(node, ast.Subscript):
        return Index(object=_convert_expr(node.value), index=_convert_expr(node.slice))

    if isinstance(node, ast.List):
        return ListLiteral(elements=[_convert_expr(e) for e in node.elts])

    if isinstance(node, ast.Dict):
        pairs = []
        for k, v in zip(node.keys, node.values):
            if k is None:
                return RawExpr(text=ast.unparse(node))
            pairs.append((_convert_expr(k), _convert_expr(v)))
        return DictLiteral(pairs=pairs)

    if isinstance(node, ast.Set):
        return RawExpr(text=ast.unparse(node))

    if isinstance(node, ast.Tuple):
        return TupleLiteral(elements=[_convert_expr(e) for e in node.elts])

    if isinstance(node, ast.IfExp):
        return Conditional(
            cond=_convert_expr(node.test),
            then_expr=_convert_expr(node.body),
            else_expr=_convert_expr(node.orelse),
        )

    if isinstance(node, ast.Lambda):
        params = [a.arg for a in node.args.args]
        return Lambda(params=params, body=_convert_expr(node.body))

    if isinstance(node, ast.Await):
        return Await(expr=_convert_expr(node.value))

    if isinstance(node, ast.JoinedStr):
        return RawExpr(text=ast.unparse(node))

    # Comprehensions, generators, walrus, starred, yield → raw
    return RawExpr(text=ast.unparse(node))
