"""Parse Rust source → unified AST using tree-sitter-rust."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

try:
    import tree_sitter_rust as ts_rust
    from tree_sitter import Language, Parser, Node
    RUST_LANGUAGE = Language(ts_rust.language())
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

_IMPORT_COUNTER: List[int] = [0]


def parse(source: str, filename: str = "<string>") -> Module:
    if not _TS_AVAILABLE:
        raise RuntimeError("tree-sitter-rust is not installed. Run: pip install tree-sitter-rust")

    _IMPORT_COUNTER[0] = 0
    parser = Parser(RUST_LANGUAGE)
    tree = parser.parse(bytes(source, "utf8"))
    root = tree.root_node

    # First pass: collect structs/enums/traits by name
    types: Dict[str, TypeDefNode] = {}
    imports: List[ImportNode] = []
    variables: List[VariableNode] = []
    functions: List[FunctionNode] = []
    ordered_ids: List[str] = []  # preserve declaration order

    for child in root.children:
        ctype = child.type
        if ctype in ("line_comment", "block_comment", "attribute_item", "inner_attribute_item"):
            continue

        if ctype == "use_declaration":
            imp = _parse_use(child, source)
            if imp:
                imports.extend(imp if isinstance(imp, list) else [imp])

        elif ctype in ("const_item", "static_item"):
            v = _parse_const(child, source)
            if v:
                variables.append(v)
                ordered_ids.append(v.id)

        elif ctype == "struct_item":
            td = _parse_struct(child, source)
            types[td.name] = td
            ordered_ids.append(td.id)

        elif ctype == "enum_item":
            td = _parse_enum(child, source)
            types[td.name] = td
            ordered_ids.append(td.id)

        elif ctype == "trait_item":
            td = _parse_trait(child, source)
            types[td.name] = td
            ordered_ids.append(td.id)

        elif ctype == "function_item":
            fn = _parse_function(child, source, scope="")
            functions.append(fn)
            ordered_ids.append(fn.id)

        elif ctype == "impl_item":
            _attach_impl(child, source, types)

    # Second pass: build ordered node list
    id_to_node: Dict[str, ASTNode] = {}
    for imp in imports:
        id_to_node[imp.id] = imp
    for v in variables:
        id_to_node[v.id] = v
    for td in types.values():
        id_to_node[td.id] = td
    for fn in functions:
        id_to_node[fn.id] = fn

    nodes: List[ASTNode] = []
    seen = set()
    # imports first (they don't appear in ordered_ids)
    for imp in imports:
        nodes.append(imp)
        seen.add(imp.id)
    for nid in ordered_ids:
        if nid in id_to_node and nid not in seen:
            nodes.append(id_to_node[nid])
            seen.add(nid)

    return Module(source_language="rust", source_file=filename, nodes=nodes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(node: Node, source: str) -> str:
    return source[node.start_byte:node.end_byte]


def _child(node: Node, *types: str) -> Optional[Node]:
    for c in node.children:
        if c.type in types:
            return c
    return None


def _children(node: Node, *types: str) -> List[Node]:
    return [c for c in node.children if c.type in types]


def _named_children(node: Node) -> List[Node]:
    return [c for c in node.children if c.is_named]


def _vis(node: Node) -> Visibility:
    for c in node.children:
        if c.type == "visibility_modifier":
            return Visibility.PUBLIC
    return Visibility.PRIVATE


# ---------------------------------------------------------------------------
# Imports (use declarations)
# ---------------------------------------------------------------------------

def _parse_use(node: Node, source: str) -> List[ImportNode]:
    idx = _IMPORT_COUNTER[0]
    _IMPORT_COUNTER[0] += 1
    # Flatten use tree into import paths
    paths = _flatten_use_tree(node, source)
    results = []
    for i, (module, items, alias) in enumerate(paths):
        results.append(ImportNode(
            id=f"import:{idx + i}:{module}",
            module=module,
            items=items,
            alias=alias,
        ))
    _IMPORT_COUNTER[0] += len(paths) - 1
    return results


def _flatten_use_tree(node: Node, source: str) -> List[Tuple[str, Optional[List[str]], Optional[str]]]:
    """Recursively flatten a use_declaration into (module, items, alias) tuples."""
    # use_declaration: use <use_tree> ;
    # use_tree: scoped_use_list | use_wildcard | use_as_clause | identifier | scoped_identifier
    results = []
    for child in node.children:
        if child.type in ("use_tree", "scoped_use_list", "use_as_clause", "use_wildcard",
                          "identifier", "scoped_identifier"):
            results.extend(_use_tree(child, source, prefix=""))
    return results or [(_text(node, source).replace("use ", "").rstrip(";").strip(), None, None)]


def _use_tree(node: Node, source: str, prefix: str) -> List[Tuple[str, Optional[List[str]], Optional[str]]]:
    t = node.type

    if t == "scoped_identifier":
        # e.g. std::io  or  std::io::Read
        text = _text(node, source)
        if prefix:
            return [(f"{prefix}::{text}", None, None)]
        return [(text, None, None)]

    if t == "identifier":
        name = _text(node, source)
        if name == "self":
            return [(prefix, None, None)]
        full = f"{prefix}::{name}" if prefix else name
        return [(full, None, None)]

    if t == "use_wildcard":
        # prefix::*
        return [(prefix, ["*"], None)]

    if t == "use_as_clause":
        # X as Y
        parts = [c for c in node.children if c.type not in ("::", "as")]
        if len(parts) >= 2:
            path = _text(parts[0], source)
            alias = _text(parts[-1], source)
            full = f"{prefix}::{path}" if prefix else path
            return [(full, None, alias)]
        return [(prefix, None, None)]

    if t == "scoped_use_list":
        # path::{a, b, c}
        path_node = _child(node, "scoped_identifier", "identifier")
        list_node = _child(node, "use_list")
        if path_node:
            new_prefix = _text(path_node, source)
            full_prefix = f"{prefix}::{new_prefix}" if prefix else new_prefix
        else:
            full_prefix = prefix
        if list_node:
            results = []
            for c in list_node.children:
                if c.type not in ("{", "}", ","):
                    results.extend(_use_tree(c, source, full_prefix))
            return results
        return [(full_prefix, None, None)]

    if t == "use_list":
        results = []
        for c in node.children:
            if c.type not in ("{", "}", ","):
                results.extend(_use_tree(c, source, prefix))
        return results

    if t == "use_tree":
        # Recurse into children
        results = []
        for c in node.children:
            if c.type not in ("::", ";", "use"):
                results.extend(_use_tree(c, source, prefix))
        return results or [(prefix, None, None)]

    return [(prefix or _text(node, source), None, None)]


# ---------------------------------------------------------------------------
# Constants / statics
# ---------------------------------------------------------------------------

def _parse_const(node: Node, source: str) -> Optional[VariableNode]:
    name_node = _child(node, "identifier")
    if not name_node:
        return None
    name = _text(name_node, source)
    type_node = _child(node, "type_identifier", "primitive_type", "generic_type",
                       "scoped_type_identifier", "reference_type", "tuple_type",
                       "array_type", "pointer_type", "never_type", "abstract_type",
                       "dynamic_type", "bounded_type", "function_type", "optional_type")
    ty = _parse_type(type_node, source) if type_node else T_INFERRED

    val_node = None
    # value is the expression after =
    found_eq = False
    for c in node.children:
        if c.type == "=":
            found_eq = True
            continue
        if found_eq and c.type not in (";",):
            val_node = c
            break
    value = _text(val_node, source) if val_node else None
    vis = _vis(node)
    is_const = node.type == "const_item"
    return VariableNode(
        id=f"var:{name}",
        name=name,
        visibility=vis,
        is_const=is_const,
        is_static=node.type == "static_item",
        type=ty,
        value=value,
    )


# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------

def _parse_struct(node: Node, source: str) -> TypeDefNode:
    name_node = _child(node, "type_identifier")
    name = _text(name_node, source) if name_node else "Unknown"
    vis = _vis(node)

    type_params = _parse_type_params(node, source)
    fields: List[FieldNode] = []

    field_list = _child(node, "field_declaration_list")
    if field_list:
        for c in field_list.children:
            if c.type == "field_declaration":
                fvis_node = _child(c, "visibility_modifier")
                fvis = Visibility.PUBLIC if fvis_node else Visibility.PRIVATE
                fname_node = _child(c, "field_identifier")
                fname = _text(fname_node, source) if fname_node else "?"
                ftype_nodes = [ch for ch in c.children if ch.type not in
                               ("field_identifier", "visibility_modifier", ":", ",", "attribute_item")]
                fty = _parse_type(ftype_nodes[0], source) if ftype_nodes else T_INFERRED
                fields.append(FieldNode(
                    id=f"field:{name}.{fname}",
                    name=fname,
                    type=fty,
                    visibility=fvis,
                ))

    return TypeDefNode(
        id=f"type:{name}",
        name=name,
        category=TypeDefCategory.STRUCT,
        visibility=vis,
        type_params=type_params,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

def _parse_enum(node: Node, source: str) -> TypeDefNode:
    name_node = _child(node, "type_identifier")
    name = _text(name_node, source) if name_node else "Unknown"
    vis = _vis(node)
    type_params = _parse_type_params(node, source)

    # Enum variants stored as fields with raw text
    fields: List[FieldNode] = []
    body = _child(node, "enum_variant_list")
    if body:
        for c in body.children:
            if c.type == "enum_variant":
                vname_node = _child(c, "identifier")
                vname = _text(vname_node, source) if vname_node else "?"
                fields.append(FieldNode(
                    id=f"field:{name}.{vname}",
                    name=vname,
                    type=T_NAMED(name),
                    visibility=Visibility.PUBLIC,
                    default=_text(c, source),
                ))

    return TypeDefNode(
        id=f"type:{name}",
        name=name,
        category=TypeDefCategory.ENUM,
        visibility=vis,
        type_params=type_params,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# Trait
# ---------------------------------------------------------------------------

def _parse_trait(node: Node, source: str) -> TypeDefNode:
    name_node = _child(node, "type_identifier")
    name = _text(name_node, source) if name_node else "Unknown"
    vis = _vis(node)
    type_params = _parse_type_params(node, source)

    methods: List[FunctionNode] = []
    body = _child(node, "declaration_list")
    if body:
        for c in body.children:
            if c.type == "function_item":
                methods.append(_parse_function(c, source, scope=f"{name}."))
            elif c.type == "function_signature_item":
                fn = _parse_function_signature(c, source, scope=f"{name}.")
                fn.is_abstract = True
                methods.append(fn)

    return TypeDefNode(
        id=f"type:{name}",
        name=name,
        category=TypeDefCategory.TRAIT,
        visibility=vis,
        type_params=type_params,
        methods=methods,
    )


# ---------------------------------------------------------------------------
# Impl block — attach to existing TypeDefNode
# ---------------------------------------------------------------------------

def _attach_impl(node: Node, source: str, types: Dict[str, TypeDefNode]) -> None:
    # impl [Trait for] TypeName { ... }
    # Find the type being implemented
    trait_node = _child(node, "scoped_type_identifier", "generic_type")
    type_id_nodes = _children(node, "type_identifier", "generic_type")

    # impl Trait for Type: last type_identifier is the target
    target_name = None
    for c in node.children:
        if c.type == "type_identifier":
            target_name = _text(c, source)  # keep updating — last one is impl target
        elif c.type == "generic_type":
            inner = _child(c, "type_identifier")
            if inner:
                target_name = _text(inner, source)

    # Check if this is "impl Trait for Type" — the "for" keyword signals it
    has_for = any(c.type == "for" for c in node.children)

    if target_name and target_name in types:
        td = types[target_name]
        body = _child(node, "declaration_list")
        if body:
            for c in body.children:
                if c.type == "function_item":
                    fn = _parse_function(c, source, scope=f"{target_name}.")
                    # Mark trait impl methods
                    if has_for:
                        fn.attributes["trait_impl"] = True
                    td.methods.append(fn)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _parse_function(node: Node, source: str, scope: str) -> FunctionNode:
    name_node = _child(node, "identifier")
    name = _text(name_node, source) if name_node else "unknown"
    fn_id = f"fn:{scope}{name}" if scope else f"fn:{name}"
    vis = _vis(node)
    is_async = any(c.type == "async" for c in node.children)

    type_params = _parse_type_params(node, source)
    params = _parse_fn_params(node, source)
    return_type = _parse_return_type(node, source)

    body_node = _child(node, "block")
    body = _parse_block(body_node, source) if body_node else Block()

    return FunctionNode(
        id=fn_id,
        name=name,
        params=params,
        return_type=return_type,
        body=body,
        visibility=vis,
        is_async=is_async,
        type_params=type_params,
    )


def _parse_function_signature(node: Node, source: str, scope: str) -> FunctionNode:
    """Parse a trait method signature (no body)."""
    fn = _parse_function.__wrapped__(node, source, scope) if hasattr(_parse_function, "__wrapped__") else None
    name_node = _child(node, "identifier")
    name = _text(name_node, source) if name_node else "unknown"
    fn_id = f"fn:{scope}{name}" if scope else f"fn:{name}"
    vis = _vis(node)
    type_params = _parse_type_params(node, source)
    params = _parse_fn_params(node, source)
    return_type = _parse_return_type(node, source)
    return FunctionNode(
        id=fn_id, name=name, params=params, return_type=return_type, body=Block(),
        visibility=vis, type_params=type_params, is_abstract=True,
    )


def _parse_fn_params(node: Node, source: str) -> List[Param]:
    params = []
    params_node = _child(node, "parameters")
    if not params_node:
        return params

    for c in params_node.children:
        if c.type == "self_parameter":
            params.append(Param(name="self", type=T_SELF, is_self=True))
        elif c.type == "parameter":
            pname_node = _child(c, "identifier", "pattern_identifier")
            # Handle mut self, &self, &mut self
            pattern = _child(c, "identifier")
            if not pattern:
                # Skip complex patterns like destructuring
                params.append(Param(name=_text(c, source).split(":")[0].strip().lstrip("&mut ").strip(), type=T_INFERRED))
                continue
            pname = _text(pattern, source)
            # Find type — everything after ":"
            found_colon = False
            ptype_node = None
            for cc in c.children:
                if cc.type == ":":
                    found_colon = True
                    continue
                if found_colon and cc.is_named:
                    ptype_node = cc
                    break
            pty = _parse_type(ptype_node, source) if ptype_node else T_INFERRED
            params.append(Param(name=pname, type=pty))
        elif c.type == "variadic_parameter":
            params.append(Param(name="...", type=T_INFERRED, is_variadic=True))

    return params


def _parse_return_type(node: Node, source: str) -> UnifiedType:
    # Look for -> after parameters
    found_arrow = False
    for c in node.children:
        if c.type == "->":
            found_arrow = True
            continue
        if found_arrow and c.is_named and c.type not in ("where_clause", "block"):
            return _parse_type(c, source)
    return T_VOID


def _parse_type_params(node: Node, source: str) -> List[str]:
    tp_node = _child(node, "type_parameters")
    if not tp_node:
        return []
    return [_text(c, source) for c in tp_node.children
            if c.type in ("type_identifier", "lifetime", "constrained_type_parameter")]


# ---------------------------------------------------------------------------
# Type parsing
# ---------------------------------------------------------------------------

_RUST_PRIM = {
    "i8": lambda: T_NUMBER(8, signed=True),
    "i16": lambda: T_NUMBER(16, signed=True),
    "i32": lambda: T_NUMBER(32, signed=True),
    "i64": lambda: T_NUMBER(64, signed=True),
    "i128": lambda: T_NUMBER(128, signed=True),
    "isize": lambda: T_NUMBER(64, signed=True),
    "u8": lambda: T_NUMBER(8, signed=False),
    "u16": lambda: T_NUMBER(16, signed=False),
    "u32": lambda: T_NUMBER(32, signed=False),
    "u64": lambda: T_NUMBER(64, signed=False),
    "u128": lambda: T_NUMBER(128, signed=False),
    "usize": lambda: T_NUMBER(64, signed=False),
    "f32": lambda: T_NUMBER(32, signed=True, float=True),
    "f64": lambda: T_NUMBER(64, signed=True, float=True),
    "bool": lambda: T_BOOLEAN,
    "str": lambda: T_STRING,
    "String": lambda: T_STRING,
    "char": lambda: T_STRING,
    "()": lambda: T_VOID,
}


def _parse_type(node: Optional[Node], source: str) -> UnifiedType:
    if node is None:
        return T_INFERRED

    t = node.type
    text = _text(node, source)

    if t == "primitive_type":
        factory = _RUST_PRIM.get(text)
        return factory() if factory else T_INFERRED

    if t == "type_identifier":
        factory = _RUST_PRIM.get(text)
        if factory:
            return factory()
        if text in ("T", "K", "V", "S", "E", "U"):
            return T_GENERIC(text)
        return T_NAMED(text)

    if t == "reference_type":
        # &T or &mut T — unwrap the inner type
        inner = [c for c in node.children if c.type not in ("&", "lifetime", "mutable_specifier")]
        return _parse_type(inner[0] if inner else None, source)

    if t in ("mutable_specifier",):
        return T_INFERRED

    if t == "generic_type":
        # Vec<T>, Option<T>, HashMap<K,V>, etc.
        name_node = _child(node, "type_identifier", "scoped_type_identifier")
        args_node = _child(node, "type_arguments")
        name = _text(name_node, source) if name_node else text

        type_args: List[UnifiedType] = []
        if args_node:
            for c in args_node.children:
                if c.type not in ("<", ">", ",", "lifetime"):
                    type_args.append(_parse_type(c, source))

        if name in ("Vec", "VecDeque", "LinkedList"):
            return T_LIST(type_args[0] if type_args else T_INFERRED)
        if name in ("HashMap", "BTreeMap", "IndexMap"):
            k = type_args[0] if len(type_args) > 0 else T_INFERRED
            v = type_args[1] if len(type_args) > 1 else T_INFERRED
            return T_MAP(k, v)
        if name in ("HashSet", "BTreeSet"):
            return T_SET(type_args[0] if type_args else T_INFERRED)
        if name == "Option":
            return T_OPTIONAL(type_args[0] if type_args else T_INFERRED)
        if name == "Result":
            # Result<T, E> — treat as T (the success type)
            return type_args[0] if type_args else T_INFERRED
        if name == "Box":
            return type_args[0] if type_args else T_INFERRED
        if name in ("Arc", "Rc", "Mutex", "RwLock", "Cell", "RefCell"):
            return type_args[0] if type_args else T_INFERRED
        if name in ("Fn", "FnMut", "FnOnce"):
            return UnifiedType(TypeKind.FUNCTION, params=type_args[:-1] if len(type_args) > 1 else [], ret=type_args[-1] if type_args else T_VOID)
        return T_NAMED(name)

    if t == "tuple_type":
        inner = [_parse_type(c, source) for c in node.children if c.type not in ("(", ")", ",")]
        if not inner:
            return T_VOID
        return T_TUPLE(*inner)

    if t == "array_type":
        elem_node = [c for c in node.children if c.type not in ("[", "]", ";") and c.is_named]
        return T_LIST(_parse_type(elem_node[0], source) if elem_node else T_INFERRED)

    if t == "pointer_type":
        inner = [c for c in node.children if c.type not in ("*", "const", "mut")]
        return _parse_type(inner[0] if inner else None, source)

    if t == "scoped_type_identifier":
        # e.g. std::string::String
        parts = text.split("::")
        last = parts[-1]
        factory = _RUST_PRIM.get(last)
        if factory:
            return factory()
        return T_NAMED(last)

    if t == "never_type":
        return T_VOID

    if t == "abstract_type" or t == "dynamic_type":
        # impl Trait / dyn Trait
        inner = [c for c in node.children if c.type not in ("impl", "dyn", "+")]
        return _parse_type(inner[0] if inner else None, source)

    if text == "()":
        return T_VOID

    return T_INFERRED


# ---------------------------------------------------------------------------
# Block / statements
# ---------------------------------------------------------------------------

def _parse_block(node: Node, source: str) -> Block:
    stmts: List[Stmt] = []
    children = [c for c in node.children if c.type not in ("{", "}") and c.type != "line_comment"]

    for i, c in enumerate(children):
        is_last = (i == len(children) - 1)
        stmt = _parse_stmt(c, source, implicit_return=is_last)
        if stmt is not None:
            stmts.append(stmt)

    return Block(stmts=stmts)


def _parse_stmt(node: Node, source: str, implicit_return: bool = False) -> Optional[Stmt]:
    t = node.type

    if t in ("line_comment", "block_comment", "attribute_item", "inner_attribute_item"):
        return None

    if t == "let_declaration":
        return _parse_let(node, source)

    if t == "expression_statement":
        # Expression followed by ;
        expr_children = [c for c in node.children if c.type != ";"]
        if not expr_children:
            return None
        inner = expr_children[0]
        # Delegate statement-like inner nodes back through _parse_stmt
        _STMT_EXPR_TYPES = frozenset({
            "for_expression", "while_expression", "loop_expression",
            "if_expression", "match_expression",
            "return_expression",
            "assignment_expression", "compound_assignment_expr",
        })
        if inner.type in _STMT_EXPR_TYPES:
            result = _parse_stmt(inner, source, implicit_return=False)
            if result is not None:
                return result
        return ExprStmt(expr=_parse_expr(inner, source))

    if t == "return_expression":
        val_children = [c for c in node.children if c.type not in ("return",)]
        val = _parse_expr(val_children[0], source) if val_children else None
        return Return(value=val)

    if t == "if_expression":
        return _parse_if(node, source)

    if t == "while_expression":
        return _parse_while(node, source)

    if t == "loop_expression":
        body_node = _child(node, "block")
        body = _parse_block(body_node, source) if body_node else Block()
        # Wrap in while(true)
        return WhileLoop(cond=Literal(value=True, lit_kind="bool"), body=body)

    if t == "for_expression":
        return _parse_for(node, source)

    if t == "match_expression":
        return _parse_match(node, source)

    if t == "break_expression":
        return Break()

    if t == "continue_expression":
        return Continue()

    if t == "block":
        inner = _parse_block(node, source)
        return inner if inner.stmts else None

    if t == "macro_invocation":
        text = _text(node, source)
        # panic! → Raise
        if text.startswith("panic!"):
            arg = text[6:].strip("!()\n ;")
            return Raise(expr=RawExpr(text=arg))
        return Raw(text=text)

    # assignment expressions appearing as statements
    if t in ("assignment_expression", "compound_assignment_expr"):
        return _parse_assignment_stmt(node, source)

    # Last expr in block without ; is implicit return
    if node.is_named:
        expr = _parse_expr(node, source)
        return ExprStmt(expr=expr, is_implicit_return=implicit_return)

    return Raw(text=_text(node, source))


def _parse_let(node: Node, source: str) -> VarDecl:
    is_mut = any(c.type == "mutable_specifier" for c in node.children)

    # Pattern (variable name)
    pat = _child(node, "identifier")
    if not pat:
        # tuple destructuring or complex pattern
        return VarDecl(name=_text(node, source), type=T_INFERRED, is_mutable=is_mut)
    name = _text(pat, source)

    # Type annotation
    found_colon = False
    type_node = None
    for c in node.children:
        if c.type == ":":
            found_colon = True
            continue
        if found_colon and c.type not in ("=", ";") and c.is_named and not isinstance(c, type(None)):
            type_node = c
            break
    ty = _parse_type(type_node, source) if type_node else T_INFERRED

    # Value
    found_eq = False
    val_node = None
    for c in node.children:
        if c.type == "=":
            found_eq = True
            continue
        if found_eq and c.type != ";" and c.is_named:
            val_node = c
            break
    val = _parse_expr(val_node, source) if val_node else None

    return VarDecl(name=name, type=ty, value=val, is_mutable=is_mut)


def _parse_assignment_stmt(node: Node, source: str) -> Stmt:
    children = [c for c in node.children if c.type not in (",",)]
    if node.type == "compound_assignment_expr":
        # left op= right
        op_node = next((c for c in node.children if c.type not in ("{", "}") and not c.is_named), None)
        named = [c for c in node.children if c.is_named]
        if len(named) >= 2 and op_node:
            return Assign(
                target=_parse_expr(named[0], source),
                op=_text(op_node, source),
                value=_parse_expr(named[1], source),
            )
    # Simple assignment
    named = [c for c in node.children if c.is_named]
    if len(named) >= 2:
        return Assign(target=_parse_expr(named[0], source), op="=", value=_parse_expr(named[1], source))
    return Raw(text=_text(node, source))


def _parse_if(node: Node, source: str) -> If:
    children = list(node.children)
    cond_node = None
    then_node = None
    else_node = None

    i = 0
    while i < len(children):
        c = children[i]
        if c.type == "if":
            i += 1
            continue
        if c.type in ("let_chain", "block"):
            if cond_node is None and c.type != "block":
                cond_node = c
            elif then_node is None and c.type == "block":
                then_node = c
        elif c.is_named and cond_node is None and c.type != "block":
            cond_node = c
        elif c.is_named and then_node is None and c.type == "block":
            then_node = c
        elif c.type in ("else_clause",):
            else_node = c
        i += 1

    # For simple if: first named non-block = cond, then block = then
    named = [c for c in children if c.is_named and c.type not in ("else_clause",)]
    if not cond_node and len(named) >= 2:
        cond_node = named[0]
        then_node = named[1]
    elif not cond_node and named:
        cond_node = named[0]

    cond = _parse_expr(cond_node, source) if cond_node else Literal(value=True, lit_kind="bool")
    then = _parse_block(then_node, source) if then_node and then_node.type == "block" else Block()

    elif_branches = []
    else_block = None

    if else_node:
        else_children = [c for c in else_node.children if c.type not in ("else",)]
        if else_children:
            ec = else_children[0]
            if ec.type == "if_expression":
                sub_if = _parse_if(ec, source)
                elif_branches.append((sub_if.cond, sub_if.then_block))
                elif_branches.extend(sub_if.elif_branches)
                else_block = sub_if.else_block
            elif ec.type == "block":
                else_block = _parse_block(ec, source)

    return If(cond=cond, then_block=then, elif_branches=elif_branches, else_block=else_block)


def _parse_while(node: Node, source: str) -> WhileLoop:
    named = [c for c in node.children if c.is_named]
    cond = _parse_expr(named[0], source) if named else Literal(value=True, lit_kind="bool")
    body_node = _child(node, "block")
    body = _parse_block(body_node, source) if body_node else Block()
    return WhileLoop(cond=cond, body=body)


def _parse_for(node: Node, source: str) -> ForEach:
    # for <pattern> in <expr> <block>
    pat_node = _child(node, "identifier")
    var = _text(pat_node, source) if pat_node else "_"

    named = [c for c in node.children if c.is_named]
    iter_node = None
    block_node = None
    for c in named:
        if c.type == "block":
            block_node = c
        elif c != pat_node and iter_node is None:
            iter_node = c

    iter_expr = _parse_expr(iter_node, source) if iter_node else RawExpr(text="")
    body = _parse_block(block_node, source) if block_node else Block()
    return ForEach(var=var, iter_expr=iter_expr, body=body)


def _parse_match(node: Node, source: str) -> Match:
    named = [c for c in node.children if c.is_named]
    subject_node = named[0] if named else None
    subject = _parse_expr(subject_node, source) if subject_node else RawExpr(text="")

    arms: List[MatchArm] = []
    body_node = _child(node, "match_block")
    if body_node:
        for arm_node in body_node.children:
            if arm_node.type == "match_arm":
                pat_node = _child(arm_node, "match_pattern")
                guard_node = _child(arm_node, "match_guard")
                body_child = [c for c in arm_node.children if c.type in ("block", "expression_statement") or (c.is_named and c != pat_node and c != guard_node)]
                pattern = _text(pat_node, source) if pat_node else "?"
                guard = None
                if guard_node:
                    gcond = [c for c in guard_node.children if c.type != "if" and c.is_named]
                    guard = _parse_expr(gcond[0], source) if gcond else None
                arm_body_node = body_child[-1] if body_child else None
                if arm_body_node and arm_body_node.type == "block":
                    arm_body = _parse_block(arm_body_node, source)
                elif arm_body_node:
                    arm_body = Block(stmts=[ExprStmt(expr=_parse_expr(arm_body_node, source), is_implicit_return=True)])
                else:
                    arm_body = Block()
                arms.append(MatchArm(pattern=pattern, guard=guard, body=arm_body))

    return Match(subject=subject, arms=arms)


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

_RUST_BINOP_MAP = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=",
    "&&": "&&", "||": "||", "&": "&", "|": "|", "^": "^", "<<": "<<", ">>": ">>",
}


def _parse_expr(node: Optional[Node], source: str) -> Expr:
    if node is None:
        return RawExpr(text="")

    t = node.type
    text = _text(node, source)

    if t == "integer_literal":
        try:
            val = int(text.rstrip("uils0123456789").replace("_", "") or text.replace("_", ""))
        except ValueError:
            val = text
        return Literal(value=val, lit_kind="int")

    if t == "float_literal":
        try:
            val = float(text.rstrip("f0123456789").replace("_", "") or text)
        except ValueError:
            val = text
        return Literal(value=val, lit_kind="float")

    if t == "string_literal":
        return Literal(value=text.strip('"'), lit_kind="string")

    if t == "char_literal":
        return Literal(value=text.strip("'"), lit_kind="string")

    if t == "boolean_literal":
        return Literal(value=(text == "true"), lit_kind="bool")

    if t == "identifier":
        if text in ("true", "false"):
            return Literal(value=(text == "true"), lit_kind="bool")
        return Identifier(name=text)

    if t == "self":
        return Identifier(name="self")

    if t == "binary_expression":
        named = [c for c in node.children if c.is_named]
        ops = [c for c in node.children if not c.is_named and c.type not in ("(", ")", " ")]
        if len(named) >= 2 and ops:
            op = _text(ops[0], source)
            return BinaryOp(left=_parse_expr(named[0], source), op=op, right=_parse_expr(named[1], source))

    if t == "unary_expression":
        op_node = next((c for c in node.children if not c.is_named), None)
        operand = next((c for c in node.children if c.is_named), None)
        op = _text(op_node, source) if op_node else "-"
        op = "not" if op == "!" else op
        return UnaryOp(op=op, operand=_parse_expr(operand, source))

    if t == "reference_expression":
        # &expr or &mut expr — just unwrap
        inner = next((c for c in node.children if c.is_named), None)
        return _parse_expr(inner, source)

    if t == "call_expression":
        named = [c for c in node.children if c.is_named]
        func = _parse_expr(named[0], source) if named else RawExpr(text=text)
        args_node = _child(node, "arguments")
        args = []
        if args_node:
            args = [_parse_expr(c, source) for c in args_node.children
                    if c.is_named and c.type not in (")", "(")]
        return Call(func=func, args=args)

    if t == "method_call_expression":
        # obj.method(args)
        receiver = _child(node, *[c.type for c in node.children if c.is_named])
        named = [c for c in node.children if c.is_named]
        obj = _parse_expr(named[0], source) if named else RawExpr(text="")
        method_node = _child(node, "field_identifier")
        method = _text(method_node, source) if method_node else "call"
        func = FieldAccess(object=obj, field_name=method)
        args_node = _child(node, "arguments")
        args = []
        if args_node:
            args = [_parse_expr(c, source) for c in args_node.children
                    if c.is_named and c.type not in (")", "(")]
        return Call(func=func, args=args)

    if t == "field_expression":
        named = [c for c in node.children if c.is_named]
        obj = _parse_expr(named[0], source) if named else RawExpr(text="")
        field_node = _child(node, "field_identifier")
        field = _text(field_node, source) if field_node else "?"
        return FieldAccess(object=obj, field_name=field)

    if t == "index_expression":
        named = [c for c in node.children if c.is_named]
        obj = _parse_expr(named[0], source) if len(named) > 0 else RawExpr(text="")
        idx = _parse_expr(named[1], source) if len(named) > 1 else RawExpr(text="")
        return Index(object=obj, index=idx)

    if t == "array_expression":
        elems = [_parse_expr(c, source) for c in node.children if c.is_named]
        return ListLiteral(elements=elems)

    if t == "tuple_expression":
        elems = [_parse_expr(c, source) for c in node.children if c.is_named]
        return TupleLiteral(elements=elems)

    if t == "if_expression":
        sub = _parse_if(node, source)
        # As expression (conditional)
        if sub.then_block.stmts and not sub.elif_branches:
            then_expr = sub.then_block.stmts[-1]
            else_expr = sub.else_block.stmts[-1] if sub.else_block and sub.else_block.stmts else Literal(value=None, lit_kind="none")
            if isinstance(then_expr, ExprStmt) and isinstance(else_expr, ExprStmt):
                return Conditional(cond=sub.cond, then_expr=then_expr.expr, else_expr=else_expr.expr)
        return RawExpr(text=text)

    if t == "await_expression":
        inner = next((c for c in node.children if c.is_named), None)
        return Await(expr=_parse_expr(inner, source))

    if t == "type_cast_expression" or t == "as_expression":
        named = [c for c in node.children if c.is_named]
        expr_node = named[0] if named else None
        type_node = named[1] if len(named) > 1 else None
        return Cast(
            expr=_parse_expr(expr_node, source),
            target_type=_parse_type(type_node, source),
        )

    if t == "block":
        inner = _parse_block(node, source)
        # A block as expression
        return RawExpr(text=text)

    if t == "closure_expression":
        # |params| body
        params_node = _child(node, "closure_parameters")
        params = []
        if params_node:
            params = [_text(c, source) for c in params_node.children
                      if c.is_named and c.type in ("identifier",)]
        body_children = [c for c in node.children if c.is_named and c != params_node]
        body_expr = _parse_expr(body_children[0], source) if body_children else RawExpr(text="")
        return Lambda(params=params, body=body_expr)

    if t in ("paren_expression",):
        inner = next((c for c in node.children if c.is_named), None)
        return _parse_expr(inner, source)

    if t == "macro_invocation":
        macro_name = ""
        name_node = _child(node, "identifier")
        if name_node:
            macro_name = _text(name_node, source)
        if macro_name in ("vec", "array"):
            # vec![a, b, c]
            token_tree = _child(node, "token_tree")
            if token_tree:
                elems = [_parse_expr(c, source) for c in token_tree.children if c.is_named]
                return ListLiteral(elements=elems)
        if macro_name in ("format", "println", "print", "eprintln"):
            return RawExpr(text=text)
        return RawExpr(text=text)

    if t in ("range_expression",):
        return RawExpr(text=text)

    if t == "assignment_expression":
        named = [c for c in node.children if c.is_named]
        if len(named) >= 2:
            return RawExpr(text=text)

    if t == "compound_assignment_expr":
        return RawExpr(text=text)

    if t == "struct_expression":
        # Counter { field: value, ... } → Call(Counter, kwargs={field: value})
        type_node = next((c for c in node.children if c.type in ("type_identifier", "scoped_type_identifier")), None)
        func_name = _text(type_node, source) if type_node else "Unknown"
        func = Identifier(name=func_name)
        field_init_list = _child(node, "field_initializer_list")
        kwargs = {}
        if field_init_list:
            for c in field_init_list.children:
                if c.type == "field_initializer":
                    fname_node = _child(c, "field_identifier")
                    val_children = [ch for ch in c.children if ch.is_named and ch != fname_node]
                    if fname_node:
                        fname = _text(fname_node, source)
                        val = _parse_expr(val_children[0], source) if val_children else Identifier(name=fname)
                        kwargs[fname] = val
                elif c.type == "shorthand_field_initializer":
                    # `name` shorthand — field name == variable name
                    fname_node = _child(c, "identifier")
                    if fname_node:
                        fname = _text(fname_node, source)
                        kwargs[fname] = Identifier(name=fname)
        return Call(func=func, args=[], kwargs=kwargs)

    return RawExpr(text=text)
