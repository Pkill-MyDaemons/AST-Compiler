"""Parse TypeScript source → unified AST using tree-sitter-typescript."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

try:
    import tree_sitter_typescript as ts_ts
    from tree_sitter import Language, Parser, Node
    TS_LANGUAGE = Language(ts_ts.language_typescript())
    TSX_LANGUAGE = Language(ts_ts.language_tsx())
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


def parse(source: str, filename: str = "<string>", tsx: bool = False) -> Module:
    if not _TS_AVAILABLE:
        raise RuntimeError("tree-sitter-typescript is not installed. Run: pip install tree-sitter-typescript")

    _IMPORT_COUNTER[0] = 0
    lang = TSX_LANGUAGE if tsx else TS_LANGUAGE
    parser = Parser(lang)
    tree = parser.parse(bytes(source, "utf8"))
    root = tree.root_node

    nodes: List[ASTNode] = []
    for child in root.children:
        converted = _convert_top(child, source)
        if converted is None:
            continue
        if isinstance(converted, list):
            nodes.extend(converted)
        else:
            nodes.append(converted)

    language = "tsx" if tsx else "typescript"
    return Module(source_language=language, source_file=filename, nodes=nodes)


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


def _named(node: Node) -> List[Node]:
    return [c for c in node.children if c.is_named]


def _is_exported(node: Node) -> bool:
    """Check if a node is wrapped in an export_statement or has export modifier."""
    if node.type == "export_statement":
        return True
    # Check accessibility/export modifiers
    for c in node.children:
        if c.type in ("export", "public"):
            return True
    return False


def _vis_from_modifiers(node: Node, exported: bool = False) -> Visibility:
    for c in node.children:
        if c.type == "accessibility_modifier":
            t = c.children[0].type if c.children else ""
            if t == "private":
                return Visibility.PRIVATE
            if t == "protected":
                return Visibility.PROTECTED
    return Visibility.PUBLIC if exported else Visibility.PUBLIC


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def _convert_top(node: Node, source: str) -> Optional[ASTNode | List[ASTNode]]:
    t = node.type

    if t in ("comment", "hash_bang_line"):
        return None

    # export statement wraps the real declaration
    if t == "export_statement":
        return _unwrap_export(node, source)

    if t in ("import_declaration", "import_statement"):
        return _parse_import(node, source)

    if t in ("function_declaration", "generator_function_declaration"):
        return _parse_function(node, source, scope="", exported=False)

    if t in ("class_declaration", "abstract_class_declaration"):
        return _parse_class(node, source, exported=False)

    if t == "interface_declaration":
        return _parse_interface(node, source, exported=False)

    if t in ("lexical_declaration", "variable_declaration"):
        return _parse_module_var(node, source, exported=False)

    if t == "type_alias_declaration":
        return None  # skip type aliases for now

    if t == "enum_declaration":
        return _parse_enum(node, source, exported=False)

    if t == "ambient_declaration":
        # declare ... — skip ambient declarations
        return None

    return None


def _unwrap_export(node: Node, source: str) -> Optional[ASTNode | List[ASTNode]]:
    for c in node.children:
        if c.type in ("function_declaration", "generator_function_declaration"):
            return _parse_function(c, source, scope="", exported=True)
        if c.type in ("class_declaration", "abstract_class_declaration"):
            return _parse_class(c, source, exported=True)
        if c.type == "interface_declaration":
            return _parse_interface(c, source, exported=True)
        if c.type in ("lexical_declaration", "variable_declaration"):
            return _parse_module_var(c, source, exported=True)
        if c.type == "enum_declaration":
            return _parse_enum(c, source, exported=True)
        if c.type == "export_clause":
            return None  # re-export: skip
    return None


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def _parse_import(node: Node, source: str) -> List[ImportNode]:
    idx = _IMPORT_COUNTER[0]
    _IMPORT_COUNTER[0] += 1

    # Module path: in import_statement, the string is a direct child (no from_clause wrapper)
    module_path = ""
    str_node = _child(node, "string") or (_child(_child(node, "from_clause"), "string") if _child(node, "from_clause") else None)
    if str_node:
        raw = _text(str_node, source)
        module_path = raw.strip('"\'')

    import_clause = _child(node, "import_clause")
    if not import_clause:
        # side-effect import: import 'module'
        return [ImportNode(id=f"import:{idx}:{module_path}", module=module_path, items=[])]

    results: List[ImportNode] = []

    # Default import: import X from 'module'
    default_id = _child(import_clause, "identifier")
    if default_id:
        alias = _text(default_id, source)
        results.append(ImportNode(
            id=f"import:{idx}:{module_path}",
            module=module_path,
            items=None,
            alias=alias,
        ))
        idx += 1

    # Named imports: import { X, Y as Z } from 'module'
    named_imports = _child(import_clause, "named_imports")
    if named_imports:
        items = []
        for spec in _children(named_imports, "import_specifier"):
            names = [c for c in spec.children if c.type == "identifier"]
            if len(names) == 1:
                items.append(_text(names[0], source))
            elif len(names) == 2:
                items.append(f"{_text(names[0], source)} as {_text(names[1], source)}")
        results.append(ImportNode(
            id=f"import:{idx}:{module_path}",
            module=module_path,
            items=items,
        ))
        idx += 1

    # Namespace import: import * as X from 'module'
    namespace_import = _child(import_clause, "namespace_import")
    if namespace_import:
        alias_node = _child(namespace_import, "identifier")
        alias = _text(alias_node, source) if alias_node else None
        results.append(ImportNode(
            id=f"import:{idx}:{module_path}",
            module=module_path,
            items=["*"],
            alias=alias,
        ))

    _IMPORT_COUNTER[0] = idx
    return results or [ImportNode(id=f"import:{idx}:{module_path}", module=module_path)]


# ---------------------------------------------------------------------------
# Module-level variables
# ---------------------------------------------------------------------------

def _parse_module_var(node: Node, source: str, exported: bool) -> List[VariableNode]:
    is_const = any(c.type == "const" for c in node.children)
    results = []
    for decl in _children(node, "variable_declarator"):
        name_node = _child(decl, "identifier")
        if not name_node:
            continue
        name = _text(name_node, source)

        type_ann = _child(decl, "type_annotation")
        ty = _parse_type_annotation(type_ann, source) if type_ann else _infer_literal_type(decl, source)

        val_node = None
        found_eq = False
        for c in decl.children:
            if c.type == "=":
                found_eq = True
                continue
            if found_eq and c.is_named:
                val_node = c
                break
        value = _text(val_node, source) if val_node else None

        vis = Visibility.PUBLIC if exported else Visibility.PRIVATE
        results.append(VariableNode(
            id=f"var:{name}",
            name=name,
            visibility=vis,
            is_const=is_const,
            type=ty,
            value=value,
        ))
    return results


def _infer_literal_type(node: Node, source: str) -> UnifiedType:
    # Find value node
    val = None
    found_eq = False
    for c in node.children:
        if c.type == "=":
            found_eq = True
            continue
        if found_eq:
            val = c
            break
    if val is None:
        return T_INFERRED
    t = val.type
    if t == "number":
        return T_NUMBER()
    if t == "string":
        return T_STRING
    if t == "true" or t == "false":
        return T_BOOLEAN
    if t == "null":
        return T_VOID
    if t == "array":
        return T_LIST(T_INFERRED)
    if t == "object":
        return T_MAP(T_STRING, T_INFERRED)
    return T_INFERRED


# ---------------------------------------------------------------------------
# Type annotations
# ---------------------------------------------------------------------------

def _parse_type_annotation(node: Node, source: str) -> UnifiedType:
    """node is type_annotation: contains ':' then the type."""
    for c in node.children:
        if c.type not in (":", "readonly"):
            return _parse_ts_type(c, source)
    return T_INFERRED


_TS_PRIM = {
    "number": lambda: T_NUMBER(),
    "string": lambda: T_STRING,
    "boolean": lambda: T_BOOLEAN,
    "bigint": lambda: T_NUMBER(64, signed=True),
    "void": lambda: T_VOID,
    "any": lambda: T_ANY,
    "unknown": lambda: T_ANY,
    "never": lambda: T_VOID,
    "null": lambda: T_VOID,
    "undefined": lambda: T_VOID,
    "object": lambda: T_ANY,
    "symbol": lambda: T_ANY,
}


def _parse_ts_type(node: Optional[Node], source: str) -> UnifiedType:
    if node is None:
        return T_INFERRED

    t = node.type

    if t == "predefined_type":
        name = _text(node, source)
        factory = _TS_PRIM.get(name)
        return factory() if factory else T_NAMED(name)

    if t == "type_identifier":
        name = _text(node, source)
        factory = _TS_PRIM.get(name)
        if factory:
            return factory()
        if name in ("T", "K", "V", "U", "S", "E", "R"):
            return T_GENERIC(name)
        return T_NAMED(name)

    if t == "array_type":
        # T[]
        elem = [c for c in node.children if c.type not in ("[", "]")]
        return T_LIST(_parse_ts_type(elem[0] if elem else None, source))

    if t == "generic_type":
        # Array<T>, Map<K,V>, etc.
        name_node = _child(node, "type_identifier")
        args_node = _child(node, "type_arguments")
        name = _text(name_node, source) if name_node else ""

        args: List[UnifiedType] = []
        if args_node:
            for c in args_node.children:
                if c.type not in ("<", ">", ","):
                    args.append(_parse_ts_type(c, source))

        if name in ("Array", "ReadonlyArray"):
            return T_LIST(args[0] if args else T_INFERRED)
        if name in ("Map", "ReadonlyMap"):
            return T_MAP(args[0] if len(args) > 0 else T_INFERRED, args[1] if len(args) > 1 else T_INFERRED)
        if name in ("Set", "ReadonlySet"):
            return T_SET(args[0] if args else T_INFERRED)
        if name in ("Promise",):
            return args[0] if args else T_INFERRED
        if name == "Optional":
            return T_OPTIONAL(args[0] if args else T_INFERRED)
        if name in ("Record",):
            return T_MAP(args[0] if len(args) > 0 else T_STRING, args[1] if len(args) > 1 else T_INFERRED)
        if name in ("Partial", "Required", "Readonly"):
            return args[0] if args else T_INFERRED
        return T_NAMED(name)

    if t == "union_type":
        # T | null  /  T | undefined  →  Optional<T>
        # Use is_named to skip the `|` punctuation tokens
        members = [_parse_ts_type(c, source) for c in node.children if c.is_named]
        null_like = {TypeKind.VOID}
        non_null = [m for m in members if m.kind not in null_like]
        has_null = any(m.kind in null_like for m in members)
        if has_null and len(non_null) == 1:
            return T_OPTIONAL(non_null[0])
        if len(non_null) == 1:
            return non_null[0]
        return T_ANY

    if t == "intersection_type":
        return T_ANY

    if t == "tuple_type":
        elems = [_parse_ts_type(c, source) for c in node.children if c.type not in ("[", "]", ",")]
        return T_TUPLE(*elems) if elems else T_TUPLE()

    if t == "parenthesized_type":
        inner = [c for c in node.children if c.type not in ("(", ")")]
        return _parse_ts_type(inner[0] if inner else None, source)

    if t == "function_type":
        # (params) => RetType
        params_node = _child(node, "formal_parameters")
        ret_node = [c for c in node.children if c.type not in
                    ("(", ")", "=>", "formal_parameters", ",")]
        params = _parse_params(params_node, source) if params_node else []
        param_types = [p.type for p in params]
        ret = _parse_ts_type(ret_node[0] if ret_node else None, source)
        return UnifiedType(TypeKind.FUNCTION, params=param_types, ret=ret)

    if t == "readonly_type":
        inner = [c for c in node.children if c.type != "readonly"]
        return _parse_ts_type(inner[0] if inner else None, source)

    if t == "literal_type":
        # null, undefined, or string/number/boolean literals used as types
        inner = next((c for c in node.children if c.is_named), None)
        if inner and inner.type in ("null", "undefined"):
            return T_VOID
        return T_ANY

    if t in ("template_literal_type", "conditional_type",
             "mapped_type", "index_type_query", "lookup_type",
             "infer_type", "template_literal_body"):
        return T_ANY

    if t == "nested_type_identifier":
        # A.B — treat as named
        return T_NAMED(_text(node, source).replace(".", "::"))

    return T_INFERRED


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _parse_function(node: Node, source: str, scope: str, exported: bool,
                    is_method: bool = False, accessibility: Visibility = Visibility.PUBLIC) -> FunctionNode:
    name_node = _child(node, "identifier", "property_identifier")
    name = _text(name_node, source) if name_node else "anonymous"
    fn_id = f"fn:{scope}{name}" if scope else f"fn:{name}"

    is_async = any(c.type == "async" for c in node.children)
    is_static = any(c.type == "static" for c in node.children)
    is_abstract = any(c.type == "abstract" for c in node.children)

    params_node = _child(node, "formal_parameters")
    params = _parse_params(params_node, source) if params_node else []

    # Return type annotation
    ret_type_node = _child(node, "type_annotation")
    # In function declarations the return type comes after the params
    return_type = _parse_type_annotation(ret_type_node, source) if ret_type_node else T_INFERRED

    # Body
    body_node = _child(node, "statement_block")
    body = _parse_block(body_node, source) if body_node else Block()

    # Docstring — leading comment on node (simplified: look for JSDoc)
    docstring = _extract_jsdoc(node, source)

    vis = accessibility if is_method else (Visibility.PUBLIC if exported else Visibility.PRIVATE)

    return FunctionNode(
        id=fn_id,
        name=name,
        params=params,
        return_type=return_type,
        body=body,
        visibility=vis,
        is_async=is_async,
        is_static=is_static,
        is_abstract=is_abstract,
        docstring=docstring,
        is_constructor=(name == "constructor"),
        attributes={"is_method": is_method},
    )


def _extract_jsdoc(node: Node, source: str) -> Optional[str]:
    """Try to grab the preceding JSDoc comment."""
    # tree-sitter doesn't attach comments to nodes, so we look at preceding siblings
    # by checking the parent's children list for a comment immediately before this node
    if node.parent is None:
        return None
    siblings = node.parent.children
    for i, c in enumerate(siblings):
        if c.id == node.id and i > 0:
            prev = siblings[i - 1]
            if prev.type == "comment":
                txt = _text(prev, source)
                if txt.startswith("/**"):
                    return txt.strip("/** \n").strip("*/").strip()
    return None


def _parse_params(node: Node, source: str) -> List[Param]:
    params = []
    for c in node.children:
        if c.type in ("(", ")", ","):
            continue
        if c.type in ("required_parameter", "optional_parameter"):
            params.append(_parse_one_param(c, source))
        elif c.type == "rest_element":
            name_node = _child(c, "identifier")
            name = _text(name_node, source) if name_node else "args"
            type_ann = _child(c, "type_annotation")
            ty = _parse_type_annotation(type_ann, source) if type_ann else T_INFERRED
            params.append(Param(name=f"...{name}", type=ty, is_variadic=True))
        elif c.type == "identifier":
            # No type annotation
            params.append(Param(name=_text(c, source), type=T_INFERRED))
        elif c.type in ("assignment_pattern",):
            # param = default
            children = [ch for ch in c.children if ch.is_named]
            name_node = children[0] if children else None
            name = _text(name_node, source) if name_node else "?"
            default_node = children[1] if len(children) > 1 else None
            default = _text(default_node, source) if default_node else None
            params.append(Param(name=name, type=T_INFERRED, default=default))
    return params


def _parse_one_param(node: Node, source: str) -> Param:
    # accessibility modifier (public param shorthand in TS)
    acc = _child(node, "accessibility_modifier")
    is_self = False

    # name: could be identifier, object_pattern, array_pattern, this
    name_node = next(
        (c for c in node.children if c.type in ("identifier", "this")), None
    )
    name = _text(name_node, source) if name_node else _text(node, source).split(":")[0].strip()
    if name == "this":
        is_self = True

    type_ann = _child(node, "type_annotation")
    ty = _parse_type_annotation(type_ann, source) if type_ann else T_INFERRED

    # Default value
    default_node = None
    found_eq = False
    for c in node.children:
        if c.type == "=":
            found_eq = True
            continue
        if found_eq and c.is_named:
            default_node = c
            break
    default = _text(default_node, source) if default_node else None

    is_optional = any(c.type == "?" for c in node.children)
    if is_optional and ty.kind != TypeKind.OPTIONAL:
        ty = T_OPTIONAL(ty)

    return Param(name=name, type=ty, default=default, is_self=is_self)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

def _parse_class(node: Node, source: str, exported: bool) -> TypeDefNode:
    name_node = _child(node, "type_identifier", "identifier")
    name = _text(name_node, source) if name_node else "Anonymous"
    cls_id = f"type:{name}"
    vis = Visibility.PUBLIC if exported else Visibility.PRIVATE
    is_abstract = any(c.type == "abstract" for c in node.children)

    # Type parameters
    tp_node = _child(node, "type_parameters")
    type_params = _parse_type_params_list(tp_node, source) if tp_node else []

    # Extends / implements
    bases: List[str] = []
    interfaces: List[str] = []
    for clause in _children(node, "class_heritage"):
        for c in clause.children:
            if c.type == "extends_clause":
                for t in _named(c):
                    if t.type not in ("extends",):
                        bases.append(_text(t, source))
            elif c.type == "implements_clause":
                for t in _named(c):
                    if t.type not in ("implements",):
                        interfaces.append(_text(t, source))
    # Also check direct children
    for c in node.children:
        if c.type == "extends_clause":
            for t in _named(c):
                bases.append(_text(t, source))
        elif c.type == "implements_clause":
            for t in _named(c):
                interfaces.append(_text(t, source))

    body = _child(node, "class_body")
    fields, methods = _parse_class_body(body, source, class_name=name)

    docstring = _extract_jsdoc(node, source)
    category = TypeDefCategory.CLASS
    if is_abstract:
        category = TypeDefCategory.INTERFACE

    return TypeDefNode(
        id=cls_id,
        name=name,
        category=category,
        visibility=vis,
        bases=bases,
        interfaces=interfaces,
        type_params=type_params,
        fields=fields,
        methods=methods,
        docstring=docstring,
    )


def _parse_class_body(node: Optional[Node], source: str, class_name: str) -> Tuple[List[FieldNode], List[FunctionNode]]:
    fields: List[FieldNode] = []
    methods: List[FunctionNode] = []
    if node is None:
        return fields, methods

    for c in node.children:
        if c.type in ("{", "}"):
            continue

        if c.type in ("method_definition", "abstract_method_signature"):
            acc = _accessibility(c)
            fn = _parse_function(c, source, scope=f"{class_name}.", exported=False,
                                 is_method=True, accessibility=acc)
            fn.is_abstract = c.type == "abstract_method_signature"
            methods.append(fn)

        elif c.type in ("public_field_definition", "field_definition"):
            acc = _accessibility(c)
            is_static = any(ch.type == "static" for ch in c.children)
            name_node = _child(c, "property_identifier", "private_property_identifier")
            if not name_node:
                continue
            fname = _text(name_node, source).lstrip("#")
            type_ann = _child(c, "type_annotation")
            fty = _parse_type_annotation(type_ann, source) if type_ann else T_INFERRED
            # Default value
            val_node = None
            found_eq = False
            for ch in c.children:
                if ch.type == "=":
                    found_eq = True
                    continue
                if found_eq and ch.is_named:
                    val_node = ch
                    break
            default = _text(val_node, source) if val_node else None
            is_readonly = any(ch.type == "readonly" for ch in c.children)
            fields.append(FieldNode(
                id=f"field:{class_name}.{fname}",
                name=fname,
                type=fty,
                visibility=acc,
                default=default,
                is_mutable=not is_readonly,
            ))

    return fields, methods


def _accessibility(node: Node) -> Visibility:
    acc = _child(node, "accessibility_modifier")
    if acc is None:
        return Visibility.PUBLIC
    for c in acc.children:
        if c.type == "private":
            return Visibility.PRIVATE
        if c.type == "protected":
            return Visibility.PROTECTED
    return Visibility.PUBLIC


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

def _parse_interface(node: Node, source: str, exported: bool) -> TypeDefNode:
    name_node = _child(node, "type_identifier")
    name = _text(name_node, source) if name_node else "Unknown"
    vis = Visibility.PUBLIC if exported else Visibility.PRIVATE

    tp_node = _child(node, "type_parameters")
    type_params = _parse_type_params_list(tp_node, source) if tp_node else []

    bases: List[str] = []
    for c in node.children:
        if c.type == "extends_type_clause":
            for t in _named(c):
                bases.append(_text(t, source))

    body = _child(node, "interface_body")
    fields: List[FieldNode] = []
    methods: List[FunctionNode] = []

    if body:
        for c in body.children:
            if c.type in ("{", "}"):
                continue
            if c.type == "property_signature":
                fname_node = _child(c, "property_identifier")
                if not fname_node:
                    continue
                fname = _text(fname_node, source)
                type_ann = _child(c, "type_annotation")
                fty = _parse_type_annotation(type_ann, source) if type_ann else T_INFERRED
                is_optional = any(ch.type == "?" for ch in c.children)
                if is_optional:
                    fty = T_OPTIONAL(fty)
                fields.append(FieldNode(
                    id=f"field:{name}.{fname}",
                    name=fname,
                    type=fty,
                    visibility=Visibility.PUBLIC,
                ))
            elif c.type in ("method_signature", "abstract_method_signature"):
                fn = _parse_function(c, source, scope=f"{name}.", exported=False,
                                     is_method=True)
                fn.is_abstract = True
                methods.append(fn)
            elif c.type in ("call_signature", "construct_signature",
                            "index_signature"):
                pass  # skip for now

    return TypeDefNode(
        id=f"type:{name}",
        name=name,
        category=TypeDefCategory.INTERFACE,
        visibility=vis,
        bases=bases,
        type_params=type_params,
        fields=fields,
        methods=methods,
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

def _parse_enum(node: Node, source: str, exported: bool) -> TypeDefNode:
    name_node = _child(node, "identifier")
    name = _text(name_node, source) if name_node else "Unknown"
    vis = Visibility.PUBLIC if exported else Visibility.PRIVATE
    is_const = any(c.type == "const" for c in node.children)

    body = _child(node, "enum_body")
    fields: List[FieldNode] = []
    if body:
        for c in body.children:
            if c.type == "enum_assignment":
                members = [ch for ch in c.children if ch.is_named]
                if members:
                    vname = _text(members[0], source)
                    val = _text(members[1], source) if len(members) > 1 else None
                    fields.append(FieldNode(
                        id=f"field:{name}.{vname}",
                        name=vname,
                        type=T_NAMED(name),
                        visibility=Visibility.PUBLIC,
                        default=val,
                    ))
            elif c.type == "property_identifier":
                vname = _text(c, source)
                fields.append(FieldNode(
                    id=f"field:{name}.{vname}",
                    name=vname,
                    type=T_NAMED(name),
                    visibility=Visibility.PUBLIC,
                ))

    return TypeDefNode(
        id=f"type:{name}",
        name=name,
        category=TypeDefCategory.ENUM,
        visibility=vis,
        fields=fields,
        attributes={"is_const_enum": is_const},
    )


def _parse_type_params_list(node: Node, source: str) -> List[str]:
    return [_text(c, source) for c in node.children
            if c.type in ("type_identifier", "type_parameter")]


# ---------------------------------------------------------------------------
# Block / statements
# ---------------------------------------------------------------------------

def _parse_block(node: Node, source: str) -> Block:
    stmts: List[Stmt] = []
    for c in node.children:
        if c.type in ("{", "}"):
            continue
        s = _parse_stmt(c, source)
        if s is not None:
            stmts.append(s)
    return Block(stmts=stmts)


def _parse_stmt(node: Node, source: str) -> Optional[Stmt]:
    t = node.type

    if t in ("comment", "empty_statement"):
        return None

    if t == "return_statement":
        val_children = [c for c in node.children if c.type not in ("return", ";")]
        val = _parse_expr(val_children[0], source) if val_children else None
        return Return(value=val)

    if t in ("lexical_declaration", "variable_declaration"):
        return _parse_local_var(node, source)

    if t == "expression_statement":
        expr_children = [c for c in node.children if c.type != ";"]
        if not expr_children:
            return None
        expr_node = expr_children[0]
        # assignment and augmented assignment are expression statements
        if expr_node.type in ("assignment_expression", "augmented_assignment_expression"):
            return _parse_assignment(expr_node, source)
        return ExprStmt(expr=_parse_expr(expr_node, source))

    if t == "if_statement":
        return _parse_if(node, source)

    if t == "while_statement":
        cond_node = _child(node, "parenthesized_expression")
        body_node = _child(node, "statement_block")
        cond = _parse_expr(_unwrap_paren(cond_node, source), source) if cond_node else Literal(value=True, lit_kind="bool")
        body = _parse_block(body_node, source) if body_node else Block()
        return WhileLoop(cond=cond, body=body)

    if t == "do_while_statement":
        body_node = _child(node, "statement_block")
        cond_node = _child(node, "parenthesized_expression")
        body = _parse_block(body_node, source) if body_node else Block()
        cond = _parse_expr(_unwrap_paren(cond_node, source), source) if cond_node else Literal(value=True, lit_kind="bool")
        # do { body } while (cond) → body first, then while
        return Block(stmts=[*body.stmts, WhileLoop(cond=cond, body=body)])

    if t == "for_in_statement":
        return _parse_for_in(node, source)

    if t == "for_statement":
        # C-style for loop → fall back to Raw
        return Raw(text=_text(node, source))

    if t == "switch_statement":
        return _parse_switch(node, source)

    if t == "throw_statement":
        val_children = [c for c in node.children if c.type not in ("throw", ";")]
        val = _parse_expr(val_children[0], source) if val_children else None
        return Raise(expr=val)

    if t == "break_statement":
        return Break()

    if t == "continue_statement":
        return Continue()

    if t == "try_statement":
        return Raw(text=_text(node, source))

    if t == "statement_block":
        return _parse_block(node, source)

    if t in ("function_declaration", "generator_function_declaration"):
        # Nested function
        return Raw(text=_text(node, source))

    if t == "class_declaration":
        return Raw(text=_text(node, source))

    if t == "labeled_statement":
        return Raw(text=_text(node, source))

    if node.is_named:
        return ExprStmt(expr=_parse_expr(node, source))

    return Raw(text=_text(node, source))


def _parse_local_var(node: Node, source: str) -> Stmt:
    is_const = any(c.type == "const" for c in node.children)
    decls = _children(node, "variable_declarator")
    if len(decls) == 1:
        decl = decls[0]
        name_node = _child(decl, "identifier")
        if name_node:
            name = _text(name_node, source)
            type_ann = _child(decl, "type_annotation")
            ty = _parse_type_annotation(type_ann, source) if type_ann else T_INFERRED
            val_node = None
            found_eq = False
            for c in decl.children:
                if c.type == "=":
                    found_eq = True
                    continue
                if found_eq and c.is_named:
                    val_node = c
                    break
            val = _parse_expr(val_node, source) if val_node else None
            return VarDecl(name=name, type=ty, value=val, is_mutable=not is_const)
    return Raw(text=_text(node, source))


def _parse_assignment(node: Node, source: str) -> Stmt:
    named = [c for c in node.children if c.is_named]
    ops = [c for c in node.children if not c.is_named and c.type not in ("(", ")", " ")]
    if len(named) >= 2 and ops:
        op = _text(ops[0], source)
        return Assign(target=_parse_expr(named[0], source), op=op, value=_parse_expr(named[1], source))
    if len(named) >= 2:
        return Assign(target=_parse_expr(named[0], source), op="=", value=_parse_expr(named[1], source))
    return Raw(text=_text(node, source))


def _parse_if(node: Node, source: str) -> If:
    cond_node = _child(node, "parenthesized_expression")
    cond = _parse_expr(_unwrap_paren(cond_node, source), source) if cond_node else Literal(value=True, lit_kind="bool")

    bodies = _children(node, "statement_block")
    then_node = bodies[0] if bodies else None
    then = _parse_block(then_node, source) if then_node else Block()

    elif_branches = []
    else_block = None

    else_clause = _child(node, "else_clause")
    if else_clause:
        else_children = [c for c in else_clause.children if c.type not in ("else",)]
        if else_children:
            ec = else_children[0]
            if ec.type == "if_statement":
                sub = _parse_if(ec, source)
                elif_branches.append((sub.cond, sub.then_block))
                elif_branches.extend(sub.elif_branches)
                else_block = sub.else_block
            elif ec.type == "statement_block":
                else_block = _parse_block(ec, source)

    return If(cond=cond, then_block=then, elif_branches=elif_branches, else_block=else_block)


def _parse_for_in(node: Node, source: str) -> Stmt:
    # for (const x of iterable) / for (const x in object)
    has_of = any(c.type == "of" for c in node.children)
    has_in = any(c.type == "in" for c in node.children)

    # Find the variable name
    var = "_"
    for c in node.children:
        if c.type in ("identifier",):
            var = _text(c, source)
            break
        if c.type in ("lexical_declaration", "variable_declaration"):
            id_node = _child(c, "identifier") or _child(c, "variable_declarator")
            if id_node:
                inner = _child(id_node, "identifier") or id_node
                var = _text(inner, source)
            break

    # Find the iterable (after 'of' or 'in')
    found_kw = False
    iter_node = None
    for c in node.children:
        if c.type in ("of", "in"):
            found_kw = True
            continue
        if found_kw and c.is_named and c.type != "statement_block":
            iter_node = c
            break

    iter_expr = _parse_expr(iter_node, source) if iter_node else RawExpr(text="")

    body_node = _child(node, "statement_block")
    body = _parse_block(body_node, source) if body_node else Block()

    if has_of:
        return ForEach(var=var, iter_expr=iter_expr, body=body)
    # for...in iterates keys — fall back to raw
    return Raw(text=_text(node, source))


def _parse_switch(node: Node, source: str) -> Match:
    val_node = _child(node, "parenthesized_expression")
    subject = _parse_expr(_unwrap_paren(val_node, source), source) if val_node else RawExpr(text="")

    body_node = _child(node, "switch_body")
    arms: List[MatchArm] = []

    if body_node:
        current_stmts: List[Stmt] = []
        current_pattern: Optional[str] = None

        for c in body_node.children:
            if c.type == "switch_case":
                if current_pattern is not None:
                    arms.append(MatchArm(
                        pattern=current_pattern,
                        body=Block(stmts=current_stmts),
                    ))
                val_children = [ch for ch in c.children if ch.type not in ("case", ":")]
                current_pattern = _text(val_children[0], source) if val_children else "?"
                # Collect statements after the colon
                current_stmts = []
                found_colon = False
                for ch in c.children:
                    if ch.type == ":":
                        found_colon = True
                        continue
                    if found_colon:
                        s = _parse_stmt(ch, source)
                        if s:
                            current_stmts.append(s)
            elif c.type == "switch_default":
                if current_pattern is not None:
                    arms.append(MatchArm(pattern=current_pattern, body=Block(stmts=current_stmts)))
                current_pattern = "_"
                current_stmts = []
                found_colon = False
                for ch in c.children:
                    if ch.type == ":":
                        found_colon = True
                        continue
                    if found_colon:
                        s = _parse_stmt(ch, source)
                        if s:
                            current_stmts.append(s)

        if current_pattern is not None:
            arms.append(MatchArm(pattern=current_pattern, body=Block(stmts=current_stmts)))

    return Match(subject=subject, arms=arms)


def _unwrap_paren(node: Optional[Node], source: str) -> Optional[Node]:
    if node is None:
        return None
    inner = [c for c in node.children if c.type not in ("(", ")")]
    return inner[0] if inner else node


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

_TS_BINOP_REMAP = {
    "===": "==",
    "!==": "!=",
    "**": "**",
    "??": "??",
    "instanceof": "instanceof",
    "in": "in",
}


def _parse_expr(node: Optional[Node], source: str) -> Expr:
    if node is None:
        return RawExpr(text="")

    t = node.type

    if t == "number":
        raw = _text(node, source)
        try:
            val = int(raw)
            return Literal(value=val, lit_kind="int")
        except ValueError:
            try:
                val = float(raw)
                return Literal(value=val, lit_kind="float")
            except ValueError:
                return RawExpr(text=raw)

    if t == "string":
        raw = _text(node, source)
        return Literal(value=raw.strip('"\'`'), lit_kind="string")

    if t in ("true", "false"):
        return Literal(value=(t == "true"), lit_kind="bool")

    if t in ("null", "undefined"):
        return Literal(value=None, lit_kind="none")

    if t == "identifier":
        name = _text(node, source)
        if name == "true":
            return Literal(value=True, lit_kind="bool")
        if name == "false":
            return Literal(value=False, lit_kind="bool")
        if name in ("null", "undefined", "None"):
            return Literal(value=None, lit_kind="none")
        return Identifier(name=name)

    if t == "this":
        return Identifier(name="self")

    if t == "binary_expression":
        named = [c for c in node.children if c.is_named]
        ops = [c for c in node.children if not c.is_named]
        if len(named) >= 2 and ops:
            op = _text(ops[0], source)
            op = _TS_BINOP_REMAP.get(op, op)
            return BinaryOp(left=_parse_expr(named[0], source), op=op, right=_parse_expr(named[1], source))

    if t == "unary_expression":
        ops = [c for c in node.children if not c.is_named]
        operand = next((c for c in node.children if c.is_named), None)
        op = _text(ops[0], source) if ops else "-"
        op = "not" if op == "!" else op
        return UnaryOp(op=op, operand=_parse_expr(operand, source))

    if t == "call_expression":
        named = [c for c in node.children if c.is_named]
        func = _parse_expr(named[0], source) if named else RawExpr(text="")
        args_node = _child(node, "arguments")
        args = []
        if args_node:
            args = [_parse_expr(c, source) for c in args_node.children
                    if c.is_named and c.type not in (")", "(")]
        return Call(func=func, args=args)

    if t == "new_expression":
        # new Foo(args) → Call(Foo, args)
        constructor_node = next((c for c in node.children if c.type in ("identifier", "member_expression")), None)
        func = _parse_expr(constructor_node, source) if constructor_node else RawExpr(text="")
        args_node = _child(node, "arguments")
        args = []
        if args_node:
            args = [_parse_expr(c, source) for c in args_node.children
                    if c.is_named and c.type not in (")", "(")]
        return Call(func=func, args=args)

    if t == "member_expression":
        named = [c for c in node.children if c.is_named]
        obj = _parse_expr(named[0], source) if named else RawExpr(text="")
        prop_node = _child(node, "property_identifier", "identifier")
        field = _text(prop_node, source) if prop_node else "?"
        return FieldAccess(object=obj, field_name=field)

    if t == "subscript_expression":
        named = [c for c in node.children if c.is_named]
        obj = _parse_expr(named[0], source) if named else RawExpr(text="")
        idx = _parse_expr(named[1], source) if len(named) > 1 else RawExpr(text="")
        return Index(object=obj, index=idx)

    if t == "array":
        elems = [_parse_expr(c, source) for c in node.children
                 if c.is_named and c.type not in ("[", "]", ",")]
        return ListLiteral(elements=elems)

    if t == "object":
        pairs = []
        for c in node.children:
            if c.type == "pair":
                kv = [ch for ch in c.children if ch.is_named and ch.type not in (":",)]
                if len(kv) >= 2:
                    k = _parse_expr(kv[0], source)
                    v = _parse_expr(kv[1], source)
                    pairs.append((k, v))
                elif len(kv) == 1:
                    # shorthand: { x } → { x: x }
                    k = _parse_expr(kv[0], source)
                    pairs.append((k, k))
            elif c.type == "shorthand_property_identifier":
                name = _text(c, source)
                k = Literal(value=name, lit_kind="string")
                v = Identifier(name=name)
                pairs.append((k, v))
        return DictLiteral(pairs=pairs)

    if t == "arrow_function":
        params_node = _child(node, "formal_parameters")
        if params_node:
            params = [p.name for p in _parse_params(params_node, source)]
        else:
            # Single param without parens
            id_node = _child(node, "identifier")
            params = [_text(id_node, source)] if id_node else []
        body_children = [c for c in node.children if c.type not in ("=>", "formal_parameters") and c.type != "identifier" or (c.type == "identifier" and not params)]
        body_node = next((c for c in node.children if c.type in ("statement_block",)), None)
        if body_node:
            return RawExpr(text=_text(node, source))  # block arrow fn → Raw
        expr_node = next((c for c in node.children if c.is_named and c.type not in ("formal_parameters",)), None)
        body_expr = _parse_expr(expr_node, source) if expr_node else RawExpr(text="")
        return Lambda(params=params, body=body_expr)

    if t == "parenthesized_expression":
        inner = _unwrap_paren(node, source)
        return _parse_expr(inner, source)

    if t == "conditional_expression":
        named = [c for c in node.children if c.is_named]
        if len(named) >= 3:
            return Conditional(
                cond=_parse_expr(named[0], source),
                then_expr=_parse_expr(named[1], source),
                else_expr=_parse_expr(named[2], source),
            )

    if t == "await_expression":
        inner = next((c for c in node.children if c.is_named), None)
        return Await(expr=_parse_expr(inner, source))

    if t in ("as_expression", "type_assertion"):
        # expr as Type  or  <Type>expr
        named = [c for c in node.children if c.is_named]
        expr_node = named[0] if named else None
        type_node = named[1] if len(named) > 1 else None
        return Cast(
            expr=_parse_expr(expr_node, source),
            target_type=_parse_ts_type(type_node, source) if type_node else T_INFERRED,
        )

    if t == "template_string":
        return RawExpr(text=_text(node, source))

    if t == "non_null_expression":
        # expr! — unwrap
        inner = next((c for c in node.children if c.is_named), None)
        return _parse_expr(inner, source)

    if t == "optional_chain":
        return RawExpr(text=_text(node, source))

    if t == "assignment_expression":
        named = [c for c in node.children if c.is_named]
        if len(named) >= 2:
            return RawExpr(text=_text(node, source))

    if t == "augmented_assignment_expression":
        return RawExpr(text=_text(node, source))

    if t in ("spread_element", "sequence_expression", "yield_expression",
             "typeof_expression", "delete_expression", "void_expression"):
        return RawExpr(text=_text(node, source))

    if t == "property_identifier":
        return Identifier(name=_text(node, source))

    if node.is_named:
        return RawExpr(text=_text(node, source))

    return RawExpr(text=_text(node, source))
