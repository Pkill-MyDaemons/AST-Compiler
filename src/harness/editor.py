"""Edit operations on an in-memory Module.

All operations return the modified Module (mutations are in-place too).
Raises ValueError with a descriptive message on failure.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple

# Expression kinds that live inside statements (not statement-level nodes)
_EXPR_KINDS = frozenset({
    "literal", "identifier", "binary_op", "unary_op", "call",
    "field_access", "index", "list_literal", "dict_literal",
    "tuple_literal", "lambda", "conditional", "await", "cast", "raw_expr",
})

from ..unified_ast.nodes import (
    Module, ASTNode, ImportNode, VariableNode, FieldNode, FunctionNode, TypeDefNode,
    Param, Visibility, TypeDefCategory, _node_from_dict, _field_from_dict,
)
from ..unified_ast.types import UnifiedType
from ..unified_ast.expr import Block, Stmt, stmt_from_dict, expr_from_dict


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_SKIP_IF_EMPTY = frozenset({"kwargs", "elif", "decorators", "type_params", "attributes"})
_SKIP_IF_NONE  = frozenset({"docstring", "else", "default", "value"})
_SKIP_IF_FALSE = frozenset({"is_async", "is_static", "is_constructor", "is_implicit_return", "is_self"})
_SKIP_IF_TRUE  = frozenset({"is_mutable"})
_SKIP_IF_PUB   = frozenset({"visibility"})


def _prune(d: Any) -> Any:
    """Strip default/empty fields to reduce token count by ~50%."""
    if isinstance(d, list):
        return [_prune(x) for x in d]
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        v = _prune(v)
        if k in _SKIP_IF_EMPTY and v in ({}, []):
            continue
        if k in _SKIP_IF_NONE and v is None:
            continue
        if k in _SKIP_IF_FALSE and v is False:
            continue
        if k in _SKIP_IF_TRUE and v is True:
            continue
        if k in _SKIP_IF_PUB and v == "public":
            continue
        out[k] = v
    return out


def get_node(module: Module, node_id: str) -> Dict[str, Any]:
    """Return the full dict of a node by ID (searches recursively)."""
    node = _find_node(module, node_id)
    if node is None:
        raise ValueError(f"Node not found: {node_id!r}")
    return _prune(node.to_dict())


def _find_node(module: Module, node_id: str) -> Optional[ASTNode]:
    for node in module.nodes:
        found = _search(node, node_id)
        if found is not None:
            return found
    return None


def _search(node: ASTNode, node_id: str) -> Optional[ASTNode]:
    if node.id == node_id:
        return node
    if isinstance(node, TypeDefNode):
        for m in node.methods:
            if m.id == node_id:
                return m
        for f in node.fields:
            if f.id == node_id:
                return f
        for inner in node.inner_types:
            found = _search(inner, node_id)
            if found is not None:
                return found
    return None


def _find_function(module: Module, fn_id: str) -> FunctionNode:
    node = _find_node(module, fn_id)
    if node is None:
        raise ValueError(f"Node not found: {fn_id!r}")
    if not isinstance(node, FunctionNode):
        raise ValueError(f"Node {fn_id!r} is not a function (got {node.kind!r})")
    return node


def _find_typedef(module: Module, type_id: str) -> TypeDefNode:
    node = _find_node(module, type_id)
    if node is None:
        raise ValueError(f"Node not found: {type_id!r}")
    if not isinstance(node, TypeDefNode):
        raise ValueError(f"Node {type_id!r} is not a type_def (got {node.kind!r})")
    return node


# ---------------------------------------------------------------------------
# str_replace_body — replace one statement in a function body
# ---------------------------------------------------------------------------

def str_replace_body(
    module: Module,
    fn_id: str,
    old_stmt_json: str,
    new_stmt_json: str,
) -> Module:
    """Find the statement matching old_stmt_json in fn's body and replace it with new_stmt_json.

    Matching is done by comparing the serialized dict of each statement. The match
    must be unique (exactly one occurrence). Both arguments should be valid JSON
    that deserialises to a statement dict.
    """
    fn = _find_function(module, fn_id)

    try:
        old_dict = json.loads(old_stmt_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"old_stmt_json is not valid JSON: {e}") from e
    try:
        new_dict = json.loads(new_stmt_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"new_stmt_json is not valid JSON: {e}") from e

    indices = _find_stmt_indices(fn.body, old_dict)

    if not indices and old_dict.get("kind") in _EXPR_KINDS:
        # Expression-level fallback: deep-search the body dict for the expression
        # and patch it in-place, then reconstruct the whole body.
        body_dict = fn.body.to_dict()
        new_body_dict, changed = _deep_replace_matching(body_dict, old_dict, new_dict)
        if not changed:
            raise ValueError(
                f"Expression not found anywhere in body of {fn_id!r}.\n"
                f"Searched for: {json.dumps(old_dict)}\n"
                f"Tip: use `ast-harness get {fn_id}` to see the exact JSON."
            )
        fn.body = stmt_from_dict(new_body_dict)
        return module

    if not indices:
        # Show the first few statements so the model can correct its JSON
        preview = json.dumps([_prune(s.to_dict()) for s in fn.body.stmts[:3]], separators=(",", ":"))
        raise ValueError(
            f"Statement not found in {fn_id!r}.\n"
            f"Searched for: {json.dumps(old_dict)}\n"
            f"First statements in body:\n{preview}"
        )
    if len(indices) > 1:
        raise ValueError(
            f"Statement is ambiguous ({len(indices)} matches) in {fn_id!r}. "
            "Make old_stmt_json more specific."
        )

    new_stmt = stmt_from_dict(new_dict)
    _replace_at_path(fn.body, indices[0], new_stmt)
    return module


def _find_stmt_indices(block: Block, target: dict, prefix: tuple = ()) -> List[tuple]:
    """Recursively find all paths to statements matching target dict."""
    results = []
    for i, stmt in enumerate(block.stmts):
        path = prefix + (i,)
        stmt_dict = stmt.to_dict()
        if _dict_matches(stmt_dict, target):
            results.append(path)
        # Recurse into nested blocks
        nested = _get_nested_blocks(stmt)
        for j, nested_block in enumerate(nested):
            results.extend(_find_stmt_indices(nested_block, target, path + ("nested", j)))
    return results


def _dict_matches(actual: dict, pattern: dict) -> bool:
    """Check if pattern is a subset of actual (partial matching for convenience)."""
    for k, v in pattern.items():
        if k not in actual:
            return False
        if isinstance(v, dict) and isinstance(actual[k], dict):
            if not _dict_matches(actual[k], v):
                return False
        elif actual[k] != v:
            return False
    return True


def _deep_replace_matching(d: Any, old_pattern: dict, new_fields: dict) -> tuple:
    """Find the first sub-dict matching old_pattern and replace it with {**match, **new_fields}.

    Returns (result, was_replaced). Stops at the first match (depth-first left-to-right).
    """
    if isinstance(d, dict):
        if _dict_matches(d, old_pattern):
            return {**d, **new_fields}, True
        new_d = {}
        replaced = False
        for k, v in d.items():
            if not replaced:
                new_v, replaced = _deep_replace_matching(v, old_pattern, new_fields)
                new_d[k] = new_v
            else:
                new_d[k] = v
        return new_d, replaced
    if isinstance(d, list):
        new_list = []
        replaced = False
        for item in d:
            if not replaced:
                new_item, replaced = _deep_replace_matching(item, old_pattern, new_fields)
                new_list.append(new_item)
            else:
                new_list.append(item)
        return new_list, replaced
    return d, False


def _get_nested_blocks(stmt: Stmt) -> List[Block]:
    from ..unified_ast.expr import If, WhileLoop, ForEach, Match, Block as B
    blocks = []
    if isinstance(stmt, If):
        blocks.append(stmt.then_block)
        for _, eb in stmt.elif_branches:
            blocks.append(eb)
        if stmt.else_block:
            blocks.append(stmt.else_block)
    elif isinstance(stmt, (WhileLoop, ForEach)):
        blocks.append(stmt.body)
    elif isinstance(stmt, Match):
        for arm in stmt.arms:
            blocks.append(arm.body)
    elif isinstance(stmt, B):
        blocks.append(stmt)
    return blocks


def _replace_at_path(block: Block, path: tuple, new_stmt: Stmt) -> None:
    if len(path) == 1:
        idx = path[0]
        if isinstance(new_stmt, Block):
            # Unpack: replace single stmt with the block's contents
            block.stmts[idx : idx + 1] = new_stmt.stmts
        else:
            block.stmts[idx] = new_stmt
        return
    # Navigate into nested block
    idx = path[0]
    _, nested_idx = path[1], path[2]
    parent_stmt = block.stmts[idx]
    nested_blocks = _get_nested_blocks(parent_stmt)
    _replace_at_path(nested_blocks[nested_idx], path[3:], new_stmt)


# ---------------------------------------------------------------------------
# rename_node
# ---------------------------------------------------------------------------

def rename_node(module: Module, node_id: str, new_name: str) -> Module:
    """Rename a node and update its ID. Does not rewrite body references."""
    node = _find_node(module, node_id)
    if node is None:
        raise ValueError(f"Node not found: {node_id!r}")

    old_name = node.name
    node.name = new_name

    # Update the node ID to reflect the new name
    new_id = node_id.replace(f":{old_name}", f":{new_name}", 1)
    # Handle scoped IDs like fn:ClassName.method
    if "." in node_id:
        parts = node_id.split(".")
        parts[-1] = new_name
        new_id = ".".join(parts)
    node.id = new_id

    return module


# ---------------------------------------------------------------------------
# update_return_type
# ---------------------------------------------------------------------------

def update_return_type(module: Module, fn_id: str, type_dict: dict) -> Module:
    fn = _find_function(module, fn_id)
    fn.return_type = UnifiedType.from_dict(type_dict)
    return module


# ---------------------------------------------------------------------------
# update_param
# ---------------------------------------------------------------------------

def update_param(module: Module, fn_id: str, param_name: str, type_dict: dict) -> Module:
    fn = _find_function(module, fn_id)
    for p in fn.params:
        if p.name == param_name:
            p.type = UnifiedType.from_dict(type_dict)
            return module
    # Check if it's a local variable — give a specific redirect
    if fn.body:
        for stmt in fn.body.stmts:
            if hasattr(stmt, "name") and stmt.name == param_name:
                raise ValueError(
                    f"'{param_name}' is a local variable in {fn_id!r}, not a parameter. "
                    f"Use `str-replace {fn_id} '{{\"kind\":\"literal\",\"value\":OLD}}' "
                    f"'{{\"kind\":\"literal\",\"value\":NEW}}'` to change its value."
                )
    raise ValueError(f"Param {param_name!r} not found in {fn_id!r}")


# ---------------------------------------------------------------------------
# add_method
# ---------------------------------------------------------------------------

def add_method(module: Module, type_id: str, fn_node_dict: dict) -> Module:
    td = _find_typedef(module, type_id)
    fn = _node_from_dict(fn_node_dict)
    if not isinstance(fn, FunctionNode):
        raise ValueError("fn_node_dict must describe a function node")
    td.methods.append(fn)
    return module


# ---------------------------------------------------------------------------
# add_field
# ---------------------------------------------------------------------------

def add_field(module: Module, type_id: str, field_dict: dict) -> Module:
    td = _find_typedef(module, type_id)
    field = _field_from_dict(field_dict)
    td.fields.append(field)
    return module


# ---------------------------------------------------------------------------
# remove_node
# ---------------------------------------------------------------------------

def remove_node(module: Module, node_id: str) -> Module:
    # Try top-level first
    for i, node in enumerate(module.nodes):
        if node.id == node_id:
            module.nodes.pop(i)
            return module

    # Try inside type_defs
    for node in module.nodes:
        if isinstance(node, TypeDefNode):
            for i, m in enumerate(node.methods):
                if m.id == node_id:
                    node.methods.pop(i)
                    return module
            for i, f in enumerate(node.fields):
                if f.id == node_id:
                    node.fields.pop(i)
                    return module

    raise ValueError(f"Node not found: {node_id!r}")


# ---------------------------------------------------------------------------
# append_stmt / insert_stmt_at
# ---------------------------------------------------------------------------

def append_stmt(module: Module, fn_id: str, stmt_dict: dict) -> Module:
    fn = _find_function(module, fn_id)
    fn.body.stmts.append(stmt_from_dict(stmt_dict))
    return module


def insert_stmt_at(module: Module, fn_id: str, index: int, stmt_dict: dict) -> Module:
    """Insert a statement at position `index` in the function body (0 = prepend)."""
    fn = _find_function(module, fn_id)
    if index < 0 or index > len(fn.body.stmts):
        raise ValueError(
            f"Index {index} out of range for {fn_id!r} "
            f"(body has {len(fn.body.stmts)} stmts; valid range 0–{len(fn.body.stmts)})"
        )
    fn.body.stmts.insert(index, stmt_from_dict(stmt_dict))
    return module


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(module: Module, path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(module.to_dict(), f, indent=2, ensure_ascii=False)


def load(path: str) -> Module:
    import json
    with open(path, encoding="utf-8") as f:
        return Module.from_dict(json.load(f))
