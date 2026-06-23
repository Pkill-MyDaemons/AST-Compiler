"""Build a skeleton view of a Module — names and signatures only, no bodies."""
from __future__ import annotations
from typing import Any, Dict, List

from ..unified_ast.nodes import (
    Module, ASTNode, ImportNode, VariableNode, FieldNode, FunctionNode, TypeDefNode,
    Param, Visibility,
)
from ..unified_ast.types import UnifiedType


def build_skeleton(module: Module) -> Dict[str, Any]:
    return {
        "version": module.version,
        "source_language": module.source_language,
        "source_file": module.source_file,
        "nodes": [_skeleton_node(n) for n in module.nodes],
    }


def _skeleton_node(node: ASTNode) -> Dict[str, Any]:
    if isinstance(node, ImportNode):
        return _skel_import(node)
    if isinstance(node, VariableNode):
        return _skel_variable(node)
    if isinstance(node, FunctionNode):
        return _skel_function(node)
    if isinstance(node, TypeDefNode):
        return _skel_typedef(node)
    return {"id": "?", "kind": "unknown"}


def _skel_import(node: ImportNode) -> Dict[str, Any]:
    if node.items is None:
        summary = f"import {node.module}"
    else:
        summary = f"from {node.module} import {', '.join(node.items)}"
    if node.alias:
        summary += f" as {node.alias}"
    return {"id": node.id, "kind": "import", "summary": summary}


def _skel_variable(node: VariableNode) -> Dict[str, Any]:
    ty = node.type.render()
    kw = "const" if node.is_const else "var"
    summary = f"{kw} {node.name}: {ty}"
    if node.value:
        summary += f" = {node.value}"
    return {"id": node.id, "kind": "variable", "summary": summary}


def _skel_function(node: FunctionNode) -> Dict[str, Any]:
    sig = _render_signature(node)
    body_stmts = len(node.body.stmts) if node.body else 0
    d: Dict[str, Any] = {
        "id": node.id,
        "kind": "function",
        "signature": sig,
        "body_stmts": body_stmts,
    }
    if node.docstring:
        d["docstring"] = node.docstring[:120] + ("..." if len(node.docstring) > 120 else "")
    return d


def _skel_typedef(node: TypeDefNode) -> Dict[str, Any]:
    members: List[Dict[str, Any]] = []
    for f in node.fields:
        members.append(_skel_field(f))
    for m in node.methods:
        members.append(_skel_function(m))
    for inner in node.inner_types:
        members.append(_skel_typedef(inner))

    d: Dict[str, Any] = {
        "id": node.id,
        "kind": "type_def",
        "category": node.category.value,
        "name": node.name,
        "members": members,
    }
    if node.bases:
        d["bases"] = node.bases
    if node.interfaces:
        d["interfaces"] = node.interfaces
    if node.type_params:
        d["type_params"] = node.type_params
    if node.docstring:
        d["docstring"] = node.docstring[:120] + ("..." if len(node.docstring) > 120 else "")
    return d


def _skel_field(node: FieldNode) -> Dict[str, Any]:
    ty = node.type.render()
    summary = f"{node.name}: {ty}"
    if node.default is not None:
        summary += f" = {node.default}"
    return {"id": node.id, "kind": "field", "summary": summary}


def _render_signature(fn: FunctionNode) -> str:
    params = []
    for p in fn.params:
        if p.is_self:
            params.append(p.name)
            continue
        ty = p.type.render()
        part = f"{p.name}: {ty}" if ty != "_" else p.name
        if p.default:
            part += f" = {p.default}"
        if p.is_variadic:
            part = f"*{part}"
        params.append(part)
    ret = fn.return_type.render()
    ret_str = f" -> {ret}" if ret and ret != "_" else ""
    prefix = "async " if fn.is_async else ""
    return f"{prefix}{fn.name}({', '.join(params)}){ret_str}"
