"""
Execute harness commands in-process and return their output as strings.

This is the same logic as the CLI but without subprocess overhead, so
we can generate thousands of training examples quickly.
"""
from __future__ import annotations
import json
import copy
from typing import Tuple

from src.unified_ast.nodes import Module
from src.harness.skeleton import build_skeleton
from src.harness.editor import (
    get_node, str_replace_body, rename_node,
    update_return_type, update_param,
    add_method, add_field, remove_node, append_stmt, insert_stmt_at,
)
from src.generators import generate as gen_code


def run_command(command: str, module: Module) -> Tuple[str, Module]:
    """
    Parse and execute a harness/compiler command string.
    Returns (output_text, updated_module).
    The module is mutated in-place for editing commands.
    """
    parts = _split_command(command)
    if not parts:
        return "# (no command)", module

    tool = parts[0]
    if tool == "ast-harness":
        return _run_harness(parts[1:], module)
    if tool == "ast-compiler":
        return _run_compiler(parts[1:], module)
    return f"# unknown tool: {tool}", module


def _run_harness(args: list, module: Module) -> Tuple[str, Module]:
    if not args:
        return "# missing subcommand", module

    sub = args[0]
    # args[1] is the ast_file path (e.g. "code.json") — we ignore it since
    # the module is already in-process; everything after it is the real args.
    rest = args[2:]

    _J = lambda d: json.dumps(d, separators=(",", ":"), ensure_ascii=False)

    if sub == "skeleton":
        skel = build_skeleton(module)
        return _J(skel), module

    if sub == "get":
        if not rest:
            return "# get requires node_id", module
        node_id = rest[0]
        try:
            node = get_node(module, node_id)
            return _J(node), module
        except ValueError as e:
            return f"Error: {e}", module

    if sub == "rename":
        if len(rest) < 2:
            return "# rename requires node_id new_name", module
        node_id, new_name = rest[0], rest[1]
        try:
            rename_node(module, node_id, new_name)
            return f"Renamed {node_id} → {new_name}", module
        except ValueError as e:
            return f"Error: {e}", module

    if sub == "set-return-type":
        if len(rest) < 2:
            return "# set-return-type requires fn_id type_json", module
        fn_id = rest[0]
        type_json = _rejoin(rest[1:])
        try:
            type_dict = json.loads(type_json)
            update_return_type(module, fn_id, type_dict)
            return f"Updated return type of {fn_id}", module
        except (ValueError, json.JSONDecodeError) as e:
            return f"Error: {e}", module

    if sub == "set-param-type":
        if len(rest) < 3:
            return "# set-param-type requires fn_id param_name type_json", module
        fn_id, param_name = rest[0], rest[1]
        type_json = _rejoin(rest[2:])
        try:
            type_dict = json.loads(type_json)
            update_param(module, fn_id, param_name, type_dict)
            return f"Updated type of {fn_id} param {param_name!r}", module
        except (ValueError, json.JSONDecodeError) as e:
            return f"Error: {e}", module

    if sub == "str-replace":
        if len(rest) < 3:
            return "# str-replace requires fn_id old_stmt new_stmt", module
        fn_id = rest[0]
        old_json = _rejoin_quoted(rest[1])
        new_json = _rejoin_quoted(rest[2] if len(rest) > 2 else "")
        try:
            str_replace_body(module, fn_id, old_json, new_json)
            return f"Replaced statement in {fn_id}", module
        except ValueError as e:
            return f"Error: {e}", module

    if sub == "append-stmt":
        if len(rest) < 2:
            return "# append-stmt requires fn_id stmt_json", module
        fn_id = rest[0]
        stmt_json = _rejoin(rest[1:])
        try:
            stmt_dict = json.loads(stmt_json)
            append_stmt(module, fn_id, stmt_dict)
            return f"Appended statement to {fn_id}", module
        except (ValueError, json.JSONDecodeError) as e:
            return f"Error: {e}", module

    if sub == "insert-before":
        if len(rest) < 3:
            return "# insert-before requires fn_id index stmt_json", module
        fn_id = rest[0]
        try:
            index = int(rest[1])
        except ValueError:
            return "# insert-before: index must be an integer", module
        stmt_json = _rejoin(rest[2:])
        try:
            stmt_dict = json.loads(stmt_json)
            insert_stmt_at(module, fn_id, index, stmt_dict)
            return f"Inserted statement at index {index} in {fn_id}", module
        except (ValueError, json.JSONDecodeError) as e:
            return f"Error: {e}", module

    if sub == "add-method":
        if len(rest) < 2:
            return "# add-method requires type_id fn_json", module
        type_id = rest[0]
        fn_json = _rejoin(rest[1:])
        try:
            fn_dict = json.loads(fn_json)
            add_method(module, type_id, fn_dict)
            return f"Added method to {type_id}", module
        except (ValueError, json.JSONDecodeError) as e:
            return f"Error: {e}", module

    if sub == "remove":
        if not rest:
            return "# remove requires node_id", module
        node_id = rest[0]
        try:
            remove_node(module, node_id)
            return f"Removed {node_id}", module
        except ValueError as e:
            return f"Error: {e}", module

    return f"# unknown harness subcommand: {sub}", module


def _run_compiler(args: list, module: Module) -> Tuple[str, Module]:
    if not args:
        return "# missing subcommand", module

    sub = args[0]
    # args[1] is the ast_file path — skip it for in-process execution
    if sub == "compile":
        lang = None
        for i, a in enumerate(args):
            if a == "--lang" and i + 1 < len(args):
                lang = args[i + 1]
                break
        if not lang:
            return "# compile requires --lang", module
        try:
            source = gen_code(module, lang)
            # Truncate long outputs in training data
            if len(source) > 3000:
                lines = source.splitlines()[:60]
                source = "\n".join(lines) + "\n# ... (truncated)"
            return source, module
        except Exception as e:
            return f"Error: {e}", module

    return f"# unknown compiler subcommand: {sub}", module


def _split_command(cmd: str) -> list:
    """Naive shell-like split that handles single-quoted JSON args."""
    parts = []
    current = []
    in_single = False
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if c == "'" and not in_single:
            in_single = True
            i += 1
            continue
        if c == "'" and in_single:
            in_single = False
            i += 1
            continue
        if c == " " and not in_single:
            if current:
                parts.append("".join(current))
                current = []
            i += 1
            continue
        current.append(c)
        i += 1
    if current:
        parts.append("".join(current))
    return parts


def _rejoin(parts: list) -> str:
    return " ".join(parts).strip("'\"")


def _rejoin_quoted(s: str) -> str:
    return s.strip("'\"")
