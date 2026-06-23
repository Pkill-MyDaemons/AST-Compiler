"""
Targeted task generators for the two benchmark failures:

  - Test 8: multi-step (rename + retype in one session)
  - Test 9: prepend-guard (str-replace first stmt with [guard, original])
"""
from __future__ import annotations
import json
import random
from typing import List, Optional

from src.unified_ast.nodes import (
    Module, FunctionNode, TypeDefNode, Visibility, TypeDefCategory,
)
from src.unified_ast.types import (
    UnifiedType, TypeKind,
    T_NUMBER, T_STRING, T_BOOLEAN, T_VOID, T_ANY, T_INFERRED,
    T_OPTIONAL, T_NAMED, T_LIST,
)
from src.unified_ast.expr import (
    Block, Stmt, Expr,
    Literal, Identifier, BinaryOp, Return, Assign, VarDecl,
    If, ExprStmt, Raw,
)
from .tasks import (
    Task, Step, _all_functions, _all_types, _new_name, _alt_type,
    _type_label, _stmt_json, _has_body, _zero_for,
    change_return_type_tasks, change_param_type_tasks,
)


def _rename_id(old_id: str, new_name: str) -> str:
    """Compute the new node ID after a rename operation."""
    if "." in old_id:
        prefix = old_id.rsplit(".", 1)[0]
        return f"{prefix}.{new_name}"
    kind = old_id.split(":")[0]
    return f"{kind}:{new_name}"


# ---------------------------------------------------------------------------
# Multi-step: rename + retype in the same session
# ---------------------------------------------------------------------------

def rename_and_retype_tasks(module: Module, ast_file: str,
                             rng: random.Random) -> List[Task]:
    """
    Generate tasks that require TWO commands in one session:
      1. ast-harness rename ...
      2. ast-harness set-return-type ...
    """
    tasks = []
    fns = _all_functions(module)

    for fn in fns:
        if fn.is_constructor or fn.name.startswith("_"):
            continue
        alt = _alt_type(fn.return_type)
        if alt is None:
            continue
        new_name = _new_name(fn.name)
        old_label = _type_label(fn.return_type)
        new_label = _type_label(alt)

        tasks.append(Task(
            description=(
                f"Rename `{fn.name}` to `{new_name}` "
                f"and change its return type from `{old_label}` to `{new_label}`."
            ),
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Get the skeleton to confirm `{fn.id}` and its signature.",
                ),
                Step(
                    command=f"ast-harness rename {ast_file} {fn.id} {new_name}",
                    rationale=f"Rename `{fn.name}` → `{new_name}`.",
                ),
                Step(
                    command=f"ast-harness set-return-type {ast_file} {_rename_id(fn.id, new_name)} '{json.dumps(alt.to_dict())}'",
                    rationale=f"Change return type to `{new_label}` using the new ID.",
                ),
            ],
            category="multi_step",
            difficulty="hard",
            tags=["rename", "retype", "multi_step"],
        ))

    return tasks


# ---------------------------------------------------------------------------
# Prepend-guard: str-replace first stmt → [if-guard, original-first-stmt]
# ---------------------------------------------------------------------------

def prepend_guard_tasks(module: Module, ast_file: str,
                         rng: random.Random) -> List[Task]:
    """
    Generate tasks where the model prepends an early-return guard before the
    first statement of a function.

    The model must:
      1. skeleton
      2. get (to read current first stmt)
      3. str-replace first-stmt → block([guard, first-stmt])
    """
    tasks = []
    fns = _all_functions(module)

    for fn in fns:
        if not _has_body(fn):
            continue
        if fn.return_type.kind == TypeKind.VOID:
            continue  # only makes sense for functions that return something

        non_self = [p for p in fn.params if not p.is_self]
        if not non_self:
            continue

        param = non_self[0]
        first_stmt = fn.body.stmts[0]

        # Guard: if param == zero/None → return zero
        zero_val = _zero_for(fn.return_type)
        if zero_val is None:
            continue

        param_zero = _zero_for(param.type) or Literal(value=None, lit_kind="none")

        guard = If(
            cond=BinaryOp(
                left=Identifier(name=param.name),
                op="==",
                right=param_zero,
            ),
            then_block=Block(stmts=[Return(value=zero_val)]),
        )

        # Build the replacement: a block containing [guard, first_stmt]
        new_block = Block(stmts=[guard, first_stmt])
        old_json = json.dumps(first_stmt.to_dict())
        new_json = json.dumps(new_block.to_dict())

        tasks.append(Task(
            description=(
                f"In `{fn.name}`, add a guard at the very start: "
                f"if `{param.name}` is zero or null, return immediately. "
                f"Prepend this check before the first statement."
            ),
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Locate `{fn.id}` in the skeleton.",
                ),
                Step(
                    command=f"ast-harness get {ast_file} {fn.id}",
                    rationale="Read the existing body to find the exact first statement.",
                ),
                Step(
                    command=(
                        f"ast-harness str-replace {ast_file} {fn.id} "
                        f"'{old_json}' '{new_json}'"
                    ),
                    rationale=(
                        "Replace the first statement with a block that contains "
                        "[guard, original-first-statement]."
                    ),
                ),
            ],
            category="body_edit",
            difficulty="hard",
            tags=["body_edit", "add_stmt", "guard", "prepend", "str_replace"],
        ))

    return tasks


# ---------------------------------------------------------------------------
# Rename + remove: rename a method then remove a different one
# ---------------------------------------------------------------------------

def rename_then_remove_tasks(module: Module, ast_file: str,
                              rng: random.Random) -> List[Task]:
    """Two structural changes in sequence."""
    tasks = []
    for td in _all_types(module):
        candidates = [m for m in td.methods if not m.is_constructor and not m.name.startswith("_")]
        if len(candidates) < 2:
            continue
        rng.shuffle(candidates)
        rename_target = candidates[0]
        remove_target = candidates[1]
        new_name = _new_name(rename_target.name)

        tasks.append(Task(
            description=(
                f"In `{td.name}`, rename `{rename_target.name}` to `{new_name}`, "
                f"then remove the method `{remove_target.name}`."
            ),
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Confirm both method IDs in `{td.name}`.",
                ),
                Step(
                    command=f"ast-harness rename {ast_file} {rename_target.id} {new_name}",
                    rationale=f"Rename `{rename_target.name}` → `{new_name}`.",
                ),
                Step(
                    command=f"ast-harness remove {ast_file} {remove_target.id}",
                    rationale=f"Remove `{remove_target.name}`.",
                ),
            ],
            category="multi_step",
            difficulty="hard",
            tags=["rename", "remove", "multi_step"],
        ))
    return tasks


def _find_all_literals_in_dict(d) -> list:
    """Recursively collect all literal dicts from an AST dict."""
    results = []
    if isinstance(d, dict):
        if d.get("kind") == "literal" and d.get("value") is not None:
            results.append(d)
        for v in d.values():
            results.extend(_find_all_literals_in_dict(v))
    elif isinstance(d, list):
        for item in d:
            results.extend(_find_all_literals_in_dict(item))
    return results


def _find_all_binary_ops_in_dict(d) -> list:
    """Recursively collect all binary_op dicts from an AST dict."""
    results = []
    if isinstance(d, dict):
        if d.get("kind") == "binary_op":
            results.append(d)
        for v in d.values():
            results.extend(_find_all_binary_ops_in_dict(v))
    elif isinstance(d, list):
        for item in d:
            results.extend(_find_all_binary_ops_in_dict(item))
    return results


def change_literal_expr_tasks(module: Module, ast_file: str,
                               rng: random.Random) -> List[Task]:
    """
    Train the model to use expression-level str-replace for literal changes.
    Uses minimal pattern: {"kind":"literal","value":OLD} → {"kind":"literal","value":NEW}
    Works even when the literal is nested inside control flow.
    """
    import random as _random
    tasks = []
    fns = _all_functions(module)

    mutate_map = {
        "int":   lambda v: v + _random.choice([-1, 1, 5, -5, 10]),
        "float": lambda v: round(v * 2.0, 2) if v != 0.0 else 1.0,
        "bool":  lambda v: not v,
    }

    for fn in fns:
        if not _has_body(fn):
            continue
        body_dict = fn.body.to_dict()
        lits = _find_all_literals_in_dict(body_dict)
        seen_values = set()
        for lit_d in lits[:3]:
            lk = lit_d.get("lit_kind", "int")
            v = lit_d["value"]
            if lk not in mutate_map or v in seen_values:
                continue
            seen_values.add(v)
            new_v = mutate_map[lk](v)
            old_pattern = json.dumps({"kind": "literal", "value": v, "lit_kind": lk})
            new_expr    = json.dumps({"kind": "literal", "value": new_v, "lit_kind": lk})
            tasks.append(Task(
                description=(
                    f"In `{fn.name}`, change the literal value `{v}` to `{new_v}`."
                ),
                steps=[
                    Step(
                        command=f"ast-harness skeleton {ast_file}",
                        rationale=f"Locate `{fn.id}` in the skeleton.",
                    ),
                    Step(
                        command=f"ast-harness get {ast_file} {fn.id}",
                        rationale="Read the body to find the exact literal.",
                    ),
                    Step(
                        command=(
                            f"ast-harness str-replace {ast_file} {fn.id} "
                            f"'{old_pattern}' '{new_expr}'"
                        ),
                        rationale=(
                            f"Replace the literal `{v}` with `{new_v}` using "
                            f"an expression-level pattern."
                        ),
                    ),
                ],
                category="body_edit",
                difficulty="medium",
                tags=["literal", "str_replace", "expr_level"],
            ))
    return tasks


def change_operator_expr_tasks(module: Module, ast_file: str,
                                rng: random.Random) -> List[Task]:
    """
    Train the model to use expression-level str-replace for operator changes.
    Uses minimal pattern: {"kind":"binary_op","op":"X"} → {"kind":"binary_op","op":"Y"}
    Finds operators anywhere in the body — including inside loops/conditionals.
    """
    alt_ops = {
        "+": "-", "-": "+", "*": "/", "/": "*",
        "==": "!=", "!=": "==", "<": "<=", "<=": "<", ">": ">=", ">=": ">",
        "&&": "||", "||": "&&", "and": "or", "or": "and",
    }
    tasks = []
    fns = _all_functions(module)

    for fn in fns:
        if not _has_body(fn):
            continue
        body_dict = fn.body.to_dict()
        ops = _find_all_binary_ops_in_dict(body_dict)
        seen_ops = set()
        for op_d in ops[:3]:
            op = op_d.get("op", "")
            alt = alt_ops.get(op)
            if alt is None or op in seen_ops:
                continue
            seen_ops.add(op)
            old_pattern = json.dumps({"kind": "binary_op", "op": op})
            new_expr    = json.dumps({"kind": "binary_op", "op": alt})
            tasks.append(Task(
                description=(
                    f"In `{fn.name}`, change the `{op}` operator to `{alt}`."
                ),
                steps=[
                    Step(
                        command=f"ast-harness skeleton {ast_file}",
                        rationale=f"Locate `{fn.id}` in the skeleton.",
                    ),
                    Step(
                        command=f"ast-harness get {ast_file} {fn.id}",
                        rationale="Read the body to see the binary operation.",
                    ),
                    Step(
                        command=(
                            f"ast-harness str-replace {ast_file} {fn.id} "
                            f"'{old_pattern}' '{new_expr}'"
                        ),
                        rationale=(
                            f"Replace `{op}` with `{alt}` using a minimal "
                            f"expression-level pattern — the harness finds it "
                            f"anywhere in the body, including inside loops."
                        ),
                    ),
                ],
                category="body_edit",
                difficulty="medium",
                tags=["operator", "str_replace", "expr_level"],
            ))
    return tasks


def insert_guard_tasks(module: Module, ast_file: str,
                       rng: random.Random) -> List[Task]:
    """
    Teach the model to use `insert-before` to prepend a guard.
    Simpler and more direct than the str-replace block-wrapping approach.
    """
    tasks = []
    fns = _all_functions(module)

    for fn in fns:
        if not _has_body(fn):
            continue
        if fn.return_type.kind == TypeKind.VOID:
            continue
        non_self = [p for p in fn.params if not p.is_self]
        if not non_self:
            continue
        param = non_self[0]
        zero_val = _zero_for(fn.return_type)
        if zero_val is None:
            continue
        param_zero = _zero_for(param.type) or Literal(value=None, lit_kind="none")

        guard = If(
            cond=BinaryOp(
                left=Identifier(name=param.name),
                op="==",
                right=param_zero,
            ),
            then_block=Block(stmts=[Return(value=zero_val)]),
        )
        guard_json = json.dumps(guard.to_dict())

        tasks.append(Task(
            description=(
                f"In `{fn.name}`, prepend a guard at position 0: "
                f"if `{param.name}` equals zero/null, return immediately."
            ),
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Locate `{fn.id}` in the skeleton.",
                ),
                Step(
                    command=f"ast-harness get {ast_file} {fn.id}",
                    rationale="Read the body to confirm position 0 is the right insertion point.",
                ),
                Step(
                    command=(
                        f"ast-harness insert-before {ast_file} {fn.id} 0 '{guard_json}'"
                    ),
                    rationale=(
                        "Insert the guard at index 0 (prepend before all existing statements)."
                    ),
                ),
            ],
            category="body_edit",
            difficulty="medium",
            tags=["guard", "insert_before", "prepend"],
        ))
    return tasks


TARGETED_GENERATORS = [
    rename_and_retype_tasks,
    prepend_guard_tasks,
    rename_then_remove_tasks,
    # Single-step retype examples — balance against multi-step to prevent overfitting
    change_return_type_tasks,
    change_param_type_tasks,
    # Expression-level body edits — teach minimal pattern str-replace
    change_literal_expr_tasks,
    change_operator_expr_tasks,
    # insert-before command
    insert_guard_tasks,
]
