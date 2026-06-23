"""
Task generators — each returns a list of Task objects for a given Module.

A Task has:
  - description: natural-language instruction for the user turn
  - steps: list of (command_str, description) pairs the model should execute
  - verify: callable that checks the edited module is correct
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from src.unified_ast.nodes import (
    Module, FunctionNode, TypeDefNode, VariableNode, FieldNode, Param,
    Visibility, TypeDefCategory,
)
from src.unified_ast.types import (
    UnifiedType, TypeKind,
    T_NUMBER, T_STRING, T_BOOLEAN, T_VOID, T_ANY, T_INFERRED,
    T_LIST, T_OPTIONAL, T_NAMED,
)
from src.unified_ast.expr import (
    Block, Stmt, Expr,
    Literal, Identifier, BinaryOp, UnaryOp, Return, Assign, VarDecl,
    If, WhileLoop, ForEach, ExprStmt, Raw, RawExpr, Call, FieldAccess,
    Break, Continue,
)


@dataclass
class Step:
    command: str          # exact CLI command string
    rationale: str        # one-line explanation for the assistant turn


@dataclass
class Task:
    description: str      # user-facing instruction
    steps: List[Step]     # ordered harness commands to apply
    category: str         # rename | retype | body_edit | add | remove | structural
    difficulty: str = "easy"   # easy | medium | hard
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_functions(module: Module) -> List[FunctionNode]:
    fns = []
    for n in module.nodes:
        if isinstance(n, FunctionNode):
            fns.append(n)
        elif isinstance(n, TypeDefNode):
            fns.extend(n.methods)
    return fns


def _all_types(module: Module) -> List[TypeDefNode]:
    return [n for n in module.nodes if isinstance(n, TypeDefNode)]


def _has_body(fn: FunctionNode) -> bool:
    return bool(fn.body and fn.body.stmts)


def _new_name(old: str) -> str:
    """Produce a plausible renamed version of a name."""
    synonyms = {
        "get": "fetch", "fetch": "retrieve", "retrieve": "get",
        "add": "insert", "insert": "put", "put": "add",
        "remove": "delete", "delete": "erase", "erase": "clear",
        "calc": "compute", "compute": "calculate", "calculate": "calc",
        "run": "execute", "execute": "invoke", "invoke": "run",
        "check": "validate", "validate": "verify", "verify": "check",
        "find": "search", "search": "lookup", "lookup": "find",
    }
    for prefix, replacement in synonyms.items():
        if old.lower().startswith(prefix):
            return replacement + old[len(prefix):]
    if old.endswith("_v2"):
        return old[:-3]
    return old + "_v2"


def _alt_type(ty: UnifiedType) -> Optional[UnifiedType]:
    """Return a plausible alternative type for mutation tasks."""
    k = ty.kind
    if k == TypeKind.NUMBER:
        if ty.float:
            return T_NUMBER(64, float=False)  # f64 → i64
        bits = ty.bits or 64
        new_bits = 32 if bits == 64 else 64
        return T_NUMBER(new_bits, signed=ty.signed if ty.signed is not None else True)
    if k == TypeKind.STRING:
        return T_OPTIONAL(T_STRING)
    if k == TypeKind.BOOLEAN:
        return T_OPTIONAL(T_BOOLEAN)
    if k == TypeKind.LIST:
        return T_OPTIONAL(ty)
    if k == TypeKind.INFERRED or k == TypeKind.VOID or k == TypeKind.ANY:
        return None
    return None


def _type_label(ty: UnifiedType) -> str:
    return ty.render()


def _find_literals(block: Block) -> List[Tuple[int, Literal]]:
    """Return (stmt_index, literal) pairs for top-level statements that contain literals."""
    results = []
    for i, stmt in enumerate(block.stmts):
        lit = _extract_literal_from_stmt(stmt)
        if lit is not None:
            results.append((i, lit))
    return results


def _extract_literal_from_stmt(stmt: Stmt) -> Optional[Literal]:
    if isinstance(stmt, Return) and isinstance(stmt.value, Literal):
        return stmt.value
    if isinstance(stmt, Assign) and isinstance(stmt.value, Literal):
        return stmt.value
    if isinstance(stmt, VarDecl) and isinstance(stmt.value, Literal):
        return stmt.value
    return None


def _find_binary_ops(block: Block) -> List[Tuple[int, BinaryOp]]:
    """Return (stmt_index, binop) pairs where the stmt's top-level value is a BinaryOp."""
    results = []
    for i, stmt in enumerate(block.stmts):
        bo = _extract_binop_from_stmt(stmt)
        if bo:
            results.append((i, bo))
    return results


def _extract_binop_from_stmt(stmt: Stmt) -> Optional[BinaryOp]:
    if isinstance(stmt, Return) and isinstance(stmt.value, BinaryOp):
        return stmt.value
    if isinstance(stmt, Assign) and isinstance(stmt.value, BinaryOp):
        return stmt.value
    if isinstance(stmt, VarDecl) and isinstance(stmt.value, BinaryOp):
        return stmt.value
    return None


def _stmt_json(stmt: Stmt) -> str:
    return json.dumps(stmt.to_dict())


_ALT_OPS = {
    "+": "-", "-": "+", "*": "/", "/": "*",
    "==": "!=", "!=": "==", "<": "<=", "<=": "<", ">": ">=", ">=": ">",
    "&&": "||", "||": "&&",
}


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def rename_function_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    tasks = []
    for fn in _all_functions(module):
        if fn.name.startswith("_") or fn.is_constructor:
            continue
        new_name = _new_name(fn.name)
        tasks.append(Task(
            description=f"Rename the function `{fn.name}` to `{new_name}`.",
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Get the skeleton to confirm `{fn.id}` exists.",
                ),
                Step(
                    command=f"ast-harness rename {ast_file} {fn.id} {new_name}",
                    rationale=f"Rename `{fn.name}` → `{new_name}`.",
                ),
            ],
            category="rename",
            difficulty="easy",
            tags=["rename", "function"],
        ))
    return tasks


def rename_type_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    tasks = []
    for td in _all_types(module):
        new_name = _new_name(td.name)
        tasks.append(Task(
            description=f"Rename the {td.category.value} `{td.name}` to `{new_name}`.",
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Confirm `{td.id}` in the skeleton.",
                ),
                Step(
                    command=f"ast-harness rename {ast_file} {td.id} {new_name}",
                    rationale=f"Rename `{td.name}` → `{new_name}`.",
                ),
            ],
            category="rename",
            difficulty="easy",
            tags=["rename", "type"],
        ))
    return tasks


def change_return_type_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    tasks = []
    for fn in _all_functions(module):
        alt = _alt_type(fn.return_type)
        if alt is None:
            continue
        old_label = _type_label(fn.return_type)
        new_label = _type_label(alt)
        tasks.append(Task(
            description=f"Change the return type of `{fn.name}` from `{old_label}` to `{new_label}`.",
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Confirm `{fn.id}` signature.",
                ),
                Step(
                    command=f"ast-harness set-return-type {ast_file} {fn.id} '{json.dumps(alt.to_dict())}'",
                    rationale=f"Update return type to `{new_label}`.",
                ),
            ],
            category="retype",
            difficulty="easy",
            tags=["return_type"],
        ))
    return tasks


def change_param_type_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    tasks = []
    for fn in _all_functions(module):
        for param in fn.params:
            if param.is_self or param.name in ("self", "cls"):
                continue
            alt = _alt_type(param.type)
            if alt is None:
                continue
            old_label = _type_label(param.type)
            new_label = _type_label(alt)
            tasks.append(Task(
                description=(
                    f"Change the type of parameter `{param.name}` in `{fn.name}` "
                    f"from `{old_label}` to `{new_label}`."
                ),
                steps=[
                    Step(
                        command=f"ast-harness skeleton {ast_file}",
                        rationale=f"Find `{fn.id}` and confirm param `{param.name}` type.",
                    ),
                    Step(
                        command=(
                            f"ast-harness set-param-type {ast_file} {fn.id} "
                            f"{param.name} '{json.dumps(alt.to_dict())}'"
                        ),
                        rationale=f"Change `{param.name}` to `{new_label}`.",
                    ),
                ],
                category="retype",
                difficulty="easy",
                tags=["param_type"],
            ))
    return tasks


def change_literal_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    tasks = []
    for fn in _all_functions(module):
        if not _has_body(fn):
            continue
        literals = _find_literals(fn.body)
        for stmt_idx, lit in literals[:2]:  # at most 2 per function
            old_stmt = fn.body.stmts[stmt_idx]
            new_lit, description = _mutate_literal(lit, rng)
            if new_lit is None:
                continue
            # Build new stmt with mutated literal
            new_stmt = _replace_literal_in_stmt(old_stmt, new_lit)
            if new_stmt is None:
                continue
            tasks.append(Task(
                description=f"In function `{fn.name}`, {description}",
                steps=[
                    Step(
                        command=f"ast-harness skeleton {ast_file}",
                        rationale=f"Find `{fn.id}`.",
                    ),
                    Step(
                        command=f"ast-harness get {ast_file} {fn.id}",
                        rationale="Read the body to find the exact statement.",
                    ),
                    Step(
                        command=(
                            f"ast-harness str-replace {ast_file} {fn.id} "
                            f"'{_stmt_json(old_stmt)}' '{_stmt_json(new_stmt)}'"
                        ),
                        rationale=f"Replace the literal.",
                    ),
                ],
                category="body_edit",
                difficulty="medium",
                tags=["literal", "str_replace"],
            ))
    return tasks


def change_operator_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    tasks = []
    for fn in _all_functions(module):
        if not _has_body(fn):
            continue
        ops = _find_binary_ops(fn.body)
        for stmt_idx, bo in ops[:2]:
            alt_op = _ALT_OPS.get(bo.op)
            if alt_op is None:
                continue
            old_stmt = fn.body.stmts[stmt_idx]
            new_bo = BinaryOp(left=bo.left, op=alt_op, right=bo.right)
            new_stmt = _replace_expr_in_stmt(old_stmt, bo, new_bo)
            if new_stmt is None:
                continue
            tasks.append(Task(
                description=(
                    f"In function `{fn.name}`, change the `{bo.op}` operator to `{alt_op}`."
                ),
                steps=[
                    Step(
                        command=f"ast-harness skeleton {ast_file}",
                        rationale=f"Locate `{fn.id}`.",
                    ),
                    Step(
                        command=f"ast-harness get {ast_file} {fn.id}",
                        rationale="Read the body to find the binary operation.",
                    ),
                    Step(
                        command=(
                            f"ast-harness str-replace {ast_file} {fn.id} "
                            f"'{_stmt_json(old_stmt)}' '{_stmt_json(new_stmt)}'"
                        ),
                        rationale=f"Replace `{bo.op}` with `{alt_op}`.",
                    ),
                ],
                category="body_edit",
                difficulty="medium",
                tags=["operator", "str_replace"],
            ))
    return tasks


def add_return_none_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    """Add `return None` at the end of void functions that don't have an explicit return."""
    tasks = []
    for fn in _all_functions(module):
        if fn.return_type.kind != TypeKind.VOID:
            continue
        # Skip if already ends with a return
        if fn.body.stmts and isinstance(fn.body.stmts[-1], Return):
            continue
        stmt = Return(value=Literal(value=None, lit_kind="none"))
        tasks.append(Task(
            description=f"Add an explicit `return None` at the end of `{fn.name}`.",
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Confirm `{fn.id}` exists.",
                ),
                Step(
                    command=f"ast-harness append-stmt {ast_file} {fn.id} '{_stmt_json(stmt)}'",
                    rationale="Append the return statement.",
                ),
            ],
            category="add",
            difficulty="easy",
            tags=["append", "return"],
        ))
    return tasks


def add_early_return_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    """For functions with a non-void return and non-empty body, add a guard return at the top."""
    tasks = []
    for fn in _all_functions(module):
        if fn.return_type.kind in (TypeKind.VOID, TypeKind.INFERRED):
            continue
        if not _has_body(fn):
            continue
        # Build: if (some param == 0/null/false) return <zero>
        non_self_params = [p for p in fn.params if not p.is_self]
        if not non_self_params:
            continue
        p = non_self_params[0]
        zero = _zero_for(fn.return_type)
        if zero is None:
            continue
        guard_stmt = If(
            cond=BinaryOp(
                left=Identifier(name=p.name),
                op="==",
                right=_zero_for(p.type) or Literal(value=None, lit_kind="none"),
            ),
            then_block=Block(stmts=[Return(value=zero)]),
        )
        tasks.append(Task(
            description=(
                f"Add a guard at the start of `{fn.name}` that returns "
                f"immediately if `{p.name}` is zero/null."
            ),
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Confirm `{fn.id}` and its signature.",
                ),
                Step(
                    command=f"ast-harness get {ast_file} {fn.id}",
                    rationale="Read the existing body.",
                ),
                Step(
                    command=(
                        f"ast-harness str-replace {ast_file} {fn.id} "
                        f"'{_stmt_json(fn.body.stmts[0])}' "
                        + f"'{json.dumps({'kind': 'block', 'stmts': [guard_stmt.to_dict(), fn.body.stmts[0].to_dict()]})}'"
                    ),
                    rationale="Prepend the guard before the first statement.",
                ),
            ],
            category="add",
            difficulty="medium",
            tags=["guard", "early_return", "str_replace"],
        ))
    return tasks


def remove_method_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    tasks = []
    for td in _all_types(module):
        # Only remove non-constructor methods
        candidates = [m for m in td.methods if not m.is_constructor]
        if len(candidates) < 2:
            continue  # don't remove the only method
        m = rng.choice(candidates)
        tasks.append(Task(
            description=f"Remove the method `{m.name}` from `{td.name}`.",
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Confirm `{m.id}` is a method of `{td.name}`.",
                ),
                Step(
                    command=f"ast-harness remove {ast_file} {m.id}",
                    rationale=f"Delete `{m.name}` from the type.",
                ),
            ],
            category="remove",
            difficulty="easy",
            tags=["remove", "method"],
        ))
    return tasks


def cross_compile_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    """Generate a task to compile the AST to a different language."""
    src_lang = module.source_language
    targets = [l for l in ("python", "rust", "typescript") if l != src_lang]
    tasks = []
    for target in targets:
        tasks.append(Task(
            description=f"Compile this {src_lang} codebase to {target}.",
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale="Review the module structure before compiling.",
                ),
                Step(
                    command=f"ast-compiler compile {ast_file} --lang {target}",
                    rationale=f"Emit {target} source from the unified AST.",
                ),
            ],
            category="structural",
            difficulty="easy",
            tags=["cross_compile", target],
        ))
    return tasks


def error_recovery_tasks(module: Module, ast_file: str, rng: random.Random) -> List[Task]:
    """
    Generate tasks whose FIRST step uses a wrong node ID, then corrects it.
    Teaches the model to read skeleton output and fix bad assumptions.
    """
    tasks = []
    fns = _all_functions(module)
    # Only top-level functions (not methods) — these are what the model most often
    # mis-addresses as fn:ClassName.name instead of fn:name
    top_level = [fn for fn in fns
                 if not any(fn.id.startswith(f"fn:{td.name}.") for td in _all_types(module))]
    for fn in top_level[:3]:
        # Simulate what model does wrong: prepend a class name that doesn't exist
        wrong_id = f"fn:Module.{fn.name}"
        new_name = _new_name(fn.name)
        tasks.append(Task(
            description=f"Rename the function `{fn.name}` to `{new_name}`.",
            steps=[
                Step(
                    command=f"ast-harness skeleton {ast_file}",
                    rationale=f"Get the skeleton to look up the correct ID for `{fn.name}`.",
                ),
                # Intentionally correct step — teaches model to use the right ID
                Step(
                    command=f"ast-harness rename {ast_file} {fn.id} {new_name}",
                    rationale=f"Use the exact ID `{fn.id}` from the skeleton.",
                ),
            ],
            category="rename",
            difficulty="easy",
            tags=["rename", "top_level_fn", "id_lookup"],
        ))
    return tasks


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def _mutate_literal(lit: Literal, rng: random.Random) -> Tuple[Optional[Literal], str]:
    if lit.lit_kind == "int" and isinstance(lit.value, int):
        new_val = lit.value + rng.choice([-1, 1, 2, -2, 10, -10])
        return Literal(value=new_val, lit_kind="int"), f"change the literal `{lit.value}` to `{new_val}`."
    if lit.lit_kind == "float" and isinstance(lit.value, float):
        new_val = round(lit.value * rng.choice([2.0, 0.5, -1.0]) + rng.uniform(-1, 1), 2)
        return Literal(value=new_val, lit_kind="float"), f"change `{lit.value}` to `{new_val}`."
    if lit.lit_kind == "string" and isinstance(lit.value, str) and lit.value:
        new_val = lit.value.upper() if lit.value == lit.value.lower() else lit.value.lower()
        return Literal(value=new_val, lit_kind="string"), f'change the string `"{lit.value}"` to `"{new_val}"`.'
    if lit.lit_kind == "bool":
        new_val = not lit.value
        return Literal(value=new_val, lit_kind="bool"), f"change `{lit.value}` to `{new_val}`."
    return None, ""


def _replace_literal_in_stmt(stmt: Stmt, new_lit: Literal) -> Optional[Stmt]:
    if isinstance(stmt, Return) and isinstance(stmt.value, Literal):
        return Return(value=new_lit)
    if isinstance(stmt, Assign) and isinstance(stmt.value, Literal):
        return Assign(target=stmt.target, op=stmt.op, value=new_lit)
    if isinstance(stmt, VarDecl) and isinstance(stmt.value, Literal):
        return VarDecl(name=stmt.name, type=stmt.type, value=new_lit, is_mutable=stmt.is_mutable)
    return None


def _replace_expr_in_stmt(stmt: Stmt, old_expr: Expr, new_expr: Expr) -> Optional[Stmt]:
    if isinstance(stmt, Return) and stmt.value is old_expr:
        return Return(value=new_expr)
    if isinstance(stmt, Assign) and stmt.value is old_expr:
        return Assign(target=stmt.target, op=stmt.op, value=new_expr)
    if isinstance(stmt, VarDecl) and stmt.value is old_expr:
        return VarDecl(name=stmt.name, type=stmt.type, value=new_expr, is_mutable=stmt.is_mutable)
    if isinstance(stmt, ExprStmt) and stmt.expr is old_expr:
        return ExprStmt(expr=new_expr)
    return None


def _zero_for(ty: UnifiedType) -> Optional[Literal]:
    k = ty.kind
    if k == TypeKind.NUMBER:
        return Literal(value=0, lit_kind="int")
    if k == TypeKind.STRING:
        return Literal(value="", lit_kind="string")
    if k == TypeKind.BOOLEAN:
        return Literal(value=False, lit_kind="bool")
    if k in (TypeKind.OPTIONAL, TypeKind.NAMED, TypeKind.ANY):
        return Literal(value=None, lit_kind="none")
    return None


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

ALL_GENERATORS = [
    rename_function_tasks,
    rename_type_tasks,
    change_return_type_tasks,
    change_param_type_tasks,
    change_literal_tasks,
    change_operator_tasks,
    add_return_none_tasks,
    add_early_return_tasks,
    remove_method_tasks,
    cross_compile_tasks,
    error_recovery_tasks,
]


def generate_all_tasks(
    module: Module,
    ast_file: str,
    rng: Optional[random.Random] = None,
    max_per_generator: int = 3,
) -> List[Task]:
    if rng is None:
        rng = random.Random(42)
    tasks = []
    for gen in ALL_GENERATORS:
        batch = gen(module, ast_file, rng)
        # Shuffle and cap to avoid bloat from very large files
        rng.shuffle(batch)
        tasks.extend(batch[:max_per_generator])
    return tasks
