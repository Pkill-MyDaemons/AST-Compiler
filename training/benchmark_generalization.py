"""
Generalization benchmark — tests the v4 adapter on bookstore.py,
a file the model has NEVER seen during training.

Same 10 task categories as benchmark.py, different domain.
"""
from __future__ import annotations
import copy
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.parsers import parse
from src.generators import generate as gen_code
from src.unified_ast.nodes import Module, FunctionNode, TypeDefNode
from src.unified_ast.types import TypeKind
from training.dataset import compact_skeleton
from training.executor import run_command
from training.system_prompt import SYSTEM_PROMPT

MAX_TURNS = 8
MODEL_PATH = "models/ast-editor-mlx-v6"
BASE_MODEL  = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
SOURCE_FILE = "examples/bookstore.py"


# ---------------------------------------------------------------------------
# Helpers (identical to benchmark.py)
# ---------------------------------------------------------------------------

def get_fn(module: Module, fn_id: str) -> Optional[FunctionNode]:
    for node in module.nodes:
        if isinstance(node, FunctionNode) and node.id == fn_id:
            return node
        if isinstance(node, TypeDefNode):
            for m in node.methods:
                if m.id == fn_id:
                    return m
    return None


def get_type(module: Module, type_id: str) -> Optional[TypeDefNode]:
    for node in module.nodes:
        if isinstance(node, TypeDefNode) and node.id == type_id:
            return node
    return None


def method_names(module: Module, type_id: str) -> List[str]:
    td = get_type(module, type_id)
    return [m.name for m in td.methods] if td else []


def _body_contains_literal(fn: FunctionNode, value: float) -> bool:
    import json
    body_json = json.dumps(fn.body.to_dict())
    return str(value) in body_json or str(int(value)) in body_json


def _body_contains_op(fn: FunctionNode, op: str) -> bool:
    import json
    body_json = json.dumps(fn.body.to_dict())
    return f'"op": "{op}"' in body_json


# ---------------------------------------------------------------------------
# Test definitions — bookstore.py analogues
# ---------------------------------------------------------------------------

@dataclass
class Test:
    number: int
    difficulty: int
    description: str
    verify: Callable[[Module], Tuple[bool, str]]
    tags: List[str] = field(default_factory=list)


TESTS: List[Test] = [

    # ── 1 ── trivial rename (method)
    Test(
        number=1, difficulty=1,
        description="Rename the method `revenue` in class `Book` to `total_price`.",
        tags=["rename", "method"],
        verify=lambda m: (
            get_fn(m, "fn:Book.total_price") is not None,
            "fn:Book.total_price exists" if get_fn(m, "fn:Book.total_price") else "method not renamed"
        ),
    ),

    # ── 2 ── rename top-level function
    Test(
        number=2, difficulty=2,
        description="Rename the top-level function `most_expensive` to `find_priciest`.",
        tags=["rename", "top_level_fn"],
        verify=lambda m: (
            get_fn(m, "fn:find_priciest") is not None,
            "fn:find_priciest exists" if get_fn(m, "fn:find_priciest") else "top-level fn not renamed"
        ),
    ),

    # ── 3 ── change return type
    Test(
        number=3, difficulty=3,
        description="Change the return type of `is_available` in `Book` to `optional<boolean>`.",
        tags=["retype", "return_type"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:Book.is_available")) is not None
            and fn.return_type.kind == TypeKind.OPTIONAL
            and fn.return_type.element is not None
            and fn.return_type.element.kind == TypeKind.BOOLEAN,
            f"return type is {get_fn(m,'fn:Book.is_available').return_type.render() if get_fn(m,'fn:Book.is_available') else 'missing'}"
        ),
    ),

    # ── 4 ── change param type
    Test(
        number=4, difficulty=4,
        description="Change the type of the `threshold` parameter in `out_of_stock` to `optional<number>`.",
        tags=["retype", "param_type"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:BookStore.out_of_stock")) is not None
            and any(
                p.name == "threshold" and p.type.kind == TypeKind.OPTIONAL
                for p in fn.params
            ),
            f"threshold type: {next((p.type.render() for p in (get_fn(m,'fn:BookStore.out_of_stock') or type('x',[],{'params':[]})()).params if p.name=='threshold'), 'missing')}"
            if get_fn(m, "fn:BookStore.out_of_stock") else "fn missing"
        ),
    ),

    # ── 5 ── remove a method
    Test(
        number=5, difficulty=4,
        description="Remove the method `by_genre` from the `BookStore` class.",
        tags=["remove", "method"],
        verify=lambda m: (
            "by_genre" not in method_names(m, "type:BookStore"),
            f"methods: {method_names(m, 'type:BookStore')}"
        ),
    ),

    # ── 6 ── body edit: change a literal value
    Test(
        number=6, difficulty=5,
        description=(
            "In the function `total_revenue` (inside `BookStore`), "
            "change the initial value of `result` from `0.0` to `50.0`."
        ),
        tags=["body_edit", "literal"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:BookStore.total_revenue")) is not None
            and _body_contains_literal(fn, 50.0),
            "50.0 found in body" if (fn := get_fn(m, "fn:BookStore.total_revenue")) and _body_contains_literal(fn, 50.0) else "literal not changed"
        ),
    ),

    # ── 7 ── body edit: change an operator
    Test(
        number=7, difficulty=6,
        description=(
            "In `apply_sale`, change the `-` operator in the price calculation "
            "to `+` (apply a surcharge instead of a discount)."
        ),
        tags=["body_edit", "operator"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:apply_sale")) is not None
            and _body_contains_op(fn, "+"),
            "+ operator found" if (fn := get_fn(m, "fn:apply_sale")) and _body_contains_op(fn, "+") else "operator not changed"
        ),
    ),

    # ── 8 ── rename + retype (two commands)
    Test(
        number=8, difficulty=7,
        description=(
            "Rename the method `remove_book` in `BookStore` to `delete_book`, "
            "and also change its return type from `boolean` to `optional<boolean>`."
        ),
        tags=["rename", "retype", "multi_step"],
        verify=lambda m: (
            get_fn(m, "fn:BookStore.delete_book") is not None
            and get_fn(m, "fn:BookStore.delete_book").return_type.kind == TypeKind.OPTIONAL,
            (
                f"fn:BookStore.delete_book exists={get_fn(m,'fn:BookStore.delete_book') is not None}, "
                f"return={get_fn(m,'fn:BookStore.delete_book').return_type.render() if get_fn(m,'fn:BookStore.delete_book') else 'missing'}"
            )
        ),
    ),

    # ── 9 ── prepend guard
    Test(
        number=9, difficulty=8,
        description=(
            "In `add_book` (inside `BookStore`), add a guard at the start: "
            "if `book` is None, return immediately. "
            "Prepend this check before the existing `self.books.append(book)` statement."
        ),
        tags=["body_edit", "add_stmt", "guard"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:BookStore.add_book")) is not None
            and len(fn.body.stmts) >= 2,
            f"body has {len(get_fn(m,'fn:BookStore.add_book').body.stmts)} stmts" if get_fn(m,'fn:BookStore.add_book') else "fn missing"
        ),
    ),

    # ── 10 ── cross-compile to TypeScript
    Test(
        number=10, difficulty=10,
        description="Compile this Python bookstore system to TypeScript.",
        tags=["cross_compile", "typescript"],
        verify=lambda m: (True, "checked via output"),
    ),
]


# ---------------------------------------------------------------------------
# Agent loop (identical to benchmark.py)
# ---------------------------------------------------------------------------

def extract_commands(text: str) -> List[str]:
    blocks = re.findall(r"```bash\s*(.*?)```", text, re.DOTALL)
    cmds = []
    for block in blocks:
        for line in block.strip().splitlines():
            line = line.strip()
            if line and (line.startswith("ast-harness") or line.startswith("ast-compiler")):
                cmds.append(line)
    return cmds


def is_done(text: str) -> bool:
    phrases = ["done.", "done!", "complete", "finished", "successfully",
               "has been renamed", "has been updated", "has been changed",
               "has been added", "has been removed"]
    lower = text.lower()
    return any(p in lower for p in phrases) and not extract_commands(text)


def run_test(test: Test, base_module: Module, model, tokenizer, ast_file: str) -> dict:
    from mlx_lm import generate as mlx_generate

    module = copy.deepcopy(base_module)
    skel   = compact_skeleton(module)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"{test.description}\n\n"
            f"The unified AST is at `{ast_file}`.\n\n"
            f"Skeleton:\n```\n{skel}\n```"
        )},
    ]

    t0 = time.time()
    turns_used = 0
    commands_run = []

    for turn in range(MAX_TURNS):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = mlx_generate(model, tokenizer, prompt=prompt,
                                max_tokens=700, verbose=False)
        for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            response = response.split(tok)[0]
        response = response.strip()

        turns_used += 1
        messages.append({"role": "assistant", "content": response})

        if is_done(response) and not extract_commands(response):
            break

        cmds = extract_commands(response)
        if not cmds:
            if is_done(response):
                break
            messages.append({"role": "user", "content": "Please issue the next command or say Done."})
            continue

        tool_parts = []
        for cmd in cmds:
            commands_run.append(cmd)
            output, module = run_command(cmd, module)
            tool_parts.append(f"$ {cmd}\n{output}")
        messages.append({"role": "user", "content": "[Tool output]\n" + "\n\n".join(tool_parts)})

    elapsed = time.time() - t0

    if test.number == 10:
        try:
            ts_source = gen_code(module, "typescript")
            passed = "class Book" in ts_source and "class BookStore" in ts_source
            detail = "TS output has Book + BookStore classes" if passed else "TS output missing classes"
        except Exception as e:
            passed, detail = False, str(e)
    else:
        passed, detail = test.verify(module)

    return {
        "number":     test.number,
        "difficulty": test.difficulty,
        "passed":     passed,
        "detail":     detail,
        "turns":      turns_used,
        "commands":   commands_run,
        "elapsed":    elapsed,
        "tags":       test.tags,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading source file…")
    source = Path(SOURCE_FILE).read_text()
    base_module = parse(source, "python", Path(SOURCE_FILE).name)
    ast_file = Path(SOURCE_FILE).name + ".json"

    print(f"Loading model {BASE_MODEL} + adapter {MODEL_PATH}…")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(BASE_MODEL, adapter_path=MODEL_PATH)

    results = []
    for test in TESTS:
        bar = "█" * test.difficulty + "░" * (10 - test.difficulty)
        print(f"\n{'─'*60}")
        print(f"  Test {test.number:02d}/10  [{bar}]  {test.description[:60]}")
        print(f"{'─'*60}")
        sys.stdout.flush()

        result = run_test(test, base_module, model, tokenizer, ast_file)
        results.append(result)

        icon = "✅" if result["passed"] else "❌"
        print(f"  {icon}  {result['detail']}  ({result['turns']} turns, {result['elapsed']:.1f}s)")
        for cmd in result["commands"]:
            print(f"     $ {cmd[:90]}")

    passed = sum(1 for r in results if r["passed"])
    total  = len(results)
    print(f"\n{'═'*60}")
    print(f"  GENERALIZATION RESULTS: {passed}/{total} passed  (bookstore.py — never seen in training)")
    print(f"{'═'*60}")
    print(f"  {'#':>2}  {'Diff':>4}  {'Pass':>4}  {'Turns':>5}  {'Time':>6}  Tags")
    print(f"  {'─'*56}")
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        tags = ", ".join(r["tags"])
        print(f"  {r['number']:>2}  {r['difficulty']:>4}  {icon:>4}  {r['turns']:>5}  {r['elapsed']:>5.1f}s  {tags}")

    avg_diff_passed = (
        sum(r["difficulty"] for r in results if r["passed"]) / passed
        if passed else 0
    )
    print(f"\n  Avg difficulty of passing tests: {avg_diff_passed:.1f}/10")
    print(f"  Total time: {sum(r['elapsed'] for r in results):.0f}s")


if __name__ == "__main__":
    main()
