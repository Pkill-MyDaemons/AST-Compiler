"""
Rust generalization benchmark — tests v5 on counter.rs.
The model was trained on counter.rs/wallet.rs/scoreboard.rs,
so this tests whether Rust editing patterns transferred.
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
SOURCE_FILE = "examples/synth/wallet.rs"   # test on wallet — similar structure to counter


def get_fn(module, fn_id):
    for node in module.nodes:
        if isinstance(node, FunctionNode) and node.id == fn_id:
            return node
        if isinstance(node, TypeDefNode):
            for m in node.methods:
                if m.id == fn_id:
                    return m
    return None

def get_type(module, type_id):
    for node in module.nodes:
        if isinstance(node, TypeDefNode) and node.id == type_id:
            return node
    return None

def method_names(module, type_id):
    td = get_type(module, type_id)
    return [m.name for m in td.methods] if td else []

def _body_contains_literal(fn, value):
    import json
    body_json = json.dumps(fn.body.to_dict())
    return str(value) in body_json

def _body_contains_op(fn, op):
    import json
    body_json = json.dumps(fn.body.to_dict())
    return f'"op": "{op}"' in body_json


@dataclass
class Test:
    number: int
    difficulty: int
    description: str
    verify: Callable[[Module], Tuple[bool, str]]
    tags: List[str] = field(default_factory=list)


TESTS: List[Test] = [

    # ── 1 ── rename method
    Test(
        number=1, difficulty=1,
        description="Rename the method `deposit` in struct `Wallet` to `credit`.",
        tags=["rename", "method"],
        verify=lambda m: (
            get_fn(m, "fn:Wallet.credit") is not None,
            "fn:Wallet.credit exists" if get_fn(m, "fn:Wallet.credit") else "method not renamed"
        ),
    ),

    # ── 2 ── rename top-level function
    Test(
        number=2, difficulty=2,
        description="Rename the top-level function `richest` to `find_richest`.",
        tags=["rename", "top_level_fn"],
        verify=lambda m: (
            get_fn(m, "fn:find_richest") is not None,
            "fn:find_richest exists" if get_fn(m, "fn:find_richest") else "fn not renamed"
        ),
    ),

    # ── 3 ── change return type
    Test(
        number=3, difficulty=3,
        description="Change the return type of `is_solvent` in `Wallet` to `optional<boolean>`.",
        tags=["retype", "return_type"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:Wallet.is_solvent")) is not None
            and fn.return_type.kind == TypeKind.OPTIONAL,
            f"return type is {get_fn(m,'fn:Wallet.is_solvent').return_type.render() if get_fn(m,'fn:Wallet.is_solvent') else 'missing'}"
        ),
    ),

    # ── 4 ── change param type
    Test(
        number=4, difficulty=4,
        description="Change the type of the `fee` parameter in `apply_fee` to `optional<number>`.",
        tags=["retype", "param_type"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:apply_fee")) is not None
            and any(p.name == "fee" and p.type.kind == TypeKind.OPTIONAL for p in fn.params),
            "fee param is optional<number>" if get_fn(m, "fn:apply_fee") else "fn missing"
        ),
    ),

    # ── 5 ── remove method
    Test(
        number=5, difficulty=4,
        description="Remove the method `total_debits` from the `Wallet` struct.",
        tags=["remove", "method"],
        verify=lambda m: (
            "total_debits" not in method_names(m, "type:Wallet"),
            f"methods: {method_names(m, 'type:Wallet')}"
        ),
    ),

    # ── 6 ── body edit: literal
    Test(
        number=6, difficulty=5,
        description=(
            "In `total_credits` (inside `Wallet`), change the initial value of `total` from `0.0` to `10.0`."
        ),
        tags=["body_edit", "literal"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:Wallet.total_credits")) is not None
            and _body_contains_literal(fn, 10.0),
            "10.0 found" if (fn := get_fn(m, "fn:Wallet.total_credits")) and _body_contains_literal(fn, 10.0) else "literal not changed"
        ),
    ),

    # ── 7 ── body edit: operator
    Test(
        number=7, difficulty=6,
        description="In `apply_fee`, change the `>` operator to `>=`.",
        tags=["body_edit", "operator"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:apply_fee")) is not None
            and _body_contains_op(fn, ">="),
            ">= found" if (fn := get_fn(m, "fn:apply_fee")) and _body_contains_op(fn, ">=") else "op not changed"
        ),
    ),

    # ── 8 ── rename + retype
    Test(
        number=8, difficulty=7,
        description=(
            "Rename `withdraw` in `Wallet` to `debit`, "
            "and change its return type from `boolean` to `optional<boolean>`."
        ),
        tags=["rename", "retype", "multi_step"],
        verify=lambda m: (
            get_fn(m, "fn:Wallet.debit") is not None
            and get_fn(m, "fn:Wallet.debit").return_type.kind == TypeKind.OPTIONAL,
            f"debit exists={get_fn(m,'fn:Wallet.debit') is not None}, "
            f"return={get_fn(m,'fn:Wallet.debit').return_type.render() if get_fn(m,'fn:Wallet.debit') else 'missing'}"
        ),
    ),

    # ── 9 ── prepend guard
    Test(
        number=9, difficulty=8,
        description=(
            "In `deposit` (inside `Wallet`), add a guard at the start: "
            "if `amount` is zero or negative, return immediately. "
            "Prepend this check before the existing first statement."
        ),
        tags=["body_edit", "add_stmt", "guard"],
        verify=lambda m: (
            (fn := get_fn(m, "fn:Wallet.deposit")) is not None
            and len(fn.body.stmts) >= 3,
            f"body has {len(get_fn(m,'fn:Wallet.deposit').body.stmts)} stmts"
            if get_fn(m, "fn:Wallet.deposit") else "fn missing"
        ),
    ),

    # ── 10 ── cross-compile to Python
    Test(
        number=10, difficulty=10,
        description="Compile this Rust wallet system to Python.",
        tags=["cross_compile", "python"],
        verify=lambda m: (True, "checked via output"),
    ),
]


def extract_commands(text):
    blocks = re.findall(r"```bash\s*(.*?)```", text, re.DOTALL)
    cmds = []
    for block in blocks:
        for line in block.strip().splitlines():
            line = line.strip()
            if line and (line.startswith("ast-harness") or line.startswith("ast-compiler")):
                cmds.append(line)
    return cmds


def is_done(text):
    phrases = ["done.", "done!", "complete", "finished", "successfully",
               "has been renamed", "has been updated", "has been changed",
               "has been added", "has been removed"]
    lower = text.lower()
    return any(p in lower for p in phrases) and not extract_commands(text)


def run_test(test, base_module, model, tokenizer, ast_file):
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
            py_source = gen_code(module, "python")
            passed = "class Wallet" in py_source and "class Transaction" in py_source
            detail = "Python output has Wallet + Transaction" if passed else "missing classes"
        except Exception as e:
            passed, detail = False, str(e)
    else:
        passed, detail = test.verify(module)

    return {
        "number": test.number, "difficulty": test.difficulty,
        "passed": passed, "detail": detail,
        "turns": turns_used, "commands": commands_run,
        "elapsed": elapsed, "tags": test.tags,
    }


def main():
    print("Loading source file…")
    source = Path(SOURCE_FILE).read_text()
    base_module = parse(source, "rust", Path(SOURCE_FILE).name)
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
    print(f"  RUST RESULTS: {passed}/{total} passed  (wallet.rs)")
    print(f"{'═'*60}")
    print(f"  {'#':>2}  {'Diff':>4}  {'Pass':>4}  {'Turns':>5}  {'Time':>6}  Tags")
    print(f"  {'─'*56}")
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {r['number']:>2}  {r['difficulty']:>4}  {icon:>4}  {r['turns']:>5}  {r['elapsed']:>5.1f}s  {', '.join(r['tags'])}")

    avg = sum(r["difficulty"] for r in results if r["passed"]) / passed if passed else 0
    print(f"\n  Avg difficulty of passing: {avg:.1f}/10")
    print(f"  Total time: {sum(r['elapsed'] for r in results):.0f}s")


if __name__ == "__main__":
    main()
