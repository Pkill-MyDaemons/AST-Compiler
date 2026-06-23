"""
MLX-based agent loop — same workflow as agent.py but using MLX for inference.
Inference is 3-5x faster than PyTorch MPS on Apple Silicon.

Usage:
    python -m training.agent_mlx \\
        --source examples/inventory.py \\
        --task "Rename total_value to compute_total_cost" \\
        --adapter models/ast-editor-mlx \\
        --base-model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
"""
from __future__ import annotations
import copy
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import click

from src.parsers import parse, detect_language
from src.generators import generate as gen_code
from src.unified_ast.nodes import Module
from training.system_prompt import SYSTEM_PROMPT
from training.executor import run_command
from training.dataset import compact_skeleton

MAX_TURNS = 10


# ---------------------------------------------------------------------------
# MLX model loading & generation
# ---------------------------------------------------------------------------

def load_mlx_model(adapter_path: str, base_model: str):
    from mlx_lm import load
    print(f"  Loading {base_model} + adapter {adapter_path} …", file=sys.stderr)
    model, tokenizer = load(base_model, adapter_path=adapter_path)
    return model, tokenizer


def model_step_mlx(model, tokenizer, messages: list, max_tokens: int = 512) -> str:
    from mlx_lm import generate

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    # Strip Qwen chat delimiters that sometimes bleed through
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        response = response.split(tok)[0]
    return response.strip()


# ---------------------------------------------------------------------------
# Shared agent logic (same as agent.py)
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
    done_phrases = ["done.", "done!", "complete", "finished",
                    "successfully", "has been renamed", "has been updated",
                    "has been changed", "has been added", "has been removed"]
    lower = text.lower()
    return any(p in lower for p in done_phrases) and not extract_commands(text)


DIV = "─" * 60

def pr(role: str, text: str):
    colors = {"assistant": "\033[92m", "tool": "\033[90m", "user": "\033[94m"}
    reset = "\033[0m"
    c = colors.get(role, "")
    print(f"\n[{c}{role.upper()}{reset}]")
    print(text[:1000] + (" …" if len(text) > 1000 else ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _verify_module(module, language: str) -> str:
    """Compile module back to source and syntax-check. Returns error string or '' on success."""
    try:
        source = gen_code(module, language)
    except Exception as e:
        return f"Compile error: {e}"
    if language == "python":
        import ast as _ast
        try:
            _ast.parse(source)
        except SyntaxError as e:
            return f"Syntax error: {e}"
    return ""


def run_agent(source_path: Path, task: str, adapter: str, base_model: str,
              target_lang: str, verbose: bool, verify: bool = False):
    # Parse
    source = source_path.read_text()
    language = detect_language(source_path.name)
    module = parse(source, language, filename=source_path.name)
    skel_text = compact_skeleton(module)

    print(f"\n{DIV}")
    print(f"  Source   : {source_path.name}  ({language}, {len(module.nodes)} nodes)")
    print(f"  Task     : {task}")
    print(DIV)
    print(skel_text)

    # Load MLX model
    print(f"\n{DIV}  Loading model  {DIV}")
    model_obj, tokenizer = load_mlx_model(adapter, base_model)

    # Build initial prompt
    ast_file = source_path.name + ".json"
    user_content = (
        f"{task}\n\n"
        f"The unified AST is at `{ast_file}`.\n\n"
        f"Skeleton:\n```\n{skel_text}\n```"
    )
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
    ]

    if verify:
        print(f"  Verify mode ON — will compile after each body edit", file=sys.stderr)
    print(f"\n{DIV}  Agent loop  {DIV}")

    for turn in range(MAX_TURNS):
        response = model_step_mlx(model_obj, tokenizer, messages)
        pr("assistant", response)
        messages.append({"role": "assistant", "content": response})

        if is_done(response) and not extract_commands(response):
            break

        cmds = extract_commands(response)
        if not cmds:
            if is_done(response):
                break
            messages.append({"role": "user", "content": "Please issue the next command or say Done."})
            continue

        tool_outputs = []
        for cmd in cmds:
            output, module = run_command(cmd, module)
            pr("tool", f"$ {cmd}\n{output}")
            tool_outputs.append(f"$ {cmd}\n{output}")
            # Verify after body-edit commands
            if verify and any(sub in cmd for sub in ("str-replace", "append-stmt", "insert-before")):
                verify_msg = _verify_module(module, language)
                if verify_msg:
                    tool_outputs.append(f"[Verify] {verify_msg}")
                    pr("tool", f"[Verify] {verify_msg}")
                else:
                    tool_outputs.append("[Verify] Compiled successfully — no syntax errors.")

        messages.append({"role": "user", "content": "[Tool output]\n" + "\n\n".join(tool_outputs)})

    # Compile result
    print(f"\n{DIV}  Final {target_lang}  {DIV}")
    try:
        print(gen_code(module, target_lang))
    except Exception as e:
        print(f"Compile error: {e}", file=sys.stderr)


@click.command()
@click.option("--source", "-s", required=True, type=click.Path(exists=True))
@click.option("--task",   "-t", required=True)
@click.option("--adapter", default="models/ast-editor-mlx", show_default=True)
@click.option("--base-model", default="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
              show_default=True)
@click.option("--lang", "target_lang", default=None)
@click.option("--verbose/--no-verbose", default=False)
@click.option("--verify/--no-verify", default=False,
              help="After each body edit, compile and syntax-check; feed errors back to the model.")
def main(source, task, adapter, base_model, target_lang, verbose, verify):
    """Run the MLX-based AST-editor agent."""
    path = Path(source)
    lang = target_lang or detect_language(path.name)
    run_agent(path, task, adapter, base_model, lang, verbose, verify)


if __name__ == "__main__":
    main()
