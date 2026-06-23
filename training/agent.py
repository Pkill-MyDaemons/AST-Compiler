"""
AST-editor agent loop.

Drives the trained model through a real edit session:
  1. Decompile source → AST
  2. Build skeleton → first user message
  3. Loop: model generates command → harness executes → output fed back
  4. Compile AST → show final source

Usage:
    python -m training.agent \\
        --source examples/inventory.py \\
        --task "Rename the method total_value to compute_total_cost" \\
        --adapter models/ast-editor-1.5b
"""
from __future__ import annotations
import copy
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple, Optional

import click
import torch

from src.parsers import parse, detect_language
from src.generators import generate as gen_code
from src.harness.skeleton import build_skeleton
from src.unified_ast.nodes import Module
from training.system_prompt import SYSTEM_PROMPT
from training.executor import run_command
from training.dataset import compact_skeleton

MAX_TURNS = 10   # safety limit


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(adapter_path: str, base_model: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) \
             else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device : {device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Single inference step
# ---------------------------------------------------------------------------

def model_step(model, tokenizer, device: str, messages: list,
               max_new_tokens: int = 400) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

def extract_commands(text: str) -> List[str]:
    """Pull every ```bash ... ``` block out of model text."""
    blocks = re.findall(r"```bash\s*(.*?)```", text, re.DOTALL)
    cmds = []
    for block in blocks:
        for line in block.strip().splitlines():
            line = line.strip()
            if line and (line.startswith("ast-harness") or line.startswith("ast-compiler")):
                cmds.append(line)
    return cmds


def is_done(text: str) -> bool:
    done_phrases = [
        "done.", "done!", "complete", "finished", "all done",
        "successfully", "has been renamed", "has been updated",
        "has been changed", "has been added", "has been removed",
    ]
    lower = text.lower()
    return any(p in lower for p in done_phrases) and not extract_commands(text)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIV = "─" * 60

def print_header(title: str):
    print(f"\n{DIV}")
    print(f"  {title}")
    print(DIV)

def print_role(role: str, text: str, color: str = ""):
    colors = {"user": "\033[94m", "assistant": "\033[92m", "tool": "\033[90m", "": ""}
    reset = "\033[0m" if color else ""
    tag = f"{colors.get(role,'')}{role.upper()}{reset}"
    print(f"\n[{tag}]")
    # Wrap long tool output
    if role == "tool" and len(text) > 800:
        print(text[:800] + "\n  … (truncated)")
    else:
        print(text)


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(
    source_path: Path,
    task: str,
    adapter_path: str,
    base_model: str,
    target_lang: str,
    verbose: bool,
):
    # 1. Parse source
    print_header("Parsing source")
    source = source_path.read_text(encoding="utf-8")
    language = detect_language(source_path.name)
    module = parse(source, language, filename=source_path.name)
    print(f"  language : {language}")
    print(f"  nodes    : {len(module.nodes)} top-level")

    # 2. Build compact skeleton for first turn (fits fully in context, shows IDs explicitly)
    skel_text = compact_skeleton(module)

    # 3. Load model
    print_header("Loading model")
    print(f"  adapter  : {adapter_path}")
    print(f"  base     : {base_model}")
    model, tokenizer, device = load_model(adapter_path, base_model)

    # 4. Construct initial messages
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

    print_header(f"Task: {task}")
    if verbose:
        print_role("user", user_content)

    # 5. Agent loop
    print_header("Agent loop")
    turn = 0
    while turn < MAX_TURNS:
        turn += 1

        # Model generates next response
        assistant_text = model_step(model, tokenizer, device, messages)
        print_role("assistant", assistant_text, "assistant")
        messages.append({"role": "assistant", "content": assistant_text})

        # Check if done
        if is_done(assistant_text) and not extract_commands(assistant_text):
            break

        # Extract and execute commands
        commands = extract_commands(assistant_text)
        if not commands:
            # Model didn't emit a command — it may be done or confused
            if is_done(assistant_text):
                break
            # Nudge
            messages.append({
                "role": "user",
                "content": "Please issue the next harness command, or say 'Done' if finished."
            })
            continue

        # Execute each command and collect outputs
        tool_outputs = []
        for cmd in commands:
            output, module = run_command(cmd, module)
            print_role("tool", f"$ {cmd}\n{output}", "tool")
            tool_outputs.append(f"$ {cmd}\n{output}")

        # Feed all outputs back as a user/tool message
        combined = "\n\n".join(tool_outputs)
        messages.append({"role": "user", "content": f"[Tool output]\n{combined}"})

    # 6. Compile final result
    print_header(f"Final source ({target_lang})")
    try:
        final_source = gen_code(module, target_lang)
        print(final_source)
    except Exception as e:
        print(f"Compile error: {e}", file=sys.stderr)

    # 7. Show diff vs original
    print_header("Summary")
    orig_lines = set(source.splitlines())
    new_lines  = set(final_source.splitlines()) if 'final_source' in dir() else set()
    added   = [l for l in new_lines   if l not in orig_lines and l.strip()]
    removed = [l for l in orig_lines  if l not in new_lines  and l.strip()]
    if removed:
        for l in removed[:8]:
            print(f"  - {l}")
    if added:
        for l in added[:8]:
            print(f"  + {l}")
    if not removed and not added:
        print("  (no textual changes — model may need more training data)")

    return module, final_source if 'final_source' in dir() else ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--source", "-s", required=True, type=click.Path(exists=True))
@click.option("--task",   "-t", required=True, help="Natural-language editing instruction")
@click.option("--adapter", default="models/ast-editor-1.5b", show_default=True)
@click.option("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", show_default=True)
@click.option("--lang", "target_lang", default=None,
              help="Output language (default: same as input)")
@click.option("--verbose/--no-verbose", default=False)
def main(source, task, adapter, base_model, target_lang, verbose):
    """Run the AST-editor agent on a source file."""
    source_path = Path(source)
    if target_lang is None:
        target_lang = detect_language(source_path.name)
    run_agent(source_path, task, adapter, base_model, target_lang, verbose)


if __name__ == "__main__":
    main()
