"""
Dataset generation pipeline.

For each source file:
  1. Decompile to unified AST
  2. Generate tasks (rename, retype, body_edit, …)
  3. Execute each task's steps against a fresh copy of the module
  4. Format into conversation turns
  5. Write to JSONL

Output format: ShareGPT (compatible with HuggingFace TRL/SFT, LLaMA-Factory, Axolotl).
"""
from __future__ import annotations
import copy
import json
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional

from src.parsers import parse, detect_language
from src.harness.skeleton import build_skeleton
from src.unified_ast.nodes import Module
from .system_prompt import SYSTEM_PROMPT
from .tasks import Task, Step, generate_all_tasks
from .executor import run_command


def compact_skeleton(module: Module) -> str:
    """One-line-per-node skeleton that fits in context without truncation."""
    lines = [f"# {module.source_language}  file={module.source_file}"]
    for node in module.nodes:
        if node.kind == "import":
            lines.append(f"[import]  id={node.id}  {node.module}")
        elif node.kind == "variable":
            lines.append(f"[var]     id={node.id}  {node.name}: {node.type.render()}")
        elif node.kind == "function":
            params = ", ".join(
                (f"{p.name}: {p.type.render()}" if not p.is_self else p.name)
                for p in node.params
            )
            ret = node.return_type.render()
            lines.append(f"[fn]      id={node.id}  {node.name}({params}) -> {ret}  [{len(node.body.stmts)} stmts]")
        elif node.kind == "type_def":
            lines.append(f"[{node.category.value}]   id={node.id}  {node.name}")
            for f in node.fields:
                lines.append(f"  [field]  id={f.id}  {f.name}: {f.type.render()}")
            for m in node.methods:
                params = ", ".join(
                    (f"{p.name}: {p.type.render()}" if not p.is_self else p.name)
                    for p in m.params
                )
                ret = m.return_type.render()
                lines.append(f"  [fn]     id={m.id}  {m.name}({params}) -> {ret}  [{len(m.body.stmts)} stmts]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversation formatting
# ---------------------------------------------------------------------------

def task_to_conversation(
    task: Task,
    module: Module,
    ast_file: str = "code.json",
) -> Optional[dict]:
    """
    Build a ShareGPT-format conversation dict for one task.

    The trajectory is:
      user:       task description + "The AST is at code.json."
      assistant:  rationale + bash block for step 1
      tool:       output of step 1
      assistant:  rationale + bash block for step 2
      tool:       output of step 2
      ...
      assistant:  final summary
    """
    # Work on a deep copy so tasks don't interfere with each other
    mod = copy.deepcopy(module)
    turns = []

    # Include compact skeleton in the user turn so the model always has IDs
    skel = compact_skeleton(module)
    turns.append({
        "from": "human",
        "value": (
            f"{task.description}\n\n"
            f"The unified AST is at `{ast_file}`.\n\n"
            f"Skeleton:\n```\n{skel}\n```"
        ),
    })

    tool_outputs = []

    for i, step in enumerate(task.steps):
        # Execute command against the live (copied) module
        output, mod = run_command(step.command, mod)

        # Skip on error (don't include broken examples)
        if output.startswith("Error:"):
            return None

        tool_outputs.append((step, output))

    if not tool_outputs:
        return None

    # Build assistant turns interleaved with tool outputs
    for i, (step, output) in enumerate(tool_outputs):
        is_last = i == len(tool_outputs) - 1

        # Assistant speaks then issues the command
        assistant_text = f"{step.rationale}\n\n```bash\n{step.command}\n```"
        turns.append({"from": "gpt", "value": assistant_text})

        # Tool result
        turns.append({"from": "tool", "value": output})

    # Final assistant summary
    turns.append({
        "from": "gpt",
        "value": _final_summary(task),
    })

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            *turns,
        ],
        "metadata": {
            "category": task.category,
            "difficulty": task.difficulty,
            "tags": task.tags,
            "source_language": module.source_language,
        },
    }


def _final_summary(task: Task) -> str:
    cat = task.category
    if cat == "rename":
        return "Done. The node has been renamed successfully."
    if cat == "retype":
        return "Done. The type has been updated."
    if cat == "body_edit":
        return "Done. The function body has been updated."
    if cat == "add":
        return "Done. The new statement/node has been added."
    if cat == "remove":
        return "Done. The node has been removed."
    if cat == "structural":
        return "Done. The structural change has been applied."
    return "Done."


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------

def process_file(
    source_path: Path,
    ast_file: str = "code.json",
    rng: Optional[random.Random] = None,
    max_per_generator: int = 3,
) -> List[dict]:
    if rng is None:
        rng = random.Random(42)

    source = source_path.read_text(encoding="utf-8", errors="replace")
    try:
        language = detect_language(source_path.name)
        module = parse(source, language, filename=source_path.name)
    except Exception as e:
        print(f"  ⚠  Parse error {source_path.name}: {e}", file=sys.stderr)
        return []

    tasks = generate_all_tasks(module, ast_file, rng, max_per_generator)
    conversations = []
    for task in tasks:
        conv = task_to_conversation(task, module, ast_file)
        if conv is not None:
            conversations.append(conv)

    return conversations


# ---------------------------------------------------------------------------
# Multi-file pipeline
# ---------------------------------------------------------------------------

def process_directory(
    directory: Path,
    extensions: tuple = (".py", ".rs", ".ts"),
    rng: Optional[random.Random] = None,
    max_per_generator: int = 3,
    max_files: Optional[int] = None,
) -> Iterator[dict]:
    if rng is None:
        rng = random.Random(42)

    files = [p for p in sorted(directory.rglob("*")) if p.suffix in extensions]
    if max_files:
        files = files[:max_files]

    for i, path in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {path.name}", file=sys.stderr)
        convs = process_file(path, ast_file=f"{path.name}.json", rng=rng,
                             max_per_generator=max_per_generator)
        yield from convs


def write_jsonl(conversations: List[dict], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    return len(conversations)


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def dataset_stats(conversations: List[dict]) -> dict:
    from collections import Counter
    cats = Counter(c["metadata"]["category"] for c in conversations)
    langs = Counter(c["metadata"]["source_language"] for c in conversations)
    diffs = Counter(c["metadata"]["difficulty"] for c in conversations)
    avg_turns = sum(len(c["conversations"]) for c in conversations) / max(len(conversations), 1)
    return {
        "total": len(conversations),
        "by_category": dict(cats),
        "by_language": dict(langs),
        "by_difficulty": dict(diffs),
        "avg_turns": round(avg_turns, 1),
    }
