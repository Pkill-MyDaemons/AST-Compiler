"""
Generate the targeted top-up dataset using TARGETED_GENERATORS.

Usage:
    python -m training.generate_targeted
"""
from __future__ import annotations
import copy
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

from src.parsers import parse, detect_language
from src.unified_ast.nodes import Module
from training.dataset import (
    compact_skeleton, task_to_conversation, write_jsonl, dataset_stats,
)
from training.executor import run_command
from training.tasks_targeted import TARGETED_GENERATORS
from training.mlx_export import export as mlx_export


def generate_targeted_tasks(module: Module, ast_file: str, rng: random.Random,
                             max_per_generator: int = 5) -> list:
    tasks = []
    for gen in TARGETED_GENERATORS:
        batch = gen(module, ast_file, rng)
        rng.shuffle(batch)
        tasks.extend(batch[:max_per_generator])
    return tasks


def main():
    rng = random.Random(42)
    synth_dir = Path("examples/synth")
    out_train = Path("data/targeted_v4_train.jsonl")
    out_val   = Path("data/targeted_v4_val.jsonl")
    mlx_out   = Path("data/mlx_targeted_v4")

    source_files = list(synth_dir.glob("*.py")) + list(synth_dir.glob("*.rs"))
    if not source_files:
        print("No synth files found in examples/synth/", file=sys.stderr)
        sys.exit(1)

    conversations = []
    for path in source_files:
        source = path.read_text(encoding="utf-8")
        try:
            lang = detect_language(path.name)
            module = parse(source, lang, filename=path.name)
        except Exception as e:
            print(f"  Parse error {path.name}: {e}", file=sys.stderr)
            continue

        ast_file = path.name + ".json"
        tasks = generate_targeted_tasks(module, ast_file, rng, max_per_generator=5)
        print(f"  {path.name}: {len(tasks)} targeted tasks", file=sys.stderr)

        for task in tasks:
            conv = task_to_conversation(task, module, ast_file)
            if conv is not None:
                conversations.append(conv)

    rng.shuffle(conversations)
    print(f"\nTotal: {len(conversations)} conversations", file=sys.stderr)

    # Train/val split (90/10)
    cut = max(1, int(len(conversations) * 0.9))
    train_convs = conversations[:cut]
    val_convs   = conversations[cut:]

    write_jsonl(train_convs, out_train)
    write_jsonl(val_convs,   out_val)
    print(f"Wrote {len(train_convs)} train → {out_train}", file=sys.stderr)
    print(f"Wrote {len(val_convs)} val   → {out_val}", file=sys.stderr)

    # Export to MLX format
    n_tr, n_val = mlx_export(str(out_train), str(out_val), str(mlx_out))
    print(f"MLX export: {n_tr} train + {n_val} val → {mlx_out}/", file=sys.stderr)

    # Stats
    if conversations:
        s = dataset_stats(conversations)
        print(json.dumps(s, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
