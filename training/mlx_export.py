"""
Convert ShareGPT JSONL → MLX-LM chat format.

MLX-LM (0.20+) accepts:
  data/train.jsonl  — {"messages": [...]}  one conversation per line
  data/valid.jsonl

The messages format is identical to OpenAI chat: role in {system, user, assistant}.
Tool outputs are merged into the preceding user turn to keep roles compatible.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List


def sharegpt_to_mlx_messages(conv: dict) -> dict:
    """Convert one ShareGPT example to MLX messages format."""
    role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
    raw = conv["conversations"]
    messages = []

    i = 0
    while i < len(raw):
        turn = raw[i]
        role = role_map.get(turn["from"], turn["from"])

        if role == "tool":
            # Merge tool output into the next user turn (or append as user if last)
            tool_text = f"[Tool output]\n{turn['value']}"
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += f"\n\n{tool_text}"
            else:
                messages.append({"role": "user", "content": tool_text})
            i += 1
            continue

        if role == "user" and messages and messages[-1]["role"] == "user":
            # Merge consecutive user turns (can happen after tool output merging)
            messages[-1]["content"] += f"\n\n{turn['value']}"
            i += 1
            continue

        # MLX only accepts system / user / assistant
        if role not in ("system", "user", "assistant"):
            role = "user"

        messages.append({"role": role, "content": turn["value"]})
        i += 1

    return {"messages": messages}


def export(
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
) -> tuple[int, int]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def convert_file(src: str, dst: Path) -> int:
        if not Path(src).exists():
            return 0
        count = 0
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                mlx_example = sharegpt_to_mlx_messages(example)
                # Keep metadata for debugging but MLX ignores extra keys
                if "metadata" in example:
                    mlx_example["metadata"] = example["metadata"]
                fout.write(json.dumps(mlx_example, ensure_ascii=False) + "\n")
                count += 1
        return count

    n_train = convert_file(train_jsonl, out / "train.jsonl")
    n_valid = convert_file(val_jsonl,   out / "valid.jsonl")
    return n_train, n_valid


if __name__ == "__main__":
    import sys
    train = sys.argv[1] if len(sys.argv) > 1 else "data/dataset_train.jsonl"
    val   = sys.argv[2] if len(sys.argv) > 2 else "data/dataset_val.jsonl"
    out   = sys.argv[3] if len(sys.argv) > 3 else "data/mlx"
    n_tr, n_val = export(train, val, out)
    print(f"Wrote {n_tr} train + {n_val} val examples to {out}/")
