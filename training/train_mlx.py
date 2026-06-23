"""
MLX-LM fine-tuning — runs on Apple Silicon (M1/M2/M3) via Metal.
Typically 5-10x faster than PyTorch MPS for transformer LoRA training.

Steps:
  1. Generate dataset          ast-gen-data ...
  2. Export to MLX format      python -m training.mlx_export ...
  3. Run this script           python -m training.train_mlx ...
  4. Fuse adapter (optional)   mlx_lm.fuse --model ... --adapter ...

Usage:
    python -m training.train_mlx \\
        --data data/mlx \\
        --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \\
        --output models/ast-editor-mlx \\
        --iters 600
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

import click


# Default 4-bit quantized model — fastest + smallest RAM footprint on M1
DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"


@click.command()
@click.option("--data",   "-d", default="data/mlx",     show_default=True,
              help="Directory with train.jsonl / valid.jsonl (MLX format).")
@click.option("--model",  "-m", default=DEFAULT_MODEL,   show_default=True,
              help="MLX model repo (must be mlx-community/* or local mlx dir).")
@click.option("--output", "-o", default="models/ast-editor-mlx", show_default=True,
              help="Where to save the LoRA adapter.")
@click.option("--iters",        default=600,  type=int,  show_default=True,
              help="Training iterations (steps).")
@click.option("--batch-size",   default=4,    type=int,  show_default=True,
              help="Batch size. M1 Max handles 4-8 easily at 4-bit.")
@click.option("--lora-layers",  default=16,   type=int,  show_default=True,
              help="Number of transformer layers to apply LoRA to.")
@click.option("--lr",           default=1e-4, type=float, show_default=True)
@click.option("--max-seq-len",  default=2048, type=int,  show_default=True)
@click.option("--steps-per-report", default=10, type=int)
@click.option("--save-every",   default=200,  type=int,
              help="Save adapter checkpoint every N steps.")
@click.option("--grad-checkpoint/--no-grad-checkpoint", default=True,
              help="Gradient checkpointing — saves ~40% memory, ~10% slower.")
@click.option("--seed", default=42, type=int)
def main(data, model, output, iters, batch_size, lora_layers, lr,
         max_seq_len, steps_per_report, save_every, grad_checkpoint, seed):
    """Fine-tune with MLX LoRA — much faster than PyTorch MPS on Apple Silicon."""
    data_path = Path(data)
    if not (data_path / "train.jsonl").exists():
        raise SystemExit(
            f"No train.jsonl in {data}. "
            "Run: python -m training.mlx_export data/dataset_train.jsonl "
            "data/dataset_val.jsonl data/mlx"
        )

    n_train = sum(1 for _ in open(data_path / "train.jsonl"))
    n_valid = sum(1 for _ in open(data_path / "valid.jsonl")) if (data_path / "valid.jsonl").exists() else 0
    click.echo(f"Dataset  : {n_train} train, {n_valid} valid")
    click.echo(f"Model    : {model}")
    click.echo(f"Output   : {output}")
    click.echo(f"Iters    : {iters}  batch={batch_size}  lr={lr}")
    click.echo(f"LoRA     : {lora_layers} layers")

    Path(output).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model",               model,
        "--train",
        "--data",                str(data_path),
        "--adapter-path",        output,
        "--iters",               str(iters),
        "--batch-size",          str(batch_size),
        "--num-layers",          str(lora_layers),
        "--learning-rate",       str(lr),
        "--max-seq-length",      str(max_seq_len),
        "--steps-per-report",    str(steps_per_report),
        "--save-every",          str(save_every),
        "--seed",                str(seed),
    ]
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")

    click.echo(f"\n$ {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise SystemExit(f"MLX training failed (exit {result.returncode})")

    click.echo(f"\nTraining finished in {elapsed/60:.1f} min.")
    click.echo(f"Adapter saved to {output}/")
    click.echo("\nTo fuse adapter into a standalone model:")
    click.echo(f"  mlx_lm.fuse --model {model} --adapter-path {output} --save-path models/ast-editor-fused")
    click.echo("\nTo run inference:")
    click.echo(f"  python -m training.agent_mlx --source examples/inventory.py --task '...'")


if __name__ == "__main__":
    main()
