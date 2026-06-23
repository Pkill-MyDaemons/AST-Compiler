"""
CLI: generate a training dataset from source files.

Usage:
    python -m training.generate --input tests/fixtures/ --output data/train.jsonl
    python -m training.generate --input myproject/ --output data/train.jsonl --max-files 200
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import click

from .dataset import process_file, process_directory, write_jsonl, dataset_stats


@click.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Source file or directory to process.")
@click.option("--output", "-o", "output_path", required=True, type=click.Path(),
              help="Output JSONL file.")
@click.option("--max-files", default=None, type=int,
              help="Maximum number of files to process (for quick runs).")
@click.option("--max-per-gen", default=3, type=int,
              help="Max tasks per generator per file (controls density).")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--split", default=0.9, type=float,
              help="Train/validation split ratio (default: 0.9 = 90% train).")
@click.option("--stats/--no-stats", default=True, help="Print dataset statistics.")
def main(input_path, output_path, max_files, max_per_gen, seed, split, stats):
    """Generate an AST-editing instruction-tuning dataset."""
    rng = random.Random(seed)
    input_path = Path(input_path)
    output_path = Path(output_path)

    click.echo(f"Generating dataset from {input_path} …", err=True)

    conversations = []
    if input_path.is_file():
        conversations = process_file(input_path, rng=rng, max_per_generator=max_per_gen)
    else:
        for conv in process_directory(input_path, rng=rng,
                                      max_per_generator=max_per_gen,
                                      max_files=max_files):
            conversations.append(conv)

    rng.shuffle(conversations)

    if split < 1.0 and len(conversations) > 1:
        cut = int(len(conversations) * split)
        train = conversations[:cut]
        val = conversations[cut:]

        train_path = output_path.with_suffix("") .parent / (output_path.stem + "_train.jsonl")
        val_path = output_path.with_suffix("").parent / (output_path.stem + "_val.jsonl")

        write_jsonl(train, train_path)
        write_jsonl(val, val_path)
        click.echo(f"Wrote {len(train)} train examples → {train_path}", err=True)
        click.echo(f"Wrote {len(val)} val examples → {val_path}", err=True)
    else:
        n = write_jsonl(conversations, output_path)
        click.echo(f"Wrote {n} examples → {output_path}", err=True)

    if stats and conversations:
        s = dataset_stats(conversations)
        click.echo("\nDataset statistics:", err=True)
        click.echo(json.dumps(s, indent=2), err=True)


if __name__ == "__main__":
    main()
