"""
Fine-tuning script using HuggingFace TRL SFTTrainer.

Install deps first:
    pip install trl transformers datasets accelerate bitsandbytes peft torch

Recommended base models (smallest → largest):
    Qwen/Qwen2.5-Coder-1.5B-Instruct   ← good default, fast, understands JSON
    Qwen/Qwen2.5-Coder-3B-Instruct
    Qwen/Qwen2.5-Coder-7B-Instruct
    mistralai/Mistral-7B-Instruct-v0.3
    meta-llama/Llama-3.2-3B-Instruct

Usage:
    python -m training.train \\
        --dataset data/train.jsonl \\
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --output models/ast-editor-1.5b \\
        --epochs 3
"""
from __future__ import annotations
import json
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------

def load_sharegpt_jsonl(path: str):
    """Load ShareGPT-format JSONL into HuggingFace Dataset."""
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("pip install datasets")

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def apply_chat_template(example: dict, tokenizer) -> dict:
    """Convert ShareGPT conversations to the model's chat template."""
    role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
    messages = []
    for turn in example["conversations"]:
        role = role_map.get(turn["from"], turn["from"])
        # Most models don't have a "tool" role — merge into user
        if role == "tool":
            role = "user"
            content = f"[Tool output]\n{turn['value']}"
        else:
            content = turn["value"]
        messages.append({"role": role, "content": content})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@click.command()
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True),
              help="Path to training JSONL (ShareGPT format).")
@click.option("--val-dataset", default=None, type=click.Path(),
              help="Optional validation JSONL.")
@click.option("--model", "-m", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
              help="Base model to fine-tune.")
@click.option("--output", "-o", default="models/ast-editor", help="Output directory.")
@click.option("--epochs", default=3, type=int)
@click.option("--batch-size", default=2, type=int, help="Per-device train batch size.")
@click.option("--grad-accum", default=8, type=int, help="Gradient accumulation steps.")
@click.option("--lr", default=2e-4, type=float, help="Learning rate.")
@click.option("--max-seq-len", default=4096, type=int, help="Max sequence length.")
@click.option("--lora/--no-lora", default=True, help="Use LoRA (default: yes).")
@click.option("--lora-r", default=64, type=int, help="LoRA rank.")
@click.option("--lora-alpha", default=128, type=int, help="LoRA alpha.")
@click.option("--load-in-4bit/--no-4bit", default=False, help="QLoRA (4-bit quant).")
@click.option("--flash-attn/--no-flash-attn", default=False,
              help="Use Flash Attention 2 (requires flash-attn package).")
def main(dataset, val_dataset, model, output, epochs, batch_size, grad_accum,
         lr, max_seq_len, lora, lora_r, lora_alpha, load_in_4bit, flash_attn):
    """Fine-tune a model on the AST-editing dataset."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        raise SystemExit(f"Missing dependency: {e}\n  pip install trl transformers accelerate bitsandbytes")

    # Detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    click.echo(f"Device: {device}")

    click.echo(f"Loading tokenizer from {model} …")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    click.echo(f"Loading dataset from {dataset} …")
    train_ds = load_sharegpt_jsonl(dataset)
    train_ds = train_ds.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    click.echo(f"  {len(train_ds)} training examples")

    eval_ds = None
    if val_dataset and Path(val_dataset).exists():
        eval_ds = load_sharegpt_jsonl(val_dataset)
        eval_ds = eval_ds.map(
            lambda ex: apply_chat_template(ex, tokenizer),
            remove_columns=eval_ds.column_names,
        )
        click.echo(f"  {len(eval_ds)} validation examples")

    # Model
    model_kwargs: dict = {"trust_remote_code": True}
    if device == "mps":
        # MPS: no bitsandbytes, no flash-attn; use float16
        model_kwargs["dtype"] = torch.float16
        load_in_4bit = False
        flash_attn = False
    elif load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    click.echo(f"Loading model {model} …")
    base_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    if device == "mps":
        base_model = base_model.to("mps")

    # LoRA
    peft_config = None
    if lora:
        try:
            from peft import LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                # Target all linear layers — works for most architectures
                target_modules="all-linear",
                bias="none",
            )
            click.echo(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
        except ImportError:
            raise SystemExit("pip install peft")

    # Precision flags — MPS uses fp16 via model dtype, not trainer flags
    use_fp16 = (device == "cuda") and not load_in_4bit
    use_bf16 = load_in_4bit and (device == "cuda")

    # Training config
    # MPS can't handle entropy computation in chunked_nll eval — pin to nll
    loss_type = "nll"
    # Also skip eval on MPS to avoid the INT_MAX tensor dim bug in log_softmax
    eval_strategy = "no" if device == "mps" else ("epoch" if eval_ds else "no")
    load_best = eval_ds is not None and device != "mps"

    sft_config = SFTConfig(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_length=max_seq_len,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy=eval_strategy,
        load_best_model_at_end=load_best,
        report_to="none",
        dataset_text_field="text",
        dataloader_pin_memory=(device == "cuda"),
        loss_type=loss_type,
    )

    trainer = SFTTrainer(
        model=base_model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    click.echo("Starting training …")
    trainer.train()

    click.echo(f"Saving to {output} …")
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    click.echo("Done.")


if __name__ == "__main__":
    main()
