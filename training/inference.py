"""Quick inference test for the trained AST-editor model."""
from __future__ import annotations
import json
import sys
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

from training.system_prompt import SYSTEM_PROMPT


def load_model(adapter_path: str, base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading on {device}…", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()
    return model, tokenizer, device


def ask(model, tokenizer, device: str, user_message: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_demo(adapter_path: str):
    model, tokenizer, device = load_model(adapter_path)

    # Build a mini AST from sample.py for the demo
    from src.parsers import parse
    from src.harness.skeleton import build_skeleton

    source = Path("tests/fixtures/sample.py").read_text()
    module = parse(source, "python", "sample.py")
    skeleton = json.dumps(build_skeleton(module), indent=2)

    prompts = [
        # 1. Simple rename
        "Rename the function `greet` to `say_hello`. The AST is at `code.json`.",
        # 2. Type change
        "Change the return type of `fibonacci` to `optional<number>`. The AST is at `code.json`.",
        # 3. Cross-compile
        "Compile the Python codebase to TypeScript. The AST is at `code.json`.",
    ]

    print("\n" + "="*60)
    print("Skeleton shown to model:")
    print("="*60)
    # The model would first call skeleton — simulate that tool output in context
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"[TEST {i}] {prompt}")
        print("="*60)
        # Inject skeleton as if the model already called it (to save time)
        user_with_ctx = (
            f"{prompt}\n\n"
            f"[Skeleton already retrieved]\n```json\n{skeleton[:1500]}\n```"
        )
        response = ask(model, tokenizer, device, user_with_ctx, max_new_tokens=300)
        print(response)


if __name__ == "__main__":
    adapter = sys.argv[1] if len(sys.argv) > 1 else "models/ast-editor-1.5b"
    run_demo(adapter)
