#!/usr/bin/env python3
"""
Test passkey retrieval at EXTENDED context lengths (beyond training window).

Llama-2 was trained at 4096 tokens. DroPE paper tested at 2× (8192).
This should show RoPE failing and DroPE working.
"""

import sys
from pathlib import Path
import torch
import random

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

ROPE_MODEL = "meta-llama/Llama-2-7b-hf"
DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"


def generate_passkey_prompt(context_length, tokenizer, passkey=None):
    """Generate RULER-style prompt at specified context length."""
    if passkey is None:
        passkey = str(random.randint(1000000, 9999999))  # 7 digits like RULER

    key = "city"

    # Paul Graham essay style filler (like the paper uses)
    filler_sentences = [
        "The most important thing is to be working on something you're interested in.",
        "It's hard to do a really good job on something you don't like.",
        "The way to get startup ideas is not to try to think of startup ideas.",
        "The best way to come up with startup ideas is to ask what you wish someone would build for you.",
        "If you want to start a startup, you have to force yourself to think of ideas.",
        "The very best ideas tend to have three things in common.",
        "They're something the founders themselves want.",
        "That they themselves can build and that few others realize are worth doing.",
        "When you find an idea you can imagine yourself working on for years, you've found something valuable.",
        "The initial idea is just a starting point, not a blueprint.",
    ]

    # Build filler to target length
    filler = ""
    while len(tokenizer.encode(filler)) < context_length - 200:
        filler += random.choice(filler_sentences) + " "

    # Insert needle at ~50% depth
    filler_parts = filler.split(". ")
    insert_pos = len(filler_parts) // 2
    needle = f"The special magic number for {key} is: {passkey}"
    filler_parts.insert(insert_pos, needle)
    context = ". ".join(filler_parts)

    # Trim to target
    tokens = tokenizer.encode(context)
    if len(tokens) > context_length - 150:
        tokens = tokens[:context_length - 150]
        context = tokenizer.decode(tokens, skip_special_tokens=True)

    prompt = f"""Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.

{context}

What is the special magic number for {key} mentioned in the provided text? The special magic number for {key} is:"""

    return prompt, passkey


def test_retrieval(model, tokenizer, context_length, num_trials=10):
    """Test passkey retrieval at given context length."""
    correct = 0

    for i in range(num_trials):
        prompt, target = generate_passkey_prompt(context_length, tokenizer)

        # Don't truncate - we want to test at full length
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        actual_len = input_ids.shape[1]

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=15,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

            if target in generated:
                correct += 1
                status = "✓"
            else:
                status = "✗"

            print(f"  Trial {i+1}: {status} (len={actual_len}, target={target}, got={generated[:20]}...)")

        except Exception as e:
            print(f"  Trial {i+1}: ERROR - {e}")

    accuracy = correct / num_trials
    return accuracy


def main():
    print("=" * 70)
    print("Testing at EXTENDED Context Lengths (Beyond Training Window)")
    print("Llama-2 trained at 4096. Testing at 4096, 6144, 8192.")
    print("=" * 70)

    # Context lengths: within training, 1.5×, 2×
    context_lengths = [4096, 6144, 8192]

    results = {}

    for model_name, model_key in [(ROPE_MODEL, "RoPE"), (DROPE_MODEL, "DroPE")]:
        print(f"\n{'='*70}")
        print(f"Testing {model_key}")
        print(f"{'='*70}")

        # Load WITHOUT rope_scaling to match paper's baseline
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        print(f"Max position embeddings: {model.config.max_position_embeddings}")

        results[model_key] = {}

        for ctx_len in context_lengths:
            print(f"\n--- Context length: {ctx_len} ({ctx_len/4096:.1f}× training) ---")

            acc = test_retrieval(model, tokenizer, ctx_len, num_trials=5)
            results[model_key][ctx_len] = acc
            print(f"  Accuracy: {acc*100:.0f}%")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("COMPARISON: Extended Context Retrieval")
    print("=" * 70)

    print(f"\n{'Context':<12} {'Multiple':<10} {'RoPE':<12} {'DroPE':<12}")
    print("-" * 46)
    for ctx in context_lengths:
        mult = f"{ctx/4096:.1f}×"
        rope = f"{results['RoPE'].get(ctx, 0)*100:.0f}%"
        drope = f"{results['DroPE'].get(ctx, 0)*100:.0f}%"
        print(f"{ctx:<12} {mult:<10} {rope:<12} {drope:<12}")


if __name__ == "__main__":
    main()
