#!/usr/bin/env python3
"""
Test DroPE with RULER-style prompt format vs simple passkey.

The DroPE paper used RULER benchmark which has specific prompt templates.
Our simple passkey test may not be a fair comparison.
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


def generate_ruler_style_prompt(context_length, tokenizer, magic_number=None):
    """
    Generate RULER-style NIAH prompt.

    Template: "Some special magic numbers are hidden within the following text.
    Make sure to memorize it. I will quiz you about the numbers afterwards.
    {context}
    What is the special magic number for {key} mentioned in the provided text?"
    """
    if magic_number is None:
        magic_number = str(random.randint(1000000, 9999999))  # 7 digits like RULER

    key = "city"  # Simple key

    # Filler sentences
    filler_sentences = [
        "The grass is green. The sky is blue. The sun is yellow.",
        "Trees grow in forests. Birds fly in the sky. Fish swim in water.",
        "Books contain knowledge. Libraries store books. People read to learn.",
        "Mountains are tall. Valleys are low. Rivers flow between them.",
        "Cities have buildings. Roads connect places. Cars drive on roads.",
    ]

    # Build filler
    filler = ""
    while len(tokenizer.encode(filler)) < context_length - 150:
        filler += random.choice(filler_sentences) + " "

    # Insert needle at ~50% position
    filler_parts = filler.split(". ")
    insert_pos = len(filler_parts) // 2
    needle = f"The special magic number for {key} is: {magic_number}"
    filler_parts.insert(insert_pos, needle)
    context = ". ".join(filler_parts)

    # Trim to target length
    tokens = tokenizer.encode(context)
    if len(tokens) > context_length - 100:
        tokens = tokens[:context_length - 100]
        context = tokenizer.decode(tokens, skip_special_tokens=True)

    prompt = f"""Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.

{context}

What is the special magic number for {key} mentioned in the provided text? The special magic number for {key} is:"""

    return prompt, magic_number


def generate_simple_passkey_prompt(context_length, tokenizer, passkey=None):
    """Our simple passkey format."""
    if passkey is None:
        passkey = str(random.randint(10000, 99999))

    filler_sentences = [
        "The grass is green. The sky is blue. The sun is yellow.",
        "Trees grow in forests. Birds fly in the sky. Fish swim in water.",
        "Books contain knowledge. Libraries store books. People read to learn.",
        "Mountains are tall. Valleys are low. Rivers flow between them.",
        "Cities have buildings. Roads connect places. Cars drive on roads.",
    ]

    filler = ""
    while len(tokenizer.encode(filler)) < context_length - 100:
        filler += random.choice(filler_sentences) + " "

    filler_tokens = filler.split(". ")
    insert_pos = len(filler_tokens) // 2
    passkey_sentence = f"The secret passkey is {passkey}"
    filler_tokens.insert(insert_pos, passkey_sentence)
    context = ". ".join(filler_tokens)

    tokens = tokenizer.encode(context)
    if len(tokens) > context_length - 50:
        tokens = tokens[:context_length - 50]
        context = tokenizer.decode(tokens, skip_special_tokens=True)

    prompt = f"{context}\n\nWhat is the secret passkey mentioned above? The passkey is"

    return prompt, passkey


def test_format(model, tokenizer, prompt_fn, format_name, context_length, num_trials=10):
    """Test a specific prompt format."""
    correct = 0

    for i in range(num_trials):
        prompt, target = prompt_fn(context_length, tokenizer)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length + 200)
        input_ids = inputs["input_ids"].to(model.device)

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
            result = "✓"
        else:
            result = "✗"

        print(f"  Trial {i+1}: {result} (target: {target}, got: {generated[:30]}...)")

    accuracy = correct / num_trials
    print(f"  {format_name}: {accuracy*100:.0f}% ({correct}/{num_trials})")
    return accuracy


def main():
    print("=" * 60)
    print("Testing RULER-style vs Simple Passkey Format")
    print("=" * 60)

    # Load WITHOUT quantization
    print("\nLoading models WITHOUT quantization...")

    context_lengths = [512, 1024, 2048]

    all_results = {}

    for model_name, model_key in [(ROPE_MODEL, "RoPE"), (DROPE_MODEL, "DroPE")]:
        print(f"\n{'='*60}")
        print(f"Testing {model_key}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        all_results[model_key] = {"ruler": {}, "simple": {}}

        for context_length in context_lengths:
            print(f"\n--- Context length: {context_length} ---")

            print("\n1. RULER-style format:")
            ruler_acc = test_format(model, tokenizer, generate_ruler_style_prompt,
                                    "RULER", context_length, num_trials=5)
            all_results[model_key]["ruler"][context_length] = ruler_acc

            print("\n2. Simple passkey format:")
            simple_acc = test_format(model, tokenizer, generate_simple_passkey_prompt,
                                     "Simple", context_length, num_trials=5)
            all_results[model_key]["simple"][context_length] = simple_acc

        print(f"\n{model_key} Summary:")
        for ctx in context_lengths:
            print(f"  {ctx}: RULER {all_results[model_key]['ruler'][ctx]*100:.0f}%, Simple {all_results[model_key]['simple'][ctx]*100:.0f}%")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    print(f"\n{'Context':<10} {'RoPE RULER':<12} {'RoPE Simple':<12} {'DroPE RULER':<12} {'DroPE Simple':<12}")
    print("-" * 58)
    for ctx in context_lengths:
        rope_ruler = all_results["RoPE"]["ruler"].get(ctx, 0) * 100
        rope_simple = all_results["RoPE"]["simple"].get(ctx, 0) * 100
        drope_ruler = all_results["DroPE"]["ruler"].get(ctx, 0) * 100
        drope_simple = all_results["DroPE"]["simple"].get(ctx, 0) * 100
        print(f"{ctx:<10} {rope_ruler:<12.0f} {rope_simple:<12.0f} {drope_ruler:<12.0f} {drope_simple:<12.0f}")


if __name__ == "__main__":
    main()
