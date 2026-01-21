#!/usr/bin/env python3
"""
Experiment 10: Context Length Scaling

The whole point of DroPE is longer contexts. Does the Layer 0-1 restructuring
actually help at longer context lengths?

We test passkey retrieval at various context lengths:
- 512, 1K, 2K, 4K, 8K tokens

Hypothesis: DroPE may underperform at short contexts (60% vs 100% at 512)
but catch up or surpass RoPE at longer contexts.
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np
from tqdm import tqdm
import random

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import custom_models
import custom_models.attention
import custom_models.drope

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROPE_MODEL = "meta-llama/Llama-2-7b-hf"
DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"
RESULTS_FILE = REPO_ROOT / "results" / "phase_metrics" / "rope_vs_drope_phase_metrics.json"
OUTPUT_DIR = REPO_ROOT / "results" / "phase_metrics"


def generate_passkey_prompt(context_length, tokenizer, passkey=None):
    """
    Generate a passkey retrieval prompt with specified context length.

    The passkey is hidden at a random position in filler text.
    """
    if passkey is None:
        passkey = str(random.randint(10000, 99999))

    # Filler text that will be repeated
    filler_sentences = [
        "The grass is green. The sky is blue. The sun is yellow.",
        "Trees grow in forests. Birds fly in the sky. Fish swim in water.",
        "Books contain knowledge. Libraries store books. People read to learn.",
        "Mountains are tall. Valleys are low. Rivers flow between them.",
        "Cities have buildings. Roads connect places. Cars drive on roads.",
    ]

    # Build filler to approximate target length
    filler = ""
    while len(tokenizer.encode(filler)) < context_length - 100:  # Leave room for passkey + question
        filler += random.choice(filler_sentences) + " "

    # Insert passkey at random position (not too early, not too late)
    filler_tokens = filler.split(". ")
    if len(filler_tokens) > 10:
        insert_pos = random.randint(len(filler_tokens) // 4, 3 * len(filler_tokens) // 4)
        passkey_sentence = f"The secret passkey is {passkey}."
        filler_tokens.insert(insert_pos, passkey_sentence)
    else:
        passkey_sentence = f"The secret passkey is {passkey}."
        filler_tokens.insert(len(filler_tokens) // 2, passkey_sentence)

    context = ". ".join(filler_tokens)

    # Trim to exact length
    tokens = tokenizer.encode(context)
    if len(tokens) > context_length - 50:
        tokens = tokens[:context_length - 50]
        context = tokenizer.decode(tokens, skip_special_tokens=True)

    prompt = f"{context}\n\nWhat is the secret passkey mentioned above? The passkey is"

    return prompt, passkey


def evaluate_passkey(model, tokenizer, context_length, num_samples=20):
    """Evaluate passkey retrieval at a specific context length."""

    correct = 0
    total = 0

    for _ in tqdm(range(num_samples), desc=f"Passkey @ {context_length}"):
        prompt, true_passkey = generate_passkey_prompt(context_length, tokenizer)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length + 100)
        input_ids = inputs["input_ids"].to(model.device)

        actual_length = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Check if the passkey is in the generated text
        if true_passkey in generated:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, actual_length


def main():
    print("=" * 60)
    print("Experiment 10: Context Length Scaling")
    print("Does DroPE's restructuring help at longer contexts?")
    print("=" * 60)

    # Load existing results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Context lengths to test
    context_lengths = [512, 1024, 2048, 4096]

    # Check if model supports longer contexts
    # Llama-2 base is 4096, DroPE claims to extend this

    num_samples = 20  # Per context length

    for model_name, model_key in [(ROPE_MODEL, "RoPE"), (DROPE_MODEL, "DroPE")]:
        print(f"\n{'='*60}")
        print(f"Testing {model_key}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        max_length = model.config.max_position_embeddings
        print(f"Max position embeddings: {max_length}")

        if "context_scaling" not in results[model_key]:
            results[model_key]["context_scaling"] = {}

        for ctx_len in context_lengths:
            if ctx_len > max_length:
                print(f"Skipping {ctx_len} (exceeds max {max_length})")
                continue

            print(f"\n--- Testing context length: {ctx_len} ---")

            accuracy, actual_len = evaluate_passkey(model, tokenizer, ctx_len, num_samples)

            results[model_key]["context_scaling"][str(ctx_len)] = {
                "accuracy": accuracy,
                "actual_length": actual_len,
                "num_samples": num_samples,
            }

            print(f"Accuracy: {accuracy*100:.1f}% (actual length: {actual_len})")

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Passkey Accuracy by Context Length")
    print("=" * 60)

    print(f"\n{'Context':<12} {'RoPE':<15} {'DroPE':<15} {'Difference':<15}")
    print("-" * 57)

    for ctx_len in context_lengths:
        ctx_str = str(ctx_len)
        rope_acc = results["RoPE"]["context_scaling"].get(ctx_str, {}).get("accuracy", None)
        drope_acc = results["DroPE"]["context_scaling"].get(ctx_str, {}).get("accuracy", None)

        if rope_acc is not None and drope_acc is not None:
            diff = drope_acc - rope_acc
            rope_str = f"{rope_acc*100:.1f}%"
            drope_str = f"{drope_acc*100:.1f}%"
            diff_str = f"{diff*100:+.1f}%"
            print(f"{ctx_len:<12} {rope_str:<15} {drope_str:<15} {diff_str:<15}")
        elif rope_acc is not None:
            print(f"{ctx_len:<12} {rope_acc*100:.1f}%{'':10} —{'':14} —")
        elif drope_acc is not None:
            print(f"{ctx_len:<12} —{'':14} {drope_acc*100:.1f}%{'':10} —")


if __name__ == "__main__":
    main()
