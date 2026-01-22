#!/usr/bin/env python3
"""
Analyze near-misses in long context passkey retrieval.

Categories:
1. Exact match - correct
2. Near-miss - same length, 1-2 digits different
3. Length error - wrong number of digits
4. Complete failure - gibberish, no number found
"""

import sys
from pathlib import Path
import torch
import random
import re

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

ROPE_MODEL = "meta-llama/Llama-2-7b-hf"
DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"


def classify_response(target, response):
    """
    Classify the response into categories.

    Returns: (category, extracted_number, edit_distance)
    """
    # Extract all digit sequences from response
    numbers = re.findall(r'\d+', response)

    if not numbers:
        return "no_number", None, None

    # Find the number closest to target length
    target_len = len(target)
    best_match = None
    best_diff = float('inf')

    for num in numbers:
        diff = abs(len(num) - target_len)
        if diff < best_diff:
            best_diff = diff
            best_match = num

    if best_match is None:
        return "no_number", None, None

    # Calculate edit distance (digit-level)
    def digit_edit_distance(s1, s2):
        """Simple edit distance for digit strings."""
        if len(s1) != len(s2):
            return max(len(s1), len(s2))  # Length difference as baseline
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    edit_dist = digit_edit_distance(target, best_match)

    # Classify
    if best_match == target:
        return "exact", best_match, 0
    elif len(best_match) == len(target) and edit_dist <= 2:
        return "near_miss", best_match, edit_dist
    elif len(best_match) == len(target):
        return "wrong_digits", best_match, edit_dist
    elif abs(len(best_match) - len(target)) == 1:
        return "off_by_one_length", best_match, edit_dist
    else:
        return "wrong_length", best_match, edit_dist


def generate_passkey_prompt(context_length, tokenizer, passkey=None):
    """Generate RULER-style prompt."""
    if passkey is None:
        passkey = str(random.randint(1000000, 9999999))  # 7 digits

    filler_sentences = [
        "The grass is green. The sky is blue. The sun is yellow.",
        "Trees grow in forests. Birds fly in the sky. Fish swim in water.",
        "Books contain knowledge. Libraries store books. People read to learn.",
        "Mountains are tall. Valleys are low. Rivers flow between them.",
        "Cities have buildings. Roads connect places. Cars drive on roads.",
    ]

    filler = ""
    while len(tokenizer.encode(filler)) < context_length - 200:
        filler += random.choice(filler_sentences) + " "

    filler_parts = filler.split(". ")
    insert_pos = len(filler_parts) // 2
    needle = f"The special magic number is: {passkey}"
    filler_parts.insert(insert_pos, needle)
    context = ". ".join(filler_parts)

    tokens = tokenizer.encode(context)
    if len(tokens) > context_length - 150:
        tokens = tokens[:context_length - 150]
        context = tokenizer.decode(tokens, skip_special_tokens=True)

    prompt = f"""Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.

{context}

What is the special magic number mentioned in the provided text? The special magic number is:"""

    return prompt, passkey


def test_and_classify(model, tokenizer, context_length, num_trials=20):
    """Run trials and classify each response."""
    results = {
        "exact": 0,
        "near_miss": 0,
        "wrong_digits": 0,
        "off_by_one_length": 0,
        "wrong_length": 0,
        "no_number": 0,
    }

    details = []

    for i in range(num_trials):
        passkey = str(random.randint(1000000, 9999999))
        prompt, _ = generate_passkey_prompt(context_length, tokenizer, passkey)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        category, extracted, edit_dist = classify_response(passkey, generated)
        results[category] += 1

        # Status symbol
        if category == "exact":
            status = "✓"
        elif category == "near_miss":
            status = "≈"
        else:
            status = "✗"

        details.append({
            "target": passkey,
            "extracted": extracted,
            "category": category,
            "edit_distance": edit_dist,
            "raw": generated[:40]
        })

        print(f"  {i+1:2d}. {status} {category:<18} target={passkey} got={str(extracted):<10} edit={edit_dist} raw='{generated[:30]}...'")

    return results, details


def main():
    print("=" * 80)
    print("Near-Miss Analysis: Categorizing Long Context Retrieval Errors")
    print("=" * 80)

    context_lengths = [2048, 4096, 8192]

    all_results = {}

    for model_name, model_key in [(ROPE_MODEL, "RoPE"), (DROPE_MODEL, "DroPE")]:
        print(f"\n{'='*80}")
        print(f"Testing {model_key}")
        print(f"{'='*80}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        all_results[model_key] = {}

        for ctx_len in context_lengths:
            print(f"\n--- Context length: {ctx_len} ({ctx_len/4096:.1f}× training) ---\n")

            results, details = test_and_classify(model, tokenizer, ctx_len, num_trials=20)
            all_results[model_key][ctx_len] = results

            print(f"\n  Summary for {ctx_len}:")
            total = sum(results.values())
            for cat, count in sorted(results.items(), key=lambda x: -x[1]):
                pct = count / total * 100
                bar = "█" * int(pct / 5)
                print(f"    {cat:<20} {count:2d} ({pct:5.1f}%) {bar}")

        del model
        torch.cuda.empty_cache()

    # Final comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: Error Categories by Model and Context Length")
    print("=" * 80)

    categories = ["exact", "near_miss", "wrong_digits", "off_by_one_length", "wrong_length", "no_number"]

    for ctx_len in context_lengths:
        print(f"\n--- {ctx_len} tokens ({ctx_len/4096:.1f}× training) ---")
        print(f"{'Category':<20} {'RoPE':<15} {'DroPE':<15}")
        print("-" * 50)

        for cat in categories:
            rope_count = all_results["RoPE"][ctx_len].get(cat, 0)
            drope_count = all_results["DroPE"][ctx_len].get(cat, 0)
            rope_pct = rope_count / 20 * 100
            drope_pct = drope_count / 20 * 100
            print(f"{cat:<20} {rope_count:2d} ({rope_pct:5.1f}%)    {drope_count:2d} ({drope_pct:5.1f}%)")

    # Key insight
    print("\n" + "=" * 80)
    print("KEY METRIC: Near-Miss Rate (of non-exact responses)")
    print("=" * 80)

    for ctx_len in context_lengths:
        for model_key in ["RoPE", "DroPE"]:
            results = all_results[model_key][ctx_len]
            exact = results.get("exact", 0)
            near_miss = results.get("near_miss", 0)
            total_errors = 20 - exact

            if total_errors > 0:
                near_miss_rate = near_miss / total_errors * 100
                print(f"{model_key} @ {ctx_len}: {near_miss}/{total_errors} errors are near-misses ({near_miss_rate:.0f}%)")
            else:
                print(f"{model_key} @ {ctx_len}: No errors (100% exact)")


if __name__ == "__main__":
    main()
