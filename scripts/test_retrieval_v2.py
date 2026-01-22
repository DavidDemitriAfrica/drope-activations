#!/usr/bin/env python3
"""
Test v2: Better test for retrieval vs copying.

Key insight from v1: MCQ has position bias (RoPE→A, DroPE→D).
Solution: Look at probability of correct answer regardless of which letter it is.

Also test: Can the model identify the correct number when we give it the exact one?
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


def generate_near_miss(passkey):
    """Generate a near-miss: same length, 1-2 digits different."""
    passkey_list = list(passkey)
    num_changes = random.randint(1, 2)
    positions = random.sample(range(len(passkey_list)), num_changes)
    for pos in positions:
        old_digit = passkey_list[pos]
        new_digit = random.choice([d for d in '0123456789' if d != old_digit])
        passkey_list[pos] = new_digit
    return ''.join(passkey_list)


def build_context(context_length, tokenizer, passkey):
    """Build context with hidden passkey."""
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

    return context


def test_verification(model, tokenizer, context_length, num_trials=20):
    """
    Test: "Is X the magic number?" for exact and near-misses.
    Measure P(Yes) for exact vs P(Yes) for near-miss.

    If retrieval works: P(Yes|exact) >> P(Yes|near-miss)
    If retrieval fails: P(Yes|exact) ≈ P(Yes|near-miss)
    """
    yes_token = tokenizer.encode('Yes', add_special_tokens=False)[0]
    no_token = tokenizer.encode('No', add_special_tokens=False)[0]

    exact_probs = []
    nearmiss_probs = []

    for i in range(num_trials):
        passkey = str(random.randint(1000000, 9999999))
        context = build_context(context_length, tokenizer, passkey)
        near_miss = generate_near_miss(passkey)

        # Test exact match
        prompt_exact = f"""Remember the magic number in the text below.

{context}

Is {passkey} the magic number from the text? Answer Yes or No:"""

        inputs = tokenizer(prompt_exact, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(torch.tensor([logits[yes_token].item(), logits[no_token].item()]), dim=0)
            p_yes_exact = probs[0].item()
            exact_probs.append(p_yes_exact)

        # Test near-miss
        prompt_near = f"""Remember the magic number in the text below.

{context}

Is {near_miss} the magic number from the text? Answer Yes or No:"""

        inputs = tokenizer(prompt_near, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(torch.tensor([logits[yes_token].item(), logits[no_token].item()]), dim=0)
            p_yes_near = probs[0].item()
            nearmiss_probs.append(p_yes_near)

        print(f"  Trial {i+1}: P(Yes|exact)={p_yes_exact:.2f}, P(Yes|near)={p_yes_near:.2f}, Δ={p_yes_exact-p_yes_near:+.2f}")

    avg_exact = sum(exact_probs) / len(exact_probs)
    avg_near = sum(nearmiss_probs) / len(nearmiss_probs)
    discrimination = avg_exact - avg_near

    return avg_exact, avg_near, discrimination


def test_ranking(model, tokenizer, context_length, num_trials=20):
    """
    Test: Given exact + 3 near-misses, does the model assign highest P to exact?

    This avoids position bias by looking at probabilities, not generation.
    """
    results = []

    for i in range(num_trials):
        passkey = str(random.randint(1000000, 9999999))
        context = build_context(context_length, tokenizer, passkey)

        # Generate 3 near-misses
        candidates = [passkey]
        for _ in range(3):
            nm = generate_near_miss(passkey)
            while nm in candidates:
                nm = generate_near_miss(passkey)
            candidates.append(nm)

        # For each candidate, get P(Yes) for "Is X the number?"
        probs = {}
        yes_token = tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token = tokenizer.encode('No', add_special_tokens=False)[0]

        for cand in candidates:
            prompt = f"""Remember the magic number in the text below.

{context}

Is {cand} the magic number from the text? Answer Yes or No:"""

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]
                p = torch.softmax(torch.tensor([logits[yes_token].item(), logits[no_token].item()]), dim=0)
                probs[cand] = p[0].item()

        # Check if exact has highest probability
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        correct_rank = [c for c, _ in ranked].index(passkey) + 1
        is_top = (correct_rank == 1)
        results.append(is_top)

        status = "✓" if is_top else "✗"
        print(f"  Trial {i+1}: {status} Correct rank={correct_rank}, P(exact)={probs[passkey]:.2f}, P(best near)={ranked[0][1] if ranked[0][0] != passkey else ranked[1][1]:.2f}")

    accuracy = sum(results) / len(results)
    return accuracy


def main():
    print("=" * 70)
    print("Test v2: Retrieval Analysis (Probability-Based)")
    print("=" * 70)

    context_lengths = [2048, 4096, 8192]

    for model_name, model_key in [(ROPE_MODEL, "RoPE"), (DROPE_MODEL, "DroPE")]:
        print(f"\n{'='*70}")
        print(f"Testing {model_key}")
        print(f"{'='*70}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        for ctx_len in context_lengths:
            print(f"\n--- Context length: {ctx_len} ({ctx_len/4096:.1f}× training) ---")

            print("\n1. Verification (P(Yes) for exact vs near-miss):")
            avg_exact, avg_near, discrim = test_verification(model, tokenizer, ctx_len, num_trials=10)
            print(f"   Avg P(Yes|exact)={avg_exact:.2f}, P(Yes|near)={avg_near:.2f}")
            print(f"   Discrimination (Δ) = {discrim:+.2f}")
            if discrim > 0.1:
                print(f"   → Model CAN distinguish exact from near-miss")
            else:
                print(f"   → Model CANNOT distinguish (retrieval failure)")

            print("\n2. Ranking (Is exact ranked #1 among 4 candidates?):")
            rank_acc = test_ranking(model, tokenizer, ctx_len, num_trials=10)
            print(f"   Ranking accuracy: {rank_acc*100:.0f}%")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If Discrimination > 0.1 and Ranking > 50%:
  → RETRIEVAL WORKS - model finds the needle
  → Copying failures are in the generation/decoding phase

If Discrimination ≈ 0 and Ranking ≈ 25% (chance):
  → RETRIEVAL FAILS - model can't locate the information
""")


if __name__ == "__main__":
    main()
