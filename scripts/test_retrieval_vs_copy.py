#!/usr/bin/env python3
"""
Test: Is DroPE's failure in RETRIEVAL or COPYING?

Hypothesis: DroPE attends correctly but fails at precise token copying.

Tests:
1. MCQ - "Which is the passkey? A) ... B) ... C) ... D) ..."
   → If DroPE gets this right, retrieval works, copying fails
2. Yes/No - "Is the passkey X?"
   → Even simpler verification

Near-misses as distractors: same length, off by 1-2 digits
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


def generate_near_miss(passkey):
    """Generate a near-miss: same length, 1-2 digits different."""
    passkey_list = list(passkey)
    # Change 1-2 random positions
    num_changes = random.randint(1, 2)
    positions = random.sample(range(len(passkey_list)), num_changes)
    for pos in positions:
        # Change to a different digit
        old_digit = passkey_list[pos]
        new_digit = random.choice([d for d in '0123456789' if d != old_digit])
        passkey_list[pos] = new_digit
    return ''.join(passkey_list)


def generate_mcq_prompt(context_length, tokenizer, passkey=None):
    """
    Generate MCQ format: hide passkey, then ask which option is correct.
    Distractors are near-misses (same length, 1-2 digits different).
    """
    if passkey is None:
        passkey = str(random.randint(1000000, 9999999))  # 7 digits

    # Generate 3 near-miss distractors
    distractors = []
    for _ in range(3):
        d = generate_near_miss(passkey)
        while d in distractors or d == passkey:
            d = generate_near_miss(passkey)
        distractors.append(d)

    # Shuffle options
    options = [passkey] + distractors
    random.shuffle(options)
    correct_letter = ['A', 'B', 'C', 'D'][options.index(passkey)]

    # Build filler
    filler_sentences = [
        "The grass is green. The sky is blue. The sun is yellow.",
        "Trees grow in forests. Birds fly in the sky. Fish swim in water.",
        "Books contain knowledge. Libraries store books. People read to learn.",
        "Mountains are tall. Valleys are low. Rivers flow between them.",
        "Cities have buildings. Roads connect places. Cars drive on roads.",
    ]

    filler = ""
    while len(tokenizer.encode(filler)) < context_length - 300:
        filler += random.choice(filler_sentences) + " "

    # Insert needle at ~50%
    filler_parts = filler.split(". ")
    insert_pos = len(filler_parts) // 2
    needle = f"The special magic number is: {passkey}"
    filler_parts.insert(insert_pos, needle)
    context = ". ".join(filler_parts)

    # Trim
    tokens = tokenizer.encode(context)
    if len(tokens) > context_length - 250:
        tokens = tokens[:context_length - 250]
        context = tokenizer.decode(tokens, skip_special_tokens=True)

    prompt = f"""Remember the magic number hidden in the following text.

{context}

Which of the following is the magic number from the text above?
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Answer with the letter only (A, B, C, or D):"""

    return prompt, correct_letter, passkey, options


def generate_yesno_prompt(context_length, tokenizer, passkey=None, is_correct=True):
    """
    Generate Yes/No format: "Is the passkey X?"
    """
    if passkey is None:
        passkey = str(random.randint(1000000, 9999999))

    # Build filler
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

    # Query with correct or near-miss
    if is_correct:
        query_number = passkey
        expected = "Yes"
    else:
        query_number = generate_near_miss(passkey)
        expected = "No"

    prompt = f"""Remember the magic number hidden in the following text.

{context}

Is {query_number} the magic number from the text above? Answer Yes or No only:"""

    return prompt, expected, passkey, query_number


def test_mcq(model, tokenizer, context_length, num_trials=10):
    """Test MCQ format with logit analysis."""
    correct = 0
    logit_diffs = []

    # Get token IDs for A, B, C, D
    letter_tokens = {
        'A': tokenizer.encode('A', add_special_tokens=False)[0],
        'B': tokenizer.encode('B', add_special_tokens=False)[0],
        'C': tokenizer.encode('C', add_special_tokens=False)[0],
        'D': tokenizer.encode('D', add_special_tokens=False)[0],
    }

    for i in range(num_trials):
        prompt, correct_letter, passkey, options = generate_mcq_prompt(context_length, tokenizer)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Get logits for A, B, C, D
            letter_logits = {k: logits[v].item() for k, v in letter_tokens.items()}

            # Softmax for probabilities
            letter_logit_tensor = torch.tensor([letter_logits['A'], letter_logits['B'],
                                                 letter_logits['C'], letter_logits['D']])
            probs = torch.softmax(letter_logit_tensor, dim=0)
            letter_probs = {'A': probs[0].item(), 'B': probs[1].item(),
                           'C': probs[2].item(), 'D': probs[3].item()}

            # Find predicted and logit diff
            predicted_letter = max(letter_logits, key=letter_logits.get)
            sorted_logits = sorted(letter_logits.values(), reverse=True)
            logit_diff = sorted_logits[0] - sorted_logits[1]  # Top - second
            logit_diffs.append(logit_diff)

        if predicted_letter == correct_letter:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        prob_str = f"P({correct_letter})={letter_probs[correct_letter]:.2f}"
        print(f"  Trial {i+1}: {status} correct={correct_letter}, pred={predicted_letter}, {prob_str}, Δlogit={logit_diff:.1f}")

    accuracy = correct / num_trials
    avg_logit_diff = sum(logit_diffs) / len(logit_diffs)
    return accuracy, avg_logit_diff


def test_yesno(model, tokenizer, context_length, num_trials=10):
    """Test Yes/No format with logit analysis."""
    correct = 0
    logit_diffs = []

    # Get token IDs for Yes/No
    yes_token = tokenizer.encode('Yes', add_special_tokens=False)[0]
    no_token = tokenizer.encode('No', add_special_tokens=False)[0]

    for i in range(num_trials):
        is_correct_query = (i % 2 == 0)  # Alternate
        prompt, expected, passkey, query = generate_yesno_prompt(
            context_length, tokenizer, is_correct=is_correct_query
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]

            yes_logit = logits[yes_token].item()
            no_logit = logits[no_token].item()

            # Softmax
            probs = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
            yes_prob, no_prob = probs[0].item(), probs[1].item()

            predicted = "Yes" if yes_logit > no_logit else "No"
            logit_diff = abs(yes_logit - no_logit)
            logit_diffs.append(logit_diff)

        if predicted == expected:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        match_type = "exact" if is_correct_query else "near-miss"
        print(f"  Trial {i+1}: {status} ({match_type}) expect={expected}, pred={predicted}, P(Yes)={yes_prob:.2f}, Δ={logit_diff:.1f}")

    accuracy = correct / num_trials
    avg_logit_diff = sum(logit_diffs) / len(logit_diffs)
    return accuracy, avg_logit_diff


def main():
    print("=" * 70)
    print("Testing: Is DroPE's failure in RETRIEVAL or COPYING?")
    print("=" * 70)
    print("\nIf MCQ accuracy >> Copy accuracy → Retrieval works, copying fails")
    print("If MCQ accuracy ≈ Copy accuracy → Retrieval itself fails\n")

    context_lengths = [2048, 4096, 8192]

    results = {}

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

        results[model_key] = {"mcq": {}, "yesno": {}}

        for ctx_len in context_lengths:
            print(f"\n--- Context length: {ctx_len} ({ctx_len/4096:.1f}× training) ---")

            print("\n1. MCQ (no copying required):")
            mcq_acc, mcq_logit = test_mcq(model, tokenizer, ctx_len, num_trials=10)
            results[model_key]["mcq"][ctx_len] = (mcq_acc, mcq_logit)
            print(f"  MCQ Accuracy: {mcq_acc*100:.0f}%, Avg Δlogit: {mcq_logit:.1f}")

            print("\n2. Yes/No (verification only):")
            yesno_acc, yesno_logit = test_yesno(model, tokenizer, ctx_len, num_trials=10)
            results[model_key]["yesno"][ctx_len] = (yesno_acc, yesno_logit)
            print(f"  Yes/No Accuracy: {yesno_acc*100:.0f}%, Avg Δlogit: {yesno_logit:.1f}")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("RESULTS: Retrieval vs Copying")
    print("=" * 70)

    print(f"\n{'Context':<10} {'RoPE MCQ':<15} {'RoPE Y/N':<15} {'DroPE MCQ':<15} {'DroPE Y/N':<15}")
    print("-" * 70)
    for ctx in context_lengths:
        rope_mcq_acc, rope_mcq_logit = results["RoPE"]["mcq"].get(ctx, (0, 0))
        rope_yn_acc, rope_yn_logit = results["RoPE"]["yesno"].get(ctx, (0, 0))
        drope_mcq_acc, drope_mcq_logit = results["DroPE"]["mcq"].get(ctx, (0, 0))
        drope_yn_acc, drope_yn_logit = results["DroPE"]["yesno"].get(ctx, (0, 0))
        print(f"{ctx:<10} {rope_mcq_acc*100:.0f}% (Δ{rope_mcq_logit:.1f})   {rope_yn_acc*100:.0f}% (Δ{rope_yn_logit:.1f})   {drope_mcq_acc*100:.0f}% (Δ{drope_mcq_logit:.1f})   {drope_yn_acc*100:.0f}% (Δ{drope_yn_logit:.1f})")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If DroPE MCQ/Yes-No >> DroPE Copy (from earlier tests):
  → DroPE RETRIEVAL works, COPYING fails
  → The 100× Q/K amplification finds the needle
  → But without position embeddings, precise token-by-token decoding fails

If DroPE MCQ/Yes-No ≈ DroPE Copy:
  → DroPE RETRIEVAL itself fails
  → The attention mechanism can't locate the information
""")


if __name__ == "__main__":
    main()
