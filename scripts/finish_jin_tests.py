#!/usr/bin/env python3
"""
Finish Jin et al. tests - complete AQUA, Passkey for RoPE, then all DroPE tests.
Use hardcoded RoPE results we already have.
"""

import json
import random
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

import torch
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.model_loading import load_model

# Hardcoded RoPE results from completed tests
ROPE_RESULTS = {
    'parametric': {
        'cities': {'baseline': 0.895, 'disrupted': 0.575, 'degradation': 35.8},
        'sports': {'baseline': 0.60, 'disrupted': 0.52, 'degradation': 13.3},
    },
    'contextual': {
        'imdb': {'baseline': 0.44, 'disrupted': 0.05, 'degradation': 88.6},
    }
}


def register_mean_replacement_hooks(model, num_heads: int = 32, head_dim: int = 128) -> List:
    """Register hooks to replace top-1 massive dimension per head with mean."""
    hooks = []
    layers = model.model.layers if hasattr(model, 'model') else model.layers

    for layer in layers:
        attn_layer = layer.self_attn

        def make_hook(n_heads, h_dim):
            def hook(module, input, output):
                bsz, seq_len, hidden = output.shape
                states = output.view(bsz, seq_len, n_heads, h_dim)
                norms = states.norm(dim=1)
                modified = states.clone()
                for head_idx in range(n_heads):
                    _, top_idx = torch.topk(norms[0, head_idx, :], 1)
                    dim_values = states[:, :, head_idx, top_idx[0]]
                    modified[:, :, head_idx, top_idx[0]] = dim_values.mean()
                return modified.view(bsz, seq_len, hidden)
            return hook

        q_hook = attn_layer.q_proj.register_forward_hook(make_hook(num_heads, head_dim))
        k_hook = attn_layer.k_proj.register_forward_hook(make_hook(num_heads, head_dim))
        hooks.extend([q_hook, k_hook])
    return hooks


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def run_passkey_test(model, tokenizer, n: int = 20, disrupted: bool = False) -> Dict:
    """Run passkey retrieval (contextual) test."""
    hooks = register_mean_replacement_hooks(model) if disrupted else []
    correct = 0

    try:
        desc = f"Passkey ({'disrupted' if disrupted else 'baseline'})"
        for i in tqdm(range(n), desc=desc):
            random.seed(42 + i)
            passkey = str(random.randint(10000, 99999))

            filler = "The grass is green. The sky is blue. The sun is yellow. " * 30
            insert_pos = len(filler) // 2
            text = filler[:insert_pos] + f" The secret passkey is {passkey}. Remember this number. " + filler[insert_pos:]

            prompt = f"{text}\n\nWhat is the secret passkey mentioned above? Answer with just the number:"
            response = generate_response(model, tokenizer, prompt, max_new_tokens=20)

            if passkey in response:
                correct += 1
    finally:
        for hook in hooks:
            hook.remove()

    return {'accuracy': correct / n, 'correct': correct, 'total': n}


def check_yes_no(response: str) -> int:
    """Check if response indicates yes (1) or no (0)."""
    text = response.lower().strip()
    first_word = text.split()[0] if text.split() else ""

    if first_word.startswith('yes'):
        return 1
    if first_word.startswith('no'):
        return 0

    yes_words = ['yes', 'true', 'correct', 'right']
    no_words = ['no', 'false', 'incorrect', 'wrong']
    yes_count = sum(1 for w in yes_words if w in text[:100])
    no_count = sum(1 for w in no_words if w in text[:100])

    if yes_count > no_count:
        return 1
    elif no_count > yes_count:
        return 0
    return -1


def convert_to_question(statement: str) -> str:
    """Convert statement to yes/no question."""
    if statement[0].isupper():
        statement = statement[0].lower() + statement[1:]
    parts = statement.split()
    try:
        is_index = parts.index('is')
        parts.pop(is_index)
        question = 'Is ' + ' '.join(parts)
        if question.endswith('.'):
            question = question[:-1] + '?'
        else:
            question += '?'
        return question
    except ValueError:
        return statement


def load_cities(path: str, n: int = 200) -> List[Dict]:
    """Load cities dataset."""
    import csv
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({'statement': row['statement'], 'label': int(row['label'])})

    true_samples = [d for d in data if d['label'] == 1]
    false_samples = [d for d in data if d['label'] == 0]
    random.seed(42)
    n_each = n // 2
    sampled = random.sample(true_samples, min(n_each, len(true_samples)))
    sampled += random.sample(false_samples, min(n_each, len(false_samples)))
    random.shuffle(sampled)
    return sampled


def load_qa_json(path: str, n: int = 100) -> List[Dict]:
    """Load QA JSON dataset."""
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    random.seed(42)
    return random.sample(data, min(n, len(data)))


def run_parametric_test(model, tokenizer, data: List[Dict], name: str, disrupted: bool = False) -> Dict:
    """Run parametric (yes/no factual) test."""
    hooks = register_mean_replacement_hooks(model) if disrupted else []
    correct = total = 0

    try:
        desc = f"{name} ({'disrupted' if disrupted else 'baseline'})"
        for item in tqdm(data, desc=desc):
            if 'statement' in item:
                question = convert_to_question(item['statement'])
                label = item['label']
            else:
                question = item['question']
                label = 1 if item['answer'] else 0

            prompt = f"Answer with only 'Yes' or 'No'.\n\nQuestion: {question}\nAnswer:"
            response = generate_response(model, tokenizer, prompt, max_new_tokens=20)
            predicted = check_yes_no(response)

            if predicted != -1:
                total += 1
                if predicted == label:
                    correct += 1
    finally:
        for hook in hooks:
            hook.remove()

    return {'accuracy': correct / total if total > 0 else 0, 'correct': correct, 'total': total}


def load_imdb(n: int = 100) -> List[Dict]:
    """Load IMDB sentiment dataset."""
    dataset = load_dataset('imdb', split='test')
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    return [{'text': dataset[i]['text'], 'label': dataset[i]['label']} for i in indices]


def run_imdb_test(model, tokenizer, data: List[Dict], disrupted: bool = False) -> Dict:
    """Run IMDB sentiment (contextual) test."""
    hooks = register_mean_replacement_hooks(model) if disrupted else []
    correct = 0
    total = len(data)

    try:
        desc = f"IMDB ({'disrupted' if disrupted else 'baseline'})"
        for item in tqdm(data, desc=desc):
            text = item['text'][:1500]
            prompt = f"Read this movie review and answer: Is it Positive or Negative?\n\nReview: {text}\n\nSentiment (Positive/Negative):"

            response = generate_response(model, tokenizer, prompt, max_new_tokens=20)
            response_lower = response.lower()

            has_pos = 'positive' in response_lower
            has_neg = 'negative' in response_lower

            if has_pos and not has_neg:
                predicted = 1
            elif has_neg and not has_pos:
                predicted = 0
            else:
                pos_idx = response_lower.find('positive')
                neg_idx = response_lower.find('negative')
                if pos_idx >= 0 and (neg_idx < 0 or pos_idx < neg_idx):
                    predicted = 1
                elif neg_idx >= 0:
                    predicted = 0
                else:
                    predicted = -1

            if predicted == item['label']:
                correct += 1
    finally:
        for hook in hooks:
            hook.remove()

    return {'accuracy': correct / total if total > 0 else 0, 'correct': correct, 'total': total}


def run_drope_tests(model, tokenizer) -> Dict:
    """Run all tests for DroPE model."""
    results = {'parametric': {}, 'contextual': {}}

    # ===== PARAMETRIC TESTS =====
    print("\n" + "="*60)
    print("PARAMETRIC TESTS - DroPE")
    print("="*60)

    # Cities
    cities_data = load_cities("Rope_with_LLM/datasets/cities.csv", n=200)
    print(f"\nCities: {len(cities_data)} samples")
    baseline = run_parametric_test(model, tokenizer, cities_data, "Cities", disrupted=False)
    disrupted = run_parametric_test(model, tokenizer, cities_data, "Cities", disrupted=True)
    deg = (baseline['accuracy'] - disrupted['accuracy']) / baseline['accuracy'] * 100 if baseline['accuracy'] > 0 else 0
    results['parametric']['cities'] = {'baseline': baseline['accuracy'], 'disrupted': disrupted['accuracy'], 'degradation': deg}
    print(f"Cities: {baseline['accuracy']:.1%} → {disrupted['accuracy']:.1%} ({deg:.1f}% degradation)")

    # Sports
    sports_data = load_qa_json("Rope_with_LLM/datasets/sports_qa.json", n=100)
    print(f"\nSports: {len(sports_data)} samples")
    baseline = run_parametric_test(model, tokenizer, sports_data, "Sports", disrupted=False)
    disrupted = run_parametric_test(model, tokenizer, sports_data, "Sports", disrupted=True)
    deg = (baseline['accuracy'] - disrupted['accuracy']) / baseline['accuracy'] * 100 if baseline['accuracy'] > 0 else 0
    results['parametric']['sports'] = {'baseline': baseline['accuracy'], 'disrupted': disrupted['accuracy'], 'degradation': deg}
    print(f"Sports: {baseline['accuracy']:.1%} → {disrupted['accuracy']:.1%} ({deg:.1f}% degradation)")

    # ===== CONTEXTUAL TESTS =====
    print("\n" + "="*60)
    print("CONTEXTUAL TESTS - DroPE")
    print("="*60)

    # IMDB
    imdb_data = load_imdb(n=100)
    print(f"\nIMDB: {len(imdb_data)} samples")
    baseline = run_imdb_test(model, tokenizer, imdb_data, disrupted=False)
    disrupted = run_imdb_test(model, tokenizer, imdb_data, disrupted=True)
    deg = (baseline['accuracy'] - disrupted['accuracy']) / baseline['accuracy'] * 100 if baseline['accuracy'] > 0 else 0
    results['contextual']['imdb'] = {'baseline': baseline['accuracy'], 'disrupted': disrupted['accuracy'], 'degradation': deg}
    print(f"IMDB: {baseline['accuracy']:.1%} → {disrupted['accuracy']:.1%} ({deg:.1f}% degradation)")

    # Passkey
    print(f"\nPasskey: 20 samples")
    baseline = run_passkey_test(model, tokenizer, n=20, disrupted=False)
    disrupted = run_passkey_test(model, tokenizer, n=20, disrupted=True)
    deg = (baseline['accuracy'] - disrupted['accuracy']) / baseline['accuracy'] * 100 if baseline['accuracy'] > 0 else 0
    results['contextual']['passkey'] = {'baseline': baseline['accuracy'], 'disrupted': disrupted['accuracy'], 'degradation': deg}
    print(f"Passkey: {baseline['accuracy']:.1%} → {disrupted['accuracy']:.1%} ({deg:.1f}% degradation)")

    return results


def main():
    output_dir = Path("results/knowledge_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {'RoPE': ROPE_RESULTS}

    # Complete RoPE contextual tests - just passkey since we have IMDB
    print("\n" + "="*70)
    print("COMPLETING ROPE PASSKEY TEST")
    print("="*70)
    model, tokenizer = load_model("llama2-7b", device="cuda")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nPasskey: 20 samples")
    baseline = run_passkey_test(model, tokenizer, n=20, disrupted=False)
    disrupted = run_passkey_test(model, tokenizer, n=20, disrupted=True)
    deg = (baseline['accuracy'] - disrupted['accuracy']) / baseline['accuracy'] * 100 if baseline['accuracy'] > 0 else 0
    all_results['RoPE']['contextual']['passkey'] = {'baseline': baseline['accuracy'], 'disrupted': disrupted['accuracy'], 'degradation': deg}
    print(f"Passkey: {baseline['accuracy']:.1%} → {disrupted['accuracy']:.1%} ({deg:.1f}% degradation)")

    del model
    torch.cuda.empty_cache()

    # Run all DroPE tests
    print("\n" + "="*70)
    print("LOADING DROPE MODEL")
    print("="*70)
    model, tokenizer = load_model("llama2-7b-drope", device="cuda")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results['DroPE'] = run_drope_tests(model, tokenizer)

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print("\n{:<10} {:<12} {:<10} {:<10} {:<10} {:<12}".format(
        "Model", "Category", "Task", "Baseline", "Disrupted", "Degradation"))
    print("-"*70)

    for model_name in ['RoPE', 'DroPE']:
        for category in ['parametric', 'contextual']:
            for task, r in all_results[model_name][category].items():
                print("{:<10} {:<12} {:<10} {:<10.1%} {:<10.1%} {:<12.1f}%".format(
                    model_name, category, task, r['baseline'], r['disrupted'], r['degradation']))

    # Compute averages
    print("\n" + "="*70)
    print("AVERAGE DEGRADATION BY CATEGORY")
    print("="*70)

    for model_name in ['RoPE', 'DroPE']:
        param_degs = [r['degradation'] for r in all_results[model_name]['parametric'].values() if r['baseline'] > 0]
        ctx_degs = [r['degradation'] for r in all_results[model_name]['contextual'].values() if r['baseline'] > 0]

        param_avg = sum(param_degs) / len(param_degs) if param_degs else 0
        ctx_avg = sum(ctx_degs) / len(ctx_degs) if ctx_degs else 0

        print(f"\n{model_name}:")
        print(f"  Parametric avg degradation: {param_avg:.1f}%")
        print(f"  Contextual avg degradation: {ctx_avg:.1f}%")
        if param_avg > 0:
            print(f"  Ratio (contextual/parametric): {ctx_avg/param_avg:.1f}x")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"jin_tests_final_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
