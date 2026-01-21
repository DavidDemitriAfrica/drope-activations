#!/usr/bin/env python3
"""
Run functional tests (cities, sports, passkey, IMDB) for all intervention conditions.
Uses real data from Jin et al. datasets and HuggingFace.

Conditions:
1. Baseline
2. BOS-MLP ablation
3. Q/K massive value disruption
4. Combined (both)

Fixed version: Hooks are registered once before all tests and removed once after.
"""

import sys
from pathlib import Path
import torch
import json
import random
import csv
from tqdm import tqdm
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

CITIES_PATH = REPO_ROOT / "Rope_with_LLM" / "datasets" / "cities.csv"
SPORTS_PATH = REPO_ROOT / "Rope_with_LLM" / "datasets" / "sports_qa.json"

import custom_models
import custom_models.attention
import custom_models.drope

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROPE_MODEL = "meta-llama/Llama-2-7b-hf"
DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"
RESULTS_FILE = REPO_ROOT / "results" / "phase_metrics" / "rope_vs_drope_phase_metrics.json"


class BOSMLPAblationHook:
    """Hook that zeros out MLP output for BOS token (position 0)."""
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.handle = None

    def hook_fn(self, module, input, output):
        modified = output.clone()
        modified[:, 0, :] = 0
        return modified

    def register(self, model):
        if self.handle is not None:
            return self  # Already registered
        mlp = model.model.layers[self.layer_idx].mlp
        self.handle = mlp.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def register_qk_disruption_hooks(model, num_heads=32, head_dim=128):
    """Register hooks to replace top-1 massive dimension per head with mean (Jin et al. method)."""
    hooks = []
    layers = model.model.layers

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


def generate_response(model, tokenizer, prompt, max_new_tokens=20):
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


def load_cities(n=200):
    """Load cities dataset from Jin et al."""
    data = []
    with open(CITIES_PATH, 'r') as f:
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


def load_sports(n=100):
    """Load sports QA dataset from Jin et al."""
    with open(SPORTS_PATH, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    random.seed(42)
    return random.sample(data, min(n, len(data)))


def convert_to_question(statement):
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


def check_yes_no(response):
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


def run_parametric(model, tokenizer, data, name):
    """Run parametric (yes/no factual) test. No hook management - hooks should be registered externally."""
    correct = total = 0

    for item in tqdm(data, desc=name, leave=False):
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

    return correct / total if total > 0 else 0


def run_passkey(model, tokenizer, n_samples=20):
    """Run passkey retrieval test (same format as Jin et al.)."""
    correct = 0

    for i in tqdm(range(n_samples), desc="Passkey", leave=False):
        random.seed(42 + i)
        passkey = str(random.randint(10000, 99999))

        filler = "The grass is green. The sky is blue. The sun is yellow. " * 30
        insert_pos = len(filler) // 2
        text = filler[:insert_pos] + f" The secret passkey is {passkey}. Remember this number. " + filler[insert_pos:]

        prompt = f"{text}\n\nWhat is the secret passkey mentioned above? Answer with just the number:"
        response = generate_response(model, tokenizer, prompt, max_new_tokens=20)

        if passkey in response:
            correct += 1

    return correct / n_samples


def load_imdb(n=100):
    """Load real IMDB data from HuggingFace."""
    dataset = load_dataset('imdb', split='test')

    pos = [x for x in dataset if x['label'] == 1][:n//2]
    neg = [x for x in dataset if x['label'] == 0][:n//2]

    data = []
    for item in pos:
        data.append({'text': item['text'], 'label': 1})
    for item in neg:
        data.append({'text': item['text'], 'label': 0})

    random.seed(42)
    random.shuffle(data)
    return data


def run_imdb(model, tokenizer, n_samples=100):
    """Run IMDB sentiment test (same format as Jin et al.)."""
    data = load_imdb(n_samples)
    correct = 0

    for item in tqdm(data, desc="IMDB", leave=False):
        text = item['text'][:1500]
        label = item['label']

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

        if predicted == label:
            correct += 1

    return correct / len(data)


def main():
    print("=" * 60)
    print("Running Functional Tests for All Conditions")
    print("=" * 60)

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

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

        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads

        bos_spike_layer = results[model_key].get("bos_spike_layer", 1 if model_key == "RoPE" else 2)

        # Initialize functional results
        if "functional" not in results[model_key]:
            results[model_key]["functional"] = {}

        # Load parametric data once
        cities_data = load_cities(n=200)
        sports_data = load_sports(n=100)

        conditions = ["baseline", "bos_mlp_ablation", "qk_disruption", "combined"]

        for condition in conditions:
            print(f"\n--- {condition} ---")

            # Set up and REGISTER hooks BEFORE running tests
            bos_hook = None
            qk_hooks = []

            if condition in ["bos_mlp_ablation", "combined"]:
                bos_hook = BOSMLPAblationHook(bos_spike_layer)
                bos_hook.register(model)
                print(f"  Registered BOS-MLP ablation hook at layer {bos_spike_layer}")

            if condition in ["qk_disruption", "combined"]:
                qk_hooks = register_qk_disruption_hooks(model, num_heads, head_dim)
                print(f"  Registered {len(qk_hooks)} Q/K disruption hooks")

            # Run all tests with hooks active
            cities_acc = run_parametric(model, tokenizer, cities_data, "Cities")
            sports_acc = run_parametric(model, tokenizer, sports_data, "Sports")
            passkey_acc = run_passkey(model, tokenizer, n_samples=20)
            imdb_acc = run_imdb(model, tokenizer, n_samples=100)

            # Clean up hooks AFTER all tests complete
            if bos_hook:
                bos_hook.remove()
            for h in qk_hooks:
                h.remove()

            print(f"  Cities: {cities_acc:.0%}")
            print(f"  Sports: {sports_acc:.0%}")
            print(f"  Passkey: {passkey_acc:.0%}")
            print(f"  IMDB: {imdb_acc:.0%}")

            # Get existing perplexity if available
            existing_ppl = results[model_key].get("functional", {}).get(condition, {}).get("perplexity", 0)

            results[model_key]["functional"][condition] = {
                "perplexity": existing_ppl,
                "cities": cities_acc,
                "sports": sports_acc,
                "passkey": passkey_acc,
                "imdb": imdb_acc,
            }

        del model
        torch.cuda.empty_cache()

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    conditions = ["baseline", "bos_mlp_ablation", "qk_disruption", "combined"]

    print("\nParametric Tasks (Cities, Sports):")
    print(f"{'Condition':<20} {'RoPE Cities':<12} {'RoPE Sports':<12} {'DroPE Cities':<12} {'DroPE Sports':<12}")
    print("-" * 68)
    for cond in conditions:
        rc = results["RoPE"]["functional"].get(cond, {}).get("cities", 0)
        rs = results["RoPE"]["functional"].get(cond, {}).get("sports", 0)
        dc = results["DroPE"]["functional"].get(cond, {}).get("cities", 0)
        ds = results["DroPE"]["functional"].get(cond, {}).get("sports", 0)
        print(f"{cond:<20} {rc:<12.0%} {rs:<12.0%} {dc:<12.0%} {ds:<12.0%}")

    print("\nContextual Tasks (Passkey, IMDB):")
    print(f"{'Condition':<20} {'RoPE Pass':<12} {'RoPE IMDB':<12} {'DroPE Pass':<12} {'DroPE IMDB':<12}")
    print("-" * 68)
    for cond in conditions:
        rp = results["RoPE"]["functional"].get(cond, {}).get("passkey", 0)
        ri = results["RoPE"]["functional"].get(cond, {}).get("imdb", 0)
        dp = results["DroPE"]["functional"].get(cond, {}).get("passkey", 0)
        di = results["DroPE"]["functional"].get(cond, {}).get("imdb", 0)
        print(f"{cond:<20} {rp:<12.0%} {ri:<12.0%} {dp:<12.0%} {di:<12.0%}")


if __name__ == "__main__":
    main()
