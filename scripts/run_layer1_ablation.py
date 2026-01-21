#!/usr/bin/env python3
"""
Experiment 6: Layer 1 Ablation Study

DroPE Layer 1 shows three unusual properties:
- 37× more massive values than RoPE (101.3 vs 2.7)
- Maximum BOS write score (1.14)
- Extreme Q/K activations (±394 vs RoPE's ±5)

Hypothesis: Layer 1 serves as a "positional substitute" mechanism in DroPE.

Ablation conditions:
1. baseline - No ablation
2. layer1_mlp - Zero MLP output at layer 1
3. layer1_attn - Zero attention output at layer 1
4. layer1_both - Zero both MLP and attention at layer 1
5. layer1_bos_only - Zero only BOS token's layer 1 outputs (MLP + attn)

Metrics: Perplexity + functional tasks (cities, sports, passkey, IMDB)
"""

import sys
from pathlib import Path
import torch
import json
import random
import csv
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
import numpy as np

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


class Layer1MLPAblationHook:
    """Hook that zeros out MLP output at layer 1."""
    def __init__(self, bos_only=False):
        self.bos_only = bos_only
        self.handle = None

    def hook_fn(self, module, input, output):
        modified = output.clone()
        if self.bos_only:
            modified[:, 0, :] = 0  # Only BOS token
        else:
            modified[:, :, :] = 0  # All tokens
        return modified

    def register(self, model, layer_idx=1):
        if self.handle is not None:
            return self
        mlp = model.model.layers[layer_idx].mlp
        self.handle = mlp.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


class Layer1AttnAblationHook:
    """Hook that zeros out attention output at layer 1."""
    def __init__(self, bos_only=False):
        self.bos_only = bos_only
        self.handle = None

    def hook_fn(self, module, input, output):
        # LlamaAttention returns tuple: (attn_output, attn_weights, past_kv)
        # or just attn_output depending on config
        if isinstance(output, tuple):
            attn_out = output[0].clone()
            if self.bos_only:
                attn_out[:, 0, :] = 0
            else:
                attn_out[:, :, :] = 0
            return (attn_out,) + output[1:]
        else:
            modified = output.clone()
            if self.bos_only:
                modified[:, 0, :] = 0
            else:
                modified[:, :, :] = 0
            return modified

    def register(self, model, layer_idx=1):
        if self.handle is not None:
            return self
        attn = model.model.layers[layer_idx].self_attn
        self.handle = attn.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on a set of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(model.device)

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        total_loss += loss.item() * (input_ids.shape[1] - 1)
        total_tokens += input_ids.shape[1] - 1

    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


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
    """Run parametric (yes/no factual) test."""
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
    """Run passkey retrieval test."""
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
    """Run IMDB sentiment test."""
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


def load_test_texts():
    """Load test texts for perplexity evaluation."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > 100][:50]
    return texts


def main():
    print("=" * 60)
    print("Experiment 6: Layer 1 Ablation Study")
    print("=" * 60)

    # Load existing results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    test_texts = load_test_texts()
    print(f"Loaded {len(test_texts)} test texts for perplexity")

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

        # Initialize layer1 results
        if "layer1_ablation" not in results[model_key]:
            results[model_key]["layer1_ablation"] = {}

        # Load parametric data once
        cities_data = load_cities(n=200)
        sports_data = load_sports(n=100)

        # Define ablation conditions
        conditions = {
            "baseline": {"mlp": False, "attn": False, "bos_only": False},
            "layer1_mlp": {"mlp": True, "attn": False, "bos_only": False},
            "layer1_attn": {"mlp": False, "attn": True, "bos_only": False},
            "layer1_both": {"mlp": True, "attn": True, "bos_only": False},
            "layer1_bos_only": {"mlp": True, "attn": True, "bos_only": True},
        }

        for condition, config in conditions.items():
            print(f"\n--- {condition} ---")

            # Set up hooks
            mlp_hook = None
            attn_hook = None

            if config["mlp"]:
                mlp_hook = Layer1MLPAblationHook(bos_only=config["bos_only"])
                mlp_hook.register(model, layer_idx=1)
                print(f"  Registered Layer 1 MLP ablation (bos_only={config['bos_only']})")

            if config["attn"]:
                attn_hook = Layer1AttnAblationHook(bos_only=config["bos_only"])
                attn_hook.register(model, layer_idx=1)
                print(f"  Registered Layer 1 Attention ablation (bos_only={config['bos_only']})")

            # Compute perplexity
            ppl = compute_perplexity(model, tokenizer, test_texts[:20])
            print(f"  Perplexity: {ppl:.2f}")

            # Run functional tasks
            cities_acc = run_parametric(model, tokenizer, cities_data, "Cities")
            sports_acc = run_parametric(model, tokenizer, sports_data, "Sports")
            passkey_acc = run_passkey(model, tokenizer, n_samples=20)
            imdb_acc = run_imdb(model, tokenizer, n_samples=100)

            # Clean up hooks
            if mlp_hook:
                mlp_hook.remove()
            if attn_hook:
                attn_hook.remove()

            print(f"  Cities: {cities_acc:.0%}")
            print(f"  Sports: {sports_acc:.0%}")
            print(f"  Passkey: {passkey_acc:.0%}")
            print(f"  IMDB: {imdb_acc:.0%}")

            results[model_key]["layer1_ablation"][condition] = {
                "perplexity": float(ppl),
                "cities": float(cities_acc),
                "sports": float(sports_acc),
                "passkey": float(passkey_acc),
                "imdb": float(imdb_acc),
            }

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Layer 1 Ablation")
    print("=" * 60)

    print("\nPerplexity:")
    print(f"{'Condition':<20} {'RoPE':>12} {'DroPE':>12} {'RoPE ratio':>12} {'DroPE ratio':>12}")
    print("-" * 68)

    rope_baseline = results["RoPE"]["layer1_ablation"]["baseline"]["perplexity"]
    drope_baseline = results["DroPE"]["layer1_ablation"]["baseline"]["perplexity"]

    for cond in conditions:
        rope_ppl = results["RoPE"]["layer1_ablation"][cond]["perplexity"]
        drope_ppl = results["DroPE"]["layer1_ablation"][cond]["perplexity"]
        rope_ratio = rope_ppl / rope_baseline
        drope_ratio = drope_ppl / drope_baseline
        print(f"{cond:<20} {rope_ppl:>12.2f} {drope_ppl:>12.2f} {rope_ratio:>12.2f}x {drope_ratio:>12.2f}x")

    print("\nParametric Tasks (Cities, Sports):")
    print(f"{'Condition':<20} {'RoPE Cities':>12} {'RoPE Sports':>12} {'DroPE Cities':>12} {'DroPE Sports':>12}")
    print("-" * 68)
    for cond in conditions:
        rc = results["RoPE"]["layer1_ablation"][cond]["cities"]
        rs = results["RoPE"]["layer1_ablation"][cond]["sports"]
        dc = results["DroPE"]["layer1_ablation"][cond]["cities"]
        ds = results["DroPE"]["layer1_ablation"][cond]["sports"]
        print(f"{cond:<20} {rc:>12.0%} {rs:>12.0%} {dc:>12.0%} {ds:>12.0%}")

    print("\nContextual Tasks (Passkey, IMDB):")
    print(f"{'Condition':<20} {'RoPE Pass':>12} {'RoPE IMDB':>12} {'DroPE Pass':>12} {'DroPE IMDB':>12}")
    print("-" * 68)
    for cond in conditions:
        rp = results["RoPE"]["layer1_ablation"][cond]["passkey"]
        ri = results["RoPE"]["layer1_ablation"][cond]["imdb"]
        dp = results["DroPE"]["layer1_ablation"][cond]["passkey"]
        di = results["DroPE"]["layer1_ablation"][cond]["imdb"]
        print(f"{cond:<20} {rp:>12.0%} {ri:>12.0%} {dp:>12.0%} {di:>12.0%}")


if __name__ == "__main__":
    main()
