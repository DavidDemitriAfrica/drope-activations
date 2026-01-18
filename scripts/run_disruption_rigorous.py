#!/usr/bin/env python3
"""
Rigorous disruption experiment with multiple seeds and statistical testing.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import sys

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.massive_values.extraction import extract_qkv_all_layers
from src.utils.model_loading import load_model


class ProjectionDisruptor:
    """Disrupts Q/K projection outputs by zeroing massive value dimensions."""

    def __init__(self, model, massive_dims, target="both"):
        self.model = model
        self.massive_dims = massive_dims
        self.target = target
        self.hooks = []

    def _create_hook(self, mask):
        def hook(module, input, output):
            zero_mask = (~mask).to(output.dtype).to(output.device)
            return output * zero_mask.unsqueeze(0).unsqueeze(0)
        return hook

    def register_hooks(self):
        self.remove_hooks()
        if hasattr(self.model, "model"):
            layers = self.model.model.layers
        else:
            layers = self.model.layers

        for idx, layer in enumerate(layers):
            if idx not in self.massive_dims:
                continue
            attn = layer.self_attn

            if self.target in ["query", "both"]:
                mask = self.massive_dims[idx].get("q")
                if mask is not None and mask.any():
                    handle = attn.q_proj.register_forward_hook(self._create_hook(mask))
                    self.hooks.append(handle)

            if self.target in ["key", "both"]:
                mask = self.massive_dims[idx].get("k")
                if mask is not None and mask.any():
                    handle = attn.k_proj.register_forward_hook(self._create_hook(mask))
                    self.hooks.append(handle)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on texts."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            outputs = model(input_ids, labels=input_ids)
            num_tokens = input_ids.shape[1] - 1
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


def identify_massive_dims(model, tokenizer, texts, lambda_threshold=5.0, seed=42):
    """Identify massive value dimensions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = next(model.parameters()).device
    all_q_norms = {}
    all_k_norms = {}

    with torch.no_grad():
        for text in texts[:3]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            qkv_by_layer = extract_qkv_all_layers(model, input_ids)

            for layer_idx, qkv in qkv_by_layer.items():
                q_norm = qkv.query.squeeze(0).norm(dim=0)
                k_norm = qkv.key.squeeze(0).norm(dim=0)

                if layer_idx not in all_q_norms:
                    all_q_norms[layer_idx] = []
                    all_k_norms[layer_idx] = []

                all_q_norms[layer_idx].append(q_norm)
                all_k_norms[layer_idx].append(k_norm)

    massive_dims = {}
    total_count = 0

    for layer_idx in all_q_norms.keys():
        q_norm_avg = torch.stack(all_q_norms[layer_idx]).mean(dim=0)
        k_norm_avg = torch.stack(all_k_norms[layer_idx]).mean(dim=0)

        q_mask = q_norm_avg > lambda_threshold * q_norm_avg.mean()
        k_mask = k_norm_avg > lambda_threshold * k_norm_avg.mean()

        if q_mask.any() or k_mask.any():
            massive_dims[layer_idx] = {"q": q_mask.flatten(), "k": k_mask.flatten()}
            total_count += q_mask.sum().item() + k_mask.sum().item()

    return massive_dims, total_count


def create_random_dims(massive_dims, seed=42):
    """Create random dimension masks with same count."""
    torch.manual_seed(seed)
    random_dims = {}

    for layer_idx, masks in massive_dims.items():
        random_dims[layer_idx] = {}
        for key in ["q", "k"]:
            mask = masks[key]
            num_massive = mask.sum().item()
            total_dims = mask.numel()
            random_mask = torch.zeros(total_dims, dtype=torch.bool, device=mask.device)
            random_indices = torch.randperm(total_dims)[:int(num_massive)]
            random_mask[random_indices] = True
            random_dims[layer_idx][key] = random_mask

    return random_dims


def run_single_trial(model, tokenizer, eval_texts, massive_dims, random_seed):
    """Run a single disruption trial."""
    # Baseline
    ppl_baseline = compute_perplexity(model, tokenizer, eval_texts)

    # Massive disruption
    with ProjectionDisruptor(model, massive_dims, target="both"):
        ppl_massive = compute_perplexity(model, tokenizer, eval_texts)

    # Random disruption (with seed)
    random_dims = create_random_dims(massive_dims, seed=random_seed)
    with ProjectionDisruptor(model, random_dims, target="both"):
        ppl_random = compute_perplexity(model, tokenizer, eval_texts)

    return {
        "baseline": ppl_baseline,
        "massive": ppl_massive,
        "random": ppl_random,
        "massive_increase_pct": (ppl_massive - ppl_baseline) / ppl_baseline * 100,
        "random_increase_pct": (ppl_random - ppl_baseline) / ppl_baseline * 100,
    }


def main():
    output_dir = Path("results/disruption_rigorous")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    N_SEEDS = 10  # Number of random seeds for control
    SEEDS = list(range(42, 42 + N_SEEDS))

    print("=" * 70)
    print("RIGOROUS DISRUPTION EXPERIMENT")
    print(f"Running {N_SEEDS} seeds for statistical validation")
    print("=" * 70)

    # Different evaluation text sets
    eval_sets = {
        "literary": [
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole. " * 10,
            "It was the best of times, it was the worst of times. " * 15,
            "Call me Ishmael. Some years ago, never mind how long precisely. " * 10,
        ],
        "technical": [
            "The transformer architecture uses self-attention mechanisms to process sequences. " * 15,
            "Machine learning models learn patterns from data through optimization. " * 15,
            "Neural networks consist of layers of interconnected nodes. " * 15,
        ],
        "repetitive": [
            "The quick brown fox jumps over the lazy dog. " * 30,
            "Pack my box with five dozen liquor jugs. " * 30,
            "How vexingly quick daft zebras jump. " * 30,
        ],
    }

    all_results = {}

    for model_name, model_key in [("RoPE", "llama2-7b"), ("DroPE", "llama2-7b-drope")]:
        print(f"\n{'='*70}")
        print(f"Loading {model_name}...")
        print("=" * 70)

        model, tokenizer = load_model(model_key, device="cuda")
        model.eval()

        # Identify massive dims (consistent across trials)
        print("Identifying massive value dimensions...")
        massive_dims, total_massive = identify_massive_dims(
            model, tokenizer, eval_sets["literary"], seed=42
        )
        print(f"Found {total_massive} massive dimensions")

        model_results = {
            "total_massive_dims": total_massive,
            "trials": [],
            "by_eval_set": {},
        }

        # Run trials with different seeds
        print(f"\nRunning {N_SEEDS} trials...")
        for seed in tqdm(SEEDS, desc="Seeds"):
            trial = run_single_trial(model, tokenizer, eval_sets["literary"], massive_dims, seed)
            trial["seed"] = seed
            model_results["trials"].append(trial)

        # Also test on different eval sets
        print("\nTesting on different text types...")
        for set_name, texts in eval_sets.items():
            set_result = run_single_trial(model, tokenizer, texts, massive_dims, 42)
            model_results["by_eval_set"][set_name] = set_result

        all_results[model_name] = model_results

        # Free memory
        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # Statistical Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    for model_name in ["RoPE", "DroPE"]:
        trials = all_results[model_name]["trials"]

        baseline = np.mean([t["baseline"] for t in trials])
        massive_pcts = [t["massive_increase_pct"] for t in trials]
        random_pcts = [t["random_increase_pct"] for t in trials]

        massive_mean = np.mean(massive_pcts)
        massive_std = np.std(massive_pcts)
        random_mean = np.mean(random_pcts)
        random_std = np.std(random_pcts)

        # Paired t-test: massive vs random
        t_stat, p_value = stats.ttest_rel(massive_pcts, random_pcts)

        # Effect size (Cohen's d)
        diff = np.array(massive_pcts) - np.array(random_pcts)
        cohens_d = np.mean(diff) / np.std(diff)

        print(f"\n{model_name}:")
        print(f"  Baseline PPL: {baseline:.2f}")
        print(f"  Massive disruption: {massive_mean:+.1f}% ± {massive_std:.1f}%")
        print(f"  Random disruption:  {random_mean:+.1f}% ± {random_std:.1f}%")
        print(f"  Difference (M-R):   {massive_mean - random_mean:+.1f}%")
        print(f"  Paired t-test: t={t_stat:.2f}, p={p_value:.2e}")
        print(f"  Cohen's d: {cohens_d:.2f}")

        all_results[model_name]["stats"] = {
            "baseline_ppl": baseline,
            "massive_mean": massive_mean,
            "massive_std": massive_std,
            "random_mean": random_mean,
            "random_std": random_std,
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
        }

    # Compare RoPE vs DroPE
    print("\n" + "-" * 70)
    print("COMPARISON: RoPE vs DroPE")
    print("-" * 70)

    rope_mr = all_results["RoPE"]["stats"]["massive_mean"] - all_results["RoPE"]["stats"]["random_mean"]
    drope_mr = all_results["DroPE"]["stats"]["massive_mean"] - all_results["DroPE"]["stats"]["random_mean"]

    # Independent t-test on M-R differences
    rope_diffs = [t["massive_increase_pct"] - t["random_increase_pct"]
                  for t in all_results["RoPE"]["trials"]]
    drope_diffs = [t["massive_increase_pct"] - t["random_increase_pct"]
                   for t in all_results["DroPE"]["trials"]]

    t_stat, p_value = stats.ttest_ind(rope_diffs, drope_diffs)

    print(f"RoPE M-R difference:  {rope_mr:+.1f}%")
    print(f"DroPE M-R difference: {drope_mr:+.1f}%")
    print(f"Ratio: RoPE relies {rope_mr/drope_mr:.1f}x more on massive values")
    print(f"Independent t-test: t={t_stat:.2f}, p={p_value:.2e}")

    if p_value < 0.001:
        print("\n*** HIGHLY SIGNIFICANT (p < 0.001) ***")
    elif p_value < 0.01:
        print("\n** SIGNIFICANT (p < 0.01) **")
    elif p_value < 0.05:
        print("\n* SIGNIFICANT (p < 0.05) *")

    all_results["comparison"] = {
        "rope_mr_diff": rope_mr,
        "drope_mr_diff": drope_mr,
        "ratio": rope_mr / drope_mr,
        "t_statistic": t_stat,
        "p_value": p_value,
    }

    # =========================================================================
    # Visualization
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Box plots of increases
    ax = axes[0]
    rope_massive = [t["massive_increase_pct"] for t in all_results["RoPE"]["trials"]]
    rope_random = [t["random_increase_pct"] for t in all_results["RoPE"]["trials"]]
    drope_massive = [t["massive_increase_pct"] for t in all_results["DroPE"]["trials"]]
    drope_random = [t["random_increase_pct"] for t in all_results["DroPE"]["trials"]]

    # Use log scale for visualization
    data = [rope_massive, rope_random, drope_massive, drope_random]
    positions = [1, 2, 4, 5]
    colors = ['#e74c3c', '#3498db', '#e74c3c', '#3498db']

    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yscale('log')
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(['RoPE', 'DroPE'])
    ax.set_ylabel('PPL Increase (%)')
    ax.set_title(f'A. Distribution Across {N_SEEDS} Seeds')
    ax.legend([bp['boxes'][0], bp['boxes'][1]], ['Massive', 'Random'], loc='upper right')

    # Panel B: M-R differences with error bars
    ax = axes[1]
    x = [0, 1]
    means = [rope_mr, drope_mr]
    stds = [np.std(rope_diffs), np.std(drope_diffs)]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#0077BB', '#EE7733'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['RoPE', 'DroPE'])
    ax.set_ylabel('M-R Difference (%)')
    ax.set_title('B. Reliance on Massive Values')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Add significance stars
    if p_value < 0.001:
        ax.annotate('***', xy=(0.5, max(means) * 0.9), ha='center', fontsize=14)

    # Panel C: By text type
    ax = axes[2]
    set_names = list(eval_sets.keys())
    x = np.arange(len(set_names))
    width = 0.35

    rope_by_set = [all_results["RoPE"]["by_eval_set"][s]["massive_increase_pct"] for s in set_names]
    drope_by_set = [all_results["DroPE"]["by_eval_set"][s]["massive_increase_pct"] for s in set_names]

    ax.bar(x - width/2, rope_by_set, width, label='RoPE', color='#0077BB')
    ax.bar(x + width/2, drope_by_set, width, label='DroPE', color='#EE7733')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(set_names)
    ax.set_ylabel('PPL Increase (%)')
    ax.set_title('C. Consistency Across Text Types')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / f"rigorous_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f"rigorous_results_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save results
    # Convert numpy types for JSON
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    with open(output_dir / f"rigorous_results_{timestamp}.json", "w") as f:
        json.dump(convert_types(all_results), f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
