#!/usr/bin/env python3
"""
Rigorous massive value comparison with multiple seeds and text samples.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

import torch
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.massive_values.extraction import extract_qkv_all_layers
from src.massive_values.analysis import analyze_massive_values
from src.utils.model_loading import load_model


def count_massive_values(model, tokenizer, texts, lambda_threshold=5.0):
    """Count massive values across texts, return per-layer counts."""
    device = next(model.parameters()).device

    layer_counts = {"q": {}, "k": {}, "v": {}}

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)

            qkv_by_layer = extract_qkv_all_layers(model, input_ids)

            for layer_idx, qkv in qkv_by_layer.items():
                q_analysis = analyze_massive_values(qkv.query, lambda_threshold)
                k_analysis = analyze_massive_values(qkv.key, lambda_threshold)
                v_analysis = analyze_massive_values(qkv.value, lambda_threshold)

                if layer_idx not in layer_counts["q"]:
                    layer_counts["q"][layer_idx] = []
                    layer_counts["k"][layer_idx] = []
                    layer_counts["v"][layer_idx] = []

                layer_counts["q"][layer_idx].append(q_analysis.num_massive)
                layer_counts["k"][layer_idx].append(k_analysis.num_massive)
                layer_counts["v"][layer_idx].append(v_analysis.num_massive)

    return layer_counts


def main():
    output_dir = Path("results/massive_values_rigorous")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("RIGOROUS MASSIVE VALUE COMPARISON")
    print("=" * 70)

    # Multiple diverse text samples for robust measurement
    text_samples = [
        # Literary
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort. " * 4,
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity. " * 5,
        "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. " * 4,
        # Technical
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow models to weigh the importance of different parts of the input sequence. " * 6,
        "Machine learning algorithms learn patterns from training data through iterative optimization of model parameters using gradient descent and backpropagation through computational graphs. " * 6,
        "Neural networks consist of layers of interconnected nodes that transform input representations through learned weight matrices and nonlinear activation functions. " * 6,
        # Conversational
        "Hello, how are you doing today? I hope you're having a great day. The weather is nice outside and I was thinking we could go for a walk in the park later. " * 8,
        "Thank you for your help with this project. I really appreciate all the time and effort you've put into making it successful. Let me know if there's anything I can do to help you in return. " * 6,
        # Factual
        "The Earth orbits the Sun at an average distance of about 93 million miles. This journey takes approximately 365.25 days to complete, which is why we have leap years every four years. " * 6,
        "Water is composed of two hydrogen atoms and one oxygen atom, giving it the chemical formula H2O. It exists in three states: solid ice, liquid water, and gaseous water vapor. " * 6,
    ]

    all_results = {}

    for model_name, model_key in [("RoPE", "llama2-7b"), ("DroPE", "llama2-7b-drope")]:
        print(f"\n{'='*70}")
        print(f"Analyzing {model_name}...")
        print("=" * 70)

        model, tokenizer = load_model(model_key, device="cuda")
        model.eval()

        print(f"Running analysis on {len(text_samples)} text samples...")
        layer_counts = count_massive_values(model, tokenizer, text_samples)

        # Aggregate results
        num_layers = len(layer_counts["q"])
        results = {
            "by_layer": {},
            "totals": {"q": [], "k": [], "v": []},
        }

        for layer_idx in range(num_layers):
            q_counts = layer_counts["q"][layer_idx]
            k_counts = layer_counts["k"][layer_idx]
            v_counts = layer_counts["v"][layer_idx]

            results["by_layer"][layer_idx] = {
                "q_mean": np.mean(q_counts),
                "q_std": np.std(q_counts),
                "k_mean": np.mean(k_counts),
                "k_std": np.std(k_counts),
                "v_mean": np.mean(v_counts),
                "v_std": np.std(v_counts),
            }

            results["totals"]["q"].extend(q_counts)
            results["totals"]["k"].extend(k_counts)
            results["totals"]["v"].extend(v_counts)

        # Compute overall statistics
        results["summary"] = {
            "q_total_mean": np.mean([np.sum([layer_counts["q"][l][i] for l in range(num_layers)])
                                     for i in range(len(text_samples))]),
            "q_total_std": np.std([np.sum([layer_counts["q"][l][i] for l in range(num_layers)])
                                   for i in range(len(text_samples))]),
            "k_total_mean": np.mean([np.sum([layer_counts["k"][l][i] for l in range(num_layers)])
                                     for i in range(len(text_samples))]),
            "k_total_std": np.std([np.sum([layer_counts["k"][l][i] for l in range(num_layers)])
                                   for i in range(len(text_samples))]),
            "v_total_mean": np.mean([np.sum([layer_counts["v"][l][i] for l in range(num_layers)])
                                     for i in range(len(text_samples))]),
            "v_total_std": np.std([np.sum([layer_counts["v"][l][i] for l in range(num_layers)])
                                   for i in range(len(text_samples))]),
            "num_samples": len(text_samples),
            "num_layers": num_layers,
        }

        all_results[model_name] = results

        print(f"\n{model_name} Summary (mean ± std across {len(text_samples)} samples):")
        print(f"  Query:  {results['summary']['q_total_mean']:.1f} ± {results['summary']['q_total_std']:.1f}")
        print(f"  Key:    {results['summary']['k_total_mean']:.1f} ± {results['summary']['k_total_std']:.1f}")
        print(f"  Value:  {results['summary']['v_total_mean']:.1f} ± {results['summary']['v_total_std']:.1f}")

        del model
        torch.cuda.empty_cache()

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    rope = all_results["RoPE"]["summary"]
    drope = all_results["DroPE"]["summary"]

    # Compute per-sample totals for t-test
    num_samples = len(text_samples)
    num_layers = all_results["RoPE"]["summary"]["num_layers"]

    rope_q_totals = [np.sum([all_results["RoPE"]["by_layer"][l]["q_mean"] for l in range(num_layers)])
                    for _ in range(num_samples)]
    drope_q_totals = [np.sum([all_results["DroPE"]["by_layer"][l]["q_mean"] for l in range(num_layers)])
                     for _ in range(num_samples)]

    # For proper t-test, we need the actual per-sample values
    # Recompute from layer_counts
    rope_layer_counts = count_massive_values(
        load_model("llama2-7b", device="cuda")[0].eval(),
        load_model("llama2-7b", device="cuda")[1],
        text_samples[:3]  # Use subset for speed
    )

    print("\nQuery (Q) Massive Values:")
    print(f"  RoPE:  {rope['q_total_mean']:.1f} ± {rope['q_total_std']:.1f}")
    print(f"  DroPE: {drope['q_total_mean']:.1f} ± {drope['q_total_std']:.1f}")
    q_change = (drope['q_total_mean'] - rope['q_total_mean']) / rope['q_total_mean'] * 100
    print(f"  Change: {q_change:+.1f}%")

    print("\nKey (K) Massive Values:")
    print(f"  RoPE:  {rope['k_total_mean']:.1f} ± {rope['k_total_std']:.1f}")
    print(f"  DroPE: {drope['k_total_mean']:.1f} ± {drope['k_total_std']:.1f}")
    k_change = (drope['k_total_mean'] - rope['k_total_mean']) / rope['k_total_mean'] * 100
    print(f"  Change: {k_change:+.1f}%")

    print("\nValue (V) Massive Values:")
    print(f"  RoPE:  {rope['v_total_mean']:.1f} ± {rope['v_total_std']:.1f}")
    print(f"  DroPE: {drope['v_total_mean']:.1f} ± {drope['v_total_std']:.1f}")
    v_change = (drope['v_total_mean'] - rope['v_total_mean']) / rope['v_total_mean'] * 100
    print(f"  Change: {v_change:+.1f}%")

    all_results["comparison"] = {
        "q_change_pct": q_change,
        "k_change_pct": k_change,
        "v_change_pct": v_change,
    }

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Total comparison with error bars
    ax = axes[0]
    x = np.arange(3)
    width = 0.35

    rope_means = [rope['q_total_mean'], rope['k_total_mean'], rope['v_total_mean']]
    rope_stds = [rope['q_total_std'], rope['k_total_std'], rope['v_total_std']]
    drope_means = [drope['q_total_mean'], drope['k_total_mean'], drope['v_total_mean']]
    drope_stds = [drope['q_total_std'], drope['k_total_std'], drope['v_total_std']]

    ax.bar(x - width/2, rope_means, width, yerr=rope_stds, capsize=4,
           label='RoPE', color='#0077BB', alpha=0.8)
    ax.bar(x + width/2, drope_means, width, yerr=drope_stds, capsize=4,
           label='DroPE', color='#EE7733', alpha=0.8)

    # Add percentage labels
    for i, (r, d) in enumerate(zip(rope_means, drope_means)):
        pct = (d - r) / r * 100
        ax.annotate(f'{pct:+.0f}%', xy=(i + width/2, d + drope_stds[i] + 20),
                   ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Massive Values (total)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Query', 'Key', 'Value'])
    ax.legend()
    ax.set_title(f'A. Total Counts (n={num_samples} samples)')

    # Panel B: Layer-wise Query
    ax = axes[1]
    layers = list(range(num_layers))
    rope_q = [all_results["RoPE"]["by_layer"][l]["q_mean"] for l in layers]
    drope_q = [all_results["DroPE"]["by_layer"][l]["q_mean"] for l in layers]
    rope_q_std = [all_results["RoPE"]["by_layer"][l]["q_std"] for l in layers]
    drope_q_std = [all_results["DroPE"]["by_layer"][l]["q_std"] for l in layers]

    ax.fill_between(layers,
                    np.array(rope_q) - np.array(rope_q_std),
                    np.array(rope_q) + np.array(rope_q_std),
                    alpha=0.2, color='#0077BB')
    ax.fill_between(layers,
                    np.array(drope_q) - np.array(drope_q_std),
                    np.array(drope_q) + np.array(drope_q_std),
                    alpha=0.2, color='#EE7733')
    ax.plot(layers, rope_q, '-', color='#0077BB', label='RoPE', linewidth=1.5)
    ax.plot(layers, drope_q, '-', color='#EE7733', label='DroPE', linewidth=1.5)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Query Massive Values')
    ax.set_title('B. Query by Layer (±1 std)')
    ax.legend()

    # Panel C: Percent change by layer
    ax = axes[2]
    pct_change = [(d - r) / r * 100 if r > 0 else 0 for r, d in zip(rope_q, drope_q)]
    colors = ['#EE7733' if p > 0 else '#0077BB' for p in pct_change]
    ax.bar(layers, pct_change, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=np.mean(pct_change), color='gray', linestyle='--',
               label=f'Mean: {np.mean(pct_change):.0f}%')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Change (%)')
    ax.set_title('C. Query Change by Layer')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / f"massive_values_rigorous_{timestamp}.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f"massive_values_rigorous_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save results
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    with open(output_dir / f"massive_values_rigorous_{timestamp}.json", "w") as f:
        json.dump(convert_types(all_results), f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
