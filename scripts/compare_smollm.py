#!/usr/bin/env python3
"""
Compare SmolLM-360M RoPE vs DroPE (unconverted - RoPE removed, no recalibration).
This shows what happens immediately when RoPE is removed.
"""

import json
from pathlib import Path
from datetime import datetime

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.massive_values.extraction import extract_qkv_all_layers
from src.massive_values.analysis import analyze_massive_values
from src.utils.model_loading import load_model


def analyze_model(model, tokenizer, model_name: str, num_samples: int = 5, lambda_threshold: float = 5.0):
    """Run massive value analysis."""
    print(f"\nAnalyzing {model_name}...")

    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "In the beginning, there was only darkness. Then came light. " * 10,
        "Machine learning models have revolutionized artificial intelligence. " * 15,
        "Natural language processing enables computers to understand human language. " * 15,
        "The history of computing dates back to the early mechanical calculators. " * 15,
    ]

    all_layers = {}
    device = next(model.parameters()).device

    for text in tqdm(sample_texts[:num_samples], desc="Samples"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            qkv_by_layer = extract_qkv_all_layers(model, input_ids)

        for layer_idx, qkv in qkv_by_layer.items():
            if layer_idx not in all_layers:
                all_layers[layer_idx] = {"q": [], "k": [], "v": []}
            all_layers[layer_idx]["q"].append(qkv.query)
            all_layers[layer_idx]["k"].append(qkv.key)
            all_layers[layer_idx]["v"].append(qkv.value)

    results = []
    for layer_idx in sorted(all_layers.keys()):
        q_analyses = [analyze_massive_values(q, lambda_threshold) for q in all_layers[layer_idx]["q"]]
        k_analyses = [analyze_massive_values(k, lambda_threshold) for k in all_layers[layer_idx]["k"]]
        v_analyses = [analyze_massive_values(v, lambda_threshold) for v in all_layers[layer_idx]["v"]]

        results.append({
            "layer": layer_idx,
            "num_massive_q": sum(a.num_massive for a in q_analyses) / len(q_analyses),
            "num_massive_k": sum(a.num_massive for a in k_analyses) / len(k_analyses),
            "num_massive_v": sum(a.num_massive for a in v_analyses) / len(v_analyses),
        })

    return results


def main():
    output_dir = Path("results/smollm_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("SmolLM-360M: RoPE vs DroPE (Unconverted) Comparison")
    print("=" * 60)
    print("\nNote: 'Unconverted' means RoPE removed but NO recalibration.")
    print("This tests if massive values are in projection weights.\n")

    # Load SmolLM RoPE
    print("Loading SmolLM-360M (RoPE)...")
    rope_model, tokenizer = load_model("smollm-360m", device="cuda")
    rope_model.eval()

    # Analyze RoPE
    rope_results = analyze_model(rope_model, tokenizer, "SmolLM-RoPE")

    # Since we hook Q/K/V projections BEFORE RoPE is applied,
    # the results should be IDENTICAL for RoPE vs unconverted DroPE
    # (because the projections are the same, RoPE is applied after)

    # But let's verify by actually removing RoPE and checking
    from src.drope.conversion import convert_rope_to_drope

    print("\nConverting to DroPE (removing RoPE)...")
    drope_model = convert_rope_to_drope(rope_model, copy_model=True)
    drope_model.eval()

    # Analyze DroPE (unconverted)
    drope_results = analyze_model(drope_model, tokenizer, "SmolLM-DroPE-Unconverted")

    # Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    rope_total_q = sum(r["num_massive_q"] for r in rope_results)
    rope_total_k = sum(r["num_massive_k"] for r in rope_results)
    rope_total_v = sum(r["num_massive_v"] for r in rope_results)

    drope_total_q = sum(r["num_massive_q"] for r in drope_results)
    drope_total_k = sum(r["num_massive_k"] for r in drope_results)
    drope_total_v = sum(r["num_massive_v"] for r in drope_results)

    print(f"\nRoPE  - Q: {rope_total_q:.1f}, K: {rope_total_k:.1f}, V: {rope_total_v:.1f}")
    print(f"DroPE - Q: {drope_total_q:.1f}, K: {drope_total_k:.1f}, V: {drope_total_v:.1f}")

    diff_q = abs(drope_total_q - rope_total_q)
    diff_k = abs(drope_total_k - rope_total_k)

    print(f"\nDifference: Q: {diff_q:.1f}, K: {diff_k:.1f}")

    if diff_q < 1 and diff_k < 1:
        print("\n✓ IDENTICAL! Massive values are in the projection weights.")
        print("  Since we hook BEFORE RoPE, removing RoPE doesn't change Q/K/V outputs.")
        print("  This confirms: massive values are learned into weights, not caused by RoPE at inference.")
    else:
        print("\n! Values differ - unexpected result.")

    # Save results
    results = {
        "rope": rope_results,
        "drope_unconverted": drope_results,
        "note": "Unconverted DroPE = RoPE removed, no recalibration"
    }
    with open(output_dir / f"smollm_comparison_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create figure
    try:
        plt.style.use(['science', 'ieee', 'no-latex'])
    except:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    layers = [r["layer"] for r in rope_results]

    for ax, (tensor, key) in zip(axes, [("Query", "num_massive_q"), ("Key", "num_massive_k"), ("Value", "num_massive_v")]):
        rope_vals = [r[key] for r in rope_results]
        drope_vals = [r[key] for r in drope_results]

        ax.plot(layers, rope_vals, 'o-', color='#0077BB', label='RoPE', markersize=3, linewidth=1.5)
        ax.plot(layers, drope_vals, 's--', color='#EE7733', label='DroPE (unconverted)', markersize=3, linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Massive Values')
        ax.set_title(tensor)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('SmolLM-360M: RoPE vs DroPE (Unconverted)\nMassive values are IDENTICAL (in projection weights)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f"smollm_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f"smollm_comparison_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
