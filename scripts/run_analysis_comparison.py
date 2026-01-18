#!/usr/bin/env python3
"""
Analyze massive values in SmolLM and compare with an immediate DroPE conversion.

This tests whether massive values are:
1. Baked into the projection weights (persist after RoPE removal)
2. Or caused by RoPE during inference (would change after removal)

Note: Since we hook Q/K/V projection outputs BEFORE RoPE is applied,
the patterns should be identical if massive values are in the weights.
This validates our hypothesis about where massive values originate.
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
from src.drope.conversion import convert_rope_to_drope


def analyze_model(model, tokenizer, model_name: str, num_samples: int = 5, lambda_threshold: float = 5.0):
    """Run massive value analysis on a model."""
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

    for i, text in enumerate(tqdm(sample_texts[:num_samples], desc="Samples")):
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
            "concentration_q": sum(a.concentration_score for a in q_analyses) / len(q_analyses),
            "concentration_k": sum(a.concentration_score for a in k_analyses) / len(k_analyses),
            "concentration_v": sum(a.concentration_score for a in v_analyses) / len(v_analyses),
        })

    total_q = sum(r["num_massive_q"] for r in results)
    total_k = sum(r["num_massive_k"] for r in results)
    total_v = sum(r["num_massive_v"] for r in results)

    print(f"\n{model_name} Summary:")
    print(f"  Total massive values Q: {total_q:.1f}")
    print(f"  Total massive values K: {total_k:.1f}")
    print(f"  Total massive values V: {total_v:.1f}")
    if total_v > 0:
        print(f"  Q/V ratio: {total_q/total_v:.2f}x")

    return results


def test_generation(model, tokenizer, model_name: str):
    """Quick generation test to see if model still works."""
    print(f"\nGeneration test for {model_name}:")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Output: {generated}")


def main():
    output_dir = Path("results/smollm_drope_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("SmolLM-360M: RoPE vs DroPE (Immediate Conversion) Analysis")
    print("="*60)
    print("\nThis tests whether massive values are in the weights vs. caused by RoPE.")
    print("Since we hook Q/K/V projections BEFORE RoPE, patterns should match if")
    print("massive values are in the learned weights.\n")

    # Load RoPE model
    print("Loading SmolLM-360M (RoPE)...")
    rope_model, tokenizer = load_model("smollm-360m", device="cuda")
    rope_model.eval()

    # Test generation before analysis
    test_generation(rope_model, tokenizer, "RoPE model")

    # Analyze RoPE model
    rope_results = analyze_model(rope_model, tokenizer, "SmolLM-360M-RoPE")

    # Convert to DroPE (no recalibration)
    print("\n" + "="*60)
    print("Converting to DroPE (removing RoPE)...")
    print("="*60)

    drope_model = convert_rope_to_drope(rope_model, copy_model=True)
    drope_model.eval()

    # Test generation after conversion
    test_generation(drope_model, tokenizer, "DroPE model (unconverted)")

    # Analyze DroPE model
    drope_results = analyze_model(drope_model, tokenizer, "SmolLM-360M-DroPE-Unconverted")

    # Generate comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = [r["layer"] for r in rope_results]

    # Query massive values
    axes[0, 0].plot(layers, [r["num_massive_q"] for r in rope_results], "o-", label="RoPE", linewidth=2)
    axes[0, 0].plot(layers, [r["num_massive_q"] for r in drope_results], "s--", label="DroPE (unconverted)", linewidth=2)
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Number of Massive Values")
    axes[0, 0].set_title("Query (Q) Massive Values")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Key massive values
    axes[0, 1].plot(layers, [r["num_massive_k"] for r in rope_results], "o-", label="RoPE", linewidth=2)
    axes[0, 1].plot(layers, [r["num_massive_k"] for r in drope_results], "s--", label="DroPE (unconverted)", linewidth=2)
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Number of Massive Values")
    axes[0, 1].set_title("Key (K) Massive Values")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Value massive values
    axes[1, 0].plot(layers, [r["num_massive_v"] for r in rope_results], "o-", label="RoPE", linewidth=2)
    axes[1, 0].plot(layers, [r["num_massive_v"] for r in drope_results], "s--", label="DroPE (unconverted)", linewidth=2)
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Number of Massive Values")
    axes[1, 0].set_title("Value (V) Massive Values")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Concentration comparison
    axes[1, 1].plot(layers, [r["concentration_q"] for r in rope_results], "o-", label="RoPE Q", linewidth=2)
    axes[1, 1].plot(layers, [r["concentration_q"] for r in drope_results], "s--", label="DroPE Q (unconverted)", linewidth=2)
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Concentration (Gini)")
    axes[1, 1].set_title("Query Concentration")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("SmolLM-360M: RoPE vs DroPE (Unconverted) Comparison\n(Tests if massive values are in weights)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f"smollm_comparison_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save results
    results = {
        "rope": rope_results,
        "drope_unconverted": drope_results,
        "note": "DroPE model is NOT recalibrated - this tests if massive values are in projection weights"
    }
    with open(output_dir / f"comparison_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    rope_total_q = sum(r["num_massive_q"] for r in rope_results)
    rope_total_k = sum(r["num_massive_k"] for r in rope_results)

    drope_total_q = sum(r["num_massive_q"] for r in drope_results)
    drope_total_k = sum(r["num_massive_k"] for r in drope_results)

    print(f"\nRoPE  - Q: {rope_total_q:.1f}, K: {rope_total_k:.1f}")
    print(f"DroPE - Q: {drope_total_q:.1f}, K: {drope_total_k:.1f}")

    diff_q = abs(rope_total_q - drope_total_q) / rope_total_q * 100
    diff_k = abs(rope_total_k - drope_total_k) / rope_total_k * 100

    print(f"\nDifference: Q: {diff_q:.1f}%, K: {diff_k:.1f}%")

    if diff_q < 5 and diff_k < 5:
        print("\n✓ Massive value patterns are IDENTICAL - they are in the projection weights!")
        print("  This supports the Persistence Hypothesis: massive values are learned into weights")
        print("  during RoPE training and persist after RoPE removal.")
    else:
        print("\n! Massive value patterns CHANGED after RoPE removal.")
        print("  This suggests RoPE influences the patterns during inference.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
