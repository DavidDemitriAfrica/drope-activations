#!/usr/bin/env python3
"""
Compare massive values between Llama-2-7B RoPE and DroPE models.
Uses 4-bit quantization to fit in memory.
"""

import json
from pathlib import Path
from datetime import datetime

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.massive_values.extraction import extract_qkv_all_layers
from src.massive_values.analysis import analyze_massive_values


def load_model_quantized(model_path: str, device: str = "cuda"):
    """Load a model with 4-bit quantization."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def analyze_model(model, tokenizer, model_name: str, num_samples: int = 3, lambda_threshold: float = 5.0):
    """Run massive value analysis on a model."""
    print(f"\nAnalyzing {model_name}...")

    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning models have revolutionized artificial intelligence. " * 10,
        "Natural language processing enables computers to understand human language. " * 10,
    ]

    all_layers = {}

    for i, text in enumerate(tqdm(sample_texts[:num_samples], desc="Samples")):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(model.device)

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


def main():
    output_dir = Path("results/llama_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("Llama-2-7B: RoPE vs DroPE Comparison")
    print("="*60)

    # Load and analyze RoPE model
    print("\nLoading Llama-2-7B (RoPE)...")
    rope_model, rope_tokenizer = load_model_quantized("meta-llama/Llama-2-7b-hf")
    rope_results = analyze_model(rope_model, rope_tokenizer, "Llama-2-7B-RoPE")

    # Free memory
    del rope_model
    torch.cuda.empty_cache()

    # Load and analyze DroPE model
    print("\nLoading Llama-2-7B (DroPE)...")
    drope_model, drope_tokenizer = load_model_quantized("SakanaAI/Llama-2-7b-hf-DroPE")
    drope_results = analyze_model(drope_model, drope_tokenizer, "Llama-2-7B-DroPE")

    # Generate comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = [r["layer"] for r in rope_results]

    # Query massive values
    axes[0, 0].plot(layers, [r["num_massive_q"] for r in rope_results], "o-", label="RoPE", linewidth=2)
    axes[0, 0].plot(layers, [r["num_massive_q"] for r in drope_results], "s--", label="DroPE", linewidth=2)
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Number of Massive Values")
    axes[0, 0].set_title("Query (Q) Massive Values")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Key massive values
    axes[0, 1].plot(layers, [r["num_massive_k"] for r in rope_results], "o-", label="RoPE", linewidth=2)
    axes[0, 1].plot(layers, [r["num_massive_k"] for r in drope_results], "s--", label="DroPE", linewidth=2)
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Number of Massive Values")
    axes[0, 1].set_title("Key (K) Massive Values")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Value massive values
    axes[1, 0].plot(layers, [r["num_massive_v"] for r in rope_results], "o-", label="RoPE", linewidth=2)
    axes[1, 0].plot(layers, [r["num_massive_v"] for r in drope_results], "s--", label="DroPE", linewidth=2)
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Number of Massive Values")
    axes[1, 0].set_title("Value (V) Massive Values")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Concentration comparison (Query)
    axes[1, 1].plot(layers, [r["concentration_q"] for r in rope_results], "o-", label="RoPE Q", linewidth=2)
    axes[1, 1].plot(layers, [r["concentration_q"] for r in drope_results], "s--", label="DroPE Q", linewidth=2)
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Concentration (Gini)")
    axes[1, 1].set_title("Query Concentration")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Llama-2-7B: RoPE vs DroPE Massive Value Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f"llama_comparison_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save results
    results = {
        "rope": rope_results,
        "drope": drope_results,
    }
    with open(output_dir / f"comparison_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    rope_total_q = sum(r["num_massive_q"] for r in rope_results)
    rope_total_k = sum(r["num_massive_k"] for r in rope_results)
    rope_total_v = sum(r["num_massive_v"] for r in rope_results)

    drope_total_q = sum(r["num_massive_q"] for r in drope_results)
    drope_total_k = sum(r["num_massive_k"] for r in drope_results)
    drope_total_v = sum(r["num_massive_v"] for r in drope_results)

    print(f"\nRoPE  - Q: {rope_total_q:.1f}, K: {rope_total_k:.1f}, V: {rope_total_v:.1f}")
    print(f"DroPE - Q: {drope_total_q:.1f}, K: {drope_total_k:.1f}, V: {drope_total_v:.1f}")

    print(f"\nChange after DroPE:")
    print(f"  Query:  {(drope_total_q - rope_total_q) / rope_total_q * 100:+.1f}%")
    print(f"  Key:    {(drope_total_k - rope_total_k) / rope_total_k * 100:+.1f}%")
    if rope_total_v > 0:
        print(f"  Value:  {(drope_total_v - rope_total_v) / rope_total_v * 100:+.1f}%")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
