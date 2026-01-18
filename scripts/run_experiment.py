#!/usr/bin/env python3
"""
Main experiment runner for DroPE massive value analysis.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.massive_values.extraction import extract_qkv_all_layers
from src.massive_values.analysis import (
    analyze_massive_values,
    analyze_layer_progression,
    compare_massive_value_positions,
)
from src.massive_values.visualization import (
    plot_massive_value_heatmap,
    plot_concentration_by_layer,
    plot_qkv_comparison,
    plot_model_comparison,
)
from src.evaluation.passkey import PasskeyRetrievalEvaluator, PasskeyConfig
from src.utils.model_loading import load_model, get_model_config


def run_massive_value_analysis(
    model_name: str,
    output_dir: Path,
    num_samples: int = 10,
    lambda_threshold: float = 5.0,
    device: str = "cuda",
):
    """Extract and analyze massive values from a model."""
    print(f"\n{'='*60}")
    print(f"Massive Value Analysis: {model_name}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading {model_name}...")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()

    # Sample texts for analysis
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "In the beginning, there was only darkness. Then came light, and with it, the dawn of a new era. " * 10,
        "Machine learning models have revolutionized the field of artificial intelligence. " * 15,
        "The history of computing dates back to the early mechanical calculators. " * 15,
        "Natural language processing enables computers to understand human language. " * 15,
    ]

    all_results = {
        "model": model_name,
        "lambda_threshold": lambda_threshold,
        "layers": {},
    }

    # Analyze multiple samples and aggregate
    print(f"\nAnalyzing {num_samples} samples...")
    aggregated_q = {}
    aggregated_k = {}
    aggregated_v = {}

    for i, text in enumerate(tqdm(sample_texts[:num_samples], desc="Samples")):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            qkv_by_layer = extract_qkv_all_layers(model, input_ids)

        for layer_idx, qkv in qkv_by_layer.items():
            if layer_idx not in aggregated_q:
                aggregated_q[layer_idx] = []
                aggregated_k[layer_idx] = []
                aggregated_v[layer_idx] = []

            aggregated_q[layer_idx].append(qkv.query)
            aggregated_k[layer_idx].append(qkv.key)
            aggregated_v[layer_idx].append(qkv.value)

    # Analyze aggregated results
    print("\nComputing massive value statistics...")
    layer_results = []

    for layer_idx in sorted(aggregated_q.keys()):
        # Analyze each sample separately and average results
        q_analyses = [analyze_massive_values(q, lambda_threshold) for q in aggregated_q[layer_idx]]
        k_analyses = [analyze_massive_values(k, lambda_threshold) for k in aggregated_k[layer_idx]]
        v_analyses = [analyze_massive_values(v, lambda_threshold) for v in aggregated_v[layer_idx]]

        # Use the last sample's analysis for detailed info, average for counts
        q_analysis = q_analyses[-1]
        k_analysis = k_analyses[-1]
        v_analysis = v_analyses[-1]

        # Average the counts across samples
        avg_q_massive = sum(a.num_massive for a in q_analyses) / len(q_analyses)
        avg_k_massive = sum(a.num_massive for a in k_analyses) / len(k_analyses)
        avg_v_massive = sum(a.num_massive for a in v_analyses) / len(v_analyses)

        avg_q_conc = sum(a.concentration_score for a in q_analyses) / len(q_analyses)
        avg_k_conc = sum(a.concentration_score for a in k_analyses) / len(k_analyses)
        avg_v_conc = sum(a.concentration_score for a in v_analyses) / len(v_analyses)

        layer_result = {
            "layer": layer_idx,
            "query": {
                "num_massive": avg_q_massive,
                "concentration": avg_q_conc,
                "threshold": q_analysis.threshold,
                "top_dims": q_analysis.top_dimensions[:5],
            },
            "key": {
                "num_massive": avg_k_massive,
                "concentration": avg_k_conc,
                "threshold": k_analysis.threshold,
                "top_dims": k_analysis.top_dimensions[:5],
            },
            "value": {
                "num_massive": avg_v_massive,
                "concentration": avg_v_conc,
                "threshold": v_analysis.threshold,
                "top_dims": v_analysis.top_dimensions[:5],
            },
        }
        all_results["layers"][layer_idx] = layer_result

        layer_results.append({
            "layer": layer_idx,
            "num_massive_q": avg_q_massive,
            "num_massive_k": avg_k_massive,
            "num_massive_v": avg_v_massive,
            "concentration_q": avg_q_conc,
            "concentration_k": avg_k_conc,
            "concentration_v": avg_v_conc,
        })

        # Generate visualizations for select layers
        if layer_idx in [0, len(aggregated_q) // 2, len(aggregated_q) - 1]:
            fig = plot_qkv_comparison(
                q_analysis, k_analysis, v_analysis,
                layer_idx,
                save_path=output_dir / f"qkv_layer_{layer_idx}.png"
            )
            plt.close(fig)

    # Plot layer progression

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Massive value counts
    layers = [r["layer"] for r in layer_results]
    axes[0].plot(layers, [r["num_massive_q"] for r in layer_results], "o-", label="Query")
    axes[0].plot(layers, [r["num_massive_k"] for r in layer_results], "s-", label="Key")
    axes[0].plot(layers, [r["num_massive_v"] for r in layer_results], "^-", label="Value")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Number of Massive Values")
    axes[0].set_title(f"Massive Values by Layer ({model_name})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Concentration scores
    axes[1].plot(layers, [r["concentration_q"] for r in layer_results], "o-", label="Query")
    axes[1].plot(layers, [r["concentration_k"] for r in layer_results], "s-", label="Key")
    axes[1].plot(layers, [r["concentration_v"] for r in layer_results], "^-", label="Value")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Concentration (Gini)")
    axes[1].set_title(f"Concentration by Layer ({model_name})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "layer_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save results
    with open(output_dir / "analysis_results.json", "w") as f:
        # Convert non-serializable items
        serializable_results = json.loads(json.dumps(all_results, default=str))
        json.dump(serializable_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_q = sum(r["num_massive_q"] for r in layer_results)
    total_k = sum(r["num_massive_k"] for r in layer_results)
    total_v = sum(r["num_massive_v"] for r in layer_results)

    print(f"Total massive values in Query:  {total_q}")
    print(f"Total massive values in Key:    {total_k}")
    print(f"Total massive values in Value:  {total_v}")
    print(f"\nRatio Q:K:V = {total_q}:{total_k}:{total_v}")

    if total_v > 0:
        print(f"Q/V ratio: {total_q/total_v:.2f}x")
        print(f"K/V ratio: {total_k/total_v:.2f}x")

    print(f"\nResults saved to: {output_dir}")

    return all_results, layer_results


def run_passkey_evaluation(
    model_name: str,
    output_dir: Path,
    context_lengths: list = [512, 1024],
    num_samples: int = 10,
    device: str = "cuda",
):
    """Run passkey retrieval evaluation."""
    print(f"\n{'='*60}")
    print(f"Passkey Retrieval: {model_name}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading {model_name}...")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()

    all_results = {
        "model": model_name,
        "context_lengths": {},
    }

    for ctx_len in context_lengths:
        print(f"\nContext length: {ctx_len}")

        config = PasskeyConfig(
            context_length=ctx_len,
            num_samples=num_samples,
        )
        evaluator = PasskeyRetrievalEvaluator(model, tokenizer, config)

        results = evaluator.evaluate(
            depths=[10, 50, 90],
            num_samples_per_depth=num_samples // 3,
        )

        all_results["context_lengths"][ctx_len] = {
            "overall_accuracy": results["overall_accuracy"],
            "per_depth": results["per_depth"],
        }

        print(f"  Overall accuracy: {results['overall_accuracy']:.2%}")
        for depth, stats in results["per_depth"].items():
            print(f"    Depth {depth}%: {stats['accuracy']:.2%}")

    # Save results
    with open(output_dir / "passkey_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return all_results


def run_comparison(
    rope_model: str,
    drope_model: str,
    output_dir: Path,
    device: str = "cuda",
):
    """Compare RoPE and DroPE models."""
    print(f"\n{'='*60}")
    print(f"Model Comparison: {rope_model} vs {drope_model}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis on both models
    rope_dir = output_dir / "rope"
    drope_dir = output_dir / "drope"

    print("\n--- RoPE Model ---")
    rope_results, rope_layers = run_massive_value_analysis(
        rope_model, rope_dir, device=device
    )

    print("\n--- DroPE Model ---")
    drope_results, drope_layers = run_massive_value_analysis(
        drope_model, drope_dir, device=device
    )

    # Generate comparison plots

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = [r["layer"] for r in rope_layers]

    # Query comparison
    axes[0].plot(layers, [r["num_massive_q"] for r in rope_layers], "o-", label="RoPE Query")
    axes[0].plot(layers, [r["num_massive_q"] for r in drope_layers], "s--", label="DroPE Query")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Number of Massive Values")
    axes[0].set_title("Query: RoPE vs DroPE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Key comparison
    axes[1].plot(layers, [r["num_massive_k"] for r in rope_layers], "o-", label="RoPE Key")
    axes[1].plot(layers, [r["num_massive_k"] for r in drope_layers], "s--", label="DroPE Key")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Number of Massive Values")
    axes[1].set_title("Key: RoPE vs DroPE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "rope_vs_drope_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nComparison saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run DroPE massive value experiments")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Phase to run")
    parser.add_argument("--model", type=str, default="smollm-360m", help="Model name")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.phase == 1:
        # Phase 1: Baseline analysis
        run_massive_value_analysis(
            args.model,
            output_dir / f"phase1/{args.model}_{timestamp}",
            num_samples=args.num_samples,
            device=args.device,
        )
        run_passkey_evaluation(
            args.model,
            output_dir / f"phase1/{args.model}_{timestamp}/passkey",
            context_lengths=[512],
            num_samples=9,
            device=args.device,
        )

    elif args.phase == 2:
        # Phase 2: RoPE vs DroPE comparison
        drope_model = f"{args.model}-drope"
        run_comparison(
            args.model,
            drope_model,
            output_dir / f"phase2/{args.model}_{timestamp}",
            device=args.device,
        )


if __name__ == "__main__":
    main()
