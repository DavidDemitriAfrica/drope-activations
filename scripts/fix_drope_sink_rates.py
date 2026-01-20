#!/usr/bin/env python3
"""
Fix DroPE sink rate measurements by rerunning with eager attention enabled.

This script:
1. Loads DroPE with eager attention (required for output_attentions=True)
2. Computes baseline metrics including proper sink rates
3. Updates the existing JSON file with corrected values
"""

import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import numpy as np

# Add repo root to path for custom_models
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# Pre-import custom_models for DroPE
import custom_models
import custom_models.attention
import custom_models.drope

DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"
RESULTS_FILE = REPO_ROOT / "results" / "phase_metrics" / "rope_vs_drope_phase_metrics.json"

# Same eval prompts as main script
EVAL_PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent is used to",
    "The theory of relativity states that",
    "Once upon a time in a land far away",
    "The most important thing about programming is",
    "When analyzing data, it is crucial to",
    "The fundamental theorem of calculus shows that",
    "In the context of neural networks, backpropagation",
    "The quick brown fox jumps over",
    "To be or not to be, that is",
]


def compute_sink_metrics(attn_weights, tau=0.3):
    """
    Compute attention sink metrics for a single layer.

    Args:
        attn_weights: [batch, heads, seq, seq] attention weights
        tau: threshold for sink classification

    Returns:
        sink_rate: fraction of heads that are sinks
        sink_scores: per-head sink scores
    """
    # attn_weights[b, h, i, j] = attention from position i to position j
    # BOS is at position 0
    # sink_score = average attention to BOS across all query positions

    bos_attention = attn_weights[0, :, :, 0]  # [heads, seq] - attention to position 0
    sink_scores = bos_attention.mean(dim=1)  # [heads] - average across query positions
    sink_scores_list = sink_scores.tolist()

    # A head is a "sink" if its average attention to BOS >= tau
    sink_rate = (sink_scores >= tau).float().mean().item()

    return sink_rate, sink_scores_list


def compute_entropy(hidden_states):
    """Compute normalized entropy of singular values."""
    # hidden_states: [batch, seq, hidden]
    h = hidden_states[0]  # [seq, hidden]

    # SVD
    try:
        _, s, _ = torch.linalg.svd(h, full_matrices=False)
        # Normalize singular values to get distribution
        s_norm = s / s.sum()
        # Compute entropy
        entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum().item()
        # Normalize by max entropy (log of rank)
        max_entropy = np.log(len(s))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return normalized_entropy
    except:
        return 0.0


def compute_anisotropy(hidden_states):
    """Compute anisotropy (1 - normalized entropy)."""
    return 1.0 - compute_entropy(hidden_states)


def analyze_drope_baseline(model, tokenizer, prompts, seq_len=128, tau=0.3):
    """
    Analyze DroPE baseline metrics with proper attention capture.
    """
    all_metrics = {
        "bos_norm": [],
        "bos_ratio": [],
        "entropy": [],
        "anisotropy": [],
        "sink_rate": [],
    }

    for prompt in tqdm(prompts, desc="DroPE Baseline (with attention)"):
        # Tokenize without padding to avoid DroPE issues
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
            )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        n_layers = len(hidden_states) - 1

        # Per-layer metrics for this prompt
        bos_norms = []
        bos_ratios = []
        entropies = []
        anisotropies = []
        sink_rates = []

        for layer_idx in range(n_layers):
            h = hidden_states[layer_idx + 1]  # +1 to skip embedding layer

            # BOS norm
            bos_repr = h[0, 0, :]  # [hidden]
            bos_norm = torch.norm(bos_repr).item()
            bos_norms.append(bos_norm)

            # BOS ratio
            other_norms = torch.norm(h[0, 1:, :], dim=1)
            mean_other = other_norms.mean().item() if len(other_norms) > 0 else 1.0
            bos_ratio = bos_norm / mean_other if mean_other > 0 else 0.0
            bos_ratios.append(bos_ratio)

            # Entropy
            entropy = compute_entropy(h)
            entropies.append(entropy)

            # Anisotropy
            anisotropy = 1.0 - entropy
            anisotropies.append(anisotropy)

            # Sink rate (now properly computed!)
            attn = attentions[layer_idx]
            sink_rate, _ = compute_sink_metrics(attn, tau=tau)
            sink_rates.append(sink_rate)

        all_metrics["bos_norm"].append(bos_norms)
        all_metrics["bos_ratio"].append(bos_ratios)
        all_metrics["entropy"].append(entropies)
        all_metrics["anisotropy"].append(anisotropies)
        all_metrics["sink_rate"].append(sink_rates)

    # Aggregate across prompts
    aggregated = {}
    for key in all_metrics:
        values = np.array(all_metrics[key])  # [n_prompts, n_layers]
        aggregated[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }

    return aggregated


def main():
    print("=" * 60)
    print("Fixing DroPE Sink Rate Measurements")
    print("=" * 60)

    # Load existing results
    print(f"\nLoading existing results from {RESULTS_FILE}")
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # Check current DroPE sink rates
    drope_sink = results["DroPE"]["baseline"]["sink_rate"]["mean"]
    print(f"Current DroPE sink rates (first 5 layers): {drope_sink[:5]}")
    print(f"Current DroPE average sink rate: {np.mean(drope_sink):.2%}")

    # Load DroPE with eager attention
    print(f"\nLoading {DROPE_MODEL} with eager attention...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        DROPE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Required for output_attentions=True
    )
    tokenizer = AutoTokenizer.from_pretrained(DROPE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Run baseline analysis with attention capture
    print("\nComputing baseline metrics with attention capture...")
    new_baseline = analyze_drope_baseline(model, tokenizer, EVAL_PROMPTS)

    # Show comparison
    print("\n" + "=" * 60)
    print("Sink Rate Comparison (Old vs New)")
    print("=" * 60)
    print(f"{'Layer':<8} {'Old':<12} {'New':<12}")
    print("-" * 32)
    for i in range(32):
        old_val = results["DroPE"]["baseline"]["sink_rate"]["mean"][i]
        new_val = new_baseline["sink_rate"]["mean"][i]
        print(f"{i:<8} {old_val:<12.2%} {new_val:<12.2%}")

    old_avg = np.mean(results["DroPE"]["baseline"]["sink_rate"]["mean"])
    new_avg = np.mean(new_baseline["sink_rate"]["mean"])
    print("-" * 32)
    print(f"{'Average':<8} {old_avg:<12.2%} {new_avg:<12.2%}")

    # Update results
    print("\nUpdating JSON with corrected values...")
    results["DroPE"]["baseline"] = new_baseline

    # Save updated results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULTS_FILE}")

    # Also print RoPE comparison
    rope_avg = np.mean(results["RoPE"]["baseline"]["sink_rate"]["mean"])
    print(f"\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)
    print(f"RoPE average sink rate:  {rope_avg:.2%}")
    print(f"DroPE average sink rate: {new_avg:.2%}")
    print(f"Reduction: {rope_avg/new_avg:.1f}x" if new_avg > 0 else "DroPE has 0% sinks")


if __name__ == "__main__":
    main()
