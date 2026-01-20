#!/usr/bin/env python3
"""
Rerun DroPE phase metrics with proper handling.

Fixes bugs from original run:
1. Entropy/anisotropy returning 0.0/1.0 due to silent failures
2. Sink rates computed via manual Q/K hooks (eager attention produces NaN)
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import custom_models
import custom_models.attention
import custom_models.drope

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"
ROPE_MODEL = "meta-llama/Llama-2-7b-hf"
RESULTS_FILE = REPO_ROOT / "results" / "phase_metrics" / "rope_vs_drope_phase_metrics.json"

EVAL_PROMPTS = [
    "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols.",
    "In physics, the theory of relativity encompasses two interrelated physics theories by Albert Einstein: special relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity. General relativity explains the law of gravitation and its relation to the forces of nature.",
    "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest.",
    "To implement a transformer model, you need to understand the self-attention mechanism. The attention function computes a weighted sum of values, where the weights are determined by the compatibility of queries and keys. This can be expressed as Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V.",
    "Machine learning models can be broadly categorized into supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the model learns from labeled examples. In unsupervised learning, the model discovers patterns in unlabeled data.",
    "User: What is the capital of France? Assistant: The capital of France is Paris. Paris is not only the capital but also the largest city in France, located in the north-central part of the country along the Seine River.",
    "User: How do I make pasta? Assistant: To make pasta, you'll need flour, eggs, and a pinch of salt. Mix the ingredients to form a dough, knead it until smooth, roll it out thin, and cut it into your desired shape.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.",
    "Problem: If a train travels at 60 miles per hour for 2.5 hours, how far does it travel? Solution: Distance equals speed multiplied by time. So the distance is 60 * 2.5 = 150 miles.",
]

EPS = 1e-10


def compute_bos_metrics(hidden_states):
    """Compute BOS norm and ratio."""
    X = hidden_states[0].float()
    bos_norm = X[0].norm(p=2).item()
    if X.shape[0] > 1:
        non_bos_norms = X[1:].norm(p=2, dim=-1)
        non_bos_mean = non_bos_norms.mean().item()
        bos_ratio = bos_norm / (non_bos_mean + 1e-8)
    else:
        bos_ratio = 1.0
    return bos_norm, bos_ratio


def compute_entropy_metrics(hidden_states):
    """Compute entropy and anisotropy from singular values."""
    X = hidden_states[0].float()
    C = X @ X.T
    frobenius_sq = torch.trace(C).item()

    if frobenius_sq < EPS:
        print(f"  WARNING: frobenius_sq={frobenius_sq} < eps")
        return 0.0, 1.0

    try:
        eigenvalues = torch.linalg.eigvalsh(C)
        eigenvalues = eigenvalues.flip(0)  # descending
        eigenvalues = eigenvalues.clamp(min=0)
    except Exception as e:
        print(f"  WARNING: eigenvalue computation failed: {e}")
        return 0.0, 1.0

    if eigenvalues.isnan().any():
        print(f"  WARNING: NaN in eigenvalues")
        return 0.0, 1.0

    p = eigenvalues / frobenius_sq
    p = p.clamp(min=EPS)

    entropy = -torch.sum(p * torch.log(p)).item()
    anisotropy = p[0].item()

    return entropy, anisotropy


def compute_sink_rate_manual(model, tokenizer, prompt, layer_idx, tau=0.3):
    """Compute sink rate using manual Q/K hooks (avoids eager attention NaN)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    captured = {}

    def capture_hook(name):
        def hook(module, input, output):
            captured[name] = output.detach()
        return hook

    layer = model.model.layers[layer_idx]
    h1 = layer.self_attn.q_proj.register_forward_hook(capture_hook('q'))
    h2 = layer.self_attn.k_proj.register_forward_hook(capture_hook('k'))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        h1.remove()
        h2.remove()

    q = captured['q'].view(1, seq_len, num_heads, head_dim).transpose(1, 2).float()
    k = captured['k'].view(1, seq_len, num_heads, head_dim).transpose(1, 2).float()

    scale = head_dim ** -0.5
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=model.device, dtype=torch.bool), diagonal=1)
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Sink rate: fraction of heads with mean attention to BOS >= tau
    bos_attention = attn_weights[0, :, :, 0].mean(dim=1)  # [heads]
    sink_rate = (bos_attention >= tau).float().mean().item()

    return sink_rate


def run_analysis(model_name, model_key, recompute_sink=False):
    """Run full analysis for a model."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_key}")
    print(f"{'='*60}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers

    all_metrics = {
        "bos_norm": [],
        "bos_ratio": [],
        "entropy": [],
        "anisotropy": [],
        "sink_rate": [],
    }

    for prompt in tqdm(EVAL_PROMPTS, desc=f"{model_key} Baseline"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        bos_norms = []
        bos_ratios = []
        entropies = []
        anisotropies = []
        sink_rates = []

        for layer_idx in range(n_layers):
            hs = hidden_states[layer_idx + 1]

            bos_norm, bos_ratio = compute_bos_metrics(hs)
            entropy, anisotropy = compute_entropy_metrics(hs)

            bos_norms.append(bos_norm)
            bos_ratios.append(bos_ratio)
            entropies.append(entropy)
            anisotropies.append(anisotropy)

        # Sink rates via manual method for DroPE, or standard for RoPE
        if recompute_sink:
            for layer_idx in range(n_layers):
                sink_rate = compute_sink_rate_manual(model, tokenizer, prompt, layer_idx)
                sink_rates.append(sink_rate)
        else:
            sink_rates = [0.0] * n_layers  # Will use existing data

        all_metrics["bos_norm"].append(bos_norms)
        all_metrics["bos_ratio"].append(bos_ratios)
        all_metrics["entropy"].append(entropies)
        all_metrics["anisotropy"].append(anisotropies)
        if recompute_sink:
            all_metrics["sink_rate"].append(sink_rates)

    # Aggregate
    aggregated = {}
    for key in all_metrics:
        if key == "sink_rate" and not recompute_sink:
            continue
        values = np.array(all_metrics[key])
        aggregated[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }

    del model
    torch.cuda.empty_cache()

    return aggregated


def main():
    print("="*60)
    print("Rerunning DroPE Phase Metrics")
    print("="*60)

    # Load existing results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # Show current suspicious values
    print("\nCurrent DroPE values (suspicious):")
    print(f"  Entropy first 5: {results['DroPE']['baseline']['entropy']['mean'][:5]}")
    print(f"  Anisotropy first 5: {results['DroPE']['baseline']['anisotropy']['mean'][:5]}")

    # Rerun DroPE (sink rates already fixed, just need entropy/anisotropy/bos)
    drope_metrics = run_analysis(DROPE_MODEL, "DroPE", recompute_sink=False)

    # Update results (keep existing sink_rate which was fixed separately)
    existing_sink = results["DroPE"]["baseline"]["sink_rate"]
    results["DroPE"]["baseline"].update(drope_metrics)
    results["DroPE"]["baseline"]["sink_rate"] = existing_sink

    # Print corrected values
    print("\n" + "="*60)
    print("Corrected DroPE values:")
    print("="*60)
    print(f"  Entropy first 5: {[f'{v:.3f}' for v in drope_metrics['entropy']['mean'][:5]]}")
    print(f"  Entropy min: {min(drope_metrics['entropy']['mean']):.3f}")
    print(f"  Entropy max: {max(drope_metrics['entropy']['mean']):.3f}")
    print(f"  Anisotropy first 5: {[f'{v:.3f}' for v in drope_metrics['anisotropy']['mean'][:5]]}")
    print(f"  Anisotropy min: {min(drope_metrics['anisotropy']['mean']):.3f}")
    print(f"  Anisotropy max: {max(drope_metrics['anisotropy']['mean']):.3f}")

    # Compare with RoPE
    print("\n" + "="*60)
    print("Comparison with RoPE:")
    print("="*60)
    rope_entropy = results["RoPE"]["baseline"]["entropy"]["mean"]
    drope_entropy = drope_metrics["entropy"]["mean"]
    print(f"  RoPE entropy range: {min(rope_entropy):.3f} - {max(rope_entropy):.3f}")
    print(f"  DroPE entropy range: {min(drope_entropy):.3f} - {max(drope_entropy):.3f}")

    rope_aniso = results["RoPE"]["baseline"]["anisotropy"]["mean"]
    drope_aniso = drope_metrics["anisotropy"]["mean"]
    print(f"  RoPE anisotropy range: {min(rope_aniso):.3f} - {max(rope_aniso):.3f}")
    print(f"  DroPE anisotropy range: {min(drope_aniso):.3f} - {max(drope_aniso):.3f}")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
