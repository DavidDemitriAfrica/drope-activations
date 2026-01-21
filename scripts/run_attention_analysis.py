#!/usr/bin/env python3
"""
Experiment 7: Attention Pattern Analysis

Goes beyond sink rates to understand actual attention patterns.
Both models have ~97% sink rates but behave completely differently
under BOS-MLP ablation (1249× PPL for RoPE vs 1.00× for DroPE).

Analyses:
1. Attention entropy per head (focused vs distributed)
2. Head clustering (sink heads vs local heads vs content heads)
3. Full attention matrices for visualization
4. Position-dependent attention profiles
5. Attention decay with distance
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import custom_models
import custom_models.attention
import custom_models.drope

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROPE_MODEL = "meta-llama/Llama-2-7b-hf"
DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"
RESULTS_FILE = REPO_ROOT / "results" / "phase_metrics" / "rope_vs_drope_phase_metrics.json"
OUTPUT_DIR = REPO_ROOT / "results" / "phase_metrics"


class FullAttentionCapture:
    """Capture full attention matrices by computing from Q/K projections."""

    def __init__(self, model, num_heads=32, head_dim=128):
        self.model = model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_outputs = {}
        self.k_outputs = {}
        self.handles = []

    def _q_hook(self, layer_idx):
        def hook(module, input, output):
            self.q_outputs[layer_idx] = output.detach()
        return hook

    def _k_hook(self, layer_idx):
        def hook(module, input, output):
            self.k_outputs[layer_idx] = output.detach()
        return hook

    def register_hooks(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            h_q = layer.self_attn.q_proj.register_forward_hook(self._q_hook(layer_idx))
            h_k = layer.self_attn.k_proj.register_forward_hook(self._k_hook(layer_idx))
            self.handles.extend([h_q, h_k])

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def compute_attention_weights(self):
        """Compute full attention weight matrices from Q, K."""
        attention_weights = {}

        for layer_idx in self.q_outputs:
            q = self.q_outputs[layer_idx]  # [batch, seq, hidden]
            k = self.k_outputs[layer_idx]  # [batch, seq, hidden]

            batch, seq_len, hidden = q.shape

            # Reshape to per-head
            q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute attention scores (scaled dot product)
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Clamp extreme scores (necessary for DroPE's massive activations)
            scores = torch.clamp(scores, min=-100, max=100)

            # Apply causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)  # [batch, heads, seq, seq]
            attention_weights[layer_idx] = attn_weights[0].cpu()  # [heads, seq, seq]

        return attention_weights

    def clear(self):
        self.q_outputs = {}
        self.k_outputs = {}


def compute_attention_entropy(attn_weights):
    """
    Compute entropy of attention distribution for each query position.

    Args:
        attn_weights: [heads, seq, seq] attention weights

    Returns:
        entropy: [heads] mean entropy per head
    """
    # attn_weights[h, q, k] = attention from query q to key k
    # For each query position, compute entropy of attention distribution
    eps = 1e-10
    # Mask out zeros (from causal mask)
    attn_safe = attn_weights.clamp(min=eps)
    entropy = -torch.sum(attn_weights * torch.log(attn_safe), dim=-1)  # [heads, seq]
    # Average over query positions
    mean_entropy = entropy.mean(dim=-1)  # [heads]
    return mean_entropy.numpy()


def compute_bos_attention(attn_weights):
    """
    Compute mean attention to BOS (position 0) per head.

    Args:
        attn_weights: [heads, seq, seq]

    Returns:
        bos_attn: [heads] mean attention to BOS
    """
    # Attention to BOS = weights[:, :, 0]
    attn_to_bos = attn_weights[:, 1:, 0]  # Exclude BOS querying itself
    return attn_to_bos.mean(dim=1).numpy()


def compute_local_attention(attn_weights, window=5):
    """
    Compute mean attention to local window (last `window` tokens).

    Args:
        attn_weights: [heads, seq, seq]
        window: size of local window

    Returns:
        local_attn: [heads] mean local attention
    """
    heads, seq, _ = attn_weights.shape
    local_attn = []

    for h in range(heads):
        total_local = 0
        count = 0
        for q in range(1, seq):  # Skip BOS
            # Local window: max(0, q-window) to q (exclusive of q itself due to causal)
            start = max(0, q - window)
            end = q
            if end > start:
                total_local += attn_weights[h, q, start:end].sum().item()
                count += 1
        local_attn.append(total_local / count if count > 0 else 0)

    return np.array(local_attn)


def compute_attention_decay(attn_weights, max_distance=50):
    """
    Compute attention decay profile with distance.

    Args:
        attn_weights: [heads, seq, seq]
        max_distance: maximum distance to consider

    Returns:
        decay_profile: [heads, max_distance] attention at each distance
    """
    heads, seq, _ = attn_weights.shape
    decay_profile = np.zeros((heads, max_distance))
    counts = np.zeros((heads, max_distance))

    for h in range(heads):
        for q in range(seq):
            for k in range(q + 1):  # k <= q due to causal mask
                dist = q - k
                if dist < max_distance:
                    decay_profile[h, dist] += attn_weights[h, q, k].item()
                    counts[h, dist] += 1

    # Average
    decay_profile = np.divide(decay_profile, counts, where=counts > 0)
    return decay_profile


def classify_heads(bos_attn, local_attn, entropy, bos_threshold=0.3, local_threshold=0.3):
    """
    Classify heads into categories.

    Categories:
    - sink: High BOS attention (>= bos_threshold)
    - local: High local attention (>= local_threshold) and not sink
    - distributed: High entropy, not sink or local
    - mixed: Other

    Returns:
        classifications: [heads] category labels
    """
    n_heads = len(bos_attn)
    classifications = []

    for h in range(n_heads):
        if bos_attn[h] >= bos_threshold:
            classifications.append("sink")
        elif local_attn[h] >= local_threshold:
            classifications.append("local")
        elif entropy[h] > np.median(entropy):
            classifications.append("distributed")
        else:
            classifications.append("mixed")

    return classifications


def analyze_attention_patterns(model, tokenizer, texts, num_heads=32, head_dim=128):
    """Analyze attention patterns across multiple texts."""

    capture = FullAttentionCapture(model, num_heads=num_heads, head_dim=head_dim)
    capture.register_hooks()

    num_layers = len(model.model.layers)

    # Accumulators
    all_entropy = defaultdict(list)
    all_bos_attn = defaultdict(list)
    all_local_attn = defaultdict(list)
    all_decay = defaultdict(list)
    sample_matrices = {}  # Store sample attention matrices for visualization

    for i, text in enumerate(tqdm(texts, desc="Analyzing attention")):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.device)

        capture.clear()

        with torch.no_grad():
            outputs = model(input_ids)

        attn_weights = capture.compute_attention_weights()

        for layer_idx in range(num_layers):
            if layer_idx not in attn_weights:
                continue

            aw = attn_weights[layer_idx]  # [heads, seq, seq]

            # Compute metrics
            entropy = compute_attention_entropy(aw)
            bos_attn = compute_bos_attention(aw)
            local_attn = compute_local_attention(aw)
            decay = compute_attention_decay(aw)

            all_entropy[layer_idx].append(entropy)
            all_bos_attn[layer_idx].append(bos_attn)
            all_local_attn[layer_idx].append(local_attn)
            all_decay[layer_idx].append(decay)

            # Store sample matrices for key layers (first text only)
            if i == 0 and layer_idx in [0, 1, 15, 31]:
                sample_matrices[layer_idx] = aw.numpy()

    capture.remove_hooks()

    # Average across texts
    results = {
        "entropy": {},
        "bos_attention": {},
        "local_attention": {},
        "decay_profile": {},
        "head_classifications": {},
        "sample_matrices": sample_matrices,
    }

    for layer_idx in range(num_layers):
        if layer_idx in all_entropy:
            results["entropy"][layer_idx] = np.mean(all_entropy[layer_idx], axis=0).tolist()
            results["bos_attention"][layer_idx] = np.mean(all_bos_attn[layer_idx], axis=0).tolist()
            results["local_attention"][layer_idx] = np.mean(all_local_attn[layer_idx], axis=0).tolist()
            results["decay_profile"][layer_idx] = np.mean(all_decay[layer_idx], axis=0).tolist()

            # Classify heads
            entropy = np.mean(all_entropy[layer_idx], axis=0)
            bos = np.mean(all_bos_attn[layer_idx], axis=0)
            local = np.mean(all_local_attn[layer_idx], axis=0)
            results["head_classifications"][layer_idx] = classify_heads(bos, local, entropy)

    return results


def compute_summary_stats(results, num_layers=32):
    """Compute summary statistics across layers."""

    summary = {
        "mean_entropy_by_layer": [],
        "mean_bos_attn_by_layer": [],
        "mean_local_attn_by_layer": [],
        "head_type_counts": {"sink": 0, "local": 0, "distributed": 0, "mixed": 0},
        "head_type_by_layer": {},
    }

    for layer_idx in range(num_layers):
        if layer_idx in results["entropy"]:
            summary["mean_entropy_by_layer"].append(np.mean(results["entropy"][layer_idx]))
            summary["mean_bos_attn_by_layer"].append(np.mean(results["bos_attention"][layer_idx]))
            summary["mean_local_attn_by_layer"].append(np.mean(results["local_attention"][layer_idx]))

            # Count head types
            classifications = results["head_classifications"][layer_idx]
            layer_counts = {"sink": 0, "local": 0, "distributed": 0, "mixed": 0}
            for c in classifications:
                summary["head_type_counts"][c] += 1
                layer_counts[c] += 1
            summary["head_type_by_layer"][layer_idx] = layer_counts

    return summary


def main():
    print("=" * 60)
    print("Experiment 7: Attention Pattern Analysis")
    print("=" * 60)

    # Load existing results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Test texts
    texts = [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms.",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
        "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse.",
        "All happy families are alike; each unhappy family is unhappy in its own way. Everything was in confusion.",
        "The Nellie, a cruising yawl, swung to her anchor without a flutter of the sails, and was at rest.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune must be in want of a wife.",
        "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer.",
    ]

    for model_name, model_key in [(ROPE_MODEL, "RoPE"), (DROPE_MODEL, "DroPE")]:
        print(f"\n{'='*60}")
        print(f"Analyzing {model_key}")
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
        num_layers = model.config.num_hidden_layers

        # Analyze attention patterns
        attn_results = analyze_attention_patterns(
            model, tokenizer, texts,
            num_heads=num_heads, head_dim=head_dim
        )

        # Compute summary
        summary = compute_summary_stats(attn_results, num_layers=num_layers)

        # Store results (excluding large matrices for JSON)
        if "attention_analysis" not in results[model_key]:
            results[model_key]["attention_analysis"] = {}

        results[model_key]["attention_analysis"]["entropy_by_layer_head"] = attn_results["entropy"]
        results[model_key]["attention_analysis"]["bos_attention_by_layer_head"] = attn_results["bos_attention"]
        results[model_key]["attention_analysis"]["local_attention_by_layer_head"] = attn_results["local_attention"]
        results[model_key]["attention_analysis"]["decay_profile_by_layer"] = attn_results["decay_profile"]
        results[model_key]["attention_analysis"]["head_classifications"] = attn_results["head_classifications"]
        results[model_key]["attention_analysis"]["summary"] = summary

        # Save attention matrices separately (numpy format)
        np.savez(
            OUTPUT_DIR / f"{model_key.lower()}_attention_matrices.npz",
            **{f"layer_{k}": v for k, v in attn_results["sample_matrices"].items()}
        )

        # Print summary
        print(f"\n--- Summary for {model_key} ---")
        print(f"Mean entropy across layers: {np.mean(summary['mean_entropy_by_layer']):.3f}")
        print(f"Mean BOS attention: {np.mean(summary['mean_bos_attn_by_layer']):.3f}")
        print(f"Mean local attention: {np.mean(summary['mean_local_attn_by_layer']):.3f}")
        print(f"\nHead type counts (across all layers):")
        for htype, count in summary["head_type_counts"].items():
            pct = count / (num_layers * num_heads) * 100
            print(f"  {htype}: {count} ({pct:.1f}%)")

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Attention Pattern Analysis")
    print("=" * 60)

    rope_summary = results["RoPE"]["attention_analysis"]["summary"]
    drope_summary = results["DroPE"]["attention_analysis"]["summary"]

    print("\nMean Metrics (averaged across layers):")
    print(f"{'Metric':<25} {'RoPE':>15} {'DroPE':>15}")
    print("-" * 55)
    print(f"{'Entropy':<25} {np.mean(rope_summary['mean_entropy_by_layer']):>15.3f} {np.mean(drope_summary['mean_entropy_by_layer']):>15.3f}")
    print(f"{'BOS Attention':<25} {np.mean(rope_summary['mean_bos_attn_by_layer']):>15.3f} {np.mean(drope_summary['mean_bos_attn_by_layer']):>15.3f}")
    print(f"{'Local Attention':<25} {np.mean(rope_summary['mean_local_attn_by_layer']):>15.3f} {np.mean(drope_summary['mean_local_attn_by_layer']):>15.3f}")

    print("\nHead Type Distribution:")
    print(f"{'Type':<15} {'RoPE':>15} {'DroPE':>15}")
    print("-" * 45)
    total_heads = 32 * 32  # 32 layers × 32 heads
    for htype in ["sink", "local", "distributed", "mixed"]:
        rope_pct = rope_summary["head_type_counts"][htype] / total_heads * 100
        drope_pct = drope_summary["head_type_counts"][htype] / total_heads * 100
        print(f"{htype:<15} {rope_pct:>14.1f}% {drope_pct:>14.1f}%")

    # Layer 1 comparison (the key layer for DroPE)
    print("\nLayer 1 Attention (Key DroPE Layer):")
    print(f"{'Metric':<25} {'RoPE':>15} {'DroPE':>15}")
    print("-" * 55)

    rope_l1 = results["RoPE"]["attention_analysis"]
    drope_l1 = results["DroPE"]["attention_analysis"]

    if "1" in rope_l1["entropy_by_layer_head"] or 1 in rope_l1["entropy_by_layer_head"]:
        l1_key = "1" if "1" in rope_l1["entropy_by_layer_head"] else 1
        print(f"{'Mean Entropy':<25} {np.mean(rope_l1['entropy_by_layer_head'][l1_key]):>15.3f} {np.mean(drope_l1['entropy_by_layer_head'][l1_key]):>15.3f}")
        print(f"{'Mean BOS Attention':<25} {np.mean(rope_l1['bos_attention_by_layer_head'][l1_key]):>15.3f} {np.mean(drope_l1['bos_attention_by_layer_head'][l1_key]):>15.3f}")
        print(f"{'Mean Local Attention':<25} {np.mean(rope_l1['local_attention_by_layer_head'][l1_key]):>15.3f} {np.mean(drope_l1['local_attention_by_layer_head'][l1_key]):>15.3f}")


if __name__ == "__main__":
    main()
