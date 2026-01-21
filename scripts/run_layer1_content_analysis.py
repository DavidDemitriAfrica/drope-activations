#!/usr/bin/env python3
"""
Experiment 8: Layer 1 Attention Content Analysis

Why is DroPE Layer 1 attention critical (201×) when RoPE's is not (2.5×)?
Both have ~93% sink heads, yet DroPE needs Layer 1 attention while RoPE doesn't.

Key question: What information is Layer 1 attention writing to the residual stream?

Analyses:
1. Attention output norms (how much is being written)
2. MLP output norms (comparison)
3. Attention/MLP contribution ratio to residual stream
4. Per-head contribution analysis
5. Cosine similarity between attention output and residual stream
6. Information content: does attention output correlate with final predictions?
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

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


class Layer1ContentCapture:
    """Capture Layer 1 attention and MLP outputs to analyze what's being written."""

    def __init__(self, model, num_heads=32, head_dim=128):
        self.model = model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim

        # Storage
        self.attn_output = None  # After o_proj
        self.mlp_output = None   # MLP output
        self.hidden_before = None  # Input to layer 1
        self.hidden_after = None   # Output of layer 1
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

        self.handles = []

    def _attn_output_hook(self, module, input, output):
        """Capture attention module output (before residual add)."""
        if isinstance(output, tuple):
            self.attn_output = output[0].detach()
        else:
            self.attn_output = output.detach()

    def _mlp_output_hook(self, module, input, output):
        """Capture MLP output (before residual add)."""
        self.mlp_output = output.detach()

    def _layer_input_hook(self, module, input, output):
        """Capture input to layer 1."""
        if isinstance(input, tuple):
            self.hidden_before = input[0].detach()
        else:
            self.hidden_before = input.detach()

    def _layer_output_hook(self, module, input, output):
        """Capture output of layer 1."""
        if isinstance(output, tuple):
            self.hidden_after = output[0].detach()
        else:
            self.hidden_after = output.detach()

    def _q_hook(self, module, input, output):
        self.q_proj = output.detach()

    def _k_hook(self, module, input, output):
        self.k_proj = output.detach()

    def _v_hook(self, module, input, output):
        self.v_proj = output.detach()

    def register_hooks(self, layer_idx=1):
        """Register hooks on layer 1 components."""
        layer = self.model.model.layers[layer_idx]

        # Attention output (after o_proj, before residual)
        h1 = layer.self_attn.register_forward_hook(self._attn_output_hook)

        # MLP output
        h2 = layer.mlp.register_forward_hook(self._mlp_output_hook)

        # Layer input/output for residual analysis
        h3 = layer.register_forward_hook(self._layer_output_hook)

        # Q, K, V projections
        h4 = layer.self_attn.q_proj.register_forward_hook(self._q_hook)
        h5 = layer.self_attn.k_proj.register_forward_hook(self._k_hook)
        h6 = layer.self_attn.v_proj.register_forward_hook(self._v_hook)

        # Input to layer (use input_layernorm as proxy)
        h7 = layer.input_layernorm.register_forward_hook(
            lambda m, i, o: setattr(self, 'hidden_before', i[0].detach() if isinstance(i, tuple) else i.detach())
        )

        self.handles = [h1, h2, h3, h4, h5, h6, h7]

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self):
        self.attn_output = None
        self.mlp_output = None
        self.hidden_before = None
        self.hidden_after = None
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None


def compute_content_metrics(capture, num_heads=32, head_dim=128):
    """Compute metrics about what Layer 1 is writing."""
    metrics = {}

    # Basic norms
    if capture.attn_output is not None:
        attn_out = capture.attn_output[0]  # [seq, hidden]
        metrics["attn_output_norm_mean"] = attn_out.norm(dim=-1).mean().item()
        metrics["attn_output_norm_std"] = attn_out.norm(dim=-1).std().item()
        metrics["attn_output_norm_bos"] = attn_out[0].norm().item()
        metrics["attn_output_norm_content"] = attn_out[1:].norm(dim=-1).mean().item()

    if capture.mlp_output is not None:
        mlp_out = capture.mlp_output[0]  # [seq, hidden]
        metrics["mlp_output_norm_mean"] = mlp_out.norm(dim=-1).mean().item()
        metrics["mlp_output_norm_std"] = mlp_out.norm(dim=-1).std().item()
        metrics["mlp_output_norm_bos"] = mlp_out[0].norm().item()
        metrics["mlp_output_norm_content"] = mlp_out[1:].norm(dim=-1).mean().item()

    # Contribution ratio (attention vs MLP)
    if capture.attn_output is not None and capture.mlp_output is not None:
        attn_norm = capture.attn_output[0].norm(dim=-1).mean().item()
        mlp_norm = capture.mlp_output[0].norm(dim=-1).mean().item()
        metrics["attn_mlp_ratio"] = attn_norm / (mlp_norm + 1e-10)
        metrics["attn_contribution_frac"] = attn_norm / (attn_norm + mlp_norm + 1e-10)

    # Cosine similarity: attention output direction vs MLP output direction
    if capture.attn_output is not None and capture.mlp_output is not None:
        attn_flat = capture.attn_output[0].flatten()
        mlp_flat = capture.mlp_output[0].flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            attn_flat.unsqueeze(0), mlp_flat.unsqueeze(0)
        ).item()
        metrics["attn_mlp_cosine_sim"] = cos_sim

    # Q, K, V norms (to understand attention mechanism)
    if capture.q_proj is not None:
        q = capture.q_proj[0]  # [seq, hidden]
        metrics["q_norm_mean"] = q.norm(dim=-1).mean().item()
        metrics["q_norm_bos"] = q[0].norm().item()
        metrics["q_norm_max"] = q.norm(dim=-1).max().item()

    if capture.k_proj is not None:
        k = capture.k_proj[0]
        metrics["k_norm_mean"] = k.norm(dim=-1).mean().item()
        metrics["k_norm_bos"] = k[0].norm().item()
        metrics["k_norm_max"] = k.norm(dim=-1).max().item()

    if capture.v_proj is not None:
        v = capture.v_proj[0]
        metrics["v_norm_mean"] = v.norm(dim=-1).mean().item()
        metrics["v_norm_bos"] = v[0].norm().item()
        metrics["v_norm_max"] = v.norm(dim=-1).max().item()

        # Per-head V analysis at BOS
        v_heads = v[0].view(num_heads, head_dim)  # [heads, dim]
        v_head_norms = v_heads.norm(dim=-1).cpu().numpy()
        metrics["v_bos_head_norms"] = v_head_norms.tolist()
        metrics["v_bos_head_norm_max"] = float(v_head_norms.max())
        metrics["v_bos_head_norm_mean"] = float(v_head_norms.mean())

    # Residual stream analysis
    if capture.hidden_before is not None and capture.hidden_after is not None:
        before = capture.hidden_before[0]  # [seq, hidden]
        after = capture.hidden_after[0]
        delta = after - before  # Total change from layer 1

        metrics["residual_delta_norm_mean"] = delta.norm(dim=-1).mean().item()
        metrics["residual_delta_norm_bos"] = delta[0].norm().item()

        # How much of the change is from attention vs MLP?
        if capture.attn_output is not None:
            attn_contrib = capture.attn_output[0].norm(dim=-1).mean().item()
            total_change = delta.norm(dim=-1).mean().item()
            # Note: not exactly additive due to layernorms, but gives intuition
            metrics["attn_frac_of_delta"] = attn_contrib / (total_change + 1e-10)

    return metrics


def analyze_per_head_contributions(capture, num_heads=32, head_dim=128):
    """Analyze which heads contribute most to attention output."""
    if capture.v_proj is None or capture.attn_output is None:
        return {}

    # We need attention weights to compute per-head contribution
    # For now, analyze V norms as proxy for potential contribution
    v = capture.v_proj[0]  # [seq, hidden]
    seq_len = v.shape[0]

    # Reshape to per-head
    v_heads = v.view(seq_len, num_heads, head_dim)  # [seq, heads, dim]

    # Per-head norms averaged over sequence
    head_norms = v_heads.norm(dim=-1).mean(dim=0).cpu().numpy()  # [heads]

    # BOS-specific per-head norms
    bos_head_norms = v_heads[0].norm(dim=-1).cpu().numpy()  # [heads]

    return {
        "head_v_norms": head_norms.tolist(),
        "head_v_norms_bos": bos_head_norms.tolist(),
        "head_v_norm_gini": compute_gini(head_norms),
        "head_v_norm_bos_gini": compute_gini(bos_head_norms),
    }


def compute_gini(x):
    """Compute Gini coefficient (0 = equal, 1 = concentrated)."""
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n


def run_content_analysis(model, tokenizer, texts, layer_idx=1):
    """Run content analysis on multiple texts."""
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    capture = Layer1ContentCapture(model, num_heads=num_heads, head_dim=head_dim)
    capture.register_hooks(layer_idx=layer_idx)

    all_metrics = defaultdict(list)
    all_head_metrics = defaultdict(list)

    for text in tqdm(texts, desc=f"Analyzing Layer {layer_idx}"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.device)

        capture.clear()

        with torch.no_grad():
            outputs = model(input_ids)

        # Compute metrics
        metrics = compute_content_metrics(capture, num_heads, head_dim)
        for k, v in metrics.items():
            if not isinstance(v, list):
                all_metrics[k].append(v)

        head_metrics = analyze_per_head_contributions(capture, num_heads, head_dim)
        for k, v in head_metrics.items():
            if isinstance(v, list):
                all_head_metrics[k].append(v)
            else:
                all_metrics[k].append(v)

    capture.remove_hooks()

    # Average metrics
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    # Average head metrics
    for k, v in all_head_metrics.items():
        avg_metrics[k] = np.mean(v, axis=0).tolist()

    return avg_metrics


def main():
    print("=" * 60)
    print("Experiment 8: Layer 1 Attention Content Analysis")
    print("Why is DroPE Layer 1 attention critical but RoPE's is not?")
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
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.",
        "It was the best of times, it was the worst of times, it was the age of wisdom.",
        "Call me Ishmael. Some years ago, never mind how long precisely.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "The Nellie, a cruising yawl, swung to her anchor without a flutter of the sails.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune.",
        "You don't know about me without you have read a book by the name of Tom Sawyer.",
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

        # Analyze Layer 1
        print("\n--- Layer 1 Content Analysis ---")
        layer1_metrics = run_content_analysis(model, tokenizer, texts, layer_idx=1)

        # Also analyze Layer 0 and Layer 2 for comparison
        print("\n--- Layer 0 Content Analysis (comparison) ---")
        layer0_metrics = run_content_analysis(model, tokenizer, texts, layer_idx=0)

        print("\n--- Layer 2 Content Analysis (comparison) ---")
        layer2_metrics = run_content_analysis(model, tokenizer, texts, layer_idx=2)

        # Store results
        if "layer_content_analysis" not in results[model_key]:
            results[model_key]["layer_content_analysis"] = {}

        results[model_key]["layer_content_analysis"]["layer_0"] = layer0_metrics
        results[model_key]["layer_content_analysis"]["layer_1"] = layer1_metrics
        results[model_key]["layer_content_analysis"]["layer_2"] = layer2_metrics

        # Print key metrics
        print(f"\n--- Key Metrics for {model_key} ---")
        print(f"\nLayer 1:")
        print(f"  Attention output norm (mean): {layer1_metrics.get('attn_output_norm_mean', 0):.2f}")
        print(f"  MLP output norm (mean): {layer1_metrics.get('mlp_output_norm_mean', 0):.2f}")
        print(f"  Attention/MLP ratio: {layer1_metrics.get('attn_mlp_ratio', 0):.3f}")
        print(f"  Attention contribution frac: {layer1_metrics.get('attn_contribution_frac', 0):.3f}")
        print(f"  Q norm (mean): {layer1_metrics.get('q_norm_mean', 0):.2f}")
        print(f"  K norm (mean): {layer1_metrics.get('k_norm_mean', 0):.2f}")
        print(f"  V norm (mean): {layer1_metrics.get('v_norm_mean', 0):.2f}")
        print(f"  V BOS head norm (max): {layer1_metrics.get('v_bos_head_norm_max', 0):.2f}")
        print(f"  Head V norm Gini: {layer1_metrics.get('head_v_norm_gini', 0):.3f}")

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Layer 1 Content Analysis")
    print("=" * 60)

    rope_l1 = results["RoPE"]["layer_content_analysis"]["layer_1"]
    drope_l1 = results["DroPE"]["layer_content_analysis"]["layer_1"]

    print(f"\n{'Metric':<35} {'RoPE':>15} {'DroPE':>15}")
    print("-" * 65)

    key_metrics = [
        ("attn_output_norm_mean", "Attn output norm (mean)"),
        ("attn_output_norm_bos", "Attn output norm (BOS)"),
        ("mlp_output_norm_mean", "MLP output norm (mean)"),
        ("mlp_output_norm_bos", "MLP output norm (BOS)"),
        ("attn_mlp_ratio", "Attn/MLP ratio"),
        ("attn_contribution_frac", "Attn contribution frac"),
        ("q_norm_mean", "Q norm (mean)"),
        ("q_norm_max", "Q norm (max)"),
        ("k_norm_mean", "K norm (mean)"),
        ("k_norm_max", "K norm (max)"),
        ("v_norm_mean", "V norm (mean)"),
        ("v_bos_head_norm_max", "V BOS head norm (max)"),
        ("head_v_norm_gini", "Head V norm Gini"),
        ("attn_mlp_cosine_sim", "Attn-MLP cosine sim"),
    ]

    for key, label in key_metrics:
        rope_val = rope_l1.get(key, 0)
        drope_val = drope_l1.get(key, 0)
        if isinstance(rope_val, float):
            print(f"{label:<35} {rope_val:>15.3f} {drope_val:>15.3f}")

    # Layer comparison
    print("\n" + "=" * 60)
    print("LAYER COMPARISON: Attention Output Norms")
    print("=" * 60)

    print(f"\n{'Layer':<10} {'RoPE Attn Norm':>15} {'DroPE Attn Norm':>15} {'RoPE MLP Norm':>15} {'DroPE MLP Norm':>15}")
    print("-" * 75)

    for layer in [0, 1, 2]:
        rope_l = results["RoPE"]["layer_content_analysis"][f"layer_{layer}"]
        drope_l = results["DroPE"]["layer_content_analysis"][f"layer_{layer}"]
        print(f"Layer {layer:<4} {rope_l.get('attn_output_norm_mean', 0):>15.2f} {drope_l.get('attn_output_norm_mean', 0):>15.2f} {rope_l.get('mlp_output_norm_mean', 0):>15.2f} {drope_l.get('mlp_output_norm_mean', 0):>15.2f}")


if __name__ == "__main__":
    main()
