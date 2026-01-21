#!/usr/bin/env python3
"""
Experiment 5: Are sinks no-ops in DroPE but not in RoPE?

Measures "effective BOS write" and tests BOS-V ablation to explain why:
- Both models have ~97% sink rates
- But only RoPE depends catastrophically on BOS-MLP

Key metrics:
- BOS attention mass: mean attention to BOS token per head
- BOS V norm: L2 norm of V vector at BOS position per head
- Effective BOS write score: attention_mass × V_norm (proxy for actual contribution)

Interventions:
- BOS-V ablation: zero V at BOS position
- Compare to BOS-MLP ablation (already done)
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
from datasets import load_dataset

ROPE_MODEL = "meta-llama/Llama-2-7b-hf"
DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"
RESULTS_FILE = REPO_ROOT / "results" / "phase_metrics" / "rope_vs_drope_phase_metrics.json"
OUTPUT_DIR = REPO_ROOT / "results" / "phase_metrics"


class BOSMetricsCapture:
    """Capture V norms and attention weights for BOS analysis."""

    def __init__(self, model, num_layers=32, num_heads=32, head_dim=128):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Storage for captured values
        self.v_outputs = {}  # layer -> V projection output
        self.attn_weights = {}  # layer -> attention weights

        self.handles = []

    def _v_hook(self, layer_idx):
        def hook(module, input, output):
            # output shape: [batch, seq, num_heads * head_dim]
            self.v_outputs[layer_idx] = output.detach()
        return hook

    def _attn_hook(self, layer_idx):
        def hook(module, input, output):
            # For LlamaAttention, we need to capture attention weights
            # This is trickier - we'll compute them from Q, K
            pass
        return hook

    def register_hooks(self):
        """Register hooks on v_proj for all layers."""
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Hook V projection
            h = layer.self_attn.v_proj.register_forward_hook(self._v_hook(layer_idx))
            self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_bos_v_norms(self):
        """Get BOS V norm per layer per head."""
        bos_v_norms = {}
        for layer_idx, v_out in self.v_outputs.items():
            # v_out: [batch, seq, num_heads * head_dim]
            # BOS is position 0
            bos_v = v_out[0, 0, :]  # [num_heads * head_dim]
            # Reshape to per-head
            bos_v_heads = bos_v.view(self.num_heads, self.head_dim)
            # Compute norm per head
            norms = bos_v_heads.norm(dim=1).cpu().numpy()  # [num_heads]
            bos_v_norms[layer_idx] = norms
        return bos_v_norms

    def clear(self):
        self.v_outputs = {}
        self.attn_weights = {}


class AttentionWeightCapture:
    """Capture attention weights by hooking into attention computation."""

    def __init__(self, model, num_layers=32, num_heads=32):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attn_weights = {}
        self.handles = []

    def _make_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            # LlamaAttention forward returns (attn_output, attn_weights, past_kv)
            # But attn_weights is only returned if output_attentions=True
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.attn_weights[layer_idx] = output[1].detach()
        return hook

    def register_hooks(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            h = layer.self_attn.register_forward_hook(self._make_hook(layer_idx), with_kwargs=True)
            self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_bos_attention_mass(self):
        """Get mean attention to BOS (position 0) per layer per head."""
        bos_attn = {}
        for layer_idx, weights in self.attn_weights.items():
            # weights: [batch, num_heads, seq, seq]
            # Attention to BOS = weights[:, :, :, 0]
            attn_to_bos = weights[0, :, :, 0]  # [num_heads, seq]
            # Mean over query positions (excluding BOS itself)
            mean_attn = attn_to_bos[:, 1:].mean(dim=1).cpu().numpy()  # [num_heads]
            bos_attn[layer_idx] = mean_attn
        return bos_attn

    def clear(self):
        self.attn_weights = {}


class BOSVAblationHook:
    """Hook that zeros out V at BOS position."""

    def __init__(self, layer_idx=None):
        """
        Args:
            layer_idx: If None, ablate all layers. Otherwise, ablate specific layer.
        """
        self.layer_idx = layer_idx
        self.handles = []

    def _hook(self, module, input, output):
        # Zero V at position 0 (BOS)
        modified = output.clone()
        modified[:, 0, :] = 0
        return modified

    def register(self, model):
        if self.layer_idx is not None:
            # Single layer
            layer = model.model.layers[self.layer_idx]
            h = layer.self_attn.v_proj.register_forward_hook(self._hook)
            self.handles.append(h)
        else:
            # All layers
            for layer in model.model.layers:
                h = layer.self_attn.v_proj.register_forward_hook(self._hook)
                self.handles.append(h)
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on a set of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(model.device)

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        total_loss += loss.item() * (input_ids.shape[1] - 1)
        total_tokens += input_ids.shape[1] - 1

    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


class QKCapture:
    """Capture Q and K projections to compute attention weights manually."""

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

    def compute_attention_to_bos(self, debug=False):
        """Compute attention weights to BOS (position 0) from Q, K."""
        bos_attn = {}
        nan_layers = []
        for layer_idx in self.q_outputs:
            q = self.q_outputs[layer_idx]  # [batch, seq, hidden]
            k = self.k_outputs[layer_idx]  # [batch, seq, hidden]

            batch, seq_len, hidden = q.shape

            if debug and layer_idx in [0, 1]:
                print(f"  Layer {layer_idx}: Q shape: {q.shape}, K shape: {k.shape}")
                print(f"  Layer {layer_idx}: Q has NaN: {torch.isnan(q).any()}, K has NaN: {torch.isnan(k).any()}")
                print(f"  Layer {layer_idx}: Q range: [{q.min():.2f}, {q.max():.2f}], K range: [{k.min():.2f}, {k.max():.2f}]")

            # Reshape to per-head
            q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq, dim]
            k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute attention scores (scaled dot product)
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, heads, seq, seq]

            if debug and layer_idx in [0, 1]:
                print(f"  Layer {layer_idx}: Scores has NaN: {torch.isnan(scores).any()}, has Inf: {torch.isinf(scores).any()}")
                finite_scores = scores[~torch.isinf(scores) & ~torch.isnan(scores)]
                if finite_scores.numel() > 0:
                    print(f"  Layer {layer_idx}: Scores range: [{finite_scores.min():.2f}, {finite_scores.max():.2f}]")

            # Clamp extreme scores to prevent overflow in softmax
            # This is necessary for DroPE which has massive Q/K activations
            scores = torch.clamp(scores, min=-100, max=100)

            # Apply causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)  # [batch, heads, seq, seq]

            if torch.isnan(attn_weights).any():
                nan_layers.append(layer_idx)
                if debug:
                    # Debug why softmax produced NaN
                    nan_count = torch.isnan(attn_weights).sum().item()
                    total = attn_weights.numel()
                    print(f"  Layer {layer_idx}: NaN count: {nan_count}/{total}")

            if debug and layer_idx in [0, 1]:
                print(f"  Layer {layer_idx}: Attn has NaN: {torch.isnan(attn_weights).any()}")

            # Get attention to BOS (column 0), excluding BOS itself
            attn_to_bos = attn_weights[0, :, 1:, 0].mean(dim=1).cpu().numpy()  # [num_heads]
            bos_attn[layer_idx] = attn_to_bos

        if debug and nan_layers:
            print(f"  Layers with NaN attention: {nan_layers}")

        return bos_attn

    def clear(self):
        self.q_outputs = {}
        self.k_outputs = {}


def measure_bos_metrics(model, tokenizer, num_heads=32, head_dim=128):
    """Measure BOS V norms and attention mass across all layers.

    Uses manual Q/K capture to compute attention weights (compatible with all attention implementations).
    """

    # Sample text for measurement
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago, never mind how long precisely.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
    ]

    v_capture = BOSMetricsCapture(model, num_heads=num_heads, head_dim=head_dim)
    v_capture.register_hooks()

    qk_capture = QKCapture(model, num_heads=num_heads, head_dim=head_dim)
    qk_capture.register_hooks()

    all_bos_v_norms = defaultdict(list)
    all_bos_attn = defaultdict(list)

    for i, text in enumerate(tqdm(texts, desc="Measuring BOS metrics")):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.device)

        v_capture.clear()
        qk_capture.clear()

        with torch.no_grad():
            # Run WITHOUT output_attentions (manual Q/K capture instead)
            outputs = model(input_ids)

        # Get V norms
        bos_v_norms = v_capture.get_bos_v_norms()
        for layer_idx, norms in bos_v_norms.items():
            all_bos_v_norms[layer_idx].append(norms)

        # Compute attention weights from Q, K (debug on first iteration)
        bos_attn = qk_capture.compute_attention_to_bos(debug=(i == 0))
        for layer_idx, attn in bos_attn.items():
            all_bos_attn[layer_idx].append(attn)

    v_capture.remove_hooks()
    qk_capture.remove_hooks()

    # Average across texts
    avg_bos_v_norms = {}
    avg_bos_attn = {}

    for layer_idx in range(len(model.model.layers)):
        if layer_idx in all_bos_v_norms:
            avg_bos_v_norms[layer_idx] = np.mean(all_bos_v_norms[layer_idx], axis=0)
        if layer_idx in all_bos_attn:
            avg_bos_attn[layer_idx] = np.mean(all_bos_attn[layer_idx], axis=0)

    return avg_bos_v_norms, avg_bos_attn


def compute_effective_write_score(bos_v_norms, bos_attn):
    """Compute effective BOS write score = attention × V_norm."""
    write_scores = {}
    for layer_idx in bos_v_norms:
        if layer_idx in bos_attn:
            write_scores[layer_idx] = bos_v_norms[layer_idx] * bos_attn[layer_idx]
    return write_scores


def run_bos_v_ablation_ppl(model, tokenizer, texts, layer_idx=None):
    """Run perplexity test with BOS-V ablation."""
    hook = BOSVAblationHook(layer_idx=layer_idx)
    hook.register(model)

    try:
        ppl = compute_perplexity(model, tokenizer, texts)
    finally:
        hook.remove()

    return ppl


def load_test_texts():
    """Load test texts for perplexity evaluation."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > 100][:50]
    return texts


def main():
    print("=" * 60)
    print("Experiment 5: BOS Write Analysis")
    print("Are sinks no-ops in DroPE but not in RoPE?")
    print("=" * 60)

    # Load existing results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    test_texts = load_test_texts()
    print(f"Loaded {len(test_texts)} test texts")

    for model_name, model_key in [(ROPE_MODEL, "RoPE"), (DROPE_MODEL, "DroPE")]:
        print(f"\n{'='*60}")
        print(f"Analyzing {model_key}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            # Don't use eager attention - causes NaN in DroPE
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads
        num_layers = model.config.num_hidden_layers

        # Get BOS spike layer from previous results
        bos_spike_layer = results[model_key].get("bos_spike_layer", 1 if model_key == "RoPE" else 2)

        print(f"\n--- Measuring BOS metrics ---")
        bos_v_norms, bos_attn = measure_bos_metrics(model, tokenizer, num_heads, head_dim)

        # Compute effective write scores
        write_scores = compute_effective_write_score(bos_v_norms, bos_attn)

        # Compute summary statistics
        all_v_norms = []
        all_attn = []
        all_write = []

        for layer_idx in range(num_layers):
            if layer_idx in bos_v_norms:
                all_v_norms.append(bos_v_norms[layer_idx].mean())
            if layer_idx in bos_attn:
                all_attn.append(bos_attn[layer_idx].mean())
            if layer_idx in write_scores:
                all_write.append(write_scores[layer_idx].mean())

        print(f"\nBOS V Norm (mean across layers): {np.mean(all_v_norms):.2f}")
        print(f"BOS Attention Mass (mean): {np.mean(all_attn):.4f}")
        print(f"Effective Write Score (mean): {np.mean(all_write):.4f}")

        # Find layer with max write score
        if write_scores:
            layer_write_means = [(l, write_scores[l].mean()) for l in write_scores]
            max_write_layer = max(layer_write_means, key=lambda x: x[1])
            print(f"Max write layer: {max_write_layer[0]} (score: {max_write_layer[1]:.4f})")
        else:
            max_write_layer = (0, 0.0)
            print("Warning: No write scores computed")

        # Store detailed metrics
        if "bos_write_analysis" not in results[model_key]:
            results[model_key]["bos_write_analysis"] = {}

        results[model_key]["bos_write_analysis"]["bos_v_norm_by_layer"] = {
            int(k): float(v.mean()) for k, v in bos_v_norms.items()
        }
        results[model_key]["bos_write_analysis"]["bos_attn_mass_by_layer"] = {
            int(k): float(v.mean()) for k, v in bos_attn.items()
        }
        results[model_key]["bos_write_analysis"]["write_score_by_layer"] = {
            int(k): float(v.mean()) for k, v in write_scores.items()
        }
        results[model_key]["bos_write_analysis"]["mean_v_norm"] = float(np.mean(all_v_norms))
        results[model_key]["bos_write_analysis"]["mean_attn_mass"] = float(np.mean(all_attn))
        results[model_key]["bos_write_analysis"]["mean_write_score"] = float(np.mean(all_write))
        results[model_key]["bos_write_analysis"]["max_write_layer"] = int(max_write_layer[0])

        # Run BOS-V ablation experiments
        print(f"\n--- BOS-V Ablation ---")

        # Baseline PPL
        baseline_ppl = compute_perplexity(model, tokenizer, test_texts[:20])
        print(f"Baseline PPL: {baseline_ppl:.2f}")

        # BOS-V ablation at spike layer
        spike_ppl = run_bos_v_ablation_ppl(model, tokenizer, test_texts[:20], layer_idx=bos_spike_layer)
        print(f"BOS-V ablation (layer {bos_spike_layer}): {spike_ppl:.2f} ({spike_ppl/baseline_ppl:.2f}x)")

        # BOS-V ablation at all layers
        all_layers_ppl = run_bos_v_ablation_ppl(model, tokenizer, test_texts[:20], layer_idx=None)
        print(f"BOS-V ablation (all layers): {all_layers_ppl:.2f} ({all_layers_ppl/baseline_ppl:.2f}x)")

        results[model_key]["bos_write_analysis"]["bos_v_ablation"] = {
            "baseline_ppl": float(baseline_ppl),
            "spike_layer_ppl": float(spike_ppl),
            "spike_layer_ratio": float(spike_ppl / baseline_ppl),
            "all_layers_ppl": float(all_layers_ppl),
            "all_layers_ratio": float(all_layers_ppl / baseline_ppl),
        }

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_FILE}")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("SUMMARY: BOS Write Analysis")
    print("=" * 60)

    print("\nMetric Comparison:")
    print(f"{'Metric':<30} {'RoPE':>15} {'DroPE':>15}")
    print("-" * 60)

    rope_write = results["RoPE"]["bos_write_analysis"]
    drope_write = results["DroPE"]["bos_write_analysis"]

    print(f"{'Mean BOS V Norm':<30} {rope_write['mean_v_norm']:>15.2f} {drope_write['mean_v_norm']:>15.2f}")
    print(f"{'Mean BOS Attention Mass':<30} {rope_write['mean_attn_mass']:>15.4f} {drope_write['mean_attn_mass']:>15.4f}")
    print(f"{'Mean Effective Write Score':<30} {rope_write['mean_write_score']:>15.4f} {drope_write['mean_write_score']:>15.4f}")
    print(f"{'Max Write Layer':<30} {rope_write['max_write_layer']:>15} {drope_write['max_write_layer']:>15}")

    print("\nBOS-V Ablation (PPL ratio):")
    print(f"{'Condition':<30} {'RoPE':>15} {'DroPE':>15}")
    print("-" * 60)
    print(f"{'Spike layer only':<30} {rope_write['bos_v_ablation']['spike_layer_ratio']:>15.2f}x {drope_write['bos_v_ablation']['spike_layer_ratio']:>15.2f}x")
    print(f"{'All layers':<30} {rope_write['bos_v_ablation']['all_layers_ratio']:>15.2f}x {drope_write['bos_v_ablation']['all_layers_ratio']:>15.2f}x")


if __name__ == "__main__":
    main()
