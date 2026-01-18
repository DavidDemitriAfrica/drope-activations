#!/usr/bin/env python3
"""
Disruption Experiments: Test if DroPE is less reliant on massive values.

Hypothesis: If DroPE learned alternative attention mechanisms, disrupting
massive values should cause LESS degradation in DroPE than in RoPE.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.massive_values.extraction import extract_qkv_all_layers
from src.massive_values.analysis import identify_massive_values
from src.utils.model_loading import load_model


@dataclass
class DisruptionResult:
    """Results from a disruption experiment."""
    model_name: str
    disruption_type: str  # "massive", "random", "none"
    target: str  # "query", "key", "both"
    perplexity_before: float
    perplexity_after: float
    ppl_increase: float
    ppl_increase_pct: float
    num_positions_disrupted: int


class ProjectionDisruptor:
    """
    Disrupts Q/K projection outputs by zeroing massive value dimensions.

    This hooks the output of q_proj and k_proj and zeros out the dimensions
    identified as containing massive values.
    """

    def __init__(
        self,
        model: nn.Module,
        massive_dims: Dict[int, Dict[str, torch.Tensor]],  # layer -> {"q": mask, "k": mask}
        target: str = "both",  # "query", "key", or "both"
    ):
        self.model = model
        self.massive_dims = massive_dims
        self.target = target
        self.hooks = []

    def _create_q_hook(self, layer_idx: int):
        """Create hook for Q projection."""
        mask = self.massive_dims.get(layer_idx, {}).get("q")
        if mask is None:
            return None

        def hook(module, input, output):
            # output: [batch, seq_len, hidden_dim]
            # mask: [hidden_dim] boolean
            # Use multiplication to zero out masked dimensions
            zero_mask = (~mask).to(output.dtype).to(output.device)  # 0 where massive, 1 elsewhere
            return output * zero_mask.unsqueeze(0).unsqueeze(0)
        return hook

    def _create_k_hook(self, layer_idx: int):
        """Create hook for K projection."""
        mask = self.massive_dims.get(layer_idx, {}).get("k")
        if mask is None:
            return None

        def hook(module, input, output):
            zero_mask = (~mask).to(output.dtype).to(output.device)
            return output * zero_mask.unsqueeze(0).unsqueeze(0)
        return hook

    def register_hooks(self):
        """Register hooks on all Q/K projections."""
        self.remove_hooks()

        if hasattr(self.model, "model"):
            base = self.model.model
        else:
            base = self.model

        if hasattr(base, "layers"):
            layers = base.layers
        else:
            raise ValueError("Cannot find transformer layers")

        for idx, layer in enumerate(layers):
            if idx not in self.massive_dims:
                continue

            attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attention

            if self.target in ["query", "both"]:
                hook = self._create_q_hook(idx)
                if hook is not None:
                    handle = attn.q_proj.register_forward_hook(hook)
                    self.hooks.append(handle)

            if self.target in ["key", "both"]:
                hook = self._create_k_hook(idx)
                if hook is not None:
                    handle = attn.k_proj.register_forward_hook(hook)
                    self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()


def compute_perplexity(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
) -> float:
    """Compute perplexity on a set of texts."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            input_ids = inputs["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            num_tokens = input_ids.shape[1] - 1  # Exclude first token
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    return perplexity


def identify_massive_dims(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    lambda_threshold: float = 5.0,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], int]:
    """
    Identify which dimensions contain massive values.

    Returns:
        massive_dims: Dict[layer_idx, {"q": mask, "k": mask}]
                      Masks are flattened to match q_proj/k_proj output [hidden_dim]
        total_count: Total number of massive dimensions
    """
    device = next(model.parameters()).device

    # Collect Q/K outputs across samples
    all_q_norms = {}  # layer -> list of norm matrices
    all_k_norms = {}

    with torch.no_grad():
        for text in texts[:3]:  # Use 3 samples to identify positions
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)

            qkv_by_layer = extract_qkv_all_layers(model, input_ids)

            for layer_idx, qkv in qkv_by_layer.items():
                # qkv.query: [batch, seq_len, num_heads, head_dim]
                # Compute L2 norm across sequence dimension -> [num_heads, head_dim]
                q_norm = qkv.query.squeeze(0).norm(dim=0)  # [num_heads, head_dim]
                k_norm = qkv.key.squeeze(0).norm(dim=0)

                if layer_idx not in all_q_norms:
                    all_q_norms[layer_idx] = []
                    all_k_norms[layer_idx] = []

                all_q_norms[layer_idx].append(q_norm)
                all_k_norms[layer_idx].append(k_norm)

    # Identify massive dimensions
    massive_dims = {}
    total_count = 0

    for layer_idx in all_q_norms.keys():
        # Average norms across samples: [num_heads, head_dim]
        q_norm_avg = torch.stack(all_q_norms[layer_idx]).mean(dim=0)
        k_norm_avg = torch.stack(all_k_norms[layer_idx]).mean(dim=0)

        # Identify massive dimensions (norm > lambda * mean)
        q_threshold = lambda_threshold * q_norm_avg.mean()
        k_threshold = lambda_threshold * k_norm_avg.mean()

        q_mask = q_norm_avg > q_threshold  # [num_heads, head_dim]
        k_mask = k_norm_avg > k_threshold

        if q_mask.any() or k_mask.any():
            # Flatten to match q_proj output shape [hidden_dim]
            # q_proj output is [batch, seq, num_heads * head_dim]
            massive_dims[layer_idx] = {
                "q": q_mask.flatten(),  # [hidden_dim]
                "k": k_mask.flatten(),
            }
            total_count += q_mask.sum().item() + k_mask.sum().item()

    return massive_dims, total_count


def create_random_dims(
    massive_dims: Dict[int, Dict[str, torch.Tensor]],
    seed: int = 42,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Create random dimension masks with same count as massive dims."""
    torch.manual_seed(seed)
    random_dims = {}

    for layer_idx, masks in massive_dims.items():
        random_dims[layer_idx] = {}

        for key in ["q", "k"]:
            mask = masks[key]
            num_massive = mask.sum().item()
            total_dims = mask.numel()

            # Create random mask with same count
            random_mask = torch.zeros(total_dims, dtype=torch.bool, device=mask.device)
            random_indices = torch.randperm(total_dims)[:int(num_massive)]
            random_mask[random_indices] = True
            random_dims[layer_idx][key] = random_mask

    return random_dims


def run_disruption_experiment(
    model_name: str,
    model: nn.Module,
    tokenizer,
    eval_texts: List[str],
    massive_dims: Dict[int, Dict[str, torch.Tensor]],
    target: str = "both",
) -> Tuple[DisruptionResult, DisruptionResult, DisruptionResult]:
    """
    Run disruption experiment with massive, random, and no disruption.

    Returns:
        (baseline_result, massive_result, random_result)
    """
    device = next(model.parameters()).device

    # Count total disrupted positions
    total_disrupted = sum(
        masks["q"].sum().item() + masks["k"].sum().item()
        for masks in massive_dims.values()
    )

    # Baseline: no disruption
    print(f"  Computing baseline perplexity...")
    ppl_baseline = compute_perplexity(model, tokenizer, eval_texts)

    baseline_result = DisruptionResult(
        model_name=model_name,
        disruption_type="none",
        target=target,
        perplexity_before=ppl_baseline,
        perplexity_after=ppl_baseline,
        ppl_increase=0,
        ppl_increase_pct=0,
        num_positions_disrupted=0,
    )

    # Disrupt massive values
    print(f"  Disrupting massive values ({total_disrupted} dimensions)...")
    with ProjectionDisruptor(model, massive_dims, target=target):
        ppl_massive = compute_perplexity(model, tokenizer, eval_texts)

    massive_result = DisruptionResult(
        model_name=model_name,
        disruption_type="massive",
        target=target,
        perplexity_before=ppl_baseline,
        perplexity_after=ppl_massive,
        ppl_increase=ppl_massive - ppl_baseline,
        ppl_increase_pct=(ppl_massive - ppl_baseline) / ppl_baseline * 100,
        num_positions_disrupted=total_disrupted,
    )

    # Disrupt random values (control)
    print(f"  Disrupting random values (control)...")
    random_dims = create_random_dims(massive_dims)
    with ProjectionDisruptor(model, random_dims, target=target):
        ppl_random = compute_perplexity(model, tokenizer, eval_texts)

    random_result = DisruptionResult(
        model_name=model_name,
        disruption_type="random",
        target=target,
        perplexity_before=ppl_baseline,
        perplexity_after=ppl_random,
        ppl_increase=ppl_random - ppl_baseline,
        ppl_increase_pct=(ppl_random - ppl_baseline) / ppl_baseline * 100,
        num_positions_disrupted=total_disrupted,
    )

    return baseline_result, massive_result, random_result


def main():
    output_dir = Path("results/disruption")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("DISRUPTION EXPERIMENT: Are Massive Values Functionally Important?")
    print("=" * 70)
    print()
    print("Hypothesis: If DroPE learned alternative mechanisms, disrupting")
    print("massive values should cause LESS degradation than in RoPE.")
    print()

    # Evaluation texts
    eval_texts = [
        "The quick brown fox jumps over the lazy dog. " * 30,
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort. " * 5,
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness. " * 10,
        "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. " * 5,
        "The history of every major galactic civilization tends to pass through three distinct and recognizable phases, those of Survival, Inquiry and Sophistication. " * 10,
    ]

    all_results = {}

    # =========================================================================
    # Llama-2-7B RoPE
    # =========================================================================
    print("\n" + "=" * 70)
    print("Loading Llama-2-7B (RoPE)...")
    print("=" * 70)

    rope_model, tokenizer = load_model("llama2-7b", device="cuda")
    rope_model.eval()

    print("\nIdentifying massive value dimensions...")
    rope_massive_dims, rope_total = identify_massive_dims(rope_model, tokenizer, eval_texts)
    print(f"Found {rope_total} massive dimensions across {len(rope_massive_dims)} layers")

    print("\nRunning disruption experiments on RoPE model...")
    rope_baseline, rope_massive, rope_random = run_disruption_experiment(
        "Llama-2-7B-RoPE", rope_model, tokenizer, eval_texts, rope_massive_dims
    )

    all_results["rope"] = {
        "baseline": rope_baseline.__dict__,
        "massive": rope_massive.__dict__,
        "random": rope_random.__dict__,
    }

    print(f"\n  Baseline PPL:     {rope_baseline.perplexity_after:.2f}")
    print(f"  Massive disrupted: {rope_massive.perplexity_after:.2f} (+{rope_massive.ppl_increase_pct:.1f}%)")
    print(f"  Random disrupted:  {rope_random.perplexity_after:.2f} (+{rope_random.ppl_increase_pct:.1f}%)")

    # Free memory
    del rope_model
    torch.cuda.empty_cache()

    # =========================================================================
    # Llama-2-7B DroPE
    # =========================================================================
    print("\n" + "=" * 70)
    print("Loading Llama-2-7B (DroPE)...")
    print("=" * 70)

    drope_model, _ = load_model("llama2-7b-drope", device="cuda")
    drope_model.eval()

    print("\nIdentifying massive value dimensions...")
    drope_massive_dims, drope_total = identify_massive_dims(drope_model, tokenizer, eval_texts)
    print(f"Found {drope_total} massive dimensions across {len(drope_massive_dims)} layers")

    print("\nRunning disruption experiments on DroPE model...")
    drope_baseline, drope_massive, drope_random = run_disruption_experiment(
        "Llama-2-7B-DroPE", drope_model, tokenizer, eval_texts, drope_massive_dims
    )

    all_results["drope"] = {
        "baseline": drope_baseline.__dict__,
        "massive": drope_massive.__dict__,
        "random": drope_random.__dict__,
    }

    print(f"\n  Baseline PPL:     {drope_baseline.perplexity_after:.2f}")
    print(f"  Massive disrupted: {drope_massive.perplexity_after:.2f} (+{drope_massive.ppl_increase_pct:.1f}%)")
    print(f"  Random disrupted:  {drope_random.perplexity_after:.2f} (+{drope_random.ppl_increase_pct:.1f}%)")

    # =========================================================================
    # Summary and Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<20} {'Baseline':>10} {'Massive':>12} {'Random':>12} {'M-R Diff':>10}")
    print("-" * 70)

    rope_diff = rope_massive.ppl_increase_pct - rope_random.ppl_increase_pct
    drope_diff = drope_massive.ppl_increase_pct - drope_random.ppl_increase_pct

    print(f"{'RoPE':<20} {rope_baseline.perplexity_after:>10.2f} "
          f"{rope_massive.ppl_increase_pct:>+11.1f}% {rope_random.ppl_increase_pct:>+11.1f}% "
          f"{rope_diff:>+9.1f}%")
    print(f"{'DroPE':<20} {drope_baseline.perplexity_after:>10.2f} "
          f"{drope_massive.ppl_increase_pct:>+11.1f}% {drope_random.ppl_increase_pct:>+11.1f}% "
          f"{drope_diff:>+9.1f}%")

    print("\n" + "-" * 70)
    print("Key metric: M-R Diff = (Massive disruption) - (Random disruption)")
    print("Higher M-R Diff = massive values are MORE important than random positions")
    print("-" * 70)

    if rope_diff > drope_diff:
        print(f"\n✓ HYPOTHESIS SUPPORTED: RoPE relies MORE on massive values")
        print(f"  RoPE M-R Diff: {rope_diff:+.1f}%")
        print(f"  DroPE M-R Diff: {drope_diff:+.1f}%")
        print(f"  Difference: {rope_diff - drope_diff:.1f}% more reliance in RoPE")
    else:
        print(f"\n✗ HYPOTHESIS NOT SUPPORTED: DroPE still relies on massive values")
        print(f"  RoPE M-R Diff: {rope_diff:+.1f}%")
        print(f"  DroPE M-R Diff: {drope_diff:+.1f}%")

    # Save results
    all_results["summary"] = {
        "rope_massive_dims": rope_total,
        "drope_massive_dims": drope_total,
        "rope_mr_diff": float(rope_diff),
        "drope_mr_diff": float(drope_diff),
        "hypothesis_supported": bool(rope_diff > drope_diff),
    }

    with open(output_dir / f"disruption_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: PPL comparison
    ax = axes[0]
    x = np.arange(2)
    width = 0.25

    baseline_ppls = [rope_baseline.perplexity_after, drope_baseline.perplexity_after]
    massive_ppls = [rope_massive.perplexity_after, drope_massive.perplexity_after]
    random_ppls = [rope_random.perplexity_after, drope_random.perplexity_after]

    ax.bar(x - width, baseline_ppls, width, label='Baseline', color='#2ecc71')
    ax.bar(x, massive_ppls, width, label='Massive disrupted', color='#e74c3c')
    ax.bar(x + width, random_ppls, width, label='Random disrupted', color='#3498db')

    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity After Disruption')
    ax.set_xticks(x)
    ax.set_xticklabels(['RoPE', 'DroPE'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Right: % increase comparison
    ax = axes[1]

    massive_pcts = [rope_massive.ppl_increase_pct, drope_massive.ppl_increase_pct]
    random_pcts = [rope_random.ppl_increase_pct, drope_random.ppl_increase_pct]

    ax.bar(x - width/2, massive_pcts, width, label='Massive disrupted', color='#e74c3c')
    ax.bar(x + width/2, random_pcts, width, label='Random disrupted', color='#3498db')

    ax.set_ylabel('PPL Increase (%)')
    ax.set_title('Relative Degradation from Disruption')
    ax.set_xticks(x)
    ax.set_xticklabels(['RoPE', 'DroPE'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / f"disruption_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f"disruption_results_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
