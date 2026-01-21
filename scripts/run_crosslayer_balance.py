#!/usr/bin/env python3
"""
Experiment 9: Cross-Layer Attention/MLP Balance

We found that DroPE inverts the attention/MLP balance at Layer 1:
- RoPE: 0.9% attention, 99.1% MLP
- DroPE: 68.8% attention, 31.2% MLP

Question: Is Layer 1 special, or does this inversion persist across all layers?
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np
from tqdm import tqdm

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


class LayerOutputCapture:
    """Capture attention and MLP outputs for a single layer."""

    def __init__(self):
        self.attn_output = None
        self.mlp_output = None
        self.handles = []

    def _attn_hook(self, module, input, output):
        if isinstance(output, tuple):
            self.attn_output = output[0].detach()
        else:
            self.attn_output = output.detach()

    def _mlp_hook(self, module, input, output):
        self.mlp_output = output.detach()

    def register(self, layer):
        h1 = layer.self_attn.register_forward_hook(self._attn_hook)
        h2 = layer.mlp.register_forward_hook(self._mlp_hook)
        self.handles = [h1, h2]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self):
        self.attn_output = None
        self.mlp_output = None


class AllLayerCapture:
    """Capture attention and MLP outputs for all layers."""

    def __init__(self, model):
        self.model = model
        self.num_layers = len(model.model.layers)
        self.captures = [LayerOutputCapture() for _ in range(self.num_layers)]

    def register_all(self):
        for i, layer in enumerate(self.model.model.layers):
            self.captures[i].register(layer)

    def remove_all(self):
        for c in self.captures:
            c.remove()

    def clear_all(self):
        for c in self.captures:
            c.clear()

    def get_norms(self):
        """Get attention and MLP output norms for all layers."""
        attn_norms = []
        mlp_norms = []

        for c in self.captures:
            if c.attn_output is not None:
                # Mean norm across sequence
                attn_norm = c.attn_output[0].norm(dim=-1).mean().item()
                attn_norms.append(attn_norm)
            else:
                attn_norms.append(0)

            if c.mlp_output is not None:
                mlp_norm = c.mlp_output[0].norm(dim=-1).mean().item()
                mlp_norms.append(mlp_norm)
            else:
                mlp_norms.append(0)

        return attn_norms, mlp_norms


def analyze_model(model, tokenizer, texts):
    """Analyze attention/MLP balance across all layers."""

    capture = AllLayerCapture(model)
    capture.register_all()

    all_attn_norms = []
    all_mlp_norms = []

    for text in tqdm(texts, desc="Processing texts"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.device)

        capture.clear_all()

        with torch.no_grad():
            model(input_ids)

        attn_norms, mlp_norms = capture.get_norms()
        all_attn_norms.append(attn_norms)
        all_mlp_norms.append(mlp_norms)

    capture.remove_all()

    # Average across texts
    avg_attn_norms = np.mean(all_attn_norms, axis=0)
    avg_mlp_norms = np.mean(all_mlp_norms, axis=0)

    # Compute contribution fractions
    attn_fracs = avg_attn_norms / (avg_attn_norms + avg_mlp_norms + 1e-10)
    mlp_fracs = avg_mlp_norms / (avg_attn_norms + avg_mlp_norms + 1e-10)

    return {
        "attn_norms": avg_attn_norms.tolist(),
        "mlp_norms": avg_mlp_norms.tolist(),
        "attn_contribution_frac": attn_fracs.tolist(),
        "mlp_contribution_frac": mlp_fracs.tolist(),
    }


def main():
    print("=" * 60)
    print("Experiment 9: Cross-Layer Attention/MLP Balance")
    print("Is the Layer 1 inversion unique or does it persist?")
    print("=" * 60)

    # Load existing results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

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

        layer_results = analyze_model(model, tokenizer, texts)

        # Store results
        if "crosslayer_balance" not in results[model_key]:
            results[model_key]["crosslayer_balance"] = {}

        results[model_key]["crosslayer_balance"] = layer_results

        # Print summary
        print(f"\n--- {model_key} Attention Contribution by Layer ---")
        print(f"{'Layer':<8} {'Attn Norm':<12} {'MLP Norm':<12} {'Attn %':<10} {'MLP %':<10}")
        print("-" * 52)

        for i in range(len(layer_results["attn_norms"])):
            attn_n = layer_results["attn_norms"][i]
            mlp_n = layer_results["mlp_norms"][i]
            attn_f = layer_results["attn_contribution_frac"][i] * 100
            mlp_f = layer_results["mlp_contribution_frac"][i] * 100
            print(f"{i:<8} {attn_n:<12.2f} {mlp_n:<12.2f} {attn_f:<10.1f} {mlp_f:<10.1f}")

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Attention Contribution Fraction by Layer")
    print("=" * 60)

    rope_attn = results["RoPE"]["crosslayer_balance"]["attn_contribution_frac"]
    drope_attn = results["DroPE"]["crosslayer_balance"]["attn_contribution_frac"]

    print(f"\n{'Layer':<8} {'RoPE Attn %':<15} {'DroPE Attn %':<15} {'Difference':<15}")
    print("-" * 53)

    for i in range(len(rope_attn)):
        rope_pct = rope_attn[i] * 100
        drope_pct = drope_attn[i] * 100
        diff = drope_pct - rope_pct
        marker = "***" if abs(diff) > 10 else ""
        print(f"{i:<8} {rope_pct:<15.1f} {drope_pct:<15.1f} {diff:>+10.1f} {marker}")

    # Summary statistics
    print("\n--- Summary ---")
    print(f"RoPE mean attention contribution: {np.mean(rope_attn)*100:.1f}%")
    print(f"DroPE mean attention contribution: {np.mean(drope_attn)*100:.1f}%")
    print(f"Layers where DroPE attention > RoPE attention: {sum(d > r for d, r in zip(drope_attn, rope_attn))}/32")

    # Find the crossover point
    for i in range(len(rope_attn)):
        if drope_attn[i] < rope_attn[i]:
            print(f"First layer where RoPE attention > DroPE attention: Layer {i}")
            break


if __name__ == "__main__":
    main()
