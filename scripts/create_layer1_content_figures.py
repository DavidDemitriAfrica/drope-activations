#!/usr/bin/env python3
"""
Create figures for Experiment 8: Layer 1 Content Analysis

Visualizes the key finding: DroPE inverts the attention/MLP balance at Layer 1.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_FILE = REPO_ROOT / "results" / "phase_metrics" / "rope_vs_drope_phase_metrics.json"
OUTPUT_DIR = REPO_ROOT / "results" / "phase_metrics"

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
})

ROPE_COLOR = '#2ecc71'  # Green
DROPE_COLOR = '#e74c3c'  # Red


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def fig_layer1_content_analysis(results):
    """Figure: Layer 1 content analysis - attention/MLP balance inversion."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    rope_l1 = results["RoPE"]["layer_content_analysis"]["layer_1"]
    drope_l1 = results["DroPE"]["layer_content_analysis"]["layer_1"]

    # 1. Attention vs MLP output norms (bar chart)
    ax1 = axes[0, 0]
    x = np.arange(2)
    width = 0.35

    rope_vals = [rope_l1["attn_output_norm_mean"], rope_l1["mlp_output_norm_mean"]]
    drope_vals = [drope_l1["attn_output_norm_mean"], drope_l1["mlp_output_norm_mean"]]

    ax1.bar(x - width/2, rope_vals, width, label='RoPE', color=ROPE_COLOR)
    ax1.bar(x + width/2, drope_vals, width, label='DroPE', color=DROPE_COLOR)
    ax1.set_ylabel('Output Norm')
    ax1.set_title('Layer 1 Output Norms')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Attention', 'MLP'])
    ax1.legend()
    ax1.set_yscale('log')

    # 2. Contribution fractions (pie charts)
    ax2 = axes[0, 1]

    # RoPE pie
    rope_attn_frac = rope_l1["attn_contribution_frac"]
    rope_mlp_frac = 1 - rope_attn_frac
    drope_attn_frac = drope_l1["attn_contribution_frac"]
    drope_mlp_frac = 1 - drope_attn_frac

    # Stacked bar chart instead for better comparison
    x = np.arange(2)
    ax2.bar(x, [rope_attn_frac * 100, drope_attn_frac * 100], label='Attention',
            color='#3498db', alpha=0.8)
    ax2.bar(x, [rope_mlp_frac * 100, drope_mlp_frac * 100],
            bottom=[rope_attn_frac * 100, drope_attn_frac * 100],
            label='MLP', color='#e67e22', alpha=0.8)
    ax2.set_ylabel('Contribution (%)')
    ax2.set_title('Layer 1 Residual Stream Contribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['RoPE', 'DroPE'])
    ax2.legend(loc='upper right')

    # Add percentage labels
    ax2.text(0, rope_attn_frac * 50, f'{rope_attn_frac*100:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(0, rope_attn_frac * 100 + rope_mlp_frac * 50, f'{rope_mlp_frac*100:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(1, drope_attn_frac * 50, f'{drope_attn_frac*100:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(1, drope_attn_frac * 100 + drope_mlp_frac * 50, f'{drope_mlp_frac*100:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')

    # 3. Q/K norms comparison
    ax3 = axes[1, 0]
    x = np.arange(2)

    rope_qk = [rope_l1["q_norm_mean"], rope_l1["k_norm_mean"]]
    drope_qk = [drope_l1["q_norm_mean"], drope_l1["k_norm_mean"]]

    ax3.bar(x - width/2, rope_qk, width, label='RoPE', color=ROPE_COLOR)
    ax3.bar(x + width/2, drope_qk, width, label='DroPE', color=DROPE_COLOR)
    ax3.set_ylabel('Norm')
    ax3.set_title('Layer 1 Q/K Projection Norms')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Q', 'K'])
    ax3.legend()
    ax3.set_yscale('log')

    # Add ratio annotations
    q_ratio = drope_l1["q_norm_mean"] / rope_l1["q_norm_mean"]
    k_ratio = drope_l1["k_norm_mean"] / rope_l1["k_norm_mean"]
    ax3.annotate(f'{q_ratio:.0f}×', xy=(0 + width/2, drope_qk[0]), xytext=(0.3, drope_qk[0] * 1.5),
                 fontsize=10, ha='center')
    ax3.annotate(f'{k_ratio:.0f}×', xy=(1 + width/2, drope_qk[1]), xytext=(1.3, drope_qk[1] * 1.5),
                 fontsize=10, ha='center')

    # 4. Layer comparison (0, 1, 2)
    ax4 = axes[1, 1]

    layers = [0, 1, 2]
    rope_attn_norms = []
    drope_attn_norms = []

    for layer in layers:
        rope_l = results["RoPE"]["layer_content_analysis"][f"layer_{layer}"]
        drope_l = results["DroPE"]["layer_content_analysis"][f"layer_{layer}"]
        rope_attn_norms.append(rope_l["attn_output_norm_mean"])
        drope_attn_norms.append(drope_l["attn_output_norm_mean"])

    x = np.arange(len(layers))
    ax4.bar(x - width/2, rope_attn_norms, width, label='RoPE', color=ROPE_COLOR)
    ax4.bar(x + width/2, drope_attn_norms, width, label='DroPE', color=DROPE_COLOR)
    ax4.set_ylabel('Attention Output Norm')
    ax4.set_title('Attention Output by Layer')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Layer {l}' for l in layers])
    ax4.legend()
    ax4.set_yscale('log')

    plt.suptitle('Experiment 8: Layer 1 Content Analysis\nDroPE Inverts Attention/MLP Balance', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_layer1_content.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_layer1_content.png")


def main():
    print("=" * 60)
    print("Creating Layer 1 Content Analysis Figures")
    print("=" * 60)

    results = load_results()

    # Check if layer content analysis exists
    if "layer_content_analysis" not in results.get("RoPE", {}):
        print("Error: layer_content_analysis not found in results.")
        print("Run scripts/run_layer1_content_analysis.py first.")
        return

    fig_layer1_content_analysis(results)

    print("\nDone! Figure saved to results/phase_metrics/")


if __name__ == "__main__":
    main()
