#!/usr/bin/env python3
"""
Create figures for Experiment 9: Cross-Layer Attention/MLP Balance
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

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
})

ROPE_COLOR = '#2ecc71'
DROPE_COLOR = '#e74c3c'


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def fig_crosslayer_balance(results):
    """Figure: Attention contribution fraction across all layers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    rope_attn = results["RoPE"]["crosslayer_balance"]["attn_contribution_frac"]
    drope_attn = results["DroPE"]["crosslayer_balance"]["attn_contribution_frac"]
    rope_attn_norm = results["RoPE"]["crosslayer_balance"]["attn_norms"]
    drope_attn_norm = results["DroPE"]["crosslayer_balance"]["attn_norms"]
    rope_mlp_norm = results["RoPE"]["crosslayer_balance"]["mlp_norms"]
    drope_mlp_norm = results["DroPE"]["crosslayer_balance"]["mlp_norms"]

    layers = range(len(rope_attn))

    # 1. Attention contribution fraction
    ax1 = axes[0, 0]
    ax1.plot(layers, [x * 100 for x in rope_attn], 'o-', color=ROPE_COLOR,
             label='RoPE', markersize=4, linewidth=1.5)
    ax1.plot(layers, [x * 100 for x in drope_attn], 's-', color=DROPE_COLOR,
             label='DroPE', markersize=4, linewidth=1.5)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Attention Contribution (%)')
    ax1.set_title('Attention vs MLP Balance by Layer')
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Highlight layers 0-1
    ax1.axvspan(-0.5, 1.5, alpha=0.2, color='yellow', label='Layers 0-1')

    # 2. Difference (DroPE - RoPE)
    ax2 = axes[0, 1]
    diff = [(d - r) * 100 for d, r in zip(drope_attn, rope_attn)]
    colors = [DROPE_COLOR if d > 0 else ROPE_COLOR for d in diff]
    ax2.bar(layers, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-10, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Difference (DroPE - RoPE) %')
    ax2.set_title('Attention Contribution Difference')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Attention output norms
    ax3 = axes[1, 0]
    ax3.plot(layers, rope_attn_norm, 'o-', color=ROPE_COLOR,
             label='RoPE', markersize=4, linewidth=1.5)
    ax3.plot(layers, drope_attn_norm, 's-', color=DROPE_COLOR,
             label='DroPE', markersize=4, linewidth=1.5)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Attention Output Norm')
    ax3.set_title('Attention Output Magnitude by Layer')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. MLP output norms
    ax4 = axes[1, 1]
    ax4.plot(layers, rope_mlp_norm, 'o-', color=ROPE_COLOR,
             label='RoPE', markersize=4, linewidth=1.5)
    ax4.plot(layers, drope_mlp_norm, 's-', color=DROPE_COLOR,
             label='DroPE', markersize=4, linewidth=1.5)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('MLP Output Norm')
    ax4.set_title('MLP Output Magnitude by Layer')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Experiment 9: Cross-Layer Attention/MLP Balance\n'
                 'The inversion is localized to Layers 0-1', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_crosslayer_balance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_crosslayer_balance.png")


def fig_layer01_spotlight(results):
    """Figure: Spotlight on Layers 0-1 showing the swap."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    rope_attn = results["RoPE"]["crosslayer_balance"]["attn_contribution_frac"]
    drope_attn = results["DroPE"]["crosslayer_balance"]["attn_contribution_frac"]

    # Layer 0 and 1 comparison
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.35

    rope_vals = [rope_attn[0] * 100, rope_attn[1] * 100]
    drope_vals = [drope_attn[0] * 100, drope_attn[1] * 100]

    bars1 = ax1.bar(x - width/2, rope_vals, width, label='RoPE', color=ROPE_COLOR)
    bars2 = ax1.bar(x + width/2, drope_vals, width, label='DroPE', color=DROPE_COLOR)

    ax1.set_ylabel('Attention Contribution (%)')
    ax1.set_title('Layers 0-1: The Swap')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Layer 0', 'Layer 1'])
    ax1.legend()
    ax1.set_ylim(0, 80)

    # Add value labels
    for bar, val in zip(bars1, rope_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, drope_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Layers 2-31 average
    ax2 = axes[1]
    rope_rest = np.mean(rope_attn[2:]) * 100
    drope_rest = np.mean(drope_attn[2:]) * 100

    x = np.arange(1)
    bars1 = ax2.bar(x - width/2, [rope_rest], width, label='RoPE', color=ROPE_COLOR)
    bars2 = ax2.bar(x + width/2, [drope_rest], width, label='DroPE', color=DROPE_COLOR)

    ax2.set_ylabel('Attention Contribution (%)')
    ax2.set_title('Layers 2-31: Nearly Identical')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Average'])
    ax2.legend()
    ax2.set_ylim(0, 50)

    # Add value labels
    ax2.text(bars1[0].get_x() + bars1[0].get_width()/2, bars1[0].get_height() + 1,
            f'{rope_rest:.1f}%', ha='center', va='bottom', fontsize=11)
    ax2.text(bars2[0].get_x() + bars2[0].get_width()/2, bars2[0].get_height() + 1,
            f'{drope_rest:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.suptitle('DroPE Rewires Only the First Two Layers', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_layer01_spotlight.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_layer01_spotlight.png")


def main():
    print("=" * 60)
    print("Creating Cross-Layer Balance Figures")
    print("=" * 60)

    results = load_results()

    if "crosslayer_balance" not in results.get("RoPE", {}):
        print("Error: crosslayer_balance not found in results.")
        print("Run scripts/run_crosslayer_balance.py first.")
        return

    fig_crosslayer_balance(results)
    fig_layer01_spotlight(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
