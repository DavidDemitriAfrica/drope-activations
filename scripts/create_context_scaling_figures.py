#!/usr/bin/env python3
"""
Create figures for Experiment 10: Context Length Scaling
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


def fig_context_scaling(results):
    """Figure: Passkey accuracy by context length."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    rope_data = results["RoPE"]["context_scaling"]
    drope_data = results["DroPE"]["context_scaling"]

    contexts = sorted([int(k) for k in rope_data.keys()])

    rope_acc = [rope_data[str(c)]["accuracy"] * 100 for c in contexts]
    drope_acc = [drope_data[str(c)]["accuracy"] * 100 for c in contexts]

    x = np.arange(len(contexts))
    width = 0.35

    bars1 = ax.bar(x - width/2, rope_acc, width, label='RoPE', color=ROPE_COLOR)
    bars2 = ax.bar(x + width/2, drope_acc, width, label='DroPE', color=DROPE_COLOR)

    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Passkey Retrieval Accuracy (%)')
    ax.set_title('Experiment 10: Context Length Scaling\nDroPE fails at passkey retrieval across all context lengths')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in contexts])
    ax.legend()
    ax.set_ylim(0, 110)

    # Add value labels
    for bar, val in zip(bars1, rope_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, drope_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_context_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_context_scaling.png")


def fig_context_scaling_line(results):
    """Figure: Line plot of passkey accuracy by context length."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    rope_data = results["RoPE"]["context_scaling"]
    drope_data = results["DroPE"]["context_scaling"]

    contexts = sorted([int(k) for k in rope_data.keys()])

    rope_acc = [rope_data[str(c)]["accuracy"] * 100 for c in contexts]
    drope_acc = [drope_data[str(c)]["accuracy"] * 100 for c in contexts]

    ax.plot(contexts, rope_acc, 'o-', color=ROPE_COLOR, label='RoPE',
            markersize=10, linewidth=2)
    ax.plot(contexts, drope_acc, 's-', color=DROPE_COLOR, label='DroPE',
            markersize=10, linewidth=2)

    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Passkey Retrieval Accuracy (%)')
    ax.set_title('Passkey Retrieval: RoPE vs DroPE')
    ax.legend()
    ax.set_ylim(-5, 110)
    ax.set_xscale('log', base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([str(c) for c in contexts])
    ax.grid(True, alpha=0.3)

    # Annotate the gap
    for i, c in enumerate(contexts):
        gap = rope_acc[i] - drope_acc[i]
        mid_y = (rope_acc[i] + drope_acc[i]) / 2
        ax.annotate(f'{gap:.0f}%\ngap', xy=(c, mid_y), fontsize=8,
                   ha='center', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_context_scaling_line.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_context_scaling_line.png")


def main():
    print("=" * 60)
    print("Creating Context Scaling Figures")
    print("=" * 60)

    results = load_results()

    if "context_scaling" not in results.get("RoPE", {}):
        print("Error: context_scaling not found in results.")
        print("Run scripts/run_context_scaling.py first.")
        return

    fig_context_scaling(results)
    fig_context_scaling_line(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
