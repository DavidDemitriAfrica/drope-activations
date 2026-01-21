#!/usr/bin/env python3
"""
Create figures for Experiment 10: Extended Context Length Comparison
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

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


def fig_extended_context():
    """Figure: Passkey accuracy across full context range including extended."""

    # Data from our experiments
    contexts = [512, 1024, 2048, 4096, 6144, 8192]
    multiples = [0.125, 0.25, 0.5, 1.0, 1.5, 2.0]

    # RoPE: 100% within training, 0% beyond
    rope_acc = [100, 100, 100, 100, 0, 0]

    # DroPE: varies within training, strong beyond
    drope_acc = [80, 80, 40, 100, 100, 80]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(contexts))

    ax.plot(x, rope_acc, 'o-', color=ROPE_COLOR, label='RoPE',
            markersize=12, linewidth=2.5)
    ax.plot(x, drope_acc, 's-', color=DROPE_COLOR, label='DroPE',
            markersize=12, linewidth=2.5)

    # Add training boundary
    ax.axvline(x=3.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(3.5, 105, 'Training\nBoundary', ha='center', va='bottom',
            fontsize=10, color='gray')

    # Shade regions
    ax.axvspan(-0.5, 3.5, alpha=0.1, color='green', label='Within training')
    ax.axvspan(3.5, 5.5, alpha=0.1, color='red', label='Beyond training')

    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Passkey Retrieval Accuracy (%)')
    ax.set_title('RoPE vs DroPE: The Crossover at Training Boundary\n'
                 'RoPE excels within training but collapses beyond. DroPE generalizes.')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n({m}×)' for c, m in zip(contexts, multiples)])
    ax.legend(loc='lower left')
    ax.set_ylim(-5, 115)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (r, d) in enumerate(zip(rope_acc, drope_acc)):
        if r > 0:
            ax.annotate(f'{r}%', (i, r), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9, color=ROPE_COLOR)
        else:
            ax.annotate('0%\n(gibberish)', (i, r), textcoords="offset points",
                       xytext=(0, -25), ha='center', fontsize=8, color=ROPE_COLOR)

        ax.annotate(f'{d}%', (i, d), textcoords="offset points",
                   xytext=(0, -20 if d < r else 10), ha='center', fontsize=9, color=DROPE_COLOR)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_extended_context.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_extended_context.png")


def fig_tradeoff_summary():
    """Figure: Summary of the trade-off between short and long context."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Within training context
    ax1 = axes[0]
    contexts = ['512', '1024', '2048', '4096']
    rope_within = [100, 100, 100, 100]
    drope_within = [80, 80, 40, 100]

    x = np.arange(len(contexts))
    width = 0.35

    bars1 = ax1.bar(x - width/2, rope_within, width, label='RoPE', color=ROPE_COLOR)
    bars2 = ax1.bar(x + width/2, drope_within, width, label='DroPE', color=DROPE_COLOR)

    ax1.set_xlabel('Context Length')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Within Training Context (≤4096)\nRoPE: Perfect | DroPE: Variable')
    ax1.set_xticks(x)
    ax1.set_xticklabels(contexts)
    ax1.legend()
    ax1.set_ylim(0, 115)

    for bar, val in zip(bars1, rope_within):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, drope_within):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=9)

    # Right: Beyond training context
    ax2 = axes[1]
    contexts_ext = ['6144\n(1.5×)', '8192\n(2.0×)']
    rope_beyond = [0, 0]
    drope_beyond = [100, 80]

    x = np.arange(len(contexts_ext))

    bars1 = ax2.bar(x - width/2, rope_beyond, width, label='RoPE', color=ROPE_COLOR)
    bars2 = ax2.bar(x + width/2, drope_beyond, width, label='DroPE', color=DROPE_COLOR)

    ax2.set_xlabel('Context Length')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Beyond Training Context (>4096)\nRoPE: Collapse | DroPE: Functional')
    ax2.set_xticks(x)
    ax2.set_xticklabels(contexts_ext)
    ax2.legend()
    ax2.set_ylim(0, 115)

    for bar, val in zip(bars1, rope_beyond):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                '0%' if val == 0 else f'{val}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, drope_beyond):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=9)

    plt.suptitle('The DroPE Trade-Off: Short-Context Precision vs Long-Context Capability',
                 y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_tradeoff_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_tradeoff_summary.png")


def main():
    print("=" * 60)
    print("Creating Extended Context Figures")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_extended_context()
    fig_tradeoff_summary()

    print("\nDone!")


if __name__ == "__main__":
    main()
