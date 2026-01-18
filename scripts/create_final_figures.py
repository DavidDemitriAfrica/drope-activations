#!/usr/bin/env python3
"""
Create clean publication figures for the two main findings.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try science styles
try:
    plt.style.use(['science', 'ieee', 'no-latex'])
except:
    plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

# Colors
ROPE_COLOR = '#0077BB'
DROPE_COLOR = '#EE7733'


def fig1_massive_value_reduction():
    """Finding 1: DroPE has fewer massive values."""
    # Load data
    llama_dir = Path("results/llama_comparison")
    latest = max(llama_dir.glob("comparison_results_*.json"), key=lambda x: x.stat().st_mtime)
    with open(latest) as f:
        data = json.load(f)

    rope = data['rope']
    drope = data['drope']

    # Totals
    rope_q = sum(r['num_massive_q'] for r in rope)
    rope_k = sum(r['num_massive_k'] for r in rope)
    rope_v = sum(r['num_massive_v'] for r in rope)

    drope_q = sum(r['num_massive_q'] for r in drope)
    drope_k = sum(r['num_massive_k'] for r in drope)
    drope_v = sum(r['num_massive_v'] for r in drope)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    # Panel A: Bar chart
    ax = axes[0]
    x = np.arange(3)
    width = 0.35

    rope_vals = [rope_q, rope_k, rope_v]
    drope_vals = [drope_q, drope_k, drope_v]

    bars1 = ax.bar(x - width/2, rope_vals, width, label='RoPE', color=ROPE_COLOR)
    bars2 = ax.bar(x + width/2, drope_vals, width, label='DroPE', color=DROPE_COLOR)

    # Add percentage labels
    for i, (r, d) in enumerate(zip(rope_vals, drope_vals)):
        pct = (d - r) / r * 100
        ax.annotate(f'{pct:+.0f}%', xy=(i + width/2, d + 40), ha='center',
                    fontsize=8, fontweight='bold', color=DROPE_COLOR if pct < 0 else 'black')

    ax.set_ylabel('Massive Values (total)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Query', 'Key', 'Value'])
    ax.legend(loc='upper right')
    ax.set_title('A. Massive Value Count')
    ax.set_ylim(0, max(rope_vals) * 1.15)

    # Panel B: Layer-wise Query
    ax = axes[1]
    layers = [r['layer'] for r in rope]
    rope_q_layers = [r['num_massive_q'] for r in rope]
    drope_q_layers = [r['num_massive_q'] for r in drope]

    ax.fill_between(layers, rope_q_layers, drope_q_layers, alpha=0.3, color=ROPE_COLOR)
    ax.plot(layers, rope_q_layers, 'o-', color=ROPE_COLOR, label='RoPE', markersize=3, linewidth=1.2)
    ax.plot(layers, drope_q_layers, 's-', color=DROPE_COLOR, label='DroPE', markersize=3, linewidth=1.2)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Query Massive Values')
    ax.set_title('B. Query by Layer')
    ax.legend(loc='upper right')
    ax.set_xlim(-1, 32)

    plt.tight_layout()
    return fig


def fig2_disruption_experiment():
    """Finding 2: RoPE relies more on massive values."""
    # Data from experiment
    rope_baseline = 1.15
    rope_massive = 2946.0
    rope_random = 1.19

    drope_baseline = 1.29
    drope_massive = 31.72
    drope_random = 1.32

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    # Panel A: PPL after disruption (log scale)
    ax = axes[0]
    x = np.arange(2)
    width = 0.25

    baseline = [rope_baseline, drope_baseline]
    massive = [rope_massive, drope_massive]
    random = [rope_random, drope_random]

    ax.bar(x - width, baseline, width, label='Baseline', color='#2ecc71')
    ax.bar(x, massive, width, label='Massive zeroed', color='#e74c3c')
    ax.bar(x + width, random, width, label='Random zeroed', color='#3498db')

    ax.set_yscale('log')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(['RoPE', 'DroPE'])
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title('A. Perplexity After Disruption')
    ax.set_ylim(1, 5000)

    # Panel B: Relative increase
    ax = axes[1]

    rope_increase = (rope_massive - rope_baseline) / rope_baseline * 100
    drope_increase = (drope_massive - drope_baseline) / drope_baseline * 100

    bars = ax.bar(['RoPE', 'DroPE'], [rope_increase, drope_increase],
                  color=[ROPE_COLOR, DROPE_COLOR], width=0.5)

    # Add value labels
    ax.annotate(f'{rope_increase/1000:.0f}k%', xy=(0, rope_increase),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.annotate(f'{drope_increase:.0f}%', xy=(1, drope_increase + 5000),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('PPL Increase (%)')
    ax.set_title('B. Degradation from Disruption')
    ax.set_ylim(0, rope_increase * 1.1)

    # Add ratio annotation
    ratio = rope_increase / drope_increase
    ax.annotate(f'RoPE relies\n{ratio:.0f}× more', xy=(0.5, rope_increase * 0.5),
                ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    return fig


def fig_combined():
    """Combined 2-panel figure for paper."""
    fig = plt.figure(figsize=(7, 5.5))

    # Load massive values data
    llama_dir = Path("results/llama_comparison")
    latest = max(llama_dir.glob("comparison_results_*.json"), key=lambda x: x.stat().st_mtime)
    with open(latest) as f:
        data = json.load(f)

    rope = data['rope']
    drope = data['drope']

    # ===== Top row: Massive value analysis =====
    ax1 = fig.add_subplot(2, 2, 1)

    # Totals
    rope_q = sum(r['num_massive_q'] for r in rope)
    rope_k = sum(r['num_massive_k'] for r in rope)
    drope_q = sum(r['num_massive_q'] for r in drope)
    drope_k = sum(r['num_massive_k'] for r in drope)

    x = np.arange(2)
    width = 0.35

    ax1.bar(x - width/2, [rope_q, rope_k], width, label='RoPE', color=ROPE_COLOR)
    ax1.bar(x + width/2, [drope_q, drope_k], width, label='DroPE', color=DROPE_COLOR)

    # Percentage labels
    for i, (r, d) in enumerate(zip([rope_q, rope_k], [drope_q, drope_k])):
        pct = (d - r) / r * 100
        ax1.annotate(f'{pct:+.0f}%', xy=(i + width/2, d + 30), ha='center',
                    fontsize=8, fontweight='bold', color=DROPE_COLOR)

    ax1.set_ylabel('Massive Values')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Query', 'Key'])
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('A. Massive Value Count (Llama-2-7B)')

    # Layer-wise
    ax2 = fig.add_subplot(2, 2, 2)
    layers = [r['layer'] for r in rope]
    rope_q_layers = [r['num_massive_q'] for r in rope]
    drope_q_layers = [r['num_massive_q'] for r in drope]

    ax2.fill_between(layers, rope_q_layers, drope_q_layers, alpha=0.25, color=ROPE_COLOR)
    ax2.plot(layers, rope_q_layers, '-', color=ROPE_COLOR, label='RoPE', linewidth=1.5)
    ax2.plot(layers, drope_q_layers, '-', color=DROPE_COLOR, label='DroPE', linewidth=1.5)

    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Query Massive Values')
    ax2.set_title('B. Query Distribution by Layer')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0, 31)

    # ===== Bottom row: Disruption experiment =====
    ax3 = fig.add_subplot(2, 2, 3)

    # Data
    rope_baseline, rope_massive, rope_random = 1.15, 2946.0, 1.19
    drope_baseline, drope_massive, drope_random = 1.29, 31.72, 1.32

    x = np.arange(2)
    width = 0.25

    ax3.bar(x - width, [rope_baseline, drope_baseline], width, label='Baseline', color='#2ecc71')
    ax3.bar(x, [rope_massive, drope_massive], width, label='Massive zeroed', color='#e74c3c')
    ax3.bar(x + width, [rope_random, drope_random], width, label='Random zeroed', color='#3498db')

    ax3.set_yscale('log')
    ax3.set_ylabel('Perplexity (log)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['RoPE', 'DroPE'])
    ax3.legend(loc='upper right', fontsize=7)
    ax3.set_title('C. Disruption Experiment')
    ax3.set_ylim(1, 5000)

    # Relative increase
    ax4 = fig.add_subplot(2, 2, 4)

    rope_inc = (rope_massive - rope_baseline) / rope_baseline * 100
    drope_inc = (drope_massive - drope_baseline) / drope_baseline * 100

    bars = ax4.bar(['RoPE', 'DroPE'], [rope_inc / 1000, drope_inc / 1000],
                   color=[ROPE_COLOR, DROPE_COLOR], width=0.5)

    ax4.set_ylabel('PPL Increase (×1000%)')
    ax4.set_title('D. Reliance on Massive Values')

    # Ratio
    ratio = rope_inc / drope_inc
    ax4.annotate(f'{ratio:.0f}× more\nreliant', xy=(0, rope_inc/1000 * 0.6),
                ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    output_dir = Path("results/final_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating publication figures...")

    # Figure 1: Massive value reduction
    fig1 = fig1_massive_value_reduction()
    fig1.savefig(output_dir / 'fig1_massive_values.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / 'fig1_massive_values.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Created: fig1_massive_values")

    # Figure 2: Disruption experiment
    fig2 = fig2_disruption_experiment()
    fig2.savefig(output_dir / 'fig2_disruption.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / 'fig2_disruption.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("Created: fig2_disruption")

    # Combined figure
    fig3 = fig_combined()
    fig3.savefig(output_dir / 'fig_combined.pdf', dpi=300, bbox_inches='tight')
    fig3.savefig(output_dir / 'fig_combined.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("Created: fig_combined")

    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
