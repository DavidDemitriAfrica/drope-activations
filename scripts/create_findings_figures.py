#!/usr/bin/env python3
"""
Create figures for FINDINGS.md - clean, publication-ready.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use science style if available
try:
    plt.style.use(['science', 'ieee', 'no-latex'])
except:
    plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
})

ROPE_COLOR = '#0077BB'
DROPE_COLOR = '#EE7733'
MASSIVE_COLOR = '#CC3311'
RANDOM_COLOR = '#009988'
BASELINE_COLOR = '#33BBEE'

output_dir = Path("results/findings_figures")
output_dir.mkdir(parents=True, exist_ok=True)


def fig1_massive_value_counts():
    """
    Figure 1: Massive value counts comparison (Experiment 1)
    Shows Q/K/V with error bars
    """
    # Data from rigorous experiment (10 samples)
    rope = {'q': (1475.5, 22.6), 'k': (1496.8, 69.8), 'v': (174.0, 10.7)}
    drope = {'q': (901.4, 36.0), 'k': (1331.5, 74.1), 'v': (176.6, 5.7)}

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(3)
    width = 0.35

    rope_means = [rope['q'][0], rope['k'][0], rope['v'][0]]
    rope_stds = [rope['q'][1], rope['k'][1], rope['v'][1]]
    drope_means = [drope['q'][0], drope['k'][0], drope['v'][0]]
    drope_stds = [drope['q'][1], drope['k'][1], drope['v'][1]]

    bars1 = ax.bar(x - width/2, rope_means, width, yerr=rope_stds, capsize=5,
                   label='RoPE', color=ROPE_COLOR, alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, drope_means, width, yerr=drope_stds, capsize=5,
                   label='DroPE', color=DROPE_COLOR, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Add percentage change labels
    changes = [
        (drope['q'][0] - rope['q'][0]) / rope['q'][0] * 100,
        (drope['k'][0] - rope['k'][0]) / rope['k'][0] * 100,
        (drope['v'][0] - rope['v'][0]) / rope['v'][0] * 100,
    ]

    for i, (change, drope_val, drope_std) in enumerate(zip(changes, drope_means, drope_stds)):
        color = DROPE_COLOR if change < 0 else 'black'
        weight = 'bold' if abs(change) > 10 else 'normal'
        ax.annotate(f'{change:+.0f}%',
                    xy=(i + width/2, drope_val + drope_std + 40),
                    ha='center', va='bottom', fontsize=11, fontweight=weight, color=color)

    ax.set_ylabel('Massive Values (total across 32 layers)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Query (Q)', 'Key (K)', 'Value (V)'])
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.set_ylim(0, max(rope_means) * 1.2)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_massive_value_counts.png', bbox_inches='tight')
    fig.savefig(output_dir / 'fig1_massive_value_counts.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Created: fig1_massive_value_counts")


def fig2_layer_distribution():
    """
    Figure 2: Layer-wise Query distribution (Experiment 1)
    """
    # Load actual layer data
    llama_dir = Path("results/llama_comparison")
    latest = max(llama_dir.glob("comparison_results_*.json"), key=lambda x: x.stat().st_mtime)
    with open(latest) as f:
        data = json.load(f)

    rope = data['rope']
    drope = data['drope']

    layers = [r['layer'] for r in rope]
    rope_q = [r['num_massive_q'] for r in rope]
    drope_q = [r['num_massive_q'] for r in drope]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Fill between to show difference
    ax.fill_between(layers, rope_q, drope_q, alpha=0.3, color=ROPE_COLOR,
                    where=[r >= d for r, d in zip(rope_q, drope_q)], label='_')
    ax.fill_between(layers, rope_q, drope_q, alpha=0.3, color=DROPE_COLOR,
                    where=[r < d for r, d in zip(rope_q, drope_q)], label='_')

    ax.plot(layers, rope_q, 'o-', color=ROPE_COLOR, label='RoPE',
            markersize=5, linewidth=1.5, markeredgecolor='white', markeredgewidth=0.5)
    ax.plot(layers, drope_q, 's-', color=DROPE_COLOR, label='DroPE',
            markersize=5, linewidth=1.5, markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Query Massive Values')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(0, max(max(rope_q), max(drope_q)) * 1.1)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Add annotation for the gap
    mid_layer = 15
    gap = rope_q[mid_layer] - drope_q[mid_layer]
    ax.annotate('', xy=(mid_layer, drope_q[mid_layer]), xytext=(mid_layer, rope_q[mid_layer]),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.annotate(f'~{int(np.mean([r-d for r,d in zip(rope_q, drope_q)]))} fewer\nper layer',
                xy=(mid_layer + 1, (rope_q[mid_layer] + drope_q[mid_layer])/2),
                fontsize=9, va='center')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_layer_distribution.png', bbox_inches='tight')
    fig.savefig(output_dir / 'fig2_layer_distribution.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Created: fig2_layer_distribution")


def fig3_disruption_perplexity():
    """
    Figure 3: Perplexity after disruption (Experiment 2)
    """
    # Data from rigorous experiment
    data = {
        'RoPE': {'baseline': 1.30, 'massive': 1508.5, 'random': 1.31},
        'DroPE': {'baseline': 1.49, 'massive': 22.7, 'random': 1.49},
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))

    x = np.arange(2)
    width = 0.25

    baseline = [data['RoPE']['baseline'], data['DroPE']['baseline']]
    massive = [data['RoPE']['massive'], data['DroPE']['massive']]
    random = [data['RoPE']['random'], data['DroPE']['random']]

    bars1 = ax.bar(x - width, baseline, width, label='Baseline',
                   color=BASELINE_COLOR, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, massive, width, label='Massive zeroed',
                   color=MASSIVE_COLOR, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, random, width, label='Random zeroed',
                   color=RANDOM_COLOR, edgecolor='black', linewidth=0.5)

    ax.set_yscale('log')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(['RoPE', 'DroPE'])
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.set_ylim(0.5, 3000)

    # Add value labels on bars
    for bar, val in zip(bars2, massive):
        ax.annotate(f'{val:.0f}' if val > 100 else f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.yaxis.grid(True, linestyle='--', alpha=0.7, which='both')
    ax.set_axisbelow(True)

    # Add "broken" and "degraded" labels
    ax.annotate('BROKEN', xy=(0, 800), ha='center', fontsize=10,
                color=MASSIVE_COLOR, fontweight='bold')
    ax.annotate('degraded', xy=(1, 35), ha='center', fontsize=10,
                color=MASSIVE_COLOR, style='italic')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_disruption_perplexity.png', bbox_inches='tight')
    fig.savefig(output_dir / 'fig3_disruption_perplexity.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Created: fig3_disruption_perplexity")


def fig4_reliance_comparison():
    """
    Figure 4: The 82x reliance comparison (Experiment 2)
    """
    # Data: M-R difference (massive disruption % - random disruption %)
    rope_mr = 115929
    drope_mr = 1421
    ratio = rope_mr / drope_mr

    fig, ax = plt.subplots(figsize=(5, 4.5))

    bars = ax.bar(['RoPE', 'DroPE'], [rope_mr/1000, drope_mr/1000],
                  color=[ROPE_COLOR, DROPE_COLOR], edgecolor='black', linewidth=0.5, width=0.5)

    ax.set_ylabel('PPL Increase from Disruption (×1000%)')
    ax.set_ylim(0, rope_mr/1000 * 1.15)

    # Add value labels
    ax.annotate(f'{rope_mr/1000:.0f}k%', xy=(0, rope_mr/1000),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.annotate(f'{drope_mr/1000:.1f}k%', xy=(1, drope_mr/1000 + 5),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add the ratio annotation with box
    ax.annotate(f'RoPE relies\n{ratio:.0f}× more\non massive values',
                xy=(0.5, rope_mr/1000 * 0.55), ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                         edgecolor='orange', linewidth=2))

    # Add significance
    ax.annotate('p < 10⁻⁸⁷', xy=(0.5, rope_mr/1000 * 0.25),
                ha='center', fontsize=10, style='italic', color='gray')

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_reliance_comparison.png', bbox_inches='tight')
    fig.savefig(output_dir / 'fig4_reliance_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Created: fig4_reliance_comparison")


def fig5_combined_summary():
    """
    Figure 5: Combined 2x2 summary figure
    """
    fig = plt.figure(figsize=(10, 8))

    # Data
    rope_mv = {'q': (1475.5, 22.6), 'k': (1496.8, 69.8), 'v': (174.0, 10.7)}
    drope_mv = {'q': (901.4, 36.0), 'k': (1331.5, 74.1), 'v': (176.6, 5.7)}

    disruption = {
        'RoPE': {'baseline': 1.30, 'massive': 1508.5, 'random': 1.31},
        'DroPE': {'baseline': 1.49, 'massive': 22.7, 'random': 1.49},
    }

    # Panel A: Massive value counts
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(3)
    width = 0.35

    rope_means = [rope_mv['q'][0], rope_mv['k'][0], rope_mv['v'][0]]
    rope_stds = [rope_mv['q'][1], rope_mv['k'][1], rope_mv['v'][1]]
    drope_means = [drope_mv['q'][0], drope_mv['k'][0], drope_mv['v'][0]]
    drope_stds = [drope_mv['q'][1], drope_mv['k'][1], drope_mv['v'][1]]

    ax1.bar(x - width/2, rope_means, width, yerr=rope_stds, capsize=4,
            label='RoPE', color=ROPE_COLOR, alpha=0.85)
    ax1.bar(x + width/2, drope_means, width, yerr=drope_stds, capsize=4,
            label='DroPE', color=DROPE_COLOR, alpha=0.85)

    for i, (r, d) in enumerate(zip(rope_means, drope_means)):
        pct = (d - r) / r * 100
        ax1.annotate(f'{pct:+.0f}%', xy=(i + width/2, d + drope_stds[i] + 30),
                     ha='center', fontsize=9, fontweight='bold', color=DROPE_COLOR)

    ax1.set_ylabel('Massive Values')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Query', 'Key', 'Value'])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title('A. Massive Value Counts', fontweight='bold', loc='left')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Panel B: Layer distribution
    ax2 = fig.add_subplot(2, 2, 2)

    llama_dir = Path("results/llama_comparison")
    latest = max(llama_dir.glob("comparison_results_*.json"), key=lambda x: x.stat().st_mtime)
    with open(latest) as f:
        layer_data = json.load(f)

    layers = [r['layer'] for r in layer_data['rope']]
    rope_q = [r['num_massive_q'] for r in layer_data['rope']]
    drope_q = [r['num_massive_q'] for r in layer_data['drope']]

    ax2.fill_between(layers, rope_q, drope_q, alpha=0.3, color=ROPE_COLOR)
    ax2.plot(layers, rope_q, '-', color=ROPE_COLOR, label='RoPE', linewidth=1.5)
    ax2.plot(layers, drope_q, '-', color=DROPE_COLOR, label='DroPE', linewidth=1.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Query Massive Values')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('B. Query by Layer', fontweight='bold', loc='left')
    ax2.set_xlim(0, 31)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Panel C: Disruption perplexity
    ax3 = fig.add_subplot(2, 2, 3)
    x = np.arange(2)
    width = 0.25

    baseline = [disruption['RoPE']['baseline'], disruption['DroPE']['baseline']]
    massive = [disruption['RoPE']['massive'], disruption['DroPE']['massive']]
    random = [disruption['RoPE']['random'], disruption['DroPE']['random']]

    ax3.bar(x - width, baseline, width, label='Baseline', color=BASELINE_COLOR)
    ax3.bar(x, massive, width, label='Massive zeroed', color=MASSIVE_COLOR)
    ax3.bar(x + width, random, width, label='Random zeroed', color=RANDOM_COLOR)

    ax3.set_yscale('log')
    ax3.set_ylabel('Perplexity (log)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['RoPE', 'DroPE'])
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_title('C. Disruption Experiment', fontweight='bold', loc='left')
    ax3.set_ylim(0.5, 3000)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.5, which='both')

    # Panel D: Reliance ratio
    ax4 = fig.add_subplot(2, 2, 4)
    rope_mr = 115929
    drope_mr = 1421
    ratio = rope_mr / drope_mr

    bars = ax4.bar(['RoPE', 'DroPE'], [rope_mr/1000, drope_mr/1000],
                   color=[ROPE_COLOR, DROPE_COLOR], width=0.5)
    ax4.set_ylabel('M-R Difference (×1000%)')
    ax4.set_title('D. Reliance on Massive Values', fontweight='bold', loc='left')

    ax4.annotate(f'{ratio:.0f}× more\nreliant', xy=(0.5, rope_mr/1000 * 0.5),
                 ha='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    ax4.yaxis.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_combined_summary.png', bbox_inches='tight')
    fig.savefig(output_dir / 'fig5_combined_summary.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Created: fig5_combined_summary")


def main():
    print("Creating figures for FINDINGS.md...")
    print("=" * 50)

    fig1_massive_value_counts()
    fig2_layer_distribution()
    fig3_disruption_perplexity()
    fig4_reliance_comparison()
    fig5_combined_summary()

    print("=" * 50)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nTo insert in FINDINGS.md, use:")
    print("  ![Figure 1](findings_figures/fig1_massive_value_counts.png)")
    print("  ![Figure 2](findings_figures/fig2_layer_distribution.png)")
    print("  ![Figure 3](findings_figures/fig3_disruption_perplexity.png)")
    print("  ![Figure 4](findings_figures/fig4_reliance_comparison.png)")
    print("  ![Figure 5](findings_figures/fig5_combined_summary.png)")


if __name__ == "__main__":
    main()
