#!/usr/bin/env python3
"""Create figures for Jin et al. knowledge test results."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results from the tests
RESULTS = {
    'RoPE': {
        'parametric': {
            'cities': {'baseline': 0.895, 'disrupted': 0.575, 'degradation': 35.8},
            'sports': {'baseline': 0.60, 'disrupted': 0.52, 'degradation': 13.3},
        },
        'contextual': {
            'imdb': {'baseline': 0.44, 'disrupted': 0.05, 'degradation': 88.6},
            'passkey': {'baseline': 1.0, 'disrupted': 0.0, 'degradation': 100.0},
        }
    },
    'DroPE': {
        'parametric': {
            'cities': {'baseline': 0.79, 'disrupted': 0.885, 'degradation': -12.0},
            'sports': {'baseline': 0.73, 'disrupted': 0.67, 'degradation': 8.2},
        },
        'contextual': {
            'imdb': {'baseline': 0.30, 'disrupted': 0.15, 'degradation': 50.0},
            'passkey': {'baseline': 0.60, 'disrupted': 0.60, 'degradation': 0.0},
        }
    }
}

def setup_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def fig7_degradation_comparison(output_dir: Path):
    """Figure 7: Degradation comparison - parametric vs contextual."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Parametric\n(Cities)', 'Parametric\n(Sports)', 'Contextual\n(IMDB)', 'Contextual\n(Passkey)']
    rope_deg = [35.8, 13.3, 88.6, 100.0]
    drope_deg = [-12.0, 8.2, 50.0, 0.0]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, rope_deg, width, label='RoPE', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, drope_deg, width, label='DroPE', color='#3498db', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add text annotations
    ax.text(0.5, 105, 'Parametric Tasks', ha='center', fontsize=12, style='italic')
    ax.text(2.5, 105, 'Contextual Tasks', ha='center', fontsize=12, style='italic')

    ax.set_ylabel('Degradation (%)', fontsize=14)
    ax.set_title('Accuracy Degradation When Massive Values Are Disrupted', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    ax.set_ylim(-20, 115)

    # Add value labels on bars
    for bar, val in zip(bars1, rope_deg):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, color='#e74c3c')
    for bar, val in zip(bars2, drope_deg):
        ypos = bar.get_height() + 2 if val >= 0 else bar.get_height() - 8
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, color='#3498db')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_degradation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig7_degradation_comparison.png")

def fig8_passkey_spotlight(output_dir: Path):
    """Figure 8: Passkey retrieval - the most dramatic result."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Baseline vs Disrupted accuracy
    ax1 = axes[0]
    models = ['RoPE', 'DroPE']
    baseline = [100, 60]
    disrupted = [0, 60]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline, width, label='Baseline', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, disrupted, width, label='Disrupted', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    ax1.set_title('Passkey Retrieval Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=13)
    ax1.legend()
    ax1.set_ylim(0, 115)

    # Add value labels
    for bar, val in zip(bars1, baseline):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    for bar, val in zip(bars2, disrupted):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Right: Degradation comparison with annotation
    ax2 = axes[1]
    degradation = [100, 0]
    colors = ['#e74c3c', '#3498db']

    bars = ax2.bar(models, degradation, color=colors, alpha=0.8, width=0.5)
    ax2.set_ylabel('Degradation (%)', fontsize=14)
    ax2.set_title('Passkey Degradation from Disruption', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 120)

    # Add dramatic annotations
    ax2.annotate('COMPLETE\nCOLLAPSE', xy=(0, 100), xytext=(0.3, 85),
                fontsize=11, fontweight='bold', color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b'))
    ax2.annotate('ZERO\nDEGRADATION', xy=(1, 0), xytext=(0.7, 25),
                fontsize=11, fontweight='bold', color='#2980b9',
                arrowprops=dict(arrowstyle='->', color='#2980b9'))

    # Add value labels
    for bar, val in zip(bars, degradation):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_passkey_spotlight.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig8_passkey_spotlight.png")

def fig9_category_averages(output_dir: Path):
    """Figure 9: Average degradation by category."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Parametric', 'Contextual']
    rope_avg = [24.5, 94.3]
    drope_avg = [-1.9, 25.0]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, rope_avg, width, label='RoPE', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, drope_avg, width, label='DroPE', color='#3498db', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_ylabel('Average Degradation (%)', fontsize=14)
    ax.set_title('Average Degradation by Knowledge Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.legend(loc='upper left')
    ax.set_ylim(-15, 110)

    # Add value labels
    for bar, val in zip(bars1, rope_avg):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#e74c3c')
    for bar, val in zip(bars2, drope_avg):
        ypos = bar.get_height() + 2 if val >= 0 else bar.get_height() - 10
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#3498db')

    # Add ratio annotation
    ax.annotate('3.8x more\ncontextual\ndegradation', xy=(1.175, 94.3), xytext=(1.5, 70),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig9_category_averages.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig9_category_averages.png")

def fig10_baseline_vs_disrupted(output_dir: Path):
    """Figure 10: All tasks - baseline vs disrupted accuracy."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tasks = ['Cities', 'Sports', 'IMDB', 'Passkey']

    # RoPE
    ax1 = axes[0]
    rope_baseline = [89.5, 60.0, 44.0, 100.0]
    rope_disrupted = [57.5, 52.0, 5.0, 0.0]

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax1.bar(x - width/2, rope_baseline, width, label='Baseline', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rope_disrupted, width, label='Disrupted', color='#e74c3c', alpha=0.8)

    ax1.axvline(x=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    ax1.set_title('RoPE: Baseline vs Disrupted', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 115)
    ax1.text(0.5, 108, 'Parametric', ha='center', fontsize=11, style='italic')
    ax1.text(2.5, 108, 'Contextual', ha='center', fontsize=11, style='italic')

    # DroPE
    ax2 = axes[1]
    drope_baseline = [79.0, 73.0, 30.0, 60.0]
    drope_disrupted = [88.5, 67.0, 15.0, 60.0]

    bars3 = ax2.bar(x - width/2, drope_baseline, width, label='Baseline', color='#2ecc71', alpha=0.8)
    bars4 = ax2.bar(x + width/2, drope_disrupted, width, label='Disrupted', color='#3498db', alpha=0.8)

    ax2.axvline(x=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_title('DroPE: Baseline vs Disrupted', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 115)
    ax2.text(0.5, 108, 'Parametric', ha='center', fontsize=11, style='italic')
    ax2.text(2.5, 108, 'Contextual', ha='center', fontsize=11, style='italic')

    # Highlight Cities improvement in DroPE
    ax2.annotate('Improved!', xy=(0.175, 88.5), xytext=(0.5, 100),
                fontsize=10, fontweight='bold', color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60'))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig10_baseline_vs_disrupted.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig10_baseline_vs_disrupted.png")

def fig11_jin_summary(output_dir: Path):
    """Figure 11: Combined summary figure for Jin et al. replication."""
    setup_style()
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top left: Category averages
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Parametric', 'Contextual']
    rope_avg = [24.5, 94.3]
    drope_avg = [-1.9, 25.0]
    x = np.arange(len(categories))
    width = 0.35
    ax1.bar(x - width/2, rope_avg, width, label='RoPE', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, drope_avg, width, label='DroPE', color='#3498db', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Average Degradation (%)')
    ax1.set_title('A. Average Degradation by Category', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(loc='upper left')
    ax1.set_ylim(-15, 105)

    # Top right: Passkey spotlight
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['RoPE', 'DroPE']
    degradation = [100, 0]
    colors = ['#e74c3c', '#3498db']
    bars = ax2.bar(models, degradation, color=colors, alpha=0.8, width=0.5)
    ax2.set_ylabel('Degradation (%)')
    ax2.set_title('B. Passkey Retrieval Degradation', fontweight='bold')
    ax2.set_ylim(0, 120)
    for bar, val in zip(bars, degradation):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Bottom: All tasks comparison
    ax3 = fig.add_subplot(gs[1, :])
    tasks = ['Cities\n(Param)', 'Sports\n(Param)', 'IMDB\n(Context)', 'Passkey\n(Context)']
    rope_deg = [35.8, 13.3, 88.6, 100.0]
    drope_deg = [-12.0, 8.2, 50.0, 0.0]
    x = np.arange(len(tasks))
    width = 0.35
    bars1 = ax3.bar(x - width/2, rope_deg, width, label='RoPE', color='#e74c3c', alpha=0.8)
    bars2 = ax3.bar(x + width/2, drope_deg, width, label='DroPE', color='#3498db', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Degradation (%)')
    ax3.set_title('C. Degradation by Task', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tasks)
    ax3.legend(loc='upper left')
    ax3.set_ylim(-25, 115)

    # Value labels
    for bar, val in zip(bars1, rope_deg):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, color='#c0392b')
    for bar, val in zip(bars2, drope_deg):
        ypos = bar.get_height() + 2 if val >= 0 else bar.get_height() - 12
        ax3.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, color='#2980b9')

    fig.suptitle('Jin et al. Knowledge Test Results: RoPE vs DroPE', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'fig11_jin_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig11_jin_summary.png")


def main():
    output_dir = Path("results/findings_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating Jin et al. knowledge test figures...")
    print("=" * 50)

    fig7_degradation_comparison(output_dir)
    fig8_passkey_spotlight(output_dir)
    fig9_category_averages(output_dir)
    fig10_baseline_vs_disrupted(output_dir)
    fig11_jin_summary(output_dir)

    print("=" * 50)
    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
