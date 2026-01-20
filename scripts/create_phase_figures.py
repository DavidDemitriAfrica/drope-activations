#!/usr/bin/env python3
"""
Create figures for phase metrics analysis (Queipo-style).

Generates:
- fig_bos_norm.png: BOS norm and ratio comparison
- fig_entropy.png: Representation entropy comparison
- fig_sink_rate.png: Attention sink rate comparison
- fig_interventions.png: Effect of interventions on metrics
- fig_functional.png: Functional evaluation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def setup_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_results(results_path: str) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_bos_metrics(results: dict, output_dir: Path):
    """Plot BOS norm and BOS ratio across layers."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'RoPE': '#e74c3c', 'DroPE': '#3498db'}

    for model_key in ['RoPE', 'DroPE']:
        baseline = results[model_key]['baseline']
        n_layers = len(baseline['bos_norm']['mean'])
        layers = np.arange(n_layers)

        # BOS Norm
        ax = axes[0]
        mean = np.array(baseline['bos_norm']['mean'])
        std = np.array(baseline['bos_norm']['std'])
        ax.plot(layers, mean, color=colors[model_key], label=model_key, linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, color=colors[model_key], alpha=0.2)

        # BOS Ratio
        ax = axes[1]
        mean = np.array(baseline['bos_ratio']['mean'])
        std = np.array(baseline['bos_ratio']['std'])
        ax.plot(layers, mean, color=colors[model_key], label=model_key, linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, color=colors[model_key], alpha=0.2)

    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('BOS Norm')
    axes[0].set_title('BOS Token L2 Norm by Layer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('BOS Ratio')
    axes[1].set_title('BOS Ratio (BOS Norm / Mean Non-BOS Norm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_bos_norm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_bos_norm.png")


def plot_entropy_metrics(results: dict, output_dir: Path):
    """Plot representation entropy and anisotropy across layers."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'RoPE': '#e74c3c', 'DroPE': '#3498db'}

    for model_key in ['RoPE', 'DroPE']:
        baseline = results[model_key]['baseline']
        n_layers = len(baseline['entropy']['mean'])
        layers = np.arange(n_layers)

        # Entropy
        ax = axes[0]
        mean = np.array(baseline['entropy']['mean'])
        std = np.array(baseline['entropy']['std'])
        ax.plot(layers, mean, color=colors[model_key], label=model_key, linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, color=colors[model_key], alpha=0.2)

        # Anisotropy
        ax = axes[1]
        mean = np.array(baseline['anisotropy']['mean'])
        std = np.array(baseline['anisotropy']['std'])
        ax.plot(layers, mean, color=colors[model_key], label=model_key, linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, color=colors[model_key], alpha=0.2)

    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Entropy H(X)')
    axes[0].set_title('Representation Entropy (Compression Valley)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Anisotropy (p1)')
    axes[1].set_title('Anisotropy (Top Singular Value Fraction)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_entropy.png")


def plot_sink_metrics(results: dict, output_dir: Path):
    """Plot attention sink rate across layers."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {'RoPE': '#e74c3c', 'DroPE': '#3498db'}

    for model_key in ['RoPE', 'DroPE']:
        baseline = results[model_key]['baseline']
        n_layers = len(baseline['sink_rate']['mean'])
        layers = np.arange(n_layers)

        mean = np.array(baseline['sink_rate']['mean'])
        std = np.array(baseline['sink_rate']['std'])
        ax.plot(layers, mean, color=colors[model_key], label=model_key, linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, color=colors[model_key], alpha=0.2)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Sink Rate')
    ax.set_title('Attention Sink Rate by Layer (τ=0.3)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_sink_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_sink_rate.png")


def plot_interventions(results: dict, output_dir: Path):
    """Plot effect of interventions on key metrics."""
    setup_style()

    # Check if intervention data exists
    if 'interventions' not in results['RoPE'] or not results['RoPE']['interventions']:
        print("Skipping intervention plots - no intervention data")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = {'RoPE': '#e74c3c', 'DroPE': '#3498db'}
    intervention_styles = {
        'baseline': '-',
        'bos_mlp_ablation': '--',
        'qk_disruption': ':',
        'combined': '-.',
    }

    for col, model_key in enumerate(['RoPE', 'DroPE']):
        # Get available interventions
        interventions = ['baseline'] + list(results[model_key].get('interventions', {}).keys())

        for intervention in interventions:
            if intervention == 'baseline':
                data = results[model_key]['baseline']
            else:
                data = results[model_key]['interventions'].get(intervention, {})
                if not data:
                    continue

            n_layers = len(data['bos_ratio']['mean'])
            layers = np.arange(n_layers)

            style = intervention_styles.get(intervention, '-')
            label = intervention.replace('_', ' ').title()

            # BOS Ratio
            axes[0, col].plot(layers, data['bos_ratio']['mean'],
                            linestyle=style, label=label, linewidth=2)

            # Entropy
            axes[1, col].plot(layers, data['entropy']['mean'],
                            linestyle=style, label=label, linewidth=2)

        axes[0, col].set_title(f'{model_key}: BOS Ratio')
        axes[0, col].set_xlabel('Layer')
        axes[0, col].set_ylabel('BOS Ratio')
        axes[0, col].legend(loc='upper right')
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].set_title(f'{model_key}: Entropy')
        axes[1, col].set_xlabel('Layer')
        axes[1, col].set_ylabel('Entropy')
        axes[1, col].legend(loc='lower right')
        axes[1, col].grid(True, alpha=0.3)

    # Hide unused subplot
    axes[0, 2].axis('off')
    axes[1, 2].axis('off')

    # Add summary text in the empty subplot area
    axes[0, 2].text(0.5, 0.5, 'Intervention Effects:\n\n'
                   '- Baseline: Normal operation\n'
                   '- BOS-MLP Ablation: Zero MLP output for BOS\n'
                   '- Q/K Disruption: Zero massive values\n'
                   '- Combined: Both interventions',
                   transform=axes[0, 2].transAxes,
                   ha='center', va='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_interventions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_interventions.png")


def plot_functional(results: dict, output_dir: Path):
    """Plot functional evaluation results."""
    setup_style()

    # Check if functional data exists
    if 'functional' not in results['RoPE'] or not results['RoPE']['functional']:
        print("Skipping functional plots - no functional data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = {'RoPE': '#e74c3c', 'DroPE': '#3498db'}

    # Get available conditions
    conditions = list(results['RoPE']['functional'].keys())

    # Perplexity
    ax = axes[0]
    x = np.arange(len(conditions))
    width = 0.35

    rope_ppl = [results['RoPE']['functional'][c]['perplexity'] for c in conditions]
    drope_ppl = [results['DroPE']['functional'][c]['perplexity'] for c in conditions]

    ax.bar(x - width/2, rope_ppl, width, label='RoPE', color=colors['RoPE'], alpha=0.8)
    ax.bar(x + width/2, drope_ppl, width, label='DroPE', color=colors['DroPE'], alpha=0.8)
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity by Condition')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=9)
    ax.legend()
    ax.set_yscale('log')

    # Passkey
    ax = axes[1]
    rope_pass = [results['RoPE']['functional'][c]['passkey'] * 100 for c in conditions]
    drope_pass = [results['DroPE']['functional'][c]['passkey'] * 100 for c in conditions]

    ax.bar(x - width/2, rope_pass, width, label='RoPE', color=colors['RoPE'], alpha=0.8)
    ax.bar(x + width/2, drope_pass, width, label='DroPE', color=colors['DroPE'], alpha=0.8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Passkey Retrieval')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 105)

    # IMDB
    ax = axes[2]
    rope_imdb = [results['RoPE']['functional'][c]['imdb'] * 100 for c in conditions]
    drope_imdb = [results['DroPE']['functional'][c]['imdb'] * 100 for c in conditions]

    ax.bar(x - width/2, rope_imdb, width, label='RoPE', color=colors['RoPE'], alpha=0.8)
    ax.bar(x + width/2, drope_imdb, width, label='DroPE', color=colors['DroPE'], alpha=0.8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('IMDB Sentiment')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_functional.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_functional.png")


def plot_combined_summary(results: dict, output_dir: Path):
    """Create a combined summary figure."""
    setup_style()
    fig = plt.figure(figsize=(16, 12))

    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    colors = {'RoPE': '#e74c3c', 'DroPE': '#3498db'}

    # Top row: BOS metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for model_key in ['RoPE', 'DroPE']:
        baseline = results[model_key]['baseline']
        n_layers = len(baseline['bos_norm']['mean'])
        layers = np.arange(n_layers)

        ax1.plot(layers, baseline['bos_norm']['mean'], color=colors[model_key],
                label=model_key, linewidth=2)
        ax2.plot(layers, baseline['bos_ratio']['mean'], color=colors[model_key],
                label=model_key, linewidth=2)
        ax3.plot(layers, baseline['sink_rate']['mean'], color=colors[model_key],
                label=model_key, linewidth=2)

    ax1.set_title('A. BOS Norm')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('L2 Norm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('B. BOS Ratio')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    ax3.set_title('C. Sink Rate')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Middle row: Entropy and anisotropy
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])

    for model_key in ['RoPE', 'DroPE']:
        baseline = results[model_key]['baseline']
        layers = np.arange(len(baseline['entropy']['mean']))

        ax4.plot(layers, baseline['entropy']['mean'], color=colors[model_key],
                label=model_key, linewidth=2)
        ax5.plot(layers, baseline['anisotropy']['mean'], color=colors[model_key],
                label=model_key, linewidth=2)

    ax4.set_title('D. Representation Entropy')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Entropy H(X)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5.set_title('E. Anisotropy')
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('p1 (Top SV Fraction)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Summary stats in remaining subplot
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_text = "Summary Statistics:\n\n"
    for model_key in ['RoPE', 'DroPE']:
        baseline = results[model_key]['baseline']
        max_bos_ratio = max(baseline['bos_ratio']['mean'])
        max_bos_layer = baseline['bos_ratio']['mean'].index(max_bos_ratio)
        min_entropy = min(baseline['entropy']['mean'])
        min_entropy_layer = baseline['entropy']['mean'].index(min_entropy)
        max_sink = max(baseline['sink_rate']['mean'])
        max_sink_layer = baseline['sink_rate']['mean'].index(max_sink)

        summary_text += f"{model_key}:\n"
        summary_text += f"  Max BOS ratio: {max_bos_ratio:.2f} (L{max_bos_layer})\n"
        summary_text += f"  Min entropy: {min_entropy:.2f} (L{min_entropy_layer})\n"
        summary_text += f"  Max sink rate: {max_sink:.1%} (L{max_sink_layer})\n\n"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')

    # Bottom row: Functional results if available
    if 'functional' in results['RoPE'] and results['RoPE']['functional']:
        ax7 = fig.add_subplot(gs[2, :])

        conditions = list(results['RoPE']['functional'].keys())
        x = np.arange(len(conditions))
        width = 0.15

        metrics = ['perplexity', 'passkey', 'imdb']
        metric_labels = ['Perplexity (log)', 'Passkey (%)', 'IMDB (%)']

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            rope_vals = [results['RoPE']['functional'][c][metric] for c in conditions]
            drope_vals = [results['DroPE']['functional'][c][metric] for c in conditions]

            if metric == 'perplexity':
                rope_vals = [np.log10(v) for v in rope_vals]
                drope_vals = [np.log10(v) for v in drope_vals]
            else:
                rope_vals = [v * 100 for v in rope_vals]
                drope_vals = [v * 100 for v in drope_vals]

            offset = (i - 1) * width * 2.5
            ax7.bar(x + offset - width/2, rope_vals, width, label=f'RoPE {label}',
                   color=colors['RoPE'], alpha=0.4 + i*0.2)
            ax7.bar(x + offset + width/2, drope_vals, width, label=f'DroPE {label}',
                   color=colors['DroPE'], alpha=0.4 + i*0.2)

        ax7.set_title('F. Functional Evaluations')
        ax7.set_xticks(x)
        ax7.set_xticklabels([c.replace('_', ' ').title() for c in conditions])
        ax7.legend(loc='upper right', ncol=3)
        ax7.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Phase Metrics Analysis: RoPE vs DroPE', fontsize=16, fontweight='bold')

    plt.savefig(output_dir / 'fig_phase_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_phase_summary.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create phase metrics figures")
    parser.add_argument("--results", type=str, default="results/phase_metrics/rope_vs_drope_phase_metrics.json",
                       help="Path to results JSON")
    parser.add_argument("--output_dir", type=str, default="results/phase_metrics",
                       help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results}")
    results = load_results(args.results)

    print("\nCreating figures...")
    plot_bos_metrics(results, output_dir)
    plot_entropy_metrics(results, output_dir)
    plot_sink_metrics(results, output_dir)
    plot_interventions(results, output_dir)
    plot_functional(results, output_dir)
    plot_combined_summary(results, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
