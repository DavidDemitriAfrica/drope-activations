#!/usr/bin/env python3
"""
Create publication-quality figures using scienceplots.
Comprehensive visualization suite for massive values analysis.
"""

import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Try to use science plots styles
try:
    plt.style.use(['science', 'ieee', 'no-latex'])
except:
    try:
        plt.style.use(['science', 'no-latex'])
    except:
        print("Warning: scienceplots styles not fully available, using default")
        plt.style.use('seaborn-v0_8-whitegrid')

# Set consistent style parameters
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

# Color palette (colorblind-friendly)
COLORS = {
    'rope': '#0077BB',      # Blue
    'drope': '#EE7733',     # Orange
    'query': '#009988',     # Teal
    'key': '#CC3311',       # Red
    'value': '#33BBEE',     # Cyan
    'smollm': '#EE3377',    # Magenta
    'llama': '#0077BB',     # Blue
}


def load_results():
    """Load all available results."""
    results = {}

    # Load Llama comparison results
    llama_dir = Path("results/llama_comparison")
    llama_files = list(llama_dir.glob("comparison_results_*.json"))
    if llama_files:
        latest = max(llama_files, key=lambda x: x.stat().st_mtime)
        with open(latest) as f:
            data = json.load(f)
            results['llama_rope'] = data['rope']
            results['llama_drope'] = data['drope']

    # Load SmolLM results
    smollm_dir = Path("results/phase1")
    smollm_dirs = list(smollm_dir.glob("smollm-360m_*"))
    if smollm_dirs:
        latest = max(smollm_dirs, key=lambda x: x.stat().st_mtime)
        analysis_file = latest / "analysis_results.json"
        if analysis_file.exists():
            with open(analysis_file) as f:
                data = json.load(f)
                # Convert to same format as llama results
                smollm_results = []
                for layer_idx in sorted(data['layers'].keys(), key=int):
                    layer_data = data['layers'][layer_idx]
                    smollm_results.append({
                        'layer': int(layer_idx),
                        'num_massive_q': layer_data['query']['num_massive'],
                        'num_massive_k': layer_data['key']['num_massive'],
                        'num_massive_v': layer_data['value']['num_massive'],
                        'concentration_q': layer_data['query']['concentration'],
                        'concentration_k': layer_data['key']['concentration'],
                        'concentration_v': layer_data['value']['concentration'],
                    })
                results['smollm_rope'] = smollm_results

    # Load SmolLM RoPE vs DroPE (unconverted) comparison
    smollm_comparison_dir = Path("results/smollm_comparison")
    smollm_comparison_files = list(smollm_comparison_dir.glob("smollm_comparison_*.json"))
    if smollm_comparison_files:
        latest = max(smollm_comparison_files, key=lambda x: x.stat().st_mtime)
        with open(latest) as f:
            data = json.load(f)
            results['smollm_rope_comparison'] = data['rope']
            results['smollm_drope_unconverted'] = data['drope_unconverted']

    return results


def fig1_rope_vs_drope_comparison(results, output_dir):
    """Figure 1: Main RoPE vs DroPE comparison for Llama-2-7B."""
    rope = results['llama_rope']
    drope = results['llama_drope']

    layers = [r['layer'] for r in rope]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Query
    ax = axes[0]
    rope_q = [r['num_massive_q'] for r in rope]
    drope_q = [r['num_massive_q'] for r in drope]
    ax.fill_between(layers, rope_q, drope_q, alpha=0.3, color=COLORS['rope'])
    ax.plot(layers, rope_q, 'o-', color=COLORS['rope'], label='RoPE', markersize=3)
    ax.plot(layers, drope_q, 's-', color=COLORS['drope'], label='DroPE', markersize=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Massive Values')
    ax.set_title('Query (Q)')
    ax.legend(loc='upper right')
    ax.set_xlim(-1, 32)

    # Key
    ax = axes[1]
    rope_k = [r['num_massive_k'] for r in rope]
    drope_k = [r['num_massive_k'] for r in drope]
    ax.fill_between(layers, rope_k, drope_k, alpha=0.3, color=COLORS['rope'],
                    where=[r > d for r, d in zip(rope_k, drope_k)])
    ax.fill_between(layers, rope_k, drope_k, alpha=0.3, color=COLORS['drope'],
                    where=[r <= d for r, d in zip(rope_k, drope_k)])
    ax.plot(layers, rope_k, 'o-', color=COLORS['rope'], label='RoPE', markersize=3)
    ax.plot(layers, drope_k, 's-', color=COLORS['drope'], label='DroPE', markersize=3)
    ax.set_xlabel('Layer')
    ax.set_title('Key (K)')
    ax.legend(loc='upper right')
    ax.set_xlim(-1, 32)

    # Value
    ax = axes[2]
    rope_v = [r['num_massive_v'] for r in rope]
    drope_v = [r['num_massive_v'] for r in drope]
    ax.plot(layers, rope_v, 'o-', color=COLORS['rope'], label='RoPE', markersize=3)
    ax.plot(layers, drope_v, 's-', color=COLORS['drope'], label='DroPE', markersize=3)
    ax.set_xlabel('Layer')
    ax.set_title('Value (V)')
    ax.legend(loc='upper right')
    ax.set_xlim(-1, 32)

    fig.suptitle('Llama-2-7B: Massive Values in RoPE vs DroPE', fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_rope_vs_drope.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig1_rope_vs_drope.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig1_rope_vs_drope")


def fig2_qkv_comparison(results, output_dir):
    """Figure 2: Q/K/V comparison within each model."""
    rope = results['llama_rope']
    drope = results['llama_drope']

    layers = [r['layer'] for r in rope]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # RoPE model
    ax = axes[0]
    ax.plot(layers, [r['num_massive_q'] for r in rope], 'o-', color=COLORS['query'],
            label='Query', markersize=3)
    ax.plot(layers, [r['num_massive_k'] for r in rope], 's-', color=COLORS['key'],
            label='Key', markersize=3)
    ax.plot(layers, [r['num_massive_v'] for r in rope], '^-', color=COLORS['value'],
            label='Value', markersize=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Massive Values')
    ax.set_title('Llama-2-7B with RoPE')
    ax.legend()
    ax.set_xlim(-1, 32)

    # DroPE model
    ax = axes[1]
    ax.plot(layers, [r['num_massive_q'] for r in drope], 'o-', color=COLORS['query'],
            label='Query', markersize=3)
    ax.plot(layers, [r['num_massive_k'] for r in drope], 's-', color=COLORS['key'],
            label='Key', markersize=3)
    ax.plot(layers, [r['num_massive_v'] for r in drope], '^-', color=COLORS['value'],
            label='Value', markersize=3)
    ax.set_xlabel('Layer')
    ax.set_title('Llama-2-7B with DroPE')
    ax.legend()
    ax.set_xlim(-1, 32)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_qkv_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig2_qkv_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig2_qkv_comparison")


def fig3_percent_change(results, output_dir):
    """Figure 3: Percent change by layer."""
    rope = results['llama_rope']
    drope = results['llama_drope']

    layers = [r['layer'] for r in rope]

    pct_q = [(d['num_massive_q'] - r['num_massive_q']) / r['num_massive_q'] * 100
             if r['num_massive_q'] > 0 else 0 for r, d in zip(rope, drope)]
    pct_k = [(d['num_massive_k'] - r['num_massive_k']) / r['num_massive_k'] * 100
             if r['num_massive_k'] > 0 else 0 for r, d in zip(rope, drope)]

    # Clip extreme values for visualization
    pct_q_clipped = [max(min(p, 100), -100) for p in pct_q]
    pct_k_clipped = [max(min(p, 100), -100) for p in pct_k]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Query
    ax = axes[0]
    colors_q = [COLORS['drope'] if p > 0 else COLORS['rope'] for p in pct_q_clipped]
    bars = ax.bar(layers, pct_q_clipped, color=colors_q, alpha=0.8, width=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=-38.5, color='gray', linestyle='--', linewidth=1, label=f'Mean: -38.5%')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Change (%)')
    ax.set_title('Query: % Change (DroPE vs RoPE)')
    ax.legend(loc='upper right')
    ax.set_xlim(-1, 32)

    # Key
    ax = axes[1]
    colors_k = [COLORS['drope'] if p > 0 else COLORS['rope'] for p in pct_k_clipped]
    bars = ax.bar(layers, pct_k_clipped, color=colors_k, alpha=0.8, width=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=-9.2, color='gray', linestyle='--', linewidth=1, label=f'Mean: -9.2%')
    ax.set_xlabel('Layer')
    ax.set_title('Key: % Change (DroPE vs RoPE)')
    ax.legend(loc='upper right')
    ax.set_xlim(-1, 32)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_percent_change.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig3_percent_change.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig3_percent_change")


def fig4_summary_bars(results, output_dir):
    """Figure 4: Summary bar chart with totals."""
    rope = results['llama_rope']
    drope = results['llama_drope']

    rope_totals = {
        'Q': sum(r['num_massive_q'] for r in rope),
        'K': sum(r['num_massive_k'] for r in rope),
        'V': sum(r['num_massive_v'] for r in rope),
    }
    drope_totals = {
        'Q': sum(r['num_massive_q'] for r in drope),
        'K': sum(r['num_massive_k'] for r in drope),
        'V': sum(r['num_massive_v'] for r in drope),
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width/2, list(rope_totals.values()), width,
                   label='RoPE', color=COLORS['rope'], alpha=0.85)
    bars2 = ax.bar(x + width/2, list(drope_totals.values()), width,
                   label='DroPE', color=COLORS['drope'], alpha=0.85)

    ax.set_ylabel('Total Massive Values')
    ax.set_title('Llama-2-7B: Total Massive Values by Tensor')
    ax.set_xticks(x)
    ax.set_xticklabels(['Query (Q)', 'Key (K)', 'Value (V)'])
    ax.legend()

    # Add percentage labels
    for i, (tensor, rope_val) in enumerate(rope_totals.items()):
        drope_val = drope_totals[tensor]
        pct = (drope_val - rope_val) / rope_val * 100
        ax.annotate(f'{pct:+.1f}%',
                    xy=(i + width/2, drope_val + 30),
                    ha='center', fontsize=9, fontweight='bold',
                    color=COLORS['drope'] if pct > 0 else COLORS['rope'])

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_summary_bars.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig4_summary_bars.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig4_summary_bars")


def fig5_correlation_scatter(results, output_dir):
    """Figure 5: Correlation scatter plots."""
    rope = results['llama_rope']
    drope = results['llama_drope']

    rope_q = [r['num_massive_q'] for r in rope]
    drope_q = [r['num_massive_q'] for r in drope]
    rope_k = [r['num_massive_k'] for r in rope]
    drope_k = [r['num_massive_k'] for r in drope]

    corr_q, _ = stats.pearsonr(rope_q, drope_q)
    corr_k, _ = stats.pearsonr(rope_k, drope_k)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Query
    ax = axes[0]
    ax.scatter(rope_q, drope_q, alpha=0.7, s=30, c=COLORS['query'], edgecolors='white', linewidth=0.5)
    max_val = max(max(rope_q), max(drope_q)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1, label='y = x')

    # Fit line
    z = np.polyfit(rope_q, drope_q, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max(rope_q), 100)
    ax.plot(x_line, p(x_line), '-', color=COLORS['query'], alpha=0.7, linewidth=1.5)

    ax.set_xlabel('RoPE Massive Values')
    ax.set_ylabel('DroPE Massive Values')
    ax.set_title(f'Query (r = {corr_q:.2f})')
    ax.legend(loc='upper left')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Key
    ax = axes[1]
    ax.scatter(rope_k, drope_k, alpha=0.7, s=30, c=COLORS['key'], edgecolors='white', linewidth=0.5)
    max_val = max(max(rope_k), max(drope_k)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1, label='y = x')

    # Fit line
    z = np.polyfit(rope_k, drope_k, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max(rope_k), 100)
    ax.plot(x_line, p(x_line), '-', color=COLORS['key'], alpha=0.7, linewidth=1.5)

    ax.set_xlabel('RoPE Massive Values')
    ax.set_ylabel('DroPE Massive Values')
    ax.set_title(f'Key (r = {corr_k:.2f})')
    ax.legend(loc='upper left')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_correlation.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig5_correlation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig5_correlation")


def fig6_smollm_vs_llama(results, output_dir):
    """Figure 6: Compare SmolLM-360M to Llama-2-7B patterns."""
    if 'smollm_rope' not in results:
        print("Skipping fig6: SmolLM results not available")
        return

    smollm = results['smollm_rope']
    llama = results['llama_rope']

    # Normalize by number of layers for comparison
    smollm_layers = [r['layer'] for r in smollm]
    llama_layers = [r['layer'] for r in llama]

    # Normalize layer index to [0, 1]
    smollm_norm_layers = [l / max(smollm_layers) for l in smollm_layers]
    llama_norm_layers = [l / max(llama_layers) for l in llama_layers]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Query
    ax = axes[0]
    ax.plot(smollm_norm_layers, [r['num_massive_q'] for r in smollm],
            'o-', color=COLORS['smollm'], label='SmolLM-360M', markersize=3, alpha=0.8)
    ax.plot(llama_norm_layers, [r['num_massive_q'] for r in llama],
            's-', color=COLORS['llama'], label='Llama-2-7B', markersize=3, alpha=0.8)
    ax.set_xlabel('Normalized Layer Position')
    ax.set_ylabel('Massive Values')
    ax.set_title('Query (Q)')
    ax.legend()

    # Key
    ax = axes[1]
    ax.plot(smollm_norm_layers, [r['num_massive_k'] for r in smollm],
            'o-', color=COLORS['smollm'], label='SmolLM-360M', markersize=3, alpha=0.8)
    ax.plot(llama_norm_layers, [r['num_massive_k'] for r in llama],
            's-', color=COLORS['llama'], label='Llama-2-7B', markersize=3, alpha=0.8)
    ax.set_xlabel('Normalized Layer Position')
    ax.set_title('Key (K)')
    ax.legend()

    # Value
    ax = axes[2]
    ax.plot(smollm_norm_layers, [r['num_massive_v'] for r in smollm],
            'o-', color=COLORS['smollm'], label='SmolLM-360M', markersize=3, alpha=0.8)
    ax.plot(llama_norm_layers, [r['num_massive_v'] for r in llama],
            's-', color=COLORS['llama'], label='Llama-2-7B', markersize=3, alpha=0.8)
    ax.set_xlabel('Normalized Layer Position')
    ax.set_title('Value (V)')
    ax.legend()

    fig.suptitle('Model Comparison: SmolLM-360M vs Llama-2-7B (RoPE)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_smollm_vs_llama.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig6_smollm_vs_llama.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig6_smollm_vs_llama")


def fig7_qv_ratio(results, output_dir):
    """Figure 7: Q/V ratio across layers."""
    rope = results['llama_rope']
    drope = results['llama_drope']

    layers = [r['layer'] for r in rope]

    rope_ratio = [r['num_massive_q'] / r['num_massive_v'] if r['num_massive_v'] > 0 else 0
                  for r in rope]
    drope_ratio = [r['num_massive_q'] / r['num_massive_v'] if r['num_massive_v'] > 0 else 0
                   for r in drope]

    # Smooth for visualization
    rope_smooth = gaussian_filter1d(rope_ratio, sigma=1)
    drope_smooth = gaussian_filter1d(drope_ratio, sigma=1)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(layers, rope_ratio, 'o', color=COLORS['rope'], alpha=0.3, markersize=4)
    ax.plot(layers, rope_smooth, '-', color=COLORS['rope'], linewidth=2, label='RoPE')
    ax.plot(layers, drope_ratio, 's', color=COLORS['drope'], alpha=0.3, markersize=4)
    ax.plot(layers, drope_smooth, '-', color=COLORS['drope'], linewidth=2, label='DroPE')

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Q/V Ratio')
    ax.set_title('Query/Value Massive Value Ratio')
    ax.legend()
    ax.set_xlim(-1, 32)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig7_qv_ratio.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig7_qv_ratio.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig7_qv_ratio")


def fig8_heatmap_difference(results, output_dir):
    """Figure 8: Heatmap of layer-wise differences."""
    rope = results['llama_rope']
    drope = results['llama_drope']

    layers = [r['layer'] for r in rope]

    # Create difference matrix
    diff_q = [(d['num_massive_q'] - r['num_massive_q']) for r, d in zip(rope, drope)]
    diff_k = [(d['num_massive_k'] - r['num_massive_k']) for r, d in zip(rope, drope)]
    diff_v = [(d['num_massive_v'] - r['num_massive_v']) for r, d in zip(rope, drope)]

    diff_matrix = np.array([diff_q, diff_k, diff_v])

    fig, ax = plt.subplots(figsize=(10, 2))

    im = ax.imshow(diff_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-max(abs(diff_matrix.min()), abs(diff_matrix.max())),
                   vmax=max(abs(diff_matrix.min()), abs(diff_matrix.max())))

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Query', 'Key', 'Value'])
    ax.set_xlabel('Layer')
    ax.set_title('DroPE - RoPE: Massive Value Difference by Layer')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Δ Massive Values')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig8_heatmap.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig8_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig8_heatmap")


def fig9_cumulative(results, output_dir):
    """Figure 9: Cumulative massive values."""
    rope = results['llama_rope']
    drope = results['llama_drope']

    layers = [r['layer'] for r in rope]

    rope_q_cum = np.cumsum([r['num_massive_q'] for r in rope])
    drope_q_cum = np.cumsum([r['num_massive_q'] for r in drope])
    rope_k_cum = np.cumsum([r['num_massive_k'] for r in rope])
    drope_k_cum = np.cumsum([r['num_massive_k'] for r in drope])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Query
    ax = axes[0]
    ax.fill_between(layers, rope_q_cum, drope_q_cum, alpha=0.3, color=COLORS['rope'])
    ax.plot(layers, rope_q_cum, '-', color=COLORS['rope'], linewidth=2, label='RoPE')
    ax.plot(layers, drope_q_cum, '-', color=COLORS['drope'], linewidth=2, label='DroPE')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cumulative Massive Values')
    ax.set_title('Query: Cumulative Sum')
    ax.legend()
    ax.set_xlim(-1, 32)

    # Key
    ax = axes[1]
    ax.fill_between(layers, rope_k_cum, drope_k_cum, alpha=0.3, color=COLORS['rope'],
                    where=[r > d for r, d in zip(rope_k_cum, drope_k_cum)])
    ax.fill_between(layers, rope_k_cum, drope_k_cum, alpha=0.3, color=COLORS['drope'],
                    where=[r <= d for r, d in zip(rope_k_cum, drope_k_cum)])
    ax.plot(layers, rope_k_cum, '-', color=COLORS['rope'], linewidth=2, label='RoPE')
    ax.plot(layers, drope_k_cum, '-', color=COLORS['drope'], linewidth=2, label='DroPE')
    ax.set_xlabel('Layer')
    ax.set_title('Key: Cumulative Sum')
    ax.legend()
    ax.set_xlim(-1, 32)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig9_cumulative.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig9_cumulative.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig9_cumulative")


def fig_smollm_drope_comparison(results, output_dir):
    """SmolLM RoPE vs DroPE (unconverted) comparison - shows where massive values originate."""
    if 'smollm_rope_comparison' not in results:
        print("Skipping fig_smollm_drope: SmolLM comparison results not available")
        return

    rope = results['smollm_rope_comparison']
    drope = results['smollm_drope_unconverted']

    layers = [r['layer'] for r in rope]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    # Top row: Absolute values
    for ax, (tensor, key, title) in zip(axes[0], [
        ('Query', 'num_massive_q', 'Query (Q)'),
        ('Key', 'num_massive_k', 'Key (K)'),
        ('Value', 'num_massive_v', 'Value (V)')
    ]):
        rope_vals = [r[key] for r in rope]
        drope_vals = [r[key] for r in drope]

        ax.plot(layers, rope_vals, 'o-', color=COLORS['rope'], label='RoPE', markersize=4, linewidth=1.5)
        ax.plot(layers, drope_vals, 's--', color=COLORS['drope'], label='DroPE (unconverted)', markersize=4, linewidth=1.5, alpha=0.8)

        # Highlight layer 0 with a box
        ax.axvspan(-0.5, 0.5, alpha=0.2, color='green')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Massive Values')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom row: Differences
    for ax, (tensor, key, title) in zip(axes[1], [
        ('Query', 'num_massive_q', 'Query Δ'),
        ('Key', 'num_massive_k', 'Key Δ'),
        ('Value', 'num_massive_v', 'Value Δ')
    ]):
        rope_vals = [r[key] for r in rope]
        drope_vals = [r[key] for r in drope]
        diffs = [d - r for r, d in zip(rope_vals, drope_vals)]

        colors = [COLORS['drope'] if d > 0 else COLORS['rope'] for d in diffs]
        ax.bar(layers, diffs, color=colors, alpha=0.7, width=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Highlight layer 0
        ax.axvspan(-0.5, 0.5, alpha=0.2, color='green')
        ax.set_xlabel('Layer')
        ax.set_ylabel('DroPE - RoPE')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('SmolLM-360M: RoPE vs DroPE (Unconverted)\n'
                 'Layer 0 identical = Massive values are in projection weights\n'
                 'Later layers differ = Attention patterns affect residual stream',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_smollm_drope_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_smollm_drope_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig_smollm_drope_comparison")


def fig_layer0_insight(results, output_dir):
    """Focused figure showing the Layer 0 insight - projections are identical."""
    if 'smollm_rope_comparison' not in results:
        print("Skipping fig_layer0_insight: SmolLM comparison results not available")
        return

    rope = results['smollm_rope_comparison']
    drope = results['smollm_drope_unconverted']

    # Calculate cumulative absolute difference from layer 0
    layers = [r['layer'] for r in rope]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, (tensor, key, title) in zip(axes, [
        ('Query', 'num_massive_q', 'Query'),
        ('Key', 'num_massive_k', 'Key'),
        ('Value', 'num_massive_v', 'Value')
    ]):
        rope_vals = [r[key] for r in rope]
        drope_vals = [r[key] for r in drope]
        abs_diffs = [abs(d - r) for r, d in zip(rope_vals, drope_vals)]

        # Cumulative absolute difference
        cum_diff = np.cumsum(abs_diffs)

        ax.bar(layers, abs_diffs, color=COLORS['query' if tensor == 'Query' else 'key' if tensor == 'Key' else 'value'],
               alpha=0.6, width=0.8, label='Layer difference')
        ax.plot(layers, cum_diff / 10, 'k-', linewidth=2, label='Cumulative (÷10)')

        # Annotate layer 0
        ax.annotate(f'Layer 0: {abs_diffs[0]:.1f}',
                    xy=(0, abs_diffs[0]),
                    xytext=(5, abs_diffs[0] + 3),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

        ax.set_xlabel('Layer')
        ax.set_ylabel('|DroPE - RoPE|')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('SmolLM: How Differences Accumulate Through Layers\n'
                 'Layer 0 ~0 difference confirms massive values originate in weights',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_layer0_insight.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_layer0_insight.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig_layer0_insight")


def fig10_main_figure(results, output_dir):
    """Figure 10: Main summary figure for paper."""
    rope = results['llama_rope']
    drope = results['llama_drope']

    layers = [r['layer'] for r in rope]

    fig = plt.figure(figsize=(12, 8))

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # Panel A: Q comparison
    ax_a = fig.add_subplot(gs[0, :2])
    rope_q = [r['num_massive_q'] for r in rope]
    drope_q = [r['num_massive_q'] for r in drope]
    ax_a.fill_between(layers, rope_q, drope_q, alpha=0.3, color=COLORS['rope'])
    ax_a.plot(layers, rope_q, 'o-', color=COLORS['rope'], label='RoPE', markersize=3)
    ax_a.plot(layers, drope_q, 's-', color=COLORS['drope'], label='DroPE', markersize=3)
    ax_a.set_xlabel('Layer')
    ax_a.set_ylabel('Massive Values')
    ax_a.set_title('A. Query Massive Values')
    ax_a.legend(loc='upper right')
    ax_a.text(-0.1, 1.05, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold')

    # Panel B: K comparison
    ax_b = fig.add_subplot(gs[0, 2:])
    rope_k = [r['num_massive_k'] for r in rope]
    drope_k = [r['num_massive_k'] for r in drope]
    ax_b.plot(layers, rope_k, 'o-', color=COLORS['rope'], label='RoPE', markersize=3)
    ax_b.plot(layers, drope_k, 's-', color=COLORS['drope'], label='DroPE', markersize=3)
    ax_b.set_xlabel('Layer')
    ax_b.set_ylabel('Massive Values')
    ax_b.set_title('B. Key Massive Values')
    ax_b.legend(loc='upper right')
    ax_b.text(-0.1, 1.05, 'B', transform=ax_b.transAxes, fontsize=14, fontweight='bold')

    # Panel C: Summary bars
    ax_c = fig.add_subplot(gs[1, :2])
    rope_totals = [sum(r['num_massive_q'] for r in rope),
                   sum(r['num_massive_k'] for r in rope),
                   sum(r['num_massive_v'] for r in rope)]
    drope_totals = [sum(r['num_massive_q'] for r in drope),
                    sum(r['num_massive_k'] for r in drope),
                    sum(r['num_massive_v'] for r in drope)]
    x = np.arange(3)
    width = 0.35
    ax_c.bar(x - width/2, rope_totals, width, label='RoPE', color=COLORS['rope'], alpha=0.85)
    ax_c.bar(x + width/2, drope_totals, width, label='DroPE', color=COLORS['drope'], alpha=0.85)
    ax_c.set_ylabel('Total Massive Values')
    ax_c.set_title('C. Totals by Tensor Type')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(['Q', 'K', 'V'])
    ax_c.legend()
    for i, (r, d) in enumerate(zip(rope_totals, drope_totals)):
        pct = (d - r) / r * 100
        ax_c.annotate(f'{pct:+.0f}%', xy=(i + width/2, d + 30), ha='center', fontsize=8,
                      color=COLORS['drope'] if pct > 0 else COLORS['rope'])
    ax_c.text(-0.1, 1.05, 'C', transform=ax_c.transAxes, fontsize=14, fontweight='bold')

    # Panel D: Scatter correlation
    ax_d = fig.add_subplot(gs[1, 2:])
    corr_q, _ = stats.pearsonr(rope_q, drope_q)
    ax_d.scatter(rope_q, drope_q, alpha=0.6, s=25, c=COLORS['query'], edgecolors='white', linewidth=0.5)
    max_val = max(max(rope_q), max(drope_q)) * 1.1
    ax_d.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1)
    ax_d.set_xlabel('RoPE Query')
    ax_d.set_ylabel('DroPE Query')
    ax_d.set_title(f'D. Layer Correlation (r = {corr_q:.2f})')
    ax_d.text(-0.1, 1.05, 'D', transform=ax_d.transAxes, fontsize=14, fontweight='bold')

    # Panel E: Heatmap
    ax_e = fig.add_subplot(gs[2, :])
    diff_q = [(d['num_massive_q'] - r['num_massive_q']) for r, d in zip(rope, drope)]
    diff_k = [(d['num_massive_k'] - r['num_massive_k']) for r, d in zip(rope, drope)]
    diff_v = [(d['num_massive_v'] - r['num_massive_v']) for r, d in zip(rope, drope)]
    diff_matrix = np.array([diff_q, diff_k, diff_v])
    im = ax_e.imshow(diff_matrix, aspect='auto', cmap='RdBu_r',
                     vmin=-max(abs(diff_matrix.min()), abs(diff_matrix.max())),
                     vmax=max(abs(diff_matrix.min()), abs(diff_matrix.max())))
    ax_e.set_yticks([0, 1, 2])
    ax_e.set_yticklabels(['Query', 'Key', 'Value'])
    ax_e.set_xlabel('Layer')
    ax_e.set_title('E. DroPE − RoPE: Massive Value Difference')
    cbar = plt.colorbar(im, ax=ax_e, shrink=0.5, pad=0.02)
    cbar.set_label('Δ')
    ax_e.text(-0.05, 1.05, 'E', transform=ax_e.transAxes, fontsize=14, fontweight='bold')

    fig.suptitle('Massive Values in Llama-2-7B: RoPE vs DroPE Comparison',
                 fontsize=14, fontweight='bold', y=0.98)

    fig.savefig(output_dir / 'fig10_main_figure.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig10_main_figure.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Created: fig10_main_figure")


def main():
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results()

    print(f"\nAvailable data: {list(results.keys())}")
    print(f"\nGenerating publication-quality figures...")
    print("=" * 50)

    # Generate all figures
    fig1_rope_vs_drope_comparison(results, output_dir)
    fig2_qkv_comparison(results, output_dir)
    fig3_percent_change(results, output_dir)
    fig4_summary_bars(results, output_dir)
    fig5_correlation_scatter(results, output_dir)
    fig6_smollm_vs_llama(results, output_dir)
    fig7_qv_ratio(results, output_dir)
    fig8_heatmap_difference(results, output_dir)
    fig9_cumulative(results, output_dir)
    fig10_main_figure(results, output_dir)

    # SmolLM DroPE (unconverted) comparison figures
    fig_smollm_drope_comparison(results, output_dir)
    fig_layer0_insight(results, output_dir)

    print("=" * 50)
    print(f"\nAll figures saved to: {output_dir}")
    print(f"  - PDF versions for papers")
    print(f"  - PNG versions for quick viewing")


if __name__ == "__main__":
    main()
