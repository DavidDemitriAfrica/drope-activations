#!/usr/bin/env python3
"""
Create figures for Experiment 7: Attention Pattern Analysis

Figures:
1. Attention entropy by layer (RoPE vs DroPE)
2. Head type distribution comparison
3. Attention heatmaps for key layers (Layer 1, 15, 31)
4. Attention decay profiles
5. Layer 1 detailed comparison
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def fig_entropy_by_layer(results):
    """Figure: Attention entropy by layer."""
    fig, ax = plt.subplots(figsize=(10, 4))

    rope = results["RoPE"]["attention_analysis"]["summary"]["mean_entropy_by_layer"]
    drope = results["DroPE"]["attention_analysis"]["summary"]["mean_entropy_by_layer"]

    layers = range(len(rope))
    ax.plot(layers, rope, 'o-', color=ROPE_COLOR, label='RoPE', markersize=4)
    ax.plot(layers, drope, 's-', color=DROPE_COLOR, label='DroPE', markersize=4)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Attention Entropy')
    ax.set_title('Attention Entropy by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Highlight layer 1
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='Layer 1')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_attn_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_attn_entropy.png")


def fig_head_type_distribution(results):
    """Figure: Head type distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    head_types = ["sink", "local", "distributed", "mixed"]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']

    for idx, (model_key, ax) in enumerate([("RoPE", axes[0]), ("DroPE", axes[1])]):
        counts = results[model_key]["attention_analysis"]["summary"]["head_type_counts"]
        total = sum(counts.values())
        sizes = [counts[ht] / total * 100 for ht in head_types]

        ax.pie(sizes, labels=head_types, colors=colors, autopct='%1.1f%%',
               startangle=90)
        ax.set_title(f'{model_key} Head Types')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_head_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_head_types.png")


def fig_attention_heatmaps(results):
    """Figure: Attention heatmaps for key layers."""
    # Load numpy matrices
    rope_matrices = np.load(OUTPUT_DIR / 'rope_attention_matrices.npz')
    drope_matrices = np.load(OUTPUT_DIR / 'drope_attention_matrices.npz')

    layers = [1, 15, 31]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for col, layer in enumerate(layers):
        key = f'layer_{layer}'
        if key not in rope_matrices or key not in drope_matrices:
            continue

        # Average across heads for visualization
        rope_avg = rope_matrices[key].mean(axis=0)  # [seq, seq]
        drope_avg = drope_matrices[key].mean(axis=0)

        # RoPE
        im0 = axes[0, col].imshow(rope_avg[:50, :50], cmap='viridis', aspect='auto')
        axes[0, col].set_title(f'RoPE Layer {layer}')
        axes[0, col].set_xlabel('Key Position')
        axes[0, col].set_ylabel('Query Position')
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

        # DroPE
        im1 = axes[1, col].imshow(drope_avg[:50, :50], cmap='viridis', aspect='auto')
        axes[1, col].set_title(f'DroPE Layer {layer}')
        axes[1, col].set_xlabel('Key Position')
        axes[1, col].set_ylabel('Query Position')
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

    plt.suptitle('Attention Patterns (averaged across heads, first 50 positions)', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_attn_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_attn_heatmaps.png")


def fig_decay_profiles(results):
    """Figure: Attention decay profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, (model_key, ax) in enumerate([("RoPE", axes[0]), ("DroPE", axes[1])]):
        decay = results[model_key]["attention_analysis"]["decay_profile_by_layer"]

        # Average across all layers and heads
        all_decay = []
        for layer_key in decay:
            layer_decay = np.array(decay[layer_key])  # [heads, distance]
            all_decay.append(layer_decay.mean(axis=0))  # Average across heads

        avg_decay = np.mean(all_decay, axis=0)
        std_decay = np.std(all_decay, axis=0)

        distances = range(len(avg_decay))
        color = ROPE_COLOR if model_key == "RoPE" else DROPE_COLOR

        ax.plot(distances, avg_decay, color=color, linewidth=2)
        ax.fill_between(distances, avg_decay - std_decay, avg_decay + std_decay,
                        color=color, alpha=0.2)
        ax.set_xlabel('Distance (query - key)')
        ax.set_ylabel('Mean Attention')
        ax.set_title(f'{model_key} Attention Decay')
        ax.set_xlim(0, 50)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_attn_decay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_attn_decay.png")


def fig_layer1_comparison(results):
    """Figure: Detailed Layer 1 comparison."""
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Get layer 1 data
    l1_key = 1  # Integer key

    rope_entropy = results["RoPE"]["attention_analysis"]["entropy_by_layer_head"].get(str(l1_key), [])
    drope_entropy = results["DroPE"]["attention_analysis"]["entropy_by_layer_head"].get(str(l1_key), [])

    rope_bos = results["RoPE"]["attention_analysis"]["bos_attention_by_layer_head"].get(str(l1_key), [])
    drope_bos = results["DroPE"]["attention_analysis"]["bos_attention_by_layer_head"].get(str(l1_key), [])

    rope_local = results["RoPE"]["attention_analysis"]["local_attention_by_layer_head"].get(str(l1_key), [])
    drope_local = results["DroPE"]["attention_analysis"]["local_attention_by_layer_head"].get(str(l1_key), [])

    # 1. Entropy per head
    ax1 = fig.add_subplot(gs[0, 0])
    heads = range(len(rope_entropy))
    ax1.bar([h - 0.2 for h in heads], rope_entropy, 0.4, label='RoPE', color=ROPE_COLOR, alpha=0.7)
    ax1.bar([h + 0.2 for h in heads], drope_entropy, 0.4, label='DroPE', color=DROPE_COLOR, alpha=0.7)
    ax1.set_xlabel('Head')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Layer 1: Entropy per Head')
    ax1.legend()

    # 2. BOS attention per head
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar([h - 0.2 for h in heads], rope_bos, 0.4, label='RoPE', color=ROPE_COLOR, alpha=0.7)
    ax2.bar([h + 0.2 for h in heads], drope_bos, 0.4, label='DroPE', color=DROPE_COLOR, alpha=0.7)
    ax2.set_xlabel('Head')
    ax2.set_ylabel('BOS Attention')
    ax2.set_title('Layer 1: BOS Attention per Head')
    ax2.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Sink threshold')
    ax2.legend()

    # 3. Local attention per head
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar([h - 0.2 for h in heads], rope_local, 0.4, label='RoPE', color=ROPE_COLOR, alpha=0.7)
    ax3.bar([h + 0.2 for h in heads], drope_local, 0.4, label='DroPE', color=DROPE_COLOR, alpha=0.7)
    ax3.set_xlabel('Head')
    ax3.set_ylabel('Local Attention')
    ax3.set_title('Layer 1: Local Attention per Head')
    ax3.legend()

    # 4. Layer 1 heatmaps
    try:
        rope_matrices = np.load(OUTPUT_DIR / 'rope_attention_matrices.npz')
        drope_matrices = np.load(OUTPUT_DIR / 'drope_attention_matrices.npz')

        if 'layer_1' in rope_matrices:
            ax4 = fig.add_subplot(gs[1, 0])
            rope_l1 = rope_matrices['layer_1'].mean(axis=0)[:30, :30]
            im4 = ax4.imshow(rope_l1, cmap='viridis', aspect='auto')
            ax4.set_title('RoPE Layer 1 Attention')
            ax4.set_xlabel('Key')
            ax4.set_ylabel('Query')
            plt.colorbar(im4, ax=ax4, fraction=0.046)

            ax5 = fig.add_subplot(gs[1, 1])
            drope_l1 = drope_matrices['layer_1'].mean(axis=0)[:30, :30]
            im5 = ax5.imshow(drope_l1, cmap='viridis', aspect='auto')
            ax5.set_title('DroPE Layer 1 Attention')
            ax5.set_xlabel('Key')
            ax5.set_ylabel('Query')
            plt.colorbar(im5, ax=ax5, fraction=0.046)

            # Difference
            ax6 = fig.add_subplot(gs[1, 2])
            diff = drope_l1 - rope_l1
            im6 = ax6.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
            ax6.set_title('Difference (DroPE - RoPE)')
            ax6.set_xlabel('Key')
            ax6.set_ylabel('Query')
            plt.colorbar(im6, ax=ax6, fraction=0.046)
    except FileNotFoundError:
        print("Attention matrices not found, skipping heatmaps")

    plt.suptitle('Layer 1 Detailed Comparison', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_layer1_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_layer1_comparison.png")


def fig_summary(results):
    """Figure: Summary comparison of attention patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Entropy by layer
    ax1 = axes[0, 0]
    rope_entropy = results["RoPE"]["attention_analysis"]["summary"]["mean_entropy_by_layer"]
    drope_entropy = results["DroPE"]["attention_analysis"]["summary"]["mean_entropy_by_layer"]
    layers = range(len(rope_entropy))
    ax1.plot(layers, rope_entropy, 'o-', color=ROPE_COLOR, label='RoPE', markersize=3)
    ax1.plot(layers, drope_entropy, 's-', color=DROPE_COLOR, label='DroPE', markersize=3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Entropy')
    ax1.set_title('Attention Entropy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. BOS attention by layer
    ax2 = axes[0, 1]
    rope_bos = results["RoPE"]["attention_analysis"]["summary"]["mean_bos_attn_by_layer"]
    drope_bos = results["DroPE"]["attention_analysis"]["summary"]["mean_bos_attn_by_layer"]
    ax2.plot(layers, rope_bos, 'o-', color=ROPE_COLOR, label='RoPE', markersize=3)
    ax2.plot(layers, drope_bos, 's-', color=DROPE_COLOR, label='DroPE', markersize=3)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Mean BOS Attention')
    ax2.set_title('BOS Attention')
    ax2.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Local attention by layer
    ax3 = axes[1, 0]
    rope_local = results["RoPE"]["attention_analysis"]["summary"]["mean_local_attn_by_layer"]
    drope_local = results["DroPE"]["attention_analysis"]["summary"]["mean_local_attn_by_layer"]
    ax3.plot(layers, rope_local, 'o-', color=ROPE_COLOR, label='RoPE', markersize=3)
    ax3.plot(layers, drope_local, 's-', color=DROPE_COLOR, label='DroPE', markersize=3)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Mean Local Attention')
    ax3.set_title('Local Attention (window=5)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Head type bar chart
    ax4 = axes[1, 1]
    head_types = ["sink", "local", "distributed", "mixed"]
    x = np.arange(len(head_types))
    width = 0.35

    rope_counts = results["RoPE"]["attention_analysis"]["summary"]["head_type_counts"]
    drope_counts = results["DroPE"]["attention_analysis"]["summary"]["head_type_counts"]
    total = 32 * 32

    rope_pcts = [rope_counts[ht] / total * 100 for ht in head_types]
    drope_pcts = [drope_counts[ht] / total * 100 for ht in head_types]

    ax4.bar(x - width/2, rope_pcts, width, label='RoPE', color=ROPE_COLOR)
    ax4.bar(x + width/2, drope_pcts, width, label='DroPE', color=DROPE_COLOR)
    ax4.set_xlabel('Head Type')
    ax4.set_ylabel('Percentage')
    ax4.set_title('Head Type Distribution')
    ax4.set_xticks(x)
    ax4.set_xticklabels(head_types)
    ax4.legend()

    plt.suptitle('Experiment 7: Attention Pattern Analysis Summary', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_attention_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig_attention_summary.png")


def main():
    print("=" * 60)
    print("Creating Attention Analysis Figures")
    print("=" * 60)

    results = load_results()

    # Check if attention analysis exists
    if "attention_analysis" not in results.get("RoPE", {}):
        print("Error: attention_analysis not found in results.")
        print("Run scripts/run_attention_analysis.py first.")
        return

    fig_entropy_by_layer(results)
    fig_head_type_distribution(results)
    fig_decay_profiles(results)
    fig_summary(results)

    # These require the .npz files
    try:
        fig_attention_heatmaps(results)
        fig_layer1_comparison(results)
    except FileNotFoundError as e:
        print(f"Warning: Could not create heatmap figures: {e}")
        print("Run scripts/run_attention_analysis.py to generate attention matrices.")

    print("\nDone! Figures saved to results/phase_metrics/")


if __name__ == "__main__":
    main()
