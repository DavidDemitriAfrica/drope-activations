"""
Visualization functions for massive value analysis.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def plot_massive_value_heatmap(
    M: torch.Tensor,
    massive_mask: Optional[torch.Tensor] = None,
    title: str = "Massive Value Distribution",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "viridis",
    show_threshold: bool = True,
    lambda_threshold: float = 5.0,
) -> plt.Figure:
    """
    Plot heatmap of M[h,d] norm matrix with massive values highlighted.

    Args:
        M: Norm matrix of shape [num_heads, head_dim]
        massive_mask: Boolean mask highlighting massive values
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        cmap: Colormap name
        show_threshold: Whether to show threshold line in colorbar
        lambda_threshold: Threshold used for massive value identification

    Returns:
        Matplotlib figure
    """
    M_np = M.cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(M_np, aspect="auto", cmap=cmap)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("L2 Norm", rotation=270, labelpad=15)

    if show_threshold:
        threshold = lambda_threshold * M_np.mean()
        cbar.ax.axhline(y=threshold, color="red", linewidth=2)
        cbar.ax.text(1.5, threshold, f"λ={lambda_threshold}", color="red", va="center")

    # Highlight massive values if mask provided
    if massive_mask is not None:
        mask_np = massive_mask.cpu().numpy()
        # Overlay markers on massive value positions
        massive_y, massive_x = np.where(mask_np)
        ax.scatter(massive_x, massive_y, c="red", marker="x", s=20, alpha=0.7, label="Massive")
        ax.legend(loc="upper right")

    ax.set_xlabel("Head Dimension")
    ax.set_ylabel("Attention Head")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_concentration_by_layer(
    layer_results: List[Dict],
    metric: str = "concentration",
    title: str = "Concentration Across Layers",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot how a metric evolves across transformer layers.

    Args:
        layer_results: List of dicts with 'layer' and metric keys
        metric: Which metric to plot
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    layers = [r["layer"] for r in layer_results]
    values = [r[metric] for r in layer_results]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(layers, values, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_qkv_comparison(
    q_analysis: "MassiveValueAnalysis",
    k_analysis: "MassiveValueAnalysis",
    v_analysis: "MassiveValueAnalysis",
    layer_idx: int,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plot side-by-side comparison of Q, K, V massive value patterns.

    Reproduces Figure 2 style from the Massive Values paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    analyses = [
        (q_analysis, "Query (Q)", axes[0]),
        (k_analysis, "Key (K)", axes[1]),
        (v_analysis, "Value (V)", axes[2]),
    ]

    vmin = min(a.norm_matrix.min().item() for a, _, _ in analyses)
    vmax = max(a.norm_matrix.max().item() for a, _, _ in analyses)

    for analysis, title, ax in analyses:
        M_np = analysis.norm_matrix.cpu().numpy()
        im = ax.imshow(M_np, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

        # Highlight massive values
        mask_np = analysis.massive_mask.cpu().numpy()
        massive_y, massive_x = np.where(mask_np)
        ax.scatter(massive_x, massive_y, c="red", marker="x", s=15, alpha=0.7)

        ax.set_xlabel("Dimension")
        ax.set_ylabel("Head")
        ax.set_title(f"{title}\n({analysis.num_massive} massive values)")

    # Shared colorbar
    fig.colorbar(im, ax=axes, label="L2 Norm", shrink=0.8)

    fig.suptitle(f"Layer {layer_idx}: Q/K/V Massive Value Patterns", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(
    rope_results: List[Dict],
    drope_results: List[Dict],
    metric: str = "num_massive",
    labels: Tuple[str, str] = ("RoPE", "DroPE"),
    title: str = "Massive Values: RoPE vs DroPE",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Compare massive value metrics between RoPE and DroPE models.
    """
    layers_rope = [r["layer"] for r in rope_results]
    values_rope = [r[metric] for r in rope_results]

    layers_drope = [r["layer"] for r in drope_results]
    values_drope = [r[metric] for r in drope_results]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(layers_rope, values_rope, "o-", linewidth=2, markersize=8, label=labels[0])
    ax.plot(layers_drope, values_drope, "s--", linewidth=2, markersize=8, label=labels[1])

    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_disruption_results(
    results: Dict[str, Dict[str, float]],
    baseline_key: str = "baseline",
    title: str = "Effect of Massive Value Disruption",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot results from disruption experiments.

    Args:
        results: Dict mapping condition name to metric dict
                 e.g., {"baseline": {"accuracy": 0.9}, "disrupted_massive": {"accuracy": 0.3}}
        baseline_key: Key for baseline results
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    conditions = list(results.keys())
    metrics = list(results[conditions[0]].keys())

    x = np.arange(len(metrics))
    width = 0.8 / len(conditions)

    fig, ax = plt.subplots(figsize=figsize)

    for i, condition in enumerate(conditions):
        values = [results[condition][m] for m in metrics]
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=condition)

        # Highlight baseline
        if condition == baseline_key:
            for bar in bars:
                bar.set_edgecolor("black")
                bar.set_linewidth(2)

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_recalibration_evolution(
    checkpoints: List[str],
    metrics_by_checkpoint: Dict[str, List[float]],
    title: str = "Massive Values During DroPE Recalibration",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot how massive value metrics evolve during DroPE recalibration.

    Args:
        checkpoints: List of checkpoint names (x-axis)
        metrics_by_checkpoint: Dict mapping metric name to list of values
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(checkpoints))

    for metric_name, values in metrics_by_checkpoint.items():
        ax.plot(x, values, "o-", linewidth=2, markersize=8, label=metric_name)

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
