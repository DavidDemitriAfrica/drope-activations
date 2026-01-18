"""
Analysis functions for identifying and measuring massive values in Q/K tensors.

Based on "Massive Values in Self-Attention Modules are the Key to Contextual
Knowledge Understanding" (Jin, Sun et al., ICML 2025).
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from scipy import stats


@dataclass
class MassiveValueAnalysis:
    """Results from massive value analysis on a single tensor."""
    norm_matrix: torch.Tensor  # [num_heads, head_dim] - L2 norms
    massive_mask: torch.Tensor  # [num_heads, head_dim] - bool mask of massive values
    threshold: float
    num_massive: int
    concentration_score: float
    top_dimensions: List[Tuple[int, int, float]]  # (head, dim, norm) sorted by norm


def compute_massive_value_matrix(
    tensor: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute M[h,d] = ||tensor[:,h,d]||_2 along sequence dimension.

    Args:
        tensor: Shape [batch, seq_len, num_heads, head_dim]
        normalize: Whether to normalize by sequence length

    Returns:
        M matrix of shape [num_heads, head_dim] (averaged over batch)
    """
    # Compute L2 norm along sequence dimension
    # tensor: [batch, seq_len, num_heads, head_dim]
    norms = torch.norm(tensor, p=2, dim=1)  # [batch, num_heads, head_dim]

    if normalize:
        seq_len = tensor.shape[1]
        norms = norms / np.sqrt(seq_len)

    # Average over batch
    M = norms.mean(dim=0)  # [num_heads, head_dim]

    return M


def identify_massive_values(
    M: torch.Tensor,
    lambda_threshold: float = 5.0,
) -> Tuple[torch.Tensor, int]:
    """
    Identify massive values using threshold from Definition 1 in the paper.

    A value is "massive" if M[h,d] > lambda * mean(M)

    Args:
        M: Norm matrix of shape [num_heads, head_dim]
        lambda_threshold: Multiplier for mean to determine threshold

    Returns:
        Tuple of (boolean mask, count of massive values)
    """
    mean_val = M.mean()
    threshold = lambda_threshold * mean_val

    massive_mask = M > threshold
    num_massive = massive_mask.sum().item()

    return massive_mask, int(num_massive)


def measure_concentration(
    M: torch.Tensor,
    massive_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute concentration metrics for the massive value matrix.

    Args:
        M: Norm matrix of shape [num_heads, head_dim]
        massive_mask: Optional pre-computed massive value mask

    Returns:
        Dictionary with concentration metrics:
        - gini: Gini coefficient (higher = more concentrated)
        - top_k_ratio: Ratio of top-k values to total
        - entropy: Entropy of normalized distribution (lower = more concentrated)
        - max_to_mean: Ratio of max to mean value
    """
    M_flat = M.flatten().cpu().numpy()

    # Gini coefficient
    sorted_vals = np.sort(M_flat)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_vals))) / (n * np.sum(sorted_vals)) - (n + 1) / n

    # Top-k ratio (top 1% of dimensions)
    k = max(1, int(0.01 * n))
    top_k_sum = np.sum(sorted_vals[-k:])
    top_k_ratio = top_k_sum / np.sum(sorted_vals)

    # Entropy of normalized distribution
    M_norm = M_flat / M_flat.sum()
    M_norm = M_norm[M_norm > 0]  # Avoid log(0)
    entropy = -np.sum(M_norm * np.log(M_norm))
    max_entropy = np.log(n)  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy

    # Max to mean ratio
    max_to_mean = M_flat.max() / M_flat.mean()

    return {
        "gini": float(gini),
        "top_k_ratio": float(top_k_ratio),
        "entropy": float(normalized_entropy),
        "max_to_mean": float(max_to_mean),
    }


def analyze_massive_values(
    tensor: torch.Tensor,
    lambda_threshold: float = 5.0,
    top_n: int = 20,
) -> MassiveValueAnalysis:
    """
    Complete analysis of massive values in a Q or K tensor.

    Args:
        tensor: Shape [batch, seq_len, num_heads, head_dim]
        lambda_threshold: Threshold for identifying massive values
        top_n: Number of top dimensions to return

    Returns:
        MassiveValueAnalysis with all metrics
    """
    M = compute_massive_value_matrix(tensor)
    massive_mask, num_massive = identify_massive_values(M, lambda_threshold)
    concentration = measure_concentration(M, massive_mask)

    # Find top dimensions
    M_flat = M.flatten()
    top_indices = torch.argsort(M_flat, descending=True)[:top_n]

    num_heads, head_dim = M.shape
    top_dimensions = []
    for idx in top_indices:
        head = idx.item() // head_dim
        dim = idx.item() % head_dim
        norm = M[head, dim].item()
        top_dimensions.append((head, dim, norm))

    return MassiveValueAnalysis(
        norm_matrix=M,
        massive_mask=massive_mask,
        threshold=lambda_threshold * M.mean().item(),
        num_massive=num_massive,
        concentration_score=concentration["gini"],
        top_dimensions=top_dimensions,
    )


def compare_massive_value_positions(
    M1: torch.Tensor,
    M2: torch.Tensor,
    lambda_threshold: float = 5.0,
) -> Dict[str, float]:
    """
    Compare massive value positions between two models (e.g., RoPE vs DroPE).

    Args:
        M1, M2: Norm matrices of same shape [num_heads, head_dim]
        lambda_threshold: Threshold for identifying massive values

    Returns:
        Dictionary with comparison metrics:
        - jaccard: Jaccard similarity of massive value positions
        - cosine: Cosine similarity of full M matrices
        - rank_correlation: Spearman correlation of value rankings
    """
    mask1, _ = identify_massive_values(M1, lambda_threshold)
    mask2, _ = identify_massive_values(M2, lambda_threshold)

    # Jaccard similarity
    intersection = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()
    jaccard = intersection / union if union > 0 else 0.0

    # Cosine similarity
    M1_flat = M1.flatten()
    M2_flat = M2.flatten()
    cosine = torch.nn.functional.cosine_similarity(
        M1_flat.unsqueeze(0), M2_flat.unsqueeze(0)
    ).item()

    # Spearman rank correlation
    rank_corr, _ = stats.spearmanr(M1_flat.cpu().numpy(), M2_flat.cpu().numpy())

    return {
        "jaccard": float(jaccard),
        "cosine": float(cosine),
        "rank_correlation": float(rank_corr),
    }


def analyze_layer_progression(
    qkv_by_layer: Dict[int, "QKVTensors"],
    tensor_type: str = "query",
    lambda_threshold: float = 5.0,
) -> List[Dict]:
    """
    Analyze how massive values evolve across layers.

    Args:
        qkv_by_layer: Dictionary mapping layer index to QKVTensors
        tensor_type: "query", "key", or "value"
        lambda_threshold: Threshold for massive value identification

    Returns:
        List of analysis results per layer
    """
    results = []

    for layer_idx in sorted(qkv_by_layer.keys()):
        qkv = qkv_by_layer[layer_idx]
        tensor = getattr(qkv, tensor_type)

        analysis = analyze_massive_values(tensor, lambda_threshold)

        results.append({
            "layer": layer_idx,
            "num_massive": analysis.num_massive,
            "concentration": analysis.concentration_score,
            "threshold": analysis.threshold,
            "top_dimensions": analysis.top_dimensions[:5],
        })

    return results
