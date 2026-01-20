#!/usr/bin/env python3
"""
Phase Metrics Analysis: RoPE vs DroPE

Computes Queipo-style metrics:
- BOS residual norm and ratio
- Representation entropy (compression valleys)
- Attention sink rates

Supports interventions:
- BOS-MLP ablation (Queipo-style)
- Q/K massive value disruption (Jin-style)
- Combined

References:
- Jin et al. 2025: Massive Values in Self-Attention Modules
- Queipo-de-Llano et al. 2025: Attention sinks and compression valleys
- Gelberg et al. 2025: DroPE
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings

# Add custom_models to path for DroPE model loading
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
CUSTOM_MODELS_DIR = REPO_ROOT / "custom_models"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pre-import custom_models so transformers can find it during model loading
try:
    import custom_models
    import custom_models.attention
    import custom_models.drope
except ImportError as e:
    print(f"Warning: Could not pre-import custom_models: {e}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class PhaseMetrics:
    """Per-layer phase metrics."""
    bos_norm: List[float]           # ||X^(l)[0]||_2
    bos_ratio: List[float]          # BOS_norm / mean(non-BOS norms)
    entropy: List[float]            # H(X^(l)) representation entropy
    anisotropy: List[float]         # p1 = max singular value fraction
    sink_rate: List[float]          # fraction of heads with sink_score >= tau
    sink_scores: List[List[float]]  # per-head sink scores [layer][head]


@dataclass
class InterventionConfig:
    """Configuration for interventions."""
    bos_mlp_ablation: bool = False
    bos_mlp_ablation_layers: List[int] = None  # layers to ablate
    qk_massive_disruption: bool = False
    qk_disruption_method: str = "zero"  # "zero" or "mean"


# ============================================================================
# PROMPTS FOR EVALUATION
# ============================================================================

EVAL_PROMPTS = [
    # Wiki-style paragraphs
    "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols.",

    "In physics, the theory of relativity encompasses two interrelated physics theories by Albert Einstein: special relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity. General relativity explains the law of gravitation and its relation to the forces of nature.",

    "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest.",

    # Technical content
    "To implement a transformer model, you need to understand the self-attention mechanism. The attention function computes a weighted sum of values, where the weights are determined by the compatibility of queries and keys. This can be expressed as Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V.",

    "Machine learning models can be broadly categorized into supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the model learns from labeled examples. In unsupervised learning, the model discovers patterns in unlabeled data.",

    # Conversational
    "User: What is the capital of France?\nAssistant: The capital of France is Paris. Paris is not only the capital but also the largest city in France, located in the north-central part of the country along the Seine River.",

    "User: How do I make pasta?\nAssistant: To make pasta, you'll need flour, eggs, and a pinch of salt. Mix the ingredients to form a dough, knead it until smooth, roll it out thin, and cut it into your desired shape.",

    # Book excerpts
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness.",

    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.",

    # Math/reasoning
    "Problem: If a train travels at 60 miles per hour for 2.5 hours, how far does it travel? Solution: Distance equals speed multiplied by time. So the distance is 60 * 2.5 = 150 miles.",
]


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def compute_bos_metrics(hidden_states: torch.Tensor) -> Tuple[float, float]:
    """
    Compute BOS norm and BOS ratio from hidden states.

    Args:
        hidden_states: [batch, seq, hidden_dim] tensor

    Returns:
        bos_norm: L2 norm of BOS token representation
        bos_ratio: BOS norm / mean non-BOS norm
    """
    # Take first sample in batch
    X = hidden_states[0].float()  # [seq, hidden_dim]

    # BOS is position 0
    bos_norm = X[0].norm(p=2).item()

    # Non-BOS mean norm
    if X.shape[0] > 1:
        non_bos_norms = X[1:].norm(p=2, dim=-1)
        non_bos_mean = non_bos_norms.mean().item()
        bos_ratio = bos_norm / (non_bos_mean + 1e-8)
    else:
        bos_ratio = 1.0

    return bos_norm, bos_ratio


def compute_entropy_metrics(hidden_states: torch.Tensor, eps: float = 1e-10) -> Tuple[float, float]:
    """
    Compute representation entropy and anisotropy from hidden states.

    Uses Queipo's matrix-based entropy from singular values:
    - Compute SVD of X (or eigenvalues of X @ X.T for efficiency)
    - p_j = sigma_j^2 / ||X||_F^2
    - H = -sum(p_j * log(p_j))
    - anisotropy p1 = max(p_j)

    Args:
        hidden_states: [batch, seq, hidden_dim] tensor
        eps: small constant for numerical stability

    Returns:
        entropy: representation entropy H(X)
        anisotropy: p1 (max singular value fraction)
    """
    X = hidden_states[0].float()  # [seq, hidden_dim]
    T, d = X.shape

    # Compute covariance C = X @ X.T (shape [T, T]) - more efficient when T < d
    C = X @ X.T

    # Frobenius norm squared = trace(C)
    frobenius_sq = torch.trace(C).item()

    if frobenius_sq < eps:
        return 0.0, 1.0

    # Eigenvalues of C give sigma^2
    try:
        eigenvalues = torch.linalg.eigvalsh(C)  # sorted ascending
        eigenvalues = eigenvalues.flip(0)  # descending order
        eigenvalues = eigenvalues.clamp(min=0)  # numerical stability
    except Exception:
        return 0.0, 1.0

    # Compute probability distribution
    p = eigenvalues / frobenius_sq
    p = p.clamp(min=eps)  # avoid log(0)

    # Entropy
    entropy = -torch.sum(p * torch.log(p)).item()

    # Anisotropy (fraction of variance in top singular value)
    anisotropy = p[0].item()

    return entropy, anisotropy


def compute_sink_metrics(
    attentions: torch.Tensor,
    tau: float = 0.3
) -> Tuple[float, List[float]]:
    """
    Compute attention sink metrics.

    sink_score_BOS(l,h) = mean_t alpha[t, 0] (average attention to BOS)
    sink_rate(l) = fraction of heads with sink_score >= tau

    Args:
        attentions: [batch, heads, seq, seq] attention weights
        tau: threshold for sink classification

    Returns:
        sink_rate: fraction of heads that are sinks
        sink_scores: per-head sink scores
    """
    # Take first sample
    attn = attentions[0].float()  # [heads, seq, seq]
    n_heads = attn.shape[0]

    # Sink score = average attention to BOS (column 0)
    # For each head, average over all query positions
    sink_scores = attn[:, :, 0].mean(dim=1)  # [heads]
    sink_scores_list = sink_scores.tolist()

    # Sink rate = fraction of heads with sink_score >= tau
    sink_rate = (sink_scores >= tau).float().mean().item()

    return sink_rate, sink_scores_list


# ============================================================================
# INTERVENTION HOOKS
# ============================================================================

class BOSMLPAblationHook:
    """Hook to ablate MLP output for BOS token only."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.handle = None

    def hook_fn(self, module, input, output):
        """Set MLP output to zero for BOS position only."""
        # output shape: [batch, seq, hidden_dim]
        modified = output.clone()
        modified[:, 0, :] = 0  # Zero out BOS position
        return modified

    def register(self, model):
        """Register the hook on the MLP module."""
        mlp = model.model.layers[self.layer_idx].mlp
        self.handle = mlp.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class QKMassiveDisruptionHook:
    """Hook to disrupt massive values in Q/K projections."""

    def __init__(self, layer_idx: int, num_heads: int = 32, head_dim: int = 128,
                 method: str = "zero", threshold: float = 5.0):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.method = method
        self.threshold = threshold
        self.q_handle = None
        self.k_handle = None

    def make_hook(self):
        """Create hook function for Q or K projection."""
        def hook_fn(module, input, output):
            # output shape: [batch, seq, hidden_dim]
            bsz, seq_len, hidden = output.shape

            # Reshape to [batch, seq, heads, head_dim]
            states = output.view(bsz, seq_len, self.num_heads, self.head_dim)

            # Compute L2 norm across sequence for each [head, dim]
            norms = states.float().norm(dim=1)  # [batch, heads, head_dim]

            # Find massive dimensions per head
            modified = states.clone()
            for head_idx in range(self.num_heads):
                head_norms = norms[0, head_idx, :]  # [head_dim]
                mean_norm = head_norms.mean()
                threshold = self.threshold * mean_norm

                # Find top-1 massive dimension for this head
                max_idx = head_norms.argmax()
                if head_norms[max_idx] > threshold:
                    if self.method == "zero":
                        modified[:, :, head_idx, max_idx] = 0
                    elif self.method == "mean":
                        dim_mean = states[:, :, head_idx, max_idx].mean()
                        modified[:, :, head_idx, max_idx] = dim_mean

            return modified.view(bsz, seq_len, hidden)
        return hook_fn

    def register(self, model):
        """Register hooks on Q and K projections."""
        layer = model.model.layers[self.layer_idx].self_attn
        self.q_handle = layer.q_proj.register_forward_hook(self.make_hook())
        self.k_handle = layer.k_proj.register_forward_hook(self.make_hook())
        return self

    def remove(self):
        """Remove hooks."""
        if self.q_handle is not None:
            self.q_handle.remove()
            self.q_handle = None
        if self.k_handle is not None:
            self.k_handle.remove()
            self.k_handle = None


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def load_model(model_name: str, device: str = "cuda", use_eager_attention: bool = True):
    """Load model with 4-bit quantization.

    Args:
        model_name: HuggingFace model name
        device: Device to load on
        use_eager_attention: If True, use eager attention (required for output_attentions).
                            DroPE models may need this disabled.
    """
    print(f"Loading {model_name}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if use_eager_attention:
        load_kwargs["attn_implementation"] = "eager"  # Required for output_attentions=True

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    return model, tokenizer


def analyze_prompt(
    model,
    tokenizer,
    prompt: str,
    seq_len: int = 512,
    intervention_hooks: List = None,
    tau: float = 0.3,
    capture_attention: bool = True,
    use_padding: bool = True,
) -> PhaseMetrics:
    """
    Analyze a single prompt and compute phase metrics.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text
        seq_len: Sequence length to use
        intervention_hooks: List of intervention hooks to apply
        tau: Threshold for sink classification
        capture_attention: If True, capture attention weights for sink metrics.
                          Set to False for models that don't support output_attentions.
        use_padding: If True, pad to max_length. Set to False for DroPE models.

    Returns:
        PhaseMetrics object with per-layer metrics
    """
    # Tokenize and truncate/pad to seq_len
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": seq_len,
    }
    if use_padding:
        tokenizer_kwargs["padding"] = "max_length"

    inputs = tokenizer(prompt, **tokenizer_kwargs).to(model.device)

    # Apply intervention hooks if any
    if intervention_hooks:
        for hook in intervention_hooks:
            hook.register(model)

    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=capture_attention,
            )
    finally:
        # Always remove hooks
        if intervention_hooks:
            for hook in intervention_hooks:
                hook.remove()

    # hidden_states: tuple of (n_layers + 1) tensors
    # hidden_states[0] = embeddings, hidden_states[i] = after layer i-1
    hidden_states = outputs.hidden_states
    attentions = outputs.attentions if capture_attention else None

    n_layers = len(hidden_states) - 1  # exclude embedding layer

    # Initialize metrics
    bos_norms = []
    bos_ratios = []
    entropies = []
    anisotropies = []
    sink_rates = []
    sink_scores_all = []

    # Compute metrics for each layer
    for layer_idx in range(n_layers):
        # Use hidden_states[layer_idx + 1] = output after layer layer_idx
        hs = hidden_states[layer_idx + 1]

        # BOS metrics
        bos_norm, bos_ratio = compute_bos_metrics(hs)
        bos_norms.append(bos_norm)
        bos_ratios.append(bos_ratio)

        # Entropy metrics
        entropy, anisotropy = compute_entropy_metrics(hs)
        entropies.append(entropy)
        anisotropies.append(anisotropy)

        # Sink metrics (only if attention is available)
        if attentions is not None:
            attn = attentions[layer_idx]
            sink_rate, sink_scores = compute_sink_metrics(attn, tau=tau)
            sink_rates.append(sink_rate)
            sink_scores_all.append(sink_scores)
        else:
            sink_rates.append(0.0)
            sink_scores_all.append([])

    return PhaseMetrics(
        bos_norm=bos_norms,
        bos_ratio=bos_ratios,
        entropy=entropies,
        anisotropy=anisotropies,
        sink_rate=sink_rates,
        sink_scores=sink_scores_all,
    )


def find_bos_spike_layer(metrics_list: List[PhaseMetrics]) -> int:
    """
    Find the layer with the largest BOS ratio spike.

    Args:
        metrics_list: List of PhaseMetrics from multiple prompts

    Returns:
        Layer index with highest average BOS ratio
    """
    n_layers = len(metrics_list[0].bos_ratio)
    avg_ratios = []

    for layer_idx in range(n_layers):
        layer_ratios = [m.bos_ratio[layer_idx] for m in metrics_list]
        avg_ratios.append(np.mean(layer_ratios))

    return int(np.argmax(avg_ratios))


def aggregate_metrics(metrics_list: List[PhaseMetrics]) -> Dict:
    """
    Aggregate metrics across prompts (mean and std).

    Args:
        metrics_list: List of PhaseMetrics from multiple prompts

    Returns:
        Dict with mean and std for each metric
    """
    n_layers = len(metrics_list[0].bos_norm)
    n_prompts = len(metrics_list)

    result = {
        "bos_norm": {"mean": [], "std": []},
        "bos_ratio": {"mean": [], "std": []},
        "entropy": {"mean": [], "std": []},
        "anisotropy": {"mean": [], "std": []},
        "sink_rate": {"mean": [], "std": []},
    }

    for layer_idx in range(n_layers):
        for metric_name in result.keys():
            values = [getattr(m, metric_name)[layer_idx] for m in metrics_list]
            result[metric_name]["mean"].append(float(np.mean(values)))
            result[metric_name]["std"].append(float(np.std(values)))

    return result


# ============================================================================
# FUNCTIONAL EVALUATIONS
# ============================================================================

def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    seq_len: int = 512,
    intervention_hooks: List = None,
) -> float:
    """Compute average perplexity on texts."""
    total_loss = 0
    total_tokens = 0

    for text in tqdm(texts, desc="Computing PPL"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
        ).to(model.device)

        if intervention_hooks:
            for hook in intervention_hooks:
                hook.register(model)

        try:
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
        finally:
            if intervention_hooks:
                for hook in intervention_hooks:
                    hook.remove()

        n_tokens = inputs["input_ids"].shape[1]
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def run_passkey_retrieval(
    model,
    tokenizer,
    n_samples: int = 20,
    context_len: int = 500,
    intervention_hooks: List = None,
) -> float:
    """Run passkey retrieval test."""
    import random
    random.seed(42)

    correct = 0

    for _ in tqdm(range(n_samples), desc="Passkey"):
        # Generate random passkey
        passkey = str(random.randint(10000, 99999))

        # Generate filler text
        filler = "The quick brown fox jumps over the lazy dog. " * 50

        # Insert passkey at random position
        insert_pos = random.randint(100, 300)
        text_before = filler[:insert_pos]
        text_after = filler[insert_pos:]

        prompt = f"{text_before} The secret passkey is {passkey}. Remember it. {text_after}\n\nWhat is the secret passkey mentioned above? The passkey is"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=context_len,
        ).to(model.device)

        if intervention_hooks:
            for hook in intervention_hooks:
                hook.register(model)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            if intervention_hooks:
                for hook in intervention_hooks:
                    hook.remove()

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        if passkey in generated:
            correct += 1

    return correct / n_samples


def run_imdb_sentiment(
    model,
    tokenizer,
    n_samples: int = 50,
    intervention_hooks: List = None,
) -> float:
    """Run IMDB sentiment classification."""
    # Simple sentiment examples
    samples = [
        ("This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.", "positive"),
        ("Terrible waste of time. The acting was wooden and the story made no sense.", "negative"),
        ("A masterpiece of cinema. Every scene was beautifully crafted.", "positive"),
        ("I couldn't even finish watching this garbage. Worst movie I've seen.", "negative"),
        ("An entertaining and thoughtful film that I would highly recommend.", "positive"),
        ("Boring, predictable, and poorly executed. Don't bother.", "negative"),
        ("The performances were outstanding and the direction was flawless.", "positive"),
        ("A complete disaster from start to finish. Save your money.", "negative"),
        ("Brilliant storytelling with memorable characters.", "positive"),
        ("Painfully slow and utterly forgettable.", "negative"),
    ] * 5  # Repeat to get 50 samples

    samples = samples[:n_samples]
    correct = 0

    for review, label in tqdm(samples, desc="IMDB"):
        prompt = f"Review: {review}\n\nSentiment (positive or negative):"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(model.device)

        if intervention_hooks:
            for hook in intervention_hooks:
                hook.register(model)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            if intervention_hooks:
                for hook in intervention_hooks:
                    hook.remove()

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()

        predicted = "positive" if "positive" in generated else "negative" if "negative" in generated else "unknown"
        if predicted == label:
            correct += 1

    return correct / n_samples


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase Metrics Analysis: RoPE vs DroPE")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--tau", type=float, default=0.3, help="Sink threshold")
    parser.add_argument("--output_dir", type=str, default="results/phase_metrics", help="Output directory")
    parser.add_argument("--skip_interventions", action="store_true", help="Skip intervention experiments")
    parser.add_argument("--skip_functional", action="store_true", help="Skip functional evaluations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model configs
    ROPE_MODEL = "meta-llama/Llama-2-7b-hf"
    DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"

    results = {}

    for model_name, model_key in [(ROPE_MODEL, "RoPE"), (DROPE_MODEL, "DroPE")]:
        print(f"\n{'='*60}")
        print(f"Analyzing {model_key}")
        print(f"{'='*60}")

        # DroPE model doesn't support eager attention well, so skip attention capture
        # DroPE also has issues with padding, so skip that too
        is_drope = (model_key == "DroPE")
        use_eager = not is_drope
        capture_attn = not is_drope
        use_padding = not is_drope

        model, tokenizer = load_model(model_name, use_eager_attention=use_eager)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // n_heads

        results[model_key] = {"baseline": {}, "interventions": {}, "functional": {}}

        # ==========================================
        # BASELINE METRICS
        # ==========================================
        print("\n--- Baseline Metrics ---")
        baseline_metrics = []

        for prompt in tqdm(EVAL_PROMPTS, desc="Baseline"):
            metrics = analyze_prompt(
                model, tokenizer, prompt,
                seq_len=args.seq_len,
                tau=args.tau,
                capture_attention=capture_attn,
                use_padding=use_padding,
            )
            baseline_metrics.append(metrics)

        results[model_key]["baseline"] = aggregate_metrics(baseline_metrics)

        # Find BOS spike layer for interventions
        bos_spike_layer = find_bos_spike_layer(baseline_metrics)
        print(f"BOS spike layer: {bos_spike_layer}")
        results[model_key]["bos_spike_layer"] = bos_spike_layer

        if not args.skip_interventions:
            # ==========================================
            # INTERVENTION A: BOS-MLP ABLATION
            # ==========================================
            print(f"\n--- Intervention A: BOS-MLP Ablation (layer {bos_spike_layer}) ---")
            intervention_a_metrics = []

            for prompt in tqdm(EVAL_PROMPTS, desc="BOS-MLP Ablation"):
                hook = BOSMLPAblationHook(bos_spike_layer)
                metrics = analyze_prompt(
                    model, tokenizer, prompt,
                    seq_len=args.seq_len,
                    tau=args.tau,
                    intervention_hooks=[hook],
                    capture_attention=capture_attn,
                    use_padding=use_padding,
                )
                intervention_a_metrics.append(metrics)

            results[model_key]["interventions"]["bos_mlp_ablation"] = aggregate_metrics(intervention_a_metrics)

            # ==========================================
            # INTERVENTION B: Q/K MASSIVE DISRUPTION
            # ==========================================
            print("\n--- Intervention B: Q/K Massive Value Disruption ---")
            intervention_b_metrics = []

            for prompt in tqdm(EVAL_PROMPTS, desc="Q/K Disruption"):
                # Apply to all layers
                hooks = [
                    QKMassiveDisruptionHook(layer_idx, n_heads, head_dim, method="mean")
                    for layer_idx in range(n_layers)
                ]
                metrics = analyze_prompt(
                    model, tokenizer, prompt,
                    seq_len=args.seq_len,
                    tau=args.tau,
                    intervention_hooks=hooks,
                    capture_attention=capture_attn,
                    use_padding=use_padding,
                )
                intervention_b_metrics.append(metrics)

            results[model_key]["interventions"]["qk_disruption"] = aggregate_metrics(intervention_b_metrics)

            # ==========================================
            # INTERVENTION C: COMBINED
            # ==========================================
            print(f"\n--- Intervention C: Combined (BOS-MLP + Q/K) ---")
            intervention_c_metrics = []

            for prompt in tqdm(EVAL_PROMPTS, desc="Combined"):
                hooks = [BOSMLPAblationHook(bos_spike_layer)]
                hooks += [
                    QKMassiveDisruptionHook(layer_idx, n_heads, head_dim, method="mean")
                    for layer_idx in range(n_layers)
                ]
                metrics = analyze_prompt(
                    model, tokenizer, prompt,
                    seq_len=args.seq_len,
                    tau=args.tau,
                    intervention_hooks=hooks,
                    capture_attention=capture_attn,
                    use_padding=use_padding,
                )
                intervention_c_metrics.append(metrics)

            results[model_key]["interventions"]["combined"] = aggregate_metrics(intervention_c_metrics)

        if not args.skip_functional:
            # ==========================================
            # FUNCTIONAL EVALUATIONS
            # ==========================================
            print("\n--- Functional Evaluations ---")

            # Baseline
            print("Baseline functional tests...")
            ppl_baseline = compute_perplexity(model, tokenizer, EVAL_PROMPTS[:5], seq_len=args.seq_len)
            passkey_baseline = run_passkey_retrieval(model, tokenizer, n_samples=20)
            imdb_baseline = run_imdb_sentiment(model, tokenizer, n_samples=50)

            results[model_key]["functional"]["baseline"] = {
                "perplexity": ppl_baseline,
                "passkey": passkey_baseline,
                "imdb": imdb_baseline,
            }
            print(f"  Baseline - PPL: {ppl_baseline:.2f}, Passkey: {passkey_baseline:.1%}, IMDB: {imdb_baseline:.1%}")

            if not args.skip_interventions:
                # BOS-MLP ablation
                print("BOS-MLP ablation functional tests...")
                hook_a = BOSMLPAblationHook(bos_spike_layer)
                ppl_a = compute_perplexity(model, tokenizer, EVAL_PROMPTS[:5], seq_len=args.seq_len,
                                          intervention_hooks=[hook_a])
                passkey_a = run_passkey_retrieval(model, tokenizer, n_samples=20,
                                                  intervention_hooks=[hook_a])
                imdb_a = run_imdb_sentiment(model, tokenizer, n_samples=50,
                                           intervention_hooks=[hook_a])

                results[model_key]["functional"]["bos_mlp_ablation"] = {
                    "perplexity": ppl_a,
                    "passkey": passkey_a,
                    "imdb": imdb_a,
                }
                print(f"  BOS-MLP - PPL: {ppl_a:.2f}, Passkey: {passkey_a:.1%}, IMDB: {imdb_a:.1%}")

                # Q/K disruption
                print("Q/K disruption functional tests...")
                hooks_b = [
                    QKMassiveDisruptionHook(layer_idx, n_heads, head_dim, method="mean")
                    for layer_idx in range(n_layers)
                ]
                ppl_b = compute_perplexity(model, tokenizer, EVAL_PROMPTS[:5], seq_len=args.seq_len,
                                          intervention_hooks=hooks_b)
                passkey_b = run_passkey_retrieval(model, tokenizer, n_samples=20,
                                                  intervention_hooks=hooks_b)
                imdb_b = run_imdb_sentiment(model, tokenizer, n_samples=50,
                                           intervention_hooks=hooks_b)

                results[model_key]["functional"]["qk_disruption"] = {
                    "perplexity": ppl_b,
                    "passkey": passkey_b,
                    "imdb": imdb_b,
                }
                print(f"  Q/K Dis - PPL: {ppl_b:.2f}, Passkey: {passkey_b:.1%}, IMDB: {imdb_b:.1%}")

                # Combined
                print("Combined functional tests...")
                hooks_c = [BOSMLPAblationHook(bos_spike_layer)] + hooks_b
                ppl_c = compute_perplexity(model, tokenizer, EVAL_PROMPTS[:5], seq_len=args.seq_len,
                                          intervention_hooks=hooks_c)
                passkey_c = run_passkey_retrieval(model, tokenizer, n_samples=20,
                                                  intervention_hooks=hooks_c)
                imdb_c = run_imdb_sentiment(model, tokenizer, n_samples=50,
                                           intervention_hooks=hooks_c)

                results[model_key]["functional"]["combined"] = {
                    "perplexity": ppl_c,
                    "passkey": passkey_c,
                    "imdb": imdb_c,
                }
                print(f"  Combined - PPL: {ppl_c:.2f}, Passkey: {passkey_c:.1%}, IMDB: {imdb_c:.1%}")

        # Clean up model
        del model
        torch.cuda.empty_cache()

    # Save results
    output_file = output_dir / "rope_vs_drope_phase_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for model_key in ["RoPE", "DroPE"]:
        print(f"\n{model_key}:")
        print(f"  BOS spike layer: {results[model_key]['bos_spike_layer']}")

        baseline = results[model_key]["baseline"]
        max_bos_ratio = max(baseline["bos_ratio"]["mean"])
        max_bos_layer = baseline["bos_ratio"]["mean"].index(max_bos_ratio)
        min_entropy = min(baseline["entropy"]["mean"])
        min_entropy_layer = baseline["entropy"]["mean"].index(min_entropy)
        max_sink_rate = max(baseline["sink_rate"]["mean"])
        max_sink_layer = baseline["sink_rate"]["mean"].index(max_sink_rate)

        print(f"  Max BOS ratio: {max_bos_ratio:.2f} at layer {max_bos_layer}")
        print(f"  Min entropy: {min_entropy:.2f} at layer {min_entropy_layer}")
        print(f"  Max sink rate: {max_sink_rate:.2%} at layer {max_sink_layer}")

        if "functional" in results[model_key] and results[model_key]["functional"]:
            func = results[model_key]["functional"]
            if "baseline" in func:
                print(f"  Baseline - PPL: {func['baseline']['perplexity']:.2f}, "
                      f"Passkey: {func['baseline']['passkey']:.1%}, "
                      f"IMDB: {func['baseline']['imdb']:.1%}")


if __name__ == "__main__":
    main()
