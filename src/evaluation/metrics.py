"""
Common evaluation metrics for language models.
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_length: int = 2048,
    stride: int = 512,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Compute perplexity on a list of texts using sliding window.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of text strings
        max_length: Maximum sequence length
        stride: Sliding window stride
        batch_size: Batch size for processing

    Returns:
        Dict with perplexity and average negative log likelihood
    """
    device = next(model.parameters()).device
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()

            # Only compute loss on the new tokens (not the overlapping context)
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood.item())

            prev_end_loc = end_loc
            if end_loc >= seq_len:
                break

        total_nll += sum(nlls)
        total_tokens += seq_len

    avg_nll = total_nll / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
    }


def compute_ngram_diversity(
    texts: List[str],
    n: int = 2,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> Dict[str, float]:
    """
    Compute n-gram diversity metrics.

    From the Massive Values paper: "The significant reduction of 2-gram diversity
    is a clear indicator that the model's generation struggles to form complete
    and coherent responses."

    Args:
        texts: List of generated texts
        n: n-gram size (default 2 for bigrams)
        tokenizer: Optional tokenizer for token-level ngrams. If None, uses word-level.

    Returns:
        Dict with diversity metrics
    """
    all_ngrams = []
    ngram_counts = Counter()

    for text in texts:
        if tokenizer is not None:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens = [str(t) for t in tokens]
        else:
            tokens = text.split()

        # Extract n-grams
        text_ngrams = list(zip(*[tokens[i:] for i in range(n)]))
        all_ngrams.extend(text_ngrams)
        ngram_counts.update(text_ngrams)

    total_ngrams = len(all_ngrams)
    unique_ngrams = len(ngram_counts)

    # Diversity = unique / total
    diversity = unique_ngrams / total_ngrams if total_ngrams > 0 else 0

    # Self-BLEU style repetition metric
    repetition_rate = 1 - diversity

    return {
        f"{n}gram_diversity": diversity,
        f"{n}gram_unique": unique_ngrams,
        f"{n}gram_total": total_ngrams,
        f"{n}gram_repetition": repetition_rate,
    }


def compute_accuracy(
    predictions: List[str],
    targets: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute accuracy for classification/generation tasks.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        normalize: Whether to lowercase and strip whitespace

    Returns:
        Dict with accuracy and counts
    """
    if normalize:
        predictions = [p.lower().strip() for p in predictions]
        targets = [t.lower().strip() for t in targets]

    correct = sum(p == t for p, t in zip(predictions, targets))
    total = len(predictions)

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
    }


def compute_exact_match(
    predictions: List[str],
    targets: List[str],
) -> Dict[str, float]:
    """
    Compute exact match accuracy (stricter than accuracy).
    """
    exact_matches = sum(p.strip() == t.strip() for p, t in zip(predictions, targets))
    total = len(predictions)

    return {
        "exact_match": exact_matches / total if total > 0 else 0,
        "matches": exact_matches,
        "total": total,
    }


def compute_generation_quality(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> Dict[str, any]:
    """
    Generate text and compute quality metrics.

    Returns perplexity, diversity, and sample generations.
    """
    device = next(model.parameters()).device
    model.eval()

    generations = []

    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        generations.append(generated)

    # Compute diversity
    diversity_2gram = compute_ngram_diversity(generations, n=2, tokenizer=tokenizer)
    diversity_3gram = compute_ngram_diversity(generations, n=3, tokenizer=tokenizer)

    # Compute perplexity on generations
    if generations:
        ppl_results = compute_perplexity(model, tokenizer, generations)
    else:
        ppl_results = {"perplexity": float("inf")}

    return {
        "generations": generations,
        "perplexity": ppl_results["perplexity"],
        **diversity_2gram,
        **diversity_3gram,
    }
