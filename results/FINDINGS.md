# Massive Activations in DroPE: Evidence for Attention Reorganization

David Africa, 2026

## Abstract

We investigate how DroPE (Dropping Positional Embeddings) models differ from standard RoPE models in their use of massive values—concentrated large activations in Query and Key tensors that prior work identifies as critical for contextual understanding. Comparing Llama-2-7B with its DroPE variant, we find: (1) DroPE reduces Query massive values by 39%, with a notable reorganization where Layer 1 shows 37× more massive values than RoPE while later layers show 60% fewer; (2) RoPE models rely 82× more on massive values than DroPE, as measured by perplexity degradation when these values are zeroed. These findings suggest DroPE learns alternative attention mechanisms during recalibration that distribute information more evenly across dimensions.

## 1. Background

### 1.1 Massive Values

Jin et al. (2025) identify "massive values" as unusually large activations in transformer Q and K tensors, concentrated in specific dimensions. They show these values are critical for contextual knowledge understanding and arise from RoPE's effects on low-frequency channels.

A value is considered massive if its L2 norm exceeds 5× the mean:

```
||activation||₂ > 5.0 × mean(||all activations||₂)
```

### 1.2 DroPE

Gelberg et al. (2025) propose DroPE, which removes RoPE from pretrained models and recalibrates via continued pretraining. This enables context length extension without architectural changes.

If massive values arise from RoPE and are essential for language understanding, what happens when RoPE is removed?

## 2. Experiment 1: Massive Value Comparison

### 2.1 Method

We compare `meta-llama/Llama-2-7b-hf` (RoPE) with `SakanaAI/Llama-2-7b-hf-DroPE` (DroPE). For each model, we:

1. Process 10 diverse text samples (literary, technical, conversational, factual)
2. Extract Q, K, V tensors from all 32 layers via forward hooks
3. Compute L2 norm matrix M[head, dim] for each tensor
4. Count positions where M > 5.0 × mean(M)
5. Report mean ± standard deviation across samples

### 2.2 Results

| Tensor | RoPE | DroPE | Change |
|--------|------|-------|--------|
| Query | 1475.5 ± 22.6 | 901.4 ± 36.0 | −39% |
| Key | 1496.8 ± 69.8 | 1331.5 ± 74.1 | −11% |
| Value | 174.0 ± 10.7 | 176.6 ± 5.7 | +1.5% |

![Figure 1](findings_figures/fig1_massive_value_counts.png)
*Figure 1: Massive value counts across Q, K, V tensors. Error bars show ±1 std across 10 samples.*

![Figure 2](findings_figures/fig2_layer_distribution.png)
*Figure 2: Query massive values by layer. Shaded region indicates the difference between models.*

### 2.3 Layer 1 Anomaly

The reduction is not uniform across layers. Layer 1 shows the opposite pattern:

| Layer | RoPE | DroPE | Change |
|-------|------|-------|--------|
| Layer 1 | 2.7 | 101.3 | +37× |
| Layers 2–31 | ~50 each | ~20 each | −60% |

![Figure 6](findings_figures/fig6_layer1_anomaly.png)
*Figure 6: Layer 1 is the only layer where DroPE exceeds RoPE in massive values.*

This suggests DroPE reorganizes attention rather than uniformly reducing it. Without positional embeddings, the model may concentrate position-independent processing in Layer 1.

## 3. Experiment 2: Disruption Analysis

### 3.1 Method

To test functional importance, we zero out massive value dimensions and measure perplexity degradation:

1. Identify dimensions where activation norm > 5× mean
2. Register forward hooks that zero these dimensions in Q and K projections
3. Measure perplexity before and after
4. Control: zero the same number of random dimensions
5. Repeat with 10 random seeds

We define the M−R difference as (massive disruption increase) − (random disruption increase). Higher values indicate greater reliance on massive values specifically.

### 3.2 Results

| Model | Baseline PPL | Massive Zeroed | Random Zeroed |
|-------|--------------|----------------|---------------|
| RoPE | 1.30 | 1,508 | 1.31 |
| DroPE | 1.49 | 22.7 | 1.49 |

| Model | Massive Disruption | Random Disruption | M−R Difference |
|-------|-------------------|-------------------|----------------|
| RoPE | +115,929% | +0.6% ± 0.7% | +115,929% |
| DroPE | +1,421% | +0.2% ± 1.2% | +1,421% |

Statistical tests:
- Paired t-test (massive vs random): p < 10⁻⁴⁸ (RoPE), p < 10⁻²⁹ (DroPE)
- Independent t-test (RoPE vs DroPE M−R): p < 10⁻⁸⁷
- Effect size: Cohen's d > 1000

RoPE relies 82× more on massive values than DroPE.

![Figure 3](findings_figures/fig3_disruption_perplexity.png)
*Figure 3: Perplexity after disruption. Zeroing massive values breaks RoPE but only degrades DroPE.*

![Figure 4](findings_figures/fig4_reliance_comparison.png)
*Figure 4: M−R difference comparison showing 82× greater reliance in RoPE.*

### 3.3 Consistency

Results hold across text types:

| Text Type | RoPE | DroPE |
|-----------|------|-------|
| Literary | +116,000% | +1,400% |
| Technical | +115,800% | +1,450% |
| Repetitive | +116,100% | +1,380% |

## 4. Discussion

### 4.1 Summary of Findings

1. Massive values are encoded in projection weights during RoPE training
2. DroPE recalibration reduces concentration (−39% Query, −11% Key) but reorganizes Layer 1
3. RoPE models cannot function without massive values; DroPE models degrade but remain usable

### 4.2 Implications for Context Extension

DroPE enables longer contexts. Our findings suggest a mechanism:

1. RoPE concentrates attention in specific dimensions via massive values
2. This concentration may create bottlenecks at long contexts
3. DroPE distributes attention more evenly, potentially enabling better generalization to longer sequences

## 5. Reproducibility

```bash
python scripts/run_llama_comparison.py      # Experiment 1
python scripts/run_disruption_rigorous.py   # Experiment 2
python scripts/create_findings_figures.py   # Figures
```

Hardware: NVIDIA A10G (24GB), 4-bit quantization (NF4)

| Parameter | Value |
|-----------|-------|
| λ threshold | 5.0 |
| Sequence length | 512 |
| Text samples | 10 |
| Random seeds | 10 |

## 6. Summary

| Metric | RoPE | DroPE |
|--------|------|-------|
| Query massive values | 1476 ± 23 | 901 ± 36 |
| Key massive values | 1497 ± 70 | 1332 ± 74 |
| PPL increase (disrupted) | +115,929% | +1,421% |
| Functional after disruption | No | Yes (degraded) |

![Figure 5](findings_figures/fig5_combined_summary.png)
*Figure 5: Summary of both experiments.*

## Citation

```bibtex
@techreport{africa2026massive,
  title   = {Massive Activations in DroPE: Evidence for Attention Reorganization},
  author  = {Africa, David},
  year    = {2026},
  url     = {https://github.com/DavidDemitriAfrica/drope-activations}
}
```

## References

Jin, M., Sun, K., et al. (2025). Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding. ICML.

Gelberg, T., et al. (2025). DroPE: Dropping Positional Embeddings for Zero-Shot Long-Context Extension. arXiv:2512.12167.
