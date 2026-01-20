# Phase Metrics Analysis: RoPE vs DroPE

## Executive Summary

We analyzed Queipo-style phase metrics and performed ablation experiments to understand how DroPE reorganizes attention compared to RoPE. Our key finding:

**Both RoPE and DroPE have nearly identical attention sink rates (~96-98%), but only RoPE is destroyed by BOS-MLP ablation.** BOS-MLP ablation causes catastrophic failure in RoPE (1249× PPL increase) but has zero effect on DroPE. This reveals that while both models attend to BOS, they use it fundamentally differently.

## Results Summary

### Functional Impact of Interventions

| Model | Baseline PPL | BOS-MLP Ablation | Change |
|-------|-------------|------------------|--------|
| RoPE  | 10.2        | 12766 | **1249×↑** |
| DroPE | 18.6        | 18.5  | **1.00×** (no change) |

### Attention Sink Rates (Corrected Measurement)

Using manual Q/K hook method to avoid NaN issues with DroPE's eager attention:

| Layer Range | RoPE | DroPE |
|-------------|------|-------|
| 0 (first) | 66% | 54% |
| 1 (early) | 78% | 31% |
| 2-30 (middle/late) | 97-100% | 94-99% |
| 31 (final) | 92% | 91% |
| **Average** | **97.8%** | **95.6%** |

**Both models have nearly identical sink rates.** The difference in BOS-MLP ablation impact cannot be explained by sink rate differences.

### Phase Metrics

| Metric | RoPE | DroPE |
|--------|------|-------|
| BOS Spike Layer | 1 | 2 |
| Peak BOS Norm | 982.1 (layer 23) | 632.2 (layer 26) |
| Min Entropy (Compression Valley) | 0.0081 (layer 1) | 0.0895 (layer 2) |

## Key Findings

### 1. Sink Rate Is Not the Key Difference

Both RoPE (97.8%) and DroPE (95.6%) have nearly identical attention sink rates. Despite high attention to BOS in both models, only RoPE catastrophically fails when BOS-MLP is ablated.

This means the critical difference is **what information is stored in BOS**, not how much attention flows to it.

### 2. BOS-MLP Ablation Reveals Different BOS Functionality

| Model | BOS-MLP Ablation Impact |
|-------|-------------------------|
| RoPE | 1249× PPL increase - catastrophic failure |
| DroPE | 1.00× - zero impact |

In RoPE, the BOS token at the spike layer (layer 1) encodes critical position-related information that the entire network depends on. Zeroing the MLP output removes this information and destroys the model.

In DroPE, despite attention flowing to BOS, the information stored there is not critical for model function. This suggests DroPE has reorganized what information is stored in BOS, making it expendable.

### 3. DroPE has More Isotropic Representations

The entropy at the compression valley is 11× higher in DroPE (0.0895 vs 0.0081), indicating more isotropic (less compressed) representations. This suggests DroPE maintains richer, more distributed representations instead of collapsing information into specific tokens.

### 4. Q/K Disruption Impact

| Model | Q/K Disruption Impact |
|-------|----------------------|
| RoPE | 6.7× PPL increase |
| DroPE | 29% PPL increase |

DroPE is more robust to Q/K massive value disruption, consistent with its less position-dependent attention patterns.

## Interpretation

### Why Is DroPE Immune to BOS-MLP Ablation?

RoPE creates position-dependent attention patterns through rotary embeddings. The BOS token at position 0 develops unique rotational properties that make it special. The MLP at the BOS spike layer (layer 1) encodes critical positional information into BOS that all subsequent layers depend on.

DroPE removes rotary embeddings, eliminating the positional asymmetry that makes BOS special. While attention still flows to BOS (perhaps due to the causal mask making it always visible), the information stored there is not critical. DroPE has learned to distribute critical information elsewhere in the sequence.

### Implications for Understanding Transformers

1. **Attention sink rate ≠ functional importance** - High attention to a token doesn't mean the token stores critical information
2. **RoPE creates position-dependent critical tokens** - The BOS token becomes a critical repository for position information in RoPE
3. **DroPE achieves redundancy** - By removing positional encoding, DroPE learns more distributed representations that don't depend on any single token

## Experimental Notes

### DroPE Eager Attention NaN Issue

DroPE produces NaN values in hidden states and attention weights when using eager attention (`attn_implementation="eager"`). This is a compatibility issue between DroPE's NoPE attention wrapper and the transformers library's eager attention path.

**Impact**: Standard `output_attentions=True` cannot be used for DroPE sink rate measurement.

**Solution**: We compute attention weights manually using forward hooks on Q and K projection layers, then applying softmax(QK^T / sqrt(d)) ourselves. This produces valid attention weights.

### Verification

All key findings have been verified:
- BOS-MLP ablation results confirmed with independent PPL measurements
- Sink rates computed using manual hook method to avoid NaN issues
- Both models tested under identical conditions

## Methods

### Phase Metrics (Queipo-de-Llano et al.)
- **BOS Norm**: L2 norm of BOS token residual at each layer
- **Entropy**: Normalized entropy of singular values from SVD
- **Sink Rate**: Fraction of heads where average attention to BOS ≥ τ (0.3)

### Interventions
- **BOS-MLP Ablation**: Zero the MLP output for BOS token at BOS spike layer
- **Q/K Disruption**: Replace massive Q/K values (>10 std) with mean (Jin-style)

### Functional Evaluations
- **Perplexity**: WikiText-2 samples

## Citation

```bibtex
@techreport{africa2026massive,
  title   = {Massive Activations in DroPE: BOS Attention Without BOS Dependence},
  author  = {Africa, David},
  year    = {2026},
  url     = {https://github.com/DavidDemitriAfrica/drope-activations}
}
```
