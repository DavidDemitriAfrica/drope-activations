# Phase Metrics Analysis: RoPE vs DroPE

## Summary

We extend our analysis with metrics from Queipo-de-Llano et al. (2025), who showed that transformer models develop "attention sinks" (tokens that receive high attention regardless of content) and "compression valleys" (low-entropy representations in middle layers). They trace both phenomena to massive activations in the residual stream.

**Main finding:** Both RoPE and DroPE have nearly identical attention sink rates (~97%), but only RoPE is destroyed by BOS-MLP ablation. This reveals that attention flowing to BOS does not imply functional dependence on BOS.

| Model | Sink Rate | BOS-MLP Ablation |
|-------|-----------|------------------|
| RoPE  | 97.8%     | 1249× PPL increase |
| DroPE | 95.6%     | 1.00× (no effect) |

## Background

### Attention Sinks

Attention sinks are tokens (typically BOS) that receive disproportionate attention across many heads. Queipo-de-Llano et al. define the sink rate as the fraction of attention heads where average attention to BOS exceeds a threshold τ=0.3.

### BOS-MLP Ablation

The BOS token develops extreme activation norms in early layers via MLP processing. Ablating (zeroing) the MLP output for BOS at its "spike layer" reveals whether the model depends on whatever information is stored there.

## Experiment 4: Attention Sinks and BOS-MLP Ablation

### Methodology

**Sink Rate Measurement:**

Standard `output_attentions=True` produces NaN for DroPE (see Technical Notes below). We compute attention weights manually using forward hooks:

```python
# Hook into Q/K projections
layer.self_attn.q_proj.register_forward_hook(capture_hook('q'))
layer.self_attn.k_proj.register_forward_hook(capture_hook('k'))

# Compute attention manually
scale = head_dim ** -0.5
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
attn_weights = F.softmax(attn_scores.masked_fill(causal_mask, -inf), dim=-1)

# Sink rate: fraction of heads with mean attention to BOS >= 0.3
bos_attention = attn_weights[:, :, :, 0].mean(dim=2)  # average across positions
sink_rate = (bos_attention >= 0.3).float().mean()
```

**BOS-MLP Ablation:**

We zero the MLP output for the BOS token at the BOS spike layer (layer 1 for RoPE, layer 2 for DroPE):

```python
def bos_mlp_ablation_hook(module, input, output):
    output[0, 0, :] = 0  # Zero BOS token's MLP output
    return output

model.layers[spike_layer].mlp.register_forward_hook(bos_mlp_ablation_hook)
```

### Results

**Sink Rates by Layer:**

| Layer | RoPE | DroPE |
|-------|------|-------|
| 0 | 66% | 54% |
| 1 | 78% | 31% |
| 2-30 | 97-100% | 94-99% |
| 31 | 92% | 91% |
| **Average** | **97.8%** | **95.6%** |

Both models have high sink rates across nearly all layers.

**BOS-MLP Ablation:**

| Model | Baseline PPL | Ablated PPL | Change |
|-------|--------------|-------------|--------|
| RoPE  | 10.2 | 12,766 | **1249×** |
| DroPE | 18.6 | 18.5 | **1.00×** |

### Interpretation

The results are striking: despite nearly identical sink rates, the models respond completely differently to BOS-MLP ablation.

**RoPE:** Catastrophic failure. The model cannot function without BOS-MLP processing, suggesting BOS stores critical position-dependent information.

**DroPE:** Zero effect. Despite 95.6% of heads attending to BOS, the information stored there is expendable.

This means **sink rate ≠ functional importance**. Both models route attention to BOS, but only RoPE stores critical information there. DroPE has learned to make BOS a "garbage collector" that receives attention but stores nothing essential.

## Other Phase Metrics

### BOS Norm and Compression

| Metric | RoPE | DroPE |
|--------|------|-------|
| BOS Spike Layer | 1 | 2 |
| Peak BOS Norm | 982 (layer 23) | 632 (layer 26) |
| Min Entropy | 0.008 (layer 1) | 0.090 (layer 2) |

DroPE has:
- 35% lower peak BOS norm
- 11× higher minimum entropy (less compressed representations)

### Q/K Disruption (from Experiment 3)

| Model | Baseline PPL | Disrupted PPL | Change |
|-------|--------------|---------------|--------|
| RoPE | 1.30 | 25.6 | 6.7× |
| DroPE | 1.49 | 5.46 | 29% |

DroPE is more robust to Q/K massive value disruption, consistent with its reduced dependence on concentrated features.

## Technical Notes: DroPE Compatibility

DroPE requires specific handling:

| Issue | Cause | Solution |
|-------|-------|----------|
| Eager attention NaN | DroPE produces NaN from layer 2+ with `attn_implementation="eager"` | Use SDPA (default) or manual Q/K hooks |
| Padding CUDA errors | Index out of bounds with padded sequences | Use variable-length inputs |

The eager attention issue appears to be a compatibility problem between DroPE's NoPE attention wrapper (which sets cos=1, sin=0 to nullify RoPE) and the transformers library's eager attention path.

## Implications

1. **Attention sinks are not the key difference** between RoPE and DroPE. Both have ~97% sink rates.

2. **What matters is what's stored in BOS**, not how much attention flows there. RoPE uses BOS to encode position-dependent information; DroPE makes BOS expendable.

3. **DroPE achieves redundancy** by distributing critical information across the sequence rather than concentrating it in specific tokens.

4. **This explains the massive value findings**: DroPE's massive values appear vestigial because the model has reorganized to not depend on any single token or dimension.

## Reproducibility

```bash
# Compute sink rates (manual Q/K method)
python scripts/fix_drope_sink_rates.py

# Generate figures
python scripts/create_phase_figures.py
```

## References

- Queipo-de-Llano et al. (2025) - "Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin" ([arXiv:2510.06477](https://arxiv.org/abs/2510.06477))
- Jin et al. (2025) - "Massive Values in Self-Attention Modules" ([arXiv:2502.01563](https://arxiv.org/abs/2502.01563))
- Gelberg et al. (2025) - "DroPE: Dropping Positional Embeddings" ([arXiv:2512.12167](https://arxiv.org/abs/2512.12167))
