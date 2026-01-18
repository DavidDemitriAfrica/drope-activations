# Massive Values in DroPE: Key Findings

## Finding 1: DroPE Reduces Massive Value Concentration

**Llama-2-7B comparison (RoPE vs DroPE after recalibration):**

| Tensor | RoPE | DroPE | Change |
|--------|------|-------|--------|
| Query  | 1488 | 915   | **-38%** |
| Key    | 1623 | 1474  | -9% |
| Value  | 185  | 187   | ~0% |

DroPE recalibration significantly reduces Query massive values while preserving Value patterns.

## Finding 2: DroPE Is Less Reliant on Massive Values

**Disruption experiment (zeroing massive value dimensions):**

| Model | Baseline PPL | After Disruption | Increase |
|-------|-------------|------------------|----------|
| RoPE  | 1.15        | 2,946            | +255,000% |
| DroPE | 1.29        | 31.72            | +2,356% |

**RoPE relies 109× more on massive values than DroPE.**

- RoPE is completely broken when massive values are zeroed
- DroPE is degraded but still functional
- Control (random positions): ~3% increase for both

## Implications

1. **Massive values are critical for RoPE** — the model cannot function without them
2. **DroPE learned alternative mechanisms** — it compensates when massive values are disrupted
3. **This may explain DroPE's context extension** — less concentrated attention could generalize to longer sequences

## Citation

```
Massive Values: Jin et al., "Massive Values in Self-Attention Modules
                are the Key to Contextual Knowledge Understanding" (ICML 2025)

DroPE: Gelberg et al., "Dropping Positional Embeddings for
       Zero-Shot Long-Context Extension" (arXiv:2512.12167)
```
