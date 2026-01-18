# drope-activations

Investigating the connection between RoPE-induced massive activations and context extension in LLMs.

## Background

This project combines insights from two papers:

1. **Massive Values in Self-Attention** (Jin, Sun et al., ICML 2025)
   - Paper: [arxiv.org/abs/2502.01563](https://arxiv.org/abs/2502.01563)
   - Finding: Massive values in Q and K matrices are concentrated in low-frequency dimensions
   - These values are responsible for contextual knowledge understanding (not parametric knowledge)
   - Root cause: RoPE's effects on low-frequency channels

2. **DroPE: Dropping Positional Embeddings** (Gelberg et al., 2025)
   - Paper: [arxiv.org/abs/2512.12167](https://arxiv.org/abs/2512.12167)
   - Method: CPT models to drop RoPE, extending context length
   - Models available on HuggingFace: [SakanaAI/DroPE](https://huggingface.co/collections/SakanaAI/drope)

## Hypothesis

If massive activations arise from RoPE and are key to contextual understanding, then:
- What happens to these activations when RoPE is dropped?
- Does the model develop alternative mechanisms for contextual knowledge?
- Can we understand *why* dropping RoPE extends context?

## Reference Repos

- `DroPE/` - Training code for DroPE models
- `Rope_with_LLM/` - Analysis code for massive activations

## Setup

```bash
conda create -n drope-activations python=3.11 -y && conda activate drope-activations
# Install deps from both projects as needed
```
