# Massive Activations in DroPE

Code and experiments investigating massive activations in DroPE (Dropping Positional Embeddings) models.

## Summary

We compare Llama-2-7B with and without RoPE to understand how removing positional embeddings affects the concentrated activations ("massive values") that prior work identifies as critical for contextual understanding.

**Main findings:**
- DroPE reduces Query massive values by 39% and Key by 11%
- RoPE models rely 82× more on massive values than DroPE (disruption causes 116,000% vs 1,400% perplexity increase)
- DroPE reorganizes rather than uniformly reduces: Layer 1 has 37× more massive values in DroPE

See [results/FINDINGS.md](results/FINDINGS.md) for full experimental details.

## Current Work: Phase Metrics Analysis

We are extending our analysis with Queipo-style phase metrics (from "Attention Sinks and Representation Compression Valleys"):

### Motivation

Jin et al. showed that massive values in Q/K are critical for contextual knowledge, and Queipo-de-Llano et al. showed that early layers develop "attention sinks" (BOS tokens as garbage collectors) and "compression valleys" (anisotropic representations). We hypothesize these phenomena are related:

1. **RoPE creates massive values** → enables attention sinks → functional but brittle
2. **DroPE removes RoPE** → different/fewer massive values → different attention patterns?

### Phase Metrics Being Computed

1. **BOS Norm & Ratio**: How much does the BOS token representation dominate?
2. **Representation Entropy**: Do we see compression valleys (anisotropic middle layers)?
3. **Attention Sink Rate**: What fraction of heads use BOS as a sink (τ=0.3)?

### Interventions

- **BOS-MLP Ablation**: Zero MLP output for BOS at the BOS spike layer
- **Q/K Massive Disruption**: Replace massive values with mean (Jin-style)
- **Combined**: Both interventions together

### Key Questions

- Does DroPE have attention sinks? If not, what replaces them?
- Does removing BOS-MLP destroy DroPE models like it does RoPE models?
- Are the "reorganized" massive values in DroPE Layer 1 serving the same function?

## Background

This work builds on:

1. **Jin et al. (2025)** - "Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding" ([arXiv:2502.01563](https://arxiv.org/abs/2502.01563))
   - Massive values in Q/K are concentrated in low-frequency dimensions
   - These values enable contextual (not parametric) knowledge understanding
   - Root cause: RoPE's effects on low-frequency channels

2. **Gelberg et al. (2025)** - "DroPE: Dropping Positional Embeddings" ([arXiv:2512.12167](https://arxiv.org/abs/2512.12167))
   - Removes RoPE from pretrained models via continued pretraining
   - Enables context length extension without architectural changes

3. **Queipo-de-Llano et al. (2025)** - "Attention Sinks and Representation Compression Valleys"
   - BOS tokens serve as attention sinks in early layers
   - Middle layers show compression valleys (anisotropic representations)
   - BOS-MLP ablation causes catastrophic failures in some models

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
# Massive value comparison (RoPE vs DroPE)
python scripts/run_llama_comparison.py

# Disruption experiment (10 seeds)
python scripts/run_disruption_rigorous.py

# Phase metrics analysis (BOS norm, entropy, sink rates)
python scripts/run_phase_metrics.py

# Generate figures
python scripts/create_findings_figures.py
python scripts/create_phase_figures.py
```

## Project Structure

```
drope-activations/
├── src/
│   ├── massive_values/      # Extraction and analysis
│   ├── drope/               # DroPE conversion utilities
│   └── utils/               # Model loading
├── scripts/                 # Experiment scripts
├── results/
│   ├── FINDINGS.md          # Full writeup with figures
│   └── findings_figures/    # Publication figures
├── DroPE/                   # Submodule: SakanaAI DroPE code
└── Rope_with_LLM/           # Submodule: Jin et al. analysis code
```

## Citation

```bibtex
@techreport{africa2026massive,
  title   = {Massive Activations in DroPE: Evidence for Attention Reorganization},
  author  = {Africa, David},
  year    = {2026},
  url     = {https://github.com/DavidDemitriAfrica/drope-activations}
}
```

Please also cite the underlying work:

```bibtex
@article{jin2025massive,
  title   = {Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding},
  author  = {Jin, Mingyu and Sun, Kai and others},
  journal = {ICML},
  year    = {2025}
}

@article{gelberg2025drope,
  title   = {DroPE: Dropping Positional Embeddings for Zero-Shot Long-Context Extension},
  author  = {Gelberg, Tal and others},
  journal = {arXiv preprint arXiv:2512.12167},
  year    = {2025}
}
```
