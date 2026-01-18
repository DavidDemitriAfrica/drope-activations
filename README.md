# Massive Activations in DroPE

Code and experiments investigating massive activations in DroPE (Dropping Positional Embeddings) models.

## Summary

We compare Llama-2-7B with and without RoPE to understand how removing positional embeddings affects the concentrated activations ("massive values") that prior work identifies as critical for contextual understanding.

Main findings:
- DroPE reduces Query massive values by 39% and Key by 11%
- RoPE models rely 82× more on massive values than DroPE (disruption causes 116,000% vs 1,400% perplexity increase)
- DroPE reorganizes rather than uniformly reduces: Layer 1 has 37× more massive values in DroPE

See [results/FINDINGS.md](results/FINDINGS.md) for full experimental details.

## Background

This work builds on:

1. **Jin et al. (2025)** - "Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding" ([arXiv:2502.01563](https://arxiv.org/abs/2502.01563))
   - Massive values in Q/K are concentrated in low-frequency dimensions
   - These values enable contextual (not parametric) knowledge understanding
   - Root cause: RoPE's effects on low-frequency channels

2. **Gelberg et al. (2025)** - "DroPE: Dropping Positional Embeddings" ([arXiv:2512.12167](https://arxiv.org/abs/2512.12167))
   - Removes RoPE from pretrained models via continued pretraining
   - Enables context length extension without architectural changes

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

# Generate figures
python scripts/create_findings_figures.py
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
