# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DroPE Activations is a research project investigating how removing RoPE (Rotary Positional Embeddings) affects "massive values"—concentrated large activations in language model attention mechanisms. The project compares RoPE-based models with DroPE variants to understand attention reorganization and contextual knowledge understanding.

**Key Research Finding**: DroPE models learn alternative attention mechanisms that don't rely on concentrated features. RoPE depends 82× more on massive values than DroPE, and passkey retrieval (pure contextual task) is completely unaffected by massive value disruption in DroPE but collapses entirely in RoPE.

## Commands

```bash
# Setup
pip install -r requirements.txt
pip install -e .                    # Editable install
pip install -e ".[dev]"             # With dev tools
pip install -e ".[all]"             # Full install with flash-attn, quantization

# Code quality
black src/
isort src/
mypy src/
pytest tests/

# Main experiments
python scripts/run_llama_comparison.py           # Massive value comparison (RoPE vs DroPE)
python scripts/run_disruption_rigorous.py        # Statistical disruption analysis
python scripts/run_phase_metrics.py              # Phase metrics: BOS residual, entropy, sinks
python scripts/run_functional_tests.py           # Knowledge tests (4 tasks × conditions)
python scripts/fix_drope_sink_rates.py           # Attention sink rates (manual Q/K method)

# Visualization
python scripts/create_findings_figures.py        # Main publication figures
python scripts/create_phase_figures.py           # Phase metrics figures

# Phase workflows
bash scripts/run_phase1.sh [model] [output_dir]  # Baseline analysis
bash scripts/run_phase2.sh [model] [output_dir]  # DroPE analysis
bash scripts/run_phase3.sh [model] [output_dir]  # Mechanistic analysis
bash scripts/run_phase4.sh [model] [output_dir]  # Extended analysis
```

## Architecture

```
src/
├── drope/                    # DroPE model conversion
│   ├── conversion.py         # Remove RoPE from attention layers
│   └── recalibration.py      # DroPE recalibration training
├── massive_values/           # Core analysis module
│   ├── extraction.py         # QKVHookedExtractor - extracts Q/K/V tensors via forward hooks
│   ├── analysis.py           # identify_massive_values(), compute_massive_value_matrix()
│   ├── disruption.py         # MassiveValueDisruptor - zeros/perturbs dimensions
│   └── visualization.py      # Heatmaps, layer progression plots
├── evaluation/
│   ├── passkey.py            # PasskeyRetrievalEvaluator - in-context memory test
│   └── metrics.py            # compute_perplexity() - sliding window perplexity
└── utils/
    ├── model_loading.py      # load_model() - handles RoPE/DroPE, quantization
    └── data_loading.py       # Dataset utilities
```

### Configuration Files
- `configs/models.yaml` - Model definitions (SmolLM, Llama-2, Llama-3, Qwen, Mistral)
- `configs/experiments.yaml` - Phase 1-4 experiment configurations
- `configs/evaluation.yaml` - Evaluation task settings

### Results Structure
- `results/FINDINGS.md` - Full technical findings document
- `results/phase_metrics/` - Phase metrics JSON data and figures
- `results/findings_figures/` - Publication figures

## DroPE Compatibility Notes

1. **Eager attention NaN bug**: DroPE produces NaN from layer 2+ with `attn_implementation="eager"`. Use SDPA (default) or manual Q/K hooks.

2. **Padding CUDA errors**: Index out of bounds with padded sequences. Use variable-length inputs only.

## External Submodules

- `DroPE/` - SakanaAI's DroPE implementation with custom attention modules
- `Rope_with_LLM/` - Jin et al. massive values code (ICML 2025)
