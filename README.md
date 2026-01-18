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

## Research Questions

If massive activations arise from RoPE and are key to contextual understanding, then:
- What happens to these activations when RoPE is dropped?
- Does the model develop alternative mechanisms for contextual knowledge?
- Can we understand *why* dropping RoPE extends context?

## Hypotheses

1. **Persistence Hypothesis**: Massive value patterns persist after RoPE removal
2. **Reorganization Hypothesis**: DroPE develops alternative attention mechanisms
3. **Decoupling Hypothesis**: Massive values and RoPE serve partially independent functions

## Project Structure

```
drope-activations/
├── src/
│   ├── massive_values/      # Core massive value analysis
│   │   ├── extraction.py    # Extract Q, K, V tensors
│   │   ├── analysis.py      # Compute massive value metrics
│   │   ├── disruption.py    # Disruption experiments
│   │   └── visualization.py # Generate figures
│   ├── drope/               # DroPE model handling
│   │   ├── conversion.py    # Convert RoPE → DroPE
│   │   └── recalibration.py # Training for recalibration
│   ├── evaluation/          # Task evaluation
│   │   ├── passkey.py       # Passkey retrieval task
│   │   └── metrics.py       # PPL, diversity, accuracy
│   └── utils/               # Utilities
│       ├── model_loading.py # Load models
│       └── data_loading.py  # Data pipelines
├── configs/                 # YAML configurations
│   ├── models.yaml          # Model definitions
│   ├── experiments.yaml     # Experiment configs
│   └── evaluation.yaml      # Evaluation settings
├── scripts/                 # Experiment runners
│   ├── run_phase1.sh        # Baseline analysis
│   ├── run_phase2.sh        # DroPE comparison
│   ├── run_phase3.sh        # Mechanistic analysis
│   └── run_phase4.sh        # Extended analysis
├── notebooks/               # Jupyter notebooks
├── results/                 # Output directory
│   ├── figures/
│   └── tables/
├── DroPE/                   # Submodule: DroPE training code
└── Rope_with_LLM/           # Submodule: Massive values analysis code
```

## Setup

```bash
# Create conda environment
conda create -n drope-activations python=3.11 -y
conda activate drope-activations

# Install dependencies
pip install -e ".[all]"

# Or minimal install
pip install -e .
```

## Quick Start

```python
from src.massive_values import extract_qkv_all_layers, analyze_massive_values
from src.utils import load_model

# Load a model
model, tokenizer = load_model("smollm-360m")

# Extract Q, K, V tensors
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.cuda()
qkv_by_layer = extract_qkv_all_layers(model, input_ids)

# Analyze massive values
for layer_idx, qkv in qkv_by_layer.items():
    q_analysis = analyze_massive_values(qkv.query)
    print(f"Layer {layer_idx}: {q_analysis.num_massive} massive values in Q")
```

## Running Experiments

### Phase 1: Baseline Analysis
```bash
./scripts/run_phase1.sh smollm-360m results/phase1
```

### Phase 2: DroPE Comparison
```bash
./scripts/run_phase2.sh smollm-360m results/phase2
```

### Phase 3: Mechanistic Analysis
```bash
./scripts/run_phase3.sh smollm-360m results/phase3
```

### Phase 4: Extended Analysis
```bash
./scripts/run_phase4.sh llama2-7b results/phase4
```

## Reference Repos

- `DroPE/` - Training code for DroPE models (SakanaAI)
- `Rope_with_LLM/` - Analysis code for massive activations (Jin et al.)

## Citation

If you use this code, please cite the original papers:

```bibtex
@article{jin2025massive,
  title={Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding},
  author={Jin, Mingyu and Sun, Kai and others},
  journal={ICML},
  year={2025}
}

@article{gelberg2025drope,
  title={DroPE: Dropping Positional Embeddings for Zero-Shot Long-Context Extension},
  author={Gelberg, Tal and others},
  journal={arXiv preprint arXiv:2512.12167},
  year={2025}
}
```
