# Massive Activations in DroPE

Investigating how removing RoPE affects massive activations and attention sinks in language models.

## Key Finding

**Both RoPE and DroPE have ~97% attention sink rates, but only RoPE depends on BOS-MLP processing.**

| Model | Sink Rate | BOS-MLP Ablation |
|-------|-----------|------------------|
| RoPE  | 98.9%     | 1249× PPL increase (catastrophic) |
| DroPE | 96.7%     | 1.00× (no effect) |

This reveals that attention flowing to BOS does not imply functional dependence on BOS. DroPE has learned to make BOS expendable despite high attention to it.

## Results

See detailed findings in:
- [results/FINDINGS.md](results/FINDINGS.md) - Massive value comparison (Jin et al. replication)
- [results/phase_metrics/PHASE_FINDINGS.md](results/phase_metrics/PHASE_FINDINGS.md) - Phase metrics and BOS-MLP ablation

## DroPE Compatibility Notes

DroPE requires specific handling due to compatibility issues:

| Issue | Cause | Solution |
|-------|-------|----------|
| **Eager attention NaN** | DroPE produces NaN from layer 2+ with `attn_implementation="eager"` | Use SDPA (default) or compute attention manually via Q/K hooks |
| **Padding CUDA errors** | Index out of bounds with padded sequences | Disable padding or use variable-length inputs |

### Computing Sink Rates for DroPE

Standard `output_attentions=True` fails for DroPE. Use manual computation:

```python
# Hook into Q/K projections
layer.self_attn.q_proj.register_forward_hook(capture_hook('q'))
layer.self_attn.k_proj.register_forward_hook(capture_hook('k'))

# Compute attention manually
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
attn_weights = F.softmax(attn_scores.masked_fill(causal_mask, -inf), dim=-1)
```

See `scripts/fix_drope_sink_rates.py` for full implementation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
# Massive value comparison
python scripts/run_llama_comparison.py

# Disruption experiment
python scripts/run_disruption_rigorous.py

# Phase metrics (BOS norm, entropy, sink rates)
python scripts/run_phase_metrics.py

# Fix DroPE sink rates (manual Q/K method)
python scripts/fix_drope_sink_rates.py

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
│   ├── FINDINGS.md          # Massive value findings
│   ├── findings_figures/    # Figures for FINDINGS.md
│   └── phase_metrics/       # Phase metrics results
├── DroPE/                   # Submodule: SakanaAI DroPE code
└── Rope_with_LLM/           # Submodule: Jin et al. code
```

## Citation

```bibtex
@techreport{africa2026massive,
  title   = {Massive Activations in DroPE: BOS Attention Without BOS Dependence},
  author  = {Africa, David},
  year    = {2026},
  url     = {https://github.com/DavidDemitriAfrica/drope-activations}
}
```

## References

- Jin et al. (2025) - [Massive Values in Self-Attention Modules](https://arxiv.org/abs/2502.01563)
- Gelberg et al. (2025) - [DroPE: Dropping Positional Embeddings](https://arxiv.org/abs/2512.12167)
- Queipo-de-Llano et al. (2025) - Attention Sinks and Representation Compression Valleys
