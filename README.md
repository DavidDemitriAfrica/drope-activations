# Massive Activations in DroPE

Investigating how removing RoPE affects massive activations and attention sinks in language models.

## Summary

We compare Llama-2-7B with and without RoPE to understand how removing positional embeddings affects the concentrated activations ("massive values") that prior work identifies as critical for contextual understanding.

**Main findings**

1. DroPE reduces Query massive values by 39% and Key by 11%
2. RoPE relies 82× more on massive values than DroPE (disruption causes 116,000% vs 1,400% PPL increase)
3. Disrupting massive values degrades RoPE's contextual knowledge by 94% but DroPE's by only 25%
4. Passkey retrieval collapses completely in RoPE (100%→0%) but is unaffected in DroPE (60%→60%)
5. Both models have ~97% attention sink rates, but only RoPE depends on BOS-MLP processing (1249× vs 1.00× PPL change)

These findings suggest DroPE learns alternative attention mechanisms that don't rely on concentrated features or specific tokens.

See [results/FINDINGS.md](results/FINDINGS.md) for full details.

## DroPE Compatibility

DroPE requires specific handling due to bugs in eager attention mode.

| Issue | Cause | Solution |
|-------|-------|----------|
| Eager attention NaN | DroPE produces NaN from layer 2+ with `attn_implementation="eager"` | Use SDPA (default) or manual Q/K hooks |
| Padding CUDA errors | Index out of bounds with padded sequences | Use variable-length inputs |

For attention analysis, compute weights manually via hooks on Q/K projections. See `scripts/fix_drope_sink_rates.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
python scripts/run_llama_comparison.py       # Massive value comparison
python scripts/run_disruption_rigorous.py    # Disruption experiment
python scripts/run_phase_metrics.py          # Phase metrics
python scripts/fix_drope_sink_rates.py       # Sink rates (manual method)
python scripts/create_findings_figures.py    # Figures
```

## Project Structure

```
drope-activations/
├── src/                    # Analysis library
├── scripts/                # Experiment scripts
├── results/
│   ├── FINDINGS.md         # Full writeup
│   └── findings_figures/   # Figures
├── DroPE/                  # SakanaAI DroPE code
└── Rope_with_LLM/          # Jin et al. code
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

## References

- Jin et al. (2025) - [Massive Values in Self-Attention Modules](https://arxiv.org/abs/2502.01563)
- Gelberg et al. (2025) - [DroPE: Dropping Positional Embeddings](https://arxiv.org/abs/2512.12167)
- Queipo-de-Llano et al. (2025) - [Attention Sinks and Compression Valleys](https://arxiv.org/abs/2510.06477)
