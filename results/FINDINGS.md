# Massive Activations in DroPE: Evidence for Attention Reorganization

David Africa, 2026

## Abstract

We investigate how DroPE (Dropping Positional Embeddings) models differ from standard RoPE models in their use of massive values—concentrated large activations in Query and Key tensors that prior work identifies as critical for contextual understanding. Comparing Llama-2-7B with its DroPE variant, we find: (1) DroPE reduces Query massive values by 39%, with a notable reorganization where Layer 1 shows 37× more massive values than RoPE while later layers show 60% fewer; (2) RoPE models rely 82× more on massive values than DroPE, as measured by perplexity degradation when these values are zeroed; (3) While RoPE follows Jin et al.'s pattern where disruption degrades contextual knowledge (94.3%) far more than parametric (24.5%), DroPE shows dramatically different behavior—contextual degradation is 73% lower (25% vs 94.3%), and parametric accuracy actually *improves* when disrupted. Most strikingly, passkey retrieval—a pure contextual task—collapses completely under disruption in RoPE (100%→0%) but is entirely unaffected in DroPE (60%→60%). These findings suggest DroPE's massive values may be vestigial artifacts rather than functional features, and that the model learns fundamentally different attention mechanisms during recalibration.

## 1. Background

### 1.1 Massive Values

Jin et al. (2025) identify "massive values" as unusually large activations in transformer Q and K tensors, concentrated in specific dimensions. They show these values are critical for contextual knowledge understanding and arise from RoPE's effects on low-frequency channels.

A value is considered massive if its L2 norm exceeds 5× the mean:

```
||activation||₂ > 5.0 × mean(||all activations||₂)
```

### 1.2 DroPE

Gelberg et al. (2025) propose DroPE, which removes RoPE from pretrained models and recalibrates via continued pretraining. This enables context length extension without architectural changes.

If massive values arise from RoPE and are essential for language understanding, what happens when RoPE is removed?

## 2. Experiment 1: Massive Value Comparison

### 2.1 Method

We compare `meta-llama/Llama-2-7b-hf` (RoPE) with `SakanaAI/Llama-2-7b-hf-DroPE` (DroPE). For each model, we:

1. Process 10 diverse text samples (literary, technical, conversational, factual)
2. Extract Q, K, V tensors from all 32 layers via forward hooks
3. Compute L2 norm matrix M[head, dim] for each tensor
4. Count positions where M > 5.0 × mean(M)
5. Report mean ± standard deviation across samples

### 2.2 Results

| Tensor | RoPE | DroPE | Change |
|--------|------|-------|--------|
| Query | 1475.5 ± 22.6 | 901.4 ± 36.0 | −39% |
| Key | 1496.8 ± 69.8 | 1331.5 ± 74.1 | −11% |
| Value | 174.0 ± 10.7 | 176.6 ± 5.7 | +1.5% |

![Figure 1](findings_figures/fig1_massive_value_counts.png)
*Figure 1: Massive value counts across Q, K, V tensors. Error bars show ±1 std across 10 samples.*

![Figure 2](findings_figures/fig2_layer_distribution.png)
*Figure 2: Query massive values by layer. Shaded region indicates the difference between models.*

### 2.3 Layer 1 Anomaly

The reduction is not uniform across layers. Layer 1 shows the opposite pattern:

| Layer | RoPE | DroPE | Change |
|-------|------|-------|--------|
| Layer 1 | 2.7 | 101.3 | +37× |
| Layers 2–31 | ~50 each | ~20 each | −60% |

![Figure 6](findings_figures/fig6_layer1_anomaly.png)
*Figure 6: Layer 1 is the only layer where DroPE exceeds RoPE in massive values.*

This suggests DroPE reorganizes attention rather than uniformly reducing it. Without positional embeddings, the model may concentrate position-independent processing in Layer 1.

## 3. Experiment 2: Disruption Analysis

### 3.1 Method

To test functional importance, we zero out massive value dimensions and measure perplexity degradation:

1. Identify dimensions where activation norm > 5× mean
2. Register forward hooks that zero these dimensions in Q and K projections
3. Measure perplexity before and after
4. Control: zero the same number of random dimensions
5. Repeat with 10 random seeds

We define the M−R difference as (massive disruption increase) − (random disruption increase). Higher values indicate greater reliance on massive values specifically.

### 3.2 Results

| Model | Baseline PPL | Massive Zeroed | Random Zeroed |
|-------|--------------|----------------|---------------|
| RoPE | 1.30 | 1,508 | 1.31 |
| DroPE | 1.49 | 22.7 | 1.49 |

| Model | Massive Disruption | Random Disruption | M−R Difference |
|-------|-------------------|-------------------|----------------|
| RoPE | +115,929% | +0.6% ± 0.7% | +115,929% |
| DroPE | +1,421% | +0.2% ± 1.2% | +1,421% |

Statistical tests:
- Paired t-test (massive vs random): p < 10⁻⁴⁸ (RoPE), p < 10⁻²⁹ (DroPE)
- Independent t-test (RoPE vs DroPE M−R): p < 10⁻⁸⁷
- Effect size: Cohen's d > 1000

RoPE relies 82× more on massive values than DroPE.

![Figure 3](findings_figures/fig3_disruption_perplexity.png)
*Figure 3: Perplexity after disruption. Zeroing massive values breaks RoPE but only degrades DroPE.*

![Figure 4](findings_figures/fig4_reliance_comparison.png)
*Figure 4: M−R difference comparison showing 82× greater reliance in RoPE.*

### 3.3 Consistency

Results hold across text types:

| Text Type | RoPE | DroPE |
|-----------|------|-------|
| Literary | +116,000% | +1,400% |
| Technical | +115,800% | +1,450% |
| Repetitive | +116,100% | +1,380% |

## 4. Experiment 3: Parametric vs Contextual Knowledge

### 4.1 Background

Jin et al. (2025) demonstrate that massive value disruption affects contextual knowledge (~95% degradation) far more than parametric knowledge (~11% degradation). We replicate their methodology on both RoPE and DroPE using the same task categories:

**Parametric Tasks** (facts stored in model weights):
- **Cities** (n=200): Yes/no factual statements ("Paris is in France")
- **Sports** (n=100): Yes/no sports knowledge questions

**Contextual Tasks** (information extracted from input context):
- **IMDB** (n=100): Sentiment classification from movie reviews
- **Passkey** (n=20): Retrieve 5-digit number hidden in filler text

**Disruption Method**: Replace top-1 massive dimension per head with mean value (per Jin et al.)

### 4.2 Results

| Model | Category | Task | Baseline | Disrupted | Degradation |
|-------|----------|------|----------|-----------|-------------|
| RoPE | parametric | cities | 89.5% | 57.5% | 35.8% |
| RoPE | parametric | sports | 60.0% | 52.0% | 13.3% |
| RoPE | contextual | imdb | 44.0% | 5.0% | **88.6%** |
| RoPE | contextual | passkey | 100.0% | 0.0% | **100.0%** |
| DroPE | parametric | cities | 79.0% | 88.5% | **−12.0%** |
| DroPE | parametric | sports | 73.0% | 67.0% | 8.2% |
| DroPE | contextual | imdb | 30.0% | 15.0% | 50.0% |
| DroPE | contextual | passkey | 60.0% | 60.0% | **0.0%** |

### 4.3 Average Degradation by Category

| Model | Parametric Avg | Contextual Avg | Ratio (ctx/param) |
|-------|----------------|----------------|-------------------|
| **RoPE** | 24.5% | **94.3%** | 3.8× |
| **DroPE** | −1.9% | 25.0% | — |

![Figure 7](findings_figures/fig7_degradation_comparison.png)
*Figure 7: Degradation comparison across all tasks. RoPE shows severe contextual degradation (88-100%), while DroPE is dramatically more robust.*

### 4.4 Key Findings

**RoPE confirms Jin et al.'s pattern:**
- Contextual knowledge degrades 94.3% on average
- Parametric knowledge degrades only 24.5%
- Passkey retrieval collapses completely (100% → 0%)
- Massive values are critical for contextual understanding

**DroPE shows dramatically different behavior:**
- Contextual degradation is 73% lower than RoPE (25% vs 94.3%)
- Parametric accuracy actually **improves** when disrupted (−1.9%)
- **Passkey is completely unaffected** (60% → 60%, 0% degradation)
- Massive values appear non-functional for information storage

### 4.5 The Passkey Result

The passkey task is particularly revealing:
- **RoPE**: 100% baseline → 0% disrupted = complete collapse
- **DroPE**: 60% baseline → 60% disrupted = **zero degradation**

This demonstrates that DroPE's contextual retrieval mechanism is entirely independent of massive values. The model has learned alternative attention patterns that don't rely on value concentration.

![Figure 8](findings_figures/fig8_passkey_spotlight.png)
*Figure 8: Passkey retrieval results. RoPE completely collapses (100% degradation) while DroPE is entirely unaffected (0% degradation).*

### 4.6 Interpretation

The DroPE improvement under disruption suggests massive values are **vestigial artifacts** from the original RoPE training:

1. During RoPE pretraining, massive values encode positional information
2. DroPE removes positional embeddings and recalibrates
3. Massive values persist structurally but lose their function
4. They may actually create interference, explaining the improvement when disrupted

The stark contrast in passkey results (100% collapse vs 0% degradation) provides the clearest evidence that DroPE develops fundamentally different attention mechanisms.

![Figure 9](findings_figures/fig9_category_averages.png)
*Figure 9: Average degradation by knowledge category. RoPE's contextual knowledge degrades 3.8× more than parametric.*

![Figure 10](findings_figures/fig10_baseline_vs_disrupted.png)
*Figure 10: Baseline vs disrupted accuracy for all tasks. Note DroPE's Cities accuracy actually improves under disruption.*

![Figure 11](findings_figures/fig11_jin_summary.png)
*Figure 11: Combined summary of Jin et al. replication results.*

## 5. Experiment 4: Attention Sinks and BOS-MLP Ablation

### 5.1 Background

Prior work has identified several related phenomena in transformer models. Queipo-de-Llano et al. (2025) connect these phenomena, arguing they share a common cause in massive activations.

**Attention Sinks**

Attention sinks are tokens that receive disproportionate attention regardless of content. In autoregressive models, the BOS token typically becomes a sink because it is always visible to all positions. A head is classified as a "sink head" if its average attention to BOS exceeds a threshold τ. We use τ=0.3. The sink rate is the fraction of heads that qualify as sink heads.

```
sink_score(head) = mean attention to BOS across all query positions
sink_head = 1 if sink_score >= 0.3 else 0
sink_rate = mean(sink_head) across all heads
```

**BOS Norm and Spike Layer**

The BOS token develops unusually large activation norms in early layers, primarily through MLP processing. The "spike layer" is where the BOS norm first exceeds the mean norm of other tokens by a large margin.

**Compression Valleys**

Compression valleys are layers where representation entropy drops sharply, indicating that information is being compressed into fewer dimensions. We measure this via singular value entropy of the hidden state matrix.

```
entropy = -Σ (σᵢ/Σσ) × log(σᵢ/Σσ)
```

where σᵢ are the singular values. Lower entropy means more concentrated (compressed) representations.

We hypothesized that DroPE might eliminate attention sinks since RoPE creates the positional asymmetry that makes BOS special.

### 5.2 Method

Standard `output_attentions=True` produces NaN for DroPE due to a compatibility issue between DroPE's NoPE attention wrapper and the transformers library's eager attention path. We compute attention weights manually using forward hooks on Q/K projections.

```python
# Hook into Q/K projections
layer.self_attn.q_proj.register_forward_hook(capture_hook('q'))
layer.self_attn.k_proj.register_forward_hook(capture_hook('k'))

# Compute attention manually
scale = head_dim ** -0.5
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
attn_weights = F.softmax(attn_scores.masked_fill(causal_mask, -inf), dim=-1)

# Sink rate is fraction of heads with mean attention to BOS >= 0.3
bos_attention = attn_weights[:, :, :, 0].mean(dim=2)
sink_rate = (bos_attention >= 0.3).float().mean()
```

For BOS-MLP ablation, we zero the MLP output for the BOS token at the BOS spike layer (layer 1 for RoPE, layer 2 for DroPE).

```python
def bos_mlp_ablation_hook(module, input, output):
    output[0, 0, :] = 0  # Zero BOS token's MLP output
    return output

model.layers[spike_layer].mlp.register_forward_hook(bos_mlp_ablation_hook)
```

### 5.3 Results

**Sink Rates by Layer**

| Layer | RoPE | DroPE |
|-------|------|-------|
| 0 | 66% | 54% |
| 1 | 78% | 31% |
| 2-30 | 97-100% | 94-99% |
| 31 | 92% | 91% |
| **Average** | **97.8%** | **95.6%** |

Both models have high sink rates across nearly all layers, contrary to our hypothesis.

![Figure 12](phase_metrics/fig_sink_rate.png)
*Figure 12: Attention sink rates by layer. Both models converge to ~97% by layer 2.*

**BOS-MLP Ablation**

| Model | Baseline PPL | Ablated PPL | Change |
|-------|--------------|-------------|--------|
| RoPE  | 10.2 | 12,766 | **1249×** |
| DroPE | 18.6 | 18.5 | **1.00×** |

![Figure 13](phase_metrics/fig_interventions.png)
*Figure 13: BOS-MLP ablation results. RoPE perplexity explodes while DroPE is unaffected.*

### 5.4 Additional Phase Metrics

| Metric | RoPE | DroPE |
|--------|------|-------|
| BOS Spike Layer | 1 | 2 |
| Peak BOS Norm | 982 (layer 23) | 519 (layer 26) |
| Min Entropy | 0.008 (layer 1) | 0.129 (layer 2) |

The BOS spike layer is the first layer where the BOS token's norm exceeds other tokens by more than 2×. RoPE spikes at layer 1; DroPE delays this to layer 2.

Peak BOS norm measures the maximum L2 norm the BOS token reaches across all layers. DroPE's peak is 47% lower than RoPE's, suggesting less extreme concentration in the BOS representation.

Minimum entropy identifies the layer with the most compressed representations. RoPE reaches near-zero entropy (0.008) at layer 1, indicating extreme compression. DroPE's minimum is 16× higher (0.129), meaning representations stay more distributed even at their most compressed point.

![Figure 14](phase_metrics/fig_bos_norm.png)
*Figure 14: BOS token norm by layer. Both models show similar profiles, with DroPE peaking slightly later.*

![Figure 15](phase_metrics/fig_entropy.png)
*Figure 15: Representation entropy by layer. DroPE maintains higher entropy throughout.*

### 5.5 Interpretation

The results are striking. Despite nearly identical sink rates, the models respond completely differently to BOS-MLP ablation.

**BOS-MLP Ablation Results:**

| Condition | RoPE Cities | RoPE Sports | RoPE Pass | RoPE IMDB | DroPE Cities | DroPE Sports | DroPE Pass | DroPE IMDB |
|-----------|-------------|-------------|-----------|-----------|--------------|--------------|------------|------------|
| baseline | 99% | 69% | 100% | 50% | 60% | 70% | 30% | 9% |
| bos_mlp_ablation | **0%** | **0%** | **0%** | **0%** | 60% | 71% | 25% | 11% |
| qk_disruption | 58% | 60% | 0% | 2% | 66% | 56% | 0% | 9% |
| combined | 49% | 64% | 0% | 0% | 62% | 56% | 0% | 4% |

RoPE suffers catastrophic failure under BOS-MLP ablation—**0% accuracy on all tasks**. This is not merely wrong answers; the model produces outputs so incoherent that they contain no valid yes/no responses. The model cannot function without BOS-MLP processing, confirming that BOS stores critical position-dependent information essential for coherent generation.

DroPE shows remarkable resilience. Despite 95.6% of heads attending to BOS, performance is virtually unchanged (parametric: 65%→65.5%, contextual: 19.5%→18%). DroPE has learned to make BOS a "garbage collector" that receives attention but stores nothing essential.

Q/K disruption affects both models for passkey retrieval (100%→0% for RoPE, 30%→0% for DroPE), but DroPE's parametric tasks remain stable. This confirms that massive values serve different roles: RoPE uses them for critical information storage, while DroPE uses them as vestigial attention routing.

**Sink rate does not imply functional dependence.** Both models route attention to BOS, but only RoPE stores critical information there. This explains why DroPE's massive values appear vestigial—the model has reorganized to distribute critical information across the sequence rather than concentrating it in specific tokens or dimensions.

![Figure 16](phase_metrics/fig_functional.png)
*Figure 16: Functional evaluation across all 4 intervention conditions for all 4 tasks (Cities, Sports, Passkey, IMDB). Top row: parametric tasks; bottom row: contextual tasks. RoPE collapses completely under BOS-MLP ablation (0% on all tasks—model outputs are incoherent), while DroPE maintains parametric performance (60-71%). Q/K disruption destroys passkey retrieval for both models but leaves DroPE's parametric tasks intact.*

![Figure 17](phase_metrics/fig_phase_summary.png)
*Figure 17: Summary of phase metrics analysis.*

## 6. Experiment 5: BOS Value Analysis

### 6.1 Motivation

Experiment 4 revealed a striking puzzle: both models have ~97% attention sink rates, yet only RoPE depends on BOS-MLP ablation (1249× PPL increase vs 1.00×). If both models attend to BOS equally, why does only RoPE store critical information there?

We hypothesized that DroPE's attention sinks might be "no-ops"—heads that route attention to BOS but don't actually write meaningful content to the residual stream. To test this, we measure the **effective BOS write score**, which combines:
- **BOS attention mass**: How much attention each head pays to BOS
- **BOS V norm**: The L2 norm of the Value vector at BOS position

```
effective_write = attention_to_BOS × V_norm_at_BOS
```

A low write score would indicate sinks that attend but don't write. We also perform **BOS-V ablation**—zeroing the V projection at BOS—to measure functional dependence on BOS value content.

### 6.2 Method

We hook Q, K, and V projections to capture pre-attention representations:

```python
# Capture V norms at BOS position
v_out = layer.self_attn.v_proj(hidden_states)
bos_v = v_out[0, 0, :].view(num_heads, head_dim)
bos_v_norms = bos_v.norm(dim=1)  # [num_heads]

# Compute attention from Q, K manually (DroPE incompatible with output_attentions)
scores = torch.matmul(q, k.T) * scale
attn_weights = F.softmax(scores.masked_fill(causal_mask, -inf), dim=-1)
bos_attn_mass = attn_weights[:, :, 1:, 0].mean()  # attention to BOS from non-BOS tokens
```

**Note**: DroPE layer 1 has massive Q/K activations (±394 vs RoPE's ±5), causing attention scores to overflow. We clamp scores to [-100, 100] before softmax to prevent NaN.

For BOS-V ablation:
```python
def bos_v_ablation_hook(module, input, output):
    output[:, 0, :] = 0  # Zero V at BOS position
    return output
```

### 6.3 Results

**BOS Write Metrics**

| Metric | RoPE | DroPE |
|--------|------|-------|
| Mean BOS V Norm | 0.36 | **0.61** |
| Mean BOS Attention Mass | 71.7% | 69.0% |
| Mean Effective Write Score | 0.22 | **0.34** |
| Max Write Layer | 31 | **1** |

Contrary to our hypothesis, DroPE has **higher** BOS write scores than RoPE (0.34 vs 0.22). DroPE also has higher BOS V norms (0.61 vs 0.36), meaning it writes *more* content to BOS, not less.

The key difference is in **where** the write happens: RoPE's max write is at layer 31 (late), while DroPE's max write is at layer 1—the same layer where massive activations concentrate.

![Figure 18](phase_metrics/fig_bos_write_analysis.png)
*Figure 18: BOS write analysis. Top left: V norms per layer. Top right: attention mass per layer. Bottom left: effective write score per layer. Bottom right: BOS-V ablation effects.*

**BOS-V Ablation**

| Condition | RoPE | DroPE |
|-----------|------|-------|
| Baseline PPL | 10.3 | 17.7 |
| BOS-V ablation (spike layer) | 0.93× | 1.02× |
| BOS-V ablation (all layers) | **4.12×** | **3.39×** |

Both models show similar sensitivity to BOS-V ablation (~4× PPL increase when ablating all layers). This contrasts sharply with BOS-MLP ablation, which affects RoPE 1249× more than DroPE.

### 6.4 Interpretation

The results refute the "no-op sink" hypothesis. DroPE's attention sinks do write meaningful content to the residual stream—more than RoPE, in fact. Both models depend similarly on BOS value content (BOS-V ablation: ~4× degradation).

The key difference lies in **MLP processing**, not attention routing or V content:

| Ablation | RoPE Effect | DroPE Effect |
|----------|-------------|--------------|
| BOS-V (all layers) | 4.12× | 3.39× |
| BOS-MLP (spike layer) | 1249× | 1.00× |

RoPE's BOS-MLP stores critical position-dependent information that cannot be recovered from other sources. DroPE's BOS-MLP, despite receiving similar V content, does not encode essential information—the model has learned to distribute critical computations elsewhere.

### 6.5 The Massive Activation Connection

DroPE layer 1 shows massive Q/K activations (±394) that cause attention score overflow. This is the same layer where:
- DroPE has its max BOS write score (1.14)
- DroPE reorganizes massive values (37× more than RoPE at layer 1)
- DroPE delays its BOS spike (layer 2 vs layer 1)

The massive activations at layer 1 may serve as a positional substitute—extreme values that create strong, stable attention patterns without requiring RoPE's rotary embeddings. The model concentrates this processing early, leaving later layers free for content-based attention.

## 7. Experiment 6: Layer 1 Ablation Study

### 7.1 Motivation

DroPE Layer 1 shows three unusual properties:
- 37× more massive values than RoPE (101.3 vs 2.7)
- Maximum BOS write score (1.14)
- Extreme Q/K activations (±394 vs RoPE's ±5)

What function does this concentrated Layer 1 processing serve? Is it a "positional substitute" mechanism?

### 7.2 Method

We ablate Layer 1 components and measure functional impact:

| Condition | Description |
|-----------|-------------|
| `baseline` | No ablation |
| `layer1_mlp` | Zero MLP output at layer 1 |
| `layer1_attn` | Zero attention output at layer 1 |
| `layer1_both` | Zero both MLP and attention at layer 1 |
| `layer1_bos_only` | Zero only BOS token's layer 1 outputs |

### 7.3 Results

**Perplexity Impact**

| Condition | RoPE PPL | DroPE PPL | RoPE Ratio | DroPE Ratio |
|-----------|----------|-----------|------------|-------------|
| baseline | 10.26 | 17.65 | 1.0× | 1.0× |
| layer1_mlp | 18,630 | 3,563 | **1815×** | **202×** |
| layer1_attn | 25.77 | 3,557 | **2.5×** | **201×** |
| layer1_both | 13,824 | 3,603 | 1347× | 204× |
| layer1_bos_only | 14,422 | 3,018 | 1405× | 171× |

**Task Accuracy**

| Condition | RoPE Cities | RoPE Sports | RoPE Pass | RoPE IMDB | DroPE Cities | DroPE Sports | DroPE Pass | DroPE IMDB |
|-----------|-------------|-------------|-----------|-----------|--------------|--------------|------------|------------|
| baseline | 99% | 69% | **100%** | 50% | 60% | 70% | 30% | 9% |
| layer1_mlp | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| layer1_attn | **60%** | **70%** | 0% | 5% | 0% | 0% | 0% | 0% |
| layer1_both | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| layer1_bos_only | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

### 7.4 Key Findings

1. **RoPE Layer 1 MLP is 725× more critical than attention** (1815× vs 2.5× PPL increase)
2. **DroPE shows uniform criticality** — both MLP and attention are equally important (~201×)
3. **RoPE attention ablation preserves parametric tasks** (60% cities, 70% sports) but destroys passkey (0%)
4. **DroPE Layer 1 is uniformly critical** — any ablation destroys all tasks
5. **DroPE is 9× more resilient overall** to Layer 1 MLP ablation (202× vs 1815×)

### 7.5 Interpretation

The dramatic difference in Layer 1 architecture explains the "positional substitute" hypothesis:

**RoPE**: Layer 1 MLP stores critical position-dependent information. Attention at Layer 1 is relatively unimportant (2.5× vs 1815×). The model concentrates positional processing in the MLP.

**DroPE**: Without RoPE, the model redistributes positional processing across both MLP and attention at Layer 1. Both components become equally critical (~200×), but the total criticality is 9× lower than RoPE's MLP alone.

This supports the hypothesis that DroPE's 37× increase in Layer 1 massive values serves as an alternative positional encoding mechanism, distributed across both MLP and attention rather than concentrated in MLP alone.

## 8. Experiment 7: Attention Pattern Analysis

### 8.1 Motivation

Both models have ~97% sink rates but respond completely differently to ablations. What do the actual attention patterns look like beyond sink rates?

### 8.2 Method

We analyze attention patterns across 8 diverse texts:
- Compute attention entropy per head (focused vs distributed)
- Classify heads as sink (BOS attention ≥ 0.3), local, or distributed
- Measure attention decay with distance
- Compare Layer 1 patterns specifically

### 8.3 Results

**Attention Pattern Comparison**

| Metric | RoPE | DroPE |
|--------|------|-------|
| Mean BOS Attention | 65.1% | 67.5% |
| Mean Local Attention | 29.7% | 26.4% |
| **Sink heads** | **93.5%** | **93.1%** |
| Local heads | 6.4% | 6.2% |

**Layer 1 Attention**

| Metric | RoPE | DroPE |
|--------|------|-------|
| BOS Attention | 16.9% | 19.6% |
| Local Attention | 40.0% | 39.4% |

![Figure 19](phase_metrics/fig_attention_summary.png)
*Figure 19: Attention pattern analysis summary.*

![Figure 20](phase_metrics/fig_layer1_comparison.png)
*Figure 20: Detailed Layer 1 attention comparison.*

### 8.4 Key Finding

**Attention patterns are nearly identical** (~93% sink heads in both models), yet **Layer 1 functional importance is completely different**:
- RoPE: MLP 725× more critical than attention
- DroPE: Equal criticality (~200× each)

This demonstrates that **attention pattern similarity does not imply functional equivalence**. The models route attention identically but process it completely differently.

## 9. Experiment 8: Layer 1 Attention Content Analysis

### 9.1 Motivation

Experiment 6 revealed a striking asymmetry: RoPE Layer 1 MLP is 725× more critical than attention (1815× vs 2.5×), while DroPE shows equal criticality (~200× each). Why is DroPE's Layer 1 attention suddenly critical when RoPE's is not?

### 9.2 Method

We capture Layer 1 attention and MLP outputs to analyze what each component actually writes to the residual stream:

```python
class Layer1ContentCapture:
    """Capture Layer 1 outputs to analyze residual stream contributions."""

    def __init__(self, model):
        # Hook attention output (after o_proj)
        # Hook MLP output
        # Capture hidden states before and after layer 1
```

Metrics:
- **Attention output norm**: L2 norm of attention output tensor
- **MLP output norm**: L2 norm of MLP output tensor
- **Contribution ratio**: Relative magnitude of attention vs MLP to residual stream
- **Q/K norms**: Magnitude of query and key projections

### 9.3 Results

**Residual Stream Contributions**

| Metric | RoPE | DroPE | Ratio |
|--------|------|-------|-------|
| Attention output norm | 1.52 | **79.19** | **52×** |
| MLP output norm | 176.30 | 36.27 | 0.2× |
| **Attention contribution** | **0.9%** | **68.8%** | **76×** |
| **MLP contribution** | **99.1%** | **31.2%** | 0.3× |

**Q/K Activation Norms**

| Metric | RoPE | DroPE | Ratio |
|--------|------|-------|-------|
| Q norm | 45.5 | **6,586** | **145×** |
| K norm | 52.0 | **5,514** | **106×** |

### 9.4 Key Finding

**DroPE has completely inverted the attention/MLP balance at Layer 1.**

- **RoPE**: Attention contributes only 0.9% to the residual stream; MLP dominates at 99.1%
- **DroPE**: Attention contributes 68.8% to the residual stream; MLP is only 31.2%

This explains the ablation asymmetry from Experiment 6:
- RoPE attention ablation (2.5× PPL) has minimal impact because attention contributes only 0.9%
- DroPE attention ablation (201× PPL) is catastrophic because attention contributes 68.8%

### 9.5 The Q/K Magnitude Connection

DroPE's massive Q/K norms (145× and 106× larger than RoPE) directly cause the increased attention output. The attention mechanism:

```
attention_output = softmax(Q @ K.T / sqrt(d)) @ V @ W_o
```

With Q/K norms ~100× larger, the attention scores before softmax are much larger, creating sharper attention patterns. The resulting attention output has 52× the magnitude of RoPE's.

### 9.6 Interpretation

DroPE compensates for the removal of positional embeddings by:

1. **Amplifying Q/K projections** at Layer 1 (~100× larger norms)
2. **Shifting processing from MLP to attention** (0.9% → 68.8% contribution)
3. **Creating strong initial attention patterns** via massive activations

This is the mechanistic explanation for the "positional substitute" hypothesis: without RoPE's positional encoding, DroPE uses extreme Q/K values at Layer 1 to establish position-independent attention routing that serves a similar organizational function.

## 10. Discussion

### 10.1 Summary of Findings

1. Massive values are encoded in projection weights during RoPE training
2. DroPE recalibration reduces concentration (−39% Query, −11% Key) but reorganizes Layer 1
3. RoPE models cannot function without massive values; DroPE models degrade but remain usable
4. RoPE follows Jin et al.'s pattern: contextual knowledge (94.3% degradation) affected 3.8× more than parametric (24.5%)
5. DroPE shows dramatically different behavior:
   - Parametric accuracy stable or improved under BOS-MLP ablation (65%→65.5%)
   - Contextual degradation minimal under BOS-MLP ablation (19.5%→18%)
   - Q/K disruption affects passkey retrieval similarly to RoPE (both drop to 0%)
6. **Both models have ~97% attention sink rates, but only RoPE depends on BOS-MLP** (1249× PPL increase vs 1.00×)
7. **BOS-MLP ablation = 0% accuracy on ALL tasks for RoPE** vs maintained performance for DroPE
8. **DroPE has higher BOS write scores (0.34 vs 0.22)** but doesn't depend on BOS-MLP—the critical difference is in MLP processing, not attention routing
9. **DroPE concentrates BOS writes at layer 1** (max write layer), where massive activations are 37× higher than RoPE
10. **Layer 1 MLP ablation is 725× more critical for RoPE than Layer 1 attention** (1815× vs 2.5×)
11. **DroPE shows uniform Layer 1 criticality** — both MLP and attention equally important (~200×)
12. **Attention patterns are nearly identical** (~93% sink heads) yet functional importance differs completely
13. **DroPE inverts Layer 1 attention/MLP balance** — attention contributes 68.8% vs RoPE's 0.9%
14. **DroPE Q/K norms are 100× larger at Layer 1** (6586/5514 vs 45/52)

### 10.2 Implications for Context Extension

DroPE enables longer contexts. Our findings suggest a mechanism:

1. RoPE concentrates attention in specific dimensions via massive values
2. This concentration may create bottlenecks at long contexts
3. DroPE distributes attention more evenly, potentially enabling better generalization to longer sequences

## 11. Reproducibility

```bash
python scripts/run_llama_comparison.py      # Experiment 1
python scripts/run_disruption_rigorous.py   # Experiment 2
python scripts/finish_jin_tests.py          # Experiment 3 (Jin et al. replication)
python scripts/run_phase_metrics.py         # Experiment 4 (phase metrics)
python scripts/rerun_drope_metrics.py       # Experiment 4 (fix DroPE entropy)
python scripts/fix_drope_sink_rates.py      # Experiment 4 (fix DroPE sink rates)
python scripts/run_functional_tests.py      # Experiment 4 (functional tests)
python scripts/run_bos_write_analysis.py    # Experiment 5 (BOS value analysis)
python scripts/run_layer1_ablation.py       # Experiment 6 (Layer 1 ablation)
python scripts/run_attention_analysis.py    # Experiment 7 (attention patterns)
python scripts/run_layer1_content_analysis.py # Experiment 8 (Layer 1 content)
python scripts/create_phase_figures.py      # Phase figures
python scripts/create_bos_write_figures.py  # BOS write figures
python scripts/create_attention_figures.py  # Attention figures
python scripts/create_findings_figures.py   # Main figures
```

Hardware: NVIDIA A10G (24GB), 4-bit quantization (NF4)

| Parameter | Value |
|-----------|-------|
| λ threshold | 5.0 |
| Sequence length | 512 |
| Text samples | 10 |
| Random seeds | 10 |

## 12. Summary

| Metric | RoPE | DroPE |
|--------|------|-------|
| Query massive values | 1476 ± 23 | 901 ± 36 |
| Key massive values | 1497 ± 70 | 1332 ± 74 |
| PPL increase (Q/K disrupted) | +115,929% | +1,421% |
| Parametric avg (baseline) | 84% | 65% |
| Contextual avg (baseline) | 75% | 19.5% |
| Attention sink rate | 97.8% | 95.6% |
| BOS-MLP ablation PPL | **1249×** | **1.00×** |
| BOS-MLP ablation accuracy | **0% (all tasks)** | **Maintained** |
| Q/K disruption passkey | 0% | 0% |
| Functional after BOS ablation | **No** | **Yes** |
| BOS V norm (mean) | 0.36 | **0.61** |
| BOS effective write score | 0.22 | **0.34** |
| BOS-V ablation (all layers) | 4.12× | 3.39× |
| Max write layer | 31 | **1** |
| **Layer 1 MLP ablation PPL** | **1815×** | **202×** |
| **Layer 1 Attn ablation PPL** | **2.5×** | **201×** |
| **Layer 1 MLP/Attn ratio** | **725:1** | **1:1** |
| Sink heads (attention) | 93.5% | 93.1% |
| **Layer 1 Attn contribution** | **0.9%** | **68.8%** |
| **Layer 1 Q norm** | 45.5 | **6,586** |
| **Layer 1 K norm** | 52.0 | **5,514** |

![Figure 5](findings_figures/fig5_combined_summary.png)
*Figure 5: Summary of both experiments.*

## 13. Citation

```bibtex
@techreport{africa2026massive,
  title   = {Massive Activations in DroPE: Evidence for Attention Reorganization},
  author  = {Africa, David},
  year    = {2026},
  url     = {https://github.com/DavidDemitriAfrica/drope-activations}
}
```

## 14. References

Jin, M., Sun, K., et al. (2025). Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding. ICML.

Gelberg, T., et al. (2025). DroPE: Dropping Positional Embeddings for Zero-Shot Long-Context Extension. arXiv:2512.12167.

Queipo-de-Llano, A., et al. (2025). Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin. arXiv:2510.06477.
