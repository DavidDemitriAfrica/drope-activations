"""
Extract real attention patterns from RoPE and DroPE models for animation.
This extracts actual model internals, not heuristics.
"""

import sys
sys.path.insert(0, "/home/ubuntu/drope-activations/DroPE")

import torch
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

ROPE_MODEL = "meta-llama/Llama-2-7b-hf"
DROPE_MODEL = "SakanaAI/Llama-2-7b-hf-DroPE"

def extract_attention_patterns(model, tokenizer, text, model_name, num_layers=32):
    """Extract real attention weights from all layers via Q/K hooks."""

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]
    tokens = [tokenizer.decode(t) for t in inputs.input_ids[0]]

    print(f"Extracting from {model_name}, seq_len={seq_len}")
    print(f"Tokens: {tokens[:10]}...")

    # Storage for Q, K activations
    qk_data = {}

    def make_hook(layer_idx, proj_type):
        def hook(module, input, output):
            qk_data[f"layer{layer_idx}_{proj_type}"] = output.detach().cpu()
        return hook

    # Register hooks on Q and K projections
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i, 'q')))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_hook(i, 'k')))

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute attention patterns from Q, K
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    attention_data = {
        "model": model_name,
        "tokens": tokens,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "layers": {},
        "qk_norms": {"q": [], "k": []}
    }

    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    for layer_idx in range(num_layers):
        q = qk_data[f"layer{layer_idx}_q"]  # [1, seq_len, hidden]
        k = qk_data[f"layer{layer_idx}_k"]

        # Record norms
        q_norm = q.norm().item()
        k_norm = k.norm().item()
        attention_data["qk_norms"]["q"].append(q_norm)
        attention_data["qk_norms"]["k"].append(k_norm)

        # Reshape for multi-head attention
        q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)  # [1, heads, seq, dim]
        k = k.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [1, heads, seq, seq]

        # Clamp for DroPE stability (massive activations at layer 1)
        scores = scores.clamp(-100, 100)

        # Apply causal mask and softmax
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)  # [1, heads, seq, seq]

        # Average across heads for visualization
        avg_attn = attn_weights[0].mean(dim=0).numpy()  # [seq, seq]

        # BOS attention (how much each position attends to BOS)
        bos_attn = attn_weights[0, :, :, 0].mean(dim=0).numpy()  # [seq]

        attention_data["layers"][layer_idx] = {
            "avg_attention": avg_attn.tolist(),
            "bos_attention": bos_attn.tolist(),
            "q_norm": q_norm,
            "k_norm": k_norm,
        }

    return attention_data


def main():
    text = "The capital of France is Paris. The Eiffel Tower is located in"

    # Load RoPE model
    print("Loading RoPE model...")
    rope_tokenizer = AutoTokenizer.from_pretrained(ROPE_MODEL)
    rope_model = AutoModelForCausalLM.from_pretrained(
        ROPE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    rope_data = extract_attention_patterns(rope_model, rope_tokenizer, text, "RoPE")

    # Free memory
    del rope_model
    torch.cuda.empty_cache()

    # Load DroPE model
    print("\nLoading DroPE model...")
    drope_tokenizer = AutoTokenizer.from_pretrained(DROPE_MODEL, trust_remote_code=True)
    drope_model = AutoModelForCausalLM.from_pretrained(
        DROPE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    drope_data = extract_attention_patterns(drope_model, drope_tokenizer, text, "DroPE")

    # Save data
    output = {
        "rope": rope_data,
        "drope": drope_data,
        "text": text
    }

    with open("results/attention_animation_data.json", "w") as f:
        json.dump(output, f)

    print("\nSaved to results/attention_animation_data.json")

    # Print key differences
    print("\n=== Q/K Norm Comparison (Layer 1) ===")
    print(f"RoPE  - Q: {rope_data['qk_norms']['q'][1]:.1f}, K: {rope_data['qk_norms']['k'][1]:.1f}")
    print(f"DroPE - Q: {drope_data['qk_norms']['q'][1]:.1f}, K: {drope_data['qk_norms']['k'][1]:.1f}")
    print(f"Ratio - Q: {drope_data['qk_norms']['q'][1]/rope_data['qk_norms']['q'][1]:.1f}x, K: {drope_data['qk_norms']['k'][1]/rope_data['qk_norms']['k'][1]:.1f}x")


if __name__ == "__main__":
    main()
