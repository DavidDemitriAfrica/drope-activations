"""
Convert RoPE models to DroPE by removing positional embeddings.

Based on the DroPE paper (Gelberg et al., 2025).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel
import copy


def remove_rope_from_layer(
    attention_module: nn.Module,
    model_type: str = "llama",
) -> nn.Module:
    """
    Remove RoPE from a single attention layer.

    This modifies the attention computation to skip rotary position embedding
    application to Q and K tensors.

    Args:
        attention_module: The attention module to modify
        model_type: Type of model architecture

    Returns:
        Modified attention module (modified in-place)
    """
    # Store original forward method
    original_forward = attention_module.forward

    def forward_without_rope(
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Forward pass that skips RoPE application."""
        input_shape = hidden_states.shape[:-1]

        # Get head_dim from module
        head_dim = attention_module.head_dim
        hidden_shape = (*input_shape, -1, head_dim)

        # Project to Q, K, V and reshape
        query_states = attention_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attention_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attention_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # NOTE: We skip the RoPE application here - this is the key change
        # Original code would do:
        # cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache if present (new Cache API)
        if past_key_values is not None:
            # For static cache, we still need cos/sin even though we don't use them
            cos, sin = position_embeddings if position_embeddings is not None else (None, None)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, attention_module.layer_idx, cache_kwargs)

        # Use eager attention (simplified for DroPE conversion)
        num_heads = query_states.shape[1]
        num_kv_heads = key_states.shape[1]

        # Repeat KV for GQA if needed
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        # Compute attention with scaling
        scaling = getattr(attention_module, 'scaling', head_dim ** -0.5)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attention_module.o_proj(attn_output)

        # Return format matches new transformers API: (attn_output, attn_weights)
        return attn_output, attn_weights

    # Replace forward method
    attention_module.forward = forward_without_rope
    attention_module._drope_converted = True

    return attention_module


def convert_rope_to_drope(
    model: PreTrainedModel,
    copy_model: bool = True,
) -> PreTrainedModel:
    """
    Convert a RoPE model to DroPE by removing positional embeddings.

    Args:
        model: The pretrained model with RoPE
        copy_model: Whether to create a copy (True) or modify in-place (False)

    Returns:
        Model with RoPE removed from all attention layers
    """
    if copy_model:
        model = copy.deepcopy(model)

    # Detect model type
    model_name = model.__class__.__name__.lower()
    if "llama" in model_name:
        model_type = "llama"
    elif "qwen" in model_name:
        model_type = "qwen"
    elif "mistral" in model_name:
        model_type = "mistral"
    else:
        model_type = "llama"  # Default to llama-style

    # Get transformer layers
    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    if hasattr(base, "layers"):
        transformer_layers = base.layers
    elif hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        transformer_layers = base.decoder.layers
    else:
        raise ValueError(f"Cannot find transformer layers in {type(model)}")

    # Remove RoPE from each layer
    for layer in transformer_layers:
        if hasattr(layer, "self_attn"):
            remove_rope_from_layer(layer.self_attn, model_type)
        elif hasattr(layer, "attention"):
            remove_rope_from_layer(layer.attention, model_type)

    # Mark model as converted
    model.config._drope_converted = True

    return model


def is_drope_model(model: PreTrainedModel) -> bool:
    """Check if a model has been converted to DroPE."""
    return getattr(model.config, "_drope_converted", False)
