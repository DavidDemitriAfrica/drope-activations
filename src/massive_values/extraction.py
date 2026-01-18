"""
Extract Q, K, V tensors from transformer models.

This module provides utilities to hook into attention layers and extract
the query, key, and value tensors for massive value analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import PreTrainedModel


@dataclass
class QKVTensors:
    """Container for Q, K, V tensors from a single layer."""
    query: torch.Tensor  # [batch, seq_len, num_heads, head_dim]
    key: torch.Tensor    # [batch, seq_len, num_heads, head_dim]
    value: torch.Tensor  # [batch, seq_len, num_heads, head_dim]
    layer_idx: int


class QKVExtractor:
    """
    Hook-based extractor for Q, K, V tensors from transformer attention layers.

    Supports Llama, Qwen, Mistral, and other HuggingFace transformer architectures.
    Uses hooks on Q/K/V projection layers to capture tensors reliably.
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.captured_outputs: Dict[str, torch.Tensor] = {}
        self._model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """Detect the model architecture type."""
        model_name = self.model.__class__.__name__.lower()
        if "llama" in model_name:
            return "llama"
        elif "qwen" in model_name:
            return "qwen"
        elif "mistral" in model_name:
            return "mistral"
        elif "gemma" in model_name:
            return "gemma"
        else:
            # Default to llama-style architecture
            return "llama"

    def _get_transformer_layers(self) -> nn.ModuleList:
        """Get the transformer layers from the model."""
        # Handle different model architectures
        if hasattr(self.model, "model"):
            base = self.model.model
        else:
            base = self.model

        if hasattr(base, "layers"):
            return base.layers
        elif hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
            return base.decoder.layers
        else:
            raise ValueError(f"Cannot find transformer layers in {type(self.model)}")

    def _get_attention_module(self, layer: nn.Module) -> nn.Module:
        """Get the attention module from a transformer layer."""
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        elif hasattr(layer, "attention"):
            return layer.attention
        else:
            raise ValueError(f"Cannot find attention module in {type(layer)}")

    def _create_projection_hook(self, key: str):
        """Create a hook to capture projection output."""
        def hook(module, input, output):
            self.captured_outputs[key] = output.detach().clone()
        return hook

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """
        Register forward hooks on Q, K, V projection layers.

        Args:
            layer_indices: Specific layers to hook. If None, hooks all layers.
        """
        self.remove_hooks()
        self.captured_outputs.clear()

        transformer_layers = self._get_transformer_layers()

        for idx, layer in enumerate(transformer_layers):
            if layer_indices is not None and idx not in layer_indices:
                continue

            attn = self._get_attention_module(layer)

            # Hook Q, K, V projections
            if hasattr(attn, "q_proj"):
                handle_q = attn.q_proj.register_forward_hook(
                    self._create_projection_hook(f"layer_{idx}_q")
                )
                handle_k = attn.k_proj.register_forward_hook(
                    self._create_projection_hook(f"layer_{idx}_k")
                )
                handle_v = attn.v_proj.register_forward_hook(
                    self._create_projection_hook(f"layer_{idx}_v")
                )
                self.hooks.extend([handle_q, handle_k, handle_v])
            elif hasattr(attn, "qkv_proj"):
                # Fused QKV - would need different handling
                raise NotImplementedError("Fused QKV projection not yet supported")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_captured_qkv(self) -> Dict[int, QKVTensors]:
        """
        Return the captured Q, K, V tensors organized by layer.
        """
        result = {}

        # Get model config for reshaping
        transformer_layers = self._get_transformer_layers()

        # Find all captured layers
        layer_indices = set()
        for key in self.captured_outputs:
            parts = key.split("_")
            layer_idx = int(parts[1])
            layer_indices.add(layer_idx)

        for layer_idx in sorted(layer_indices):
            q_key = f"layer_{layer_idx}_q"
            k_key = f"layer_{layer_idx}_k"
            v_key = f"layer_{layer_idx}_v"

            if q_key not in self.captured_outputs:
                continue

            q = self.captured_outputs[q_key]
            k = self.captured_outputs[k_key]
            v = self.captured_outputs[v_key]

            # Get attention module for config
            attn = self._get_attention_module(transformer_layers[layer_idx])

            # Get head configuration
            num_heads = getattr(attn, "num_heads", None)
            if num_heads is None:
                num_heads = getattr(attn.config, "num_attention_heads", None) if hasattr(attn, "config") else None
            if num_heads is None:
                # Try to infer from model config
                if hasattr(self.model, "config"):
                    num_heads = self.model.config.num_attention_heads

            num_kv_heads = getattr(attn, "num_key_value_heads", None)
            if num_kv_heads is None and hasattr(self.model, "config"):
                num_kv_heads = getattr(self.model.config, "num_key_value_heads", num_heads)
            if num_kv_heads is None:
                num_kv_heads = num_heads

            # Reshape: [batch, seq_len, hidden] -> [batch, seq_len, num_heads, head_dim]
            batch_size, seq_len, _ = q.shape

            # Calculate head dimensions from actual tensor sizes
            head_dim = q.shape[-1] // num_heads
            kv_head_dim = k.shape[-1] // num_kv_heads

            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, num_kv_heads, kv_head_dim)
            v = v.view(batch_size, seq_len, num_kv_heads, kv_head_dim)

            result[layer_idx] = QKVTensors(
                query=q,
                key=k,
                value=v,
                layer_idx=layer_idx,
            )

        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def extract_qkv_tensors(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    layer_idx: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> QKVTensors:
    """
    Extract Q, K, V tensors from a specific layer.

    Args:
        model: The transformer model
        input_ids: Input token IDs [batch, seq_len]
        layer_idx: Which layer to extract from
        attention_mask: Optional attention mask

    Returns:
        QKVTensors containing Q, K, V each of shape [batch, seq_len, num_heads, head_dim]
    """
    extractor = QKVExtractor(model)
    extractor.register_hooks(layer_indices=[layer_idx])

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    qkv = extractor.get_captured_qkv()
    extractor.remove_hooks()

    if layer_idx not in qkv:
        raise RuntimeError(f"Failed to capture QKV tensors from layer {layer_idx}")

    return qkv[layer_idx]


def extract_qkv_all_layers(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[int, QKVTensors]:
    """
    Extract Q, K, V tensors from all layers.

    Args:
        model: The transformer model
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Optional attention mask

    Returns:
        Dictionary mapping layer index to QKVTensors
    """
    extractor = QKVExtractor(model)
    extractor.register_hooks()

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    qkv = extractor.get_captured_qkv()
    extractor.remove_hooks()

    return qkv
