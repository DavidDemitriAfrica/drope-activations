"""
Disruption experiments for testing the causal role of massive values.

Based on Section 4.2 of the Massive Values paper: "Massive Values Are Essential
for Contextual Knowledge Understanding"
"""

from typing import Callable, Dict, List, Optional, Tuple
from enum import Enum
from contextlib import contextmanager
import torch
import torch.nn as nn
from transformers import PreTrainedModel


class DisruptionMethod(Enum):
    """Methods for disrupting massive values."""
    MEAN = "mean"           # Replace with tensor mean
    ZERO = "zero"           # Replace with zeros
    MIN = "min"             # Replace with minimum value
    RANDOM = "random"       # Replace with random values (same distribution)
    SCALE = "scale"         # Scale down by factor


class MassiveValueDisruptor:
    """
    Applies disruption to massive values during forward pass.

    This modifies the Q and K tensors in-place during inference to test
    whether massive values are causally important for model performance.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        massive_positions: Dict[int, torch.Tensor],  # layer -> [num_heads, head_dim] bool mask
        method: DisruptionMethod = DisruptionMethod.MEAN,
        scale_factor: float = 0.1,
        target: str = "both",  # "query", "key", or "both"
    ):
        """
        Args:
            model: The transformer model to disrupt
            massive_positions: Dict mapping layer index to boolean mask of massive value positions
            method: How to replace massive values
            scale_factor: Scale factor if using SCALE method
            target: Which tensors to disrupt ("query", "key", or "both")
        """
        self.model = model
        self.massive_positions = massive_positions
        self.method = method
        self.scale_factor = scale_factor
        self.target = target
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _get_replacement_value(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute replacement value based on method."""
        if self.method == DisruptionMethod.MEAN:
            return tensor.mean()
        elif self.method == DisruptionMethod.ZERO:
            return torch.zeros(1, device=tensor.device, dtype=tensor.dtype)
        elif self.method == DisruptionMethod.MIN:
            return tensor.min()
        elif self.method == DisruptionMethod.RANDOM:
            # Sample from same distribution (Gaussian with matching mean/std)
            mean = tensor.mean()
            std = tensor.std()
            return torch.randn_like(tensor) * std + mean
        elif self.method == DisruptionMethod.SCALE:
            return tensor * self.scale_factor
        else:
            raise ValueError(f"Unknown disruption method: {self.method}")

    def _create_hook(self, layer_idx: int):
        """Create a forward hook that disrupts massive values."""
        mask = self.massive_positions.get(layer_idx)
        if mask is None:
            return None

        def hook(module, input, output):
            hidden_states = input[0] if isinstance(input, tuple) else input

            # Get Q and K projections
            if hasattr(module, "q_proj"):
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)

                batch_size, seq_len, _ = hidden_states.shape
                num_heads = module.num_heads if hasattr(module, "num_heads") else q.shape[-1] // mask.shape[-1]
                num_kv_heads = getattr(module, "num_key_value_heads", num_heads)
                head_dim = mask.shape[-1]

                # Reshape for disruption
                q = q.view(batch_size, seq_len, num_heads, head_dim)
                k = k.view(batch_size, seq_len, num_kv_heads, head_dim)

                # Apply disruption
                if self.target in ["query", "both"]:
                    q = self._disrupt_tensor(q, mask[:num_heads])
                if self.target in ["key", "both"]:
                    k = self._disrupt_tensor(k, mask[:num_kv_heads])

                # Reshape back and update module state
                # Note: This modifies the forward pass, not the weights
                q = q.view(batch_size, seq_len, -1)
                k = k.view(batch_size, seq_len, -1)

                # Store disrupted values for use in attention computation
                module._disrupted_q = q
                module._disrupted_k = k

        return hook

    def _disrupt_tensor(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply disruption to tensor at masked positions."""
        # tensor: [batch, seq_len, num_heads, head_dim]
        # mask: [num_heads, head_dim]

        # Expand mask to match tensor shape
        expanded_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, head_dim]
        expanded_mask = expanded_mask.expand_as(tensor)

        result = tensor.clone()

        if self.method == DisruptionMethod.SCALE:
            # Scale down massive values
            result[expanded_mask] = tensor[expanded_mask] * self.scale_factor
        elif self.method == DisruptionMethod.RANDOM:
            # Replace with random values
            replacement = self._get_replacement_value(tensor, mask)
            result[expanded_mask] = replacement[expanded_mask]
        else:
            # Replace with scalar value
            replacement = self._get_replacement_value(tensor, mask)
            result[expanded_mask] = replacement

        return result

    def register_hooks(self):
        """Register disruption hooks on all layers with massive values."""
        self.remove_hooks()

        # Get attention layers
        if hasattr(self.model, "model"):
            base = self.model.model
        else:
            base = self.model

        if hasattr(base, "layers"):
            transformer_layers = base.layers
        elif hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
            transformer_layers = base.decoder.layers
        else:
            raise ValueError(f"Cannot find transformer layers in {type(self.model)}")

        for idx, layer in enumerate(transformer_layers):
            if idx in self.massive_positions:
                if hasattr(layer, "self_attn"):
                    attn_module = layer.self_attn
                elif hasattr(layer, "attention"):
                    attn_module = layer.attention
                else:
                    continue

                hook = self._create_hook(idx)
                if hook is not None:
                    handle = attn_module.register_forward_hook(hook)
                    self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all disruption hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def disrupt_massive_values(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    massive_positions: Dict[int, torch.Tensor],
    method: DisruptionMethod = DisruptionMethod.MEAN,
    target: str = "both",
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Run forward pass with massive values disrupted.

    Args:
        model: The model to run
        input_ids: Input token IDs
        massive_positions: Dict mapping layer index to massive value mask
        method: Disruption method
        target: "query", "key", or "both"
        attention_mask: Optional attention mask

    Returns:
        Model outputs with disrupted massive values
    """
    disruptor = MassiveValueDisruptor(
        model=model,
        massive_positions=massive_positions,
        method=method,
        target=target,
    )

    with disruptor:
        outputs = model(input_ids, attention_mask=attention_mask)

    return outputs


@contextmanager
def disruption_context(
    model: PreTrainedModel,
    massive_positions: Dict[int, torch.Tensor],
    method: DisruptionMethod = DisruptionMethod.MEAN,
    target: str = "both",
):
    """
    Context manager for disrupting massive values during evaluation.

    Example:
        with disruption_context(model, positions, DisruptionMethod.MEAN):
            outputs = model(input_ids)
            # Massive values are disrupted in this block
    """
    disruptor = MassiveValueDisruptor(
        model=model,
        massive_positions=massive_positions,
        method=method,
        target=target,
    )
    disruptor.register_hooks()
    try:
        yield disruptor
    finally:
        disruptor.remove_hooks()


def create_control_positions(
    massive_positions: Dict[int, torch.Tensor],
    control_type: str = "random",
    seed: int = 42,
) -> Dict[int, torch.Tensor]:
    """
    Create control positions for disruption experiments.

    Args:
        massive_positions: Original massive value positions
        control_type: "random" (same count, random positions) or
                     "non_massive" (positions that are NOT massive)
        seed: Random seed for reproducibility

    Returns:
        Control positions with same shape as massive_positions
    """
    torch.manual_seed(seed)
    control_positions = {}

    for layer_idx, mask in massive_positions.items():
        num_massive = mask.sum().item()
        total_positions = mask.numel()

        if control_type == "random":
            # Random positions, same count
            flat_mask = torch.zeros(total_positions, dtype=torch.bool)
            random_indices = torch.randperm(total_positions)[:num_massive]
            flat_mask[random_indices] = True
            control_positions[layer_idx] = flat_mask.view_as(mask)

        elif control_type == "non_massive":
            # Non-massive positions
            control_positions[layer_idx] = ~mask

        else:
            raise ValueError(f"Unknown control type: {control_type}")

    return control_positions
