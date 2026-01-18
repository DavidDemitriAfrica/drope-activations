"""
Utilities for loading models for massive value analysis.
"""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)


# Known DroPE models from the paper
DROPE_MODELS = {
    "smollm-360m-drope": "SakanaAI/SmolLM-360M-DroPE",
    "smollm-1.7b-drope": "SakanaAI/SmolLM-1.7B-DroPE",
    "llama2-7b-drope": "SakanaAI/Llama-2-7B-DroPE",
}

# RoPE baselines
ROPE_BASELINES = {
    "smollm-360m": "HuggingFaceTB/SmolLM-360M",
    "smollm-1.7b": "HuggingFaceTB/SmolLM-1.7B",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}


def get_model_config(model_name: str) -> Dict[str, any]:
    """
    Get configuration for a model by name.

    Args:
        model_name: Short name or HF path

    Returns:
        Dict with model_path, is_drope, base_model (for DroPE models)
    """
    model_name_lower = model_name.lower()

    # Check DroPE models
    if model_name_lower in DROPE_MODELS:
        return {
            "model_path": DROPE_MODELS[model_name_lower],
            "is_drope": True,
            "base_model": model_name_lower.replace("-drope", ""),
        }

    # Check RoPE baselines
    if model_name_lower in ROPE_BASELINES:
        return {
            "model_path": ROPE_BASELINES[model_name_lower],
            "is_drope": False,
            "base_model": None,
        }

    # Assume it's a HF path
    return {
        "model_path": model_name,
        "is_drope": "drope" in model_name_lower,
        "base_model": None,
    }


def load_model(
    model_name_or_path: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer.

    Args:
        model_name_or_path: Model name (short name or HF path) or local path
        device: Device to load on ("cuda", "cpu", or None for auto)
        dtype: Model dtype
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        trust_remote_code: Allow custom code
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")

    Returns:
        Tuple of (model, tokenizer)
    """
    # Resolve model path
    config = get_model_config(model_name_or_path)
    model_path = config["model_path"]

    # Setup quantization config
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Determine device map
    if device is None:
        device_map = "auto"
    elif device == "cpu":
        device_map = {"": "cpu"}
    else:
        device_map = {"": device}

    # Load model
    model_kwargs = {
        "dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_drope_model(
    model_name: str,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a DroPE model specifically.

    Validates that the model is indeed a DroPE model.
    """
    config = get_model_config(model_name)

    if not config["is_drope"]:
        raise ValueError(f"{model_name} is not a DroPE model. Use load_model instead.")

    return load_model(model_name, **kwargs)


def load_model_pair(
    model_name: str,
    **kwargs,
) -> Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]]:
    """
    Load both RoPE and DroPE versions of a model for comparison.

    Args:
        model_name: Base model name (e.g., "smollm-360m")

    Returns:
        Dict with "rope" and "drope" keys, each containing (model, tokenizer)
    """
    # Load RoPE baseline
    rope_model, rope_tokenizer = load_model(model_name, **kwargs)

    # Load DroPE version
    drope_name = f"{model_name}-drope"
    drope_model, drope_tokenizer = load_model(drope_name, **kwargs)

    return {
        "rope": (rope_model, rope_tokenizer),
        "drope": (drope_model, drope_tokenizer),
    }
