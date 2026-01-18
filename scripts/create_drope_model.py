#!/usr/bin/env python3
"""
Create a DroPE model from a RoPE model by removing positional embeddings.
This creates a model that has NOT been recalibrated - for testing purposes.
"""

import argparse
from pathlib import Path
import torch

from src.utils.model_loading import load_model
from src.drope.conversion import convert_rope_to_drope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="smollm-360m")
    parser.add_argument("--output-dir", type=str, default="models/drope")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f"{args.model}-drope-unconverted"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device=args.device)

    print("Converting to DroPE (removing RoPE)...")
    drope_model = convert_rope_to_drope(model, copy_model=False)

    print(f"Saving to {output_dir}...")
    drope_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done!")
    print(f"\nNOTE: This model has NOT been recalibrated.")
    print("It will likely perform poorly until recalibration training is done.")


if __name__ == "__main__":
    main()
