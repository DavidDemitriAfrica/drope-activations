#!/bin/bash
# Phase 1: Baseline Massive Value Analysis
# Reproduce findings from the Massive Values paper on RoPE models

set -e

# Default settings
MODEL=${1:-"smollm-360m"}
OUTPUT_DIR=${2:-"results/phase1"}
DEVICE=${3:-"cuda"}

echo "=== Phase 1: Baseline Massive Value Analysis ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"

mkdir -p "$OUTPUT_DIR"

# Step 1: Extract and visualize massive values
echo ""
echo "Step 1: Extracting massive values..."
python -m src.scripts.extract_massive_values \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR/massive_values" \
    --device "$DEVICE" \
    --lambda-threshold 5.0 \
    --num-samples 100

# Step 2: Run baseline evaluations
echo ""
echo "Step 2: Running baseline evaluations..."
python -m src.scripts.run_evaluation \
    --model "$MODEL" \
    --tasks passkey imdb \
    --output-dir "$OUTPUT_DIR/evaluation" \
    --device "$DEVICE"

# Step 3: Generate visualizations
echo ""
echo "Step 3: Generating visualizations..."
python -m src.scripts.visualize_results \
    --input-dir "$OUTPUT_DIR/massive_values" \
    --output-dir "$OUTPUT_DIR/figures"

echo ""
echo "=== Phase 1 Complete ==="
echo "Results saved to: $OUTPUT_DIR"
