#!/bin/bash
# Phase 2: DroPE Model Analysis
# Compare massive values between RoPE and DroPE models

set -e

# Default settings
BASE_MODEL=${1:-"smollm-360m"}
OUTPUT_DIR=${2:-"results/phase2"}
DEVICE=${3:-"cuda"}

DROPE_MODEL="${BASE_MODEL}-drope"

echo "=== Phase 2: DroPE Model Analysis ==="
echo "RoPE Model: $BASE_MODEL"
echo "DroPE Model: $DROPE_MODEL"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Step 1: Extract massive values from both models
echo ""
echo "Step 1: Extracting massive values from RoPE model..."
python -m src.scripts.extract_massive_values \
    --model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR/rope" \
    --device "$DEVICE"

echo ""
echo "Step 2: Extracting massive values from DroPE model..."
python -m src.scripts.extract_massive_values \
    --model "$DROPE_MODEL" \
    --output-dir "$OUTPUT_DIR/drope" \
    --device "$DEVICE"

# Step 3: Compare massive value positions
echo ""
echo "Step 3: Comparing massive value positions..."
python -m src.scripts.compare_models \
    --rope-dir "$OUTPUT_DIR/rope" \
    --drope-dir "$OUTPUT_DIR/drope" \
    --output-dir "$OUTPUT_DIR/comparison"

# Step 4: Run disruption experiments
echo ""
echo "Step 4: Running disruption experiments..."
for MODEL in "$BASE_MODEL" "$DROPE_MODEL"; do
    for METHOD in mean zero; do
        echo "  Disrupting $MODEL with method=$METHOD..."
        python -m src.scripts.run_disruption \
            --model "$MODEL" \
            --method "$METHOD" \
            --output-dir "$OUTPUT_DIR/disruption/${MODEL}_${METHOD}" \
            --device "$DEVICE"
    done
done

echo ""
echo "=== Phase 2 Complete ==="
echo "Results saved to: $OUTPUT_DIR"
