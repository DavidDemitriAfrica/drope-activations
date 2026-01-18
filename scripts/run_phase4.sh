#!/bin/bash
# Phase 4: Extended Analysis
# Context length scaling and advanced experiments

set -e

# Default settings
BASE_MODEL=${1:-"llama2-7b"}
OUTPUT_DIR=${2:-"results/phase4"}
DEVICE=${3:-"cuda"}

DROPE_MODEL="${BASE_MODEL}-drope"

echo "=== Phase 4: Extended Analysis ==="
echo "RoPE Model: $BASE_MODEL"
echo "DroPE Model: $DROPE_MODEL"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Step 1: Context length scaling experiment
echo ""
echo "Step 1: Context length scaling..."
for CTX_MULT in 1 2 4 8; do
    echo "  Testing context multiplier: ${CTX_MULT}x"

    # RoPE model
    python -m src.scripts.run_context_scaling \
        --model "$BASE_MODEL" \
        --context-multiplier "$CTX_MULT" \
        --output-dir "$OUTPUT_DIR/context_scaling/rope_${CTX_MULT}x" \
        --device "$DEVICE"

    # DroPE model
    python -m src.scripts.run_context_scaling \
        --model "$DROPE_MODEL" \
        --context-multiplier "$CTX_MULT" \
        --output-dir "$OUTPUT_DIR/context_scaling/drope_${CTX_MULT}x" \
        --device "$DEVICE"
done

# Step 2: Massive values at extended context
echo ""
echo "Step 2: Massive values at extended context..."
python -m src.scripts.analyze_extended_context \
    --rope-model "$BASE_MODEL" \
    --drope-model "$DROPE_MODEL" \
    --context-lengths 4096 8192 16384 32768 \
    --output-dir "$OUTPUT_DIR/extended_context" \
    --device "$DEVICE"

# Step 3: Quantization interaction (optional)
if command -v python -c "import bitsandbytes" &> /dev/null; then
    echo ""
    echo "Step 3: Quantization experiments..."
    for QUANT in 8bit 4bit; do
        python -m src.scripts.run_quantization_experiment \
            --rope-model "$BASE_MODEL" \
            --drope-model "$DROPE_MODEL" \
            --quantization "$QUANT" \
            --output-dir "$OUTPUT_DIR/quantization/${QUANT}" \
            --device "$DEVICE"
    done
else
    echo ""
    echo "Step 3: Skipping quantization (bitsandbytes not installed)"
fi

echo ""
echo "=== Phase 4 Complete ==="
echo "Results saved to: $OUTPUT_DIR"
