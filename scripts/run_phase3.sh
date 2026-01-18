#!/bin/bash
# Phase 3: Mechanistic Analysis
# Track massive value evolution during recalibration

set -e

# Default settings
BASE_MODEL=${1:-"smollm-360m"}
OUTPUT_DIR=${2:-"results/phase3"}
DEVICE=${3:-"cuda"}
NUM_TOKENS=${4:-"10000000000"}  # 10B tokens

echo "=== Phase 3: Mechanistic Analysis ==="
echo "Base Model: $BASE_MODEL"
echo "Training Tokens: $NUM_TOKENS"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Step 1: Create DroPE model (remove RoPE)
echo ""
echo "Step 1: Converting RoPE model to DroPE..."
python -m src.scripts.convert_to_drope \
    --model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR/drope_initial" \
    --device "$DEVICE"

# Step 2: Run recalibration with checkpointing
echo ""
echo "Step 2: Running recalibration training..."
python -m src.scripts.run_recalibration \
    --model-path "$OUTPUT_DIR/drope_initial" \
    --output-dir "$OUTPUT_DIR/checkpoints" \
    --num-tokens "$NUM_TOKENS" \
    --save-every "1000000000" \
    --device "$DEVICE" \
    --analyze-checkpoints

# Step 3: Analyze massive value evolution
echo ""
echo "Step 3: Analyzing evolution across checkpoints..."
python -m src.scripts.analyze_evolution \
    --checkpoints-dir "$OUTPUT_DIR/checkpoints" \
    --output-dir "$OUTPUT_DIR/evolution"

# Step 4: Attention pattern analysis
echo ""
echo "Step 4: Analyzing attention patterns..."
python -m src.scripts.analyze_attention \
    --rope-model "$BASE_MODEL" \
    --drope-model "$OUTPUT_DIR/checkpoints/final" \
    --output-dir "$OUTPUT_DIR/attention" \
    --device "$DEVICE"

echo ""
echo "=== Phase 3 Complete ==="
echo "Results saved to: $OUTPUT_DIR"
