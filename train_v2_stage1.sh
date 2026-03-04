#!/bin/bash
# =============================================================================
# V2 Stage 1: MLFIM Pre-training (50 epochs, faster than V1's 80)
# =============================================================================
# Usage: bash train_v2_stage1.sh
# Runs on Colab with pre-generated training data.
# =============================================================================

set -e

echo "=========================================="
echo "  MLFIM V2 — Stage 1: Pre-training"
echo "=========================================="

# ---- CONFIG ----
MODEL_NAME="MyEfficientLFNetV2_MLFIM"
ANGRES=5
SCALE=4
EPOCHS=50           # V2: 50 (V1 used 80 — saturation after 40 means wasted epochs)
BATCH=4
LR=2e-4
MASK_RATIO=0.25
NUM_WORKERS=20

# ---- PATHS (adjust for your setup) ----
TRAIN_DATA="./data_for_training/"
TEST_DATA="./data_for_test/"

echo ""
echo "Model:      $MODEL_NAME"
echo "Epochs:     $EPOCHS (reduced from 80 — pretrain saturates early)"
echo "Batch:      $BATCH"
echo "LR:         $LR"
echo "Mask ratio: $MASK_RATIO"
echo ""

# ---- VERIFY MODEL ----
echo "Verifying model efficiency..."
python -c "
import torch
import sys
sys.path.insert(0, '.')
from model.SR.${MODEL_NAME} import get_model

class A:
    angRes_in = $ANGRES
    scale_factor = $SCALE

m = get_model(A())
params = sum(p.numel() for p in m.parameters())
print(f'Parameters: {params:,} ({params/1e6:.3f}M)')
assert params < 1_000_000, f'OVER BUDGET: {params} > 1M'
print('✅ Under 1M param limit')
"

echo ""
echo "Starting pre-training..."

python train_mlfim_v2.py \
    --stage pretrain \
    --model_name "$MODEL_NAME" \
    --angRes "$ANGRES" \
    --scale_factor "$SCALE" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --epoch "$EPOCHS" \
    --mlfim_mask_ratio "$MASK_RATIO" \
    --num_workers "$NUM_WORKERS" \
    --path_for_train "$TRAIN_DATA" \
    --path_for_test "$TEST_DATA" \
    --data_name ALL

echo ""
echo "=========================================="
echo "  Stage 1 Complete!"
echo "  Best checkpoint saved to log/ directory"
echo "=========================================="
