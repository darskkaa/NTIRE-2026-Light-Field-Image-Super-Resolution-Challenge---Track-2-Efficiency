#!/bin/bash
# =============================================================================
# V2 Stage 1: MLFIM Pre-training (60 epochs, Charbonnier loss)
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
EPOCHS=60           # V2: 60 (was 50; extra 10ep for better feature init)
BATCH=4
LR=2e-4             # LFMamba's proven pretrain LR
MASK_RATIO=0.25
LOSS_TYPE=charbonnier  # SOTA: pure pixel loss for max PSNR
NUM_WORKERS=16

# ---- PATHS (adjust for your setup) ----
TRAIN_DATA="./data_for_training/"
TEST_DATA="./data_for_test/"

echo ""
echo "Model:      $MODEL_NAME"
echo "Epochs:     $EPOCHS"
echo "Batch:      $BATCH"
echo "LR:         $LR"
echo "Loss:       $LOSS_TYPE (pure pixel loss)"
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
    --loss_type "$LOSS_TYPE" \
    --num_workers "$NUM_WORKERS" \
    --path_for_train "$TRAIN_DATA" \
    --path_for_test "$TEST_DATA" \
    --data_name ALL

echo ""
echo "=========================================="
echo "  Stage 1 Complete!"
echo "  Best checkpoint saved to log/ directory"
echo "=========================================="
