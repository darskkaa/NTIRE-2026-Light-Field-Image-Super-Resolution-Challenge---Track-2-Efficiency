#!/bin/bash
# =============================================================================
# V2 Stage 2: Fine-tuning (100 epochs, optimized LR + loss scheduling)
# =============================================================================
# Usage: bash train_v2_stage2.sh <path_to_stage1_best.pth>
#
# Key V2 improvements:
#   - LR: 1e-4 (was 5e-5) — escapes pretrain local minimum
#   - Grad accum 2 steps → effective batch=8 (was 4) — smoother gradients
#   - Loss warmup: L1 for first 20 epochs, then composite loss
# =============================================================================

set -e

echo "=========================================="
echo "  MLFIM V2 — Stage 2: Fine-tuning"
echo "=========================================="

# ---- ARGS ----
if [ -z "$1" ]; then
    echo "Usage: bash train_v2_stage2.sh <path_to_stage1_checkpoint.pth>"
    echo ""
    echo "Example:"
    echo "  bash train_v2_stage2.sh ./log/SR_5x5_4x/ALL/MyEfficientLFNetV2_MLFIM/checkpoints/MyEfficientLFNetV2_MLFIM_pretrain_best.pth"
    exit 1
fi

PRETRAIN_CKPT="$1"

# ---- CONFIG ----
MODEL_NAME="MyEfficientLFNetV2_MLFIM"
ANGRES=5
SCALE=4
EPOCHS=100           # V2: 100 (LFTransMamba uses 100; 200 risks Colab timeout)
BATCH=4
LR=2e-4              # V2: 2e-4 (research: 2e-4 to 3e-4 is sweet spot for LFSR)
GRAD_ACCUM=2         # Effective batch = 4 × 2 = 8
LOSS_WARMUP=10       # 10% of 100 epochs (was 20 for 200 epochs)
NUM_WORKERS=2

# ---- PATHS ----
TRAIN_DATA="./data_for_training/"
TEST_DATA="./data_for_test/"

echo ""
echo "Model:          $MODEL_NAME"
echo "Checkpoint:     $PRETRAIN_CKPT"
echo "Epochs:         $EPOCHS"
echo "Batch:          $BATCH × $GRAD_ACCUM accum = $((BATCH * GRAD_ACCUM)) effective"
echo "LR:             $LR → cosine → 1e-6"
echo "Loss warmup:    $LOSS_WARMUP epochs (L1-only before composite)"
echo ""

python train_mlfim_v2.py \
    --stage finetune \
    --model_name "$MODEL_NAME" \
    --angRes "$ANGRES" \
    --scale_factor "$SCALE" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --epoch "$EPOCHS" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --loss_warmup_epochs "$LOSS_WARMUP" \
    --use_pre_ckpt \
    --path_pre_pth "$PRETRAIN_CKPT" \
    --num_workers "$NUM_WORKERS" \
    --path_for_train "$TRAIN_DATA" \
    --path_for_test "$TEST_DATA" \
    --data_name ALL

echo ""
echo "=========================================="
echo "  Stage 2 Complete!"
echo "  Best checkpoint: log/.../checkpoints/${MODEL_NAME}_finetune_best.pth"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run inference with the best checkpoint"
echo "  2. Submit to CodaLab test server"
echo ""
echo "  python inference.py \\"
echo "    --model_name $MODEL_NAME \\"
echo "    --scale_factor $SCALE \\"
echo "    --use_pre_ckpt \\"
echo "    --path_pre_pth <best_checkpoint.pth> \\"
echo "    --data_name NTIRE_Val_Real"
