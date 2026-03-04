#!/bin/bash
# =============================================================================
# V2 Stage 2: Fine-tuning (120 epochs, max-PSNR recipe)
# =============================================================================
# Usage: bash train_v2_stage2.sh <path_to_stage1_best.pth>
#
# Research-backed hyperparameters (see implementation_plan.md):
#   - LR: 3e-4 (LFTransMamba 1st-place NTIRE 2025 recipe)
#   - β2: 0.99 (LFTransMamba — faster 2nd-moment adaptation)
#   - Weight decay: 5e-5 (reduced — less regularization for fitting)
#   - Loss: Pure Charbonnier (SwinIR/HAT/LFMamba/LFTransMamba all use L1/Charb)
#   - Grad accum 2 steps → effective batch=8
#   - Cosine eta_min: 5e-7 (deeper tail for EMA distillation)
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

# ---- CONFIG (Research-backed max-PSNR recipe) ----
MODEL_NAME="MyEfficientLFNetV2_MLFIM"
ANGRES=5
SCALE=4
EPOCHS=120           # Extended: 120ep for deep cosine tail + EMA polish
BATCH=4
LR=3e-4              # LFTransMamba recipe (1st NTIRE 2025)
BETA2=0.99            # LFTransMamba: faster adaptation for finetune
WEIGHT_DECAY=5e-5     # Reduced: less regularization = more fitting capacity
ETA_MIN=5e-7          # Deep cosine tail for EMA distillation
GRAD_ACCUM=2          # Effective batch = 4 × 2 = 8
LOSS_TYPE=charbonnier  # SOTA: pure pixel loss for max PSNR
NUM_WORKERS=2

# ---- PATHS ----
TRAIN_DATA="./data_for_training/"
TEST_DATA="./data_for_test/"

echo ""
echo "Model:          $MODEL_NAME"
echo "Checkpoint:     $PRETRAIN_CKPT"
echo "Epochs:         $EPOCHS"
echo "Batch:          $BATCH × $GRAD_ACCUM accum = $((BATCH * GRAD_ACCUM)) effective"
echo "LR:             $LR → cosine → $ETA_MIN"
echo "β2:             $BETA2 (LFTransMamba)"
echo "Weight decay:   $WEIGHT_DECAY"
echo "Loss:           $LOSS_TYPE (pure pixel loss)"
echo ""

python train_mlfim_v2.py \
    --stage finetune \
    --model_name "$MODEL_NAME" \
    --angRes "$ANGRES" \
    --scale_factor "$SCALE" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --epoch "$EPOCHS" \
    --beta2 "$BETA2" \
    --weight_decay "$WEIGHT_DECAY" \
    --eta_min "$ETA_MIN" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --loss_type "$LOSS_TYPE" \
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
