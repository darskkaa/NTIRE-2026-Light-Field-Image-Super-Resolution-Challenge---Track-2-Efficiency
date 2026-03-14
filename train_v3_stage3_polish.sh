#!/bin/bash
# =============================================================================
# V3 Stage 3: Polish Run (30 epochs, max-PSNR last mile)
# =============================================================================
# Usage: bash train_v3_stage3_polish.sh [path_to_stage2_best.pth]
#
# This is a short "polish" run that sits on top of the converged Stage 2 model.
# It uses an extremely low LR, ZERO augmentation, and Charbonnier loss to let
# the model memorize the clean training distribution with sub-pixel precision.
#
# Expected improvement: +0.02–0.05 dB on top of Stage 2 best.
# =============================================================================

set -e

# ---- ANSI Colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}==========================================${NC}\n${BOLD}${CYAN}$1${NC}\n${BOLD}${CYAN}==========================================${NC}"; }

header "MLFIM V3 — Stage 3: Polish Run"

# ---- ARGS ----
DEFAULT_CKPT="./log/SR_5x5_4x/ALL/MyEfficientLFNetV3_MLFIM/checkpoints/MyEfficientLFNetV3_MLFIM_finetune_best.pth"

if [ -n "$1" ]; then
    FINETUNE_CKPT="$1"
else
    FINETUNE_CKPT="$DEFAULT_CKPT"
    info "No checkpoint arg provided, using default: $DEFAULT_CKPT"
fi

# ---- CONFIG (Polish recipe: sub-pixel settling) ----
MODEL_NAME="MyEfficientLFNetV3_MLFIM"
ANGRES=5
SCALE=4
EPOCHS=30              # Short burst — already converged
BATCH=4                # Can use larger batch since no augmentation
LR=5e-6               # 50x lower than Stage 2 — sub-pixel settling only
GRAD_ACCUM=1           # No accumulation needed at this LR
LOSS_TYPE=charbonnier  # Smooth L1 for precise minimum
SCHED_TYPE=cosine      # 5e-6 → 1e-7
NUM_WORKERS=16

# ---- PATHS ----
TRAIN_DATA="./data_for_training/"
TEST_DATA="./data_for_test/"

echo ""
info "Model:          $MODEL_NAME"
info "Checkpoint:     $FINETUNE_CKPT"
info "Epochs:         $EPOCHS"
info "Batch:          $BATCH (no accumulation)"
info "LR:             $LR → 1e-7 (CosineAnnealingLR ultra-gentle)"
info "Loss:           $LOSS_TYPE"
info "Augmentation:   DISABLED (zero noise for clean memorization)"
info "SWA:            DISABLED (too few epochs)"
echo ""

# ---- VERIFY ----
header "🔍 Pre-flight: Verify"

if [ -z "$(find data_for_training -name "*.h5" 2>/dev/null | head -1)" ]; then
    error "Training data not found!"
fi
if [ ! -f "$FINETUNE_CKPT" ]; then
    error "Checkpoint not found: $FINETUNE_CKPT"
fi
success "Data and checkpoint found."

# ---- STAGE 3: POLISH ----
header "✨ Stage 3: Polish ($EPOCHS Epochs)"

python train_mlfim_v3.py \
    --stage finetune \
    --model_name "$MODEL_NAME" \
    --angRes "$ANGRES" \
    --scale_factor "$SCALE" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --epoch "$EPOCHS" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --loss_type "$LOSS_TYPE" \
    --scheduler_type "$SCHED_TYPE" \
    --warmup_epochs 0 \
    --val_step 5 \
    --swa_start_frac 1.1 \
    --use_pre_ckpt \
    --path_pre_pth "$FINETUNE_CKPT" \
    --num_workers "$NUM_WORKERS" \
    --path_for_train "$TRAIN_DATA" \
    --path_for_test "$TEST_DATA" \
    --data_name ALL \
    --no_augmentation

success "Stage 3 (Polish) complete"

# ---- INFERENCE WITH SELF-ENSEMBLE ----
header "📊 Inference + Self-Ensemble"

CKPT_DIR="log/SR_5x5_4x/ALL/${MODEL_NAME}/checkpoints"
POLISH_BEST="${CKPT_DIR}/${MODEL_NAME}_finetune_best.pth"

if [ -f "$POLISH_BEST" ]; then
    info "Running inference with self-ensemble..."
    python inference.py --model_name $MODEL_NAME --angRes $ANGRES --scale_factor $SCALE \
        --use_pre_ckpt --path_pre_pth "$POLISH_BEST" \
        --path_for_test ./data_for_test/ --data_name ALL \
        --self_ensemble
    success "Self-ensemble inference complete"
else
    warn "No polished checkpoint found at: $POLISH_BEST"
fi

header "🏁 ALL DONE!"
echo ""
echo "Expected improvements over Stage 2 (31.90 dB):"
echo "  [+0.02-0.05 dB] Polish run (ultra-low LR, zero augmentation)"
echo "  [+0.10-0.15 dB] Self-ensemble (8x geometric averaging)"
echo "  Total expected: ~32.05-32.10 dB"
echo ""
