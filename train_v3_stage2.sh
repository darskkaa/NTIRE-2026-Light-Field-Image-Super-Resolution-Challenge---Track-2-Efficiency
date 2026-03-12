#!/bin/bash
# =============================================================================
# V3 Stage 2: Fine-tuning (200 epochs, max-PSNR recipe)
# =============================================================================
# Usage: bash train_v3_stage2.sh [path_to_stage1_best.pth]
#
# Optimized hyperparameters (fixes all legacy finetune bottlenecks):
#   - LR: 1e-4 (half of pretrain — gentle entry preserves learned features)
#   - Optimizer: Adam β=(0.99, 0.999) (LFTransMamba exact)
#   - Scheduler: CosineAnnealingLR 1e-4 → 1e-6 (smooth decay, no shocks)
#   - Loss: Charbonnier eps=1e-9 (gradient shrinks near zero for sub-pixel settling)
#   - SWA: Final 10% of training (flatter optima → domain-shift resilience)
#   - EMA: 0.997 → 0.999 at 75% (built into train_mlfim_v3.py)
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

header "MLFIM V3 — Stage 2: Fine-tuning"

# ---- ARGS ----
DEFAULT_CKPT="../NTIRE-2026-Light-Field-Image-Super-Resolution-Challenge---Track-2-Efficiency/MyEfficientLFNetV3_MLFIM_pretrain_best.pth"

if [ -n "$1" ]; then
    PRETRAIN_CKPT="$1"
else
    PRETRAIN_CKPT="$DEFAULT_CKPT"
    info "No checkpoint arg provided, using default: $DEFAULT_CKPT"
fi

# ---- CONFIG (Max-PSNR recipe: fixes L1/StepLR/high-LR legacy bottlenecks) ----
MODEL_NAME="MyEfficientLFNetV3_MLFIM"
ANGRES=5
SCALE=4
EPOCHS=200
BATCH=3               # LFTransMamba Track 2 exact value
LR=1e-4               # Half of pretrain (2e-4): gentle entry preserves 31.85 dB features
GRAD_ACCUM=2           # Effective BS=6 (matches legacy, smoother gradients)
LOSS_TYPE=charbonnier  # Gradient shrinks near zero → settles into sub-pixel minimum
SCHED_TYPE=cosine      # CosineAnnealingLR: smooth 1e-4 → 1e-6 over 200 epochs
NUM_WORKERS=16

# ---- PATHS ----
TRAIN_DATA="./data_for_training/"
TEST_DATA="./data_for_test/"

echo ""
info "Model:          $MODEL_NAME"
info "Checkpoint:     $PRETRAIN_CKPT"
info "Epochs:         $EPOCHS"
info "Batch:          $BATCH × $GRAD_ACCUM accum = $((BATCH * GRAD_ACCUM)) effective"
info "LR:             $LR → 1e-6 (CosineAnnealingLR smooth decay)"
info "Loss:           $LOSS_TYPE (eps=1e-9, gradient shrinks at minimum)"
info "Scheduler:      $SCHED_TYPE (T_max=$EPOCHS, eta_min=1e-6)"
info "SWA:            Last 10% of training (epoch $((EPOCHS * 9 / 10))+)"
info "Augmentation:   CutBlur=0.10, MixUp=0.15 (reduced for precision)"
echo ""

# =============================================================================
# PRE-FLIGHT: VERIFY DATA EXISTS
# =============================================================================
header "🔍 Pre-flight: Verify Data"

if [ -z "$(find data_for_training -name "*.h5" 2>/dev/null | head -1)" ]; then
    error "Training data not found! Run train_v3_stage1.sh first."
fi
if [ -z "$(find data_for_test -name "*.h5" 2>/dev/null | head -1)" ]; then
    error "Test data not found! Run train_v3_stage1.sh first."
fi
success "Training and test data found."

# =============================================================================
# VERIFY CHECKPOINT EXISTS
# =============================================================================
if [ ! -f "$PRETRAIN_CKPT" ]; then
    error "Checkpoint not found: $PRETRAIN_CKPT"
fi
success "Stage 1 checkpoint found: $PRETRAIN_CKPT"

# =============================================================================
# STAGE 2: FINE-TUNING
# =============================================================================
header "🏋️ Stage 2: Fine-tuning ($EPOCHS Epochs)"


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
    --warmup_epochs 3 \
    --val_step 5 \
    --use_pre_ckpt \
    --path_pre_pth "$PRETRAIN_CKPT" \
    --num_workers "$NUM_WORKERS" \
    --path_for_train "$TRAIN_DATA" \
    --path_for_test "$TEST_DATA" \
    --data_name ALL

success "Stage 2 (Fine-tuning) complete"

# =============================================================================
# INFERENCE & EVALUATION
# =============================================================================
header "📊 Inference and Evaluation"

CKPT_DIR="log/SR_5x5_4x/ALL/${MODEL_NAME}/checkpoints"
BEST_FINETUNE_CKPT="${CKPT_DIR}/${MODEL_NAME}_finetune_best.pth"
SWA_CKPT="${CKPT_DIR}/${MODEL_NAME}_finetune_swa.pth"

if [ -f "$BEST_FINETUNE_CKPT" ]; then
    info "Using best fine-tuned checkpoint: $BEST_FINETUNE_CKPT"
    info "Running inference on validation sets..."
    python inference.py --model_name $MODEL_NAME --angRes $ANGRES --scale_factor $SCALE \
        --use_pre_ckpt --path_pre_pth "$BEST_FINETUNE_CKPT" \
        --path_for_test ./data_for_test/ --data_name ALL
    success "Inference complete (best checkpoint)"
else
    warn "No fine-tuned best checkpoint found."
fi

if [ -f "$SWA_CKPT" ]; then
    info "SWA checkpoint also available: $SWA_CKPT"
    info "(train_mlfim_v3.py already compared SWA vs EMA and saved the winner as _best.pth)"
fi

header "📦 Generate CodaBench Submission"

if [ -f "generate_codabench_submission_v3.py" ] && [ -f "$BEST_FINETUNE_CKPT" ]; then
    info "Generating CodaBench submission..."
    python generate_codabench_submission_v3.py \
        --model_name $MODEL_NAME --angRes $ANGRES --scale_factor $SCALE \
        --use_pre_ckpt --path_pre_pth "$BEST_FINETUNE_CKPT"
    success "CodaBench submission generated"
else
    warn "Skipping submission generation (missing script or checkpoint)."
    echo "  Run manually: python generate_codabench_submission_v3.py --model_name $MODEL_NAME --angRes $ANGRES --scale_factor $SCALE --use_pre_ckpt --path_pre_pth $BEST_FINETUNE_CKPT"
fi

header "🏁 ALL DONE!"
success "Pipeline complete. Ready for CodaBench submission."
echo ""
echo "Summary of what was optimized vs legacy finetune (32.10 dB / 29.65 server):"
echo "  [FIXED] LR: 2e-4 → 1e-4 (half of pretrain, preserves learned features)"
echo "  [FIXED] Loss: L1 → Charbonnier (gradient shrinks at minimum, no bouncing)"
echo "  [FIXED] Scheduler: StepLR → CosineAnnealing (smooth decay, no shocks)"
echo "  [KEPT]  SWA: last 10% (flatter optima → domain-shift resilience)"
echo "  [KEPT]  Reduced augmentation (CutBlur=0.10, MixUp=0.15)"
echo ""
echo "Expected: 32.25-32.40 local validation → 29.95-30.10 server PSNR"
echo ""
