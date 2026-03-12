#!/bin/bash
# =============================================================================
# V3 Stage 2: Fine-tuning (200 epochs, max-PSNR recipe)
# =============================================================================
# Usage: bash train_v3_stage2.sh <path_to_stage1_best.pth>
#
# Research-backed hyperparameters:
#   - LR: 1e-4 (half of pretrain, LFTransMamba StepLR step-down)
#   - Optimizer: Adam β=(0.99, 0.999) (LFTransMamba exact)
#   - Scheduler: MultiStepLR [80, 160] ×0.5 (extended 3-phase decay)
#   - Loss: Charbonnier (smoother near-zero gradients for fine-tuning)
#   - SWA: Final 10% of training for flatter optima
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
if [ -z "$1" ]; then
    echo "Usage: bash train_v3_stage2.sh <path_to_stage1_checkpoint.pth>"
    echo ""
    echo "Example:"
    echo "  bash train_v3_stage2.sh ./log/SR_5x5_4x/ALL/MyEfficientLFNetV3_MLFIM/checkpoints/MyEfficientLFNetV3_MLFIM_pretrain_best.pth"
    exit 1
fi

PRETRAIN_CKPT="$1"

# ---- CONFIG (Research-backed max-PSNR recipe) ----
MODEL_NAME="MyEfficientLFNetV3_MLFIM"
ANGRES=5
SCALE=4
EPOCHS=200
BATCH=3               # LFTransMamba Track 2 exact value
LR=1e-4               # Half of pretrain LR (legacy-matching StepLR step-down)
GRAD_ACCUM=1           # No grad accum needed (simpler, legacy-matching)
LOSS_TYPE=charbonnier   # Charbonnier: smooth near-zero gradients for finetune
SCHED_TYPE=multistep   # MultiStepLR [80, 160] ×0.5
NUM_WORKERS=16

# ---- PATHS ----
TRAIN_DATA="./data_for_training/"
TEST_DATA="./data_for_test/"

echo ""
info "Model:          $MODEL_NAME"
info "Checkpoint:     $PRETRAIN_CKPT"
info "Epochs:         $EPOCHS"
info "Batch:          $BATCH (no accumulation)"
info "LR:             $LR (MultiStepLR [80,160] ×0.5)"
info "Loss:           $LOSS_TYPE (smooth near-zero gradients)"
info "SWA:            Last 10% of training"
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

python -c "
import os, site, ctypes
for p in site.getsitepackages():
    libdir = os.path.join(p, 'nvidia', 'cuda_runtime', 'lib')
    if os.path.exists(os.path.join(libdir, 'libcudart.so.13')):
        # Load libcudart.so.13 as a global shared library to satisfy .so.12 demands
        ctypes.CDLL(os.path.join(libdir, 'libcudart.so.13'), mode=ctypes.RTLD_GLOBAL)
"

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

BEST_FINETUNE_CKPT=$(ls -t log/SR_5x5_4x/ALL/${MODEL_NAME}/checkpoints/${MODEL_NAME}_finetune_best.pth 2>/dev/null | head -1)
if [ -n "$BEST_FINETUNE_CKPT" ]; then
    info "Using best fine-tuned checkpoint: $BEST_FINETUNE_CKPT"
    info "Running inference..."
    python inference.py --model_name $MODEL_NAME --angRes $ANGRES --scale_factor $SCALE \
        --use_pre_ckpt --path_pre_pth "$BEST_FINETUNE_CKPT" \
        --path_for_test ./data_for_test/ --data_name ALL
    success "Inference complete"
else
    warn "No fine-tuned checkpoints found. Skipping inference."
fi

header "🏁 ALL DONE!"
success "Pipeline complete. Ready for CodaBench submission."
echo ""
echo "Next steps:"
echo "  1. Submit results to CodaLab test server"
echo "  2. Fill out the Fact Sheet for co-authorship"
echo ""
