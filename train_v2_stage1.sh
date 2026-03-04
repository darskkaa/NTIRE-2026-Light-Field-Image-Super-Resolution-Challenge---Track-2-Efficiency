#!/bin/bash
# =============================================================================
# V2 Stage 1: MLFIM Pre-training (60 epochs, Charbonnier loss)
# =============================================================================
# Usage: bash train_v2_stage1.sh
# Auto-downloads datasets and generates .h5 if not present.
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

header "MLFIM V2 — Stage 1: Pre-training"

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
info "Model:      $MODEL_NAME"
info "Epochs:     $EPOCHS"
info "Batch:      $BATCH"
info "LR:         $LR"
info "Loss:       $LOSS_TYPE (pure pixel loss)"
info "Mask ratio: $MASK_RATIO"
echo ""

# =============================================================================
# STEP 1: DATASET DOWNLOAD (if not present)
# =============================================================================
header "📀 Dataset Preparation"

pip install gdown -q 2>/dev/null || true
mkdir -p datasets downloads

check_and_prepare() {
    FILE=$1
    URL=$2
    DEST="datasets/$3"

    if [ -d "$DEST" ]; then
        success "Dataset '$3' found (Skipping)"
        return
    fi

    info "Checking for $FILE..."
    if [ -f "$FILE" ]; then
        mv "$FILE" downloads/
    elif [ ! -f "downloads/$FILE" ]; then
        if [ -n "$URL" ]; then
            warn "'$FILE' not found. Downloading..."
            gdown --fuzzy "$URL" -O "downloads/$FILE"
        fi
    fi

    if [ -f "downloads/$FILE" ]; then
        info "Extracting $FILE..."
        unzip -q -o "downloads/$FILE" -d datasets/
        success "Extracted $FILE"
    fi
}

check_and_prepare "EPFL.zip" "https://drive.google.com/file/d/19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN/view?usp=drive_link" "EPFL"
check_and_prepare "HCI_new.zip" "https://drive.google.com/file/d/1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9/view?usp=drive_link" "HCI_new"
check_and_prepare "HCI_old.zip" "https://drive.google.com/file/d/1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG/view?usp=drive_link" "HCI_old"
check_and_prepare "INRIA_Lytro.zip" "https://drive.google.com/file/d/1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ/view?usp=drive_link" "INRIA_Lytro"
check_and_prepare "Stanford_Gantry.zip" "https://drive.google.com/file/d/1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3/view?usp=drive_link" "Stanford_Gantry"
success "Dataset preparation complete"

# =============================================================================
# STEP 2: GENERATE TRAINING/TEST DATA (if not present)
# =============================================================================
header "🧩 Data Generation"

mkdir -p data_for_training data_for_test

if [ -n "$(find data_for_training -name "*.h5" | head -1)" ]; then
    success "Training data (.h5) already exists. Skipping generation."
else
    info "Generating SR_5x5_4x training patches..."
    python Generate_Data_for_Training.py --angRes 5 --scale_factor 4 --src_data_path ./datasets/ --save_data_path ./
fi

if [ -n "$(find data_for_test -name "*.h5" | head -1)" ]; then
    success "Test data (.h5) already exists."
else
    info "Generating SR_5x5_4x test patches..."
    python Generate_Data_for_Test.py --angRes 5 --scale_factor 4 --src_data_path ./datasets/ --save_data_path ./
fi

success "Data generation complete"

# =============================================================================
# STEP 3: VERIFY MODEL
# =============================================================================
header "🧪 Verify Model Efficiency"

info "Running V2 model self-test..."
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
success "Model verification passed"

# =============================================================================
# STEP 4: STAGE 1 — MLFIM PRE-TRAINING
# =============================================================================
header "🏋️ Stage 1: MLFIM Pre-training ($EPOCHS Epochs)"

PRETRAIN_CKPT=$(ls -t log/SR_5x5_4x/ALL/${MODEL_NAME}/checkpoints/${MODEL_NAME}_pretrain_best.pth 2>/dev/null | head -1)

if [ -n "$PRETRAIN_CKPT" ]; then
    success "Stage 1 already complete (found best checkpoint: $PRETRAIN_CKPT)"
else
    info "Starting pre-training..."
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
fi

echo ""
header "🏁 Stage 1 Complete!"
success "Best checkpoint saved to log/ directory"
success "Now run: bash train_v2_stage2.sh <path_to_best_checkpoint.pth>"
