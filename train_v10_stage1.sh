#!/bin/bash
#===============================================================================
# V10 SOTA TRAINING PIPELINE - MyEfficientLFNetV10_MLFIM
# STAGE 1: MLFIM Pre-Training (80 Epochs, 25% Masking)
#===============================================================================

set -e  # Exit on error

# Source shared environment setup (T3 Fix: single source of truth)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

header "🚀 MyEfficientLFNetV10_MLFIM SOTA Training Workflow — STAGE 1"
info "Starting Stage 1 workflow (MLFIM Pre-training)..."

#===============================================================================
# STEP 2 & 3: DATASET PREPARATION
#===============================================================================
header "📀 STEP 2 & 3: Dataset Preparation"

mkdir -p datasets downloads

check_and_prepare() {
    FILE=$1
    URL=$2
    DEST="datasets/$3"
    
    if [ -d "$DEST" ]; then
        success "Dataset '$3' found in datasets/ (Skipping)"
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

#===============================================================================
# STEP 4: GENERATE DATA
#===============================================================================
header "🧩 STEP 4: Data Generation"

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

#===============================================================================
# STEP 5: VERIFY DATASETS & CHANNELS
#===============================================================================
header "🔍 STEP 5: Verify Datasets and Channels"

info "Running verify_datasets.py..."
python verify_datasets.py

info "Running verify_channels.py..."
python verify_channels.py

success "Data verification passed."

#===============================================================================
# STEP 6: VERIFY MODEL
#===============================================================================
header "🧪 STEP 6: Verify MyEfficientLFNetV10_MLFIM Model"

info "Running V10 structural self-test..."
if python model/SR/MyEfficientLFNetV10_MLFIM.py; then
    success "Model self-test passed."
else
    error "Model self-test failed!"
fi

#===============================================================================
# STEP 6.5: NTIRE 2026 TRACK 2 EFFICIENCY VALIDATION (MANDATORY GATE)
#===============================================================================
header "🏆 STEP 6.5: NTIRE 2026 Track 2 — Efficiency Validation"

info "Checking model against Track 2 constraints..."
info "  Param Limit:  < 1,000,000 (1M)"
info "  FLOPs Limit:  < 20G (input: 5×5×32×32, fvcore)"
info "  TTA FLOPs:    counted toward budget"
echo ""

if python check_efficiency.py --model_name MyEfficientLFNetV10_MLFIM --angRes 5 --scale_factor 4 --patch_size 32; then
    success "Efficiency validation PASSED! Model qualifies for Track 2."
else
    error "Efficiency validation FAILED! Model does NOT qualify for Track 2. Fix param/FLOPs limits before training."
fi

#===============================================================================
# STEP 7: STAGE 1 — MLFIM PRE-TRAINING
#===============================================================================
header "🏋️ STEP 7: Stage 1 — MLFIM Pre-training (80 Epochs)"

info "Starting MLFIM Pre-training with 25% masking..."
PRETRAIN_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV10_MLFIM/checkpoints/MyEfficientLFNetV10_MLFIM_pretrain_best.pth 2>/dev/null | head -1)

if [ -n "$PRETRAIN_CKPT" ]; then
    success "Stage 1 already complete (found best checkpoint: $PRETRAIN_CKPT)"
else
    # Stage 1: 80 epochs, lr=2e-4, mask_ratio=0.25
    python train_mlfim.py --stage pretrain --mlfim_mask_ratio 0.25 \
        --model_name MyEfficientLFNetV10_MLFIM --angRes 5 --scale_factor 4 \
        --batch_size 4 --lr 2e-4 --epoch 80 \
        --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ \
        --device cuda:0 --num_workers 8
fi

header "🏁 STAGE 1 DONE!"
success "Stage 1 complete. Now run train_v10_stage2.sh"
