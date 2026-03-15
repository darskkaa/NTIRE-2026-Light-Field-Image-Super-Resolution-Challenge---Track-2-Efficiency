#!/bin/bash
# =============================================================================
# V6 Final: Full Pipeline — Data → Verify → Pretrain → Finetune → SWA → Submit
# =============================================================================
# Usage: bash train_v6_final.sh [batch_size]
# =============================================================================
# // turbo-all

set -e

info() { echo -e "[INFO] $1"; }
ok() { echo -e "[OK] $1"; }
warn() { echo -e "[WARN] $1"; }
err() { echo -e "[ERR] $1"; exit 1; }

# ---- CONFIG ----
MODEL_NAME="MyEfficientLFNetV6_Final"
ANGRES=5
SCALE=4
PRETRAIN_EPOCHS=100
FINETUNE_EPOCHS=150
BATCH=${1:-4}
LR=2e-4
MASK_RATIO=0.25
NUM_WORKERS=4
TRAIN_DATA="./data_for_training/"
TEST_DATA="./data_for_test/"

echo "========================================"
echo " MyEfficientLFNet V6 Final — Full Pipeline"
echo "========================================"
info "Model:    $MODEL_NAME"
info "Batch:    $BATCH"
info "LR:       $LR (pretrain), halved for finetune"
info "Pretrain: $PRETRAIN_EPOCHS epochs (MLFIM mask=$MASK_RATIO)"
info "Finetune: $FINETUNE_EPOCHS epochs (no mask)"

# =============================================================================
# STEP 1: DATASET DOWNLOAD
# =============================================================================
echo ""
echo "=== Step 1: Dataset Preparation ==="

pip install gdown fvcore -q 2>/dev/null || true
mkdir -p datasets downloads data_for_training data_for_test

check_and_prepare() {
    FILE=$1; URL=$2; DEST="datasets/$3"
    if [ -d "$DEST" ]; then
        ok "Dataset '$3' found"
        return
    fi
    if [ -f "$FILE" ]; then mv "$FILE" downloads/; fi
    if [ ! -f "downloads/$FILE" ] && [ -n "$URL" ]; then
        warn "'$FILE' not found — downloading..."
        if [[ "$URL" == *"drive.google.com"* ]]; then
            gdown --fuzzy "$URL" -O "downloads/$FILE"
        else
            wget -O "downloads/$FILE" "$URL" || true
        fi
    fi
    if [ -f "downloads/$FILE" ]; then
        info "Extracting $FILE..."
        unzip -q -o "downloads/$FILE" -d datasets/
    fi
}

check_and_prepare "EPFL.zip" "https://huggingface.co/datasets/aaaaaa3232312/efpl/resolve/main/EPFL.zip?download=true" "EPFL"
check_and_prepare "HCI_new.zip" "https://drive.google.com/file/d/1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9/view?usp=drive_link" "HCI_new"
check_and_prepare "HCI_old.zip" "https://drive.google.com/file/d/1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG/view?usp=drive_link" "HCI_old"
check_and_prepare "INRIA_Lytro.zip" "https://huggingface.co/datasets/aaaaaa3232312/efpl/resolve/main/INRIA_Lytro.zip?download=true" "INRIA_Lytro"
check_and_prepare "Stanford_Gantry.zip" "https://huggingface.co/datasets/aaaaaa3232312/efpl/resolve/main/Stanford_Gantry.zip?download=true" "Stanford_Gantry"

# =============================================================================
# STEP 2: GENERATE TRAINING/TEST DATA (.h5 files)
# =============================================================================
echo ""
echo "=== Step 2: Data Generation ==="

if [ -n "$(find data_for_training -name '*.h5' 2>/dev/null | head -1)" ]; then
    ok "Training data (.h5) exists"
else
    info "Generating training patches..."
    python Generate_Data_for_Training.py --angRes 5 --scale_factor 4 \
        --src_data_path ./datasets/ --save_data_path ./ --n_angular_crops 1
fi

if [ -n "$(find data_for_test -name '*.h5' 2>/dev/null | head -1)" ]; then
    ok "Test data (.h5) exists"
else
    info "Generating test patches..."
    python Generate_Data_for_Test.py --angRes 5 --scale_factor 4 \
        --src_data_path ./datasets/ --save_data_path ./
fi

# =============================================================================
# STEP 3: VERIFY MODEL (param count + forward pass)
# =============================================================================
echo ""
echo "=== Step 3: Model Verification ==="

info "Running comprehensive model verification (params, FLOPs, ICNR, forward pass)..."
python verify_v6_final.py --device cuda:0 --detailed || err "Model verification FAILED!"
ok "Model verification passed"

# =============================================================================
# STEP 4: TRAIN BOTH STAGES
# =============================================================================
echo ""
echo "=== Step 4: Training ($PRETRAIN_EPOCHS + $FINETUNE_EPOCHS epochs) ==="

python train_v6_final.py \
    --model_name "$MODEL_NAME" \
    --stage both \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --mlfim_mask_ratio "$MASK_RATIO" \
    --num_workers "$NUM_WORKERS" \
    --path_for_train "$TRAIN_DATA" \
    --path_for_test "$TEST_DATA" \
    --data_name ALL

ok "Training complete!"

# =============================================================================
# STEP 5: GENERATE CODABENCH SUBMISSION
# =============================================================================
echo ""
echo "=== Step 5: Generating CodaBench Submission ==="

python generate_codabench_submission_v6.py --model_name "$MODEL_NAME"

ok "Submission generated!"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "========================================"
echo " V6 Final Pipeline COMPLETE"
echo "========================================"
echo ""
echo "Files created:"
echo "  Model:      model/SR/${MODEL_NAME}.py"
echo "  Checkpoints: log/SR_5x5_4x/ALL/${MODEL_NAME}/checkpoints/"
echo "  Submission:  submission.zip"
echo ""
echo "Next: Upload submission.zip to CodaBench"
