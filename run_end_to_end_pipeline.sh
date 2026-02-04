#!/bin/bash
# ============================================================================
# NTIRE 2026 Track 2 - END-TO-END PIPELINE (Training + Submission)
# Merges cloud_train.sh and generate_full_submission.sh
# ============================================================================
#
# Hardware Profile: RTX 5090 (Blackwell sm_120), 32GB VRAM
#
# AUTOMATION STEPS:
# 1. Setup Python Environment (PyTorch Nightly for 5090)
# 2. Download Training Datasets (EPFL, HCI, INRIA, Stanford)
# 3. Extract Validation Data (NTIRE_Val_*.zip)
# 4. Generate Patches
# 5. Train Model (MyEfficientLFNetV5)
# 6. Run Inference (Real + Synth)
# 7. Package Submission (submission.zip)
# ============================================================================

set -e  # Exit on error

echo "============================================================"
echo "🏆 NTIRE 2026 - END-TO-END PIPELINE"
echo "============================================================"

# Define python paths
PYTHON="/venv/lfsr/bin/python"
PIP="/venv/lfsr/bin/pip"

# Fallback
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
    PIP="pip3"
    echo "WARNING: /venv/lfsr not found, using system Python"
fi

# ============================================================================
# PART 1: ENVIRONMENT SETUP
# ============================================================================
echo ""
echo "[1/7] Setting up Environment..."

# Force uninstall old PyTorch (cu124 doesn't support RTX 5090)
$PIP uninstall torch torchvision -y 2>/dev/null || true

# Install PyTorch Nightly (CUDA 12.8)
echo "  Installing PyTorch Nightly for RTX 5090..."
$PIP install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
$PIP install gdown h5py einops tqdm fvcore scipy scikit-image imageio xlwt matplotlib ninja
# Mamba (optional)
$PIP install mamba-ssm causal-conv1d 2>/dev/null || echo "  (mamba-ssm skipped)"

# ============================================================================
# PART 2: DOWNLOAD TRAINING DATASETS
# ============================================================================
echo ""
echo "[2/7] Downloading Training Datasets..."

$PIP install gdown -q

download_dataset() {
    local FILE_ID="$1"
    local OUTPUT="$2"
    if [ -f "$OUTPUT" ] || [ -d "datasets/${OUTPUT%.zip}/training" ]; then
        echo "  ✓ $OUTPUT ready"
    else
        echo "  Downloading $OUTPUT..."
        gdown "$FILE_ID" -O "$OUTPUT" --fuzzy 2>/dev/null || echo "  ⚠ Download failed for $OUTPUT"
    fi
}

download_dataset "19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN" "EPFL.zip"
download_dataset "1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9" "HCI_new.zip"
download_dataset "1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG" "HCI_old.zip"
download_dataset "1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ" "INRIA_Lytro.zip"
download_dataset "1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3" "Stanford_Gantry.zip"

# Extraction Logic (Simplified)
echo "  Extracting datasets..."
mkdir -p datasets
for zip in *.zip; do
    [ -f "$zip" ] || continue
    dataset_name="${zip%.zip}"
    # Minimal check to see if extraction needed
    if [ ! -d "datasets/$dataset_name/training" ] && [[ "$dataset_name" != "NTIRE_Val_"* ]]; then
        unzip -q -o "$zip" -d "datasets/$dataset_name/"
        # Move nested training folder if it exists
        if [ -d "datasets/$dataset_name/training/training" ]; then
             mv datasets/$dataset_name/training/training/* datasets/$dataset_name/training/
             rmdir datasets/$dataset_name/training/training
        elif [ -d "datasets/$dataset_name/$dataset_name/training" ]; then
             mv datasets/$dataset_name/$dataset_name/* datasets/$dataset_name/
             rmdir datasets/$dataset_name/$dataset_name
        fi
        echo "  ✓ Extracted $dataset_name"
    fi
done

# ============================================================================
# PART 3: PREPARE VALIDATION DATA
# ============================================================================
echo ""
echo "[3/7] Preparing Validation Data..."

mkdir -p data_for_test

prepare_val_data() {
    local ZIP_FILE="$1"
    local TARGET_NAME="$2" # e.g. NTIRE_Val_Real
    
    # Check if target folder exists and is not empty
    if [ -d "data_for_test/$TARGET_NAME" ] && [ "$(ls -A data_for_test/$TARGET_NAME 2>/dev/null)" ]; then
        echo "  ✓ $TARGET_NAME ready in data_for_test/"
    else
        if [ -f "$ZIP_FILE" ]; then
            echo "  Extracting $ZIP_FILE..."
            unzip -q -o "$ZIP_FILE" -d data_for_test/
            # Handle nested folder: data_for_test/NTIRE_Val_Real/NTIRE_Val_Real
            if [ -d "data_for_test/$TARGET_NAME/$TARGET_NAME" ]; then
                mv data_for_test/$TARGET_NAME/$TARGET_NAME/* data_for_test/$TARGET_NAME/
                rmdir data_for_test/$TARGET_NAME/$TARGET_NAME
            fi
            echo "  ✓ Extracted $ZIP_FILE"
        else
            echo "  ⚠ WARNING: $ZIP_FILE not found. Submisison generation might fail later."
        fi
    fi
}

prepare_val_data "NTIRE_Val_Real.zip" "NTIRE_Val_Real"
prepare_val_data "NTIRE_Val_Synth.zip" "NTIRE_Val_Synth"

# ============================================================================
# PART 4: GENERATE PATCHES
# ============================================================================
echo ""
echo "[4/7] Generating Patches..."

if [ ! -d "data_for_training/SR_5x5_4x" ]; then
    echo "  Generating training patches..."
    $PYTHON Generate_Data_for_Training.py --angRes 5 --scale_factor 4
else
    echo "  Training patches already exist."
fi

# Symlink fix
if [ -d "data_for_training" ] && [ ! -e "data_for_train" ]; then
    ln -s data_for_training data_for_train
fi

# ============================================================================
# PART 5: TRAINING
# ============================================================================
echo ""
echo "[5/7] Starting Training..."

# Verify constraints first
$PYTHON -c "
import sys, os
sys.path.insert(0, os.getcwd())
try:
    from model.SR.MyEfficientLFNetV5 import get_model, count_parameters
    class Args: angRes_in=5; scale_factor=4; use_macpi=True; use_tta=False
    model = get_model(Args())
    params = count_parameters(model)
    print(f'  Model Params: {params:,} (Limit: 1,000,000)')
    if params > 1000000: exit(1)
except ImportError:
    print('  ⚠ Checking skipped (model file or dependencies missing)')
"

echo "  Training MyEfficientLFNetV5 (80 epochs)..."
$PYTHON train.py \
    --model_name MyEfficientLFNetV5 \
    --angRes 5 \
    --scale_factor 4 \
    --batch_size 8 \
    --lr 2e-4 \
    --epoch 80 \
    --device cuda:0 \
    --num_workers 8 \
    --data_name ALL

echo "  ✓ Training complete"

# ============================================================================
# PART 6: INFERENCE
# ============================================================================
echo ""
echo "[6/7] Running Inference..."

# Find latest checkpoint
CKPT_DIR="log/SR_5x5_4x/ALL/MyEfficientLFNetV5/checkpoints"
CKPT=$(ls -v $CKPT_DIR/*_model.pth 2>/dev/null | tail -n 1)

if [ -z "$CKPT" ]; then
    echo "  ❌ ERROR: No .pth checkpoints found in $CKPT_DIR"
    exit 1
fi
echo "  Using Checkpoint: $CKPT"

run_inference() {
    local DATA_NAME="$1"
    echo "  Infering $DATA_NAME..."
    $PYTHON inference.py \
        --model_name MyEfficientLFNetV5 \
        --angRes 5 \
        --scale_factor 4 \
        --path_pre_pth "$CKPT" \
        --path_for_test "./data_for_test/" \
        --data_name "$DATA_NAME" \
        --device cuda:0
}

run_inference "NTIRE_Val_Real"
run_inference "NTIRE_Val_Synth"

# ============================================================================
# PART 7: PACKAGE SUBMISSION
# ============================================================================
echo ""
echo "[7/7] Packaging Submission..."

# Prepare folders
rm -rf submission_final
mkdir -p submission_final/Real
mkdir -p submission_final/Synth

# Copy results (Assumes standard inference.py output path)
echo "  Copying results..."
cp -r log/SR_5x5_4x/NTIRE_Val_Real/MyEfficientLFNetV5/results/TEST/NTIRE_Val_Real/* submission_final/Real/ 2>/dev/null || echo "  ⚠ Real results copy failed"
cp -r log/SR_5x5_4x/NTIRE_Val_Synth/MyEfficientLFNetV5/results/TEST/NTIRE_Val_Synth/* submission_final/Synth/ 2>/dev/null || echo "  ⚠ Synth results copy failed"

# Zip using Python for safety
echo "  Zipping..."
$PYTHON -c "
import zipfile, os
src_dir = 'submission_final'
zip_name = 'submission.zip'
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in ['Real', 'Synth']:
        folder_path = os.path.join(src_dir, folder)
        if not os.path.exists(folder_path): continue
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, src_dir).replace(os.path.sep, '/')
                zf.write(file_path, arcname)
print(f'  ✓ Created {zip_name}')
"

echo ""
echo "============================================================"
echo "✅ PIPELINE FINISHED"
echo "submission.zip is ready for download."
echo "============================================================"
