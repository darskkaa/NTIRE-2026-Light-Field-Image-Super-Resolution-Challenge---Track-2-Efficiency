#!/bin/bash
# ============================================================================
# NTIRE 2026 Track 2 - SUBMISSION GENERATION PIPELINE
# Automates: Environment -> Data Prep -> Inference -> Zipping
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "🚀 NTIRE 2026 - SUBMISSION GENERATION"
echo "=============================================="

# Define paths
PYTHON="/venv/lfsr/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
    echo "WARNING: /venv/lfsr not found, using system Python"
fi

# ============================================================================
# STEP 1: PREPARE VALIDATION DATA
# ============================================================================
echo ""
echo "[1/4] Preparing validation data..."

# Ensure target directory exists
mkdir -p data_for_test

# Function to unzip if missing
prepare_val_data() {
    local ZIP_FILE="$1"
    local TARGET_DIR="data_for_test/$2"
    
    if [ -d "$TARGET_DIR" ] && [ "$(ls -A $TARGET_DIR)" ]; then
        echo "  ✓ $2 already exists in data_for_test/"
    else
        if [ -f "$ZIP_FILE" ]; then
            echo "  Extracting $ZIP_FILE..."
            unzip -q -o "$ZIP_FILE" -d data_for_test/
            # Fix nested folder issues if they occur
            if [ -d "data_for_test/$2/$2" ]; then
                mv data_for_test/$2/$2/* data_for_test/$2/
                rmdir data_for_test/$2/$2
            fi
            echo "  ✓ Extracted $ZIP_FILE"
        else
            echo "  ❌ ERROR: $ZIP_FILE not found! Upload it to workspace root."
            exit 1
        fi
    fi
}

prepare_val_data "NTIRE_Val_Real.zip" "NTIRE_Val_Real"
prepare_val_data "NTIRE_Val_Synth.zip" "NTIRE_Val_Synth"

# ============================================================================
# STEP 2: FIND LATEST CHECKPOINT
# ============================================================================
echo ""
echo "[2/4] Locating model checkpoint..."

CKPT_DIR="log/SR_5x5_4x/ALL/MyEfficientLFNetV5/checkpoints"
# Find latest by epoch number in filename (e.g. ...epoch_80_model.pth)
CKPT=$(ls -v $CKPT_DIR/*_model.pth 2>/dev/null | tail -n 1)

if [ -z "$CKPT" ]; then
    echo "  ❌ ERROR: No .pth checkpoints found in $CKPT_DIR"
    echo "     Is the path provided by the user correct? Check 'log' folder."
    exit 1
fi

echo "  ✓ Found checkpoint: $CKPT"

# ============================================================================
# STEP 3: RUN INFERENCE
# ============================================================================
echo ""
echo "[3/4] Running inference..."

run_inference() {
    local DATA_NAME="$1"
    echo "  Processing $DATA_NAME..."
    
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

echo "  ✓ Inference complete"

# ============================================================================
# STEP 4: PACKAGE SUBMISSION
# ============================================================================
echo ""
echo "[4/4] creating submission.zip..."

# Clean previous submission folder
rm -rf submission_final
mkdir -p submission_final/Real
mkdir -p submission_final/Synth

# Copy results (Adjust source path based on inference.py output structure)
# Log path structure: log/SR_5x5_4x/[DATASET]/MyEfficientLFNetV5/results/TEST/[DATASET]/...
# REAL
cp -r log/SR_5x5_4x/NTIRE_Val_Real/MyEfficientLFNetV5/results/TEST/NTIRE_Val_Real/* submission_final/Real/
# SYNTH
cp -r log/SR_5x5_4x/NTIRE_Val_Synth/MyEfficientLFNetV5/results/TEST/NTIRE_Val_Synth/* submission_final/Synth/

# Verify counts
REAL_COUNT=$(find submission_final/Real -name "*.bmp" | wc -l)
SYNTH_COUNT=$(find submission_final/Synth -name "*.bmp" | wc -l)

echo "  Files collected: Real=$REAL_COUNT, Synth=$SYNTH_COUNT"

if [ "$REAL_COUNT" -ne 400 ] || [ "$SYNTH_COUNT" -ne 400 ]; then
    echo "  ⚠ WARNING: Expected 400 files each, found differents counts."
fi

# Zip using Python to ensure Linux-style forward slashes and simple root structure
$PYTHON -c "
import zipfile
import os

src_dir = 'submission_final'
zip_name = 'submission.zip'

print(f'  Zipping {src_dir} to {zip_name}...')
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in ['Real', 'Synth']:
        folder_path = os.path.join(src_dir, folder)
        if not os.path.exists(folder_path):
            continue
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate arcname to be relative to submission_final (e.g. Real/scene/view.bmp)
                arcname = os.path.relpath(file_path, start=src_dir)
                # Force forward slashes
                arcname = arcname.replace(os.path.sep, '/')
                zf.write(file_path, arcname)

print('  ✓ submission.zip created successfully')
"

echo ""
echo "=============================================="
echo "✅ DONE! Download 'submission.zip'"
echo "=============================================="
# Verify structure list
unzip -l submission.zip | head -n 10
