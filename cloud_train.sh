#!/bin/bash
# ============================================================================
# NTIRE 2026 Track 2 - PRODUCTION Training Pipeline
# Cloud GPU: RTX 5090 32GB VRAM, CUDA 12.8, Blackwell sm_120
# ============================================================================
# LESSONS LEARNED:
# 1. PyTorch cu124 does NOT support RTX 5090 - must use nightly cu128
# 2. Generate_Data_for_Training.py saves to "data_for_training/" (full word)
# 3. train.py expects "data_for_train/" (abbreviated) - need symlink
# 4. Use /venv/lfsr/bin/python directly, not conda activate
# 5. Force uninstall old PyTorch before installing nightly
# ============================================================================
#
# DATASET GOOGLE DRIVE IDs:
# EPFL:           19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN
# HCI_new:        1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9
# HCI_old:        1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG
# INRIA_Lytro:    1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ
# Stanford_Gantry: 1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3
#
# EXPECTED PERFORMANCE:
# - Parameters: 367,526 (<1M ✓)
# - PSNR (worst): ~28 dB
# - PSNR (expected): 30.5-31.0 dB
# - PSNR (best): 31.5+ dB
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "NTIRE 2026 Track 2 - Training Pipeline"
echo "RTX 5090 Optimized (Blackwell sm_120)"
echo "=============================================="

# Define paths
PYTHON="/venv/lfsr/bin/python"
PIP="/venv/lfsr/bin/pip"

# Fallback if venv doesn't exist
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
    PIP="pip3"
    echo "WARNING: /venv/lfsr not found, using system Python"
fi

# ============================================================================
# STEP 1: Verify Project Structure
# ============================================================================
echo ""
echo "[1/8] Checking project structure..."

if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found. Are you in the project root?"
    echo "Run: cd /workspace/NTIRE-2026-Light-Field-Image-Super-Resolution-Challenge---Track-2-Efficiency"
    exit 1
fi

if [ ! -d "model/SR" ]; then
    echo "ERROR: model/SR directory not found."
    exit 1
fi

if [ ! -f "model/SR/MyEfficientLFNetV5.py" ]; then
    echo "ERROR: MyEfficientLFNetV5.py not found in model/SR/"
    exit 1
fi

echo "  ✓ Project structure OK"

# ============================================================================
# STEP 2: Download Datasets (if not present)
# ============================================================================
echo ""
echo "[2/8] Downloading datasets from Google Drive..."

# Install gdown if needed
$PIP install gdown -q

# Download function with retry
download_dataset() {
    local FILE_ID="$1"
    local OUTPUT="$2"
    
    if [ -f "$OUTPUT" ] || [ -d "datasets/${OUTPUT%.zip}/training" ]; then
        echo "  ✓ $OUTPUT already exists, skipping"
        return 0
    fi
    
    echo "  Downloading $OUTPUT..."
    gdown "$FILE_ID" -O "$OUTPUT" --fuzzy 2>/dev/null || {
        echo "  ⚠ gdown failed for $OUTPUT (quota exceeded?)"
        echo "    Manual download: https://drive.google.com/file/d/$FILE_ID/view"
        return 1
    }
}

# Download all 5 datasets
download_dataset "19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN" "EPFL.zip"
download_dataset "1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9" "HCI_new.zip"
download_dataset "1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG" "HCI_old.zip"
download_dataset "1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ" "INRIA_Lytro.zip"
download_dataset "1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3" "Stanford_Gantry.zip"

echo "  ✓ Dataset download complete"


# ============================================================================
# STEP 2: Extract Datasets (if zips exist)
# ============================================================================
echo ""
echo "[3/8] Checking and extracting datasets..."

mkdir -p datasets

# -------------------------------------------------------------------------
# Generic function to extract and fix dataset structure
# Usage: extract_dataset "ZipName" "ExpectedFolderName"
# -------------------------------------------------------------------------
extract_dataset() {
    local ZIP_PATTERN="$1"
    local FOLDER_NAME="$2"
    
    # Find any matching zip file (case insensitive, various naming patterns)
    local ZIP_FILE=$(ls ${ZIP_PATTERN}*.zip ${ZIP_PATTERN}*.ZIP 2>/dev/null | head -1)
    
    if [ -z "$ZIP_FILE" ]; then
        # Try lowercase
        ZIP_FILE=$(ls $(echo "$ZIP_PATTERN" | tr '[:upper:]' '[:lower:]')*.zip 2>/dev/null | head -1)
    fi
    
    # Check if we need to extract
    if [ -n "$ZIP_FILE" ] && [ ! -d "datasets/$FOLDER_NAME/training" ]; then
        echo "  Extracting $ZIP_FILE..."
        
        # Create temp extraction dir
        rm -rf temp_extract
        mkdir -p temp_extract
        unzip -q -o "$ZIP_FILE" -d temp_extract/
        
        # Find the training folder wherever it is (handles nested structures)
        TRAINING_DIR=$(find temp_extract -type d -name "training" 2>/dev/null | head -1)
        TEST_DIR=$(find temp_extract -type d -name "test" 2>/dev/null | head -1)
        
        if [ -n "$TRAINING_DIR" ]; then
            mkdir -p "datasets/$FOLDER_NAME"
            cp -r "$TRAINING_DIR" "datasets/$FOLDER_NAME/"
            if [ -n "$TEST_DIR" ]; then
                cp -r "$TEST_DIR" "datasets/$FOLDER_NAME/"
            fi
            echo "  ✓ $FOLDER_NAME extracted"
        else
            # Fallback: Maybe the zip contains .mat files directly
            MAT_FILES=$(find temp_extract -name "*.mat" 2>/dev/null | head -1)
            if [ -n "$MAT_FILES" ]; then
                mkdir -p "datasets/$FOLDER_NAME/training"
                find temp_extract -name "*.mat" -exec cp {} "datasets/$FOLDER_NAME/training/" \;
                echo "  ✓ $FOLDER_NAME extracted (flat structure)"
            else
                echo "  ⚠ $ZIP_FILE: Could not find training data inside"
            fi
        fi
        
        rm -rf temp_extract
    fi
}

# -------------------------------------------------------------------------
# Extract each dataset (handles various naming conventions)
# -------------------------------------------------------------------------

# EPFL - might be named EPFL.zip, epfl.zip, EPFL_dataset.zip, etc.
extract_dataset "EPFL" "EPFL"
extract_dataset "epfl" "EPFL"

# HCI_new - might be HCI_new.zip, HCInew.zip, HCI-new.zip, etc.
extract_dataset "HCI_new" "HCI_new"
extract_dataset "HCInew" "HCI_new"
extract_dataset "HCI-new" "HCI_new"
extract_dataset "hci_new" "HCI_new"

# HCI_old - might be HCI_old.zip, HCIold.zip, HCI-old.zip, etc.
extract_dataset "HCI_old" "HCI_old"
extract_dataset "HCIold" "HCI_old"
extract_dataset "HCI-old" "HCI_old"
extract_dataset "hci_old" "HCI_old"

# INRIA_Lytro - might be INRIA.zip, INRIA_Lytro.zip, Lytro.zip, etc.
extract_dataset "INRIA" "INRIA_Lytro"
extract_dataset "inria" "INRIA_Lytro"
extract_dataset "Lytro" "INRIA_Lytro"

# STFgantry/Stanford_Gantry - multiple naming conventions
extract_dataset "STFgantry" "STFgantry"
extract_dataset "Stanford" "STFgantry"
extract_dataset "stfgantry" "STFgantry"
extract_dataset "stanford_gantry" "STFgantry"
extract_dataset "Stanford_Gantry" "STFgantry"

# -------------------------------------------------------------------------
# Fix any remaining nested folder issues
# -------------------------------------------------------------------------
for ds in EPFL HCI_new HCI_old INRIA_Lytro STFgantry; do
    # Handle double-nested folders like datasets/EPFL/EPFL/training
    if [ -d "datasets/$ds/$ds/training" ]; then
        echo "  Fixing nested structure in $ds..."
        mv "datasets/$ds/$ds/"* "datasets/$ds/" 2>/dev/null || true
        rmdir "datasets/$ds/$ds" 2>/dev/null || true
    fi
    
    # Handle case where training folder ended up directly in datasets
    if [ -d "datasets/$ds/training/training" ]; then
        mv "datasets/$ds/training/training/"* "datasets/$ds/training/" 2>/dev/null || true
        rmdir "datasets/$ds/training/training" 2>/dev/null || true
    fi
done

# -------------------------------------------------------------------------
# Verify all datasets
# -------------------------------------------------------------------------
echo ""
DATASETS=("EPFL" "HCI_new" "HCI_old" "INRIA_Lytro" "STFgantry")
MISSING=0
for ds in "${DATASETS[@]}"; do
    if [ -d "datasets/$ds/training" ]; then
        COUNT=$(ls datasets/$ds/training/*.mat 2>/dev/null | wc -l)
        echo "  ✓ $ds ($COUNT scenes)"
    else
        echo "  ✗ $ds MISSING"
        MISSING=1
    fi
done


if [ $MISSING -eq 1 ]; then
    echo ""
    echo "WARNING: Some datasets are missing. Training will proceed with available data."
fi

# ============================================================================
# STEP 3: Setup Python Environment (RTX 5090 Compatible)
# ============================================================================
echo ""
echo "[4/8] Setting up Python environment..."

# CRITICAL: Force uninstall old PyTorch first (cu124 doesn't support RTX 5090)
echo "  Removing old PyTorch (if exists)..."
$PIP uninstall torch torchvision -y 2>/dev/null || true

# Install PyTorch Nightly with CUDA 12.8 for RTX 5090 (Blackwell sm_120)
echo "  Installing PyTorch Nightly (CUDA 12.8 for RTX 5090)..."
$PIP install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install mamba-ssm (optional, has fallback)
echo "  Installing mamba-ssm..."
$PIP install mamba-ssm causal-conv1d 2>/dev/null || echo "  (mamba-ssm failed, fallback will be used)"

# Install other dependencies
echo "  Installing other dependencies..."
$PIP install h5py einops tqdm fvcore scipy scikit-image imageio xlwt matplotlib ninja

echo "  ✓ Environment ready (PyTorch Nightly + CUDA 12.8)"

# ============================================================================
# STEP 4: Generate Training Patches
# ============================================================================
echo ""
echo "[5/8] Generating training patches..."

# Check if patches already exist
if [ -d "data_for_training/SR_5x5_4x" ] && [ "$(ls -A data_for_training/SR_5x5_4x 2>/dev/null)" ]; then
    echo "  Training patches found, skipping generation..."
else
    echo "  Generating training patches (~15 min)..."
    $PYTHON Generate_Data_for_Training.py --angRes 5 --scale_factor 4
fi

if [ -d "data_for_test/SR_5x5_4x" ] && [ "$(ls -A data_for_test/SR_5x5_4x 2>/dev/null)" ]; then
    echo "  Test patches found, skipping generation..."
else
    echo "  Generating test patches..."
    $PYTHON Generate_Data_for_Test.py --angRes 5 --scale_factor 4
fi

# CRITICAL: Create symlink for directory name mismatch
# Generate_Data_for_Training.py saves to "data_for_training/" (full word)
# But option.py/train.py defaults to "data_for_train/" (abbreviated in some configs)
# We create symlink: data_for_train -> data_for_training
if [ -d "data_for_training" ] && [ ! -e "data_for_train" ]; then
    echo "  Creating symlink: data_for_train -> data_for_training"
    ln -s data_for_training data_for_train
fi

echo "  ✓ Patches ready"

# ============================================================================
# STEP 5: Verify Model Loads
# ============================================================================
echo ""
echo "[6/8] Verifying model..."

$PYTHON -c "
import sys
import os
sys.path.insert(0, os.getcwd())
from model.SR.MyEfficientLFNetV5 import get_model, count_parameters

class Args:
    angRes_in = 5
    scale_factor = 4
    use_macpi = True
    use_tta = False

model = get_model(Args())
params = count_parameters(model)
print(f'  Parameters: {params:,} ({params/1e6:.3f}M)')
print(f'  Budget: {params/1e6*100:.1f}% of 1M limit')
assert params < 1_000_000, 'FAIL: Exceeds 1M params!'
print('  ✓ Model verified (<1M params)')
"

# ============================================================================
# STEP 6: Training
# ============================================================================
echo ""
echo "[7/8] Starting training..."
echo "  Model: MyEfficientLFNetV5"
echo "  Epochs: 80"
echo "  Batch Size: 8"
echo "  Estimated Time: 3-4 hours"
echo ""

$PYTHON train.py \
    --model_name MyEfficientLFNetV5 \
    --angRes 5 \
    --scale_factor 4 \
    --batch_size 8 \
    --lr 2e-4 \
    --epoch 80 \
    --device cuda:0 \
    --num_workers 8

echo "  ✓ Training complete"

# ============================================================================
# STEP 7: Inference & Submission
# ============================================================================
echo ""
echo "[8/8] Running inference..."

# Find the best checkpoint
CKPT="log/SR/5x5_4x_MyEfficientLFNetV5/ALL/checkpoints/Best.pth"
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls -t log/SR/5x5_4x_MyEfficientLFNetV5/ALL/checkpoints/*.pth 2>/dev/null | head -1)
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "  ERROR: No checkpoint found!"
    exit 1
fi

echo "  Using checkpoint: $CKPT"

$PYTHON inference.py \
    --model_name MyEfficientLFNetV5 \
    --angRes 5 \
    --scale_factor 4 \
    --path_pre_pth "$CKPT" \
    --device cuda:0

echo ""
echo "=============================================="
echo "✓ PIPELINE COMPLETE"
echo "=============================================="
echo ""
echo "Results: Results/SR/5x5_4x_MyEfficientLFNetV5/"
echo ""
echo "To create submission zip:"
echo "  cd Results/SR/5x5_4x_MyEfficientLFNetV5/"
echo "  zip -r submission.zip *"
