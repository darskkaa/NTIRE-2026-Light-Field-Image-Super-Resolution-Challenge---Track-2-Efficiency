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
echo "[1/7] Checking project structure..."

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
# STEP 2: Extract Datasets (if zips exist)
# ============================================================================
echo ""
echo "[2/7] Checking datasets..."

mkdir -p datasets

# Extract INRIA if zip exists but folder doesn't have training data
if [ -f "INRIA.zip" ] && [ ! -d "datasets/INRIA_Lytro/training" ]; then
    echo "  Extracting INRIA.zip (~8.5GB)..."
    unzip -q -o INRIA.zip -d datasets/
    # Handle nested folder structure
    if [ -d "datasets/INRIA_Lytro/INRIA_Lytro" ]; then
        mv datasets/INRIA_Lytro/INRIA_Lytro/* datasets/INRIA_Lytro/ 2>/dev/null || true
        rmdir datasets/INRIA_Lytro/INRIA_Lytro 2>/dev/null || true
    fi
    echo "  ✓ INRIA extracted"
fi

# Extract STFgantry if zip exists but folder doesn't have training data
if [ -f "STFgantry.zip" ] && [ ! -d "datasets/STFgantry/training" ]; then
    echo "  Extracting STFgantry.zip (~4.2GB)..."
    mkdir -p datasets/STFgantry
    unzip -q -o STFgantry.zip -d temp_stf/
    # Handle nested folder structure - move contents up
    find temp_stf -name "training" -type d -exec cp -r {} datasets/STFgantry/ \; 2>/dev/null || true
    find temp_stf -name "test" -type d -exec cp -r {} datasets/STFgantry/ \; 2>/dev/null || true
    rm -rf temp_stf
    echo "  ✓ STFgantry extracted"
fi

# Check all 5 datasets
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
echo "[3/7] Setting up Python environment..."

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
echo "[4/7] Generating training patches..."

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
echo "[5/7] Verifying model..."

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
echo "[6/7] Starting training..."
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
echo "[7/7] Running inference..."

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
