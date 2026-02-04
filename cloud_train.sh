#!/bin/bash
# ============================================================================
# NTIRE 2026 Track 2 - FINAL Training Pipeline (Bulletproof Edition)
# Cloud GPU: RTX 5090 32GB VRAM, CUDA 12.8
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "NTIRE 2026 Track 2 - Training Pipeline"
echo "=============================================="

# ============================================================================
# STEP 1: Verify We're In The Right Place
# ============================================================================
echo ""
echo "[1/7] Checking project structure..."

if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found. Are you in the project root?"
    exit 1
fi

if [ ! -d "model/SR" ]; then
    echo "ERROR: model/SR directory not found."
    exit 1
fi

echo "  ✓ Project structure OK"

# ============================================================================
# STEP 2: Extract Any Remaining Zip Files
# ============================================================================
echo ""
echo "[2/7] Checking datasets..."

mkdir -p datasets

# Extract INRIA if zip exists but folder doesn't
if [ -f "INRIA.zip" ] && [ ! -d "datasets/INRIA_Lytro" ]; then
    echo "  Extracting INRIA.zip (~8.5GB)..."
    unzip -q -o INRIA.zip -d datasets/
    # Handle nested folder structure
    if [ -d "datasets/INRIA_Lytro/INRIA_Lytro" ]; then
        mv datasets/INRIA_Lytro/INRIA_Lytro/* datasets/INRIA_Lytro/
        rmdir datasets/INRIA_Lytro/INRIA_Lytro 2>/dev/null || true
    fi
fi

# Extract STFgantry if zip exists but folder doesn't
if [ -f "STFgantry.zip" ] && [ ! -d "datasets/STFgantry" ]; then
    echo "  Extracting STFgantry.zip (~4.2GB)..."
    mkdir -p datasets/STFgantry
    unzip -q -o STFgantry.zip -d temp_stf/
    # Handle nested folder structure
    mv temp_stf/*/* datasets/STFgantry/ 2>/dev/null || mv temp_stf/* datasets/STFgantry/
    rm -rf temp_stf
fi

# Check all 5 datasets exist
DATASETS=("EPFL" "HCI_new" "HCI_old" "INRIA_Lytro" "STFgantry")
for ds in "${DATASETS[@]}"; do
    if [ -d "datasets/$ds" ]; then
        echo "  ✓ $ds found"
    else
        echo "  ✗ $ds MISSING - Please download it first!"
    fi
done

# ============================================================================
# STEP 3: Setup Python Environment
# ============================================================================
echo ""
echo "[3/7] Setting up Python environment..."

# Use venv if it exists, otherwise use system python
if [ -d "/venv/lfsr" ]; then
    PYTHON="/venv/lfsr/bin/python"
    PIP="/venv/lfsr/bin/pip"
    echo "  Using venv: /venv/lfsr"
else
    PYTHON="python3"
    PIP="pip3"
    echo "  Using system Python"
fi

# Install PyTorch Nightly for RTX 5090 (Blackwell sm_120)
echo "  Installing PyTorch Nightly (CUDA 12.8 for RTX 5090)..."
$PIP install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install mamba-ssm
echo "  Installing mamba-ssm..."
$PIP install mamba-ssm causal-conv1d 2>/dev/null || echo "  (mamba-ssm install failed, fallback will be used)"

# Install other dependencies
echo "  Installing other dependencies..."
$PIP install h5py einops tqdm fvcore scipy scikit-image imageio xlwt matplotlib ninja

echo "  ✓ Environment ready"

# ============================================================================
# STEP 4: Generate Training Patches
# ============================================================================
echo ""
echo "[4/7] Generating training patches..."

# Check if patches already exist
if [ -d "data_for_train/SR_5x5_4x" ] && [ "$(ls -A data_for_train/SR_5x5_4x 2>/dev/null)" ]; then
    echo "  Training patches found, skipping..."
else
    echo "  Generating training patches (this takes ~15 min)..."
    $PYTHON Generate_Data_for_Training.py --angRes 5 --scale_factor 4
fi

if [ -d "data_for_test/SR_5x5_4x" ] && [ "$(ls -A data_for_test/SR_5x5_4x 2>/dev/null)" ]; then
    echo "  Test patches found, skipping..."
else
    echo "  Generating test patches..."
    $PYTHON Generate_Data_for_Test.py --angRes 5 --scale_factor 4
fi

# CRITICAL: Create symlink for directory name mismatch
# Generate_Data_for_Training.py outputs to data_for_train
# But option.py expects data_for_training
if [ -d "data_for_train" ] && [ ! -e "data_for_training" ]; then
    echo "  Creating symlink: data_for_training -> data_for_train"
    ln -s data_for_train data_for_training
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
print('  ✓ Model verified')
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
    --num_workers 8 \
    --path_for_train ./data_for_train/

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

if [ -z "$CKPT" ]; then
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
echo "Results saved to: Results/SR/5x5_4x_MyEfficientLFNetV5/"
echo "To create submission: cd Results/SR/5x5_4x_MyEfficientLFNetV5/ && zip -r submission.zip *"
