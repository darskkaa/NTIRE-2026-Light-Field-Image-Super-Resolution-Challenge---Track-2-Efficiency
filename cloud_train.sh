#!/bin/bash
# ============================================================================
# NTIRE 2026 Track 2 - Complete Training Pipeline
# Cloud GPU: 32GB VRAM, CUDA 12.8, AMD EPYC 48-Core
# Estimated Cost: ~$1.50 total ($0.34/hr × 4.5 hours)
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "NTIRE 2026 Track 2 - Training Pipeline"
echo "=============================================="

# ============================================================================
# STEP 1: Clone Repository
# ============================================================================
echo ""
echo "[1/8] Cloning BasicLFSR repository..."
if [ ! -d "BasicLFSR" ]; then
    git clone https://github.com/ZhengyuLiang24/BasicLFSR.git
fi
cd BasicLFSR

# ============================================================================
# STEP 2: Environment Setup
# ============================================================================
echo ""
echo "[2/8] Setting up environment..."

# Check if conda exists
# Check if conda exists or needs installation
if [ -d "$HOME/miniconda" ]; then
    echo "  Miniconda found at $HOME/miniconda, initializing..."
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
elif [ -d "/opt/conda" ]; then
    echo "  Miniconda found at /opt/conda, initializing..."
    eval "$(/opt/conda/bin/conda shell.bash hook)"
elif ! command -v conda &> /dev/null; then
    echo "  Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    rm miniconda.sh
fi

# Create environment
conda create -n lfsr python=3.10 -y 2>/dev/null || true
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lfsr

# Install PyTorch with CUDA 12.4 (compatible with 12.8) - VERBOSE
echo "  Downloading PyTorch (~1.3GB)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install mamba-ssm (REQUIRED) - VERBOSE
echo "  Installing mamba-ssm (compiling CUDA kernels)..."
pip install mamba-ssm causal-conv1d

# Install other deps - VERBOSE
echo "  Installing dependencies..."
pip install h5py einops tqdm fvcore gdown scipy scikit-image imageio xlwt matplotlib ninja packaging

echo "✓ Environment ready"

# ============================================================================
# STEP 3: Download Datasets
# ============================================================================
echo ""
echo "[3/8] Downloading datasets..."

mkdir -p datasets
cd datasets

# EPFL
if [ ! -d "EPFL" ]; then
    echo "  Downloading EPFL (450MB)..."
    gdown --fuzzy "https://drive.google.com/file/d/19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN/view" -O EPFL.zip
    echo "  Extracting EPFL..."
    unzip -o EPFL.zip -d EPFL_tmp && mv EPFL_tmp/*/* EPFL/ 2>/dev/null || mv EPFL_tmp/* EPFL/
    rm -rf EPFL_tmp EPFL.zip
fi

# HCI_new
if [ ! -d "HCI_new" ]; then
    echo "  Downloading HCI_new (1.8GB)..."
    gdown --fuzzy "https://drive.google.com/file/d/1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9/view" -O HCI_new.zip
    echo "  Extracting HCI_new..."
    unzip -o HCI_new.zip -d HCI_new_tmp && mv HCI_new_tmp/*/* HCI_new/ 2>/dev/null || mv HCI_new_tmp/* HCI_new/
    rm -rf HCI_new_tmp HCI_new.zip
fi

# HCI_old
if [ ! -d "HCI_old" ]; then
    echo "  Downloading HCI_old (500MB)..."
    gdown --fuzzy "https://drive.google.com/file/d/1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG/view" -O HCI_old.zip
    echo "  Extracting HCI_old..."
    unzip -o HCI_old.zip -d HCI_old_tmp && mv HCI_old_tmp/*/* HCI_old/ 2>/dev/null || mv HCI_old_tmp/* HCI_old/
    rm -rf HCI_old_tmp HCI_old.zip
fi

# INRIA_Lytro
if [ ! -d "INRIA_Lytro" ]; then
    echo "  Downloading INRIA_Lytro (1.5GB)..."
    gdown --fuzzy "https://drive.google.com/file/d/1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ/view" -O INRIA.zip
    echo "  Extracting INRIA_Lytro..."
    unzip -o INRIA.zip -d INRIA_tmp && mv INRIA_tmp/*/* INRIA_Lytro/ 2>/dev/null || mv INRIA_tmp/* INRIA_Lytro/
    rm -rf INRIA_tmp INRIA.zip
fi

# STFgantry
if [ ! -d "STFgantry" ]; then
    echo "  Downloading STFgantry (3.2GB)..."
    gdown --fuzzy "https://drive.google.com/file/d/1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3/view" -O Stanford.zip
    echo "  Extracting STFgantry..."
    unzip -o Stanford.zip -d Stanford_tmp && mv Stanford_tmp/*/* STFgantry/ 2>/dev/null || mv Stanford_tmp/* STFgantry/
    rm -rf Stanford_tmp Stanford.zip
fi

cd ..
echo "✓ Datasets downloaded"

# ============================================================================
# STEP 4: Generate Training Patches
# ============================================================================
echo ""
# Check if patches exist (BasicLFSR/data_for_train/SR_5x5_4x)
if [ -d "data_for_train/SR_5x5_4x" ] && [ "$(ls -A data_for_train/SR_5x5_4x)" ]; then
    echo "  Training patches found, skipping generation..."
else
    echo "  Generating training patches..."
    python Generate_Data_for_Training.py --angRes 5 --scale_factor 4
fi

if [ -d "data_for_test/SR_5x5_4x" ] && [ "$(ls -A data_for_test/SR_5x5_4x)" ]; then
    echo "  Test patches found, skipping generation..."
else
    echo "  Generating test patches..."
    python Generate_Data_for_Test.py --angRes 5 --scale_factor 4
fi

echo "✓ Training patches generated"

# ============================================================================
# STEP 5: Verify Model
# ============================================================================
echo ""
echo "[5/8] Verifying model..."
# We are already in BasicLFSR directory from step 1

python -c "
import torch
import sys
import os
sys.path.append(os.getcwd())
from model.SR.MyEfficientLFNetV5 import get_model, count_parameters

class Args:
    angRes_in = 5
    scale_factor = 4
    use_macpi = True
    use_tta = False  # TTA DISABLED

model = get_model(Args())
params = count_parameters(model)

print(f'Parameters: {params:,} ({params/1e6:.3f}M)')
print(f'Budget Used: {params/1e6*100:.1f}% of 1M')

# Test forward pass
model.eval()
x = torch.randn(1, 1, 160, 160)
with torch.no_grad():
    out = model(x)
print(f'Forward: {x.shape} -> {out.shape}')

assert params < 1_000_000, 'FAIL: Exceeds 1M params!'
assert out.shape == (1, 1, 640, 640), 'FAIL: Wrong output shape!'
print('✓ Model verified')
"

# ============================================================================
# STEP 6: Training
# ============================================================================
echo ""
echo "[6/8] Starting training..."
echo "  Model: MyEfficientLFNetV5"
echo "  Epochs: 80"
echo "  Batch Size: 8 (optimized for 32GB VRAM)"
echo "  Estimated Time: 3.5-4.5 hours"
echo ""

python train.py \
    --model_name MyEfficientLFNetV5 \
    --angRes 5 \
    --scale_factor 4 \
    --batch_size 8 \
    --lr 2e-4 \
    --epoch 80 \
    --device cuda:0 \
    --num_workers 8

echo "✓ Training complete"

# ============================================================================
# STEP 7: Inference
# ============================================================================
echo ""
echo "[7/8] Running inference..."

# Find best checkpoint
CKPT=$(ls -t log/SR/5x5_4x_MyEfficientLFNetV5/ALL/checkpoints/*.pth | head -1)
echo "Using checkpoint: $CKPT"

# Run on validation set
python inference.py \
    --model_name MyEfficientLFNetV5 \
    --angRes 5 \
    --scale_factor 4 \
    --path_pre_pth "$CKPT" \
    --device cuda:0

echo "✓ Inference complete"

# ============================================================================
# STEP 8: Create Submission
# ============================================================================
echo ""
echo "[8/8] Creating submission..."

cd Results/SR/5x5_4x_MyEfficientLFNetV5/

# Validate before zipping
python ../../../validate_submission.py . || echo "Warning: Validation issues"

# Create submission
zip -r submission.zip Real Synth

echo ""
echo "=============================================="
echo "✓ PIPELINE COMPLETE"
echo "=============================================="
echo ""
echo "Submission: $(pwd)/submission.zip"
echo "Upload to: https://codalab.lisn.upsaclay.fr/"
echo ""
echo "Expected Results:"
echo "  - PSNR (val): 31-32 dB"
echo "  - PSNR (test): 30-31 dB"
echo "  - Parameters: ~550K (55% of budget)"
echo "  - FLOPs: ~12G (60% of budget)"
echo "=============================================="
