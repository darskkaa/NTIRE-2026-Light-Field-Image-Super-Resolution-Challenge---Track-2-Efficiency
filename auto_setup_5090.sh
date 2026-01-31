#!/bin/bash
#============================================================
# RTX 5090 ULTRA-FAST AUTO-SETUP
# Complete One-Command Setup for High-End GPU
# WITH GOOGLE DRIVE DATASET DOWNLOAD
# UPDATED: PyTorch Nightly for Blackwell (sm_120) support
#============================================================

set -e

GDRIVE_LINK="https://drive.google.com/file/d/1WUY0WuJsfQWA4ibeLLAcskDJR6c4Es4B/view?usp=sharing"
GDRIVE_FILE_ID="1WUY0WuJsfQWA4ibeLLAcskDJR6c4Es4B"

echo "============================================================"
echo "🚀 RTX 5090 ULTRA-FAST AUTO-SETUP (Blackwell Edition)"
echo "============================================================"
echo ""

# Step 1: System Dependencies
echo "[1/8] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip unzip wget git > /dev/null 2>&1
echo "✓ System dependencies installed"

# Step 2: Virtual Environment
echo "[2/8] Creating Python virtual environment..."
if [ ! -d "venv_lfsr" ]; then
    python3 -m venv venv_lfsr
fi
source venv_lfsr/bin/activate
echo "✓ Virtual environment activated"

# Step 3: Install PyTorch NIGHTLY (Required for RTX 5090 Blackwell sm_120)
echo "[3/8] Installing PyTorch Nightly with CUDA 12.6 (sm_120 support)..."
pip install --quiet --upgrade pip
pip install --quiet --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
echo "✓ PyTorch Nightly installed (Blackwell sm_120 supported)"

# Step 4: Install dependencies (including scikit-image)
echo "[4/8] Installing project dependencies..."
pip install --quiet -r requirements.txt
pip install --quiet fvcore einops h5py scipy xlwt gdown scikit-image imageio
echo "✓ All dependencies installed"

# Step 5: Verify GPU
echo "[5/8] Verifying GPU..."
python -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'   Compute Capability: {torch.cuda.get_device_capability(0)}')
"

# Step 6: Download Dataset from Google Drive
echo "[6/8] Downloading dataset from Google Drive..."
echo "      File ID: ${GDRIVE_FILE_ID}"
cd datasets 2>/dev/null || mkdir -p datasets && cd datasets

if [ ! -d "EPFL" ]; then
    if [ ! -f "dataset.zip" ]; then
        gdown --fuzzy "${GDRIVE_LINK}" -O dataset.zip
    fi
    echo "      Extracting..."
    unzip -q dataset.zip
    rm -f dataset.zip
    echo "✓ Dataset downloaded and extracted"
else
    echo "✓ Dataset already exists, skipping download"
fi
cd ..

# Step 7: Generate Training Patches
echo "[7/8] Generating training patches..."
python Generate_Data_for_Training.py --angRes 5 --scale_factor 4
echo "✓ Training patches generated"

# Step 8: Verify Model
echo "[8/8] Verifying model efficiency..."
python check_efficiency.py --model_name MyEfficientLFNet

echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "🎯 Ready to train! Run:"
echo "   ./train_5090.sh"
echo ""
echo "Expected training time: ~2 hours on RTX 5090"
echo ""
