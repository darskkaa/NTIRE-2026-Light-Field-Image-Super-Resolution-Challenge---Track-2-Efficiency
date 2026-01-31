#!/bin/bash
#============================================================
# RTX 5090 ULTRA-FAST AUTO-SETUP
# Complete One-Command Setup for High-End GPU
#============================================================

set -e

echo "============================================================"
echo "🚀 RTX 5090 ULTRA-FAST AUTO-SETUP"
echo "============================================================"
echo ""

# Step 1: System Dependencies
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip unzip wget git > /dev/null 2>&1
echo "✓ System dependencies installed"

# Step 2: Virtual Environment
echo "[2/6] Creating Python virtual environment..."
if [ ! -d "venv_lfsr" ]; then
    python3 -m venv venv_lfsr
fi
source venv_lfsr/bin/activate
echo "✓ Virtual environment activated"

# Step 3: Install PyTorch (CUDA 12.4 for RTX 5090)
echo "[3/6] Installing PyTorch with CUDA 12.4..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo "✓ PyTorch installed"

# Step 4: Install dependencies
echo "[4/6] Installing project dependencies..."
pip install --quiet -r requirements.txt
pip install --quiet fvcore einops h5py scipy xlwt
echo "✓ All dependencies installed"

# Step 5: Verify GPU
echo "[5/6] Verifying GPU..."
python -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Step 6: Verify Model
echo "[6/6] Verifying model efficiency..."
python check_efficiency.py --model_name MyEfficientLFNet

echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""

# Check for datasets
if [ -d "datasets/EPFL" ] && [ -d "datasets/HCI_new" ]; then
    echo "✓ Datasets found! Generating training patches..."
    python Generate_Data_for_Training.py --angRes 5 --scale_factor 4
    echo ""
    echo "🎯 Ready to train! Run: ./train_5090.sh"
else
    echo "⚠ Datasets not found in ./datasets/"
    echo ""
    echo "📁 Upload your datasets to ./datasets/ and run:"
    echo "   ./prepare_data.sh && ./train_5090.sh"
fi

echo ""
