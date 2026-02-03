#!/bin/bash
# ============================================================================
# NTIRE 2026 LF-SR Challenge - Track 2 (Efficiency) Setup Script
# ============================================================================
# This script sets up the complete environment for training MyEfficientLFNet
# on a clean VM with a strong GPU (e.g., RTX 3090, A100, V100)
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# After running, manually download datasets from OneDrive and run:
#   ./prepare_data.sh
# ============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "NTIRE 2026 LF-SR Track 2 - Environment Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# STEP 1: System Check
# ============================================================================
echo -e "\n${YELLOW}[1/6] Checking system requirements...${NC}"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}✗ No NVIDIA GPU detected. This setup requires a CUDA-capable GPU.${NC}"
    echo "  If running on a VM, ensure GPU passthrough is enabled."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

# ============================================================================
# STEP 2: Create Virtual Environment
# ============================================================================
echo -e "\n${YELLOW}[2/6] Creating Python virtual environment...${NC}"

# Install venv if not present
if ! python3 -m venv --help &> /dev/null; then
    echo "Installing python3-venv..."
    sudo apt-get update && sudo apt-get install -y python3-venv
fi

# Create venv
VENV_DIR="./venv_lfsr"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
    echo -e "${GREEN}✓ Virtual environment created at $VENV_DIR${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate venv
source $VENV_DIR/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# ============================================================================
# STEP 3: Install PyTorch with CUDA
# ============================================================================
echo -e "\n${YELLOW}[3/6] Installing PyTorch with CUDA support...${NC}"

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "Detected CUDA version: $CUDA_VERSION"
fi

# Install PyTorch (CUDA 11.8 - widely compatible)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo -e "${GREEN}✓ PyTorch installed successfully${NC}"

# ============================================================================
# STEP 4: Install Project Dependencies
# ============================================================================
echo -e "\n${YELLOW}[4/6] Installing project dependencies...${NC}"

pip install \
    einops \
    h5py \
    scipy \
    numpy \
    imageio \"D:\DownloadsD\OneDrive_1_1-30-2026.zip"
    xlwt \
    tqdm \
    fvcore \
    matplotlib \
    Pillow

echo -e "${GREEN}✓ All dependencies installed${NC}"

# ============================================================================
# STEP 5: Create Directory Structure
# ============================================================================
echo -e "\n${YELLOW}[5/6] Creating directory structure...${NC}"

# Create required directories
mkdir -p datasets/EPFL/training
mkdir -p datasets/HCI_new/training
mkdir -p datasets/HCI_old/training
mkdir -p datasets/INRIA_Lytro/training
mkdir -p datasets/Stanford_Gantry/training
mkdir -p datasets/NTIRE_Val_Real/inference
mkdir -p datasets/NTIRE_Val_Synth/inference
mkdir -p data_for_training
mkdir -p data_for_inference
mkdir -p log
mkdir -p pth

echo -e "${GREEN}✓ Directory structure created${NC}"

# ============================================================================
# STEP 6: Verify Model
# ============================================================================
echo -e "\n${YELLOW}[6/6] Verifying MyEfficientLFNet model...${NC}"

python3 -c "
import sys
sys.path.insert(0, '.')
from model.SR.MyEfficientLFNet import get_model
import torch

class Args:
    angRes_in = 5
    scale_factor = 4

model = get_model(Args())
params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {params:,}')
print(f'Parameter limit:  1,000,000')
print(f'Status: {\"PASS\" if params < 1_000_000 else \"FAIL\"}')"

echo -e "${GREEN}✓ Model verification complete${NC}"

# ============================================================================
# Final Instructions
# ============================================================================
echo ""
echo "=============================================="
echo -e "${GREEN}Environment setup complete!${NC}"
echo "=============================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. DOWNLOAD TRAINING DATA (manual - requires browser):"
echo "   Open: https://stuxidianeducn-my.sharepoint.com/personal/zyliang_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzyliang%5Fstu%5Fxidian%5Fedu%5Fcn%2FDocuments%2Fdatasets&ga=1"
echo ""
echo "   Download each folder and extract to:"
echo "   - datasets/EPFL/training/"
echo "   - datasets/HCI_new/training/"
echo "   - datasets/HCI_old/training/"
echo "   - datasets/INRIA_Lytro/training/"
echo "   - datasets/Stanford_Gantry/training/"
echo ""
echo "2. PREPARE DATA (after downloading):"
echo "   source venv_lfsr/bin/activate"
echo "   ./prepare_data.sh"
echo ""
echo "3. TRAIN MODEL:"
echo "   source venv_lfsr/bin/activate"
echo "   ./train.sh"
echo ""
echo "4. ACTIVATE ENVIRONMENT (for manual commands):"
echo "   source venv_lfsr/bin/activate"
echo ""
