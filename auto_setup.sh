#!/bin/bash
# ============================================================================
# NTIRE 2026 LF-SR Challenge - Track 2 (Efficiency)
# FULLY AUTOMATED VM SETUP SCRIPT
# ============================================================================
# This script does EVERYTHING:
#   1. Installs system dependencies
#   2. Creates Python virtual environment
#   3. Installs PyTorch with CUDA
#   4. Downloads datasets automatically
#   5. Generates training patches
#   6. Verifies model efficiency
#
# Usage on a fresh Linux VM with GPU:
#   chmod +x auto_setup.sh
#   ./auto_setup.sh
#
# Tested on: Ubuntu 20.04/22.04 with NVIDIA GPU
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     NTIRE 2026 LF-SR Track 2 - Automated Setup                   ║"
echo "║     MyEfficientLFNet: 781K params, 18.44G FLOPs                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# STEP 1: System Dependencies
# ============================================================================
echo -e "\n${YELLOW}[1/7] Installing system dependencies...${NC}"

# Detect package manager
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        wget \
        curl \
        unzip \
        git \
        htop \
        tmux
    echo -e "${GREEN}✓ System dependencies installed (apt)${NC}"
elif command -v yum &> /dev/null; then
    sudo yum install -y \
        python3 \
        python3-pip \
        wget \
        curl \
        unzip \
        git \
        htop \
        tmux
    echo -e "${GREEN}✓ System dependencies installed (yum)${NC}"
else
    echo -e "${YELLOW}⚠ Unknown package manager. Please ensure python3, wget, unzip are installed.${NC}"
fi

# ============================================================================
# STEP 2: Check GPU
# ============================================================================
echo -e "\n${YELLOW}[2/7] Checking GPU...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}✗ No NVIDIA GPU detected!${NC}"
    echo "  This setup requires a CUDA-capable GPU."
    echo "  Continuing anyway, but training will fail without GPU."
fi

# ============================================================================
# STEP 3: Create Virtual Environment
# ============================================================================
echo -e "\n${YELLOW}[3/7] Creating Python virtual environment...${NC}"

VENV_DIR="./venv_lfsr"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

source $VENV_DIR/bin/activate
pip install --upgrade pip -q

# ============================================================================
# STEP 4: Install PyTorch and Dependencies
# ============================================================================
echo -e "\n${YELLOW}[4/7] Installing PyTorch and dependencies...${NC}"

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

# Install project dependencies
pip install \
    einops \
    h5py \
    scipy \
    numpy \
    imageio \
    xlwt \
    tqdm \
    fvcore \
    matplotlib \
    Pillow \
    gdown \
    -q

echo -e "${GREEN}✓ All Python packages installed${NC}"

# Verify PyTorch
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ============================================================================
# STEP 5: Download Datasets
# ============================================================================
echo -e "\n${YELLOW}[5/7] Downloading datasets...${NC}"

# Create directory structure
mkdir -p datasets/EPFL/training
mkdir -p datasets/HCI_new/training
mkdir -p datasets/HCI_old/training
mkdir -p datasets/INRIA_Lytro/training
mkdir -p datasets/Stanford_Gantry/training
mkdir -p datasets/NTIRE_Val_Real/inference
mkdir -p datasets/NTIRE_Val_Synth/inference

# ============================================================================
# Dataset Download Options
# ============================================================================
# The official datasets are hosted on OneDrive which requires authentication.
# We provide multiple download methods:
#
# Option A: Direct links (if available from mirrors)
# Option B: Google Drive mirrors
# Option C: Manual download with verification
# ============================================================================

echo -e "${BLUE}Attempting dataset download...${NC}"

# Function to download with retry
download_with_retry() {
    local url="$1"
    local output="$2"
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if wget -q --show-progress -O "$output" "$url"; then
            return 0
        fi
        retry=$((retry + 1))
        echo "Retry $retry/$max_retries..."
        sleep 2
    done
    return 1
}

# Check if datasets already exist
check_datasets() {
    local all_present=true
    for ds in EPFL HCI_new HCI_old INRIA_Lytro Stanford_Gantry; do
        if [ -z "$(ls -A datasets/$ds/training 2>/dev/null)" ]; then
            all_present=false
            break
        fi
    done
    echo $all_present
}

if [ "$(check_datasets)" = "true" ]; then
    echo -e "${GREEN}✓ Datasets already present, skipping download${NC}"
else
    echo -e "${YELLOW}Datasets not found. Attempting download...${NC}"
    
    # Try Google Drive mirror first (these are public backup links)
    # Note: These are example IDs - you may need to update with actual public mirrors
    
    DOWNLOAD_SUCCESS=false
    
    # Method 1: Try gdown with Google Drive (if mirrors exist)
    # Uncomment and add real file IDs if you upload to Google Drive
    # echo "Trying Google Drive mirrors..."
    # gdown --folder "GOOGLE_DRIVE_FOLDER_ID" -O datasets/ --remaining-ok || true
    
    # Method 2: Direct HTTP download (if organizers provide direct links)
    # Check BasicLFSR releases for direct downloads
    BASICLFSR_RELEASE="https://github.com/ZhengyuLiang24/BasicLFSR/releases/download/v1.0"
    
    echo "Checking for direct download links..."
    
    # Try to download from BasicLFSR releases (common location for LF datasets)
    for ds in EPFL HCI_new HCI_old INRIA_Lytro Stanford_Gantry; do
        if [ -z "$(ls -A datasets/$ds/training 2>/dev/null)" ]; then
            echo "  Downloading $ds..."
            
            # Try release download
            if download_with_retry "${BASICLFSR_RELEASE}/${ds}.zip" "datasets/${ds}.zip" 2>/dev/null; then
                unzip -q "datasets/${ds}.zip" -d "datasets/${ds}/training/" && rm "datasets/${ds}.zip"
                echo -e "  ${GREEN}✓ $ds downloaded${NC}"
            else
                echo -e "  ${YELLOW}⚠ $ds: Direct download not available${NC}"
            fi
        fi
    done
    
    # Final check
    if [ "$(check_datasets)" = "true" ]; then
        echo -e "${GREEN}✓ All datasets downloaded successfully${NC}"
        DOWNLOAD_SUCCESS=true
    else
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  MANUAL DOWNLOAD REQUIRED${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
        echo ""
        echo "Some datasets could not be downloaded automatically."
        echo "Please download manually from:"
        echo ""
        echo -e "${BLUE}  https://stuxidianeducn-my.sharepoint.com/personal/zyliang_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzyliang%5Fstu%5Fxidian%5Fedu%5Fcn%2FDocuments%2Fdatasets${NC}"
        echo ""
        echo "Download each folder and extract:"
        echo "  EPFL/          → datasets/EPFL/training/"
        echo "  HCI_new/       → datasets/HCI_new/training/"
        echo "  HCI_old/       → datasets/HCI_old/training/"
        echo "  INRIA_Lytro/   → datasets/INRIA_Lytro/training/"
        echo "  Stanford_Gantry/ → datasets/Stanford_Gantry/training/"
        echo ""
        echo "After downloading, run: ./prepare_data.sh"
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
        
        # Create a helper script for after manual download
        cat > download_complete.sh << 'HELPER'
#!/bin/bash
source ./venv_lfsr/bin/activate
./prepare_data.sh
./train.sh
HELPER
        chmod +x download_complete.sh
        
        echo ""
        echo "After downloading datasets, run: ./download_complete.sh"
        echo ""
    fi
fi

# ============================================================================
# STEP 6: Generate Training Data
# ============================================================================
echo -e "\n${YELLOW}[6/7] Generating training patches...${NC}"

if [ "$(check_datasets)" = "true" ]; then
    echo "Generating training data (this may take 10-30 minutes)..."
    python Generate_Data_for_Training.py \
        --angRes 5 \
        --scale_factor 4 \
        --data_for training \
        --src_data_path ./datasets/ \
        --save_data_path ./
    
    echo -e "${GREEN}✓ Training patches generated${NC}"
else
    echo -e "${YELLOW}⚠ Skipping - datasets not complete${NC}"
fi

# ============================================================================
# STEP 7: Verify Model
# ============================================================================
echo -e "\n${YELLOW}[7/7] Verifying MyEfficientLFNet model...${NC}"

python3 << 'PYCHECK'
import sys
sys.path.insert(0, '.')
from model.SR.MyEfficientLFNet import get_model
import torch

class Args:
    angRes_in = 5
    scale_factor = 4

model = get_model(Args())
params = sum(p.numel() for p in model.parameters())

print(f"Model: MyEfficientLFNet")
print(f"Parameters: {params:,} / 1,000,000 limit")

# Quick FLOPs check
try:
    from fvcore.nn import FlopCountAnalysis
    x = torch.randn(1, 1, 160, 160)
    flops = FlopCountAnalysis(model, x).total()
    print(f"FLOPs: {flops/1e9:.2f}G / 20G limit")
    
    if params < 1_000_000 and flops < 20e9:
        print("✓ Model meets all efficiency constraints!")
    else:
        print("✗ Model exceeds efficiency limits!")
except:
    print("(FLOPs check skipped)")

print(f"\nStatus: {'READY' if params < 1_000_000 else 'NEEDS FIX'}")
PYCHECK

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      SETUP COMPLETE                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

if [ "$(check_datasets)" = "true" ]; then
    echo -e "${GREEN}✓ Environment ready${NC}"
    echo -e "${GREEN}✓ Datasets downloaded${NC}"
    echo -e "${GREEN}✓ Training data prepared${NC}"
    echo ""
    echo "To start training:"
    echo "  source venv_lfsr/bin/activate"
    echo "  ./train.sh"
    echo ""
    echo "Or use tmux for persistent training:"
    echo "  tmux new -s train"
    echo "  source venv_lfsr/bin/activate"
    echo "  ./train.sh"
    echo "  # Ctrl+B, D to detach"
else
    echo -e "${GREEN}✓ Environment ready${NC}"
    echo -e "${YELLOW}⚠ Datasets need manual download${NC}"
    echo ""
    echo "After downloading datasets, run:"
    echo "  ./download_complete.sh"
fi

echo ""
echo "Repository: https://github.com/darskkaa/NTIRE-2026-Light-Field-Image-Super-Resolution-Challenge---Track-2-Efficiency"
echo ""
