#!/bin/bash
#===============================================================================
# V9 SOTA TRAINING PIPELINE - MyEfficientLFNetV9
# NTIRE 2026 Track 2 Efficiency Challenge
#===============================================================================

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper Functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}============================================================${NC}\n${BOLD}${CYAN}$1${NC}\n${BOLD}${CYAN}============================================================${NC}"; }

set -e  # Exit on error

header "🚀 MyEfficientLFNetV9 Training Workflow"
info "Starting workflow for SOTA Architecture..."

#===============================================================================
# STEP 1: VM ENVIRONMENT SETUP
#===============================================================================
header "📦 STEP 1: VM Environment Setup"

if ! conda env list | grep -q "lfsr"; then
    info "Creating conda environment 'lfsr'..."
    conda create -n lfsr python=3.10 -y
fi

info "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lfsr

if python -c "import transformers; from packaging import version; assert version.parse(transformers.__version__) < version.parse('4.45.0')" &> /dev/null; then
    success "Compatible transformers version already installed."
else
    info "Ensuring compatible transformers version for mamba-ssm..."
    pip install "transformers<4.45.0"
fi

if python -c "import mamba_ssm" &> /dev/null; then
    success "mamba-ssm is already installed in 'lfsr' environment! Skipping installation."
else
    warn "mamba-ssm not found in environment. Installing (one-time only)..."
    
    info "Installing PyTorch 2.4.0 (cu121)..."
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

    info "Installing mamba-ssm via pre-built wheels (no compilation needed)..."
    CAUSAL_CONV1D_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
    MAMBA_SSM_WHL="https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

    pip install "$CAUSAL_CONV1D_WHL"
    pip install "$MAMBA_SSM_WHL"
fi

info "Installing other dependencies..."
pip install numpy scipy h5py imageio einops xlwt tqdm scikit-image fvcore matplotlib gdown

success "All packages installed! This won't run again on future executions."

info "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from mamba_ssm import Mamba; print('mamba-ssm: OK')"
success "Environment setup complete"

#===============================================================================
# STEP 2 & 3: DATASET PREPARATION
#===============================================================================
header "📀 STEP 2 & 3: Dataset Preparation"

mkdir -p datasets downloads

check_and_prepare() {
    FILE=$1
    URL=$2
    DEST="datasets/$3"
    
    if [ -d "$DEST" ]; then
        success "Dataset '$3' found in datasets/ (Skipping)"
        return
    fi
    
    info "Checking for $FILE..."
    if [ -f "$FILE" ]; then
        mv "$FILE" downloads/
    elif [ ! -f "downloads/$FILE" ]; then
        if [ -n "$URL" ]; then
            warn "'$FILE' not found. Downloading..."
            gdown --fuzzy "$URL" -O "downloads/$FILE"
        fi
    fi

    if [ -f "downloads/$FILE" ]; then
        info "Extracting $FILE..."
        unzip -q -o "downloads/$FILE" -d datasets/
        success "Extracted $FILE"
    fi
}

check_and_prepare "EPFL.zip" "https://drive.google.com/file/d/19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN/view?usp=drive_link" "EPFL"
check_and_prepare "HCI_new.zip" "https://drive.google.com/file/d/1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9/view?usp=drive_link" "HCI_new"
check_and_prepare "HCI_old.zip" "https://drive.google.com/file/d/1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG/view?usp=drive_link" "HCI_old"
check_and_prepare "INRIA_Lytro.zip" "https://drive.google.com/file/d/1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ/view?usp=drive_link" "INRIA_Lytro"
check_and_prepare "Stanford_Gantry.zip" "https://drive.google.com/file/d/1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3/view?usp=drive_link" "Stanford_Gantry"
success "Dataset preparation complete"

#===============================================================================
# STEP 4: GENERATE DATA
#===============================================================================
header "🧩 STEP 4: Data Generation"

mkdir -p data_for_training data_for_test

if [ -n "$(find data_for_training -name "*.h5" | head -1)" ]; then
    success "Training data (.h5) already exists. Skipping generation."
else
    info "Generating SR_5x5_4x training patches..."
    python Generate_Data_for_Training.py --angRes 5 --scale_factor 4 --src_data_path ./datasets/ --save_data_path ./data_for_training/
fi

if [ -n "$(find data_for_test -name "*.h5" | head -1)" ]; then
    success "Test data (.h5) already exists."
else
    info "Generating SR_5x5_4x test patches..."
    python Generate_Data_for_Test.py --angRes 5 --scale_factor 4 --src_data_path ./datasets/ --save_data_path ./data_for_test/
fi

success "Data generation complete"

#===============================================================================
# STEP 5: VERIFY DATASETS & CHANNELS
#===============================================================================
header "🔍 STEP 5: Verify Datasets and Channels"

info "Running verify_datasets.py..."
python verify_datasets.py

info "Running verify_channels.py..."
python verify_channels.py

success "Data verification passed."

#===============================================================================
# STEP 6: VERIFY MODEL
#===============================================================================
header "🧪 STEP 6: Verify MyEfficientLFNetV9 Model"

info "Running V9 structural self-test..."
if python model/SR/MyEfficientLFNetV9.py; then
    success "Model self-test passed."
else
    error "Model self-test failed!"
fi

#===============================================================================
# STEP 7: TRAINING
#===============================================================================
header "🏋️ STEP 7: Training MyEfficientLFNetV9"

LAST_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV9/checkpoints/*.pth 2>/dev/null | head -1)

if [ -n "$LAST_CKPT" ]; then
    warn "Found existing checkpoint: $LAST_CKPT"
    read -p "Resume training? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        info "Starting FRESH training..."
        python train.py --model_name MyEfficientLFNetV9 --angRes 5 --scale_factor 4 --batch_size 8 --lr 2e-4 --epoch 150 --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ --device cuda:0 --num_workers 8
    else
        info "Resuming training..."
        python train.py --model_name MyEfficientLFNetV9 --angRes 5 --scale_factor 4 --batch_size 8 --lr 2e-4 --epoch 150 --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ --device cuda:0 --num_workers 8
    fi
else
    info "Starting V9 training (150 epochs)..."
    python train.py --model_name MyEfficientLFNetV9 --angRes 5 --scale_factor 4 --batch_size 8 --lr 2e-4 --epoch 150 --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ --device cuda:0 --num_workers 8
fi

success "Training complete"

#===============================================================================
# STEP 8: INFERENCE & EVALUATION
#===============================================================================
header "📊 STEP 8: Inference and Evaluation"

BEST_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV9/checkpoints/*.pth 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    info "Using best checkpoint: $BEST_CKPT"
    info "Running inference..."
    python inference.py --model_name MyEfficientLFNetV9 --angRes 5 --scale_factor 4 --use_pre_ckpt True --path_pre_pth "$BEST_CKPT" --path_for_test ./data_for_test/ --data_name ALL
    success "Inference complete"
else
    warn "No checkpoints found. Skipping inference."
fi

header "🏆 TRAINING WORKFLOW COMPLETE!"
success "V9 Results available in log/SR_5x5_4x/ALL/MyEfficientLFNetV9/"
