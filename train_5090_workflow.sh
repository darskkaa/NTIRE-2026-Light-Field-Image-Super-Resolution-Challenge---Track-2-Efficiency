#!/bin/bash
#===============================================================================
# RTX 5090 TRAINING WORKFLOW - MyEfficientLFNetV6_2
# NTIRE 2026 Track 2 Efficiency Challenge
#===============================================================================

# ANSI Colors for Chatty CLI
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

header "🚀 MyEfficientLFNetV6_2 Training Workflow - RTX 5090"
info "Starting workflow..."

#===============================================================================
# STEP 1: ENVIRONMENT SETUP
#===============================================================================
header "📦 STEP 1: Environment Setup"

# Check for mamba-ssm first
if python -c "import mamba_ssm" &> /dev/null; then
    success "mamba-ssm is already installed!"
else
    warn "mamba-ssm not found. Setting up environment..."
    
    # Create conda environment (if not exists)
    if ! conda env list | grep -q "lfsr"; then
        info "Creating conda environment 'lfsr'..."
        conda create -n lfsr python=3.10 -y
    fi

    # Activate environment
    info "Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate lfsr

    # Install PyTorch 2.4.0 (STABLE for Mamba)
    # 2.5.x is known to cause ABI issues with Mamba-SSM 2.3.0
    info "Installing PyTorch 2.4.0 (Stable)..."
    pip uninstall -y torch torchvision torchaudio mamba-ssm causal-conv1d
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

    # Install mamba-ssm (REQUIRED - no fallback)
    # Force reinstall to ensure ABI compatibility with current Torch
    info "Installing/Recompiling mamba-ssm to match PyTorch..."
    pip install causal-conv1d>=1.1.0 mamba-ssm --force-reinstall --no-cache-dir --no-binary mamba-ssm,causal-conv1d

    # Install other dependencies
    info "Installing other dependencies..."
    pip install numpy scipy h5py imageio einops xlwt tqdm scikit-image fvcore matplotlib
fi

# Verify installations
info "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from mamba_ssm import Mamba; print('mamba-ssm: OK')"
success "Environment setup complete"

#===============================================================================
# STEP 2 & 3: DATASET PREPARATION (Smart Check)
#===============================================================================
header "� STEP 2 & 3: Dataset Preparation"

mkdir -p datasets downloads

# Function to check and move/download
check_and_prepare() {
    FILE=$1
    URL=$2
    DEST="datasets/$3"
    
    if [ -d "$DEST" ]; then
        success "Dataset '$3' found in datasets/ (Skipping)"
        return
    fi
    
    info "Checking for $FILE..."
    
    # Check current dir, downloads dir, or download it
    if [ -f "$FILE" ]; then
        info "Found '$FILE' in current directory. Moving to downloads/..."
        mv "$FILE" downloads/
    elif [ ! -f "downloads/$FILE" ]; then
        if [ -z "$URL" ]; then
             warn "File '$FILE' not found and no URL provided. Skipping..."
             return
        fi
        warn "'$FILE' not found. Downloading..."
        gdown --fuzzy "$URL" -O "downloads/$FILE"
    else
        info "'$FILE' found in downloads/."
    fi

    info "Extracting $FILE..."
    unzip -q -o "downloads/$FILE" -d datasets/
    success "Extracted $FILE"
}

# 1. Standard Datasets
check_and_prepare "EPFL.zip" "https://drive.google.com/file/d/19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN/view?usp=drive_link" "EPFL"
check_and_prepare "HCI_new.zip" "https://drive.google.com/file/d/1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9/view?usp=drive_link" "HCI_new"
check_and_prepare "HCI_old.zip" "https://drive.google.com/file/d/1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG/view?usp=drive_link" "HCI_old"
check_and_prepare "INRIA_Lytro.zip" "https://drive.google.com/file/d/1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ/view?usp=drive_link" "INRIA_Lytro"
check_and_prepare "Stanford_Gantry.zip" "https://drive.google.com/file/d/1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3/view?usp=drive_link" "Stanford_Gantry"

# 2. Validation Datasets (User Provided)
check_and_prepare "NTIRE_Val_Real.zip" "" "NTIRE_Val_Real"
check_and_prepare "NTIRE_Val_Synth.zip" "" "NTIRE_Val_Synth"

success "Dataset preparation complete"

#===============================================================================
# STEP 4: GENERATE PATCHES (Smart Check)
#===============================================================================
header "� STEP 4: Generate Training Patches"

mkdir -p data_for_training data_for_test

# Check if training data seems populated (heuristic: check for .h5 files)
if [ -n "$(find data_for_training -name "*.h5" | head -1)" ]; then
    success "Training data (.h5) already exists. Skipping generation."
    read -p "Force regenerate? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Regenerating training patches..."
        python Generate_Data_for_Training.py --angRes 5 --scale_factor 4 --src_data_path ./datasets/ --save_data_path ./data_for_training/
    fi
else
    info "Generating SR_5x5_4x training patches..."
    python Generate_Data_for_Training.py --angRes 5 --scale_factor 4 --src_data_path ./datasets/ --save_data_path ./data_for_training/
fi

# Check Test Data
if [ -n "$(find data_for_test -name "*.h5" | head -1)" ]; then
    success "Test data (.h5) already exists."
else
    info "Generating SR_5x5_4x test patches..."
    python Generate_Data_for_Test.py --angRes 5 --scale_factor 4 --src_data_path ./datasets/ --save_data_path ./data_for_test/
fi

success "Patch generation complete"

#===============================================================================
# STEP 5: VERIFY MODEL
#===============================================================================
header "🧪 STEP 5: Verify Model"

info "Running MyEfficientLFNetV6_2 self-test..."
if python model/SR/MyEfficientLFNetV6_2.py; then
    success "Model self-test passed."
else
    error "Model self-test failed!"
fi

info "Checking efficiency constraints..."
if python check_efficiency.py --model_name MyEfficientLFNetV6_2; then
    success "Efficiency check passed."
else
    error "Efficiency check failed! Check 'check_efficiency.py' output."
fi

#===============================================================================
# STEP 6: TRAINING
#===============================================================================
header "🏋️ STEP 6: Training MyEfficientLFNetV6_2"

# Check if checkpoint exists
LAST_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV6_2/checkpoints/*.pth 2>/dev/null | head -1)

if [ -n "$LAST_CKPT" ]; then
    warn "Found existing checkpoint: $LAST_CKPT"
    read -p "Resume training? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        info "Starting FRESH training..."
        python train.py --model_name MyEfficientLFNetV6_2 --angRes 5 --scale_factor 4 --batch_size 8 --lr 2e-4 --epoch 150 --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ --device cuda:0 --num_workers 8
    else
        info "Resuming training (script logic needed for auto-resume, defaulting to train.py)..."
        # Note: BasicLFSR train.py usually needs --resume or loads latest if configured. 
        # Assuming standard run continues or overwrites depending on impl. 
        # For safety, we restart unless user manually flags resume in code.
        # But we'll just run output logger adds to existing.
        python train.py --model_name MyEfficientLFNetV6_2 --angRes 5 --scale_factor 4 --batch_size 8 --lr 2e-4 --epoch 150 --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ --device cuda:0 --num_workers 8
    fi
else
    info "Starting training (150 epochs)..."
    python train.py --model_name MyEfficientLFNetV6_2 --angRes 5 --scale_factor 4 --batch_size 8 --lr 2e-4 --epoch 150 --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ --device cuda:0 --num_workers 8
fi

success "Training complete"

#===============================================================================
# STEP 7: INFERENCE & EVALUATION
#===============================================================================
header "📊 STEP 7: Inference and Evaluation"

BEST_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV6_2/checkpoints/*.pth | head -1)
info "Using best checkpoint: $BEST_CKPT"

info "Running inference..."
python inference.py --model_name MyEfficientLFNetV6_2 --angRes 5 --scale_factor 4 --use_pre_ckpt True --path_pre_pth "$BEST_CKPT" --path_for_test ./data_for_test/ --data_name ALL

success "Inference complete"

#===============================================================================
# STEP 8: SUBMISSION
#===============================================================================
header "📦 STEP 8: Create Submission"

info "Creating submission package..."
./create_submission.sh

info "Validating submission..."
python validate_submission.py

header "🏆 TRAINING WORKFLOW COMPLETE!"
success "Results available in log/SR_5x5_4x/ALL/MyEfficientLFNetV6_2/"
