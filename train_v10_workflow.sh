#!/bin/bash
#===============================================================================
# V10 SOTA TRAINING PIPELINE - MyEfficientLFNetV10
# NTIRE 2026 Light Field Image Super-Resolution
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

header "🚀 MyEfficientLFNetV10 SOTA Training Workflow"
info "Starting workflow for V10 SOTA Architecture..."

#===============================================================================
# STEP 1: VM ENVIRONMENT SETUP
#===============================================================================
header "📦 STEP 1: VM Environment Setup"

if python -c "import mamba_ssm" &> /dev/null; then
    success "mamba-ssm is already installed!"
else
    warn "mamba-ssm not found. Setting up environment..."
    
    if ! conda env list | grep -q "lfsr"; then
        info "Creating conda environment 'lfsr'..."
        conda create -n lfsr python=3.10 -y
    fi

    info "Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate lfsr

    info "Installing PyTorch 2.4.0 (Stable)..."
    pip uninstall -y torch torchvision torchaudio mamba-ssm causal-conv1d
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

    info "Installing mamba-ssm (REQUIRED)..."
    pip install causal-conv1d>=1.1.0 mamba-ssm --force-reinstall --no-cache-dir --no-binary mamba-ssm,causal-conv1d

    info "Installing other dependencies..."
    pip install numpy scipy h5py imageio einops xlwt tqdm scikit-image fvcore matplotlib
fi

# Verify einops (required for V10 rearranges)
if ! python -c "import einops" &> /dev/null; then
    info "Installing einops (required for V10 tensor rearranges)..."
    pip install einops
fi

info "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from mamba_ssm import Mamba; print('mamba-ssm: OK')"
python -c "from einops import rearrange; print('einops: OK')"
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
header "🧪 STEP 6: Verify MyEfficientLFNetV10 Model"

info "Running V10 structural self-test..."
if python model/SR/MyEfficientLFNetV10.py; then
    success "Model self-test passed."
else
    error "Model self-test failed!"
fi

#===============================================================================
# STEP 6.5: NTIRE 2026 TRACK 2 EFFICIENCY VALIDATION (MANDATORY GATE)
#===============================================================================
header "🏆 STEP 6.5: NTIRE 2026 Track 2 — Efficiency Validation"

info "Checking model against Track 2 constraints..."
info "  Param Limit:  < 1,000,000 (1M)"
info "  FLOPs Limit:  < 20G (input: 5×5×32×32, fvcore)"
info "  TTA FLOPs:    counted toward budget"
echo ""

if python check_efficiency.py --model_name MyEfficientLFNetV10 --angRes 5 --scale_factor 4 --patch_size 32; then
    success "Efficiency validation PASSED! Model qualifies for Track 2."
else
    error "Efficiency validation FAILED! Model does NOT qualify for Track 2. Fix param/FLOPs limits before training."
fi

#===============================================================================
# STEP 7: TRAINING
#===============================================================================
header "🏋️ STEP 7: Training MyEfficientLFNetV10"

LAST_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV10/checkpoints/*.pth 2>/dev/null | head -1)

if [ -n "$LAST_CKPT" ]; then
    warn "Found existing checkpoint: $LAST_CKPT"
    read -p "Resume training? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        info "Starting FRESH training..."
        python train.py --model_name MyEfficientLFNetV10 --angRes 5 --scale_factor 4 --batch_size 4 --lr 2e-4 --epoch 150 --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ --device cuda:0 --num_workers 8
    else
        info "Resuming training from $LAST_CKPT..."
        python train.py --model_name MyEfficientLFNetV10 --angRes 5 --scale_factor 4 --batch_size 4 --lr 2e-4 --epoch 150 --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ --device cuda:0 --num_workers 8 --use_pre_ckpt --path_pre_pth "$LAST_CKPT"
    fi
else
    info "Starting V10 training (150 epochs)..."
    python train.py --model_name MyEfficientLFNetV10 --angRes 5 --scale_factor 4 --batch_size 4 --lr 2e-4 --epoch 150 --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ --device cuda:0 --num_workers 8
fi

success "Training complete"

#===============================================================================
# STEP 8: INFERENCE & EVALUATION
#===============================================================================
header "📊 STEP 8: Inference and Evaluation"

BEST_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV10/checkpoints/*.pth 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    info "Using best checkpoint: $BEST_CKPT"
    info "Running inference..."
    python inference.py --model_name MyEfficientLFNetV10 --angRes 5 --scale_factor 4 --use_pre_ckpt --path_pre_pth "$BEST_CKPT" --path_for_test ./data_for_test/ --data_name ALL
    success "Inference complete"
else
    warn "No checkpoints found. Skipping inference."
fi

header "🏆 TRAINING WORKFLOW COMPLETE!"
success "V10 Results available in log/SR_5x5_4x/ALL/MyEfficientLFNetV10/"

#===============================================================================
# STEP 9: VALIDATE SUBMISSION (CodaBench Compliance)
#===============================================================================
header "✅ STEP 9: Validate Submission"

if [ -f "validate_submission.py" ]; then
    info "Running submission validator..."
    if python validate_submission.py --model_name MyEfficientLFNetV10 --angRes 5 --scale_factor 4; then
        success "Submission validation passed!"
    else
        warn "Submission validation had issues — check output above."
    fi
else
    warn "validate_submission.py not found, skipping validation."
fi

info "Re-running efficiency check (final gate)..."
python check_efficiency.py --model_name MyEfficientLFNetV10 --angRes 5 --scale_factor 4 --patch_size 32

header "🏁 ALL DONE!"
success "Pipeline complete. Ready for CodaBench submission."
