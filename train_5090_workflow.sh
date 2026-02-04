#!/bin/bash
#===============================================================================
# RTX 5090 TRAINING WORKFLOW - MyEfficientLFNetV5
# NTIRE 2026 Track 2 Efficiency Challenge
#===============================================================================
#
# This script provides a complete workflow for training MyEfficientLFNetV5
# on a Linux VM with RTX 5090 GPU.
#
# Prerequisites:
#   - Ubuntu 20.04/22.04 LTS
#   - NVIDIA Driver 545+
#   - CUDA 12.x
#   - Python 3.10+
#   - RTX 5090 GPU (24GB+ VRAM)
#
# Usage:
#   chmod +x train_5090_workflow.sh
#   ./train_5090_workflow.sh
#
#===============================================================================

set -e  # Exit on error

echo "=============================================================================="
echo "🚀 MyEfficientLFNetV5 Training Workflow - RTX 5090"
echo "=============================================================================="

#===============================================================================
# STEP 1: ENVIRONMENT SETUP
#===============================================================================
echo ""
echo "📦 STEP 1: Environment Setup"
echo "------------------------------------------------------------------------------"

# Create conda environment (if not exists)
if ! conda env list | grep -q "lfsr"; then
    echo "Creating conda environment 'lfsr'..."
    conda create -n lfsr python=3.10 -y
fi

# Activate environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lfsr

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install mamba-ssm (REQUIRED - no fallback)
echo "Installing mamba-ssm..."
pip install causal-conv1d>=1.1.0
pip install mamba-ssm

# Install other dependencies
echo "Installing other dependencies..."
pip install numpy scipy h5py imageio einops xlwt tqdm scikit-image fvcore matplotlib

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from mamba_ssm import Mamba; print('mamba-ssm: OK')"

echo "✓ Environment setup complete"

#===============================================================================
# STEP 2: DATASET DOWNLOAD
#===============================================================================
echo ""
echo "📥 STEP 2: Dataset Download"
echo "------------------------------------------------------------------------------"

# Install gdown for Google Drive downloads
pip install gdown

# Create directories
mkdir -p data_for_training
mkdir -p data_for_test
mkdir -p downloads

cd downloads

# Download datasets from Google Drive
echo "Downloading EPFL dataset..."
# EPFL: https://drive.google.com/file/d/19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN/view
gdown --fuzzy "https://drive.google.com/file/d/19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN/view?usp=drive_link" -O EPFL.zip

echo "Downloading HCI_new dataset..."
# HCI_new: https://drive.google.com/file/d/1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9/view
gdown --fuzzy "https://drive.google.com/file/d/1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9/view?usp=drive_link" -O HCI_new.zip

echo "Downloading HCI_old dataset..."
# HCI_old: https://drive.google.com/file/d/1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG/view
gdown --fuzzy "https://drive.google.com/file/d/1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG/view?usp=drive_link" -O HCI_old.zip

echo "Downloading INRIA_Lytro dataset..."
# INRIA_Lytro: https://drive.google.com/file/d/1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ/view
gdown --fuzzy "https://drive.google.com/file/d/1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ/view?usp=drive_link" -O INRIA_Lytro.zip

echo "Downloading Stanford_Gantry dataset..."
# Stanford_Gantry: https://drive.google.com/file/d/1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3/view
gdown --fuzzy "https://drive.google.com/file/d/1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3/view?usp=drive_link" -O Stanford_Gantry.zip

cd ..

echo "✓ Dataset download complete"

#===============================================================================
# STEP 3: EXTRACT AND ORGANIZE DATASETS
#===============================================================================
echo ""
echo "📂 STEP 3: Extract and Organize Datasets"
echo "------------------------------------------------------------------------------"

cd downloads

# Extract all datasets
for zip in *.zip; do
    echo "Extracting $zip..."
    unzip -o "$zip" -d ../datasets/
done

cd ..

# Verify dataset structure
echo "Verifying dataset structure..."
ls -la datasets/

echo "✓ Dataset extraction complete"

#===============================================================================
# STEP 4: GENERATE TRAINING PATCHES (H5 format)
#===============================================================================
echo ""
echo "🔧 STEP 4: Generate Training Patches"
echo "------------------------------------------------------------------------------"

# Generate training data using BasicLFSR's script
echo "Generating SR_5x5_4x training patches..."
python Generate_Data_for_Training.py \
    --angRes 5 \
    --scale_factor 4 \
    --src_data_path ./datasets/ \
    --save_data_path ./data_for_training/

# Generate test data
echo "Generating SR_5x5_4x test patches..."
python Generate_Data_for_Test.py \
    --angRes 5 \
    --scale_factor 4 \
    --src_data_path ./datasets/ \
    --save_data_path ./data_for_test/

# Verify generated data
echo "Verifying generated data..."
ls -la data_for_training/SR_5x5_4x/
ls -la data_for_test/SR_5x5_4x/

echo "✓ Training patch generation complete"

#===============================================================================
# STEP 5: VERIFY MODEL BEFORE TRAINING
#===============================================================================
echo ""
echo "🧪 STEP 5: Verify Model"
echo "------------------------------------------------------------------------------"

# Run model self-test
echo "Running MyEfficientLFNetV5 self-test..."
python model/SR/MyEfficientLFNetV5.py

# Check efficiency constraints
echo "Checking efficiency constraints..."
python check_efficiency.py --model_name MyEfficientLFNetV5

echo "✓ Model verification complete"

#===============================================================================
# STEP 6: TRAINING
#===============================================================================
echo ""
echo "🏋️ STEP 6: Training MyEfficientLFNetV5"
echo "------------------------------------------------------------------------------"

# Training configuration for RTX 5090
# - Batch size 8 (5090 has 32GB VRAM)
# - 80 epochs for full training
# - Mixed precision enabled automatically

echo "Starting training..."
python train.py \
    --model_name MyEfficientLFNetV5 \
    --angRes 5 \
    --scale_factor 4 \
    --batch_size 8 \
    --lr 2e-4 \
    --epoch 80 \
    --path_for_train ./data_for_training/ \
    --path_for_test ./data_for_test/ \
    --device cuda:0 \
    --num_workers 8

echo "✓ Training complete"

#===============================================================================
# STEP 7: INFERENCE AND EVALUATION
#===============================================================================
echo ""
echo "📊 STEP 7: Inference and Evaluation"
echo "------------------------------------------------------------------------------"

# Find the best checkpoint
BEST_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV5/checkpoints/*.pth | head -1)
echo "Using checkpoint: $BEST_CKPT"

# Run inference on test set
echo "Running inference..."
python inference.py \
    --model_name MyEfficientLFNetV5 \
    --angRes 5 \
    --scale_factor 4 \
    --use_pre_ckpt True \
    --path_pre_pth "$BEST_CKPT" \
    --path_for_test ./data_for_test/ \
    --data_name ALL

echo "✓ Inference complete"

#===============================================================================
# STEP 8: CREATE SUBMISSION
#===============================================================================
echo ""
echo "📦 STEP 8: Create Submission"
echo "------------------------------------------------------------------------------"

# Create submission zip for CodaBench
echo "Creating submission package..."
./create_submission.sh

# Validate submission
echo "Validating submission..."
python validate_submission.py

echo "✓ Submission created"

#===============================================================================
# COMPLETE
#===============================================================================
echo ""
echo "=============================================================================="
echo "🏆 TRAINING WORKFLOW COMPLETE!"
echo "=============================================================================="
echo ""
echo "Results:"
echo "  - Checkpoints: log/SR_5x5_4x/ALL/MyEfficientLFNetV5/checkpoints/"
echo "  - Validation: log/SR_5x5_4x/ALL/MyEfficientLFNetV5/results/"
echo "  - Submission: submission.zip"
echo ""
echo "Next steps:"
echo "  1. Check validation PSNR in log files"
echo "  2. Submit to CodaBench for official evaluation"
echo "  3. Fine-tune if needed (adjust lr, epochs)"
echo ""
echo "=============================================================================="
