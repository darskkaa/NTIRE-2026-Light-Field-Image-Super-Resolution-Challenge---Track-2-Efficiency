#!/bin/bash
#===============================================================================
# NTIRE 2026 LF-SR Track 2 - Full Pipeline & VM Validation Script
# Includes: Downloading Validation Data, Checking Environment, Inference, 
#           Zip Formatting, and CodaBench Validation.
#===============================================================================

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "\n${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

set -e

# ==========================================
# 0. Prep Environment
# ==========================================
info "Checking dependencies..."
if ! command -v gdown &> /dev/null; then
    info "Installing gdown..."
    pip install gdown
fi

# ==========================================
# 1. Download Validation Data from GDrive
# ==========================================
info "Checking for Validation data..."
mkdir -p datasets/NTIRE_Val_Real/inference datasets/NTIRE_Val_Synth/inference downloads/NTIRE_Val

if [ -f "datasets/NTIRE_Val_Real/inference/Val_Real_00.mat" ] || [ -f "datasets/NTIRE_Val_Real/inference/Val_Real_00.h5" ] || \
   [ "$(ls -A datasets/NTIRE_Val_Real/inference 2>/dev/null)" ] ; then
    success "Validation data already exists!"
else
    info "Downloading Validation data from Google Drive folder..."
    # User's provided validation folder link
    gdown --folder "https://drive.google.com/drive/folders/1LfPTTTtTDOPyNg3D-B_RfzwBZd4D0-HH?usp=sharing" -O downloads/NTIRE_Val
    
    info "Organizing Validation data into datasets/ folder for standard pipeline..."
    # The GDrive folder likely contains the Real/Synth subset, or raw files.
    # We will safely find all .mat and .h5 files and distribute them if they have Real/Synth in their path.
    find downloads/NTIRE_Val -type f \( -name "*.mat" -o -name "*.h5" \) | while read FILE; do
        if [[ "$FILE" == *"Real"* ]]; then
            cp "$FILE" datasets/NTIRE_Val_Real/inference/
        elif [[ "$FILE" == *"Synth"* ]]; then
            cp "$FILE" datasets/NTIRE_Val_Synth/inference/
        else
            # If the name doesn't specify, just drop it in Real as fallback, but print a warning.
            echo "Warning: Couldn't determine Real/Synth category for $FILE. Placing in Real."
            cp "$FILE" datasets/NTIRE_Val_Real/inference/
        fi
    done
    success "Validation data downloaded and structured."
fi

# ==========================================
# 2. Generate Validation Patches
# ==========================================
info "Generating test patches (LR/HR)..."
# We invoke Generate_Data_for_Test.py targeting ONLY the NTIRE_Val datasets
python Generate_Data_for_Test.py --angRes 5 --scale_factor 4 --data_for inference --src_data_path ./datasets/ --save_data_path ./data_for_test/
success "Data generation complete."

# ==========================================
# 3. Run Inference on Validation Set
# ==========================================
# You can change the model name if you're evaluating a different version
MODEL="MyEfficientLFNetV10"
info "Finding best model checkpoint for $MODEL..."

BEST_CKPT=$(ls -t log/SR_5x5_4x/ALL/$MODEL/checkpoints/*.pth 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    error "No checkpoint found for $MODEL! Have you trained the model yet?"
fi

info "Using checkpoint: $BEST_CKPT"
info "Running inference on ALL test data (including newly downloaded validation sets)..."

# Inference generates BMPs in log/SR_5x5_4x/ALL/$MODEL/results/TEST/
python inference.py \
    --model_name "$MODEL" \
    --angRes 5 \
    --scale_factor 4 \
    --use_pre_ckpt True \
    --path_pre_pth "$BEST_CKPT" \
    --path_for_test ./data_for_test/ \
    --data_name ALL

success "Inference complete!"

# ==========================================
# 4. Format Submission & Validate
# ==========================================
RESULTS_DIR="log/SR_5x5_4x/ALL/$MODEL/results/TEST"
ZIP_NAME="submission_${MODEL}.zip"

if [ -d "$RESULTS_DIR" ]; then
    info "Formatting submission from $RESULTS_DIR into required Real/ Synth/ structure..."
    python format_submission.py "$RESULTS_DIR" --output "$ZIP_NAME"
    
    info "Running strict CodaBench validation script..."
    python validate_submission.py "$ZIP_NAME"
    
    success "🚀 PIPELINE FINISHED!" 
    echo -e "${GREEN}Your submission file is ready for upload:${NC} $ZIP_NAME"
    echo "To push your changes to GitHub, run:"
    echo "git add ."
    echo "git commit -m 'Added validation pipeline'"
    echo "git push"
else
    error "Results directory $RESULTS_DIR not found. Inference might have failed."
fi
