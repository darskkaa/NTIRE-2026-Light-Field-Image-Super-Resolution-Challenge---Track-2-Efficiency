#!/bin/bash
#===============================================================================
# NTIRE 2026 LF-SR Track 2 — FULL VALIDATION & SUBMISSION PIPELINE
#===============================================================================
# Run this on the VM after training is complete.
#
# What it does:
#   1. Downloads NTIRE Validation .mat files from Google Drive
#   2. Generates test patches via Generate_Validation_Data.py
#   3. Runs inference with your latest checkpoint
#   4. Formats output into Real/ + Synth/ structure (CodaBench format)
#   5. Zips and validates for submission
#
# Usage:
#   chmod +x eval_validation.sh
#   ./eval_validation.sh
#===============================================================================

set -e

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "\n${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

MODEL="MyEfficientLFNetV10"

#===============================================================================
# STEP 1: Download Validation Data
#===============================================================================
info "=== STEP 1: Download Validation Data ==="

pip install gdown 2>/dev/null

# Google Drive folder ID from user's link
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1LfPTTTtTDOPyNg3D-B_RfzwBZd4D0-HH"

mkdir -p downloads/NTIRE_Val
mkdir -p datasets/NTIRE_Val_Real/inference
mkdir -p datasets/NTIRE_Val_Synth/inference

# Check if we already have validation data
REAL_COUNT=$(find datasets/NTIRE_Val_Real/inference -name "*.mat" -o -name "*.h5" 2>/dev/null | wc -l)
SYNTH_COUNT=$(find datasets/NTIRE_Val_Synth/inference -name "*.mat" -o -name "*.h5" 2>/dev/null | wc -l)

if [ "$REAL_COUNT" -ge 8 ] && [ "$SYNTH_COUNT" -ge 8 ]; then
    success "Validation data already present (${REAL_COUNT} Real, ${SYNTH_COUNT} Synth). Skipping download."
else
    info "Downloading validation data from Google Drive..."
    gdown --folder "$GDRIVE_FOLDER" -O downloads/NTIRE_Val

    info "Downloaded contents:"
    find downloads/NTIRE_Val -type f | head -30

    # Organize into the correct folder structure
    # The competition provides Real and Synth .mat files
    # Move them into datasets/NTIRE_Val_Real/inference/ and datasets/NTIRE_Val_Synth/inference/
    info "Organizing files..."

    # Strategy: check for Real/Synth subfolders first, then fall back to filename matching
    if [ -d "downloads/NTIRE_Val/NTIRE_Val_Real" ]; then
        cp downloads/NTIRE_Val/NTIRE_Val_Real/*.mat datasets/NTIRE_Val_Real/inference/ 2>/dev/null || true
        cp downloads/NTIRE_Val/NTIRE_Val_Real/*.h5  datasets/NTIRE_Val_Real/inference/ 2>/dev/null || true
    fi
    if [ -d "downloads/NTIRE_Val/NTIRE_Val_Synth" ]; then
        cp downloads/NTIRE_Val/NTIRE_Val_Synth/*.mat datasets/NTIRE_Val_Synth/inference/ 2>/dev/null || true
        cp downloads/NTIRE_Val/NTIRE_Val_Synth/*.h5  datasets/NTIRE_Val_Synth/inference/ 2>/dev/null || true
    fi
    # Also check for Real/ and Synth/ subfolders
    if [ -d "downloads/NTIRE_Val/Real" ]; then
        cp downloads/NTIRE_Val/Real/*.mat datasets/NTIRE_Val_Real/inference/ 2>/dev/null || true
        cp downloads/NTIRE_Val/Real/*.h5  datasets/NTIRE_Val_Real/inference/ 2>/dev/null || true
    fi
    if [ -d "downloads/NTIRE_Val/Synth" ]; then
        cp downloads/NTIRE_Val/Synth/*.mat datasets/NTIRE_Val_Synth/inference/ 2>/dev/null || true
        cp downloads/NTIRE_Val/Synth/*.h5  datasets/NTIRE_Val_Synth/inference/ 2>/dev/null || true
    fi

    # Fallback: if files are flat, sort by name
    find downloads/NTIRE_Val -maxdepth 2 -name "*Real*" \( -name "*.mat" -o -name "*.h5" \) -exec cp {} datasets/NTIRE_Val_Real/inference/ \; 2>/dev/null || true
    find downloads/NTIRE_Val -maxdepth 2 -name "*Synth*" \( -name "*.mat" -o -name "*.h5" \) -exec cp {} datasets/NTIRE_Val_Synth/inference/ \; 2>/dev/null || true

    REAL_COUNT=$(find datasets/NTIRE_Val_Real/inference -name "*.mat" -o -name "*.h5" 2>/dev/null | wc -l)
    SYNTH_COUNT=$(find datasets/NTIRE_Val_Synth/inference -name "*.mat" -o -name "*.h5" 2>/dev/null | wc -l)
    success "Organized: ${REAL_COUNT} Real scenes, ${SYNTH_COUNT} Synth scenes"

    if [ "$REAL_COUNT" -eq 0 ] || [ "$SYNTH_COUNT" -eq 0 ]; then
        warn "Could not auto-sort files. Please manually place .mat files into:"
        warn "  datasets/NTIRE_Val_Real/inference/"
        warn "  datasets/NTIRE_Val_Synth/inference/"
        info "Downloaded files are in downloads/NTIRE_Val/"
        find downloads/NTIRE_Val -type f
        error "Cannot continue without validation data."
    fi
fi

#===============================================================================
# STEP 2: Generate Test Patches (h5 files for the dataloader)
#===============================================================================
info "=== STEP 2: Generate Test Patches ==="

# Generate_Validation_Data.py creates h5 files in data_for_test/data_for_inference/SR_5x5_4x/
# It only processes NTIRE_Val_Real and NTIRE_Val_Synth datasets
python Generate_Validation_Data.py \
    --angRes 5 \
    --scale_factor 4 \
    --data_for inference \
    --src_data_path ./datasets/ \
    --save_data_path ./

success "Test patches generated."

#===============================================================================
# STEP 3: Run Inference
#===============================================================================
info "=== STEP 3: Run Inference with $MODEL ==="

BEST_CKPT=$(ls -t log/SR_5x5_4x/ALL/$MODEL/checkpoints/*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    error "No checkpoint found for $MODEL! Train the model first."
fi

info "Checkpoint: $BEST_CKPT"

# Run inference on the validation data
# --path_for_test points to data_for_inference/ which contains the h5 files
# --data_name ALL picks up all datasets in that folder (NTIRE_Val_Real, NTIRE_Val_Synth)
python inference.py \
    --model_name "$MODEL" \
    --angRes 5 \
    --scale_factor 4 \
    --use_pre_ckpt \
    --path_pre_pth "$BEST_CKPT" \
    --path_for_test ./data_for_inference/ \
    --data_name ALL \
    --device cuda:0

success "Inference complete!"

#===============================================================================
# STEP 4: Format & Validate Submission
#===============================================================================
info "=== STEP 4: Format & Validate Submission ==="

# Inference saves BMPs to: log/SR_5x5_4x/ALL/$MODEL/results/TEST/
# Inside: NTIRE_Val_Real/<scene>/View_i_j.bmp, NTIRE_Val_Synth/<scene>/View_i_j.bmp
RESULTS_DIR="log/SR_5x5_4x/ALL/$MODEL/results/TEST"
ZIP_NAME="submission_${MODEL}.zip"

if [ ! -d "$RESULTS_DIR" ]; then
    error "Results directory not found: $RESULTS_DIR"
fi

info "Results directory contents:"
ls "$RESULTS_DIR"

# format_submission.py maps:
#   NTIRE_Val_Real  -> Real/
#   NTIRE_Val_Synth -> Synth/
# Then zips them with Real/ and Synth/ at the root
info "Formatting submission..."
python format_submission.py "$RESULTS_DIR" --output "$ZIP_NAME"

info "Validating submission..."
python validate_submission.py "$ZIP_NAME"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}🚀 SUBMISSION READY: ${ZIP_NAME}${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Upload $ZIP_NAME to CodaBench to get your validation score."
