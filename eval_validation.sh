#!/bin/bash
#===============================================================================
# NTIRE 2026 LF-SR Track 2 — VALIDATION SUBMISSION PIPELINE
#===============================================================================
# Generates a valid submission zip from NTIRE validation data.
#
# Expected NTIRE validation data structure (from Google Drive):
#   datasets/
#   ├── NTIRE_Val_Real/inference/   ← 16 .mat files (Real scenes)
#   └── NTIRE_Val_Synth/inference/  ← 16 .mat files (Synth scenes)
#
# Output: submission_MyEfficientLFNetV10.zip
#   ├── Real/
#   │   ├── 01/ (25 BMPs: View_0_0.bmp ... View_4_4.bmp)
#   │   ├── 02/
#   │   └── ... (16 scene folders)
#   └── Synth/
#       ├── 01/
#       └── ... (16 scene folders)
#===============================================================================

set -e

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
# STEP 1: Download NTIRE Validation Data
#===============================================================================
info "=== STEP 1: Download NTIRE Validation Data ==="

pip install gdown 2>/dev/null || true

mkdir -p datasets/NTIRE_Val_Real/inference datasets/NTIRE_Val_Synth/inference downloads

REAL_COUNT=$(find datasets/NTIRE_Val_Real/inference -name "*.mat" 2>/dev/null | wc -l)
SYNTH_COUNT=$(find datasets/NTIRE_Val_Synth/inference -name "*.mat" 2>/dev/null | wc -l)

if [ "$REAL_COUNT" -ge 1 ] && [ "$SYNTH_COUNT" -ge 1 ]; then
    success "Validation data already present (${REAL_COUNT} Real, ${SYNTH_COUNT} Synth)."
else
    info "Downloading from Google Drive..."
    gdown --folder "https://drive.google.com/drive/folders/1LfPTTTtTDOPyNg3D-B_RfzwBZd4D0-HH" -O downloads/NTIRE_Val

    info "Downloaded contents:"
    find downloads/NTIRE_Val -type f | sort

    # Auto-organize: try multiple possible folder structures
    # Case 1: NTIRE_Val_Real / NTIRE_Val_Synth subfolders
    for src in downloads/NTIRE_Val/NTIRE_Val_Real downloads/NTIRE_Val/Real; do
        [ -d "$src" ] && find "$src" -name "*.mat" -exec cp {} datasets/NTIRE_Val_Real/inference/ \;
    done
    for src in downloads/NTIRE_Val/NTIRE_Val_Synth downloads/NTIRE_Val/Synth; do
        [ -d "$src" ] && find "$src" -name "*.mat" -exec cp {} datasets/NTIRE_Val_Synth/inference/ \;
    done

    # Case 2: Flat .mat files with Real/Synth in filename
    find downloads/NTIRE_Val -maxdepth 1 -name "*Real*" -name "*.mat" -exec cp {} datasets/NTIRE_Val_Real/inference/ \; 2>/dev/null || true
    find downloads/NTIRE_Val -maxdepth 1 -name "*Synth*" -name "*.mat" -exec cp {} datasets/NTIRE_Val_Synth/inference/ \; 2>/dev/null || true

    REAL_COUNT=$(find datasets/NTIRE_Val_Real/inference -name "*.mat" 2>/dev/null | wc -l)
    SYNTH_COUNT=$(find datasets/NTIRE_Val_Synth/inference -name "*.mat" 2>/dev/null | wc -l)
    info "After organizing: ${REAL_COUNT} Real, ${SYNTH_COUNT} Synth .mat files"

    if [ "$REAL_COUNT" -eq 0 ] && [ "$SYNTH_COUNT" -eq 0 ]; then
        warn "Auto-sort couldn't find .mat files. Listing what was downloaded:"
        find downloads/NTIRE_Val -type f
        warn "Please manually place .mat files into:"
        warn "  datasets/NTIRE_Val_Real/inference/"
        warn "  datasets/NTIRE_Val_Synth/inference/"
        error "Cannot continue without validation data."
    fi
fi

info "Real .mat files:"
ls datasets/NTIRE_Val_Real/inference/
info "Synth .mat files:"
ls datasets/NTIRE_Val_Synth/inference/

#===============================================================================
# STEP 2: Generate Test Patches (h5 files for dataloader)
#===============================================================================
info "=== STEP 2: Generate Test Patches ==="

# Generate_Validation_Data.py processes NTIRE_Val_Real and NTIRE_Val_Synth
# and saves h5 files to data_for_inference/SR_5x5_4x/NTIRE_Val_Real/ etc.
python Generate_Validation_Data.py \
    --angRes 5 \
    --scale_factor 4 \
    --data_for inference \
    --src_data_path ./datasets/ \
    --save_data_path ./

success "Test patches generated in data_for_inference/SR_5x5_4x/"
ls data_for_inference/SR_5x5_4x/

#===============================================================================
# STEP 3: Run Inference (ONLY on NTIRE validation data)
#===============================================================================
info "=== STEP 3: Run Inference ==="

BEST_CKPT=$(ls -t log/SR_5x5_4x/ALL/$MODEL/checkpoints/*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    error "No checkpoint found for $MODEL!"
fi
info "Using checkpoint: $BEST_CKPT"

# CRITICAL: --path_for_test points to data_for_inference/ (NOT data_for_test/)
# This ensures we only process the NTIRE validation scenes
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
# STEP 4: Format into Real/ + Synth/ and Validate
#===============================================================================
info "=== STEP 4: Format & Validate ==="

RESULTS_DIR="log/SR_5x5_4x/ALL/$MODEL/results/TEST"
ZIP_NAME="submission_${MODEL}.zip"

if [ ! -d "$RESULTS_DIR" ]; then
    error "Results directory not found: $RESULTS_DIR"
fi

info "Inference output:"
ls "$RESULTS_DIR"
for d in "$RESULTS_DIR"/*/; do
    scene=$(basename "$d")
    count=$(find "$d" -name "*.bmp" | wc -l)
    echo "  $scene: $count BMPs"
done

# format_submission.py maps NTIRE_Val_Real → Real/, NTIRE_Val_Synth → Synth/
python format_submission.py "$RESULTS_DIR" --output "$ZIP_NAME"

# Validate the zip
python validate_submission.py "$ZIP_NAME"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}🚀 SUBMISSION READY: ${ZIP_NAME}${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Upload $ZIP_NAME to CodaBench!"
