#!/bin/bash
#===============================================================================
# NTIRE 2026 LF-SR Track 2 — VALIDATION SUBMISSION PIPELINE
#===============================================================================
# Downloads NTIRE Val data, runs inference, formats submission zip.
#
# Google Drive folder contains:
#   NTIRE_Val_Real.zip  → NTIRE_Val_Real/inference/ (16 .mat files)
#   NTIRE_Val_Synth.zip → NTIRE_Val_Synth/inference/ (16 .mat files: 01-16)
#
# Final submission zip:
#   Real/<scene_name>/View_0_0.bmp ... View_4_4.bmp  (16 scenes)
#   Synth/<scene_name>/View_0_0.bmp ... View_4_4.bmp (16 scenes)
#===============================================================================

set -e

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "\n${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

MODEL="MyEfficientLFNetV10"

#===============================================================================
# STEP 1: Download & Extract NTIRE Validation Data
#===============================================================================
info "=== STEP 1: Download NTIRE Validation Data ==="

pip install gdown 2>/dev/null || true
mkdir -p downloads datasets

REAL_COUNT=$(find datasets/NTIRE_Val_Real/inference -name "*.mat" 2>/dev/null | wc -l)
SYNTH_COUNT=$(find datasets/NTIRE_Val_Synth/inference -name "*.mat" 2>/dev/null | wc -l)

if [ "$REAL_COUNT" -ge 16 ] && [ "$SYNTH_COUNT" -ge 16 ]; then
    success "Validation data already present (${REAL_COUNT} Real, ${SYNTH_COUNT} Synth)."
else
    info "Downloading validation zips from Google Drive..."
    gdown --folder "https://drive.google.com/drive/folders/1LfPTTTtTDOPyNg3D-B_RfzwBZd4D0-HH" -O downloads/

    info "Downloaded files:"
    ls -la downloads/

    # Extract the zips into datasets/
    if [ -f "downloads/NTIRE_Val_Real.zip" ]; then
        info "Extracting NTIRE_Val_Real.zip..."
        unzip -o -q downloads/NTIRE_Val_Real.zip -d datasets/
    fi
    if [ -f "downloads/NTIRE_Val_Synth.zip" ]; then
        info "Extracting NTIRE_Val_Synth.zip..."
        unzip -o -q downloads/NTIRE_Val_Synth.zip -d datasets/
    fi

    # Verify
    REAL_COUNT=$(find datasets/NTIRE_Val_Real/inference -name "*.mat" 2>/dev/null | wc -l)
    SYNTH_COUNT=$(find datasets/NTIRE_Val_Synth/inference -name "*.mat" 2>/dev/null | wc -l)
    info "Found: ${REAL_COUNT} Real .mat, ${SYNTH_COUNT} Synth .mat"

    if [ "$REAL_COUNT" -eq 0 ] || [ "$SYNTH_COUNT" -eq 0 ]; then
        error "Missing validation data! Check downloads/ and manually extract."
    fi
fi

info "Real scenes:"; ls datasets/NTIRE_Val_Real/inference/
info "Synth scenes:"; ls datasets/NTIRE_Val_Synth/inference/

#===============================================================================
# STEP 2: Generate h5 patches for the dataloader
#===============================================================================
info "=== STEP 2: Generate Test Patches ==="

# Clear old inference data to avoid mixing with training datasets
rm -rf data_for_inference/SR_5x5_4x/NTIRE_Val_Real data_for_inference/SR_5x5_4x/NTIRE_Val_Synth

python Generate_Validation_Data.py \
    --angRes 5 \
    --scale_factor 4 \
    --data_for inference \
    --src_data_path ./datasets/ \
    --save_data_path ./

success "Patches generated:"
ls data_for_inference/SR_5x5_4x/

#===============================================================================
# STEP 3: Clear old results & Run Inference on NTIRE validation ONLY
#===============================================================================
info "=== STEP 3: Run Inference ==="

BEST_CKPT=$(ls -t log/SR_5x5_4x/ALL/$MODEL/checkpoints/*.pth 2>/dev/null | head -1)
[ -z "$BEST_CKPT" ] && error "No checkpoint found for $MODEL!"
info "Checkpoint: $BEST_CKPT"

# IMPORTANT: Clear previous results to avoid mixing with standard datasets
rm -rf log/SR_5x5_4x/ALL/$MODEL/results/TEST
info "Cleared old results."

# --path_for_test = data_for_inference/ (ONLY contains NTIRE val data)
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
info "=== STEP 4: Format & Validate Submission ==="

RESULTS_DIR="log/SR_5x5_4x/ALL/$MODEL/results/TEST"
ZIP_NAME="submission_${MODEL}.zip"

[ ! -d "$RESULTS_DIR" ] && error "Results dir not found: $RESULTS_DIR"

info "Result scenes:"
for d in "$RESULTS_DIR"/*/; do
    scene=$(basename "$d")
    count=$(find "$d" -name "*.bmp" | wc -l)
    echo "  $scene/ → $count BMPs"
done

# NTIRE_Val_Real → Real/, NTIRE_Val_Synth → Synth/
python format_submission.py "$RESULTS_DIR" --output "$ZIP_NAME"
python validate_submission.py "$ZIP_NAME"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 READY: ${ZIP_NAME}${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Upload to CodaBench!"
