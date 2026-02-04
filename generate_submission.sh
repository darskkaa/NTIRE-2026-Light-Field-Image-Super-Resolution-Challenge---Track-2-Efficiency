#!/bin/bash
# ============================================================
# generate_submission.sh
# Automates inference and packaging for NTIRE 2026 Track 2
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "NTIRE 2026 - Generating Submission"
echo "============================================================"

# 1. Activate environment
if [ -f "venv_lfsr/bin/activate" ]; then
    source venv_lfsr/bin/activate
    echo "✓ Activated venv_lfsr"
else
    echo "⚠ venv_lfsr not found, using current environment"
fi

# 2. Define paths
# Use the best checkpoint (epoch 50 by default, adjust if needed)
MODEL_NAME="MyEfficientLFNetV4_5"
CHECKPOINT_DIR="log/SR_5x5_4x/ALL/${MODEL_NAME}/checkpoints"
OUTPUT_DIR="log/SR_5x5_4x/ALL/${MODEL_NAME}/TEST"

# Find latest checkpoint
LATEST_CKPT=$(ls -t ${CHECKPOINT_DIR}/*.pth 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "❌ Error: No checkpoints found in ${CHECKPOINT_DIR}"
    exit 1
fi
echo "✓ Using checkpoint: ${LATEST_CKPT}"

# 3. Check data directories
VAL_DATA_PATH="./data_for_inference/"
if [ ! -d "${VAL_DATA_PATH}" ]; then
    # Fallback to test path
    VAL_DATA_PATH="./data_for_test/SR_5x5_4x/"
fi
echo "✓ Using data path: ${VAL_DATA_PATH}"

# 4. Run inference on validation sets
echo ""
echo "[1/4] Running inference on NTIRE_Val_Real..."
python inference.py \
    --model_name ${MODEL_NAME} \
    --angRes 5 \
    --scale_factor 4 \
    --task SR \
    --use_pre_ckpt True \
    --path_pre_pth "${LATEST_CKPT}" \
    --path_for_test "${VAL_DATA_PATH}" \
    --data_name NTIRE_Val_Real \
    --device cuda:0

echo ""
echo "[2/4] Running inference on NTIRE_Val_Synth..."
python inference.py \
    --model_name ${MODEL_NAME} \
    --angRes 5 \
    --scale_factor 4 \
    --task SR \
    --use_pre_ckpt True \
    --path_pre_pth "${LATEST_CKPT}" \
    --path_for_test "${VAL_DATA_PATH}" \
    --data_name NTIRE_Val_Synth \
    --device cuda:0

# 5. Package results
echo ""
echo "[3/4] Packaging results..."
cd "${OUTPUT_DIR}"

# Create proper folder structure (Real/ and Synth/)
mkdir -p Real Synth 2>/dev/null || true

# Move results if they exist
if [ -d "NTIRE_Val_Real" ]; then
    mv NTIRE_Val_Real/* Real/ 2>/dev/null || cp -r NTIRE_Val_Real/* Real/
fi
if [ -d "NTIRE_Val_Synth" ]; then
    mv NTIRE_Val_Synth/* Synth/ 2>/dev/null || cp -r NTIRE_Val_Synth/* Synth/
fi

# Create zip
echo "[4/4] Creating submission.zip..."
zip -r submission.zip Real Synth

echo ""
echo "============================================================"
echo "✓ SUCCESS!"
echo "============================================================"
echo "Submission file: ${OUTPUT_DIR}/submission.zip"
echo ""
echo "Next steps:"
echo "  1. Go to CodaBench NTIRE 2026 Track 2"
echo "  2. Upload submission.zip"
echo "  3. Wait for PSNR/FLOPs verification"
echo "============================================================"
