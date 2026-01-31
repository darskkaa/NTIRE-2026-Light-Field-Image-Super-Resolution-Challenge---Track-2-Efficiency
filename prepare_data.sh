#!/bin/bash
# ============================================================================
# Data Preparation Script for NTIRE 2026 LF-SR Challenge
# ============================================================================
# Run this AFTER downloading the datasets from OneDrive
# ============================================================================

set -e

echo "=============================================="
echo "NTIRE 2026 LF-SR - Data Preparation"
echo "=============================================="

# Activate virtual environment if exists
if [ -d "./venv_lfsr" ]; then
    source ./venv_lfsr/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check if datasets exist
echo ""
echo "Checking for datasets..."

DATASETS=("EPFL" "HCI_new" "HCI_old" "INRIA_Lytro" "Stanford_Gantry")
MISSING=0

for ds in "${DATASETS[@]}"; do
    if [ -d "datasets/$ds/training" ] && [ "$(ls -A datasets/$ds/training 2>/dev/null)" ]; then
        echo "  ✓ $ds found"
    else
        echo "  ✗ $ds MISSING or empty"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "WARNING: Some datasets are missing!"
    echo "Training will proceed with available datasets only."
    echo "This is fine for testing, but download all for submission."
    echo ""
fi

echo ""
echo "All datasets found!"
echo ""

# ============================================================================
# Generate Training Patches
# ============================================================================
echo "=============================================="
echo "Generating training patches (5x5 angular, 4x SR)..."
echo "This may take 10-30 minutes depending on your storage speed."
echo "=============================================="

python generate_data_fix.py \
    --angRes 5 \
    --scale_factor 4 \
    --data_for training \
    --src_data_path ./datasets/ \
    --save_data_path ./

echo ""
echo "✓ Training data generated at: ./data_for_training/SR_5x5_4x/"

# ============================================================================
# Check for validation data (optional)
# ============================================================================
echo ""
echo "Checking for validation data..."

VAL_FOUND=0
if [ -d "datasets/NTIRE_Val_Real/inference" ] && [ "$(ls -A datasets/NTIRE_Val_Real/inference 2>/dev/null)" ]; then
    echo "  ✓ NTIRE_Val_Real found"
    VAL_FOUND=1
else
    echo "  - NTIRE_Val_Real not found (optional)"
fi

if [ -d "datasets/NTIRE_Val_Synth/inference" ] && [ "$(ls -A datasets/NTIRE_Val_Synth/inference 2>/dev/null)" ]; then
    echo "  ✓ NTIRE_Val_Synth found"
    VAL_FOUND=1
else
    echo "  - NTIRE_Val_Synth not found (optional)"
fi

# Generate validation data if present
if [ $VAL_FOUND -eq 1 ]; then
    echo ""
    echo "Generating validation/inference data..."
    python Generate_Data_for_inference.py \
        --angRes 5 \
        --scale_factor 4 \
        --data_for inference \
        --src_data_path ./datasets/ \
        --save_data_path ./
    echo "✓ Validation data generated at: ./data_for_inference/SR_5x5_4x/"
fi

echo ""
echo "=============================================="
echo "Data preparation complete!"
echo "=============================================="
echo ""
echo "Ready to train! Run:"
echo "  ./train.sh"
echo ""
echo "Or manually:"
echo "  python train.py --model_name MyEfficientLFNet --angRes 5 --scale_factor 4 --batch_size 4 --epoch 51"
echo ""
