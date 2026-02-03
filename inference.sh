#!/bin/bash
# ============================================================================
# Inference Script for MyEfficientLFNet - NTIRE 2026 Track 2
# ============================================================================

set -e

echo "=============================================="
echo "Running Inference - MyEfficientLFNet"
echo "=============================================="

# Activate virtual environment if exists
if [ -d "./venv_lfsr" ]; then
    source ./venv_lfsr/bin/activate
    echo "✓ Virtual environment activated"
fi

# Configuration
MODEL_NAME="MyEfficientLFNetV4_5"
ANG_RES=5
SCALE=4
DEVICE="cuda:0"

# Find the best checkpoint (latest epoch)
CKPT_DIR="./log/SR_${ANG_RES}x${ANG_RES}_${SCALE}x/ALL/$MODEL_NAME/checkpoints"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_DIR"
    echo "Please train the model first: ./train.sh"
    exit 1
fi

# Get latest checkpoint
CKPT=$(ls -t $CKPT_DIR/*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in $CKPT_DIR"
    exit 1
fi

echo "Using checkpoint: $CKPT"
echo ""

# Check for validation data
if [ ! -d "data_for_inference/SR_${ANG_RES}x${ANG_RES}_${SCALE}x" ]; then
    echo "ERROR: Inference data not found!"
    echo "Ensure validation data is in datasets/NTIRE_Val_Real/ and datasets/NTIRE_Val_Synth/"
    echo "Then run ./prepare_data.sh"
    exit 1
fi

# Run inference on Real validation set
echo "=============================================="
echo "Inference on NTIRE_Val_Real..."
echo "=============================================="
python inference.py \
    --model_name $MODEL_NAME \
    --angRes $ANG_RES \
    --scale_factor $SCALE \
    --data_name NTIRE_Val_Real \
    --path_pre_pth "$CKPT" \
    --device $DEVICE

# Run inference on Synthetic validation set
echo ""
echo "=============================================="
echo "Inference on NTIRE_Val_Synth..."
echo "=============================================="
python inference.py \
    --model_name $MODEL_NAME \
    --angRes $ANG_RES \
    --scale_factor $SCALE \
    --data_name NTIRE_Val_Synth \
    --path_pre_pth "$CKPT" \
    --device $DEVICE

echo ""
echo "=============================================="
echo "Inference complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  ./log/SR_${ANG_RES}x${ANG_RES}_${SCALE}x/ALL/$MODEL_NAME/results/TEST/"
echo ""
echo "To create submission ZIP:"
echo "  ./create_submission.sh"
echo ""
