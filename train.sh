#!/bin/bash
# ============================================================================
# Training Script for MyEfficientLFNet - NTIRE 2026 Track 2
# ============================================================================

set -e

echo "=============================================="
echo "Training MyEfficientLFNet"
echo "=============================================="

# Activate virtual environment if exists
if [ -d "./venv_lfsr" ]; then
    source ./venv_lfsr/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check if training data exists
if [ ! -d "data_for_training/SR_5x5_4x" ]; then
    echo "ERROR: Training data not found!"
    echo "Run ./prepare_data.sh first"
    exit 1
fi

# Display GPU info
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Verify model efficiency before training
echo "Verifying model efficiency..."
python check_efficiency.py --model_name MyEfficientLFNet
echo ""

# Training configuration
MODEL_NAME="MyEfficientLFNet"
ANG_RES=5
SCALE=4
BATCH_SIZE=4
LR=0.0002
EPOCHS=51
DEVICE="cuda:0"

echo "=============================================="
echo "Training Configuration:"
echo "  Model:       $MODEL_NAME"
echo "  Angular:     ${ANG_RES}x${ANG_RES}"
echo "  Scale:       ${SCALE}x"
echo "  Batch Size:  $BATCH_SIZE"
echo "  LR:          $LR"
echo "  Epochs:      $EPOCHS"
echo "  Device:      $DEVICE"
echo "=============================================="
echo ""
echo "Starting training..."
echo "Logs will be saved to: ./log/SR_${ANG_RES}x${ANG_RES}_${SCALE}x/ALL/$MODEL_NAME/"
echo ""

# Run training
python train.py \
    --model_name $MODEL_NAME \
    --angRes $ANG_RES \
    --scale_factor $SCALE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epoch $EPOCHS \
    --device $DEVICE \
    --use_pre_ckpt False

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Checkpoints saved to:"
echo "  ./log/SR_${ANG_RES}x${ANG_RES}_${SCALE}x/ALL/$MODEL_NAME/checkpoints/"
echo ""
echo "To run inference:"
echo "  ./inference.sh"
echo ""
