#!/bin/bash
# ============================================================
# train_v3_championship.sh
# Training script for MyEfficientLFNetV3 - Championship Model
# ============================================================

set -e

echo "============================================================"
echo "🏆 CHAMPIONSHIP MODEL TRAINING - MyEfficientLFNetV3"
echo "============================================================"

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f "venv_lfsr/bin/activate" ]; then
    source venv_lfsr/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Handle memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "[1/2] Verifying model efficiency..."
python check_efficiency.py --model_name MyEfficientLFNetV3

echo ""
echo "[2/2] Starting training..."
echo "      Model: MyEfficientLFNetV3 (Championship)"
echo "      Batch Size: 24 | Workers: 12 | Epochs: 80"
echo "      Learning Rate: 2e-4 (Cosine Annealing)"
echo "      Estimated Time: ~8 hours"
echo ""

python train.py \
    --model_name MyEfficientLFNetV3 \
    --angRes 5 \
    --scale_factor 4 \
    --batch_size 24 \
    --lr 0.0002 \
    --epoch 80 \
    --num_workers 12 \
    --device cuda:0 \
    --data_name ALL

echo ""
echo "============================================================"
echo "🏆 CHAMPIONSHIP TRAINING COMPLETE!"
echo "============================================================"
