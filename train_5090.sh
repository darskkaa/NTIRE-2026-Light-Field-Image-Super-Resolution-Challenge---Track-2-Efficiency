#!/bin/bash
#============================================================
# RTX 5090 ULTRA-OPTIMIZED TRAINING SCRIPT
# Based on NTIRE 2024/2025 winning solutions
#============================================================
#
# Hardware Profile:
#   - GPU: RTX 5090 (108 TFLOPS, 32GB VRAM)
#   - CPU: AMD EPYC 9K84 (24 cores allocated)
#   - Disk: NVMe 2GB/s
#
# Optimizations Applied:
#   1. Batch Size 32 (uses ~28GB VRAM)
#   2. 16 DataLoader workers (parallel data loading)
#   3. Mixed Precision (AMP) - 2x speed boost
#   4. AdamW optimizer (better generalization)
#   5. Cosine Annealing + Warmup LR schedule
#   6. 100 epochs (fast enough on RTX 5090)
#   7. pin_memory + prefetch for faster data transfer
#
# Expected Time: ~2 hours for full training
#============================================================

set -e  # Exit on error

echo "============================================================"
echo "🚀 RTX 5090 ULTRA-OPTIMIZED TRAINING"
echo "============================================================"

# Activate environment
if [ -d "venv_lfsr" ]; then
    source venv_lfsr/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No virtual environment found. Using system Python."
fi

# Check CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Verify model before training
echo ""
echo "[1/2] Verifying model efficiency..."
python check_efficiency.py --model_name MyEfficientLFNet

# Start training
echo ""
echo "[2/2] Starting training..."
echo "      Batch Size: 32 | Workers: 16 | Epochs: 50"
echo "      Estimated Time: ~6 hours"
echo ""

# Handle memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --model_name MyEfficientLFNet \
    --angRes 5 \
    --scale_factor 4 \
    --batch_size 16 \
    --lr 0.0005 \
    --epoch 50 \
    --num_workers 8 \
    --device cuda:0

echo ""
echo "============================================================"
echo "✅ TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Model saved at: log/SR_5x5_4x/ALL/MyEfficientLFNet/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Run inference:  ./inference.sh"
echo "  2. Create submission: ./create_submission.sh"
echo ""
