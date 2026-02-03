#!/bin/bash
# ============================================================
# train_v4_5_championship.sh
# Ultimate Championship Training - MyEfficientLFNetV4_5 (FINAL)
# Based on NTIRE 2025 SOTA Analysis + Mamba Integration
# ============================================================

set -e

echo "============================================================"
echo "🏆 ULTIMATE CHAMPIONSHIP - MyEfficientLFNetV4_5 (FINAL)"
echo "============================================================"
echo "Based on 2025 SOTA: LFTransMamba, MCMamba, TriFormer"
echo "Features: Mamba SSM, Multi-Scale Spatial, Channel Attention"
echo ""

cd "$(dirname "$0")"

# Virtual environment
if [ -f "venv_lfsr/bin/activate" ]; then
    source venv_lfsr/bin/activate
    echo "✓ Virtual environment activated"
fi

# CUDA setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Device check
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "[1/2] Verifying model constraints..."
python check_efficiency.py --model_name MyEfficientLFNetV4_5

echo ""
echo "[2/2] Starting championship training..."
echo "      Model: MyEfficientLFNetV4_5 (Mamba + Multi-Scale)"
echo "      Batch Size: 20 | Workers: 10 | Epochs: 80"
echo "      LR: 2e-4 with Cosine Annealing"
echo "      Loss: L1 + FFT (0.05) + Edge (0.02)"
echo ""

python train.py \
    --model_name MyEfficientLFNetV4_5 \
    --angRes 5 \
    --scale_factor 4 \
    --batch_size 20 \
    --lr 0.0002 \
    --epoch 80 \
    --num_workers 10 \
    --device cuda:0 \
    --data_name ALL

echo ""
echo "============================================================"
echo "🏆 CHAMPIONSHIP TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run inference: ./inference.sh"
echo "  2. Create submission: ./create_submission.sh"
echo "  3. Validate: python validate_submission.py submission.zip"
echo "============================================================"
