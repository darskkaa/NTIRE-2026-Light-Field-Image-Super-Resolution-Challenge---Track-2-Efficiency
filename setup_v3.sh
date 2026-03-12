#!/bin/bash
# =============================================================================
# Automated VM Setup for MyEfficientLFNetV3 (Mamba)
# =============================================================================
# Run this on your Vast.ai RTX 5090 instance.
# Note: Since you use 'vastai/pytorch_cuda-13.0.2-auto/jupyter', PyTorch is
# likely already installed. We just install the deps and compile Mamba.

set -e

echo "======================================================="
echo "1. Installing base requirements..."
echo "======================================================="
pip install packaging ninja wheel setuptools xlwt
pip install einops h5py scipy imageio tqdm scikit-image fvcore

echo ""
echo "======================================================="
echo "2. Compiling causal-conv1d (Step 1/2 for Mamba)..."
echo "   NOTE: This will print a ton of warnings. Ignore them!"
echo "   NOTE: Using --no-build-isolation to use your system's CUDA 13 PyTorch."
echo "======================================================="
MAX_JOBS=4 pip install causal-conv1d>=1.2.0 --no-build-isolation

echo ""
echo "======================================================="
echo "3. Compiling mamba-ssm (Step 2/2 for Mamba)..."
echo "   NOTE: This will also print scary CUDA/nvcc warnings."
echo "         It will fall back to native PyTorch when run."
echo "======================================================="
MAX_JOBS=4 pip install mamba-ssm --no-build-isolation

echo ""
echo "======================================================="
echo "✅ SETUP COMPLETE!"
echo "You can now run: bash train_v3_stage1.sh"
echo "======================================================="
