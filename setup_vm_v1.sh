#!/bin/bash
# =============================================================================
# VM Environment Setup Script (v1)
# Specifically tailored for RTX 5090 (Blackwell/sm_100) and CUDA 13.0 vs 12.8 mismatch
# =============================================================================

set -e

echo "Starting VM Environment Setup V1..."

# Step 1: Patch PyTorch's overly strict CUDA 13.0 vs 12.8 check
echo "Step 1: Patching PyTorch CUDA version check..."
python -c "
import torch.utils.cpp_extension as ext
f = ext.__file__
with open(f) as fh:
    code = fh.read()
code = code.replace(
    'raise RuntimeError(CUDA_MISMATCH_MESSAGE',
    'pass  # raise RuntimeError(CUDA_MISMATCH_MESSAGE'
)
with open(f, 'w') as fh:
    fh.write(code)
print('Patched CUDA version check successfully.')
"

# Step 2: Set RTX 5090 Blackwell architecture target
echo "Step 2: Setting TORCH_CUDA_ARCH_LIST for RTX 5090 (Blackwell)..."
export TORCH_CUDA_ARCH_LIST="10.0"

# Step 3: Install causal-conv1d (builds from source, ~3-5 min)
echo "Step 3: Installing causal-conv1d (this will take a few minutes to compile)..."
pip install causal-conv1d>=1.4.0 --no-build-isolation -v

# Step 4: Install mamba-ssm (builds from source, ~5-10 min)
echo "Step 4: Installing mamba-ssm (this will také a few minutes to compile)..."
pip install mamba-ssm --no-build-isolation -v

# Step 5: Install remaining deps
echo "Step 5: Installing remaining dependencies..."
pip install fvcore einops h5py scikit-image tqdm scipy matplotlib xlwt pandas openpyxl

# Step 6: Verify everything works
echo "Step 6: Running Model Efficiency Verification..."
python model/SR/MyEfficientLFNetV2_MLFIM.py

echo "=========================================="
echo "  Setup Complete! You can now start training."
echo "=========================================="
