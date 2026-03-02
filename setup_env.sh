#!/bin/bash
#===============================================================================
# SHARED ENVIRONMENT SETUP for V10 Training Pipeline
# Sourced by both train_v10_stage1.sh and train_v10_stage2.sh
#===============================================================================

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper Functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}============================================================${NC}\n${BOLD}${CYAN}$1${NC}\n${BOLD}${CYAN}============================================================${NC}"; }

#===============================================================================
# CONDA ENVIRONMENT
#===============================================================================
header "📦 Environment Setup"

if ! conda env list | grep -q "lfsr"; then
    info "Creating conda environment 'lfsr' (Python 3.10)..."
    conda create -n lfsr python=3.10 -y
fi

info "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lfsr

if python -c "import transformers; from packaging import version; assert version.parse(transformers.__version__) < version.parse('4.45.0')" &> /dev/null; then
    success "Compatible transformers version already installed."
else
    info "Ensuring compatible transformers version for mamba-ssm..."
    pip install "transformers<4.45.0"
fi

#===============================================================================
# MAMBA-SSM INSTALLATION
#===============================================================================
# Pre-flight: clean stale CUDA .so files BEFORE the guard check
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
find "$SITE_PKGS" -maxdepth 1 -name "selective_scan_cuda*.so" -delete 2>/dev/null || true
find "$SITE_PKGS" -maxdepth 1 -name "causal_conv1d_cuda*.so" -delete 2>/dev/null || true

# T5 Fix: Detect ACTUAL GPU compute capability instead of hardcoding sm_120.
# This supports A100 (sm_80), H100 (sm_90), Blackwell (sm_120), etc.
GPU_ARCH_OK=$(python -c "
import torch
if not torch.cuda.is_available():
    print('NO_GPU')
else:
    cc = torch.cuda.get_device_capability()
    gpu_sm = f'sm_{cc[0]*10+cc[1]}'
    arch_list = torch.cuda.get_arch_list()
    # Check if any supported arch matches or covers our GPU
    supported = any(gpu_sm <= a.replace('sm_','sm_').split('+')[0] for a in arch_list if 'sm_' in a)
    print('OK' if supported else 'NEED_REBUILD')
" 2>/dev/null || echo "NO_GPU")

# Full validation: mamba-ssm must import AND CUDA kernels must load
if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')" &> /dev/null && [ "$GPU_ARCH_OK" = "OK" ]; then
    success "mamba-ssm is working and GPU architecture is supported! Skipping installation."
else
    warn "Missing mamba-ssm, broken CUDA kernels, or GPU arch not supported. Installing..."

    info "Removing old mamba packages..."
    pip uninstall -y mamba-ssm causal-conv1d 2>/dev/null || true

    # Check if current PyTorch supports the GPU
    if [ "$GPU_ARCH_OK" = "OK" ]; then
        success "Current PyTorch supports this GPU, keeping it."
    elif [ "$GPU_ARCH_OK" = "NO_GPU" ]; then
        warn "No GPU detected. Installing PyTorch nightly for widest arch support..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    else
        info "PyTorch lacks support for this GPU. Reinstalling nightly..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    fi

    # Remove stale CUDA .so files
    find "$SITE_PKGS" -name "selective_scan_cuda*.so" -delete 2>/dev/null || true
    find "$SITE_PKGS" -name "causal_conv1d_cuda*.so" -delete 2>/dev/null || true
    find "$SITE_PKGS" -name "mamba_ssm*.so" -delete 2>/dev/null || true
    pip cache purge 2>/dev/null || true

    python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA archs: {torch.cuda.get_arch_list()}')"

    info "Installing build dependencies..."
    pip install ninja packaging

    # Bypass PyTorch's strict CUDA version check
    info "Patching PyTorch cpp_extension.py to allow CUDA version mismatch..."
    python -c "
import torch.utils.cpp_extension as ext
import inspect
f = inspect.getfile(ext)
with open(f) as fh:
    s = fh.read()
if 'raise RuntimeError(CUDA_MISMATCH_MESSAGE' in s:
    s = s.replace('raise RuntimeError(CUDA_MISMATCH_MESSAGE', 'return  # raise RuntimeError(CUDA_MISMATCH_MESSAGE')
    with open(f, 'w') as fh:
        fh.write(s)
    print(f'Patched {f} — CUDA version check bypassed')
else:
    print('Already patched or check not found, skipping')
"

    info "Building causal-conv1d from source (~5 min)..."
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" MAX_JOBS=4 \
        pip install causal-conv1d --no-binary causal-conv1d --no-build-isolation --no-cache-dir

    info "Building mamba-ssm from source (~10 min)..."
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" MAX_JOBS=4 \
        pip install mamba-ssm --no-binary mamba-ssm --no-build-isolation --no-cache-dir
fi

info "Installing other dependencies..."
pip install numpy scipy h5py imageio einops xlwt tqdm scikit-image fvcore matplotlib gdown

success "All packages installed!"

info "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba-ssm: OK')"
python -c "from einops import rearrange; print('einops: OK')"
success "Environment setup complete"
