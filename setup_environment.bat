@echo off
REM ============================================================================
REM NTIRE 2026 LF-SR Challenge - Track 2 (Efficiency) Setup Script for Windows
REM ============================================================================
REM This script sets up the complete environment for training MyEfficientLFNet
REM on a Windows machine with a CUDA-capable GPU
REM
REM Usage: Run in Command Prompt or PowerShell
REM   setup_environment.bat
REM ============================================================================

echo ==============================================
echo NTIRE 2026 LF-SR Track 2 - Environment Setup
echo ==============================================

REM ============================================================================
REM STEP 1: Check for GPU
REM ============================================================================
echo.
echo [1/5] Checking for NVIDIA GPU...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if %errorlevel% neq 0 (
    echo ERROR: No NVIDIA GPU detected or nvidia-smi not found.
    echo Please ensure CUDA drivers are installed.
    pause
    exit /b 1
)
echo GPU detected!

REM ============================================================================
REM STEP 2: Create Virtual Environment
REM ============================================================================
echo.
echo [2/5] Creating Python virtual environment...

if not exist "venv_lfsr" (
    python -m venv venv_lfsr
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate
call venv_lfsr\Scripts\activate.bat
echo Virtual environment activated.

REM ============================================================================
REM STEP 3: Install PyTorch with CUDA
REM ============================================================================
echo.
echo [3/5] Installing PyTorch with CUDA support...
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Verify
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

REM ============================================================================
REM STEP 4: Install Dependencies
REM ============================================================================
echo.
echo [4/5] Installing project dependencies...
pip install -r requirements.txt

REM ============================================================================
REM STEP 5: Create Directory Structure
REM ============================================================================
echo.
echo [5/5] Creating directory structure...

mkdir datasets\EPFL\training 2>nul
mkdir datasets\HCI_new\training 2>nul
mkdir datasets\HCI_old\training 2>nul
mkdir datasets\INRIA_Lytro\training 2>nul
mkdir datasets\Stanford_Gantry\training 2>nul
mkdir datasets\NTIRE_Val_Real\inference 2>nul
mkdir datasets\NTIRE_Val_Synth\inference 2>nul
mkdir data_for_training 2>nul
mkdir data_for_inference 2>nul
mkdir log 2>nul
mkdir pth 2>nul

echo Directories created.

REM ============================================================================
REM Verify Model
REM ============================================================================
echo.
echo Verifying MyEfficientLFNet model...
python check_efficiency.py --model_name MyEfficientLFNet

echo.
echo ==============================================
echo Environment setup complete!
echo ==============================================
echo.
echo NEXT STEPS:
echo.
echo 1. DOWNLOAD TRAINING DATA (open in browser):
echo    https://stuxidianeducn-my.sharepoint.com/personal/zyliang_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%%2Fpersonal%%2Fzyliang%%5Fstu%%5Fxidian%%5Fedu%%5Fcn%%2FDocuments%%2Fdatasets
echo.
echo    Extract each folder contents to:
echo    - datasets\EPFL\training\
echo    - datasets\HCI_new\training\
echo    - datasets\HCI_old\training\
echo    - datasets\INRIA_Lytro\training\
echo    - datasets\Stanford_Gantry\training\
echo.
echo 2. PREPARE DATA (after downloading):
echo    venv_lfsr\Scripts\activate
echo    python Generate_Data_for_Training.py --angRes 5 --scale_factor 4 --src_data_path ./datasets/ --save_data_path ./
echo.
echo 3. TRAIN MODEL:
echo    python train.py --model_name MyEfficientLFNet --angRes 5 --scale_factor 4 --batch_size 4 --epoch 51
echo.

pause
