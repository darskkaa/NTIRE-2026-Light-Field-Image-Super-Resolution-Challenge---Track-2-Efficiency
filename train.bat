@echo off
REM ============================================================================
REM Training Script for MyEfficientLFNet - Windows
REM ============================================================================

echo ==============================================
echo Training MyEfficientLFNet
echo ==============================================

REM Activate virtual environment
if exist "venv_lfsr\Scripts\activate.bat" (
    call venv_lfsr\Scripts\activate.bat
    echo Virtual environment activated.
)

REM Check training data
if not exist "data_for_training\SR_5x5_4x" (
    echo ERROR: Training data not found!
    echo Please run data preparation first:
    echo   python Generate_Data_for_Training.py --angRes 5 --scale_factor 4 --src_data_path ./datasets/ --save_data_path ./
    pause
    exit /b 1
)

REM Display GPU info
echo.
echo GPU Info:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo.

REM Verify model
echo Verifying model efficiency...
python check_efficiency.py --model_name MyEfficientLFNet
echo.

echo ==============================================
echo Training Configuration:
echo   Model:       MyEfficientLFNet
echo   Angular:     5x5
echo   Scale:       4x
echo   Batch Size:  4
echo   LR:          0.0002
echo   Epochs:      51
echo ==============================================
echo.
echo Starting training... (Ctrl+C to stop)
echo Logs saved to: ./log/SR_5x5_4x/ALL/MyEfficientLFNet/
echo.

python train.py ^
    --model_name MyEfficientLFNet ^
    --angRes 5 ^
    --scale_factor 4 ^
    --batch_size 4 ^
    --lr 0.0002 ^
    --epoch 51 ^
    --device cuda:0 ^
    --use_pre_ckpt False

echo.
echo ==============================================
echo Training complete!
echo ==============================================
pause
