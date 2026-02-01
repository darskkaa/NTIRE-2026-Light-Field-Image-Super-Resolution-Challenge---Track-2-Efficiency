#!/bin/bash
# Helper script to process downloaded NTIRE Validation Set and create submission

set -e

# Activate virtual environment if exists
if [ -d "./venv_lfsr" ]; then
    source ./venv_lfsr/bin/activate
    echo "✓ Virtual environment activated"
fi

# 0. Download validation set if missing
if [ ! -f "validation_set.zip" ]; then
    echo "Attempting to download validation_set.zip from Google Drive..."
    FILE_ID="1ik0yzmuuIG2SbPyw-PtukmRztZL24cMk"
    
    # Try using gdown (more reliable)
    if python -c "import gdown" &> /dev/null; then
        python -m gdown "$FILE_ID" -O validation_set.zip
    else
        echo "Installing gdown..."
        pip install gdown
        python -m gdown "$FILE_ID" -O validation_set.zip
    fi
fi

# 1. Unzip the validation set
    unzip -o validation_set.zip -d datasets/
    
    # Ensure folder structure is correct (sometimes zips have different internal structures)
    # Goal: datasets/NTIRE_Val_Real/inference/*.png and datasets/NTIRE_Val_Synth/inference/*.png
    
    # Simple fixup if they unzipped into a subfolder
    if [ -d "datasets/validation_set/NTIRE_Val_Real" ]; then
        mv datasets/validation_set/* datasets/
        rmdir datasets/validation_set
    fi
else
    echo "WARNING: validation_set.zip not found. Assuming you manually placed files in datasets/"
fi

# 2. Prepare Data (Generate H5) for Inference Only
echo "Running data preparation (Inference only)..."
python Generate_Data_for_inference.py \
    --angRes 5 \
    --scale_factor 4 \
    --data_for inference \
    --src_data_path ./datasets/ \
    --save_data_path ./

# 3. Run Inference
echo "Running inference on validation sets..."
bash inference.sh

# 4. Create Submission
echo "Creating submission zip..."
bash create_submission.sh
