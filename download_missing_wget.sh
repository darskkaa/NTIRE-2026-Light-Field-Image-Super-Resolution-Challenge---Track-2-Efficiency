#!/bin/bash
# ============================================================================
# Emergency Download Script - Wget Method
# ============================================================================
# Uses wget with confirm-token extraction to try and bypass gdown issues.
# Only downloads missing: EPFL.zip (20GB) and INRIA_Lytro.zip (8GB).
# ============================================================================

set -e

mkdir -p datasets
cd datasets

download_gdrive_wget() {
    FILEID=$1
    FILENAME=$2
    echo "Attempting to download $FILENAME with wget..."
    
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$FILEID" -O "$FILENAME"
    rm -rf /tmp/cookies.txt
}

# 1. Download EPFL (Missing)
if [ ! -f "EPFL.zip" ] && [ ! -d "EPFL/training" ]; then
    download_gdrive_wget "19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN" "EPFL.zip"
else
    echo "✓ EPFL already exists"
fi

# 2. Download INRIA_Lytro (Missing)
if [ ! -f "INRIA_Lytro.zip" ] && [ ! -d "INRIA_Lytro/training" ]; then
    download_gdrive_wget "1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ" "INRIA_Lytro.zip"
else
    echo "✓ INRIA_Lytro already exists"
fi

cd ..
echo "Download attempt complete. Running extraction..."
./download_datasets.sh
