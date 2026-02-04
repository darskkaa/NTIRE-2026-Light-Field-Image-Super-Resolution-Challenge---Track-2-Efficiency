#!/bin/bash
set -e

# Install gdown if not present
if ! command -v gdown &> /dev/null; then
    echo "gdown could not be found, installing..."
    pip install gdown
fi

# Create directory
mkdir -p data_for_inference

# Download the folder
echo "Downloading validation data..."
gdown --folder "https://drive.google.com/drive/folders/1LfPTTTtTDOPyNg3D-B_RfzwBZd4D0-HH?usp=drive_link" -O data_for_inference

echo "Download complete. Data is in data_for_inference/"
ls -R data_for_inference
