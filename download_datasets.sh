#!/bin/bash
# ============================================================================
# NTIRE 2026 LF-SR - Dataset Download Script
# ============================================================================
# Downloads all training datasets from Google Drive and extracts them.
#
# Source: https://drive.google.com/drive/folders/1kxAGVzMTg4R-qncuj8x2qWqBmlQdd3zC
#
# Expected files:
#   - EPFL.zip           (20.46 GB)
#   - HCI_new.zip        (1.54 GB)
#   - HCI_old.zip        (3.2 GB)
#   - INRIA_Lytro.zip    (7.97 GB)
#   - Stanford_Gantry.zip(3.86 GB)
#   Total: ~37 GB
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     NTIRE 2026 LF-SR - Downloading Datasets from Google Drive    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Activate venv if exists
if [ -d "./venv_lfsr" ]; then
    source ./venv_lfsr/bin/activate
fi

# Install gdown if not present
if ! command -v gdown &> /dev/null; then
    echo -e "${YELLOW}Installing gdown...${NC}"
    pip install gdown -q
fi

# Google Drive folder ID (NEW LINK)
GDRIVE_FOLDER="1j3F9iD5SqAl14LEAaOdN8wTnHrYuKwGd"

# Create datasets directory
mkdir -p datasets

echo -e "${YELLOW}[1/3] Downloading datasets from Google Drive...${NC}"
echo "      This will download ~37 GB. Please be patient."
echo ""

# Download entire folder
cd datasets
gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER}" --remaining-ok

echo ""
echo -e "${YELLOW}[2/3] Extracting ZIP files...${NC}"

# Fix for gdown creating a subfolder (e.g. training_nicre/)
# Move all ZIPs to current directory
mv */*.zip . 2>/dev/null || true

# Extract each dataset
for ZIP in EPFL.zip HCI_new.zip HCI_old.zip INRIA_Lytro.zip Stanford_Gantry.zip; do
    if [ -f "$ZIP" ]; then
        DS_NAME="${ZIP%.zip}"
        echo "  Extracting $ZIP..."
        unzip -q -o "$ZIP" -d "./${DS_NAME}_temp"
        
        # Handle nested structure (Dataset/Dataset/training -> Dataset/training)
        if [ -d "./${DS_NAME}_temp/${DS_NAME}/${DS_NAME}/training" ]; then
            mkdir -p "./${DS_NAME}/training"
            mv "./${DS_NAME}_temp/${DS_NAME}/${DS_NAME}/training/"* "./${DS_NAME}/training/" 2>/dev/null || true
        elif [ -d "./${DS_NAME}_temp/${DS_NAME}/training" ]; then
            mkdir -p "./${DS_NAME}/training"
            mv "./${DS_NAME}_temp/${DS_NAME}/training/"* "./${DS_NAME}/training/" 2>/dev/null || true
        elif [ -d "./${DS_NAME}_temp/training" ]; then
            mkdir -p "./${DS_NAME}/training"
            mv "./${DS_NAME}_temp/training/"* "./${DS_NAME}/training/" 2>/dev/null || true
        fi
        
        # Cleanup temp dir and zip
        rm -rf "./${DS_NAME}_temp"
        rm -f "$ZIP"
        echo -e "  ${GREEN}✓${NC} $DS_NAME extracted"
    else
        echo -e "  ${RED}✗${NC} $ZIP not found"
    fi
done

cd ..

echo ""
echo -e "${YELLOW}[3/3] Verifying datasets...${NC}"

# Quick verification
TOTAL=0
for DS in EPFL HCI_new HCI_old INRIA_Lytro Stanford_Gantry; do
    COUNT=$(find "datasets/${DS}/training" -name "*.mat" 2>/dev/null | wc -l)
    TOTAL=$((TOTAL + COUNT))
    echo "  ${DS}: ${COUNT} files"
done

echo ""
echo "  Total: ${TOTAL}/144 files"

if [ "$TOTAL" -eq 144 ]; then
    echo ""
    echo -e "${GREEN}🎉 ALL DATASETS DOWNLOADED SUCCESSFULLY!${NC}"
    echo ""
    echo "Next steps:"
    echo "  ./prepare_data.sh   # Generate training patches"
    echo "  ./train.sh          # Start training"
else
    echo ""
    echo -e "${RED}⚠ Some files may be missing. Run ./verify_all.sh for details.${NC}"
fi
