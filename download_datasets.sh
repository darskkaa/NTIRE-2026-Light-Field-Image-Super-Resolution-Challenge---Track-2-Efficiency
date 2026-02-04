#!/bin/bash
# ============================================================================
# NTIRE 2026 LF-SR - Dataset Download Script (Individual Links)
# ============================================================================
# Downloads all training datasets from Google Drive individually.
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

# Create datasets directory
mkdir -p datasets
cd datasets

download_file() {
    local id=$1
    local name=$2
    echo -e "${YELLOW}Downloading $name...${NC}"
    if [ -f "$name" ]; then
        echo "  $name already exists. Skipping."
    else
        # Try gdown with ID
        gdown "$id" -O "$name" || echo -e "${RED}Failed to download $name${NC}"
    fi
}

echo -e "${YELLOW}[1/3] Downloading datasets...${NC}"

# Stanford_Gantry
download_file "1stqpt2c0LCbglZg8rjipCoPP4o-NC9q3" "Stanford_Gantry.zip"

# INRIA_Lytro
download_file "1XNMTwczPpooktQUjVWLjgQpXRi-Gf4RQ" "INRIA_Lytro.zip"

# HCI_old
download_file "1bNYAizmiAqcxiCEjoNM_g9VDkU0RgNRG" "HCI_old.zip"

# HCI_new
download_file "1IasKKF8ivxE_H6Gm7RGdci-cvi-BHfl9" "HCI_new.zip"

# EPFL
download_file "19aBn1DvW4ynSLjAPhDeB30p_umwBO8EN" "EPFL.zip"

echo ""
echo -e "${YELLOW}[2/3] Extracting ZIP files...${NC}"

# Extract each dataset
for ZIP in EPFL.zip HCI_new.zip HCI_old.zip INRIA_Lytro.zip Stanford_Gantry.zip; do
    if [ -f "$ZIP" ]; then
        DS_NAME="${ZIP%.zip}"
        echo "  Extracting $ZIP..."
        unzip -q -o "$ZIP" -d "./${DS_NAME}_temp"
        
        # Handle nested structure (Dataset/Dataset/training -> Dataset/training)
        # Using a loop to check different depths
        TARGET="./${DS_NAME}/training"
        mkdir -p "$TARGET"
        
        MOVED=0
        for SRC in "./${DS_NAME}_temp/${DS_NAME}/${DS_NAME}/training" \
                   "./${DS_NAME}_temp/${DS_NAME}/training" \
                   "./${DS_NAME}_temp/training"; do
            if [ -d "$SRC" ]; then
                mv "$SRC/"* "$TARGET/" 2>/dev/null || true
                MOVED=1
                break
            fi
        done
        
        if [ $MOVED -eq 0 ]; then
             echo -e "  ${RED}⚠ Could not find training folder inside $ZIP${NC}"
        else
             echo -e "  ${GREEN}✓${NC} $DS_NAME extracted"
        fi

        # Cleanup temp dir and zip
        rm -rf "./${DS_NAME}_temp"
        rm -f "$ZIP"
    else
        echo -e "  ${RED}✗${NC} $ZIP not found (Download failed?)"
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
else
    echo ""
    echo -e "${RED}⚠ Some files may be missing. Run ./verify_all.sh for details.${NC}"
fi
