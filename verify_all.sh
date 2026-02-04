#!/bin/bash
# ============================================================================
# NTIRE 2026 LF-SR Challenge - Complete VM Verification Script
# ============================================================================
# Run this AFTER downloading datasets from Google Drive to verify everything.
#
# Usage:
#   chmod +x verify_all.sh
#   ./verify_all.sh
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
echo "║     NTIRE 2026 LF-SR Track 2 - Complete Verification             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

ALL_PASSED=true

# ============================================================================
# 1. Check Python Environment
# ============================================================================
echo -e "\n${YELLOW}[1/4] Python Environment${NC}"

if [ -d "./venv_lfsr" ]; then
    source ./venv_lfsr/bin/activate
    echo -e "  ${GREEN}✓${NC} Virtual environment activated"
else
    echo -e "  ${YELLOW}⚠${NC} No venv found - using system Python"
fi

python3 -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo -e "  ${RED}✗${NC} PyTorch not installed"
    ALL_PASSED=false
}

# ============================================================================
# 2. Check Datasets (144 total .mat files)
# ============================================================================
echo -e "\n${YELLOW}[2/4] Dataset Verification${NC}"
echo "  Expected: 144 .mat files across 5 datasets"
echo ""

declare -A EXPECTED_COUNTS
EXPECTED_COUNTS["EPFL"]=70
EXPECTED_COUNTS["HCI_new"]=20
EXPECTED_COUNTS["HCI_old"]=10
EXPECTED_COUNTS["INRIA_Lytro"]=35
EXPECTED_COUNTS["Stanford_Gantry"]=9

TOTAL_EXPECTED=144
TOTAL_FOUND=0

for DS in EPFL HCI_new HCI_old INRIA_Lytro Stanford_Gantry; do
    EXPECTED=${EXPECTED_COUNTS[$DS]}
    TRAINING_PATH="datasets/${DS}/training"
    
    if [ -d "$TRAINING_PATH" ]; then
        FOUND=$(find "$TRAINING_PATH" -maxdepth 1 -name "*.mat" 2>/dev/null | wc -l)
        TOTAL_FOUND=$((TOTAL_FOUND + FOUND))
        
        if [ "$FOUND" -eq "$EXPECTED" ]; then
            echo -e "  ${GREEN}✓${NC} ${DS}: ${FOUND}/${EXPECTED} files"
        else
            echo -e "  ${RED}✗${NC} ${DS}: ${FOUND}/${EXPECTED} files"
            ALL_PASSED=false
        fi
    else
        echo -e "  ${RED}✗${NC} ${DS}: MISSING (expected ${EXPECTED} files)"
        ALL_PASSED=false
    fi
done

echo ""
echo "  Total: ${TOTAL_FOUND}/${TOTAL_EXPECTED} files"

# ============================================================================
# 3. Check Model Efficiency
# ============================================================================
echo -e "\n${YELLOW}[3/4] Model Efficiency Check${NC}"

if python3 check_efficiency.py --model_name MyEfficientLFNet 2>&1 | grep -q "Model meets all"; then
    echo -e "  ${GREEN}✓${NC} MyEfficientLFNet passes efficiency constraints"
else
    echo -e "  ${RED}✗${NC} Model efficiency check failed"
    ALL_PASSED=false
fi

# ============================================================================
# 4. Check Training Data (generated patches)
# ============================================================================
echo -e "\n${YELLOW}[4/4] Training Patches${NC}"

PATCHES_DIR="data_for_training/SR_5x5_4x"
if [ -d "$PATCHES_DIR" ] && [ "$(ls -A $PATCHES_DIR 2>/dev/null)" ]; then
    PATCH_COUNT=$(find "$PATCHES_DIR" -name "*.h5" 2>/dev/null | wc -l)
    echo -e "  ${GREEN}✓${NC} Training patches exist: ${PATCH_COUNT} .h5 files"
else
    echo -e "  ${YELLOW}⚠${NC} Training patches not generated yet"
    echo "      Run: ./prepare_data.sh"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                         SUMMARY                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

if [ "$ALL_PASSED" = true ] && [ "$TOTAL_FOUND" -eq "$TOTAL_EXPECTED" ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED!${NC}"
    echo ""
    echo "Ready to train. Run:"
    echo "  ./prepare_data.sh   # Generate training patches (if not done)"
    echo "  ./train.sh          # Start training"
else
    echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
    echo ""
    if [ "$TOTAL_FOUND" -ne "$TOTAL_EXPECTED" ]; then
        echo "Missing datasets. Expected structure:"
        echo "  datasets/"
        echo "  ├── EPFL/training/          (70 .mat files)"
        echo "  ├── HCI_new/training/       (20 .mat files)"
        echo "  ├── HCI_old/training/       (10 .mat files)"
        echo "  ├── INRIA_Lytro/training/   (35 .mat files)"
        echo "  └── Stanford_Gantry/training/ (9 .mat files)"
    fi
fi

echo ""
