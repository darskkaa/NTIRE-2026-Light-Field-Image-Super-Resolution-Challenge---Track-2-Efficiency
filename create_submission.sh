#!/bin/bash
# ============================================================================
# Create Submission ZIP for CodaBench - NTIRE 2026 Track 2
# ============================================================================

set -e

echo "=============================================="
echo "Creating CodaBench Submission Package"
echo "=============================================="

# Configuration
MODEL_NAME="MyEfficientLFNet"
ANG_RES=5
SCALE=4

# Results directory
# Results handling now split per dataset below

# Create submission directory
SUBMISSION_DIR="./submission_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SUBMISSION_DIR/Real"
mkdir -p "$SUBMISSION_DIR/Synth"

echo "Copying results..."

# Copy Real results
REAL_RESULTS_DIR="./log/SR_${ANG_RES}x${ANG_RES}_${SCALE}x/NTIRE_Val_Real/$MODEL_NAME/results/TEST/NTIRE_Val_Real"
if [ -d "$REAL_RESULTS_DIR" ]; then
    cp -r "$REAL_RESULTS_DIR"/* "$SUBMISSION_DIR/Real/"
    echo "  ✓ NTIRE_Val_Real copied"
else
    echo "  ✗ NTIRE_Val_Real not found at $REAL_RESULTS_DIR"
fi

# Copy Synthetic results
SYNTH_RESULTS_DIR="./log/SR_${ANG_RES}x${ANG_RES}_${SCALE}x/NTIRE_Val_Synth/$MODEL_NAME/results/TEST/NTIRE_Val_Synth"
if [ -d "$SYNTH_RESULTS_DIR" ]; then
    cp -r "$SYNTH_RESULTS_DIR"/* "$SUBMISSION_DIR/Synth/"
    echo "  ✓ NTIRE_Val_Synth copied"
else
    echo "  ✗ NTIRE_Val_Synth not found at $SYNTH_RESULTS_DIR"
fi

# Create ZIP
cd "$SUBMISSION_DIR"
ZIP_NAME="${MODEL_NAME}_submission.zip"
zip -r "../$ZIP_NAME" Real/ Synth/
cd ..

echo ""
echo "=============================================="
echo "Submission package created!"
echo "=============================================="
echo ""
echo "ZIP file: ./$ZIP_NAME"
echo ""
echo "NEXT STEPS:"
echo "1. Upload to CodaBench Track 2:"
echo "   https://www.codabench.org/competitions/12927/"
echo ""
echo "2. Email fact sheet and code to:"
echo "   ntire.lfsr@outlook.com"
echo ""
echo "Submission directory preserved at: $SUBMISSION_DIR"
echo ""
