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
RESULTS_DIR="./log/SR_${ANG_RES}x${ANG_RES}_${SCALE}x/ALL/$MODEL_NAME/results/TEST"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    echo "Please run inference first: ./inference.sh"
    exit 1
fi

# Create submission directory
SUBMISSION_DIR="./submission_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SUBMISSION_DIR/Real"
mkdir -p "$SUBMISSION_DIR/Synth"

echo "Copying results..."

# Copy Real results
if [ -d "$RESULTS_DIR/NTIRE_Val_Real" ]; then
    cp -r "$RESULTS_DIR/NTIRE_Val_Real"/* "$SUBMISSION_DIR/Real/"
    echo "  ✓ NTIRE_Val_Real copied"
else
    echo "  ✗ NTIRE_Val_Real not found"
fi

# Copy Synthetic results
if [ -d "$RESULTS_DIR/NTIRE_Val_Synth" ]; then
    cp -r "$RESULTS_DIR/NTIRE_Val_Synth"/* "$SUBMISSION_DIR/Synth/"
    echo "  ✓ NTIRE_Val_Synth copied"
else
    echo "  ✗ NTIRE_Val_Synth not found"
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
