#!/bin/bash
# ============================================================================
# Fix Dataset Structure Script
# ============================================================================
# usage: ./fix_structure.sh
#
# Run this AFTER manually uploading datasets (e.g. via SCP) to fix 
# nested directory structures like:
# datasets/EPFL/EPFL/training -> datasets/EPFL/training
# ============================================================================

set -e

echo "Fixing dataset structure..."

cd datasets 2>/dev/null || { echo "datasets/ dir not found"; exit 1; }

for DS in EPFL HCI_new HCI_old INRIA_Lytro Stanford_Gantry; do
    # Case 1: Nested DS/DS/training (Common in backups)
    if [ -d "${DS}/${DS}/training" ]; then
        echo "  Fixing nested ${DS}..."
        # Move training up
        mv "${DS}/${DS}/training" "${DS}/"
        # Remove empty nested folder
        rmdir "${DS}/${DS}" 2>/dev/null || true
        echo "  ✓ ${DS} fixed"
    
    # Case 2: Nested DS/training (Correct, do nothing)
    elif [ -d "${DS}/training" ]; then
        echo "  ✓ ${DS} structure is correct"
        
    else
        echo "  - ${DS}/training not found (checking deeper...)"
        # Try to find 'training' anywhere inside and move it
        FOUND=$(find "${DS}" -type d -name "training" | head -n 1)
        if [ -n "$FOUND" ]; then
             echo "    Found at: $FOUND"
             mv "$FOUND" "${DS}/"
             echo "    ✓ moved to ${DS}/training"
        fi
    fi
done

cd ..
echo "Structure fix complete."
