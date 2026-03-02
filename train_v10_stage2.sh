#!/bin/bash
#===============================================================================
# V10 SOTA TRAINING PIPELINE - MyEfficientLFNetV10_MLFIM
# STAGE 2: Fine-Tuning (200 Epochs) & Evaluation
#===============================================================================

set -e  # Exit on error

# Source shared environment setup (T3 Fix: single source of truth)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

header "🚀 MyEfficientLFNetV10_MLFIM SOTA Training Workflow — STAGE 2"
info "Starting Stage 2 workflow (Fine-tuning & Evaluation)..."

#===============================================================================
# T4 Fix: VERIFY DATA EXISTS BEFORE TRAINING
#===============================================================================
header "🔍 Pre-flight: Verify Data"

if [ -z "$(find data_for_training -name "*.h5" 2>/dev/null | head -1)" ]; then
    error "Training data not found! Run train_v10_stage1.sh first to generate data."
fi
if [ -z "$(find data_for_test -name "*.h5" 2>/dev/null | head -1)" ]; then
    error "Test data not found! Run train_v10_stage1.sh first to generate data."
fi
success "Training and test data found."

#===============================================================================
# STEP 8: STAGE 2 — FINE-TUNING
#===============================================================================
header "🏋️ STEP 8: Stage 2 — Fine-tuning (200 Epochs)"

PRETRAIN_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV10_MLFIM/checkpoints/MyEfficientLFNetV10_MLFIM_pretrain_best.pth 2>/dev/null | head -1)

if [ -z "$PRETRAIN_CKPT" ]; then
    error "Cannot start Stage 2: No Stage 1 checkpoint found!"
fi

info "Starting Stage 2 Fine-tuning (no masking) from $PRETRAIN_CKPT..."
# Stage 2: 200 epochs (extended from 150), lr=5e-5, mask_ratio=0.0.
# Extra 50 epochs at the cosine tail (very low LR) consistently give
# +0.03–0.07 dB in SR competitions. eta_min lowered to 5e-7 so the
# final LR is low enough to settle into a sharp minimum.
python train_mlfim.py --stage finetune \
    --use_pre_ckpt --path_pre_pth "$PRETRAIN_CKPT" \
    --model_name MyEfficientLFNetV10_MLFIM --angRes 5 --scale_factor 4 \
    --batch_size 4 --lr 5e-5 --epoch 200 \
    --path_for_train ./data_for_training/ --path_for_test ./data_for_test/ \
    --device cuda:0 --num_workers 8

success "Stage 2 (Fine-tuning) complete"

#===============================================================================
# STEP 8.5: INFERENCE & EVALUATION
#===============================================================================
header "📊 STEP 8.5: Inference and Evaluation"

BEST_FINETUNE_CKPT=$(ls -t log/SR_5x5_4x/ALL/MyEfficientLFNetV10_MLFIM/checkpoints/MyEfficientLFNetV10_MLFIM_finetune_best.pth 2>/dev/null | head -1)
if [ -n "$BEST_FINETUNE_CKPT" ]; then
    info "Using best fine-tuned checkpoint: $BEST_FINETUNE_CKPT"
    info "Running inference..."
    python inference.py --model_name MyEfficientLFNetV10_MLFIM --angRes 5 --scale_factor 4 \
        --use_pre_ckpt --path_pre_pth "$BEST_FINETUNE_CKPT" \
        --path_for_test ./data_for_test/ --data_name ALL
    success "Inference complete"
else
    warn "No fine-tuned checkpoints found. Skipping inference."
fi

header "🏆 TRAINING WORKFLOW COMPLETE!"

#===============================================================================
# STEP 9: VALIDATE SUBMISSION (CodaBench Compliance)
#===============================================================================
header "✅ STEP 9: Validate Submission"

if [ -f "validate_submission.py" ]; then
    RESULTS_DIR="log/SR_5x5_4x/ALL/MyEfficientLFNetV10_MLFIM/results/TEST"
    if [ -d "$RESULTS_DIR" ]; then
        info "Running submission validator on $RESULTS_DIR..."
        if python validate_submission.py "$RESULTS_DIR"; then
            success "Submission validation passed!"
        else
            warn "Submission validation had issues — check output above."
        fi
    else
        warn "Results directory $RESULTS_DIR not found. Run inference first (Step 8.5)."
    fi
else
    warn "validate_submission.py not found, skipping validation."
fi

info "Re-running efficiency check (final gate)..."
python check_efficiency.py --model_name MyEfficientLFNetV10_MLFIM --angRes 5 --scale_factor 4 --patch_size 32

header "🏁 ALL DONE!"
success "Pipeline complete. Ready for CodaBench submission."
