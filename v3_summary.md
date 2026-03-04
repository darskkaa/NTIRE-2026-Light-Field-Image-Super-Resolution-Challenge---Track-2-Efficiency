# MyEfficientLFNet V3 & MLFIM Pipeline Summary

This document summarizes the V3 updates for the NTIRE 2026 Light Field Image Super-Resolution Challenge (Track 2: Efficiency).

## 1. Architectural Updates (MyEfficientLFNetV3_MLFIM.py)
The V3 model heavily refines the V2 MLFIM baseline to squeeze maximum PSNR out of the strictly constrained 20G FLOPs / 1M parameters budget.

### Efficiency Budget
- **Parameters:** 799,676 (0.800M / 1.0M)
- **FLOPs (5x5x32x32):** 19.930G / 20.0G (99.6% utilized)
- **Core Config:** Channels=48, SA Groups=4, EPI Groups=3, VSS Depth=2.

### Key Refinements
- **BMDMambaLayer (Batched Multi-Directional Mamba):** 4-way channel splitting (H, W, H-rev, W-rev) for true 4-directional 2D scanning without increasing FLOPs vs single-scan. V3 fix clarifies square-only assumptions for reverse index flipping.
- **Adaptive Stream Gating (ASG):** Learns content-aware fusion of IFE, SpaAng, and EPI streams, replacing standard concat+conv.
- **Local Contrast Enhancement (LCE):** Lightweight high-pass filter + 1x1 conv right before reconstruction to counter the low-pass bias inherent in consecutive Mamba processing.
- **MicroCAB:** Depthwise-separable channel attention replaces massive 3x3 dense channel attention blocks to save 28% of EPI FLOPs, re-allocated to deeper SA layers.

## 2. Training Pipeline Updates (train_mlfim_v3.py)
The training scripts adopt SOTA recipes from NTIRE 2025 winners (LFTransMamba, LFMamba).

- **Loss Function:** Pure Charbonnier loss (`eps=1e-9`). (V3 code removes fragile `hasattr` dispatch logic).
- **Optimizer:** AdamW. LR=3e-4 (Finetune), weight_decay=5e-5.
- **Momentum:** β2=0.99 for finetuning (allows faster 2nd-moment adaptation).
- **EMA:** Exponential Moving Average of weights (decay=0.9995 -> 0.99995 at 75% epochs). V3 fixes a shallow-copy bug in `state_dict()` via deep tensor cloning.
- **Gradient Accumulation:** Accum=2 (Effective Batch=8) to smooth 4x SR gradient variance while keeping GPU VRAM useable.

## 3. Workflow Shell Scripts
- **Stage 1 (train_v3_stage1.sh):** Masked LF Image Modeling (MLFIM) pre-training for 90 epochs.
- **Stage 2 (train_v3_stage2.sh):** Fine-tuning for 200 epochs to push extreme PSNR convergence in the deep cosine scheduler tail.

## 4. Notable V3 Bug Fixes
- **Model:** Square-only assumption documented for BMDMambaLayer 1D-flattening reverse paths.
- **Train Script:** Replaced fragile `hasattr` criterion dispatch with standard passing of `data_info`.
- **Train Script:** EMA deep-copy `state_dict` implemented.
- **Data Pipeline:** Clarified that `CutBlur` augmentation acts probabilistically across the mosaic, behaving as additional stochastic regularization.
