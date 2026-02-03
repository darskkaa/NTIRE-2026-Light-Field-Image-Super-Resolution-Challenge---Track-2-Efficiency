# NTIRE Light Field SR: SOTA Summary

> **Track 2 Efficiency Constraints**: `Params < 1M` · `FLOPs < 20G` · `Input: 5×5×32×32` · `4× SR`

---

## NTIRE 2024/2025 Winners

| Track | Winner | Method | PSNR |
|-------|--------|--------|------|
| Track 1 (Fidelity) 2024 | BNU&TMU-AI-TRY | **BigEPIT** | 43.22 dB |
| Track 2 (Efficiency) 2024 | BITSMBU | **LGFN** | ~40 dB |
| Track 1 (Fidelity) 2025 | — | — | 43.58 dB (+0.36) |

---

## SOTA Methods

### BigEPIT (Track 1 Winner)
- Scaled EPIT with more channels/blocks
- Augmented data resampling for disparity variation
- **Limitation**: High computational cost, not suitable for efficiency track

### LGFN (Track 2 Winner)
- 0.45M params, 19.33G FLOPs
- Lightweight yet competitive
- **Limitation**: Lower PSNR than Track 1 methods

### DistgSSR (Baseline)
- Disentangles spatial/angular information
- Uses domain-specific convolutions
- **Limitation**: Outperformed by newer methods

### LFT (Transformer)
- Angular + Spatial Transformers
- Captures long-range dependencies
- **Limitation**: Transformer overhead, memory hungry

---

## Key Techniques (Use These)

| Technique | Why It Works |
|-----------|-------------|
| **Progressive Disentangling** | Separates spatial/angular/EPI branches |
| **RepConv Blocks** | Multi-branch train → single conv inference |
| **EPI Processing** | Captures disparity across views |
| **Channel Attention** | Weights important features |
| **PixelShuffle Upsampling** | Efficient 4× upscale |

---

## Known Limitations (Track 2)

### Architecture Constraints
- **1M param limit** → forces shallow networks
- **20G FLOPs** → limits Transformer depth
- **Disparity variation** → hard to handle with small models

### Data Challenges
- Limited training scenes (144 total)
- Imbalanced disparity distribution
- Real vs Synthetic domain gap

### Inference Issues
- Mixed precision can cause NaN on some inputs
- TTA counted toward FLOPs budget
- Patch overlap tuning affects PSNR

---

## Your Model Status

| Metric | Value | Limit | Status |
|--------|-------|-------|--------|
| Params | 547K | 1M | ✓ 55% |
| FLOPs | 19.54G | 20G | ✓ 98% |
| Expected PSNR | 35-40 dB | — | Competitive |

---

## References

1. NTIRE 2024 LF-SR Challenge Report (CVPR Workshop)
2. NTIRE 2025 LF-SR Challenge Report (CVPR Workshop)
3. BigEPIT: Scalable EPI Transformer for LFSR
4. DistgSSR: Disentangling Light Field Features
5. LGFN: Lightweight Network for Track 2
