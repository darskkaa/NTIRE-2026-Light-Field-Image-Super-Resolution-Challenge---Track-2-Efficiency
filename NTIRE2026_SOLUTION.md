# NTIRE 2026 Light Field SR - Track 2 (Efficiency) Solution

This is a complete solution for the **NTIRE 2026 Light Field Image Super-Resolution Challenge - Track 2: Efficiency**.

## 🎯 Efficiency Metrics (Verified)

| Metric | Limit | Achieved | Status |
|--------|-------|----------|--------|
| **Parameters** | < 1,000,000 | **547,540** | ✅ PASS |
| **FLOPs** | < 20G | **19.60G** | ✅ PASS |

## 🔬 v2.0 SOTA-Inspired Architecture

**MyEfficientLFNet v2.0** is a novel architecture combining cutting-edge techniques:

| Technique | Inspiration | What It Does |
|-----------|-------------|--------------|
| **Progressive Disentangling** | CVPR 2024 | Channel-wise domain-specific processing |
| **Lightweight Angular Attention** | LFT/Transformer | Efficient cross-view interaction |
| **RepConv Blocks** | RepVGG/DBB | Multi-branch training → single-branch inference |
| **Multi-scale EPI** | BigEPIT | H/V/Diagonal EPI for varying disparities |
| **SA Modulator** | L²FMamba | Spatial-Angular combined attention |

## 📁 Custom Files Added

```
BasicLFSR/
├── model/SR/
│   └── MyEfficientLFNet.py      # v2.0 SOTA-inspired model
├── auto_setup.sh                 # ONE-COMMAND setup
├── check_efficiency.py           # Verify params & FLOPs
├── requirements.txt              # Python dependencies
│
├── setup_environment.sh          # Linux/VM setup
├── prepare_data.sh               # Generate training patches
├── train.sh                      # Training script
├── inference.sh                  # Run inference
├── create_submission.sh          # Create CodaBench ZIP
│
├── setup_environment.bat         # Windows setup
├── train.bat                     # Windows training
└── NTIRE2026_SOLUTION.md         # This file
```

## 🚀 Quick Start (Linux VM with GPU)

### ONE-COMMAND SETUP
```bash
git clone https://github.com/darskkaa/NTIRE-2026-Light-Field-Image-Super-Resolution-Challenge---Track-2-Efficiency.git
cd NTIRE-2026-Light-Field-Image-Super-Resolution-Challenge---Track-2-Efficiency
chmod +x *.sh
./auto_setup.sh
```

This single script will:
1. ✅ Install system dependencies
2. ✅ Create Python virtual environment
3. ✅ Install PyTorch with CUDA
4. ✅ Attempt to download datasets automatically
5. ✅ Generate training patches
6. ✅ Verify model efficiency

> **Note**: If automatic dataset download fails (OneDrive requires auth), the script will provide manual download instructions. After downloading, run `./download_complete.sh`

### After Setup - Train!
```bash
source venv_lfsr/bin/activate
./train.sh
```

### Dataset Download (if needed manually)
```bash
source venv_lfsr/bin/activate
./prepare_data.sh
```

### 4. Train
```bash
./train.sh
```

Training: ~12-24 hours on RTX 3060/3070, ~8-12 hours on RTX 3090/A100.

### 5. Inference & Submit
```bash
./inference.sh
./create_submission.sh
```

Upload the generated ZIP to [CodaBench Track 2](https://www.codabench.org/competitions/12927/).

---

## 🏗️ Architecture Details

```
Input [B, 1, 5×H, 5×W]
    │
    ▼
┌─────────────────────────────────────────┐
│ RepConv Shallow Feature Extraction      │
│ (dilated 3×3, d=5) → 54 channels       │
└─────────────────────────────────────────┘
    │
    ▼ × 5 Stages
┌─────────────────────────────────────────┐
│ Progressive Disentangling Stage         │
│ ┌───────────────────────────────────┐   │
│ │ Channel Split [18, 18, 18]        │   │
│ │ ├─ Spatial Branch (RepConv)       │   │
│ │ ├─ Angular Branch (LightAttn)     │   │
│ │ └─ EPI Branch (H/V/Diag)          │   │
│ │ Learned Gates + SA Modulator      │   │
│ └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Global Fusion + Residual               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ PixelShuffle 2× → 2× = 4× Upsampling   │
└─────────────────────────────────────────┘
    │
    ▼
Output [B, 1, 5×H×4, 5×W×4] + Bicubic Skip
```

---

## 📋 Manual Commands

### Verify Efficiency
```bash
python check_efficiency.py --model_name MyEfficientLFNet
```

### Generate Training Data
```bash
python Generate_Data_for_Training.py \
    --angRes 5 \
    --scale_factor 4 \
    --src_data_path ./datasets/ \
    --save_data_path ./
```

### Train
```bash
python train.py \
    --model_name MyEfficientLFNet \
    --angRes 5 \
    --scale_factor 4 \
    --batch_size 4 \
    --lr 0.0002 \
    --epoch 51 \
    --device cuda:0
```

### Resume Training
```bash
python train.py \
    --model_name MyEfficientLFNet \
    --angRes 5 \
    --scale_factor 4 \
    --batch_size 4 \
    --use_pre_ckpt True \
    --path_pre_pth ./log/SR_5x5_4x/ALL/MyEfficientLFNet/checkpoints/MyEfficientLFNet_5x5_4x_epoch_XX_model.pth
```

### Inference
```bash
python inference.py \
    --model_name MyEfficientLFNet \
    --angRes 5 \
    --scale_factor 4 \
    --data_name NTIRE_Val_Real \
    --path_pre_pth ./log/SR_5x5_4x/ALL/MyEfficientLFNet/checkpoints/MyEfficientLFNet_5x5_4x_epoch_51_model.pth
```

---

## 📊 Expected Results

Target validation PSNR: **≥29.5 dB** average to be competitive.

Based on similar architectures (LFT, EPIT):
- Expected Lytro (Real): ~29.5-30.0 dB
- Expected Synthetic: ~29.8-30.2 dB
- Expected Average: ~29.7-30.1 dB

---

## 📝 Submission Checklist

- [ ] Train model for 51 epochs
- [ ] Run inference on NTIRE_Val_Real and NTIRE_Val_Synth
- [ ] Upload to CodaBench
- [ ] Prepare fact sheet
- [ ] Email code + fact sheet to ntire.lfsr@outlook.com

---

## 🔗 Links

- **CodaBench Track 2**: https://www.codabench.org/competitions/12927/
- **NTIRE 2026 Page**: https://cvlai.net/ntire/2026
- **BasicLFSR Toolbox**: https://github.com/ZhengyuLiang24/BasicLFSR
- **Dataset (OneDrive)**: [Download Link](https://stuxidianeducn-my.sharepoint.com/personal/zyliang_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzyliang%5Fstu%5Fxidian%5Fedu%5Fcn%2FDocuments%2Fdatasets)

---

## 📅 Important Dates

- Train + Val data released: **Jan 20, 2026**
- Validation server: **Online now**
- Test data release: **March 10, 2026**
- Submission deadline: **March 17, 2026**

Good luck! 🚀
