"""
Targeted diagnostic: Test trained model with all combinations to find the validation bug.
Run: python debug_val2.py --ckpt ./log/<path_to_latest_checkpoint.pth>

If no checkpoint, test on a SINGLE TRAINING PATCH to compare train vs val metrics.
"""
import torch
import numpy as np
import importlib
import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.dirname(__file__))

from option import args as base_args
from utils.utils import LFdivide, LFintegrate, LFintegrate_gaussian, cal_metrics
from utils.utils_datasets import MultiTestSetDataLoader, TrainSetDataLoader
from torch.utils.data import DataLoader
from einops import rearrange

# Parse checkpoint path
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
cli_args, _ = parser.parse_known_args()

# Setup
device = torch.device('cuda:0')
base_args.model_name = 'MyEfficientLFNetV3_MLFIM'
base_args.scale_factor = 4
base_args.angRes_in = 5
base_args.angRes_out = 5
base_args.patch_size_for_test = 32
base_args.stride_for_test = 16
base_args.task = 'SR'

MODEL_PATH = 'model.SR.MyEfficientLFNetV3_MLFIM'
MODEL = importlib.import_module(MODEL_PATH)
net = MODEL.get_model(base_args)

# Find checkpoint
ckpt_path = cli_args.ckpt
if ckpt_path is None:
    # Auto-find latest checkpoint
    candidates = glob.glob('./log/**/MyEfficientLFNetV3_MLFIM*pretrain*best*.pth', recursive=True)
    candidates += glob.glob('./log/**/MyEfficientLFNetV3_MLFIM*pretrain*epoch*.pth', recursive=True)
    if candidates:
        ckpt_path = max(candidates, key=os.path.getmtime)
        print(f"Auto-found checkpoint: {ckpt_path}")

if ckpt_path and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(ckpt['state_dict'])
    ema_state = ckpt.get('ema_state_dict', None)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Has EMA: {ema_state is not None}")
    HAS_CKPT = True
else:
    print("WARNING: No checkpoint found. Using RANDOM weights.")
    HAS_CKPT = False
    ema_state = None

net = net.to(device)
print(f"Model params: {sum(p.numel() for p in net.parameters()):,}")

# ======================================================================
# TEST 1: Single training patch — does model produce correct output?
# ======================================================================
print("\n" + "="*60)
print("TEST 1: Training patch inference (SHOULD match training PSNR)")
print("="*60)

train_dataset = TrainSetDataLoader(base_args)
train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=1, shuffle=False)

# Get one training sample
for data, label, data_info in train_loader:
    [Lr_angRes_in, Lr_angRes_out] = data_info
    data_info[0] = Lr_angRes_in[0].item()
    data_info[1] = Lr_angRes_out[0].item()
    break

data = data.to(device)
label = label.to(device)

print(f"  Train LR shape: {data.shape}, range: [{data.min():.4f}, {data.max():.4f}]")
print(f"  Train HR shape: {label.shape}, range: [{label.min():.4f}, {label.max():.4f}]")

# Test in TRAIN mode (how training evaluates PSNR)
net.train()
with torch.no_grad():
    out_train = net(data, data_info)
    print(f"\n  [TRAIN mode] Output range: [{out_train.min():.4f}, {out_train.max():.4f}]")
    psnr_t, ssim_t = cal_metrics(base_args, label.detach(), out_train.detach().float())
    print(f"  [TRAIN mode] PSNR: {psnr_t:.2f} dB, SSIM: {ssim_t:.4f}")

    # Clamped
    out_train_c = out_train.clamp(0, 1)
    psnr_tc, ssim_tc = cal_metrics(base_args, label.detach(), out_train_c.detach().float())
    print(f"  [TRAIN mode, clamped] PSNR: {psnr_tc:.2f} dB, SSIM: {ssim_tc:.4f}")

# Test in EVAL mode
net.eval()
with torch.no_grad():
    out_eval = net(data, data_info)
    print(f"\n  [EVAL mode] Output range: [{out_eval.min():.4f}, {out_eval.max():.4f}]")
    psnr_e, ssim_e = cal_metrics(base_args, label.detach(), out_eval.detach().float())
    print(f"  [EVAL mode] PSNR: {psnr_e:.2f} dB, SSIM: {ssim_e:.4f}")

    # Clamped
    out_eval_c = out_eval.clamp(0, 1)
    psnr_ec, ssim_ec = cal_metrics(base_args, label.detach(), out_eval_c.detach().float())
    print(f"  [EVAL mode, clamped] PSNR: {psnr_ec:.2f} dB, SSIM: {ssim_ec:.4f}")

# Compare train vs eval element-wise
diff = (out_train - out_eval).abs()
print(f"\n  Train vs Eval diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
if diff.max() > 1e-3:
    print("  *** CRITICAL: Train and Eval produce DIFFERENT outputs! ***")
    print("  This means some module behaves differently in eval mode.")
else:
    print("  ✓ Train and Eval produce identical outputs (as expected)")

# ======================================================================
# TEST 2: EMA weights (if available)
# ======================================================================
if ema_state is not None:
    print("\n" + "="*60)
    print("TEST 2: EMA weights vs raw weights")
    print("="*60)
    
    # Save current weights
    backup = {k: v.clone() for k, v in net.state_dict().items()}
    
    # Load EMA weights
    ema_sd = {}
    raw_sd = net.state_dict()
    for name, param in ema_state.items():
        if name in raw_sd:
            ema_sd[name] = param
        else:
            print(f"  WARNING: EMA key {name} not in model state dict")
    
    # Check how different EMA is from raw weights
    total_diff = 0
    n_params = 0
    for name in ema_sd:
        if name in raw_sd:
            d = (raw_sd[name].float() - ema_sd[name].float()).abs().mean().item()
            total_diff += d
            n_params += 1
    avg_diff = total_diff / max(n_params, 1)
    print(f"  Avg weight diff (raw vs EMA): {avg_diff:.6f}")
    
    # Apply EMA weights
    for name, param in net.named_parameters():
        if name in ema_state:
            param.data.copy_(ema_state[name])
    
    net.eval()
    with torch.no_grad():
        out_ema = net(data, data_info)
        print(f"  [EMA + EVAL] Output range: [{out_ema.min():.4f}, {out_ema.max():.4f}]")
        psnr_ema, ssim_ema = cal_metrics(base_args, label.detach(), out_ema.detach().float())
        print(f"  [EMA + EVAL] PSNR: {psnr_ema:.2f} dB, SSIM: {ssim_ema:.4f}")
    
    # Restore raw weights
    net.load_state_dict(backup)

# ======================================================================
# TEST 3: Full validation pipeline on one image
# ======================================================================
print("\n" + "="*60)
print("TEST 3: Full validation (LFdivide -> model -> LFintegrate)")
print("="*60)

base_args.data_name = 'EPFL'
test_Names, test_Loaders, _ = MultiTestSetDataLoader(base_args)

net.eval()
for idx, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in enumerate(test_Loaders[0]):
    if idx > 0:
        break

    [Lr_angRes_in, Lr_angRes_out] = data_info
    data_info[0] = Lr_angRes_in[0].item()
    data_info[1] = Lr_angRes_out[0].item()

    print(f"\n  Image: {LF_name[0]}")
    Lr = Lr_SAI_y.squeeze().to(device)
    
    subLFin = LFdivide(Lr, base_args.angRes_in, base_args.patch_size_for_test, base_args.stride_for_test)
    numU, numV, H, W = subLFin.size()
    subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')

    # Process all patches and track output range per patch
    patch_mins, patch_maxs = [], []
    subLFout = torch.zeros(numU * numV, 1,
                           base_args.angRes_in * base_args.patch_size_for_test * base_args.scale_factor,
                           base_args.angRes_in * base_args.patch_size_for_test * base_args.scale_factor)
    
    with torch.no_grad():
        for i in range(numU * numV):
            tmp = subLFin[i:i+1].to(device)
            out = net(tmp, data_info)
            patch_mins.append(out.min().item())
            patch_maxs.append(out.max().item())
            subLFout[i:i+1] = out.cpu()
    
    print(f"  Patch output ranges: min={min(patch_mins):.4f}, max={max(patch_maxs):.4f}")
    print(f"  Patches with min < -0.1: {sum(1 for m in patch_mins if m < -0.1)}/{len(patch_mins)}")
    print(f"  Patches with max >  1.1: {sum(1 for m in patch_maxs if m > 1.1)}/{len(patch_maxs)}")
    
    subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)
    
    target_h = Hr_SAI_y.size(-2) // base_args.angRes_out
    target_w = Hr_SAI_y.size(-1) // base_args.angRes_out
    sr_pz = base_args.patch_size_for_test * base_args.scale_factor
    sr_stride = base_args.stride_for_test * base_args.scale_factor
    
    Sr_4D_y = LFintegrate_gaussian(subLFout, base_args.angRes_out, sr_pz, sr_stride, target_h, target_w)
    Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')
    
    # WITHOUT clamping
    psnr_raw, ssim_raw = cal_metrics(base_args, Hr_SAI_y, Sr_SAI_y)
    print(f"\n  [EVAL, no clamp] PSNR: {psnr_raw:.2f} dB")
    
    # WITH clamping
    Sr_SAI_y_c = Sr_SAI_y.clamp(0, 1)
    psnr_clamp, ssim_clamp = cal_metrics(base_args, Hr_SAI_y, Sr_SAI_y_c)
    print(f"  [EVAL, clamped] PSNR: {psnr_clamp:.2f} dB")
    
    # Bicubic baseline
    import torch.nn.functional as Func
    bic = Func.interpolate(Lr_SAI_y.float(), scale_factor=base_args.scale_factor, mode='bicubic', align_corners=False).clamp(0,1)
    psnr_bic, _ = cal_metrics(base_args, Hr_SAI_y, bic)
    print(f"  [Bicubic baseline] PSNR: {psnr_bic:.2f} dB")

print("\n=== DONE ===")
