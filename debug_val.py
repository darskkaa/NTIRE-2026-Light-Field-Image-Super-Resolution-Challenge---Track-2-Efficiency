"""
Quick diagnostic to find the validation PSNR bug.
Run on VM: python debug_val.py

Tests one validation image and prints shapes/values at every step.
"""
import torch
import numpy as np
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from option import args
from utils.utils import LFdivide, LFintegrate, LFintegrate_gaussian, cal_metrics
from utils.utils_datasets import MultiTestSetDataLoader
from einops import rearrange

# Load model
args.model_name = 'MyEfficientLFNetV3_MLFIM'
args.scale_factor = 4
args.angRes_in = 5
args.angRes_out = 5
args.patch_size_for_test = 32
args.stride_for_test = 16

device = torch.device('cuda:0')

MODEL_PATH = 'model.SR.MyEfficientLFNetV3_MLFIM'
MODEL = importlib.import_module(MODEL_PATH)
net = MODEL.get_model(args)
net = net.to(device)
net.eval()

print(f"Model params: {sum(p.numel() for p in net.parameters()):,}")
print(f"Model on device: {next(net.parameters()).device}")

# Load one test image
args.data_name = 'EPFL'
test_Names, test_Loaders, _ = MultiTestSetDataLoader(args)
test_loader = test_Loaders[0]

for idx, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in enumerate(test_loader):
    if idx > 0:
        break

    [Lr_angRes_in, Lr_angRes_out] = data_info
    data_info[0] = Lr_angRes_in[0].item()
    data_info[1] = Lr_angRes_out[0].item()

    print(f"\n=== Image: {LF_name[0]} ===")
    print(f"Lr_SAI_y shape: {Lr_SAI_y.shape}, dtype: {Lr_SAI_y.dtype}")
    print(f"Lr_SAI_y range: [{Lr_SAI_y.min():.4f}, {Lr_SAI_y.max():.4f}]")
    print(f"Hr_SAI_y shape: {Hr_SAI_y.shape}, dtype: {Hr_SAI_y.dtype}")
    print(f"Hr_SAI_y range: [{Hr_SAI_y.min():.4f}, {Hr_SAI_y.max():.4f}]")

    # Step 1: Squeeze
    Lr_squeezed = Lr_SAI_y.squeeze().to(device)
    print(f"\nStep 1 - After squeeze: {Lr_squeezed.shape}")

    # Step 2: LFdivide
    subLFin = LFdivide(Lr_squeezed, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
    numU, numV, H, W = subLFin.size()
    print(f"Step 2 - LFdivide: subLFin={subLFin.shape}, numU={numU}, numV={numV}")
    print(f"  subLFin range: [{subLFin.min():.4f}, {subLFin.max():.4f}]")

    # Step 3: Rearrange for model input
    subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
    print(f"Step 3 - Model input: {subLFin.shape}")

    # Step 4: Run model on first patch
    with torch.no_grad():
        first_patch = subLFin[0:1].to(device)
        print(f"\nStep 4 - First patch input: {first_patch.shape}")
        print(f"  Input range: [{first_patch.min():.4f}, {first_patch.max():.4f}]")
        out_patch = net(first_patch, data_info)
        print(f"  Output shape: {out_patch.shape}")
        print(f"  Output range: [{out_patch.min():.4f}, {out_patch.max():.4f}]")
        print(f"  Output mean: {out_patch.mean():.4f}")
        print(f"  Output std: {out_patch.std():.4f}")

    # Step 5: Run model on ALL patches
    subLFout = torch.zeros(numU * numV, 1,
                           args.angRes_in * args.patch_size_for_test * args.scale_factor,
                           args.angRes_in * args.patch_size_for_test * args.scale_factor)
    print(f"\nStep 5 - subLFout allocation: {subLFout.shape}")

    with torch.no_grad():
        for i in range(0, numU * numV, args.minibatch_for_test):
            tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV)]
            out = net(tmp.to(device), data_info)
            subLFout[i:min(i + args.minibatch_for_test, numU * numV)] = out.cpu()

    print(f"  subLFout range: [{subLFout.min():.4f}, {subLFout.max():.4f}]")
    subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

    # Step 6: LFintegrate
    target_h = Hr_SAI_y.size(-2) // args.angRes_out
    target_w = Hr_SAI_y.size(-1) // args.angRes_out
    sr_pz = args.patch_size_for_test * args.scale_factor
    sr_stride = args.stride_for_test * args.scale_factor
    print(f"\nStep 6 - LFintegrate: target_h={target_h}, target_w={target_w}, sr_pz={sr_pz}, sr_stride={sr_stride}")

    Sr_4D_y = LFintegrate_gaussian(subLFout, args.angRes_out, sr_pz, sr_stride, target_h, target_w)
    print(f"  Sr_4D_y shape: {Sr_4D_y.shape}")
    print(f"  Sr_4D_y range: [{Sr_4D_y.min():.4f}, {Sr_4D_y.max():.4f}]")

    # Step 7: Reshape for metrics
    Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')
    print(f"\nStep 7 - Sr_SAI_y shape: {Sr_SAI_y.shape}")
    print(f"  Sr_SAI_y range: [{Sr_SAI_y.min():.4f}, {Sr_SAI_y.max():.4f}]")
    print(f"  Hr_SAI_y shape: {Hr_SAI_y.shape}")
    print(f"  Hr_SAI_y range: [{Hr_SAI_y.min():.4f}, {Hr_SAI_y.max():.4f}]")

    # Step 8: Compute PSNR
    psnr, ssim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
    print(f"\n=== RESULT ===")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

    # Step 9: Also compute PSNR WITH clamping
    Sr_SAI_y_clamped = Sr_SAI_y.clamp(0, 1)
    psnr_clamped, ssim_clamped = cal_metrics(args, Hr_SAI_y, Sr_SAI_y_clamped)
    print(f"\nPSNR (clamped): {psnr_clamped:.2f} dB")
    print(f"SSIM (clamped): {ssim_clamped:.4f}")

    # Step 10: Compute bicubic baseline PSNR
    import torch.nn.functional as F
    Lr_up = F.interpolate(Lr_SAI_y.float(), scale_factor=args.scale_factor, mode='bicubic', align_corners=False)
    Lr_up = Lr_up.clamp(0, 1)
    psnr_bicubic, _ = cal_metrics(args, Hr_SAI_y, Lr_up)
    print(f"PSNR (bicubic): {psnr_bicubic:.2f} dB")

    # Step 11: Check per-view error
    print(f"\n=== Per-View Analysis (center view [2,2]) ===")
    Sr_6D = rearrange(Sr_SAI_y, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=5, a2=5)
    Hr_6D = rearrange(Hr_SAI_y, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=5, a2=5)
    center_sr = Sr_6D[0, 0, 2, :, 2, :].numpy()
    center_hr = Hr_6D[0, 0, 2, :, 2, :].numpy()
    print(f"  Center SR range: [{center_sr.min():.4f}, {center_sr.max():.4f}], mean={center_sr.mean():.4f}")
    print(f"  Center HR range: [{center_hr.min():.4f}, {center_hr.max():.4f}], mean={center_hr.mean():.4f}")
    mse = np.mean((center_sr - center_hr) ** 2)
    print(f"  MSE: {mse:.6f}")
    if mse > 0:
        print(f"  PSNR: {10 * np.log10(1.0 / mse):.2f} dB")

    print("\n=== DONE ===")
    break
