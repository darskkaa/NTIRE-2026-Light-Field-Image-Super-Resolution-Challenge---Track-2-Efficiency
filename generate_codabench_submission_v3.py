"""
CodaBench Submission Generator for V3 (MyEfficientLFNetV3_MLFIM)
================================================================
Generates submission_v3.zip ready for NTIRE 2026 LF-SR Track 2 CodaBench upload.

Pipeline:
  1. Download NTIRE_Test_Real / NTIRE_Test_Synth .mat files via gdown
  2. Load V3 model checkpoint (EMA preferred, falls back to state_dict)
  3. For each .mat: RGB→YCbCr, SR on Y channel (NO TTA), bicubic-upsample CbCr
  4. Gaussian-weighted patch stitching (matches train.py validation exactly)
  5. YCbCr→RGB, save 5x5 views as View_{i}_{j}.bmp per scene
  6. Zip into submission_v3.zip  (Real/<scene>/View_*.bmp, Synth/<scene>/View_*.bmp)

AUDIT NOTES (2026-03-07):
  - LFdivide uses V3-fixed numU/numV from padded dims (matches utils.py)
  - imresize handles multi-channel CbCr correctly (per-channel resize)
  - subLFin patches are moved to device before net() call
  - LFintegrate_gaussian matches utils.py exactly
  - No TTA whatsoever — pure forward pass
  - BMP output format matches colab_submission.py and train.py test()
"""

import os
import sys
import glob
import subprocess
import shutil
import zipfile
import numpy as np
import h5py
import scipy.io as scio
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import importlib
import argparse
from pathlib import Path
from math import ceil

# ==============================================================================
# 0. Set up dependencies and paths
# ==============================================================================

TEST_REAL_LINK = "https://drive.google.com/drive/folders/1rjPxwBjdg8JeGnMHacDsVwDbc2PiFqhF?usp=drive_link"
TEST_SYNTH_LINK = "https://drive.google.com/drive/folders/1eHiYKJ72R2_6Ci6I0QQ0UW2pjblGNvVy?usp=drive_link"

def run_cmd(cmd, check=True):
    print(f"\n[EXEC] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}: {cmd}")
        sys.exit(result.returncode)

def download_test_data():
    print("\n=== STEP 1: Checking/Downloading NTIRE Test Data ===")
    os.makedirs('datasets_test', exist_ok=True)
    
    real_dir = "datasets_test/NTIRE_Test_Real"
    synth_dir = "datasets_test/NTIRE_Test_Synth"
    
    if not os.path.exists(real_dir) or len(glob.glob(f'{real_dir}/**/*.mat', recursive=True)) < 16:
        print(f"Downloading Test Real to {real_dir}...")
        os.makedirs(real_dir, exist_ok=True)
        folder_id = TEST_REAL_LINK.split('/')[-1].split('?')[0]
        run_cmd(f'gdown --folder {folder_id} -O "{real_dir}"', check=False)
        
    if not os.path.exists(synth_dir) or len(glob.glob(f'{synth_dir}/**/*.mat', recursive=True)) < 16:
        print(f"Downloading Test Synth to {synth_dir}...")
        os.makedirs(synth_dir, exist_ok=True)
        folder_id = TEST_SYNTH_LINK.split('/')[-1].split('?')[0]
        run_cmd(f'gdown --folder {folder_id} -O "{synth_dir}"', check=False)

    real_count = len(glob.glob(f'{real_dir}/**/*.mat', recursive=True))
    synth_count = len(glob.glob(f'{synth_dir}/**/*.mat', recursive=True))
    print(f"[INFO] Found: {real_count} Real .mat, {synth_count} Synth .mat")
    if real_count == 0 and synth_count == 0:
        print("\n❌ No test data found! gdown failed (likely due to Google Drive rate limits).")
        print("Please download the test data manually via your browser, upload to your VM, and run:")
        print("  python generate_codabench_submission_v3.py --real_dir /path/to/Real/inference --synth_dir /path/to/Synth/inference\n")
        sys.exit(1)

# ==============================================================================
# MATLAB-compatible bicubic resize (matches colab_submission.py / BasicLFSR)
# ==============================================================================
# NOTE: The PIL-based imresize(mode='F') only handles single-channel images.
# CbCr has 2 channels. We must resize each channel independently OR use the
# MATLAB-compatible imresize from colab_submission.py which handles multi-channel.

def _triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f

def _cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + \
        np.multiply(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (1 < absx) & (absx <= 2))
    return f

def _contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def _imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None):
    """MATLAB-compatible bicubic resize. Handles 2D and multi-channel arrays."""
    kernel = _cubic if method == 'bicubic' else _triangle
    kernel_width = 4.0
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = [int(ceil(scalar_scale * I.shape[0])), int(ceil(scalar_scale * I.shape[1]))]
    elif output_shape is not None:
        scale = [1.0 * output_shape[k] / I.shape[k] for k in range(2)]
        output_size = list(output_shape)
    else:
        raise ValueError('scalar_scale OR output_shape should be defined!')
    
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = _contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = _imresizevec(B, weights[dim], indices[dim], dim)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


# ==============================================================================
# Color conversion (identical to utils.py)
# ==============================================================================

def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0
    y = y / 255.0
    return y

def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y


# ==============================================================================
# Patch divide/integrate (matches utils.py V3-fixed version EXACTLY)
# ==============================================================================

def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]
    return Im_out


def LFdivide(data, angRes, patch_size, stride):
    """V3-FIXED: Compute numU/numV from actual padded dimensions (matches utils.py)."""
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    # V3 FIX: Compute numU/numV from the actual padded dimensions
    h_pad, w_pad = data_pad.shape[2], data_pad.shape[3]
    numU = (h_pad - patch_size) // stride + 1
    numV = (w_pad - patch_size) // stride + 1
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)
    return subLF


def LFintegrate_gaussian(subLF, angRes, pz, stride, h, w):
    """Gaussian-weighted patch stitching (matches utils.py exactly)."""
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w',
                          a1=angRes, a2=angRes)

    n1, n2, a1, a2, pH, pW = subLF.shape

    # Build 2D Gaussian weight map for one patch
    sigma = pz / 4.0
    ax = torch.arange(pz, dtype=torch.float32, device=subLF.device) - (pz - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    gauss_2d = gauss_2d / gauss_2d.max()

    # Accumulate weighted patches into output canvas
    canvas_h = (n1 - 1) * stride + pz
    canvas_w = (n2 - 1) * stride + pz

    outLF = torch.zeros(a1, a2, canvas_h, canvas_w, dtype=subLF.dtype, device=subLF.device)
    weight_map = torch.zeros(1, 1, canvas_h, canvas_w, dtype=subLF.dtype, device=subLF.device)

    for i in range(n1):
        for j in range(n2):
            top = i * stride
            left = j * stride
            outLF[:, :, top:top+pz, left:left+pz] += subLF[i, j] * gauss_2d
            weight_map[:, :, top:top+pz, left:left+pz] += gauss_2d

    weight_map = weight_map.clamp(min=1e-8)
    outLF = outLF / weight_map

    # Crop to target size, EXCLUDING the padded border
    bdr_hr = (pz - stride) // 2
    outLF = outLF[:, :, bdr_hr : bdr_hr + h, bdr_hr : bdr_hr + w]
    return outLF


# ==============================================================================
# Model Inference (NO TTA — pure forward pass)
# ==============================================================================

def process_file_direct(mat_file_path, save_dir, net, device, args):
    """
    Process a single .mat file → 5x5 View BMP files.
    Matches train.py test() logic exactly. No TTA.
    """
    import imageio
    filename = Path(mat_file_path).name
    
    # 1. Load LF data from .mat
    try:
        data = h5py.File(mat_file_path, 'r')
        LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
    except:
        data = scio.loadmat(mat_file_path)
        LF = np.array(data['LF'])

    (U, V, H, W, _) = LF.shape
    angRes = args.angRes_in
    scale_factor = args.scale_factor
    
    # Extract central angRes×angRes views
    LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, 0:H, 0:W, 0:3]
    LF = LF.astype('double')
    (U, V, H, W, _) = LF.shape

    # Pre-allocate output arrays
    Sr_SAI_cbcr = np.zeros((U * H * scale_factor, V * W * scale_factor, 2), dtype='single')
    Lr_SAI_y = np.zeros((U * H, V * W), dtype='single')

    # 2. Convert RGB→YCbCr, extract Y, bicubic-upsample CbCr
    for u in range(U):
        for v in range(V):
            tmp_Lr_rgb = LF[u, v, :, :, :]
            tmp_Lr_ycbcr = rgb2ycbcr(tmp_Lr_rgb)
            Lr_SAI_y[u * H: (u+1) * H, v * W: (v+1) * W] = tmp_Lr_ycbcr[:, :, 0]

            # Bicubic upsample CbCr channels (MATLAB-compatible, multi-channel safe)
            tmp_Lr_cbcr = tmp_Lr_ycbcr[:, :, 1:3]
            tmp_Sr_cbcr = imresize(tmp_Lr_cbcr, scalar_scale=scale_factor)
            Sr_SAI_cbcr[u * H * scale_factor: (u+1) * H * scale_factor,
                        v * W * scale_factor: (v+1) * W * scale_factor, :] = tmp_Sr_cbcr

    # 3. Prepare for model: convert to torch tensors
    Lr_SAI_y_tensor = torch.from_numpy(Lr_SAI_y).unsqueeze(0).unsqueeze(0)  # (1, 1, U*H, V*W)
    Sr_SAI_cbcr_tensor = torch.from_numpy(Sr_SAI_cbcr).permute(2, 0, 1).unsqueeze(0)  # (1, 2, U*H*S, V*W*S)
    data_info = [args.angRes_in, args.angRes_out]

    # 4. Divide LF into overlapping patches
    subLFin = LFdivide(Lr_SAI_y_tensor.squeeze(), angRes, args.patch_size_for_test, args.stride_for_test)
    numU, numV, pH, pW = subLFin.size()
    subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
    
    # Allocate output on CPU (same as train.py test())
    subLFout = torch.zeros(numU * numV, 1, angRes * args.patch_size_for_test * scale_factor,
                           angRes * args.patch_size_for_test * scale_factor)

    # 5. SR each patch (No TTA — pure forward pass, matching train.py test() exactly)
    net.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in range(0, numU * numV, args.minibatch_for_test):
            end_idx = min(i + args.minibatch_for_test, numU * numV)
            tmp = subLFin[i:end_idx, :, :, :]
            out = net(tmp.to(device), data_info)
            subLFout[i:end_idx, :, :, :] = out.cpu()  # Move back to CPU (matches train.py)

    subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

    # 6. Restore patches to full LF using Gaussian blending (matches train.py test())
    sr_pz = args.patch_size_for_test * scale_factor
    sr_stride = args.stride_for_test * scale_factor
    target_h = H * scale_factor
    target_w = W * scale_factor
    Sr_4D_y = LFintegrate_gaussian(subLFout, args.angRes_out, sr_pz, sr_stride, target_h, target_w)
    Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')
    
    # 7. Recombine Y with CbCr and convert to RGB (matches train.py test() exactly)
    Sr_SAI_ycbcr = torch.cat((Sr_SAI_y.cpu(), Sr_SAI_cbcr_tensor), dim=1)
    Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0, 1) * 255).astype('uint8')
    Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

    # 8. Save BMP per view (matches train.py test() and colab_submission.py format)
    scene_name = filename.replace('.mat', '').replace('.h5', '')
    scene_dir = os.path.join(save_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    
    for i in range(args.angRes_out):
        for j in range(args.angRes_out):
            img = Sr_4D_rgb[i, j, :, :, :]
            path = os.path.join(scene_dir, f'View_{i}_{j}.bmp')
            imageio.imwrite(path, img)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser("CodaBench Submission Generator for V3")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint. Defaults to auto-search in log dir")
    parser.add_argument("--real_dir", type=str, default=None,
                        help="Path to Real test .mat files (overrides download)")
    parser.add_argument("--synth_dir", type=str, default=None,
                        help="Path to Synth test .mat files (overrides download)")
    args = parser.parse_args()

    # Step 1: Download or locate test data
    if args.real_dir is None or args.synth_dir is None:
        download_test_data()
    
    real_dir = args.real_dir or "datasets_test/NTIRE_Test_Real"
    synth_dir = args.synth_dir or "datasets_test/NTIRE_Test_Synth"

    # Step 2: Model & Args Definition
    class Config:
        model_name = "MyEfficientLFNetV3_MLFIM"
        task = "SR"
        angRes_in = 5
        angRes_out = 5
        scale_factor = 4
        patch_size_for_test = 32
        stride_for_test = 16
        minibatch_for_test = 1
        mlfim_mask_ratio = 0.0  # CRITICAL: 0.0 at inference (no masking)
    
    config = Config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")

    print("\n=== STEP 2: Loading V3 Model ===")
    MODEL_PATH = 'model.SR.' + config.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(config).to(device)

    # Auto-find checkpoint
    if args.ckpt is None:
        search_patterns = [
            f'log/SR_5x5_4x/ALL/{config.model_name}/checkpoints/*.pth',
            f'log/SR_5x5_4x/*/{config.model_name}/checkpoints/*.pth',
            'log/**/*.pth',
        ]
        pth_files = []
        for pattern in search_patterns:
            pth_files = glob.glob(pattern, recursive=True)
            if pth_files:
                break
        
        if not pth_files:
            print("❌ No .pth checkpoint found! Please provide it explicitly with --ckpt <path>")
            sys.exit(1)
            
        pth_files.sort(key=os.path.getmtime, reverse=True)
        best_ckpt = pth_files[0]
        print(f"Auto-selected checkpoint: {best_ckpt}")
    else:
        best_ckpt = args.ckpt

    print(f"Loading checkpoint: {best_ckpt}")
    checkpoint = torch.load(best_ckpt, map_location=device)
    
    # Priority: ema_state_dict > state_dict > raw dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('ema_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    # Clean module. prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        cleaned_state_dict[new_k] = v
    
    net.load_state_dict(cleaned_state_dict, strict=False)
    net.eval()
    print(f"✅ Checkpoint loaded successfully! ({len(cleaned_state_dict)} params)")

    # Step 3: Run Inference
    print("\n=== STEP 3: Running Inference & Generating .bmp files ===")
    out_base = "submission_temp"
    shutil.rmtree(out_base, ignore_errors=True)
    os.makedirs(f"{out_base}/Real", exist_ok=True)
    os.makedirs(f"{out_base}/Synth", exist_ok=True)

    real_files = sorted(glob.glob(f"{real_dir}/**/*.mat", recursive=True))
    synth_files = sorted(glob.glob(f"{synth_dir}/**/*.mat", recursive=True))

    print(f"Found {len(real_files)} Real + {len(synth_files)} Synth .mat files")
    if len(real_files) == 0 and len(synth_files) == 0:
        print("❌ No .mat files found! Check your data paths.")
        sys.exit(1)

    print(f"\nProcessing {len(real_files)} Real scenes...")
    for f in tqdm(real_files, ncols=70):
        process_file_direct(f, f"{out_base}/Real", net, device, config)

    print(f"\nProcessing {len(synth_files)} Synth scenes...")
    for f in tqdm(synth_files, ncols=70):
        process_file_direct(f, f"{out_base}/Synth", net, device, config)

    # Step 4: Zip Submission
    print("\n=== STEP 4: Creating submission.zip ===")
    zip_path = "submission.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    total_files = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(out_base):
            for file in files:
                file_path = os.path.join(root, file)
                
                # We want the zip to contain ONLY `Real/...` and `Synth/...` at the root.
                # `out_base` is "submission_temp", so we strip it.
                rel_path = os.path.relpath(file_path, out_base)
                
                if rel_path.startswith('/') or rel_path.startswith('\\'):
                    rel_path = rel_path[1:]
                
                arcname = rel_path
                
                zipf.write(file_path, arcname)
                total_files += 1

    # Verify
    real_scenes = len(os.listdir(f"{out_base}/Real")) if os.path.exists(f"{out_base}/Real") else 0
    synth_scenes = len(os.listdir(f"{out_base}/Synth")) if os.path.exists(f"{out_base}/Synth") else 0
    print(f"\n📊 Summary:")
    print(f"   Real scenes:  {real_scenes}")
    print(f"   Synth scenes: {synth_scenes}")
    print(f"   Total BMP files in zip: {total_files}")
    print(f"   Expected: {(real_scenes + synth_scenes) * 25} ({real_scenes + synth_scenes} scenes × 25 views)")
    
    if total_files == (real_scenes + synth_scenes) * 25:
        print(f"\n✅ Submission successfully created: {zip_path}")
        print("   You can now upload this file to CodaBench.")
    else:
        print(f"\n⚠️  Warning: File count mismatch! Expected {(real_scenes + synth_scenes) * 25}, got {total_files}")

if __name__ == "__main__":
    main()
