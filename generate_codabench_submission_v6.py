"""
CodaBench Submission Generator for MyEfficientLFNetV6_Final
============================================================
Generates submission.zip for NTIRE 2026 LF-SR Track 2 CodaBench.

Pipeline:
  1. Download NTIRE_Test_Real / NTIRE_Test_Synth .mat files via gdown
  2. Load V6_Final model (import from model/SR/ or use embedded fallback)
  3. Checkpoint loading: SWA > finetune_best > pretrain_best > auto-find
  4. For each .mat: RGB→YCbCr, SR on Y (NO TTA), bicubic CbCr upsample
  5. Gaussian-weighted patch stitching (matches training validation)
  6. YCbCr→RGB, save 5×5 views as View_i_j.bmp
  7. Zip into submission.zip and validate structure

NO TTA — pure FP32 forward pass.

Usage:
  python generate_codabench_submission_v6.py
  python generate_codabench_submission_v6.py --ckpt path/to/checkpoint.pth
  python generate_codabench_submission_v6.py --real_dir /path/to/Real --synth_dir /path/to/Synth
"""

import os, sys, glob, subprocess, shutil, zipfile, argparse, re, importlib
import numpy as np
import h5py
import scipy.io as scio
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from pathlib import Path
from math import ceil
from collections import OrderedDict

# ============================================================================
# Test Data Download
# ============================================================================
TEST_REAL_LINK = "https://drive.google.com/drive/folders/1FxWmbrbH2mYQgApjOmj-2UM1Yu7fQ1Rg"
TEST_SYNTH_LINK = "https://drive.google.com/drive/folders/120fxXLA20jI7tWrZ-YGn14e4B41cIPq7"

EXPECTED_REAL_FILES = [
    "IMG_0199__Decoded.mat", "IMG_0214__Decoded.mat", "IMG_0238__Decoded.mat",
    "IMG_0256__Decoded.mat", "IMG_0262__Decoded.mat", "IMG_0268__Decoded.mat",
    "IMG_0271__Decoded.mat", "IMG_0280__Decoded.mat", "IMG_0289__Decoded.mat",
    "IMG_0308__Decoded.mat", "IMG_0329__Decoded.mat", "IMG_0363__Decoded.mat",
    "IMG_0371__Decoded.mat", "IMG_0389__Decoded.mat", "IMG_0394__Decoded.mat",
    "IMG_0404__Decoded.mat",
]
EXPECTED_SYNTH_FILES = [
    "aquarium.mat", "bookcase.mat", "camellia.mat", "cat 2.mat", "cat.mat",
    "courtyard.mat", "dining table.mat", "headboard.mat", "ivy.mat",
    "obius cube.mat", "pinetree.mat", "plants.mat", "shelf.mat",
    "ship model.mat", "stationery.mat", "washer.mat",
]


def run_cmd(cmd, check=True):
    print(f"\n[EXEC] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(result.returncode)


def verify_all_files(base_dir, expected_files, label):
    found = {os.path.basename(f) for f in glob.glob(f'{base_dir}/**/*.mat', recursive=True)}
    missing = [f for f in expected_files if f not in found]
    if missing:
        print(f"  MISSING {len(missing)} {label} files: {missing[:3]}...")
        return False
    print(f"  All {len(expected_files)} {label} .mat files verified")
    return True


def download_test_data(force=False):
    print("\n=== Checking/Downloading NTIRE Test Data ===")
    os.makedirs('datasets_test', exist_ok=True)
    real_dir = "datasets_test/NTIRE_Test_Real"
    synth_dir = "datasets_test/NTIRE_Test_Synth"

    for data_dir, link, label in [
        (real_dir, TEST_REAL_LINK, "Real"),
        (synth_dir, TEST_SYNTH_LINK, "Synth"),
    ]:
        n_existing = len(glob.glob(f'{data_dir}/**/*.mat', recursive=True))
        if not force and n_existing >= 16:
            continue
        print(f"Downloading Test {label} to {data_dir}...")
        if force and os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        folder_id = link.split('/')[-1].split('?')[0]
        run_cmd(f'gdown --folder {folder_id} -O "{data_dir}"', check=False)

    real_count = len(glob.glob(f'{real_dir}/**/*.mat', recursive=True))
    synth_count = len(glob.glob(f'{synth_dir}/**/*.mat', recursive=True))
    print(f"\nFound: {real_count} Real .mat, {synth_count} Synth .mat")

    ok1 = verify_all_files(real_dir, EXPECTED_REAL_FILES, "Real")
    ok2 = verify_all_files(synth_dir, EXPECTED_SYNTH_FILES, "Synth")
    if not (ok1 and ok2):
        print("\nSome test files MISSING! gdown may have failed.")
        print("Download manually and use --real_dir / --synth_dir flags.")
        sys.exit(1)


# ============================================================================
# MATLAB-compatible bicubic resize
# ============================================================================
def _cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + \
        np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
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
    aux = np.concatenate((np.arange(in_length),
                          np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
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
    kernel = _cubic
    kernel_width = 4.0
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = [int(ceil(scalar_scale * I.shape[0])),
                       int(ceil(scalar_scale * I.shape[1]))]
    elif output_shape is not None:
        scale = [1.0 * output_shape[k] / I.shape[k] for k in range(2)]
        output_size = list(output_shape)
    else:
        raise ValueError('scalar_scale OR output_shape required')
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights, indices = [], []
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


# ============================================================================
# Color conversion
# ============================================================================
def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481*x[:,:,0] + 128.553*x[:,:,1] +  24.966*x[:,:,2] + 16.0
    y[:,:,1] = -37.797*x[:,:,0] -  74.203*x[:,:,1] + 112.000*x[:,:,2] + 128.0
    y[:,:,2] = 112.000*x[:,:,0] -  93.786*x[:,:,1] -  18.214*x[:,:,2] + 128.0
    return y / 255.0


def ycbcr2rgb(x):
    mat = np.array([[ 65.481, 128.553,  24.966],
                    [-37.797, -74.203, 112.0  ],
                    [112.0,   -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] = mat_inv[0,0]*x[:,:,0] + mat_inv[0,1]*x[:,:,1] + mat_inv[0,2]*x[:,:,2] - offset[0]
    y[:,:,1] = mat_inv[1,0]*x[:,:,0] + mat_inv[1,1]*x[:,:,1] + mat_inv[1,2]*x[:,:,2] - offset[1]
    y[:,:,2] = mat_inv[2,0]*x[:,:,0] + mat_inv[2,1]*x[:,:,1] + mat_inv[2,2]*x[:,:,2] - offset[2]
    return y


# ============================================================================
# Patch divide/integrate (matches training validation exactly)
# ============================================================================
def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])
    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    return Im_Ext[:, :, h - bdr[0]: 2*h + bdr[1], w - bdr[2]: 2*w + bdr[3]]


def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()
    bdr = (patch_size - stride) // 2
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    h_pad, w_pad = data_pad.shape[2], data_pad.shape[3]
    numU = (h_pad - patch_size) // stride + 1
    numV = (w_pad - patch_size) // stride + 1
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)
    return subLF


def LFintegrate_gaussian(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w',
                          a1=angRes, a2=angRes)
    n1, n2, a1, a2, pH, pW = subLF.shape
    sigma = pz / 3.0
    ax = torch.arange(pz, dtype=torch.float32, device=subLF.device) - (pz - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    gauss_2d = gauss_2d / gauss_2d.max()

    canvas_h = (n1 - 1) * stride + pz
    canvas_w = (n2 - 1) * stride + pz
    outLF = torch.zeros(a1, a2, canvas_h, canvas_w, dtype=subLF.dtype, device=subLF.device)
    weight_map = torch.zeros(1, 1, canvas_h, canvas_w, dtype=subLF.dtype, device=subLF.device)

    for i in range(n1):
        for j in range(n2):
            top, left = i * stride, j * stride
            outLF[:, :, top:top+pz, left:left+pz] += subLF[i, j] * gauss_2d
            weight_map[:, :, top:top+pz, left:left+pz] += gauss_2d

    weight_map = weight_map.clamp(min=1e-8)
    outLF = outLF / weight_map
    bdr_hr = (pz - stride) // 2
    outLF = outLF[:, :, bdr_hr:bdr_hr+h, bdr_hr:bdr_hr+w]
    return outLF


# ============================================================================
# Model loading
# ============================================================================
def load_v6_model(args, device):
    """Load MyEfficientLFNetV6_Final model."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        model_module = importlib.import_module('model.SR.MyEfficientLFNetV6_Final')
        print("Loaded model from model/SR/MyEfficientLFNetV6_Final.py")
    except ImportError as e:
        print(f"Cannot import model: {e}")
        sys.exit(1)
    net = model_module.get_model(args).to(device)
    return net


def find_best_checkpoint(model_name):
    """Auto-find the best checkpoint with priority: SWA > finetune_best > pretrain_best."""
    search_order = [
        f'log/SR_5x5_4x/ALL/{model_name}/checkpoints/{model_name}_finetune_swa.pth',
        f'log/SR_5x5_4x/ALL/{model_name}/checkpoints/{model_name}_finetune_best.pth',
        f'log/SR_5x5_4x/ALL/{model_name}/checkpoints/{model_name}_pretrain_best.pth',
        f'log/SR_5x5_4x/*/{model_name}/checkpoints/*best*.pth',
        f'log/SR_5x5_4x/*/{model_name}/checkpoints/*swa*.pth',
        'log/**/*.pth',
    ]
    for pattern in search_order:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return None


def load_checkpoint(net, ckpt_path, device):
    """Load checkpoint into model with proper key cleaning."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('state_dict', ckpt)
    else:
        state_dict = ckpt
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k.replace('module.', '')] = v
    result = net.load_state_dict(cleaned, strict=False)
    print(f"  Loaded {len(cleaned)} params")
    if result.missing_keys:
        print(f"  Missing: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"  Unexpected: {result.unexpected_keys[:5]}")
    net.float()
    net.eval()
    return net


# ============================================================================
# Inference (NO TTA, FP32)
# ============================================================================
def run_sr_inference(Lr_SAI_y_tensor, net, device, args):
    """SR inference on Y channel with Gaussian-weighted patch stitching."""
    angRes = args.angRes_in
    scale_factor = args.scale_factor

    net.float()
    net.eval()

    subLFin = LFdivide(Lr_SAI_y_tensor.squeeze(), angRes,
                       args.patch_size_for_test, args.stride_for_test)
    numU, numV, pH, pW = subLFin.size()
    subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')

    subLFout = torch.zeros(numU * numV, 1,
                           angRes * args.patch_size_for_test * scale_factor,
                           angRes * args.patch_size_for_test * scale_factor)

    data_info = [args.angRes_in, args.angRes_out]
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i in range(0, numU * numV, args.minibatch_for_test):
            end_idx = min(i + args.minibatch_for_test, numU * numV)
            tmp = subLFin[i:end_idx].float().to(device)
            out = net(tmp, data_info)
            subLFout[i:end_idx] = out.float().cpu()

    subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w',
                         n1=numU, n2=numV)

    sr_pz = args.patch_size_for_test * scale_factor
    sr_stride = args.stride_for_test * scale_factor
    total_h = Lr_SAI_y_tensor.squeeze().shape[-2] // angRes
    total_w = Lr_SAI_y_tensor.squeeze().shape[-1] // angRes
    target_h = total_h * scale_factor
    target_w = total_w * scale_factor

    Sr_4D_y = LFintegrate_gaussian(subLFout, args.angRes_out,
                                   sr_pz, sr_stride, target_h, target_w)
    return Sr_4D_y


def process_file(mat_path, save_dir, net, device, args):
    """Process a single .mat file → 5×5 View BMP files."""
    import imageio
    filename = Path(mat_path).name

    # Load LF data
    try:
        data = h5py.File(mat_path, 'r')
        LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
    except:
        data = scio.loadmat(mat_path)
        LF = np.array(data['LF'])

    (U, V, H, W, _) = LF.shape
    angRes = args.angRes_in
    scale_factor = args.scale_factor

    # Central views
    LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, 0:H, 0:W, 0:3]
    LF = LF.astype('double')
    (U, V, H, W, _) = LF.shape

    Sr_SAI_cbcr = np.zeros((U*H*scale_factor, V*W*scale_factor, 2), dtype='single')
    Lr_SAI_y = np.zeros((U*H, V*W), dtype='single')

    for u in range(U):
        for v in range(V):
            tmp_rgb = LF[u, v, :, :, :]
            tmp_ycbcr = rgb2ycbcr(tmp_rgb)
            Lr_SAI_y[u*H:(u+1)*H, v*W:(v+1)*W] = tmp_ycbcr[:, :, 0]
            tmp_cbcr = tmp_ycbcr[:, :, 1:3]
            tmp_Sr_cbcr = imresize(tmp_cbcr, scalar_scale=scale_factor)
            Sr_SAI_cbcr[u*H*scale_factor:(u+1)*H*scale_factor,
                        v*W*scale_factor:(v+1)*W*scale_factor, :] = tmp_Sr_cbcr

    Lr_SAI_y_tensor = torch.from_numpy(Lr_SAI_y).unsqueeze(0).unsqueeze(0)
    Sr_SAI_cbcr_tensor = torch.from_numpy(Sr_SAI_cbcr).permute(2, 0, 1).unsqueeze(0)

    Sr_4D_y = run_sr_inference(Lr_SAI_y_tensor, net, device, args)
    Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

    Sr_SAI_ycbcr = torch.cat((Sr_SAI_y.cpu(), Sr_SAI_cbcr_tensor), dim=1)
    Sr_SAI_rgb = np.round(
        ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0, 1) * 255.0
    ).astype('uint8')
    Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c',
                          a1=args.angRes_out, a2=args.angRes_out)

    scene_name = filename.replace('.mat', '').replace('.h5', '')
    scene_dir = os.path.join(save_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    for i in range(args.angRes_out):
        for j in range(args.angRes_out):
            img = Sr_4D_rgb[i, j, :, :, :]
            path = os.path.join(scene_dir, f'View_{i}_{j}.bmp')
            imageio.imwrite(path, img)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser("CodaBench Submission Generator — V6 Final")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--real_dir", type=str, default=None)
    parser.add_argument("--synth_dir", type=str, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--model_name", type=str, default="MyEfficientLFNetV6_Final")
    args, _ = parser.parse_known_args()

    # Step 1: Download test data
    if args.real_dir is None or args.synth_dir is None:
        download_test_data(force=args.force_download)
    real_dir = args.real_dir or "datasets_test/NTIRE_Test_Real"
    synth_dir = args.synth_dir or "datasets_test/NTIRE_Test_Synth"

    # Step 2: Set up model config
    class Config:
        model_name = args.model_name
        angRes_in = 5
        angRes_out = 5
        scale_factor = 4
        patch_size_for_test = 48
        stride_for_test = 4
        minibatch_for_test = 16
        mlfim_mask_ratio = 0.0  # No masking at inference

    config = Config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Step 3: Load model
    print("\n=== Loading Model ===")
    net = load_v6_model(config, device)
    params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {params:,}")

    # Step 4: Load checkpoint
    if args.ckpt is None:
        ckpt_path = find_best_checkpoint(args.model_name)
        if ckpt_path is None:
            print("No checkpoint found! Use --ckpt to specify.")
            sys.exit(1)
        print(f"Auto-selected: {ckpt_path}")
    else:
        ckpt_path = args.ckpt

    net = load_checkpoint(net, ckpt_path, device)

    # Step 5: Inference
    print("\n=== Running Inference ===")
    print(f"Config: patch={config.patch_size_for_test}, stride={config.stride_for_test}")
    out_base = "submission_temp"
    shutil.rmtree(out_base, ignore_errors=True)
    os.makedirs(f"{out_base}/Real", exist_ok=True)
    os.makedirs(f"{out_base}/Synth", exist_ok=True)

    real_files = sorted(glob.glob(f"{real_dir}/**/*.mat", recursive=True))
    synth_files = sorted(glob.glob(f"{synth_dir}/**/*.mat", recursive=True))
    print(f"Found {len(real_files)} Real + {len(synth_files)} Synth .mat files")

    if len(real_files) == 0 and len(synth_files) == 0:
        print("No .mat files found!")
        sys.exit(1)

    failed = []
    for label, files, out_dir in [
        ("Real", real_files, f"{out_base}/Real"),
        ("Synth", synth_files, f"{out_base}/Synth"),
    ]:
        print(f"\nProcessing {len(files)} {label} scenes...")
        for f in tqdm(files, ncols=70):
            try:
                process_file(f, out_dir, net, device, config)
                torch.cuda.empty_cache()
            except Exception as e:
                scene = Path(f).stem
                print(f"\n  FAILED {label}/{scene}: {e}")
                import traceback; traceback.print_exc()
                failed.append(f"{label}/{scene}")

    if failed:
        print(f"\nWARNING: {len(failed)} scenes FAILED: {failed}")

    # Step 6: Create zip
    print("\n=== Creating submission.zip ===")
    zip_path = "submission.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    total_files = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        registered_dirs = set()
        for root, dirs, files in os.walk(out_base):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, out_base)
                arcname = rel_path.replace('\\', '/')
                if arcname.startswith('/'):
                    arcname = arcname[1:]
                # Add directory entries
                parts = arcname.split('/')
                for i in range(1, len(parts)):
                    dir_path = '/'.join(parts[:i]) + '/'
                    if dir_path not in registered_dirs:
                        zipf.writestr(zipfile.ZipInfo(dir_path), '')
                        registered_dirs.add(dir_path)
                zipf.write(file_path, arcname)
                total_files += 1

    print(f"Packaged {total_files} files into {zip_path}")

    # Step 7: Validate
    print("\n=== Validating submission.zip ===")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        entries = zf.namelist()
        has_real = any(f.startswith('Real/') for f in entries)
        has_synth = any(f.startswith('Synth/') for f in entries)
        bmp_count = sum(1 for f in entries if f.endswith('.bmp'))
        real_scenes = {f.split('/')[1] for f in entries
                       if f.startswith('Real/') and f.endswith('.bmp')}
        synth_scenes = {f.split('/')[1] for f in entries
                        if f.startswith('Synth/') and f.endswith('.bmp')}

        print(f"  Real/: {has_real} ({len(real_scenes)} scenes)")
        print(f"  Synth/: {has_synth} ({len(synth_scenes)} scenes)")
        print(f"  BMPs: {bmp_count} (expected: {(len(real_scenes)+len(synth_scenes))*25})")

        # View check
        expected_views = {f"View_{i}_{j}.bmp" for i in range(5) for j in range(5)}
        errors = []
        if not has_real: errors.append("Missing Real/")
        if not has_synth: errors.append("Missing Synth/")
        for st, scenes in [("Real", real_scenes), ("Synth", synth_scenes)]:
            for scene in scenes:
                scene_files = {f.split('/')[-1] for f in entries
                               if f.startswith(f"{st}/{scene}/") and f.endswith('.bmp')}
                missing = expected_views - scene_files
                if missing:
                    errors.append(f"{st}/{scene} missing: {missing}")

        if errors:
            print(f"\n  VALIDATION FAILED:")
            for e in errors: print(f"    {e}")
        else:
            print(f"\n  VALIDATION PASSED — submission.zip ready for CodaBench!")


if __name__ == "__main__":
    main()
