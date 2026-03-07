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

# ==============================================================================
# 0. Set up dependencies and paths
# ==============================================================================

# User provided links for Test Data
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
    
    # We use gdown to download folders
    if not os.path.exists(real_dir) or len(glob.glob(f'{real_dir}/inference/*.mat')) < 16:
        print(f"Downloading Test Real to {real_dir}...")
        os.makedirs(real_dir, exist_ok=True)
        # Extract folder ID from link
        folder_id = TEST_REAL_LINK.split('/')[-1].split('?')[0]
        run_cmd(f'gdown --folder {folder_id} -O "{real_dir}"', check=False)
        
    if not os.path.exists(synth_dir) or len(glob.glob(f'{synth_dir}/inference/*.mat')) < 16:
        print(f"Downloading Test Synth to {synth_dir}...")
        os.makedirs(synth_dir, exist_ok=True)
        # Extract folder ID from link
        folder_id = TEST_SYNTH_LINK.split('/')[-1].split('?')[0]
        run_cmd(f'gdown --folder {folder_id} -O "{synth_dir}"', check=False)

    real_count = len(glob.glob(f'{real_dir}/inference/*.mat'))
    synth_count = len(glob.glob(f'{synth_dir}/inference/*.mat'))
    print(f"[INFO] Found: {real_count} Real .mat, {synth_count} Synth .mat")

# ==============================================================================
# Helper functions for Model Inference
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

def imresize(img, scalar_scale):
    from PIL import Image
    h, w = img.shape[:2]
    new_h, new_w = int(h * scalar_scale), int(w * scalar_scale)
    pil_img = Image.fromarray(img.astype(np.float32), mode='F')
    resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
    return np.array(resized)

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
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subLF

def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF


# ==============================================================================
# Model Inference 
# ==============================================================================

def process_file_direct(mat_file_path, save_dir, net, device, args):
    """
    Directly processes a .mat file to an output .mat file without saving intermediate h5s.
    No TTA (Test Time Augmentation), just standard forward pass.
    """
    filename = Path(mat_file_path).name
    
    # 1. Load Data
    try:
        data = h5py.File(mat_file_path, 'r')
        LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
    except:
        data = scio.loadmat(mat_file_path)
        LF = np.array(data['LF'])

    (U, V, H, W, _) = LF.shape
    angRes = args.angRes_in
    scale_factor = args.scale_factor
    
    # Extract central angRes * angRes views
    LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, 0:H, 0:W, 0:3]
    LF = LF.astype('double')
    (U, V, H, W, _) = LF.shape

    Sr_SAI_cbcr = np.zeros((U * H * scale_factor, V * W * scale_factor, 2), dtype='single')
    Lr_SAI_y = np.zeros((U * H, V * W), dtype='single')

    # Convert RGB to YCbCr, Extract Y, interpolate CbCr
    for u in range(U):
        for v in range(V):
            tmp_Lr_rgb = LF[u, v, :, :, :]
            tmp_Lr_ycbcr = rgb2ycbcr(tmp_Lr_rgb)
            Lr_SAI_y[u * H: (u+1) * H, v * W: (v+1)* W] = tmp_Lr_ycbcr[:, :, 0]

            tmp_Lr_cbcr = tmp_Lr_ycbcr[:,:,1:3]
            tmp_Sr_cbcr = imresize(tmp_Lr_cbcr, scalar_scale=scale_factor)
            Sr_SAI_cbcr[u * H * scale_factor: (u+1) * H * scale_factor,
                        v * W * scale_factor: (v+1) * W * scale_factor, :] = tmp_Sr_cbcr

    # Prepare for Model: Torch format
    Lr_SAI_y = torch.from_numpy(Lr_SAI_y).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, U*H, V*W)
    Sr_SAI_cbcr_tensor = torch.from_numpy(Sr_SAI_cbcr).permute(2, 0, 1).unsqueeze(0) # (1, 2, U*H*S, V*W*S)
    data_info = [args.angRes_in, args.angRes_out]

    # Divide LFs into Patches
    subLFin = LFdivide(Lr_SAI_y.squeeze(0), args.angRes_in, args.patch_size_for_test, args.stride_for_test)
    numU, numV, pH, pW = subLFin.size()
    subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
    subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
                           args.angRes_in * args.patch_size_for_test * args.scale_factor, device=device)

    # SR the Patches (No TTA)
    net.eval()
    with torch.no_grad():
        for i in range(0, numU * numV, args.minibatch_for_test):
            tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
            out = net(tmp, data_info)
            subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out

    subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

    # Restore Patches to LFs
    Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
                          args.stride_for_test * args.scale_factor, H, W)
    Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')
    
    # Recombine Y with CbCr and convert to RGB
    Sr_SAI_ycbcr = torch.cat((Sr_SAI_y.cpu(), Sr_SAI_cbcr_tensor), dim=1)
    Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0, 1) * 255).astype('uint8')
    Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

    # Save format required for CodaBench (.bmp per view)
    import imageio
    scene_dir = os.path.join(save_dir, filename.replace('.mat', '').replace('.h5', ''))
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
    parser.add_argument("--ckpt", type=str, default=None, help="Path to best Stage 1 checkpoint. Defaults to auto-search in log dir")
    args = parser.parse_args()

    # Step 1: Download or locate tests
    download_test_data()

    # Step 2: Model & Args Definition
    class Args:
        model_name = "MyEfficientLFNetV3_MLFIM"
        task = "SR"
        angRes_in = 5
        angRes_out = 5
        scale_factor = 4
        patch_size_for_test = 32
        stride_for_test = 16
        minibatch_for_test = 1
        # Model specific args
        mlfim_mask_ratio = 0.0
    
    config = Args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("\n=== STEP 2: Loading V3 Model ===")
    MODEL_PATH = 'model.SR.' + config.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(config).to(device)

    # Auto-find checkpoint if not provided. We want Stage 1 (best validation PSNR)
    # Search in standard train.py saving location for track2
    if args.ckpt is None:
        pth_files = glob.glob(f'log/SR_5x5_4x/ALL/{config.model_name}/checkpoints/*.pth')
        
        if not pth_files:
            # Let's try looking for any .pth in the log dir
            pth_files = glob.glob(f'log/*.pth')
        
        if not pth_files:
            print("❌ No .pth checkpoint found! Please provide it explicitly --ckpt <path>")
            sys.exit(1)
            
        pth_files.sort(key=os.path.getmtime, reverse=True) # Usually the last one
        best_ckpt = pth_files[0]
        print(f"Auto-selected checkpoint: {best_ckpt}")
    else:
        best_ckpt = args.ckpt

    print(f"Loading checkpoint: {best_ckpt}")
    checkpoint = torch.load(best_ckpt, map_location=device)
    state_dict = checkpoint.get('ema_state_dict', checkpoint.get('state_dict', checkpoint))
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(cleaned_state_dict, strict=True)
    net.eval()
    print("Checkpoint loaded successfully!")

    # Step 3: Run Inference directly to Target directory
    print("\n=== STEP 3: Running Inference & Generating Output .mat files ===")
    out_base = "submission_v3"
    shutil.rmtree(out_base, ignore_errors=True)
    os.makedirs(f"{out_base}/Real", exist_ok=True)
    os.makedirs(f"{out_base}/Synth", exist_ok=True)

    real_files = glob.glob("datasets_test/NTIRE_Test_Real/inference/*.mat")
    synth_files = glob.glob("datasets_test/NTIRE_Test_Synth/inference/*.mat")

    print(f"Processing {len(real_files)} Real instances...")
    for f in tqdm(real_files, ncols=70):
        process_file_direct(f, f"{out_base}/Real", net, device, config)

    print(f"Processing {len(synth_files)} Synth instances...")
    for f in tqdm(synth_files, ncols=70):
        process_file_direct(f, f"{out_base}/Synth", net, device, config)

    # Step 4: Zip Submission
    print("\n=== STEP 4: Creating submission.zip ===")
    zip_path = "submission_v3.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(out_base):
            for file in files:
                file_path = os.path.join(root, file)
                # Ensure the structure inside the zip is Real/... and Synth/... directly
                arcname = os.path.relpath(file_path, out_base)
                zipf.write(file_path, arcname)

    print(f"\n✅ Submission successfully created: {zip_path}")
    print("You can now upload this file to CodaBench.")

if __name__ == "__main__":
    main()
