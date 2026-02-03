#!/usr/bin/env python3
"""
Standalone training data generator for NTIRE 2026 LF-SR Challenge.
This script has NO external option.py dependencies.

Usage:
    python generate_training_patches.py
"""

import os
import h5py
import numpy as np
import scipy.io as scio
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these if needed
# ============================================================================
ANG_RES = 5           # Angular resolution (5x5 views)
SCALE_FACTOR = 4      # Super-resolution scale (4x)
SRC_DATA_PATH = './datasets/'
SAVE_DATA_PATH = './'
DATA_FOR = 'training'  # 'training' or 'test'
# ============================================================================


def rgb2ycbcr(rgb_image):
    """Convert RGB to YCbCr, return only Y channel normalized to [0,1]."""
    rgb_image = rgb_image.astype(np.float64)
    # ITU-R BT.601 conversion
    transform_matrix = np.array([
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.0],
        [112.0, -93.786, -18.214]
    ])
    shift = np.array([16, 128, 128])
    
    ycbcr = np.zeros_like(rgb_image)
    for i in range(3):
        ycbcr[:, :, i] = (transform_matrix[i, 0] * rgb_image[:, :, 0] +
                          transform_matrix[i, 1] * rgb_image[:, :, 1] +
                          transform_matrix[i, 2] * rgb_image[:, :, 2]) / 255.0 + shift[i]
    
    return ycbcr


def imresize(img, scalar_scale):
    """Simple bicubic downsampling using PIL."""
    from PIL import Image
    h, w = img.shape[:2]
    new_h, new_w = int(h * scalar_scale), int(w * scalar_scale)
    
    if len(img.shape) == 2:
        pil_img = Image.fromarray(img.astype(np.float32), mode='F')
        resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
        return np.array(resized)
    else:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
        return np.array(resized).astype(np.float64) / 255.0


def main():
    angRes = ANG_RES
    scale_factor = SCALE_FACTOR
    patchsize = scale_factor * 32
    stride = patchsize // 2
    downRatio = 1 / scale_factor

    # Create output directory
    save_dir = Path(SAVE_DATA_PATH + 'data_for_' + DATA_FOR)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath('SR_' + str(angRes) + 'x' + str(angRes) + '_' + str(scale_factor) + 'x')
    save_dir.mkdir(exist_ok=True)

    src_datasets = os.listdir(SRC_DATA_PATH)
    src_datasets.sort()
    
    valid_datasets = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']
    
    for name_dataset in src_datasets:
        if name_dataset not in valid_datasets:
            continue
            
        idx_save = 0
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)

        src_sub_dataset = SRC_DATA_PATH + name_dataset + '/' + DATA_FOR + '/'
        if not os.path.isdir(src_sub_dataset):
            print(f"Skipping {name_dataset}: {src_sub_dataset} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {name_dataset}...")
        print(f"{'='*60}")

        for root, dirs, files in os.walk(src_sub_dataset):
            for file in files:
                if not file.endswith('.mat'):
                    continue
                    
                idx_scene_save = 0
                print(f'  Processing {file}...', end=' ', flush=True)
                
                try:
                    try:
                        data = h5py.File(os.path.join(root, file), 'r')
                        LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                    except:
                        data = scio.loadmat(os.path.join(root, file))
                        LF = np.array(data['LF'])

                    (U, V, _, _, _) = LF.shape
                    LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, :, :, 0:3]
                    LF = LF.astype('double')
                    (U, V, H, W, _) = LF.shape

                    for h in range(0, H - patchsize + 1, stride):
                        for w in range(0, W - patchsize + 1, stride):
                            idx_save += 1
                            idx_scene_save += 1
                            Hr_SAI_y = np.zeros((U * patchsize, V * patchsize), dtype='single')
                            Lr_SAI_y = np.zeros((U * patchsize // scale_factor, V * patchsize // scale_factor), dtype='single')

                            for u in range(U):
                                for v in range(V):
                                    tmp_Hr_rgb = LF[u, v, h: h + patchsize, w: w + patchsize, :]
                                    tmp_Hr_ycbcr = rgb2ycbcr(tmp_Hr_rgb)
                                    tmp_Hr_y = tmp_Hr_ycbcr[:, :, 0]

                                    patchsize_Lr = patchsize // scale_factor
                                    Hr_SAI_y[u * patchsize: (u+1) * patchsize, v * patchsize: (v+1) * patchsize] = tmp_Hr_y
                                    tmp_Sr_y = imresize(tmp_Hr_y, scalar_scale=downRatio)
                                    Lr_SAI_y[u*patchsize_Lr: (u+1)*patchsize_Lr, v*patchsize_Lr: (v+1)*patchsize_Lr] = tmp_Sr_y

                            # Save patch
                            file_name = str(sub_save_dir) + '/' + '%06d' % idx_save + '.h5'
                            with h5py.File(file_name, 'w') as hf:
                                hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y.transpose((1, 0)), dtype='single')
                                hf.create_dataset('Hr_SAI_y', data=Hr_SAI_y.transpose((1, 0)), dtype='single')

                    print(f'{idx_scene_save} patches')
                    
                except Exception as e:
                    print(f'ERROR: {e}')
                    continue

    print(f"\n{'='*60}")
    print(f"Done! Training data saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
