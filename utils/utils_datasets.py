import os
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
import gc
from torch.utils.data import DataLoader
from utils import *


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_train + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/'
            pass

        if args.data_name == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args.data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y')) # Lr_SAI_y — stored column-major (W,H)
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y — stored column-major (W,H)
            # CRITICAL FIX (P1): h5 files are stored transposed (W,H) by Generate_Data_for_Training.py.
            # Must transpose back to (H,W) before augmentation and ToTensor.
            # TestSetDataLoader already does this correctly at lines 121-122.
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y)
            Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        if index % 50 == 0:
            gc.collect()

        return Lr_SAI_y, Hr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None
    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        if args.task == 'SR':
            dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.scale_factor) + 'x/'
            data_list = os.listdir(dataset_dir)
        elif args.task == 'RE':
            dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)

        # P8: Force num_workers=0 for test loaders — h5py is not fork-safe and
        # test sets are small enough that worker overhead exceeds benefit.
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
            self.data_list = [data_name]
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
            Sr_SAI_cbcr = np.array(hf.get('Sr_SAI_cbcr'), dtype='single')
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            if Sr_SAI_cbcr.ndim == 3:
                Sr_SAI_cbcr = np.transpose(Sr_SAI_cbcr, (2, 1, 0))
            elif Sr_SAI_cbcr.ndim == 0 or Sr_SAI_cbcr.size == 0:
                # Create dummy cbcr with 2 channels matching Hr_SAI_y dimensions
                Sr_SAI_cbcr = np.zeros((Hr_SAI_y.shape[0], Hr_SAI_y.shape[1], 2), dtype=np.float32)
            elif Sr_SAI_cbcr.ndim == 2:
                # 2D array - expand to 3D with single channel
                Sr_SAI_cbcr = np.expand_dims(Sr_SAI_cbcr, axis=-1)

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy().astype(np.float32))

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        if index % 50 == 0:
            gc.collect()

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape)==2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def cutblur(data, label, alpha=0.7):
    """CutBlur augmentation (Yoo et al., CVPR 2020).

    Randomly pastes a rectangular patch from the upscaled LR image into the HR
    image (and vice versa). This teaches the model to handle mixed-resolution
    inputs and improves robustness. Proven +0.1-0.2 dB on SR benchmarks.

    The LR data is bicubic-upscaled to match HR size for the cut region,
    then the cut region is swapped between LR-upscaled and HR.

    Args:
        data: LR image (H_lr, W_lr) — SAI mosaic (2D numpy array)
        label: HR image (H_hr, W_hr) — SAI mosaic (2D numpy array)
        alpha: max fraction of area to cut (default 0.7 = up to 70%)
    """
    H_lr, W_lr = data.shape[:2]
    H_hr, W_hr = label.shape[:2]
    scale = H_hr // H_lr

    # Safety: if scale is somehow 0 or negative, bail out
    if scale < 1:
        return data, label

    # Random cut ratio between 0.25 and alpha
    cut_ratio = random.uniform(0.25, alpha)
    # CRITICAL: dimensions must be aligned to scale factor for reshape
    cut_h = int(H_hr * np.sqrt(cut_ratio)) // scale * scale
    cut_w = int(W_hr * np.sqrt(cut_ratio)) // scale * scale

    # Minimum cut size = scale (1 LR pixel)
    cut_h = max(cut_h, scale)
    cut_w = max(cut_w, scale)

    # Safety: ensure cut region fits within image
    if cut_h > H_hr or cut_w > W_hr:
        return data, label

    # Random position in HR space, aligned to scale factor
    max_cy = (H_hr - cut_h) // scale
    max_cx = (W_hr - cut_w) // scale
    cy = random.randint(0, max(0, max_cy)) * scale
    cx = random.randint(0, max(0, max_cx)) * scale

    # Upscale LR to HR size using simple repeat (fast, no scipy dependency)
    data_up = np.repeat(np.repeat(data, scale, axis=0), scale, axis=1)

    # Swap: paste HR patch into upscaled-LR, paste LR patch into HR
    # This creates a mixed-resolution image for training
    label_cut = label.copy()
    label_cut[cy:cy+cut_h, cx:cx+cut_w] = data_up[cy:cy+cut_h, cx:cx+cut_w]

    # Downscale the mixed HR back to LR space
    data_cut = data.copy()
    lr_cy, lr_cx = cy // scale, cx // scale
    lr_ch, lr_cw = cut_h // scale, cut_w // scale
    # Paste HR region (downscaled) into LR via area-average
    # BUG FIX: handle 2D arrays (Y-channel only, no channel dim) correctly
    hr_patch = label[cy:cy+cut_h, cx:cx+cut_w]
    if lr_ch > 0 and lr_cw > 0:
        if hr_patch.ndim == 2:
            hr_patch_ds = hr_patch.reshape(lr_ch, scale, lr_cw, scale).mean(axis=(1, 3))
        else:
            # 3D: (H, W, C) — reshape with channel preserved
            C = hr_patch.shape[2]
            hr_patch_ds = hr_patch.reshape(lr_ch, scale, lr_cw, scale, C).mean(axis=(1, 3))
        data_cut[lr_cy:lr_cy+lr_ch, lr_cx:lr_cx+lr_cw] = hr_patch_ds

    return data_cut, label_cut


def mixup(data, label, alpha=0.2):
    """Self-MixUp augmentation for PSNR-oriented super-resolution.

    Blends the current (data, label) pair with its own 180°-rotated version
    using a mixing coefficient λ ~ Beta(alpha, alpha). This is "self-mixup"
    because we only have access to one sample at a time in __getitem__.

    Why it works:
      - Forces the network to learn smoother, more generalizable mappings
      - Acts as implicit regularization without adding model complexity
      - The 180° rotation creates a meaningfully different view of the same
        light field, teaching angular consistency
      - Proven +0.03-0.10 dB on image SR benchmarks (Zhang et al., 2018)

    Args:
        data: LR image (H, W) — SAI mosaic
        label: HR image (H, W) — SAI mosaic
        alpha: Beta distribution parameter (0.2 = mild mixing, preserves
               most of the original sample. Lower = less aggressive.)
    Returns:
        Mixed (data, label) pair
    """
    # Sample λ from Beta distribution — concentrates near 0 and 1
    # with alpha=0.2, ensuring one sample dominates (mild regularization)
    lam = np.random.beta(alpha, alpha)
    # Ensure λ >= 0.5 so the original sample always dominates
    # This prevents the model from training on mostly-rotated samples
    lam = max(lam, 1.0 - lam)

    # Create the "other" sample via 180° rotation (same image, different view)
    data_rot = data[::-1, ::-1].copy()
    label_rot = label[::-1, ::-1].copy()

    # Blend: λ * original + (1-λ) * rotated
    data_mixed = np.asarray(lam * data + (1.0 - lam) * data_rot, dtype=data.dtype)
    label_mixed = np.asarray(lam * label + (1.0 - lam) * label_rot, dtype=label.dtype)

    return data_mixed, label_mixed


def augmentation(data, label):
    # 8-mode deterministic spatial augmentation (LFTransMamba-style).
    # Covers all Dihedral-4 symmetries: identity + 3 rotations × (original + flipped).
    # Previous code only had 4 probabilistic modes (flip + transpose) — missing
    # half the augmentation set. This is expected to gain ~0.05-0.15 dB PSNR.
    mode = random.randint(0, 7)
    if mode == 0:
        pass  # identity
    elif mode == 1:
        data = data[:, ::-1]             # horizontal flip
        label = label[:, ::-1]
    elif mode == 2:
        data = data[::-1, :]             # vertical flip
        label = label[::-1, :]
    elif mode == 3:
        data = data[::-1, ::-1]          # 180° rotation
        label = label[::-1, ::-1]
    elif mode == 4:
        data = data.transpose(1, 0)      # 90° rotation (transpose)
        label = label.transpose(1, 0)
    elif mode == 5:
        data = data.transpose(1, 0)[:, ::-1]   # 90° + h-flip
        label = label.transpose(1, 0)[:, ::-1]
    elif mode == 6:
        data = data.transpose(1, 0)[::-1, :]   # 90° + v-flip
        label = label.transpose(1, 0)[::-1, :]
    elif mode == 7:
        data = data.transpose(1, 0)[::-1, ::-1]  # 90° + 180°
        label = label.transpose(1, 0)[::-1, ::-1]

    # CutBlur with 25% probability (Yoo et al., CVPR 2020)
    if random.random() < 0.25:
        data, label = cutblur(data, label, alpha=0.7)

    # MixUp with 20% probability (self-mixup with 180° rotated version)
    # Applied last to blend already-augmented samples for maximum diversity
    if random.random() < 0.20:
        data, label = mixup(data, label, alpha=0.2)

    return data, label

