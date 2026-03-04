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
        data: LR image (H_lr, W_lr) — SAI mosaic
        label: HR image (H_hr, W_hr) — SAI mosaic
        alpha: max fraction of area to cut (default 0.7 = up to 70%)
    """
    H_lr, W_lr = data.shape[:2]
    H_hr, W_hr = label.shape[:2]
    scale = H_hr // H_lr

    # Random cut ratio between 0.25 and alpha
    cut_ratio = random.uniform(0.25, alpha)
    # CRITICAL: dimensions must be aligned to scale factor for reshape
    cut_h = int(H_hr * np.sqrt(cut_ratio)) // scale * scale
    cut_w = int(W_hr * np.sqrt(cut_ratio)) // scale * scale

    # Minimum cut size = scale (1 LR pixel)
    cut_h = max(cut_h, scale)
    cut_w = max(cut_w, scale)

    # Random position in HR space, aligned to scale factor
    cy = random.randint(0, (H_hr - cut_h) // scale) * scale
    cx = random.randint(0, (W_hr - cut_w) // scale) * scale

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
    hr_patch = label[cy:cy+cut_h, cx:cx+cut_w]
    if lr_ch > 0 and lr_cw > 0:
        hr_patch_ds = hr_patch.reshape(lr_ch, scale, lr_cw, scale).mean(axis=(1, 3))
        data_cut[lr_cy:lr_cy+lr_ch, lr_cx:lr_cx+lr_cw] = hr_patch_ds

    return data_cut, label_cut


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction (axis=1)
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along H-U direction (axis=0)
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    # Random gamma correction — simulates varying exposure/lighting.
    # Applied to BOTH LR and HR so the target remains consistent.
    # Helps the model generalize on real-world Lytro test images (different
    # tonal distributions than synthetic training data).
    # LR-only gamma is wrong — it changes the target mapping.
    if random.random() < 0.3:  # 30% probability, conservative
        gamma = random.uniform(0.7, 1.4)
        # Clip to [0,1] after gamma to prevent overflow in float32 Y-channel
        data = np.clip(np.power(np.clip(data, 1e-8, 1.0), gamma), 0.0, 1.0)
        label = np.clip(np.power(np.clip(label, 1e-8, 1.0), gamma), 0.0, 1.0)
    # CutBlur augmentation (Yoo et al., CVPR 2020) — mixes LR↔HR patches
    # to improve feature robustness. 30% probability, conservative.
    if random.random() < 0.3:
        data, label = cutblur(data, label)
    return data, label

