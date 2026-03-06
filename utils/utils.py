"""
Utility functions for BasicLFSR / MyEfficientLFNet V3.

Adapted from the original BasicLFSR utils.py with V3-specific fixes:
  - Logger.log_string() uses instance attribute instead of global `args`
  - cal_metrics() uses data_range=1.0 (required by newer scikit-image)
  - cal_metrics() handles division-by-zero safely
  - LFintegrate_gaussian() added for Gaussian-weighted patch stitching
  - rearrange re-exported from einops for downstream `from utils.utils import *`
"""

import numpy as np
import os
from skimage import metrics
import torch
from pathlib import Path
import logging
from einops import rearrange
import xlwt
import torch.nn.functional as F


class ExcelFile():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10
        self.sum = 1

    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, LF_name[i], psnr_iter_test[i], ssim_iter_test[i])

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test)
        self.sum = self.sum + 1

    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.6f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.6f' % ssim_iter_test)
        self.sum = self.sum + 1


def get_logger(log_dir, args):
    '''LOG '''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def create_dir(args):
    log_dir = Path(args.path_log)
    log_dir.mkdir(exist_ok=True)
    if args.task == 'SR':
        task_path = 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x'
    elif args.task == 'RE':
        task_path = 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.angRes_out) + 'x' + str(args.angRes_out)
    else:
        task_path = args.task

    log_dir = log_dir.joinpath(task_path)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.data_name)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.model_name)
    log_dir.mkdir(exist_ok=True)

    checkpoints_dir = log_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    results_dir = log_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)

    return log_dir, checkpoints_dir, results_dir


class Logger():
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)
        # V3 FIX: Store local_rank as instance attribute instead of relying
        # on module-level global `args`. The original BasicLFSR code used
        # `if args.local_rank <= 0:` which only worked because option.py
        # was imported at module level. This breaks when train_mlfim_v3.py
        # uses its own args namespace.
        self.local_rank = getattr(args, 'local_rank', 0)

    def log_string(self, str):
        if self.local_rank <= 0:
            self.logger.info(str)
            print(str)


def cal_metrics(args, label, out,):
    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    for b in range(B):
        for u in range(U):
            for v in range(V):
                # V3 FIX: Added data_range=1.0 — required by newer scikit-image
                # versions (>=0.18) where data_range is no longer auto-detected.
                # Without this, PSNR values are wildly wrong on [0,1] float data.
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(
                    label_y[b, u, :, v, :].numpy(),
                    out_y[b, u, :, v, :].numpy(),
                    data_range=1.0)
                if args.task == 'RE':
                    SSIM[b, u, v] = metrics.structural_similarity(
                        label_y[b, u, :, v, :].numpy(),
                        out_y[b, u, :, v, :].numpy(),
                        gaussian_weights=True,
                        sigma=1.5, use_sample_covariance=False,
                        data_range=1.0)
                else:
                    SSIM[b, u, v] = metrics.structural_similarity(
                        label_y[b, u, :, v, :].numpy(),
                        out_y[b, u, :, v, :].numpy(),
                        gaussian_weights=True,
                        data_range=1.0)
                pass

    if args.task == 'RE':
        for u in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
            for v in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
                PSNR[:, u, v] = 0
                SSIM[:, u, v] = 0

    # V3 FIX: Guard against division-by-zero when all PSNR/SSIM values are 0
    valid_psnr = np.sum(PSNR > 0)
    PSNR_mean = PSNR.sum() / valid_psnr if valid_psnr > 0 else 0.0

    valid_ssim = np.sum(SSIM > 0)
    SSIM_mean = SSIM.sum() / valid_ssim if valid_ssim > 0 else 0.0

    return PSNR_mean, SSIM_mean


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
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    # V3 FIX: Compute numU/numV from the actual padded dimensions instead of
    # the old formula `(h0 + bdr*2 - 1) // stride` which is wrong when
    # stride != patch_size/2 (e.g., stride=8, patch_size=32).
    h_pad, w_pad = data_pad.shape[2], data_pad.shape[3]
    numU = (h_pad - patch_size) // stride + 1
    numV = (w_pad - patch_size) // stride + 1
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


def LFintegrate_gaussian(subLF, angRes, pz, stride, h, w):
    """Gaussian-weighted patch stitching for LF reconstruction.

    Instead of hard-cropping overlapping borders (like LFintegrate), this
    method blends overlapping patches using a 2D Gaussian weight map.
    Patches near the center of each tile contribute more than those at
    borders, producing smoother seams without visible stitching artifacts.

    This is used in train.py's test() function when use_gaussian_psw=True.

    Args:
        subLF: (n1, n2, a1*pz, a2*pz) or (n1, n2, a1, a2, pz, pz)
        angRes: angular resolution (e.g. 5)
        pz: patch size in SR space (patch_size * scale)
        stride: stride in SR space (stride_for_test * scale)
        h: target spatial height per view
        w: target spatial width per view

    Returns:
        outLF: (a1, a2, h, w) — stitched LF
    """
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w',
                          a1=angRes, a2=angRes)

    n1, n2, a1, a2, pH, pW = subLF.shape

    # Build 2D Gaussian weight map for one patch
    sigma = pz / 4.0
    ax = torch.arange(pz, dtype=torch.float32) - (pz - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)  # (pz, pz)
    gauss_2d = gauss_2d / gauss_2d.max()  # normalize peak to 1.0

    # Accumulate weighted patches into output canvas
    canvas_h = (n1 - 1) * stride + pz
    canvas_w = (n2 - 1) * stride + pz

    outLF = torch.zeros(a1, a2, canvas_h, canvas_w, dtype=subLF.dtype)
    weight_map = torch.zeros(1, 1, canvas_h, canvas_w, dtype=subLF.dtype)

    for i in range(n1):
        for j in range(n2):
            top = i * stride
            left = j * stride
            outLF[:, :, top:top+pz, left:left+pz] += subLF[i, j] * gauss_2d
            weight_map[:, :, top:top+pz, left:left+pz] += gauss_2d

    # Normalize by accumulated weights
    weight_map = weight_map.clamp(min=1e-8)
    outLF = outLF / weight_map

    # Crop to target size, EXCLUDING the padded border!
    # LFdivide added `bdr` padding to the top/left of the LR image.
    # In HR space, this is `bdr * scale` padding at the top/left.
    # We must start cropping AFTER this padding!
    lr_pz = pz // (h // subLF.shape[2]) # Infer scale factor (target h is per-view, but wait, scale isn't passed)
    # Actually, a better way is to deduce bdr from the overlap:
    # We know pz = lr_pz * scale, stride = lr_stride * scale.
    # The padding added in LR space was `bdr_lr = (lr_pz - lr_stride) // 2`.
    # In HR space, `bdr_hr = (pz - stride) // 2`.
    bdr_hr = (pz - stride) // 2
    
    outLF = outLF[:, :, bdr_hr : bdr_hr + h, bdr_hr : bdr_hr + w]

    return outLF


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
