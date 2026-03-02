"""
MLFIM Pre-training for MyEfficientLFNetV10.3 (Audit-Hardened)
=============================================================
Masked Light Field Image Modeling (LFTransMamba-style)

This script implements a 2-stage training pipeline:
  Stage 1: MLFIM pre-training (self-supervised, 50-100 epochs)
           — Randomly masks 25% of spatial tokens AFTER IFE
           — Model must reconstruct the full output (including masked regions)
           — Teaches the network to infer missing spatial information
           — Uses L1 loss (simpler objective — composite destabilizes masking)

  Stage 2: Fine-tuning (standard SR, 200 epochs)
           — Disable masking (mlfim_mask_ratio=0.0)
           — Load Stage 1 checkpoint
           — Train with full composite loss (Charb+FFT+SSIM+Grad+Angular)
           — Uses EMA (decay=0.999) for ~0.05-0.1 dB free PSNR gain

Reference:
  LFTransMamba (CVPRW 2025, 1st NTIRE 2025)
  — Pre-training with random masking enables better feature representations
  — Official: 25% mask ratio, applied at feature level after IFE
  — Zero inference cost (masking disabled at inference via self.training)

Usage:
  # Stage 1: MLFIM Pre-training
  python train_mlfim.py --stage pretrain --mlfim_mask_ratio 0.25 --epoch 80 \\
      --lr 2e-4 --model_name MyEfficientLFNetV10

  # Stage 2: Fine-tuning (loads Stage 1 checkpoint)
  python train_mlfim.py --stage finetune --epoch 150 --lr 5e-5 \\
      --path_pre_pth <stage1_checkpoint.pth> --model_name MyEfficientLFNetV10
"""

import argparse
import importlib
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader


def parse_mlfim_args():
    """Parse MLFIM-specific arguments on top of standard training args."""
    from option import args as base_args

    parser = argparse.ArgumentParser(description="MLFIM Pre-training for V10.3")
    parser.add_argument('--stage', type=str, choices=['pretrain', 'finetune'],
                        required=True, help='Training stage')
    parser.add_argument('--mlfim_mask_ratio', type=float, default=0.25,
                        help='Mask ratio for MLFIM pre-training (default: 0.25)')

    mlfim_args, _ = parser.parse_known_args()

    # Merge MLFIM args into base args
    base_args.stage = mlfim_args.stage
    base_args.mlfim_mask_ratio = (
        mlfim_args.mlfim_mask_ratio if mlfim_args.stage == 'pretrain' else 0.0
    )

    return base_args


def train_one_epoch(train_loader, device, net, criterion, optimizer, args, stage, ema=None):
    """Train one epoch with optional MLFIM masking and per-step EMA updates."""
    net.train()
    psnr_list, loss_list, ssim_list = [], [], []

    for idx_iter, (data, label, data_info) in tqdm(
        enumerate(train_loader), total=len(train_loader), ncols=70
    ):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = net(data, data_info)
            loss = criterion(out, label, data_info) if hasattr(criterion, 'angular_loss') else criterion(out, label)

        if torch.isnan(loss):
            print(f"Warning: NaN loss at iter {idx_iter}, skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        # Per-step EMA update — critical for correct smoothing.
        if ema is not None:
            ema.update(net)

        loss_list.append(loss.data.cpu())
        psnr, ssim = cal_metrics(args, label.detach(), out.detach().float())
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    return (
        float(np.array(loss_list).mean()),
        float(np.array(psnr_list).mean()),
        float(np.array(ssim_list).mean()),
    )


# ============================================================================
# EMA (Exponential Moving Average) — free +0.05-0.1 dB PSNR
# ============================================================================
class ModelEMA:
    """Maintains an exponential moving average of model parameters.
    EMA smooths noisy SGD updates, producing a more generalizable model.
    Standard in SR: EDSR, SwinIR, HAT all use EMA for final submissions."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model):
        """Replace model params with EMA params (for validation/inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model params after validation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return self.shadow.copy()

    def load_state_dict(self, state_dict):
        self.shadow = state_dict.copy()


def _load_state_dict_flexible(net, state_dict):
    """Load state dict handling module. prefix in either direction.
    B4 Fix: Previous code always tried module. prefix first and relied
    on exception fallback. This handles both cases cleanly in one pass."""
    model_keys = set(net.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    if model_keys == ckpt_keys:
        net.load_state_dict(state_dict)
        return

    # Try stripping 'module.' from checkpoint keys
    stripped = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace('module.', '', 1) if k.startswith('module.') else k
        stripped[new_k] = v
    if set(stripped.keys()) == model_keys:
        net.load_state_dict(stripped)
        return

    # Try adding 'module.' to checkpoint keys
    prefixed = OrderedDict()
    for k, v in state_dict.items():
        prefixed['module.' + k] = v
    if set(prefixed.keys()) == model_keys:
        net.load_state_dict(prefixed)
        return

    # Fallback: strict=False for partial loading
    net.load_state_dict(state_dict, strict=False)


def main():
    args = parse_mlfim_args()

    # Reproducibility
    seed = getattr(args, 'seed', 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Dirs
    log_dir, checkpoints_dir, val_dir = create_dir(args)
    logger = Logger(log_dir, args)

    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    # Data
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    train_loader = DataLoader(
        dataset=train_Dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)

    # ---- Model ----
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    stage = args.stage
    logger.log_string(f'\n{"="*60}')
    logger.log_string(f'MLFIM Training — Stage: {stage.upper()}')
    logger.log_string(f'Mask ratio: {args.mlfim_mask_ratio}')
    logger.log_string(f'{"="*60}\n')

    # ---- Load checkpoint (B4 Fix: flexible prefix handling) ----
    start_epoch = 0
    if args.use_pre_ckpt and hasattr(args, 'path_pre_pth'):
        try:
            ckpt = torch.load(args.path_pre_pth, map_location='cpu')
            start_epoch = ckpt.get('epoch', 0)
            _load_state_dict_flexible(net, ckpt['state_dict'])
            logger.log_string(f'Loaded checkpoint from epoch {start_epoch}')

            if stage == 'finetune':
                logger.log_string('Fine-tuning: masking DISABLED (mask_ratio=0.0)')
                start_epoch = 0  # Reset epoch counter for fine-tuning
        except Exception as e:
            logger.log_string(f'Checkpoint load failed: {e}')
            net.apply(MODEL.weights_init)
            start_epoch = 0
    else:
        net.apply(MODEL.weights_init)

    net = net.to(device)
    torch.backends.cudnn.benchmark = True

    # ---- EMA (T7: free +0.05-0.1 dB PSNR) ----
    ema = ModelEMA(net, decay=0.999)
    logger.log_string('EMA enabled (decay=0.999)')

    # ---- Loss (B2 Fix: Stage 1 uses L1, Stage 2 uses full composite) ----
    if stage == 'pretrain':
        # Stage 1: Simple L1 loss for pre-training (LFTransMamba design).
        # Composite loss destabilizes when 25% of tokens are masked —
        # SSIM windows span masked regions, gradient Sobel hits missing edges.
        criterion = nn.L1Loss().to(device)
        logger.log_string('Pre-training loss: L1 (stable for masked inputs)')
    else:
        # Stage 2: Full composite loss for fine-tuning
        criterion = MODEL.get_loss(args).to(device)
        logger.log_string('Fine-tuning loss: full composite (Charb+FFT+SSIM+Grad+Ang)')

    # ---- Optimizer ----
    lr = args.lr
    if stage == 'finetune':
        lr = min(args.lr, 5e-5)  # Lower LR for fine-tuning
        logger.log_string(f'Fine-tuning LR capped at: {lr}')

    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
    )

    # ---- Scheduler ----
    total_epochs = args.epoch
    warmup_epochs = min(5, total_epochs // 10)
    # Stage 2 fine-tuning uses tighter eta_min so the cosine tail can settle
    # into a very flat minimum for max PSNR. Stage 1 uses 1e-6 (faster decay).
    eta_min = 5e-7 if stage == 'finetune' else 1e-6
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    # ---- Training Loop ----
    logger.log_string('\nStart training...')
    best_psnr = 0.0

    for epoch in range(start_epoch, args.epoch):
        logger.log_string(f'\nEpoch {epoch + 1}/{args.epoch} '
                          f'[{stage.upper()}, mask={args.mlfim_mask_ratio}]:')

        # Bump EMA decay in the final 25% of training for tighter smoothing.
        # sigmoid of decay is not relevant here — this is a direct scalar.
        # 0.999 during early training (fast adaptation), 0.9999 late (stable average).
        if epoch >= int(args.epoch * 0.75):
            ema.decay = 0.9999

        loss_train, psnr_train, ssim_train = train_one_epoch(
            train_loader, device, net, criterion, optimizer, args, stage, ema=ema
        )

        # EMA is now updated per-step inside train_one_epoch — do NOT call here.

        logger.log_string(
            f'Train — loss: {loss_train:.5f}, '
            f'psnr: {psnr_train:.5f}, ssim: {ssim_train:.5f}'
        )

        # Save checkpoint (includes EMA state)
        save_path = str(checkpoints_dir) + (
            f'/{args.model_name}_{stage}_{args.angRes_in}x{args.angRes_in}'
            f'_{args.scale_factor}x_epoch_{epoch+1:02d}_model.pth'
        )
        state = {
            'epoch': epoch + 1,
            'stage': stage,
            'mlfim_mask_ratio': args.mlfim_mask_ratio,
            'state_dict': (
                net.module.state_dict() if hasattr(net, 'module')
                else net.state_dict()
            ),
            'ema_state_dict': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(state, save_path)

        # Validation every 5 epochs. The `% step == 0` covers epoch 5,10,...,200.
        # No separate near-end guard needed — epoch 200 is already divisible by 5.
        step = 5
        if (epoch + 1) % step == 0:
            # Use EMA weights for validation (better generalization)
            ema.apply_shadow(net)
            net.eval()

            # B3 Fix: Collect PSNR across ALL test sets, use aggregate for best
            all_psnrs = []
            with torch.no_grad():
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]
                    from train import test
                    psnr_iter, ssim_iter, _ = test(
                        test_loader, device, net, args
                    )
                    psnr_avg = float(np.array(psnr_iter).mean())
                    ssim_avg = float(np.array(ssim_iter).mean())
                    all_psnrs.append(psnr_avg)
                    logger.log_string(
                        f'  Val {test_name}: '
                        f'psnr={psnr_avg:.2f}, ssim={ssim_avg:.3f}'
                    )

            # B3 Fix: Best checkpoint based on mean PSNR across ALL test sets
            aggregate_psnr = float(np.mean(all_psnrs))
            logger.log_string(f'  Aggregate PSNR: {aggregate_psnr:.2f} dB')
            if aggregate_psnr > best_psnr:
                best_psnr = aggregate_psnr
                best_path = str(checkpoints_dir) + (
                    f'/{args.model_name}_{stage}_best.pth'
                )
                torch.save(state, best_path)
                logger.log_string(f'  ★ New best: {aggregate_psnr:.2f} dB (aggregate)')

            # Restore training weights after validation
            ema.restore(net)

        scheduler.step()

    logger.log_string(f'\n{"="*60}')
    logger.log_string(f'Stage {stage.upper()} complete! Best PSNR: {best_psnr:.2f} dB (aggregate)')
    logger.log_string(f'{"="*60}')


if __name__ == '__main__':
    main()
