"""
MLFIM Training V2 — Improved Pipeline for Maximum PSNR
=======================================================
Key improvements over train_mlfim.py:
  1. Finetune LR: 2e-4 (research-backed sweet spot for LFSR)
  2. Gradient accumulation: effective batch_size=8 (2 accum steps × batch=4)
  3. Loss scheduling: L1-only for first 10 finetune epochs, then composite
  4. Longer warmup (10 epochs for finetune) — stabilizes after L1→composite switch
  5. Cosine eta_min 1e-6 (was 5e-7) — slightly more LR in the tail

Usage:
  # Stage 1: MLFIM Pre-training
  python train_mlfim_v2.py --stage pretrain --mlfim_mask_ratio 0.25 --epoch 50 \\
      --lr 2e-4 --model_name MyEfficientLFNetV2_MLFIM

  # Stage 2: Fine-tuning (optimized)
  python train_mlfim_v2.py --stage finetune --epoch 100 --lr 2e-4 \\
      --path_pre_pth <stage1_best.pth> --model_name MyEfficientLFNetV2_MLFIM \\
      --use_pre_ckpt
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
from train import test as run_validation_test


def parse_mlfim_args():
    """Parse MLFIM-specific arguments on top of standard training args."""
    from option import args as base_args

    parser = argparse.ArgumentParser(description="MLFIM Training V2")
    parser.add_argument('--stage', type=str, choices=['pretrain', 'finetune'],
                        required=True, help='Training stage')
    parser.add_argument('--mlfim_mask_ratio', type=float, default=0.25,
                        help='Mask ratio for MLFIM pre-training (default: 0.25)')
    parser.add_argument('--grad_accum_steps', type=int, default=2,
                        help='Gradient accumulation steps for effective batch size '
                             '(effective_bs = batch_size * grad_accum_steps)')
    parser.add_argument('--loss_warmup_epochs', type=int, default=20,
                        help='Number of finetune epochs to use L1-only before '
                             'switching to composite loss')

    mlfim_args, _ = parser.parse_known_args()

    # Merge MLFIM args into base args
    base_args.stage = mlfim_args.stage
    base_args.mlfim_mask_ratio = (
        mlfim_args.mlfim_mask_ratio if mlfim_args.stage == 'pretrain' else 0.0
    )
    base_args.grad_accum_steps = mlfim_args.grad_accum_steps
    base_args.loss_warmup_epochs = mlfim_args.loss_warmup_epochs

    return base_args


def train_one_epoch(train_loader, device, net, criterion, optimizer, args,
                    stage, ema=None, grad_accum_steps=1):
    """Train one epoch with gradient accumulation and per-step EMA updates."""
    net.train()
    psnr_list, loss_list, ssim_list = [], [], []

    optimizer.zero_grad()

    for idx_iter, (data, label, data_info) in tqdm(
        enumerate(train_loader), total=len(train_loader), ncols=70
    ):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        data = data.to(device)
        label = label.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = net(data, data_info)
            loss = criterion(out, label, data_info) if hasattr(criterion, 'angular_loss') else criterion(out, label)

        # BUG FIX: Check NaN BEFORE dividing and accumulating.
        # If we divide NaN and then continue, stale NaN gradients remain
        # in the accumulator from any previous backward() calls.
        if torch.isnan(loss):
            print(f"Warning: NaN loss at iter {idx_iter}, skipping")
            # Zero out accumulated gradients to prevent NaN contamination
            optimizer.zero_grad()
            continue

        # Scale loss by accumulation steps so effective gradient magnitude
        # is the same as if we used a larger batch directly
        loss = loss / grad_accum_steps

        loss.backward()

        # Step optimizer every grad_accum_steps iterations
        if (idx_iter + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Per-step EMA update — after optimizer.step()
            if ema is not None:
                ema.update(net)

        # Unscale loss for logging
        loss_list.append((loss.data.cpu() * grad_accum_steps).item())
        # BUG FIX 13: Only compute metrics every 50 iters to avoid skimage overhead
        # (~50ms per iter × 1000 iters = 50 seconds wasted per epoch)
        if (idx_iter + 1) % 50 == 0 or idx_iter == len(train_loader) - 1:
            psnr, ssim = cal_metrics(args, label.detach(), out.detach().float())
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    # Handle any remaining accumulated gradients
    if len(train_loader) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update(net)

    return (
        float(np.array(loss_list).mean()),
        float(np.array(psnr_list).mean()),
        float(np.array(ssim_list).mean()),
    )


# ============================================================================
# EMA (Exponential Moving Average) — free +0.05-0.1 dB PSNR
# ============================================================================
class ModelEMA:
    """Maintains an exponential moving average of model parameters."""

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


def _load_state_dict_flexible(net, state_dict, logger=None):
    """Load state dict handling module. prefix and architecture changes."""
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

    # BUG FIX: Previous code called load_state_dict(strict=False) twice,
    # which corrupts model state because the first call partially loads.
    # Now: determine which key set has more overlap, then load once.
    raw_overlap = len(ckpt_keys & model_keys)
    stripped_overlap = len(set(stripped.keys()) & model_keys)
    best_dict = stripped if stripped_overlap >= raw_overlap else state_dict

    missing, unexpected = net.load_state_dict(best_dict, strict=False)

    if logger:
        if missing:
            logger.log_string(f'  Missing keys ({len(missing)}): {missing[:5]}...')
        if unexpected:
            logger.log_string(f'  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...')


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
    logger.log_string(f"Training samples: {len(train_Dataset)}")
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
    grad_accum = args.grad_accum_steps if stage == 'finetune' else 1
    effective_bs = args.batch_size * grad_accum

    logger.log_string(f'\n{"="*60}')
    logger.log_string(f'MLFIM Training V2 — Stage: {stage.upper()}')
    logger.log_string(f'Mask ratio: {args.mlfim_mask_ratio}')
    logger.log_string(f'Batch size: {args.batch_size} × {grad_accum} accum = {effective_bs} effective')
    logger.log_string(f'Loss warmup: {args.loss_warmup_epochs} epochs (L1-only before composite)')
    logger.log_string(f'{"="*60}\n')

    # ---- Load checkpoint ----
    start_epoch = 0
    if args.use_pre_ckpt and hasattr(args, 'path_pre_pth'):
        try:
            ckpt = torch.load(args.path_pre_pth, map_location='cpu')
            start_epoch = ckpt.get('epoch', 0)
            _load_state_dict_flexible(net, ckpt['state_dict'], logger)
            logger.log_string(f'Loaded checkpoint from epoch {start_epoch}')

            if stage == 'finetune':
                logger.log_string('Fine-tuning: masking DISABLED (mask_ratio=0.0)')
                start_epoch = 0  # Reset epoch counter for fine-tuning
                # Don't load optimizer/scheduler — finetune has different LR schedule
            else:
                # Pretrain resume: restore optimizer/scheduler state for seamless continuation
                _resume_optimizer = ckpt.get('optimizer', None)
                _resume_scheduler = ckpt.get('scheduler', None)
                _resume_ema = ckpt.get('ema_state_dict', None)
        except Exception as e:
            logger.log_string(f'Checkpoint load failed: {e}')
            net.apply(MODEL.weights_init)
            start_epoch = 0
            _resume_optimizer = None
            _resume_scheduler = None
            _resume_ema = None
    else:
        net.apply(MODEL.weights_init)
        _resume_optimizer = None
        _resume_scheduler = None
        _resume_ema = None

    net = net.to(device)
    torch.backends.cudnn.benchmark = True

    # Print param count
    params = sum(p.numel() for p in net.parameters())
    logger.log_string(f'Parameters: {params:,} ({params/1e6:.3f}M)')

    # ---- EMA ----
    ema = ModelEMA(net, decay=0.999)
    # BUG FIX 12: Restore EMA state on pretrain resume
    if _resume_ema is not None and stage == 'pretrain':
        ema.load_state_dict(_resume_ema)
        logger.log_string('EMA state restored from checkpoint')
    else:
        logger.log_string('EMA enabled (decay=0.999, fresh start)')

    # ---- Loss functions ----
    # L1 criterion (used for pretrain and loss warmup period)
    l1_criterion = nn.L1Loss().to(device)
    # Composite criterion (used after warmup in finetune stage)
    composite_criterion = MODEL.get_loss(args).to(device) if stage == 'finetune' else None

    if stage == 'pretrain':
        logger.log_string('Pre-training loss: L1 (entire stage)')
    else:
        logger.log_string(f'Fine-tuning loss: L1 for epochs 1-{args.loss_warmup_epochs}, '
                         f'then composite (Charb+FFT+SSIM+Grad+Ang)')

    # ---- Optimizer ----
    # Key change: finetune LR = 1e-4 (was 5e-5 in v1)
    # This allows the model to escape the pretrain local minimum before settling.
    lr = args.lr  # V2: let the script arg control LR directly (no cap)

    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
    )
    logger.log_string(f'Optimizer: AdamW, LR={lr}, weight_decay=1e-4')

    # ---- Scheduler ----
    total_epochs = args.epoch
    warmup_epochs = min(10 if stage == 'finetune' else 5, max(total_epochs // 10, 5))
    eta_min = 1e-6  # V2: 1e-6 (was 5e-7) — more LR in the tail
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
    logger.log_string(f'Scheduler: {warmup_epochs}ep warmup → cosine to {eta_min}')

    # BUG FIX 12: Restore optimizer/scheduler state on pretrain resume
    if _resume_optimizer is not None and stage == 'pretrain':
        try:
            optimizer.load_state_dict(_resume_optimizer)
            logger.log_string('Optimizer state restored from checkpoint')
        except Exception as e:
            logger.log_string(f'Optimizer restore failed (new arch?): {e}')
    if _resume_scheduler is not None and stage == 'pretrain':
        try:
            scheduler.load_state_dict(_resume_scheduler)
            logger.log_string('Scheduler state restored from checkpoint')
        except Exception as e:
            logger.log_string(f'Scheduler restore failed: {e}')

    # ---- Training Loop ----
    logger.log_string('\nStart training...')
    best_psnr = 0.0

    for epoch in range(start_epoch, args.epoch):
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_string(f'\nEpoch {epoch + 1}/{args.epoch} '
                          f'[{stage.upper()}, mask={args.mlfim_mask_ratio}, '
                          f'lr={current_lr:.2e}]:')

        # Select loss function based on loss warmup schedule
        if stage == 'pretrain':
            criterion = l1_criterion
        elif epoch < args.loss_warmup_epochs:
            criterion = l1_criterion
            if epoch == 0:
                logger.log_string(f'  → Using L1 loss (warmup: epochs 1-{args.loss_warmup_epochs})')
        else:
            criterion = composite_criterion
            if epoch == args.loss_warmup_epochs:
                logger.log_string('  → Switching to composite loss')

        # EMA decay bump in final 25% of training
        if epoch >= int(args.epoch * 0.75):
            ema.decay = 0.9999

        loss_train, psnr_train, ssim_train = train_one_epoch(
            train_loader, device, net, criterion, optimizer, args,
            stage, ema=ema, grad_accum_steps=grad_accum
        )

        logger.log_string(
            f'Train — loss: {loss_train:.5f}, '
            f'psnr: {psnr_train:.2f}, ssim: {ssim_train:.4f}'
        )

        # Save checkpoint
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
        # Save checkpoint every 10 epochs (saves Colab disk space)
        if (epoch + 1) % 10 == 0:
            torch.save(state, save_path)

        # Validation every 5 epochs
        step = 5
        if (epoch + 1) % step == 0:
            ema.apply_shadow(net)
            net.eval()

            all_psnrs = []
            with torch.no_grad():
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]
                    psnr_iter, ssim_iter, _ = run_validation_test(
                        test_loader, device, net, args
                    )
                    psnr_avg = float(np.array(psnr_iter).mean())
                    ssim_avg = float(np.array(ssim_iter).mean())
                    all_psnrs.append(psnr_avg)
                    logger.log_string(
                        f'  Val {test_name}: '
                        f'psnr={psnr_avg:.2f}, ssim={ssim_avg:.3f}'
                    )

            aggregate_psnr = float(np.mean(all_psnrs))
            logger.log_string(f'  Aggregate PSNR: {aggregate_psnr:.2f} dB')
            if aggregate_psnr > best_psnr:
                best_psnr = aggregate_psnr
                best_path = str(checkpoints_dir) + (
                    f'/{args.model_name}_{stage}_best.pth'
                )
                torch.save(state, best_path)
                logger.log_string(f'  ★ New best: {aggregate_psnr:.2f} dB (aggregate)')

            ema.restore(net)

        scheduler.step()

    logger.log_string(f'\n{"="*60}')
    logger.log_string(f'Stage {stage.upper()} complete! Best PSNR: {best_psnr:.2f} dB')
    logger.log_string(f'{"="*60}')


if __name__ == '__main__':
    main()
