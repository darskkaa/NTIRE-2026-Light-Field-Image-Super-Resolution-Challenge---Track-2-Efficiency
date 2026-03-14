"""
MLFIM Training V3 — Max-PSNR Pipeline (Fixed)
===============================================
Research-backed training recipe for NTIRE 2026 Track 2 Efficiency.

Key design decisions (all backed by LFTransMamba 1st NTIRE 2025):
  1. Loss: L1 for pretrain (LFTransMamba default), Charbonnier for finetune
  2. Optimizer: Adam with β1=0.99, β2=0.999 (LFTransMamba exact)
  3. Scheduler: StepLR ×0.5 @80ep for pretrain, MultiStepLR([80,160]) for finetune
  4. LR: 2e-4 pretrain, 1e-4 finetune (LFTransMamba default)
  5. Warmup: 5-epoch linear warmup (proven in SwinIR/HAT/MambaIR)
  6. EMA: decay=0.997 (LFTransMamba exact), bumps to 0.999 at 75%
  7. SWA: Stochastic Weight Averaging in final 10% epochs (+0.05-0.1 dB)
  8. No bfloat16: full float32 training

Usage:
  # Stage 1: MLFIM Pre-training (100 epochs)
  python train_mlfim_v3.py --stage pretrain --mlfim_mask_ratio 0.25 --epoch 100 \\
      --lr 2e-4 --model_name MyEfficientLFNetV3_MLFIM --loss_type l1

  # Stage 2: Fine-tuning (200 epochs, max-PSNR recipe)
  python train_mlfim_v3.py --stage finetune --epoch 200 --lr 1e-4 \\
      --loss_type charbonnier --scheduler_type multistep \\
      --path_pre_pth <stage1_best.pth> --model_name MyEfficientLFNetV3_MLFIM \\
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
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from collections import OrderedDict
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader, AUG_CONFIG
from train import test as run_validation_test


def parse_mlfim_args():
    """Parse MLFIM-specific arguments on top of standard training args."""
    from option import args as base_args

    parser = argparse.ArgumentParser(description="MLFIM Training V3 — Max PSNR")
    parser.add_argument('--stage', type=str, choices=['pretrain', 'finetune'],
                        required=True, help='Training stage')
    parser.add_argument('--mlfim_mask_ratio', type=float, default=0.25,
                        help='Mask ratio for MLFIM pre-training (default: 0.25, paper Table 4 optimal)')
    parser.add_argument('--grad_accum_steps', type=int, default=2,
                        help='Gradient accumulation steps for effective batch size '
                             '(effective_bs = batch_size * grad_accum_steps)')
    parser.add_argument('--loss_warmup_epochs', type=int, default=0,
                        help='Number of finetune epochs to use L1-only before '
                             'switching to composite loss (0 = no warmup)')
    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['l1', 'charbonnier', 'composite'],
                        help='Loss function: l1 (LFTransMamba default), '
                             'charbonnier (smooth L1), or composite')
    parser.add_argument('--scheduler_type', type=str, default='step',
                        choices=['step', 'cosine', 'multistep'],
                        help='LR scheduler: step (StepLR x0.5@80ep, default), '
                             'multistep (MultiStepLR [80,160]), or '
                             'cosine (CosineAnnealingLR)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Linear LR warmup epochs (0=disabled, 5=recommended)')
    parser.add_argument('--swa_start_frac', type=float, default=0.9,
                        help='Start SWA at this fraction of total epochs (0.9=last 10%%)')
    parser.add_argument('--val_step', type=int, default=10,
                        help='Validation frequency in epochs (default: 10, use 5 for finetune)')
    parser.add_argument('--no_augmentation', action='store_true', default=False,
                        help='Disable ALL data augmentation (for Stage 3 polish runs)')

    mlfim_args, _ = parser.parse_known_args()

    # Merge MLFIM args into base args
    base_args.stage = mlfim_args.stage
    base_args.mlfim_mask_ratio = (
        mlfim_args.mlfim_mask_ratio if mlfim_args.stage == 'pretrain' else 0.0
    )
    base_args.grad_accum_steps = mlfim_args.grad_accum_steps
    base_args.loss_warmup_epochs = mlfim_args.loss_warmup_epochs
    base_args.loss_type = mlfim_args.loss_type
    base_args.scheduler_type = mlfim_args.scheduler_type
    base_args.warmup_epochs = mlfim_args.warmup_epochs
    base_args.swa_start_frac = mlfim_args.swa_start_frac
    base_args.val_step = mlfim_args.val_step
    base_args.no_augmentation = mlfim_args.no_augmentation

    return base_args


def train_one_epoch(train_loader, device, net, criterion, optimizer, args,
                    stage, ema=None, grad_accum_steps=1):
    """Train one epoch with gradient accumulation and per-step EMA updates."""
    net.train()
    psnr_list, loss_list, ssim_list = [], [], []

    optimizer.zero_grad(set_to_none=True)

    for idx_iter, (data, label, data_info) in tqdm(
        enumerate(train_loader), total=len(train_loader), ncols=70
    ):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # Full float32 — no autocast. 800K model doesn't need mixed precision,
        # and bfloat16 caused a critical train/val precision mismatch (~20 dB gap).
        out = net(data, data_info)
        loss = criterion(out, label, data_info)

        # BUG FIX: Check NaN BEFORE dividing and accumulating.
        # If we divide NaN and then continue, stale NaN gradients remain
        # in the accumulator from any previous backward() calls.
        if torch.isnan(loss):
            print(f"Warning: NaN loss at iter {idx_iter}, skipping")
            # Zero out accumulated gradients to prevent NaN contamination
            optimizer.zero_grad(set_to_none=True)
            continue

        # Scale loss by accumulation steps so effective gradient magnitude
        # is the same as if we used a larger batch directly
        loss = loss / grad_accum_steps

        loss.backward()

        # Step optimizer every grad_accum_steps iterations
        if (idx_iter + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

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
        optimizer.zero_grad(set_to_none=True)
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
        # V3 Bug Fix 6: Deep copy — clone tensors to prevent external mutation
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        # V3 Bug Fix 6: Deep copy — clone tensors to prevent external mutation
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


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
    # NOTE: cudnn.deterministic removed — it kills PSNR by 0.1-0.2 dB and
    # slows training 15-30%. All NTIRE winners use non-deterministic cuDNN.
    # cudnn.benchmark (set below) is sufficient.

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
        drop_last=True,  # LFTransMamba match: prevents incomplete last batch
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
    warmup_epochs = getattr(args, 'warmup_epochs', 5)
    swa_start_frac = getattr(args, 'swa_start_frac', 0.9)

    logger.log_string(f'\n{"="*60}')
    logger.log_string(f'MLFIM Training V3 — Stage: {stage.upper()}')
    logger.log_string(f'Mask ratio: {args.mlfim_mask_ratio}')
    logger.log_string(f'Batch size: {args.batch_size} × {grad_accum} accum = {effective_bs} effective')
    logger.log_string(f'LR warmup: {warmup_epochs} epochs')
    logger.log_string(f'SWA: starts at {swa_start_frac*100:.0f}% of training')
    logger.log_string(f'{"="*60}\n')

    # ---- Configure augmentation for this stage ----
    # Pretrain: aggressive augmentation to build robust features
    # Finetune: reduced augmentation so optimizer settles into precise minimum
    # Polish (--no_augmentation): ZERO augmentation for clean memorization
    if getattr(args, 'no_augmentation', False):
        AUG_CONFIG['cutblur_prob'] = 0.0
        AUG_CONFIG['mixup_prob'] = 0.0
        AUG_CONFIG['mixup_alpha'] = 0.0
        logger.log_string(f'Augmentation: DISABLED (polish mode)')
    elif stage == 'finetune':
        AUG_CONFIG['cutblur_prob'] = 0.10   # Reduced from 0.25 (Z3 audit)
        AUG_CONFIG['mixup_prob'] = 0.15     # Reduced from 0.20 for precision
        AUG_CONFIG['mixup_alpha'] = 0.15    # Milder blending for fine-tuning
        logger.log_string(f'Augmentation: CutBlur={AUG_CONFIG["cutblur_prob"]}, '
                          f'MixUp={AUG_CONFIG["mixup_prob"]} '
                          f'(alpha={AUG_CONFIG["mixup_alpha"]})')
    else:
        AUG_CONFIG['cutblur_prob'] = 0.25
        AUG_CONFIG['mixup_prob'] = 0.20
        AUG_CONFIG['mixup_alpha'] = 0.2
        logger.log_string(f'Augmentation: CutBlur={AUG_CONFIG["cutblur_prob"]}, '
                          f'MixUp={AUG_CONFIG["mixup_prob"]} '
                          f'(alpha={AUG_CONFIG["mixup_alpha"]})')

    # ---- Load checkpoint ----
    start_epoch = 0
    _resume_optimizer = None
    _resume_scheduler = None
    _resume_ema = None
    if args.use_pre_ckpt and hasattr(args, 'path_pre_pth'):
        try:
            ckpt = torch.load(args.path_pre_pth, map_location='cpu')
            start_epoch = ckpt.get('epoch', 0)
            ckpt_stage = ckpt.get('stage', 'pretrain')
            _load_state_dict_flexible(net, ckpt['state_dict'], logger)
            logger.log_string(f'Loaded checkpoint from epoch {start_epoch}')

            if stage == 'finetune':
                logger.log_string('Fine-tuning: masking DISABLED (mask_ratio=0.0)')
                # If the checkpoint is from pretrain, we are STARTING fine-tuning.
                if ckpt_stage == 'pretrain':
                    start_epoch = 0  # Reset epoch counter for fine-tuning
                    logger.log_string('Starting fresh fine-tune from pretrain weights')
                else:
                    # If the checkpoint is already finetune, we are RESUMING fine-tuning.
                    logger.log_string('Resuming existing fine-tune run')
                    _resume_optimizer = ckpt.get('optimizer', None)
                    _resume_scheduler = ckpt.get('scheduler', None)
                    _resume_ema = ckpt.get('ema_state_dict', None)
            else:
                # Pretrain resume: restore optimizer/scheduler state
                _resume_optimizer = ckpt.get('optimizer', None)
                _resume_scheduler = ckpt.get('scheduler', None)
                _resume_ema = ckpt.get('ema_state_dict', None)

            # Stage 3 Polish Fix: If we are in polish mode (signaled by no_augmentation),
            # args.epoch (e.g., 40) means "40 MORE epochs", not "end at epoch 40".
            # Otherwise, range(195, 40) is empty and training skips.
            if getattr(args, 'no_augmentation', False):
                logger.log_string(f'Polish Mode Detected: Extending absolute total epochs from {args.epoch} to {start_epoch + args.epoch}')
                args.epoch = start_epoch + args.epoch

            # Fix #4: Free checkpoint memory immediately after extracting what we need
            del ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            logger.log_string(f'Checkpoint load failed: {e}')
            net.apply(MODEL.weights_init)
            start_epoch = 0
    else:
        net.apply(MODEL.weights_init)

    net = net.to(device)
    torch.backends.cudnn.benchmark = True

    # Print param count
    params = sum(p.numel() for p in net.parameters())
    logger.log_string(f'Parameters: {params:,} ({params/1e6:.3f}M)')

    # ---- EMA ----
    ema = ModelEMA(net, decay=0.997)  # LFTransMamba exact value: 0.997 (line 627 of lfsr.py)
    # BUG FIX 12: Restore EMA state on pretrain resume
    if _resume_ema is not None and stage == 'pretrain':
        ema.load_state_dict(_resume_ema)
        logger.log_string('EMA state restored from checkpoint')
    else:
        logger.log_string('EMA enabled (decay=0.997, fresh start)')

    # ---- Loss functions ----
    # LFTransMamba ground truth: plain L1Loss for all training.
    # Wrapped to accept optional data_info arg (training loop passes 3 args)
    class L1WithInfo(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.L1Loss()
        def forward(self, pred, target, data_info=None):
            return self.l1(pred, target)
    l1_criterion = L1WithInfo().to(device)
    # Charbonnier kept as alternative (not default)
    charb_eps = getattr(args, 'charbonnier_eps', 1e-9)
    class CharbonnierLoss(nn.Module):
        def __init__(self, eps):
            super().__init__()
            self.eps = eps
        def forward(self, pred, target, data_info=None):
            pred, target = pred.float(), target.float()
            return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))
    charbonnier_criterion = CharbonnierLoss(charb_eps).to(device)
    # Composite criterion (only created if needed)
    composite_criterion = None
    if args.loss_type == 'composite' and stage == 'finetune':
        composite_criterion = MODEL.get_loss(args).to(device)

    logger.log_string(f'Loss type: {args.loss_type}')
    if args.loss_type == 'l1':
        logger.log_string('  → L1Loss (LFTransMamba default)')
    elif args.loss_type == 'charbonnier':
        logger.log_string('  → Charbonnier (eps=1e-9)')
    else:
        logger.log_string(f'  → Composite (Charb+FFT+SSIM+Grad+Ang)')

    # ---- Optimizer ----
    # LFTransMamba (1st NTIRE 2025) exact config from lfsr.py:
    #   Adam(betas=(0.99, 0.999), eps=1e-08, weight_decay=0.0)
    lr = args.lr

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.99, 0.999),
        eps=1e-08,
    )
    logger.log_string(f'Optimizer: Adam, LR={lr}, β=(0.99, 0.999)')

    # ---- Scheduler (with built-in LinearLR warmup via SequentialLR) ----
    sched_type = getattr(args, 'scheduler_type', 'step')
    
    # Stage 3 Polish Fix: the Cosine curve should span the newly requested epochs (e.g., 40),
    # not the massive combined absolute length (235).
    if getattr(args, 'no_augmentation', False):
        # We previously modified args.epoch to be start_epoch + original args.epoch
        # So the duration of the curve is exactly (args.epoch - start_epoch)
        main_epochs = args.epoch - start_epoch
        main_epochs -= warmup_epochs
    else:
        main_epochs = args.epoch - warmup_epochs
        
    if sched_type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=1e-6
        )
        logger.log_string(f'Scheduler: CosineAnnealingLR (T_max={main_epochs}, eta_min=1e-6)')
    elif sched_type == 'multistep':
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[80, 160], gamma=0.5
        )
        logger.log_string(f'Scheduler: MultiStepLR [80, 160] ×0.5')
    else:
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=80, gamma=0.5
        )
        logger.log_string(f'Scheduler: StepLR ×0.5 every 80 epochs')

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-5 / lr, end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
        logger.log_string(f'Warmup: LinearLR {warmup_epochs} epochs (1e-5 → {lr})')
    else:
        scheduler = main_scheduler

    # ---- SWA (Stochastic Weight Averaging) ----
    # Averages weights over final epochs for flatter minima → better generalization
    # Research: Izmailov et al., "Averaging Weights Leads to Wider Optima" (UAI 2018)
    swa_start_epoch = int(args.epoch * swa_start_frac)
    swa_model = AveragedModel(net)
    
    # FIX: Don't jump LR back up to 1e-5 if cosine already decayed to 1e-6
    swa_lr = 1e-6 if sched_type == 'cosine' else 1e-5
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=5, anneal_strategy='cos')
    
    swa_active = False
    logger.log_string(f'SWA: will activate at epoch {swa_start_epoch + 1} with lr={swa_lr}')

    # Restore optimizer/scheduler state on pretrain resume
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
            if start_epoch > 0:
                for _ in range(start_epoch):
                    scheduler.step()
                logger.log_string(f'Scheduler fast-forwarded {start_epoch} steps')

    # ---- Training Loop ----
    logger.log_string('\nStart training...')
    best_psnr = 0.0
    step = getattr(args, 'val_step', 10)  # Z4 audit: configurable validation frequency
    logger.log_string(f'Validation every {step} epochs')

    for epoch in range(start_epoch, args.epoch):
        # ---- SWA activation (Check before logging epoch) ----
        if epoch >= swa_start_epoch and not swa_active:
            swa_active = True
            logger.log_string(f'\n  ★ SWA activated (epoch {epoch + 1})')
            # Force optimizer to SWA LR immediately upon activation
            for param_group in optimizer.param_groups:
                param_group['lr'] = swa_lr

        current_lr = optimizer.param_groups[0]['lr']
        logger.log_string(f'\nEpoch {epoch + 1}/{args.epoch} '
                          f'[{stage.upper()}, mask={args.mlfim_mask_ratio}, '
                          f'lr={current_lr:.2e}]:')

        # Select loss function based on loss_type and warmup schedule
        if args.loss_type == 'l1':
            criterion = l1_criterion
        elif args.loss_type == 'charbonnier':
            criterion = charbonnier_criterion
        elif stage == 'pretrain':
            criterion = l1_criterion
        elif args.loss_warmup_epochs > 0 and epoch < args.loss_warmup_epochs:
            criterion = l1_criterion
            if epoch == 0:
                logger.log_string(f'  → Using L1 loss (warmup: epochs 1-{args.loss_warmup_epochs})')
        else:
            assert composite_criterion is not None, (
                "composite_criterion is None — use --loss_type l1 or charbonnier."
            )
            criterion = composite_criterion
            if epoch == args.loss_warmup_epochs and args.loss_warmup_epochs > 0:
                logger.log_string('  → Switching to composite loss')

        # EMA decay bump in final 25% of training
        # V3 FIX: Target 0.999 (was 0.9999, too aggressive for 0.997 base)
        if epoch >= int(args.epoch * 0.75):
            ema.decay = 0.999

        loss_train, psnr_train, ssim_train = train_one_epoch(
            train_loader, device, net, criterion, optimizer, args,
            stage, ema=ema, grad_accum_steps=grad_accum
        )

        logger.log_string(
            f'Train — loss: {loss_train:.5f}, '
            f'psnr: {psnr_train:.2f}, ssim: {ssim_train:.4f}'
        )

        # Fix #5: Only build state dict on save/val epochs to avoid unnecessary CPU copies
        need_save = (epoch + 1) % 20 == 0
        # Validate every `step` epochs + final epoch
        # Z1 FIX: use configurable `step` variable (was hardcoded 10)
        idx_e = epoch + 1
        need_val = (idx_e % step == 0) or (idx_e == args.epoch)
        state = None
        if need_save or need_val:
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
            if need_save:
                torch.save(state, save_path)

        if need_val:
            torch.cuda.empty_cache()  # Free training VRAM before validation

            # For first 5 epochs, skip EMA and use raw model weights.
            # EMA shadow at early epochs is a weighted avg from random init
            # (decay=0.997) which may not have converged, potentially
            # producing garbage predictions (observed 12 dB val PSNR).
            use_ema_for_val = (epoch >= 5)
            if use_ema_for_val:
                ema.apply_shadow(net)
                logger.log_string('  [VAL] Using EMA shadow weights')
            else:
                logger.log_string(f'  [VAL] Skipping EMA (epoch {epoch+1} < 5)')

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

            if use_ema_for_val:
                ema.restore(net)

        # ---- Scheduler step ----
        if swa_active:
            swa_model.update_parameters(net)
            swa_scheduler.step()
        else:
            scheduler.step()

    # ---- Post-training: update SWA batch norm if SWA was used ----
    if swa_active:
        logger.log_string('\nUpdating SWA batch norm statistics...')
        # SWA BN update needs the training loader
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Z5 FIX: Validate SWA model to see if it beats best EMA checkpoint
        logger.log_string('Validating SWA model...')
        swa_model.eval()
        swa_psnrs = []
        with torch.no_grad():
            for index, test_name in enumerate(test_Names):
                test_loader = test_Loaders[index]
                psnr_iter, ssim_iter, _ = run_validation_test(
                    test_loader, device, swa_model.module, args
                )
                swa_psnr_avg = float(np.array(psnr_iter).mean())
                swa_ssim_avg = float(np.array(ssim_iter).mean())
                swa_psnrs.append(swa_psnr_avg)
                logger.log_string(
                    f'  SWA Val {test_name}: '
                    f'psnr={swa_psnr_avg:.2f}, ssim={swa_ssim_avg:.3f}'
                )
        swa_aggregate_psnr = float(np.mean(swa_psnrs))
        logger.log_string(f'  SWA Aggregate PSNR: {swa_aggregate_psnr:.2f} dB '
                          f'(vs best EMA: {best_psnr:.2f} dB)')

        # Save SWA model
        swa_path = str(checkpoints_dir) + f'/{args.model_name}_{stage}_swa.pth'
        swa_state = {
            'epoch': args.epoch,
            'stage': stage,
            'state_dict': swa_model.module.state_dict(),
            'ema_state_dict': ema.state_dict(),
        }
        torch.save(swa_state, swa_path)
        logger.log_string(f'SWA model saved to {swa_path}')

        if swa_aggregate_psnr > best_psnr:
            best_psnr = swa_aggregate_psnr
            best_path = str(checkpoints_dir) + (
                f'/{args.model_name}_{stage}_best.pth'
            )
            torch.save(swa_state, best_path)
            logger.log_string(f'  ★ SWA is NEW BEST: {swa_aggregate_psnr:.2f} dB!')
        else:
            logger.log_string(f'  SWA did not beat EMA best ({best_psnr:.2f} dB)')

    logger.log_string(f'\n{"="*60}')
    logger.log_string(f'Stage {stage.upper()} complete! Best PSNR: {best_psnr:.2f} dB')
    logger.log_string(f'{"="*60}')


if __name__ == '__main__':
    main()
