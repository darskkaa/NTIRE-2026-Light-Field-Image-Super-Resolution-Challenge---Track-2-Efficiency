"""
train_v6_final.py — Combined Pretrain + Finetune for MyEfficientLFNetV6_Final
=============================================================================
Single script that runs Stage 1 (MLFIM pretrain) → Stage 2 (finetune) → submit.

LR SCHEDULE AUDIT:
  Stage 1 (pretrain, 100 epochs):
    - Adam(lr=2e-4)  ← LFMamba default
    - StepLR(step_size=15, gamma=0.5)
    - LR decomposition:
        Epoch  1-15:  2.0e-4
        Epoch 16-30:  1.0e-4
        Epoch 31-45:  5.0e-5
        Epoch 46-60:  2.5e-5
        Epoch 61-75:  1.25e-5
        Epoch 76-90:  6.25e-6
        Epoch 91-100: 3.125e-6
    - scheduler.step() called ONCE per epoch AFTER training loop ✓

  Stage 2 (finetune, 150 epochs):
    - Adam(lr=1e-4)  ← halved from pretrain
    - StepLR(step_size=15, gamma=0.5)
    - LR decomposition:
        Epoch   1-15:  1.0e-4
        Epoch  16-30:  5.0e-5
        Epoch  31-45:  2.5e-5
        Epoch  46-60:  1.25e-5
        Epoch  61-75:  6.25e-6
        Epoch  76-90:  3.125e-6
        Epoch  91-105: 1.5625e-6
        Epoch 106-120: 7.8125e-7
        Epoch 121-135: 3.906e-7
        Epoch 136-150: 1.953e-7
    - scheduler.step() called ONCE per epoch AFTER training loop ✓

  RESUME LR AUDIT:
    - If scheduler_state_dict is in checkpoint → restore it directly
    - If NOT (e.g. periodic checkpoint) → manually step scheduler N times
      to reconstruct the correct LR for that epoch
    - Example: resuming from epoch 30 without scheduler state:
        scheduler.step() called 30 times → LR = 2e-4 * 0.5^(30//15) = 5e-5 ✓

SHAPE AUDIT:
  - train data: (B, 1, angRes*patch, angRes*patch) for both LR and HR
  - data_info: [angRes_in, angRes_out] = [5, 5]
  - model input: (B, 1, 5*32, 5*32) = (B, 1, 160, 160) for patch_size=32
  - model output: (B, 1, 5*128, 5*128) = (B, 1, 640, 640) for scale=4

Usage:
  python train_v6_final.py                          # Both stages
  python train_v6_final.py --stage pretrain          # Pretrain only
  python train_v6_final.py --stage finetune --path_pre_pth <ckpt>
  python train_v6_final.py --stage pretrain --resume_ckpt <ckpt>  # Resume
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys, time, argparse, importlib, glob, re, traceback
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from einops import rearrange

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TeeLogger:
    """Tee stdout/stderr to both console and a log file.
    Ensures training output is never lost if SSH drops."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, 'a', buffering=1)  # line-buffered
        self.log.write(f"\n{'='*60}\n")
        self.log.write(f"Log started: {datetime.now().isoformat()}\n")
        self.log.write(f"{'='*60}\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def parse_args():
    parser = argparse.ArgumentParser("V6 Final Training")
    parser.add_argument('--model_name', type=str, default='MyEfficientLFNetV6_Final')
    parser.add_argument('--stage', type=str, default='both',
                        choices=['pretrain', 'finetune', 'both'])
    parser.add_argument('--angRes', type=int, default=5)
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Base LR for pretrain. Finetune uses lr/2.')
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=150)
    parser.add_argument('--mlfim_mask_ratio', type=float, default=0.25)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--path_for_train', type=str, default='./data_for_training/')
    parser.add_argument('--path_for_test', type=str, default='./data_for_test/')
    parser.add_argument('--data_name', type=str, default='ALL')
    parser.add_argument('--path_log', type=str, default='./log/')
    parser.add_argument('--path_pre_pth', type=str, default=None,
                        help='Checkpoint path for finetune stage')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Checkpoint to resume training from (restores epoch, optimizer, scheduler)')
    parser.add_argument('--task', type=str, default='SR')
    parser.add_argument('--generate_submission', action='store_true',
                        help='Generate CodaBench submission after training')
    args, _ = parser.parse_known_args()

    # Derived attributes expected by data loaders and model
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 1
    return args


def import_model(model_name):
    """Dynamically import model module from model/SR/<name>.py."""
    return importlib.import_module(f'model.SR.{model_name}')


def create_dataloaders(args):
    """
    Create train + test data loaders.

    AUDIT: AUG_CONFIG is set BEFORE creating loaders.
    Both CutBlur and MixUp are disabled — LFMamba uses only geometric aug.

    BUG FIX (P0-A): MultiTestSetDataLoader returns 3 values, not 2.
    BUG FIX (P0-B): TrainSetDataLoader is a Dataset — must wrap in DataLoader
                     for batching, shuffling, and correct tensor dimensions.
    """
    from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader

    # Disable all fancy augmentations — LFMamba recipe
    try:
        from utils.utils_datasets import AUG_CONFIG
        AUG_CONFIG['cutblur_prob'] = 0.0
        AUG_CONFIG['mixup_prob'] = 0.0
    except ImportError:
        pass

    # P0-B FIX: TrainSetDataLoader is a Dataset, NOT a DataLoader.
    # Must wrap in DataLoader for: batching, shuffling, multi-worker prefetch,
    # and correct tensor dimensions (adds batch dim via collation).
    train_dataset = TrainSetDataLoader(args)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    print(f"  Train loader: {len(train_dataset)} samples, "
          f"{len(train_loader)} batches/epoch (batch_size={args.batch_size}, "
          f"num_workers={args.num_workers})")

    # P0-A FIX: MultiTestSetDataLoader returns 3 values (names, loaders, length)
    test_names, test_loaders, _ = MultiTestSetDataLoader(args)
    return train_loader, test_names, test_loaders


def cal_psnr(label, out, angRes):
    """
    Per-view PSNR calculation.

    AUDIT:
      - label, out: (B, 1, angRes*H, angRes*W) — SAI format
      - Rearranged to (B, 1, U, H, V, W) for per-view extraction
      - PSNR computed on Y channel with data_range=1.0
    """
    from skimage.metrics import peak_signal_noise_ratio
    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w',
                          a1=angRes, a2=angRes)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w',
                        a1=angRes, a2=angRes)
    B, C, U, h, V, w = label.size()
    label_y = label[:, 0].data.cpu().numpy()
    out_y = out[:, 0].data.cpu().numpy()

    psnr_list = []
    for b in range(B):
        for u in range(U):
            for v in range(V):
                psnr_list.append(peak_signal_noise_ratio(
                    label_y[b, u, :, v, :], out_y[b, u, :, v, :],
                    data_range=1.0))
    return float(np.mean(psnr_list))


def vram_report(label=""):
    """Print VRAM usage stats. Safe to call even without CUDA."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  VRAM {label}: alloc={allocated:.0f}MB reserved={reserved:.0f}MB peak={max_alloc:.0f}MB")


@torch.no_grad()
def validate(net, test_loaders, test_names, args, device):
    """Run validation and return average PSNR across all test sets.
    
    Wrapped with memory cleanup to prevent OOM during validation.
    """
    net.eval()
    # Clear CUDA cache before validation to free training buffers
    torch.cuda.empty_cache()
    
    psnr_all = []
    for name, loader in zip(test_names, test_loaders):
        psnrs = []
        for data in loader:
            lr_data = data[0].to(device)
            hr_data = data[1].to(device)
            data_info = [args.angRes_in, args.angRes_out]

            # AUDIT: model returns (B, 1, angRes*H_sr, angRes*W_sr)
            sr = net(lr_data, data_info)
            sr = sr.clamp(0, 1)

            # AUDIT: hr_data shape must match sr shape
            assert sr.shape == hr_data.shape, \
                f"Shape mismatch! SR: {sr.shape}, HR: {hr_data.shape}"

            psnrs.append(cal_psnr(hr_data, sr, args.angRes_in))
            
            # Free validation tensors immediately
            del lr_data, hr_data, sr
        
        avg = float(np.mean(psnrs))
        psnr_all.append(avg)
        print(f"  {name}: {avg:.4f} dB")

    overall = float(np.mean(psnr_all))
    print(f"  AVERAGE: {overall:.4f} dB")
    
    # Cleanup after validation
    torch.cuda.empty_cache()
    net.train()
    return overall


def save_checkpoint(ckpt_dir, model_name, stage, epoch, net, optimizer, scheduler,
                    psnr=None, is_best=False, suffix=None):
    """Save a FULL checkpoint with all state needed for resume."""
    if suffix:
        fname = f'{model_name}_{stage}_{suffix}.pth'
    elif is_best:
        fname = f'{model_name}_{stage}_best.pth'
    else:
        fname = f'{model_name}_{stage}_epoch_{epoch:03d}.pth'
    
    save_path = Path(ckpt_dir) / fname
    ckpt = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'stage': stage,
        'lr': optimizer.param_groups[0]['lr'],
    }
    if psnr is not None:
        ckpt['psnr'] = psnr
    torch.save(ckpt, save_path)
    return str(save_path)


def train_one_stage(args, model_module, stage, device, pretrain_ckpt=None, resume_ckpt=None):
    """
    Train one stage (pretrain or finetune).

    LR AUDIT:
      - pretrain: lr=args.lr (default 2e-4)
      - finetune: lr=args.lr / 2 (default 1e-4)
      - StepLR: step_size=15, gamma=0.5
      - scheduler.step() called ONCE per epoch at end of training loop
      - NOT inside the batch loop (this was a bug in previous versions)

    RESUME LR AUDIT:
      When resuming from a checkpoint:
      1. If checkpoint has scheduler_state_dict → restore it directly
         (LR is automatically correct because state encodes last_epoch)
      2. If checkpoint has NO scheduler_state_dict (e.g. old periodic ckpt) →
         manually call scheduler.step() N times where N = checkpoint epoch.
         This reconstructs the correct LR decay schedule.
      
      Example: Resume pretrain from epoch 30 (no scheduler state):
        Initial LR = 2e-4
        scheduler.step() called 30 times → last_epoch=30
        LR = 2e-4 * 0.5^floor(30/15) = 2e-4 * 0.25 = 5e-5 ✓
    """
    epochs = args.pretrain_epochs if stage == 'pretrain' else args.finetune_epochs

    # LR AUDIT: pretrain=2e-4, finetune=1e-4 (halved)
    if stage == 'pretrain':
        lr = args.lr           # 2e-4
        args.mlfim_mask_ratio = 0.25
    else:
        lr = args.lr / 2.0     # 1e-4
        args.mlfim_mask_ratio = 0.0  # No masking for finetune

    print(f"\n{'='*60}")
    print(f"STAGE: {stage.upper()}")
    print(f"  Epochs:     {epochs}")
    print(f"  Init LR:    {lr:.1e}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Mask ratio: {args.mlfim_mask_ratio}")
    print(f"  Workers:    {args.num_workers}")
    print(f"  Scheduler:  StepLR(step=15, gamma=0.5)")
    if resume_ckpt:
        print(f"  RESUMING:   {resume_ckpt}")
    print(f"{'='*60}\n")

    # Print LR decay schedule
    print("  LR Schedule:")
    _lr = lr
    for i in range(1, epochs + 1):
        if i % 15 == 1 or i == 1:
            print(f"    Epoch {i:3d}-{min(i+14, epochs):3d}: {_lr:.2e}")
        if i % 15 == 0:
            _lr *= 0.5

    # Create model
    # AUDIT: args.mlfim_mask_ratio is already set above before model creation
    net = model_module.get_model(args).to(device)
    criterion = nn.L1Loss()

    # AUDIT: optimizer gets lr from the stage-specific calculation above
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # AUDIT: StepLR steps epoch counter, gamma=0.5 halves LR every 15 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    start_epoch = 1
    best_psnr = 0.0
    best_epoch = 0

    # ===== RESUME FROM CHECKPOINT =====
    if resume_ckpt is not None:
        print(f"\n  RESUMING from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        state_dict = ckpt.get('state_dict', ckpt)
        # Clean module. prefix (from DDP training)
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            cleaned[k.replace('module.', '')] = v
        result = net.load_state_dict(cleaned, strict=False)
        print(f"    Loaded {len(cleaned)} params")
        if result.missing_keys:
            print(f"    Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"    Unexpected keys: {result.unexpected_keys[:5]}...")

        # Restore epoch counter FIRST (needed for scheduler fallback)
        resume_epoch = ckpt.get('epoch', 0)
        start_epoch = resume_epoch + 1
        print(f"    Checkpoint epoch: {resume_epoch}")
        print(f"    Will resume from epoch: {start_epoch}")

        # Restore optimizer state if available
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"    ✓ Restored optimizer state (momentum buffers intact)")
        else:
            print(f"    ⚠ No optimizer state in checkpoint — fresh Adam momentum")

        # Restore scheduler state — CRITICAL for correct LR
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"    ✓ Restored scheduler state (last_epoch={scheduler.last_epoch})")
        else:
            # FALLBACK: manually step scheduler to reconstruct correct LR
            # Without this, LR would reset to initial value (e.g. 2e-4 instead of 5e-5)
            print(f"    ⚠ No scheduler state — reconstructing LR by stepping {resume_epoch} times")
            for _step in range(resume_epoch):
                scheduler.step()
            print(f"    ✓ Scheduler reconstructed: last_epoch={scheduler.last_epoch}")

        # Restore best PSNR
        if 'psnr' in ckpt:
            best_psnr = ckpt['psnr']
            best_epoch = resume_epoch
            print(f"    Best PSNR so far: {best_psnr:.4f} dB @ epoch {best_epoch}")

        current_lr = optimizer.param_groups[0]['lr']
        expected_lr = lr * (0.5 ** (resume_epoch // 15))
        print(f"\n    === LR VERIFICATION ===")
        print(f"    Current LR:  {current_lr:.2e}")
        print(f"    Expected LR: {expected_lr:.2e} (lr={lr:.1e} * 0.5^{resume_epoch // 15})")
        if abs(current_lr - expected_lr) / expected_lr > 0.01:
            print(f"    ⚠ LR MISMATCH! This may indicate a scheduler restore bug.")
        else:
            print(f"    ✓ LR matches expected value")
        print(f"    === RESUME COMPLETE ===\n")

    # Load pretrain checkpoint for finetune stage (only if NOT resuming)
    elif pretrain_ckpt is not None:
        print(f"\nLoading pretrain checkpoint: {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location=device)
        state_dict = ckpt.get('state_dict', ckpt)
        # Clean module. prefix (from DDP training)
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            cleaned[k.replace('module.', '')] = v
        result = net.load_state_dict(cleaned, strict=False)
        print(f"  Loaded {len(cleaned)} params")
        if result.missing_keys:
            print(f"  Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  Unexpected keys: {result.unexpected_keys[:5]}...")

    # Parameter count audit
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")
    assert total_params < 1_000_000, f"OVER 1M BUDGET: {total_params:,} params!"

    # === LR FINAL CHECK ===
    actual_lr = optimizer.param_groups[0]['lr']
    print(f"\n  LR FINAL CHECK:")
    print(f"    optimizer LR = {actual_lr:.2e}")
    print(f"    Scheduler last_epoch: {scheduler.last_epoch}")
    print(f"    Starting from epoch:  {start_epoch}")

    # Data loaders
    train_loader, test_names, test_loaders = create_dataloaders(args)

    # Checkpoint directory
    log_dir = Path(args.path_log) / f'SR_{args.angRes}x{args.angRes}_{args.scale_factor}x'
    log_dir = log_dir / args.data_name / args.model_name
    ckpt_dir = log_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vram_report("before training")

    for epoch in range(start_epoch, epochs + 1):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        total_batches = len(train_loader)
        grad_norm_sum = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch}/{epochs} START [{datetime.now().strftime('%H:%M:%S')}] LR={current_lr:.2e} ---")

        for batch_idx, data in enumerate(train_loader):
            lr_data = data[0].to(device)
            hr_data = data[1].to(device)
            data_info = [args.angRes_in, args.angRes_out]

            # Forward pass
            sr = net(lr_data, data_info)

            # AUDIT: sr and hr_data must have the same shape
            assert sr.shape == hr_data.shape, \
                f"Train shape mismatch! SR: {sr.shape}, HR: {hr_data.shape}"

            loss = criterion(sr, hr_data)

            # NaN/Inf detection — save emergency checkpoint and bail
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  ‼ FATAL: Loss is {loss.item()} at epoch {epoch} batch {batch_idx}!")
                print(f"    Saving emergency checkpoint...")
                save_checkpoint(ckpt_dir, args.model_name, stage, epoch, net,
                                optimizer, scheduler, psnr=best_psnr, suffix='emergency_nan')
                print(f"    ABORTING training due to NaN/Inf loss.")
                raise RuntimeError(f"NaN/Inf loss at epoch {epoch} batch {batch_idx}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents explosion with Mamba SSMs)
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            grad_norm_sum += grad_norm.item()

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

            # Free forward pass intermediates
            del sr, lr_data, hr_data, loss

            # Print progress only at end of epoch (once per epoch)
            if batch_idx + 1 == total_batches:
                elapsed_so_far = time.time() - t0
                speed_b = n_batches / elapsed_so_far if elapsed_so_far > 0 else 0
                
                # Calculate ETA for the rest of the STAGE (not just this epoch)
                remaining_epochs = epochs - epoch
                time_per_epoch = elapsed_so_far
                eta_seconds = (remaining_epochs * time_per_epoch)
                eta_hours = int(eta_seconds // 3600)
                eta_mins = int((eta_seconds % 3600) // 60)
                
                avg_loss_so_far = epoch_loss / n_batches
                avg_gnorm_so_far = grad_norm_sum / n_batches
                vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                print(f"  E{epoch:03d} [{n_batches}/{total_batches}] "
                      f"loss={avg_loss_so_far:.5f} |g|={avg_gnorm_so_far:.2f} "
                      f"lr={current_lr:.1e} | "
                      f"{speed_b:.1f} batch/s | "
                      f"Epoch Time: {time_per_epoch:.0f}s | "
                      f"Stage ETA: {eta_hours}h{eta_mins}m | "
                      f"VRAM={vram_mb:.0f}MB")

        # AUDIT: scheduler.step() called ONCE per epoch, AFTER training loop
        # This is critical — calling inside batch loop causes LR to decay
        # n_batches times per epoch instead of once
        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        avg_gnorm = grad_norm_sum / max(n_batches, 1)
        speed = n_batches * args.batch_size / elapsed  # samples/sec

        print(f"--- Epoch {epoch:3d}/{epochs} DONE [{datetime.now().strftime('%H:%M:%S')}] | "
              f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | |g|: {avg_gnorm:.2f} | "
              f"{speed:.1f} img/s | {elapsed:.0f}s ---")

        # Validate: every 5 epochs, every epoch in last 20, or first epoch
        do_validate = (epoch % 5 == 0 or epoch > epochs - 20 or epoch == 1)

        if do_validate:
            try:
                psnr = validate(net, test_loaders, test_names, args, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  ⚠ Validation OOM! Clearing cache and continuing training.")
                    torch.cuda.empty_cache()
                    psnr = best_psnr  # Use last known best
                else:
                    raise  # Re-raise non-OOM errors

            if psnr > best_psnr:
                best_psnr = psnr
                best_epoch = epoch
                save_checkpoint(ckpt_dir, args.model_name, stage, epoch, net,
                                optimizer, scheduler, psnr=psnr, is_best=True)
                print(f"  ** New best: {psnr:.4f} dB @ epoch {epoch} **")

        # Save periodic checkpoint every 10 epochs (with FULL state for resume)
        if epoch % 10 == 0:
            save_checkpoint(ckpt_dir, args.model_name, stage, epoch, net,
                            optimizer, scheduler, psnr=best_psnr)
            print(f"  Checkpoint saved: epoch {epoch}")

        # Memory cleanup every epoch
        torch.cuda.empty_cache()

    # Always save final epoch
    save_checkpoint(ckpt_dir, args.model_name, stage, epochs, net,
                    optimizer, scheduler, psnr=best_psnr, suffix='final')

    print(f"\n{'='*60}")
    print(f"Stage {stage.upper()} complete!")
    print(f"  Best PSNR: {best_psnr:.4f} dB @ epoch {best_epoch}")
    print(f"  Checkpoints in: {ckpt_dir}")
    print(f"{'='*60}")

    best_ckpt = ckpt_dir / f'{args.model_name}_{stage}_best.pth'
    return str(best_ckpt) if best_ckpt.exists() else str(ckpt_dir / f'{args.model_name}_{stage}_final.pth')


def swa_average(ckpt_dir, model_name, stage='finetune', n_last=10):
    """
    Stochastic Weight Averaging: average the last N checkpoints.

    AUDIT:
      - Finds all epoch_*.pth files for the given stage
      - Sorts by epoch number (extracted from filename)
      - Averages state_dict values (FP32)
      - Returns averaged state_dict

    This squeezes extra performance without any training cost.
    """
    import re
    pattern = str(Path(ckpt_dir) / f'{model_name}_{stage}_epoch_*.pth')
    ckpt_files = glob.glob(pattern)

    if len(ckpt_files) < 2:
        print(f"SWA: Only {len(ckpt_files)} checkpoints, skipping SWA")
        return None

    # Sort by epoch number
    def extract_epoch(path):
        m = re.search(r'epoch_(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else 0
    ckpt_files.sort(key=extract_epoch)

    # Take last N
    to_average = ckpt_files[-n_last:]
    print(f"\nSWA: Averaging {len(to_average)} checkpoints:")
    for p in to_average:
        print(f"  - {os.path.basename(p)}")

    avg_sd = OrderedDict()
    for path in to_average:
        ckpt = torch.load(path, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        for k, v in sd.items():
            if k in avg_sd:
                avg_sd[k] += v.float()
            else:
                avg_sd[k] = v.float().clone()

    for k in avg_sd:
        avg_sd[k] /= float(len(to_average))

    # Save SWA checkpoint
    swa_path = str(Path(ckpt_dir) / f'{model_name}_{stage}_swa.pth')
    torch.save({'state_dict': avg_sd, 'stage': stage}, swa_path)
    print(f"SWA checkpoint saved: {swa_path}")
    return swa_path


def main():
    args = parse_args()

    # Setup file logging — tee all output to a timestamped log file
    log_dir = Path(args.path_log) / f'SR_{args.angRes}x{args.angRes}_{args.scale_factor}x'
    log_dir = log_dir / args.data_name / args.model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'train_{args.stage}_{timestamp}.log'
    tee = TeeLogger(str(log_file))
    sys.stdout = tee
    sys.stderr = tee
    print(f"Logging to: {log_file}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Args: {vars(args)}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Set memory fraction to leave headroom
        torch.cuda.empty_cache()

    model_module = import_model(args.model_name)

    # Verify model before training
    print("\n--- Model Verification ---")
    test_args = argparse.Namespace(**vars(args))
    test_args.mlfim_mask_ratio = 0.0
    test_model = model_module.get_model(test_args).to(device)
    params = sum(p.numel() for p in test_model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.3f}M)")
    assert params < 1_000_000, f"OVER BUDGET: {params:,} > 1M"
    # Quick forward test (must be on CUDA — Mamba kernels require it)
    test_x = torch.randn(1, 1, args.angRes * 32, args.angRes * 32).to(device)
    with torch.no_grad():
        test_y = test_model(test_x, [args.angRes, args.angRes])
    expected = (1, 1, args.angRes * 32 * args.scale_factor,
                args.angRes * 32 * args.scale_factor)
    assert test_y.shape == torch.Size(expected), \
        f"Shape error! Got {test_y.shape}, expected {expected}"
    print(f"Forward pass: {test_x.shape} → {test_y.shape} ✓")
    del test_model, test_x, test_y
    torch.cuda.empty_cache()
    print("--- Verification passed ---\n")

    pretrain_ckpt = None
    swa_ckpt = None
    final_ckpt = None

    try:
        # Stage 1: Pretrain with MLFIM masking
        if args.stage in ('pretrain', 'both'):
            # Check if we are resuming pretrain
            resume = args.resume_ckpt if args.stage == 'pretrain' else None
            pretrain_ckpt = train_one_stage(
                args, model_module, 'pretrain', device,
                resume_ckpt=resume)

        # Stage 2: Finetune without masking
        if args.stage in ('finetune', 'both'):
            finetune_ckpt = pretrain_ckpt or args.path_pre_pth
            if finetune_ckpt is None:
                print("ERROR: No pretrain checkpoint for finetune!")
                print("Run: python train_v6_final.py --stage pretrain")
                print("Or:  python train_v6_final.py --stage finetune --path_pre_pth <ckpt>")
                sys.exit(1)

            # Check if we are resuming finetune
            resume = args.resume_ckpt if args.stage == 'finetune' else None
            final_ckpt = train_one_stage(
                args, model_module, 'finetune', device,
                pretrain_ckpt=finetune_ckpt,
                resume_ckpt=resume)

            # SWA post-processing
            log_dir = Path(args.path_log) / f'SR_{args.angRes}x{args.angRes}_{args.scale_factor}x'
            ckpt_dir = log_dir / args.data_name / args.model_name / 'checkpoints'
            swa_ckpt = swa_average(str(ckpt_dir), args.model_name,
                                   stage='finetune', n_last=10)

        # Generate submission if requested
        if args.generate_submission:
            print("\n\n=== Generating CodaBench Submission ===")
            # Prefer SWA > best > final
            ckpt_to_use = swa_ckpt or final_ckpt
            print(f"Using checkpoint: {ckpt_to_use}")
            os.system(f"python generate_codabench_submission_v6.py --ckpt \"{ckpt_to_use}\"")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user (Ctrl+C)")
        print("Checkpoints have been saved periodically — use --resume_ckpt to continue.")
    except Exception as e:
        print(f"\n\n‼ FATAL ERROR: {type(e).__name__}: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        print("Checkpoints have been saved periodically — use --resume_ckpt to continue.")
        raise

    print(f"\n=== Pipeline Complete [{datetime.now().isoformat()}] ===")


if __name__ == '__main__':
    main()
