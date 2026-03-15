#!/usr/bin/env python3
"""
Stochastic Weight Averaging (SWA) for MyEfficientLFNetV3_MLFIM checkpoints.

Averages the model state_dict weights from the last N finetune checkpoints
to produce a smoother, more generalizable model. This is a zero-cost
inference-time optimization (only affects weights, not architecture).

Usage:
    python average_checkpoints.py \
        --ckpt_dir /path/to/checkpoints \
        --output /path/to/output_SWA.pth \
        --last_n 5

The script will automatically find finetune checkpoints, sort them by epoch,
and average the last N. By default it averages the last 5 (epochs 120-200).
"""

import argparse
import glob
import os
import re
import sys
import torch
from collections import OrderedDict


def find_finetune_checkpoints(ckpt_dir):
    """Find all finetune epoch checkpoints (not 'best') and sort by epoch number."""
    pattern = os.path.join(ckpt_dir, '*finetune*epoch*model.pth')
    files = glob.glob(pattern)
    
    if not files:
        print(f"[ERROR] No finetune epoch checkpoints found in: {ckpt_dir}")
        print(f"  Searched pattern: {pattern}")
        sys.exit(1)
    
    # Extract epoch number from filename
    epoch_file_pairs = []
    for f in files:
        match = re.search(r'epoch_(\d+)_model\.pth', os.path.basename(f))
        if match:
            epoch = int(match.group(1))
            epoch_file_pairs.append((epoch, f))
    
    # Sort by epoch
    epoch_file_pairs.sort(key=lambda x: x[0])
    return epoch_file_pairs


def average_checkpoints(ckpt_paths, output_path):
    """Average model state dicts from multiple checkpoint files."""
    print(f"\n{'='*60}")
    print(f"  Stochastic Weight Averaging (SWA)")
    print(f"  Averaging {len(ckpt_paths)} checkpoints")
    print(f"{'='*60}\n")
    
    avg_state_dict = None
    n = len(ckpt_paths)
    
    for i, path in enumerate(ckpt_paths):
        print(f"  [{i+1}/{n}] Loading: {os.path.basename(path)}")
        ckpt = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        
        if avg_state_dict is None:
            # Initialize with first checkpoint
            avg_state_dict = OrderedDict()
            for key, value in state_dict.items():
                avg_state_dict[key] = value.float().clone()
        else:
            # Accumulate
            for key, value in state_dict.items():
                if key in avg_state_dict:
                    avg_state_dict[key] += value.float()
                else:
                    print(f"  [WARN] Key '{key}' not in first checkpoint, skipping")
    
    # Divide by N to get the average
    for key in avg_state_dict:
        avg_state_dict[key] /= float(n)
    
    # Save as a clean state_dict (compatible with generate_codabench_submission_v3.py)
    output_ckpt = {
        'state_dict': avg_state_dict,
        'swa_info': {
            'num_checkpoints': n,
            'source_files': [os.path.basename(p) for p in ckpt_paths],
        }
    }
    
    torch.save(output_ckpt, output_path)
    print(f"\n  ✅ SWA checkpoint saved to: {output_path}")
    print(f"  Averaged {n} checkpoints")
    print(f"{'='*60}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Average finetune checkpoints (SWA)')
    parser.add_argument('--ckpt_dir', type=str, 
                        default=r'C:\Users\darkz\Downloads\MyEfficientLFNetV3_MLFIM\MyEfficientLFNetV3_MLFIM\checkpoints',
                        help='Directory containing finetune checkpoint .pth files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for averaged checkpoint (default: same dir as ckpt_dir)')
    parser.add_argument('--last_n', type=int, default=5,
                        help='Number of last checkpoints to average (default: 5)')
    parser.add_argument('--epochs', type=str, default=None,
                        help='Comma-separated list of specific epochs to average (e.g., "120,140,160,180,200")')
    args = parser.parse_args()
    
    # Find checkpoints
    all_ckpts = find_finetune_checkpoints(args.ckpt_dir)
    
    print(f"\nFound {len(all_ckpts)} finetune epoch checkpoints:")
    for epoch, path in all_ckpts:
        print(f"  Epoch {epoch:3d}: {os.path.basename(path)}")
    
    # Select checkpoints to average
    if args.epochs:
        # Use specific epochs
        target_epochs = set(int(e.strip()) for e in args.epochs.split(','))
        selected = [(e, p) for e, p in all_ckpts if e in target_epochs]
        if len(selected) != len(target_epochs):
            found_epochs = {e for e, _ in selected}
            missing = target_epochs - found_epochs
            print(f"\n[WARN] Missing epochs: {missing}")
    else:
        # Use last N
        selected = all_ckpts[-args.last_n:]
    
    if len(selected) < 2:
        print(f"\n[ERROR] Need at least 2 checkpoints to average, found {len(selected)}")
        sys.exit(1)
    
    print(f"\nSelected {len(selected)} checkpoints for averaging:")
    for epoch, path in selected:
        print(f"  Epoch {epoch:3d}: {os.path.basename(path)}")
    
    # Set output path
    if args.output is None:
        epochs_str = '_'.join(str(e) for e, _ in selected)
        args.output = os.path.join(args.ckpt_dir, 
                                    f'MyEfficientLFNetV3_MLFIM_finetune_SWA_{epochs_str}.pth')
    
    # Average
    average_checkpoints([p for _, p in selected], args.output)


if __name__ == '__main__':
    main()
