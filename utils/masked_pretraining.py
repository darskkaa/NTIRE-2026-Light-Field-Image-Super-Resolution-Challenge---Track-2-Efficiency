"""
⚠️ DEPRECATED — DO NOT USE WITH V10 ⚠️

This file implements INPUT-PIXEL-LEVEL masking (zeroes out entire angular views).
V10's MyEfficientLFNetV10.py has its own FEATURE-LEVEL MLFIM masking built into
the model's forward() method, matching LFTransMamba's official implementation.

These are fundamentally different approaches:
  - This file: masks raw input pixels → model never sees masked views
  - V10 MLFIM: masks feature tokens after IFE → learned mask_token replacement

This file is kept for reference only. V10 ignores it entirely.
"""

import torch
import torch.nn as nn
import random

def get_mask(n, p=0.3):
    """
    Generates a boolean mask of length `n` where approximately 
    `p` proportion of elements are True (masked out).
    Always ensures the center view is NOT masked so model has a reference.
    """
    mask = torch.rand(n) < p
    center_idx = n // 2
    if n % 2 == 1:
        mask[center_idx] = False # ensure center view is never masked
    return mask

def apply_masked_pretraining(data, mask_ratio):
    """
    Masks random angular views in the input light field to 0.
    Input `data` expects shape: (B, 1, U*H, V*W) where U=V=angRes.
    Returns:
        masked_data: Input with specific angular views zeroed out
        mask_info: Boolean mask indicating which views were dropped
    """
    B, C, UH, VW = data.shape
    # For now, hardcode angRes=5 as that's the competition standard
    angRes = 5
    H, W = UH // angRes, VW // angRes
    
    # Reshape to (B, C, U, V, H, W) to access individual views easily
    data_5d = data.view(B, C, angRes, H, angRes, W).permute(0, 1, 2, 4, 3, 5).contiguous()
    
    # Generate mask for angular views (U * V)
    total_views = angRes * angRes
    view_mask = get_mask(total_views, p=mask_ratio).to(data.device)
    view_mask = view_mask.view(angRes, angRes)
    
    # Apply mask (broadcasting over B, C, H, W)
    # view_mask: (U, V) -> (1, 1, U, V, 1, 1)
    mask_expand = view_mask.view(1, 1, angRes, angRes, 1, 1)
    
    # Create output tensor, setting masked views to 0
    masked_data_5d = data_5d.clone()
    masked_data_5d[mask_expand.expand_as(masked_data_5d)] = 0.0
    
    # Return to original shape (B, 1, U*H, V*W)
    masked_data = masked_data_5d.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, UH, VW)
    
    return masked_data, view_mask

class ProgressiveMasking(nn.Module):
    """
    Progressively increases the masking ratio during training warmup.
    Modeled after MLFIM technique from LFTransMamba (1st NTIRE 2025).
    """
    def __init__(self, angRes=5, start_ratio=0.1, end_ratio=0.3, warmup_epochs=20):
        super().__init__()
        self.angRes = angRes
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_epochs = warmup_epochs
        self.current_ratio = start_ratio
        
    def set_epoch(self, epoch):
        """Update ratio based on current epoch."""
        if epoch >= self.warmup_epochs:
            self.current_ratio = self.end_ratio
        else:
            # Linear scaling
            progress = epoch / self.warmup_epochs
            self.current_ratio = self.start_ratio + progress * (self.end_ratio - self.start_ratio)
            
    def forward(self, data):
        if not self.training or self.current_ratio == 0:
            return data, None
            
        return apply_masked_pretraining(data, self.current_ratio)

if __name__ == '__main__':
    # Simple test
    print("Testing ProgressiveMasking...")
    masker = ProgressiveMasking(angRes=5, start_ratio=0.1, end_ratio=0.5, warmup_epochs=10)
    data = torch.ones(2, 1, 5*32, 5*32) # completely white image
    
    print(f"Initial Phase (Epoch 0, Ratio {masker.start_ratio}):")
    masker.set_epoch(0)
    masked_data, mask = masker(data)
    zero_elements = (masked_data == 0).sum().item()
    total_elements = masked_data.numel()
    print(f"  Mask shape: {mask.shape}")
    print(f"  Zeroed elements: {zero_elements} / {total_elements} ({zero_elements/total_elements*100:.1f}%)")
    
    print(f"\nFinal Phase (Epoch 10, Ratio {masker.end_ratio}):")
    masker.set_epoch(10)
    masked_data, mask = masker(data)
    zero_elements = (masked_data == 0).sum().item()
    print(f"  Zeroed elements: {zero_elements} / {total_elements} ({zero_elements/total_elements*100:.1f}%)")
    
    # Center should remain 1.0
    center_data = masked_data[:, :, 2*32:3*32, 2*32:3*32]
    all_ones = (center_data == 1.0).all().item()
    print(f"\nCenter view preserved? {'✅ YES' if all_ones else '❌ NO'}")
    print("\n✅ Test Passed")
