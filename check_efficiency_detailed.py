#!/usr/bin/env python
"""
NTIRE 2026 Track 2 - Extreme Efficiency Analysis (Detailed)
Run: python check_efficiency_detailed.py
"""

import torch
import sys
import os
import torch.nn as nn
sys.path.insert(0, os.getcwd())

print("="*80)
print("🔬 NTIRE 2026 Track 2 - EXTREME EFFICIENCY ANALYSIS")
print("="*80)

# ============================================================================
# LOAD MODEL
# ============================================================================
class Args:
    angRes_in = 5
    scale_factor = 4
    use_macpi = True
    use_tta = False

try:
    from model.SR.MyEfficientLFNetV6_2 import get_model, count_parameters
    print("✓ Loaded MyEfficientLFNetV6_2 (SOTA)")
except ImportError:
    try:
        from model.SR.MyEfficientLFNetV6_1 import get_model, count_parameters
        print("✓ Loaded MyEfficientLFNetV6_1")
    except ImportError:
        try:
            from model.SR.MyEfficientLFNetV5 import get_model, count_parameters
            print("✓ Loaded MyEfficientLFNetV5")
        except ImportError:
            print("❌ Could not load model file. Make sure you are in the project root.")
            sys.exit(1)

model = get_model(Args())
model.eval()

# ============================================================================
# 1. PARAMETER ANALYSIS (LIMIT: 1,000,000)
# ============================================================================
print("\n" + "="*80)
print("📊 1. PARAMETER ANALYSIS (Rule: < 1M params)")
print("="*80)

total_params = 0
trainable_params = 0
layer_params = {}

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
    
    # Group by module
    module_name = name.split('.')[0]
    if module_name not in layer_params:
        layer_params[module_name] = 0
    layer_params[module_name] += param.numel()

print(f"\n{'Module':<30} {'Params':>15} {'%':>8}")
print("-"*55)
for name, params in sorted(layer_params.items(), key=lambda x: -x[1]):
    pct = params/total_params*100
    print(f"{name:<30} {params:>15,} {pct:>7.2f}%")

print("-"*55)
print(f"{'TOTAL':<30} {total_params:>15,} {100:>7.1f}%")
print(f"\n✓ Trainable: {trainable_params:,}")
print(f"✓ Budget Used: {total_params/1_000_000*100:.2f}%")
print(f"✓ Headroom: {1_000_000 - total_params:,} params available")
print(f"{'✅ PASS' if total_params < 1_000_000 else '❌ FAIL'}: {'<1M' if total_params < 1_000_000 else '>1M'}")

# ============================================================================
# 2. FLOPS ANALYSIS (LIMIT: 20G for 5x5x32x32 input)
# ============================================================================
print("\n" + "="*80)
print("📈 2. FLOPS ANALYSIS (Rule: < 20G FLOPs on 5x5x32x32 input)")
print("="*80)

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    # NTIRE spec: 5x5 angular, 32x32 spatial = 160x160 input tensor
    # Input shape: [1, 1, 5*32, 5*32] = [1, 1, 160, 160]
    x = torch.randn(1, 1, 160, 160)
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        
    flops = FlopCountAnalysis(model, x)
    total_flops = flops.total()
    
    print(f"\n📐 Input Shape: {x.shape} (represents 5x5 views of 32x32 patches)")
    print(f"📐 Output Shape: [1, 1, 640, 640] (4× upscale)")
    print(f"\n{'='*60}")
    print(f"TOTAL FLOPs: {total_flops/1e9:.4f} G")
    print(f"Budget Used: {total_flops/20e9*100:.2f}%")
    print(f"Headroom: {(20e9 - total_flops)/1e9:.2f} G available")
    print(f"{'='*60}")
    print(f"{'✅ PASS' if total_flops < 20e9 else '❌ FAIL'}: {'<20G' if total_flops < 20e9 else '>20G'}")
    
except ImportError:
    print("⚠ fvcore not installed. Run: pip install fvcore")
except Exception as e:
    print(f"⚠ FLOPs calculation failed: {e}")

# ============================================================================
# 3. ARCHITECTURE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🏗️ 3. ARCHITECTURE SUMMARY")
print("="*80)

print(f"Name: {model.__class__.__name__}")
if hasattr(model, 'channels'): print(f"Channels: {model.channels}")
if hasattr(model, 'n_blocks'): print(f"Blocks: {model.n_blocks}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
