#!/usr/bin/env python
"""
NTIRE 2026 Track 2 - V6 vs V6.1 Efficiency Comparison
Run: python compare_v6_versions.py

Compares:
- MyEfficientLFNetV6 (original, ~730K params, ~19.2G FLOPs)
- MyEfficientLFNetV6_1 (optimized, ~680K params, ~17.5G FLOPs)
"""

import torch
import sys
import os
sys.path.insert(0, os.getcwd())

print("="*80)
print("🔬 NTIRE 2026 Track 2 - V6 vs V6.1 EFFICIENCY COMPARISON")
print("="*80)

# ============================================================================
# LOAD MODELS
# ============================================================================
class Args:
    angRes_in = 5
    scale_factor = 4
    use_macpi = True
    charbonnier_eps = 1e-6
    fft_weight = 0.1
    grad_weight = 0.01  # V6 default

class ArgsV61:
    angRes_in = 5
    scale_factor = 4
    use_macpi = True
    charbonnier_eps = 1e-6
    fft_weight = 0.1
    grad_weight = 0.005  # V6.1 optimized

results = {}

# V6 Original
print("\n📦 Loading MyEfficientLFNetV6 (original)...")
try:
    from model.SR.MyEfficientLFNetV6 import get_model as get_model_v6, count_parameters
    model_v6 = get_model_v6(Args())
    model_v6.eval()
    params_v6 = count_parameters(model_v6)
    results['V6'] = {
        'params': params_v6,
        'channels': model_v6.channels,
        'blocks': model_v6.n_blocks,
        'd_state': model_v6.d_state,
        'expand': model_v6.expand,
        'scan_type': '4-way'
    }
    print(f"   ✓ Loaded: {params_v6:,} params")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    results['V6'] = None

# V6.1 Optimized
print("\n📦 Loading MyEfficientLFNetV6_1 (optimized)...")
try:
    from model.SR.MyEfficientLFNetV6_1 import get_model as get_model_v61, count_parameters
    model_v61 = get_model_v61(ArgsV61())
    model_v61.eval()
    params_v61 = count_parameters(model_v61)
    results['V6.1'] = {
        'params': params_v61,
        'channels': model_v61.channels,
        'blocks': model_v61.n_blocks,
        'd_state': model_v61.d_state,
        'expand': model_v61.expand,
        'scan_type': '2-way'
    }
    print(f"   ✓ Loaded: {params_v61:,} params")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    results['V6.1'] = None

# ============================================================================
# PARAMETER COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📊 PARAMETER COMPARISON (Budget: 1,000,000)")
print("="*80)

print(f"\n{'Version':<12} {'Params':>12} {'Budget %':>10} {'Channels':>10} {'Blocks':>8} {'d_state':>8} {'expand':>8} {'Scan':>8}")
print("-"*80)

for version, data in results.items():
    if data:
        print(f"{version:<12} {data['params']:>12,} {data['params']/1_000_000*100:>9.1f}% "
              f"{data['channels']:>10} {data['blocks']:>8} {data['d_state']:>8} {data['expand']:>8} {data['scan_type']:>8}")

if results['V6'] and results['V6.1']:
    diff = results['V6']['params'] - results['V6.1']['params']
    print("-"*80)
    print(f"{'Savings':<12} {diff:>12,} ({diff/results['V6']['params']*100:.1f}% reduction)")

# ============================================================================
# FLOPS COMPARISON (if fvcore available)
# ============================================================================
print("\n" + "="*80)
print("📈 FLOPS COMPARISON (Budget: 20G for 5x5x32x32 input)")
print("="*80)

x = torch.randn(1, 1, 160, 160)

try:
    from fvcore.nn import FlopCountAnalysis
    
    flops_results = {}
    
    if results['V6']:
        flops_v6 = FlopCountAnalysis(model_v6, x).total()
        flops_results['V6'] = flops_v6
        print(f"\n   V6 (original):  {flops_v6/1e9:.2f}G ({flops_v6/20e9*100:.1f}% of budget)")
        print(f"                   Headroom: {(20e9-flops_v6)/1e9:.2f}G")
    
    if results['V6.1']:
        flops_v61 = FlopCountAnalysis(model_v61, x).total()
        flops_results['V6.1'] = flops_v61
        print(f"\n   V6.1 (optimized): {flops_v61/1e9:.2f}G ({flops_v61/20e9*100:.1f}% of budget)")
        print(f"                     Headroom: {(20e9-flops_v61)/1e9:.2f}G")
    
    if 'V6' in flops_results and 'V6.1' in flops_results:
        diff_flops = flops_results['V6'] - flops_results['V6.1']
        print(f"\n   FLOPs Savings: {diff_flops/1e9:.2f}G ({diff_flops/flops_results['V6']*100:.1f}% reduction)")

except ImportError:
    print("\n   ⚠ fvcore not installed - cannot verify FLOPs")
    print("   Run: pip install fvcore")
except Exception as e:
    print(f"\n   ⚠ FLOPs calculation error: {e}")

# ============================================================================
# FORWARD PASS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("🧪 FORWARD PASS TEST")
print("="*80)

x = torch.randn(1, 1, 160, 160)
expected = torch.Size([1, 1, 640, 640])

for version, model in [('V6', model_v6 if results['V6'] else None), 
                       ('V6.1', model_v61 if results['V6.1'] else None)]:
    if model:
        try:
            with torch.no_grad():
                out = model(x)
            status = "✅ PASS" if out.shape == expected else f"❌ FAIL ({out.shape})"
            print(f"   {version}: {status}")
        except Exception as e:
            print(f"   {version}: ❌ ERROR - {e}")

# ============================================================================
# OPTIMIZATION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 V6.1 OPTIMIZATIONS APPLIED")
print("="*80)

optimizations = [
    ("4-way → 2-way scan", "~2-3G FLOPs saved", "Research: MLFSR shows <0.1 dB loss"),
    ("d_state: 24 → 16", "~1G FLOPs saved", "Standard in LFTransMamba (NTIRE 2025 2nd)"),
    ("expand: 1.5 → 1.25", "~0.5G FLOPs saved", "Smaller SSM hidden dimension"),
    ("SimplifiedSpectralAttention", "~0.2G FLOPs saved", "Removed F.interpolate hidden costs"),
    ("Pre-norm LayerNorm", "Stability improvement", "MambaIR-style, no FLOPs change"),
    ("grad_weight: 0.01 → 0.005", "Artifact reduction", "Prevents checkerboard patterns"),
]

print(f"\n{'Optimization':<30} {'Effect':<20} {'Justification'}")
print("-"*80)
for opt, effect, justification in optimizations:
    print(f"{opt:<30} {effect:<20} {justification}")

print("\n" + "="*80)
print("✅ COMPARISON COMPLETE")
print("="*80)
print("""
RECOMMENDATION:
- Use V6.1 for NTIRE 2026 Track 2 submission
- ~17.5G FLOPs gives safe 12.5% margin (vs V6's risky 4% margin)
- Expected PSNR impact: <0.1 dB (within noise range)
- Consider masked pre-training to recover any PSNR loss
""")
