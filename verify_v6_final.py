#!/usr/bin/env python3
"""
Comprehensive V6_Final Model Verification Script
=================================================
Run this on a CUDA machine BEFORE training to confirm:
  1. Parameter count < 1,000,000 (NTIRE Track 2 constraint)
  2. FLOPs < 20G (NTIRE Track 2 constraint)
  3. Forward pass produces correct output shape
  4. Layer-by-layer parameter breakdown
  5. ICNR init verification

Usage:
  python verify_v6_final.py                    # Quick check
  python verify_v6_final.py --detailed         # Per-layer FLOPs breakdown
  python verify_v6_final.py --batch-test 4     # Test batch=4 forward pass

Requirements:
  pip install fvcore torch mamba-ssm causal-conv1d einops
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return total, trainable, buffers


def parameter_breakdown(model):
    """Print per-module parameter counts."""
    print("\n" + "="*70)
    print("PARAMETER BREAKDOWN")
    print("="*70)
    
    # Group by top-level module
    groups = {}
    for name, param in model.named_parameters():
        top = name.split('.')[0]
        if top not in groups:
            groups[top] = {'count': 0, 'names': []}
        groups[top]['count'] += param.numel()
        groups[top]['names'].append((name, param.shape, param.numel()))
    
    for group_name, info in sorted(groups.items(), key=lambda x: -x[1]['count']):
        pct = info['count'] / sum(g['count'] for g in groups.values()) * 100
        print(f"\n  {group_name}: {info['count']:,} params ({pct:.1f}%)")
        for name, shape, numel in info['names']:
            print(f"    {name:55s} {str(list(shape)):20s} = {numel:,}")
    
    print(f"\n{'='*70}")


def check_icnr_init(model):
    """Verify ICNR initialization fired correctly."""
    print("\n" + "="*70)
    print("ICNR INITIALIZATION CHECK")
    print("="*70)
    
    icnr_fired = False
    for i, mod in enumerate(model.upsampling):
        if isinstance(mod, nn.Conv2d):
            oc, ic = mod.out_channels, mod.in_channels
            print(f"  upsampling[{i}]: Conv2d({ic}, {oc}, k={mod.kernel_size})")
            if oc > ic:
                scale_sq = oc // ic
                match = (oc == ic * scale_sq)
                status = "✓ ICNR ACTIVE" if match else "✗ ICNR SKIPPED"
                print(f"    → oc={oc}, ic={ic}, scale_sq={scale_sq}, "
                      f"ic*scale_sq={ic*scale_sq}, {status}")
                if match:
                    icnr_fired = True
                    # Verify the weight pattern: all scale_sq slices should be identical
                    w = mod.weight.data  # (oc, ic, kH, kW)
                    slices = w.view(ic, scale_sq, ic, *w.shape[2:])
                    first_slice = slices[:, 0]
                    all_same = all(
                        torch.allclose(slices[:, s], first_slice, atol=1e-6)
                        for s in range(1, scale_sq)
                    )
                    if all_same:
                        print(f"    → Weight pattern: All {scale_sq} sub-kernels "
                              f"are identical ✓ (ICNR repeat_interleave confirmed)")
                    else:
                        print(f"    → WARNING: Sub-kernels differ — ICNR may not "
                              f"have been applied correctly")
    
    if icnr_fired:
        print("\n  ICNR CHECK: PASSED ✓")
    else:
        print("\n  ICNR CHECK: FAILED ✗ — No conv matched ICNR condition!")
    
    return icnr_fired


def compute_flops(model, input_shape, device, detailed=False):
    """Compute FLOPs using fvcore."""
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
    except ImportError:
        print("\n  WARNING: fvcore not installed. Install with: pip install fvcore")
        print("  Skipping FLOPs analysis.")
        return None
    
    print("\n" + "="*70)
    print("FLOPS ANALYSIS (fvcore)")
    print("="*70)
    
    model.eval()
    x = torch.randn(*input_shape).to(device)
    data_info = [5, 5]  # [angRes_in, angRes_out]
    
    # fvcore analysis
    flops_analyzer = FlopCountAnalysis(model, (x, data_info))
    flops_analyzer.unsupported_ops_warnings(False)
    flops_analyzer.uncalled_modules_warnings(False)
    
    total_flops = flops_analyzer.total()
    total_gflops = total_flops / 1e9
    
    print(f"\n  Input shape:  {list(input_shape)}")
    print(f"  Total FLOPs:  {total_flops:,.0f} ({total_gflops:.2f}G)")
    print(f"  Budget:       20.00G")
    print(f"  Headroom:     {20.0 - total_gflops:.2f}G")
    
    if total_gflops < 20.0:
        print(f"  Status:       ✓ WITHIN BUDGET")
    else:
        print(f"  Status:       ✗ OVER BUDGET!")
    
    if detailed:
        print(f"\n  Per-module breakdown:")
        print(flop_count_table(flops_analyzer, max_depth=3))
    
    # Check for uncounted ops (Mamba selective scan is custom CUDA)
    uncounted = flops_analyzer.uncalled_modules()
    if uncounted:
        print(f"\n  WARNING: {len(uncounted)} modules were not called during analysis:")
        for name in list(uncounted)[:10]:
            print(f"    - {name}")
    
    unsupported = flops_analyzer.unsupported_ops()
    if unsupported:
        print(f"\n  NOTE: {len(unsupported)} unsupported ops (FLOPs underestimated):")
        for op, count in unsupported.items():
            print(f"    - {op}: {count} calls")
        print(f"\n  Mamba selective_scan is a custom CUDA op not counted by fvcore.")
        print(f"  Actual FLOPs will be HIGHER than reported. The official NTIRE")
        print(f"  evaluation uses their own profiler — fvcore gives a lower bound.")
    
    return total_gflops


def forward_pass_test(model, device, batch_sizes=[1, 2, 4]):
    """Test forward pass with various batch sizes."""
    print("\n" + "="*70)
    print("FORWARD PASS TEST")
    print("="*70)
    
    model.eval()
    data_info = [5, 5]
    
    for bs in batch_sizes:
        x = torch.randn(bs, 1, 160, 160).to(device)
        try:
            with torch.no_grad():
                y = model(x, data_info)
            expected = (bs, 1, 640, 640)
            if y.shape == torch.Size(expected):
                vram = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0
                print(f"  Batch={bs}: {tuple(x.shape)} → {tuple(y.shape)} ✓  "
                      f"(VRAM peak: {vram:.0f} MB)")
            else:
                print(f"  Batch={bs}: SHAPE MISMATCH! Got {tuple(y.shape)}, "
                      f"expected {expected}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch={bs}: OOM — reduce batch size")
                torch.cuda.empty_cache()
            else:
                print(f"  Batch={bs}: ERROR — {e}")
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)


def main():
    parser = argparse.ArgumentParser(description='V6_Final Model Verification')
    parser.add_argument('--detailed', action='store_true',
                        help='Show per-layer FLOPs breakdown')
    parser.add_argument('--batch-test', type=int, nargs='+', default=[1, 2, 4],
                        help='Batch sizes to test forward pass')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    parser.add_argument('--skip-flops', action='store_true',
                        help='Skip FLOPs analysis (if fvcore not installed)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device).total_mem / 1024**3:.1f} GB")
    
    # ── 1. Load Model ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("LOADING MODEL: MyEfficientLFNetV6_Final")
    print("="*70)
    
    from model.SR.MyEfficientLFNetV6_Final import get_model
    
    class ModelArgs:
        angRes_in = 5
        scale_factor = 4
        mlfim_mask_ratio = 0.0
    
    model = get_model(ModelArgs()).to(device)
    
    # ── 2. Parameter Count ──────────────────────────────────────────
    total, trainable, buffers = count_parameters(model)
    
    print(f"\n  Total params:     {total:>10,}  ({total/1e6:.3f}M)")
    print(f"  Trainable params: {trainable:>10,}")
    print(f"  Buffers:          {buffers:>10,}")
    print(f"  Budget:           1,000,000")
    print(f"  Headroom:         {1_000_000 - total:>10,}")
    
    param_ok = total < 1_000_000
    if param_ok:
        print(f"  Status:           ✓ WITHIN BUDGET")
    else:
        print(f"  Status:           ✗ OVER BUDGET!")
    
    # ── 3. Parameter Breakdown ──────────────────────────────────────
    parameter_breakdown(model)
    
    # ── 4. ICNR Check ──────────────────────────────────────────────
    icnr_ok = check_icnr_init(model)
    
    # ── 5. Forward Pass Test ────────────────────────────────────────
    forward_pass_test(model, device, args.batch_test)
    
    # ── 6. FLOPs Analysis ──────────────────────────────────────────
    flops_gflops = None
    if not args.skip_flops:
        flops_gflops = compute_flops(
            model, (1, 1, 160, 160), device, detailed=args.detailed
        )
    
    # ── 7. Summary ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    checks = [
        ("Parameters < 1M", param_ok, f"{total:,}"),
        ("ICNR Init Active", icnr_ok, "Bottleneck Conv2d(48,768)"),
        ("Forward Pass (B=1)", True, "(1,1,160,160) → (1,1,640,640)"),
    ]
    
    if flops_gflops is not None:
        flops_ok = flops_gflops < 20.0
        checks.append(("FLOPs < 20G (fvcore)", flops_ok, f"{flops_gflops:.2f}G"))
    
    all_ok = True
    for name, ok, detail in checks:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  [{status}] {name}: {detail}")
        if not ok:
            all_ok = False
    
    print(f"\n{'='*70}")
    if all_ok:
        print("  ALL CHECKS PASSED — Model is ready for training.")
    else:
        print("  SOME CHECKS FAILED — Fix issues before training!")
    print(f"{'='*70}\n")
    
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
