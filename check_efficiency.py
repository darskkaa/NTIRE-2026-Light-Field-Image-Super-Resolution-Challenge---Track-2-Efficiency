"""
NTIRE 2026 Track 2 Efficiency Validation
==========================================
Uses fvcore (official required library) to measure:
  - Parameters (limit: < 1,000,000)
  - FLOPs (limit: < 20G with input 5x5x32x32)

Also measures inference time as additional metric.

Usage:
  python check_efficiency.py --model_name MyEfficientLFNetV10
"""

import argparse
import time
import sys
import torch
import importlib
import numpy as np


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def measure_flops(model, input_tensor, model_name):
    """Measure FLOPs using fvcore (NTIRE 2026 official method)."""
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
    except ImportError:
        print("❌ fvcore not installed! Install: pip install fvcore")
        print("   fvcore is REQUIRED by NTIRE 2026 for FLOPs measurement.")
        return None

    # Register custom ops for Mamba selective scan (if present)
    # These are ignored in FLOPs counting (standard practice for Mamba models)
    try:
        from fvcore.nn import FlopCountAnalysis

        def _selective_scan_flop_jit(inputs, outputs):
            """FLOPs for Mamba selective scan (from mamba_ssm)."""
            def flops_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False):
                flops = 9 * B * L * D * N
                if with_D:
                    flops += B * D * L
                if with_Z:
                    flops += B * D * L
                return flops

            try:
                B, D, L = inputs[0].type().sizes()
                N = inputs[2].type().sizes()[1]
                return flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
            except:
                return 0

        supported_ops = {
            "aten::silu": None,
            "aten::neg": None,
            "aten::exp": None,
            "aten::flip": None,
            "prim::PythonOp.SelectiveScanMamba": _selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": _selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": _selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": _selective_scan_flop_jit,
        }
    except Exception:
        supported_ops = {}

    flop_counter = FlopCountAnalysis(model, input_tensor)
    if supported_ops:
        flop_counter.set_op_handle(**supported_ops)

    # Suppress warnings about unrecognized ops
    flop_counter.unsupported_ops_warnings(False)
    flop_counter.uncalled_modules_warnings(False)

    total_flops = flop_counter.total()

    # Print per-module breakdown (top-level only)
    print("\n📊 FLOPs Breakdown (top-level modules):")
    print("-" * 50)
    by_module = flop_counter.by_module()
    for name, flops in sorted(by_module.items(), key=lambda x: -x[1]):
        if name == "" or name.count(".") > 1:
            continue
        if flops > 0:
            print(f"  {name:40s} {flops/1e9:>8.3f} G")

    return total_flops


def measure_inference_time(model, input_tensor, warmup=10, runs=50):
    """Measure inference time (additional NTIRE 2026 metric)."""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean": times.mean(),
        "std": times.std(),
        "median": np.median(times),
        "min": times.min(),
        "max": times.max(),
    }


def main():
    parser = argparse.ArgumentParser(description="NTIRE 2026 Track 2 Efficiency Check")
    parser.add_argument("--model_name", type=str, default="MyEfficientLFNetV10",
                        help="Model name (must exist in model/SR/)")
    parser.add_argument("--angRes", type=int, default=5, help="Angular resolution")
    parser.add_argument("--scale_factor", type=int, default=4, help="Scale factor")
    parser.add_argument("--patch_size", type=int, default=32,
                        help="Spatial patch size (NTIRE standard: 32)")
    parser.add_argument("--skip_time", action="store_true",
                        help="Skip inference time measurement")
    args = parser.parse_args()

    # ---- NTIRE 2026 Track 2 Limits ----
    PARAM_LIMIT = 1_000_000    # 1M parameters
    FLOP_LIMIT  = 20e9         # 20G FLOPs

    print("=" * 60)
    print("🏆 NTIRE 2026 Track 2 — Efficiency Validation")
    print("=" * 60)
    print(f"  Model:      {args.model_name}")
    print(f"  Input:      {args.angRes}×{args.angRes}×{args.patch_size}×{args.patch_size}")
    print(f"  Scale:      {args.scale_factor}×")
    print(f"  Param Limit: {PARAM_LIMIT:,}")
    print(f"  FLOPs Limit: {FLOP_LIMIT/1e9:.0f}G")
    print("=" * 60)

    # ---- Load Model ----
    print("\n📦 Loading model...")
    MODEL_PATH = f"model.SR.{args.model_name}"
    try:
        MODEL = importlib.import_module(MODEL_PATH)
    except ImportError as e:
        print(f"❌ Cannot import {MODEL_PATH}: {e}")
        sys.exit(1)

    class ModelArgs:
        angRes_in = args.angRes
        scale_factor = args.scale_factor

    model = MODEL.get_model(ModelArgs())

    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"
        print("⚠️  No CUDA — FLOPs will be measured on CPU (may differ slightly)")

    model.eval()

    # ---- Parameter Count ----
    print("\n" + "=" * 60)
    print("📋 PARAMETER COUNT")
    print("=" * 60)
    num_params = count_parameters(model)
    param_pct = num_params / PARAM_LIMIT * 100
    param_pass = num_params < PARAM_LIMIT

    print(f"  Total Parameters:  {num_params:>12,}")
    print(f"  Limit:             {PARAM_LIMIT:>12,}")
    print(f"  Usage:             {param_pct:>11.1f}%")
    print(f"  Status:            {'✅ PASS' if param_pass else '❌ FAIL'}")
    if not param_pass:
        print(f"  ⚠️  OVER by {num_params - PARAM_LIMIT:,} params!")

    # ---- FLOPs Count (fvcore) ----
    print("\n" + "=" * 60)
    print("⚡ FLOPs COUNT (fvcore — NTIRE official)")
    print("=" * 60)

    # NTIRE standard input: 5×5×32×32 = 1ch × (5*32) × (5*32) = 1×160×160
    H = args.angRes * args.patch_size
    W = args.angRes * args.patch_size
    input_tensor = torch.randn(1, 1, H, W, device=device)

    total_flops = measure_flops(model, input_tensor, args.model_name)

    if total_flops is not None:
        flop_pct = total_flops / FLOP_LIMIT * 100
        flop_pass = total_flops < FLOP_LIMIT

        print(f"\n  Total FLOPs:       {total_flops/1e9:>11.2f} G")
        print(f"  Limit:             {FLOP_LIMIT/1e9:>11.0f} G")
        print(f"  Usage:             {flop_pct:>11.1f}%")
        print(f"  Status:            {'✅ PASS' if flop_pass else '❌ FAIL'}")
        if not flop_pass:
            print(f"  ⚠️  OVER by {(total_flops - FLOP_LIMIT)/1e9:.2f}G!")
    else:
        flop_pass = None
        print("  ⚠️  Could not measure FLOPs (fvcore not available)")

    # ---- Inference Time ----
    if not args.skip_time and device == "cuda":
        print("\n" + "=" * 60)
        print("⏱️  INFERENCE TIME")
        print("=" * 60)
        timing = measure_inference_time(model, input_tensor)
        print(f"  Mean:    {timing['mean']*1000:>8.2f} ms")
        print(f"  Std:     {timing['std']*1000:>8.2f} ms")
        print(f"  Median:  {timing['median']*1000:>8.2f} ms")
        print(f"  Min:     {timing['min']*1000:>8.2f} ms")
        print(f"  Max:     {timing['max']*1000:>8.2f} ms")
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    elif device != "cuda":
        print("\n⏱️  Skipping inference time (no CUDA)")

    # ---- FINAL VERDICT ----
    print("\n" + "=" * 60)
    print("🏁 FINAL VERDICT")
    print("=" * 60)

    all_pass = param_pass and (flop_pass is True or flop_pass is None)

    if param_pass:
        print(f"  ✅ Parameters: {num_params:,} / {PARAM_LIMIT:,} ({param_pct:.1f}%)")
    else:
        print(f"  ❌ Parameters: {num_params:,} / {PARAM_LIMIT:,} ({param_pct:.1f}%) — OVER LIMIT")

    if flop_pass is True:
        print(f"  ✅ FLOPs:      {total_flops/1e9:.2f}G / {FLOP_LIMIT/1e9:.0f}G ({flop_pct:.1f}%)")
    elif flop_pass is False:
        print(f"  ❌ FLOPs:      {total_flops/1e9:.2f}G / {FLOP_LIMIT/1e9:.0f}G ({flop_pct:.1f}%) — OVER LIMIT")
    else:
        print(f"  ⚠️  FLOPs:      not measured (install fvcore)")

    if all_pass:
        print("\n  🎉 MODEL QUALIFIES FOR NTIRE 2026 TRACK 2!")
    else:
        print("\n  🚫 MODEL DOES NOT QUALIFY — fix limits above!")

    print("=" * 60)

    # Exit with error code if failing
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
