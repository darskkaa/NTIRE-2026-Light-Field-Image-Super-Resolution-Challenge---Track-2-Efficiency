"""
NTIRE 2026 Track 2 Efficiency Validation
==========================================
Uses fvcore (official required library) to measure:
  - Parameters (limit: < 1,000,000)
  - FLOPs (limit: < 20G with input 5x5x32x32)

Also validates:
  - Output shape correctness at multiple resolutions
  - Forward/backward gradient flow
  - THOP cross-validation of FLOPs
  - Inference time (GPU)

Usage:
  python check_efficiency.py
  python check_efficiency.py --model_name MyEfficientLFNetV3_MLFIM
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
    parser.add_argument("--model_name", type=str, default="MyEfficientLFNetV3_MLFIM",
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

    # ---- THOP Cross-Validation ----
    thop_pass = True
    print("\n" + "=" * 60)
    print("🔄 THOP FLOPs (cross-validation)")
    print("=" * 60)
    try:
        from thop import profile, clever_format
        thop_input = torch.randn(1, 1, H, W, device=device)
        flops_thop, params_thop = profile(model, inputs=(thop_input, ), verbose=False)
        flops_str, params_str = clever_format([flops_thop, params_thop], "%.3f")
        print(f"  THOP FLOPs:   {flops_str}  ({flops_thop/1e9:.3f}G)")
        print(f"  THOP Params:  {params_str}  ({params_thop/1e6:.3f}M)")
        if total_flops is not None:
            ratio = flops_thop / total_flops if total_flops > 0 else 0
            print(f"  THOP/fvcore:  {ratio:.2f}x  (expected ~1.0x)")
    except ImportError:
        print("  ⚠️  thop not installed (pip install thop) — skipping")
    except Exception as e:
        print(f"  ⚠️  THOP error: {e}")

    # ---- Dimension Validation ----
    print("\n" + "=" * 60)
    print("📐 DIMENSION VALIDATION")
    print("=" * 60)
    dim_pass = True
    test_sizes = [
        (32, 32, "NTIRE standard (32×32 per view)"),
        (64, 64, "Double patch (64×64 per view)"),
        (128, 128, "Training patch (128×128 per view)"),
    ]
    angRes = args.angRes
    scale = args.scale_factor
    for h, w, desc in test_sizes:
        H_in, W_in = angRes * h, angRes * w
        H_out, W_out = angRes * h * scale, angRes * w * scale
        x_test = torch.randn(1, 1, H_in, W_in, device=device)
        try:
            with torch.no_grad():
                y_test = model(x_test)
            ok = y_test.shape == (1, 1, H_out, W_out)
            status = "✅" if ok else "❌"
            print(f"  {status} {desc}: (1,1,{H_in},{W_in}) → {tuple(y_test.shape)}"
                  f"  expected (1,1,{H_out},{W_out})")
            if not ok:
                dim_pass = False
        except Exception as e:
            print(f"  ❌ {desc}: CRASHED — {e}")
            dim_pass = False
    print(f"  Status: {'✅ All shapes correct' if dim_pass else '❌ Shape mismatch detected'}")

    # ---- Forward / Backward Sanity ----
    print("\n" + "=" * 60)
    print("🧪 FORWARD / BACKWARD SANITY")
    print("=" * 60)
    grad_pass = True
    try:
        model.eval()
        x_fwd = torch.randn(1, 1, H, W, device=device)
        with torch.no_grad():
            y_fwd = model(x_fwd)
        expected_shape = (1, 1, H * scale, W * scale)
        fwd_ok = y_fwd.shape == expected_shape
        nan_ok = not torch.isnan(y_fwd).any().item()
        inf_ok = not torch.isinf(y_fwd).any().item()
        print(f"  Forward shape: {'✅' if fwd_ok else '❌'} {tuple(y_fwd.shape)}")
        print(f"  No NaN:        {'✅' if nan_ok else '❌'}")
        print(f"  No Inf:        {'✅' if inf_ok else '❌'}")
        if not (fwd_ok and nan_ok and inf_ok):
            grad_pass = False

        model.train()
        x_bwd = torch.randn(1, 1, H, W, device=device, requires_grad=True)
        y_bwd = model(x_bwd)
        y_bwd.mean().backward()
        grad_exists = x_bwd.grad is not None
        grad_no_nan = grad_exists and not torch.isnan(x_bwd.grad).any().item()
        print(f"  Backward grad: {'✅' if grad_exists else '❌'}")
        print(f"  Grad no NaN:   {'✅' if grad_no_nan else '❌'}")
        if not (grad_exists and grad_no_nan):
            grad_pass = False
        model.eval()
    except Exception as e:
        print(f"  ❌ CRASHED: {e}")
        grad_pass = False

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

    all_pass = param_pass and (flop_pass is True or flop_pass is None) and dim_pass and grad_pass

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

    print(f"  {'✅' if dim_pass else '❌'} Dimensions: all output shapes correct" if dim_pass
          else f"  ❌ Dimensions: shape mismatch detected")
    print(f"  {'✅' if grad_pass else '❌'} Gradients: forward/backward clean")

    if all_pass:
        print(f"\n  🎉 MODEL QUALIFIES FOR NTIRE 2026 TRACK 2!")
    else:
        print(f"\n  🚫 MODEL DOES NOT QUALIFY — fix failing checks above!")

    print("=" * 60)

    # Exit with error code if failing
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
