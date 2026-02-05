"""
NTIRE 2025/2026 Official Efficiency Check Script
================================================
Based on OFFICIAL guidelines:
- FLOPs: Computed using fvcore library
- Input: 5×5 angular × 32×32 spatial LF (SAI format: [1, 1, 160, 160])
- Limits: Parameters < 1M, FLOPs < 20G
- TTA operations count toward FLOPs if used

This script provides:
1. fvcore-based FLOPs counting (official method)
2. ptflops/thop fallback for comparison
3. Per-module breakdown to identify bottlenecks
4. GPU memory profiling
5. Inference speed benchmarking

Usage:
    python check_efficiency_official.py --model_name MyEfficientLFNetV6_4
    python check_efficiency_official.py --model_name MyEfficientLFNetV6_4 --detailed
"""

import argparse
import sys
import time
import importlib
from collections import defaultdict

import torch
import torch.nn as nn


# =============================================================================
# OFFICIAL NTIRE 2025 CONSTRAINTS
# =============================================================================
PARAM_LIMIT = 1_000_000      # 1M parameters
FLOPS_LIMIT = 20e9           # 20 GFLOPs
ANG_RES = 5                  # 5×5 angular resolution
PATCH_SIZE = 32              # 32×32 spatial patches
SCALE_FACTOR = 4             # 4× upscaling


def parse_args():
    parser = argparse.ArgumentParser(description='NTIRE 2025/2026 Official Efficiency Check')
    parser.add_argument('--model_name', type=str, default='MyEfficientLFNetV6_4',
                        help='Model name (must exist in model/SR/)')
    parser.add_argument('--angRes', type=int, default=ANG_RES,
                        help='Angular resolution (default: 5)')
    parser.add_argument('--patch_size', type=int, default=PATCH_SIZE,
                        help='Spatial patch size (default: 32)')
    parser.add_argument('--scale_factor', type=int, default=SCALE_FACTOR,
                        help='Upscaling factor (default: 4)')
    parser.add_argument('--detailed', action='store_true',
                        help='Show per-module FLOPs breakdown')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference speed benchmark')
    parser.add_argument('--deploy', action='store_true',
                        help='Test in deploy mode (fused blocks)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    return parser.parse_args()


def load_model(model_name, ang_res, sf, device, deploy=False):
    """Load model with proper configuration."""
    class ModelArgs:
        angRes_in = ang_res
        angRes_out = ang_res
        scale_factor = sf
    
    MODEL_PATH = f'model.SR.{model_name}'
    try:
        MODEL = importlib.import_module(MODEL_PATH)
        model = MODEL.get_model(ModelArgs())
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    if deploy and hasattr(model, 'switch_to_deploy'):
        print("🔧 Switching to deploy mode...")
        model.switch_to_deploy()
    
    model = model.to(device)
    model.eval()
    return model


def count_parameters(model):
    """Count parameters with breakdown by type."""
    total = 0
    trainable = 0
    frozen = 0
    
    param_by_type = defaultdict(int)
    
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        
        if param.requires_grad:
            trainable += count
        else:
            frozen += count
        
        # Categorize
        if 'mamba' in name.lower() or 'ssm' in name.lower():
            param_by_type['Mamba/SSM'] += count
        elif 'conv' in name.lower():
            param_by_type['Convolutions'] += count
        elif 'attn' in name.lower() or 'attention' in name.lower():
            param_by_type['Attention'] += count
        elif 'norm' in name.lower() or 'bn' in name.lower() or 'ln' in name.lower():
            param_by_type['Normalization'] += count
        else:
            param_by_type['Other'] += count
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'by_type': dict(param_by_type)
    }


def count_flops_fvcore(model, input_tensor, detailed=False):
    """
    Count FLOPs using fvcore (OFFICIAL NTIRE method).
    Returns total FLOPs and optionally per-module breakdown.
    """
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        from fvcore.nn.jit_handles import Handle
    except ImportError:
        print("❌ fvcore not installed. Install with: pip install fvcore")
        return None, None
    
    try:
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        
        total_flops = flops.total()
        
        # Get per-module breakdown
        breakdown = None
        if detailed:
            breakdown = {}
            for name, module in model.named_modules():
                if name:
                    try:
                        module_flops = flops.by_module().get(name, 0)
                        if module_flops > 0:
                            breakdown[name] = module_flops
                    except:
                        pass
        
        # Check for unsupported ops
        unsupported = flops.unsupported_ops()
        if unsupported:
            print(f"\n⚠️  Unsupported ops (FLOPs may be undercounted):")
            for op, count in unsupported.items():
                print(f"   - {op}: {count} occurrences")
        
        return total_flops, breakdown
        
    except Exception as e:
        print(f"⚠️  fvcore analysis failed: {e}")
        return None, None


def count_flops_ptflops(model, input_tensor):
    """
    Count FLOPs using ptflops/thop (alternative method).
    Often better at handling custom ops.
    """
    try:
        from ptflops import get_model_complexity_info
        
        def input_constructor(input_res):
            return {'x': torch.randn(1, *input_res).to(next(model.parameters()).device)}
        
        # ptflops needs input resolution without batch
        input_res = tuple(input_tensor.shape[1:])
        
        macs, params = get_model_complexity_info(
            model, 
            input_res,
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        # MACs to FLOPs (1 MAC = 2 FLOPs)
        flops = macs * 2
        return flops
        
    except ImportError:
        print("ℹ️  ptflops not installed. Install with: pip install ptflops")
        return None
    except Exception as e:
        print(f"⚠️  ptflops failed: {e}")
        return None


def count_flops_thop(model, input_tensor):
    """
    Count FLOPs using thop (another alternative).
    """
    try:
        from thop import profile, clever_format
        
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops = macs * 2  # MACs to FLOPs
        return flops
        
    except ImportError:
        print("ℹ️  thop not installed. Install with: pip install thop")
        return None
    except Exception as e:
        print(f"⚠️  thop failed: {e}")
        return None


def count_flops_manual(model, input_tensor):
    """
    Manual FLOPs counting with hooks (handles custom Mamba ops better).
    Counts Conv2d, Linear, and estimates Mamba SSM operations.
    """
    total_flops = [0]  # Use list for mutability in closure
    mamba_flops = [0]
    
    def conv_hook(module, input, output):
        batch_size = input[0].size(0)
        output_channels = output.size(1)
        output_h = output.size(2)
        output_w = output.size(3)
        
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        in_channels = module.in_channels // module.groups
        
        # FLOPs = 2 * (kernel_ops * in_channels) * (output size)
        flops = 2 * kernel_ops * in_channels * output_channels * output_h * output_w * batch_size
        total_flops[0] += flops
    
    def linear_hook(module, input, output):
        batch_size = input[0].size(0)
        flops = 2 * module.in_features * module.out_features * batch_size
        total_flops[0] += flops
    
    def mamba_hook(module, input, output):
        # Estimate Mamba SSM FLOPs
        # Mamba complexity: O(B * L * D * N) where N is state size
        # Typical state_size=16, expand_ratio=2
        try:
            if hasattr(module, 'd_model'):
                d_model = module.d_model
            elif hasattr(module, 'd_inner'):
                d_model = module.d_inner // 2
            else:
                d_model = 64  # fallback
            
            if len(input[0].shape) >= 3:
                batch = input[0].size(0)
                seq_len = input[0].size(1) if len(input[0].shape) > 2 else input[0].numel() // batch // d_model
            else:
                batch, seq_len = 1, input[0].numel()
            
            state_size = getattr(module, 'd_state', 16)
            expand = getattr(module, 'expand', 2)
            
            # SSM scan: 4 * B * L * D * N (for selectivity operations)
            ssm_flops = 4 * batch * seq_len * d_model * expand * state_size
            mamba_flops[0] += ssm_flops
            total_flops[0] += ssm_flops
        except:
            pass
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif 'mamba' in module.__class__.__name__.lower():
            hooks.append(module.register_forward_hook(mamba_hook))
    
    with torch.no_grad():
        model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    return total_flops[0], mamba_flops[0]


def estimate_fft_flops(h, w, channels, num_fft_layers=1):
    """
    Estimate FFT operation FLOPs.
    FFT complexity: O(N * log(N)) where N = H * W
    """
    n = h * w
    # 2D FFT: 5 * N * log2(N) per channel, forward and backward
    fft_flops_per_channel = 2 * 5 * n * (n.bit_length() - 1)
    return fft_flops_per_channel * channels * num_fft_layers * 2  # forward + backward


def benchmark_inference(model, input_tensor, warmup=10, iterations=100):
    """Benchmark inference speed."""
    device = next(model.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time = elapsed / iterations * 1000  # ms
    
    return avg_time


def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
        }
    return None


def main():
    args = parse_args()
    
    print("=" * 70)
    print("  NTIRE 2025/2026 OFFICIAL EFFICIENCY CHECK")
    print("=" * 70)
    
    # Official input specification
    H = W = args.patch_size
    angRes = args.angRes
    sai_h = angRes * H  # 160 for 5×32
    sai_w = angRes * W  # 160 for 5×32
    
    print(f"\n📋 Configuration:")
    print(f"   Model:          {args.model_name}")
    print(f"   Angular Res:    {angRes}×{angRes}")
    print(f"   Patch Size:     {H}×{W}")
    print(f"   SAI Input:      [1, 1, {sai_h}, {sai_w}]")
    print(f"   Scale Factor:   {args.scale_factor}×")
    print(f"   Device:         {args.device}")
    print(f"   Deploy Mode:    {args.deploy}")
    
    # Load model
    print(f"\n🔄 Loading model...")
    model = load_model(args.model_name, angRes, args.scale_factor, args.device, args.deploy)
    
    # Create official input tensor
    input_tensor = torch.randn(1, 1, sai_h, sai_w, device=args.device)
    
    # ==========================================================================
    # PARAMETER COUNT
    # ==========================================================================
    print(f"\n{'='*70}")
    print("  PARAMETER COUNT")
    print(f"{'='*70}")
    
    param_info = count_parameters(model)
    
    print(f"\n   Total:      {param_info['total']:>12,}")
    print(f"   Trainable:  {param_info['trainable']:>12,}")
    print(f"   Frozen:     {param_info['frozen']:>12,}")
    print(f"   Size (fp32):{param_info['total'] * 4 / 1024**2:>10.2f} MB")
    print(f"   Size (fp16):{param_info['total'] * 2 / 1024**2:>10.2f} MB")
    
    print(f"\n   By Type:")
    for ptype, count in sorted(param_info['by_type'].items(), key=lambda x: -x[1]):
        pct = count / param_info['total'] * 100
        print(f"      {ptype:<20} {count:>10,} ({pct:>5.1f}%)")
    
    param_pass = param_info['total'] < PARAM_LIMIT
    print(f"\n   ✅ Parameter Limit:  {param_info['total']:,} / {PARAM_LIMIT:,}", end="")
    print(f" {'✓ PASS' if param_pass else '✗ FAIL'}")
    
    # ==========================================================================
    # FLOPS COUNT
    # ==========================================================================
    print(f"\n{'='*70}")
    print("  FLOPS COUNT (Multiple Methods)")
    print(f"{'='*70}")
    
    flops_results = {}
    
    # Method 1: fvcore (OFFICIAL)
    print("\n📊 Method 1: fvcore (OFFICIAL)")
    fvcore_flops, breakdown = count_flops_fvcore(model, input_tensor, args.detailed)
    if fvcore_flops:
        flops_results['fvcore'] = fvcore_flops
        print(f"   Total: {fvcore_flops:,.0f} ({fvcore_flops/1e9:.2f}G)")
    
    # Method 2: Manual with Mamba estimation
    print("\n📊 Method 2: Manual + Mamba Estimation")
    manual_flops, mamba_flops = count_flops_manual(model, input_tensor)
    if manual_flops:
        flops_results['manual'] = manual_flops
        print(f"   Conv/Linear:  {(manual_flops - mamba_flops):,.0f} ({(manual_flops - mamba_flops)/1e9:.2f}G)")
        print(f"   Mamba Est:    {mamba_flops:,.0f} ({mamba_flops/1e9:.2f}G)")
        print(f"   Total:        {manual_flops:,.0f} ({manual_flops/1e9:.2f}G)")
    
    # Method 3: ptflops
    print("\n📊 Method 3: ptflops")
    ptflops_result = count_flops_ptflops(model, input_tensor)
    if ptflops_result:
        flops_results['ptflops'] = ptflops_result
        print(f"   Total: {ptflops_result:,.0f} ({ptflops_result/1e9:.2f}G)")
    
    # Method 4: thop
    print("\n📊 Method 4: thop")
    thop_result = count_flops_thop(model, input_tensor)
    if thop_result:
        flops_results['thop'] = thop_result
        print(f"   Total: {thop_result:,.0f} ({thop_result/1e9:.2f}G)")
    
    # FFT estimation if spectral attention is present
    print("\n📊 FFT Operations Estimate (if spectral attention present):")
    fft_flops = estimate_fft_flops(sai_h, sai_w, 64, num_fft_layers=1)
    print(f"   ~{fft_flops:,.0f} ({fft_flops/1e9:.3f}G) per spectral attention layer")
    
    # Summary
    print(f"\n{'─'*70}")
    print("  FLOPS Summary:")
    print(f"{'─'*70}")
    
    for method, flops in flops_results.items():
        status = "✓ PASS" if flops < FLOPS_LIMIT else "✗ FAIL"
        pct = flops / FLOPS_LIMIT * 100
        print(f"   {method:<12} {flops/1e9:>8.2f}G / 20.00G ({pct:>5.1f}%) {status}")
    
    # Use fvcore as official, or max of available
    official_flops = flops_results.get('fvcore') or max(flops_results.values()) if flops_results else 0
    flops_pass = official_flops < FLOPS_LIMIT
    
    # Detailed breakdown
    if args.detailed and breakdown:
        print(f"\n{'='*70}")
        print("  DETAILED MODULE BREAKDOWN (Top 20)")
        print(f"{'='*70}")
        
        sorted_breakdown = sorted(breakdown.items(), key=lambda x: -x[1])[:20]
        for name, flops in sorted_breakdown:
            print(f"   {name[:50]:<50} {flops/1e9:>8.3f}G")
    
    # ==========================================================================
    # FORWARD PASS TEST
    # ==========================================================================
    print(f"\n{'='*70}")
    print("  FORWARD PASS VALIDATION")
    print(f"{'='*70}")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    expected_h = sai_h * args.scale_factor
    expected_w = sai_w * args.scale_factor
    
    print(f"\n   Input:    {list(input_tensor.shape)}")
    print(f"   Output:   {list(output.shape)}")
    print(f"   Expected: [1, 1, {expected_h}, {expected_w}]")
    
    shape_pass = list(output.shape) == [1, 1, expected_h, expected_w]
    print(f"\n   {'✓ PASS' if shape_pass else '✗ FAIL'}: Output shape {'correct' if shape_pass else 'MISMATCH'}")
    
    # ==========================================================================
    # GPU MEMORY PROFILING
    # ==========================================================================
    if args.device == 'cuda':
        print(f"\n{'='*70}")
        print("  GPU MEMORY PROFILING")
        print(f"{'='*70}")
        
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        mem = get_gpu_memory()
        print(f"\n   Allocated:     {mem['allocated']:>8.1f} MB")
        print(f"   Reserved:      {mem['reserved']:>8.1f} MB")
        print(f"   Peak:          {mem['max_allocated']:>8.1f} MB")
    
    # ==========================================================================
    # INFERENCE BENCHMARK
    # ==========================================================================
    if args.benchmark:
        print(f"\n{'='*70}")
        print("  INFERENCE SPEED BENCHMARK")
        print(f"{'='*70}")
        
        avg_ms = benchmark_inference(model, input_tensor)
        fps = 1000 / avg_ms
        
        print(f"\n   Average time:  {avg_ms:.2f} ms/frame")
        print(f"   Throughput:    {fps:.1f} FPS")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print(f"\n{'='*70}")
    print("  📋 FINAL SUMMARY")
    print(f"{'='*70}")
    
    all_pass = param_pass and flops_pass and shape_pass
    
    print(f"""
   Model:        {args.model_name}
   Parameters:   {param_info['total']:,} / {PARAM_LIMIT:,} ({param_info['total']/PARAM_LIMIT*100:.1f}%)  {'✓' if param_pass else '✗'}
   FLOPs:        {official_flops/1e9:.2f}G / 20.00G ({official_flops/FLOPS_LIMIT*100:.1f}%)  {'✓' if flops_pass else '✗'}
   Output Shape: {'Correct' if shape_pass else 'WRONG'}  {'✓' if shape_pass else '✗'}
   Deploy Mode:  {'Yes' if args.deploy else 'No'}
""")
    
    if all_pass:
        print("   🎉 MODEL MEETS ALL NTIRE 2025/2026 TRACK 2 CONSTRAINTS!")
    else:
        print("   ⚠️  MODEL DOES NOT MEET ALL CONSTRAINTS!")
        if not param_pass:
            print(f"      → Reduce parameters by {param_info['total'] - PARAM_LIMIT:,}")
        if not flops_pass:
            print(f"      → Reduce FLOPs by {(official_flops - FLOPS_LIMIT)/1e9:.2f}G")
    
    print("=" * 70)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
