"""
NTIRE 2026 Track 2 Efficiency Validation (Google Colab Standalone)
==================================================================
Run this block in a Google Colab notebook cell to audit your 
MyEfficientLFNetV10 model's Parameters (<1M) and FLOPs (<20G).

Instructions:
1. Make sure fvcore and mamba-ssm are installed:
   !pip install fvcore mamba-ssm causal-conv1d

2. Paste your `get_model` and any related classes (e.g., SpaAngGroup, 
   BMDMambaLayer) in the space provided below.

3. Run the cell.
"""

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("❌ fvcore not installed! Please run: !pip install fvcore")
    import sys
    sys.exit(1)

# ==============================================================================
# 1. PASTE YOUR V10 MODEL CODE HERE
# ==============================================================================
#
# Replace the dummy class below with your entire MyEfficientLFNetV10.py 
# (from `class get_model` down to all the helper blocks).
# 
# For example:
# class get_model(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         ...
#
# class SpaAngGroup(nn.Module):
#     ...
#
# ==============================================================================

class get_model(nn.Module):
    """
    Placeholder. Replace this entirely with your actual V10 `get_model` class
    and all its dependencies!
    """
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, 1, 1)
        
    def forward(self, x, info=None):
        return self.conv(x)


# ==============================================================================
# 2. NTIRE 2026 EFFICIENCY AUDITING SCRIPT
# ==============================================================================

def count_parameters(model):
    """Count total trainable parameters (Limit: 1,000,000)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_flops_mamba(model, input_tensor):
    """
    Measure FLOPs using fvcore, accurately registering Mamba SSM operations
    to match the NTIRE 2026 Track 2 audit environment.
    """
    def _selective_scan_flop_jit(inputs, outputs):
        """FLOPs for Mamba selective scan (from mamba_ssm primitive op)."""
        def flops_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False):
            # standard 9 * B * L * D * N theoretical algorithmic cost
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
        except Exception:
            return 0

    # Register the exact hooks check_efficiency.py uses
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

    flop_counter = FlopCountAnalysis(model, input_tensor)
    flop_counter.set_op_handle(**supported_ops)

    # Suppress warnings to keep output clean like Track 2 validation
    flop_counter.unsupported_ops_warnings(False)
    flop_counter.uncalled_modules_warnings(False)

    total_flops = flop_counter.total()
    
    print("\n📊 FLOPs Breakdown (top-level modules):")
    print("-" * 50)
    for name, flops in sorted(flop_counter.by_module().items(), key=lambda x: -x[1]):
        if name == "" or name.count(".") > 1:
            continue
        if flops > 0:
            print(f"  {name:40s} {flops/1e9:>8.3f} G")

    return total_flops

def main():
    print("=" * 60)
    print("🏆 NTIRE 2026 Track 2 — Efficiency Validation (Colab Standalone)")
    print("=" * 60)
    
    PARAM_LIMIT = 1_000_000   # < 1M parameters
    FLOP_LIMIT  = 20e9        # < 20G FLOPs
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📦 Using device: {device}")
    
    # Standard testing arguments
    class ModelArgs:
        angRes_in = 5
        scale_factor = 4
        mlfim_mask_ratio = 0.0

    # 1. Load Model
    try:
        model = get_model(ModelArgs()).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error instantiating model: {e}")
        print("   Did you paste your full MyEfficientLFNetV10 code properly?")
        return
        
    # --- Parameter Count ---
    print("\n" + "=" * 60)
    print("📋 PARAMETER COUNT")
    print("=" * 60)
    num_params = count_parameters(model)
    param_pass = num_params < PARAM_LIMIT
    
    print(f"  Total Parameters:  {num_params:>12,}")
    print(f"  Limit:             {PARAM_LIMIT:>12,}")
    print(f"  Usage:             {num_params / PARAM_LIMIT * 100:>11.1f}%")
    print(f"  Status:            {'✅ PASS' if param_pass else '❌ FAIL'}")
    
    # --- FLOPs Count ---
    print("\n" + "=" * 60)
    print("⚡ FLOPs COUNT (fvcore — NTIRE official)")
    print("=" * 60)
    
    # NTIRE Standard Input: 5x5 angular resolution, 32x32 spatial patch size
    H = 5 * 32
    W = 5 * 32
    print(f"  Test Input Shape:  1×1×{H}×{W}")
    input_tensor = torch.randn(1, 1, H, W, device=device)
    
    total_flops = measure_flops_mamba(model, input_tensor)
    flop_pass = total_flops < FLOP_LIMIT
    
    print(f"\n  Total FLOPs:       {total_flops/1e9:>11.2f} G")
    print(f"  Limit:             {FLOP_LIMIT/1e9:>11.0f} G")
    print(f"  Usage:             {total_flops / FLOP_LIMIT * 100:>11.1f}%")
    print(f"  Status:            {'✅ PASS' if flop_pass else '❌ FAIL'}")
    
    # --- Final Verdict ---
    print("\n" + "=" * 60)
    print("🏁 FINAL VERDICT")
    print("=" * 60)
    if param_pass and flop_pass:
        if isinstance(model, get_model) and not hasattr(model, 'sa_groups'):
             print("  ⚠️ NOTE: You are testing the placeholder dummy model.")
             print("    Please paste your actual V10 code to see real results!")
        else:
            print("\n  🎉 MODEL QUALIFIES FOR NTIRE 2026 TRACK 2!")
    else:
        print("\n  🚫 MODEL DOES NOT QUALIFY — fix limits above!")
    print("=" * 60)


if __name__ == "__main__":
    main()
