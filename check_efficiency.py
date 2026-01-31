"""
Efficiency Check Script for NTIRE 2026 LF-SR Challenge Track 2

Verifies that the model meets the efficiency constraints:
- Parameters: < 1,000,000 (< 1MB)
- FLOPs: < 20G (measured on 1x25x32x32x3 MacPI format input)

Usage:
    python check_efficiency.py --model_name MyEfficientLFNet
"""

import argparse
import torch
import importlib
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Check model efficiency for NTIRE 2026')
    parser.add_argument('--model_name', type=str, default='MyEfficientLFNet',
                        help='Model name (must exist in model/SR/)')
    parser.add_argument('--angRes', type=int, default=5,
                        help='Angular resolution')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Spatial upscaling factor')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Spatial patch size (LR)')
    parser.add_argument('--deploy', action='store_true',
                        help='Check in deploy mode (fused RepVGG blocks)')
    return parser.parse_args()


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops_fvcore(model, input_tensor):
    """Count FLOPs using fvcore (recommended by NTIRE)."""
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        
        flops = FlopCountAnalysis(model, input_tensor)
        total_flops = flops.total()
        
        return total_flops
    except ImportError:
        print("Warning: fvcore not installed. Install with: pip install fvcore")
        return None


def count_flops_manual(model, input_tensor):
    """Manual FLOPs counting using hooks (fallback)."""
    from torch.nn.modules.conv import Conv2d
    from torch.nn.modules.linear import Linear
    
    total_flops = 0
    
    def conv_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        output_channels = output.size(1)
        output_height = output.size(2)
        output_width = output.size(3)
        
        kernel_size = module.kernel_size[0] * module.kernel_size[1]
        in_channels = module.in_channels
        groups = module.groups
        
        flops_per_instance = kernel_size * in_channels // groups
        total_instances = batch_size * output_channels * output_height * output_width
        total_flops += flops_per_instance * total_instances * 2  # multiply-add
    
    def linear_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        total_flops += batch_size * module.in_features * module.out_features * 2
    
    hooks = []
    for module in model.modules():
        if isinstance(module, Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    
    with torch.no_grad():
        model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    return total_flops


def main():
    args = parse_args()
    
    print("=" * 60)
    print("NTIRE 2026 LF-SR Challenge - Track 2 Efficiency Check")
    print("=" * 60)
    
    # Create args object for model
    class ModelArgs:
        angRes_in = args.angRes
        angRes_out = args.angRes
        scale_factor = args.scale_factor
    
    model_args = ModelArgs()
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    MODEL_PATH = 'model.SR.' + args.model_name
    try:
        MODEL = importlib.import_module(MODEL_PATH)
        model = MODEL.get_model(model_args)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Switch to deploy mode if requested
    if args.deploy and hasattr(model, 'switch_to_deploy'):
        print("Switching to deploy mode (fusing RepVGG blocks)...")
        model.switch_to_deploy()
    
    model.eval()
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    param_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    print(f"\n{'='*40}")
    print("PARAMETER COUNT")
    print(f"{'='*40}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size:           {param_mb:.2f} MB (float32)")
    
    # Check constraint
    PARAM_LIMIT = 1_000_000
    if total_params < PARAM_LIMIT:
        print(f"✓ PASS: Parameters ({total_params:,}) < {PARAM_LIMIT:,}")
    else:
        print(f"✗ FAIL: Parameters ({total_params:,}) >= {PARAM_LIMIT:,}")
    
    # Create input tensor (SAI format: [B, 1, angRes*H, angRes*W])
    H = W = args.patch_size
    angRes = args.angRes
    input_tensor = torch.randn(1, 1, angRes * H, angRes * W)
    
    print(f"\n{'='*40}")
    print("FLOPS COUNT")
    print(f"{'='*40}")
    print(f"Input shape: {list(input_tensor.shape)}")
    print(f"  (SAI format: [B=1, C=1, H={angRes}x{H}={angRes*H}, W={angRes}x{W}={angRes*W}])")
    
    # Count FLOPs
    flops = count_flops_fvcore(model, input_tensor)
    
    if flops is None:
        print("\nUsing manual FLOPs counting (fallback)...")
        flops = count_flops_manual(model, input_tensor)
    
    flops_g = flops / 1e9
    
    print(f"\nTotal FLOPs: {flops:,}")
    print(f"             {flops_g:.2f} G")
    
    # Check constraint
    FLOPS_LIMIT = 20e9
    if flops < FLOPS_LIMIT:
        print(f"✓ PASS: FLOPs ({flops_g:.2f}G) < 20G")
    else:
        print(f"✗ FAIL: FLOPs ({flops_g:.2f}G) >= 20G")
    
    # Test forward pass
    print(f"\n{'='*40}")
    print("FORWARD PASS TEST")
    print(f"{'='*40}")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    expected_h = angRes * H * args.scale_factor
    expected_w = angRes * W * args.scale_factor
    
    print(f"Input shape:    {list(input_tensor.shape)}")
    print(f"Output shape:   {list(output.shape)}")
    print(f"Expected shape: [1, 1, {expected_h}, {expected_w}]")
    
    if list(output.shape) == [1, 1, expected_h, expected_w]:
        print("✓ PASS: Output shape correct")
    else:
        print("✗ FAIL: Output shape mismatch")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_pass = total_params < PARAM_LIMIT and flops < FLOPS_LIMIT
    
    print(f"Model:       {args.model_name}")
    print(f"Parameters:  {total_params:,} / {PARAM_LIMIT:,} ({'PASS' if total_params < PARAM_LIMIT else 'FAIL'})")
    print(f"FLOPs:       {flops_g:.2f}G / 20.00G ({'PASS' if flops < FLOPS_LIMIT else 'FAIL'})")
    print(f"Deploy mode: {'Yes' if args.deploy else 'No'}")
    print()
    
    if all_pass:
        print("🎉 Model meets all Track 2 efficiency constraints!")
    else:
        print("⚠️  Model does NOT meet efficiency constraints. Please optimize.")
    
    print("=" * 60)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
