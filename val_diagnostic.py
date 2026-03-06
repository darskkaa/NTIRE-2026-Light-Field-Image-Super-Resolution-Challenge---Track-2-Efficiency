import torch
import torch.nn as nn
import numpy as np
import os
import glob
from collections import defaultdict
from einops import rearrange
import sys

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Import the model and the interpolation function
try:
    from model.SR.MyEfficientLFNetV3_MLFIM import get_model, LF_interpolate
except ImportError as e:
    print(f"FAILED TO IMPORT: {e}. Make sure you run this from the project root!")
    sys.exit(1)

class DummyArgs:
    def __init__(self):
        self.angRes_in = 5
        self.scale_factor = 4
        self.channels = 48
        self.mlfim_mask_ratio = 0.25

print("="*60)
print("🔍 ULTIMATE V3 VALIDATION DIAGNOSTIC TOOL 🔍")
print("="*60)
print("This tool will aggressively hunt down why Train PSNR is 32dB but Val is 13dB.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = DummyArgs()
model = get_model(args).to(device)

# Attempt to load the best/latest checkpoint
checkpoint_dir = './log/SR_NTIRE_NTIRE_MyEfficientLFNetV3_MLFIM/checkpoints'
if os.path.exists(checkpoint_dir):
    ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getctime)
        print(f"Loading weights from {latest_ckpt}...")
        try:
            state = torch.load(latest_ckpt, map_location=device)
            state_dict = state.get('state_dict', state)
            # handle module. prefix
            new_state = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                new_state[new_key] = v
            model.load_state_dict(new_state, strict=False)
            print("✅ Weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load weights: {e}. Using random weights.")
else:
    print("⚠️ No checkpoints found. Using random initialized weights.")

print("\n" + "="*60)
print("TEST 1: FORWARD PASS TENSOR STATISTICS (.train() vs .eval())")
print("="*60)

# Hook mechanism to catch exploding activations
activation_stats = defaultdict(dict)

def get_hook(name, mode):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_stats[mode][name] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'has_nan': torch.isnan(output).any().item(),
                'shape': list(output.shape)
            }
        elif isinstance(output, tuple):
            activation_stats[mode][name] = {
                'type': 'tuple',
                'len': len(output)
            }
    return hook

# Register hooks on major blocks
tracked_modules = {
    'IFE': model.conv_init,
    'SA_Group_0': model.sa_groups[0] if len(model.sa_groups) > 0 else None,
    'EPI_Group_0': model.epi_groups[0] if len(model.epi_groups) > 0 else None,
    'Window_Attn': model.win_attn,
    'ASG': model.asg,
    'LCE': model.lce,
    'Recon_Head': model.hlfr
}

hooks = []
for name, mod in tracked_modules.items():
    if mod is not None:
        hooks.append(mod.register_forward_hook(get_hook(name, 'train')))

# Generate a fake training patch (typically 32x32 per view)
print("Running .train() forward pass...")
model.train()
x_train = torch.rand(2, 1, 5*32, 5*32).to(device) # Batch of 2
y_train = model(x_train)
print(f"Train Output -> Min: {y_train.min().item():.4f}, Max: {y_train.max().item():.4f}, Mean: {y_train.mean().item():.4f}")

# Remove train hooks, add eval hooks
for h in hooks: h.remove()
hooks = []
for name, mod in tracked_modules.items():
    if mod is not None:
        hooks.append(mod.register_forward_hook(get_hook(name, 'eval')))

print("\nRunning .eval() forward pass...")
model.eval()
with torch.no_grad():
    x_eval = torch.rand(2, 1, 5*32, 5*32).to(device)
    y_eval = model(x_eval)
print(f"Eval Output -> Min: {y_eval.min().item():.4f}, Max: {y_eval.max().item():.4f}, Mean: {y_eval.mean().item():.4f}")

print("\n--- Layer-by-Layer Comparison ---")
discrepancy_found = False
for name in tracked_modules.keys():
    if name not in activation_stats['train'] or name not in activation_stats['eval']:
        continue
    t = activation_stats['train'][name]
    e = activation_stats['eval'][name]
    if t.get('type') == 'tuple':
        continue
    
    mean_diff = abs(t['mean'] - e['mean'])
    std_diff = abs(t['std'] - e['std'])
    
    # If the statistics change drastically between train and eval on identical random noise distributions, we have a problem.
    if mean_diff > 1.0 or std_diff > 1.0 or e['has_nan']:
        discrepancy_found = True
        print(f"🚨 DISCREPANCY IN {name}!")
        print(f"   Train : Mean={t['mean']:7.3f}, Std={t['std']:7.3f}, Range=[{t['min']:7.3f}, {t['max']:7.3f}]")
        print(f"   Eval  : Mean={e['mean']:7.3f}, Std={e['std']:7.3f}, Range=[{e['min']:7.3f}, {e['max']:7.3f}]")
    else:
        print(f"✅ {name} looks stable.")

if not discrepancy_found:
    print("No severe layer discrepancies found between .train() and .eval() modes.")
    
for h in hooks: h.remove()


print("\n" + "="*60)
print("TEST 2: LFINTEGRATE_GAUSSIAN AND DATA SHAPES (Validation Pipeline)")
print("="*60)
# Often, validation scripts cut images into overlapping patches and stitch them back.
# If the reshape logic expects `(B, U, V, H, W)` but gets `(B, U*H, V*W)`, it scrambles the image resulting in ~13dB.

# Simulate a validation full-size image (e.g., HCI dataset is 512x512 per view)
# Because angRes=5, total input is 512*5 = 2560
H_val, W_val = 512, 512
angRes = 5
fake_val_hq = torch.rand(1, 1, angRes*H_val, angRes*W_val).to(device)

print(f"Simulated full validation image shape: {fake_val_hq.shape}")
# In standard LFMamba/LFTransMamba validation code, they often reshape:
try:
    fake_val_6d = rearrange(fake_val_hq, 'b c (u h) (v w) -> b c u v h w', u=angRes, v=angRes)
    print(f"Reshapes to 6D successfully: {fake_val_6d.shape}")
except Exception as e:
    print(f"🚨 FATAL SHAPE ERROR: Failed to reshape validation image. {e}")

# Test the bicubic baseline
print("\nTesting the LF_interpolate (bicubic baseline)...")
try:
    sr_y_6d = LF_interpolate(fake_val_6d, scale_factor=4, mode='bicubic')
    sr_y_macPI = rearrange(sr_y_6d, 'b c u v h w -> b c (u h) (v w)')
    print(f"Bicubic baseline produces shape: {sr_y_macPI.shape}")
    print(f"Bicubic range: Min={sr_y_macPI.min().item():.3f}, Max={sr_y_macPI.max().item():.3f}")
except Exception as e:
    print(f"🚨 FATAL BICUBIC ERROR: {e}")

print("\n" + "="*60)
print("TEST 3: THE CLAMPING BUG CHECK")
print("="*60)
print("In V3, the output is `return (out + sr_y).clamp(0, 1)`")
print("If the model output `out` (the residual) is massive, it pins the output to 0 or 1, destroying the image.")

model.eval()
with torch.no_grad():
    test_patch = torch.rand(1, 1, 5*64, 5*64).to(device)
    # Let's intercept BEFORE the clamp by grabbing the HLFR output directly.
    # We bypass the final forward() clamp by running the modules manually.
    
    # 1. Bicubic
    x_6d_t = rearrange(test_patch, 'b c (u h) (v w) -> b c u v h w', u=5, v=5)
    sr_y_t = LF_interpolate(x_6d_t, scale_factor=4, mode='bicubic')
    sr_y_t = rearrange(sr_y_t, 'b c u v h w -> b c (u h) (v w)')
    
    # 2. Extract features manually
    feat = model.conv_init0(rearrange(test_patch, 'b c (u h) (v w) -> b c (u v) h w', u=5, v=5))
    feat = model.conv_init(feat) + feat
    feat = feat + model.ang_embed
    for sa in model.sa_groups: feat = sa(feat, 5) + feat
    for epi in model.epi_groups: feat = epi(feat, 5) + feat
    
    f_2d = rearrange(feat, 'b c (u v) h w -> b c (u h) (v w)', u=5, v=5)
    f_wa = model.win_attn(f_2d)
    
    f_wa_6d = rearrange(f_wa, 'b c (u h) (v w) -> b c (u v) h w', u=5, v=5)
    f_ife = rearrange(model.conv_init0(rearrange(test_patch, 'b c (u h) (v w) -> b c (u v) h w', u=5, v=5)), 'b c (u v) h w -> b c (u h) (v w)', u=5, v=5)
    
    # Use fake inputs for ASG to just test the end
    out_res = model.hlfr(f_wa) 

    # Analyze exactly what the Mamba network attempts to ADD to the bicubic image
    print(f"RAW Network Residual ('out') Statistics:")
    print(f"  Mean: {out_res.mean().item():.4f}")
    print(f"  Min : {out_res.min().item():.4f}")
    print(f"  Max : {out_res.max().item():.4f}")
    print(f"  Std : {out_res.std().item():.4f}")
    
    if out_res.mean().abs() > 0.5 or out_res.std() > 2.0:
        print("\n🚨 CRITICAL FINDING: Your network residual is exploding!")
        print("Because the model adds this to `sr_y` and clamps to [0,1], a massive residual will turn the entire image into pure white (1.0) or pure black (0.0).")
        print("This causes a ~10-14 dB PSNR (worse than completely random noise).")
    else:
        print("\n✅ Network residual magnitude looks reasonable. It is not exploding.")


print("\n" + "="*60)
print("TEST 4: VALIDATION DATALOADER NORMALIZATION")
print("="*60)
print("If training images are loaded as [0, 1] floats, but `val_loader` loads them as [0, 255] uint8,")
print("the model will receive inputs 255x larger than it expects, causing chaotic activations.")
print("ACTION FOR YOU: Check `train.py` def test() around line 290.")
print("Add `print(lr.max(), hr.max())`. If they are > 1.0, you have found the bug.")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE.")
print("Run this script using: python val_diagnostic.py")
print("="*60)
