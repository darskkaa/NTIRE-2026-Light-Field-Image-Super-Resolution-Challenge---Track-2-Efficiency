'''
MyEfficientLFNet v4.2 - TRUE MAMBA CHAMPIONSHIP

Deep Research Implementation (Jan 31, 2026):

CRITICAL FIXES from v4.1 Audit:
1. TRUE Selective SSM with A/B/C/D matrices and Δ discretization (ZOH)
2. SS2D Cross-Scan (4-way) for 2D image processing
3. Multi-scale spatial (1/3/5/7) matching MCMamba winner
4. Fixed hierarchical fusion (no more wasted features)
5. EPSW removed from training (inference-only technique)

Based on:
- Mamba: Selective SSM with input-dependent parameters
- VMamba: SS2D 4-way cross-scan for vision
- MambaIR: RSSB block with local enhancement + channel attention
- MCMamba: Multi-size parallel convolutions

Target: 30.5+ dB, <1M params, <20G FLOPs
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class get_model(nn.Module):
    """
    MyEfficientLFNet v4.2 - True Mamba Championship
    
    Proper implementation of selective state space model.
    """
    
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes = args.angRes_in
        self.scale = args.scale_factor
        
        # Architecture config (MCMamba-like)
        self.channels = 64
        self.n_blocks = 8
        
        # Shallow with local pixel enhancement (MambaIR-style)
        self.shallow = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            LocalPixelEnhancement(self.channels),
        )
        
        # Deep: True Mamba blocks with multi-scale spatial
        self.blocks = nn.ModuleList([
            TrueMambaBlock(self.channels, self.angRes)
            for _ in range(self.n_blocks)
        ])
        
        # Progressive fusion (fixed - no wasted features)
        self.fuse_early = nn.Conv2d(self.channels * 4, self.channels, 1, bias=False)
        self.fuse_late = nn.Conv2d(self.channels * 4, self.channels, 1, bias=False)
        self.fuse_final = nn.Conv2d(self.channels * 2, self.channels, 1, bias=False)
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Upsampler
        self.upsampler = EfficientUpsampler(self.channels, self.scale)
        
        # Output
        self.output = nn.Conv2d(self.channels, 1, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x, info=None):
        # Bicubic baseline
        x_up = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        # Shallow
        feat = self.shallow(x)
        shallow = feat
        
        # Deep with progressive fusion (ALL features used)
        early_outs = []
        late_outs = []
        for i, block in enumerate(self.blocks):
            feat = block(feat)
            if i < 4:
                early_outs.append(feat)
            else:
                late_outs.append(feat)
        
        # Proper fusion (no wasted features!)
        early = self.fuse_early(torch.cat(early_outs, dim=1))
        late = self.fuse_late(torch.cat(late_outs, dim=1))
        feat = self.fuse_final(torch.cat([early, late], dim=1))
        
        # Shallow residual
        feat = feat + shallow
        
        # Refinement
        feat = self.refine(feat)
        
        # Upsample
        feat = self.upsampler(feat)
        
        # Output
        return self.output(feat) + x_up


class TrueMambaBlock(nn.Module):
    """
    True Mamba block combining:
    1. Multi-scale spatial (MCMamba-style)
    2. Selective SSM with SS2D cross-scan
    3. Channel attention (MambaIR-style)
    """
    
    def __init__(self, channels, angRes):
        super(TrueMambaBlock, self).__init__()
        
        # Multi-scale spatial (MCMamba winner design)
        self.ms_spatial = MultiScaleSpatial(channels)
        
        # TRUE selective SSM with SS2D
        self.ssm = SS2DBlock(channels)
        
        # Fusion + Channel attention
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.ca = ChannelAttention(channels)
    
    def forward(self, x):
        # Multi-scale local features
        f_local = self.ms_spatial(x)
        
        # Global features via selective SSM
        f_global = self.ssm(x)
        
        # Fuse
        fused = self.fuse(torch.cat([f_local, f_global], dim=1))
        
        # Channel attention + residual
        return self.ca(fused) + x


class SS2DBlock(nn.Module):
    """
    SS2D: 2D Selective Scan Block (VMamba-style)
    
    Cross-scan strategy:
    1. Unfold image into 4 directional sequences
    2. Apply S6 selective SSM to each
    3. Merge back to 2D
    
    This is TRUE Mamba with proper A/B/C/Δ parameters.
    """
    
    def __init__(self, channels, d_state=16):
        super(SS2DBlock, self).__init__()
        self.channels = channels
        self.d_state = d_state  # SSM state dimension
        
        # Input projection
        self.in_proj = nn.Linear(channels, channels * 2, bias=False)
        
        # S6 core parameters (learned, input-independent base)
        # A: state transition (diagonal for stability)
        self.A_log = nn.Parameter(torch.log(torch.randn(channels, d_state).abs() + 1e-4))
        
        # D: skip connection
        self.D = nn.Parameter(torch.ones(channels))
        
        # Input-dependent parameter generation
        # B, C, Δ are input-dependent (selective)
        self.x_proj = nn.Linear(channels, d_state * 2 + 1, bias=False)  # B, C, Δ
        
        # Output projection
        self.out_proj = nn.Linear(channels, channels, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape to sequence [B, L, C]
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_flat = self.norm(x_flat)
        
        # Input projection: split into x and gate
        xz = self.in_proj(x_flat)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # 4-way cross-scan
        y = self.cross_scan_ssm(x_ssm, H, W)
        
        # Output with gating
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        # Reshape back to 2D
        out = rearrange(y, 'b (h w) c -> b c h w', h=H, w=W)
        
        return out
    
    def cross_scan_ssm(self, x, H, W):
        """Apply SSM in 4 directions and merge."""
        B, L, C = x.shape
        
        # Reshape to 2D
        x_2d = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        # 4 scanning directions
        # 1. Left-to-right, top-to-bottom
        y1 = self.ssm_scan(rearrange(x_2d, 'b h w c -> b (h w) c'))
        
        # 2. Right-to-left, bottom-to-top (reverse)
        y2 = self.ssm_scan(rearrange(x_2d.flip([1, 2]), 'b h w c -> b (h w) c'))
        y2 = rearrange(y2, 'b (h w) c -> b h w c', h=H, w=W).flip([1, 2])
        y2 = rearrange(y2, 'b h w c -> b (h w) c')
        
        # 3. Top-to-bottom, left-to-right (transpose)
        y3 = self.ssm_scan(rearrange(x_2d.transpose(1, 2), 'b w h c -> b (w h) c'))
        y3 = rearrange(y3, 'b (w h) c -> b w h c', w=W, h=H).transpose(1, 2)
        y3 = rearrange(y3, 'b h w c -> b (h w) c')
        
        # 4. Bottom-to-top, right-to-left (transpose + reverse)
        y4 = self.ssm_scan(rearrange(x_2d.transpose(1, 2).flip([1, 2]), 'b w h c -> b (w h) c'))
        y4 = rearrange(y4, 'b (w h) c -> b w h c', w=W, h=H).flip([1, 2]).transpose(1, 2)
        y4 = rearrange(y4, 'b h w c -> b (h w) c')
        
        # Merge (average - can also use learned weights)
        y = (y1 + y2 + y3 + y4) / 4
        
        return y
    
    def ssm_scan(self, x):
        """
        True selective SSM scan with ZOH discretization.
        
        Continuous: dx/dt = Ax + Bu, y = Cx + Du
        Discrete (ZOH): x_k = Ā x_{k-1} + B̄ u_k, y_k = C x_k + D u_k
        
        Where:
            Ā = exp(Δ A)
            B̄ = (Δ A)^{-1} (exp(Δ A) - I) Δ B ≈ Δ B (first-order approx)
        """
        B_batch, L, C = x.shape
        D = self.d_state
        
        # Get base A (negative for stability)
        A = -torch.exp(self.A_log)  # [C, D]
        
        # Generate input-dependent B, C, Δ
        x_proj = self.x_proj(x)  # [B, L, D*2+1]
        
        delta = F.softplus(x_proj[..., :1])  # [B, L, 1] - discretization step
        B_inp = x_proj[..., 1:D+1]  # [B, L, D]
        C_inp = x_proj[..., D+1:2*D+1]  # [B, L, D]
        
        # Discretize A and B using ZOH (first-order approximation)
        # Ā = exp(Δ * A) ≈ 1 + Δ * A (for efficiency)
        # B̄ ≈ Δ * B
        delta = delta.expand(-1, -1, C)  # [B, L, C]
        
        # Sequential scan (true recurrence)
        y = self._selective_scan(x, A, B_inp, C_inp, delta, self.D)
        
        return y
    
    def _selective_scan(self, u, A, B, C, delta, D):
        """
        Selective scan with proper SSM recurrence.
        
        Args:
            u: input [B, L, C]
            A: state matrix [C, D]
            B: input projection [B, L, D]
            C: output projection [B, L, D]
            delta: discretization step [B, L, C]
            D: skip connection [C]
        """
        B_batch, L, C_channels = u.shape
        D_state = A.shape[1]
        
        # Initialize state
        h = torch.zeros(B_batch, C_channels, D_state, device=u.device, dtype=u.dtype)
        
        # Output container
        ys = []
        
        # Discretized A: exp(delta * A) ≈ 1 + delta * A
        # For efficiency, compute per-step
        
        for i in range(L):
            # Get current step
            u_i = u[:, i, :]  # [B, C]
            delta_i = delta[:, i, :]  # [B, C]
            B_i = B[:, i, :]  # [B, D]
            C_i = C[:, i, :]  # [B, D]
            
            # Discretize A: Ā = exp(Δ * diag(A))
            # Simplified: Ā ≈ 1 + Δ * A (first-order Taylor)
            A_bar = 1 + delta_i.unsqueeze(-1) * A.unsqueeze(0)  # [B, C, D]
            
            # Discretize B: B̄ ≈ Δ * B
            B_bar = delta_i.unsqueeze(-1) * B_i.unsqueeze(1)  # [B, C, D]
            
            # State update: h = Ā * h + B̄ * u
            h = A_bar * h + B_bar * u_i.unsqueeze(-1)  # [B, C, D]
            
            # Output: y = C * h + D * u
            y_i = (h * C_i.unsqueeze(1)).sum(-1) + D * u_i  # [B, C]
            
            ys.append(y_i)
        
        y = torch.stack(ys, dim=1)  # [B, L, C]
        return y


class MultiScaleSpatial(nn.Module):
    """Multi-scale spatial (1/3/5/7) matching MCMamba winner."""
    
    def __init__(self, channels):
        super(MultiScaleSpatial, self).__init__()
        
        c_split = channels // 4
        
        self.conv1 = nn.Conv2d(c_split, c_split, 1, bias=False)
        self.conv3 = nn.Conv2d(c_split, c_split, 3, padding=1, groups=c_split, bias=False)
        self.conv5 = nn.Conv2d(c_split, c_split, 5, padding=2, groups=c_split, bias=False)
        self.conv7 = nn.Conv2d(c_split, c_split, 7, padding=3, groups=c_split, bias=False)
        
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        x1, x3, x5, x7 = x.chunk(4, dim=1)
        
        y1 = self.conv1(x1)
        y3 = self.conv3(x3)
        y5 = self.conv5(x5)
        y7 = self.conv7(x7)
        
        y = torch.cat([y1, y3, y5, y7], dim=1)
        y = self.act(self.bn(self.pw(y)))
        
        return y + x


class LocalPixelEnhancement(nn.Module):
    """Local enhancement to address SSM's local forgetting (MambaIR)."""
    
    def __init__(self, channels):
        super(LocalPixelEnhancement, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
    
    def forward(self, x):
        return x + self.conv(x)


class ChannelAttention(nn.Module):
    """Channel attention (MambaIR RSSB style)."""
    
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        
        hidden = max(channels // reduction, 16)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(self.pool(x))


class EfficientUpsampler(nn.Module):
    """PixelShuffle upsampler."""
    
    def __init__(self, channels, scale):
        super(EfficientUpsampler, self).__init__()
        
        if scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        elif scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * scale * scale, 3, padding=1, bias=False),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),
            )
    
    def forward(self, x):
        return self.up(x)


class get_loss(nn.Module):
    """L1 + FFT loss."""
    
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.l1 = nn.L1Loss()
        self.fft_weight = 0.05
    
    def forward(self, SR, HR, criterion_data=[]):
        loss = self.l1(SR, HR)
        
        sr_fft = torch.fft.rfft2(SR)
        hr_fft = torch.fft.rfft2(HR)
        fft_loss = F.l1_loss(torch.abs(sr_fft), torch.abs(hr_fft))
        
        return loss + self.fft_weight * fft_loss


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    class Args:
        angRes_in = 5
        scale_factor = 4
    
    print("=" * 60)
    print("🏆 MyEfficientLFNet v4.2 - TRUE MAMBA CHAMPIONSHIP")
    print("   Based on: VMamba SS2D + MCMamba multi-scale + MambaIR")
    print("=" * 60)
    
    model = get_model(Args())
    params = count_parameters(model)
    
    print(f"\nParams: {params:,} ({params/1_000_000*100:.1f}% of 1M)")
    print(f"Check: {'PASS ✓' if params < 1_000_000 else 'FAIL ✗'}")
    
    # Test forward
    x = torch.randn(1, 1, 5*32, 5*32)
    print(f"\nInput: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Output: {out.shape}")
    print("✓ Forward pass OK")
    
    # FLOPs
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x)
        print(f"\nFLOPs: {flops.total()/1e9:.2f}G ({flops.total()/20e9*100:.1f}% of 20G)")
        print(f"Check: {'PASS ✓' if flops.total() < 20e9 else 'FAIL ✗'}")
    except ImportError:
        print("\n(fvcore not installed)")
    
    print("\n" + "=" * 60)
    print("⚠️  NOTE: Sequential SSM scan is SLOW during training!")
    print("   For production, use CUDA-optimized mamba-ssm package")
    print("=" * 60)
