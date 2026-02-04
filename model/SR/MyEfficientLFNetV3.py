'''
MyEfficientLFNet v3.0 - Championship Architecture for NTIRE 2026 Track 2

Deep Research-Based Design (Jan 31, 2026):
- LGFN-inspired: DGCE, ESAM, ECAM modules (Track 2 winner 2024)
- GhostNet: Cheap feature generation for FLOPs efficiency
- Large Kernel Attention: Decomposed K×1 + 1×K for disparity
- MambaIR: Efficient channel mixing ideas

Target: ~920K params, <20G FLOPs, 32+ dB PSNR (Beat LGFN's 30.05 dB)

Author: NTIRE 2026 Submission - Championship Entry
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class get_model(nn.Module):
    """
    MyEfficientLFNet v3.0 - Championship Architecture
    
    Key innovations over v2:
    1. Ghost Modules (2x more params for same FLOPs)
    2. DGCE: Double-Gated Convolution Extraction
    3. ESAM: Efficient Spatial Attention with Large Kernels
    4. ECAM: Efficient Channel Attention
    5. Wider channels (96 vs 54)
    6. More depth (6 blocks vs 5)
    """
    
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes = args.angRes_in
        self.scale = args.scale_factor
        
        # Architecture hyperparameters - Championship config
        self.channels = 80  # Wider for more capacity
        self.n_blocks = 6   # Deeper for better features
        
        # Shallow feature extraction with Ghost Module
        self.shallow_feat = nn.Sequential(
            GhostModule(1, self.channels, kernel_size=3, ratio=2),
            nn.LeakyReLU(0.1, inplace=True),
            GhostModule(self.channels, self.channels, kernel_size=3, ratio=2),
        )
        
        # Deep feature extraction - LGFM blocks (LGFN-inspired)
        self.blocks = nn.ModuleList([
            LGFMBlock(self.channels, self.angRes)
            for _ in range(self.n_blocks)
        ])
        
        # Global feature aggregation
        self.global_fusion = nn.Sequential(
            nn.Conv2d(self.channels * self.n_blocks, self.channels, 
                     kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Final feature refinement
        self.refine = nn.Sequential(
            GhostModule(self.channels, self.channels, kernel_size=3, ratio=2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # PixelShuffle upsampling
        self.upsampler = EfficientUpsampler(self.channels, self.scale)
        
        # Output projection
        self.output_conv = nn.Conv2d(
            self.channels, 1, kernel_size=3, stride=1,
            padding=1, bias=True
        )
    
    def forward(self, x, info=None):
        """
        Forward pass.
        
        Args:
            x: Input LF [B, 1, angRes*H, angRes*W]
            info: Optional data info (compatibility)
        
        Returns:
            Super-resolved LF [B, 1, angRes*H*scale, angRes*W*scale]
        """
        # Bicubic baseline for residual learning
        x_up = F.interpolate(
            x, scale_factor=self.scale, mode='bicubic', align_corners=False
        )
        
        # Shallow features
        feat = self.shallow_feat(x)
        shallow = feat
        
        # Deep feature extraction with dense connections
        block_outputs = []
        for block in self.blocks:
            feat = block(feat)
            block_outputs.append(feat)
        
        # Aggregate all block outputs
        aggregated = torch.cat(block_outputs, dim=1)
        feat = self.global_fusion(aggregated)
        
        # Residual from shallow
        feat = feat + shallow
        
        # Refinement
        feat = self.refine(feat)
        
        # Upsampling
        feat = self.upsampler(feat)
        
        # Output with global residual
        out = self.output_conv(feat) + x_up
        
        return out


class LGFMBlock(nn.Module):
    """
    Local and Global Feature Module (LGFN-inspired).
    
    Combines three key components:
    1. DGCE: Double-Gated Convolution Extraction (local features)
    2. ESAM: Efficient Spatial Attention (global spatial)
    3. ECAM: Efficient Channel Attention (global channel)
    """
    
    def __init__(self, channels, angRes):
        super(LGFMBlock, self).__init__()
        self.channels = channels
        self.angRes = angRes
        
        # DGCE: Local feature extraction with gating
        self.dgce = DGCE(channels, angRes)
        
        # ESAM: Efficient Spatial Attention
        self.esam = ESAM(channels, angRes)
        
        # ECAM: Efficient Channel Attention
        self.ecam = ECAM(channels)
        
        # Local fusion
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        # DGCE for local features
        local_feat = self.dgce(x)
        
        # ESAM for spatial attention
        spatial_feat = self.esam(local_feat)
        
        # ECAM for channel attention
        channel_feat = self.ecam(spatial_feat)
        
        # Fusion with residual
        out = self.fusion(channel_feat) + x
        
        return out


class GhostModule(nn.Module):
    """
    Ghost Module (GhostNet-inspired).
    
    Generates features using cheap linear operations:
    - Primary features: standard conv on half the channels
    - Ghost features: depthwise conv on primary features
    
    Result: ~2x more features for same FLOPs as standard conv.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, 
                 dilation=1, stride=1):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        self.ratio = ratio
        
        # Primary features (intrinsic)
        self.primary_channels = out_channels // ratio
        self.cheap_channels = out_channels - self.primary_channels
        
        padding = (kernel_size // 2) * dilation
        
        # Primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.primary_channels, kernel_size=1, 
                     stride=stride, bias=False),
            nn.BatchNorm2d(self.primary_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Cheap operation (depthwise conv)
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(self.primary_channels, self.cheap_channels, 
                     kernel_size=kernel_size, stride=1, 
                     padding=padding, dilation=dilation,
                     groups=self.primary_channels, bias=False),
            nn.BatchNorm2d(self.cheap_channels),
        )
    
    def forward(self, x):
        # Primary features
        primary = self.primary_conv(x)
        
        # Ghost features via cheap operation
        cheap = self.cheap_conv(primary)
        
        # Concatenate
        out = torch.cat([primary, cheap], dim=1)
        
        return out


class DGCE(nn.Module):
    """
    Double-Gated Convolution Extraction (LGFN-inspired).
    
    Two levels of gating for adaptive feature modulation:
    1. Spatial gating: location-wise importance
    2. Channel gating: feature-wise importance
    """
    
    def __init__(self, channels, angRes):
        super(DGCE, self).__init__()
        self.channels = channels
        
        # Feature extraction with Ghost modules
        self.extract = nn.Sequential(
            GhostModule(channels, channels, kernel_size=3, ratio=2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Spatial gate
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=angRes,
                     dilation=angRes, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Channel gate (SE-like)
        hidden = max(channels // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        feat = self.extract(x)
        
        # Apply spatial gate
        s_gate = self.spatial_gate(feat)
        feat = feat * s_gate
        
        # Apply channel gate
        c_gate = self.channel_gate(feat)
        feat = feat * c_gate
        
        return feat + x


class ESAM(nn.Module):
    """
    Efficient Spatial Attention Module (LGFN-inspired).
    
    Uses decomposed large-kernel convolution:
    - K×1 + 1×K instead of K×K
    - Captures large receptive field for disparity handling
    - K = 2*angRes + 1 for full angular coverage
    """
    
    def __init__(self, channels, angRes):
        super(ESAM, self).__init__()
        self.channels = channels
        
        # Large kernel size for disparity coverage
        k_size = 2 * angRes + 1  # 11 for angRes=5
        
        # Decomposed large kernel (K×1 + 1×K)
        self.conv_h = nn.Conv2d(
            channels, channels, kernel_size=(1, k_size),
            stride=1, padding=(0, k_size // 2),
            groups=channels, bias=False
        )
        self.conv_v = nn.Conv2d(
            channels, channels, kernel_size=(k_size, 1),
            stride=1, padding=(k_size // 2, 0),
            groups=channels, bias=False
        )
        
        # Pointwise conv for channel mixing
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Attention generation
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Learnable scale
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        # Decomposed large kernel
        h = self.conv_h(x)
        v = self.conv_v(x)
        spatial = h + v
        
        # Channel mixing
        spatial = self.pw_conv(spatial)
        
        # Generate attention
        attn = self.attention(spatial)
        
        # Apply attention with residual
        out = x + self.scale * (x * attn)
        
        return out


class ECAM(nn.Module):
    """
    Efficient Channel Attention Module (LGFN-inspired).
    
    Uses 1D convolution on channel statistics:
    - Global average pooling
    - 1D conv for local channel correlation
    - Efficient O(C) instead of O(C²)
    """
    
    def __init__(self, channels):
        super(ECAM, self).__init__()
        
        # 1D conv for channel correlation
        k_size = max(3, channels // 8)
        if k_size % 2 == 0:
            k_size += 1
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        # 1D conv on channels
        self.conv1d = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=k_size // 2, bias=False
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Global pooling
        y_avg = self.gap(x).view(B, 1, C)  # [B, 1, C]
        y_max = self.gmp(x).view(B, 1, C)  # [B, 1, C]
        
        # 1D conv for channel correlation
        y_avg = self.conv1d(y_avg)
        y_max = self.conv1d(y_max)
        
        # Combine and sigmoid
        y = self.sigmoid(y_avg + y_max).view(B, C, 1, 1)
        
        return x * y


class EfficientUpsampler(nn.Module):
    """
    Efficient PixelShuffle-based upsampler.
    
    Uses Ghost modules for efficient feature expansion.
    """
    
    def __init__(self, channels, scale):
        super(EfficientUpsampler, self).__init__()
        self.scale = scale
        
        if scale == 4:
            self.up = nn.Sequential(
                # First 2x upsample
                nn.Conv2d(channels, channels * 4, kernel_size=3, 
                         padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
                # Second 2x upsample
                nn.Conv2d(channels, channels * 4, kernel_size=3, 
                         padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        elif scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * 4, kernel_size=3, 
                         padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * scale * scale, kernel_size=3, 
                         padding=1, bias=False),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),
            )
    
    def forward(self, x):
        return self.up(x)


class get_loss(nn.Module):
    """
    Enhanced loss function with L1 + Frequency + Edge components.
    """
    
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.freq_weight = 0.05
        self.edge_weight = 0.02
    
    def forward(self, SR, HR, criterion_data=[]):
        # Primary L1 loss
        loss = self.l1_loss(SR, HR)
        
        # Frequency domain loss
        sr_fft = torch.fft.rfft2(SR)
        hr_fft = torch.fft.rfft2(HR)
        freq_loss = F.l1_loss(torch.abs(sr_fft), torch.abs(hr_fft))
        
        # Edge loss (Sobel)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=SR.dtype, device=SR.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=SR.dtype, device=SR.device).view(1, 1, 3, 3)
        
        sr_edge_x = F.conv2d(SR, sobel_x, padding=1)
        sr_edge_y = F.conv2d(SR, sobel_y, padding=1)
        hr_edge_x = F.conv2d(HR, sobel_x, padding=1)
        hr_edge_y = F.conv2d(HR, sobel_y, padding=1)
        
        edge_loss = F.l1_loss(sr_edge_x, hr_edge_x) + F.l1_loss(sr_edge_y, hr_edge_y)
        
        total_loss = loss + self.freq_weight * freq_loss + self.edge_weight * edge_loss
        
        return total_loss


def weights_init(m):
    """Kaiming initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Verification test
    class Args:
        angRes_in = 5
        scale_factor = 4
    
    print("=" * 60)
    print("MyEfficientLFNet v3.0 - Championship Architecture")
    print("=" * 60)
    
    model = get_model(Args())
    params = count_parameters(model)
    
    print(f"\nParameters: {params:,}")
    print(f"Param limit: 1,000,000")
    print(f"Param usage: {params/1_000_000*100:.1f}%")
    print(f"Param check: {'PASS ✓' if params < 1_000_000 else 'FAIL ✗'}")
    
    # Forward pass test
    x = torch.randn(1, 1, 5*32, 5*32)
    with torch.no_grad():
        out = model(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    expected = torch.Size([1, 1, 5*32*4, 5*32*4])
    print(f"Expected:     {expected}")
    
    assert out.shape == expected, "Output shape mismatch!"
    print("✓ Forward pass test PASSED")
    
    # FLOPs check
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x)
        total_flops = flops.total()
        print(f"\nFLOPs: {total_flops/1e9:.2f}G")
        print(f"FLOPs limit: 20G")
        print(f"FLOPs usage: {total_flops/20e9*100:.1f}%")
        print(f"FLOPs check: {'PASS ✓' if total_flops < 20e9 else 'FAIL ✗'}")
    except ImportError:
        print("\n(fvcore not installed, skipping FLOPs check)")
    
    print("\n" + "=" * 60)
    print("🏆 Championship Model Ready!")
    print("=" * 60)
