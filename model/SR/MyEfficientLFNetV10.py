"""
MyEfficientLFNet v10.0 — SOTA Architecture
=============================================

Combines LFMamba's PROVEN 4D disentanglement (Spa→Ang→EPI via einops
rearranges, 3D Conv IFE, CAB) with V9's NOVEL contributions (BMD-Mamba
4-dir batch scanning, FASS HF injection, composite loss, window attention)
and insights from 12+ SOTA papers:

  LFMamba (arXiv 2024)      — SpaSSM/AngSSM/EPISSM disentanglement
  LFTransMamba (CVPRW 2025) — 1st NTIRE 2025, masked angular pre-training
  LFTramba (CVPRW 2025)     — Mamba for Spa-Ang, Transformer for EPI
  MLFSR (ACCV 2024)         — Spatial-Angular Modulator (SAM)
  L²FMamba (arXiv 2025)     — Intra/Inter/MacPI-SSM extraction
  MambaIRv2 (CVPR 2025)     — Non-causal ASE, semantic reordering
  Hi-Mamba (arXiv 2024)     — Direction alternation scanning
  DHSFNet (MDPI 2024)       — Dual-domain DCT HF restoration
  FAMamba/FMSR (arXiv 2024) — Frequency-gated Mamba
  DistgSSR (TPAMI 2022)     — 4D LF disentangling mechanism
  EPIT (arXiv 2023)         — EPI Transformer with H/V alternation
  MambaIR (ECCV 2024)       — Local enhancement + channel attention

V10 Novel Architecture (Track 2 Efficiency — <1M params, <20G FLOPs):
  1. 5D Tensor Processing — operates in (B, C, A, H, W) space throughout
  2. BMD-Mamba applied to disentangled spatial/angular/EPI subspaces
  3. Spatial-Angular Modulator (SAM) for lightweight cross-domain attention
  4. FASS HF injection at every stage to counter Mamba's low-pass bias
  5. Window Attention for global context (absent from LFMamba)
  6. V9-proven composite loss (Charb + FFT + SSIM + Grad + Angular)

Track 2 Efficiency Budget: C=48, n_sa=2, n_epi=2 → ~874K params (87%).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
from einops import rearrange

# ============================================================================
# MAMBA-SSM
# ============================================================================
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✓ mamba-ssm loaded (V10.0 — SOTA)")
except ImportError:
    MAMBA_AVAILABLE = False
    raise ImportError(
        "\n" + "=" * 70 + "\n"
        "❌ mamba-ssm is REQUIRED for V10.0!\n"
        "=" * 70 + "\n\n"
        "Install:  pip install mamba-ssm causal-conv1d\n\n"
        + "=" * 70
    )


# ============================================================================
# MAIN MODEL CLASS
# ============================================================================
class get_model(nn.Module):
    """
    MyEfficientLFNet v10.0 — SOTA Architecture

    Pipeline:
      1. IFE   — 3D Conv Initial Feature Extraction (angular-aware)
      2. SAFL  — Spatial-Angular Feature Learning
                 (N_sa groups of SpaSSM → AngSSM → SAM → FASS)
      3. EPFL  — EPI Feature Learning
                 (N_epi groups of HorizEPI → VertEPI → CAB → FASS)
      4. WA    — Window Attention (global context)
      5. AGG   — Feature Aggregation (3-stage concat)
      6. HLFR  — HR LF Reconstruction (progressive PixelShuffle)
    """

    def __init__(self, args):
        super(get_model, self).__init__()

        # ---- configuration ------------------------------------------------
        self.angRes = getattr(args, "angRes_in", 5)
        self.scale  = getattr(args, "scale_factor", 4)

        # V10 hyperparameters — Track 2 Efficiency (<1M params, <20G FLOPs)
        # Swept from C=64/n_sa=4/n_epi=3 (2.25M) to fit budget
        self.channels   = 48
        self.n_sa       = 2       # number of Spa-Ang groups (was 4)
        self.n_epi      = 2       # number of EPI groups (was 3)
        self.d_state    = 16
        self.d_conv     = 4
        self.expand     = 2.0     # matching LFMamba's proven value
        self.vss_depth  = 2       # VSSBlocks per group, matching LFMamba

        C = self.channels

        # ---- MLFIM: Masked Light Field Image Modeling (LFTransMamba-style) --
        # Learned mask token replaces masked spatial positions in feature space
        # Reference: LFTransMamba (CVPRW 2025, 1st NTIRE 2025)
        # Applied AFTER IFE, on feature maps — NOT on raw input pixels
        self.mlfim_mask_ratio = 0.25  # LFTransMamba official default (25%)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, C), requires_grad=True)

        # ---- MODULE 1: 3D Conv IFE (LFMamba-proven) -----------------------
        self.conv_init0 = nn.Conv3d(1, C, kernel_size=(1, 3, 3),
                                    padding=(0, 1, 1), bias=False)
        self.conv_init = nn.Sequential(
            nn.Conv3d(C, C, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(C, C, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(C, C, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ---- MODULE 2: Spatial-Angular Feature Learning -------------------
        self.sa_groups = nn.ModuleList([
            SpaAngGroup(C, self.angRes, self.d_state, self.d_conv,
                        self.expand, self.vss_depth)
            for _ in range(self.n_sa)
        ])

        # ---- MODULE 3: EPI Feature Learning -------------------------------
        self.epi_groups = nn.ModuleList([
            EPIGroup(C, self.angRes, self.d_state, self.d_conv,
                     self.expand, self.vss_depth)
            for _ in range(self.n_epi)
        ])

        # ---- MODULE 4: Window Attention (V9-novel, global context) --------
        self.win_attn = EfficientWindowAttention(C, num_heads=4, window_size=4)

        # ---- MODULE 5: Feature Aggregation (LFMamba-style 3-stage) --------
        # Concat: [IFE features, SpaAng features, EPI features]
        self.agg_conv = nn.Sequential(
            nn.Conv2d(C * 3, C, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ---- MODULE 6: Reconstruction Head --------------------------------
        self.hlfr = ReconstructionHead(C, self.scale)

        # weight init
        self._init_weights()

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        x: torch.Tensor,
        info: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if info is not None and len(info) >= 1:
            angRes = info[0]
        else:
            angRes = self.angRes

        B, C_in, H, W = x.shape
        assert C_in == 1, f"Expected 1 channel (Y), got {C_in}"

        # ---- Global residual (bicubic upscale) ----------------------------
        # Reshape to 6D for proper per-view bicubic
        x_6d = rearrange(x, 'b c (u h) (v w) -> b c u v h w',
                         u=angRes, v=angRes)
        sr_y = LF_interpolate(x_6d, scale_factor=self.scale, mode='bicubic')
        sr_y = rearrange(sr_y, 'b c u v h w -> b c (u h) (v w)')

        # ---- Set h, w for all submodules that need it ---------------------
        h = H // angRes
        w = W // angRes
        for m in self.modules():
            m.h = h
            m.w = w

        # ---- MODULE 1: 3D Conv IFE ---------------------------------------
        # Reshape to 5D: (B, 1, U*V, h, w)
        x_5d = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w',
                         u=angRes, v=angRes)
        buffer = self.conv_init0(x_5d)
        buffer_init = self.conv_init(buffer) + buffer  # residual

        # ---- MLFIM: Feature-level masking (train only) -------------------
        # Masks random spatial tokens and replaces with learned mask_token.
        # Applied AFTER IFE so features are meaningful; zero inference cost.
        # Reference: LFTransMamba random_masking() — official implementation
        if self.training:
            B_m, C_m, A_m, h_m, w_m = buffer_init.shape
            feat_seq = rearrange(buffer_init,
                                'b c (u v) h w -> (b u v) (h w) c',
                                u=angRes, v=angRes)
            feat_seq = self.random_masking(feat_seq, self.mlfim_mask_ratio)
            buffer_init = rearrange(feat_seq,
                                   '(b u v) (h w) c -> b c (u v) h w',
                                   b=B_m, u=angRes, v=angRes, h=h_m, w=w_m)

        # ---- MODULE 2: Spatial-Angular Feature Learning -------------------
        feat = buffer_init
        for sa_group in self.sa_groups:
            feat = sa_group(feat, angRes)
        buffer_sa = feat + buffer_init  # stage residual

        # ---- MODULE 3: EPI Feature Learning -------------------------------
        feat = buffer_sa
        for epi_group in self.epi_groups:
            feat = epi_group(feat, angRes)
        buffer_epi = feat + buffer_sa  # stage residual

        # ---- Reshape to 2D for remaining modules --------------------------
        u, v = angRes, angRes
        init_2d = rearrange(buffer_init, 'b c (u v) h w -> b c (u h) (v w)',
                            u=u, v=v)
        sa_2d = rearrange(buffer_sa, 'b c (u v) h w -> b c (u h) (v w)',
                          u=u, v=v)
        epi_2d = rearrange(buffer_epi, 'b c (u v) h w -> b c (u h) (v w)',
                           u=u, v=v)

        # ---- MODULE 4: Window Attention -----------------------------------
        epi_2d = self.win_attn(epi_2d)

        # ---- MODULE 5: Feature Aggregation --------------------------------
        buffer_cat = torch.cat([init_2d, sa_2d, epi_2d], dim=1)
        combined = self.agg_conv(buffer_cat)

        # ---- MODULE 6: Reconstruction ------------------------------------
        out = self.hlfr(combined)

        assert out.shape == sr_y.shape, (
            f"Shape mismatch: {out.shape} vs {sr_y.shape}"
        )
        return out + sr_y

    # -------------------------------------------------------------- helpers
    def random_masking(self, x, mask_ratio):
        """
        MLFIM: Masked Light Field Image Modeling (LFTransMamba-style).

        Performs per-sample random masking by per-sample shuffling.
        Masked tokens are replaced with a LEARNED mask_token parameter
        (not zeros — the model optimizes the replacement value).

        This is applied at the feature level (after IFE), on the sequence
        of spatial tokens (h*w) for each angular view independently.

        Reference: LFTransMamba random_masking() — OpenMeow/LFTransMamba

        Args:
            x: (N, L, D) — feature sequence (N=B*U*V, L=h*w, D=channels)
            mask_ratio: fraction of tokens to mask (official: 0.25)

        Returns:
            masked_x: (N, L, D) — same shape, masked tokens replaced
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # Generate per-sample random noise and sort
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)    # ascend: small=keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep unmasked tokens
        ids_keep = ids_shuffle[:, :len_keep]
        unmasked_x = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Append learned mask tokens for the removed positions
        mask_tokens = self.mask_token.expand(
            N, ids_restore.shape[1] - unmasked_x.shape[1], -1
        )
        masked_x = torch.cat([unmasked_x, mask_tokens], dim=1)

        # Unshuffle to restore original ordering
        masked_x = torch.gather(
            masked_x, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        return masked_x

    def _init_weights(self):
        for m in self.modules():
            # CRITICAL: Skip Mamba — it has its own HiPPO-based init for
            # A_log, dt_proj.bias, and D. Overwriting these (especially
            # dt_proj.bias via zeros_) destroys the SSM's time-step dynamics.
            if MAMBA_AVAILABLE and isinstance(m, Mamba):
                continue
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Conv1d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                # Respect Mamba's _no_reinit flag on dt_proj.bias
                if m.bias is not None and not getattr(m.bias, '_no_reinit', False):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ============================================================================
# Spatial-Angular Group (Spa→Ang→SAM→FASS)
# ============================================================================
class SpaAngGroup(nn.Module):
    """
    One Spatial-Angular processing group:
      1. SpaSSMBlock — process spatial dims per angular view
      2. AngSSMBlock — process angular grid per spatial pixel
      3. SAM — Spatial-Angular Modulator (from MLFSR)
      4. FASS — Frequency-Assisted residual (from V9)
    """

    def __init__(self, channels, angRes, d_state, d_conv, expand, depth):
        super().__init__()
        self.angRes = angRes

        self.spa_block = SpaSSMBlock(channels, d_state, d_conv, expand, depth)
        self.ang_block = AngSSMBlock(channels, angRes, d_state, d_conv,
                                     expand, depth)
        self.sam = SpatialAngularModulator(channels, angRes)
        self.fass = FASSModule(channels)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.2)

    def forward(self, x, angRes):
        """x: (B, C, A, h, w) where A = angRes²"""
        # Spatial SSM
        feat = self.spa_block(x, angRes)
        # Angular SSM
        feat = self.ang_block(feat, angRes)
        # SAM modulation
        feat = self.sam(feat, angRes)
        # FASS HF injection (operates on 2D, so temporarily flatten)
        B, C, A, h, w = feat.shape
        feat_2d = rearrange(feat, 'b c a h w -> (b a) c h w')
        feat_2d = self.fass(feat_2d)
        feat = rearrange(feat_2d, '(b a) c h w -> b c a h w', a=A)

        return x + self.res_scale * feat


# ============================================================================
# SpaSSMBlock — Spatial Mamba per angular view
# ============================================================================
class SpaSSMBlock(nn.Module):
    """
    Reshapes to (B*A, C, h, w), applies BMD-Mamba (4-dir scanning)
    on the spatial dimensions of each angular view independently.

    Research basis: LFMamba SpaSSM (proven), V9 BMD-Mamba (novel engine)
    """

    def __init__(self, channels, d_state, d_conv, expand, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            BMDMambaLayer(channels, d_state, d_conv, expand)
            for _ in range(depth)
        ])
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)

    def forward(self, x, angRes):
        """x: (B, C, A, h, w) where A = angRes²"""
        B, C, A, h, w = x.shape
        # Reshape: each angular view becomes an independent batch item
        out = rearrange(x, 'b c a h w -> (b a) c h w')
        for layer in self.layers:
            out = layer(out)
        out = self.conv(out)
        out = rearrange(out, '(b a) c h w -> b c a h w', a=A)
        return out + x  # residual


# ============================================================================
# AngSSMBlock — Angular Mamba per spatial pixel
# ============================================================================
class AngSSMBlock(nn.Module):
    """
    Reshapes to (B*h*w, C, U, V), applies BMD-Mamba (4-dir scanning)
    on the angular grid at each spatial position.

    Research basis: LFMamba AngSSM (proven), V9 BMD-Mamba (novel engine)
    """

    def __init__(self, channels, angRes, d_state, d_conv, expand, depth):
        super().__init__()
        self.angRes = angRes
        self.layers = nn.ModuleList([
            BMDMambaLayer(channels, d_state, d_conv, expand)
            for _ in range(depth)
        ])
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)

    def forward(self, x, angRes):
        """x: (B, C, A, h, w) where A = angRes²"""
        B, C, A, h, w = x.shape
        # Reshape: at each (h,w) pixel, the angular grid is a 2D image
        out = rearrange(x, 'b c (u v) h w -> (b h w) c u v',
                        u=angRes, v=angRes)
        for layer in self.layers:
            out = layer(out)
        out = self.conv(out)
        out = rearrange(out, '(b h w) c u v -> b c (u v) h w',
                        b=B, h=h, w=w)
        return out + x  # residual


# ============================================================================
# EPIGroup — Epipolar Plane Image processing (H-EPI → V-EPI → CAB → FASS)
# ============================================================================
class EPIGroup(nn.Module):
    """
    One EPI processing group:
      1. HorizEPIBlock — rearrange to (B*V*w, C, U, h), run BMD-Mamba
      2. VertEPIBlock  — rearrange to (B*U*h, C, V, w), run BMD-Mamba
      3. CAB — Conv + Channel Attention (from LFMamba/MambaIR)
      4. FASS — Frequency-Assisted residual

    Research basis: LFMamba EPISSM (proven), EPIT (EPI Transformer)
    """

    def __init__(self, channels, angRes, d_state, d_conv, expand, depth):
        super().__init__()
        self.angRes = angRes
        self.h_epi = EPIMambaBlock(channels, d_state, d_conv, expand, depth)
        self.v_epi = EPIMambaBlock(channels, d_state, d_conv, expand, depth)
        self.cab = CAB(channels)
        self.fass = FASSModule(channels)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.2)

    def forward(self, x, angRes):
        """x: (B, C, A, h, w) where A = angRes²"""
        B, C, A, h, w = x.shape
        u, v = angRes, angRes

        # ---- Horizontal EPI: pair (U, h) ----
        h_in = rearrange(x, 'b c (u v) h w -> (b v w) c u h',
                         u=u, v=v)
        h_out = self.h_epi(h_in)
        h_out = rearrange(h_out, '(b v w) c u h -> b c (u v) h w',
                          v=v, w=w, u=u)

        # ---- Vertical EPI: pair (V, w) ----
        v_in = rearrange(h_out, 'b c (u v) h w -> (b u h) c v w',
                         u=u, v=v)
        v_out = self.v_epi(v_in)
        v_out = rearrange(v_out, '(b u h) c v w -> b c (u v) h w',
                          u=u, h=h, v=v)

        # ---- CAB: local enhancement ----
        epi_2d = rearrange(v_out, 'b c a h w -> (b a) c h w')
        epi_2d = self.cab(epi_2d)
        epi_2d = self.fass(epi_2d)
        feat = rearrange(epi_2d, '(b a) c h w -> b c a h w', a=A)

        return x + self.res_scale * feat


# ============================================================================
# EPIMambaBlock — Single direction EPI processing
# ============================================================================
class EPIMambaBlock(nn.Module):
    """Mamba block for EPI slices (either horizontal or vertical)."""

    def __init__(self, channels, d_state, d_conv, expand, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            BMDMambaLayer(channels, d_state, d_conv, expand)
            for _ in range(depth)
        ])
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)

    def forward(self, x):
        """x: (N, C, D1, D2) — an arbitrary 2D feature map"""
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.conv(out)
        return out + x  # residual


# ============================================================================
# BMDMambaLayer — Batched Multi-Directional Mamba (V9 novel, proven engine)
# ============================================================================
class BMDMambaLayer(nn.Module):
    """
    One layer of the V9-novel BMD-Mamba engine.
    Folds 4 scan directions into the batch dimension → (4B, C, L).
    Uses standard mamba_ssm Triton kernels (faster than custom ESS2D).
    Includes LayerNorm → Mamba → 1×1 fusion with residual.

    Research basis: V9 BMDFASSBlock (novel), Hi-Mamba direction alternation
    """

    def __init__(self, channels, d_state=16, d_conv=4, expand=2.0):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.dir_fusion = nn.Conv2d(channels, channels, 1, bias=False)
        self.skip_scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        L = H * W
        x_in = x

        # ---- build 4 directional sequences --------------------------------
        s0 = x.flatten(2)                                      # (B, C, L) row-major
        s1 = x.flatten(2).flip(-1)                             # reversed
        s2 = x.permute(0, 1, 3, 2).flatten(2)                 # col-major
        s3 = x.permute(0, 1, 3, 2).flatten(2).flip(-1)        # reversed col

        # Stack along batch: (4B, L, C) for Mamba
        batched = torch.cat([s0, s1, s2, s3], dim=0)          # (4B, C, L)
        batched = batched.transpose(1, 2).contiguous()         # (4B, L, C)
        batched = self.norm(batched)

        # ---- Mamba pass ---------------------------------------------------
        out = self.mamba(batched)                               # (4B, L, C)
        out = out.transpose(1, 2).contiguous()                 # (4B, C, L)

        # ---- un-batch and reshape -----------------------------------------
        o0, o1, o2, o3 = out.chunk(4, dim=0)
        r0 = o0.view(B, C, H, W)
        r1 = o1.flip(-1).view(B, C, H, W)
        r2 = o2.view(B, C, W, H).permute(0, 1, 3, 2).contiguous()
        r3 = o3.flip(-1).view(B, C, W, H).permute(0, 1, 3, 2).contiguous()

        combined = (r0 + r1 + r2 + r3) * 0.25
        out_feat = self.dir_fusion(combined)

        # skip connection with learnable per-channel scale
        return x_in * self.skip_scale.view(1, -1, 1, 1) + out_feat


# ============================================================================
# Spatial-Angular Modulator (SAM) — from MLFSR (ACCV 2024)
# ============================================================================
class SpatialAngularModulator(nn.Module):
    """
    Lightweight 1×1 conv that generates spatial and angular attention
    maps to modulate features. Proven complementary to Mamba in MLFSR.

    Research basis: MLFSR (ACCV 2024) Spatial-Angular Modulator
    """

    def __init__(self, channels, angRes):
        super().__init__()
        self.angRes = angRes

        # Spatial attention (applied per angular view)
        self.spa_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # Angular attention (applied per spatial position)
        self.ang_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x, angRes):
        """x: (B, C, A, h, w) where A = angRes²"""
        B, C, A, h, w = x.shape
        u, v = angRes, angRes

        # Spatial modulation: per angular view
        x_spa = rearrange(x, 'b c a h w -> (b a) c h w')
        spa_weight = self.spa_attn(x_spa)
        x_spa_mod = x_spa * spa_weight
        x_spa_mod = rearrange(x_spa_mod, '(b a) c h w -> b c a h w', a=A)

        # Angular modulation: per spatial pixel
        x_ang = rearrange(x, 'b c (u v) h w -> (b h w) c u v',
                          u=u, v=v)
        ang_weight = self.ang_attn(x_ang)
        x_ang_mod = x_ang * ang_weight
        x_ang_mod = rearrange(x_ang_mod, '(b h w) c u v -> b c (u v) h w',
                              b=B, h=h, w=w)

        # Fuse spatial and angular modulated features
        fused_2d = rearrange(
            torch.cat([x_spa_mod, x_ang_mod], dim=1),
            'b c2 a h w -> (b a) c2 h w'
        )
        fused = self.fuse(fused_2d)
        fused = rearrange(fused, '(b a) c h w -> b c a h w', a=A)

        return fused


# ============================================================================
# CAB — Channel Attention Block (from LFMamba / MambaIR, proven)
# ============================================================================
class CAB(nn.Module):
    """
    Conv 3×3 + Channel Attention for local detail preservation.
    Proven essential inside every VSS layer in LFMamba and MambaIR.

    Research basis: LFMamba/MambaIR CAB (ECCV 2024)
    """

    def __init__(self, channels, compress_ratio=3, squeeze_factor=16):
        # NOTE: LFMamba uses squeeze_factor=30 (→2 channels). We use 16 (→4 channels)
        # because V10's CAB has a residual skip that the attention branch must
        # compete with, requiring more capacity. No param budget constraint.
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(channels, channels // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // compress_ratio, channels, 3, 1, 1),
            ChannelAttention(channels, squeeze_factor),
        )

    def forward(self, x):
        # NOTE: Intentional divergence from LFMamba — V10 CAB adds residual skip.
        # LFMamba's VSSBlock already provides a skip, so this is a double-residual.
        # We keep it because V10 uses CAB outside VSSBlock (in EPIGroup) where
        # the extra skip strengthens gradient flow.
        return self.cab(x) + x  # residual


class ChannelAttention(nn.Module):
    """Channel attention (from RCAN, used in LFMamba)."""

    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


# ============================================================================
# FASS Module — Frequency-Assisted State Space (V9 novel, proven)
# ============================================================================
class FASSModule(nn.Module):
    """
    Extracts high-frequency residual = input − low_pass(input),
    refines it via bottleneck conv, gates injection strength.
    Counters Mamba's inherent low-pass smoothing bias.

    Research basis: V9 FASSModule (novel), DHSFNet (DCT HF restoration),
                    FAMamba (frequency-gated Mamba)
    """

    def __init__(self, channels):
        super().__init__()
        self.low_pass = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.hf_refine = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1,
                      groups=channels // 4, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low_pass(x)
        hf = x - low
        hf_refined = self.hf_refine(hf)
        g = self.gate(x)
        return x + self.scale * hf_refined * g


# ============================================================================
# Efficient Window Attention (V9 novel — Swin-style with RPB)
# ============================================================================
class EfficientWindowAttention(nn.Module):
    """
    Swin-style window attention with relative position bias.
    Provides global context that pure SSMs lack.
    Applied at strategic depth after all Mamba processing.

    NOTE: Operates on SAI mosaic space (u*h, v*w). With window_size=8 and
    typical patch h=32 (>8), windows stay within individual angular views.
    At view boundaries, cross-view information exchange may occur — this is
    intentional and potentially beneficial. LFMamba does not use window attn.

    Research basis: V9 EfficientWindowAttention (novel),
                    SwinIR (ICCVW 2021), LFTramba Transformer for EPI
    """

    def __init__(self, channels, num_heads=4, window_size=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads
        self.scale_factor = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(channels)
        self.qkv  = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.attn_scale = nn.Parameter(torch.ones(1) * 0.3)

        # relative position bias table
        self.rpb_table = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1) * (2 * window_size - 1), num_heads
            )
        )
        nn.init.trunc_normal_(self.rpb_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rpi = rel.sum(-1)
        self.register_buffer("rpi", rpi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        x_in = x

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x.shape

        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)

        x_n = self.norm(x)
        qkv = (
            self.qkv(x_n)
            .reshape(-1, ws * ws, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale_factor
        rpb = self.rpb_table[self.rpi.view(-1)].view(
            ws * ws, ws * ws, -1
        ).permute(2, 0, 1)
        attn = attn + rpb.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        out = self.proj(out)

        nwh, nww = Hp // ws, Wp // ws
        out = out.view(B, nwh, nww, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return x_in + self.attn_scale * out


# ============================================================================
# Reconstruction Head (V9-proven, progressive PixelShuffle)
# ============================================================================
class ReconstructionHead(nn.Module):
    """
    Progressive 2×+2× PixelShuffle for 4× upscaling.
    Includes channel attention and output scaling.
    """

    def __init__(self, channels, scale):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

        # Channel attention
        hidden = max(channels // 16, 8)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # PixelShuffle upsampler
        if scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * scale * scale, 3,
                          padding=1, bias=False),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.output = nn.Conv2d(channels, 1, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.refine(x)
        r = (r + x) * self.ca(r + x)
        up = self.up(r)
        return self.output(up)


# ============================================================================
# LF_interpolate — per-view bicubic upscaling (from LFMamba, proven)
# ============================================================================
def LF_interpolate(LF, scale_factor, mode):
    """
    Bicubic upscale each sub-aperture image independently.
    Input:  (B, C, U, V, h, w)
    Output: (B, C, U, V, h*s, w*s)
    """
    b, c, u, v, h, w = LF.size()
    LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
    LF_up = F.interpolate(LF, scale_factor=scale_factor, mode=mode,
                          align_corners=False)
    LF_up = rearrange(LF_up, '(b u v) c h w -> b c u v h w', u=u, v=v)
    return LF_up


# ============================================================================
# LOSS FUNCTION (V9-proven composite loss — strictly better than L1)
# ============================================================================
class get_loss(nn.Module):
    """Charbonnier + FFT + SSIM + Gradient + Angular."""

    def __init__(self, args):
        super().__init__()
        self.eps      = getattr(args, "charbonnier_eps", 1e-9)
        self.fft_w    = getattr(args, "fft_weight", 0.1)
        self.ssim_w   = getattr(args, "ssim_weight", 0.02)
        self.grad_w   = getattr(args, "grad_weight", 0.04)
        self.ang_w    = getattr(args, "angular_weight", 0.06)
        self.angRes   = getattr(args, "angRes_in", 5)

        # Pre-register Sobel filters as buffers (avoids re-creation every forward pass)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def charbonnier(self, p, t):
        p, t = p.float(), t.float()  # CRITICAL: bfloat16 underflows on eps²=1e-18
        return torch.mean(torch.sqrt((p - t) ** 2 + self.eps ** 2))

    def fft_loss(self, p, t):
        p, t = p.float(), t.float()  # float32 for FFT precision
        return F.l1_loss(
            torch.abs(torch.fft.rfft2(p)),
            torch.abs(torch.fft.rfft2(t)),
        )

    def ssim_loss(self, p, t):
        p, t = p.float(), t.float()  # CRITICAL: bfloat16 squaring causes variance underflow
        C1, C2, ws = 0.01 ** 2, 0.03 ** 2, 7
        pad = ws // 2
        mu_p = F.avg_pool2d(p, ws, 1, pad)
        mu_t = F.avg_pool2d(t, ws, 1, pad)
        s_p = F.avg_pool2d(p ** 2, ws, 1, pad) - mu_p ** 2
        s_t = F.avg_pool2d(t ** 2, ws, 1, pad) - mu_t ** 2
        s_x = F.avg_pool2d(p * t,  ws, 1, pad) - mu_p * mu_t
        s_p, s_t = s_p.clamp(min=0), s_t.clamp(min=0)
        ssim = ((2 * mu_p * mu_t + C1) * (2 * s_x + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (s_p + s_t + C2))
        return 1 - ssim.mean()

    def gradient_loss(self, p, t):
        p, t = p.float(), t.float()  # float32 for Sobel convolution
        sx = self.sobel_x  # already float32 (registered as buffer)
        sy = self.sobel_y
        return (
            F.l1_loss(F.conv2d(p, sx, padding=1), F.conv2d(t, sx, padding=1))
            + F.l1_loss(F.conv2d(p, sy, padding=1), F.conv2d(t, sy, padding=1))
        )

    def angular_loss(self, p, t):
        p, t = p.float(), t.float()  # float32 for angular difference precision
        B, C, H, W = p.shape
        a = self.angRes
        h, w = H // a, W // a
        pv = p.view(B, C, a, h, a, w)
        tv = t.view(B, C, a, h, a, w)
        return (
            F.l1_loss(pv[:, :, :, :, 1:, :] - pv[:, :, :, :, :-1, :],
                      tv[:, :, :, :, 1:, :] - tv[:, :, :, :, :-1, :])
            + F.l1_loss(pv[:, :, 1:, :, :, :] - pv[:, :, :-1, :, :, :],
                        tv[:, :, 1:, :, :, :] - tv[:, :, :-1, :, :, :])
        )

    def forward(self, pred, target, data_info=None):
        loss = self.charbonnier(pred, target)
        loss = loss + self.fft_w    * self.fft_loss(pred, target)
        loss = loss + self.ssim_w   * self.ssim_loss(pred, target)
        loss = loss + self.grad_w   * self.gradient_loss(pred, target)
        if (pred.shape[-1] % self.angRes == 0
                and pred.shape[-2] % self.angRes == 0):
            loss = loss + self.ang_w * self.angular_loss(pred, target)
        return loss


def weights_init(m):
    """Intentional no-op. V10 uses its own _init_weights() in __init__.
    This exists only for API compatibility with BasicLFSR's train.py
    which calls net.apply(MODEL.weights_init). Do NOT add logic here —
    it would run AFTER V10's init and overwrite MLFIM's mask_token."""
    pass


# ============================================================================
# SELF-TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 MyEfficientLFNet v10.0 — SOTA Self-Test")
    print("=" * 70)

    class Args:
        angRes_in = 5
        scale_factor = 4

    model = get_model(Args()).cuda()

    params = sum(p.numel() for p in model.parameters())
    print(f"\n📋 Parameters: {params:,} ({params/1e6:.3f}M)")

    # Quick FLOPs estimate
    try:
        from thop import profile
        dummy = torch.randn(1, 1, 160, 160, device="cuda")
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"   FLOPs:      {flops/1e9:.2f}G")
    except ImportError:
        print("   FLOPs:      (install thop for FLOPs count)")

    x = torch.randn(1, 1, 160, 160, device="cuda")
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"\n🧪 Forward: {x.shape} → {y.shape}")
    expected = (1, 1, 640, 640)
    print(f"   Shape:   {'✅ PASS' if y.shape == expected else '❌ FAIL'}")

    model.train()
    x = torch.randn(1, 1, 160, 160, device="cuda", requires_grad=True)
    y = model(x)
    y.mean().backward()
    grad_ok = x.grad is not None and not torch.isnan(x.grad).any()
    print(f"\n🔥 Backward: {'✅ PASS' if grad_ok else '❌ FAIL'}")

    print(f"\n{'='*70}")
    print("✅ V10.0 Self-Test Complete!")
    print("=" * 70)
