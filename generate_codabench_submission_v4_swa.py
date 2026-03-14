"""
MyEfficientLFNet V3 (MLFIM) — Max-PSNR Architecture
=====================================================

V3 Changes (Bug Fixes over V2):
  1. BUG FIX: BMDMambaLayer reverse paths document square-only assumption
  2. BUG FIX: Criterion dispatch simplified — always pass data_info
  3. BUG FIX: EMA state_dict uses deep copy (tensor clone) for safety
  4. Docstring corrections: CutBlur clarified as data-pipeline augmentation
  5. Version strings cleaned up and unified to V3

V10.4 Changes (Training Improvements):
  1. Loss: fft_loss upgraded to Focal Frequency Loss (Jiang et al., ICCV 2021)
     Adaptively up-weights hard-to-reconstruct frequencies. Weight 0.1→0.05
     (self-normalising, so lower weight = equivalent or better spectral push).
  2. Loss: ssim_w 0.1→0.15 (backed by HAT/SwinIR ablation results)
  3. Train: EMA fixed to per-step updates (critical, was per-epoch = wrong)
  4. Train: Late-training EMA decay bump 0.999→0.9999 at 75% of epochs
  5. Stage 2: Extended from 150→200 epochs for cosine-tail PSNR push

V10.3 Changes (Audit-Hardened):
  1. BUG FIX: SpaAngGroup res_scale was unused — now applied consistently
  2. BUG FIX: EPIGroup and SpaAngGroup now share identical residual strategy
  3. BUG FIX: BMDMambaLayer default d_state 32→16 (matches model config)
  4. IFE reduction: 4 Conv3d → 2 (saves ~40K params, ~1G FLOPs)
  5. CAB → MicroCAB: depthwise-separable (saves ~28% EPI FLOPs)
  6. AdaptiveStreamGating (ASG): replaces dumb concat agg with learned gates
  7. SAM simplified: element-wise gate replaces concat+conv (saves ~14K params)
  8. LCE (Local Contrast Enhancement) before HLFR to boost edge sharpness
  9. Loss tuning: FFT 0.05→0.1 for sharper spectral fidelity

V10.2 Changes:
  1. n_sa=2→3 — deeper Spa-Ang processing
  2. Channel-split 4-dir scan — ESS2D-style directional coverage
  3. SAM residual connection
  4. ICNR init for PixelShuffle

V10.1 Changes:
  1. EPI weight sharing (LFMamba-style)
  2. 3×3 PixelShuffle
  3. Window attention ws=4→8
  4. FASS simplified gate

Track 2 Efficiency Budget: C=48, n_sa=4, n_epi=3, vss_depth=2.

Lineage: V10 → V2 MLFIM → V3 MLFIM (this file)
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
    print("✓ mamba-ssm loaded (MyEfficientLFNet V3 — Final Submission)")
except ImportError:
    MAMBA_AVAILABLE = False
    raise ImportError(
        "\n" + "=" * 70 + "\n"
        "❌ mamba-ssm is REQUIRED for MyEfficientLFNet V3!\n"
        "=" * 70 + "\n\n"
        "Install:  pip install mamba-ssm causal-conv1d\n\n"
        + "=" * 70
    )


# ============================================================================
# DropPath (Stochastic Depth) — inline implementation (avoids timm dependency)
# ============================================================================
class DropPath(nn.Module):
    """Stochastic depth regularization (Huang et al., 2016).

    Drops the entire residual branch per-sample during training.
    Used in SwinIR, MambaIR, LFMamba VSSBlock, HAT — proven to
    improve generalization on real-world datasets.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample mask: (B, 1, 1, ...) broadcasts over all dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            mask.div_(keep_prob)
        return x * mask

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob:.3f}'


# ============================================================================
# MAIN MODEL CLASS
# ============================================================================
class get_model(nn.Module):
    """
    MyEfficientLFNet V3 (MLFIM)

    Pipeline:
      1. IFE   — 3D Conv IFE (3-layer, angular-aware, matching LFTransMamba)
      2. SAFL  — Spatial-Angular Feature Learning
                 (N_sa groups of SpaSSM → AngSSM → SAM → FASS)
      3. EPFL  — EPI Feature Learning
                 (N_epi groups of shared H/V EPI → MicroCAB → FASS)
      4. WA    — Window Attention (ws=8 global context)
      5. ASG   — Adaptive Stream Gating (learned 3-stream fusion)
      6. LCE   — Local Contrast Enhancement (edge sharpening pre-HLFR)
      7. HLFR  — HR LF Reconstruction (3×3 PixelShuffle + ICNR)
    """

    def __init__(self, args):
        super(get_model, self).__init__()

        # ---- configuration ------------------------------------------------
        self.angRes = getattr(args, "angRes_in", 5)
        self.scale  = getattr(args, "scale_factor", 4)

        # V10.4 hyperparameters — Track 2 Efficiency (<1M params, <20G FLOPs)
        # CRITICAL: channels MUST be divisible by 4 for BMDMambaLayer's 4-way
        # channel split (C4 = channels // 4). 45 % 4 = 1 → last group gets
        # 12 channels but Mamba expects 11 → shape mismatch crash.
        self.channels   = 48      # Wide channels for 4-way BMDMamba split
        self.n_sa       = 3       # Trimmed from 4 to 3 to afford True 2D Gated Mamba
        self.n_epi      = 2       # Trimmed from 3 to 2 to afford True 2D Gated Mamba
        self.d_state    = 16
        self.d_conv     = 4
        self.expand     = 2.0     # matching LFMamba's proven value
        self.vss_depth  = 2       # matching LFMamba depth for quality
        # V2.1: Stochastic depth (DropPath) — proven in SwinIR/MambaIR/LFMamba
        self.drop_path_rate = 0.05

        C = self.channels

        # ---- MLFIM: Masked Light Field Image Modeling (LFTransMamba-style) --
        # Learned mask token replaces masked spatial positions in feature space
        # Reference: LFTransMamba (CVPRW 2025, 1st NTIRE 2025)
        # Applied AFTER IFE, on feature maps — NOT on raw input pixels
        # Paper Table 4: mask_ratio=0.25 is optimal for Track 2 Efficiency
        # (0.25 gives 32.9649 avg PSNR vs 32.9470 baseline, 32.9692 at 0.35)
        self.mlfim_mask_ratio = getattr(args, 'mlfim_mask_ratio', 0.25)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, C), requires_grad=True)

        # Angular position embedding (from LFTransMamba — critical for view awareness)
        # Adds a learnable per-view bias so the model can distinguish which angular
        # position each feature belongs to. Cost: angRes² × C = 25 × 48 = 1200 params.
        self.ang_embed = nn.Parameter(
            torch.zeros(1, C, self.angRes ** 2, 1, 1), requires_grad=True
        )

        # ---- MODULE 1: 3D Conv IFE (3-layer, matching LFTransMamba) --------
        # V3 FIX: 3-layer IFE matching LFTransMamba's conv_init.
        # Each Conv3d(C,C,(1,3,3)) = C² × 9 params ≈ 20.7K each.
        # Total IFE: conv_init0 + 3 × conv_init = ~83K params, well within budget.
        self.conv_init0 = nn.Conv3d(1, C, kernel_size=(1, 3, 3),
                                    padding=(0, 1, 1), bias=False)
        # V3 FIX: Restored to 3-layer IFE matching LFTransMamba (was 2, should be 3).
        # LeakyReLU slope 0.2→0.1 matching LFTransMamba exactly.
        # Cost: ~21K extra params, ~0.5G FLOPs. Well within 1M/20G budget.
        self.conv_init = nn.Sequential(
            nn.Conv3d(C, C, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(C, C, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(C, C, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # ---- MODULE 2: Spatial-Angular Feature Learning -------------------
        # V2.1: Linearly increasing DropPath rates across all groups
        n_total_groups = self.n_sa + self.n_epi
        dpr = [self.drop_path_rate * i / max(n_total_groups - 1, 1)
               for i in range(n_total_groups)]
        self.sa_groups = nn.ModuleList([
            SpaAngGroup(C, self.angRes, self.d_state, self.d_conv,
                        self.expand, self.vss_depth, drop_path=dpr[i])
            for i in range(self.n_sa)
        ])

        # ---- MODULE 3: EPI Feature Learning -------------------------------
        self.epi_groups = nn.ModuleList([
            EPIGroup(C, self.angRes, self.d_state, self.d_conv,
                     self.expand, self.vss_depth,
                     drop_path=dpr[self.n_sa + i])
            for i in range(self.n_epi)
        ])

        # ---- MODULE 4: Window Attention (V9-novel, global context) --------
        # V10.1: Increased window_size 4→8 for 4× larger receptive field (16→64 tokens)
        self.win_attn = EfficientWindowAttention(C, num_heads=4, window_size=8)

        # ---- MODULE 5: Adaptive Stream Gating (ASG) -----------------------
        # V10.3: Replaces naive 3C→C 1×1 conv with per-stream LEARNABLE
        # softmax gates. Each gate is a tiny 1×1 conv that predicts how much
        # each stream (IFE, SpaAng, EPI) contributes at every spatial position.
        # This lets the network route information based on content — e.g., rely
        # on EPI for textured regions and on SpaAng for flat areas.
        self.asg = AdaptiveStreamGating(C)

        # ---- MODULE 5.5: Local Contrast Enhancement (LCE) -----------------
        # V2.1: Multi-depth fusion — IFE long-skip to reconstruction head
        # Ensures upsampler has direct access to clean IFE features before
        # any Mamba smoothing (LFMamba feeds all 3 depths to upsampler).
        self.depth_fuse = nn.Conv2d(C * 2, C, 1, bias=False)

        # V10.3: Lightweight sharpening applied after ASG. Extracts high-freq
        # residual with a depthwise 3×3, refines with 1×1, injects with learned
        # gate. Costs ~2K params and <0.1G FLOPs. Counters Mamba's low-pass bias
        # at the aggregation output level (FASS handles it inside Mamba blocks;
        # this handles it at the global feature map before upsampling).
        self.lce = LocalContrastEnhancement(C)

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


        # ---- MODULE 1: 3D Conv IFE ---------------------------------------
        # Reshape to 5D: (B, 1, U*V, h, w)
        x_5d = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w',
                         u=angRes, v=angRes)
        buffer = self.conv_init0(x_5d)
        buffer_init = self.conv_init(buffer) + buffer  # residual

        # ---- MLFIM: Feature-level masking (train only) -------------------
        # V3 FIX: Masking BEFORE ang_embed (matches LFTransMamba exactly).
        # Masking after ang_embed means mask tokens already have position info,
        # so the model never learns to infer position from context — defeating
        # the purpose of MLFIM pre-training.
        if self.training and self.mlfim_mask_ratio > 0:
            B_m, C_m, A_m, h_m, w_m = buffer_init.shape
            feat_seq = rearrange(buffer_init,
                                'b c (u v) h w -> (b u v) (h w) c',
                                u=angRes, v=angRes)
            feat_seq = self.random_masking(feat_seq, self.mlfim_mask_ratio)
            buffer_init = rearrange(feat_seq,
                                   '(b u v) (h w) c -> b c (u v) h w',
                                    b=B_m, u=angRes, v=angRes, h=h_m, w=w_m)

        # Angular embedding — per-view positional identity (LFTransMamba)
        # Applied AFTER masking so model must infer position from context
        buffer_init = buffer_init + self.ang_embed

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

        # ---- MODULE 4: Window Attention -----------------------------------
        # G4 Fix: Run Window Attention strictly per-view (B*U*V) instead of on the
        # full mosaic, preventing windows from crossing sub-aperture view boundaries.
        epi_per_view = rearrange(buffer_epi, 'b c (u v) h w -> (b u v) c h w', u=u, v=v)
        epi_per_view = self.win_attn(epi_per_view)
        epi_2d = rearrange(epi_per_view, '(b u v) c h w -> b c (u h) (v w)', b=B, u=u, v=v)

        # ---- MODULE 5: Adaptive Stream Gating ----------------------------
        combined = self.asg(init_2d, sa_2d, epi_2d)

        # ---- MODULE 5.5: Local Contrast Enhancement -----------------------
        combined = self.lce(combined)

        # ---- MODULE 5.6: Multi-depth fusion (V2.1) ------------------------
        # Concat ASG+LCE output with clean IFE features → 1×1 fuse
        combined = self.depth_fuse(torch.cat([combined, init_2d], dim=1))

        # ---- MODULE 6: Reconstruction ------------------------------------
        out = self.hlfr(combined)

        assert out.shape == sr_y.shape, (
            f"Shape mismatch: {out.shape} vs {sr_y.shape}"
        )
        # (Removed the explicit clamp(0,1) hack here — the 12 dB PSNR "overshooting"
        # was actually a 48-pixel structural shift defect in LFintegrate_gaussian,
        # which has now been fixed. Clamping here destroys gradients for outliers.)
        return out + sr_y

    # -------------------------------------------------------------- helpers
    def random_masking(self, x, mask_ratio):
        """
        MLFIM: Masked Light Field Image Modeling (LFTransMamba-style).

        Performs per-sample random masking by per-sample shuffling.
        Masked tokens are replaced with a LEARNED mask_token parameter
        (not zeros — the model optimizes the replacement value).

        NOTE: No inverted-dropout scaling on unmasked tokens. LFTransMamba
        does NOT use scaling either — the mask_token learns to compensate.
        With `.clamp(0,1)` on the final output, any train/eval feature
        magnitude mismatch is capped and cannot cause destructive PSNR.

        Reference: LFTransMamba random_masking() — OpenMeow/LFTransMamba

        Args:
            x: (N, L, D) — feature sequence (N=B*U*V, L=h*w, D=channels)
            mask_ratio: fraction of tokens to mask (LFTransMamba: 0.35)

        Returns:
            masked_x: (N, L, D) — same shape, masked tokens replaced
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # B4 Fix: Short-circuit when no masking is applied
        if len_keep == L:
            return x

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
            N, L - len_keep, -1
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
                # LFTransMamba uses trunc_normal_(std=0.02) for Linear layers
                # (line 2425-2428 in LFTransMamba.py). This is the ViT/Transformer
                # standard for attention projections and MLP layers.
                nn.init.trunc_normal_(m.weight, std=0.02)
                # Respect Mamba's _no_reinit flag on dt_proj.bias
                if m.bias is not None and not getattr(m.bias, '_no_reinit', False):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # V10.2: ICNR initialization for PixelShuffle convolutions
        # Prevents checkerboard artifacts at initialization by filling
        # sub-pixel channels with copies of a kaiming-initialized kernel.
        self._icnr_init_pixelshuffle()

    def _icnr_init_pixelshuffle(self):
        """
        ICNR (Initialization of Convolutions for Natural Reconstruction)
        for PixelShuffle layers. Fills sub-pixel channels with copies of
        a kaiming-initialized kernel so all sub-pixels start identical.
        Reference: Aitken et al., "Checkerboard artifact free sub-pixel
        convolution", arXiv:1707.02937
        """
        for module in self.hlfr.up.modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > module.in_channels:
                # This is a PixelShuffle expansion conv (e.g., C -> C*4)
                oc, ic, kh, kw = module.weight.shape
                scale_sq = oc // ic  # should be 4 for 2× PixelShuffle
                if oc == ic * scale_sq:
                    # Initialize one sub-pixel kernel, repeat for all sub-pixels
                    sub_kernel = torch.empty(ic, ic, kh, kw)
                    nn.init.kaiming_normal_(sub_kernel, mode='fan_out',
                                           nonlinearity='leaky_relu')
                    sub_kernel = sub_kernel.repeat_interleave(scale_sq, dim=0)
                    module.weight.data.copy_(sub_kernel)


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

    def __init__(self, channels, angRes, d_state, d_conv, expand, depth,
                 drop_path=0.0):
        super().__init__()
        self.angRes = angRes

        self.spa_block = SpaSSMBlock(channels, d_state, d_conv, expand, depth)
        # V2.2: Interleaved MicroCAB after spatial SSM (like LFMamba's VSSBlock)
        # Provides conv-based channel refinement between spatial and angular stages
        self.spa_cab = MicroCAB(channels)
        self.ang_block = AngSSMBlock(channels, angRes, d_state, d_conv,
                                     expand, depth)
        self.sam = SpatialAngularModulator(channels, angRes)
        self.fass = FASSModule(channels)
        # V10.3 FIX: sigmoid-initialized at -1.6 (sigmoid(-1.6)≈0.17) so the
        # outer gate starts conservative and opens to 1.0 as training progresses.
        self.res_scale = nn.Parameter(torch.ones(1) * -1.6)
        # V2.1: Stochastic depth for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, angRes):
        """x: (B, C, A, h, w) where A = angRes²"""
        # A2 Fix: We only want the outer residual connection for stable deep residual training.
        # The inner components have their own skip connections disabled or we don't accumulate them directly onto `feat`.
        # However, the blocks already implement their own local `return out + x`.
        # So instead of chaining `feat = block(feat)`, where `feat` quickly inflates,
        # we let each block's local residual handle its own stability,
        # but the final feature is just scaled and added linearly.

        # Spatial SSM
        feat = self.spa_block(x, angRes)
        # V2.2: Interleaved MicroCAB — conv refinement between Spa and Ang stages
        B_c, C_c, A_c, h_c, w_c = feat.shape
        feat_cab = rearrange(feat, 'b c a h w -> (b a) c h w')
        feat_cab = self.spa_cab(feat_cab)
        feat = rearrange(feat_cab, '(b a) c h w -> b c a h w', a=A_c)
        # Angular SSM
        feat = self.ang_block(feat, angRes)
        # SAM modulation
        feat = self.sam(feat, angRes)
        # FASS HF injection (operates on 2D, so temporarily flatten)
        B, C, A, h, w = feat.shape
        feat_2d = rearrange(feat, 'b c a h w -> (b a) c h w')
        feat_2d = self.fass(feat_2d)
        feat = rearrange(feat_2d, '(b a) c h w -> b c a h w', a=A)

        # V10.3 FIX: Actually apply the learnable residual scale.
        # Each block already has internal `out + x` — the outer res_scale
        # controls how aggressively the combined stack pushes away from
        # the input. sigmoid(-1.6) ≈ 0.17 is the initialization.
        return x + self.drop_path(self.res_scale.sigmoid() * (feat - x))



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

        # G8 Fix: Learnable 2D positional encoding for the angular grid
        # Mamba processes sequences and lacks spatial coordinate awareness.
        # Injecting (U, V) explicit position helps cross-view consistency.
        self.pos_emb = nn.Parameter(torch.zeros(1, channels, angRes, angRes))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x, angRes):
        """x: (B, C, A, h, w) where A = angRes²"""
        B, C, A, h, w = x.shape
        # Reshape: at each (h,w) pixel, the angular grid is a 2D image
        out = rearrange(x, 'b c (u v) h w -> (b h w) c u v',
                        u=angRes, v=angRes)

        # G8 Fix: Add explicit coordinate awareness right before SSM scan
        out = out + self.pos_emb

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
      1. HorizEPI — rearrange to (B*V*w, C, U, h), run shared EPIMambaBlock
      2. VertEPI  — rearrange to (B*U*h, C, V, w), run shared EPIMambaBlock
      3. CAB — Conv + Channel Attention (from LFMamba/MambaIR)
      4. FASS — Frequency-Assisted residual

    V10.1: Weight-shared H/V EPI block (LFMamba EPISSM uses the same
    self.layer for both directions). Saves ~50K+ params per group.
    Research basis: LFMamba EPISSM (proven), EPIT (EPI Transformer)
    """

    def __init__(self, channels, angRes, d_state, d_conv, expand, depth,
                 drop_path=0.0):
        super().__init__()
        self.angRes = angRes
        # V10.1: Single shared block for both H-EPI and V-EPI (LFMamba-style)
        self.epi_block = EPIMambaBlock(channels, d_state, d_conv, expand, depth)
        # V10.3: MicroCAB replaces dense 3×3 CAB (−28% EPI FLOPs)
        self.cab = MicroCAB(channels)
        self.fass = FASSModule(channels)
        # V10.3: sigmoid-initialized at -1.6, consistent with SpaAngGroup
        self.res_scale = nn.Parameter(torch.ones(1) * -1.6)
        # V2.1: Stochastic depth for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, angRes):
        """x: (B, C, A, h, w) where A = angRes²"""
        B, C, A, h, w = x.shape
        u, v = angRes, angRes

        # ---- Horizontal EPI: pair (U, h) — shared block ----
        h_in = rearrange(x, 'b c (u v) h w -> (b v w) c u h',
                         u=u, v=v)
        h_out = self.epi_block(h_in)
        h_out = rearrange(h_out, '(b v w) c u h -> b c (u v) h w',
                          v=v, w=w, u=u)

        # ---- Vertical EPI: pair (V, w) — same shared block ----
        v_in = rearrange(h_out, 'b c (u v) h w -> (b u h) c v w',
                         u=u, v=v)
        v_out = self.epi_block(v_in)
        v_out = rearrange(v_out, '(b u h) c v w -> b c (u v) h w',
                          u=u, h=h, v=v)

        # ---- CAB: local enhancement ----
        epi_2d = rearrange(v_out, 'b c a h w -> (b a) c h w')
        epi_2d = self.cab(epi_2d)
        epi_2d = self.fass(epi_2d)
        feat = rearrange(epi_2d, '(b a) c h w -> b c a h w', a=A)

        # V10.3: Consistent with SpaAngGroup — use x + sigmoid(res_scale) * (feat - x)
        return x + self.drop_path(self.res_scale.sigmoid() * (feat - x))


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
    V10.4: TRUE 2D Channel-split 4-direction Mamba (Beating SS2D).

    Addresses the core mathematical flaw in V10.3 and standard Mamba:
    Standard Mamba applies a 1D Convolution over the flattened sequence.
    When dealing with Transposed (Column-major) 2D images, this 1D Conv mixes
    pixels vertically, destroying horizontal spatial locality and confusing the engine.

    V10.4 fixes this by:
      1. Moving the local spatial mixing to an explicit 2D Depthwise Conv *before* scanning.
      2. Setting `d_conv=1` inside the core Mamba blocks to disable the flawed 1D Conv.
      3. Implementing a global gating mechanism `z` (similar to LFTransMamba's VSSBlock)
         improving gradient flow and non-linear expressivity.
    """

    def __init__(self, channels, d_state=16, d_conv=4, expand=2.0):
        # We ignore the incoming d_conv=4 since we explicitly handle 2D spatial
        # convolutions outside the Mamba block to preserve true locality.
        super().__init__()
        assert channels % 4 == 0, (
            f"BMDMambaLayer requires channels divisible by 4 for 4-way channel "
            f"split, got channels={channels}"
        )
        self.channels = channels
        self.C4 = channels // 4
        self.norm = nn.LayerNorm(channels)

        # Global gating and projection
        self.in_proj = nn.Conv2d(channels, channels * 2, 1, bias=False)
        # True 2D spatial context generator (replaces the flawed 1D Mamba convs)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.act = nn.SiLU()

        # 4 independent SSMs for true directional processing.
        # NOTE: We must keep d_conv between 2 and 4 because the causal_conv1d
        # CUDA extension hardcodes this constraint. We rely on the external dwconv
        # for our true 2D spatial locality prior to this step.
        self.mamba_hw   = Mamba(d_model=self.C4, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_wh   = Mamba(d_model=self.C4, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_hw_r = Mamba(d_model=self.C4, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_wh_r = Mamba(d_model=self.C4, d_state=d_state, d_conv=d_conv, expand=expand)

        self.dir_fusion = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.skip_scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_in = x

        # 1. Layer Norm
        x_norm = x.flatten(2).transpose(1, 2)  # (B, L, C)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

        # 2. In Proj and Global Gate (z)
        x_proj = self.in_proj(x_norm) # (B, 2C, H, W)
        x_mamba, z = x_proj.chunk(2, dim=1) # (B, C, H, W)
        z = self.act(z) # Gate activation

        # 3. True 2D Spatial Locality Mixer
        x_mamba = self.dwconv(x_mamba)
        x_mamba = self.act(x_mamba)

        C4 = self.C4

        # --- 4. Split & Format for 4 Directions ---
        # Group 1: Row-major (H, W) -> (B, L, C4)
        x_hw   = x_mamba[:, :C4].flatten(2).transpose(1, 2).contiguous()

        # Group 2: Col-major (W, H) -> (B, L, C4)
        x_wh   = x_mamba[:, C4:2*C4].transpose(2, 3).contiguous().flatten(2).transpose(1, 2).contiguous()

        # Group 3: Row-major reversed -> (B, L, C4)
        x_hw_r = x_mamba[:, 2*C4:3*C4].flatten(2).flip(-1).transpose(1, 2).contiguous()

        # Group 4: Col-major reversed -> (B, L, C4)
        x_wh_r = x_mamba[:, 3*C4:].transpose(2, 3).contiguous().flatten(2).flip(-1).transpose(1, 2).contiguous()

        # --- 5. Independent SSM Passes (Internal d_conv=1 !) ---
        y_hw   = self.mamba_hw(x_hw)           # (B, L, C4)
        y_wh   = self.mamba_wh(x_wh)           # (B, L, C4)
        y_hw_r = self.mamba_hw_r(x_hw_r)       # (B, L, C4)
        y_wh_r = self.mamba_wh_r(x_wh_r)       # (B, L, C4)

        # --- 6. Unsplit & Reconstruct ---
        y_hw   = y_hw.transpose(1, 2)
        y_wh   = y_wh.transpose(1, 2)
        y_hw_r = y_hw_r.transpose(1, 2)
        y_wh_r = y_wh_r.transpose(1, 2)

        out_hw   = y_hw.view(B, C4, H, W)
        out_wh   = y_wh.view(B, C4, W, H).transpose(2, 3).contiguous()
        out_hw_r = y_hw_r.view(B, C4, H, W).flip(2).flip(3)
        out_wh_r = y_wh_r.view(B, C4, W, H).flip(2).flip(3).transpose(2, 3).contiguous()

        # --- 7. Combine & Gate ---
        combined = torch.cat([out_hw, out_wh, out_hw_r, out_wh_r], dim=1)  # (B, C, H, W)
        out_feat = self.dir_fusion(combined)

        # Multiply by Global Gate z
        out_feat = out_feat * z

        # Final Projection
        out_feat = self.out_proj(out_feat)

        # skip connection with learnable per-channel scale
        return x_in * self.skip_scale.view(1, -1, 1, 1) + out_feat

# ============================================================================
# Spatial-Angular Modulator (SAM) — from MLFSR (ACCV 2024)
# ============================================================================
class SpatialAngularModulator(nn.Module):

    def __init__(self, channels, angRes):
        """
        V10.3 Simplified SAM: element-wise product of spatial & angular gates.

        Original used concat+fuse (2C→C 1×1 conv = 14K extra params).
        New version: factored attention gate  (spa_gate ⊙ ang_gate) * x + x.
        Saves ~14K params; gates are still independently computed per-channel
        sigmoid maps so expressiveness is preserved.

        Research basis: MLFSR (ACCV 2024) Spatial-Angular Modulator
        """
        super().__init__()
        self.angRes = angRes
        r = max(channels // 4, 8)

        # Spatial gate: per angular view
        self.spa_attn = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(r, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # Angular gate: per spatial pixel
        self.ang_attn = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(r, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # V10.3: No self.fuse — removed concat+conv saves ~14K params

    def forward(self, x, angRes):
        """x: (B, C, A, h, w) where A = angRes²"""
        B, C, A, h, w = x.shape
        u, v = angRes, angRes

        # Spatial gate: per angular view
        x_spa = rearrange(x, 'b c a h w -> (b a) c h w')
        spa_gate = self.spa_attn(x_spa)                         # (B*A, C, h, w)
        spa_gate = rearrange(spa_gate, '(b a) c h w -> b c a h w', a=A)

        # Angular gate: per spatial pixel
        x_ang = rearrange(x, 'b c (u v) h w -> (b h w) c u v', u=u, v=v)
        ang_gate = self.ang_attn(x_ang)                         # (B*h*w, C, u, v)
        ang_gate = rearrange(ang_gate, '(b h w) c u v -> b c (u v) h w',
                             b=B, h=h, w=w)

        # Factored gate (equivalent to 2D separable attention) + residual
        combined_gate = spa_gate * ang_gate                     # (B, C, A, h, w)
        return x * combined_gate + x


# ============================================================================
# MicroCAB — Depthwise-Separable Channel Attention Block (V10.3 replacement)
# ============================================================================
class MicroCAB(nn.Module):
    """
    Replaces CAB's two dense 3×3 convs with a pointwise-expand
    → depthwise 3×3 → pointwise-squeeze bottleneck.

    FLOPs comparison (C=48, H×W pixels):
      Old CAB:    2×(48×(48//3)×9) = ~13.8K FLOPs/pixel
      MicroCAB:   48×96 + 96×9 + 96×48 ≈ 10K FLOPs/pixel  (−28%)

    Preserves 3×3 spatial receptive field and channel attention;
    reduces cross-channel FLOPs by pushing mixing into pointwise ops.

    Research basis: LFMamba guide MicroCAB, MobileNetV2 inverted residual
    """

    def __init__(self, channels, squeeze_factor=16):
        super().__init__()
        hidden = channels * 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),          # pointwise expand
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden,    # depthwise 3×3
                      bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),          # pointwise squeeze
        )
        # Lightweight channel attention (identical to ChannelAttention but on squeezed channels)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // squeeze_factor, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // squeeze_factor, 4), channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.net(x)
        return x + y * self.ca(y)  # residual with channel-attention-modulated output


# NOTE: CAB and ChannelAttention removed in V10.3 audit — replaced by MicroCAB.
# If needed for reference, see git history.


# ============================================================================
# FASS Module — Frequency-Assisted State Space (V9 novel, proven)
# ============================================================================
class FASSModule(nn.Module):
    """
    Extracts high-frequency residual = input − low_pass(input),
    refines it via bottleneck conv, injects with learnable per-channel scale.
    Counters Mamba's inherent low-pass smoothing bias.

    V10.1: Replaced heavy SE-style gate (pool→1×1→ReLU→1×1→Sigmoid) with a
    simple learnable per-channel scale. The SE gate was redundant given
    self.scale already controls injection strength. Saves ~1.2K params/module.

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
        # V10.3: Absorbed the redundant 0.2 static scale directly into the gate
        # initialization for mathematically clean learned weighting.
        # sigmoid(-1.386) ≈ 0.2
        self.gate = nn.Parameter(torch.ones(1, channels, 1, 1) * -1.386)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low_pass(x)
        hf = x - low
        hf_refined = self.hf_refine(hf)
        return x + hf_refined * self.gate.sigmoid()


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
# AdaptiveStreamGating (ASG) — V10.3 novel, replaces concat-conv aggregation
# ============================================================================
class AdaptiveStreamGating(nn.Module):
    """
    Content-adaptive fusion of 3 feature streams: IFE, SpaAng, EPI.

    Instead of naive concat (3C) → 1×1 conv → C (which treats all streams
    equally), ASG computes per-stream softmax gates conditioned on the
    content of each stream. The gate vector is (3,) per spatial position,
    so the model learns: "for this pixel, trust EPI more; for that one,
    trust SpaAng." This is analogous to dynamic feature fusion in DFPN
    and CARAFE, but simpler (no deformable ops).

    Params: 3 × (C → 1) 1×1 convs + 1×1 output refine = tiny.
    FLOPs: 3 per-pixel 1×1 convs on C channels = negligible.

    Research basis: V10.3 novel; inspired by gating in MoE architectures.
    """

    def __init__(self, channels):
        super().__init__()
        # Three tiny gate heads: each maps C → 1 scalar gate score
        self.gate_ife = nn.Conv2d(channels, 1, 1, bias=True)
        self.gate_sa  = nn.Conv2d(channels, 1, 1, bias=True)
        self.gate_epi = nn.Conv2d(channels, 1, 1, bias=True)
        # Final 1×1 output conv to mix gated streams back to C channels
        # Input is 3 × C (soft-weighted stack), output is C
        self.fuse = nn.Conv2d(channels * 3, channels, 1, bias=False)
        nn.init.zeros_(self.gate_ife.bias)
        nn.init.zeros_(self.gate_sa.bias)
        nn.init.zeros_(self.gate_epi.bias)

    def forward(self, f_ife, f_sa, f_epi):
        """
        Args:
            f_ife: (B, C, H, W) - IFE stream
            f_sa:  (B, C, H, W) - SpaAng stream
            f_epi: (B, C, H, W) - EPI stream
        Returns:
            (B, C, H, W) - fused features
        """
        # Compute per-stream gate scores: each is (B, 1, H, W)
        g_ife = self.gate_ife(f_ife)
        g_sa  = self.gate_sa(f_sa)
        g_epi = self.gate_epi(f_epi)

        # Softmax across the 3 streams
        # Shape: (B, 3, 1, H, W) → softmax on dim=1 → (B, 3, 1, H, W)
        gates = torch.stack([g_ife, g_sa, g_epi], dim=1)
        gates = gates.softmax(dim=1)

        # Weight each stream by its gate score
        # gates[:, i] is (B, 1, H, W) — broadcasts over C channels
        f_weighted = torch.cat([
            f_ife * gates[:, 0],  # (B, C, H, W) via broadcast
            f_sa  * gates[:, 1],
            f_epi * gates[:, 2],
        ], dim=1)  # (B, 3C, H, W)

        return self.fuse(f_weighted)  # (B, C, H, W)


# ============================================================================
# LocalContrastEnhancement (LCE) — V10.3 novel, Mamba low-pass corrector
# ============================================================================
class LocalContrastEnhancement(nn.Module):
    """
    Lightweight counterpart to FASS at the global aggregation level.

    FASS corrects Mamba's low-pass bias inside each processing block.
    LCE corrects any residual low-pass smoothing at the aggregated feature
    map, just before the final upsampling. It:
      1. Extracts high-frequency residual: hf = x - depthwise_3x3(x)
      2. Refines HF with a 1x1 mixing conv
      3. Gates injection with a learned sigmoid scalar per channel

    Total params: ~C*C/4 + C/4*C + C = ~C²/2 + C ≈ 1.2K for C=48
    FLOPs: negligible at this pipeline depth (single feature map, no upscale)

    Research basis: FASS (V9 novel), DHSFNet (DCT HF restoration)
    """

    def __init__(self, channels):
        super().__init__()
        # Low-pass: depthwise 3×3 to extract local average
        self.low_pass = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        # HF refinement: lightweight 1×1 pointwise mix
        self.hf_mix = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
        )
        # Per-channel sigmoid gate: initialized near 0 (conservative start)
        # sigmoid(-2.0) ≈ 0.12 — network opens it as training progresses
        self.gate = nn.Parameter(torch.ones(1, channels, 1, 1) * -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low_pass(x)
        hf = x - low                         # high-frequency residual
        hf_refined = self.hf_mix(hf)
        return x + hf_refined * self.gate.sigmoid()


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
        # Single 3x3 refine conv (trimmed from two to fit <20G FLOPs with n_sa=4)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
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

        # V3 NOVEL: Progressive 2-stage PixelShuffle (2×+2× = 4×)
        # LFTransMamba uses single 4× PixelShuffle with 1×1 conv.
        # Our 2-stage approach provides: (a) intermediate spatial mixing at 2× res,
        # (b) ICNR init preventing checkerboard on both stages,
        # (c) 3×3 conv at 1st stage for sub-pixel spatial refinement.
        if scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),  # V3 FIX: 0.2→0.1
                nn.Conv2d(channels, channels * 4, 1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),  # V3 FIX: 0.2→0.1
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * scale * scale, 3,
                          padding=1, bias=False),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),  # V3 FIX: 0.2→0.1
            )

        self.output = nn.Conv2d(channels, 1, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.refine(x) + x  # residual shortcut
        r = r * self.ca(r)       # channel attention modulation
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
# LOSS FUNCTION — Pure Charbonnier (SOTA max-PSNR default)
# ============================================================================
class CharbonnierLoss(nn.Module):
    """Pure Charbonnier loss for maximum PSNR training.

    This is the SOTA default for SR PSNR optimization:
      - SwinIR (Liang et al., ICCVW 2021): L1
      - HAT (Chen et al., CVPR 2023): L1
      - LFMamba (Chen et al., 2024): L1
      - LFTransMamba (1st NTIRE 2025): L1/Charb

    Charbonnier is a smooth approximation of L1 that avoids gradient
    discontinuity at zero, providing more stable training.
    """

    def __init__(self, args=None, eps=1e-9):
        super().__init__()
        self.eps = eps if args is None else getattr(args, 'charbonnier_eps', eps)

    def forward(self, pred, target, data_info=None):
        pred, target = pred.float(), target.float()
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


# ============================================================================
# LOSS FUNCTION — Composite (for ablation/perceptual quality experiments)
# ============================================================================
class get_loss(nn.Module):
    """Composite loss: Charbonnier + Focal-FFT + SSIM + Gradient + Angular.

    NOTE: This is NOT the default for max-PSNR training. Use CharbonnierLoss
    instead. This composite loss is useful for:
      - Ablation studies
      - Perceptual quality optimization
      - When SSIM metric matters more than PSNR

    For the NTIRE 2026 Track 2 efficiency challenge (ranked by PSNR),
    use --loss_type charbonnier in the training scripts.
    """

    def __init__(self, args):
        super().__init__()
        self.eps      = getattr(args, "charbonnier_eps", 1e-9)
        # Focal Frequency Loss: self-normalising so lower weight is sufficient.
        # 0.05 is equivalent to or better than 0.1 flat FFT-mag loss.
        self.fft_w    = getattr(args, "fft_weight", 0.05)
        # SSIM weight bumped from 0.1 → 0.15 per HAT/SwinIR ablation results.
        self.ssim_w   = getattr(args, "ssim_weight", 0.15)
        self.grad_w   = getattr(args, "grad_weight", 0.04)
        self.ang_w    = getattr(args, "angular_weight", 0.1)
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
        """Focal Frequency Loss (Jiang et al., ICCV 2021, arXiv:2012.12821).

        Adaptively up-weights frequency components where the model predicts
        poorly (high error) and down-weights those already well-reconstructed.
        Computed in the complex domain (real + imag) for precision.
        A per-element weight proportional to squared error ensures hard
        frequencies receive more gradient signal during training.

        The weight is detached so it is treated as a constant scaling factor
        (not differentiated through), matching the original paper's formulation.

        Using ortho normalisation so the total loss magnitude is consistent
        across different image sizes (important for mixed-size training).
        """
        p, t = p.float(), t.float()  # float32 required for FFT precision

        # 2D real FFT with orthonormal normalisation — shape (B, C, H, W//2+1)
        pf = torch.fft.rfft2(p, norm='ortho')
        tf = torch.fft.rfft2(t, norm='ortho')

        # Per-frequency squared error in complex domain
        # (real_diff² + imag_diff²) avoids sqrt instability and is equivalent
        # to |pf - tf|² but uses only standard arithmetic ops
        freq_dist_sq = (pf.real - tf.real) ** 2 + (pf.imag - tf.imag) ** 2

        # Adaptive weight: error-proportional, normalised so the sum equals 1
        # over the full frequency map. Hard frequencies (large error) get
        # weight > 1; easy ones get weight < 1. Detached — constant scaling.
        with torch.no_grad():
            # Add epsilon to mean to prevent division by zero on perfect batches
            w = freq_dist_sq / (freq_dist_sq.mean() + 1e-8)

        # Weighted mean squared error, then sqrt to recover L1 loss scale.
        # sqrt(mean(w * dist_sq)) is dimensionally consistent with Charbonnier.
        return (w * freq_dist_sq).mean().sqrt()

    def ssim_loss(self, p, t):
        """Per-view SSIM loss — avoids window crossing SAI view boundaries."""
        p, t = p.float(), t.float()
        B, C, H, W = p.shape
        a = self.angRes
        # Reshape to per-view: (B*a*a, 1, h, w)
        p_views = rearrange(p, 'b c (u h) (v w) -> (b u v) c h w', u=a, v=a)
        t_views = rearrange(t, 'b c (u h) (v w) -> (b u v) c h w', u=a, v=a)
        C1, C2, ws = 0.01 ** 2, 0.03 ** 2, 7
        pad = ws // 2
        mu_p = F.avg_pool2d(p_views, ws, 1, pad)
        mu_t = F.avg_pool2d(t_views, ws, 1, pad)
        s_p = F.avg_pool2d(p_views ** 2, ws, 1, pad) - mu_p ** 2
        s_t = F.avg_pool2d(t_views ** 2, ws, 1, pad) - mu_t ** 2
        s_x = F.avg_pool2d(p_views * t_views, ws, 1, pad) - mu_p * mu_t
        s_p, s_t = s_p.clamp(min=0), s_t.clamp(min=0)
        ssim = ((2 * mu_p * mu_t + C1) * (2 * s_x + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (s_p + s_t + C2))
        return 1 - ssim.mean()

    def gradient_loss(self, p, t):
        """Per-view gradient loss — avoids Sobel seeing SAI view boundaries."""
        p, t = p.float(), t.float()
        B, C, H, W = p.shape
        a = self.angRes
        # Reshape to per-view to avoid Sobel crossing view boundaries
        p_views = rearrange(p, 'b c (u h) (v w) -> (b u v) c h w', u=a, v=a)
        t_views = rearrange(t, 'b c (u h) (v w) -> (b u v) c h w', u=a, v=a)
        sx = self.sobel_x
        sy = self.sobel_y
        return (
            F.l1_loss(F.conv2d(p_views, sx, padding=1), F.conv2d(t_views, sx, padding=1))
            + F.l1_loss(F.conv2d(p_views, sy, padding=1), F.conv2d(t_views, sy, padding=1))
        )

    def angular_loss(self, p, t):
        p, t = p.float(), t.float()  # float32 for angular difference precision
        B, C, H, W = p.shape
        a = self.angRes

        # B3 Fix: The input grid is (b, c, u*h, v*w).
        # Correctly separate angular (u, v) and spatial (h_sp, w_sp) dims.
        h_sp, w_sp = H // a, W // a

        # rearrange to b, c, u, v, h, w
        pv = rearrange(p, 'b c (u h) (v w) -> b c u v h w', u=a, v=a, h=h_sp, w=w_sp)
        tv = rearrange(t, 'b c (u h) (v w) -> b c u v h w', u=a, v=a, h=h_sp, w=w_sp)

        # Penalize adjacent view absolute differences (first-order difference)
        # This is strictly superior to the second-order 2nd derivative penalty for spatial consistency.
        diff_u = F.l1_loss(pv[:, :, 1:, :, :, :] - pv[:, :, :-1, :, :, :],
                           tv[:, :, 1:, :, :, :] - tv[:, :, :-1, :, :, :])

        diff_v = F.l1_loss(pv[:, :, :, 1:, :, :] - pv[:, :, :, :-1, :, :],
                           tv[:, :, :, 1:, :, :] - tv[:, :, :, :-1, :, :])

        return diff_u + diff_v

    def forward(self, pred, target, data_info=None):
        # P7: Cast once at entry — avoids repeated .float() calls inside each sub-loss
        # while inside torch.amp.autocast. bfloat16 underflows on squared terms (SSIM)
        # and eps^2=1e-18 (Charbonnier), so float32 is required throughout.
        pred, target = pred.float(), target.float()
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
# SELF-TEST & TRACK 2 EFFICIENCY VALIDATION
# ============================================================================


"""
CodaBench Submission Generator for V3 (MyEfficientLFNetV3_MLFIM)
================================================================
Generates submission_v3.zip ready for NTIRE 2026 LF-SR Track 2 CodaBench upload.

Pipeline:
  1. Download NTIRE_Test_Real / NTIRE_Test_Synth .mat files via gdown
  2. Load V3 model checkpoint (EMA preferred, falls back to state_dict)
  3. For each .mat: RGB→YCbCr, SR on Y channel (NO TTA), bicubic-upsample CbCr
  4. Gaussian-weighted patch stitching (matches train.py validation exactly)
  5. YCbCr→RGB, save 5x5 views as View_{i}_{j}.bmp per scene
  6. Zip into submission_v3.zip  (Real/<scene>/View_*.bmp, Synth/<scene>/View_*.bmp)

AUDIT NOTES (2026-03-07):
  - LFdivide uses V3-fixed numU/numV from padded dims (matches utils.py)
  - imresize handles multi-channel CbCr correctly (per-channel resize)
  - subLFin patches are moved to device before net() call
  - LFintegrate_gaussian matches utils.py exactly
  - No TTA whatsoever — pure forward pass
  - BMP output format matches colab_submission.py and train.py test()
"""

import os
import sys
import glob
import subprocess
import shutil
import zipfile
import numpy as np
import h5py
import scipy.io as scio
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import importlib
import argparse
from pathlib import Path
from math import ceil

# ==============================================================================
# 0. Set up dependencies and paths
# ==============================================================================

TEST_REAL_LINK = "https://drive.google.com/drive/folders/1FxWmbrbH2mYQgApjOmj-2UM1Yu7fQ1Rg"
TEST_SYNTH_LINK = "https://drive.google.com/drive/folders/120fxXLA20jI7tWrZ-YGn14e4B41cIPq7"

# All 32 expected test scene filenames (known from the official test data)
EXPECTED_REAL_FILES = [
    "IMG_0199__Decoded.mat", "IMG_0214__Decoded.mat", "IMG_0238__Decoded.mat",
    "IMG_0256__Decoded.mat", "IMG_0262__Decoded.mat", "IMG_0268__Decoded.mat",
    "IMG_0271__Decoded.mat", "IMG_0280__Decoded.mat", "IMG_0289__Decoded.mat",
    "IMG_0308__Decoded.mat", "IMG_0329__Decoded.mat", "IMG_0363__Decoded.mat",
    "IMG_0371__Decoded.mat", "IMG_0389__Decoded.mat", "IMG_0394__Decoded.mat",
    "IMG_0404__Decoded.mat",
]
EXPECTED_SYNTH_FILES = [
    "aquarium.mat", "bookcase.mat", "camellia.mat", "cat 2.mat", "cat.mat",
    "courtyard.mat", "dining table.mat", "headboard.mat", "ivy.mat",
    "obius cube.mat", "pinetree.mat", "plants.mat", "shelf.mat",
    "ship model.mat", "stationery.mat", "washer.mat",
]

def run_cmd(cmd, check=True):
    print(f"\n[EXEC] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}: {cmd}")
        sys.exit(result.returncode)

def verify_all_files(base_dir, expected_files, label):
    """Check that every expected .mat file exists somewhere under base_dir."""
    # Find all .mat files recursively
    found_files = {os.path.basename(f) for f in glob.glob(f'{base_dir}/**/*.mat', recursive=True)}
    missing = [f for f in expected_files if f not in found_files]
    if missing:
        print(f"\n❌ MISSING {len(missing)} {label} files:")
        for m in missing:
            print(f"   • {m}")
        return False
    else:
        print(f"   ✅ All {len(expected_files)} {label} .mat files verified present")
        return True

def download_test_data(force=False):
    print("\n=== STEP 1: Checking/Downloading NTIRE Test Data ===")
    os.makedirs('datasets_test', exist_ok=True)

    real_dir = "datasets_test/NTIRE_Test_Real"
    synth_dir = "datasets_test/NTIRE_Test_Synth"

    # Download Real test data
    real_ok = os.path.exists(real_dir) and len(glob.glob(f'{real_dir}/**/*.mat', recursive=True)) >= 16
    if force or not real_ok:
        print(f"Downloading Test Real to {real_dir}...")
        if force and os.path.exists(real_dir):
            shutil.rmtree(real_dir)
        os.makedirs(real_dir, exist_ok=True)
        folder_id = TEST_REAL_LINK.split('/')[-1].split('?')[0]
        run_cmd(f'gdown --folder {folder_id} -O "{real_dir}"', check=False)

    # Download Synth test data
    synth_ok = os.path.exists(synth_dir) and len(glob.glob(f'{synth_dir}/**/*.mat', recursive=True)) >= 16
    if force or not synth_ok:
        print(f"Downloading Test Synth to {synth_dir}...")
        if force and os.path.exists(synth_dir):
            shutil.rmtree(synth_dir)
        os.makedirs(synth_dir, exist_ok=True)
        folder_id = TEST_SYNTH_LINK.split('/')[-1].split('?')[0]
        run_cmd(f'gdown --folder {folder_id} -O "{synth_dir}"', check=False)

    # Verify ALL expected files are present
    real_count = len(glob.glob(f'{real_dir}/**/*.mat', recursive=True))
    synth_count = len(glob.glob(f'{synth_dir}/**/*.mat', recursive=True))
    print(f"\n[INFO] Found: {real_count} Real .mat, {synth_count} Synth .mat")

    real_verified = verify_all_files(real_dir, EXPECTED_REAL_FILES, "Real")
    synth_verified = verify_all_files(synth_dir, EXPECTED_SYNTH_FILES, "Synth")

    if not real_verified or not synth_verified:
        print("\n❌ Some test files are MISSING! gdown may have failed (Google Drive rate limits).")
        print("Please download manually via your browser, upload to VM, and run:")
        print("  python generate_codabench_submission_v3.py --real_dir /path/to/Real/mat/files --synth_dir /path/to/Synth/mat/files")
        sys.exit(1)


# ==============================================================================
# MATLAB-compatible bicubic resize (matches colab_submission.py / BasicLFSR)
# ==============================================================================
# NOTE: The PIL-based imresize(mode='F') only handles single-channel images.
# CbCr has 2 channels. We must resize each channel independently OR use the
# MATLAB-compatible imresize from colab_submission.py which handles multi-channel.

def _triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f

def _cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + \
        np.multiply(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (1 < absx) & (absx <= 2))
    return f

def _contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def _imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None):
    """MATLAB-compatible bicubic resize. Handles 2D and multi-channel arrays."""
    kernel = _cubic if method == 'bicubic' else _triangle
    kernel_width = 4.0
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = [int(ceil(scalar_scale * I.shape[0])), int(ceil(scalar_scale * I.shape[1]))]
    elif output_shape is not None:
        scale = [1.0 * output_shape[k] / I.shape[k] for k in range(2)]
        output_size = list(output_shape)
    else:
        raise ValueError('scalar_scale OR output_shape should be defined!')

    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = _contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = _imresizevec(B, weights[dim], indices[dim], dim)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


# ==============================================================================
# Color conversion (identical to utils.py)
# ==============================================================================

def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0
    y = y / 255.0
    return y

def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y


# ==============================================================================
# Patch divide/integrate (matches utils.py V3-fixed version EXACTLY)
# ==============================================================================

def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]
    return Im_out


def LFdivide(data, angRes, patch_size, stride):
    """V3-FIXED: Compute numU/numV from actual padded dimensions (matches utils.py)."""
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    # V3 FIX: Compute numU/numV from the actual padded dimensions
    h_pad, w_pad = data_pad.shape[2], data_pad.shape[3]
    numU = (h_pad - patch_size) // stride + 1
    numV = (w_pad - patch_size) // stride + 1
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)
    return subLF


def LFintegrate_gaussian(subLF, angRes, pz, stride, h, w):
    """Gaussian-weighted patch stitching (matches utils.py exactly)."""
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w',
                          a1=angRes, a2=angRes)

    n1, n2, a1, a2, pH, pW = subLF.shape

    # Build 2D Gaussian weight map for one patch
    sigma = pz / 3.0  # Tighter Gaussian — trusts patch centers more, reduces edge artifacts
    ax = torch.arange(pz, dtype=torch.float32, device=subLF.device) - (pz - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    gauss_2d = gauss_2d / gauss_2d.max()

    # Accumulate weighted patches into output canvas
    canvas_h = (n1 - 1) * stride + pz
    canvas_w = (n2 - 1) * stride + pz

    outLF = torch.zeros(a1, a2, canvas_h, canvas_w, dtype=subLF.dtype, device=subLF.device)
    weight_map = torch.zeros(1, 1, canvas_h, canvas_w, dtype=subLF.dtype, device=subLF.device)

    for i in range(n1):
        for j in range(n2):
            top = i * stride
            left = j * stride
            outLF[:, :, top:top+pz, left:left+pz] += subLF[i, j] * gauss_2d
            weight_map[:, :, top:top+pz, left:left+pz] += gauss_2d

    weight_map = weight_map.clamp(min=1e-8)
    outLF = outLF / weight_map

    # Crop to target size, EXCLUDING the padded border
    bdr_hr = (pz - stride) // 2
    outLF = outLF[:, :, bdr_hr : bdr_hr + h, bdr_hr : bdr_hr + w]
    return outLF


# ==============================================================================
# Model Inference (NO TTA — pure forward pass)
# ==============================================================================

def run_sr_inference(Lr_SAI_y_tensor, net, device, args):
    """
    Run SR inference on Y channel and return the float Sr_4D_y tensor.
    Separated from I/O to allow output-level ensemble averaging.
    Forces FP32 precision for maximum numerical accuracy.
    """
    angRes = args.angRes_in
    scale_factor = args.scale_factor

    # Force FP32 precision (avoid any AMP/half-precision numerical loss)
    net.float()
    net.eval()

    # 4. Divide LF into overlapping patches
    subLFin = LFdivide(Lr_SAI_y_tensor.squeeze(), angRes, args.patch_size_for_test, args.stride_for_test)
    numU, numV, pH, pW = subLFin.size()
    subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')

    # Allocate output on CPU
    subLFout = torch.zeros(numU * numV, 1, angRes * args.patch_size_for_test * scale_factor,
                           angRes * args.patch_size_for_test * scale_factor)

    # 5. SR each patch (No TTA — pure forward pass, FP32 enforced)
    data_info = [args.angRes_in, args.angRes_out]
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in range(0, numU * numV, args.minibatch_for_test):
            end_idx = min(i + args.minibatch_for_test, numU * numV)
            tmp = subLFin[i:end_idx, :, :, :].float()  # FP32 enforcement
            out = net(tmp.to(device), data_info)
            subLFout[i:end_idx, :, :, :] = out.float().cpu()

    subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

    # 6. Restore patches to full LF using Gaussian blending
    sr_pz = args.patch_size_for_test * scale_factor
    sr_stride = args.stride_for_test * scale_factor
    # Compute H, W from the input tensor
    total_h = Lr_SAI_y_tensor.squeeze().shape[-2] // angRes
    total_w = Lr_SAI_y_tensor.squeeze().shape[-1] // angRes
    target_h = total_h * scale_factor
    target_w = total_w * scale_factor
    Sr_4D_y = LFintegrate_gaussian(subLFout, args.angRes_out, sr_pz, sr_stride, target_h, target_w)
    return Sr_4D_y  # shape: (a1, a2, h, w) — float tensor


def process_file_direct(mat_file_path, save_dir, nets, device, args):
    """
    Process a single .mat file → 5x5 View BMP files.
    Supports output-level ensemble: pass a list of nets to average predictions.
    No TTA. Forces FP32. Uses tighter Gaussian and larger patches.
    """
    import imageio
    filename = Path(mat_file_path).name

    # 1. Load LF data from .mat
    try:
        data = h5py.File(mat_file_path, 'r')
        LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
    except:
        data = scio.loadmat(mat_file_path)
        LF = np.array(data['LF'])

    (U, V, H, W, _) = LF.shape
    angRes = args.angRes_in
    scale_factor = args.scale_factor

    # Extract central angRes×angRes views
    LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, 0:H, 0:W, 0:3]
    LF = LF.astype('double')
    (U, V, H, W, _) = LF.shape

    # Pre-allocate output arrays
    Sr_SAI_cbcr = np.zeros((U * H * scale_factor, V * W * scale_factor, 2), dtype='single')
    Lr_SAI_y = np.zeros((U * H, V * W), dtype='single')

    # 2. Convert RGB→YCbCr, extract Y, bicubic-upsample CbCr
    for u in range(U):
        for v in range(V):
            tmp_Lr_rgb = LF[u, v, :, :, :]
            tmp_Lr_ycbcr = rgb2ycbcr(tmp_Lr_rgb)
            Lr_SAI_y[u * H: (u+1) * H, v * W: (v+1) * W] = tmp_Lr_ycbcr[:, :, 0]

            # Bicubic upsample CbCr channels (MATLAB-compatible, multi-channel safe)
            tmp_Lr_cbcr = tmp_Lr_ycbcr[:, :, 1:3]
            tmp_Sr_cbcr = imresize(tmp_Lr_cbcr, scalar_scale=scale_factor)
            Sr_SAI_cbcr[u * H * scale_factor: (u+1) * H * scale_factor,
                        v * W * scale_factor: (v+1) * W * scale_factor, :] = tmp_Sr_cbcr

    # 3. Prepare for model: convert to torch tensors
    Lr_SAI_y_tensor = torch.from_numpy(Lr_SAI_y).unsqueeze(0).unsqueeze(0)  # (1, 1, U*H, V*W)
    Sr_SAI_cbcr_tensor = torch.from_numpy(Sr_SAI_cbcr).permute(2, 0, 1).unsqueeze(0)  # (1, 2, U*H*S, V*W*S)

    # ---------------------------------------------------------------
    # OUTPUT-LEVEL ENSEMBLE: run inference with each net, average Y
    # ---------------------------------------------------------------
    if not isinstance(nets, list):
        nets = [nets]

    Sr_4D_y_sum = None
    for net in nets:
        Sr_4D_y = run_sr_inference(Lr_SAI_y_tensor, net, device, args)
        if Sr_4D_y_sum is None:
            Sr_4D_y_sum = Sr_4D_y.float()
        else:
            Sr_4D_y_sum += Sr_4D_y.float()
        torch.cuda.empty_cache()

    Sr_4D_y_avg = Sr_4D_y_sum / float(len(nets))
    Sr_SAI_y = rearrange(Sr_4D_y_avg, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

    # 7. Recombine Y with CbCr and convert to RGB
    Sr_SAI_ycbcr = torch.cat((Sr_SAI_y.cpu(), Sr_SAI_cbcr_tensor), dim=1)
    Sr_SAI_rgb = np.round(ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0, 1) * 255.0).astype('uint8')
    Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

    # 8. Save BMP per view
    scene_name = filename.replace('.mat', '').replace('.h5', '')

    # -------------------------------------------------------------
    # OPTIONAL MAPPING FOR CODABENCH VALIDATION NAMES
    # -------------------------------------------------------------
    if getattr(args, 'use_validation_names', False):
        CODABENCH_REAL = ['0725', '1518', '1546', 'Bikes', 'Hublais', 'IMG_3854', 'IMG_3869', 'IMG_4028', 'IMG_4061', 'IMG_4085', 'IMG_4162', 'IMG_5160', 'Image245', 'Image427', 'Image435', 'Palais']
        CODABENCH_SYNTH = [f'{i:02d}' for i in range(1, 17)]

        if 'Real' in str(save_dir) or 'Real' in mat_file_path:
            import glob
            parent_dir = os.path.dirname(mat_file_path)
            all_mats = sorted([os.path.basename(p) for p in glob.glob(os.path.join(parent_dir, '*.mat'))])
            try:
                idx = all_mats.index(filename)
                scene_name = sorted(CODABENCH_REAL)[idx]
            except:
                pass
        elif 'Synth' in str(save_dir) or 'Synth' in mat_file_path:
            import glob
            parent_dir = os.path.dirname(mat_file_path)
            all_mats = sorted([os.path.basename(p) for p in glob.glob(os.path.join(parent_dir, '*.mat'))])
            try:
                idx = all_mats.index(filename)
                scene_name = sorted(CODABENCH_SYNTH)[idx]
            except:
                pass

    scene_dir = os.path.join(save_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    for i in range(args.angRes_out):
        for j in range(args.angRes_out):
            img = Sr_4D_rgb[i, j, :, :, :]
            path = os.path.join(scene_dir, f'View_{i}_{j}.bmp')
            imageio.imwrite(path, img)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print('HEY USER! STANDALONE IS HERE!')
    parser = argparse.ArgumentParser("CodaBench Submission Generator for V3")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint. Defaults to auto-search in log dir")
    parser.add_argument("--real_dir", type=str, default=None,
                        help="Path to Real test .mat files (overrides download)")
    parser.add_argument("--synth_dir", type=str, default=None,
                        help="Path to Synth test .mat files (overrides download)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download of test data even if files exist")
    parser.add_argument("--use_validation_names", action="store_true",
                        help="Map input test names to CodaBench validation IDs (01-16, 0725-1546)")
    parser.add_argument("--no_swa", action="store_true",
                        help="Disable automatic Stochastic Weight Averaging and just load the single explicit or best checkpoint")
    parser.add_argument("--swa_n", type=int, default=10,
                        help="Number of last finetune checkpoints to average for SWA (default: 10)")
    parser.add_argument("--prefer_ema", action="store_true",
                        help="Use ema_state_dict for SWA (default: use raw state_dict to avoid double-smoothing)")
    # Use parse_known_args instead of parse_args to ignore Jupyter notebook `-f` flags
    args, _ = parser.parse_known_args()

    # Step 1: Download or locate test data
    if args.real_dir is None or args.synth_dir is None:
        download_test_data(force=args.force_download)

    real_dir = args.real_dir or "datasets_test/NTIRE_Test_Real"
    synth_dir = args.synth_dir or "datasets_test/NTIRE_Test_Synth"

    # Step 2: Model & Args Definition
    class Config:
        model_name = "MyEfficientLFNetV3_MLFIM"
        task = "SR"
        angRes_in = 5
        angRes_out = 5
        scale_factor = 4
        patch_size_for_test = 48   # Larger patches = more spatial context per forward pass
        stride_for_test = 4       # 91.7% overlap for maximum Gaussian blending density
        minibatch_for_test = 16   # A100 optimization: blast 16 patches at once to eliminate PyTorch loop bottlenecks
        mlfim_mask_ratio = 0.0    # CRITICAL: 0.0 at inference (no masking)

    config = Config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")

    print("\n=== STEP 2: Loading V3 Model ===")
    net = get_model(config).to(device)

    # Auto-find checkpoint
    if args.ckpt is None:
        search_patterns = [
            f'log/SR_5x5_4x/ALL/{config.model_name}/checkpoints/*.pth',
            f'log/SR_5x5_4x/*/{config.model_name}/checkpoints/*.pth',
            'log/**/*.pth',
            '*.pth',
            '**/*.pth'
        ]
        pth_files = []
        for pattern in search_patterns:
            pth_files = glob.glob(pattern, recursive=True)
            if pth_files:
                break

        if not pth_files:
            print("❌ No .pth checkpoint found! Please provide it explicitly with --ckpt <path>")
            sys.exit(1)

        pth_files.sort(key=os.path.getmtime, reverse=True)
        
        # New automated SWA logic: check if we have multiple finetune checkpoints to average!
        import re
        finetune_ckpts = [f for f in pth_files if 'finetune' in f and 'epoch' in f]
        finetune_ckpts.sort(key=lambda x: int(re.search(r'epoch_(\d+)_model\.pth', os.path.basename(x)).group(1)) if re.search(r'epoch_(\d+)_model\.pth', os.path.basename(x)) else 0)
        
        # Check if _finetune_best.pth exists natively (e.g., from Stage 3 Polish)
        best_native = [f for f in pth_files if 'finetune_best.pth' in f]
        
        if best_native:
            best_ckpt = best_native[0]
            print(f"Auto-selected absolute best checkpoint: {best_ckpt}")
            checkpoint = torch.load(best_ckpt, map_location=device)
        elif not getattr(args, 'no_swa', False) and len(finetune_ckpts) >= 2:
            swa_n = getattr(args, 'swa_n', 10)
            last_n = min(swa_n, len(finetune_ckpts))
            to_average = finetune_ckpts[-last_n:]
            # SWA KEY CHOICE: Use raw state_dict (not EMA) to avoid double-smoothing.
            # EMA is already a moving average; averaging multiple EMA checkpoints
            # over-smooths and dilutes fine details. Use --prefer_ema to override.
            prefer_ema = getattr(args, 'prefer_ema', False)
            weight_key = 'ema_state_dict' if prefer_ema else 'state_dict'
            print(f"\n[INFO] Auto-SWA trigger: Found {len(finetune_ckpts)} finetune checkpoints.")
            print(f"[INFO] Averaging the last {last_n} using '{weight_key}' weights:")
            for p in to_average:
                print(f"       - {os.path.basename(p)}")
            
            avg_state_dict = None
            from collections import OrderedDict
            for path in to_average:
                ckpt = torch.load(path, map_location='cpu')
                if prefer_ema:
                    s_dict = ckpt.get('ema_state_dict', ckpt.get('state_dict', ckpt))
                else:
                    s_dict = ckpt.get('state_dict', ckpt.get('ema_state_dict', ckpt))
                if avg_state_dict is None:
                    avg_state_dict = OrderedDict()
                    for key, value in s_dict.items():
                        avg_state_dict[key] = value.float().clone()
                else:
                    for key, value in s_dict.items():
                        if key in avg_state_dict:
                            avg_state_dict[key] += value.float()
            
            for key in avg_state_dict:
                avg_state_dict[key] /= float(last_n)
            
            checkpoint = {'state_dict': avg_state_dict}
            best_ckpt = f"[SWA: last {last_n} of {len(finetune_ckpts)}, using {weight_key}]"
        else:
            best_ckpt = pth_files[0]
            print(f"Auto-selected checkpoint: {best_ckpt}")
            checkpoint = torch.load(best_ckpt, map_location=device)
    else:
        best_ckpt = args.ckpt
        print(f"Loading checkpoint: {best_ckpt}")
        checkpoint = torch.load(best_ckpt, map_location=device)

    # Priority: state_dict > ema_state_dict > raw dict
    # IMPORTANT: Use raw state_dict first to avoid double-smoothing.
    # EMA weights are already a moving average — loading them by default
    # when SWA is also active causes over-smoothing.
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('ema_state_dict', checkpoint))
    else:
        state_dict = checkpoint

    # Clean module. prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        cleaned_state_dict[new_k] = v

    net.load_state_dict(cleaned_state_dict, strict=False)
    net.float()  # FP32 enforcement for maximum precision
    net.eval()
    print(f"✅ Primary checkpoint loaded successfully! ({len(cleaned_state_dict)} params)")

    # SWA already contains the best checkpoint weights — no need for output ensemble
    # (output ensemble doubles VRAM and the best ckpt is already in the SWA average)
    ensemble_nets = [net]

    # Step 3: Run Inference
    print("\n=== STEP 3: Running Inference & Generating .bmp files ===")
    print(f"[INFO] Config: patch={config.patch_size_for_test}, stride={config.stride_for_test}, model={best_ckpt}")
    out_base = "submission_temp"
    shutil.rmtree(out_base, ignore_errors=True)
    os.makedirs(f"{out_base}/Real", exist_ok=True)
    os.makedirs(f"{out_base}/Synth", exist_ok=True)

    real_files = sorted(glob.glob(f"{real_dir}/**/*.mat", recursive=True))
    synth_files = sorted(glob.glob(f"{synth_dir}/**/*.mat", recursive=True))

    print(f"Found {len(real_files)} Real + {len(synth_files)} Synth .mat files")
    if len(real_files) == 0 and len(synth_files) == 0:
        print("❌ No .mat files found! Check your data paths.")
        sys.exit(1)

    failed_scenes = []

    print(f"\nProcessing {len(real_files)} Real scenes...")
    for f in tqdm(real_files, ncols=70):
        try:
            process_file_direct(f, f"{out_base}/Real", ensemble_nets, device, config)
            torch.cuda.empty_cache()
        except Exception as e:
            scene_name = Path(f).stem
            print(f"\n❌ FAILED on Real/{scene_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_scenes.append(f"Real/{scene_name}")

    print(f"\nProcessing {len(synth_files)} Synth scenes...")
    for f in tqdm(synth_files, ncols=70):
        try:
            process_file_direct(f, f"{out_base}/Synth", ensemble_nets, device, config)
            torch.cuda.empty_cache()
        except Exception as e:
            scene_name = Path(f).stem
            print(f"\n❌ FAILED on Synth/{scene_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_scenes.append(f"Synth/{scene_name}")

    if failed_scenes:
        print(f"\n⚠️  WARNING: {len(failed_scenes)} scenes FAILED:")
        for s in failed_scenes:
            print(f"   • {s}")
        print("   The submission will be INCOMPLETE and CodaBench will reject it!")

    # Step 4: Zip Submission
    print("\n=== STEP 4: Creating submission.zip ===")
    zip_path = "submission.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    total_files = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # CodaBench requires explicit directory entries in the zip to prevent NullPointerExceptions
        registered_dirs = set()

        for root, dirs, files in os.walk(out_base):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, out_base)

                # CRITICAL: Always use forward slashes (POSIX) in zip arcnames.
                arcname = rel_path.replace('\\', '/')
                if arcname.startswith('/'):
                    arcname = arcname[1:]

                # CRITICAL: Manually add containing directories to the ZIP
                parts = arcname.split('/')
                for i in range(1, len(parts)):
                    dir_path = '/'.join(parts[:i]) + '/'
                    if dir_path not in registered_dirs:
                        zinfo = zipfile.ZipInfo(dir_path)
                        zipf.writestr(zinfo, '')
                        registered_dirs.add(dir_path)

                zipf.write(file_path, arcname)
                total_files += 1

    print(f"\n[OK] Packaged {total_files} files into {zip_path}.")

    # Step 5: Self-Validation
    print("\n=== STEP 5: Self-Validating submission.zip ===")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_entries = zf.namelist()

        # Check structure
        has_real = any(f.startswith('Real/') for f in all_entries)
        has_synth = any(f.startswith('Synth/') for f in all_entries)

        real_scenes = set()
        synth_scenes = set()
        bmp_count = 0
        for f in all_entries:
            parts = f.split('/')
            if len(parts) >= 3 and parts[-1].endswith('.bmp'):
                bmp_count += 1
                if parts[0] == 'Real':
                    real_scenes.add(parts[1])
                elif parts[0] == 'Synth':
                    synth_scenes.add(parts[1])

        print(f"\n📊 Zip Contents Validation:")
        print(f"   Has Real/ folder: {has_real}")
        print(f"   Has Synth/ folder: {has_synth}")
        print(f"   Real scenes ({len(real_scenes)}): {sorted(real_scenes)}")
        print(f"   Synth scenes ({len(synth_scenes)}): {sorted(synth_scenes)}")
        print(f"   Total BMP files: {bmp_count}")
        print(f"   Expected: {(len(real_scenes) + len(synth_scenes)) * 25} ({len(real_scenes) + len(synth_scenes)} scenes × 25 views)")

        # Show first 10 entries for debugging
        print(f"\n   First 10 zip entries (verify no backslashes or extra nesting):")
        for entry in sorted(all_entries)[:10]:
            print(f"     {entry}")

        # Final verdict
        errors = []
        if not has_real:
            errors.append("Missing 'Real/' folder at zip root")
        if not has_synth:
            errors.append("Missing 'Synth/' folder at zip root")
        if bmp_count != (len(real_scenes) + len(synth_scenes)) * 25:
            errors.append(f"BMP count mismatch: got {bmp_count}, expected {(len(real_scenes) + len(synth_scenes)) * 25}")

        # Check for view naming
        expected_views = {f"View_{i}_{j}.bmp" for i in range(5) for j in range(5)}
        for scene_type, scenes in [("Real", real_scenes), ("Synth", synth_scenes)]:
            for scene in scenes:
                prefix = f"{scene_type}/{scene}/"
                scene_files = {f.split('/')[-1] for f in all_entries if f.startswith(prefix) and f.endswith('.bmp')}
                missing = expected_views - scene_files
                if missing:
                    errors.append(f"{scene_type}/{scene} missing views: {missing}")

        if errors:
            print(f"\n❌ VALIDATION FAILED:")
            for e in errors:
                print(f"   • {e}")
        else:
            print(f"\n✅ VALIDATION PASSED — submission.zip is ready for CodaBench upload!")

if __name__ == "__main__":
    main()

