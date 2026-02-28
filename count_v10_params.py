"""
Analytical Parameter Counter for MyEfficientLFNetV10
=====================================================
Estimates total parameters WITHOUT needing mamba-ssm installed.
Uses the exact same hyperparameters from MyEfficientLFNetV10.py.

NTIRE 2026 Track 2 Limits:
  - Parameters: < 1,000,000
  - FLOPs:      < 20G (on 5x5x32x32 input)
"""

def count_conv2d(cin, cout, k, bias=False, groups=1):
    return cout * (cin // groups) * k * k + (cout if bias else 0)

def count_conv3d(cin, cout, k_tuple, bias=False):
    kd, kh, kw = k_tuple
    return cout * cin * kd * kh * kw + (cout if bias else 0)

def count_linear(cin, cout, bias=False):
    return cin * cout + (cout if bias else 0)

def count_layernorm(dim):
    return 2 * dim  # weight + bias

def count_mamba(d_model, d_state, d_conv, expand):
    """Estimate Mamba block parameters (from mamba_ssm source)."""
    d_inner = int(d_model * expand)
    # in_proj: Linear(d_model, d_inner*2, bias=False)
    in_proj = d_model * d_inner * 2
    # conv1d: Conv1d(d_inner, d_inner, d_conv, groups=d_inner, bias=True)
    conv1d = d_inner * d_conv + d_inner  # depthwise + bias
    # x_proj: Linear(d_inner, dt_rank + 2*d_state, bias=False)
    dt_rank = max(1, d_model // 16)  # default: ceil(d_model/16)
    x_proj = d_inner * (dt_rank + 2 * d_state)
    # dt_proj: Linear(dt_rank, d_inner, bias=True)
    dt_proj = dt_rank * d_inner + d_inner
    # A_log: (d_inner, d_state) parameter
    A_log = d_inner * d_state
    # D: (d_inner,) parameter
    D = d_inner
    # out_proj: Linear(d_inner, d_model, bias=False)
    out_proj = d_inner * d_model

    total = in_proj + conv1d + x_proj + dt_proj + A_log + D + out_proj
    return total

def main():
    # ---- V10 Hyperparameters (Track 2 Efficiency) ----
    C = 48
    n_sa = 2
    n_epi = 2
    d_state = 16
    d_conv = 4
    expand = 2.0
    vss_depth = 2
    angRes = 5
    scale = 4
    num_heads = 4
    window_size = 4

    total = 0

    # ---- MLFIM mask token ----
    mask_token = C
    total += mask_token
    print(f"MLFIM mask_token:      {mask_token:>10,}")

    # ---- MODULE 1: 3D Conv IFE ----
    ife = 0
    ife += count_conv3d(1, C, (1,3,3))          # conv_init0
    ife += count_conv3d(C, C, (1,3,3)) * 3      # conv_init (3 conv3d layers)
    total += ife
    print(f"IFE (3D Conv):         {ife:>10,}")

    # ---- BMDMambaLayer params (reused) ----
    def bmd_layer_params():
        p = 0
        p += count_layernorm(C)                   # norm
        p += count_mamba(C, d_state, d_conv, expand)  # mamba
        p += count_conv2d(C, C, 1)                # dir_fusion
        p += C                                     # skip_scale
        return p

    bmd_single = bmd_layer_params()
    print(f"  (Single BMDMambaLayer: {bmd_single:,})")

    # ---- MODULE 2: Spatial-Angular Groups ----
    sa_total = 0
    for _ in range(n_sa):
        grp = 0
        # SpaSSMBlock: depth * BMDMambaLayer + 1 conv2d
        grp += vss_depth * bmd_single + count_conv2d(C, C, 3)
        # AngSSMBlock: depth * BMDMambaLayer + 1 conv2d
        grp += vss_depth * bmd_single + count_conv2d(C, C, 3)
        # SAM
        sam = 0
        sam += count_conv2d(C, C//4, 1) + count_conv2d(C//4, C, 1)  # spa_attn
        sam += count_conv2d(C, C//4, 1) + count_conv2d(C//4, C, 1)  # ang_attn
        sam += count_conv2d(C*2, C, 1)                                # fuse
        grp += sam
        # FASS
        fass = 0
        fass += count_conv2d(C, C, 5, groups=C) + count_conv2d(C, C, 1)  # low_pass
        fass += count_conv2d(C, C//4, 1) + count_conv2d(C//4, C//4, 3, groups=C//4) + count_conv2d(C//4, C, 1)  # hf_refine
        fass += count_conv2d(C, C//4, 1, bias=False) + count_conv2d(C//4, C, 1, bias=False)  # gate (inside pool)
        fass += 1  # scale param
        grp += fass
        # res_scale
        grp += 1
        sa_total += grp
    total += sa_total
    print(f"SpaAng Groups (×{n_sa}):   {sa_total:>10,}")

    # ---- MODULE 3: EPI Groups ----
    epi_total = 0
    for _ in range(n_epi):
        grp = 0
        # h_epi: EPIMambaBlock (depth * BMDMamba + conv2d)
        grp += vss_depth * bmd_single + count_conv2d(C, C, 3)
        # v_epi: EPIMambaBlock
        grp += vss_depth * bmd_single + count_conv2d(C, C, 3)
        # CAB
        cab = 0
        cab += count_conv2d(C, C//3, 3) + count_conv2d(C//3, C, 3, bias=True)  # 2 convs
        # ChannelAttention inside CAB
        squeeze = max(1, C // 16)
        cab += count_conv2d(C, squeeze, 1, bias=True) + count_conv2d(squeeze, C, 1, bias=True)
        grp += cab
        # FASS
        fass = 0
        fass += count_conv2d(C, C, 5, groups=C) + count_conv2d(C, C, 1)
        fass += count_conv2d(C, C//4, 1) + count_conv2d(C//4, C//4, 3, groups=C//4) + count_conv2d(C//4, C, 1)
        fass += count_conv2d(C, C//4, 1, bias=False) + count_conv2d(C//4, C, 1, bias=False)
        fass += 1
        grp += fass
        # res_scale
        grp += 1
        epi_total += grp
    total += epi_total
    print(f"EPI Groups (×{n_epi}):     {epi_total:>10,}")

    # ---- MODULE 4: Window Attention ----
    wa = 0
    wa += count_layernorm(C)
    wa += count_linear(C, C*3)         # qkv
    wa += count_linear(C, C)           # proj
    wa += 1                             # attn_scale
    wa += (2*window_size-1)**2 * num_heads  # rpb_table
    total += wa
    print(f"Window Attention:      {wa:>10,}")

    # ---- MODULE 5: Aggregation ----
    agg = count_conv2d(C*3, C, 1)
    total += agg
    print(f"Aggregation:           {agg:>10,}")

    # ---- MODULE 6: Reconstruction Head ----
    recon = 0
    recon += count_conv2d(C, C, 3) * 2       # refine (2 convs)
    # Channel attention
    hidden = max(C // 16, 8)
    recon += count_conv2d(C, hidden, 1, bias=True) + count_conv2d(hidden, C, 1, bias=True)
    # PixelShuffle 4x (two 2x stages)
    recon += count_conv2d(C, C*4, 3)          # first stage
    recon += count_conv2d(C, C*4, 3)          # second stage
    # Output conv
    recon += count_conv2d(C, 1, 3, bias=True)
    total += recon
    print(f"Reconstruction Head:   {recon:>10,}")

    # ---- TOTALS ----
    limit = 1_000_000
    print(f"\n{'='*50}")
    print(f"TOTAL PARAMETERS:      {total:>10,}")
    print(f"NTIRE Track 2 LIMIT:   {limit:>10,}")
    print(f"Usage:                 {total/limit*100:>9.1f}%")
    if total < limit:
        print(f"✅ PASS — {limit - total:,} params under budget")
    else:
        print(f"❌ FAIL — {total - limit:,} params OVER budget")
        print(f"\nTo fit Track 2, consider reducing:")
        print(f"  C:     64 → 32 or 24")
        print(f"  n_sa:  4  → 2 or 1")
        print(f"  n_epi: 3  → 2 or 1")

    return total

if __name__ == "__main__":
    main()
