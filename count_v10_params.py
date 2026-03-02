"""
V10.1 Analytical Parameter Counter
===================================
Reflects V10.1 changes:
  1. EPI weight sharing (h_epi/v_epi → single epi_block)
  2. FASS simplified gate (SE-block → per-channel Parameter)
  3. PixelShuffle 1×1 → 3×3
  4. Window attention ws=4 → 8
"""

def count_conv2d(cin, cout, k, bias=False, groups=1):
    return cout * (cin // groups) * k * k + (cout if bias else 0)

def count_conv3d(cin, cout, k_tuple, bias=False):
    kd, kh, kw = k_tuple
    return cout * cin * kd * kh * kw + (cout if bias else 0)

def count_linear(cin, cout, bias=False):
    return cin * cout + (cout if bias else 0)

def count_layernorm(dim):
    return 2 * dim

def count_mamba(d_model, d_state, d_conv, expand):
    d_inner = int(d_model * expand)
    in_proj = d_model * d_inner * 2
    conv1d = d_inner * d_conv + d_inner
    dt_rank = max(1, d_model // 16)
    x_proj = d_inner * (dt_rank + 2 * d_state)
    dt_proj = dt_rank * d_inner + d_inner
    A_log = d_inner * d_state
    D = d_inner
    out_proj = d_inner * d_model
    return in_proj + conv1d + x_proj + dt_proj + A_log + D + out_proj

def main():
    C = 48
    n_sa = 3       # V10.2: was 2
    n_epi = 2
    d_state = 16
    d_conv = 4
    expand = 2.0
    vss_depth = 2
    angRes = 5
    scale = 4
    num_heads = 4
    window_size = 8  # V10.1: was 4

    total = 0

    # MLFIM mask token
    mask_token = C
    total += mask_token
    print(f"MLFIM mask_token:      {mask_token:>10,}")

    # MODULE 1: 3D Conv IFE
    ife = count_conv3d(1, C, (1,3,3)) + count_conv3d(C, C, (1,3,3)) * 3
    total += ife
    print(f"IFE (3D Conv):         {ife:>10,}")

    # BMDMambaLayer
    def bmd_layer_params():
        p = count_layernorm(C) + count_mamba(C, d_state, d_conv, expand)
        p += count_conv2d(C, C, 1) + C  # dir_fusion + skip_scale
        return p

    bmd_single = bmd_layer_params()
    print(f"  (Single BMDMambaLayer: {bmd_single:,})")

    # EPIMambaBlock = vss_depth × BMDMamba + Conv2d(C,C,3)
    epi_block = vss_depth * bmd_single + count_conv2d(C, C, 3)
    print(f"  (Single EPIMambaBlock: {epi_block:,})")

    # MODULE 2: Spatial-Angular Groups
    sa_total = 0
    for _ in range(n_sa):
        grp = 0
        grp += vss_depth * bmd_single + count_conv2d(C, C, 3)  # SpaSSM
        grp += vss_depth * bmd_single + count_conv2d(C, C, 3)  # AngSSM
        # SAM
        sam = count_conv2d(C, C//4, 1) + count_conv2d(C//4, C, 1)  # spa_attn
        sam += count_conv2d(C, C//4, 1) + count_conv2d(C//4, C, 1)  # ang_attn
        sam += count_conv2d(C*2, C, 1)  # fuse
        grp += sam
        # FASS (V10.1: simplified gate = C params instead of SE block)
        fass = count_conv2d(C, C, 5, groups=C) + count_conv2d(C, C, 1)  # low_pass
        fass += count_conv2d(C, C//4, 1) + count_conv2d(C//4, C//4, 3, groups=C//4) + count_conv2d(C//4, C, 1)  # refine
        fass += C  # V10.1: gate = Parameter(1, C, 1, 1)
        fass += 1  # scale
        grp += fass
        grp += 1  # res_scale
        sa_total += grp
    total += sa_total
    print(f"SpaAng Groups (×{n_sa}):   {sa_total:>10,}")

    # MODULE 3: EPI Groups (V10.1: shared epi_block)
    epi_total = 0
    for _ in range(n_epi):
        grp = 0
        # V10.1: ONE shared EPIMambaBlock (not two!)
        grp += epi_block  # single shared block
        # CAB
        cab = count_conv2d(C, C//3, 3) + count_conv2d(C//3, C, 3, bias=True)
        squeeze = max(1, C // 16)
        cab += count_conv2d(C, squeeze, 1, bias=True) + count_conv2d(squeeze, C, 1, bias=True)
        grp += cab
        # FASS (V10.1: simplified gate)
        fass = count_conv2d(C, C, 5, groups=C) + count_conv2d(C, C, 1)
        fass += count_conv2d(C, C//4, 1) + count_conv2d(C//4, C//4, 3, groups=C//4) + count_conv2d(C//4, C, 1)
        fass += C  # V10.1: gate parameter
        fass += 1  # scale
        grp += fass
        grp += 1  # res_scale
        epi_total += grp
    total += epi_total
    print(f"EPI Groups (×{n_epi}):     {epi_total:>10,}")

    # MODULE 4: Window Attention (V10.1: ws=8)
    wa = count_layernorm(C)
    wa += count_linear(C, C*3)  # qkv
    wa += count_linear(C, C)    # proj
    wa += 1                      # attn_scale
    wa += (2*window_size-1)**2 * num_heads  # rpb_table
    total += wa
    print(f"Window Attention:      {wa:>10,}")

    # MODULE 5: Aggregation
    agg = count_conv2d(C*3, C, 1)
    total += agg
    print(f"Aggregation:           {agg:>10,}")

    # MODULE 6: Reconstruction Head (V10.1: 3×3 PixelShuffle)
    recon = count_conv2d(C, C, 3) * 2  # refine
    hidden = max(C // 16, 8)
    recon += count_conv2d(C, hidden, 1, bias=True) + count_conv2d(hidden, C, 1, bias=True)
    # V10.1: 3×3 conv before PixelShuffle (was 1×1)
    recon += count_conv2d(C, C*4, 3)  # first 2× stage
    recon += count_conv2d(C, C*4, 3)  # second 2× stage
    recon += count_conv2d(C, 1, 3, bias=True)  # output
    total += recon
    print(f"Reconstruction Head:   {recon:>10,}")

    # TOTALS
    limit = 1_000_000
    print(f"\n{'='*50}")
    print(f"TOTAL PARAMETERS:      {total:>10,}")
    print(f"NTIRE Track 2 LIMIT:   {limit:>10,}")
    print(f"Usage:                 {total/limit*100:>9.1f}%")
    if total < limit:
        print(f"✅ PASS — {limit - total:,} params under budget")
    else:
        print(f"❌ FAIL — {total - limit:,} params OVER budget")

if __name__ == "__main__":
    main()
