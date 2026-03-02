import sys
import math
import torch
import torch.nn as nn

class DummyMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2.0, **kwargs):
        super().__init__()
        d_inner = int(expand * d_model)
        dt_rank = math.ceil(d_model / 16)
        
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # d_state matrices
        self.A_log = nn.Parameter(torch.empty(d_inner, d_state))
        self.D = nn.Parameter(torch.empty(d_inner))

sys.modules['mamba_ssm'] = type('mamba_ssm', (), {'Mamba': DummyMamba})

from model.SR.MyEfficientLFNetV10 import get_model
import argparse

args = argparse.Namespace(angRes_in=5, scale_factor=4, mlfim_mask_ratio=0.0)
model = get_model(args)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,} ({params / 1e6:.6f} M)")
