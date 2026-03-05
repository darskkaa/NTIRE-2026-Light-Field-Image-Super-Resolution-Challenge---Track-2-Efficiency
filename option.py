"""
CLI argument parser for BasicLFSR / MyEfficientLFNet V3.

Adapted from the original BasicLFSR option.py with V3-specific fixes:
  - scale_factor default: 2 → 4 (Track 2 is 4× SR)
  - use_pre_ckpt: type=bool → action='store_true' (type=bool is broken in
    argparse — bool('False') returns True!)
  - parse_args() → parse_known_args() so train_mlfim_v3.py can add its own
    extra args (--stage, --mlfim_mask_ratio, etc.) without crashing
  - epoch default: 51 → 150 (reasonable for V3 training)
  - Removed decay_rate, n_steps, gamma (V3 uses StepLR ×0.5/25ep, matching LFTransMamba)
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='SR', help='SR, RE')

# LF_SR
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")

parser.add_argument('--model_name', type=str, default='MyEfficientLFNetV3_MLFIM', help="model name")
# V3 FIX: use action='store_true' instead of type=bool.
# In argparse, type=bool means bool('any_string') → True, so
# --use_pre_ckpt False would be parsed as True. This is a well-known
# Python argparse gotcha.
parser.add_argument("--use_pre_ckpt", action='store_true', default=False, help="use pre model ckpt")
parser.add_argument("--path_pre_pth", type=str, default='./pth/', help="path for pre model ckpt")
parser.add_argument('--data_name', type=str, default='ALL',
                    help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)')
parser.add_argument('--path_for_train', type=str, default='./data_for_training/')
parser.add_argument('--path_for_test', type=str, default='./data_for_test/')
parser.add_argument('--path_log', type=str, default='./log/')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--epoch', default=150, type=int, help='Epoch to run [default: 150]')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=2, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )

# NOTE: Masked pre-training (MLFIM) is now handled INSIDE the model's
# forward() method. No external args needed.

# V3 FIX: Use parse_known_args() instead of parse_args() so that
# train_mlfim_v3.py can add its own arguments (--stage, --mlfim_mask_ratio,
# --grad_accum_steps, --loss_type, --beta2, --weight_decay, --eta_min)
# without argparse raising an "unrecognized arguments" error.
args, _ = parser.parse_known_args()



if args.task == 'SR':
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    args.patch_size_for_test = 32
    args.stride_for_test = 8  # LFTransMamba: stride=8 with patch_size=32 (75% overlap)
    args.minibatch_for_test = 1

del args.angRes