import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

# Fix for einops rearrange issue when running standalone
import sys
sys.path.insert(0, '.')

from einops import rearrange
from model.SR.MyEfficientLFNetV3_MLFIM import get_model
from utils.utils import LFdivide, LFintegrate, LFintegrate_gaussian, cal_metrics
from utils.utils_datasets import MultiTestSetDataLoader

def parse_args():
    parser = argparse.ArgumentParser("Quick Validation Script")
    parser.add_argument('--model_name', type=str, default='MyEfficientLFNetV3_MLFIM')
    parser.add_argument('--ckpt', type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument('--path_for_test', type=str, default='./data_for_test/')
    parser.add_argument('--data_name', type=str, default='ALL')
    parser.add_argument('--angRes_in', type=int, default=5)
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--patch_size_for_test', type=int, default=32)
    parser.add_argument('--stride_for_test', type=int, default=8)
    parser.add_argument('--minibatch_for_test', type=int, default=1)
    parser.add_argument('--use_gaussian_psw', action='store_true', default=True)
    
    # Required for DatasetLoader
    parser.add_argument('--task', type=str, default='SR')
    # Required for model init (from train_mlfim_v3.py)
    parser.add_argument('--mlfim_mask_ratio', type=float, default=0.0)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Model
    print(f"Loading {args.model_name}...")
    net = get_model(args).to(device)
    
    checkpoint = torch.load(args.ckpt, map_location=device)
    # Handle possible DDP module prefix and EMA
    state_dict = checkpoint.get('ema_state_dict', checkpoint.get('state_dict', checkpoint))
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(cleaned_state_dict, strict=True)
    net.eval()
    print("Checkpoint loaded successfully!")

    # 2. Load Datasets
    print("Loading test datasets...")
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)

    # 3. Predict & Evaluate
    all_psnrs = []
    
    with torch.no_grad():
        for index, test_name in enumerate(test_Names):
            print(f"\nEvaluating on {test_name}:")
            test_loader = test_Loaders[index]
            psnr_dataset = []
            
            for (Lr_SAI_y, Hr_SAI_y, _, data_info, LF_name) in tqdm(test_loader, ncols=80):
                Lr_angRes_in, Lr_angRes_out = data_info[0], data_info[1]
                data_info = [Lr_angRes_in[0].item(), Lr_angRes_out[0].item()]

                Lr_SAI_y = Lr_SAI_y.squeeze().to(device) 
                
                # --- Divide ---
                subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
                numU, numV, H, W = subLFin.size()
                subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
                subLFout = torch.zeros(
                    numU * numV, 1, 
                    args.angRes_in * args.patch_size_for_test * args.scale_factor,
                    args.angRes_in * args.patch_size_for_test * args.scale_factor
                )

                # --- Predict ---
                for i in range(0, numU * numV, args.minibatch_for_test):
                    tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV)]
                    out = net(tmp.to(device), data_info)
                    subLFout[i:min(i + args.minibatch_for_test, numU * numV)] = out.cpu()
                
                subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

                # --- Integrate ---
                sr_pz = args.patch_size_for_test * args.scale_factor
                sr_stride = args.stride_for_test * args.scale_factor
                target_h = Hr_SAI_y.size(-2) // args.angRes_out
                target_w = Hr_SAI_y.size(-1) // args.angRes_out
                
                if args.use_gaussian_psw:
                    Sr_4D_y = LFintegrate_gaussian(
                        subLFout, args.angRes_out, sr_pz, sr_stride, target_h, target_w)
                else:
                    Sr_4D_y = LFintegrate(
                        subLFout, args.angRes_out, sr_pz, sr_stride, target_h, target_w)

                # --- Metrics ---
                psnr, _ = cal_metrics(args, Hr_SAI_y, Sr_4D_y)
                psnr_dataset.append(psnr)
            
            psnr_avg_dataset = float(np.mean(psnr_dataset))
            all_psnrs.append(psnr_avg_dataset)
            print(f"[{test_name}] Average PSNR: {psnr_avg_dataset:.2f} dB")

    # 4. Final Result
    print(f"\n=====================================")
    print(f"OVERALL AGGREGATE PSNR: {np.mean(all_psnrs):.2f} dB")
    print(f"=====================================")

if __name__ == '__main__':
    main()
