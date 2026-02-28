import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Comprehensive LFSR Data Pre-flight Checker")
parser.add_argument('--train_dir', type=str, default='./data_for_training/', help='Path to training data')
parser.add_argument('--test_dir', type=str, default='./data_for_test/', help='Path to test data')
parser.add_argument('--angRes', type=int, default=5, help='Angular resolution (e.g., 5 for 5x5)')
parser.add_argument('--scale_factor', type=int, default=4, help='Scale factor (e.g., 4)')
parser.add_argument('--check_values', type=str2bool, default=True, help='Check for NaNs, Infs, and value ranges (slower but safer)')
args = parser.parse_args()

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg, status="OK"):
    if status == "OK":
        print(f"[{Colors.GREEN}{Colors.BOLD} OK {Colors.ENDC}] {msg}")
    elif status == "WARN":
        print(f"[{Colors.YELLOW}{Colors.BOLD}WARN{Colors.ENDC}] {msg}")
    elif status == "FAIL":
        print(f"[{Colors.RED}{Colors.BOLD}FAIL{Colors.ENDC}] {msg}")
    else:
        print(f"[{Colors.BLUE}{Colors.BOLD}INFO{Colors.ENDC}] {msg}")

def check_h5_file(filepath, is_train, args):
    issues = []
    
    try:
        with h5py.File(filepath, 'r') as hf:
            # 1. Check required keys exist
            required_keys = ['Lr_SAI_y', 'Hr_SAI_y']
            if not is_train:
                required_keys.append('Sr_SAI_cbcr')
                
            for key in required_keys:
                if key not in hf.keys():
                    issues.append(f"Missing key: {key}")
                    continue
            
            if issues: return issues # Stop here if keys are missing
            
            # 2. Extract shapes and types
            lr = hf['Lr_SAI_y']
            hr = hf['Hr_SAI_y']
            
            lr_shape = lr.shape
            hr_shape = hr.shape
            
            if not np.issubdtype(lr.dtype, np.floating) or not np.issubdtype(hr.dtype, np.floating):
                issues.append(f"Data type is not float. Lr: {lr.dtype}, Hr: {hr.dtype}")
                
            # 3. Shape validation (Note: data in h5 is transposed W,H)
            expected_lr_pixels = (hr_shape[0] // args.scale_factor) * (hr_shape[1] // args.scale_factor)
            actual_lr_pixels = lr_shape[0] * lr_shape[1]
            
            if expected_lr_pixels != actual_lr_pixels:
                issues.append(f"Scale mismatch. Lr shape: {lr_shape}, Hr shape: {hr_shape}, factor: {args.scale_factor}")
                
            # Check angular resolution constraints (H and W should be divisible by angRes)
            # Shapes are (W*ang, H*ang)
            if hr_shape[0] % args.angRes != 0 or hr_shape[1] % args.angRes != 0:
                issues.append(f"Hr shape {hr_shape} not divisible by angRes {args.angRes}")
                
            if lr_shape[0] % args.angRes != 0 or lr_shape[1] % args.angRes != 0:
                issues.append(f"Lr shape {lr_shape} not divisible by angRes {args.angRes}")
                
            # 4. Deep value checking (NaNs, Infs, Ranges)
            if args.check_values:
                # Load a small sample to avoid memory explosion if files are huge
                # For training patches, we can load the whole thing.
                # For full test images, we load just a chunk.
                if is_train:
                    lr_data = lr[:]
                    hr_data = hr[:]
                else:
                    h_chunk = min(100, lr_shape[0])
                    w_chunk = min(100, lr_shape[1])
                    lr_data = lr[:h_chunk, :w_chunk]
                    hr_data = hr[:h_chunk * args.scale_factor, :w_chunk * args.scale_factor]

                if np.isnan(lr_data).any() or np.isnan(hr_data).any():
                    issues.append("NaN values detected in tensors!")
                if np.isinf(lr_data).any() or np.isinf(hr_data).any():
                    issues.append("Inf values detected in tensors!")
                    
                # Light field data is almost always normalized [0, 1] or [0, 255]
                # We expect [0, 1] based on standard LFSR pipelines, but warn if outside expected bounds.
                # Data generation script uses single precision float
                lr_min, lr_max = np.min(lr_data), np.max(lr_data)
                hr_min, hr_max = np.min(hr_data), np.max(hr_data)
                
                # Check for extreme out of bounds
                if lr_max > 256.0 or hr_max > 256.0 or lr_min < -1.0 or hr_min < -1.0:
                    issues.append(f"Abnormal value ranges: Lr[{lr_min:.2f}, {lr_max:.2f}], Hr[{hr_min:.2f}, {hr_max:.2f}]")
                    
    except Exception as e:
        issues.append(f"File unreadable or corrupt: {str(e)}")
        
    return issues


def main():
    print(f"\n{Colors.BOLD}🚀 Starting LFSR Comprehensive Data Audit{Colors.ENDC}")
    print("=" * 60)
    print(f"Target Scale:   {args.scale_factor}x")
    print(f"Target angRes:  {args.angRes}x{args.angRes}")
    print(f"Value Checking: {'Enabled (Detects NaN/Inf/Out-of-bounds)' if args.check_values else 'Disabled (Shape/Key checks only)'}")
    print("=" * 60 + "\n")

    # --- 1. Check Training Data ---
    train_path = Path(args.train_dir)
    print(f"🔍 {Colors.BOLD}Scanning Training Directory:{Colors.ENDC} {train_path}")
    
    if not train_path.exists():
        print_status(f"Directory {train_path} does not exist!", "FAIL")
        train_h5_files = []
    else:
        train_datasets = [d for d in train_path.iterdir() if d.is_dir()]
        train_h5_files = []
        for d in train_datasets:
            files = list(d.glob('*.h5'))
            train_h5_files.extend(files)
            print_status(f"Found {len(files):5d} patches in {d.name}", "INFO")

        if not train_h5_files:
            print_status("No .h5 training files found! Run Generate_Data_for_Training.py first.", "WARN")
        else:
            print(f"\nVerifying {len(train_h5_files)} training patches...")
            bad_train_files = 0
            for f in tqdm(train_h5_files, ncols=80):
                issues = check_h5_file(f, is_train=True, args=args)
                if issues:
                    tqdm.write(f"\n{Colors.RED}❌ Error in {f.name}:{Colors.ENDC}")
                    for issue in issues: tqdm.write(f"   - {issue}")
                    bad_train_files += 1

            if bad_train_files == 0:
                print_status(f"All {len(train_h5_files)} training patches are healthy!", "OK")
            else:
                print_status(f"{bad_train_files} corrupted/invalid training files found.", "FAIL")

    print("\n" + "-" * 60 + "\n")

    # --- 2. Check Test Data ---
    test_path = Path(args.test_dir)
    print(f"🔍 {Colors.BOLD}Scanning Test Directory:{Colors.ENDC} {test_path}")
    
    if not test_path.exists():
        print_status(f"Directory {test_path} does not exist!", "FAIL")
        test_h5_files = []
    else:
        test_datasets = [d for d in test_path.iterdir() if d.is_dir()]
        test_h5_files = []
        for d in test_datasets:
            files = list(d.glob('*.h5'))
            test_h5_files.extend(files)
            print_status(f"Found {len(files):5d} images in {d.name}", "INFO")

        if not test_h5_files:
            print_status("No .h5 test files found! Run Generate_Data_for_Test.py first.", "WARN")
        else:
            print(f"\nVerifying {len(test_h5_files)} test images...")
            bad_test_files = 0
            for f in tqdm(test_h5_files, ncols=80):
                issues = check_h5_file(f, is_train=False, args=args)
                if issues:
                    tqdm.write(f"\n{Colors.RED}❌ Error in {f.name}:{Colors.ENDC}")
                    for issue in issues: tqdm.write(f"   - {issue}")
                    bad_test_files += 1

            if bad_test_files == 0:
                print_status(f"All {len(test_h5_files)} test images are healthy!", "OK")
            else:
                print_status(f"{bad_test_files} corrupted/invalid test files found.", "FAIL")


    # --- 3. Final Verdict ---
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}🏁 FINAL VERDICT{Colors.ENDC}")
    print("=" * 60)
    
    if len(train_h5_files) == 0 and len(test_h5_files) == 0:
        print(f"  {Colors.YELLOW}⚠️  No data found to check. Generate data first.{Colors.ENDC}")
    elif bad_train_files == 0 and bad_test_files == 0:
        print(f"  {Colors.GREEN}✅ ALL DATA IS HEALTHY AND READY FOR TRAINING.{Colors.ENDC}")
        print(f"  {Colors.GREEN}✅ No NaNs, exact shape matches, valid keys found.{Colors.ENDC}")
    else:
        print(f"  {Colors.RED}❌ DATA CORRUPTION DETECTED. DO NOT START TRAINING.{Colors.ENDC}")
        print("     Please fix the errors listed above or regenerate the data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
