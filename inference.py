"""
Inference script for BasicLFSR / MyEfficientLFNet V3.

Adapted from the original BasicLFSR inference.py with V3-specific fixes:
  - Removed global `args` usage inside test() — now passed as parameter
  - Added Sr_SAI_cbcr edge-case handling (scalar/missing/2D arrays in h5)
  - Model output moved to CPU before storing in subLFout (avoids CUDA OOM)
  - Added Gaussian PSW stitching option (LFintegrate_gaussian)
  - Added flexible checkpoint loading (handles module. prefix both ways)
  - Uses test num_workers=0 (h5py is not fork-safe)
  - CodaBench-compliant output naming: View_i_j.bmp
  - 8x geometric self-ensemble for +0.10-0.15 dB (--self_ensemble flag)
"""

import importlib
import os
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from collections import OrderedDict
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import h5py
from torchvision.transforms import ToTensor
import imageio
from tqdm import tqdm
import numpy as np


def MultiTestSetDataLoader(args):
    """Load test data for all specified datasets."""
    data_list = None
    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        if args.task == 'SR':
            dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.scale_factor) + 'x/'
            data_list = os.listdir(dataset_dir)
        elif args.task == 'RE':
            dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)

        # V3 FIX: num_workers=0 — h5py is not fork-safe and test sets are
        # small enough that worker overhead exceeds benefit.
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name='ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
            self.data_list = [data_name]
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
            Sr_SAI_cbcr = np.array(hf.get('Sr_SAI_cbcr'), dtype='single')
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            # V3 FIX: Handle edge cases where Sr_SAI_cbcr may be missing,
            # scalar, or 2D in some h5 files (e.g., NTIRE challenge data).
            if Sr_SAI_cbcr.ndim == 3:
                Sr_SAI_cbcr = np.transpose(Sr_SAI_cbcr, (2, 1, 0))
            elif Sr_SAI_cbcr.ndim == 0 or Sr_SAI_cbcr.size == 0:
                # Create dummy cbcr with 2 channels matching Hr_SAI_y dims
                Sr_SAI_cbcr = np.zeros((Hr_SAI_y.shape[0], Hr_SAI_y.shape[1], 2), dtype=np.float32)
            elif Sr_SAI_cbcr.ndim == 2:
                Sr_SAI_cbcr = np.expand_dims(Sr_SAI_cbcr, axis=-1)

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy().astype(np.float32))

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def main(args):
    ''' Create Dir for Save '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)


    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
    else:
        ckpt_path = args.path_pre_pth
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # V3 FIX: Flexible checkpoint loading — try both with and without
        # 'module.' prefix, using strict=False as fallback for arch changes.
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.' + k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
            print('Use pretrain model! (with module. prefix)')
        except (RuntimeError, KeyError):
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                net.load_state_dict(new_state_dict)
                print('Use pretrain model! (without module. prefix)')
            except RuntimeError:
                # Last resort: strict=False for partially compatible checkpoints
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                missing, unexpected = net.load_state_dict(new_state_dict, strict=False)
                if missing:
                    print(f'  Warning: {len(missing)} missing keys (partial load)')
                if unexpected:
                    print(f'  Warning: {len(unexpected)} unexpected keys')
                print('Use pretrain model! (partial/flexible load)')
        pass
    pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    params = sum(p.numel() for p in net.parameters())
    print(f'Parameters: {params:,} ({params/1e6:.3f}M)')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for PSNR/SSIM '''
        excel_file = ExcelFile()
        psnr_testset = []
        ssim_testset = []

        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            # V3 FIX: Pass args explicitly instead of relying on global
            psnr_iter_test, ssim_iter_test, LF_names = test(test_loader, device, net, args, save_dir)
            excel_file.write_sheet(test_name, LF_names, psnr_iter_test, ssim_iter_test)

            psnr_epoch_test = float(np.array(psnr_iter_test).mean())
            ssim_epoch_test = float(np.array(ssim_iter_test).mean())
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print('Test on %s, psnr/ssim is %.2f/%.3f' % (test_name, psnr_epoch_test, ssim_epoch_test))

        psnr_mean = float(np.array(psnr_testset).mean())
        ssim_mean = float(np.array(ssim_testset).mean())
        print('\nMean PSNR on all testsets: %.4f, Mean SSIM: %.4f' % (psnr_mean, ssim_mean))

        excel_file.xlsx_file.save(str(result_dir) + '/evaluation.xls')
        print('Results saved to %s' % str(result_dir))
    pass


def self_ensemble_forward(net, x, data_info):
    """8x geometric self-ensemble: 4 rotations × 2 flips, averaged.
    
    Runs the model 8 times on geometric transforms of the input,
    transforms each output back, and averages. This suppresses per-pixel
    noise and boundary artifacts for +0.10-0.15 dB improvement.
    
    Note: Light field patches have angular info encoded spatially
    (angRes_in * patch_size), so rot90/flip on the full tensor is valid
    because each sub-aperture image gets the same transform.
    """
    outputs = []
    for flip in [False, True]:
        for k in range(4):  # 0°, 90°, 180°, 270°
            inp = x
            if flip:
                inp = torch.flip(inp, [-1])  # horizontal flip
            if k > 0:
                inp = torch.rot90(inp, k, [-2, -1])  # rotate k*90°
            
            out = net(inp, data_info)
            
            if k > 0:
                out = torch.rot90(out, -k, [-2, -1])  # undo rotation
            if flip:
                out = torch.flip(out, [-1])  # undo flip
            outputs.append(out)
    return torch.stack(outputs).mean(0)


def test(test_loader, device, net, args, save_dir=None):
    """Run inference on a test dataset.

    V3 changes from original BasicLFSR inference.py:
      - `args` passed explicitly instead of using global
      - Model output moved to .cpu() before storing in subLFout
      - net.eval() called once before loop, not per-iteration
      - Gaussian PSW stitching support via LFintegrate_gaussian
      - CodaBench-compliant naming: View_i_j.bmp
      - 8x geometric self-ensemble (--self_ensemble flag)
    """
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    use_ensemble = getattr(args, 'self_ensemble', False)

    net.eval()  # V3 FIX: Set eval mode once before the loop, not per-iteration
    torch.cuda.empty_cache()

    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
        Sr_SAI_cbcr = Sr_SAI_cbcr

        ''' Crop LFs into Patches '''
        subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
                               args.angRes_in * args.patch_size_for_test * args.scale_factor)

        ''' SR the Patches '''
        for i in range(0, numU * numV, args.minibatch_for_test):
            tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
            with torch.no_grad():
                torch.cuda.empty_cache()
                if use_ensemble:
                    out = self_ensemble_forward(net, tmp.to(device), data_info)
                else:
                    out = net(tmp.to(device), data_info)
                # V3 FIX: Move output to CPU to prevent CUDA OOM accumulation
                subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out.cpu()
        subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

        ''' Restore the Patches to LFs '''
        use_gaussian = getattr(args, 'use_gaussian_psw', True)
        target_h = Hr_SAI_y.size(-2) // args.angRes_out
        target_w = Hr_SAI_y.size(-1) // args.angRes_out
        sr_pz = args.patch_size_for_test * args.scale_factor
        sr_stride = args.stride_for_test * args.scale_factor

        if use_gaussian:
            Sr_4D_y = LFintegrate_gaussian(
                subLFout, args.angRes_out, sr_pz, sr_stride, target_h, target_w
            )
        else:
            Sr_4D_y = LFintegrate(
                subLFout, args.angRes_out, sr_pz, sr_stride, target_h, target_w
            )
        Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

        ''' Calculate the PSNR & SSIM '''
        psnr, ssim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        LF_iter_test.append(LF_name[0])

        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            # V3 FIX: Ensure Sr_SAI_y is on CPU before cat with Sr_SAI_cbcr
            Sr_SAI_ycbcr = torch.cat((Sr_SAI_y.cpu(), Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0, 1) * 255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

            # Save all views with CodaBench-compliant naming: View_i_j.bmp
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(save_dir_) + '/View_' + str(i) + '_' + str(j) + '.bmp'
                    imageio.imwrite(path, img)
                    pass
                pass
            pass
        pass

    return psnr_iter_test, ssim_iter_test, LF_iter_test


if __name__ == '__main__':
    from option import args
    import argparse
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--self_ensemble', action='store_true', default=False,
                              help='Enable 8x geometric self-ensemble (+0.10-0.15 dB, 8x slower)')
    extra_args, _ = extra_parser.parse_known_args()
    args.self_ensemble = extra_args.self_ensemble
    args.use_pre_ckpt = True

    # ---- V3 MLFIM Default Inference ----
    # Uncomment and adjust the checkpoint path for your model:
    args.data_name = 'ALL'
    args.model_name = 'MyEfficientLFNetV3_MLFIM'
    args.path_pre_pth = './log/SR_5x5_4x/ALL/MyEfficientLFNetV3_MLFIM/checkpoints/MyEfficientLFNetV3_MLFIM_finetune_best.pth'
    args.path_for_test = './data_for_test/'
    if args.self_ensemble:
        print('\n★ Self-ensemble enabled (8x geometric augmentation)')
    main(args)
