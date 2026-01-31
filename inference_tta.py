"""
Inference with Test-Time Augmentation (TTA) for NTIRE 2026 LF-SR Challenge

8x Geometric TTA:
- Original
- 90° rotation
- 180° rotation  
- 270° rotation
- Horizontal flip
- Vertical flip
- Horizontal flip + 90° rotation
- Vertical flip + 90° rotation

Expected gain: +0.1-0.3 dB PSNR over single inference
"""

import importlib
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.utils import *
from collections import OrderedDict
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import h5py
from torchvision.transforms import ToTensor
import imageio
from tqdm import tqdm


def apply_augmentation(x, aug_type):
    """Apply augmentation to input tensor [B, C, H, W]."""
    if aug_type == 0:  # Original
        return x
    elif aug_type == 1:  # 90° rotation
        return torch.rot90(x, k=1, dims=[2, 3])
    elif aug_type == 2:  # 180° rotation
        return torch.rot90(x, k=2, dims=[2, 3])
    elif aug_type == 3:  # 270° rotation
        return torch.rot90(x, k=3, dims=[2, 3])
    elif aug_type == 4:  # Horizontal flip
        return torch.flip(x, dims=[3])
    elif aug_type == 5:  # Vertical flip
        return torch.flip(x, dims=[2])
    elif aug_type == 6:  # Horizontal flip + 90° rotation
        return torch.rot90(torch.flip(x, dims=[3]), k=1, dims=[2, 3])
    elif aug_type == 7:  # Vertical flip + 90° rotation
        return torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[2, 3])
    return x


def reverse_augmentation(x, aug_type):
    """Reverse augmentation on output tensor [B, C, H, W]."""
    if aug_type == 0:  # Original
        return x
    elif aug_type == 1:  # 90° rotation -> -90°
        return torch.rot90(x, k=3, dims=[2, 3])
    elif aug_type == 2:  # 180° rotation -> 180°
        return torch.rot90(x, k=2, dims=[2, 3])
    elif aug_type == 3:  # 270° rotation -> 90°
        return torch.rot90(x, k=1, dims=[2, 3])
    elif aug_type == 4:  # Horizontal flip
        return torch.flip(x, dims=[3])
    elif aug_type == 5:  # Vertical flip
        return torch.flip(x, dims=[2])
    elif aug_type == 6:  # Horizontal flip + 90° rotation -> -90° + H flip
        return torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[3])
    elif aug_type == 7:  # Vertical flip + 90° rotation -> -90° + V flip
        return torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
    return x


def MultiTestSetDataLoader(args):
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
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

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
            Sr_SAI_cbcr = np.transpose(Sr_SAI_cbcr, (2, 1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def main(args):
    """Main function with TTA inference."""
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST_TTA')
    result_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    print('\n' + '='*60)
    print('🚀 TTA INFERENCE - 8x Geometric Augmentation')
    print('='*60)

    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)

    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    # Load checkpoint
    ckpt_path = args.path_pre_pth
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    try:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = 'module.' + k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k] = v
        net.load_state_dict(new_state_dict)
    print('✓ Loaded pretrained model from:', ckpt_path)

    net = net.to(device)
    net.eval()
    cudnn.benchmark = True

    print('\nPARAMETER ...')
    print(args)

    print('\nStart TTA inference...')
    with torch.no_grad():
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]
            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)
            test_with_tta(test_loader, device, net, args, save_dir)

    print('\n' + '='*60)
    print('✅ TTA INFERENCE COMPLETE!')
    print('='*60)


def test_with_tta(test_loader, device, net, args, save_dir=None):
    """Test with 8x geometric TTA."""
    num_augs = 8  # Number of augmentations
    
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(
            enumerate(test_loader), total=len(test_loader), ncols=70):
        
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)
        Sr_SAI_cbcr = Sr_SAI_cbcr

        # Crop LFs into Patches
        subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        
        # Accumulator for TTA results
        subLFout_accum = torch.zeros(
            numU * numV, 1, 
            args.angRes_in * args.patch_size_for_test * args.scale_factor,
            args.angRes_in * args.patch_size_for_test * args.scale_factor
        )

        # Apply TTA for each augmentation
        for aug_idx in range(num_augs):
            subLFout = torch.zeros(
                numU * numV, 1,
                args.angRes_in * args.patch_size_for_test * args.scale_factor,
                args.angRes_in * args.patch_size_for_test * args.scale_factor
            )

            for i in range(0, numU * numV, args.minibatch_for_test):
                tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
                
                # Apply augmentation
                tmp_aug = apply_augmentation(tmp.to(device), aug_idx)
                
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp_aug, data_info)
                
                # Reverse augmentation
                out = reverse_augmentation(out, aug_idx)
                subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out.cpu()

            subLFout_accum += subLFout

        # Average over all augmentations
        subLFout_avg = subLFout_accum / num_augs
        subLFout_avg = rearrange(subLFout_avg, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

        # Restore the Patches to LFs
        Sr_4D_y = LFintegrate(
            subLFout_avg, args.angRes_out, 
            args.patch_size_for_test * args.scale_factor,
            args.stride_for_test * args.scale_factor, 
            Hr_SAI_y.size(-2) // args.angRes_out, 
            Hr_SAI_y.size(-1) // args.angRes_out
        )
        Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

        # Save RGB
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0, 1) * 255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', 
                                  a1=args.angRes_out, a2=args.angRes_out)

            # Save all views
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(save_dir_) + '/' + 'View' + '_' + str(i) + '_' + str(j) + '.bmp'
                    imageio.imwrite(path, img)


if __name__ == '__main__':
    from option import args
    
    args.scale_factor = 4
    args.path_for_test = './data_for_inference/'
    
    # Update these for your model
    args.data_name = 'NTIRE_Val_Synth'
    args.model_name = 'MyEfficientLFNet'
    args.path_pre_pth = './log/SR_5x5_4x/ALL/MyEfficientLFNet/checkpoints/MyEfficientLFNet_5x5_4x_epoch_50_model.pth'
    
    main(args)
