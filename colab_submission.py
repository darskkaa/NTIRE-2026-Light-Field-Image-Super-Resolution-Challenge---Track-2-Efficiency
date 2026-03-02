import os
import sys
import glob
import subprocess
import shutil
import zipfile
import struct
import random
from collections import defaultdict, OrderedDict
from pathlib import Path
import numpy as np
import h5py
import imageio
import scipy.io as scio
# ==============================================================================
def install_dependencies():
    print("="*60)
    print("🚀 NTIRE 2026 LF-SR Track 2 - Colab Standalone Submission Script")
    print("="*60)
    
    print("\n=== STEP 1: Installing Dependencies ===")
    def run_cmd(cmd, check=True):
        print(f"\n[EXEC] {cmd}")
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            print(f"❌ Command failed with return code {result.returncode}: {cmd}")
            sys.exit(result.returncode)

    run_cmd('pip install --upgrade "torch==2.4.0" "torchvision==0.19.0" "torchaudio==2.4.0" --index-url https://download.pytorch.org/whl/cu121')
    
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel_suf = f"cu122torch2.4cxx11abiFALSE-{py_ver}-{py_ver}-linux_x86_64.whl"
    
    causal_url = f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+{wheel_suf}"
    mamba_url = f"https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+{wheel_suf}"
    
    run_cmd(f'pip install "{causal_url}"')
    run_cmd(f'pip install "{mamba_url}"')

    run_cmd('pip install fvcore')
    run_cmd('pip install "transformers<4.40.0"')
    run_cmd('pip install gdown h5py imageio scipy tqdm')

# ==============================================================================
# 1. GENERATE VALIDATION DATA (from Generate_Validation_Data.py)
# ==============================================================================
def generate_validation_data_inline(angRes=5, scale_factor=4, data_for='inference', src_data_path='./datasets/', save_data_path='./'):
    print("\n=== STEP 3: Generate Validation Patches (.h5) ===")
    import h5py
    import scipy.io as scio
    import numpy as np
    from math import ceil
    
    def deriveSizeFromScale(img_shape, scale):
        output_shape = []
        for k in range(2):
            output_shape.append(int(ceil(scale[k] * img_shape[k])))
        return output_shape
    
    def deriveScaleFromSize(img_shape_in, img_shape_out):
        scale = []
        for k in range(2):
            scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
        return scale
    
    def triangle(x):
        x = np.array(x).astype(np.float64)
        lessthanzero = np.logical_and((x>=-1),x<0)
        greaterthanzero = np.logical_and((x<=1),x>=0)
        f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
        return f
    
    def cubic(x):
        x = np.array(x).astype(np.float64)
        absx = np.absolute(x)
        absx2 = np.multiply(absx, absx)
        absx3 = np.multiply(absx2, absx)
        f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
        return f
    
    def contributions(in_length, out_length, scale, kernel, k_width):
        if scale < 1:
            h = lambda x: scale * kernel(scale * x)
            kernel_width = 1.0 * k_width / scale
        else:
            h = kernel
            kernel_width = k_width
        x = np.arange(1, out_length+1).astype(np.float64)
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
    
    def imresizemex(inimg, weights, indices, dim):
        in_shape = inimg.shape
        w_shape = weights.shape
        out_shape = list(in_shape)
        out_shape[dim] = w_shape[0]
        outimg = np.zeros(out_shape)
        if dim == 0:
            for i_img in range(in_shape[1]):
                for i_w in range(w_shape[0]):
                    w = weights[i_w, :]
                    ind = indices[i_w, :]
                    im_slice = inimg[ind, i_img].astype(np.float64)
                    outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
        elif dim == 1:
            for i_img in range(in_shape[0]):
                for i_w in range(w_shape[0]):
                    w = weights[i_w, :]
                    ind = indices[i_w, :]
                    im_slice = inimg[i_img, ind].astype(np.float64)
                    outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
        if inimg.dtype == np.uint8:
            outimg = np.clip(outimg, 0, 255)
            return np.around(outimg).astype(np.uint8)
        else:
            return outimg
    
    def imresizevec(inimg, weights, indices, dim):
        wshape = weights.shape
        if dim == 0:
            weights = weights.reshape((wshape[0], wshape[2], 1, 1))
            outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
        elif dim == 1:
            weights = weights.reshape((1, wshape[0], wshape[2], 1))
            outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
        if inimg.dtype == np.uint8:
            outimg = np.clip(outimg, 0, 255)
            return np.around(outimg).astype(np.uint8)
        else:
            return outimg
    
    def resizeAlongDim(A, dim, weights, indices, mode="vec"):
        if mode == "org":
            out = imresizemex(A, weights, indices, dim)
        else:
            out = imresizevec(A, weights, indices, dim)
        return out
    
    def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
        if method == 'bicubic':
            kernel = cubic
        elif method == 'bilinear':
            kernel = triangle
        else:
            print ('Error: Unidentified method supplied')
            
        kernel_width = 4.0
        if scalar_scale is not None:
            scalar_scale = float(scalar_scale)
            scale = [scalar_scale, scalar_scale]
            output_size = deriveSizeFromScale(I.shape, scale)
        elif output_shape is not None:
            scale = deriveScaleFromSize(I.shape, output_shape)
            output_size = list(output_shape)
        else:
            print ('Error: scalar_scale OR output_shape should be defined!')
            return
        scale_np = np.array(scale)
        order = np.argsort(scale_np)
        weights = []
        indices = []
        for k in range(2):
            w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
            weights.append(w)
            indices.append(ind)
        B = np.copy(I) 
        flag2D = False
        if B.ndim == 2:
            B = np.expand_dims(B, axis=2)
            flag2D = True
        for k in range(2):
            dim = order[k]
            B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
        if flag2D:
            B = np.squeeze(B, axis=2)
        return B
    
    def rgb2ycbcr(x):
        y = np.zeros(x.shape, dtype='double')
        y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
        y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
        y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0
    
        y = y / 255.0
        return y
    
    downRatio = 1 / scale_factor

    ''' dir '''
    save_dir = Path(save_data_path + 'data_for_' + data_for)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_dir = save_dir.joinpath('SR_' + str(angRes) + 'x' + str(angRes) + '_' + str(scale_factor) + 'x')
    save_dir.mkdir(exist_ok=True, parents=True)

    src_datasets = os.listdir(src_data_path)
    src_datasets.sort()
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['NTIRE_Val_Real', 'NTIRE_Val_Synth']:
            continue
        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)

        src_sub_dataset = src_data_path + name_dataset + '/' + data_for + '/'
        if not os.path.exists(src_sub_dataset):
            print(f"Skipping {name_dataset}: {src_sub_dataset} not found")
            continue

        for root, dirs, files in os.walk(src_sub_dataset):
            for file in files:
                idx_scene_save = 0
                print('Generating test data of Scene_%s in Dataset %s......\t' %(file, name_dataset))
                try:
                    data = h5py.File(os.path.join(root, file), 'r')
                    LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                except:
                    data = scio.loadmat(os.path.join(root, file))
                    LF = np.array(data['LF'])

                (U, V, H, W, _) = LF.shape

                # Extract central angRes * angRes views
                LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, 0:H, 0:W, 0:3]
                LF = LF.astype('double')
                (U, V, H, W, _) = LF.shape

                idx_save = idx_save + 1
                idx_scene_save = idx_scene_save + 1
                Sr_SAI_cbcr = np.zeros((U * H * scale_factor, V * W * scale_factor, 2), dtype='single')
                Lr_SAI_y = np.zeros((U * H, V * W), dtype='single')
                Hr_SAI_y = np.zeros((U * H * scale_factor, V * W * scale_factor), dtype='single')

                for u in range(U):
                    for v in range(V):
                        tmp_Lr_rgb = LF[u, v, :, :, :]
                        tmp_Lr_ycbcr = rgb2ycbcr(tmp_Lr_rgb)
                        Lr_SAI_y[u * H: (u+1) * H, v * W: (v+1)* W] = tmp_Lr_ycbcr[:, :, 0]

                        tmp_Lr_cbcr = tmp_Lr_ycbcr[:,:,1:3]
                        tmp_Sr_cbcr = imresize(tmp_Lr_cbcr, scalar_scale=scale_factor)
                        Sr_SAI_cbcr[u * H * scale_factor: (u+1) * H * scale_factor,
                        v * W * scale_factor: (v+1) * W * scale_factor,:] = tmp_Sr_cbcr

                        tmp_Lr_y = tmp_Lr_ycbcr[:, :, 0]
                        Hr_SAI_y[u * H * scale_factor: (u + 1) * H * scale_factor,
                        v * W * scale_factor: (v + 1) * W * scale_factor] = imresize(tmp_Lr_y, scalar_scale=scale_factor)

                file_name = [str(sub_save_dir) + '/' + '%s' % file.split('.')[0] + '.h5']
                with h5py.File(file_name[0], 'w') as hf:
                    hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y.transpose((1, 0)), dtype='single')
                    hf.create_dataset('Sr_SAI_cbcr', data=Sr_SAI_cbcr.transpose((2, 1, 0)), dtype='single')
                    hf.create_dataset('Hr_SAI_y', data=Hr_SAI_y.transpose((1, 0)), dtype='single')

                print('%d test samples have been generated' % (idx_scene_save))

# ==============================================================================
# 2. RUN INFERENCE (from inference.py)
# ==============================================================================
def run_inference_inline(model_name="MyEfficientLFNetV10", angRes=5, scale_factor=4, path_pre_pth="", path_for_test="./data_for_inference/", data_name="ALL", device="cuda:0"):
    print("\n=== STEP 4: Run Inference ===")
    import importlib
    import torch
    import torch.backends.cudnn as cudnn
    import imageio
    from tqdm import tqdm
    from einops import rearrange
    def create_dir(args):
        log_dir = Path(args.path_log)
        log_dir.mkdir(exist_ok=True)
        if args.task == 'SR':
            task_path = 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x'
        
        log_dir = log_dir.joinpath(task_path)
        log_dir.mkdir(exist_ok=True)
        log_dir = log_dir.joinpath(args.data_name)
        log_dir.mkdir(exist_ok=True)
        log_dir = log_dir.joinpath(args.model_name)
        log_dir.mkdir(exist_ok=True)
    
        checkpoints_dir = log_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
    
        results_dir = log_dir.joinpath('results/')
        results_dir.mkdir(exist_ok=True)
    
        return log_dir, checkpoints_dir, results_dir

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

    import torch.nn.functional as F

    def LFdivide(data, angRes, patch_size, stride):
        data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
        [_, _, h0, w0] = data.size()
    
        bdr = (patch_size - stride) // 2
        numU = (h0 + bdr * 2 - 1) // stride
        numV = (w0 + bdr * 2 - 1) // stride
        data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
        subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
        subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                          a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)
    
        return subLF

    def LFintegrate(subLF, angRes, pz, stride, h, w):
        if subLF.dim() == 4:
            subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
            pass
        bdr = (pz - stride) // 2
        outLF = subLF[:, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
        outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
        outLF = outLF[:, :, 0:h, 0:w]
    
        return outLF

    def ycbcr2rgb(x):
        import numpy as np
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

    import torch.utils.data as data
    import h5py

    class TestSetDataLoader(data.Dataset):
        def __init__(self, args, dataset_dir, folder_folder):
            super(TestSetDataLoader, self).__init__()
            self.angRes_in = args.angRes_in
            self.angRes_out = args.angRes_out
            self.dataset_dir = dataset_dir
            self.data_list = os.listdir(os.path.join(dataset_dir, folder_folder))
            self.data_list.sort()
            self.file_list = []
            for data_name in self.data_list:
                self.file_list.append(os.path.join(dataset_dir, folder_folder, data_name))
    
        def __getitem__(self, index):
            file_name = [self.file_list[index]]
            with h5py.File(file_name[0], 'r') as hf:
                Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
                Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
                Sr_SAI_cbcr = np.array(hf.get('Sr_SAI_cbcr'), dtype='single')
                Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
                Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
                Sr_SAI_cbcr = np.transpose(Sr_SAI_cbcr, (2, 1, 0))
            Lr_SAI_y = torch.from_numpy(Lr_SAI_y).unsqueeze(0)
            Hr_SAI_y = torch.from_numpy(Hr_SAI_y).unsqueeze(0)
            Sr_SAI_cbcr = torch.from_numpy(Sr_SAI_cbcr).permute(2, 0, 1)
            LF_name = self.data_list[index].split('.')[0]
            return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [self.angRes_in, self.angRes_out], LF_name
    
        def __len__(self):
            return len(self.data_list)
            
    def MultiTestSetDataLoader(args):
        dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x/'
        data_list = os.listdir(dataset_dir)
        data_list.sort()
        
        test_Loaders = []
        length_of_tests = 0
        test_Names = []
        for data_name in data_list:
            if args.data_name == 'ALL' or args.data_name == data_name:
                test_Names.append(data_name)
                test_Dataset = TestSetDataLoader(args, dataset_dir, data_name)
                length_of_tests += len(test_Dataset)
                test_Loaders.append(torch.utils.data.DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))
        return test_Names, test_Loaders, length_of_tests

    import argparse
    args = argparse.Namespace()
    args.task = 'SR'
    args.model_name = model_name
    args.angRes_in = angRes
    args.angRes_out = angRes
    args.scale_factor = scale_factor
    args.use_pre_ckpt = True
    args.path_pre_pth = path_pre_pth
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 1
    args.path_for_train = './data_for_training/'
    args.path_for_test = path_for_test
    args.data_name = data_name
    args.path_log = './log/'
    args.batch_size = 4
    args.lr = 2e-4
    args.epoch = 150
    args.device = device
    args.num_workers = 2
    args.local_rank = 0

    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True, parents=True)

    ''' CPU or Cuda'''
    used_device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    if 'cuda' in args.device and torch.cuda.is_available():
        try:
            device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
            torch.cuda.set_device(device_id)
        except AttributeError:
            pass # Ignore if torch._C has no attribute _cuda_setDevice (Colab CPU/driver issue)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)

    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    import importlib
    import torch

    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' Load Pre-Trained PTH '''
    print(f"Loading checkpoint from: {args.path_pre_pth}")
    ckpt_path = args.path_pre_pth
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    try:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
    except (RuntimeError, KeyError):
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k] = v
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')

    net = net.to(used_device)
    cudnn.benchmark = True

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]
            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
                [Lr_angRes_in, Lr_angRes_out] = data_info
                data_info[0] = Lr_angRes_in[0].item()
                data_info[1] = Lr_angRes_out[0].item()

                Lr_SAI_y = Lr_SAI_y.squeeze().to(used_device)

                ''' Crop LFs into Patches '''
                subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
                numU, numV, H, W = subLFin.size()
                subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
                subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
                                       args.angRes_in * args.patch_size_for_test * args.scale_factor, device=used_device)

                ''' SR the Patches '''
                net.eval()
                torch.cuda.empty_cache()
                for i in range(0, numU * numV, args.minibatch_for_test):
                    tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
                    with torch.no_grad():
                        out = net(tmp.to(used_device), data_info)
                        subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
                subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

                ''' Restore the Patches to LFs '''
                Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
                                      args.stride_for_test * args.scale_factor, Hr_SAI_y.size(-2)//args.angRes_out, Hr_SAI_y.size(-1)//args.angRes_out)
                Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

                ''' Save RGB '''
                if save_dir is not None:
                    save_dir_ = save_dir.joinpath(LF_name[0])
                    save_dir_.mkdir(exist_ok=True)
                    Sr_SAI_ycbcr = torch.cat((Sr_SAI_y.cpu(), Sr_SAI_cbcr), dim=1)
                    Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
                    Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

                    for i in range(args.angRes_out):
                        for j in range(args.angRes_out):
                            img = Sr_4D_rgb[i, j, :, :, :]
                            path = str(save_dir_) + '/' + 'View' + '_' + str(i) + '_' + str(j) + '.bmp'
                            imageio.imwrite(path, img)

# ==============================================================================
# 3. FORMAT SUBMISSION (from format_submission.py)
# ==============================================================================
def format_submission_inline(input_dir, output_zip="submission.zip"):
    print("\n=== STEP 5: Format Submission Zip ===")
    MAPPING = {
        'Real': ['EPFL', 'INRIA_Lytro', 'Stanford_Gantry', 'NTIRE_Val_Real', 'NTIRE_Test_Real'],
        'Synth': ['HCI_new', 'HCI_old', 'NTIRE_Val_Synth', 'NTIRE_Test_Synth']
    }

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Error: Input directory '{input_dir}' not found.")
        return

    # Create temporary structure
    temp_dir = Path("./temp_submission_format")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    real_dir = temp_dir / 'Real'
    synth_dir = temp_dir / 'Synth'
    real_dir.mkdir(parents=True)
    synth_dir.mkdir(parents=True)

    print(f"📦 Formatting submission from: {input_dir}")
    
    real_count = 0
    synth_count = 0

    for dataset_folder in input_path.iterdir():
        if not dataset_folder.is_dir():
            continue
            
        dataset_name = dataset_folder.name
        
        target_category = None
        if dataset_name in MAPPING['Real']:
            target_category = 'Real'
        elif dataset_name in MAPPING['Synth']:
            target_category = 'Synth'
        else:
            print(f"⚠️ Warning: Dataset '{dataset_name}' not recognized as Real or Synth. Skipping.")
            continue

        target_base_dir = real_dir if target_category == 'Real' else synth_dir

        for scene_folder in dataset_folder.iterdir():
            if not scene_folder.is_dir():
                continue
                
            scene_name = scene_folder.name
            target_scene_dir = target_base_dir / scene_name
            
            shutil.copytree(scene_folder, target_scene_dir)
            
            if target_category == 'Real':
                real_count += 1
            else:
                synth_count += 1
                
            print(f"   ✓ Copied {target_category} scene: {scene_name}")

    print(f"\n📊 Summary: {real_count} Real scenes, {synth_count} Synth scenes.")
    
    # Zip the contents
    print(f"\n🗜️ Zipping to {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    shutil.rmtree(temp_dir)
    print(f"✅ Submission successfully created: {output_zip}")

# ==============================================================================
# 4. VALIDATE SUBMISSION (from validate_submission.py)
# ==============================================================================
EXPECTED_REAL_SCENES = 16
EXPECTED_SYNTH_SCENES = 16
EXPECTED_VIEWS_PER_SCENE = 25  # 5x5 angular resolution
EXPECTED_VIEW_NAMES = [f"View_{i}_{j}.bmp" for i in range(5) for j in range(5)]
EXPECTED_REAL_DIMS = (624, 432)   # Width x Height
EXPECTED_SYNTH_DIMS = (500, 500)  # Width x Height
BMP_HEADER_SIZE = 14
BMP_INFO_HEADER_SIZE = 40  # BITMAPINFOHEADER
MIN_PIXEL_MEAN = 20.0    # Images shouldn't be too dark
MAX_PIXEL_MEAN = 235.0   # Images shouldn't be saturated
MIN_PIXEL_STD = 5.0      # Images should have some variance

class ValidationResult:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.stats = {}
        
    def error(self, msg): self.errors.append(f"❌ {msg}")
    def warning(self, msg): self.warnings.append(f"⚠️  {msg}")
    def info_msg(self, msg): self.info.append(f"ℹ️  {msg}")
    def passed(self): return len(self.errors) == 0

class FileProvider:
    def __init__(self, path):
        self.path = path
        self.is_zip = os.path.isfile(path) and path.endswith('.zip')
        self.zf = None
        if self.is_zip:
            self.zf = zipfile.ZipFile(path, 'r')
            self._files = self.zf.namelist()
        else:
            self._files = []
            for root, dirs, files in os.walk(path):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), path)
                    self._files.append(rel.replace('\\', '/'))
    def namelist(self): return self._files
    def read(self, filename):
        if self.is_zip: return self.zf.read(filename)
        else:
            with open(os.path.join(self.path, filename), 'rb') as f: return f.read()
    def close(self):
        if self.zf: self.zf.close()

def parse_bmp_header(data):
    if len(data) < 54: return None
    info = {}
    info['magic'] = data[0:2]
    info['file_size'] = struct.unpack('<I', data[2:6])[0]
    info['data_offset'] = struct.unpack('<I', data[10:14])[0]
    info['width'] = struct.unpack('<i', data[18:22])[0]
    info['height'] = struct.unpack('<i', data[22:26])[0]
    info['bits_per_pixel'] = struct.unpack('<H', data[28:30])[0]
    info['compression'] = struct.unpack('<I', data[30:34])[0]
    return info

def analyze_pixel_content(data, bmp_info):
    if bmp_info is None: return None
    offset = bmp_info['data_offset']
    pixel_data = data[offset:]
    if len(pixel_data) == 0: return None
    pixels = list(pixel_data)
    stats = {
        'mean': sum(pixels) / len(pixels),
        'min': min(pixels), 'max': max(pixels), 'std': 0.0,
    }
    mean = stats['mean']
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    stats['std'] = variance ** 0.5
    return stats

def validate_structure(provider, result):
    files = provider.namelist()
    has_real = any(f.startswith('Real/') for f in files)
    has_synth = any(f.startswith('Synth/') for f in files)
    if not has_real: result.error("Missing 'Real/' folder")
    if not has_synth: result.error("Missing 'Synth/' folder")
    return has_real, has_synth

def validate_scenes(provider, result):
    files = provider.namelist()
    real_scenes, synth_scenes = set(), set()
    for f in files:
        parts = f.split('/')
        if len(parts) >= 2:
            folder, scene = parts[0], parts[1]
            if folder == 'Real' and scene: real_scenes.add(scene)
            elif folder == 'Synth' and scene: synth_scenes.add(scene)
    
    if len(real_scenes) != EXPECTED_REAL_SCENES: result.error(f"Expected {EXPECTED_REAL_SCENES} Real scenes, found {len(real_scenes)}")
    if len(synth_scenes) != EXPECTED_SYNTH_SCENES: result.error(f"Expected {EXPECTED_SYNTH_SCENES} Synth scenes, found {len(synth_scenes)}")
    return real_scenes, synth_scenes

def validate_views(provider, result, real_scenes, synth_scenes):
    files = provider.namelist()
    expected_set = set(EXPECTED_VIEW_NAMES)
    all_scenes = [('Real', s) for s in real_scenes] + [('Synth', s) for s in synth_scenes]
    missing_views = []
    
    for folder, scene in all_scenes:
        prefix = f"{folder}/{scene}/"
        scene_files = [f.split('/')[-1] for f in files if f.startswith(prefix) and f.endswith('.bmp')]
        missing = expected_set - set(scene_files)
        if missing: missing_views.append((f"{folder}/{scene}", list(missing)))
    
    if missing_views:
        for scene, views in missing_views[:5]: result.error(f"{scene}/ missing: {views}...")

def validate_bmp_files(provider, result, real_scenes, synth_scenes):
    files = provider.namelist()
    bmp_files = [f for f in files if f.endswith('.bmp') and (f.startswith('Real/') or f.startswith('Synth/'))]
    
    for f in bmp_files:
        try:
            data = provider.read(f)
            bmp_info = parse_bmp_header(data)
            if bmp_info is None or bmp_info['magic'] != b'BM':
                result.error(f"Invalid BMP magic header: {f}")
                continue
            if bmp_info['bits_per_pixel'] != 24: result.error(f"Wrong color depth ({bmp_info['bits_per_pixel']} bpp): {f}")
            if bmp_info['compression'] != 0: result.error(f"Compressed BMP: {f}")
            
            w, h = bmp_info['width'], abs(bmp_info['height'])
            expected_dims = EXPECTED_REAL_DIMS if f.startswith('Real/') else EXPECTED_SYNTH_DIMS
            if (w, h) != expected_dims: result.warning(f"{f}: {w}x{h} (expected {expected_dims})")
        except Exception as e:
            result.error(f"Failed to read {f}: {e}")

def validate_pixel_content(provider, result, sample_size=50):
    files = provider.namelist()
    bmp_files = [f for f in files if f.endswith('.bmp') and (f.startswith('Real/') or f.startswith('Synth/'))]
    sample = random.sample(bmp_files, min(len(bmp_files), sample_size))
    
    for f in sample:
        try:
            data = provider.read(f)
            bmp_info = parse_bmp_header(data)
            if bmp_info is None: continue
            stats = analyze_pixel_content(data, bmp_info)
            if stats is None: continue
            
            if stats['mean'] < MIN_PIXEL_MEAN: result.warning(f"Dark image (mean={stats['mean']:.1f}): {f}")
            if stats['mean'] > MAX_PIXEL_MEAN: result.warning(f"Saturated image (mean={stats['mean']:.1f}): {f}")
            if stats['std'] < MIN_PIXEL_STD: result.warning(f"Low variance (std={stats['std']:.1f}): {f}")
        except Exception:
            pass

def print_summary(result):
    print("\n" + "="*60)
    print("📋 VALIDATION SUMMARY")
    print("="*60)
    if result.warnings:
        print(f"\n   ⚠️  WARNINGS ({len(result.warnings)}):")
        for w in result.warnings[:10]: print(f"      {w}")
    if result.errors:
        print(f"\n   ❌ ERRORS ({len(result.errors)}):")
        for e in result.errors[:10]: print(f"      {e}")
        print("\n❌ VALIDATION FAILED - DO NOT SUBMIT")
        return False
    else:
        print("\n✅ VALIDATION PASSED - READY TO SUBMIT!")
        return True

def validate_submission_inline(path):
    print("\n=== STEP 6: Validate Submission Zip ===")
    print("\n" + "="*60)
    print("🔍 ULTRA-RIGOROUS SUBMISSION VALIDATOR")
    print("="*60)
    
    result = ValidationResult()
    if not os.path.exists(path):
        print(f"\n❌ ERROR: Path not found: {path}")
        return False
    
    print(f"\n   Validating: {path}")
    provider = FileProvider(path)
    
    has_real, has_synth = validate_structure(provider, result)
    if has_real or has_synth:
        real_scenes, synth_scenes = validate_scenes(provider, result)
        validate_views(provider, result, real_scenes, synth_scenes)
        validate_bmp_files(provider, result, real_scenes, synth_scenes)
        validate_pixel_content(provider, result)
        
    provider.close()
    return print_summary(result)

# ==============================================================================
# 5. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    install_dependencies()

    print("\n=== STEP 2: Downloading NTIRE Validation Data ===")
    os.makedirs('downloads', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    
    def run_cmd(cmd, check=True):
        print(f"\n[EXEC] {cmd}")
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            print(f"❌ Command failed with return code {result.returncode}: {cmd}")
            sys.exit(result.returncode)

    print("Checking datasets...")
    real_count = len(glob.glob('datasets/NTIRE_Val_Real/inference/*.mat'))
    synth_count = len(glob.glob('datasets/NTIRE_Val_Synth/inference/*.mat'))

    if real_count >= 16 and synth_count >= 16:
        print(f"\n[OK] Validation data already present ({real_count} Real, {synth_count} Synth).")
    else:
        print("\n[INFO] Downloading validation zips from Google Drive...")
        run_cmd('gdown --folder "https://drive.google.com/drive/folders/1LfPTTTtTDOPyNg3D-B_RfzwBZd4D0-HH" -O downloads/', check=False)
        
        real_zip = glob.glob('downloads/**/NTIRE_Val_Real.zip', recursive=True)
        synth_zip = glob.glob('downloads/**/NTIRE_Val_Synth.zip', recursive=True)
        
        if real_zip:
            run_cmd(f'unzip -o -q "{real_zip[0]}" -d datasets/') 
        if synth_zip:
            run_cmd(f'unzip -o -q "{synth_zip[0]}" -d datasets/')
            
        real_count = len(glob.glob('datasets/NTIRE_Val_Real/inference/*.mat'))
        synth_count = len(glob.glob('datasets/NTIRE_Val_Synth/inference/*.mat'))
        print(f"[INFO] Found: {real_count} Real .mat, {synth_count} Synth .mat")

        if real_count == 0 or synth_count == 0:
            print("❌ [ERROR] Missing validation data! Check downloads/ and manually extract.")
            sys.exit(1)

    # 1. Generate Validations
    shutil.rmtree('data_for_inference/SR_5x5_4x/NTIRE_Val_Real', ignore_errors=True)
    shutil.rmtree('data_for_inference/SR_5x5_4x/NTIRE_Val_Synth', ignore_errors=True)
    generate_validation_data_inline(angRes=5, scale_factor=4, data_for='inference', src_data_path='./datasets/', save_data_path='./')

    # 2. Run Inference
    model_name = "MyEfficientLFNetV10"
    
    # Check for the specific exact file first:
    specific_pth = 'MyEfficientLFNetV10_5x5_4x_epoch_67_model.pth'
    if os.path.exists(specific_pth):
        best_ckpt = specific_pth
        print(f"✅ Found explicitly requested checkpoint: {best_ckpt}")
    else:
        # Fallback to standard globbing if not found
        pth_files = glob.glob('*.pth') + glob.glob('pth/*.pth') + glob.glob(f'log/SR_5x5_4x/ALL/{model_name}/checkpoints/*.pth')
        
        if not pth_files:
            print(f"❌ No .pth checkpoint found! Ensure '{specific_pth}' is in the current colab directory.")
            sys.exit(1)
            
        pth_files.sort(key=os.path.getmtime, reverse=True)
        best_ckpt = pth_files[0]
        print(f"⚠️ Explicit file '{specific_pth}' not found. Falling back to: {best_ckpt}")
    
    shutil.rmtree(f'log/SR_5x5_4x/ALL/{model_name}/results/TEST', ignore_errors=True)
    
    run_inference_inline(
        model_name=model_name,
        angRes=5,
        scale_factor=4,
        path_pre_pth=best_ckpt,
        path_for_test="./data_for_inference/",
        data_name="ALL",
        device="cuda:0"
    )

    # 3. Format Submission
    results_dir = f"log/SR_5x5_4x/ALL/{model_name}/results/TEST"
    zip_name = f"submission_val_{model_name}.zip"
    
    if not os.path.exists(results_dir):
        print(f"❌ Results dir not found: {results_dir}")
        sys.exit(1)
        
    format_submission_inline(results_dir, output_zip=zip_name)
    
    # 4. Validate Submission
    success = validate_submission_inline(zip_name)
    if not success:
        print("\n❌ Validation Failed.")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"✅ ALL DONE! Your submission file is ready: {zip_name}")
    print("Download this file from the Colab file browser and upload to CodaBench.")
    print("="*60)

if __name__ == "__main__":
    main()
