import importlib
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from collections import OrderedDict
import imageio
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis


from utils.utils_datasets import MultiTestSetDataLoader, TestSetDataLoader



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

    ''' FLOPs Check '''
    try:
        dummy_input = torch.randn(1, 1, 160, 160).to(args.device)
        net_check = net.to(args.device)
        flops = FlopCountAnalysis(net_check, dummy_input)
        print(f"\n[Efficiency Check] FLOPs: {flops.total()/1e9:.2f} G (Limit: 20G)")
    except Exception as e:
        print(f"\n[Efficiency Check] Failed to calculate FLOPs: {e}")
    net = net.cpu() # Reset to CPU for state dict loading logic below


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
    else:
        ckpt_path = args.path_pre_pth
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.' + k  # add `module.`
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
            pass
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():

        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            test(test_loader, device, net, args, save_dir)
    pass

def test(test_loader, device, net, args, save_dir=None):
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
                               args.angRes_in * args.patch_size_for_test * args.scale_factor, device=device)

        ''' SR the Patches '''
        net.eval()  # Set eval mode once before the loop
        torch.cuda.empty_cache()  # Once before loop, not per-iteration
        for i in range(0, numU * numV, args.minibatch_for_test):
            tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
            with torch.no_grad():
                out = net(tmp.to(device), data_info)
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
            Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

            # save all views
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(save_dir_) + '/' + 'View' + '_' + str(i) + '_' + str(j) + '.bmp'
                    imageio.imwrite(path, img)
                    pass
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    from option import args
    # Use CLI arguments (see option.py)
    # Example: python inference.py --model_name MyEfficientLFNet --scale_factor 4 --path_pre_pth ./pth/model.pth --path_for_test ./data_for_inference/ --data_name NTIRE_Val_Real
    main(args)