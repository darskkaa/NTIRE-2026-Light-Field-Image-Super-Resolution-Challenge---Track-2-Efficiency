import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='SR', help='SR, RE')

# LF_SR
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")

parser.add_argument('--model_name', type=str, default='LFT', help="model name")
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

args = parser.parse_args()



if args.task == 'SR':
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 1

del args.angRes