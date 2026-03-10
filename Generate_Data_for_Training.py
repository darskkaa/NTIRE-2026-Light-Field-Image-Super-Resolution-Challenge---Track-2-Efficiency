import argparse
import os
import h5py
from utils.imresize import *
from pathlib import Path
import scipy.io as scio
import sys
import random
from utils.utils import rgb2ycbcr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")
    parser.add_argument('--data_for', type=str, default='training', help='training or validation')
    parser.add_argument('--src_data_path', type=str, default='./datasets/', help='path to source datasets')
    parser.add_argument('--save_data_path', type=str, default='./', help='path to save converted data')
    parser.add_argument('--n_angular_crops', type=int, default=5,
                        help='Number of random angular crops per scene from 9x9 grid. '
                             'Set to 0 to use only center crop (original behavior). '
                             'For 9x9->5x5, there are 25 possible crops; we sample this many.')

    return parser.parse_args()


def generate_patches_for_angular_crop(LF_full, u_start, v_start, angRes, patchsize, stride,
                                       scale_factor, downRatio, sub_save_dir, idx_save):
    """Generate all spatial patches for one angular crop of the LF."""
    LF = LF_full[u_start:u_start+angRes, v_start:v_start+angRes, :, :, 0:3]
    LF = LF.astype('double')
    (U, V, H, W, _) = LF.shape
    count = 0

    for h in range(0, H - patchsize + 1, stride):
        for w in range(0, W - patchsize + 1, stride):
            idx_save += 1
            count += 1
            Hr_SAI_y = np.zeros((U * patchsize, V * patchsize), dtype='single')
            Lr_SAI_y = np.zeros((U * patchsize // scale_factor,
                                 V * patchsize // scale_factor), dtype='single')

            for u in range(U):
                for v in range(V):
                    tmp_Hr_rgb = LF[u, v, h: h + patchsize, w: w + patchsize, :]
                    tmp_Hr_ycbcr = rgb2ycbcr(tmp_Hr_rgb)
                    tmp_Hr_y = tmp_Hr_ycbcr[:, :, 0]

                    patchsize_Lr = patchsize // scale_factor
                    Hr_SAI_y[u * patchsize: (u+1) * patchsize,
                             v * patchsize: (v+1) * patchsize] = tmp_Hr_y
                    tmp_Sr_y = imresize(tmp_Hr_y, scalar_scale=downRatio)

                    Lr_SAI_y[u*patchsize_Lr: (u+1)*patchsize_Lr,
                             v*patchsize_Lr: (v+1)*patchsize_Lr] = tmp_Sr_y

            # save
            file_name = str(sub_save_dir) + '/' + '%06d' % idx_save + '.h5'
            with h5py.File(file_name, 'w') as hf:
                hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y.transpose((1, 0)), dtype='single')
                hf.create_dataset('Hr_SAI_y', data=Hr_SAI_y.transpose((1, 0)), dtype='single')

    return idx_save, count


def main(args):
    angRes, scale_factor = args.angRes, args.scale_factor
    patchsize = scale_factor * 32
    stride = patchsize // 2
    downRatio = 1 / scale_factor
    n_angular_crops = args.n_angular_crops

    ''' dir '''
    save_dir = Path(args.save_data_path + 'data_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath('SR_' + str(angRes) + 'x' + str(angRes) + '_' + str(scale_factor) + 'x')
    save_dir.mkdir(exist_ok=True)

    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']:
            continue
        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)

        src_sub_dataset = args.src_data_path + name_dataset + '/' + args.data_for + '/'
        for root, dirs, files in os.walk(src_sub_dataset):
            for file in files:
                idx_scene_save = 0
                print('Generating training data of Scene_%s in Dataset %s......\t' % (file, name_dataset))
                try:
                    data = h5py.File(root + file, 'r')
                    LF_full = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                except:
                    data = scio.loadmat(root + file)
                    LF_full = np.array(data['LF'])

                (U_full, V_full, _, _, _) = LF_full.shape

                # Determine angular crops
                u_range = U_full - angRes + 1  # e.g., 9-5+1=5 for 9x9->5x5
                v_range = V_full - angRes + 1

                if n_angular_crops > 0 and u_range > 1 and v_range > 1:
                    # Generate multiple random angular crops from the full grid
                    # Always include center crop first (most important)
                    u_center = (U_full - angRes) // 2
                    v_center = (V_full - angRes) // 2
                    angular_crops = [(u_center, v_center)]

                    # Generate all possible crops and sample from them
                    all_crops = [(u, v) for u in range(u_range) for v in range(v_range)
                                 if (u, v) != (u_center, v_center)]
                    random.shuffle(all_crops)
                    n_extra = min(n_angular_crops - 1, len(all_crops))
                    angular_crops.extend(all_crops[:n_extra])

                    print(f'  → {len(angular_crops)} angular crops from {U_full}x{V_full} grid '
                          f'({u_range*v_range} possible)')
                else:
                    # Original behavior: center crop only
                    u_center = (U_full - angRes) // 2
                    v_center = (V_full - angRes) // 2
                    angular_crops = [(u_center, v_center)]

                for crop_idx, (u_start, v_start) in enumerate(angular_crops):
                    idx_save, count = generate_patches_for_angular_crop(
                        LF_full, u_start, v_start, angRes, patchsize, stride,
                        scale_factor, downRatio, sub_save_dir, idx_save
                    )
                    idx_scene_save += count

                print('%d training samples have been generated\n' % (idx_scene_save))

    print(f'\nTotal: {idx_save} training samples generated across all datasets')


if __name__ == '__main__':
    args = parse_args()
    main(args)