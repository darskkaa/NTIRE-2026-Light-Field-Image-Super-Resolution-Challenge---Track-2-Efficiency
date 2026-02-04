
import os
import h5py
import scipy.io as scio
import numpy as np
from pathlib import Path
from utils.utils import rgb2ycbcr
from utils.imresize import imresize

def convert(src_path, dst_path, scale_factor=4, angRes=5):
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    
    files = sorted([f for f in os.listdir(src_path) if f.endswith('.mat')])
    print(f"Converting {len(files)} files from {src_path}...")
    
    for file in files:
        # Load MAT
        try:
            data = h5py.File(os.path.join(src_path, file), 'r')
            LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
        except:
            data = scio.loadmat(os.path.join(src_path, file))
            LF = np.array(data['LF'])
            
        # Extract params
        (U, V, H, W, _) = LF.shape
        LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, 0:H, 0:W, 0:3]
        LF = LF.astype('double')
        (U, V, H, W, _) = LF.shape
        
        # Prepare placeholders
        Sr_SAI_cbcr = np.zeros((U * H * scale_factor, V * W * scale_factor, 2), dtype='single')
        Lr_SAI_y = np.zeros((U * H, V * W), dtype='single')
        Hr_SAI_y = np.zeros((U * H * scale_factor, V * W * scale_factor), dtype='single') 

        for u in range(U):
            for v in range(V):
                tmp_Lr_rgb = LF[u, v, :, :, :]
                tmp_Lr_ycbcr = rgb2ycbcr(tmp_Lr_rgb)
                Lr_SAI_y[u*H:(u+1)*H, v*W:(v+1)*W] = tmp_Lr_ycbcr[:, :, 0]
                
                # Upscale for placeholders
                tmp_Lr_cbcr = tmp_Lr_ycbcr[:,:,1:3]
                tmp_Sr_cbcr = imresize(tmp_Lr_cbcr, scalar_scale=scale_factor)
                Sr_SAI_cbcr[u*H*scale_factor:(u+1)*H*scale_factor, v*W*scale_factor:(v+1)*W*scale_factor,:] = tmp_Sr_cbcr
                
                tmp_Lr_y = tmp_Lr_ycbcr[:, :, 0]
                Hr_SAI_y[u*H*scale_factor:(u+1)*H*scale_factor, v*W*scale_factor:(v+1)*W*scale_factor] = imresize(tmp_Lr_y, scalar_scale=scale_factor)

        # Save H5
        save_path = os.path.join(dst_path, file.split('.')[0] + '.h5')
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y.transpose((1, 0)), dtype='single')
            hf.create_dataset('Sr_SAI_cbcr', data=Sr_SAI_cbcr.transpose((2, 1, 0)), dtype='single')
            hf.create_dataset('Hr_SAI_y', data=Hr_SAI_y.transpose((1, 0)), dtype='single')
        
        print(f"Saved: {save_path}")

# Run conversion
convert("data_for_inference/SR_5x5_4x/NTIRE_Val_Real", "data_converted/SR_5x5_4x/NTIRE_Val_Real")
convert("data_for_inference/SR_5x5_4x/NTIRE_Val_Synth", "data_converted/SR_5x5_4x/NTIRE_Val_Synth")
