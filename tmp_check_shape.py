import sys
import scipy.io as scio
import numpy as np

f = sys.argv[1]
try:
    d = scio.loadmat(f)
    print("Keys:", list(d.keys()))
    print("Shape:", np.array(d['LF']).shape)
except Exception as e:
    print("Failed! File might be v7.3 HDF5, which requires h5py.")
    print(e)
