import os
import h5py
import numpy as np
from pandas import DataFrame

h5_file = os.listdir("H5_PATH")

img_id_list = []
patch_num_list = []


for i in range(len(h5_file)):
    now_sample = "H5_PATH" + h5_file[i]
    f = h5py.File(now_sample)
    patch_np = np.array(f['features'][:])
    patch_cor = np.array(f['coords'][:])

    img_id_list.append(h5_file[i])
    patch_num_list.append(patch_np.shape[0])

data = { 'img_id': img_id_list, 'patch_num': patch_num_list}
df = DataFrame(data)
df.to_excel('f1_patch_num.xlsx')
