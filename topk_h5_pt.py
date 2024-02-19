import os
import h5py
import numpy as np

import open_clip
import torch
import heapq


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path
    

sample_list = os.listdir("dataset_h5_path")
base_h5_save_dir = "h5_save_path"
base_pt_save_dir = "pt_save_path"


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained="./quilt_weight/open_clip_pytorch_model.bin")
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.cuda()
model.eval()


net_des = tokenizer("solid, trabecular, gyriform, or glandular pattern, with fairly uniform nuclei, coarsely stippled chromatin, and finely granular cytoplasm.")#NETs show a 

crc_des = tokenizer("adenocarcinoma, glandular formation, invades into the submucosa, desmoplastic reaction, inflammatory reactions, necrotic debris in glandular lumina.")


crc_des = crc_des.cuda()
crc_des = model.encode_text(crc_des)
crc_des /= crc_des.norm(dim=-1, keepdim=True)


net_des = net_des.cuda()
net_des = model.encode_text(net_des)
net_des /= net_des.norm(dim=-1, keepdim=True)



for i in range(len(sample_list)):
    now_sample = "dataset_h5_path" + sample_list[i]
    print(now_sample)
    
    info_list = []

    f = h5py.File(now_sample)
    patch_np = np.array(f['features'][:])
    patch_cor = np.array(f['coords'][:])


    image_features = torch.from_numpy(patch_np).cuda()
    image_features /= image_features.norm(dim=-1, keepdim=True)
 

    text_probs_crc = (image_features @ crc_des.T)
    text_probs_net = (image_features @ net_des.T)
    
    
    text_probs_crc = text_probs_crc.squeeze().tolist()
    text_probs_net = text_probs_net.squeeze().tolist()
    
    
    sample_num = int(patch_np.shape[0] * 0.05)

    top500_index_net = heapq.nlargest(sample_num, range(len(text_probs_net)), text_probs_net.__getitem__)
    top500_index_crc = heapq.nlargest(sample_num, range(len(text_probs_crc)), text_probs_crc.__getitem__)

    
    
    net_fea = patch_np[top500_index_net ,:]
    net_cor = patch_cor[top500_index_net ,:]
    crc_fea = patch_np[top500_index_crc ,:]
    crc_cor = patch_cor[top500_index_crc ,:]

    
    features = np.concatenate((net_fea,crc_fea),axis=0)
    coords = np.concatenate((net_cor,crc_cor),axis=0)

    
    now_h5_path = base_h5_save_dir + sample_list[i]
    now_pt_path = base_pt_save_dir + sample_list[i].replace('h5','pt')

        
 
    # save as the same h5 
    asset_dict = {'features': features, 'coords': coords}
    save_hdf5(now_h5_path, asset_dict, attr_dict= None, mode='w')
    
    # save as the same pt
    
    features = torch.from_numpy(features)
    torch.save(features, now_pt_path)



