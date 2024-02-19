import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB#, CLAM_transformer#
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger, summary, summary_cc
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from topk.svm import SmoothTop1SVM
from datasets.dataset_generic import Generic_MIL_Dataset
from torch.utils.data import DataLoader, sampler

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]
 
print('Init Model')   
instance_loss_fn = SmoothTop1SVM(n_classes = 2)
instance_loss_fn = instance_loss_fn.cuda()
model_dict = {"dropout": True, 'n_classes': 2}
model_dict.update({'k_sample': 8})

model = CLAM_MB(**model_dict)
#model = CLAM_transformer(**model_dict)
print_network(model)
ckpt = torch.load("MODEL_PATH")
ckpt_clean = {}
for key in ckpt.keys():
    if 'instance_loss_fn' in key:
        continue
    ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
model.load_state_dict(ckpt_clean, strict=True)
model.relocate()
model.eval()


print('Init Loaders')
dataset = Generic_MIL_Dataset(csv_path = "CSV_PATH",#NET_vs_CRC_tj.csv #CRC_vs_NET_dummy_clean_cc.csv #biopsy_cc.csv
                        data_dir= os.path.join("DATASET_PATH", 'tumor_vs_normal_resnet_features'),
                        shuffle = False, 
                        print_info = True,
                        label_dict = {'NET':0, 'CRC':1},
                        patient_strat=False,
                        ignore=[])

loader = DataLoader(dataset, batch_size=1, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL)

results_dict, test_error, test_auc, acc_logger, test_f1_score, test_sensitivity, test_specificity = summary(model, loader, 2)


print('acc:', 1-test_error, 'auc:',test_auc, 'f1:',test_f1_score, 'sen:',test_sensitivity, 'spe:',test_specificity)