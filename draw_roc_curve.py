import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor

import os
from utils.eval_utils import *
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_MIL_Dataset

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False,
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False,
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
args = parser.parse_args()


if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)

models_dir = "MODEL_PATH"
ckpt_paths = [os.path.join(models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]

args.n_classes=2
all_results = []
all_auc = []
all_acc = []
all_fpr = []
all_tpr = []
color_list = ["#55efc4", "#81ecec", "#74b9ff", "#a29bfe", "#ffeaa7", "#fab1a0", "#ff7675" ,"#fd79a8", "#006266", "#5758BB"] 

plt.figure()
for ckpt_idx in range(len(ckpt_paths)):

    dataset = Generic_MIL_Dataset(csv_path="CSV_PATH", #NET_vs_CRC_tj.csv #CRC_vs_NET_dummy_clean_cc.csv #biopsy_cc.csv
                                  data_dir=os.path.join("DATASET_PATH",
                                                        'tumor_vs_normal_resnet_features'), 
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'NET': 0, 'CRC': 1},
                                  patient_strat=False,
                                  ignore=[])

        
    model, patient_results, test_error, auc, df, fpr, tpr  = eval(dataset, args, ckpt_paths[ckpt_idx])
    all_results.append(all_results)
    all_auc.append(auc)
    all_acc.append(1-test_error)
    all_fpr.append(fpr)
    all_tpr.append(tpr)

    print(fpr)
    print(tpr)
    

    plt.plot(fpr, tpr,label="Fold "+str(ckpt_idx)+' (AUC=' + str(round(auc,4))+')', color = color_list[ckpt_idx])
    
    
plt.plot([0, 1], [0, 1], '--', color = 'black')
#plt.plot([0., 0., 0., 0., 0., 0., 0., 1.],[0.,0.0952381,0.19047619,0.33333333,0.38095238,0.47619048,1.,1.], '--', color = 'red')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')

plt.legend(loc=4)
plt.savefig('biopy_roc_15p.png')