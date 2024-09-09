# Deep Learning Model with Pathological Knowledge for Detection of Colorectal Neuroendocrine Tumor
=============

## 

Link.
Abstract:

## How to use our model?

### 1. Installation

First clone the repo and cd into the directory:

'''
https://github.com/LexieK7/wsi_text.git
cd wsi_text
'''

Then create a conda env and install the dependencies:

See CLAM.(https://github.com/mahmoodlab/CLAM)

### 2. Using QUILTNET(https://github.com/wisdomikezogwo/quilt1m) as Pretrained Encoders

use CLAM to segment the WHOLE SLIDE IMAGE and extract the features, note that the encoder we use is Quiltnet (other visual language foundation models are also possible).

set up your description text for the categories you need to categorize and run topk_h5_pt.py to extract the relevant features.

'''
python topk_h5_pt.py
'''

### 3. Training model

'''
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --k 5
'''
'''
CUDA_VISIBLE_DEVICES=0 python main.py --early_stopping --lr 2e-4 --k 5 --exp_code task_1_tumor_vs_normal_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_mb --log_data --data_root_dir DATA_ROOT_DIR 
'''

## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our paper:
