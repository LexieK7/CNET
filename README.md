# Deep learning model with pathological knowledge for detection of colorectal neuroendocrine tumor
#### Cell reports Medicine 


#### [Journal Link](https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(24)00532-9?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2666379124005329%3Fshowall%3Dtrue)

##Overview
####Abstract
Colorectal neuroendocrine tumors (NETs) differ significantly from colorectal carcinoma (CRCs) in terms of treatment strategy and prognosis, necessitating a cost-effective approach for accurate discrimination. Here, we propose an approach for distinguishing between colorectal NET and CRC based on pathological images by utilizing pathological prior information to facilitate the generation of robust slide-level features. By calculating the similarity between morphological descriptions and patches, our approach selects only 2% of the diagnostically relevant patches for both training and inference, achieving an area under the receiver operating characteristic curve (AUROC) of 0.9974 on the internal dataset, and AUROCs of 0.9724 and 0.9513 on two external datasets. Our model effectively identifies NETs from CRCs, reducing unnecessary immunohistochemical tests and enhancing the precise treatment for patients with colorectal tumors. Our approach also enables researchers to investigate methods with high accuracy and low computational complexity, thereby advancing the application of artificial intelligence in clinical settings.

![image](https://github.com/LexieK7/CNET/blob/main/Figure/model.png)


## How to use our model?

### 1. Installation

First clone the repo and cd into the directory:

```
https://github.com/LexieK7/wsi_text.git
cd wsi_text
```

Then create a conda env and install the dependencies:

See CLAM.(https://github.com/mahmoodlab/CLAM)

### 2. Using QUILTNET as Pretrained Encoders

use CLAM to segment the WHOLE SLIDE IMAGE and extract the features, note that the encoder we use is Quiltnet(https://github.com/wisdomikezogwo/quilt1m) (other visual language foundation models are also possible).

set up your description text for the categories you need to categorize and run topk_h5_pt.py to extract the relevant features.


```
net_des = tokenizer("CLASS A'S DESCRIPTION.")

crc_des = tokenizer("CLASS B'S DESCRIPTION.")

```

```
python topk_h5_pt.py
```

### 3. Training model

edit the code according to your needs.

```
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/NETvsCRC.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_feat_resnet'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'NET':0, 'CRC':1},
                            label_col = 'label',
                            ignore=[])
```

```
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --k 5
```
```
CUDA_VISIBLE_DEVICES=0 python main.py --early_stopping --lr 2e-4 --k 5 --exp_code task_1_tumor_vs_normal_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_mb --log_data --data_root_dir DATA_ROOT_DIR 
```

### 4. Easy to use

If you are more familiar with the use of CLAM. You can use our code to adjust the features and then train with any model.


```
python topk_h5_pt.py
```

## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```
@article{zheng2024deep,
  title={Deep learning model with pathological knowledge for detection of colorectal neuroendocrine tumor},
  author={Zheng, Ke and Duan, Jinling and Wang, Ruixuan and Chen, Haohua and He, Haiyang and Zheng, Xueyi and Zhao, Zihan and Jing, Bingzhong and Zhang, Yuqian and Liu, Shasha and others},
  journal={Cell Reports Medicine},
  volume={5},
  number={10},
  year={2024},
  publisher={Elsevier}
}
```
