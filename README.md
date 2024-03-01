# [CVPR-2024] AllSpark: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation

This repo is the official implementation of AllSpark: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation which is accepted at CVPR-2024.

<p align="left">
<img src="./docs/framework.pdf" width=90% height=90% class="center">
</p>





## 1. Environment

First, create a new environment and install the requirements:
```shell
conda create -n allspark python=3.7
conda activate allspark
cd AllSpark/
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tensorboard
pip install six
pip install pyyaml
pip install -U openmim
mim install mmcv
pip install einops
pip install timm
```

## 2. Data Preparation & Pre-trained Weights

### 2.1 Pascal VOC 2012 Dataset
Download the dataset with wget:
```shell
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EcgD_nffqThPvSVXQz6-8T0B3K9BeUiJLkY_J-NvGscBVA\?e\=2b0MdI\&download\=1 -O pascal.zip
```
Unzip the dataset:
```shell
unzip pascal.zip
```

### 2.2 Cityscapes Dataset
Download the dataset with wget:
```shell
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EcgD_nffqThPvSVXQz6-8T0B3K9BeUiJLkY_J-NvGscBVA\?e\=2b0MdI\&download\=1 -O cityscapes.zip
```
Unzip the dataset:
```shell
unzip cityscapes.zip
```

### 2.3 COCO Dataset
Download the dataset with wget:
```shell
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EcgD_nffqThPvSVXQz6-8T0B3K9BeUiJLkY_J-NvGscBVA\?e\=2b0MdI\&download\=1 -O coco.zip
```
Unzip the dataset:
```shell
unzip coco.zip
```


Then your file structure will be like:

```
├── VOC2012
    ├── JPEGImages
    └── SegmentationClass
    
├── cityscapes
    ├── leftImg8bit
    └── gtFine
    
├── coco
    ├── train2017
    ├── val2017
    └── masks
```

Next, download the following [pretrained weights](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hwanggr_connect_ust_hk/Eobv9tk6a6RJqGXEDm2D_TcB2mEn4r2-BLDkotZHkd2l6w?e=fJBy7v).

```
├── ./pretrained_weights
    ├── mit_b2.pth
    ├── mit_b3.pth
    ├── mit_b4.pth
    └── mit_b5.pth
```


## 3. Training & Evaluating

```bash
# use torch.distributed.launch
sh scripts/train.sh <num_gpu> <port>
# to fully reproduce our results, the <num_gpu> should be set as 4 on all three datasets
# otherwise, you need to adjust the learning rate accordingly

# or use slurm
# sh scripts/slurm_train.sh <num_gpu> <port> <partition>
```

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/scripts/train.sh).


## 4. Results


Model weights and training logs will be released soon.




