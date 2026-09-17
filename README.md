# Behavior-Aligned Fine-Tuning without Extra Parameters
This repository provides code for "Behavior-Aligned Fine-Tuning without Extra Parameters".

# Overview
We introduce a simple yet effective method that provides behavioral guidance during fine-tuning using signals already present during training. By extracting the relational structure of model
predictions, capturing how samples relate in the output space, and aligning it with intermediate feature representations, we impose a task-aware constraint that minimizes the discrepancy between pairwise feature similarities and pairwise prediction similarities within each mini-batch, explicitly enforcing that samples with similar class probability distributions are encoded with correspondingly similar embeddings. This alignment structures the feature space according to the model’s evolving class-separation patterns, improving representation consistency and discriminative geometry during adaptation. Our method introduces no additional learnable parameters, and can be integrated into a wide range of PEFT techniques.

# Requirements 
- Python 3.8
- torch 1.10.0
- torchvision 0.11.1
- timm 0.4.12
  

# Pretrained Model
Download the [pretrained model ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) and place it in the root folder.

# Data Preparation
1. VTAB-1k: Please refer to [SSF](https://github.com/dongzelian/SSF) or [VPT](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md) for preparing the 19 datasets included in VTAB-1K.
2. FGVC: Follow [NOAH](https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation) to download the dataset.

## Training Scripts
### 1. Train on VTAB-1K

```bash
python train.py \
  --task vtab \
  --dataset cifar \
  --method bi-adaptformer \
  --lambda_value 5e-3 \
  --feature token_mean \
  --reg_layers all \
  --epochs 300 \
  --lr 1e-3 \
  --wd 1e-4
```

### 2. Train on Few-Shot Learning

```bash
python train.py \
  --task fs \
  --dataset cifar \
  --fs_shot 16 \
  --fs_seed 0 \
  --method bi-adaptformer \
  --lambda_value 1e-3 \
  --feature token_mean \
  --reg_layers all \
  --epochs 100 \
  --lr 5e-3 \
  --wd 1e-4
```

   


