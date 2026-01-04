# Fashion DNA – Multi-task Fashion Attribute Prediction

This repository contains the source code for the paper:
"Fashion DNA: A Multi-Task Learning Model for Fashion Attribute Detection".

## Files
- train.py: Training script for the multi-task model
- eval.py: Evaluation / inference script
- fashion_hydra_weighted_best.pth: Trained model checkpoint

## Requirements
- Python 3.8+
- PyTorch
- torchvision

## Dataset
- Fashion Product Images (Small) – Kaggle (2019)  
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
## Pretrained Model

Due to GitHub file size limits, the trained model checkpoint is hosted on Google Drive.

- fashion_hydra_weighted_best.pth  
  [https://drive.google.com/xxxxxxxx](https://drive.google.com/drive/folders/1vaDP8p5yhxmp3AdRIkWkuh_CrAAz0Y2s?usp=sharing)

## Training
The model was trained using weighted loss functions to handle class imbalance.

Example:
```bash
python train.py
python eval.py

## Demo
The example image shown in the paper (e.g., public figure image) is used
for illustrative purposes only.

It is NOT included in the training or evaluation datasets.

