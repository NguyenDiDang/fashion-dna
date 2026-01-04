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
