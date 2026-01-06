# Fashion DNA – Multi-task Fashion Attribute Prediction

This repository contains the official implementation for the paper:

**“Fashion DNA: A Multi-Task Learning Model for Fashion Attribute Detection”**

The project focuses on multi-task visual recognition for fashion products,
jointly predicting multiple attributes from a single image using a shared
CNN backbone with task-specific heads.

---

## Repository Structure

- `train.py`  
  Training script for the multi-task fashion attribute model.

- `eval.py`  
  Evaluation and inference script used to compute performance on a held-out test set.

- `fashion_hydra_weighted_best.pth`  
  Trained model checkpoint (**hosted externally** due to GitHub file size limits).
  
---
## Pretrained Model

Due to GitHub file size limitations, the trained model checkpoint is hosted externally.

- fashion_hydra_weighted_best.pth  
  [https://drive.google.com/xxxxxxxx](https://drive.google.com/drive/folders/1vaDP8p5yhxmp3AdRIkWkuh_CrAAz0Y2s?usp=sharing)
---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- tqdm
- pillow

---

## Dataset

This project uses the **Fashion Product Images (Small)** dataset:

- Source: Kaggle (2019)  
  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
  
Due to licensing restrictions, the dataset is **not redistributed** in this repository.
Please download the dataset from the official source and organize it as follows:

```text
data/
 └── fashion-dna/
     ├── images/
     └── styles.csv



