# eval.py
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

DATA_DIR = "."   
MODEL_PATH = r"D:\fashion-dna\fashion_hydra_weighted_best.pth"
BATCH_SIZE = 8   
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f" Device: {DEVICE}")

TARGET_COLS = {
    'gender': 'single',
    'subCategory': 'multi',
    'articleType': 'multi',
    'baseColour': 'multi',
    'season': 'multi',
    'usage': 'multi'
}

df = pd.read_csv(os.path.join(DATA_DIR, "styles.csv"), on_bad_lines="skip")
df["image_path"] = df["id"].astype(str).apply(
    lambda x: os.path.join(DATA_DIR, "images", x + ".jpg")
)
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

print(f"📦 Total samples: {len(df)}")

encoders = {}
num_classes_dict = {}

for col in TARGET_COLS:
    le = LabelEncoder()
    df[col] = df[col].fillna("Unknown").astype(str)
    df[col + "_idx"] = le.fit_transform(df[col])
    encoders[col] = le
    num_classes_dict[col] = len(le.classes_)

df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
split_idx = int(0.8 * len(df))
df_test = df.iloc[split_idx:].reset_index(drop=True)

print(f" Test samples: {len(df_test)}")

class FashionHydraDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))

        img = self.transform(img)

        target = {}
        for col in TARGET_COLS:
            label = self.df.iloc[idx][col + "_idx"]
            if col == "gender":
                target[col] = torch.tensor(label, dtype=torch.long)
            else:
                vec = torch.zeros(num_classes_dict[col], dtype=torch.float32)
                vec[label] = 1.0
                target[col] = vec

        return img, target

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = FashionHydraDataset(df_test, transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class HydraResNet(nn.Module):
    def __init__(self, num_classes_dict):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.in_features = resnet.fc.in_features

        self.heads = nn.ModuleDict({
            col: nn.Sequential(
                nn.Linear(self.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, n)
            ) for col, n in num_classes_dict.items()
        })

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        return {k: head(feat) for k, head in self.heads.items()}

model = HydraResNet(num_classes_dict).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(" Model loaded")

all_preds = {k: [] for k in TARGET_COLS}
all_gts   = {k: [] for k in TARGET_COLS}

with torch.no_grad():
    for imgs, targets in tqdm(test_loader, desc="🔍 Evaluating"):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)

        for col in TARGET_COLS:
            if col == "gender":
                pred = outputs[col].argmax(dim=1).cpu().numpy()
                gt = targets[col].numpy()
            else:
                pred = outputs[col].argmax(dim=1).cpu().numpy()
                gt = targets[col].argmax(dim=1).numpy()

            all_preds[col].extend(pred)
            all_gts[col].extend(gt)

print("\n=== Evaluation Results (Test Set) ===")
for col in TARGET_COLS:
    acc = accuracy_score(all_gts[col], all_preds[col])
    f1  = f1_score(all_gts[col], all_preds[col], average="macro")
    print(f"{col:12s} | Acc: {acc:.4f} | Macro-F1: {f1:.4f}")
