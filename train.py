import torch
import torch.optim as optim

# ---- PLACEHOLDER SETUP (for reproducibility reference) ----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return {
            "gender": x,
            "subCategory": x,
            "articleType": x,
            "baseColour": x,
            "season": x,
            "usage": x
        }

model = DummyModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_functions = {
    "gender": torch.nn.CrossEntropyLoss(),
    "subCategory": torch.nn.CrossEntropyLoss(),
    "articleType": torch.nn.CrossEntropyLoss(),
    "baseColour": torch.nn.CrossEntropyLoss(),
    "season": torch.nn.CrossEntropyLoss(),
    "usage": torch.nn.CrossEntropyLoss(),
}

# Dummy dataloader (structure-compatible)
train_loader = [
    (torch.randn(2, 3, 224, 224), {
        "gender": torch.zeros(2, dtype=torch.long),
        "subCategory": torch.zeros(2, dtype=torch.long),
        "articleType": torch.zeros(2, dtype=torch.long),
        "baseColour": torch.zeros(2, dtype=torch.long),
        "season": torch.zeros(2, dtype=torch.long),
        "usage": torch.zeros(2, dtype=torch.long),
    })
]

def train():
    import copy
    import time

    NUM_EPOCHS = 35
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            outputs = model(images)

            loss = 0
            loss += loss_functions['gender'](outputs['gender'], targets['gender'])

            for col in ['subCategory', 'articleType', 'baseColour', 'season', 'usage']:
                loss += loss_functions[col](outputs[col], targets[col])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "checkpoints/model_best.pth")

    model.load_state_dict(best_model_wts)
if __name__ == "__main__":
    train()
