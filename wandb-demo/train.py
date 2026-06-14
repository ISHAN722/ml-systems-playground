import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# config hyperparameters
config = {
    "learning_rate": 0.01,
    "epochs": 20,
    "batch_size": 32,
    "hidden_size": 64,
    "optimizer": "adam"
}

# init W&B run
wandb.init(
    project = "wandb-demo",
    config = config
)
cfg = wandb.config

# fake data
torch.manual_seed(42)
X = torch.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).float()

dataset = TensorDataset(X, y)
train_size = 800
val_size = 200
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

# sample MLP
model = nn.Sequential(
    nn.Linear(10, cfg.hidden_size),
    nn.ReLU(),
    nn.Linear(cfg.hidden_size, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = cfg.learning_rate)


# training loop
for epoch in range(cfg.epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += ((preds > 0.5) == yb).sum().item()
        total += len(yb)

    train_acc = correct/ total

    # validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            val_loss += loss.item()
            val_correct += ((preds > 0.5) == yb).sum().item()
            val_total += len(yb)

    val_acc = val_correct / val_total

    # log to W&B
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss / len(train_loader),
        "train_acc": train_acc,
        "val_loss": val_loss / len(val_loader),
        "val_acc": val_acc
    })

    print(f"Epoch {epoch+1}/{cfg.epochs} — train_loss: {train_loss/len(train_loader):.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss/len(val_loader):.4f}, val_acc: {val_acc:.4f}")

wandb.finish()
print("\nDone. Check your W&B dashboard at https://wandb.ai")
