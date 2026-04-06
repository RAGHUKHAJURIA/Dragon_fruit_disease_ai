"""
Dragon Fruit Disease Classification - Model Training
Uses EfficientNet-B3 (or ResNet-50) for transfer learning.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR    = "../dataset"
MODEL_DIR   = "./"
MODEL_NAME  = "efficientnet_b3"   # or "resnet50"
NUM_CLASSES = 5                    # update based on your dataset classes
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Healthy",
    "Anthracnose",
    "Stem_Canker",
    "Fruit_Rot",
    "Brown_Spot",
]

# ─── DATA TRANSFORMS ─────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── BUILD MODEL ─────────────────────────────────────────────────────────────
def build_model(num_classes: int, model_name: str = MODEL_NAME) -> nn.Module:
    """Load pretrained EfficientNet/ResNet and replace classifier head."""
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc

# ─── VALIDATION LOOP ─────────────────────────────────────────────────────────
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")

    train_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),      transform=train_transforms)
    val_ds    = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"),  transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Classes: {train_ds.classes}")

    model     = build_model(len(train_ds.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} ──")
        tr_loss, tr_acc = train(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = validate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")
        print(f"Val   Loss: {vl_loss:.4f} | Val   Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"  ✅ Saved best model (Val Acc: {best_val_acc:.4f})")

    # Plot training curves
    _plot_history(history)
    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")


def _plot_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("../results/training_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
