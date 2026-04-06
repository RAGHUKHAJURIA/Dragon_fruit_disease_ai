"""
Train lightweight ConViTXSmall for dragon fruit disease classification.

Primary objective: keep trainable parameter count <= 700,000 for remote deployment.
"""

import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
try:
    import torch_directml

    _HAS_DML = True
except ImportError:
    _HAS_DML = False
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from convitx import build_convitx_base, count_parameters
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss to dynamically scale gradients based on prediction confidence.
    Heavily targets 'hard' misclassified examples (like Canker vs Anthracnose).
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.normpath(
        os.path.join(
            script_dir,
            "..",
            "dataset",
            "Dragon Fruit (Pitahaya)",
            "Dragon Fruit (Pitahaya)",
            "Converted Images",
        )
    )

    parser = argparse.ArgumentParser(description="Train ConViTXSmall on dragon fruit disease images")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Path to ImageFolder dataset")
    parser.add_argument("--save-dir", type=str, default=script_dir, help="Directory to save model and plots")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count")
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience")
    return parser.parse_args()


def build_transforms(img_size: int):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, val_transforms


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif _HAS_DML:
        device = torch_directml.device()
    else:
        device = torch.device("cpu")

    data_dir = os.path.normpath(args.data_dir)
    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_transforms, val_transforms = build_transforms(args.img_size)

    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes for training. Found classes: {class_names}")

    targets = [s[1] for s in full_dataset.samples]
    
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), 
        train_size=args.train_split, 
        random_state=42, 
        stratify=targets
    )
    
    train_size = len(train_idx)
    val_size = len(val_idx)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_dataset.dataset = copy.copy(full_dataset)
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset = copy.copy(full_dataset)
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_convitx_base(num_classes=num_classes, enforce_budget=True).to(device)
    param_count = count_parameters(model)

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    
    transformer_params = []
    cnn_params = []
    
    for name, param in model.named_parameters():
        if "transformer_blocks" in name or "patch_embed" in name or "pos_embed" in name:
            transformer_params.append(param)
        else:
            cnn_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": transformer_params, "weight_decay": 1e-3},
        {"params": cnn_params, "weight_decay": 1e-5}
    ], lr=args.lr)

    # Advanced Scheduling: 3-epoch warmup, followed by Cosine Annealing decay
    warmup_epochs = 3
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_wt = None
    patience_counter = 0

    best_model_path = os.path.join(save_dir, "best_convitx.pth")
    curve_path = os.path.join(save_dir, "convitx_training_curves.png")

    print(f"Device       : {device}")
    print(f"Data dir     : {data_dir}")
    print(f"Classes      : {class_names}")
    print(f"Train images : {train_size}")
    print(f"Val images   : {val_size}")
    print(f"Parameters   : {param_count:,} trainable (budget <= 700,000)")

    print(f"\n{'=' * 50}")
    print(f"  Starting ConViTX training — {args.epochs} epochs")
    print(f"{'=' * 50}\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss / train_size
        train_acc = correct / train_size

        model.eval()
        running_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = running_loss / val_size
        val_acc = correct / val_size
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wt = copy.deepcopy(model.state_dict())
            torch.save(best_model_wt, best_model_path)
            print(f"  New best model saved (Val Acc: {best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"  Training complete in {elapsed / 60:.1f} min")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"{'=' * 50}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {curve_path}")

    if best_model_wt is None:
        print("No improvement found; skipping final classification report.")
        return

    model.load_state_dict(best_model_wt)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\nClassification Report (Validation Set):")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
