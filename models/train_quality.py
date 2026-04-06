"""
Train lightweight ConViTXSmall for Dragon Fruit Quality Grading.
Primary objective: keep trainable parameter count <= 700,000 for remote deployment.

Default dataset layout expected:
  dataset/Dragon Fruit Quality Grading Dataset/Augmented Dataset/<class_name>/*.jpg

Outputs:
  - quality_convitx.pth
  - quality_training_curves.png
  - quality_classes.txt
"""

import argparse
import copy
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
try:
    import torch_directml
    _HAS_DML = True
except ImportError:
    _HAS_DML = False
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

import torch.nn.functional as F
from convitx import build_convitx_base, count_parameters


class FocalLoss(nn.Module):
    """Focal Loss — suppresses easy examples and focuses on hard misclassifications."""
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class RandomBrightnessContrast:
    """Torchvision equivalent of RandomBrightnessContrast augmentation."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        brightness_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
        contrast_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        return img


def _compute_class_weights(targets, num_classes: int) -> torch.Tensor:
    class_counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    weights = class_counts.sum() / (num_classes * class_counts)
    return weights


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.normpath(
        os.path.join(
            script_dir,
            "..",
            "dataset",
            "Dragon Fruit Quality Grading Dataset",
            "Augmented Dataset",
        )
    )

    parser = argparse.ArgumentParser(description="Train quality grading model (ConViTXSmall)")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="ImageFolder data directory")
    parser.add_argument("--save-dir", type=str, default=script_dir, help="Output directory for model and plots")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4) # Higher LR for ConViTX from scratch
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        RandomBrightnessContrast(brightness=0.15, contrast=0.15, p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif _HAS_DML:
        device = torch_directml.device()
    else:
        device = torch.device("cpu")

    data_dir = os.path.normpath(args.data_dir)
    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_tf, val_tf = build_transforms(args.img_size)

    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError(f"At least 2 classes required, found: {class_names}")

    targets = [s[1] for s in full_dataset.samples]
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), 
        train_size=args.train_split, 
        random_state=42, 
        stratify=targets
    )
    
    train_size = len(train_idx)
    val_size = len(val_idx)

    train_targets = [targets[i] for i in train_idx]
    class_weights = _compute_class_weights(train_targets, num_classes)
    sample_weights = [class_weights[label].item() for label in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_subset.dataset = copy.copy(full_dataset)
    train_subset.dataset.transform = train_tf
    val_subset.dataset = copy.copy(full_dataset)
    val_subset.dataset.transform = val_tf

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_convitx_base(num_classes=num_classes, enforce_budget=True).to(device)
    param_count = count_parameters(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    
    # Configure parameter groups for dynamic weight decay
    if str(device).startswith("privateuseone"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-5,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-5,
            foreach=False,
            fused=False,
        )

    # Warmup for 3 epochs then cosine decay (mirrors train_convitx.py)
    warmup_epochs = 3
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    # Start below any real accuracy so epoch 1 is always checkpointed.
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    model_path = os.path.join(save_dir, "quality_convitx.pth")
    curve_path = os.path.join(save_dir, "quality_training_curves.png")
    class_map_path = os.path.join(save_dir, "quality_classes.txt")

    print(f"Device       : {device}")
    print(f"Dataset      : {data_dir}")
    print(f"Classes      : {class_names}")
    print(f"Train/Val    : {train_size}/{val_size}")
    print(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")
    print(f"Parameters   : {param_count:,} trainable (budget <= 700,000)")

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = train_loss_sum / train_size
        train_acc = train_correct / train_size

        model.eval()
        val_loss_sum, val_correct = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss = val_loss_sum / val_size
        val_acc = val_correct / val_size
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, model_path)
            patience_counter = 0
            print(f"  New best model saved: {model_path} (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    elapsed = time.time() - start
    print(f"Training complete in {elapsed / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    with open(class_map_path, "w", encoding="utf-8") as f:
        for idx, cls in enumerate(class_names):
            f.write(f"{idx}\t{cls}\n")
    print(f"Saved class mapping: {class_map_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves: {curve_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        final_preds, final_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                preds = model(images).argmax(1).cpu().tolist()
                final_preds.extend(preds)
                final_labels.extend(labels.tolist())

        print("\nValidation classification report:")
        print(classification_report(final_labels, final_preds, target_names=class_names))


if __name__ == "__main__":
    main()
