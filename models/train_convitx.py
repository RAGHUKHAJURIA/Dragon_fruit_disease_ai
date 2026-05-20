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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
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


class ModelEMA:
    """Exponential moving average of model weights for stabler validation metrics."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                v.copy_(msd[k])
                continue
            v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)


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
    parser.add_argument("--resume-from", type=str, default="", help="Optional checkpoint path to resume fine-tuning")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay; set 0 to disable EMA")
    parser.add_argument(
        "--pretrain-data-dir",
        type=str,
        default="",
        help="Optional ImageFolder path for stage-1 domain pretraining",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=6, help="Epochs for optional stage-1 pretraining")
    parser.add_argument(
        "--pretrain-fruit-only",
        action="store_true",
        help="Use only classes containing 'fruit' in pretraining dataset",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "val_acc"],
        help="Metric used to save best checkpoint",
    )
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


def _compute_class_weights(targets, num_classes: int) -> torch.Tensor:
    class_counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    weights = class_counts.sum() / (num_classes * class_counts)
    return weights


def _build_subset_loaders(
    dataset,
    train_idx,
    val_idx,
    train_tf,
    val_tf,
    batch_size: int,
    num_workers: int,
    weighted_sampling: bool,
):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_subset.dataset = copy.copy(dataset)
    train_subset.dataset.transform = train_tf
    val_subset.dataset = copy.copy(dataset)
    val_subset.dataset.transform = val_tf

    sampler = None
    class_weights = None
    if weighted_sampling:
        targets = [dataset.samples[i][1] for i in train_idx]
        class_weights = _compute_class_weights(targets, num_classes=len(dataset.classes))
        sample_weights = [class_weights[t].item() for t in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, class_weights


def _train_one_epoch(model, loader, criterion, optimizer, device, ema=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return running_loss / max(1, total), correct / max(1, total)


def _evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(1)
            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = running_loss / max(1, total)
    val_acc = correct / max(1, total)
    macro_f1 = f1_score(all_labels, all_preds, average="macro") if all_labels else 0.0
    return val_loss, val_acc, macro_f1, all_preds, all_labels


def _stage1_pretrain_if_requested(args, model, train_tf, val_tf, device):
    pretrain_dir = os.path.normpath(args.pretrain_data_dir) if args.pretrain_data_dir else ""
    if not pretrain_dir:
        return model
    if not os.path.isdir(pretrain_dir):
        print(f"[Stage-1] Skip pretraining. Directory not found: {pretrain_dir}")
        return model

    pre_ds = datasets.ImageFolder(pretrain_dir)
    if len(pre_ds.classes) < 2:
        print("[Stage-1] Skip pretraining. Need at least 2 classes.")
        return model

    keep_idx = list(range(len(pre_ds)))
    if args.pretrain_fruit_only:
        keep_classes = {i for i, c in enumerate(pre_ds.classes) if "fruit" in c.lower()}
        keep_idx = [i for i, (_, y) in enumerate(pre_ds.samples) if y in keep_classes]
        if len(keep_idx) < 20:
            print("[Stage-1] Skip fruit-only filtering due to insufficient samples.")
            keep_idx = list(range(len(pre_ds)))

    pre_targets = [pre_ds.samples[i][1] for i in keep_idx]
    unique_pre = sorted(set(pre_targets))
    if len(unique_pre) < 2:
        print("[Stage-1] Skip pretraining. Filtered source has <2 classes.")
        return model

    tr_idx, va_idx = train_test_split(
        keep_idx,
        train_size=0.9,
        random_state=42,
        stratify=pre_targets,
    )

    tr_loader, va_loader, _ = _build_subset_loaders(
        pre_ds,
        tr_idx,
        va_idx,
        train_tf,
        val_tf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weighted_sampling=False,
    )

    old_head = model.head
    pre_classes = len(pre_ds.classes)
    model.head = nn.Linear(model.head.in_features, pre_classes).to(device)
    nn.init.xavier_uniform_(model.head.weight)
    nn.init.zeros_(model.head.bias)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.pretrain_epochs))

    print(f"\n{'=' * 50}")
    print(f"  Stage-1 domain pretraining ({args.pretrain_epochs} epochs)")
    print(f"  Dataset: {pretrain_dir}")
    print(f"  Classes: {pre_ds.classes}")
    print(f"{'=' * 50}")

    for epoch in range(1, args.pretrain_epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _ = _evaluate(model, va_loader, criterion, device)
        scheduler.step()
        print(
            f"[Stage-1] Epoch {epoch:02d}/{args.pretrain_epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f}"
        )

    model.head = old_head.to(device)
    nn.init.xavier_uniform_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    return model


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

    train_loader, val_loader, class_weights = _build_subset_loaders(
        full_dataset,
        train_idx,
        val_idx,
        train_transforms,
        val_transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weighted_sampling=True,
    )

    model = build_convitx_base(num_classes=num_classes, enforce_budget=True).to(device)
    param_count = count_parameters(model)

    if args.resume_from:
        resume_path = os.path.normpath(args.resume_from)
        if os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt, strict=False)
            print(f"Resumed model weights from: {resume_path}")
        else:
            print(f"Resume checkpoint not found, starting fresh: {resume_path}")

    model = _stage1_pretrain_if_requested(args, model, train_transforms, val_transforms, device)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    
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

    # Advanced Scheduling: warmup, followed by Cosine Annealing decay
    warmup_epochs = min(3, max(1, args.epochs))
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_metric = -1.0
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

    print(f"Selection    : {args.selection_metric}")
    print(f"Class weight : {[round(v, 3) for v in class_weights.tolist()]}")
    print(f"EMA decay    : {args.ema_decay if args.ema_decay > 0 else 'disabled'}")

    print(f"\n{'=' * 50}")
    print(f"  Starting Stage-2 disease fine-tuning — {args.epochs} epochs")
    print(f"{'=' * 50}\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device, ema=ema)
        eval_model = ema.ema if ema is not None else model
        val_loss, val_acc, val_f1, _, _ = _evaluate(eval_model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{args.epochs}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  Macro-F1: {val_f1:.4f}"
        )

        current_metric = val_f1 if args.selection_metric == "macro_f1" else val_acc
        if current_metric > best_metric:
            best_metric = current_metric
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_model_wt = copy.deepcopy(eval_model.state_dict())
            torch.save(best_model_wt, best_model_path)
            print(
                f"  New best model saved "
                f"(Val Acc: {best_val_acc:.4f}, Val Macro-F1: {best_val_f1:.4f})"
            )
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
    print(f"  Best Val Macro-F1: {best_val_f1:.4f}")
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
    axes[1].plot(history["val_f1"], label="Val Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy / Macro-F1 Curve")
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
    _, _, final_f1, all_preds, all_labels = _evaluate(model, val_loader, criterion, device)

    print("\nClassification Report (Validation Set):")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
        )
    )
    print(f"Validation Macro-F1: {final_f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))))


if __name__ == "__main__":
    main()
